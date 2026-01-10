"""
DWAL (Dynamic Windowed Alignment Loss) Forward Functions

Contains forward implementations for:
- qwen2_5_mixed_modality_forward_laser_dwal: DWAL training
- qwen2_5_mixed_modality_forward_laser_dwal_time_aware: Time-Aware DWAL with Focusing Mechanism
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from transformers.utils import is_torchdynamo_compiling
from src.constants import IGNORE_INDEX


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Custom output class for LASER models with additional loss components.
    Please refer to the original Qwen2_5_VLCausalLMOutputWithPast in
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """
    loss: Optional[torch.FloatTensor] = None
    loss_laser: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    # DWAL metrics
    dwal_confused_ratio: Optional[float] = None
    dwal_laser_end_max_ratio: Optional[float] = None


# ============================================================================
# DWAL: Dynamic Windowed Alignment Loss
# ============================================================================
'''
    DWAL (Dynamic Windowed Alignment Loss) mode
    Uses actual reasoning chain tokens instead of visual embeddings
    KL Loss for LASER region, CE Loss for text region
'''
def qwen2_5_mixed_modality_forward_laser_dwal(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    laser_tokens: Optional[torch.Tensor] = None,
    laser_tokens_thw: Optional[List[torch.Tensor]] = None,
    laser_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    question_ids: Optional[List] = None,
    original_lengths: Optional[List] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        raise ValueError("pixel_values and pixel_values_videos cannot both be None. Image input is required.")

    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)

        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id

        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

    # DWAL mode: No visual token injection for LASER positions
    # The reasoning chain is already in input_ids as actual text tokens

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_laser = None
    dwal_confused_ratio = None

    if labels is not None:
        logits = logits.float()

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()

        batch_size, seq_len = shift_input_ids.shape

        # ============== Build LASER Region Mask ==============
        # Find <|laser_start|> and <|laser_end|> positions
        laser_start_mask = shift_input_ids == self.config.laser_start_id
        laser_end_mask = shift_input_ids == self.config.laser_end_id

        # Build region mask: True for positions between start and end (inclusive of end, exclusive of start)
        # For each sequence, mark positions from start+1 to end (inclusive)
        laser_region_mask = torch.zeros_like(shift_input_ids, dtype=torch.bool)

        for b in range(batch_size):
            start_positions = torch.nonzero(laser_start_mask[b], as_tuple=True)[0]
            end_positions = torch.nonzero(laser_end_mask[b], as_tuple=True)[0]

            for start_pos, end_pos in zip(start_positions, end_positions):
                # Include positions from start+1 to end-1 (exclusive of end)
                # These are the reasoning chain tokens (c_1 to c_T), NOT including <|laser_end|>
                # <|laser_end|> should be learned by CE loss so model knows when to stop
                laser_region_mask[b, start_pos+1:end_pos] = True

        # ============== CE Loss (mask LASER region) ==============
        ce_labels = shift_labels.clone()
        ce_labels[laser_region_mask] = IGNORE_INDEX

        loss_fct = CrossEntropyLoss()
        ce_logits_flat = shift_logits.view(-1, self.config.vocab_size)
        ce_labels_flat = ce_labels.view(-1).to(ce_logits_flat.device)
        loss_ce = loss_fct(ce_logits_flat, ce_labels_flat)

        # ============== DWAL KL Loss (only LASER region) ==============
        # Get DWAL hyperparameters from config
        K_fixed = getattr(self.config, 'dwal_window_size', None)  # None = dynamic window
        tau = getattr(self.config, 'dwal_temperature', 1.0)
        entropy_threshold = getattr(self.config, 'dwal_entropy_threshold', 0.8)
        force_prob = getattr(self.config, 'dwal_force_prob', 0.8)

        # Find all LASER positions (excluding <|laser_end|> itself, we predict up to it)
        # We want positions where model predicts the next reasoning token
        laser_predict_mask = laser_region_mask.clone()
        laser_predict_mask[laser_end_mask] = False  # Don't predict after <|laser_end|>

        if laser_predict_mask.any():
            # Get flattened indices of LASER positions
            batch_indices, seq_indices = torch.nonzero(laser_predict_mask, as_tuple=True)

            # Build end_positions for each LASER position (find corresponding <|laser_end|> position)
            laser_end_positions = torch.zeros_like(seq_indices)
            for b in range(batch_size):
                b_mask = batch_indices == b
                if b_mask.any():
                    # Find all <|laser_end|> positions in this batch
                    end_pos_in_b = torch.nonzero(laser_end_mask[b], as_tuple=True)[0]
                    # For each LASER position in this batch, find its corresponding end
                    b_seq_indices = seq_indices[b_mask]
                    for i, (pos_idx, pos) in enumerate(zip(torch.nonzero(b_mask, as_tuple=True)[0], b_seq_indices)):
                        # Find the first end position > current position
                        valid_ends = end_pos_in_b[end_pos_in_b > pos]
                        if len(valid_ends) > 0:
                            laser_end_positions[pos_idx] = valid_ends[0]
                        else:
                            laser_end_positions[pos_idx] = seq_len - 1

            # Compute dynamic window size K: from current position to laser_end (inclusive)
            # Each position can see tokens from current to laser_end (inclusive)
            per_position_window_size = laser_end_positions - seq_indices + 1  # +1 to include both ends
            if K_fixed is not None:
                K = K_fixed
            else:
                K = int(per_position_window_size.max().item())  # dynamic: use max window needed

            if K <= 0:
                K = 1  # at least 1

            # Get logits at LASER positions: [N_laser, vocab_size]
            laser_logits = shift_logits[batch_indices, seq_indices]

            # Build window indices: for each position, get current and next K-1 GT token IDs
            # window_positions[i, k] = seq_indices[i] + k (current to K-1 tokens ahead)
            # Position 0 is the immediate GT token (what shift_logits[t] should predict)
            offsets = torch.arange(0, K, device=input_ids.device)  # [K]
            window_positions = seq_indices.unsqueeze(-1) + offsets  # [N_laser, K]

            # Valid mask: position must be within seq_len AND within LASER region (EXCLUDING laser_end)
            # <|laser_end|> is supervised by CE loss, not DWAL loss - this prevents shortcut learning
            valid_mask = (window_positions < seq_len) & (window_positions < laser_end_positions.unsqueeze(-1))
            window_positions_clamped = window_positions.clamp(max=seq_len - 1)

            # Get GT token IDs in the window
            # Use advanced indexing: for each (batch_idx, window_pos) get the token
            window_token_ids = shift_input_ids[batch_indices.unsqueeze(-1).expand_as(window_positions_clamped),
                                                window_positions_clamped]  # [N_laser, K]

            # Gather window logits from vocab dimension
            window_logits = laser_logits.gather(dim=-1, index=window_token_ids)  # [N_laser, K]
            # Clamp logits early to prevent numerical instability in bf16
            window_logits = window_logits.clamp(min=-1e4, max=1e4)

            # ============== Build Teacher Distribution (No Gradient) ==============
            with torch.no_grad():
                teacher_logits = window_logits.detach().clone()

                # Mask invalid positions (padding) with -inf
                teacher_logits[~valid_mask] = float('-inf')

                # Temperature softmax
                teacher_probs = F.softmax(teacher_logits / tau, dim=-1)  # [N_laser, K]

                # Handle NaN from all-inf rows (when no valid positions) - use uniform distribution
                nan_mask = torch.isnan(teacher_probs).any(dim=-1)
                if nan_mask.any():
                    teacher_probs[nan_mask] = 1.0 / K

                # Build hard target (one-hot pointing to immediate GT token, i.e., position 0)
                hard_target = torch.zeros_like(teacher_probs)
                hard_target[:, 0] = 1.0

                # Count valid positions
                valid_count = valid_mask.sum(dim=-1).float()  # [N_laser]

                # Handle valid_count == 1: only one token, just align to it directly (no KL needed)
                single_token_mask = (valid_count == 1)  # [N_laser]

                # For positions with valid_count >= 2: compute entropy-based adaptive intervention
                # For positions with valid_count == 1: directly use hard_target
                eps = 1e-8

                # Safe entropy computation (only for valid_count >= 2)
                entropy = -(teacher_probs * torch.log(teacher_probs + eps)).sum(dim=-1)  # [N_laser]
                # Avoid log(1)=0 division: only divide where valid_count >= 2
                safe_valid_count = valid_count.clamp(min=2)
                normalized_entropy = entropy / torch.log(safe_valid_count)  # [N_laser]
                normalized_entropy = normalized_entropy.clamp(min=0.0, max=1.0)

                # Mix based on entropy threshold (only applies when valid_count >= 2)
                is_confused = (normalized_entropy > entropy_threshold).float().unsqueeze(-1)  # [N_laser, 1]

                # Calculate confused ratio for logging (excluding single token positions)
                multi_token_positions = ~single_token_mask  # positions with valid_count >= 2
                if multi_token_positions.any():
                    dwal_confused_ratio = is_confused.squeeze(-1)[multi_token_positions].mean().item()
                else:
                    dwal_confused_ratio = 0.0

                # Calculate laser_end max probability ratio
                # For each position, check if argmax points to the last valid position (laser_end)
                # Last valid index = valid_count - 1 (0-indexed)
                if multi_token_positions.any():
                    argmax_indices = teacher_probs.argmax(dim=-1)  # [N_laser]
                    last_valid_indices = (valid_count - 1).long()  # [N_laser]
                    is_laser_end_max = (argmax_indices == last_valid_indices)  # [N_laser]
                    # Only count positions with multiple tokens (single token always points to laser_end)
                    dwal_laser_end_max_ratio = is_laser_end_max[multi_token_positions].float().mean().item()
                else:
                    dwal_laser_end_max_ratio = 0.0

                mixed_probs = is_confused * (force_prob * hard_target + (1 - force_prob) * teacher_probs) + \
                              (1 - is_confused) * teacher_probs

                # Final target: use hard_target for single token, mixed_probs for multiple tokens
                target_probs = torch.where(
                    single_token_mask.unsqueeze(-1),
                    hard_target,
                    mixed_probs
                )

            # ============== DWAL Loss Computation ==============
            # Get loss type from config: "mse" or "weighted_ce"
            loss_type = getattr(self.config, 'dwal_loss_type', 'weighted_ce')

            if loss_type == "mse":
                # MSE Loss: align student probability distribution to teacher distribution
                # Student uses softmax to get probability distribution (same as teacher)
                student_probs = F.softmax(window_logits / tau, dim=-1)  # [N_laser, K]

                # Extract valid positions only
                valid_student_probs = student_probs[valid_mask].float()
                valid_target_probs = target_probs[valid_mask].float()

                if valid_student_probs.numel() > 0:
                    loss_laser = F.mse_loss(valid_student_probs, valid_target_probs)
                else:
                    loss_laser = logits.sum() * 0.0

            elif loss_type == "weighted_ce":
                # Weighted CE Loss: teacher distribution acts as weights for CE loss
                # Student computes CE for each window token, weighted by teacher's confidence

                # Student: compute log_softmax over FULL vocabulary
                # laser_logits: [N_laser, vocab_size]
                student_log_probs = F.log_softmax(laser_logits, dim=-1)  # [N_laser, vocab_size]

                # Gather log probs for window tokens: -log P(c_{t+k})
                window_log_probs = student_log_probs.gather(dim=-1, index=window_token_ids)  # [N_laser, K]

                # CE loss for each window position
                ce_per_token = -window_log_probs  # [N_laser, K]

                # Mask invalid positions (set CE to 0 so they don't contribute)
                ce_per_token = ce_per_token * valid_mask.float()

                # Weighted CE: use target_probs (teacher distribution with confused handling) as weights
                weighted_ce = (target_probs * ce_per_token).sum(dim=-1)  # [N_laser]

                if weighted_ce.numel() > 0:
                    loss_laser = weighted_ce.mean()
                else:
                    loss_laser = logits.sum() * 0.0

            else:
                raise ValueError(f"Unknown dwal_loss_type: {loss_type}. Choose from: mse, weighted_ce")
        else:
            # No LASER positions in batch, return 0 but keep gradient connection
            loss_laser = logits.sum() * 0.0
            dwal_confused_ratio = 0.0
            dwal_laser_end_max_ratio = 0.0

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_laser=loss_laser,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
        dwal_confused_ratio=dwal_confused_ratio,
        dwal_laser_end_max_ratio=dwal_laser_end_max_ratio,
    )


# ============================================================================
# Time-Aware DWAL: Focusing Mechanism
# ============================================================================
'''
    Time-Aware DWAL (Focusing Mechanism) mode

    Addresses the Shortcut Learning problem in Stage 1 LASER training by introducing
    a dual-weighting mechanism that forces the model to "focus on the present while
    considering the future".

    Two orthogonal weighting dimensions:

    A. Spatial Decay (Micro): Window-level logit bias
       - Applies a decaying bias to target logits within each window
       - Logits_target[k] = Logits_teacher[k] + k * ln(gamma)
       - Closer tokens get higher probability after softmax

    B. Temporal Ramp (Macro): Step-wise loss scaling
       - Loss weight increases linearly across the reasoning chain
       - Weight(t) = alpha + (1-alpha) * (t/T)
       - Early steps: lower weight (exploration allowed)
       - Late steps: higher weight (precision required)

    Hyperparameters:
    - gamma (spatial_decay_gamma): 0.91 (each step reduces logit by ~0.094)
    - alpha (time_weight_init): 0.5 (start weight=0.5, end weight=1.0)
'''
def qwen2_5_mixed_modality_forward_laser_dwal_time_aware(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    laser_tokens: Optional[torch.Tensor] = None,
    laser_tokens_thw: Optional[List[torch.Tensor]] = None,
    laser_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    question_ids: Optional[List] = None,
    original_lengths: Optional[List] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        raise ValueError("pixel_values and pixel_values_videos cannot both be None. Image input is required.")

    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)

        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id

        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_laser = None
    dwal_confused_ratio = None
    dwal_laser_end_max_ratio = None

    if labels is not None:
        logits = logits.float()

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()

        batch_size, seq_len = shift_input_ids.shape

        # ============== Build LASER Region Mask ==============
        laser_start_mask = shift_input_ids == self.config.laser_start_id
        laser_end_mask = shift_input_ids == self.config.laser_end_id

        laser_region_mask = torch.zeros_like(shift_input_ids, dtype=torch.bool)

        for b in range(batch_size):
            start_positions = torch.nonzero(laser_start_mask[b], as_tuple=True)[0]
            end_positions = torch.nonzero(laser_end_mask[b], as_tuple=True)[0]

            for start_pos, end_pos in zip(start_positions, end_positions):
                laser_region_mask[b, start_pos+1:end_pos] = True

        # ============== CE Loss (mask LASER region) ==============
        ce_labels = shift_labels.clone()
        ce_labels[laser_region_mask] = IGNORE_INDEX

        loss_fct = CrossEntropyLoss()
        ce_logits_flat = shift_logits.view(-1, self.config.vocab_size)
        ce_labels_flat = ce_labels.view(-1).to(ce_logits_flat.device)
        loss_ce = loss_fct(ce_logits_flat, ce_labels_flat)

        # ============== Time-Aware DWAL Loss (only LASER region) ==============
        # Get hyperparameters from config
        K_fixed = getattr(self.config, 'dwal_window_size', None)
        tau = getattr(self.config, 'dwal_temperature', 1.0)
        entropy_threshold = getattr(self.config, 'dwal_entropy_threshold', 0.8)
        force_prob = getattr(self.config, 'dwal_force_prob', 0.8)

        # Time-Aware specific hyperparameters
        gamma = getattr(self.config, 'dwal_spatial_decay_gamma', 0.91)
        alpha = getattr(self.config, 'dwal_time_weight_init', 0.5)

        laser_predict_mask = laser_region_mask.clone()
        laser_predict_mask[laser_end_mask] = False

        if laser_predict_mask.any():
            batch_indices, seq_indices = torch.nonzero(laser_predict_mask, as_tuple=True)

            # ============== Build Start/End Positions ==============
            # Track both start and end positions for temporal ramp calculation
            laser_start_positions = torch.zeros_like(seq_indices)
            laser_end_positions = torch.zeros_like(seq_indices)

            for b in range(batch_size):
                b_mask = batch_indices == b
                if b_mask.any():
                    start_pos_in_b = torch.nonzero(laser_start_mask[b], as_tuple=True)[0]
                    end_pos_in_b = torch.nonzero(laser_end_mask[b], as_tuple=True)[0]
                    b_seq_indices = seq_indices[b_mask]

                    for i, (pos_idx, pos) in enumerate(zip(torch.nonzero(b_mask, as_tuple=True)[0], b_seq_indices)):
                        # Find the closest start position < current position
                        valid_starts = start_pos_in_b[start_pos_in_b < pos]
                        if len(valid_starts) > 0:
                            laser_start_positions[pos_idx] = valid_starts[-1]

                        # Find the first end position > current position
                        valid_ends = end_pos_in_b[end_pos_in_b > pos]
                        if len(valid_ends) > 0:
                            laser_end_positions[pos_idx] = valid_ends[0]
                        else:
                            laser_end_positions[pos_idx] = seq_len - 1

            # ============== Compute Temporal Weights ==============
            # t: position offset from laser_start+1 (first reasoning token)
            # T: total length of LASER region (excluding start/end markers)
            t_pos = seq_indices - laser_start_positions - 1
            T_total = laser_end_positions - laser_start_positions - 1
            T_total = T_total.clamp(min=1)  # Avoid division by zero

            # Temporal weight: alpha + (1-alpha) * (t/T)
            time_weight = alpha + (1 - alpha) * (t_pos.float() / T_total.float())  # [N_laser]

            # ============== Build Window ==============
            per_position_window_size = laser_end_positions - seq_indices + 1
            if K_fixed is not None:
                K = K_fixed
            else:
                K = int(per_position_window_size.max().item())

            if K <= 0:
                K = 1

            laser_logits = shift_logits[batch_indices, seq_indices]

            offsets = torch.arange(0, K, device=input_ids.device)
            window_positions = seq_indices.unsqueeze(-1) + offsets

            valid_mask = (window_positions < seq_len) & (window_positions < laser_end_positions.unsqueeze(-1))
            window_positions_clamped = window_positions.clamp(max=seq_len - 1)

            window_token_ids = shift_input_ids[batch_indices.unsqueeze(-1).expand_as(window_positions_clamped),
                                                window_positions_clamped]

            window_logits = laser_logits.gather(dim=-1, index=window_token_ids)
            window_logits = window_logits.clamp(min=-1e4, max=1e4)

            # ============== Build Teacher Distribution with Spatial Decay ==============
            with torch.no_grad():
                teacher_logits = window_logits.detach().clone()

                # [NEW] Apply Spatial Decay Bias: Logits[k] += k * ln(gamma)
                # This penalizes tokens further in the window
                spatial_bias = torch.arange(K, device=teacher_logits.device).float() * math.log(gamma)
                teacher_logits = teacher_logits + spatial_bias.unsqueeze(0)  # [N_laser, K]

                # Mask invalid positions
                teacher_logits[~valid_mask] = float('-inf')

                # Temperature softmax
                teacher_probs = F.softmax(teacher_logits / tau, dim=-1)

                # Handle NaN from all-inf rows
                nan_mask = torch.isnan(teacher_probs).any(dim=-1)
                if nan_mask.any():
                    teacher_probs[nan_mask] = 1.0 / K

                # Hard target (one-hot at position 0)
                hard_target = torch.zeros_like(teacher_probs)
                hard_target[:, 0] = 1.0

                # Valid count for entropy computation
                valid_count = valid_mask.sum(dim=-1).float()
                single_token_mask = (valid_count == 1)

                eps = 1e-8
                entropy = -(teacher_probs * torch.log(teacher_probs + eps)).sum(dim=-1)
                safe_valid_count = valid_count.clamp(min=2)
                normalized_entropy = entropy / torch.log(safe_valid_count)
                normalized_entropy = normalized_entropy.clamp(min=0.0, max=1.0)

                is_confused = (normalized_entropy > entropy_threshold).float().unsqueeze(-1)

                # Metrics for logging
                multi_token_positions = ~single_token_mask
                if multi_token_positions.any():
                    dwal_confused_ratio = is_confused.squeeze(-1)[multi_token_positions].mean().item()
                else:
                    dwal_confused_ratio = 0.0

                if multi_token_positions.any():
                    argmax_indices = teacher_probs.argmax(dim=-1)
                    last_valid_indices = (valid_count - 1).long()
                    is_laser_end_max = (argmax_indices == last_valid_indices)
                    dwal_laser_end_max_ratio = is_laser_end_max[multi_token_positions].float().mean().item()
                else:
                    dwal_laser_end_max_ratio = 0.0

                mixed_probs = is_confused * (force_prob * hard_target + (1 - force_prob) * teacher_probs) + \
                              (1 - is_confused) * teacher_probs

                target_probs = torch.where(
                    single_token_mask.unsqueeze(-1),
                    hard_target,
                    mixed_probs
                )

            # ============== DWAL Loss with Temporal Ramp ==============
            loss_type = getattr(self.config, 'dwal_loss_type', 'weighted_ce')

            if loss_type == "mse":
                # MSE Loss (no temporal ramp for MSE as per user request)
                student_probs = F.softmax(window_logits / tau, dim=-1)
                valid_student_probs = student_probs[valid_mask].float()
                valid_target_probs = target_probs[valid_mask].float()

                if valid_student_probs.numel() > 0:
                    loss_laser = F.mse_loss(valid_student_probs, valid_target_probs)
                else:
                    loss_laser = logits.sum() * 0.0

            elif loss_type == "weighted_ce":
                # Weighted CE Loss with Temporal Ramp
                student_log_probs = F.log_softmax(laser_logits, dim=-1)
                window_log_probs = student_log_probs.gather(dim=-1, index=window_token_ids)

                ce_per_token = -window_log_probs
                ce_per_token = ce_per_token * valid_mask.float()

                weighted_ce = (target_probs * ce_per_token).sum(dim=-1)  # [N_laser]

                # [NEW] Apply Temporal Ramp
                weighted_ce = weighted_ce * time_weight  # [N_laser]

                if weighted_ce.numel() > 0:
                    loss_laser = weighted_ce.mean()
                else:
                    loss_laser = logits.sum() * 0.0

            else:
                raise ValueError(f"Unknown dwal_loss_type: {loss_type}. Choose from: mse, weighted_ce")
        else:
            loss_laser = logits.sum() * 0.0
            dwal_confused_ratio = 0.0
            dwal_laser_end_max_ratio = 0.0

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_laser=loss_laser,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
        dwal_confused_ratio=dwal_confused_ratio,
        dwal_laser_end_max_ratio=dwal_laser_end_max_ratio,
    )

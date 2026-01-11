"""
    Implementation of LASER models based on Qwen-2.5-VL series
"""
import math
import torch.nn as nn
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.configuration_utils import PretrainedConfig

from torch.nn import CrossEntropyLoss, MSELoss, L1Loss

import inspect
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.nn import functional as F


from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    GenerationMode,
)

from transformers.generation.logits_process import (
    LogitsProcessorList,
)


from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    logger,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerateEncoderDecoderOutput, 
    GenerateDecoderOnlyOutput,
)

from transformers.generation.streamers import BaseStreamer
from transformers.cache_utils import Cache
from transformers import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled  
from transformers.integrations.fsdp import is_fsdp_managed_module


class QwenWithLaser(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _apply_relative_noise(
        h: torch.Tensor,
        noise_ratio: float,
        eps: float = 1e-8
    ) -> tuple:
        """
        Apply relative norm perturbation to hidden states.

        Formula: h' = h + scale * ε, where scale = α * ||h|| / ||ε||
        This ensures constant SNR regardless of ||h|| magnitude.

        Args:
            h: Hidden state tensor of shape (B, H) or (H,)
            noise_ratio: α, the relative noise strength (e.g., 0.01 for 1%)
            eps: Small value to prevent division by zero

        Returns:
            Tuple of (perturbed_h, cosine_similarity_before_after)
        """
        if noise_ratio <= 0:
            return h, 1.0  # No perturbation, perfect similarity

        # Step 1: Sample base noise direction ε ~ N(0, I)
        epsilon = torch.randn_like(h)

        # Step 2: Compute scaling factor
        # For batched input, compute per-sample norms
        if h.dim() == 2:
            h_norm = torch.norm(h, dim=-1, keepdim=True)  # (B, 1)
            epsilon_norm = torch.norm(epsilon, dim=-1, keepdim=True)  # (B, 1)
        else:
            h_norm = torch.norm(h)
            epsilon_norm = torch.norm(epsilon)

        scale = noise_ratio * h_norm / (epsilon_norm + eps)

        # Step 3: Inject perturbation
        h_perturbed = h + scale * epsilon

        # Step 4: Compute cosine similarity for logging
        if h.dim() == 2:
            cos_sim = torch.nn.functional.cosine_similarity(h, h_perturbed, dim=-1).mean().item()
        else:
            cos_sim = torch.nn.functional.cosine_similarity(
                h.unsqueeze(0), h_perturbed.unsqueeze(0), dim=-1
            ).item()

        return h_perturbed, cos_sim

    # Patch the generation function with laser_generate
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        decoding_strategy:Optional[str]=None,
        criterion: Optional[str]="mse",
        laser_end_threshold: Optional[float]=0.02,
        laser_steps: Optional[List[int]]=None,
        repetition_exit: Optional[bool]=False,
        save_laser_topk: Optional[int]=None,  # None=don't save, int=save top-k tokens and probabilities
        **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
        """
            Patching the generation function for LASER
        """

        # Params in 
        if decoding_strategy is None and hasattr(generation_config,'decoding_strategy'):
            decoding_strategy = generation_config.decoding_strategy
        if criterion is None and hasattr(generation_config,'criterion'):
            criterion = generation_config.criterion
        if laser_end_threshold is None and hasattr(generation_config,'laser_end_threshold'):
            laser_end_threshold = generation_config.laser_end_threshold
        if laser_steps is None and hasattr(generation_config,'laser_steps'):
            laser_steps = generation_config.laser_steps

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        # self._validate_model_kwargs()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        # generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        '''
            No other modes
            LASER decoding only
        '''
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        '''
            ._sample is patched by _laser_decoding
        '''
        if decoding_strategy == "steps":
            result = self._laser_deocding_by_steps(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                laser_steps = laser_steps,
                repetition_exit=repetition_exit,
                **model_kwargs,)
        else:
            # Vanilla decoding
            # enters LASER if it sees start
            # exits LASER if it sees end
            result = self._laser_deocding(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                criterion = criterion,
                laser_end_threshold= laser_end_threshold,
                laser_steps=laser_steps,
                repetition_exit=repetition_exit,
                save_laser_topk=save_laser_topk,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
    
    # LASER docoding logic
    def _laser_deocding(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            laser_steps: Optional[int] = None,
            repetition_exit: bool = False,
            save_laser_topk: Optional[int] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                    `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

            # CRITICAL FIX: Clear rope_deltas at the start of generation to force recomputation
            # This ensures each batch gets fresh position_ids calculation
            self.model.rope_deltas = None
            # Also clear position_ids from model_kwargs if it exists
            if 'position_ids' in model_kwargs:
                del model_kwargs['position_ids']

            # Transformer version compatibility
            try:
                model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
            except:
                model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

            model_forward = self.__call__
            if isinstance(model_kwargs.get("past_key_values"), Cache):
                is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
                if getattr(self, "hf_quantizer", None) is not None:
                    is_compileable &= self.hf_quantizer.is_compileable
                is_compileable = is_compileable and not generation_config.disable_compile
                if is_compileable and (
                    self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
                ):
                    os.environ["TOKENIZERS_PARALLELISM"] = "0"
                    model_forward = self.get_compiled_call(generation_config.compile_config)

            if generation_config.prefill_chunk_size is not None:
                model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
                is_prefill = False
            else:
                is_prefill = True

            laser_mode_switch = torch.zeros(batch_size,dtype=torch.bool,device=input_ids.device)  # switch gate for laser mode
            input_is_laser_token = torch.zeros(batch_size,dtype=torch.bool,device=input_ids.device)  # whether current input belongs to LASER region
            last_position_hidden_state = None

            # Fallback max steps for dynamic mode (in case model never generates <|laser_end|>)
            # Use laser_steps parameter if provided, otherwise default to 16
            if isinstance(laser_steps, list):
                LASER_MAX_STEPS = laser_steps[0]
            elif laser_steps is not None:
                LASER_MAX_STEPS = laser_steps
            else:
                LASER_MAX_STEPS = 16
            laser_step_counter = torch.zeros(batch_size, dtype=torch.int, device=input_ids.device)

            # Initialize LASER states buffer for storing all hidden states in LASER region
            max_new_tokens = generation_config.max_length - cur_len
            H = self.config.hidden_size
            laser_states_buffer = torch.zeros(batch_size, max_new_tokens, H, device=input_ids.device, dtype=self.dtype)
            laser_mask_buffer = torch.zeros(batch_size, max_new_tokens, dtype=torch.bool, device=input_ids.device)
            generation_step = 0  # Track generation steps (relative to start of generation)

            # Forced answer pending: when LASER timeout forces <|laser_end|>, next step should force <answer>
            # This prevents the model from continuing to generate LASER tokens after forced exit
            forced_answer_pending = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            answer_start_id = getattr(self.config, 'answer_start_id', None)

            # Relative Norm Perturbation: get noise ratio from generation_config
            laser_noise_ratio = getattr(generation_config, 'laser_noise_ratio', 0.0)
            laser_noise_cos_sims = []  # Store cosine similarities for logging

            # Top-k logging for LASER analysis (only when save_laser_topk is set)
            laser_topk_records = [] if save_laser_topk is not None else None

            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # Use input_is_laser_token to decide whether current input should use hidden state
                # This is the key: LASER region tokens always use hidden state, not token embedding
                model_inputs.update({"laser_mode_switch": input_is_laser_token})
                model_inputs.update({"last_position_hidden_state": last_position_hidden_state})
                if is_prefill:
                    outputs = self(**model_inputs, return_dict=True)
                    is_prefill = False
                else:
                    outputs = model_forward(**model_inputs, return_dict=True)

                # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue

                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # Force first token to be <|laser_start|> (skip sampling randomness)
                if generation_step == 0:
                    next_tokens = torch.full_like(next_tokens, self.config.laser_start_id)

                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # Step 1: If previous step triggered forced exit, this step forces <answer> output
                # This ensures the model transitions to answer mode after forced LASER exit
                if answer_start_id is not None and forced_answer_pending.any():
                    next_tokens = torch.where(
                        forced_answer_pending,
                        torch.full_like(next_tokens, answer_start_id),
                        next_tokens
                    )
                    forced_answer_pending = torch.zeros_like(forced_answer_pending)  # Reset after use

                '''
                    LASER reasoning mode switches:

                    When next token is <|laser_start|>, we still need to pass its token id through decoding
                    When last token is <|laser_start|>, we will start passing hidden states (enter laser mode)

                    When next token is <|laser_end|>, we will stop passing hidden states (end laser mode)

                    Key principle: LASER region tokens always use hidden state (h), never token embedding.
                    - When predicting <|laser_end|>, current step's h is the last LASER hidden state
                    - <|laser_end|> itself uses its token embedding (it's outside LASER region)
                '''
                last_tokens = input_ids[:,-1]
                laser_start_switch = (last_tokens == self.config.laser_start_id).to(device=input_ids.device)

                # Track steps in LASER mode
                # Reset counter when entering LASER, increment when in LASER mode
                just_entered = (~laser_mode_switch) & laser_start_switch
                laser_step_counter = torch.where(just_entered, torch.zeros_like(laser_step_counter), laser_step_counter)
                laser_step_counter = laser_step_counter + laser_mode_switch.int()

                # Repetition exit: detect 3 consecutive identical tokens in LASER region
                # When detected, force exit LASER to prevent infinite loops
                repetition_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
                if repetition_exit and laser_mode_switch.any() and input_ids.shape[1] >= 2:
                    last_2_tokens = input_ids[:, -2:]  # [batch, 2]
                    # Check if current token equals both of the last 2 tokens (3 consecutive identical)
                    repetition_mask = (
                        laser_mode_switch &
                        (next_tokens == last_2_tokens[:, -1]) &
                        (next_tokens == last_2_tokens[:, -2])
                    )

                # Check timeout
                laser_timeout = laser_step_counter >= LASER_MAX_STEPS
                timeout_mask = laser_timeout & laser_mode_switch

                # Combine timeout and repetition exit masks
                force_exit_mask = timeout_mask | repetition_mask

                # CRITICAL: When timeout or repetition, force output <|laser_end|> to maintain correct representation
                # This ensures the last LASER position uses hidden state, and <|laser_end|> uses token embedding
                next_tokens = torch.where(
                    force_exit_mask,
                    torch.full_like(next_tokens, self.config.laser_end_id),
                    next_tokens
                )

                # Step 2: Mark samples that need forced <answer> output in next step
                # This prevents the model from re-entering LASER mode after forced exit
                if answer_start_id is not None:
                    forced_answer_pending = forced_answer_pending | force_exit_mask

                # Now check laser_end_switch after potential forced output
                laser_end_switch = (next_tokens == self.config.laser_end_id).to(device=input_ids.device)

                # Update hidden state
                last_position_hidden_state = outputs.last_position_hidden_state

                # Apply Relative Norm Perturbation for exploration (only in LASER region)
                if laser_noise_ratio > 0 and laser_mode_switch.any():
                    # Apply noise to samples in LASER mode
                    perturbed_h, cos_sim = self._apply_relative_noise(
                        last_position_hidden_state, laser_noise_ratio
                    )
                    # Update only for samples in LASER mode (keep others unchanged)
                    last_position_hidden_state = torch.where(
                        laser_mode_switch.unsqueeze(-1),
                        perturbed_h,
                        last_position_hidden_state
                    )
                    laser_noise_cos_sims.append(cos_sim)

                # Store hidden state in buffer if currently in LASER mode
                # Exclude <|laser_end|> positions from saving (aligned with _laser_deocding_by_steps)
                if laser_mode_switch.any() and generation_step < max_new_tokens:
                    # Check which samples are in LASER mode and current token is not laser_end
                    is_laser_token = (next_tokens != self.config.laser_end_id)
                    save_mask = laser_mode_switch & is_laser_token
                    if save_mask.any():
                        laser_states_buffer[save_mask, generation_step] = last_position_hidden_state[save_mask]
                        laser_mask_buffer[save_mask, generation_step] = True

                # Update laser_mode_switch: exit if generated <|laser_end|>
                # Note: timeout now forces <|laser_end|>, so we only check laser_end_switch
                laser_mode_switch = (laser_mode_switch | laser_start_switch) & (~laser_end_switch)

                # Record top-k tokens for LASER analysis (only when save_laser_topk is set)
                # This records after mode switch update, so laser_mode_switch=True means this step is in LASER region
                # When next_tokens is <|laser_end|>, laser_mode_switch becomes False, so we don't record it
                # IMPORTANT: Use next_token_logits (raw), not next_token_scores (processed by logits_processor)
                if laser_topk_records is not None and laser_mode_switch.any():
                    with torch.no_grad():
                        # Always compute softmax from raw logits for accurate probability distribution
                        probs_for_topk = nn.functional.softmax(next_token_logits, dim=-1)
                        topk_probs, topk_ids = torch.topk(probs_for_topk, k=save_laser_topk, dim=-1)
                        # Only record samples currently in LASER mode
                        for b in range(batch_size):
                            if laser_mode_switch[b]:
                                laser_topk_records.append({
                                    'batch_idx': b,
                                    'laser_step': laser_step_counter[b].item(),
                                    'generation_step': generation_step,
                                    'topk_ids': topk_ids[b].cpu().tolist(),
                                    'topk_probs': topk_probs[b].cpu().tolist(),
                                    'selected_token': next_tokens[b].item(),
                                })

                # Determine if next step's input belongs to LASER region
                # Key insight: if this step is in LASER mode and we did NOT output <|laser_end|>,
                # then the token we output belongs to LASER region, so next input should use hidden state
                # If we output <|laser_end|>, it does NOT belong to LASER region, next input uses token embedding
                input_is_laser_token = laser_mode_switch.clone()

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                generation_step += 1
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                    
                # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                """if laser mode is unfinished, do not stop"""
                unfinished_sequences = (
                    laser_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
                )
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

            if streamer is not None:
                streamer.end()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    output = GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                    # Add LASER states for GRPO training (return_dict_in_generate=True)
                    output.laser_states = laser_states_buffer
                    output.laser_mask = laser_mask_buffer
                    # Add noise cosine similarity stats for logging
                    output.laser_noise_cos_sims = laser_noise_cos_sims
                    # Add top-k records for LASER analysis (if enabled)
                    if laser_topk_records is not None:
                        output.laser_topk_records = laser_topk_records
                    return output
            else:
                # When not using return_dict, store top-k records as model attribute for retrieval
                if laser_topk_records is not None:
                    self._last_laser_topk_records = laser_topk_records
                return input_ids

    # LASER docoding logic
    def _laser_deocding_by_steps(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        laser_steps: List[int],
        repetition_exit: bool = False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # CRITICAL FIX: Clear rope_deltas at the start of generation to force recomputation
        # This ensures each batch gets fresh position_ids calculation
        self.model.rope_deltas = None
        # Also clear position_ids from model_kwargs if it exists
        if 'position_ids' in model_kwargs:
            del model_kwargs['position_ids']

        try:
            model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        except:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        laser_mode_switch = torch.zeros(batch_size,dtype=torch.bool,device=input_ids.device)  # switch gate for laser mode
        last_position_hidden_state = None

        # For repetition_exit: track forced answer state (similar to _laser_deocding)
        forced_answer_pending = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        answer_start_id = getattr(self.config, 'answer_start_id', None)

        # Initialize LASER states buffer for saving during generation
        max_new_tokens = generation_config.max_length - cur_len
        # print(f"[GENERATION_DEBUG] Initializing LASER states buffer:")
        # print(f"[GENERATION_DEBUG]   - cur_len (prompt): {cur_len}")
        # print(f"[GENERATION_DEBUG]   - generation_config.max_length: {generation_config.max_length}")
        # print(f"[GENERATION_DEBUG]   - max_new_tokens (buffer size): {max_new_tokens}")
        # print(f"[GENERATION_DEBUG]   - batch_size: {batch_size}")

        H = self.config.hidden_size
        laser_states_buffer = torch.zeros(batch_size, max_new_tokens, H, device=input_ids.device, dtype=self.dtype)
        laser_mask_buffer = torch.zeros(batch_size, max_new_tokens, dtype=torch.bool, device=input_ids.device)
        generation_step = 0  # Track generation steps

        # Track LASER quotas
        laser_steps_orig = torch.tensor(laser_steps, dtype=torch.int, device=input_ids.device)  # original quota
        laser_remaining_steps = laser_steps_orig.clone()

        # Relative Norm Perturbation: get noise ratio from generation_config
        laser_noise_ratio = getattr(generation_config, 'laser_noise_ratio', 0.0)
        laser_noise_cos_sims = []  # Store cosine similarities for logging

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            
            model_inputs.update({"laser_mode_switch":laser_mode_switch})
            model_inputs.update({"last_position_hidden_state":last_position_hidden_state})
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Handle forced_answer_pending from previous step (for repetition_exit)
            if answer_start_id is not None and forced_answer_pending.any():
                next_tokens = torch.where(
                    forced_answer_pending,
                    torch.full_like(next_tokens, answer_start_id),
                    next_tokens
                )
                forced_answer_pending = torch.zeros_like(forced_answer_pending)  # Reset after use

            '''
                LASER reasoning mode switches:

                When next token is <|laser_start|>, we still need to pass its token id through decoding
                When last token is <|laser_start|>, we will start passing hidden states (enter laser mode)

                During LASER, keep passing hidden_states until quota uses up
            '''
            last_tokens = input_ids[:,-1]
            laser_start_switch = (last_tokens == self.config.laser_start_id).to(device=input_ids.device)

            # Repetition exit: detect 3 consecutive identical tokens in LASER region
            repetition_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            if repetition_exit and laser_mode_switch.any() and input_ids.shape[1] >= 2:
                last_2_tokens = input_ids[:, -2:]  # [batch, 2]
                # Check if current token equals both of the last 2 tokens (3 consecutive identical)
                repetition_mask = (
                    laser_mode_switch &
                    (next_tokens == last_2_tokens[:, -1]) &
                    (next_tokens == last_2_tokens[:, -2])
                )
                # When repetition detected, force exit by setting remaining steps to 0
                # and mark for forced <answer> output in next step
                if repetition_mask.any():
                    laser_remaining_steps = torch.where(repetition_mask, torch.zeros_like(laser_remaining_steps), laser_remaining_steps)
                    if answer_start_id is not None:
                        forced_answer_pending = forced_answer_pending | repetition_mask

            '''
                Goal: laser_mode_switch = laser_mode_switch + laser_start_switch
                the exit is controlled by laser quota now, not <|laser_end|>

            '''
            # Candidate new switch (no end token anymore)
            new_mode_switch = laser_mode_switch | laser_start_switch

            # Detect entry vs continuation
            just_entered = (~laser_mode_switch) & new_mode_switch
            still_in     = laser_mode_switch & new_mode_switch

            # Reset quota when entering
            laser_remaining_steps = torch.where(just_entered, laser_steps_orig, laser_remaining_steps)

            # Decrement quota only if we were already inside before this step
            laser_remaining_steps = laser_remaining_steps - laser_mode_switch.long()

            # Exit if quota used up
            laser_mode_switch = new_mode_switch & (laser_remaining_steps > 0)


            last_position_hidden_state = outputs.last_position_hidden_state

            # Apply Relative Norm Perturbation for exploration (only in LASER region)
            if laser_noise_ratio > 0 and laser_mode_switch.any():
                # Apply noise to samples in LASER mode
                perturbed_h, cos_sim = self._apply_relative_noise(
                    last_position_hidden_state, laser_noise_ratio
                )
                # Update only for samples in LASER mode (keep others unchanged)
                last_position_hidden_state = torch.where(
                    laser_mode_switch.unsqueeze(-1),
                    perturbed_h,
                    last_position_hidden_state
                )
                laser_noise_cos_sims.append(cos_sim)

            # Save LASER states during generation (for GRPO training)
            if laser_mode_switch.any() and generation_step < max_new_tokens:
                # Check which samples are in LASER mode and current token is not laser_end
                is_laser_token = (next_tokens != self.config.laser_end_id)
                save_mask = laser_mode_switch & is_laser_token

                if save_mask.any():
                    laser_states_buffer[save_mask, generation_step] = last_position_hidden_state[save_mask]
                    laser_mask_buffer[save_mask, generation_step] = True

            generation_step += 1

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            """if laser mode is unfinished, do not stop"""
            unfinished_sequences = (
                laser_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
            )

            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                output = GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
                # Add LASER states as dynamic attributes (for GRPO training)
                output.laser_states = laser_states_buffer
                output.laser_mask = laser_mask_buffer
                # Add noise cosine similarity stats for logging
                output.laser_noise_cos_sims = laser_noise_cos_sims
                return output
        else:
            return input_ids

    # @classmethod
    # def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
    #     """
    #     Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
    #     [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

    #     Args:
    #         model_config (`PretrainedConfig`):
    #             The model config that will be used to instantiate the generation config.

    #     Returns:
    #         [`GenerationConfig`]: The configuration object instantiated from those parameters.
    #     """
    #     config_dict = model_config.to_dict()
    #     config_dict.pop("_from_model_config", None)

    #     # Removes all `None` from the model config dict -- this lets the generation config defaults to take hold
    #     config_dict = {key: value for key, value in config_dict.items() if value is not None}

    #     generation_config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

    #     # Special case: some models have generation attributes set in the decoder. Use them if still unset in the
    #     # generation config (which in turn is defined from the outer attributes of model config).
    #     decoder_config = model_config.get_text_config(decoder=True)
    #     if decoder_config is not model_config:
    #         default_generation_config = GenerationConfig()
    #         decoder_config_dict = decoder_config.to_dict() if isinstance(decoder_config, PretrainedConfig) else decoder_config

    #         for attr in generation_config.to_dict().keys():
    #             is_unset = getattr(generation_config, attr) == getattr(default_generation_config, attr)
    #             if attr in decoder_config_dict and is_unset:
    #                 setattr(generation_config, attr, decoder_config_dict[attr])

    #     # If any `output_...` flag is set to `True`, we ensure `return_dict_in_generate` is set to `True`.
    #     if generation_config.return_dict_in_generate is False:
    #         if any(
    #             getattr(generation_config, extra_output_flag, False)
    #             for extra_output_flag in generation_config.extra_output_flags
    #         ):
    #             generation_config.return_dict_in_generate = True

    #     # Hash to detect whether the instance was modified
    #     generation_config._original_object_hash = hash(generation_config)
    #     return generation_config


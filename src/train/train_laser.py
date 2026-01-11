import sys
import os
import json

import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser

from src.model.qwen_laser_model import QwenWithLaser
from src.trainer import QwenLaserSFTTrainer
from src.dataset import make_dynamic_batch_data_module_laser
from src.params import DataArguments, ModelArguments, TrainingArguments

from src.train.train_utils import safe_save_model_for_hf_trainer
from src.train.monkey_patch_forward_laser import replace_qwen2_5_with_mixed_modality_forward_laser

from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None


def get_resume_global_step(checkpoint_path: str) -> int:
    """
    Read global_step from checkpoint.

    Args:
        checkpoint_path: checkpoint directory path

    Returns:
        global_step (int), returns 0 if reading fails
    """
    if not checkpoint_path:
        return 0

    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        print(f"[Resume] trainer_state.json not found in {checkpoint_path}")
        return 0

    try:
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
        global_step = state.get('global_step', 0)
        print(f"[Resume] Read global_step={global_step} from {trainer_state_path}")
        return global_step
    except Exception as e:
        print(f"[Resume] Failed to read trainer_state.json: {e}")
        return 0

# For debugging only Plese comment this during training
# torch.autograd.set_detect_anomaly(True)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with laser
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        model_pth = training_args.checkpoint_name
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)

    # DWAL config
    config.dwal_window_size = training_args.dwal_window_size
    config.dwal_temperature = training_args.dwal_temperature
    config.dwal_entropy_threshold = training_args.dwal_entropy_threshold
    config.dwal_force_prob = training_args.dwal_force_prob
    config.dwal_loss_type = training_args.dwal_loss_type

    # Time-Aware DWAL config
    config.dwal_spatial_decay_gamma = training_args.dwal_spatial_decay_gamma
    config.dwal_time_weight_init = training_args.dwal_time_weight_init

    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:
        # Patch the forward function
        replace_qwen2_5_with_mixed_modality_forward_laser(
            dwal=training_args.dwal,
            dwal_time_aware=training_args.dwal_time_aware
        )

        model = QwenWithLaser.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        ''' Patch the patch-emb with fp32; Avoid edge-case numerical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    else:
        raise("Unsupported model type. At this moment, we only support Qwen2.5LM-based Qwen2.5VL series and InternVL3 series.")

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)

    processor.tokenizer.add_tokens("<|laser_start|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|laser|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|laser_end|>",special_tokens=True)

    laser_id = processor.tokenizer.convert_tokens_to_ids("<|laser|>")
    laser_start_id = processor.tokenizer.convert_tokens_to_ids("<|laser_start|>")
    laser_end_id = processor.tokenizer.convert_tokens_to_ids("<|laser_end|>")

    model.config.laser_id = laser_id
    model.config.laser_start_id = laser_start_id
    model.config.laser_end_id = laser_end_id


    # there are some dummy tokens in newer hf version
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # configure laser loss type
    model.config.loss_laser_fct = training_args.loss_laser_fct


    '''
        Data module configurations
        use data packing for faster training due to the random input lengths of LASER
    '''
    # Get resume_step (for dynamic batch to skip already trained data)
    resume_step = 0
    if training_args.resume_from_checkpoint:
        resume_step = get_resume_global_step(training_args.resume_from_checkpoint)
        rank0_print(f"[Resume] Will skip batches corresponding to {resume_step} steps")

    # Dynamic batch mode: batch_size × max_len <= token_cap
    training_args.per_device_train_batch_size = 1  # DataLoader batch_size=1, actual batching done in DynamicBatchDataset
    data_module, total_data_len = make_dynamic_batch_data_module_laser(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        training_args=training_args,
        resume_step=resume_step,
    )
    if not training_args.max_steps:
        # Estimate steps based on average batch size
        # Actual number will vary due to dynamic batching
        estimated_avg_batch_size = training_args.dynamic_batch_max_bs // 2
        training_args.max_steps = total_data_len // (training_args.gradient_accumulation_steps
                                                     * training_args.world_size
                                                     * estimated_avg_batch_size)
    # Use modified dataloader for IterableDataset
    replace_train_dataloader()
    rank0_print(f"[Dynamic Batch] token_cap={training_args.dynamic_batch_token_cap}, "
               f"max_bs={training_args.dynamic_batch_max_bs}, "
               f"buffer_size={training_args.dynamic_batch_buffer_size}")
    
    trainer = QwenLaserSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    # Resume from checkpoint if specified
    # HF Trainer will handle loading optimizer, scheduler, and training state
    resume_from_checkpoint = None
    if training_args.resume_from_checkpoint:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        rank0_print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()

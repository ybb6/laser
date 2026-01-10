#!/bin/bash
# DWAL (Dynamic Windowed Alignment Loss) Training Script

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Model configs
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export WANDB_PROJECT="LASER-Qwen25-VL-7B-DWAL"

# Data Config
DATA_PATH="data/laser_data/meta_data_reasoning_chain_full_filter.json"
LENGTHS_PATH="data/laser_data/sample_lengths_full_filter.json"  # Precomputed lengths, 0 means skip
RANDOM_SEED=42

# ============================================================
# Dynamic Batch Configuration
# ============================================================
# - Groups samples by length to minimize padding
# - Uses rectangular memory constraint: batch_size × max_len <= TOKEN_CAP
# - Requires precomputed lengths file (LENGTHS_PATH)
DYNAMIC_TOKEN_CAP=8192      # Memory constraint: batch_size × max_len <= this
DYNAMIC_MAX_BS=16           # Max samples per batch

# ============================================================

# General training params
MAX_STEPS=500
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=8

# LLM-related params
LR=1e-5

# Loss weights
LAMBDA_LASER=1.0  # Weight for DWAL loss

# DWAL Hyperparameters

DWAL_TEMPERATURE=1.0
DWAL_ENTROPY_THRESHOLD=0.6
DWAL_FORCE_PROB=0.8
DWAL_LOSS_TYPE="weighted_ce"  # mse or weighted_ce

# Vision token config
MAX_TOKEN=8192
MIN_TOKEN=128

RUN_NAME="DWAL_Kdyn_T${DWAL_TEMPERATURE}_Eth${DWAL_ENTROPY_THRESHOLD}_Fp${DWAL_FORCE_PROB}"
OUTPUT_DIR="checkpoints/"

# Resume from checkpoint (set to checkpoint path or "auto" for latest, empty to start fresh)
RESUME_CHECKPOINT=""

deepspeed src/train/train_laser.py \
    --run_name "$RUN_NAME" \
    --dwal True \
    --dwal_temperature $DWAL_TEMPERATURE \
    --dwal_entropy_threshold $DWAL_ENTROPY_THRESHOLD \
    --dwal_force_prob $DWAL_FORCE_PROB \
    --dwal_loss_type $DWAL_LOSS_TYPE \
    --deepspeed scripts/zero3_offload_save_safe.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_laser_lambda $LAMBDA_LASER \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --adam_epsilon 1e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --disable_tqdm False \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --dataloader_num_workers 2 \
    --dynamic_batch_token_cap $DYNAMIC_TOKEN_CAP \
    --dynamic_batch_max_bs $DYNAMIC_MAX_BS \
    --random_seed $RANDOM_SEED \
    --lengths_path "$LENGTHS_PATH" \
    --stop_on_nan True \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --ignore_data_skip False

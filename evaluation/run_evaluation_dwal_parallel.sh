#!/bin/bash

# DWAL Model Evaluation Script - PARALLEL VERSION
# Evaluates DWAL checkpoints using multiple GPUs in parallel

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ============================================
# Configuration
# ============================================
# Checkpoint to evaluate (relative path from project root)
CHECKPOINT="checkpoints/checkpoint-500"

# Benchmarks to run (blink vstar mmvp realworldqa mmstar seedbench2plus hallusionbench mme hrbench muirbench visulogic)
BENCHMARKS="blink mmvp mmstar seedbench2plus hallusionbench hrbench"

# BLINK configs to evaluate (only used when BENCHMARKS contains "blink")
# Options:
#   - "all" : All 14 configs (full benchmark)
#   - Specific configs: Art_Style, Counting, Forensic_Detection, Functional_Correspondence,
#                       IQ_Test, Jigsaw, Multi-view_Reasoning, Object_Localization,
#                       Relative_Depth, Relative_Reflectance, Semantic_Correspondence,
#                       Spatial_Relation, Visual_Correspondence, Visual_Similarity
#   - Leave empty for default 5 configs: Counting, IQ_Test, Jigsaw, Relative_Reflectance, Spatial_Relation
# Note: BLINK test split answers are hidden; only val split can be evaluated locally
BLINK_CONFIGS="all"  # e.g., "all" or "Counting IQ_Test" or leave empty for default

# LASER inference steps (used for steps/latent strategy)
STEPS="8"

# Decoding strategy: steps, latent, or dynamic (exit on <|laser_end|> token with forced exit at max steps)
STRATEGY="dynamic"

# Maximum LASER steps before forced exit (only used for 'dynamic' strategy)
# When reaching this limit, model will force output <|laser_end|> and then <answer>
MAX_LASER_STEPS="8"

# Number of GPUs to use
NUM_GPUS=8  # Change this to your desired number of GPUs (4 or 8)

# Random seed for data sharding
SEED=42

# Output directory for results
OUTPUT_DIR="evaluation/results_dwal"

echo "=========================================="
echo "DWAL Model Evaluation - PARALLEL"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Benchmarks: $BENCHMARKS"
echo "BLINK Configs: ${BLINK_CONFIGS:-default (5 configs)}"
echo "LASER Steps: $STEPS"
echo "Strategy: $STRATEGY"
echo "Max LASER Steps (forced exit): $MAX_LASER_STEPS"
echo "Number of GPUs: $NUM_GPUS"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch parallel evaluation processes
echo ""
echo "Launching $NUM_GPUS parallel processes..."
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    LOG_FILE="evaluation/logs/dwal_eval_rank_${GPU_ID}.log"
    mkdir -p "evaluation/logs"

    echo "  [GPU $GPU_ID] Starting evaluation (log: $LOG_FILE)..."

    # Build blink_configs argument if set
    BLINK_CONFIGS_ARG=""
    if [ -n "$BLINK_CONFIGS" ]; then
        BLINK_CONFIGS_ARG="--blink_configs $BLINK_CONFIGS"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluation/evaluation_local.py \
        --checkpoint $CHECKPOINT \
        --benchmark $BENCHMARKS \
        --steps $STEPS \
        --strategy "$STRATEGY" \
        --max_laser_steps $MAX_LASER_STEPS \
        --rank $GPU_ID \
        --world_size $NUM_GPUS \
        --output_dir "$OUTPUT_DIR" \
        $BLINK_CONFIGS_ARG \
        > "$LOG_FILE" 2>&1 &

    # Store PID
    PIDS[$GPU_ID]=$!
done

echo ""
echo "All processes launched! Waiting for completion..."
echo "Monitor progress with: tail -f evaluation/logs/dwal_eval_rank_*.log"
echo ""

# Wait for all processes to complete
FAILED=0
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    PID=${PIDS[$GPU_ID]}
    echo "Waiting for GPU $GPU_ID (PID: $PID)..."

    if wait $PID; then
        echo "  ✓ GPU $GPU_ID completed successfully"
    else
        echo "  ✗ GPU $GPU_ID failed (exit code: $?)"
        FAILED=1
    fi
done

echo ""
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "All evaluations complete! Merging results..."
    echo "=========================================="
    echo ""

    # Merge results for each checkpoint and benchmark
    for CKPT in $CHECKPOINT; do
        for BENCH in $BENCHMARKS; do
            echo "Merging results for checkpoint: $CKPT, benchmark: $BENCH"

            # Build max_laser_steps argument for dynamic strategy
            MAX_LASER_STEPS_ARG=""
            if [ "$STRATEGY" = "dynamic" ]; then
                MAX_LASER_STEPS_ARG="--max_laser_steps $MAX_LASER_STEPS"
            fi

            python evaluation/merge_stage1_results.py \
                --benchmark "$BENCH" \
                --checkpoint "$CKPT" \
                --strategy "$STRATEGY" \
                --num_ranks $NUM_GPUS \
                --steps $STEPS \
                --output_dir "$OUTPUT_DIR" \
                $MAX_LASER_STEPS_ARG

            if [ $? -ne 0 ]; then
                echo "WARNING: Failed to merge results for $CKPT/$BENCH"
                FAILED=1
            fi
        done
    done

    if [ $FAILED -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "SUCCESS! All results merged"
        echo "=========================================="
        echo ""
        echo "Results are in: $OUTPUT_DIR"
        echo ""
        echo "Clean up rank files with:"
        echo "  find $OUTPUT_DIR -name '*_rank*.json' -delete"
    else
        echo ""
        echo "=========================================="
        echo "WARNING: Some merges failed!"
        echo "=========================================="
    fi
else
    echo "ERROR: Some evaluation processes failed!"
    echo "Check logs in: evaluation/logs/dwal_eval_rank_*.log"
    echo "=========================================="
    exit 1
fi

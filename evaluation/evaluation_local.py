"""
Simplified Local Evaluation Script for LASER Models
Modified to work with local checkpoints without cloud storage dependencies
"""

import sys
import os
import argparse

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import torch
from datasets import load_dataset
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import string
import csv
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.qwen_laser_model import QwenWithLaser
from src.train.monkey_patch_forward_laser import replace_qwen2_5_with_mixed_modality_forward_laser

# Import evaluation functions from bench_func
from bench_func import (
    evaluate_blink, evaluate_mmvp, evaluate_mmstar,
    evaluate_seedbench2plus, evaluate_hallusionbench, evaluate_hrbench,
    accuracy_reward_yesno
)

# ==== Configuration ====

# Default checkpoint paths (override with --checkpoint argument)
DEFAULT_CHKPT_PATHS = []

# LASER inference steps to test
DEFAULT_STEP_LIST = [4, 8, 16]

# Decoding strategy: "steps" or "latent"
DEFAULT_DECODING_STRATEGY = "steps"

# BLINK benchmark configs
# All 14 configs available in BLINK-Benchmark/BLINK (val split has answers, test split is hidden)
ALL_BLINK_CONFIGS = [
    'Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence',
    'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
    'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
    'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity'
]
# Default subset (original 5 configs used in early experiments)
DEFAULT_BLINK_CONFIGS = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']

# Dataset configuration
# Set enabled to False to skip that benchmark
DATASET_CONFIG = {
    'blink': {
        "enabled": True,  # Auto-downloads from HuggingFace
        "output_dir": os.path.join(PROJECT_ROOT, "evaluation/results/blink"),
    },
    'MMVP': {
        "enabled": True,  # Auto-downloads from HuggingFace (MMVP/MMVP)
        "output_dir": os.path.join(PROJECT_ROOT, "evaluation/results/mmvp"),
    },
    'mmstar': {
        "enabled": True,  # Lin-Chen/MMStar
        "output_dir": os.path.join(PROJECT_ROOT, "evaluation/results/mmstar"),
    },
    'seedbench2plus': {
        "enabled": True,  # AILab-CVC/SEED-Bench-2-plus
        "output_dir": os.path.join(PROJECT_ROOT, "evaluation/results/seedbench2plus"),
    },
    'hallusionbench': {
        "enabled": True,  # lmms-lab/HallusionBench
        "output_dir": os.path.join(PROJECT_ROOT, "evaluation/results/hallusionbench"),
    },
    'hrbench': {
        "enabled": True,  # DreamMr/HR-Bench (4K only)
        "output_dir": os.path.join(PROJECT_ROOT, "evaluation/results/hrbench"),
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Core utilities ====

def create_messages(img_path, question):
    """Create message format for the model"""
    if not isinstance(img_path, list):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
        vision_content = []
        for ip in img_path:
            vision_content.append({"type": "image", "image": ip})
        vision_content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": vision_content}]
    return messages

def load_model_and_processor(chkpt_pth):
    """Load model and processor from checkpoint"""
    print(f"Loading model from: {chkpt_pth}")

    # Extract run name from path (include parent dir to distinguish versions)
    parent_dir = os.path.basename(os.path.dirname(chkpt_pth))
    ckpt_name = os.path.basename(chkpt_pth)
    run_name = f"{parent_dir}_{ckpt_name}"

    # Load config
    config = AutoConfig.from_pretrained(chkpt_pth, trust_remote_code=True)

    # Patch forward function for DWAL inference
    replace_qwen2_5_with_mixed_modality_forward_laser(dwal=True)

    # Check if this is a DeepSpeed checkpoint (no pytorch_model.bin/safetensors)
    has_hf_weights = (
        os.path.exists(os.path.join(chkpt_pth, 'pytorch_model.bin')) or
        os.path.exists(os.path.join(chkpt_pth, 'model.safetensors')) or
        any(f.startswith('pytorch_model-') for f in os.listdir(chkpt_pth) if f.endswith('.bin')) or
        any(f.startswith('model-') for f in os.listdir(chkpt_pth) if f.endswith('.safetensors'))
    )

    if has_hf_weights:
        # Standard HuggingFace format
        model = QwenWithLaser.from_pretrained(
            chkpt_pth,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        # DeepSpeed ZeRO-2 format: use official DeepSpeed utility
        import glob
        global_step_dirs = sorted(glob.glob(os.path.join(chkpt_pth, 'global_step*')))
        if not global_step_dirs:
            raise FileNotFoundError(f"No pytorch_model.bin or global_step* found in {chkpt_pth}")

        print(f"Loading from DeepSpeed checkpoint: {chkpt_pth}")

        # Use DeepSpeed official utility to extract fp32 state dict
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(chkpt_pth)

        # Create model and load weights
        model = QwenWithLaser(config)
        model.load_state_dict(state_dict)
        model = model.to(dtype=torch.bfloat16, device='cuda')
        print(f"Loaded {len(state_dict)} parameters from DeepSpeed checkpoint")

    # Load processor
    processor = AutoProcessor.from_pretrained(chkpt_pth, trust_remote_code=True)

    # Set answer_start_id for forced exit mechanism (dynamic decoding)
    # When LASER timeout, force output <|laser_end|> then <answer>
    answer_token_ids = processor.tokenizer.encode("<answer>", add_special_tokens=False)
    if len(answer_token_ids) >= 1:
        model.config.answer_start_id = answer_token_ids[0]  # token 27 = '<'
        print(f"Set answer_start_id = {answer_token_ids[0]} for forced exit mechanism")

    print(f"Model loaded successfully.")
    return model, processor, run_name

def run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps=None, repetition_exit=False, save_laser_topk=None):
    """Run inference on a single example

    Args:
        model: The LASER model
        processor: The processor
        img_path: Image path or PIL Image
        text: Input text/question
        steps: LASER steps (used for 'steps'/'latent' strategy)
        decoding_strategy: 'steps', 'latent', or 'dynamic'
        max_laser_steps: Maximum LASER steps before forced exit (only used for 'dynamic' strategy)
        repetition_exit: Force exit LASER when 3 consecutive identical tokens detected
        save_laser_topk: If set to an integer, save top-k tokens and probs for each LASER step (for analysis)
    """
    # Ensure img_path is a PIL Image or a valid path/URL
    # Handle case where img_path might be a string path that needs to be converted
    if isinstance(img_path, str) and not (img_path.startswith('http') or img_path.startswith('/')):
        # This is a relative path string - convert to PIL Image won't work
        # We need to keep it as is for process_vision_info to handle
        pass

    messages = create_messages(img_path, text)
    text_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Debug: print full formatted prompt
    print("=" * 80)
    print("[DEBUG] Full formatted prompt:")
    print("=" * 80)
    print(text_formatted)
    print("=" * 80)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # For dynamic strategy, use max_laser_steps as the upper limit for forced exit
    # For steps/latent strategy, use steps as the fixed number of LASER steps
    if decoding_strategy == "dynamic":
        laser_steps = [max_laser_steps] if max_laser_steps is not None else [16]  # Default to 16 if not specified
    else:
        laser_steps = [steps]

    # "dynamic" strategy maps to decoding_strategy=None (exit on <|laser_end|> token with max_laser_steps limit)
    actual_decoding_strategy = None if decoding_strategy == "dynamic" else decoding_strategy

    # Determine if we need return_dict_in_generate (required for top-k logging)
    return_dict = save_laser_topk is not None

    with torch.no_grad():
        generated_output = model.generate(
            **inputs,
            max_new_tokens=32,
            decoding_strategy=actual_decoding_strategy,
            laser_steps=laser_steps,
            repetition_exit=repetition_exit,
            save_laser_topk=save_laser_topk,
            return_dict_in_generate=return_dict,
        )

        # Handle both return_dict and non-return_dict cases
        if return_dict:
            generated_ids = generated_output.sequences
            laser_topk_records = getattr(generated_output, 'laser_topk_records', None)
        else:
            generated_ids = generated_output
            laser_topk_records = None

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # Debug: print generated length
        gen_len = len(generated_ids[0]) - len(inputs.input_ids[0])
        import sys
        print(f"[DEBUG] Generated length: {gen_len}", file=sys.stderr)

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

    # Debug: print model output
    print("[DEBUG] Model output:")
    print(output_text[0])
    print("=" * 80)

    # Return with top-k records if requested
    if save_laser_topk is not None:
        return output_text, laser_topk_records
    return output_text

# ==== Evaluation Functions ====

def shard_data(data, rank=0, world_size=1, seed=42):
    """Shard data across multiple processes

    Args:
        data: List of data samples
        rank: Process rank (0-indexed)
        world_size: Total number of processes
        seed: Random seed for shuffling

    Returns:
        Sharded data list for this rank
    """
    if world_size <= 1:
        return data

    # Ensure deterministic sharding
    import random
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Shard data
    total = len(shuffled_data)
    per_rank = total // world_size
    start_idx = rank * per_rank

    if rank == world_size - 1:
        # Last rank gets remaining samples
        end_idx = total
    else:
        end_idx = start_idx + per_rank

    sharded = shuffled_data[start_idx:end_idx]
    print(f"[Rank {rank}/{world_size}] Processing samples {start_idx}-{end_idx} ({len(sharded)}/{total} samples)")

    return sharded

# ==== Main ====

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate LASER model on various benchmarks")

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to checkpoint(s) to evaluate. Can specify multiple checkpoints."
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help=f"LASER inference steps to test (default: {DEFAULT_STEP_LIST})"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["steps", "latent", "dynamic"],
        default=None,
        help=f"Decoding strategy: steps=fixed steps, latent=latent convergence, dynamic=exit on <|laser_end|> token (default: {DEFAULT_DECODING_STRATEGY})"
    )

    parser.add_argument(
        "--max_laser_steps",
        type=int,
        default=None,
        help="Maximum LASER steps before forced exit (only used for 'dynamic' strategy, default: 16)"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        choices=["blink", "vstar", "mmvp", "realworldqa", "mmstar", "seedbench2plus", "hallusionbench", "mme", "hrbench", "vsr_filtered", "cub_filtered", "muirbench", "visulogic", "geometry3k"],
        default=None,
        help="Which benchmarks to run (default: all enabled in config)"
    )

    parser.add_argument(
        "--blink_configs",
        type=str,
        nargs="+",
        default=None,
        help=f"BLINK configs to evaluate. Use 'all' for all 14 configs, or specify individual configs. "
             f"Available: {', '.join(ALL_BLINK_CONFIGS)}. Default: {DEFAULT_BLINK_CONFIGS}"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Process rank for data sharding (default: 0)"
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of processes for data sharding (default: 1)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/results",
        help="Output directory for results (default: evaluation/results)"
    )

    parser.add_argument(
        "--repetition_exit",
        action="store_true",
        default=False,
        help="Force exit LASER region when 3 consecutive identical tokens are detected (prevents repetition loops)"
    )

    parser.add_argument(
        "--save_laser_topk",
        type=int,
        default=None,
        help="Save top-k tokens and probabilities at each LASER step for analysis (default: None, disabled)"
    )

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Update output directories based on args.output_dir
    DATASET_CONFIG['blink']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "blink")
    DATASET_CONFIG['vstar']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "vstar")
    DATASET_CONFIG['MMVP']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "mmvp")
    DATASET_CONFIG['realworldqa']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "realworldqa")
    DATASET_CONFIG['mmstar']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "mmstar")
    DATASET_CONFIG['seedbench2plus']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "seedbench2plus")
    DATASET_CONFIG['hallusionbench']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "hallusionbench")
    DATASET_CONFIG['mme']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "mme")
    DATASET_CONFIG['hrbench']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "hrbench")
    DATASET_CONFIG['vsr_filtered']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "vsr_filtered")
    DATASET_CONFIG['cub_filtered']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "cub_filtered")
    DATASET_CONFIG['muirbench']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "muirbench")
    DATASET_CONFIG['visulogic']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "visulogic")
    DATASET_CONFIG['geometry3k']['output_dir'] = os.path.join(PROJECT_ROOT, args.output_dir, "geometry3k")

    # Determine checkpoint paths
    if args.checkpoint:
        chkpt_paths = [os.path.abspath(p) for p in args.checkpoint]
    else:
        chkpt_paths = DEFAULT_CHKPT_PATHS

    # Determine step list
    step_list = args.steps if args.steps else DEFAULT_STEP_LIST

    # Determine decoding strategy
    decoding_strategy = args.strategy if args.strategy else DEFAULT_DECODING_STRATEGY

    # Print header
    print(f"\n{'#'*80}")
    print("LASER Model Evaluation Script (Local Version)")
    print(f"{'#'*80}\n")

    if not chkpt_paths or chkpt_paths[0] == "":
        print("ERROR: No checkpoint specified!")
        print("Usage: python evaluation_local.py --checkpoint stage1_checkpoints/checkpoint-300")
        return

    # Check which benchmarks are enabled
    if args.benchmark:
        # Use specified benchmarks
        enabled_benchmarks = args.benchmark
    else:
        # Use all enabled in config
        enabled_benchmarks = [name for name, cfg in DATASET_CONFIG.items() if cfg.get('enabled', False)]

    # Determine BLINK configs
    if args.blink_configs:
        if args.blink_configs == ['all']:
            blink_configs = ALL_BLINK_CONFIGS
        else:
            # Validate configs
            invalid = [c for c in args.blink_configs if c not in ALL_BLINK_CONFIGS]
            if invalid:
                print(f"ERROR: Invalid BLINK configs: {invalid}")
                print(f"Available: {ALL_BLINK_CONFIGS}")
                return
            blink_configs = args.blink_configs
    else:
        blink_configs = DEFAULT_BLINK_CONFIGS

    print(f"Checkpoints: {', '.join(chkpt_paths)}")
    print(f"Enabled benchmarks: {', '.join(enabled_benchmarks)}")
    print(f"BLINK configs: {blink_configs} ({len(blink_configs)}/14)")
    print(f"LASER steps to test: {step_list}")
    print(f"Decoding strategy: {decoding_strategy}")
    if decoding_strategy == "dynamic":
        max_laser_steps = args.max_laser_steps if args.max_laser_steps else 16
        print(f"Max LASER steps (for forced exit): {max_laser_steps}")
    else:
        max_laser_steps = None
    print(f"Repetition exit: {args.repetition_exit}")
    if args.world_size > 1:
        print(f"Parallel mode: Rank {args.rank}/{args.world_size}")
    print()

    # Evaluate each checkpoint
    for checkpoint_dir in chkpt_paths:
        if not os.path.exists(checkpoint_dir):
            print(f"WARNING: Checkpoint not found: {checkpoint_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint: {checkpoint_dir}")
        print(f"{'='*80}\n")

        # Load model
        model, processor, run_name = load_model_and_processor(checkpoint_dir)

        # Build strategy directory name (include max_laser_steps for dynamic strategy)
        if decoding_strategy == "dynamic" and max_laser_steps is not None:
            strategy_dir_name = f"decoding_by_{decoding_strategy}_max{max_laser_steps}"
        else:
            strategy_dir_name = f"decoding_by_{decoding_strategy}"

        # Append repetition_exit flag to directory name
        if args.repetition_exit:
            strategy_dir_name += "_rep_exit"

        # Run benchmarks
        if 'blink' in enabled_benchmarks and DATASET_CONFIG['blink']['enabled']:
            output_dir = os.path.join(
                DATASET_CONFIG['blink']['output_dir'],
                strategy_dir_name,
                run_name
            )
            evaluate_blink(model, processor, output_dir, decoding_strategy, step_list,
                          args.rank, args.world_size, seed=42, max_laser_steps=max_laser_steps,
                          blink_configs=blink_configs,
                          repetition_exit=args.repetition_exit)

        # MMVP evaluation
        if 'mmvp' in enabled_benchmarks and DATASET_CONFIG['MMVP']['enabled']:
            output_dir = os.path.join(
                DATASET_CONFIG['MMVP']['output_dir'],
                strategy_dir_name,
                run_name
            )
            evaluate_mmvp(model, processor, output_dir, decoding_strategy, step_list,
                         args.rank, args.world_size, seed=42, max_laser_steps=max_laser_steps,
                         repetition_exit=args.repetition_exit)

        # MMStar evaluation
        if 'mmstar' in enabled_benchmarks and DATASET_CONFIG['mmstar']['enabled']:
            output_dir = os.path.join(
                DATASET_CONFIG['mmstar']['output_dir'],
                strategy_dir_name,
                run_name
            )
            evaluate_mmstar(model, processor, output_dir, decoding_strategy, step_list,
                           args.rank, args.world_size, seed=42, max_laser_steps=max_laser_steps,
                           repetition_exit=args.repetition_exit)

        # SEED-Bench-2-Plus evaluation
        if 'seedbench2plus' in enabled_benchmarks and DATASET_CONFIG['seedbench2plus']['enabled']:
            output_dir = os.path.join(
                DATASET_CONFIG['seedbench2plus']['output_dir'],
                strategy_dir_name,
                run_name
            )
            evaluate_seedbench2plus(model, processor, output_dir, decoding_strategy, step_list,
                                   args.rank, args.world_size, seed=42, max_laser_steps=max_laser_steps,
                                   repetition_exit=args.repetition_exit)

        # HallusionBench evaluation
        if 'hallusionbench' in enabled_benchmarks and DATASET_CONFIG['hallusionbench']['enabled']:
            output_dir = os.path.join(
                DATASET_CONFIG['hallusionbench']['output_dir'],
                strategy_dir_name,
                run_name
            )
            evaluate_hallusionbench(model, processor, output_dir, decoding_strategy, step_list,
                                   args.rank, args.world_size, seed=42, max_laser_steps=max_laser_steps,
                                   repetition_exit=args.repetition_exit)

        # HRBench evaluation
        if 'hrbench' in enabled_benchmarks and DATASET_CONFIG['hrbench']['enabled']:
            output_dir = os.path.join(
                DATASET_CONFIG['hrbench']['output_dir'],
                strategy_dir_name,
                run_name
            )
            evaluate_hrbench(model, processor, output_dir, decoding_strategy, step_list,
                            args.rank, args.world_size, seed=42, max_laser_steps=max_laser_steps,
                            repetition_exit=args.repetition_exit)

    print(f"\n{'#'*80}")
    print("Evaluation Complete!")
    print(f"{'#'*80}\n")

if __name__ == "__main__":
    main()

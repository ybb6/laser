"""
Merge Stage1 Evaluation Results from Multiple Parallel Processes
"""

import os
import json
import argparse
from collections import defaultdict

# Import accuracy functions from evaluation_local to avoid code duplication
from evaluation_local import accuracy_reward, accuracy_reward_yesno


def find_result_dirs(benchmark, checkpoint, strategy, output_dir="evaluation/results", max_laser_steps=None, repetition_exit=False):
    """Find all result directories for a given benchmark and checkpoint

    Args:
        repetition_exit: Whether to look for results with '_rep_exit' suffix
    """
    # Determine base directory based on benchmark
    base_dirs = {
        'blink': f'{output_dir}/blink',
        'mmvp': f'{output_dir}/mmvp',
        'mmstar': f'{output_dir}/mmstar',
        'seedbench2plus': f'{output_dir}/seedbench2plus',
        'hallusionbench': f'{output_dir}/hallusionbench',
        'hrbench': f'{output_dir}/hrbench',
    }

    if benchmark not in base_dirs:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    base_dir = base_dirs[benchmark]

    # Build strategy directory name
    if strategy == "dynamic" and max_laser_steps is not None:
        strategy_dir_name = f"decoding_by_{strategy}_max{max_laser_steps}"
    else:
        strategy_dir_name = f"decoding_by_{strategy}"

    # Append repetition_exit suffix
    if repetition_exit:
        strategy_dir_name += "_rep_exit"

    strategy_dir = os.path.join(base_dir, strategy_dir_name)

    if not os.path.exists(strategy_dir):
        return None

    # Find checkpoint directory (include parent dir to distinguish versions)
    parent_dir = os.path.basename(os.path.dirname(checkpoint))
    ckpt_name = os.path.basename(checkpoint)
    checkpoint_name = f"{parent_dir}_{ckpt_name}"

    # Check for both GenWHead_ and non-GenWHead versions
    possible_names = [
        checkpoint_name,
        f"GenWHead_{checkpoint_name}"
    ]

    for name in possible_names:
        checkpoint_dir = os.path.join(strategy_dir, name)
        if os.path.exists(checkpoint_dir):
            return checkpoint_dir

    return None


def merge_results(benchmark, checkpoint, strategy, num_ranks, steps_list, output_dir="evaluation/results", max_laser_steps=None, repetition_exit=False):
    """Merge results from multiple parallel evaluation processes

    Args:
        repetition_exit: Whether to look for results with '_rep_exit' suffix
    """

    print(f"\n{'='*80}")
    print(f"Merging results for benchmark: {benchmark}")
    if strategy == "dynamic" and max_laser_steps is not None:
        print(f"Strategy: {strategy} (max_laser_steps={max_laser_steps})")
    if repetition_exit:
        print(f"Repetition exit: enabled")
    print(f"{'='*80}")

    # Find result directory
    result_dir = find_result_dirs(benchmark, checkpoint, strategy, output_dir, max_laser_steps, repetition_exit)
    if not result_dir:
        print(f"ERROR: Result directory not found for {benchmark}/{checkpoint}")
        return False

    print(f"Result directory: {result_dir}")

    # For dynamic strategy, only one merge (no steps)
    if strategy == "dynamic":
        steps_list = [0]  # dummy value
    elif not steps_list:
        # Auto-detect steps if not provided
        steps_list = []
        for filename in os.listdir(result_dir):
            if filename.endswith("_rank0.json"):
                step_str = filename.replace(strategy, "").replace("_rank0.json", "")
                try:
                    steps_list.append(int(step_str))
                except ValueError:
                    pass
        steps_list = sorted(steps_list)

    if not steps_list:
        print("ERROR: No result files found")
        return False

    if strategy == "dynamic":
        print("Strategy: dynamic (no steps)")
    else:
        print(f"Steps to merge: {steps_list}")

    # Merge for each step
    for steps in steps_list:
        if strategy == "dynamic":
            print(f"\nMerging dynamic results...")
        else:
            print(f"\nMerging steps={steps}...")
        merged_results = []
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        total_correct = 0
        total_samples = 0

        # Collect from all ranks
        for rank in range(num_ranks):
            if strategy == "dynamic":
                result_file = os.path.join(result_dir, f"dynamic_rank{rank}.json")
            else:
                result_file = os.path.join(result_dir, f"{strategy}{steps:03d}_rank{rank}.json")

            if not os.path.exists(result_file):
                print(f"  WARNING: Missing {result_file}")
                continue

            with open(result_file, 'r') as f:
                rank_results = json.load(f)

            print(f"  Loaded {len(rank_results)} samples from rank {rank}")
            merged_results.extend(rank_results)

            # Compute statistics
            for res in rank_results:
                prediction = res['prediction'][0] if isinstance(res['prediction'], list) else res['prediction']
                label = res['label']
                category = res.get('category', 'Unknown')

                # Use appropriate accuracy function based on benchmark
                if benchmark in ['hallusionbench', 'mme']:
                    is_correct = accuracy_reward_yesno(prediction, label)
                elif benchmark == 'vsr_filtered':
                    is_correct = accuracy_reward_yesno_relaxed(prediction, label)
                elif benchmark == 'cub_filtered':
                    is_correct = accuracy_reward_choice_relaxed(prediction, label)
                else:
                    is_correct = accuracy_reward(prediction, label)

                if is_correct:
                    total_correct += 1
                    category_stats[category]['correct'] += 1

                total_samples += 1
                category_stats[category]['total'] += 1

        # Save merged results
        if strategy == "dynamic":
            output_file = os.path.join(result_dir, "dynamic.json")
        else:
            output_file = os.path.join(result_dir, f"{strategy}{steps:03d}.json")
        with open(output_file, 'w') as f:
            json.dump(merged_results, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {output_file}")
        print(f"  Total samples: {total_samples}")
        if total_samples > 0:
            print(f"  Accuracy: {total_correct}/{total_samples} = {total_correct/total_samples*100:.2f}%")

        # Print per-category stats
        if len(category_stats) > 1:
            print(f"\n  Per-category accuracy:")
            for cat in sorted(category_stats.keys()):
                stats = category_stats[cat]
                acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"    {cat}: {stats['correct']}/{stats['total']} = {acc:.2f}%")

    print(f"\n{'='*80}")
    print(f"Merge complete for {benchmark}")
    print(f"{'='*80}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Stage1 evaluation results from multiple parallel processes"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=['blink', 'vstar', 'mmvp', 'realworldqa', 'mmstar', 'seedbench2plus', 'hallusionbench', 'mme', 'hrbench', 'vsr_filtered', 'cub_filtered', 'muirbench', 'visulogic', 'geometry3k'],
        help="Benchmark name"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path or name"
    )
    parser.add_argument(
        "--strategy",
        default="steps",
        choices=['steps', 'latent', 'dynamic'],
        help="Decoding strategy"
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        required=True,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="LASER steps to merge (auto-detect if not specified)"
    )
    parser.add_argument(
        "--max_laser_steps",
        type=int,
        default=None,
        help="Maximum LASER steps (required for 'dynamic' strategy to find correct directory)"
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
        help="Look for results with repetition exit enabled (_rep_exit suffix)"
    )

    args = parser.parse_args()

    success = merge_results(
        args.benchmark,
        args.checkpoint,
        args.strategy,
        args.num_ranks,
        args.steps,
        args.output_dir,
        args.max_laser_steps,
        args.repetition_exit
    )

    if not success:
        exit(1)

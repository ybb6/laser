"""
Benchmark Evaluation Functions for LASER Models
Contains evaluation functions for various multimodal benchmarks

Supported benchmarks:
- BLINK (14 visual reasoning tasks)
- MMVP (multimodal visual perception)
- MMStar (multimodal reasoning)
- SEED-Bench-2-Plus (visual understanding)
- HallusionBench (hallucination detection)
- HRBench (high-resolution understanding)
"""

import os
import json
import string
import re
import glob
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Default step list
DEFAULT_STEP_LIST = [4, 8, 16]

# BLINK configs
ALL_BLINK_CONFIGS = [
    'Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence',
    'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
    'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
    'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity'
]
DEFAULT_BLINK_CONFIGS = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']

# ==== Helper Functions ====

def accuracy_reward(response: str, ground_truth: str) -> float:
    """Extract answer from response and compare with ground truth"""
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()

    words = given_answer.split()
    last_words = words[-10:] if len(words) > 10 else words

    extracted = None
    for word in reversed(last_words):
        match = re.search(r'(?<![a-zA-Z])([A-Da-d])(?![a-zA-Z])', word)
        if match:
            extracted = match.group(1).upper()
            break

    if extracted is None:
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', given_answer)
        extracted = cleaned[0].upper() if cleaned else ""

    ground_truth = re.sub(r'[^a-zA-Z0-9]', '', ground_truth).upper()
    return extracted == ground_truth

def accuracy_reward_yesno(response: str, ground_truth: str) -> bool:
    """Extract yes/no answer from response and compare with ground truth (0/1)"""
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip().lower()

    yes_patterns = ['yes', '1', 'true', 'correct']
    no_patterns = ['no', '0', 'false', 'incorrect']

    extracted = None
    for pattern in yes_patterns:
        if pattern in given_answer:
            extracted = 1
            break
    if extracted is None:
        for pattern in no_patterns:
            if pattern in given_answer:
                extracted = 0
                break

    if extracted is None:
        first_word = given_answer.split()[0] if given_answer.split() else ""
        if first_word in yes_patterns:
            extracted = 1
        elif first_word in no_patterns:
            extracted = 0
        else:
            extracted = -1

    gt = int(ground_truth) if str(ground_truth).isdigit() else (1 if str(ground_truth).lower() == 'yes' else 0)
    return extracted == gt

def shard_data(data, rank=0, world_size=1, seed=42):
    """Shard data across multiple processes"""
    if world_size <= 1:
        return data

    import random
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    total = len(shuffled_data)
    per_rank = total // world_size
    start_idx = rank * per_rank

    if rank == world_size - 1:
        end_idx = total
    else:
        end_idx = start_idx + per_rank

    sharded = shuffled_data[start_idx:end_idx]
    print(f"[Rank {rank}/{world_size}] Processing samples {start_idx}-{end_idx} ({len(sharded)}/{total} samples)")
    return sharded

def run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps=None, repetition_exit=False):
    """Run inference - selects correct implementation based on model type"""
    model_class_name = model.__class__.__name__

    if 'Llava' in model_class_name or 'llava' in model_class_name.lower():
        from evaluation_local_llava import run_inference as _run_inference
    else:
        from evaluation_local import run_inference as _run_inference

    return _run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

# ==== Evaluation Functions ====

def evaluate_blink(model, processor, output_dir, decoding_strategy="steps", step_list=None, rank=0, world_size=1, seed=42, max_laser_steps=None, blink_configs=None, repetition_exit=False):
    """Evaluate on BLINK benchmark

    Args:
        blink_configs: List of BLINK configs to evaluate. If None, uses DEFAULT_BLINK_CONFIGS.
                      Use ALL_BLINK_CONFIGS for full benchmark (14 configs).
    """
    if step_list is None:
        step_list = DEFAULT_STEP_LIST
    if blink_configs is None:
        blink_configs = DEFAULT_BLINK_CONFIGS

    print(f"\n{'='*80}")
    print(f"Evaluating BLINK with decoding strategy: {decoding_strategy}")
    print(f"BLINK configs: {blink_configs} ({len(blink_configs)}/14)")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load BLINK dataset (auto-downloads from HuggingFace)
    configs = blink_configs
    print(f"Loading BLINK dataset from HuggingFace ({len(configs)} configs)...")
    all_datasets = {}
    for config in configs:
        all_datasets[config] = load_dataset("BLINK-Benchmark/BLINK", config)

    # Process data
    processed_data = []
    for config in all_datasets:
        ds = all_datasets[config]['val']
        for dat in ds:
            idx = dat["idx"]
            choices = dat["choices"]
            letters = string.ascii_uppercase
            paired = list(zip(letters, choices))
            option_string = ""
            for letter, choice in paired:
                option_string += f"{letter}. {choice}\n"

            if len(dat['answer']) > 1:
                ans = dat['answer'][1].upper()
            else:
                ans = dat['answer'][0].upper()

            images = []
            for k in ['image_1', 'image_2', 'image_3', 'image_4']:
                if k in dat and dat[k] is not None:
                    images.append(dat[k])

            question = dat['question'] + "\nOptions:\n" + option_string
            buffer = {
                "question_id": idx,
                "image": images,
                "query": question,
                "label": ans,
                "category": config
            }
            processed_data.append(buffer)

    print(f"Loaded {len(processed_data)} examples from BLINK")

    # Shard data across processes
    processed_data = shard_data(processed_data, rank, world_size, seed)

    # Evaluate for each step count
    step2results_category = {}
    step2results_overall = {}

    # For dynamic strategy, only run once (steps is ignored)
    effective_step_list = [0] if decoding_strategy == "dynamic" else step_list

    for steps in effective_step_list:
        step2results_category[steps] = {}
        # Include rank in output filename for parallel processing
        if decoding_strategy == "dynamic":
            out_file = os.path.join(output_dir, f"dynamic_rank{rank}.json") if world_size > 1 else os.path.join(output_dir, "dynamic.json")
        elif world_size > 1:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}_rank{rank}.json")
        else:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")

        total, correct = 0, 0
        res_by_category = {}

        if os.path.exists(out_file):
            print(f"\nResults file exists, loading..." if decoding_strategy == "dynamic" else f"\nResults file exists for steps={steps}, loading...")
            with open(out_file, "r") as f:
                result = json.load(f)

            # Recompute accuracy
            for res in result:
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total": 0, "correct": 0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            print(f"\nRunning inference (dynamic)..." if decoding_strategy == "dynamic" else f"\nRunning inference for steps={steps}...")
            result = []
            desc = "Evaluating BLINK (dynamic)" if decoding_strategy == "dynamic" else f"Evaluating BLINK (steps={steps})"
            for dat in tqdm(processed_data, desc=desc):
                img_path = dat['image']
                text = dat['query']
                outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

                res = {
                    'id': dat['question_id'],
                    'question': dat['query'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category']
                }
                result.append(res)

                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                    if dat['category'] not in res_by_category:
                        res_by_category[dat['category']] = {"total": 0, "correct": 0}
                    res_by_category[dat['category']]["correct"] += 1
                total += 1
                if dat['category'] not in res_by_category:
                    res_by_category[dat['category']] = {"total": 0, "correct": 0}
                res_by_category[dat['category']]["total"] += 1

            json.dump(result, open(out_file, 'w+'), indent=2)
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}

        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    # Print overall results
    print(f"\n{'='*80}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))

    # Print by category
    print("\nAccuracy by category:")
    for category in configs:
        res = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct / cat_total)
        if res:
            print(category + ',' + ",".join([f"{items*100:.2f}" for items in res]))
    print(f"{'='*80}\n")


def evaluate_mmvp(model, processor, output_dir, decoding_strategy="steps", step_list=None, rank=0, world_size=1, seed=42, max_laser_steps=None, repetition_exit=False):
    """Evaluate on MMVP benchmark

    Args:
    """
    if step_list is None:
        step_list = DEFAULT_STEP_LIST

    print(f"\n{'='*80}")
    print(f"Evaluating MMVP with decoding strategy: {decoding_strategy}")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Find MMVP cache directory
    print("Loading MMVP dataset from HuggingFace cache...")
    cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--MMVP--MMVP/snapshots/*/")
    cache_dirs = glob.glob(cache_pattern)
    if not cache_dirs:
        raise FileNotFoundError("MMVP dataset cache not found. Please run: huggingface-cli download MMVP/MMVP --repo-type dataset")

    mmvp_cache_dir = cache_dirs[0]
    mmvp_images_dir = os.path.join(mmvp_cache_dir, "MMVP Images")

    # Load Questions.csv from cache
    csv_file = os.path.join(mmvp_cache_dir, "Questions.csv")
    if not os.path.exists(csv_file):
        print(f"ERROR: Questions.csv not found in cache at {csv_file}")
        print("Please run: huggingface-cli download MMVP/MMVP --repo-type dataset")
        return

    print(f"Using MMVP cache directory: {mmvp_cache_dir}")
    print(f"Loading questions from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} questions from CSV")

    # Process data
    processed_data = []
    for _, row in df.iterrows():
        img_index = int(row['Index'])
        img_path = os.path.join(mmvp_images_dir, f"{img_index}.jpg")

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        img = Image.open(img_path)

        question = row['Question']
        options = row['Options']
        answer = row['Correct Answer']

        full_query = question + " " + options

        buffer = {
            "question_id": str(img_index),
            "image": img,
            "query": full_query,
            "label": answer,
        }
        processed_data.append(buffer)

    print(f"Processed {len(processed_data)} question-image pairs")

    # Shard data across processes
    processed_data = shard_data(processed_data, rank, world_size, seed)

    # Evaluate for each step count
    step2results_overall = {}

    effective_step_list = [0] if decoding_strategy == "dynamic" else step_list

    for steps in effective_step_list:
        if decoding_strategy == "dynamic":
            out_file = os.path.join(output_dir, f"dynamic_rank{rank}.json") if world_size > 1 else os.path.join(output_dir, "dynamic.json")
        elif world_size > 1:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}_rank{rank}.json")
        else:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")

        total, correct = 0, 0

        if os.path.exists(out_file):
            print(f"\nResults file exists, loading..." if decoding_strategy == "dynamic" else f"\nResults file exists for steps={steps}, loading...")
            with open(out_file, "r") as f:
                result = json.load(f)

            for res in result:
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                total += 1
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            print(f"\nRunning inference (dynamic)..." if decoding_strategy == "dynamic" else f"\nRunning inference for steps={steps}...")
            result = []
            desc = "Evaluating MMVP (dynamic)" if decoding_strategy == "dynamic" else f"Evaluating MMVP (steps={steps})"
            for dat in tqdm(processed_data, desc=desc):
                img_path = dat['image']
                text = dat['query']
                outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

                res = {
                    'id': dat['question_id'],
                    'question': dat['query'],
                    'prediction': outputs,
                    'label': dat['label'],
                }
                result.append(res)

                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                total += 1

            json.dump(result, open(out_file, 'w+'), indent=2)
            step2results_overall[steps] = {"total": total, "correct": correct}

        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\n{'='*80}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    print(f"{'='*80}\n")


def evaluate_mmstar(model, processor, output_dir, decoding_strategy="steps", step_list=None, rank=0, world_size=1, seed=42, max_laser_steps=None, repetition_exit=False):
    """Evaluate on MMStar benchmark (Lin-Chen/MMStar)

    Args:
    """
    if step_list is None:
        step_list = DEFAULT_STEP_LIST

    print(f"\n{'='*80}")
    print(f"Evaluating MMStar with decoding strategy: {decoding_strategy}")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load MMStar from HuggingFace
    print("Loading MMStar dataset from HuggingFace...")
    dataset = load_dataset("Lin-Chen/MMStar", split="val")
    print(f"Loaded {len(dataset)} examples from MMStar")

    # Process data
    processed_data = []
    for idx, dat in enumerate(dataset):
        img = dat['image']
        question = dat['question']
        answer = dat['answer']
        category = dat.get('category', 'unknown')
        l2_category = dat.get('l2_category', 'unknown')

        full_query = question

        buffer = {
            "question_id": str(dat.get('index', idx)),
            "image": img,
            "query": full_query,
            "label": answer,
            "category": category,
            "l2_category": l2_category,
        }
        processed_data.append(buffer)

    # Shard data across processes
    processed_data = shard_data(processed_data, rank, world_size, seed)

    # Evaluate
    step2results_category = {}
    step2results_overall = {}
    effective_step_list = [0] if decoding_strategy == "dynamic" else step_list

    for steps in effective_step_list:
        step2results_category[steps] = {}
        if decoding_strategy == "dynamic":
            out_file = os.path.join(output_dir, f"dynamic_rank{rank}.json") if world_size > 1 else os.path.join(output_dir, "dynamic.json")
        elif world_size > 1:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}_rank{rank}.json")
        else:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")

        total, correct = 0, 0
        res_by_category = {}

        if os.path.exists(out_file):
            print(f"\nResults file exists, loading..." if decoding_strategy == "dynamic" else f"\nResults file exists for steps={steps}, loading...")
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                cat = res.get("category", "unknown")
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            print(f"\nRunning inference (dynamic)..." if decoding_strategy == "dynamic" else f"\nRunning inference for steps={steps}...")
            result = []
            desc = "Evaluating MMStar (dynamic)" if decoding_strategy == "dynamic" else f"Evaluating MMStar (steps={steps})"
            for dat in tqdm(processed_data, desc=desc):
                img_path = dat['image']
                text = dat['query']
                outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

                res = {
                    'id': dat['question_id'],
                    'question': dat['query'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category'],
                    'l2_category': dat['l2_category'],
                }
                result.append(res)

                cat = dat['category']
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1

            json.dump(result, open(out_file, 'w+'), indent=2)
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}

        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\n{'='*80}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))

    if res_by_category:
        print("\nAccuracy by category:")
        for category in sorted(res_by_category.keys()):
            res = []
            for steps in step2results_category:
                if category in step2results_category[steps]:
                    cat_total = step2results_category[steps][category]["total"]
                    cat_correct = step2results_category[steps][category]["correct"]
                    res.append(cat_correct / cat_total if cat_total > 0 else 0)
            if res:
                print(category + ',' + ",".join([f"{items*100:.2f}" for items in res]))
    print(f"{'='*80}\n")


def evaluate_seedbench2plus(model, processor, output_dir, decoding_strategy="steps", step_list=None, rank=0, world_size=1, seed=42, max_laser_steps=None, repetition_exit=False):
    """Evaluate on SEED-Bench-2-Plus benchmark (AILab-CVC/SEED-Bench-2-plus)

    Args:
    """
    if step_list is None:
        step_list = DEFAULT_STEP_LIST

    print(f"\n{'='*80}")
    print(f"Evaluating SEED-Bench-2-Plus with decoding strategy: {decoding_strategy}")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Find cache directory
    print("Loading SEED-Bench-2-Plus dataset from HuggingFace cache...")
    cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--AILab-CVC--SEED-Bench-2-plus/snapshots/*/")
    cache_dirs = glob.glob(cache_pattern)
    if not cache_dirs:
        raise FileNotFoundError("SEED-Bench-2-Plus dataset cache not found. Please run: huggingface-cli download AILab-CVC/SEED-Bench-2-plus --repo-type dataset")

    cache_dir = cache_dirs[0]
    json_file = os.path.join(cache_dir, "SEED-Bench-2-plus-text-rich.json")
    images_dir = os.path.join(cache_dir, "text_rich")

    # Check if images are extracted
    if not os.path.exists(images_dir):
        zip_file = os.path.join(cache_dir, "text_rich.zip")
        if os.path.exists(zip_file):
            print(f"Extracting images from {zip_file}...")
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            print(f"Extraction complete. Images extracted to {images_dir}")
        else:
            raise FileNotFoundError(f"Images not found. Please extract text_rich.zip in {cache_dir}")

    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from SEED-Bench-2-Plus")

    # Process data
    processed_data = []
    for dat in data:
        question_id = dat['question_id']
        question = dat['question']
        choices = f"A. {dat['choice_A']}\nB. {dat['choice_B']}\nC. {dat['choice_C']}\nD. {dat['choice_D']}"
        answer = dat['answer']
        category = dat.get('question_image_type', 'unknown')

        data_id = dat['data_id']
        img_path = os.path.join(cache_dir, data_id)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found for data_id={data_id} at {img_path}")
            continue

        img = Image.open(img_path)

        full_query = question + "\n" + choices

        buffer = {
            "question_id": question_id,
            "image": img,
            "query": full_query,
            "label": answer,
            "category": category,
        }
        processed_data.append(buffer)

    print(f"Processed {len(processed_data)} question-image pairs")

    # Shard data across processes
    processed_data = shard_data(processed_data, rank, world_size, seed)

    # Evaluate
    step2results_category = {}
    step2results_overall = {}
    effective_step_list = [0] if decoding_strategy == "dynamic" else step_list

    for steps in effective_step_list:
        step2results_category[steps] = {}
        if decoding_strategy == "dynamic":
            out_file = os.path.join(output_dir, f"dynamic_rank{rank}.json") if world_size > 1 else os.path.join(output_dir, "dynamic.json")
        elif world_size > 1:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}_rank{rank}.json")
        else:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")

        total, correct = 0, 0
        res_by_category = {}

        if os.path.exists(out_file):
            print(f"\nResults file exists, loading..." if decoding_strategy == "dynamic" else f"\nResults file exists for steps={steps}, loading...")
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                cat = res.get("category", "unknown")
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            print(f"\nRunning inference (dynamic)..." if decoding_strategy == "dynamic" else f"\nRunning inference for steps={steps}...")
            result = []
            desc = "Evaluating SEED-Bench-2-Plus (dynamic)" if decoding_strategy == "dynamic" else f"Evaluating SEED-Bench-2-Plus (steps={steps})"
            for dat in tqdm(processed_data, desc=desc):
                img_path = dat['image']
                text = dat['query']
                outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

                res = {
                    'id': dat['question_id'],
                    'question': dat['query'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category'],
                }
                result.append(res)

                cat = dat['category']
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1

            json.dump(result, open(out_file, 'w+'), indent=2)
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}

        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\n{'='*80}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))

    if res_by_category:
        print("\nAccuracy by category:")
        for category in sorted(res_by_category.keys()):
            res = []
            for steps in step2results_category:
                if category in step2results_category[steps]:
                    cat_total = step2results_category[steps][category]["total"]
                    cat_correct = step2results_category[steps][category]["correct"]
                    res.append(cat_correct / cat_total if cat_total > 0 else 0)
            if res:
                print(category + ',' + ",".join([f"{items*100:.2f}" for items in res]))
    print(f"{'='*80}\n")


def evaluate_hallusionbench(model, processor, output_dir, decoding_strategy="steps", step_list=None, rank=0, world_size=1, seed=42, max_laser_steps=None, repetition_exit=False):
    """Evaluate on HallusionBench benchmark (lmms-lab/HallusionBench)

    Args:
    """
    if step_list is None:
        step_list = DEFAULT_STEP_LIST

    print(f"\n{'='*80}")
    print(f"Evaluating HallusionBench with decoding strategy: {decoding_strategy}")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load HallusionBench from HuggingFace (only image split, skip non_image)
    print("Loading HallusionBench dataset from HuggingFace...")
    dataset = load_dataset("lmms-lab/HallusionBench", split="image")
    print(f"Loaded {len(dataset)} examples from HallusionBench (image split)")

    # Process data
    processed_data = []
    for idx, dat in enumerate(dataset):
        img = dat['image']
        question = dat['question']
        gt_answer = dat['gt_answer']
        category = dat.get('category', 'unknown')
        subcategory = dat.get('subcategory', 'unknown')

        full_query = question

        buffer = {
            "question_id": str(dat.get('question_id', idx)),
            "image": img,
            "query": full_query,
            "label": str(gt_answer),
            "category": category,
            "subcategory": subcategory,
        }
        processed_data.append(buffer)

    # Shard data across processes
    processed_data = shard_data(processed_data, rank, world_size, seed)

    # Evaluate
    step2results_category = {}
    step2results_overall = {}
    effective_step_list = [0] if decoding_strategy == "dynamic" else step_list

    for steps in effective_step_list:
        step2results_category[steps] = {}
        if decoding_strategy == "dynamic":
            out_file = os.path.join(output_dir, f"dynamic_rank{rank}.json") if world_size > 1 else os.path.join(output_dir, "dynamic.json")
        elif world_size > 1:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}_rank{rank}.json")
        else:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")

        total, correct = 0, 0
        res_by_category = {}

        if os.path.exists(out_file):
            print(f"\nResults file exists, loading..." if decoding_strategy == "dynamic" else f"\nResults file exists for steps={steps}, loading...")
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                cat = res.get("category", "unknown")
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward_yesno(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            print(f"\nRunning inference (dynamic)..." if decoding_strategy == "dynamic" else f"\nRunning inference for steps={steps}...")
            result = []
            desc = "Evaluating HallusionBench (dynamic)" if decoding_strategy == "dynamic" else f"Evaluating HallusionBench (steps={steps})"
            for dat in tqdm(processed_data, desc=desc):
                img_path = dat['image']
                text = dat['query']
                outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

                res = {
                    'id': dat['question_id'],
                    'question': dat['query'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category'],
                    'subcategory': dat['subcategory'],
                }
                result.append(res)

                cat = dat['category']
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward_yesno(outputs[0], dat['label']):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1

            json.dump(result, open(out_file, 'w+'), indent=2)
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}

        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\n{'='*80}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))

    if res_by_category:
        print("\nAccuracy by category:")
        for category in sorted(res_by_category.keys()):
            res = []
            for steps in step2results_category:
                if category in step2results_category[steps]:
                    cat_total = step2results_category[steps][category]["total"]
                    cat_correct = step2results_category[steps][category]["correct"]
                    res.append(cat_correct / cat_total if cat_total > 0 else 0)
            if res:
                print(category + ',' + ",".join([f"{items*100:.2f}" for items in res]))
    print(f"{'='*80}\n")


def evaluate_hrbench(model, processor, output_dir, decoding_strategy="steps", step_list=None, rank=0, world_size=1, seed=42, max_laser_steps=None, repetition_exit=False):
    """Evaluate on HRBench 4K benchmark (DreamMr/HR-Bench)

    Args:
    """
    if step_list is None:
        step_list = DEFAULT_STEP_LIST

    print(f"\n{'='*80}")
    print(f"Evaluating HRBench 4K with decoding strategy: {decoding_strategy}")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Find HRBench cache directory and load 4K data
    print("Loading HRBench 4K dataset from HuggingFace cache...")
    cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--DreamMr--HR-Bench/snapshots/*/")
    cache_dirs = glob.glob(cache_pattern)
    if not cache_dirs:
        raise FileNotFoundError("HRBench dataset cache not found. Please run: huggingface-cli download DreamMr/HR-Bench --repo-type dataset")

    cache_dir = cache_dirs[0]

    # Try parquet first, then tsv
    parquet_file = os.path.join(cache_dir, "hr_bench_4k.parquet")
    tsv_file = os.path.join(cache_dir, "hr_bench_4k.tsv")

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} examples from HRBench 4K (parquet)")
    elif os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        print(f"Loaded {len(df)} examples from HRBench 4K (tsv)")
    else:
        raise FileNotFoundError(f"HRBench 4K data not found at {parquet_file} or {tsv_file}")

    # Process data
    processed_data = []
    for idx, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        category = row.get('category', row.get('cycle_category', 'unknown'))

        options = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"

        # Decode image from base64 if needed
        img_data = row['image']
        if isinstance(img_data, str):
            import base64
            from io import BytesIO
            if img_data.startswith('data:'):
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
        else:
            img = img_data

        full_query = question + "\n" + options

        buffer = {
            "question_id": str(row.get('index', idx)),
            "image": img,
            "query": full_query,
            "label": answer,
            "category": category,
        }
        processed_data.append(buffer)

    # Shard data across processes
    processed_data = shard_data(processed_data, rank, world_size, seed)

    # Evaluate
    step2results_category = {}
    step2results_overall = {}
    effective_step_list = [0] if decoding_strategy == "dynamic" else step_list

    for steps in effective_step_list:
        step2results_category[steps] = {}
        if decoding_strategy == "dynamic":
            out_file = os.path.join(output_dir, f"dynamic_rank{rank}.json") if world_size > 1 else os.path.join(output_dir, "dynamic.json")
        elif world_size > 1:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}_rank{rank}.json")
        else:
            out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")

        total, correct = 0, 0
        res_by_category = {}

        if os.path.exists(out_file):
            print(f"\nResults file exists, loading..." if decoding_strategy == "dynamic" else f"\nResults file exists for steps={steps}, loading...")
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                cat = res.get("category", "unknown")
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            print(f"\nRunning inference (dynamic)..." if decoding_strategy == "dynamic" else f"\nRunning inference for steps={steps}...")
            result = []
            desc = "Evaluating HRBench 4K (dynamic)" if decoding_strategy == "dynamic" else f"Evaluating HRBench 4K (steps={steps})"
            for dat in tqdm(processed_data, desc=desc):
                img_path = dat['image']
                text = dat['query']
                outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy, max_laser_steps, repetition_exit)

                res = {
                    'id': dat['question_id'],
                    'question': dat['query'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category'],
                }
                result.append(res)

                cat = dat['category']
                if cat not in res_by_category:
                    res_by_category[cat] = {"total": 0, "correct": 0}
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                    res_by_category[cat]["correct"] += 1
                total += 1
                res_by_category[cat]["total"] += 1

            json.dump(result, open(out_file, 'w+'), indent=2)
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}

        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\n{'='*80}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))

    if res_by_category:
        print("\nAccuracy by category (HRBench):")
        for category in sorted(res_by_category.keys()):
            res = []
            for steps in step2results_category:
                if category in step2results_category[steps]:
                    cat_total = step2results_category[steps][category]["total"]
                    cat_correct = step2results_category[steps][category]["correct"]
                    res.append(cat_correct / cat_total if cat_total > 0 else 0)
            if res:
                print(category + ',' + ",".join([f"{items*100:.2f}" for items in res]))
    print(f"{'='*80}\n")

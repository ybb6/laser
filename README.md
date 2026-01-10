# LASER: Latent Reasoning for Vision-Language Models

Official implementation of **LASER** (Latent Reasoning), enabling vision-language models to perform implicit reasoning in continuous latent space.

## Update

- [2025/01] Code release with DWAL training

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Checkpoints](#model-checkpoints)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/[TODO]/LASER.git
cd LASER
pip install -r requirements.txt

# Optional: Flash Attention 2 (recommended)
pip install flash-attn --no-build-isolation
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.1.0
- CUDA >= 11.8
- 8× GPUs with ≥40GB VRAM (for training)

## Quick Start

### Training

```bash
bash scripts/finetune_laser_dwal_7b.sh
```

### Evaluation

```bash
bash evaluation/run_evaluation_dwal_parallel.sh
```

## Training

### Data Format

Training data should follow LLaVA format with LASER tokens:

```json
[
  {
    "id": "sample_001",
    "image": ["path/to/image.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is shown in this image?"
      },
      {
        "from": "gpt",
        "value": "<|laser_start|><laser><laser>...<laser><|laser_end|><answer>A cat sitting on a couch.</answer>"
      }
    ]
  }
]
```

- `<|laser_start|>` / `<|laser_end|>`: Mark latent reasoning region
- `<laser>`: Placeholder for each reasoning step (replaced during training)
- `<answer>`: Contains the final answer

### Precompute Sample Lengths

For efficient dynamic batching:

```bash
python scripts/precompute_lengths.py \
    --data_path data/training_data.json \
    --output_path data/sample_lengths.json \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DWAL_TEMPERATURE` | 1.0 | Softmax temperature for target distribution |
| `DWAL_ENTROPY_THRESHOLD` | 0.6 | Threshold for "confused" state detection |
| `DWAL_FORCE_PROB` | 0.8 | Force probability when confused |
| `DYNAMIC_TOKEN_CAP` | 8192 | Memory constraint (batch_size × seq_len) |
| `MAX_STEPS` | 500 | Total training steps |

## Evaluation

### Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| BLINK | Visual reasoning (14 subtasks) |
| MMVP | Multimodal visual perception |
| MMStar | Multimodal reasoning |
| SEED-Bench-2-Plus | Text-rich understanding |
| HallusionBench | Hallucination detection |
| HR-Bench | High-resolution understanding |

### Single GPU

```bash
python evaluation/evaluation_local.py \
    --checkpoint checkpoints/checkpoint-500 \
    --benchmark blink mmvp mmstar \
    --strategy dynamic \
    --max_laser_steps 8
```

### Multi-GPU Parallel

```bash
# 1. Run parallel evaluation
bash evaluation/run_evaluation_dwal_parallel.sh

# 2. Merge results
python evaluation/merge_stage1_results.py \
    --benchmark blink \
    --checkpoint checkpoints/checkpoint-500 \
    --strategy dynamic \
    --num_ranks 8
```

## Model Checkpoints

| Model | Base | Download |
|-------|------|----------|
| LASER-7B | Qwen2.5-VL-7B-Instruct | [TODO: HuggingFace] |

## Training Data

| Dataset | Download |
|---------|----------|
| LASER Training Data | [TODO: Link] |

## Results

| Model | BLINK | MMVP | MMStar | SEED-2+ | HalluBench | HR-Bench |
|-------|-------|------|--------|---------|------------|----------|
| Qwen2.5-VL-7B | - | - | - | - | - | - |
| LASER-7B | - | - | - | - | - | - |

## Citation

```bibtex
@article{laser2025,
  title={LASER: Latent Reasoning for Vision-Language Models},
  author={[TODO]},
  journal={arXiv preprint arXiv:[TODO]},
  year={2025}
}
```

## License

Apache-2.0 License

## Acknowledgement

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [InternVL](https://github.com/OpenGVLab/InternVL)

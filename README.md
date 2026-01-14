# Forest Before Trees: Latent Superposition for Efficient Visual Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2601.06803)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of **Laser** (Latent Superposition for Effective Visual Reasoning). Laser enables vision-language models to perform implicit reasoning in continuous latent space, prioritizing global understanding ("Forest") before detailed processing ("Trees").

> **Note:** Code is released for reference; weights and data will be available soon.

## 📢 News

- **[2025/01]** Code release for Laser.
- **[Coming Soon]** Pre-trained checkpoints (Laser-7B) release.
- **[Coming Soon]** Training data release.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model](#model)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/ybb6/laser.git
cd Laser
pip install -r requirements.txt

# Optional: Flash Attention 2
pip install flash-attn --no-build-isolation
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.1.0
- CUDA >= 11.8

## Quick Start

### Training

To start training with the default configuration:

```bash
bash scripts/finetune_laser_dwal_7b.sh
```

### Evaluation

To run parallel evaluation across supported benchmarks:

```bash
bash evaluation/run_evaluation_dwal_parallel.sh
```

## Training

### Data Format

Training data should follow the LLaVA-style JSON format, extended with Laser-specific tokens:

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

- `<|laser_start|>` / `<|laser_end|>`: Delimiters for the latent reasoning region.
- `<laser>`: Placeholder token for each latent reasoning step (replaced dynamically during training).
- `<answer>`: Wraps the final textual output.

### Precompute Sample Lengths

For efficient dynamic batching during training, precompute the token lengths:

```bash
python scripts/precompute_lengths.py \
    --data_path data/training_data.json \
    --output_path data/sample_lengths.json \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct
```

## Evaluation

### Supported Benchmarks

We support a comprehensive suite of visual reasoning benchmarks:

| Benchmark | Description |
|-----------|-------------|
| **BLINK** | Visual reasoning (14 subtasks) |
| **MMVP** | Multimodal visual perception |
| **MMStar** | Multimodal reasoning |
| **SEED-Bench-2-Plus** | Text-rich understanding |
| **HallusionBench** | Hallucination detection |
| **HR-Bench** | High-resolution understanding |


## Model

### Checkpoints
We will release the pre-trained weights upon acceptance or shortly after the arXiv release.

| Model | Base Model | Status | Download |
|-------|------------|--------|----------|
| **Laser-7B** | Qwen2.5-VL-7B-Instruct | *Coming Soon* | - |

### Training Data

| Dataset | Description | Status | Download |
|---------|-------------|--------|----------|
| ScanPath | Curated reasoning instruction data | *Coming Soon* | - |


## Citation

If you find our work useful, please consider citing:

```bibtex
@article{laser2026forest,
  title={Forest Before Trees: Latent Superposition for Efficient Visual Reasoning},
  author={Wang, Yubo and Zhang, Juntian and Wu, Yichen and Lin, Yankai and Lukas, Nils and Liu, Yuhan},
  journal={arXiv preprint arXiv:2601.06803},
  year={2026}
}
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Acknowledgement

We thank the authors of the following projects for their open-source contributions:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [InternVL](https://github.com/OpenGVLab/InternVL)
```

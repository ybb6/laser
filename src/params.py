from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments
# from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    loss_laser_fct: str = field(default="mse")
    loss_laser_lambda: float = field(default=1e-1)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(default="vscode debugger", metadata={"help": "Name of the run for logging purposes."})
    # True if serving the checkpoints and data on oci
    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    max_steps:Optional[int] = 2500

    # Dynamic batch parameters
    enable_dynamic_batch: bool = field(default=False, metadata={"help": "Enable dynamic batching based on rectangular memory constraint"})
    dynamic_batch_token_cap: int = field(default=10240, metadata={"help": "Maximum rectangular area (batch_size × max_len) per batch"})
    dynamic_batch_max_bs: int = field(default=16, metadata={"help": "Maximum number of samples per batch"})
    dynamic_batch_buffer_size: int = field(default=10000, metadata={"help": "Size of processed buffer for sorting and packing"})

    # DWAL (Dynamic Windowed Alignment Loss) parameters
    dwal: bool = field(default=False, metadata={"help": "Enable DWAL mode for reasoning chain training"})
    dwal_window_size: Optional[int] = field(default=None, metadata={"help": "Window size K for future token prediction. None = dynamic (all remaining tokens to laser_end)"})
    dwal_temperature: float = field(default=1.0, metadata={"help": "Temperature for teacher distribution softmax"})
    dwal_entropy_threshold: float = field(default=0.8, metadata={"help": "Entropy threshold for adaptive intervention"})
    dwal_force_prob: float = field(default=0.8, metadata={"help": "Force probability when entropy > threshold"})
    dwal_loss_type: str = field(default="weighted_ce", metadata={"help": "Loss type for DWAL: mse, weighted_ce"})

    # Time-Aware DWAL parameters (Focusing Mechanism)
    dwal_time_aware: bool = field(default=False, metadata={"help": "Enable Time-Aware DWAL with spatial decay and temporal ramp"})
    dwal_spatial_decay_gamma: float = field(default=0.91, metadata={"help": "Spatial decay factor gamma for window position bias. Each step reduces logit by ln(gamma)."})
    dwal_time_weight_init: float = field(default=0.5, metadata={"help": "Initial time weight alpha for temporal ramp. Weight = alpha + (1-alpha) * (t/T)."})

    # Debug logging
    debug_log_dir: Optional[str] = field(default=None, metadata={"help": "Directory to save debug logs"})

    # NaN handling
    stop_on_nan: bool = field(
        default=False,
        metadata={"help": "If True, stop training immediately when NaN loss is detected. If False, skip the NaN batch and continue."}
    )


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    # === Generation (Rollout) Config ===
    # Controls memory usage during generation phase (with KV cache)
    rollout_micro_batch_size: int = field(
        default=None,
        metadata={
            "help": "Number of PROMPTS per device per rollout micro-batch during generation. "
                    "Each prompt generates num_generations samples, so actual samples = rollout_micro_batch_size * G. "
                    "Smaller values reduce GPU memory usage (due to KV cache). "
                    "If None, defaults to 1 (most memory-friendly)."
        }
    )

    num_rollout_micro_batches: int = field(
        default=1,
        metadata={
            "help": "Number of rollout micro-batches per cycle. "
                    "Controls how many prompts are generated per cycle: "
                    "prompts_per_cycle = rollout_micro_batch_size * num_rollout_micro_batches. "
                    "Increase this to generate more samples per cycle, enabling more training steps "
                    "before regeneration (useful when gradient_accumulation_steps > 1)."
        }
    )

    # === Training Config ===
    # Controls batch size during training phase (no KV cache, can be larger)
    train_micro_batch_size: int = field(
        default=None,
        metadata={
            "help": "Number of SAMPLES (responses) per device per training micro-batch. "
                    "Can be larger than rollout samples since training doesn't use KV cache. "
                    "Must be divisible by num_generations to maintain group integrity. "
                    "If None, defaults to per_device_train_batch_size."
        }
    )

    # DEPRECATED: Use rollout_micro_batch_size instead
    num_mini_batches: int = field(
        default=None,
        metadata={
            "help": "[DEPRECATED] Use rollout_micro_batch_size instead. "
                    "If both are specified, rollout_micro_batch_size takes precedence."
        }
    )

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512

    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume training from. Set to 'auto' to auto-detect, or specify checkpoint path."}
    )
    decoding_strategy:str = "steps"
    laser_steps: int = 16

    # EPG-GRPO parameters (Stage 2)
    use_epg: bool = field(
        default=False,
        metadata={"help": "Enable EPG (Expected Policy Gradient) mode for LASER region. Uses Top-P weighted sum instead of single token sampling."}
    )
    top_p_threshold: float = field(
        default=0.8,
        metadata={"help": "Cumulative probability threshold for Top-P set in EPG mode (0.0-1.0)."}
    )
    max_top_k: int = field(
        default=50,
        metadata={"help": "Maximum number of tokens in Top-P set to prevent memory issues."}
    )
    novelty_beta: float = field(
        default=0.05,
        metadata={"help": "Weight coefficient for novelty reward (recommended 0.02~0.05)."}
    )

    # Efficiency bonus reward parameters (Stage 2)
    efficiency_base_bonus: float = field(
        default=0.2,
        metadata={"help": "Base bonus for efficiency reward when model exits early. R = base_bonus - step_penalty * T_actual."}
    )
    efficiency_step_penalty: float = field(
        default=0.01,
        metadata={"help": "Per-step penalty for efficiency reward. R = base_bonus - step_penalty * T_actual."}
    )

    # Forced early exit exploration (Stage 2)
    num_forced_early_exit_samples: int = field(
        default=0,
        metadata={
            "help": "Number of samples (out of G generations per prompt) that will be forced to use "
                    "random shorter laser_steps (2 to T_max). This encourages exploration of early exit "
                    "strategies. Set to 0 to disable. Recommended: 1-2 out of G=8."
        }
    )

    # Relative Norm Perturbation for LASER states (exploration)
    laser_noise_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Relative noise ratio for LASER hidden states during rollout. "
                    "Noise is scaled by ||h|| * noise_ratio / ||ε||, ensuring constant SNR. "
                    "Set to 0.0 to disable. Recommended: 0.01 (1%%) for exploration diversity."
        }
    )

    # Diversity penalty for adjacent LASER hidden states (discourages stagnation)
    diversity_penalty_beta: float = field(
        default=0.0,
        metadata={
            "help": "Penalty coefficient for high cosine similarity between adjacent LASER hidden states. "
                    "Uses squared similarity for stronger penalty on very similar states. "
                    "Formula: R = -beta * mean(cos_sim^2). Normalized by (T-1) to be fair across lengths. "
                    "Set to 0.0 to disable. Recommended: 0.05-0.1."
        }
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    random_seed: Optional[int] = field(default=None)
    nan_blacklist_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to NaN samples blacklist file (jsonl format). Samples in this list will be skipped during training."}
    )
    lengths_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to precomputed sample lengths JSON file. Enables memory-efficient global sorting."}
    )
    fixed_num_of_laser_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Fixed number of LASER tokens per ROI. If None, uses actual ROI token count."}
    )
    max_image_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum image tokens allowed per sample. Samples exceeding this will be skipped. If None, no filtering."}
    )

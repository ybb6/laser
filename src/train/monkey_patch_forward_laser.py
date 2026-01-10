"""
LASER Forward Function Monkey Patching

This module patches the Qwen2.5-VL model's forward function to support DWAL training.
"""

import torch
from typing import Optional, List, Tuple
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
import torch.distributed as dist

# Import DWAL forward functions from separate module
from src.train.forward_dwal import (
    qwen2_5_mixed_modality_forward_laser_dwal,
    qwen2_5_mixed_modality_forward_laser_dwal_time_aware,
)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def replace_qwen2_5_with_mixed_modality_forward_laser(
    dwal=False,
    dwal_time_aware=False
):
    """
    Patch Qwen2.5-VL forward function for DWAL training.

    Args:
        dwal: Enable DWAL (Dynamic Windowed Alignment Loss) mode
        dwal_time_aware: Enable Time-Aware DWAL with spatial decay and temporal ramp
    """
    print("#" * 42)
    if dwal_time_aware:
        print("Activated Time-Aware DWAL (Focusing Mechanism) mode!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_laser_dwal_time_aware
    elif dwal:
        print("Activated DWAL (Dynamic Windowed Alignment Loss) mode!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_laser_dwal
    else:
        raise ValueError("Must enable either dwal=True or dwal_time_aware=True")
    print("#" * 42)


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Custom output class for DWAL training.
    Please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.
    """
    loss: Optional[torch.FloatTensor] = None
    loss_laser: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

    # DWAL metrics
    dwal_confused_ratio: Optional[float] = None
    dwal_laser_end_max_ratio: Optional[float] = None

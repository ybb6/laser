"""
Dynamic Batch Collator for LASER Training

处理 DynamicBatchDataset 产生的动态大小 batch。
执行 padding 并正确合并多模态数据 (pixel_values, image_grid_thw 等)。

与 PackedDataCollator 的区别：
- 输入是独立样本列表（不是打包序列）
- 不需要按 input_lengths 分割
- 更简单更稳健
"""

import torch
from typing import Any, Dict, List

from src.constants import IGNORE_INDEX


def pad_to_multiple(length: int, multiple: int) -> int:
    """将长度向上取整到 multiple 的倍数。"""
    if multiple <= 1:
        return length
    return ((length + multiple - 1) // multiple) * multiple


class DynamicBatchCollator:
    """
    动态 batch 的 Collator。

    Args:
        pad_token_id: 用于填充 input_ids 的 token ID
        pad_to_multiple_of: 将序列长度填充到此值的倍数（Tensor Core 优化）
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int = 8,
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        整理一个 batch 的样本。

        Args:
            batch: DynamicBatchDataset 产生的样本列表
                   每个样本包含: input_ids, attention_mask, labels, laser_tokens,
                                pixel_values, image_grid_thw 等

        Returns:
            整理后的 batch 字典，包含填充后的张量
        """
        if not batch:
            raise ValueError("Empty batch received")

        # 处理 DataLoader batch_size=1 时可能的双重包装
        if len(batch) == 1 and isinstance(batch[0], list):
            batch = batch[0]

        # 收集各字段
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_thw = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_second_per_grid_ts = []
        batch_laser_tokens = []
        batch_indices = []  # 数组索引（用于预计算长度查询）
        batch_question_ids = []  # 原始 question_id（用于稳定标识）
        batch_original_lengths = []  # 记录每个样本的原始长度（padding前）

        for sample in batch:
            if sample is None:
                continue

            keys = sample.keys()

            batch_input_ids.append(sample["input_ids"])
            batch_labels.append(sample["labels"])
            batch_original_lengths.append(sample["input_ids"].size(0))  # 原始长度

            # 图像
            if "pixel_values" in keys:
                batch_pixel_values.append(sample["pixel_values"])
                batch_image_thw.append(sample["image_grid_thw"])

            # 视频
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(sample["pixel_values_videos"])
                batch_video_thw.append(sample["video_grid_thw"])

            # LASER tokens
            if "laser_tokens" in keys:
                batch_laser_tokens.append(sample["laser_tokens"])

            # Qwen2.5 video 的 second_per_grid_ts
            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(sample["second_per_grid_ts"])

            # 分别收集数组索引和原始 question_id
            idx_val = sample.get("_idx", -1)  # 数组索引
            qid_val = sample.get("question_id", -1)  # 原始 question_id
            batch_indices.append(idx_val)
            batch_question_ids.append(qid_val)

        if not batch_input_ids:
            raise ValueError("No valid samples in batch")

        # 计算最大长度并对齐到 multiple
        max_len = max(ids.size(0) for ids in batch_input_ids)
        if self.pad_to_multiple_of > 1:
            max_len = pad_to_multiple(max_len, self.pad_to_multiple_of)

        # Padding
        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            seq_len = input_ids.size(0)
            padding_needed = max_len - seq_len

            # Pad input_ids
            padded_input_ids.append(
                torch.nn.functional.pad(input_ids, (0, padding_needed), value=self.pad_token_id)
            )

            # Pad labels
            padded_labels.append(
                torch.nn.functional.pad(labels, (0, padding_needed), value=IGNORE_INDEX)
            )

            # 创建 attention mask
            attention_mask = torch.ones(max_len, dtype=torch.long)
            attention_mask[seq_len:] = 0
            padded_attention_masks.append(attention_mask)

        # Stack 成 batch tensor
        input_ids = torch.stack(padded_input_ids)
        labels = torch.stack(padded_labels)
        attention_mask = torch.stack(padded_attention_masks)

        # 构建输出字典
        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # indices（数组索引）和 question_ids（原始 ID）
        if batch_indices:
            data_dict["indices"] = batch_indices
        if batch_question_ids:
            data_dict["question_ids"] = batch_question_ids

        # original_lengths (padding前的长度)
        if batch_original_lengths:
            data_dict["original_lengths"] = batch_original_lengths

        # 处理 LASER tokens
        # 每个样本的 laser_tokens 是一个 tensor 列表
        # 展平成单个列表
        if batch_laser_tokens:
            laser_tokens_flat = []
            for sample_laser_tokens in batch_laser_tokens:
                for token_tensor in sample_laser_tokens:
                    # 确保是 tensor
                    if isinstance(token_tensor, torch.Tensor):
                        laser_tokens_flat.append(token_tensor)
                    else:
                        laser_tokens_flat.append(torch.tensor(token_tensor, dtype=torch.int))
            data_dict["laser_tokens"] = laser_tokens_flat

        # 合并图像 pixel values
        if batch_pixel_values:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        # 合并视频 pixel values
        if batch_pixel_video_values:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        # second_per_grid_ts
        if batch_second_per_grid_ts:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

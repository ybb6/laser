"""
Dynamic Batch Collator for LASER Training

Handles dynamic-size batches produced by DynamicBatchDataset.
Performs padding and correctly merges multimodal data (pixel_values, image_grid_thw, etc.).

Differences from PackedDataCollator:
- Input is a list of independent samples (not packed sequences)
- No need to split by input_lengths
- Simpler and more robust
"""

import torch
from typing import Any, Dict, List

from src.constants import IGNORE_INDEX


def pad_to_multiple(length: int, multiple: int) -> int:
    """Round up length to the nearest multiple."""
    if multiple <= 1:
        return length
    return ((length + multiple - 1) // multiple) * multiple


class DynamicBatchCollator:
    """
    Collator for dynamic batches.

    Args:
        pad_token_id: Token ID used for padding input_ids
        pad_to_multiple_of: Pad sequence length to multiples of this value (Tensor Core optimization)
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
        Collate a batch of samples.

        Args:
            batch: List of samples produced by DynamicBatchDataset
                   Each sample contains: input_ids, attention_mask, labels, laser_tokens,
                                        pixel_values, image_grid_thw, etc.

        Returns:
            Collated batch dictionary with padded tensors
        """
        if not batch:
            raise ValueError("Empty batch received")

        # Handle possible double wrapping when DataLoader batch_size=1
        if len(batch) == 1 and isinstance(batch[0], list):
            batch = batch[0]

        # Collect fields
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_thw = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_second_per_grid_ts = []
        batch_laser_tokens = []
        batch_indices = []  # Array indices (for precomputed length lookup)
        batch_question_ids = []  # Original question_id (for stable identification)
        batch_original_lengths = []  # Record original length of each sample (before padding)

        for sample in batch:
            if sample is None:
                continue

            keys = sample.keys()

            batch_input_ids.append(sample["input_ids"])
            batch_labels.append(sample["labels"])
            batch_original_lengths.append(sample["input_ids"].size(0))  # Original length

            # Image
            if "pixel_values" in keys:
                batch_pixel_values.append(sample["pixel_values"])
                batch_image_thw.append(sample["image_grid_thw"])

            # Video
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(sample["pixel_values_videos"])
                batch_video_thw.append(sample["video_grid_thw"])

            # LASER tokens
            if "laser_tokens" in keys:
                batch_laser_tokens.append(sample["laser_tokens"])

            # Qwen2.5 video's second_per_grid_ts
            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(sample["second_per_grid_ts"])

            # Collect array index and original question_id separately
            idx_val = sample.get("_idx", -1)  # Array index
            qid_val = sample.get("question_id", -1)  # Original question_id
            batch_indices.append(idx_val)
            batch_question_ids.append(qid_val)

        if not batch_input_ids:
            raise ValueError("No valid samples in batch")

        # Calculate max length and align to multiple
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

            # Create attention mask
            attention_mask = torch.ones(max_len, dtype=torch.long)
            attention_mask[seq_len:] = 0
            padded_attention_masks.append(attention_mask)

        # Stack into batch tensor
        input_ids = torch.stack(padded_input_ids)
        labels = torch.stack(padded_labels)
        attention_mask = torch.stack(padded_attention_masks)

        # Build output dictionary
        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # indices (array indices) and question_ids (original IDs)
        if batch_indices:
            data_dict["indices"] = batch_indices
        if batch_question_ids:
            data_dict["question_ids"] = batch_question_ids

        # original_lengths (length before padding)
        if batch_original_lengths:
            data_dict["original_lengths"] = batch_original_lengths

        # Process LASER tokens
        # Each sample's laser_tokens is a list of tensors
        # Flatten into a single list
        if batch_laser_tokens:
            laser_tokens_flat = []
            for sample_laser_tokens in batch_laser_tokens:
                for token_tensor in sample_laser_tokens:
                    # Ensure it's a tensor
                    if isinstance(token_tensor, torch.Tensor):
                        laser_tokens_flat.append(token_tensor)
                    else:
                        laser_tokens_flat.append(torch.tensor(token_tensor, dtype=torch.int))
            data_dict["laser_tokens"] = laser_tokens_flat

        # Merge image pixel values
        if batch_pixel_values:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        # Merge video pixel values
        if batch_pixel_video_values:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        # second_per_grid_ts
        if batch_second_per_grid_ts:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

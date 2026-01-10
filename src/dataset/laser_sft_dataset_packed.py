"""
Laser SFT Dataset

This file contains:
1. IterableSupervisedDatasetLaser - Map-style Dataset for LASER training

Adapted from InternVL [https://github.com/OpenGVLab/InternVL/tree/main]
"""

import copy
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, llava_to_openai_laser, llava_to_openai_dwal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class IterableSupervisedDatasetLaser(Dataset):
    """
    Map-style Dataset for LASER training.

    支持两种模式:
    - bboxes 模式: 从边界框提取视觉 token 索引
    - reasoning_chain 模式 (DWAL): 使用实际推理链文本

    Args:
        data_path: 数据 JSON 文件路径
        image_folder: 图像文件夹路径
        processor: Tokenizer/Processor
        data_args: 数据参数
        ds_name: 数据集名称
        model_id: 模型标识
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        ds_name: str,
        model_id: str,
    ):
        super().__init__()

        # 加载原始数据
        if isinstance(data_path, str):
            self.raw_data = json.load(open(data_path, "r"))
        else:
            self.raw_data = data_path

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.image_folder = image_folder
        self.ds_name = ds_name

        # 图像处理参数
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

        # NaN blacklist (可选)
        self.nan_blacklist = set()
        nan_blacklist_path = getattr(data_args, 'nan_blacklist_path', None)
        if nan_blacklist_path and os.path.exists(nan_blacklist_path):
            if get_rank() == 0:
                logger.info(f"[{self.ds_name}] Loading NaN blacklist from: {nan_blacklist_path}")
            try:
                with open(nan_blacklist_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        qid = entry.get('question_id', -1)
                        if qid != -1:
                            self.nan_blacklist.add(qid)
                if get_rank() == 0:
                    logger.info(f"[{self.ds_name}] Loaded {len(self.nan_blacklist)} NaN samples to blacklist")
            except Exception as e:
                logger.warning(f"[{self.ds_name}] Failed to load NaN blacklist: {e}")

        if get_rank() == 0:
            logger.info(f"[{self.ds_name}] Loaded {len(self.raw_data)} samples")

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, i: int) -> Optional[Dict[str, torch.Tensor]]:
        """获取单个样本。"""
        sources = self.raw_data[i]

        # 检查 NaN blacklist
        question_id = sources.get('question_id', -1)
        if question_id != -1 and question_id in self.nan_blacklist:
            return None

        processor = self.processor
        is_video = False
        videos = None
        grid_key = "image_grid_thw"
        pixel_key = "pixel_values"

        # 加载图像
        image_files = sources["image"]
        if isinstance(image_files, str):
            image_files = [image_files]

        images = []
        for image_file in image_files:
            if not os.path.exists(image_file):
                if not image_file.startswith("http"):
                    image_file = os.path.join(self.image_folder, image_file)
            images.append(get_image_info(
                image_file,
                self.image_min_pixel,
                self.image_max_pixel,
                self.image_resized_w,
                self.image_resized_h
            ))

        # 提取 LASER tokens
        image_grid_thw = processor(
            text=[""],
            images=images,
            videos=videos,
            padding=False,
            do_resize=False,
            return_tensors='pt'
        )['image_grid_thw']

        # 支持 bboxes 和 reasoning_chain 两种模式
        if 'bboxes' in sources:
            laser_token_idxs_list = self.bbox_to_token_idxs(sources['bboxes'], image_grid_thw)
            sources = copy.deepcopy(llava_to_openai_laser(
                sources['conversations'],
                is_video=is_video,
                laser_token_idxs_list=laser_token_idxs_list,
                fixed_num_of_laser_tokens=self.data_args.fixed_num_of_laser_tokens
            ))
        elif 'reasoning_chain' in sources:
            # DWAL 模式
            reasoning_chain_text = " ".join(sources['reasoning_chain'])
            laser_token_idxs_list = self.reasoning_chain_to_token_idxs(sources['reasoning_chain'])
            sources = copy.deepcopy(llava_to_openai_dwal(
                sources['conversations'],
                reasoning_chain_text=reasoning_chain_text,
                is_video=is_video
            ))
        else:
            laser_token_idxs_list = [[]]
            sources = copy.deepcopy(llava_to_openai_laser(
                sources['conversations'],
                is_video=is_video,
                laser_token_idxs_list=laser_token_idxs_list,
                fixed_num_of_laser_tokens=self.data_args.fixed_num_of_laser_tokens
            ))

        # 构建 input_ids 和 labels
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []

        # System message
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(
                system_message,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        # 处理对话轮次
        for j in range(0, len(sources), 2):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input_str = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
                f"{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            )
            gpt_response_str = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input_str:
                inputs = processor(
                    text=[user_input_str],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors='pt'
                )
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            else:
                prompt_input_ids = processor.tokenizer(
                    user_input_str,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors='pt'
                )['input_ids']

            response_input_ids = processor.tokenizer(
                gpt_response_str,
                add_special_tokens=False,
                padding=False,
                return_tensors='pt'
            )['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat([
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                response_input_ids.squeeze(0),
            ], dim=0)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # 合并所有轮次
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        # 处理 laser_tokens
        laser_tokens = [torch.tensor(group, dtype=torch.int) for group in laser_token_idxs_list]

        # 构建返回字典
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            laser_tokens=laser_tokens,
            input_lengths=torch.tensor([input_ids.size(0)]),
            question_id=self.raw_data[i].get('question_id', -1),
        )

        if pixel_key and grid_key and all_pixel_values:
            data_dict[pixel_key] = torch.cat(all_pixel_values, dim=0)
            data_dict[grid_key] = torch.cat(all_image_grid_thw, dim=0)

        return data_dict

    def bbox_to_token_idxs(
        self,
        bboxes: List,
        image_grid_thw: torch.Tensor
    ) -> List[List[int]]:
        """
        将边界框转换为视觉 token 索引。

        仅适用于 Qwen-VL 系列模型。
        """
        _, h, w = image_grid_thw[0].tolist()
        token_idxs = []

        for bbox in bboxes:
            x0, y0, x1, y1 = bbox

            # 缩放到 14x14 网格
            x0_grid = max(0, min(int(np.floor(x0 * w)), w - 1))
            x1_grid = max(0, min(int(np.ceil(x1 * w)), w))
            y0_grid = max(0, min(int(np.floor(y0 * h)), h - 1))
            y1_grid = max(0, min(int(np.ceil(y1 * h)), h))

            # 映射到 28x28 网格
            x0_token = x0_grid // 2
            x1_token = (x1_grid + 1) // 2
            y0_token = y0_grid // 2
            y1_token = (y1_grid + 1) // 2

            H2, W2 = h // 2, w // 2

            idxs = [
                int(yy * W2 + xx)
                for yy in range(y0_token, y1_token)
                for xx in range(x0_token, x1_token)
            ]
            token_idxs.append(idxs)

        return token_idxs

    def reasoning_chain_to_token_idxs(self, reasoning_chain: List[str]) -> List[List[int]]:
        """
        将 reasoning_chain 转换为 token 索引列表 (DWAL 模式)。
        """
        text = " ".join(reasoning_chain)
        token_ids = self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [list(range(len(token_ids)))]

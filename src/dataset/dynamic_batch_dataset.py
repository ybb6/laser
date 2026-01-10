"""
Dynamic Batch Dataset for LASER Training

基于 "矩形显存面积" 约束的动态批处理策略：
    (batch_size × max_seq_len) <= TOKEN_CAP

核心设计原则：
1. Buffer 只操作索引和长度，不存储完整数据
2. 长度必须预计算（通过 lengths_path 加载）
3. 数据只在最终 yield 时加载一次
4. 支持 Resume 时跳过已训练的 batches

流程:
1. 宏观缓冲: 随机 shuffle 所有索引
2. 分片: 按 rank/worker 划分索引
3. 微观排序: 按长度排序
4. 贪心封包: 双约束 (TOKEN_CAP + MAX_BS)
5. Batch 打乱: shuffle batch 顺序
6. 跳过已训练的 batches (Resume 时)
7. 加载数据: yield 时才加载真实数据
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Dataset, IterableDataset

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


class DynamicBatchDataset(IterableDataset):
    """
    动态批处理的 IterableDataset。

    核心特性：
    - 所有排序/打包操作只在 (idx, length) 对上进行
    - 数据只在 yield 时加载一次
    - 必须提供预计算的长度文件
    - 支持 Resume 时跳过已训练的 batches

    Args:
        base_dataset: 底层 Map-style Dataset (支持 __getitem__ 和 __len__)
        token_cap: 显存硬约束 - batch_size × max_len <= token_cap
        max_batch_size: 数量硬约束 - 每个 batch 最多样本数
        lengths_path: 预计算长度 JSON 文件路径 (必须提供)
        seed: 随机种子
        data_rank: 当前进程 rank (分布式训练)
        data_world_size: 总进程数
        num_workers: DataLoader 的 worker 数量 (用于预计算跳过量)
        resume_step: Resume 时的 global_step (0 表示不跳过)
        gradient_accumulation_steps: 梯度累积步数 (用于计算实际消耗的 batch 数)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        token_cap: int = 10240,
        max_batch_size: int = 16,
        lengths_path: str = None,
        seed: int = 42,
        data_rank: int = 0,
        data_world_size: int = 1,
        num_workers: int = 1,
        resume_step: int = 0,
        gradient_accumulation_steps: int = 1,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.token_cap = token_cap
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_workers = max(1, num_workers)  # 至少 1
        self.total_samples = len(base_dataset)

        # 加载预计算长度 (必须存在)
        if not lengths_path or not os.path.exists(lengths_path):
            raise ValueError(
                f"lengths_path is required and must exist. Got: {lengths_path}\n"
                f"Run scripts/precompute_lengths.py first to generate the lengths file."
            )

        if get_rank() == 0:
            logger.info(f"Loading precomputed lengths from {lengths_path}")

        with open(lengths_path, 'r') as f:
            lengths_dict = json.load(f)

        # JSON keys 是字符串，转成 int
        self.lengths = {int(k): v for k, v in lengths_dict.items()}

        # 验证长度数量匹配
        if len(self.lengths) != self.total_samples:
            logger.warning(
                f"Length mismatch: lengths file has {len(self.lengths)} entries, "
                f"but dataset has {self.total_samples} samples. "
                f"This may cause issues."
            )

        # 计算每个 worker 需要跳过的 batch 数 (用于 Resume)
        self.skip_batches_per_worker: Dict[int, int] = {}
        if resume_step > 0:
            # 每个 rank 消耗的 batch 数 = global_step * gradient_accumulation_steps
            batches_consumed_per_rank = resume_step * gradient_accumulation_steps
            self._compute_skip_for_resume(batches_consumed_per_rank)

        if get_rank() == 0:
            logger.info(
                f"DynamicBatchDataset initialized: "
                f"total_samples={self.total_samples}, "
                f"token_cap={token_cap}, "
                f"max_batch_size={max_batch_size}, "
                f"num_workers={self.num_workers}, "
                f"resume_step={resume_step}"
            )
            if self.skip_batches_per_worker:
                logger.info(f"Skip batches per worker: {self.skip_batches_per_worker}")

    def _compute_skip_for_resume(self, batches_consumed_per_rank: int):
        """
        计算每个 worker 需要跳过的 batch 数。

        原理：
        1. 预计算当前 rank 下每个 worker 会生成多少 batches
        2. 模拟 DataLoader 的 round-robin 取数逻辑
        3. 计算出每个 worker 已经贡献了多少 batches

        Args:
            batches_consumed_per_rank: 当前 rank 已消耗的 batch 数
        """
        # 预计算当前 rank 下每个 worker 的 batch 数
        batches_per_worker = self._precompute_batches_for_rank()

        if get_rank() == 0:
            logger.info(f"[Resume] Precomputed batches per worker: {batches_per_worker}")
            logger.info(f"[Resume] Batches consumed per rank: {batches_consumed_per_rank}")

        # 模拟 round-robin 取数，计算每个 worker 贡献了多少
        self.skip_batches_per_worker = self._simulate_round_robin(
            batches_consumed_per_rank,
            batches_per_worker
        )

    def _precompute_batches_for_rank(self) -> Dict[int, int]:
        """
        预计算当前 rank 下每个 worker 会生成多少 batches。

        使用与 __iter__ 相同的逻辑，但只计算数量，不加载数据。

        Returns:
            {local_worker_id: batch_count}
        """
        batches_per_worker = {}

        # 全局 RNG (与 __iter__ 保持一致)
        global_rng = np.random.default_rng(seed=self.seed)
        all_indices = list(range(self.total_samples))
        global_rng.shuffle(all_indices)

        # 计算当前 rank 下每个 worker 的 batch 数
        for local_worker_id in range(self.num_workers):
            global_worker_id = self.data_rank * self.num_workers + local_worker_id
            total_workers = self.data_world_size * self.num_workers

            # 分片
            my_indices = all_indices[global_worker_id::total_workers]

            # 获取长度，过滤无效样本
            indices_with_lengths = []
            for idx in my_indices:
                length = self.lengths.get(idx, 0)
                if length > 0:
                    indices_with_lengths.append((idx, length))

            # 排序
            indices_with_lengths.sort(key=lambda x: x[1])

            # 封包
            batches = self._greedy_pack_indices(indices_with_lengths)
            batches_per_worker[local_worker_id] = len(batches)

        return batches_per_worker

    def _simulate_round_robin(
        self,
        total_consumed: int,
        batches_per_worker: Dict[int, int]
    ) -> Dict[int, int]:
        """
        模拟 DataLoader 的 round-robin 取数逻辑。

        PyTorch DataLoader 对 IterableDataset + num_workers > 0 使用 round-robin:
        依次从 worker 0, 1, 2, ..., 0, 1, 2, ... 取数据。

        Args:
            total_consumed: 总共消耗的 batch 数
            batches_per_worker: 每个 worker 的 batch 总数

        Returns:
            {local_worker_id: skip_count}  每个 worker 需要跳过的数量
        """
        worker_ids = sorted(batches_per_worker.keys())
        remaining = {wid: batches_per_worker[wid] for wid in worker_ids}
        contributed = {wid: 0 for wid in worker_ids}

        consumed = 0
        while consumed < total_consumed:
            made_progress = False
            for wid in worker_ids:
                if consumed >= total_consumed:
                    break
                if remaining[wid] > 0:
                    contributed[wid] += 1
                    remaining[wid] -= 1
                    consumed += 1
                    made_progress = True

            if not made_progress:
                # 所有 worker 都用完了
                if get_rank() == 0:
                    logger.warning(
                        f"[Resume] All workers exhausted before reaching target. "
                        f"Consumed {consumed}/{total_consumed} batches."
                    )
                break

        return contributed

    def _greedy_pack_indices(
        self,
        sorted_indices_with_lengths: List[tuple]
    ) -> List[List[int]]:
        """
        贪心封包算法 - 双约束。

        约束:
        1. 显存约束: (count + 1) × new_max_len <= TOKEN_CAP
        2. 数量约束: count + 1 <= MAX_BS

        Args:
            sorted_indices_with_lengths: [(idx, length), ...] 按长度升序排列

        Returns:
            [[idx1, idx2, ...], [idx3, idx4, ...], ...]  索引 batch 列表
        """
        if not sorted_indices_with_lengths:
            return []

        batches = []
        current_batch = []
        current_max_len = 0

        for idx, length in sorted_indices_with_lengths:
            # 跳过无效样本
            if length <= 0:
                continue

            new_max_len = max(current_max_len, length)
            new_count = len(current_batch) + 1

            # 双约束检查
            memory_ok = (new_count * new_max_len) <= self.token_cap
            count_ok = new_count <= self.max_batch_size

            if memory_ok and count_ok:
                # 可以加入当前 batch
                current_batch.append(idx)
                current_max_len = new_max_len
            else:
                # 开始新 batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = length

        # 别忘了最后一个 batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        """
        迭代生成动态大小的 batch（无限循环模式）。

        流程:
        1. 获取 worker 信息
        2. 无限循环:
           a. 获取所有索引，shuffle (宏观随机性，每 epoch 不同)
           b. 按 rank/worker 分片
           c. 获取长度，过滤无效样本
           d. 按长度排序 (微观排序)
           e. 贪心封包
           f. Shuffle batches
           g. 跳过已训练的 batches (仅第一个 epoch)
           h. 加载数据并 yield
           i. epoch 结束，打印日志，继续下一轮
        """
        # === Step 1: 获取 worker 信息 ===
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # 组合 distributed rank 和 worker id
        global_worker_id = self.data_rank * num_workers + worker_id
        total_workers = self.data_world_size * num_workers

        # 跳过已训练的 batches (仅第一个 epoch 需要)
        skip_count = self.skip_batches_per_worker.get(worker_id, 0)
        if skip_count > 0 and global_worker_id == 0:
            logger.info(
                f"[DynamicBatch] Worker {global_worker_id}: "
                f"will skip first {skip_count} batches (resume)"
            )

        # 全局计数器
        total_batches_yielded_global = 0
        epoch = 0

        # === 无限循环 ===
        while True:
            # 每个 epoch 使用不同的 seed，保证跨 epoch 的随机性
            epoch_seed = self.seed + epoch * 1000

            # 全局 RNG: 所有 worker 使用相同 seed，保证 shuffle 结果一致
            global_rng = np.random.default_rng(seed=epoch_seed)
            # Worker-specific RNG: 用于 batch 顺序 shuffle，增加随机性
            worker_rng = np.random.default_rng(seed=epoch_seed + global_worker_id)

            # === Step 2: 获取所有索引并 shuffle (全局一致) ===
            all_indices = list(range(self.total_samples))
            global_rng.shuffle(all_indices)

            # === Step 3: 按 worker 分片 ===
            my_indices = all_indices[global_worker_id::total_workers]

            if global_worker_id == 0 and epoch == 0:
                logger.info(
                    f"[DynamicBatch] Worker {global_worker_id}/{total_workers}: "
                    f"processing {len(my_indices)}/{self.total_samples} samples"
                )

            # === Step 4: 获取长度，过滤无效样本 ===
            indices_with_lengths = []
            for idx in my_indices:
                length = self.lengths.get(idx, 0)
                if length > 0:
                    indices_with_lengths.append((idx, length))

            if global_worker_id == 0 and epoch == 0:
                logger.info(
                    f"[DynamicBatch] Worker {global_worker_id}: "
                    f"{len(indices_with_lengths)} valid samples"
                )

            # === Step 5: 按长度排序 ===
            indices_with_lengths.sort(key=lambda x: x[1])

            # === Step 6: 贪心封包 ===
            index_batches = self._greedy_pack_indices(indices_with_lengths)

            if global_worker_id == 0 and epoch == 0:
                logger.info(
                    f"[DynamicBatch] Worker {global_worker_id}: "
                    f"packed into {len(index_batches)} batches"
                )

            # === Step 7: Shuffle batches (worker-specific 顺序) ===
            worker_rng.shuffle(index_batches)

            # === Step 8: 加载数据并 yield ===
            epoch_batches_yielded = 0
            epoch_batches_skipped = 0

            for batch_idx, idx_batch in enumerate(index_batches):
                # 跳过已训练的 batches (仅第一个 epoch)
                if epoch == 0 and batch_idx < skip_count:
                    epoch_batches_skipped += 1
                    continue

                batch = []
                for idx in idx_batch:
                    try:
                        sample = self.base_dataset[idx]
                        if sample is not None:
                            sample['_idx'] = idx  # 添加索引用于调试
                            batch.append(sample)
                    except Exception as e:
                        logger.warning(f"[DynamicBatch] Failed to load sample {idx}: {e}")
                        continue

                if batch:
                    epoch_batches_yielded += 1
                    total_batches_yielded_global += 1
                    yield batch

            # === Epoch 结束，打印日志 ===
            if global_worker_id == 0:
                logger.info(
                    f"[DynamicBatch] Worker {global_worker_id}: "
                    f"epoch {epoch} exhausted after {total_batches_yielded_global} total batches "
                    f"(this epoch: {epoch_batches_yielded} yielded, {epoch_batches_skipped} skipped), "
                    f"cycling to epoch {epoch + 1}..."
                )

            epoch += 1


def make_dynamic_batch_data_module_laser(
    model_id: str,
    processor,
    data_args,
    training_args,
    resume_step: int = 0,
):
    """
    创建动态批处理的数据模块。

    Args:
        model_id: 模型标识
        processor: Tokenizer/Processor
        data_args: 数据参数
        training_args: 训练参数
        resume_step: Resume 时的 global_step (0 表示不跳过)

    Returns:
        dict: {train_dataset, eval_dataset, data_collator}, total_data_len
    """
    from .laser_sft_dataset_packed import IterableSupervisedDatasetLaser
    from .dynamic_batch_collator import DynamicBatchCollator

    # 获取分布式信息
    if is_dist_avail_and_initialized():
        data_rank = dist.get_rank()
        data_world_size = dist.get_world_size()
    else:
        data_rank = 0
        data_world_size = 1

    # 加载 meta_data
    meta_data = json.load(open(data_args.data_path))

    # 为每个数据源创建 Dataset
    datasets = []
    total_data_len = 0

    for meta in meta_data:
        ds = IterableSupervisedDatasetLaser(
            data_path=meta['data_path'],
            image_folder=meta['image_folder'],
            ds_name=meta['ds_name'],
            processor=processor,
            data_args=data_args,
            model_id=model_id,
        )
        datasets.append(ds)
        total_data_len += len(ds)

        if get_rank() == 0:
            logger.info(f"Loaded {len(ds)} samples from {meta['ds_name']}")

    # 合并成 ConcatDataset
    if len(datasets) == 1:
        base_dataset = datasets[0]
    else:
        base_dataset = ConcatDataset(datasets)

    if get_rank() == 0:
        logger.info(f"Total samples: {total_data_len}")

    # 获取 lengths_path
    lengths_path = getattr(data_args, 'lengths_path', None)

    # 获取 num_workers 和 gradient_accumulation_steps
    num_workers = getattr(training_args, 'dataloader_num_workers', 0)
    if num_workers == 0:
        num_workers = 1  # DataLoader with num_workers=0 runs in main process
    gradient_accumulation_steps = getattr(training_args, 'gradient_accumulation_steps', 1)

    if get_rank() == 0 and resume_step > 0:
        logger.info(
            f"[Resume] resume_step={resume_step}, "
            f"num_workers={num_workers}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}"
        )

    # 创建 DynamicBatchDataset
    dynamic_dataset = DynamicBatchDataset(
        base_dataset=base_dataset,
        token_cap=training_args.dynamic_batch_token_cap,
        max_batch_size=training_args.dynamic_batch_max_bs,
        lengths_path=lengths_path,
        seed=data_args.random_seed or 42,
        data_rank=data_rank,
        data_world_size=data_world_size,
        num_workers=num_workers,
        resume_step=resume_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # 创建 Collator
    data_collator = DynamicBatchCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        pad_to_multiple_of=8,  # Tensor Core 优化
    )

    if get_rank() == 0:
        logger.info(
            f"DynamicBatch config: token_cap={training_args.dynamic_batch_token_cap}, "
            f"max_bs={training_args.dynamic_batch_max_bs}"
        )

    return dict(
        train_dataset=dynamic_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    ), total_data_len

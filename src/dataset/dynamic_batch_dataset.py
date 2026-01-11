"""
Dynamic Batch Dataset for LASER Training

Dynamic batching strategy based on "rectangular memory area" constraint:
    (batch_size × max_seq_len) <= TOKEN_CAP

Core design principles:
1. Buffer only operates on indices and lengths, does not store full data
2. Lengths must be precomputed (loaded via lengths_path)
3. Data is loaded only once at final yield
4. Supports skipping already trained batches during Resume

Flow:
1. Macro buffer: randomly shuffle all indices
2. Sharding: partition indices by rank/worker
3. Micro sorting: sort by length
4. Greedy packing: dual constraints (TOKEN_CAP + MAX_BS)
5. Batch shuffle: shuffle batch order
6. Skip already trained batches (during Resume)
7. Load data: load real data only at yield time
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
    IterableDataset for dynamic batching.

    Core features:
    - All sorting/packing operations only work on (idx, length) pairs
    - Data is loaded only once at yield time
    - Precomputed lengths file must be provided
    - Supports skipping already trained batches during Resume

    Args:
        base_dataset: Underlying Map-style Dataset (supports __getitem__ and __len__)
        token_cap: Memory hard constraint - batch_size × max_len <= token_cap
        max_batch_size: Count hard constraint - max samples per batch
        lengths_path: Precomputed lengths JSON file path (required)
        seed: Random seed
        data_rank: Current process rank (distributed training)
        data_world_size: Total number of processes
        num_workers: Number of DataLoader workers (for precomputing skip count)
        resume_step: global_step during Resume (0 means no skipping)
        gradient_accumulation_steps: Gradient accumulation steps (for calculating actual batches consumed)
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
        self.num_workers = max(1, num_workers)  # At least 1
        self.total_samples = len(base_dataset)

        # Load precomputed lengths (must exist)
        if not lengths_path or not os.path.exists(lengths_path):
            raise ValueError(
                f"lengths_path is required and must exist. Got: {lengths_path}\n"
                f"Run scripts/precompute_lengths.py first to generate the lengths file."
            )

        if get_rank() == 0:
            logger.info(f"Loading precomputed lengths from {lengths_path}")

        with open(lengths_path, 'r') as f:
            lengths_dict = json.load(f)

        # JSON keys are strings, convert to int
        self.lengths = {int(k): v for k, v in lengths_dict.items()}

        # Verify length count matches
        if len(self.lengths) != self.total_samples:
            logger.warning(
                f"Length mismatch: lengths file has {len(self.lengths)} entries, "
                f"but dataset has {self.total_samples} samples. "
                f"This may cause issues."
            )

        # Calculate batches to skip per worker (for Resume)
        self.skip_batches_per_worker: Dict[int, int] = {}
        if resume_step > 0:
            # Batches consumed per rank = global_step * gradient_accumulation_steps
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
        Calculate batches to skip per worker.

        Principle:
        1. Precompute how many batches each worker under current rank will generate
        2. Simulate DataLoader's round-robin fetching logic
        3. Calculate how many batches each worker has already contributed

        Args:
            batches_consumed_per_rank: Batches already consumed by current rank
        """
        # Precompute batch count for each worker under current rank
        batches_per_worker = self._precompute_batches_for_rank()

        if get_rank() == 0:
            logger.info(f"[Resume] Precomputed batches per worker: {batches_per_worker}")
            logger.info(f"[Resume] Batches consumed per rank: {batches_consumed_per_rank}")

        # Simulate round-robin fetching, calculate how many each worker contributed
        self.skip_batches_per_worker = self._simulate_round_robin(
            batches_consumed_per_rank,
            batches_per_worker
        )

    def _precompute_batches_for_rank(self) -> Dict[int, int]:
        """
        Precompute how many batches each worker under current rank will generate.

        Uses the same logic as __iter__, but only counts, does not load data.

        Returns:
            {local_worker_id: batch_count}
        """
        batches_per_worker = {}

        # Global RNG (consistent with __iter__)
        global_rng = np.random.default_rng(seed=self.seed)
        all_indices = list(range(self.total_samples))
        global_rng.shuffle(all_indices)

        # Calculate batch count for each worker under current rank
        for local_worker_id in range(self.num_workers):
            global_worker_id = self.data_rank * self.num_workers + local_worker_id
            total_workers = self.data_world_size * self.num_workers

            # Sharding
            my_indices = all_indices[global_worker_id::total_workers]

            # Get lengths, filter invalid samples
            indices_with_lengths = []
            for idx in my_indices:
                length = self.lengths.get(idx, 0)
                if length > 0:
                    indices_with_lengths.append((idx, length))

            # Sort
            indices_with_lengths.sort(key=lambda x: x[1])

            # Packing
            batches = self._greedy_pack_indices(indices_with_lengths)
            batches_per_worker[local_worker_id] = len(batches)

        return batches_per_worker

    def _simulate_round_robin(
        self,
        total_consumed: int,
        batches_per_worker: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Simulate DataLoader's round-robin fetching logic.

        PyTorch DataLoader uses round-robin for IterableDataset + num_workers > 0:
        fetches from worker 0, 1, 2, ..., 0, 1, 2, ... in sequence.

        Args:
            total_consumed: Total batches consumed
            batches_per_worker: Total batch count for each worker

        Returns:
            {local_worker_id: skip_count}  Number of batches each worker needs to skip
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
                # All workers exhausted
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
        Greedy packing algorithm - dual constraints.

        Constraints:
        1. Memory constraint: (count + 1) × new_max_len <= TOKEN_CAP
        2. Count constraint: count + 1 <= MAX_BS

        Args:
            sorted_indices_with_lengths: [(idx, length), ...] sorted by length ascending

        Returns:
            [[idx1, idx2, ...], [idx3, idx4, ...], ...]  List of index batches
        """
        if not sorted_indices_with_lengths:
            return []

        batches = []
        current_batch = []
        current_max_len = 0

        for idx, length in sorted_indices_with_lengths:
            # Skip invalid samples
            if length <= 0:
                continue

            new_max_len = max(current_max_len, length)
            new_count = len(current_batch) + 1

            # Dual constraint check
            memory_ok = (new_count * new_max_len) <= self.token_cap
            count_ok = new_count <= self.max_batch_size

            if memory_ok and count_ok:
                # Can add to current batch
                current_batch.append(idx)
                current_max_len = new_max_len
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = length

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        """
        Iterate and generate dynamic-size batches (infinite loop mode).

        Flow:
        1. Get worker info
        2. Infinite loop:
           a. Get all indices, shuffle (macro randomness, different per epoch)
           b. Shard by rank/worker
           c. Get lengths, filter invalid samples
           d. Sort by length (micro sorting)
           e. Greedy packing
           f. Shuffle batches
           g. Skip already trained batches (first epoch only)
           h. Load data and yield
           i. Epoch ends, print log, continue to next round
        """
        # === Step 1: Get worker info ===
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Combine distributed rank and worker id
        global_worker_id = self.data_rank * num_workers + worker_id
        total_workers = self.data_world_size * num_workers

        # Skip already trained batches (first epoch only)
        skip_count = self.skip_batches_per_worker.get(worker_id, 0)
        if skip_count > 0 and global_worker_id == 0:
            logger.info(
                f"[DynamicBatch] Worker {global_worker_id}: "
                f"will skip first {skip_count} batches (resume)"
            )

        # Global counters
        total_batches_yielded_global = 0
        epoch = 0

        # === Infinite loop ===
        while True:
            # Use different seed per epoch to ensure cross-epoch randomness
            epoch_seed = self.seed + epoch * 1000

            # Global RNG: all workers use same seed to ensure consistent shuffle results
            global_rng = np.random.default_rng(seed=epoch_seed)
            # Worker-specific RNG: for batch order shuffle, adds randomness
            worker_rng = np.random.default_rng(seed=epoch_seed + global_worker_id)

            # === Step 2: Get all indices and shuffle (globally consistent) ===
            all_indices = list(range(self.total_samples))
            global_rng.shuffle(all_indices)

            # === Step 3: Shard by worker ===
            my_indices = all_indices[global_worker_id::total_workers]

            if global_worker_id == 0 and epoch == 0:
                logger.info(
                    f"[DynamicBatch] Worker {global_worker_id}/{total_workers}: "
                    f"processing {len(my_indices)}/{self.total_samples} samples"
                )

            # === Step 4: Get lengths, filter invalid samples ===
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

            # === Step 5: Sort by length ===
            indices_with_lengths.sort(key=lambda x: x[1])

            # === Step 6: Greedy packing ===
            index_batches = self._greedy_pack_indices(indices_with_lengths)

            if global_worker_id == 0 and epoch == 0:
                logger.info(
                    f"[DynamicBatch] Worker {global_worker_id}: "
                    f"packed into {len(index_batches)} batches"
                )

            # === Step 7: Shuffle batches (worker-specific order) ===
            worker_rng.shuffle(index_batches)

            # === Step 8: Load data and yield ===
            epoch_batches_yielded = 0
            epoch_batches_skipped = 0

            for batch_idx, idx_batch in enumerate(index_batches):
                # Skip already trained batches (first epoch only)
                if epoch == 0 and batch_idx < skip_count:
                    epoch_batches_skipped += 1
                    continue

                batch = []
                for idx in idx_batch:
                    try:
                        sample = self.base_dataset[idx]
                        if sample is not None:
                            sample['_idx'] = idx  # Add index for debugging
                            batch.append(sample)
                    except Exception as e:
                        logger.warning(f"[DynamicBatch] Failed to load sample {idx}: {e}")
                        continue

                if batch:
                    epoch_batches_yielded += 1
                    total_batches_yielded_global += 1
                    yield batch

            # === Epoch ends, print log ===
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
    Create dynamic batch data module.

    Args:
        model_id: Model identifier
        processor: Tokenizer/Processor
        data_args: Data arguments
        training_args: Training arguments
        resume_step: global_step during Resume (0 means no skipping)

    Returns:
        dict: {train_dataset, eval_dataset, data_collator}, total_data_len
    """
    from .laser_sft_dataset_packed import IterableSupervisedDatasetLaser
    from .dynamic_batch_collator import DynamicBatchCollator

    # Get distributed info
    if is_dist_avail_and_initialized():
        data_rank = dist.get_rank()
        data_world_size = dist.get_world_size()
    else:
        data_rank = 0
        data_world_size = 1

    # Load meta_data
    meta_data = json.load(open(data_args.data_path))

    # Create Dataset for each data source
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

    # Merge into ConcatDataset
    if len(datasets) == 1:
        base_dataset = datasets[0]
    else:
        base_dataset = ConcatDataset(datasets)

    if get_rank() == 0:
        logger.info(f"Total samples: {total_data_len}")

    # Get lengths_path
    lengths_path = getattr(data_args, 'lengths_path', None)

    # Get num_workers and gradient_accumulation_steps
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

    # Create DynamicBatchDataset
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

    # Create Collator
    data_collator = DynamicBatchCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        pad_to_multiple_of=8,  # Tensor Core optimization
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

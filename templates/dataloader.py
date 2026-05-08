"""
Production DataLoader — handles single GPU and distributed setups.
"""
import math
import multiprocessing
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


def get_num_workers(override=None):
    if override is not None:
        return override
    gpus = max(torch.cuda.device_count(), 1)
    return min(4 * gpus, multiprocessing.cpu_count())


def build_dataloader(
    dataset,
    batch_size: int,
    is_train: bool,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = None,
    seed: int = 42,
    collate_fn=None,
):
    """
    Build a production-grade DataLoader.
    Handles both single-GPU and distributed setups.
    """
    is_distributed = world_size > 1
    is_iterable = isinstance(dataset, IterableDataset)

    sampler = None
    if is_distributed and not is_iterable:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            seed=seed,
            drop_last=is_train,
        )

    nw = get_num_workers(num_workers)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train and not is_iterable),
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        drop_last=is_train,
        collate_fn=collate_fn,
    )

    return loader, sampler


class DataPrefetcher:
    """
    Async data prefetcher: loads next batch to GPU while current batch is processed.
    Use instead of raw DataLoader iteration for maximum throughput.

    Usage:
        prefetcher = DataPrefetcher(loader, device)
        for batch in prefetcher:
            # batch is already on device
            loss = model(batch["input"])
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self._iter = None
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        # Mark tensors as safe to use on current stream
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                v.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch

    def _preload(self):
        try:
            raw = next(self._iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }

    def __len__(self):
        return len(self.loader)


def benchmark_dataloader(loader, device, n_batches=50):
    """
    Measure DataLoader throughput. Run before training to detect bottlenecks.
    Target: < 50ms/batch for GPU training.
    """
    import time
    times = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        t0 = time.perf_counter()
        _ = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    print(f"DataLoader benchmark: {avg_ms:.1f}ms/batch over {len(times)} batches")
    if avg_ms > 50:
        print("WARNING: DataLoader is slow. Check num_workers, pin_memory, and storage speed.")
    return avg_ms

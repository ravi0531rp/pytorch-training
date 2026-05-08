# DataLoader: Production Configuration

## The 6 Required Settings

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def build_dataloader(dataset, batch_size, is_train, world_size, rank):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=is_train,
        seed=42,
        drop_last=is_train,  # Avoid uneven batches in training
    ) if world_size > 1 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,                   # Per-GPU batch size
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=get_num_workers(),            # See below
        pin_memory=True,                          # REQUIRED for GPU training
        persistent_workers=True,                  # Avoid worker restart overhead
        prefetch_factor=2,                        # Prefetch 2 batches per worker
        drop_last=is_train,
        collate_fn=None,                          # Custom if needed
    )

def get_num_workers():
    """4 workers per GPU is a good default. Cap at CPU count."""
    import multiprocessing
    gpus = torch.cuda.device_count()
    return min(4 * gpus, multiprocessing.cpu_count())
```

## Non-Blocking Host-to-Device Transfer

```python
# In training loop — ALWAYS use non_blocking=True
for batch in loader:
    inputs = batch["input"].cuda(non_blocking=True)
    labels = batch["label"].cuda(non_blocking=True)
    # non_blocking allows CPU to continue while transfer happens
```

## Custom Prefetching (For Maximum Throughput)

```python
class DataPrefetcher:
    """Pre-loads next batch on GPU while current batch is being processed."""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True)
                for k, v in self.next_batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

# Usage:
prefetcher = DataPrefetcher(train_loader, device)
for batch in prefetcher:
    loss = model(batch["input"])
    ...
```

## Diagnosing DataLoader Bottlenecks

```python
import time

# Quick bottleneck test — should be < 0.1s per batch for GPU training
loader = build_dataloader(...)
start = time.time()
for i, batch in enumerate(loader):
    if i == 50: break
    batch = {k: v.cuda() for k, v in batch.items()}
elapsed = time.time() - start
print(f"50 batches in {elapsed:.2f}s = {elapsed/50*1000:.1f}ms/batch")

# If > 50ms/batch with GPU: num_workers too low, or data on slow storage
```

## Checklist

- [ ] `pin_memory=True`
- [ ] `num_workers >= 4` (per GPU)
- [ ] `persistent_workers=True`
- [ ] `prefetch_factor=2` or use DataPrefetcher
- [ ] `non_blocking=True` on `.to(device)` or `.cuda()`
- [ ] `drop_last=True` for training (uneven batches cause issues with DDP)
- [ ] `set_epoch(epoch)` on DistributedSampler

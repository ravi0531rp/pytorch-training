# DataLoader: Production Tuning

## Baseline Production DataLoader

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def build_dataloader(dataset, batch_size, rank, world_size, train=True):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=train,
        drop_last=train,  # ALWAYS drop_last=True during training
    ) if world_size > 1 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and train),
        num_workers=get_optimal_workers(),
        pin_memory=True,           # async CPU→GPU transfer
        persistent_workers=True,   # keep workers alive between epochs
        prefetch_factor=2,         # prefetch 2 batches per worker
        drop_last=train,
    )

def get_optimal_workers():
    # Rule: start at 4, increase until GPU util stops improving
    # Hard cap: min(cpu_count, 16) to avoid memory pressure
    import os
    return min(os.cpu_count(), 16)
```

## Non-Blocking Data Transfer

```python
# In your training loop:
for batch in dataloader:
    # Move to GPU without blocking CPU
    inputs = batch["input"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    # GPU can start working while CPU prepares next batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(inputs, labels)
```

`non_blocking=True` only works when `pin_memory=True` on the DataLoader.

## Custom Collate for Variable-Length Sequences

```python
def collate_fn(batch):
    # Pad sequences to max length in batch (not global max)
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.stack([
        F.pad(x["input_ids"], (0, max_len - len(x["input_ids"])), value=0)
        for x in batch
    ])
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor([x["label"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

## Prefetch Queue (manual, for maximum throughput)

```python
class DataPrefetcher:
    """Overlaps GPU computation with CPU→GPU data transfer."""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in self.next_batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch
```

## Streaming Dataset (for datasets larger than RAM)

```python
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, file_paths, rank, world_size):
        self.files = file_paths[rank::world_size]  # shard files across ranks

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        if worker_info:
            files = files[worker_info.id::worker_info.num_workers]
        for path in files:
            with open(path) as f:
                for line in f:
                    yield self.process(line)

    def process(self, line):
        # parse and return tensor dict
        raise NotImplementedError
```

## Diagnosing DataLoader Bottlenecks

```python
import time

# Quick bottleneck test: time the dataloader alone
start = time.perf_counter()
for i, batch in enumerate(loader):
    if i == 100:
        break
elapsed = time.perf_counter() - start
print(f"100 batches: {elapsed:.2f}s = {elapsed/100*1000:.1f}ms/batch")

# Compare with GPU-only time (remove data loading from loop)
# If dataloader time >> forward+backward time → CPU bottleneck
```

**GPU util < 50%** → Increase `num_workers`. Try 4, 8, 16. Profile with `nvidia-smi dmon`.
**GPU util 50–70%** → Try `prefetch_factor=4` or DataPrefetcher above.
**GPU util > 80%** → DataLoader is not the bottleneck. Profile compute.

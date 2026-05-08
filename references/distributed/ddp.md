# DDP: Production Setup Guide

## Launch Command (always torchrun, never mp.spawn)

```bash
# Single node, 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --config config.yaml

# Multi-node (run on each node)
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py --config config.yaml
```

## Initialization Pattern

```python
import torch.distributed as dist

def init_distributed():
    dist.init_process_group(backend="nccl")  # always nccl for GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup():
    dist.destroy_process_group()
```

**Critical env vars set by torchrun (never set manually):**
- `LOCAL_RANK` — GPU index on this machine
- `RANK` — global rank across all nodes
- `WORLD_SIZE` — total number of processes
- `MASTER_ADDR`, `MASTER_PORT` — rendezvous

## Model Wrapping

```python
local_rank, rank, world_size = init_distributed()
device = torch.device(f"cuda:{local_rank}")

model = MyModel().to(device)

# Sync BatchNorm BEFORE wrapping (if using BN layers)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # True only if you have conditional forward paths
    gradient_as_bucket_view=True,  # reduces memory, always enable
)
```

**`find_unused_parameters=False`** — always start here. Only set True if you get errors about unused params. It adds significant overhead.

## DataLoader for DDP

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,  # CRITICAL: prevents uneven batches → deadlock
)

loader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# Each epoch: must reset sampler epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    train(loader)
```

## Checkpointing in DDP

```python
# Save: only rank 0
if rank == 0:
    torch.save({
        "model": model.module.state_dict(),  # .module unwraps DDP
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": global_step,
        "epoch": epoch,
    }, checkpoint_path)

# Load: all ranks load, but map to correct device
ckpt = torch.load(checkpoint_path, map_location=f"cuda:{local_rank}")
model.module.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
```

## Common DDP Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| `mp.spawn` instead of `torchrun` | Complex setup, harder debugging | Use torchrun always |
| Missing `set_epoch` on sampler | Same data order every epoch | Call `sampler.set_epoch(epoch)` |
| `drop_last=False` | Hang on last batch if uneven | Set `drop_last=True` |
| Logging from all ranks | 8× duplicate logs | Guard with `if rank == 0:` |
| Saving `model.state_dict()` | Saves DDP wrapper state | Save `model.module.state_dict()` |
| `find_unused_parameters=True` by default | ~10–20% slowdown | Default to False |
| No barrier before checkpoint | Race condition on slow ranks | `dist.barrier()` before save |

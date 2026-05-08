# DDP: Distributed Data Parallel

## When to Use
- Model fits in single GPU VRAM (with micro batch ≥ 1)
- < 1B parameters (general rule)
- Want near-linear scaling with minimal complexity

## Setup Rules

### 1. Process Group Initialization (ALWAYS first)
```python
import torch.distributed as dist

def init_distributed():
    dist.init_process_group(backend="nccl")  # Use "gloo" only for CPU
    local_rank = int(os.environ["LOCAL_RANK"])  # Set by torchrun
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()
```

### 2. Model Wrapping (AFTER moving to device)
```python
local_rank, rank, world_size = init_distributed()
device = torch.device(f"cuda:{local_rank}")

model = MyModel().to(device)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # Set True only if needed; has overhead
)
```

### 3. DistributedSampler (CRITICAL)
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True, seed=42)
loader = DataLoader(
    dataset,
    batch_size=per_gpu_batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# REQUIRED: call before every epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Ensures different shuffle each epoch
    train_one_epoch(...)
```

### 4. Rank-Gated Operations
```python
# Only rank 0 should: save checkpoints, log to wandb, print progress
if rank == 0:
    torch.save(checkpoint, "checkpoint.pt")
    wandb.log(metrics)

# Barrier when all ranks must sync before proceeding
dist.barrier()
```

### 5. Metric Aggregation Across Ranks
```python
def reduce_metric(tensor: torch.Tensor) -> float:
    """Average a metric across all DDP ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()

# Usage:
loss_tensor = torch.tensor(loss_val, device=device)
avg_loss = reduce_metric(loss_tensor)
```

## DDP Checklist

- [ ] `init_process_group` called before any CUDA ops
- [ ] Model on GPU *before* DDP wrapping
- [ ] `find_unused_parameters=False` (use True only if necessary)
- [ ] DistributedSampler with `set_epoch` every epoch
- [ ] Checkpoint save guarded by `rank == 0`
- [ ] `dist.destroy_process_group()` at end of script
- [ ] `torchrun` used for launch (never bare `python`)

## Common DDP Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| `find_unused_parameters=True` always | 20-30% throughput loss | Profile, only enable if needed |
| Missing `set_epoch` | All ranks see same shuffled order | Always call before epoch loop |
| Saving on all ranks | File contention / corruption | `if rank == 0: save(...)` |
| Wrong backend | Crash on CPU, slow on GPU | `nccl` for GPU, `gloo` for CPU |
| Not destroying process group | Zombie processes | `dist.destroy_process_group()` at end |

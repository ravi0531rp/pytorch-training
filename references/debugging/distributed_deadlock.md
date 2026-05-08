# Distributed Training: Deadlock Diagnosis

## Deadlock Types and Causes

| Deadlock Type | Root Cause | Detection |
|---|---|---|
| All-reduce hang | Uneven batch sizes across ranks | Some ranks finish, others wait |
| Barrier hang | One rank skips barrier (exception, conditional) | `NCCL_TIMEOUT` triggers |
| NCCL init hang | Network misconfiguration | Never reaches training loop |
| DataLoader deadlock | `num_workers > 0` + CUDA in workers | Python process freeze |

## Prevention Rules

```python
# Rule 1: Always drop_last=True in distributed DataLoader
DataLoader(..., drop_last=True)

# Rule 2: All collective ops must be called on ALL ranks
# BAD: conditional collective
if some_condition:
    dist.all_reduce(tensor)  # only rank 0 calls this → deadlock

# GOOD: compute on all ranks, use result on rank 0 only
dist.all_reduce(tensor)
if rank == 0:
    use(tensor)

# Rule 3: Every dist.barrier() must be reached by all ranks
# BAD:
if rank == 0:
    save_checkpoint()
    dist.barrier()   # rank 0 reaches barrier, others don't
# GOOD:
if rank == 0:
    save_checkpoint()
dist.barrier()       # all ranks reach this
```

## Timeout Configuration

```python
import datetime
dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(minutes=30),  # increase for large models / slow checkpointing
)
```

Set `NCCL_TIMEOUT` env var as backup: `export NCCL_TIMEOUT=1800`

## Debugging a Hang

```bash
# Step 1: Enable NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Step 2: Get Python stack traces of all processes
kill -SIGUSR1 <pid>   # dumps Python stack to stderr

# Step 3: Check which collective is hanging
# Look for "ncclAllReduce" or "ncclBarrier" in NCCL_DEBUG output
```

## Detecting Rank Divergence

```python
# Add this after potential divergence points to catch early:
def assert_ranks_equal(value, name, rank, device):
    """Ensure all ranks have the same value."""
    t = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    max_val = t.item()
    t2 = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(t2, op=dist.ReduceOp.MIN)
    min_val = t2.item()
    if max_val != min_val:
        raise RuntimeError(f"Rank divergence at {name}: rank {rank} has {value}, range=[{min_val}, {max_val}]")
```

## DataLoader + CUDA in Workers (common mistake)

```python
# NEVER do CUDA operations in DataLoader workers
# BAD:
class BadDataset(Dataset):
    def __getitem__(self, idx):
        x = torch.load(self.files[idx]).cuda()  # CUDA in worker → deadlock
        return x

# GOOD: return CPU tensors, let training loop do .to(device)
class GoodDataset(Dataset):
    def __getitem__(self, idx):
        return torch.load(self.files[idx])  # CPU tensor
```

## Async Error Detection

```python
# Set this env var to catch CUDA errors synchronously (slower, but easier to debug)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Only use during debugging — remove for production
```

## Checklist Before Multi-GPU Run

```
✅ Tested with 2 GPUs before scaling to N
✅ drop_last=True on all distributed DataLoaders
✅ No conditional collective ops
✅ NCCL connectivity test passed (see multi_node.md)
✅ Timeout set appropriately (>10min for large models)
✅ All logging/checkpointing guarded by rank==0
✅ destroy_process_group() in finally block
```

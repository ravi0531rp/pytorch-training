# Debugging: Distributed Training Deadlocks

## Immediate Response Checklist

When training hangs (no progress, no error):

```bash
# 1. Check if all processes are alive
ps aux | grep python

# 2. Enable NCCL verbose logging (restart training with this)
NCCL_DEBUG=INFO torchrun ... train.py 2>&1 | tee nccl.log

# 3. Check which operation is hanging
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL torchrun ... train.py

# 4. Get Python stack trace of hung process
kill -QUIT <pid>  # Prints stack trace to stderr
```

## Common Causes

### Cause 1: Collective Called on Subset of Ranks
```python
# WRONG: barrier only on rank 0 — all other ranks wait forever
if rank == 0:
    dist.barrier()  # Only rank 0 reaches this — DEADLOCK

# CORRECT: All ranks call collective
dist.barrier()  # All ranks call this

# Rule: Every dist.* collective must be called by ALL ranks
```

### Cause 2: Early Exit in Some Ranks
```python
# WRONG: some ranks exit early during validation, causing deadlock
if rank == 0:
    val_loss = run_validation(model, val_loader)
    if val_loss < best:
        save_checkpoint(...)
    # Other ranks are stuck waiting for DDP all-reduce from rank 0

# CORRECT: Either run validation on all ranks, or barrier before/after
dist.barrier()
if rank == 0:
    val_loss = run_validation(...)
dist.barrier()
```

### Cause 3: Exception in One Rank
```python
# If one rank throws an exception, others hang waiting for it
# Fix: Wrap training in try/except with cleanup
try:
    train()
except Exception as e:
    logger.error(f"Rank {rank} failed: {e}")
    dist.destroy_process_group()
    raise
```

### Cause 4: Uneven Batch Counts Across Ranks
```python
# If rank 0 has 101 batches and rank 1 has 100, rank 1 hangs waiting
# Fix: Always use drop_last=True with DistributedSampler
sampler = DistributedSampler(dataset, drop_last=True)
DataLoader(..., drop_last=True)
```

### Cause 5: NCCL Timeout
```python
# Default NCCL timeout is too short for large models / slow networks
dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(seconds=3600),  # 1 hour
)
```

### Cause 6: find_unused_parameters Mismatch
```python
# If model has conditional paths, some params may not get gradients
# This causes DDP to hang waiting for gradient from unused param
model = DDP(model, find_unused_parameters=True)  # Enable only if needed

# Or: register hooks to detect which params are unused
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient: {name}")
```

## Debugging Tools

```bash
# Run with timeout — process group will error out instead of hang forever
NCCL_TIMEOUT=120 torchrun ...

# Detect port conflicts (another job using same port)
lsof -i :29500

# Check InfiniBand availability
ibstat

# NCCL socket config (if InfiniBand not available)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # or ens3, ib0, etc.
```

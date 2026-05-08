# Principles

## Core Philosophy

**Throughput is the north star.** Every decision—batch size, precision, parallelism strategy—is measured in tokens/sec or samples/sec. Convergence matters, but wall-clock time to convergence is what gets models shipped.

## The 5 Laws of Production Training

### Law 1: GPU Must Never Wait
GPU idle time is money burned. The data pipeline, CPU preprocessing, and host-to-device transfers must never be the bottleneck. Profile first, optimize second.

- Target: GPU utilization > 85% (monitor with `nvidia-smi dmon` or `nvitop`)
- If utilization < 70%, the bottleneck is almost certainly the DataLoader
- Fix: increase `num_workers`, enable `pin_memory`, use prefetch factor

### Law 2: Memory is a Resource, Not a Limit
"OOM" is not a failure state—it's a signal to change strategy:
- First try: reduce batch size + gradient accumulation to maintain effective batch
- Second try: enable AMP (2x memory savings, 2-3x throughput)
- Third try: activation checkpointing (trades compute for memory, ~33% throughput cost)
- Fourth try: FSDP (shards across GPUs, linear memory scaling)

### Law 3: Distributed Training is Infrastructure, Not an Afterthought
Code must be written distributed-first. Single-GPU mode should be a `world_size=1` special case of the same code path, not a separate script.

### Law 4: Reproducibility is Non-Negotiable
Every training run must be reproducible from config + seed. This means:
- Seed everything: `torch`, `numpy`, `random`, `cuda`
- Log config, git hash, and environment to every run
- Checkpoint must restore exact training state (optimizer, scaler, scheduler, epoch)

### Law 5: Fail Fast, Fail Loudly
- Validate data pipeline before starting training (check 1 batch loads in < 1s)
- Assert tensor shapes at model boundaries in debug mode
- Log loss every N steps; alert if NaN within first 100 steps
- Use `NCCL_DEBUG=INFO` before assuming distributed deadlocks

## What We Never Do

| Anti-Pattern | Why | Fix |
|---|---|---|
| `python train.py` for multi-GPU | No process group init | `torchrun` |
| `.item()` in hot loop | GPU sync stall | Accumulate tensor, `.item()` outside loop |
| `num_workers=0` | CPU bottleneck | ≥ 4 workers per GPU |
| Saving checkpoint every rank | File corruption, slow | Rank 0 only + `dist.barrier()` |
| `float32` throughout | 2x memory, 2-3x slower | AMP everywhere |
| Hand-rolled `all_reduce` | Bug-prone | Use DDP/FSDP built-ins |
| Ignoring `set_epoch` | Non-random shuffling in DDP | Always call before epoch |

## Performance Budget (Rough Targets)

| Metric | Target | Action if Below |
|---|---|---|
| GPU Utilization | > 85% | Profile DataLoader |
| GPU Memory Usage | > 80% of VRAM | Increase batch size |
| Data Loading Time | < 10% of step time | More workers, prefetch |
| AMP Speedup | 1.5-3x over fp32 | Verify scaler is active |
| DDP Scaling Efficiency | > 90% per GPU added | Check NCCL, reduce comm overhead |

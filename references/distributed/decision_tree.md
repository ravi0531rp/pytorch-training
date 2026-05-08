# DDP vs FSDP Decision Tree

## Primary Decision: Does the model fit on one GPU?

```
Step 1: Estimate model memory
  param_bytes = num_params × bytes_per_dtype
  optimizer_bytes = param_bytes × 2   (Adam stores m + v)
  gradient_bytes = param_bytes
  activation_bytes = estimate from batch_size × seq_len × hidden × layers

  total_bytes = param_bytes + optimizer_bytes + gradient_bytes + activation_bytes
  safety_factor = 0.7  (leave 30% headroom for CUDA kernels, fragmentation)

  If total_bytes > GPU_VRAM × safety_factor → FSDP
  Else → DDP (or single-GPU)
```

## Quick Estimates by Model Size

| Model Size | dtype | Params GB | Optimizer GB | Min VRAM | Strategy |
|---|---|---|---|---|---|
| < 100M | fp32 | 0.4 | 0.8 | 2 GB | Single GPU |
| 100M–1B | bf16 | 0.2–2 | 0.4–4 | 8–16 GB | DDP |
| 1B–7B | bf16 | 2–14 | 4–28 | 40–80 GB | DDP (A100 80GB) or FSDP |
| 7B–70B | bf16 | 14–140 | 28–280 | Multi-GPU FSDP | FSDP FULL_SHARD |
| > 70B | bf16 | > 140 | > 280 | Multi-node | FSDP + tensor parallel |

## DDP: When to Use

**Use DDP when:**
- Each replica of the model fits in GPU VRAM (with batch + grads + optimizer states)
- You have 2–64 GPUs (single or multi-node)
- You want simplest possible distributed setup

**DDP characteristics:**
- Each GPU holds a full copy of model + optimizer
- Gradients are all-reduced after backward
- Linear scaling: 8× GPUs → ~8× throughput (minus comm overhead)

**DDP scaling ceiling:** When optimizer states alone exceed VRAM → switch to FSDP or ZeRO.

## FSDP: When to Use

**Use FSDP when:**
- Model + optimizer states don't fit on a single GPU
- Training models > 1B params on A100-class hardware
- You need activation checkpointing to be efficient

**FSDP Sharding Strategy Selection:**

```python
from torch.distributed.fsdp import ShardingStrategy

# Model too large to fit 1 GPU even with just params:
strategy = ShardingStrategy.FULL_SHARD  # Shard params + grads + optimizer

# Model fits params on 1 GPU but optimizer states overflow:
strategy = ShardingStrategy.SHARD_GRAD_OP  # Shard grads + optimizer only

# Multi-node: avoid inter-node all-reduce for optimizer states:
strategy = ShardingStrategy.HYBRID_SHARD  # Full shard within node, replicate across nodes
# Use HYBRID_SHARD when: num_nodes > 1 AND inter-node bandwidth << intra-node bandwidth
```

## FSDP Auto-Wrap Policy

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# For transformer models: wrap each transformer block independently
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},  # your block class
)

# For non-transformer: size-based wrapping
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1e6,  # wrap modules with > 1M params
)
```

## Gradient Accumulation with FSDP

```python
# FSDP requires no_sync() context for gradient accumulation:
for i, batch in enumerate(dataloader):
    is_last = (i + 1) % grad_accum_steps == 0
    ctx = nullcontext() if is_last else model.no_sync()
    with ctx:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch) / grad_accum_steps
        scaler.scale(loss).backward()
    if is_last:
        scaler.unscale_(optimizer)
        model.clip_grad_norm_(1.0)  # FSDP has its own clip method
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Multi-Node Decision

```
Single machine, multi-GPU → torchrun --nproc_per_node=N train.py
Multi-machine → torchrun --nnodes=M --nproc_per_node=N --rdzv_backend=c10d \
                          --rdzv_endpoint=HOST:PORT train.py
```

See `distributed/multi_node.md` for full NCCL configuration.

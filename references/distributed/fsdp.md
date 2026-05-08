# FSDP: Production Setup Guide

## Core FSDP Template

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp(model, transformer_block_cls, sharding_strategy=ShardingStrategy.FULL_SHARD):
    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,   # gradient reduction dtype
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap policy: wrap each transformer block
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_block_cls},
    )

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap comm with compute
        cpu_offload=CPUOffload(offload_params=False),     # set True only if OOM
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,   # prevents OOM from too many concurrent all-gathers
    )
    return model
```

## Sharding Strategy Selection

```python
# GPU VRAM < model params + optimizer:
ShardingStrategy.FULL_SHARD       # shard everything, highest memory savings

# GPU VRAM fits params, not optimizer:
ShardingStrategy.SHARD_GRAD_OP    # only shard gradients + optimizer states

# Multi-node, want to minimize inter-node traffic:
ShardingStrategy.HYBRID_SHARD     # full shard within node, replicate across nodes
```

## Activation Checkpointing

Add this after FSDP wrapping for large models:

```python
from torch.distributed.fsdp.wrap import apply_activation_checkpointing
from torch.utils.checkpoint import checkpoint_wrapper, CheckpointImpl

non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
```

Use activation checkpointing when: sequence length > 1024 OR model > 3B params.
Cost: ~33% more compute, saves ~60–70% activation memory.

## Checkpointing FSDP Models

FSDP requires special handling — sharded state must be gathered before saving:

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Full checkpoint (unsharded, only rank 0 gets it):
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save({"model": state_dict, "step": step}, ckpt_path)

# Sharded checkpoint (faster, each rank saves its shard):
from torch.distributed.fsdp import ShardedStateDictConfig
sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy):
    state_dict = model.state_dict()
    dist.barrier()
    # each rank saves its own shard
    torch.save(state_dict, f"ckpt_rank{rank}.pt")
```

Prefer **sharded checkpoints** for large models (faster, no memory spike from gathering).
Use **full checkpoints** when you need to port the model elsewhere.

## FSDP Optimizer

Use standard optimizers — they automatically operate on local (sharded) params:

```python
# Optimizer sees only this rank's parameter shards
optimizer = torch.optim.AdamW(
    model.parameters(),  # these are already sharded params
    lr=1e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    fused=True,  # use fused AdamW on CUDA (significant speedup)
)
```

## FSDP Gradient Accumulation

```python
from contextlib import nullcontext

for step, batch in enumerate(loader):
    is_sync_step = (step + 1) % grad_accum_steps == 0

    # no_sync() skips gradient all-reduce — critical for efficiency
    ctx = nullcontext() if is_sync_step else model.no_sync()

    with ctx:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch) / grad_accum_steps
        loss.backward()

    if is_sync_step:
        # FSDP has its own clip_grad_norm_ that handles sharded params
        grad_norm = model.clip_grad_norm_(max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

## FSDP Common Failures

| Issue | Cause | Fix |
|---|---|---|
| OOM during forward | Activations too large | Enable activation checkpointing |
| OOM during all-gather | Too many concurrent all-gathers | Set `limit_all_gathers=True` |
| Slow training | `backward_prefetch` off | Set `BackwardPrefetch.BACKWARD_PRE` |
| Wrong checkpoint | Saving without state_dict_type context | Always use FSDP.state_dict_type() |
| Optimizer state mismatch | Restoring full ckpt to sharded model | Use same StateDictType for save and load |

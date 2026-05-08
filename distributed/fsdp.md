# FSDP: Fully Sharded Data Parallel

## When to Use
- Model does NOT fit in single GPU VRAM for training
- Model ≥ 1B parameters with standard batch sizes
- Need to maximize memory efficiency across GPUs

## Sharding Strategies (Choose One)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# Strategy selection:
# FULL_SHARD: params + grads + optimizer states sharded → max memory savings
# SHARD_GRAD_OP: grads + optimizer states sharded → good balance
# NO_SHARD: equivalent to DDP (use DDP instead)
# HYBRID_SHARD: FULL_SHARD within node, replicate across nodes
```

## Minimal FSDP Setup

```python
def setup_fsdp(model, TransformerLayerClass, sharding_strategy=ShardingStrategy.FULL_SHARD):
    # Define which modules to wrap (wrap transformer blocks individually)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerLayerClass},
    )

    # Mixed precision config
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,   # bf16 preferred over fp16 (no overflow)
        reduce_dtype=torch.float32,    # fp32 for gradient reduction
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Required for torch.compile compatibility
    )
    return model
```

## Activation Checkpointing (for > 7B models)

```python
from torch.distributed.fsdp.wrap import apply_activation_checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing
)

# Apply AFTER FSDP wrapping
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ),
    check_fn=lambda module: isinstance(module, TransformerLayerClass),
)
```

## Checkpointing with FSDP

```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# Saving — gather shards on rank 0
def save_fsdp_checkpoint(model, optimizer, path, rank):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
    if rank == 0:
        torch.save({
            "model": model_state,
            "optimizer": optimizer.state_dict(),
        }, path)

# Loading — load on rank 0, scatter to all
def load_fsdp_checkpoint(model, optimizer, path, rank, device):
    if rank == 0:
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = {"model": None, "optimizer": None}
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
```

## FSDP Checklist

- [ ] `use_orig_params=True` if using `torch.compile`
- [ ] Wrap at transformer block level (not whole model)
- [ ] `bf16` preferred over `fp16` for param_dtype
- [ ] Activation checkpointing for models > 7B
- [ ] Use `FullStateDictConfig(rank0_only=True)` for saving
- [ ] No `GradScaler` needed with bf16 (unlike fp16)
- [ ] DistributedSampler still required

## Memory Savings Reference

| Config | Memory vs DDP fp32 |
|---|---|
| DDP + AMP | ~50% |
| FSDP SHARD_GRAD_OP | ~60% |
| FSDP FULL_SHARD | ~75% |
| FSDP FULL_SHARD + activation ckpt | ~85% |
| FSDP FULL_SHARD + CPU offload | ~90% (major throughput cost) |

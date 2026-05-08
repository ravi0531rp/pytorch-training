# NaN Loss Diagnosis

## Flowchart

```
Loss is NaN or Inf?
├─ Check: is it NaN from step 1?
│   YES → data issue or weight init issue (see below)
│   NO  → appeared mid-training → LR or grad issue
│
├─ Appeared mid-training:
│   ├─ Check grad norm before NaN step
│   │   Spike > 10? → Add/reduce gradient clipping (max_norm=0.5)
│   │   Fine? → Check for bad batch (outlier data)
│   │
│   └─ Is loss NaN or just very large?
│       NaN → likely division by zero, log(0), or overflow
│       Very large → LR too high, no warmup
│
└─ Reproducible on same data?
    YES → specific bad sample → add data validation
    NO  → floating point instability → switch to bf16, reduce LR
```

## Instrumentation: Catch NaN Early

```python
def check_finite(loss, step, batch):
    if not torch.isfinite(loss):
        print(f"Step {step}: Non-finite loss = {loss.item()}")
        # Log batch statistics to identify bad data
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: min={v.min()}, max={v.max()}, nan={v.isnan().any()}")
        raise RuntimeError(f"Non-finite loss at step {step}")

# In training loop, add after loss computation:
check_finite(loss, global_step, batch)
```

## Hook-Based Gradient Monitoring

```python
def register_nan_hooks(model):
    """Register hooks to detect NaN gradients by layer."""
    hooks = []
    for name, param in model.named_parameters():
        def make_hook(n):
            def hook(grad):
                if grad is not None and not torch.isfinite(grad).all():
                    raise RuntimeError(f"NaN gradient in {n}")
                return grad
            return hook
        hooks.append(param.register_hook(make_hook(name)))
    return hooks

# Use during debugging (remove for production — adds overhead):
hooks = register_nan_hooks(model)
```

## Common NaN Causes and Fixes

| Cause | Symptom | Fix |
|---|---|---|
| LR too high | NaN after a few steps | Reduce LR by 10×, add warmup |
| No gradient clipping | Exploding grad → NaN | `clip_grad_norm_(..., 1.0)` |
| `log(0)` in loss | NaN from step 1 or randomly | Add epsilon: `log(x + 1e-8)` |
| `float16` overflow | NaN specifically with fp16 | Use bfloat16 or ensure scaler is enabled |
| Bad label values | NaN from cross-entropy with -1 | Validate labels, use `ignore_index` |
| Numerical instability in softmax | NaN in attention | Use `scaled_dot_product_attention` |
| Division by zero in normalization | NaN in LayerNorm/BatchNorm | Check for zero-variance inputs |
| Corrupted checkpoint | NaN after resume | Validate checkpoint with `torch.isfinite` check |

## Pre-Training Data Validation

```python
def validate_batch(batch):
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), f"Non-finite values in {key}"
            assert val.dtype in (torch.float32, torch.float16, torch.bfloat16,
                                  torch.long, torch.int32), f"Unexpected dtype in {key}"

# Run on first N batches before full training:
for i, batch in enumerate(loader):
    validate_batch(batch)
    if i >= 10:
        break
```

## Scaler Skip Debugging (float16)

```python
# GradScaler skips optimizer.step() when NaN/Inf gradients detected
# Track skips — too many skips means scale too high or real NaN problem
skipped = 0
for step, batch in enumerate(loader):
    ...
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    new_scale = scaler.get_scale()
    if new_scale < old_scale:
        skipped += 1
        if rank == 0:
            print(f"Step {step}: Scaler reduced scale {old_scale} → {new_scale} (skips={skipped})")
```

More than 5% skip rate → investigate root cause, don't just reduce scale.

# Debugging: NaN Loss

## Immediate Triage

```python
# Add this to your training loop immediately when NaN appears
if torch.isnan(loss):
    # 1. Check inputs
    print(f"Input stats: min={inputs.min():.4f} max={inputs.max():.4f} has_nan={inputs.isnan().any()}")
    # 2. Check model outputs
    print(f"Output stats: min={outputs.min():.4f} max={outputs.max():.4f}")
    # 3. Check gradients
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"NaN gradient in: {name}")
    raise ValueError("NaN loss detected — training stopped")
```

## Common Causes and Fixes

### Cause 1: Learning Rate Too High
```python
# Symptom: NaN appears in first 100 steps
# Fix: Reduce LR by 10x; add warmup
scheduler = build_scheduler(optimizer, num_warmup_steps=1000, ...)
# Typical LR ranges: 1e-4 to 3e-4 for AdamW, 1e-3 for SGD
```

### Cause 2: Loss Explosion (fp16 Overflow)
```python
# Symptom: loss grows to inf then NaN; scaler.get_scale() keeps decreasing
# Fix 1: Use bf16 instead of fp16 (no overflow)
with autocast(dtype=torch.bfloat16):
    ...
# Fix 2: If stuck with fp16, increase initial scale
scaler = GradScaler(init_scale=2**8)  # Start smaller, let it grow
```

### Cause 3: Missing Gradient Clipping
```python
# Symptom: loss is stable then suddenly NaN; gradient norm spikes
# Fix: Always clip
scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if rank == 0:
    logger.info(f"Grad norm: {grad_norm:.4f}")  # Monitor this
```

### Cause 4: Division by Zero in Loss
```python
# Common in: cross-entropy with empty classes, custom losses
# Fix: add epsilon
loss = -torch.log(probs + 1e-8)
# Or use label smoothing
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Cause 5: Bad Data (inf/NaN in inputs)
```python
# Add assertion in Dataset.__getitem__
def __getitem__(self, idx):
    sample = self.load(idx)
    assert not torch.isnan(sample["input"]).any(), f"NaN in sample {idx}"
    assert not torch.isinf(sample["input"]).any(), f"Inf in sample {idx}"
    return sample
```

### Cause 6: Unstable BatchNorm with Small Batches
```python
# Symptom: NaN after first few steps with batch_size=1 or 2
# Fix: Use GroupNorm or LayerNorm instead of BatchNorm
# Or: set minimum batch size of 8 per GPU for BatchNorm
```

## Detection Script (Run Before Full Training)

```python
def validate_training_setup(model, loader, device, n_steps=10):
    """Run N steps; catch NaN early before long training run."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(dtype=torch.bfloat16):
            out = model(inputs)
            loss = criterion(out, labels)

        assert not torch.isnan(loss), f"NaN at step {step}"
        assert not torch.isinf(loss), f"Inf at step {step}"

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print(f"✓ Validation passed: {n_steps} steps without NaN/Inf")
```

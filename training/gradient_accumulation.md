# Gradient Accumulation

## Why and When

Use gradient accumulation when:
- Desired batch size exceeds GPU memory capacity
- Want to simulate large-batch training on limited hardware
- Formula: `effective_batch = micro_batch * accum_steps * world_size`

## Correct Implementation

```python
optimizer.zero_grad()

for step, batch in enumerate(loader):
    inputs = batch["input"].cuda(non_blocking=True)
    labels = batch["label"].cuda(non_blocking=True)

    # With DDP: avoid redundant all-reduce on non-final accumulation steps
    # model.no_sync() skips gradient synchronization across ranks
    is_last_accum = (step + 1) % cfg.grad_accum_steps == 0

    ctx = model.no_sync() if not is_last_accum else contextlib.nullcontext()

    with ctx:
        with autocast(dtype=torch.bfloat16):
            loss = model(inputs, labels)
        loss = loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()

    if is_last_accum:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()
        global_step += 1
```

## Batch Size Selection Guide

```
Target effective batch size: 2048
│
├── 8 GPUs available, 8GB VRAM each
│   micro_batch = 8 (what fits in memory)
│   accum_steps = 2048 / (8 * 8) = 32
│
├── 8 GPUs available, 80GB VRAM each
│   micro_batch = 64 (what fits in memory)
│   accum_steps = 2048 / (64 * 8) = 4
│
└── 1 GPU, 24GB VRAM
    micro_batch = 8
    accum_steps = 2048 / 8 = 256
```

## Critical: model.no_sync() for DDP

Without `no_sync()`, DDP triggers an all-reduce after every `.backward()`, wasting bandwidth on intermediate accumulation steps. Only the final step needs synchronization.

```python
# WRONG (redundant comms):
for i in range(accum_steps):
    loss.backward()  # All-reduce fires every time

# CORRECT:
for i in range(accum_steps - 1):
    with model.no_sync():
        loss.backward()  # No all-reduce
loss.backward()  # Final step: all-reduce fires
```

## Checklist

- [ ] Divide loss by `grad_accum_steps` before backward
- [ ] Use `model.no_sync()` on non-final steps (DDP only)
- [ ] Clip gradients only after final accumulation step
- [ ] Zero gradients only after optimizer step
- [ ] `scheduler.step()` aligned with optimizer step, not batch step

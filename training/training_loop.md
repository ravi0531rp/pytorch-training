# Training Loop: Production Standard

## Complete Training Loop Template

```python
import torch
from torch.cuda.amp import GradScaler, autocast
import logging

logger = logging.getLogger(__name__)

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    rank: int,
    cfg,
    sampler=None,
):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)  # CRITICAL for DDP shuffle correctness

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Non-blocking host→device transfer
        inputs = batch["input"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        # Mixed precision forward pass
        with autocast(dtype=torch.bfloat16):  # bf16 preferred; use float16 for older GPUs
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Scale for gradient accumulation
        loss = loss / cfg.grad_accum_steps

        # Backward with scaler (handles fp16/bf16 scaling)
        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # Logging (avoid .item() every step — sync stall)
        total_loss += loss.detach()  # Keep on GPU
        num_batches += 1

        if step % cfg.log_every == 0 and rank == 0:
            avg = total_loss.item() / num_batches  # .item() only for logging
            logger.info(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss: {avg:.4f}")
            total_loss = 0.0
            num_batches = 0

    return avg
```

## AMP Configuration

```python
# Initialization (do once before training)
scaler = torch.cuda.amp.GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True,
)

# bf16 vs fp16:
# bf16: no overflow, no scaler needed in practice, requires Ampere+ GPU (A100, 3090, 4090)
# fp16: works on older GPUs (V100), requires scaler
# Recommendation: bf16 whenever hardware supports it
```

## Scheduler Setup (Cosine with Warmup)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def build_scheduler(optimizer, num_warmup_steps, num_training_steps):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[num_warmup_steps])
```

## Seeding for Reproducibility

```python
def seed_everything(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: deterministic ops (slower)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# With DDP: use different seed per rank for data augmentation stochasticity
# but same seed for model init
seed_everything(cfg.seed + rank)
```

## Training Loop Checklist

- [ ] `model.train()` at start of epoch
- [ ] `sampler.set_epoch(epoch)` before iterating
- [ ] `autocast(dtype=torch.bfloat16)` wraps forward pass
- [ ] `scaler.scale(loss).backward()`
- [ ] `scaler.unscale_(optimizer)` before grad clip
- [ ] `clip_grad_norm_` with max_norm=1.0
- [ ] `scaler.step(optimizer)` + `scaler.update()`
- [ ] `optimizer.zero_grad()` (set_to_none=True for memory)
- [ ] Logging only at intervals, `.item()` only then

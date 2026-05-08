# Training Loop: AMP, Gradient Accumulation, Checkpointing

## Production Training Loop

```python
def train_epoch(model, loader, optimizer, scheduler, scaler, device, config, rank=0):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        global_step = config.epoch * len(loader) + step

        # Non-blocking data transfer
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Gradient accumulation context
        is_sync = (step + 1) % config.grad_accum_steps == 0 or (step + 1) == len(loader)
        accum_ctx = getattr(model, "no_sync", lambda: nullcontext())
        ctx = nullcontext() if is_sync else accum_ctx()

        with ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(inputs, labels=labels).loss
                loss = loss / config.grad_accum_steps  # normalize

            scaler.scale(loss).backward()

        total_loss += loss.item() * config.grad_accum_steps

        if is_sync:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory

            if rank == 0 and global_step % config.log_every == 0:
                log_metrics({
                    "loss": total_loss / (step + 1),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item(),
                    "step": global_step,
                })

            if rank == 0 and global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, global_step, config)
```

## Gradient Accumulation Details

```python
# Effective batch = batch_per_gpu × gpus × grad_accum_steps
# Loss MUST be divided by grad_accum_steps before backward

# Why: without division, each accumulated gradient has magnitude ×grad_accum_steps
# compared to no accumulation, causing LR to be effectively ×grad_accum_steps

# Correct:
loss = loss / grad_accum_steps
loss.backward()

# Wrong (don't do this):
loss.backward()
# then averaging later — too late, gradients are already summed
```

## AMP: float16 vs bfloat16

```python
# A100, H100, 3090, 4090 → use bfloat16 (no overflow, wider range)
dtype = torch.bfloat16

# V100, T4, older hardware → use float16 (needs GradScaler)
dtype = torch.float16

# GradScaler: technically only needed for float16 (bfloat16 doesn't overflow)
# But: keep GradScaler enabled even with bfloat16 for portability + safety
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
```

## Checkpointing

```python
def save_checkpoint(model, optimizer, scheduler, scaler, step, config):
    # DDP: unwrap with .module
    model_state = getattr(model, "module", model).state_dict()

    ckpt = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": vars(config),
    }

    path = Path(config.output_dir) / f"ckpt_step{step}.pt"
    tmp_path = path.with_suffix(".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.rename(path)  # atomic rename prevents corrupt checkpoints

    # Keep only last K checkpoints
    existing = sorted(Path(config.output_dir).glob("ckpt_step*.pt"))
    for old in existing[:-config.keep_checkpoints]:
        old.unlink()

def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    getattr(model, "module", model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]
```

## LR Scheduler: Warmup + Cosine Decay

```python
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return LambdaLR(optimizer, lr_lambda)

# Warmup rule: 1–5% of total steps
warmup_steps = int(0.03 * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
```

## Metric Logging

```python
import json
import logging

logger = logging.getLogger(__name__)

def log_metrics(metrics: dict):
    # Structured JSON logging — parse with jq, ship to ELK, etc.
    logger.info(json.dumps(metrics))

    # Optional: W&B
    if wandb.run:
        wandb.log(metrics)
```

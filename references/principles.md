# Core Training Principles

## The Contract
Every training run must satisfy:
1. **Deterministic startup** — seeds set before model init, dataloader init, and first forward pass
2. **Observable** — loss, lr, grad norm, GPU util logged every N steps to a structured sink (W&B, MLflow, or at minimum stdout JSON)
3. **Resumable** — checkpoint contains: model state, optimizer state, scheduler state, scaler state, step count, RNG states
4. **Safe shutdown** — SIGTERM handler saves checkpoint; distributed runs call `destroy_process_group()`

## Non-Negotiables (never skip these)

### Mixed Precision
```python
# ALWAYS use this pattern. Never manually cast.
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # prefer bfloat16 on Ampere+
    loss = model(batch)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```
Use `bfloat16` on A100/H100 (no overflow risk, no scaler needed technically but keep it for portability).
Use `float16` on older GPUs (V100, T4) — scaler is mandatory.

### Gradient Clipping
```python
# Always clip. 1.0 is a safe default. Tune down if loss spikes persist.
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Never skip this. Without clipping, a single bad batch can corrupt the entire run.

### Seeding
```python
def set_seed(seed: int, rank: int = 0):
    seed = seed + rank  # per-rank seed for data diversity
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # For max speed (non-deterministic but reproducible enough):
    torch.backends.cudnn.benchmark = True  # safe when input shapes are fixed
```

### Checkpointing Frequency
- Save every **500–2000 steps** (not just epoch-end — epochs can be hours)
- Keep last K=3 checkpoints + best-by-metric checkpoint
- On distributed: only `rank == 0` writes checkpoints

### Logging Discipline
```python
if rank == 0 and step % log_every == 0:
    logger.info(json.dumps({
        "step": step,
        "loss": loss.item(),
        "lr": scheduler.get_last_lr()[0],
        "grad_norm": grad_norm,
        "gpu_util": torch.cuda.utilization(),
    }))
```
Never call `.item()` inside the training step — it forces a CPU sync. Batch metrics collection.

## What Kills Training Runs

| Failure | Root Cause | Prevention |
|---|---|---|
| NaN loss | LR too high, bad data, no grad clip | Clip grads, validate data, warmup LR |
| OOM | Batch too large, no grad checkpoint | Use grad checkpointing, reduce batch |
| Deadlock | Uneven batch sizes across ranks | DropLast=True, check all_reduce barriers |
| Slow GPU | Data bottleneck, CPU-bound ops | Profile dataloader, increase workers |
| Divergence | LR schedule wrong, no warmup | Linear warmup 1–5% of total steps |
| Silent wrong results | Wrong loss reduction in DDP | Use `loss = loss / grad_accum_steps` |

## Batch Size Strategy

```
Effective batch size = batch_per_gpu × num_gpus × grad_accum_steps

Target effective batch: start at 256–2048 depending on task.
Scale LR linearly: lr = base_lr × (effective_batch / base_batch)
Always use warmup: 1–5% of total steps, linear from 0 to peak lr.
```

Prefer **larger physical batch** over gradient accumulation when VRAM allows — accumulation adds overhead.
Use accumulation when: single GPU, VRAM-limited, or simulating large-batch behavior.

# Debugging: Poor Convergence

## Diagnosis Framework

```
Loss not decreasing?
├── Loss is flat from step 1 → LR too low OR model init issue
├── Loss decreases then plateaus → LR decay too aggressive OR dataset issue
├── Loss oscillates wildly → LR too high OR batch size too small
└── Loss decreases slowly → Normal, or increase LR / use warmup
```

## LR Finder (Binary Search Approach)

```python
# Run for 100 steps at each LR, log final loss
# Target: find LR where loss decreases fastest without diverging
lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
for lr in lrs:
    loss = quick_train(model, loader, lr=lr, steps=100)
    print(f"LR={lr:.0e} → Loss={loss:.4f}")
# Pick the LR just before loss starts increasing
```

## Common Fixes by Symptom

### Flat Loss from Start
```python
# 1. Check model outputs (not all same value)
with torch.no_grad():
    out = model(dummy_input)
    print(f"Output range: {out.min():.3f} to {out.max():.3f}")
    # All same → dead neurons or bad init

# 2. Check gradient flow
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name}: grad_norm={p.grad.norm():.4f}")
    else:
        print(f"{name}: NO GRADIENT")

# 3. Fix: use proper weight init
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
model.apply(init_weights)
```

### Loss Oscillates
```python
# Fix 1: Lower LR by 5-10x
# Fix 2: Increase batch size (more stable gradient estimates)
# Fix 3: Add gradient clipping if not present
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Fix 4: Add label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Slow Convergence
```python
# Fix 1: Add LR warmup
scheduler = build_scheduler(optimizer, warmup_steps=1000, total_steps=100000)

# Fix 2: Use AdamW (not Adam) — decoupled weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,    # Standard for most models
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Fix 3: Increase effective batch size
# Small batches → noisy gradients → slow convergence
```

## Optimizer Quick Reference

| Model Type | Optimizer | LR | Weight Decay |
|---|---|---|---|
| CNN (ImageNet) | SGD+momentum | 0.1 (with step decay) | 1e-4 |
| Transformer | AdamW | 3e-4 | 0.01 |
| LLM fine-tune | AdamW | 1e-5 to 5e-5 | 0.01 |
| RL / online | Adam | 1e-4 | 0 |

## Monitoring Convergence Health

```python
# Log these every N steps:
metrics = {
    "loss": loss.item(),
    "grad_norm": grad_norm,
    "lr": scheduler.get_last_lr()[0],
    "loss_scale": scaler.get_scale(),  # Should stay > 1
}
# Alert if grad_norm > 10 consistently
# Alert if loss_scale < 1 (fp16 overflow)
```

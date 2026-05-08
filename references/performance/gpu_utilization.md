# GPU Utilization: Profiling and Optimization

## Quick Triage

```bash
# Watch GPU utilization in real time:
nvidia-smi dmon -s u -d 1    # util every 1 second
watch -n 0.5 nvidia-smi      # full stats every 0.5s

# Target: GPU util > 80% sustained
# < 50%: data bottleneck → fix dataloader
# 50–80%: compute underutilization → increase batch size or reduce overhead
# > 80%: good — now profile at op level
```

## torch.profiler Integration

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

profiler_schedule = schedule(
    wait=1,    # skip first step
    warmup=1,  # warmup
    active=3,  # profile 3 steps
    repeat=1,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,  # expensive, enable only when needed
) as prof:
    for step, batch in enumerate(loader):
        with record_function("forward"):
            loss = model(batch)
        with record_function("backward"):
            loss.backward()
        prof.step()
        if step >= 5:
            break

# View: tensorboard --logdir=./profiler_logs
# Or print top kernels:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## Common GPU Underutilization Causes

### 1. `.item()` in training loop (CPU sync)
```python
# BAD: forces GPU→CPU sync every step
total_loss += loss.item()

# GOOD: accumulate tensor, sync only at log time
total_loss_tensor += loss.detach()
if step % log_every == 0:
    avg = total_loss_tensor.item() / log_every  # one sync per log interval
    total_loss_tensor.zero_()
```

### 2. Small batch size
```python
# If batch is too small, GPU is idle between kernel launches
# Guideline: batch size should saturate GPU memory to ~70-80%
# Double batch size until OOM, then back off 20%
```

### 3. Synchronization barriers
```python
# BAD: unnecessary barrier
dist.barrier()  # avoid unless strictly needed

# When barriers ARE needed: before checkpoint, after data loading sync
```

### 4. torch.compile() — major throughput improvement

```python
# PyTorch 2.0+: compile model for significant speedup
model = torch.compile(model)

# Options:
model = torch.compile(model, mode="default")       # balanced
model = torch.compile(model, mode="reduce-overhead") # minimize Python overhead
model = torch.compile(model, mode="max-autotune")   # max speed, long compile time

# Note: first few steps are slow (JIT compilation). Don't benchmark them.
# FSDP + compile: wrap FSDP first, then compile
```

### 5. Flash Attention

```python
# Use F.scaled_dot_product_attention (PyTorch 2.0+) instead of manual attention
# It automatically uses Flash Attention when possible

import torch.nn.functional as F
# Replaces manual: q @ k.T / sqrt(d) → softmax → @ v
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
```

## Memory Optimization Checklist

```
✅ set_to_none=True in optimizer.zero_grad()     # frees memory vs zero-filling
✅ del intermediate tensors explicitly            # helps GC in complex models
✅ torch.cuda.empty_cache() between epochs       # reduces fragmentation
✅ Gradient checkpointing for long sequences     # trade compute for memory
✅ Use bf16 not fp32 for activations             # 2× memory savings
✅ Limit all_gathers in FSDP                     # prevents alloc spikes
```

## Throughput Benchmark Pattern

```python
def benchmark_throughput(model, loader, device, n_steps=50):
    model.train()
    # warmup
    for i, batch in enumerate(loader):
        if i >= 5: break
        batch = {k: v.to(device) for k, v in batch.items()}
        model(batch["input_ids"]).loss.backward()

    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = 0
    for i, batch in enumerate(loader):
        if i >= n_steps: break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch["input_ids"]).loss
        loss.backward()
        tokens += batch["input_ids"].numel()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Throughput: {tokens / elapsed:.0f} tokens/sec")
```

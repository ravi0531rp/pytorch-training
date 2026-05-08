# GPU Utilization & Profiling

## Quick Diagnosis

```bash
# Real-time GPU stats (1-second interval)
nvidia-smi dmon -s u -d 1

# Better: nvitop (install: pip install nvitop)
nvitop

# Target: sm% (GPU compute utilization) > 85%
# If sm% < 70%, you have a bottleneck
```

## Bottleneck Identification

```python
# Step 1: Time your DataLoader in isolation
import time
loader = build_dataloader(...)
t0 = time.time()
for i, batch in enumerate(loader):
    if i == 100: break
    _ = {k: v.cuda() for k, v in batch.items()}
data_time = (time.time() - t0) / 100

# Step 2: Time forward + backward in isolation (dummy data)
dummy = torch.randn(batch_size, *input_shape).cuda()
t0 = time.time()
for _ in range(100):
    with autocast(dtype=torch.bfloat16):
        out = model(dummy)
        loss = out.sum()
    loss.backward()
torch.cuda.synchronize()
compute_time = (time.time() - t0) / 100

print(f"Data: {data_time*1000:.1f}ms | Compute: {compute_time*1000:.1f}ms")
# If data > 20% of compute: DataLoader is bottleneck
```

## PyTorch Profiler (Full Pipeline Analysis)

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        with record_function("data_transfer"):
            inputs = batch["input"].cuda(non_blocking=True)
        with record_function("forward"):
            with autocast(dtype=torch.bfloat16):
                loss = model(inputs)
        with record_function("backward"):
            loss.backward()
        prof.step()
        if step >= 20: break

# View: tensorboard --logdir ./profiler_logs
```

## Maximizing GPU Utilization

### 1. torch.compile (PyTorch >= 2.0)
```python
model = torch.compile(model, mode="reduce-overhead")
# modes: "default" | "reduce-overhead" | "max-autotune"
# max-autotune: slower compile, fastest runtime (use for long runs)
```

### 2. cudnn.benchmark
```python
torch.backends.cudnn.benchmark = True  # Auto-tunes convolution algorithms
# Enable when: input sizes are fixed, training on CNNs
# Disable when: input sizes vary (wastes time re-tuning)
```

### 3. Increase Batch Size to Fill GPU Memory
```python
# Target: use 80-90% of GPU VRAM
# Check: nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# If memory usage < 70%: increase batch size or use gradient accumulation inversely
```

### 4. set_to_none for zero_grad
```python
optimizer.zero_grad(set_to_none=True)  # Faster than zeroing: avoids memset
```

### 5. Channels Last Memory Format (CNNs)
```python
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)
# Improves memory access pattern for conv ops; typically 10-30% speedup
```

## Common Utilization Killers

| Cause | Symptom | Fix |
|---|---|---|
| `.item()` in loop | GPU stalls | Move .item() outside loop |
| Small batch size | GPU underutilized | Increase batch or accumulate |
| `num_workers=0` | CPU bottleneck | Set ≥ 4 workers |
| Sync ops in loop | sm% spikes then drops | Use non_blocking, avoid barriers |
| Data on HDD | High wait time | Move to SSD or RAM |
| `find_unused_params=True` | Unnecessary DDP comm | Disable unless required |

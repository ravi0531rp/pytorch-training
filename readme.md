# PyTorch Training Skill — Production-Grade Deep Learning Operations

A comprehensive, production-oriented reference for training deep neural networks with PyTorch. This knowledge base covers everything from single-GPU training to large-scale distributed systems, with battle-tested patterns, decision trees, and debugging strategies.

**Core Philosophy:** Throughput is the north star. Every decision—batch size, precision, parallelism strategy—is measured in tokens/sec or samples/sec.

---

## 🎯 Quick Start: What Are You Trying to Do?

| Your Goal | Start Here | Next Steps |
|-----------|-----------|-----------|
| **Train a model on 1–8 GPUs** | [decision_tree.md](decision_tree.md) | Choose DDP or FSDP based on model size |
| **Fix OOM errors** | [decision_tree.md](decision_tree.md) → [distributed/fsdp.md](distributed/fsdp.md) | Reduce batch, enable AMP, or switch to FSDP |
| **Optimize GPU utilization** | [performance/gpu_utilization.md](performance/gpu_utilization.md) | Profile with `nvidia-smi dmon`, check DataLoader |
| **Debug NaN losses** | [debugging/nan_loss.md](debugging/nan_loss.md) | Check learning rate, gradient clipping, data normalization |
| **Multi-node distributed training** | [distributed/torchrun.md](distributed/torchrun.md) | Use torchrun with `--nnodes` and rendezvous backend |
| **Large model (>7B params)** | [decision_tree.md](decision_tree.md) | FSDP with FULL_SHARD + activation checkpointing |

---

## 📖 Core Principles: The 5 Laws of Production Training

Read [principles.md](principles.md) for the full philosophy. Here's the executive summary:

### Law 1: GPU Must Never Wait
- GPU idle = money burned. Data pipeline is almost always the bottleneck.
- Target: **GPU utilization > 85%** (monitor with `nvidia-smi dmon`)
- Fix: Increase `num_workers`, enable `pin_memory`, use prefetch

### Law 2: Memory is a Resource, Not a Limit
- "OOM" → reduce batch + gradient accumulation
- "Still OOM" → enable AMP (mixed precision)
- "Still OOM" → activation checkpointing (compute for memory trade)
- "Still OOM" → FSDP (shards across GPUs)

### Law 3: Distributed Training is Infrastructure, Not Afterthought
- Write distributed-first. Single-GPU is `world_size=1` special case.
- Always use `torchrun`, never `python train.py` for multi-GPU

### Law 4: Reproducibility is Non-Negotiable
- Seed everything (torch, numpy, random, cuda)
- Log config + git hash + environment to every run
- Checkpoint must restore exact state (optimizer, scaler, scheduler, epoch)

### Law 5: Fail Fast, Fail Loudly
- Validate data pipeline before training (1 batch loads in < 1s)
- Assert tensor shapes at model boundaries
- Log loss every N steps; alert if NaN within first 100 steps

---

## 🚦 Router: Choose Your Strategy

### Decision Tree: DDP vs FSDP vs Single GPU
See [decision_tree.md](decision_tree.md) for full flowchart.

```
Model Size?
├── < 1B params → DDP (near-linear scaling)
├── 1B–7B params → FSDP with SHARD_GRAD_OP (~2-3x memory savings)
└── > 7B params → FSDP with FULL_SHARD + activation checkpointing
```

### Routing by Query

| If You're Asking About... | Read This | Then This |
|---------------------------|-----------|-----------|
| Multi-GPU training, DDP, DistributedDataParallel | [distributed/ddp.md](distributed/ddp.md) | [distributed/torchrun.md](distributed/torchrun.md) |
| Large model, OOM on single GPU, FSDP | [distributed/fsdp.md](distributed/fsdp.md) | [distributed/torchrun.md](distributed/torchrun.md) |
| torchrun, launching, multi-node | [distributed/torchrun.md](distributed/torchrun.md) | [references/distributed/multi_node.md](references/distributed/multi_node.md) |
| DataLoader, slow data, CPU bottleneck | [data_pipeline/dataloading.md](data_pipeline/dataloading.md) | [performance/gpu_utilization.md](performance/gpu_utilization.md) |
| Streaming dataset, large dataset handling | [data_pipeline/streaming.md](data_pipeline/streaming.md) | [data_pipeline/dataloading.md](data_pipeline/dataloading.md) |
| Training loop, AMP, mixed precision | [training/training_loop.md](training/training_loop.md) | [debugging/nan_loss.md](debugging/nan_loss.md) |
| Gradient accumulation, effective batch size | [training/gradient_accumulation.md](training/gradient_accumulation.md) | [training/training_loop.md](training/training_loop.md) |
| GPU util low, profiling, nsys, bottleneck | [performance/gpu_utilization.md](performance/gpu_utilization.md) | [training/training_loop.md](training/training_loop.md) |
| NaN loss, exploding gradients, instability | [debugging/nan_loss.md](debugging/nan_loss.md) | [training/training_loop.md](training/training_loop.md) |
| Deadlock, hang, distributed stuck | [debugging/distributed_deadlock.md](debugging/distributed_deadlock.md) | [distributed/torchrun.md](distributed/torchrun.md) |
| Poor convergence, learning rate, warmup | [debugging/convergence.md](debugging/convergence.md) | [training/training_loop.md](training/training_loop.md) |

---

## 📚 Knowledge Base Structure

### 🏗️ Core Topics

#### Distributed Training
- **[distributed/ddp.md](distributed/ddp.md)** — DistributedDataParallel setup, process group initialization, sampler requirements
- **[distributed/fsdp.md](distributed/fsdp.md)** — Fully Sharded Data Parallel, sharding strategies, memory estimation
- **[distributed/torchrun.md](distributed/torchrun.md)** — Launching with torchrun, single-node, multi-node, rendezvous backends

#### Data Pipeline
- **[data_pipeline/dataloading.md](data_pipeline/dataloading.md)** — The 6 required DataLoader settings, num_workers tuning, prefetching
- **[data_pipeline/streaming.md](data_pipeline/streaming.md)** — Handling large/streaming datasets, efficient prefetching patterns

#### Training Mechanics
- **[training/training_loop.md](training/training_loop.md)** — Complete training loop template with AMP, gradient accumulation, mixed precision
- **[training/gradient_accumulation.md](training/gradient_accumulation.md)** — Effective batch size calculation, memory trade-offs

#### Performance & Profiling
- **[performance/gpu_utilization.md](performance/gpu_utilization.md)** — Measuring utilization, identifying bottlenecks, profiling with nvidia-smi, nsys

#### Debugging & Troubleshooting
- **[debugging/nan_loss.md](debugging/nan_loss.md)** — Root causes of NaN, checking data, gradient clipping, learning rates
- **[debugging/distributed_deadlock.md](debugging/distributed_deadlock.md)** — Diagnosing hangs, NCCL_DEBUG, collective operation mismatches
- **[debugging/convergence.md](debugging/convergence.md)** — Poor convergence, learning rate schedules, warmup strategies

### 🎁 Templates & Configs

#### Code Templates
- **[templates/trainer.py](templates/trainer.py)** — Full-featured trainer class (DDP/FSDP agnostic)
- **[templates/ddp_template.py](templates/ddp_template.py)** — Minimal DDP training example
- **[templates/fsdp_template.py](templates/fsdp_template.py)** — Minimal FSDP training example
- **[templates/dataloader.py](templates/dataloader.py)** — Production-grade DataLoader builder

#### Config Examples
- **[configs/single_gpu.yaml](configs/single_gpu.yaml)** — Single-GPU training config
- **[configs/ddp_8gpu.yaml](configs/ddp_8gpu.yaml)** — 8-GPU DDP config
- **[configs/fsdp_large_model.yaml](configs/fsdp_large_model.yaml)** — Large model FSDP config

### 📖 Reference Documentation
- **[principles.md](principles.md)** — Philosophy, the 5 Laws, anti-patterns, performance budgets
- **[decision_tree.md](decision_tree.md)** — Full decision flowchart with memory formulas and strategy selection matrix

---

## ⚡ Non-Negotiable Rules (Always Enforce)

1. **NEVER use `python train.py` for distributed training.** Always use `torchrun`.
2. **ALWAYS use `torch.cuda.amp.GradScaler` with AMP.** Unscaled fp16 training diverges.
3. **ALWAYS set `pin_memory=True` and `non_blocking=True`** in DataLoaders.
4. **ALWAYS clip gradients** (`torch.nn.utils.clip_grad_norm_`, max_norm=1.0).
5. **ALWAYS save checkpoints on rank 0 only** (or use `dist.barrier()` before loading).
6. **ALWAYS call `set_epoch(epoch)` on DistributedSampler** before each epoch.
7. **NEVER call `.item()` inside the training loop** (forces GPU sync, kills throughput).
8. **ALWAYS use `torch.compile()` for PyTorch ≥ 2.0** unless debugging.

---

## 🚀 Quick Reference: Common Commands

### Single Node, N GPUs
```bash
torchrun --standalone --nproc-per-node=4 train.py --config config.yaml
```

### Multi-Node Distributed Training
```bash
# Run on each node (master_host = node-0 IP)
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --rdzv-id=job42 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=master_host:29500 \
  train.py --config config.yaml
```

### Debug Distributed Hangs
```bash
NCCL_DEBUG=INFO torchrun --standalone --nproc-per-node=4 train.py
```

### Profile GPU Utilization
```bash
nvidia-smi dmon  # Real-time GPU stats
nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv -l 1  # CSV output
```

---

## 🔍 Memory Estimation Formulas

### DDP (fp32, per GPU)
```
VRAM = params * 4 * 4  # 4 bytes/param × 4 (model + grad + 2× optimizer state)
```

### FSDP FULL_SHARD (fp32, per GPU, N total GPUs)
```
VRAM = (params * 4 * 4) / N + activation_memory
```

### AMP (fp16 model, fp32 master weights)
```
VRAM = params * 6  # 2 bytes (fp16) + 4 bytes (fp32 master)
```

### Activation Memory (rough estimate)
```
activation_memory = batch_size * seq_len * hidden_dim * num_layers * 2
```

---

## 🎓 Learning Path

**New to distributed PyTorch?**
1. Start: [principles.md](principles.md) — understand the philosophy
2. Then: [decision_tree.md](decision_tree.md) — choose your strategy
3. Then: [distributed/ddp.md](distributed/ddp.md) or [distributed/fsdp.md](distributed/fsdp.md) — learn your approach
4. Then: [training/training_loop.md](training/training_loop.md) — write production code
5. Finally: [performance/gpu_utilization.md](performance/gpu_utilization.md) — optimize

**Stuck with OOM?**
1. [decision_tree.md](decision_tree.md) → Memory section
2. [distributed/fsdp.md](distributed/fsdp.md) → Sharding strategies
3. [training/gradient_accumulation.md](training/gradient_accumulation.md)

**Debugging NaN losses?**
1. [debugging/nan_loss.md](debugging/nan_loss.md)
2. [training/training_loop.md](training/training_loop.md) → Gradient clipping section

**Distributed training is hanging?**
1. [debugging/distributed_deadlock.md](debugging/distributed_deadlock.md)
2. [distributed/torchrun.md](distributed/torchrun.md)

---

## 📊 Performance Targets (Benchmarks)

| Metric | Target | How to Measure |
|--------|--------|---|
| GPU Utilization | > 85% | `nvidia-smi dmon` |
| DataLoader latency | < 100ms per batch | Log `time.perf_counter()` |
| DDP scaling efficiency | > 85% | throughput @ 8 GPUs / (throughput @ 1 GPU × 8) |
| Gradient accumulation overhead | < 5% | Compare throughput with vs without |
| Checkpoint write time | < 5 sec | Time `torch.save()` on rank 0 |
| Training loop iteration | < 1 sec | Profile with `torch.profiler` |

---

## 🛠️ Working with This Knowledge Base

### For Developers Implementing Code
1. Read the relevant `.md` file in this repo for strategy
2. Copy the template from `templates/` directory
3. Adapt config from `configs/` directory
4. Refer to routing table above when stuck

### For Debugging Live Issues
1. Check the "Non-Negotiable Rules" section above
2. Use the Router table to find relevant docs
3. If stuck in distributed deadlock: [debugging/distributed_deadlock.md](debugging/distributed_deadlock.md)
4. If performance issues: [performance/gpu_utilization.md](performance/gpu_utilization.md)

### Directory Structure
```
.
├── README.md (you are here)
├── principles.md — Philosophy and anti-patterns
├── decision_tree.md — Strategy selection flowchart
├── training/
│   ├── training_loop.md
│   └── gradient_accumulation.md
├── distributed/
│   ├── ddp.md
│   ├── fsdp.md
│   └── torchrun.md
├── data_pipeline/
│   ├── dataloading.md
│   └── streaming.md
├── debugging/
│   ├── nan_loss.md
│   ├── distributed_deadlock.md
│   └── convergence.md
├── performance/
│   └── gpu_utilization.md
├── templates/
│   ├── trainer.py
│   ├── ddp_template.py
│   ├── fsdp_template.py
│   └── dataloader.py
├── configs/
│   ├── single_gpu.yaml
│   ├── ddp_8gpu.yaml
│   └── fsdp_large_model.yaml
└── references/
    └── [Extended reference implementations]
```

---

## 📝 Contributing

This is a living knowledge base. When you discover:
- A new bug pattern or fix → add to appropriate `debugging/` doc
- A performance optimization → add to `performance/` or `principles.md`
- A new strategy or anti-pattern → update `decision_tree.md`
- New templates → add to `templates/` and reference in relevant `.md`

---

## 🔗 External Resources

- **[PyTorch DDP Docs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)**
- **[PyTorch FSDP Docs](https://pytorch.org/docs/stable/fsdp.html)**
- **[torchrun Docs](https://pytorch.org/docs/stable/elastic/run.html)**
- **[torch.cuda.amp Docs](https://pytorch.org/docs/stable/amp.html)**
- **[NVIDIA Distributed Training Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/)**

---

## 📞 Quick Troubleshooting Checklist

Before opening an issue, check:

- [ ] Did you use `torchrun`, not `python train.py`?
- [ ] Did you set `set_epoch()` on DistributedSampler every epoch?
- [ ] Did you use `non_blocking=True` for host-to-device transfers?
- [ ] Is your DataLoader using `num_workers >= 4`?
- [ ] Did you set `pin_memory=True`?
- [ ] Are you using `GradScaler` with AMP?
- [ ] Did you clip gradients before optimizer.step()?
- [ ] Is GPU utilization > 70%? (If not, check DataLoader)
- [ ] Are you saving checkpoints on rank 0 only?
- [ ] Did you set seed (torch, numpy, random, cuda) for reproducibility?

---

**Version:** 1.0 | **Last Updated:** May 2026 | **PyTorch:** ≥ 2.0


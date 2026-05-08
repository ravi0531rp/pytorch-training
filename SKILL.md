---
name: pytorch-training
description: >
  Production-grade PyTorch training SOP. Use this skill for ANY of the following:
  training deep neural networks, setting up DDP or FSDP distributed training,
  optimizing GPU utilization, fixing NaN losses or training instability, writing
  training loops with AMP/mixed precision, setting up efficient dataloaders,
  gradient accumulation, checkpointing strategies, debugging distributed training
  deadlocks, profiling GPU bottlenecks, torchrun launch configs, multi-node
  training, FSDP sharding strategies, or any PyTorch performance tuning task.
  Trigger even if the user just mentions "training a model", "GPU out of memory",
  "slow dataloader", "DDP", "FSDP", or "torchrun".
---

# PyTorch Training Skill — Central Intelligence Layer

This skill encodes production-grade training SOPs. Always read the relevant
submodule before generating code or advice. Do NOT rely on generic knowledge.

---

## Routing Table

| User Query Contains                        | Read This First                          |
|--------------------------------------------|------------------------------------------|
| DDP / DistributedDataParallel / multi-GPU  | `distributed/ddp.md`                     |
| FSDP / large model / OOM on single GPU     | `distributed/fsdp.md`                    |
| torchrun / launch / multi-node             | `distributed/torchrun.md`                |
| DataLoader / slow data / CPU bottleneck    | `data_pipeline/dataloading.md`           |
| Streaming dataset / large dataset          | `data_pipeline/streaming.md`             |
| Training loop / AMP / mixed precision      | `training/training_loop.md`              |
| Gradient accumulation / effective batch    | `training/gradient_accumulation.md`      |
| GPU util low / profiling / nsys / bottleneck | `performance/gpu_utilization.md`       |
| NaN loss / exploding gradients / instability | `debugging/nan_loss.md`               |
| Deadlock / hang / distributed stuck       | `debugging/distributed_deadlock.md`      |
| Poor convergence / lr / warmup            | `debugging/convergence.md`               |

---

## Decision Tree: Choose Your Strategy

### Step 1 — Model Size

```
Model parameters?
├── < 1B params → DDP is sufficient (see distributed/ddp.md)
├── 1B–7B params → FSDP with SHARD_GRAD_OP (see distributed/fsdp.md)
└── > 7B params → FSDP with FULL_SHARD + activation checkpointing
```

### Step 2 — GPU Count

```
How many GPUs?
├── 1 GPU → Single GPU training (training/training_loop.md)
├── 2–8 GPUs, 1 node → DDP (torchrun --nproc-per-node=N)
├── 8+ GPUs, multi-node → DDP or FSDP + torchrun --nnodes
└── Memory pressure on any config → Switch to FSDP
```

### Step 3 — Batch Size & Memory

```
OOM on GPU?
├── Yes, single GPU → Reduce batch, enable AMP, use gradient accumulation
├── Yes, multi-GPU DDP → Switch to FSDP
└── No OOM but low throughput → Profile (performance/gpu_utilization.md)
```

---

## Non-Negotiable Rules (Always Enforce)

1. **NEVER use `python train.py` for distributed training.** Always use `torchrun`.
2. **ALWAYS use `torch.cuda.amp.GradScaler` with AMP.** Unscaled training with fp16 diverges.
3. **ALWAYS set `pin_memory=True` and `non_blocking=True`** in DataLoaders.
4. **ALWAYS clip gradients** (`torch.nn.utils.clip_grad_norm_`, max_norm=1.0).
5. **ALWAYS save checkpoints on rank 0 only** (or use `dist.barrier()` before loading).
6. **ALWAYS call `set_epoch(epoch)` on DistributedSampler** before each epoch.
7. **NEVER call `.item()` inside the training loop** (forces GPU sync).
8. **ALWAYS use `torch.compile()` for PyTorch ≥ 2.0** unless debugging.

---

## Quick Reference: Launch Commands

```bash
# Single node, N GPUs
torchrun --standalone --nproc-per-node=4 train.py --config config.yaml

# Multi-node (run on each node)
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --rdzv-id=job42 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=master_host:29500 \
  train.py --config config.yaml
```

---

## Template Index

| Template                          | Use Case                              |
|-----------------------------------|---------------------------------------|
| `templates/trainer.py`            | Base trainer class (all strategies)   |
| `templates/ddp_template.py`       | DDP training script                   |
| `templates/fsdp_template.py`      | FSDP training script                  |
| `templates/dataloader.py`         | Production DataLoader setup           |

---

## Config Index

| Config                            | Use Case                              |
|-----------------------------------|---------------------------------------|
| `configs/single_gpu.yaml`         | Single GPU baseline                   |
| `configs/ddp_8gpu.yaml`           | 8-GPU DDP                             |
| `configs/fsdp_large_model.yaml`   | FSDP for large models                 |

---

## Workflow for Code Generation Tasks

1. Read relevant submodule(s) from routing table above.
2. Read the appropriate template from `templates/`.
3. Load the matching YAML config from `configs/`.
4. Generate code that strictly follows the submodule rules.
5. Include: proper rank guards, error handling, logging, checkpoint logic.
6. Never omit AMP, gradient clipping, or sampler epoch setting.

# Decision Tree: DDP vs FSDP vs Single GPU

## Full Decision Logic

```
START: How many GPUs do you have?
│
├── 1 GPU
│   └── Use: Single GPU training (training/training_loop.md)
│       ├── Enable AMP always
│       ├── Use gradient accumulation if batch doesn't fit
│       └── torch.compile() for PyTorch >= 2.0
│
└── Multiple GPUs
    │
    ├── What is your model size?
    │   │
    │   ├── < 1B parameters AND fits in single GPU VRAM with batch_size >= 8
    │   │   └── USE DDP (distributed/ddp.md)
    │   │       ├── Simple to set up, near-linear scaling
    │   │       ├── Each GPU holds full model replica
    │   │       └── Best throughput for medium models
    │   │
    │   ├── 1B–7B parameters OR getting OOM with DDP
    │   │   └── USE FSDP with ShardingStrategy.SHARD_GRAD_OP
    │   │       ├── Shards optimizer states + gradients across GPUs
    │   │       ├── Full parameters during forward/backward
    │   │       └── ~2-3x memory savings vs DDP
    │   │
    │   └── > 7B parameters OR need to maximize memory
    │       └── USE FSDP with ShardingStrategy.FULL_SHARD
    │           ├── Shards parameters + gradients + optimizer states
    │           ├── Enable activation checkpointing
    │           └── Consider CPU offload if still OOM
    │
    └── Multi-node?
        ├── Yes → torchrun with --nnodes + c10d rendezvous (distributed/torchrun.md)
        └── No  → torchrun --standalone --nproc-per-node=N
```

## Memory Estimation Formula

```python
# Rough VRAM requirement per GPU for DDP (fp32)
vram_ddp = params * 4 * 4  # 4 bytes/param * 4 (model+grad+optimizer)

# For FSDP FULL_SHARD with N GPUs (fp32)
vram_fsdp = (params * 4 * 4) / N + activation_memory

# For AMP (fp16 model, fp32 master weights)
vram_amp = params * 6  # 2 bytes (fp16) + 4 bytes (fp32 master)

# Activation memory per batch (rough estimate)
activation_memory = batch_size * seq_len * hidden_dim * num_layers * 2
```

## Batch Size Decision

```
Target effective_batch_size (e.g., 256)?
│
├── fits_in_gpu_memory(effective_batch_size)?
│   └── YES: Use it directly, no accumulation needed
│
└── NO: Use gradient accumulation
    effective_batch = micro_batch * accumulation_steps * world_size
    Example: 256 = 4 * 8 * 8  (micro_batch=4, accum=8, 8 GPUs)
```

## Quick Selection Matrix

| Scenario | Strategy | Sharding | Activation Ckpt |
|---|---|---|---|
| ResNet-50, 4x A100 | DDP | N/A | No |
| ViT-L, 8x A100 | DDP | N/A | Optional |
| LLaMA-7B, 8x A100 80GB | DDP or FSDP | SHARD_GRAD_OP | No |
| LLaMA-13B, 8x A100 40GB | FSDP | FULL_SHARD | Yes |
| LLaMA-70B, 16x A100 80GB | FSDP | FULL_SHARD | Yes |
| Any model, memory critical | FSDP | FULL_SHARD + CPU offload | Yes |

## When to Switch Strategy Mid-Training

| Signal | Action |
|---|---|
| OOM on DDP | Switch to FSDP SHARD_GRAD_OP |
| OOM on FSDP SHARD_GRAD_OP | Switch to FULL_SHARD |
| OOM on FSDP FULL_SHARD | Add activation checkpointing |
| Still OOM | Enable CPU offload (significant throughput cost) |
| GPU util < 70% | Check DataLoader before anything else |
| DDP scaling efficiency < 85% | Profile NCCL communication overhead |

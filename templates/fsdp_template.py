"""
FSDP Training Template — Production Grade
Usage: torchrun --standalone --nproc-per-node=N fsdp_template.py --config config.yaml
"""
import os
import logging
import argparse
import datetime
import functools
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Rank %(process)d][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed Init
# ---------------------------------------------------------------------------

def init_distributed():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP Model Setup
# ---------------------------------------------------------------------------

def wrap_fsdp(model, TransformerLayerClass, cfg):
    """
    Wrap model with FSDP. Wraps at transformer-block level for best performance.
    """
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerLayerClass},
    )

    # bf16 preferred: no overflow, supported on Ampere+ (A100, 3090, 4090)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,   # fp32 gradient reduction for stability
        buffer_dtype=torch.bfloat16,
    )

    sharding = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }[cfg.sharding_strategy]

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,           # Required for torch.compile
        limit_all_gathers=True,         # Reduces peak memory
        sync_module_states=True,        # Sync init across ranks
    )

    if cfg.activation_checkpointing:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, TransformerLayerClass),
        )
        logger.info("Activation checkpointing enabled")

    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_fsdp_checkpoint(model, optimizer, scheduler, epoch, cfg, rank):
    """Gather all shards to rank 0 and save."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    if rank == 0:
        path = Path(cfg.output_dir) / f"fsdp_checkpoint_epoch{epoch:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, path)
        logger.info(f"FSDP checkpoint saved: {path}")

    dist.barrier()


def load_fsdp_checkpoint(model, optimizer, scheduler, path, rank):
    """Load checkpoint (model state broadcast from rank 0)."""
    if rank == 0:
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = {"model": None, "optimizer": None, "scheduler": None, "epoch": 0}

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if rank == 0:
            model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, rank, world_size):
    from your_dataset import TrainDataset, ValDataset  # Replace

    train_ds = TrainDataset(cfg)
    val_ds = ValDataset(cfg)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=cfg.seed, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                      shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                               num_workers=cfg.num_workers, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
                             num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Training Loop (FSDP — no GradScaler needed with bf16)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, epoch, rank, cfg, sampler):
    model.train()
    sampler.set_epoch(epoch)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    running_loss = torch.tensor(0.0, device="cuda")
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        inputs = batch["input"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels) / cfg.grad_accum_steps

        loss.backward()  # No GradScaler needed with bf16 FSDP
        running_loss += loss.detach()
        steps += 1

        if (step + 1) % cfg.grad_accum_steps == 0:
            grad_norm = model.clip_grad_norm_(max_norm=1.0)  # FSDP-aware clipping
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if rank == 0 and step % cfg.log_every == 0:
                avg = running_loss.item() / steps
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"Loss: {avg:.4f} | GradNorm: {grad_norm:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss.zero_()
                steps = 0


@torch.no_grad()
def validate(model, loader, rank, world_size):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device="cuda")
    total_n = torch.tensor(0, device="cuda")

    for batch in loader:
        inputs = batch["input"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        total_loss += loss * labels.size(0)
        total_n += labels.size(0)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
    return (total_loss / total_n).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    local_rank, rank, world_size = init_distributed()

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    from your_model import MyModel, TransformerLayer  # Replace

    model = MyModel(cfg)
    model = wrap_fsdp(model, TransformerLayer, cfg)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.total_steps - cfg.warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_steps])

    train_loader, val_loader, train_sampler = build_dataloaders(cfg, rank, world_size)

    start_epoch = 0
    if cfg.resume:
        start_epoch = load_fsdp_checkpoint(model, optimizer, scheduler, cfg.resume, rank)

    for epoch in range(start_epoch, cfg.num_epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, epoch, rank, cfg, train_sampler)

        if epoch % cfg.eval_every == 0:
            val_loss = validate(model, val_loader, rank, world_size)
            if rank == 0:
                logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        save_fsdp_checkpoint(model, optimizer, scheduler, epoch, cfg, rank)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    import yaml
    from types import SimpleNamespace
    with open(args.config) as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))

    main(cfg)

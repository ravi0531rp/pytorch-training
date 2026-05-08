"""
DDP Training Template — Production Grade
Usage: torchrun --standalone --nproc-per-node=N ddp_template.py --config config.yaml
"""
import os
import logging
import argparse
import datetime
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
# Model + Optimizer + Scheduler
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    from your_model import MyModel  # Replace with actual model
    model = MyModel(cfg).to(device)
    model = torch.compile(model)  # PyTorch >= 2.0
    model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    return model


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, cfg):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.total_steps - cfg.warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_steps])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, rank, world_size):
    from your_dataset import TrainDataset, ValDataset  # Replace

    train_dataset = TrainDataset(cfg)
    val_dataset = ValDataset(cfg)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=cfg.seed, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, cfg, rank):
    if rank != 0:
        return
    ckpt = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "cfg": vars(cfg),
    }
    path = Path(cfg.output_dir) / f"checkpoint_epoch{epoch:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["step"]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, rank, cfg, sampler):
    model.train()
    sampler.set_epoch(epoch)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    running_loss = torch.tensor(0.0, device=f"cuda:{rank}")
    steps_in_epoch = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        inputs = batch["input"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        is_last_accum = (step + 1) % cfg.grad_accum_steps == 0
        sync_ctx = nullcontext() if is_last_accum else model.no_sync()

        with sync_ctx:
            with autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / cfg.grad_accum_steps
            scaler.scale(loss).backward()

        running_loss += loss.detach()
        steps_in_epoch += 1

        if is_last_accum:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if rank == 0 and step % cfg.log_every == 0:
                avg_loss = running_loss.item() / steps_in_epoch
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"Loss: {avg_loss:.4f} | GradNorm: {grad_norm:.3f} | LR: {lr:.2e}"
                )
                running_loss.zero_()
                steps_in_epoch = 0


@torch.no_grad()
def validate(model, loader, rank, world_size):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device=f"cuda:{rank}")
    total_samples = torch.tensor(0, device=f"cuda:{rank}")

    for batch in loader:
        inputs = batch["input"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        total_loss += loss * labels.size(0)
        total_samples += labels.size(0)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    return (total_loss / total_samples).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    local_rank, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Seed (rank-offset for data aug stochasticity, same for model init)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scaler = GradScaler(enabled=True)
    train_loader, val_loader, train_sampler = build_dataloaders(cfg, rank, world_size)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 0
    if cfg.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, scaler, cfg.resume, device)

    for epoch in range(start_epoch, cfg.num_epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, rank, cfg, train_sampler)

        if epoch % cfg.eval_every == 0:
            val_loss = validate(model, val_loader, rank, world_size)
            if rank == 0:
                logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        dist.barrier()
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, 0, cfg, rank)

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

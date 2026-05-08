"""
Base Trainer — strategy-agnostic wrapper.
Extend or instantiate directly. Handles: AMP, grad accum, logging, checkpointing.
"""
import os
import logging
import datetime
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: torch.nn.Module,
        cfg,
        rank: int = 0,
        world_size: int = 1,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.device = next(model.parameters()).device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
        self.global_step = 0
        self.best_val_loss = float("inf")

    @property
    def is_main(self):
        return self.rank == 0

    def train_epoch(self, loader, epoch: int, sampler=None):
        self.model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        running_loss = torch.tensor(0.0, device=self.device)
        n_steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            is_last_accum = (step + 1) % self.cfg.grad_accum_steps == 0

            # Skip DDP sync on non-final accumulation steps
            sync_ctx = (
                nullcontext()
                if (is_last_accum or not self.is_distributed)
                else self.model.no_sync()
            )

            with sync_ctx:
                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self._forward(batch) / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()

            running_loss += loss.detach()
            n_steps += 1

            if is_last_accum:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()
                self.global_step += 1

                if self.is_main and self.global_step % self.cfg.log_every == 0:
                    avg_loss = running_loss.item() / n_steps
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.cfg.lr
                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | GradNorm: {grad_norm:.3f} | LR: {lr:.2e}"
                    )
                    running_loss.zero_()
                    n_steps = 0

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_n = torch.tensor(0, device=self.device)

        for batch in loader:
            batch = self._to_device(batch)
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self._forward(batch)
            total_loss += loss * batch["label"].size(0)
            total_n += batch["label"].size(0)

        if self.is_distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_n, op=dist.ReduceOp.SUM)

        return (total_loss / total_n).item()

    def _forward(self, batch) -> torch.Tensor:
        """Override for custom forward logic."""
        outputs = self.model(batch["input"])
        return self.criterion(outputs, batch["label"])

    def _to_device(self, batch):
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        if not self.is_main:
            return
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self._get_model_state(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "val_loss": val_loss,
            "cfg": vars(self.cfg),
        }
        path = Path(self.cfg.output_dir) / f"checkpoint_epoch{epoch:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = Path(self.cfg.output_dir) / "best_checkpoint.pt"
            torch.save(ckpt, best_path)
            logger.info(f"New best model: val_loss={val_loss:.4f}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model_state = ckpt["model"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt.get("global_step", 0)
        return ckpt["epoch"]

    def _get_model_state(self):
        if hasattr(self.model, "module"):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def fit(self, train_loader, val_loader, sampler=None):
        start_epoch = 0
        if getattr(self.cfg, "resume", None):
            start_epoch = self.load_checkpoint(self.cfg.resume) + 1

        for epoch in range(start_epoch, self.cfg.num_epochs):
            self.train_epoch(train_loader, epoch, sampler)

            if epoch % self.cfg.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                if self.is_main:
                    logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
                self.save_checkpoint(epoch, val_loss)

            if self.is_distributed:
                dist.barrier()

from __future__ import annotations

import json
import math
import os
import shutil
import time
from itertools import islice
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logging import create_progress_bar, log_metrics, setup_logger
from .losses import can_multitask_loss
from .metrics import regression_metrics

HISTORY_FILENAME = "history.csv"
CONFIG_FILENAME = "train_config.json"
TRAIN_LOG_FILENAME = "train.log"


def _amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    return torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16


def _mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = set().union(*(x.keys() for x in items))
    return {
        key: float(np.mean([x[key] for x in items if key in x]))
        for key in sorted(keys)
    }


def build_parameter_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build AdamW parameter groups with no decay on biases / norm-like 1-D params.

    This mirrors the Stage 1 optimizer policy while keeping Stage 3 independent
    from Stage 1 implementation details.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    if not decay and not no_decay:
        raise ValueError("No trainable parameters found.")

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "name": "decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": float(learning_rate),
                "weight_decay": 0.0,
                "name": "no_decay",
            }
        )
    return groups


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay, matching the Stage 1 policy."""
    total = max(int(total_steps), 1)
    warmup = max(int(round(total * float(warmup_ratio))), 0)
    floor = float(min_ratio)

    if not 0.0 <= float(warmup_ratio) < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"min_ratio must be in [0, 1], got {min_ratio}")

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _flatten_history_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "epoch": row.get("epoch"),
        "minutes": row.get("minutes"),
        "learning_rate": row.get("learning_rate"),
        "global_step": row.get("global_step"),
        "max_gpu_memory_gib": row.get("max_gpu_memory_gib"),
    }
    for key, value in dict(row.get("train") or {}).items():
        flat[f"train/{key}"] = value
    for key, value in dict(row.get("val") or {}).items():
        flat[f"val/{key}"] = value
    return flat


class CANTrainer:
    """Train a Stage 3 continuous-CAN head on top of a video backbone.

    Compared with the first Stage 3 draft this trainer now carries the same
    experiment hygiene used in Stage 1: shared tqdm logging, warmup+cosine LR,
    W&B epoch logging, persistent history/config files, checkpoint resume state,
    and optional local->Drive synchronization.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any | None = None,
        device: str | torch.device = "cuda",
        grad_accum_steps: int = 1,
        grad_clip_norm: float = 1.0,
        amp_dtype: str = "bf16",
        loss_weights: Mapping[str, float] | None = None,
        stats: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        sync_dir: str | Path | None = None,
        wandb_enabled: bool = False,
        log_interval: int = 20,
        config: Mapping[str, Any] | None = None,
        logger: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.grad_clip_norm = float(grad_clip_norm)
        self.amp_dtype = _amp_dtype(amp_dtype)
        self.loss_weights = dict(loss_weights or {})
        self.stats = dict(stats or {})
        self.wandb_enabled = bool(wandb_enabled)
        self.log_interval = max(1, int(log_interval))
        self.config = dict(config or {})
        self.global_step = 0

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.sync_dir = Path(sync_dir) if sync_dir is not None else None

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.sync_dir is not None:
            self.sync_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            self.output_dir / TRAIN_LOG_FILENAME
            if self.output_dir is not None
            else None
        )
        self.logger = logger or setup_logger(
            "blackbox_detection.stage3.can",
            log_file=log_file,
        )

        self.model.to(self.device)

        if self.output_dir is not None and self.config:
            config_path = self.output_dir / CONFIG_FILENAME
            config_path.write_text(
                json.dumps(self.config, indent=2, default=str),
                encoding="utf-8",
            )
            self._sync_file(config_path)

    def _move(self, batch):
        return (
            batch["video"].to(self.device, non_blocking=True),
            batch["target"].to(self.device, non_blocking=True),
            batch["valid"].to(self.device, non_blocking=True),
        )

    def _optimizer_step(self) -> None:
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                self.grad_clip_norm,
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _sync_file(self, source: Path | None, *, retries: int = 2) -> Path | None:
        """Copy a local output to persistent storage with atomic replacement.

        Google Drive FUSE occasionally raises errno 107. Training should not
        lose the already-written local checkpoint in that case, so syncing is
        retried and a clear warning is emitted rather than deleting local state.
        """
        if source is None or self.sync_dir is None or not source.is_file():
            return None

        destination = self.sync_dir / source.name
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None

        for attempt in range(1, retries + 2):
            tmp = destination.with_name(destination.name + ".sync.tmp")
            try:
                tmp.unlink(missing_ok=True)
                shutil.copy2(source, tmp)
                os.replace(tmp, destination)
                return destination
            except OSError as exc:
                last_error = exc
                tmp.unlink(missing_ok=True)
                if attempt <= retries:
                    time.sleep(2.0 * attempt)

        self.logger.warning(
            "Could not sync %s to persistent storage %s after retries: %r. "
            "The local file is still safe until the runtime ends.",
            source,
            destination,
            last_error,
        )
        return None

    def _save_history(self, history: list[dict[str, Any]]) -> Path | None:
        if self.output_dir is None:
            return None
        path = self.output_dir / HISTORY_FILENAME
        pd.DataFrame([_flatten_history_row(row) for row in history]).to_csv(
            path,
            index=False,
            encoding="utf-8",
        )
        self._sync_file(path)
        log_path = self.output_dir / TRAIN_LOG_FILENAME
        self._sync_file(log_path)
        return path

    def _save_checkpoint(
        self,
        *,
        epoch: int,
        best_score: float,
        is_best: bool,
        history: list[dict[str, Any]],
    ) -> tuple[Path | None, Path | None]:
        if self.output_dir is None:
            return None, None

        latest = self.output_dir / "latest.pt"
        save_checkpoint(
            latest,
            model=self.model,
            epoch=epoch,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            best_score=best_score,
            config=self.config,
            extra={"history": history},
            # Stage 1 exposed notebook-specific RNG restore problems on Colab.
            # Epoch-boundary resume is robust here without serializing RNG state.
            save_rng_state=False,
        )
        self._sync_file(latest)

        best_path: Path | None = None
        if is_best:
            best_path = self.output_dir / "best.pt"
            tmp = best_path.with_name(best_path.name + ".tmp")
            tmp.unlink(missing_ok=True)
            shutil.copy2(latest, tmp)
            os.replace(tmp, best_path)
            self._sync_file(best_path)

        return latest, best_path

    def train_epoch(
        self,
        loader,
        max_steps: int | None = None,
        *,
        epoch: int | None = None,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logs: list[dict[str, float]] = []
        micro_steps = 0
        running_loss = 0.0
        total_steps = len(loader)
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        desc = f"Train {epoch}" if epoch is not None else "Train"
        progress = create_progress_bar(
            islice(loader, total_steps),
            total=total_steps,
            desc=desc,
            leave=False,
            mininterval=0.5,
        )

        for step, batch in enumerate(progress):

            video, target, valid = self._move(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                out = self.model(video)
                loss, parts = can_multitask_loss(
                    out,
                    target,
                    valid,
                    self.loss_weights,
                )
                scaled_loss = loss / self.grad_accum_steps

            scaled_loss.backward()
            micro_steps += 1

            if micro_steps % self.grad_accum_steps == 0:
                self._optimizer_step()

            logs.append(parts)
            running_loss += float(parts["total"])

            if (
                step == 0
                or (step + 1) % self.log_interval == 0
                or step + 1 == total_steps
            ):
                progress.set_postfix(
                    loss=f"{parts['total']:.4f}",
                    avg=f"{running_loss / micro_steps:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        progress.close()

        if micro_steps > 0 and micro_steps % self.grad_accum_steps != 0:
            self._optimizer_step()

        return _mean_dict(logs)

    @torch.inference_mode()
    def evaluate(
        self,
        loader,
        max_steps: int | None = None,
        *,
        epoch: int | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        logs: list[dict[str, float]] = []
        running_loss = 0.0
        num_steps = 0

        total_steps = len(loader)
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        desc = f"Val {epoch}" if epoch is not None else "Val"
        progress = create_progress_bar(
            islice(loader, total_steps),
            total=total_steps,
            desc=desc,
            leave=False,
            mininterval=0.5,
        )

        for step, batch in enumerate(progress):

            video, target, valid = self._move(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                out = self.model(video)
                _, loss_parts = can_multitask_loss(
                    out,
                    target,
                    valid,
                    self.loss_weights,
                )

            metrics = regression_metrics(out, target, valid, self.stats)
            logs.append({**loss_parts, **metrics})
            num_steps += 1
            running_loss += float(loss_parts["total"])

            if (
                step == 0
                or (step + 1) % self.log_interval == 0
                or step + 1 == total_steps
            ):
                progress.set_postfix(
                    loss=f"{loss_parts['total']:.4f}",
                    avg=f"{running_loss / num_steps:.4f}",
                )

        progress.close()
        return _mean_dict(logs)

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        epochs: int = 1,
        max_train_steps: int | None = None,
        max_val_steps: int | None = None,
        resume_from: str | Path | None = None,
        early_stopping_patience: int = 0,
    ) -> list[dict[str, Any]]:
        start_epoch = 1
        best = float("inf")
        history: list[dict[str, Any]] = []
        epochs_without_improvement = 0

        if resume_from is not None and Path(resume_from).is_file():
            try:
                metadata = load_checkpoint(
                    resume_from,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    map_location="cpu",
                    restore_rng_state=False,
                )
            except ValueError as exc:
                # The pre-patch Stage 3 optimizer used one AdamW parameter group,
                # while this trainer uses Stage-1-style decay/no-decay groups.
                # Preserve learned model weights / completed epoch instead of
                # making an older latest.pt unusable; optimizer/scheduler restart.
                self.logger.warning(
                    "Full optimizer/scheduler resume failed (%r). "
                    "Falling back to model-only resume; LR schedule restarts.",
                    exc,
                )
                metadata = load_checkpoint(
                    resume_from,
                    model=self.model,
                    optimizer=None,
                    scheduler=None,
                    map_location="cpu",
                    restore_rng_state=False,
                )
            start_epoch = int(metadata.get("epoch") or 0) + 1
            self.global_step = int(metadata.get("global_step") or 0)
            previous = dict(metadata.get("extra") or {}).get("history", [])
            if isinstance(previous, list):
                history = previous

            checkpoint_best = metadata.get("best_score")
            if checkpoint_best is not None:
                best = float(checkpoint_best)
            else:
                previous_scores = [
                    float(row.get("val", {}).get("total", float("inf")))
                    for row in history
                    if isinstance(row, dict)
                ]
                if previous_scores:
                    best = min(previous_scores)

            if history and math.isfinite(best):
                best_epochs = [
                    int(row["epoch"])
                    for row in history
                    if float(row.get("val", {}).get("total", float("inf"))) <= best + 1e-12
                ]
                if best_epochs:
                    epochs_without_improvement = max(
                        0,
                        start_epoch - 1 - max(best_epochs),
                    )

            self.logger.info(
                "Resumed from %s at epoch %d (next=%d, best val total=%.6f, global_step=%d).",
                resume_from,
                start_epoch - 1,
                start_epoch,
                best,
                self.global_step,
            )

        if start_epoch > int(epochs):
            self.logger.info(
                "Checkpoint already completed epoch %d; configured epochs=%d.",
                start_epoch - 1,
                epochs,
            )
            return history

        for epoch in range(start_epoch, int(epochs) + 1):
            t0 = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            train = self.train_epoch(
                train_loader,
                max_train_steps,
                epoch=epoch,
            )
            val = self.evaluate(
                val_loader,
                max_val_steps,
                epoch=epoch,
            )

            minutes = (time.perf_counter() - t0) / 60.0
            max_gpu_memory_gib = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
                if self.device.type == "cuda"
                else 0.0
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])

            row: dict[str, Any] = {
                "epoch": epoch,
                "minutes": minutes,
                "learning_rate": learning_rate,
                "global_step": self.global_step,
                "max_gpu_memory_gib": max_gpu_memory_gib,
                "train": train,
                "val": val,
            }
            history.append(row)

            score = float(val.get("total", float("inf")))
            improved = score < best
            if improved:
                best = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self._save_checkpoint(
                epoch=epoch,
                best_score=best,
                is_best=improved,
                history=history,
            )
            self._save_history(history)

            self.logger.info(
                "Epoch %d/%d | train %.6f | val %.6f | lr %.3e | %.1f min | peak %.2f GiB%s",
                epoch,
                epochs,
                float(train.get("total", float("nan"))),
                score,
                learning_rate,
                minutes,
                max_gpu_memory_gib,
                " <- best" if improved else "",
            )

            if self.wandb_enabled:
                payload: dict[str, Any] = {
                    "epoch": epoch,
                    "optim/learning_rate": learning_rate,
                    "system/epoch_minutes": minutes,
                    "system/max_gpu_memory_gib": max_gpu_memory_gib,
                    "system/global_step": self.global_step,
                    "val/best_total": best,
                }
                payload.update({f"train/{k}": v for k, v in train.items()})
                payload.update({f"val/{k}": v for k, v in val.items()})
                log_metrics(payload, step=epoch)

            if (
                int(early_stopping_patience) > 0
                and epochs_without_improvement >= int(early_stopping_patience)
            ):
                self.logger.info(
                    "Early stopping after %d epoch(s) without validation improvement.",
                    epochs_without_improvement,
                )
                break

        return history


__all__ = [
    "HISTORY_FILENAME",
    "CONFIG_FILENAME",
    "TRAIN_LOG_FILENAME",
    "CANTrainer",
    "build_parameter_groups",
    "build_scheduler",
]

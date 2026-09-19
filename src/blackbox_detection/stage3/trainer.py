from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .losses import can_multitask_loss
from .metrics import regression_metrics


def _amp_dtype(name: str):
    name = str(name).lower()
    return torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16


def _mean_dict(items):
    if not items:
        return {}
    keys = set().union(*(x.keys() for x in items))
    return {k: float(np.mean([x[k] for x in items if k in x])) for k in keys}


class CANTrainer:
    def __init__(
        self,
        model,
        optimizer,
        *,
        device="cuda",
        grad_accum_steps=1,
        grad_clip_norm=1.0,
        amp_dtype="bf16",
        loss_weights=None,
        stats=None,
        output_dir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.grad_clip_norm = float(grad_clip_norm)
        self.amp_dtype = _amp_dtype(amp_dtype)
        self.loss_weights = loss_weights or {}
        self.stats = stats or {}
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def _move(self, batch):
        return (
            batch["video"].to(self.device, non_blocking=True),
            batch["target"].to(self.device, non_blocking=True),
            batch["valid"].to(self.device, non_blocking=True),
        )

    def _optimizer_step(self):
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train_epoch(self, loader, max_steps=None, *, epoch=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logs = []
        micro_steps = 0
        running_loss = 0.0

        total_steps = len(loader)
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        desc = f"Epoch {epoch} train" if epoch is not None else "train"

        pbar = tqdm(
            enumerate(loader),
            total=total_steps,
            desc=desc,
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )

        for step, batch in pbar:
            if max_steps is not None and step >= max_steps:
                break

            video, target, valid = self._move(batch)

            with torch.autocast(
                "cuda",
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
            avg_loss = running_loss / micro_steps

            pbar.set_postfix(
                loss=f"{parts['total']:.4f}",
                avg=f"{avg_loss:.4f}",
            )

        if micro_steps > 0 and micro_steps % self.grad_accum_steps != 0:
            self._optimizer_step()

        return _mean_dict(logs)

    @torch.inference_mode()
    def evaluate(self, loader, max_steps=None, *, epoch=None):
        self.model.eval()

        logs = []
        running_loss = 0.0
        num_steps = 0

        total_steps = len(loader)
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        desc = f"Epoch {epoch} val" if epoch is not None else "val"

        pbar = tqdm(
            enumerate(loader),
            total=total_steps,
            desc=desc,
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )

        for step, batch in pbar:
            if max_steps is not None and step >= max_steps:
                break

            video, target, valid = self._move(batch)

            with torch.autocast(
                "cuda",
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

            metrics = regression_metrics(
                out,
                target,
                valid,
                self.stats,
            )

            logs.append({
                **loss_parts,
                **metrics,
            })

            num_steps += 1
            running_loss += float(loss_parts["total"])

            pbar.set_postfix(
                loss=f"{loss_parts['total']:.4f}",
                avg=f"{running_loss / num_steps:.4f}",
            )

        return _mean_dict(logs)

    def save(self, name, epoch, extra=None):
        if self.output_dir is None:
            return None
        path = self.output_dir / name
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "extra": extra or {},
        }
        # Write atomically so a disconnected Colab runtime does not leave a half checkpoint.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load(self, path, *, load_optimizer=True):
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return {
            "epoch": int(checkpoint.get("epoch", 0)),
            "extra": checkpoint.get("extra", {}),
        }

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        epochs=1,
        max_train_steps=None,
        max_val_steps=None,
        resume_from=None,
    ):
        start_epoch = 1
        best = float("inf")
        history = []

        if resume_from is not None and Path(resume_from).exists():
            metadata = self.load(resume_from, load_optimizer=True)
            start_epoch = metadata["epoch"] + 1
            previous = metadata.get("extra", {}).get("history", [])
            if isinstance(previous, list):
                history = previous
            previous_scores = [
                float(row.get("val", {}).get("total", float("inf")))
                for row in history
                if isinstance(row, dict)
            ]
            if previous_scores:
                best = min(previous_scores)
            print(f"Resumed from {resume_from} at epoch {metadata['epoch']}")

        for epoch in range(start_epoch, int(epochs) + 1):
            t0 = time.time()
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
            row = {
                "epoch": epoch,
                "minutes": (time.time() - t0) / 60,
                "train": train,
                "val": val,
            }
            history.append(row)
            print(row)
            score = float(val.get("total", float("inf")))
            self.save("latest.pt", epoch, {"history": history})
            if score < best:
                best = score
                self.save("best.pt", epoch, {"history": history})

        return history

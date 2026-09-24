from __future__ import annotations

from itertools import islice
from typing import Any

import numpy as np
import torch

from ..utils.logging import create_progress_bar, log_metrics
from .proxy_metrics import Stage3ValidationAccumulator
from .trainer import CANTrainer
from .v5_losses import v5_multitask_loss


def _mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = set().union(*(x.keys() for x in items))
    return {
        key: float(np.mean([x[key] for x in items if key in x]))
        for key in sorted(keys)
    }


class V5CANTrainer(CANTrainer):
    """CANTrainer with v5 auxiliary targets/losses.

    Checkpointing, resume, scheduler handling, W&B logging and Drive sync stay
    inherited from the already-tested Stage-3 trainer.
    """

    def _move_v5(self, batch):
        video = batch["video"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        valid = batch["valid"].to(self.device, non_blocking=True)
        aux = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch["aux"].items()
            if torch.is_tensor(value)
        }
        return video, target, valid, aux

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
            video, target, valid, aux = self._move_v5(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                out = self.model(video)
                loss, parts = v5_multitask_loss(
                    out,
                    target,
                    valid,
                    aux,
                    self.loss_weights,
                    stats=self.stats,
                )
                scaled_loss = loss / self.grad_accum_steps

            scaled_loss.backward()
            micro_steps += 1
            optimizer_stepped = False

            if micro_steps % self.grad_accum_steps == 0:
                self._optimizer_step()
                optimizer_stepped = True

            logs.append(parts)
            running_loss += float(parts["total"])
            running_avg = running_loss / micro_steps

            should_report = (
                step == 0
                or (step + 1) % self.log_interval == 0
                or step + 1 == total_steps
            )
            if should_report:
                progress.set_postfix(
                    loss=f"{parts['total']:.4f}",
                    avg=f"{running_avg:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

            if (
                self.wandb_enabled
                and optimizer_stepped
                and (
                    (step + 1) % self.log_interval == 0
                    or step + 1 == total_steps
                )
            ):
                payload: dict[str, Any] = {
                    "epoch": int(epoch) if epoch is not None else 0,
                    "train_step/total": float(parts["total"]),
                    "train_step/running_total": float(running_avg),
                    "optim/learning_rate": float(
                        self.optimizer.param_groups[0]["lr"]
                    ),
                    "system/global_step": int(self.global_step),
                }
                for key, value in parts.items():
                    if key != "total":
                        payload[f"train_step/{key}"] = float(value)
                log_metrics(payload)

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

        diagnostics = Stage3ValidationAccumulator(
            self.stats,
            proxy_rules=self.proxy_rules,
        )

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
            video, target, valid, aux = self._move_v5(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                out = self.model(video)
                _, parts = v5_multitask_loss(
                    out,
                    target,
                    valid,
                    aux,
                    self.loss_weights,
                    stats=self.stats,
                )

            diagnostics.update(out, target, valid)
            logs.append(parts)
            num_steps += 1
            running_loss += float(parts["total"])

            if (
                step == 0
                or (step + 1) % self.log_interval == 0
                or step + 1 == total_steps
            ):
                progress.set_postfix(
                    loss=f"{parts['total']:.4f}",
                    avg=f"{running_loss / num_steps:.4f}",
                )

        progress.close()

        result = _mean_dict(logs)
        result.update(diagnostics.compute())
        return result


__all__ = ["V5CANTrainer"]

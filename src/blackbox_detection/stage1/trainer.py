"""Stage 1 training loop.

One loop serves both branches. The video/forensic difference is confined to the
:class:`~.dataset.Stage1BatchAdapter` that flattens a batch into units, so the
loop itself contains no ``if video ... else forensic ...``: it always sees
``(N, ...)`` inputs and ``(N,)`` targets.

Provided here: AMP, gradient accumulation, AdamW with parameter-group weight
decay, warmup plus cosine schedule, lexicographic best-checkpoint selection,
optional early stopping, W&B logging and a training-history CSV.

Validation always runs through :class:`~.evaluator.Stage1Evaluator`. Checkpoint
selection uses, in order:

1. validation Macro-F1 at the searched threshold (higher is better),
2. validation Macro-F1 at the fixed 0.5 threshold (higher is better),
3. video-level validation log loss (lower is better),
4. exact ties keep the earlier checkpoint.

The first criterion remains the official Stage 1 metric. The second criterion
favours checkpoints that already perform well at the natural submission
threshold, and log loss is used only as a tertiary probability-quality
tie-breaker.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.checkpoint import save_latest_and_best
from ..utils.logging import create_progress_bar, log_metrics, setup_logger
from ..utils.seed import DEFAULT_SEED
from .dataset import Stage1BatchAdapter
from .evaluator import (
    AggregationConfig,
    EvaluationResult,
    Stage1Evaluator,
    save_predictions,
)

HISTORY_FILENAME = "history.csv"
CONFIG_FILENAME = "train_config.json"
PREDICTIONS_FILENAME = "val_predictions.csv"

# Macro-F1 values are discrete for the small Stage 1 validation folds, so exact
# ties are common. Use a very small tolerance only to avoid floating-point noise.
SELECTION_EPS = 1e-12


def video_log_loss(
    predictions: pd.DataFrame,
    *,
    eps: float = 1e-7,
) -> float:
    """Binary log loss on aggregated video-level RERECORDED probabilities.

    This is intentionally a *video-level validation* quantity derived from the
    same predictions used by the official metric. It is not the training loss
    and therefore can be used as a tertiary checkpoint-selection tie-breaker
    without favouring later epochs simply because they fit the training set
    better.

    Args:
        predictions: EvaluationResult.predictions-like table containing
            ``label`` and ``prob_rerecorded``.
        eps: Probability clipping floor for numerical stability.

    Returns:
        Mean binary negative log likelihood. Lower is better.
    """
    required = {"label", "prob_rerecorded"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            f"Predictions are missing columns required for log loss: {missing}"
        )
    if len(predictions) == 0:
        raise ValueError("Cannot compute validation log loss on an empty table.")

    labels = predictions["label"].astype(str)
    unknown = sorted(set(labels) - {"ORIGINAL", "RERECORDED"})
    if unknown:
        raise ValueError(f"Unexpected Stage 1 labels for log loss: {unknown}")

    targets = labels.eq("RERECORDED").to_numpy(dtype=np.float64)
    probabilities = predictions["prob_rerecorded"].to_numpy(dtype=np.float64)

    if not np.isfinite(probabilities).all():
        raise ValueError("Validation probabilities contain NaN or infinity.")

    probabilities = np.clip(probabilities, float(eps), 1.0 - float(eps))
    loss = -np.mean(
        targets * np.log(probabilities)
        + (1.0 - targets) * np.log(1.0 - probabilities)
    )
    return float(loss)


def _is_better_checkpoint(
    *,
    macro_f1: float,
    macro_f1_at_05: float,
    val_log_loss: float,
    best_macro_f1: float,
    best_macro_f1_at_05: float,
    best_val_log_loss: float,
    eps: float = SELECTION_EPS,
) -> tuple[bool, str]:
    """Lexicographically compare validation checkpoint metrics.

    Priority:
        1. searched-threshold Macro-F1, higher is better;
        2. Macro-F1 at threshold 0.5, higher is better;
        3. video-level log loss, lower is better;
        4. an exact tie is *not* an improvement, keeping the earlier epoch.

    Returns:
        ``(improved, reason)`` where ``reason`` is useful for logs.
    """
    macro_f1 = float(macro_f1)
    macro_f1_at_05 = float(macro_f1_at_05)
    val_log_loss = float(val_log_loss)
    best_macro_f1 = float(best_macro_f1)
    best_macro_f1_at_05 = float(best_macro_f1_at_05)
    best_val_log_loss = float(best_val_log_loss)

    if macro_f1 > best_macro_f1 + eps:
        return True, "macro_f1"
    if macro_f1 < best_macro_f1 - eps:
        return False, "macro_f1"

    if macro_f1_at_05 > best_macro_f1_at_05 + eps:
        return True, "macro_f1_at_0.5"
    if macro_f1_at_05 < best_macro_f1_at_05 - eps:
        return False, "macro_f1_at_0.5"

    if val_log_loss < best_val_log_loss - eps:
        return True, "val_log_loss"

    return False, "exact_tie_or_worse_log_loss"


@dataclass
class TrainConfig:
    """Stage 1 training configuration.

    Attributes:
        epochs: Number of epochs.
        learning_rate: Peak learning rate of the parameter groups.
        head_learning_rate: Optional separate peak rate for head parameters,
            useful when the backbone is only partially unfrozen.
        weight_decay: AdamW weight decay; never applied to biases or norms.
        warmup_ratio: Fraction of total steps spent warming up linearly.
        min_learning_rate_ratio: Floor of the cosine schedule, relative to the
            peak rate.
        grad_accum_steps: Optimiser steps are taken every N micro-batches.
        max_grad_norm: Gradient-norm clipping value; ``0`` disables it.
        amp: Enable autocast plus a gradient scaler on CUDA.
        label_smoothing: Cross-entropy label smoothing.
        class_weights: Optional per-class loss weights, ordered like
            :data:`~blackbox_detection.utils.metrics.STAGE1_LABELS`.
        early_stopping_patience: Stop after N epochs without checkpoint
            improvement under the lexicographic validation selection rule;
            ``0`` disables early stopping.
        eval_every: Validate every N epochs.
        log_interval: Log the running training loss every N micro-batches.
        seed: Seed recorded in the run configuration.
        output_dir: Directory receiving checkpoints, history and predictions.
        model_name: Model key, stored in the checkpoint.
        wandb_enabled: Log metrics to an already initialised W&B run.
        save_predictions: Write ``val_predictions.csv`` for the best epoch.
    """

    epochs: int = 10
    learning_rate: float = 1e-4
    head_learning_rate: float | None = None
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    min_learning_rate_ratio: float = 0.01
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    label_smoothing: float = 0.0
    class_weights: tuple[float, ...] | None = None
    early_stopping_patience: int = 0
    eval_every: int = 1
    log_interval: int = 20
    seed: int = DEFAULT_SEED
    output_dir: str | Path = "outputs/stage1/run"
    model_name: str = "stage1_model"
    wandb_enabled: bool = False
    save_predictions: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {self.grad_accum_steps}."
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}."
            )
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {self.eval_every}.")


@dataclass
class TrainingOutcome:
    """Summary returned by :meth:`Stage1Trainer.fit`."""

    best_macro_f1: float
    best_threshold: float
    best_epoch: int
    history: pd.DataFrame
    output_dir: Path
    best_checkpoint: Path | None
    best_result: EvaluationResult | None = field(default=None, repr=False)


def build_parameter_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    head_learning_rate: float | None = None,
    head_parameters: Iterable[nn.Parameter] | None = None,
) -> list[dict[str, Any]]:
    """Split trainable parameters into decay / no-decay (and head) groups.

    Biases and one-dimensional parameters (norm weights, learned scalars) are
    excluded from weight decay, which is standard for transformer fine-tuning
    and matters here because the head is tiny compared to the backbone.
    """
    head_ids = {id(parameter) for parameter in (head_parameters or [])}
    groups: dict[str, dict[str, Any]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = id(parameter) in head_ids
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        key = (
            f"{'head' if is_head else 'backbone'}_"
            f"{'no_decay' if no_decay else 'decay'}"
        )
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": float(
                    head_learning_rate
                    if (is_head and head_learning_rate is not None)
                    else learning_rate
                ),
                "weight_decay": 0.0 if no_decay else float(weight_decay),
                "name": key,
            }
        groups[key]["params"].append(parameter)

    if not groups:
        raise ValueError(
            "No trainable parameters found. Check the fine-tuning mode: "
            "'head_only' still requires a trainable head."
        )

    return list(groups.values())


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_ratio`` of the peak rate."""
    total = max(int(total_steps), 1)
    warmup = max(int(round(total * warmup_ratio)), 0)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup

        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Stage1Trainer:
    """Train one Stage 1 model and keep the best-scoring checkpoint.

    Args:
        model: Any :class:`~.models.base.Stage1Model`.
        config: Training configuration.
        adapter: Batch adapter matching the dataloaders.
        device: Training device; resolved automatically when ``None``.
        aggregation: Unit-to-video aggregation used during validation.
        model_config: Model configuration stored in the checkpoint so a run can
            be rebuilt from the checkpoint alone.
        logger: Optional pre-configured logger.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        *,
        adapter: Stage1BatchAdapter,
        device: torch.device | str | None = None,
        aggregation: AggregationConfig | None = None,
        model_config: Mapping[str, Any] | None = None,
        logger: Any | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.aggregation = aggregation or AggregationConfig()
        self.model_config = dict(model_config or {})
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger(
            f"blackbox_detection.stage1.{config.model_name}",
            log_file=self.output_dir / "train.log",
        )
        self.amp = bool(config.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)

        weights = (
            torch.tensor(
                config.class_weights,
                dtype=torch.float32,
                device=self.device,
            )
            if config.class_weights is not None
            else None
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=config.label_smoothing,
        )
        self.evaluator = Stage1Evaluator(
            self.model,
            adapter,
            device=self.device,
            amp=self.amp,
            aggregation=self.aggregation,
        )
        self.history: list[dict[str, Any]] = []

    # Training ---------------------------------------------------------------

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        accumulation = self.config.grad_accum_steps
        running_loss = 0.0
        seen_units = 0
        num_batches = len(loader)
        progress = create_progress_bar(
            loader,
            desc=f"Train {epoch}/{self.config.epochs}",
            leave=False,
        )
        optimizer.zero_grad(set_to_none=True)
        micro_batches_since_step = 0
        skipped_invalid_batches = 0

        for step, batch in enumerate(progress):
            adapted = self.adapter.unpack(batch, self.device)
            valid_mask = adapted.valid_mask
            has_valid_units = bool(valid_mask.any().item())

            if has_valid_units:
                inputs = adapted.inputs[valid_mask]
                targets = adapted.targets[valid_mask]

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)

                self.scaler.scale(loss / accumulation).backward()
                micro_batches_since_step += 1
                units = int(targets.numel())
                running_loss += float(loss.detach()) * units
                seen_units += units
            else:
                skipped_invalid_batches += 1

            is_last = step + 1 == num_batches
            should_step = micro_batches_since_step > 0 and (
                micro_batches_since_step >= accumulation or is_last
            )

            if should_step:
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter in self.model.parameters()
                            if parameter.requires_grad
                        ),
                        self.config.max_grad_norm,
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                micro_batches_since_step = 0

                # Models with weight constraints (Bayar) re-project here.
                hook = getattr(self.model, "on_after_optimizer_step", None)
                if callable(hook):
                    hook()

            if (
                self.config.log_interval
                and (step + 1) % self.config.log_interval == 0
            ):
                progress.set_postfix(
                    loss=f"{running_loss / max(seen_units, 1):.4f}"
                )

        progress.close()

        if seen_units == 0:
            raise RuntimeError(
                "Training epoch contained no valid decoded units. "
                "Inspect broken videos / decoder errors before continuing."
            )

        if skipped_invalid_batches:
            self.logger.warning(
                "Skipped %d batch(es) containing no valid decoded units in epoch %d.",
                skipped_invalid_batches,
                epoch,
            )

        return {
            "train_loss": running_loss / seen_units,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        subsets: Mapping[str, Sequence[str]] | None = None,
        resume_from: str | Path | None = None,
    ) -> TrainingOutcome:
        """Train, validate every ``eval_every`` epochs and keep the best model.

        Args:
            train_loader: Training loader.
            val_loader: Validation loader over held-out **videos**.
            subsets: Named ``video_id`` subsets for controlled diagnostics.
            resume_from: Optional checkpoint to resume from.

        Returns:
            A :class:`TrainingOutcome`.
        """
        config = self.config

        optimizer = torch.optim.AdamW(
            build_parameter_groups(
                self.model,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                head_learning_rate=config.head_learning_rate,
                head_parameters=(
                    self.model.head_parameters()
                    if hasattr(self.model, "head_parameters")
                    else None
                ),
            )
        )

        steps_per_epoch = max(
            math.ceil(len(train_loader) / config.grad_accum_steps),
            1,
        )
        scheduler = build_scheduler(
            optimizer,
            total_steps=steps_per_epoch * config.epochs,
            warmup_ratio=config.warmup_ratio,
            min_ratio=config.min_learning_rate_ratio,
        )

        start_epoch = 1

        # Lexicographic best-checkpoint state.
        best_score = -math.inf
        best_score_at_05 = -math.inf
        best_val_log_loss = math.inf
        best_threshold = 0.5
        best_epoch = 0

        best_checkpoint: Path | None = None
        best_result: EvaluationResult | None = None
        epochs_without_improvement = 0

        if resume_from is not None:
            from ..utils.checkpoint import load_checkpoint

            metadata = load_checkpoint(
                resume_from,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.scaler,
                map_location="cpu",
            )

            start_epoch = int(metadata.get("epoch") or 0) + 1
            best_score = float(metadata.get("best_score") or -math.inf)

            extra = dict(metadata.get("extra") or {})
            best_threshold = float(extra.get("best_threshold", 0.5))
            best_epoch = int(
                extra.get("best_epoch")
                or metadata.get("epoch")
                or 0
            )

            # New checkpoints store these explicitly. Old checkpoints do not,
            # so recover them from the row corresponding to the stored best
            # epoch when possible. We deliberately do NOT retroactively choose
            # a different old epoch because best.pt would not contain its
            # weights.
            stored_best_at_05 = extra.get("best_macro_f1_at_0.5")
            stored_best_log_loss = extra.get("best_val_log_loss")

            if stored_best_at_05 is not None:
                best_score_at_05 = float(stored_best_at_05)
            if stored_best_log_loss is not None:
                best_val_log_loss = float(stored_best_log_loss)

            resume_dir = Path(resume_from).parent
            candidate_best = resume_dir / "best.pt"
            if candidate_best.is_file():
                best_checkpoint = candidate_best

            history_path = resume_dir / HISTORY_FILENAME
            if history_path.is_file():
                previous_history = pd.read_csv(history_path)

                if len(previous_history):
                    self.history = previous_history.to_dict(orient="records")

                    # Backward-compatible recovery for checkpoints created
                    # before the tie-break metrics were saved in checkpoint
                    # metadata. Use the stored best epoch so checkpoint weights
                    # and recovered metrics stay aligned.
                    if "epoch" in previous_history.columns and best_epoch > 0:
                        best_rows = previous_history.loc[
                            previous_history["epoch"].eq(best_epoch)
                        ]
                        if len(best_rows):
                            best_row = best_rows.iloc[-1]

                            if (
                                not math.isfinite(best_score_at_05)
                                and "val_macro_f1_at_0.5" in best_row.index
                                and pd.notna(best_row["val_macro_f1_at_0.5"])
                            ):
                                best_score_at_05 = float(
                                    best_row["val_macro_f1_at_0.5"]
                                )

                            if (
                                not math.isfinite(best_val_log_loss)
                                and "val_log_loss" in best_row.index
                                and pd.notna(best_row["val_log_loss"])
                            ):
                                best_val_log_loss = float(
                                    best_row["val_log_loss"]
                                )

                            if (
                                "val_threshold" in best_row.index
                                and pd.notna(best_row["val_threshold"])
                            ):
                                best_threshold = float(
                                    best_row["val_threshold"]
                                )

            epochs_without_improvement = max(
                0,
                start_epoch - 1 - best_epoch,
            )

            self.logger.info(
                (
                    "Resumed from %s at epoch %d "
                    "(best epoch %d, F1 %.4f, F1@0.5 %.4f, "
                    "logloss %.6f, threshold %.3f)."
                ),
                resume_from,
                start_epoch,
                best_epoch,
                best_score,
                best_score_at_05,
                best_val_log_loss,
                best_threshold,
            )

        (self.output_dir / CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "train_config": {
                        key: (
                            str(value)
                            if isinstance(value, Path)
                            else value
                        )
                        for key, value in asdict(config).items()
                    },
                    "model_config": self.model_config,
                    "aggregation": asdict(self.aggregation),
                    "device": str(self.device),
                    "checkpoint_selection": {
                        "priority": [
                            "val_macro_f1",
                            "val_macro_f1_at_0.5",
                            "val_log_loss",
                            "earlier_epoch",
                        ],
                        "directions": [
                            "max",
                            "max",
                            "min",
                            "min",
                        ],
                        "eps": SELECTION_EPS,
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            metrics = self._train_one_epoch(
                train_loader,
                optimizer,
                scheduler,
                epoch,
            )

            record: dict[str, Any] = {
                "epoch": epoch,
                **metrics,
            }

            if epoch % config.eval_every == 0 or epoch == config.epochs:
                result = self.evaluator.evaluate(
                    val_loader,
                    subsets=subsets,
                )
                assert isinstance(result, EvaluationResult)

                val_loss = video_log_loss(result.predictions)

                record.update(
                    {
                        "val_macro_f1": result.macro_f1,
                        "val_macro_f1_at_0.5": result.macro_f1_at_default,
                        "val_log_loss": val_loss,
                        "val_threshold": result.threshold,
                        **{
                            f"val_f1_{name}": value
                            for name, value in result.per_class_f1.items()
                        },
                    }
                )

                for name, payload in result.subset_scores.items():
                    record[f"val_macro_f1_{name}"] = payload.get(
                        "macro_f1"
                    )

                improved, improvement_reason = _is_better_checkpoint(
                    macro_f1=float(result.macro_f1),
                    macro_f1_at_05=float(result.macro_f1_at_default),
                    val_log_loss=float(val_loss),
                    best_macro_f1=float(best_score),
                    best_macro_f1_at_05=float(best_score_at_05),
                    best_val_log_loss=float(best_val_log_loss),
                )

                if improved:
                    best_score = float(result.macro_f1)
                    best_score_at_05 = float(
                        result.macro_f1_at_default
                    )
                    best_val_log_loss = float(val_loss)
                    best_threshold = float(result.threshold)
                    best_epoch = epoch
                    best_result = result
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += config.eval_every

                paths = save_latest_and_best(
                    self.output_dir,
                    model=self.model,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=self.scaler,
                    best_score=best_score,
                    is_best=improved,
                    config={
                        "train_config": asdict(config),
                        "model_config": self.model_config,
                    },
                    extra={
                        "model_name": config.model_name,
                        "input_kind": getattr(
                            self.model,
                            "input_kind",
                            None,
                        ),
                        # Current validation metrics.
                        "val_macro_f1": float(result.macro_f1),
                        "val_macro_f1_at_0.5": float(
                            result.macro_f1_at_default
                        ),
                        "val_log_loss": float(val_loss),
                        "val_threshold": float(result.threshold),
                        # Best checkpoint metrics under the lexicographic rule.
                        "best_macro_f1": float(best_score),
                        "best_macro_f1_at_0.5": float(
                            best_score_at_05
                        ),
                        "best_val_log_loss": float(
                            best_val_log_loss
                        ),
                        "best_threshold": float(best_threshold),
                        "best_epoch": int(best_epoch),
                        "best_selection_reason": (
                            improvement_reason
                            if improved
                            else "unchanged"
                        ),
                        "checkpoint_selection_priority": [
                            "val_macro_f1",
                            "val_macro_f1_at_0.5",
                            "val_log_loss",
                            "earlier_epoch",
                        ],
                        "aggregation": asdict(self.aggregation),
                        "preprocessing": dict(
                            self.model.preprocessing()
                            if hasattr(self.model, "preprocessing")
                            else {}
                        ),
                    },
                )

                if improved:
                    best_checkpoint = paths["best"]

                    if config.save_predictions:
                        save_predictions(
                            result.predictions,
                            self.output_dir / PREDICTIONS_FILENAME,
                        )

                record["checkpoint_improved"] = bool(improved)
                record["checkpoint_improvement_reason"] = (
                    improvement_reason
                    if improved
                    else ""
                )

                self.logger.info(
                    (
                        "Epoch %d | train loss %.4f | "
                        "val Macro-F1 %.4f @ thr %.3f | "
                        "F1@0.5 %.4f | logloss %.6f%s"
                    ),
                    epoch,
                    metrics["train_loss"],
                    result.macro_f1,
                    result.threshold,
                    result.macro_f1_at_default,
                    val_loss,
                    (
                        f"  <- best ({improvement_reason})"
                        if improved
                        else ""
                    ),
                )

                if config.wandb_enabled:
                    log_metrics(
                        {
                            **result.as_metrics(),
                            "val/log_loss": float(val_loss),
                            "train/loss": metrics["train_loss"],
                        },
                        step=epoch,
                    )
            else:
                self.logger.info(
                    "Epoch %d | loss %.4f (no validation this epoch)",
                    epoch,
                    metrics["train_loss"],
                )

                if config.wandb_enabled:
                    log_metrics(
                        {"train/loss": metrics["train_loss"]},
                        step=epoch,
                    )

            record["epoch_seconds"] = time.perf_counter() - started
            self.history.append(record)

            pd.DataFrame(self.history).to_csv(
                self.output_dir / HISTORY_FILENAME,
                index=False,
                encoding="utf-8",
            )

            if (
                config.early_stopping_patience
                and epochs_without_improvement
                >= config.early_stopping_patience
            ):
                self.logger.info(
                    (
                        "Early stopping after %d epoch(s) without "
                        "lexicographic validation improvement."
                    ),
                    epochs_without_improvement,
                )
                break

        if best_epoch == 0:
            raise RuntimeError(
                "Training finished without a single validation pass; check "
                "eval_every and the number of epochs."
            )

        self.logger.info(
            (
                "Best epoch %d | Macro-F1 %.4f | F1@0.5 %.4f | "
                "logloss %.6f | threshold %.3f."
            ),
            best_epoch,
            best_score,
            best_score_at_05,
            best_val_log_loss,
            best_threshold,
        )

        return TrainingOutcome(
            best_macro_f1=float(best_score),
            best_threshold=float(best_threshold),
            best_epoch=int(best_epoch),
            history=pd.DataFrame(self.history),
            output_dir=self.output_dir,
            best_checkpoint=best_checkpoint,
            best_result=best_result,
        )


__all__ = [
    "HISTORY_FILENAME",
    "CONFIG_FILENAME",
    "PREDICTIONS_FILENAME",
    "SELECTION_EPS",
    "TrainConfig",
    "TrainingOutcome",
    "Stage1Trainer",
    "video_log_loss",
    "build_parameter_groups",
    "build_scheduler",
]

"""Video-level evaluation for Stage 1.

Everything is scored at **video** level, because that is the competition's unit
of prediction. The forensic branch predicts per patch, so its scores are
aggregated in two stages::

    patch probability -> frame aggregation -> video aggregation -> Macro-F1

The video branch predicts per clip, which the same machinery handles as a
single aggregation stage.
The official metric is never reimplemented here:
:func:`blackbox_detection.utils.metrics.stage1_score` is called directly, and
that module is not modified.

Threshold 0.5 is only a starting point. :meth:`Stage1Evaluator.evaluate`
searches the threshold that maximises validation Macro-F1 and stores it
alongside the predictions, so later threshold tuning, model comparison,
prediction-correlation analysis and late fusion all reuse the same numbers.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..utils.metrics import STAGE1_LABELS, stage1_score
from .dataset import Stage1BatchAdapter, STAGE1_INDEX_TO_LABEL, STAGE1_LABEL_TO_INDEX
PREDICTION_COLUMNS: tuple[str, ...] = (
    "video_id",
    "label",
    "prob_original",
    "prob_rerecorded",
    "prediction",
    "dataset",
)

UNIT_COLUMNS: tuple[str, ...] = (
    "video_id",
    "label",
    "dataset",
    "frame_index",
    "patch_index",
    "prob_rerecorded",
    "valid",
)

RERECORDED_INDEX = STAGE1_LABEL_TO_INDEX["RERECORDED"]
ORIGINAL_LABEL, RERECORDED_LABEL = STAGE1_LABELS


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))

def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def _maximum(values: np.ndarray) -> float:
    return float(np.max(values))


def _minimum(values: np.ndarray) -> float:
    return float(np.min(values))

def _trimmed_mean(values: np.ndarray, proportion: float = 0.2) -> float:
    if len(values) < 3:
        return float(np.mean(values))
    ordered = np.sort(values)
    cut = int(np.floor(len(ordered) * proportion))
    trimmed = ordered[cut : len(ordered) - cut] if cut else ordered
    return float(np.mean(trimmed))

def _logit_mean(values: np.ndarray, eps: float = 1e-6) -> float:
    clipped = np.clip(values, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-np.mean(logits))))


AGGREGATORS: Mapping[str, Callable[[np.ndarray], float]] = {
    "mean": _mean,
    "median": _median,
    "max": _maximum,
    "min": _minimum,
    "trimmed_mean": _trimmed_mean,
    "logit_mean": _logit_mean,
}

@dataclass(frozen=True)
class AggregationConfig:
    """How unit probabilities become one video probability.

    Attributes:
        frame_method: Patch-to-frame aggregator; ignored when the dataset does
            not report frame indices.
        video_method: Frame-to-video (or clip-to-video) aggregator.
    Notes:
        Both default to ``"mean"``, the initial Stage 1 choice. Registering a
        learned aggregator (attention pooling) later only requires adding it to
        :data:`AGGREGATORS`; no calling code changes.
    """

    frame_method: str = "mean"
    video_method: str = "mean"
    def __post_init__(self) -> None:
        for method in (self.frame_method, self.video_method):
            if method not in AGGREGATORS:
                raise ValueError(
                    f"Unknown aggregation method {method!r}. "
                    f"Available: {sorted(AGGREGATORS)}"
                )


@dataclass
class EvaluationResult:
    """Outcome of one Stage 1 evaluation pass."""
    macro_f1: float
    threshold: float
    macro_f1_at_default: float
    per_class_f1: dict[str, float]
    num_videos: int
    num_invalid_videos: int
    predictions: pd.DataFrame
    dataset_scores: dict[str, float] = field(default_factory=dict)
    subset_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    def as_metrics(self, prefix: str = "val") -> dict[str, float]:
        """Flatten the scalar metrics for logging."""
        metrics = {
            f"{prefix}/macro_f1": float(self.macro_f1),
            f"{prefix}/macro_f1@0.5": float(self.macro_f1_at_default),
            f"{prefix}/threshold": float(self.threshold),
            f"{prefix}/num_invalid_videos": float(self.num_invalid_videos),
        }
        metrics.update(
            {f"{prefix}/f1_{name}": float(value) for name, value in self.per_class_f1.items()}
        )
        metrics.update(
            {f"{prefix}/macro_f1_{name}": float(value) for name, value in self.dataset_scores.items()}
        )
        for name, payload in self.subset_scores.items():
            score = payload.get("macro_f1")
            if score is not None:
                metrics[f"{prefix}/macro_f1_{name}"] = float(score)
        return metrics

def per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = STAGE1_LABELS,
) -> dict[str, float]:
    """Per-class F1 over the official class set.
    Complements :func:`~blackbox_detection.utils.metrics.stage1_score`, which
    returns only the macro average. Kept local so ``utils/metrics.py`` stays
    untouched.
    """
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    if len(true) != len(pred):
        raise ValueError(f"Length mismatch: {len(true)} labels vs {len(pred)} predictions.")
    scores: dict[str, float] = {}
    for label in labels:
        true_is = true == label
        pred_is = pred == label
        tp = int(np.sum(true_is & pred_is))
        fp = int(np.sum(~true_is & pred_is))
        fn = int(np.sum(true_is & ~pred_is))
        denominator = 2 * tp + fp + fn
        scores[str(label)] = 0.0 if denominator == 0 else 2.0 * tp / denominator
    return scores

def probabilities_to_labels(
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> list[str]:
    """Map RERECORDED probabilities to official Stage 1 label strings."""
    values = np.asarray(probabilities, dtype=np.float64)
    return [RERECORDED_LABEL if value >= threshold else ORIGINAL_LABEL for value in values]

def search_best_threshold(
    y_true: Sequence[str],
    probabilities: Sequence[float] | np.ndarray,
    *,
    candidates: Sequence[float] | None = None,
    default: float = 0.5,
) -> tuple[float, float]:
    """Find the threshold maximising Stage 1 Macro-F1.
    Args:
        y_true: Ground-truth Stage 1 labels.
        probabilities: RERECORDED probabilities.
        candidates: Thresholds to try. Defaults to the midpoints between
            consecutive distinct probabilities, plus ``default``, which is the
            smallest set that can realise every achievable decision boundary.
        default: Threshold preferred on ties.
    Returns:
        ``(best_threshold, best_macro_f1)``.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot search a threshold on an empty prediction set.")
    if candidates is None:
        unique = np.unique(values)
        midpoints = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else np.asarray([])
        grid = np.unique(
            np.concatenate([[default], midpoints, unique, [0.0, 1.0 + 1e-9]])
        )
    else:
        grid = np.unique(np.asarray(candidates, dtype=np.float64))
    best_threshold = float(default)
    best_score = stage1_score(y_true, probabilities_to_labels(values, default))
    for threshold in grid:
        score = stage1_score(y_true, probabilities_to_labels(values, float(threshold)))
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, float(best_score)

def aggregate_unit_predictions(
    units: pd.DataFrame,
    *,
    aggregation: AggregationConfig | None = None,
) -> pd.DataFrame:
    """Aggregate unit probabilities into video-level probabilities.

    Args:
        units: Unit-level predictions with at least ``video_id``, ``label``,
            ``dataset`` and ``prob_rerecorded``. When ``frame_index`` is present
            and non-negative, patches are first aggregated per frame.
        aggregation: Aggregation methods.
    Returns:
        One row per video with ``prob_original``, ``prob_rerecorded``,
        ``num_units`` and ``num_frames``. For the video branch, whose units are
        clips rather than patches, ``num_frames`` is 1 because there is only one
        aggregation group; ``num_units`` is the clip count.
    """
    config = aggregation or AggregationConfig()
    required = {"video_id", "label", "dataset", "prob_rerecorded"}
    missing = sorted(required - set(units.columns))
    if missing:
        raise ValueError(f"Unit predictions are missing columns: {missing}")

    all_units = units.copy()
    if "valid" in all_units.columns:
        valid_mask = (
            all_units["valid"].astype(bool)
            & all_units["prob_rerecorded"].notna()
        )
        units = all_units.loc[valid_mask].copy()

        all_video_ids = set(all_units["video_id"].astype(str))
        valid_video_ids = set(units["video_id"].astype(str))
        fully_invalid = sorted(all_video_ids - valid_video_ids)
        if fully_invalid:
            raise ValueError(
                "No valid decoded unit is available for "
                f"{len(fully_invalid)} video(s): {fully_invalid[:5]}. "
                "Fix/remove broken videos instead of scoring zero-filled inputs."
            )

    frame_aggregator = AGGREGATORS[config.frame_method]
    video_aggregator = AGGREGATORS[config.video_method]
    has_frames = (
        "frame_index" in units.columns and bool((units["frame_index"] >= 0).any())
    )
    if has_frames:
        frame_level = (
            units.groupby(["video_id", "frame_index"], sort=True)["prob_rerecorded"]
            .apply(lambda values: frame_aggregator(values.to_numpy(dtype=np.float64)))
            .reset_index(name="prob_rerecorded")
        )
    else:
        frame_level = units[["video_id", "prob_rerecorded"]].copy()
        frame_level["frame_index"] = -1
    video_level = (
        frame_level.groupby("video_id", sort=True)["prob_rerecorded"]
        .apply(lambda values: video_aggregator(values.to_numpy(dtype=np.float64)))
        .reset_index(name="prob_rerecorded")
    )
    aggregations: dict[str, tuple[str, Any]] = {
        "num_units": ("prob_rerecorded", "size"),
        "label": ("label", "first"),
        "dataset": ("dataset", "first"),
    }
    if "valid" in all_units.columns:
        aggregations["num_invalid"] = (
            "valid",
            lambda values: int((~values.astype(bool)).sum()),
        )
    counts = all_units.groupby("video_id", sort=True).agg(**aggregations).reset_index()
    valid_counts = (
        units.groupby("video_id", sort=True)
        .size()
        .reset_index(name="num_valid_units")
    )
    counts = counts.merge(valid_counts, on="video_id", how="left")
    counts["num_valid_units"] = counts["num_valid_units"].fillna(0).astype("int64")
    frame_counts = (
        frame_level.groupby("video_id", sort=True)["frame_index"]
        .nunique()
        .reset_index(name="num_frames")
    )
    merged = video_level.merge(counts, on="video_id").merge(frame_counts, on="video_id")
    merged["prob_original"] = 1.0 - merged["prob_rerecorded"]
    return merged.sort_values("video_id", kind="mergesort").reset_index(drop=True)


def finalize_predictions(video_level: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add the ``prediction`` column and order the standard prediction columns.
    Missing complementary columns are derived rather than required, so this
    also works on a reduced table reloaded from disk.
    """
    frame = video_level.copy()
    if "prob_rerecorded" not in frame.columns:
        raise ValueError("Predictions must contain 'prob_rerecorded'.")
    if "prob_original" not in frame.columns:
        frame["prob_original"] = 1.0 - frame["prob_rerecorded"]
    if "dataset" not in frame.columns:
        frame["dataset"] = ""
    frame["prediction"] = probabilities_to_labels(frame["prob_rerecorded"], threshold)
    frame["threshold"] = float(threshold)

    ordered = [*PREDICTION_COLUMNS, "threshold"]
    extras = [column for column in frame.columns if column not in ordered]
    return frame[[*ordered, *extras]]

def evaluate_predictions(
    video_level: pd.DataFrame,
    *,
    threshold: float | None = None,
    search_threshold: bool = True,
    default_threshold: float = 0.5,
    subsets: Mapping[str, Sequence[str]] | None = None,
) -> EvaluationResult:
    """Score video-level predictions with the official Stage 1 metric.

    Works both on a freshly predicted frame and on a ``val_predictions.csv``
    reloaded from disk, which is what notebook 04 uses to compare and fuse
    models without retraining.
    Args:
        video_level: Video-level predictions with ``label`` and
            ``prob_rerecorded``.
        threshold: Fixed threshold. When ``None`` and ``search_threshold`` is
            true, the best validation threshold is searched.
        search_threshold: Enable threshold search.
        default_threshold: Threshold used for the reference score and as the
            tie-breaker.
        subsets: Named subsets of ``video_id`` values (VAL-A / VAL-B now,
            VAL-DLC / VAL-CCD later) to score separately.
    Returns:
        An :class:`EvaluationResult`.
    """
    if "label" not in video_level.columns or "prob_rerecorded" not in video_level.columns:
        raise ValueError("Predictions must contain 'label' and 'prob_rerecorded'.")
    if len(video_level) == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")

    labels = video_level["label"].astype(str).tolist()
    probabilities = video_level["prob_rerecorded"].to_numpy(dtype=np.float64)
    score_at_default = stage1_score(
        labels, probabilities_to_labels(probabilities, default_threshold)
    )
    if threshold is not None:
        chosen = float(threshold)
        best_score = stage1_score(labels, probabilities_to_labels(probabilities, chosen))
    elif search_threshold:
        chosen, best_score = search_best_threshold(
            labels, probabilities, default=default_threshold
        )
    else:
        chosen, best_score = float(default_threshold), float(score_at_default)
    predictions = finalize_predictions(video_level, chosen)
    class_scores = per_class_f1(labels, predictions["prediction"].tolist())
    dataset_scores: dict[str, float] = {}
    if "dataset" in predictions.columns:
        for dataset_name, group in predictions.groupby("dataset"):
            if group["label"].nunique() < len(STAGE1_LABELS):
                # A single-class subset cannot produce a meaningful Macro-F1.
                continue
            dataset_scores[str(dataset_name)] = float(
                stage1_score(group["label"].tolist(), group["prediction"].tolist())
            )
    subset_scores: dict[str, dict[str, Any]] = {}
    for name, video_ids in (subsets or {}).items():
        subset = predictions[predictions["video_id"].isin(list(video_ids))]
        if len(subset) == 0 or subset["label"].nunique() < len(STAGE1_LABELS):
            subset_scores[name] = {
                "macro_f1": None,
                "num_videos": int(len(subset)),
                "reason": "subset does not contain both Stage 1 classes",
            }
            continue
        subset_scores[name] = {
            "macro_f1": float(
                stage1_score(subset["label"].tolist(), subset["prediction"].tolist())
            ),
            "num_videos": int(len(subset)),
            "per_class_f1": per_class_f1(
                subset["label"].tolist(), subset["prediction"].tolist()
            ),
        }
    num_invalid = (
        int(predictions["num_invalid"].gt(0).sum())
        if "num_invalid" in predictions.columns
        else 0
    )
    return EvaluationResult(
        macro_f1=float(best_score),
        threshold=float(chosen),
        macro_f1_at_default=float(score_at_default),
        per_class_f1=class_scores,
        num_videos=int(len(predictions)),
        num_invalid_videos=num_invalid,
        predictions=predictions,
        dataset_scores=dataset_scores,
        subset_scores=subset_scores,
    )


class Stage1Evaluator:
    """Run a model over a loader and score it at video level.
    Args:
        model: Any :class:`~.models.base.Stage1Model`.
        adapter: Batch adapter matching the loader's dataset.
        device: Device to run on.
        amp: Use autocast on CUDA.
        aggregation: Unit-to-video aggregation configuration.
    """
    def __init__(
        self,
        model: nn.Module,
        adapter: Stage1BatchAdapter,
        *,
        device: torch.device | str = "cpu",
        amp: bool = True,
        aggregation: AggregationConfig | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.device = torch.device(device)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.aggregation = aggregation or AggregationConfig()
    @torch.inference_mode()
    def predict_units(self, loader: Any) -> pd.DataFrame:
        """Return unit-level (clip or patch) RERECORDED probabilities."""
        self.model.eval()
        records: list[dict[str, Any]] = []
        for batch in loader:
            adapted = self.adapter.unpack(batch, self.device)
            valid_mask = adapted.valid_mask
            valid_mask_cpu = valid_mask.detach().cpu().numpy().astype(bool)
            probabilities = np.full(len(valid_mask_cpu), np.nan, dtype=np.float64)

            if bool(valid_mask.any().item()):
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=self.amp
                ):
                    logits = self.model(adapted.inputs[valid_mask])
                valid_probabilities = (
                    torch.softmax(logits.float(), dim=1)[:, RERECORDED_INDEX]
                    .cpu()
                    .numpy()
                )
                probabilities[valid_mask_cpu] = valid_probabilities

            video_ids = batch["video_id"]
            datasets = batch["dataset"]
            label_names = batch["label_name"]
            frame_indices = adapted.meta.get("frame_indices")
            patch_indices = adapted.meta.get("patch_indices")
            video_index = adapted.video_index.numpy()
            for unit, video_slot in enumerate(video_index):
                records.append(
                    {
                        "video_id": video_ids[int(video_slot)],
                        "label": label_names[int(video_slot)],
                        "dataset": datasets[int(video_slot)],
                        "frame_index": (
                            int(frame_indices[unit]) if frame_indices is not None else -1
                        ),
                        "patch_index": (
                            int(patch_indices[unit])
                            if patch_indices is not None
                            else int(unit % adapted.num_units)
                        ),
                        "prob_rerecorded": float(probabilities[unit]),
                        "valid": bool(valid_mask_cpu[unit]),
                    }
                )
        if not records:
            raise ValueError("The evaluation loader produced no batches.")
        return pd.DataFrame.from_records(records, columns=list(UNIT_COLUMNS))

    def predict_videos(self, loader: Any) -> pd.DataFrame:
        """Return video-level probabilities."""
        return aggregate_unit_predictions(
            self.predict_units(loader), aggregation=self.aggregation
        )
    def evaluate(
        self,
        loader: Any,
        *,
        threshold: float | None = None,
        search_threshold: bool = True,
        subsets: Mapping[str, Sequence[str]] | None = None,
        return_units: bool = False,
    ) -> EvaluationResult | tuple[EvaluationResult, pd.DataFrame]:
        """Predict, aggregate and score in one call.
        Args:
            loader: DataLoader over a Stage 1 dataset.
            threshold: Fixed decision threshold, or ``None`` to search.
            search_threshold: Enable the threshold search.
            subsets: Named ``video_id`` subsets for diagnostics.
            return_units: Also return the unit-level predictions.
        Returns:
            The result, plus the unit-level frame when ``return_units`` is set.
        """
        units = self.predict_units(loader)
        video_level = aggregate_unit_predictions(units, aggregation=self.aggregation)
        result = evaluate_predictions(
            video_level,
            threshold=threshold,
            search_threshold=search_threshold,
            subsets=subsets,
        )
        return (result, units) if return_units else result

def save_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Write video-level predictions to CSV.
    The saved columns always start with ``video_id``, ``label``,
    ``prob_original``, ``prob_rerecorded``, ``prediction`` and ``dataset``, the
    contract relied on by threshold tuning, model comparison, prediction
    correlation and late fusion.
    """
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a ``val_predictions.csv`` back."""
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Predictions not found: {prediction_path}")
    predictions = pd.read_csv(prediction_path, encoding="utf-8")
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {missing}")
    return predictions

__all__ = [
    "PREDICTION_COLUMNS",
    "UNIT_COLUMNS",
    "AGGREGATORS",
    "AggregationConfig",
    "EvaluationResult",
    "Stage1Evaluator",
    "per_class_f1",
    "probabilities_to_labels",
    "search_best_threshold",
    "aggregate_unit_predictions",
    "finalize_predictions",
    "evaluate_predictions",
    "save_predictions",
    "load_predictions",
    "STAGE1_INDEX_TO_LABEL",
]

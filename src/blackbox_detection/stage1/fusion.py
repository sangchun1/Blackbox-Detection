"""Late fusion and model comparison for Stage 1.

The video and forensic branches are trained independently, then combined at
probability level::

    p_final = alpha * p_video + (1 - alpha) * p_forensic

Both ``alpha`` and the decision threshold are searched on validation
Macro-F1. Nothing here retrains a model: every function works from the saved
``val_predictions.csv`` files, which is what makes the comparison and fusion
notebook cheap to re-run.

Prediction correlation between models is reported alongside the scores, because
a slightly weaker but decorrelated model is often the better ensemble partner -
a distinction that matters here, where a high DLC-2021 score may simply mean a
model latched onto a document/display shortcut.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.metrics import stage1_score
from .evaluator import (
    PREDICTION_COLUMNS,
    load_predictions,
    per_class_f1,
    probabilities_to_labels,
    search_best_threshold,
)

PROBABILITY_PREFIX = "prob_"


@dataclass
class FusionResult:
    """Outcome of one fusion search."""

    models: tuple[str, ...]
    weights: tuple[float, ...]
    threshold: float
    macro_f1: float
    per_class_f1: dict[str, float]
    num_videos: int
    baseline_macro_f1: dict[str, float] = field(default_factory=dict)

    @property
    def gain_over_best_single(self) -> float:
        """Macro-F1 improvement over the best individual model."""
        if not self.baseline_macro_f1:
            return float("nan")
        return float(self.macro_f1 - max(self.baseline_macro_f1.values()))

    def as_row(self) -> dict[str, Any]:
        """Flatten into one table row."""
        row: dict[str, Any] = {
            "ensemble": " + ".join(self.models),
            "weights": ", ".join(f"{weight:.2f}" for weight in self.weights),
            "macro_f1": self.macro_f1,
            "threshold": self.threshold,
            "gain_over_best_single": self.gain_over_best_single,
            "num_videos": self.num_videos,
        }
        row.update({f"f1_{name}": value for name, value in self.per_class_f1.items()})
        return row


def load_prediction_tables(
    prediction_paths: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Merge several ``val_predictions.csv`` files into one wide table.

    Args:
        prediction_paths: Mapping from model name to prediction CSV path.

    Returns:
        A table with ``video_id``, ``label``, ``dataset`` and one
        ``prob_<model>`` column per model.

    Raises:
        ValueError: If the files disagree on the validation videos or their
            labels, which would mean they were produced from different splits.
    """
    if not prediction_paths:
        raise ValueError("At least one prediction file is required.")

    merged: pd.DataFrame | None = None
    for name, path in prediction_paths.items():
        predictions = load_predictions(path)
        frame = predictions[["video_id", "label", "dataset", "prob_rerecorded"]].rename(
            columns={"prob_rerecorded": f"{PROBABILITY_PREFIX}{name}"}
        )
        if merged is None:
            merged = frame
            continue

        before = len(merged)
        merged = merged.merge(
            frame,
            on=["video_id", "label", "dataset"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != before:
            raise ValueError(
                f"Prediction file for {name!r} does not cover the same validation "
                f"videos ({before} -> {len(merged)} rows after merging). All models "
                "must be evaluated on the same fixed split."
            )

    assert merged is not None
    return merged.sort_values("video_id", kind="mergesort").reset_index(drop=True)


def model_columns(wide: pd.DataFrame) -> list[str]:
    """Return the model names present in a wide prediction table."""
    return [
        column.removeprefix(PROBABILITY_PREFIX)
        for column in wide.columns
        if column.startswith(PROBABILITY_PREFIX)
    ]


def compare_models(
    wide: pd.DataFrame,
    *,
    search_threshold: bool = True,
) -> pd.DataFrame:
    """Score every model in a wide prediction table.

    Returns:
        One row per model with the Macro-F1 at threshold 0.5, the optimal
        threshold, the Macro-F1 there, and the per-class F1 scores.
    """
    labels = wide["label"].astype(str).tolist()
    rows: list[dict[str, Any]] = []

    for name in model_columns(wide):
        probabilities = wide[f"{PROBABILITY_PREFIX}{name}"].to_numpy(dtype=np.float64)
        score_default = stage1_score(labels, probabilities_to_labels(probabilities, 0.5))
        if search_threshold:
            threshold, score = search_best_threshold(labels, probabilities)
        else:
            threshold, score = 0.5, score_default

        class_scores = per_class_f1(
            labels, probabilities_to_labels(probabilities, threshold)
        )
        rows.append(
            {
                "model": name,
                "macro_f1": float(score),
                "optimal_threshold": float(threshold),
                "macro_f1_at_0.5": float(score_default),
                **{f"f1_{label}": value for label, value in class_scores.items()},
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("macro_f1", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def prediction_correlation(
    wide: pd.DataFrame,
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlation matrix of the models' RERECORDED probabilities.

    Args:
        wide: Wide prediction table.
        method: ``"pearson"``, ``"spearman"`` or ``"kendall"``.

    Returns:
        A square correlation matrix indexed by model name. Low correlation
        between two comparably scoring models indicates useful ensemble
        diversity.
    """
    names = model_columns(wide)
    if len(names) < 2:
        raise ValueError("Prediction correlation needs at least two models.")

    probabilities = wide[[f"{PROBABILITY_PREFIX}{name}" for name in names]]
    correlation = probabilities.corr(method=method)
    correlation.index = names
    correlation.columns = names
    return correlation


def fuse_probabilities(
    wide: pd.DataFrame,
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combine model probabilities with normalised non-negative weights."""
    if not weights:
        raise ValueError("At least one model weight is required.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"Fusion weights must be non-negative, got {dict(weights)}.")

    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Fusion weights must not sum to zero.")

    fused = np.zeros(len(wide), dtype=np.float64)
    for name, weight in weights.items():
        column = f"{PROBABILITY_PREFIX}{name}"
        if column not in wide.columns:
            raise KeyError(f"Prediction table has no column for model {name!r}.")
        fused += (weight / total) * wide[column].to_numpy(dtype=np.float64)
    return fused


def _score_weighted(
    wide: pd.DataFrame,
    models: Sequence[str],
    weights: Sequence[float],
) -> tuple[float, float]:
    fused = fuse_probabilities(wide, dict(zip(models, weights)))
    return search_best_threshold(wide["label"].astype(str).tolist(), fused)


def search_late_fusion(
    wide: pd.DataFrame,
    models: Sequence[str],
    *,
    weight_step: float = 0.05,
) -> FusionResult:
    """Grid-search fusion weights and the decision threshold.

    For two models this is exactly the ``alpha`` search of
    ``p_final = alpha * p_a + (1 - alpha) * p_b``; for more models the search
    runs over the weight simplex with the same step size.

    Args:
        wide: Wide prediction table.
        models: Models to combine.
        weight_step: Grid resolution on the simplex.

    Returns:
        The best :class:`FusionResult`.
    """
    if len(models) < 2:
        raise ValueError("Late fusion needs at least two models.")
    if not 0.0 < weight_step <= 0.5:
        raise ValueError(f"weight_step must be in (0, 0.5], got {weight_step}.")

    labels = wide["label"].astype(str).tolist()
    steps = int(round(1.0 / weight_step))
    best: tuple[float, tuple[float, ...], float] = (-np.inf, (), 0.5)

    for counts in itertools.product(range(steps + 1), repeat=len(models) - 1):
        used = sum(counts)
        if used > steps:
            continue
        weights = tuple(
            [count / steps for count in counts] + [(steps - used) / steps]
        )
        threshold, score = _score_weighted(wide, models, weights)
        if score > best[0] + 1e-12:
            best = (float(score), weights, float(threshold))

    score, weights, threshold = best
    fused = fuse_probabilities(wide, dict(zip(models, weights)))
    baselines = {
        name: float(
            search_best_threshold(
                labels, wide[f"{PROBABILITY_PREFIX}{name}"].to_numpy(dtype=np.float64)
            )[1]
        )
        for name in models
    }

    return FusionResult(
        models=tuple(models),
        weights=weights,
        threshold=threshold,
        macro_f1=score,
        per_class_f1=per_class_f1(labels, probabilities_to_labels(fused, threshold)),
        num_videos=int(len(wide)),
        baseline_macro_f1=baselines,
    )


def search_all_combinations(
    wide: pd.DataFrame,
    *,
    combinations: Sequence[Sequence[str]] | None = None,
    weight_step: float = 0.05,
) -> pd.DataFrame:
    """Search several ensembles and return a comparison table.

    Args:
        wide: Wide prediction table.
        combinations: Model groups to evaluate. Defaults to every pair and the
            full set.
        weight_step: Grid resolution on the simplex.

    Returns:
        One row per ensemble, sorted by Macro-F1.
    """
    names = model_columns(wide)
    if combinations is None:
        groups: list[Sequence[str]] = [list(pair) for pair in itertools.combinations(names, 2)]
        if len(names) > 2:
            groups.append(list(names))
    else:
        groups = [list(group) for group in combinations]

    rows = [search_late_fusion(wide, group, weight_step=weight_step).as_row() for group in groups]
    return (
        pd.DataFrame(rows)
        .sort_values("macro_f1", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def fused_predictions(
    wide: pd.DataFrame,
    result: FusionResult,
) -> pd.DataFrame:
    """Materialise a prediction table for a fusion result.

    The output uses the same columns as a single model's
    ``val_predictions.csv``, so a fused prediction can be fed straight back
    into the comparison and evaluation helpers.
    """
    fused = fuse_probabilities(wide, dict(zip(result.models, result.weights)))
    frame = wide[["video_id", "label", "dataset"]].copy()
    frame["prob_rerecorded"] = fused
    frame["prob_original"] = 1.0 - fused
    frame["prediction"] = probabilities_to_labels(fused, result.threshold)
    frame["threshold"] = float(result.threshold)
    return frame[[*PREDICTION_COLUMNS, "threshold"]]


__all__ = [
    "PROBABILITY_PREFIX",
    "FusionResult",
    "load_prediction_tables",
    "model_columns",
    "compare_models",
    "prediction_correlation",
    "fuse_probabilities",
    "search_late_fusion",
    "search_all_combinations",
    "fused_predictions",
]

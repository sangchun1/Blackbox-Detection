from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .constants import CAN_TARGETS

# DACON Stage 3 defined class sets. Macro-F1 is computed over the full defined
# class set, not only classes present in a particular validation subset.
ACCEL_CLASSES = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_CLASSES = ("LEFT", "STRAIGHT", "RIGHT")
ACCEL_WEIGHT = 0.7
STEER_WEIGHT = 0.3


def _labels(values: Sequence[str] | np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=str).reshape(-1)


def dacon_stage3_metrics(
    y_true_accel: Sequence[str] | np.ndarray | pd.Series,
    y_pred_accel: Sequence[str] | np.ndarray | pd.Series,
    y_true_steer: Sequence[str] | np.ndarray | pd.Series,
    y_pred_steer: Sequence[str] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute the official Stage 3 local metric semantics.

    DACON rules used here:
      1. accel: Macro-F1 over all 4 defined acceleration classes;
      2. steer: Macro-F1 over all 3 defined steering classes;
      3. steering rows whose *ground-truth accel label* is STOPPED are excluded;
      4. Stage 3 combines accel/steer as 0.7 / 0.3.

    Explicit ``labels=...`` is important: DACON confirmed that Macro-F1 averages
    over the complete defined class set even when a class is absent from a
    particular evaluation subset.
    """
    ta = _labels(y_true_accel)
    pa = _labels(y_pred_accel)
    ts = _labels(y_true_steer)
    ps = _labels(y_pred_steer)
    n = len(ta)
    if not (len(pa) == len(ts) == len(ps) == n):
        raise ValueError("Stage 3 metric arrays must have the same length")

    bad_pred_accel = sorted(set(pa) - set(ACCEL_CLASSES))
    bad_pred_steer = sorted(set(ps) - set(STEER_CLASSES))
    if bad_pred_accel or bad_pred_steer:
        raise ValueError(f"invalid predicted labels: accel={bad_pred_accel}, steer={bad_pred_steer}")

    accel_macro_f1 = float(
        f1_score(
            ta,
            pa,
            labels=list(ACCEL_CLASSES),
            average="macro",
            zero_division=0,
        )
    )

    moving_mask = ta != "STOPPED"
    if moving_mask.any():
        steer_macro_f1 = float(
            f1_score(
                ts[moving_mask],
                ps[moving_mask],
                labels=list(STEER_CLASSES),
                average="macro",
                zero_division=0,
            )
        )
    else:
        # The hidden evaluation set contains driving frames; this branch only
        # keeps tiny local/smoke subsets numerically well-defined.
        steer_macro_f1 = 0.0

    stage3_score = ACCEL_WEIGHT * accel_macro_f1 + STEER_WEIGHT * steer_macro_f1
    return {
        "stage3_score": float(stage3_score),
        "accel_macro_f1": accel_macro_f1,
        "steer_macro_f1": steer_macro_f1,
        "steer_eval_frames": int(moving_mask.sum()),
    }


def dacon_stage3_metrics_from_frames(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    """Score DataFrames after strict ID/sample_index alignment."""
    keys = ["ID", "sample_index"]
    required = set(keys + ["accel_label", "steer_label"])
    for name, frame in (("truth", truth), ("prediction", prediction)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} has duplicate ID/sample_index rows")

    merged = truth[keys + ["accel_label", "steer_label"]].merge(
        prediction[keys + ["accel_label", "steer_label"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_true", "_pred"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("truth/prediction ID/sample_index sets do not match exactly")

    return dacon_stage3_metrics(
        merged["accel_label_true"],
        merged["accel_label_pred"],
        merged["steer_label_true"],
        merged["steer_label_pred"],
    )


def denormalize(x: np.ndarray, stat: dict) -> np.ndarray:
    return x * float(stat["std"]) + float(stat["mean"])


def regression_metrics(outputs: dict, target: torch.Tensor, valid: torch.Tensor, stats: dict) -> dict[str, float]:
    """Auxiliary CAN metrics; these are not the DACON leaderboard metric."""
    result: dict[str, float] = {}
    target_np = target.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    for i, name in enumerate(CAN_TARGETS):
        p = outputs[name].detach().float().cpu().numpy()
        y = target_np[..., i]
        m = valid_np[..., i].astype(bool)
        if not m.any():
            continue
        p = denormalize(p[m], stats[name])
        y = denormalize(y[m], stats[name])
        result[f"{name}/mae"] = float(np.mean(np.abs(p - y)))
        result[f"{name}/rmse"] = float(np.sqrt(np.mean(np.square(p - y))))
    return result

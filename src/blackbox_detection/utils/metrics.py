"""Official-style local metrics for the DACON blackbox accident challenge.

This module follows the public competition evaluation rules.

Stage 1
-------
Macro-F1 over ORIGINAL / RERECORDED.

Stage 2
-------
- collision time accuracy : 0.35
- entry time accuracy     : 0.35
- evasion_space Macro-F1  : 0.15
- entry_side Macro-F1     : 0.15

Submitted ORIGINAL FRAME NUMBERS are converted with each video's frame-time
correspondence. A temporal prediction is correct when absolute error is <= 0.3s.
Missing, non-numeric, negative, nonexistent/out-of-range frames and invalid
categorical values are treated as incorrect.

Stage 3
-------
- acceleration Macro-F1 : 0.70
- steering Macro-F1     : 0.30

Steering is evaluated only where the GROUND-TRUTH acceleration label is not
STOPPED.

Overall stage weights are Stage1=0.20, Stage2=0.40, Stage3=0.40.
Macro-F1 always uses the full official class set, even if a class is absent
from a local validation split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np


# Official class sets ---------------------------------------------------------

STAGE1_LABELS: tuple[str, ...] = ("ORIGINAL", "RERECORDED")
STAGE2_EVASION_LABELS: tuple[int, ...] = (0, 1)
STAGE2_ENTRY_SIDE_LABELS: tuple[str, ...] = ("LEFT", "RIGHT")
STAGE3_ACCEL_LABELS: tuple[str, ...] = (
    "ACCELERATING",
    "DECELERATING",
    "CONSTANT",
    "STOPPED",
)
STAGE3_STEER_LABELS: tuple[str, ...] = ("LEFT", "STRAIGHT", "RIGHT")


# Official weights ------------------------------------------------------------

STAGE_WEIGHTS: Mapping[str, float] = {
    "stage1": 0.20,
    "stage2": 0.40,
    "stage3": 0.40,
}
STAGE2_WEIGHTS: Mapping[str, float] = {
    "collision": 0.35,
    "entry": 0.35,
    "evasion_space": 0.15,
    "entry_side": 0.15,
}
STAGE3_WEIGHTS: Mapping[str, float] = {
    "accel": 0.70,
    "steer": 0.30,
}
STAGE2_TIME_TOLERANCE_SECONDS = 0.3

FrameTimeMap: TypeAlias = Mapping[int, float] | Sequence[float] | np.ndarray


# Shared helpers --------------------------------------------------------------


def _as_1d_object_array(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={array.shape}.")
    return array


def _as_1d_float_array(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={array.shape}.")
    return array


def _check_same_length(**arrays: Sequence[Any] | np.ndarray) -> None:
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All inputs must have the same length: {lengths}")


def _encode_labels(
    values: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any],
    name: str,
    validate: bool,
) -> tuple[np.ndarray, int]:
    """Encode official labels as 0..K-1 and unknown predictions as -1."""
    array = _as_1d_object_array(values, name=name)
    encoded = np.full(len(array), -1, dtype=np.int64)

    for class_index, label in enumerate(labels):
        for idx, value in enumerate(array):
            try:
                if bool(value == label):
                    encoded[idx] = class_index
            except Exception:
                pass

    invalid_count = int(np.sum(encoded < 0))
    if validate and invalid_count:
        examples = [
            repr(value)
            for value, code in zip(array.tolist(), encoded.tolist())
            if code < 0
        ][:5]
        raise ValueError(
            f"{name} contains {invalid_count} value(s) outside the official "
            f"class set {tuple(labels)!r}. Examples: {examples}"
        )
    return encoded, invalid_count


def _encode_evasion(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    validate: bool,
) -> tuple[np.ndarray, int]:
    """Encode evasion_space; DACON explicitly allows integer 0 or 1 only."""
    array = _as_1d_object_array(values, name=name)
    encoded = np.full(len(array), -1, dtype=np.int64)

    for idx, value in enumerate(array):
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            encoded[idx] = int(value)

    invalid_count = int(np.sum(encoded < 0))
    if validate and invalid_count:
        examples = [
            repr(value)
            for value, code in zip(array.tolist(), encoded.tolist())
            if code < 0
        ][:5]
        raise ValueError(
            f"{name} must contain only integer 0 or 1. Invalid examples: {examples}"
        )
    return encoded, invalid_count


def _macro_f1_from_encoded(
    true_encoded: np.ndarray,
    pred_encoded: np.ndarray,
    *,
    num_classes: int,
) -> float:
    """Macro-F1 over the fixed official class set.

    Prediction code -1 means an invalid/out-of-domain prediction and is treated
    as wrong: it contributes a false negative to the true class.
    """
    _check_same_length(y_true=true_encoded, y_pred=pred_encoded)
    if len(true_encoded) == 0:
        raise ValueError("Cannot calculate Macro-F1 on an empty input.")

    class_f1: list[float] = []
    for class_index in range(num_classes):
        true_is_class = true_encoded == class_index
        pred_is_class = pred_encoded == class_index
        tp = int(np.sum(true_is_class & pred_is_class))
        fp = int(np.sum(~true_is_class & pred_is_class))
        fn = int(np.sum(true_is_class & ~pred_is_class))
        denominator = 2 * tp + fp + fn
        class_f1.append(0.0 if denominator == 0 else 2.0 * tp / denominator)

    return float(np.mean(class_f1))


def macro_f1(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any],
) -> float:
    """Macro-F1 over an explicit fixed class set.

    Ground truth must be valid. Unknown prediction labels are counted as wrong.
    """
    true_encoded, _ = _encode_labels(
        y_true, labels=labels, name="y_true", validate=True
    )
    pred_encoded, _ = _encode_labels(
        y_pred, labels=labels, name="y_pred", validate=False
    )
    return _macro_f1_from_encoded(
        true_encoded, pred_encoded, num_classes=len(labels)
    )


# Stage 1 ---------------------------------------------------------------------


def stage1_score(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> float:
    return macro_f1(y_true, y_pred, labels=STAGE1_LABELS)


# Stage 2: frame/time helpers -------------------------------------------------


def temporal_accuracy(
    y_true_seconds: Sequence[float] | np.ndarray,
    y_pred_seconds: Sequence[float] | np.ndarray,
    *,
    tolerance_seconds: float = STAGE2_TIME_TOLERANCE_SECONDS,
) -> float:
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative.")

    true = _as_1d_float_array(y_true_seconds, name="y_true_seconds")
    pred = _as_1d_float_array(y_pred_seconds, name="y_pred_seconds")
    _check_same_length(y_true=true, y_pred=pred)

    if len(true) == 0:
        raise ValueError("Cannot calculate temporal accuracy on an empty input.")
    if not np.all(np.isfinite(true)):
        raise ValueError("Ground-truth timestamps must all be finite.")

    valid_pred = np.isfinite(pred)
    correct = np.zeros(len(true), dtype=bool)
    correct[valid_pred] = (
        np.abs(pred[valid_pred] - true[valid_pred]) <= tolerance_seconds
    )
    return float(correct.mean())


def _coerce_frame_index(value: Any) -> int | None:
    """Return a non-negative integer frame number, or None if invalid."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        frame = int(value)
        return frame if frame >= 0 else None
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            return None
        frame = int(numeric)
        return frame if frame >= 0 else None
    return None


def frame_index_to_time(
    frame_index: Any,
    frame_times_seconds: FrameTimeMap,
) -> float:
    """Map one original frame number to timestamp; invalid input returns NaN."""
    frame = _coerce_frame_index(frame_index)
    if frame is None:
        return float("nan")

    if isinstance(frame_times_seconds, Mapping):
        if frame not in frame_times_seconds:
            return float("nan")
        try:
            seconds = float(frame_times_seconds[frame])
        except (TypeError, ValueError):
            return float("nan")
        return seconds if np.isfinite(seconds) else float("nan")

    try:
        if frame >= len(frame_times_seconds):
            return float("nan")
        seconds = float(frame_times_seconds[frame])
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return seconds if np.isfinite(seconds) else float("nan")


def frame_indices_to_times(
    frame_indices: Sequence[Any] | np.ndarray,
    frame_times_seconds: FrameTimeMap,
) -> np.ndarray:
    """Convert several frame numbers using one video's frame-time map."""
    frames = _as_1d_object_array(frame_indices, name="frame_indices")
    return np.asarray(
        [frame_index_to_time(frame, frame_times_seconds) for frame in frames],
        dtype=np.float64,
    )


def temporal_accuracy_from_frames(
    y_true_seconds: Sequence[float] | np.ndarray,
    y_pred_frames: Sequence[Any] | np.ndarray,
    frame_time_maps: Sequence[FrameTimeMap],
    *,
    tolerance_seconds: float = STAGE2_TIME_TOLERANCE_SECONDS,
) -> tuple[float, int]:
    """Temporal accuracy from submitted original frame numbers.

    frame_time_maps[i] must be the frame-time correspondence for video i.
    Returns (accuracy, invalid_frame_count).
    """
    true = _as_1d_float_array(y_true_seconds, name="y_true_seconds")
    pred_frames = _as_1d_object_array(y_pred_frames, name="y_pred_frames")
    _check_same_length(
        y_true_seconds=true,
        y_pred_frames=pred_frames,
        frame_time_maps=frame_time_maps,
    )

    if len(true) == 0:
        raise ValueError("Cannot calculate temporal accuracy on an empty input.")
    if not np.all(np.isfinite(true)):
        raise ValueError("Ground-truth timestamps must all be finite.")

    pred_seconds = np.asarray(
        [
            frame_index_to_time(frame, frame_time_map)
            for frame, frame_time_map in zip(pred_frames, frame_time_maps)
        ],
        dtype=np.float64,
    )
    invalid_count = int(np.sum(~np.isfinite(pred_seconds)))
    accuracy = temporal_accuracy(
        true, pred_seconds, tolerance_seconds=tolerance_seconds
    )
    return accuracy, invalid_count


# Stage 2: scoring ------------------------------------------------------------


def _stage2_classification_scores(
    *,
    evasion_true: Sequence[int] | np.ndarray,
    evasion_pred: Sequence[Any] | np.ndarray,
    entry_side_true: Sequence[str] | np.ndarray,
    entry_side_pred: Sequence[Any] | np.ndarray,
) -> tuple[float, float, int, int]:
    evasion_true_encoded, _ = _encode_evasion(
        evasion_true, name="evasion_true", validate=True
    )
    evasion_pred_encoded, evasion_invalid = _encode_evasion(
        evasion_pred, name="evasion_pred", validate=False
    )
    evasion_f1 = _macro_f1_from_encoded(
        evasion_true_encoded,
        evasion_pred_encoded,
        num_classes=len(STAGE2_EVASION_LABELS),
    )

    side_true_encoded, _ = _encode_labels(
        entry_side_true,
        labels=STAGE2_ENTRY_SIDE_LABELS,
        name="entry_side_true",
        validate=True,
    )
    side_pred_encoded, side_invalid = _encode_labels(
        entry_side_pred,
        labels=STAGE2_ENTRY_SIDE_LABELS,
        name="entry_side_pred",
        validate=False,
    )
    side_f1 = _macro_f1_from_encoded(
        side_true_encoded,
        side_pred_encoded,
        num_classes=len(STAGE2_ENTRY_SIDE_LABELS),
    )
    return evasion_f1, side_f1, evasion_invalid, side_invalid


def stage2_score(
    *,
    collision_true_seconds: Sequence[float] | np.ndarray,
    collision_pred_seconds: Sequence[float] | np.ndarray,
    entry_true_seconds: Sequence[float] | np.ndarray,
    entry_pred_seconds: Sequence[float] | np.ndarray,
    evasion_true: Sequence[int] | np.ndarray,
    evasion_pred: Sequence[Any] | np.ndarray,
    entry_side_true: Sequence[str] | np.ndarray,
    entry_side_pred: Sequence[Any] | np.ndarray,
    tolerance_seconds: float = STAGE2_TIME_TOLERANCE_SECONDS,
) -> dict[str, float | int]:
    """Stage 2 score for already-converted timestamps.

    Prefer stage2_score_from_frames for evaluator-faithful local validation.
    """
    _check_same_length(
        collision_true_seconds=collision_true_seconds,
        collision_pred_seconds=collision_pred_seconds,
        entry_true_seconds=entry_true_seconds,
        entry_pred_seconds=entry_pred_seconds,
        evasion_true=evasion_true,
        evasion_pred=evasion_pred,
        entry_side_true=entry_side_true,
        entry_side_pred=entry_side_pred,
    )

    collision = temporal_accuracy(
        collision_true_seconds,
        collision_pred_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    entry = temporal_accuracy(
        entry_true_seconds,
        entry_pred_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    evasion, side, evasion_invalid, side_invalid = _stage2_classification_scores(
        evasion_true=evasion_true,
        evasion_pred=evasion_pred,
        entry_side_true=entry_side_true,
        entry_side_pred=entry_side_pred,
    )

    score = (
        STAGE2_WEIGHTS["collision"] * collision
        + STAGE2_WEIGHTS["entry"] * entry
        + STAGE2_WEIGHTS["evasion_space"] * evasion
        + STAGE2_WEIGHTS["entry_side"] * side
    )
    return {
        "score": float(score),
        "collision_accuracy": collision,
        "entry_accuracy": entry,
        "evasion_space_macro_f1": evasion,
        "entry_side_macro_f1": side,
        "evasion_space_invalid_predictions": evasion_invalid,
        "entry_side_invalid_predictions": side_invalid,
    }


def stage2_score_from_frames(
    *,
    collision_true_seconds: Sequence[float] | np.ndarray,
    collision_pred_frames: Sequence[Any] | np.ndarray,
    entry_true_seconds: Sequence[float] | np.ndarray,
    entry_pred_frames: Sequence[Any] | np.ndarray,
    frame_time_maps: Sequence[FrameTimeMap],
    evasion_true: Sequence[int] | np.ndarray,
    evasion_pred: Sequence[Any] | np.ndarray,
    entry_side_true: Sequence[str] | np.ndarray,
    entry_side_pred: Sequence[Any] | np.ndarray,
    tolerance_seconds: float = STAGE2_TIME_TOLERANCE_SECONDS,
) -> dict[str, float | int]:
    """Preferred evaluator-style Stage 2 metric from original frame numbers."""
    _check_same_length(
        collision_true_seconds=collision_true_seconds,
        collision_pred_frames=collision_pred_frames,
        entry_true_seconds=entry_true_seconds,
        entry_pred_frames=entry_pred_frames,
        frame_time_maps=frame_time_maps,
        evasion_true=evasion_true,
        evasion_pred=evasion_pred,
        entry_side_true=entry_side_true,
        entry_side_pred=entry_side_pred,
    )

    collision, collision_invalid = temporal_accuracy_from_frames(
        collision_true_seconds,
        collision_pred_frames,
        frame_time_maps,
        tolerance_seconds=tolerance_seconds,
    )
    entry, entry_invalid = temporal_accuracy_from_frames(
        entry_true_seconds,
        entry_pred_frames,
        frame_time_maps,
        tolerance_seconds=tolerance_seconds,
    )
    evasion, side, evasion_invalid, side_invalid = _stage2_classification_scores(
        evasion_true=evasion_true,
        evasion_pred=evasion_pred,
        entry_side_true=entry_side_true,
        entry_side_pred=entry_side_pred,
    )

    score = (
        STAGE2_WEIGHTS["collision"] * collision
        + STAGE2_WEIGHTS["entry"] * entry
        + STAGE2_WEIGHTS["evasion_space"] * evasion
        + STAGE2_WEIGHTS["entry_side"] * side
    )
    return {
        "score": float(score),
        "collision_accuracy": collision,
        "entry_accuracy": entry,
        "evasion_space_macro_f1": evasion,
        "entry_side_macro_f1": side,
        "collision_invalid_frames": collision_invalid,
        "entry_invalid_frames": entry_invalid,
        "evasion_space_invalid_predictions": evasion_invalid,
        "entry_side_invalid_predictions": side_invalid,
    }


# Stage 3 ---------------------------------------------------------------------


def stage3_score(
    *,
    accel_true: Sequence[str] | np.ndarray,
    accel_pred: Sequence[Any] | np.ndarray,
    steer_true: Sequence[str] | np.ndarray,
    steer_pred: Sequence[Any] | np.ndarray,
) -> dict[str, float | int]:
    """Official-style Stage 3 score.

    The steering mask is based on the ground-truth acceleration label.
    """
    accel_true_arr = _as_1d_object_array(accel_true, name="accel_true")
    accel_pred_arr = _as_1d_object_array(accel_pred, name="accel_pred")
    steer_true_arr = _as_1d_object_array(steer_true, name="steer_true")
    steer_pred_arr = _as_1d_object_array(steer_pred, name="steer_pred")
    _check_same_length(
        accel_true=accel_true_arr,
        accel_pred=accel_pred_arr,
        steer_true=steer_true_arr,
        steer_pred=steer_pred_arr,
    )

    accel_true_encoded, _ = _encode_labels(
        accel_true_arr,
        labels=STAGE3_ACCEL_LABELS,
        name="accel_true",
        validate=True,
    )
    accel_pred_encoded, accel_invalid = _encode_labels(
        accel_pred_arr,
        labels=STAGE3_ACCEL_LABELS,
        name="accel_pred",
        validate=False,
    )
    accel = _macro_f1_from_encoded(
        accel_true_encoded,
        accel_pred_encoded,
        num_classes=len(STAGE3_ACCEL_LABELS),
    )

    moving_mask = accel_true_arr != "STOPPED"
    num_steer_samples = int(np.sum(moving_mask))
    if num_steer_samples == 0:
        raise ValueError(
            "No non-STOPPED ground-truth samples are available for steering."
        )

    steer_true_encoded, _ = _encode_labels(
        steer_true_arr[moving_mask],
        labels=STAGE3_STEER_LABELS,
        name="steer_true",
        validate=True,
    )
    steer_pred_encoded, steer_invalid = _encode_labels(
        steer_pred_arr[moving_mask],
        labels=STAGE3_STEER_LABELS,
        name="steer_pred",
        validate=False,
    )
    steer = _macro_f1_from_encoded(
        steer_true_encoded,
        steer_pred_encoded,
        num_classes=len(STAGE3_STEER_LABELS),
    )

    score = STAGE3_WEIGHTS["accel"] * accel + STAGE3_WEIGHTS["steer"] * steer
    return {
        "score": float(score),
        "accel_macro_f1": accel,
        "steer_macro_f1": steer,
        "steer_num_samples": num_steer_samples,
        "accel_invalid_predictions": accel_invalid,
        "steer_invalid_predictions": steer_invalid,
    }


# Overall ---------------------------------------------------------------------


def overall_score(*, stage1: float, stage2: float, stage3: float) -> float:
    scores = {
        "stage1": float(stage1),
        "stage2": float(stage2),
        "stage3": float(stage3),
    }
    for name, score in scores.items():
        if not np.isfinite(score):
            raise ValueError(f"{name} score must be finite, got {score}.")

    return float(
        STAGE_WEIGHTS["stage1"] * scores["stage1"]
        + STAGE_WEIGHTS["stage2"] * scores["stage2"]
        + STAGE_WEIGHTS["stage3"] * scores["stage3"]
    )


__all__ = [
    "STAGE1_LABELS",
    "STAGE2_EVASION_LABELS",
    "STAGE2_ENTRY_SIDE_LABELS",
    "STAGE3_ACCEL_LABELS",
    "STAGE3_STEER_LABELS",
    "STAGE_WEIGHTS",
    "STAGE2_WEIGHTS",
    "STAGE3_WEIGHTS",
    "STAGE2_TIME_TOLERANCE_SECONDS",
    "FrameTimeMap",
    "macro_f1",
    "stage1_score",
    "temporal_accuracy",
    "frame_index_to_time",
    "frame_indices_to_times",
    "temporal_accuracy_from_frames",
    "stage2_score",
    "stage2_score_from_frames",
    "stage3_score",
    "overall_score",
]

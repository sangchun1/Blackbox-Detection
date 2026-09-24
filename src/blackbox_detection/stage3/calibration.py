from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import ACCEL_CLASSES, STEER_CLASSES
from .metrics import dacon_stage3_metrics


ORDINAL_PREFIX_DECEL = "ordinal_decel_"
ORDINAL_PREFIX_ACCEL = "ordinal_accel_"


def _tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _macro_f1_numpy(
    truth: np.ndarray,
    pred: np.ndarray,
    classes: Sequence[str],
) -> float:
    truth = np.asarray(truth, dtype=str)
    pred = np.asarray(pred, dtype=str)
    scores: list[float] = []
    for cls in classes:
        tp = np.sum((truth == cls) & (pred == cls))
        fp = np.sum((truth != cls) & (pred == cls))
        fn = np.sum((truth == cls) & (pred != cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(scores))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = ["ID", "sample_index", "accel_label", "steer_label"]
    _require_columns(labels, required, "labels")
    out = labels.copy()
    out["ID"] = out["ID"].astype(str)
    out["sample_index"] = pd.to_numeric(out["sample_index"], errors="raise").astype(int)
    if out.duplicated(["ID", "sample_index"]).any():
        raise ValueError("labels contain duplicate ID/sample_index rows")
    bad_accel = sorted(set(out["accel_label"].astype(str)) - set(ACCEL_CLASSES))
    bad_steer = sorted(set(out["steer_label"].astype(str)) - set(STEER_CLASSES))
    if bad_accel or bad_steer:
        raise ValueError(
            f"labels contain invalid classes: accel={bad_accel}, steer={bad_steer}"
        )
    out["accel_label"] = out["accel_label"].astype(str)
    out["steer_label"] = out["steer_label"].astype(str)
    return out.sort_values(["ID", "sample_index"]).reset_index(drop=True)


def validate_features(
    features: pd.DataFrame,
    *,
    ordinal_thresholds_mps2: Sequence[float],
) -> pd.DataFrame:
    required = [
        "ID",
        "sample_index",
        "speed_mps",
        "accel_mps2",
        "accel_raw_mps2",
        "steering_deg",
    ]
    for threshold in ordinal_thresholds_mps2:
        required.extend(
            [
                f"{ORDINAL_PREFIX_DECEL}{_tag(threshold)}",
                f"{ORDINAL_PREFIX_ACCEL}{_tag(threshold)}",
            ]
        )
    _require_columns(features, required, "features")
    out = features.copy()
    out["ID"] = out["ID"].astype(str)
    out["sample_index"] = pd.to_numeric(out["sample_index"], errors="raise").astype(int)
    if out.duplicated(["ID", "sample_index"]).any():
        raise ValueError("features contain duplicate ID/sample_index rows")
    return out.sort_values(["ID", "sample_index"]).reset_index(drop=True)


def merge_labeled_features(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = validate_labels(labels)
    merged = labels.merge(
        features,
        on=["ID", "sample_index"],
        how="left",
        validate="one_to_one",
    )
    missing = merged["speed_mps"].isna()
    if missing.any():
        sample = merged.loc[missing, ["ID", "sample_index"]].head(10).to_dict("records")
        raise ValueError(f"missing model features for labeled rows, examples={sample}")
    return merged


def centered_smooth_features(
    features: pd.DataFrame,
    *,
    window: int,
    columns: Sequence[str],
) -> pd.DataFrame:
    window = int(window)
    if window <= 1:
        return features.copy()
    if window % 2 == 0:
        raise ValueError(f"smoothing window must be odd, got {window}")

    out = features.sort_values(["ID", "sample_index"]).copy()
    for column in columns:
        if column not in out:
            continue
        out[column] = (
            out.groupby("ID", sort=False)[column]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    center=True,
                    min_periods=1,
                ).mean()
            )
            .astype(float)
        )
    return out


def ordinal_signed_score(
    frame: pd.DataFrame,
    *,
    thresholds_mps2: Sequence[float],
) -> np.ndarray:
    accel = np.column_stack(
        [
            frame[f"{ORDINAL_PREFIX_ACCEL}{_tag(t)}"].to_numpy(dtype=float)
            for t in thresholds_mps2
        ]
    )
    decel = np.column_stack(
        [
            frame[f"{ORDINAL_PREFIX_DECEL}{_tag(t)}"].to_numpy(dtype=float)
            for t in thresholds_mps2
        ]
    )
    return (accel - decel).mean(axis=1)


@dataclass(frozen=True)
class AccelCalibration:
    source: str = "fused"
    smoothing_window: int = 1
    stop_speed_mps: float = 0.5
    accel_deadzone_pos_mps2: float = 0.2
    accel_deadzone_neg_mps2: float = 0.2
    accel_bias_mps2: float = 0.0
    ordinal_blend: float = 0.0
    ordinal_scale_mps2: float = 0.5


@dataclass(frozen=True)
class SteerCalibration:
    smoothing_window: int = 1
    steering_sign: int = 1
    steering_bias_deg: float = 0.0
    left_deadzone_deg: float = 5.0
    right_deadzone_deg: float = 5.0


@dataclass(frozen=True)
class Stage3Calibration:
    ordinal_thresholds_mps2: tuple[float, ...]
    accel: AccelCalibration
    steer: SteerCalibration
    version: int = 1
    selection: str = "lovo_consensus"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ordinal_thresholds_mps2"] = list(self.ordinal_thresholds_mps2)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage3Calibration":
        return cls(
            ordinal_thresholds_mps2=tuple(
                float(x) for x in payload["ordinal_thresholds_mps2"]
            ),
            accel=AccelCalibration(**dict(payload["accel"])),
            steer=SteerCalibration(**dict(payload["steer"])),
            version=int(payload.get("version", 1)),
            selection=str(payload.get("selection", "unknown")),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Stage3Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _smoothed_cache(
    features: pd.DataFrame,
    *,
    windows: Iterable[int],
    ordinal_thresholds_mps2: Sequence[float],
) -> dict[int, pd.DataFrame]:
    ordinal_columns = []
    for t in ordinal_thresholds_mps2:
        ordinal_columns += [
            f"{ORDINAL_PREFIX_DECEL}{_tag(t)}",
            f"{ORDINAL_PREFIX_ACCEL}{_tag(t)}",
        ]
    columns = [
        "speed_mps",
        "accel_mps2",
        "accel_raw_mps2",
        "steering_deg",
        *ordinal_columns,
    ]
    return {
        int(window): centered_smooth_features(
            features,
            window=int(window),
            columns=columns,
        )
        for window in sorted({int(w) for w in windows})
    }


def predict_accel_labels(
    labeled_or_features: pd.DataFrame,
    params: AccelCalibration,
    *,
    ordinal_thresholds_mps2: Sequence[float],
) -> np.ndarray:
    source_col = {
        "fused": "accel_mps2",
        "raw": "accel_raw_mps2",
    }.get(str(params.source))
    if source_col is None:
        raise ValueError(f"unknown accel source: {params.source}")

    speed = labeled_or_features["speed_mps"].to_numpy(dtype=float)
    scalar = labeled_or_features[source_col].to_numpy(dtype=float)
    score = ordinal_signed_score(
        labeled_or_features,
        thresholds_mps2=ordinal_thresholds_mps2,
    )
    blend = float(params.ordinal_blend)
    effective = (
        (1.0 - blend) * scalar
        + blend * float(params.ordinal_scale_mps2) * score
        + float(params.accel_bias_mps2)
    )

    pred = np.full(len(labeled_or_features), "CONSTANT", dtype=object)
    stopped = speed <= float(params.stop_speed_mps)
    moving = ~stopped
    pred[stopped] = "STOPPED"
    pred[moving & (effective > float(params.accel_deadzone_pos_mps2))] = "ACCELERATING"
    pred[moving & (effective < -float(params.accel_deadzone_neg_mps2))] = "DECELERATING"
    return pred.astype(str)


def predict_steer_labels(
    labeled_or_features: pd.DataFrame,
    params: SteerCalibration,
) -> np.ndarray:
    steer = (
        int(params.steering_sign)
        * labeled_or_features["steering_deg"].to_numpy(dtype=float)
        + float(params.steering_bias_deg)
    )
    pred = np.full(len(labeled_or_features), "STRAIGHT", dtype=object)
    pred[steer < -float(params.left_deadzone_deg)] = "LEFT"
    pred[steer > float(params.right_deadzone_deg)] = "RIGHT"
    return pred.astype(str)


def apply_calibration(
    features: pd.DataFrame,
    calibration: Stage3Calibration,
) -> pd.DataFrame:
    features = validate_features(
        features,
        ordinal_thresholds_mps2=calibration.ordinal_thresholds_mps2,
    )
    windows = {
        int(calibration.accel.smoothing_window),
        int(calibration.steer.smoothing_window),
    }
    cache = _smoothed_cache(
        features,
        windows=windows,
        ordinal_thresholds_mps2=calibration.ordinal_thresholds_mps2,
    )
    accel_frame = cache[int(calibration.accel.smoothing_window)]
    steer_frame = cache[int(calibration.steer.smoothing_window)]

    out = features[["ID", "sample_index"]].copy()
    out["accel_label"] = predict_accel_labels(
        accel_frame,
        calibration.accel,
        ordinal_thresholds_mps2=calibration.ordinal_thresholds_mps2,
    )
    out["steer_label"] = predict_steer_labels(steer_frame, calibration.steer)
    return out


def evaluate_calibration(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    calibration: Stage3Calibration,
) -> dict[str, float]:
    labels = validate_labels(labels)
    pred = apply_calibration(features, calibration)
    merged = labels.merge(
        pred,
        on=["ID", "sample_index"],
        how="left",
        validate="one_to_one",
        suffixes=("_true", "_pred"),
    )
    return dacon_stage3_metrics(
        merged["accel_label_true"],
        merged["accel_label_pred"],
        merged["steer_label_true"],
        merged["steer_label_pred"],
    )


def _candidate_penalty_accel(p: AccelCalibration) -> tuple:
    return (
        float(p.ordinal_blend),
        abs(float(p.accel_bias_mps2)),
        abs(float(p.accel_deadzone_pos_mps2) - float(p.accel_deadzone_neg_mps2)),
        abs(int(p.smoothing_window) - 1),
        0 if p.source == "fused" else 1,
    )


def _candidate_penalty_steer(p: SteerCalibration) -> tuple:
    return (
        0 if int(p.steering_sign) == 1 else 1,
        abs(float(p.steering_bias_deg)),
        abs(float(p.left_deadzone_deg) - float(p.right_deadzone_deg)),
        abs(int(p.smoothing_window) - 1),
    )


def _best_by_score(
    rows: list[tuple[float, Any]],
    penalty_fn,
):
    if not rows:
        raise RuntimeError("empty candidate set")
    rows.sort(key=lambda x: (-float(x[0]), penalty_fn(x[1])))
    return rows[0], rows


def fit_accel_calibration(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    ordinal_thresholds_mps2: Sequence[float],
    grid: Mapping[str, Sequence[Any]],
    top_k_scalar: int = 24,
) -> tuple[AccelCalibration, pd.DataFrame]:
    labels = validate_labels(labels)
    windows = [int(x) for x in grid["smoothing_window"]]
    cache = _smoothed_cache(
        features,
        windows=windows,
        ordinal_thresholds_mps2=ordinal_thresholds_mps2,
    )

    scalar_rows: list[tuple[float, AccelCalibration]] = []
    for (
        source,
        window,
        stop,
        pos_dz,
        neg_dz,
        bias,
    ) in product(
        grid["source"],
        windows,
        grid["stop_speed_mps"],
        grid["accel_deadzone_pos_mps2"],
        grid["accel_deadzone_neg_mps2"],
        grid["accel_bias_mps2"],
    ):
        labeled = merge_labeled_features(cache[int(window)], labels)
        params = AccelCalibration(
            source=str(source),
            smoothing_window=int(window),
            stop_speed_mps=float(stop),
            accel_deadzone_pos_mps2=float(pos_dz),
            accel_deadzone_neg_mps2=float(neg_dz),
            accel_bias_mps2=float(bias),
            ordinal_blend=0.0,
            ordinal_scale_mps2=float(grid["ordinal_scale_mps2"][0]),
        )
        pred = predict_accel_labels(
            labeled,
            params,
            ordinal_thresholds_mps2=ordinal_thresholds_mps2,
        )
        score = _macro_f1_numpy(
            labeled["accel_label"].to_numpy(),
            pred,
            ACCEL_CLASSES,
        )
        scalar_rows.append((score, params))

    _, ordered_scalar = _best_by_score(scalar_rows, _candidate_penalty_accel)
    top_scalar = [p for _, p in ordered_scalar[: max(int(top_k_scalar), 1)]]

    all_rows: list[tuple[float, AccelCalibration]] = list(scalar_rows)
    positive_blends = [float(x) for x in grid["ordinal_blend"] if float(x) > 0]
    for base in top_scalar:
        labeled = merge_labeled_features(
            cache[int(base.smoothing_window)],
            labels,
        )
        for blend, scale in product(
            positive_blends,
            grid["ordinal_scale_mps2"],
        ):
            params = AccelCalibration(
                source=base.source,
                smoothing_window=base.smoothing_window,
                stop_speed_mps=base.stop_speed_mps,
                accel_deadzone_pos_mps2=base.accel_deadzone_pos_mps2,
                accel_deadzone_neg_mps2=base.accel_deadzone_neg_mps2,
                accel_bias_mps2=base.accel_bias_mps2,
                ordinal_blend=float(blend),
                ordinal_scale_mps2=float(scale),
            )
            pred = predict_accel_labels(
                labeled,
                params,
                ordinal_thresholds_mps2=ordinal_thresholds_mps2,
            )
            score = _macro_f1_numpy(
                labeled["accel_label"].to_numpy(),
                pred,
                ACCEL_CLASSES,
            )
            all_rows.append((score, params))

    (best_score, best), ordered = _best_by_score(all_rows, _candidate_penalty_accel)
    leaderboard = pd.DataFrame(
        [
            {"accel_macro_f1": float(score), **asdict(params)}
            for score, params in ordered[:100]
        ]
    )
    return best, leaderboard


def fit_steer_calibration(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    ordinal_thresholds_mps2: Sequence[float],
    grid: Mapping[str, Sequence[Any]],
) -> tuple[SteerCalibration, pd.DataFrame]:
    labels = validate_labels(labels)
    windows = [int(x) for x in grid["smoothing_window"]]
    cache = _smoothed_cache(
        features,
        windows=windows,
        ordinal_thresholds_mps2=ordinal_thresholds_mps2,
    )

    rows: list[tuple[float, SteerCalibration]] = []
    for (
        window,
        sign,
        bias,
        left_dz,
        right_dz,
    ) in product(
        windows,
        grid["steering_sign"],
        grid["steering_bias_deg"],
        grid["left_deadzone_deg"],
        grid["right_deadzone_deg"],
    ):
        labeled = merge_labeled_features(cache[int(window)], labels)
        params = SteerCalibration(
            smoothing_window=int(window),
            steering_sign=int(sign),
            steering_bias_deg=float(bias),
            left_deadzone_deg=float(left_dz),
            right_deadzone_deg=float(right_dz),
        )
        pred = predict_steer_labels(labeled, params)

        moving = labeled["accel_label"].to_numpy(dtype=str) != "STOPPED"
        if moving.any():
            score = _macro_f1_numpy(
                labeled.loc[moving, "steer_label"].to_numpy(),
                pred[moving],
                STEER_CLASSES,
            )
        else:
            score = 0.0
        rows.append((score, params))

    (best_score, best), ordered = _best_by_score(rows, _candidate_penalty_steer)
    leaderboard = pd.DataFrame(
        [
            {"steer_macro_f1": float(score), **asdict(params)}
            for score, params in ordered[:100]
        ]
    )
    return best, leaderboard


def _consensus_dataclass(items: Sequence[Any], cls):
    if not items:
        raise ValueError("cannot build consensus from no items")
    records = [asdict(x) for x in items]
    result: dict[str, Any] = {}
    for key in records[0]:
        values = [r[key] for r in records]
        first = values[0]
        if isinstance(first, str):
            result[key] = Counter(values).most_common(1)[0][0]
        elif isinstance(first, (bool, np.bool_)):
            result[key] = bool(Counter(values).most_common(1)[0][0])
        elif isinstance(first, (int, np.integer)):
            # For discrete integer hyperparameters (smoothing/sign), use mode.
            result[key] = int(Counter(values).most_common(1)[0][0])
        else:
            result[key] = float(np.median(np.asarray(values, dtype=float)))
    return cls(**result)


def leave_one_video_out_calibration(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    ordinal_thresholds_mps2: Sequence[float],
    accel_grid: Mapping[str, Sequence[Any]],
    steer_grid: Mapping[str, Sequence[Any]],
    top_k_scalar: int = 24,
) -> dict[str, Any]:
    features = validate_features(
        features,
        ordinal_thresholds_mps2=ordinal_thresholds_mps2,
    )
    labels = validate_labels(labels)
    ids = sorted(labels["ID"].unique())
    if len(ids) < 2:
        raise ValueError("LOVO calibration requires at least 2 labeled videos")

    fold_rows: list[dict[str, Any]] = []
    fold_accel: list[AccelCalibration] = []
    fold_steer: list[SteerCalibration] = []

    for held_id in ids:
        train_labels = labels[labels["ID"] != held_id].reset_index(drop=True)
        held_labels = labels[labels["ID"] == held_id].reset_index(drop=True)

        accel_params, _ = fit_accel_calibration(
            features,
            train_labels,
            ordinal_thresholds_mps2=ordinal_thresholds_mps2,
            grid=accel_grid,
            top_k_scalar=top_k_scalar,
        )
        steer_params, _ = fit_steer_calibration(
            features,
            train_labels,
            ordinal_thresholds_mps2=ordinal_thresholds_mps2,
            grid=steer_grid,
        )
        calibration = Stage3Calibration(
            ordinal_thresholds_mps2=tuple(float(x) for x in ordinal_thresholds_mps2),
            accel=accel_params,
            steer=steer_params,
            selection=f"lovo_train_without_{held_id}",
        )
        score = evaluate_calibration(features, held_labels, calibration)
        fold_rows.append(
            {
                "held_id": held_id,
                **score,
                **{f"accel/{k}": v for k, v in asdict(accel_params).items()},
                **{f"steer/{k}": v for k, v in asdict(steer_params).items()},
            }
        )
        fold_accel.append(accel_params)
        fold_steer.append(steer_params)

    allfit_accel, accel_leaderboard = fit_accel_calibration(
        features,
        labels,
        ordinal_thresholds_mps2=ordinal_thresholds_mps2,
        grid=accel_grid,
        top_k_scalar=top_k_scalar,
    )
    allfit_steer, steer_leaderboard = fit_steer_calibration(
        features,
        labels,
        ordinal_thresholds_mps2=ordinal_thresholds_mps2,
        grid=steer_grid,
    )
    allfit = Stage3Calibration(
        ordinal_thresholds_mps2=tuple(float(x) for x in ordinal_thresholds_mps2),
        accel=allfit_accel,
        steer=allfit_steer,
        selection="all_labels_best",
    )
    consensus = Stage3Calibration(
        ordinal_thresholds_mps2=tuple(float(x) for x in ordinal_thresholds_mps2),
        accel=_consensus_dataclass(fold_accel, AccelCalibration),
        steer=_consensus_dataclass(fold_steer, SteerCalibration),
        selection="lovo_consensus",
    )

    fold_df = pd.DataFrame(fold_rows)
    report = {
        "folds": fold_rows,
        "cv_mean": {
            "stage3_score": float(fold_df["stage3_score"].mean()),
            "accel_macro_f1": float(fold_df["accel_macro_f1"].mean()),
            "steer_macro_f1": float(fold_df["steer_macro_f1"].mean()),
        },
        "cv_std": {
            "stage3_score": float(fold_df["stage3_score"].std(ddof=0)),
            "accel_macro_f1": float(fold_df["accel_macro_f1"].std(ddof=0)),
            "steer_macro_f1": float(fold_df["steer_macro_f1"].std(ddof=0)),
        },
        "allfit": {
            "calibration": allfit.to_dict(),
            "metrics_on_all_labels": evaluate_calibration(features, labels, allfit),
        },
        "consensus": {
            "calibration": consensus.to_dict(),
            "metrics_on_all_labels": evaluate_calibration(features, labels, consensus),
        },
        "accel_leaderboard": accel_leaderboard.to_dict("records"),
        "steer_leaderboard": steer_leaderboard.to_dict("records"),
    }
    return report


def save_calibration_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "calibration_report.json"
    report_path.write_text(
        json.dumps(dict(report), indent=2, default=str),
        encoding="utf-8",
    )

    fold_path = output_dir / "lovo_folds.csv"
    pd.DataFrame(report["folds"]).to_csv(fold_path, index=False)

    accel_path = output_dir / "accel_candidates_top100.csv"
    pd.DataFrame(report["accel_leaderboard"]).to_csv(accel_path, index=False)

    steer_path = output_dir / "steer_candidates_top100.csv"
    pd.DataFrame(report["steer_leaderboard"]).to_csv(steer_path, index=False)

    consensus = Stage3Calibration.from_dict(report["consensus"]["calibration"])
    allfit = Stage3Calibration.from_dict(report["allfit"]["calibration"])
    consensus_path = consensus.save(output_dir / "stage3_calibration_consensus.json")
    allfit_path = allfit.save(output_dir / "stage3_calibration_allfit.json")

    return {
        "report": report_path,
        "folds": fold_path,
        "accel_candidates": accel_path,
        "steer_candidates": steer_path,
        "consensus": consensus_path,
        "allfit": allfit_path,
    }


__all__ = [
    "AccelCalibration",
    "SteerCalibration",
    "Stage3Calibration",
    "validate_labels",
    "validate_features",
    "centered_smooth_features",
    "ordinal_signed_score",
    "predict_accel_labels",
    "predict_steer_labels",
    "apply_calibration",
    "evaluate_calibration",
    "fit_accel_calibration",
    "fit_steer_calibration",
    "leave_one_video_out_calibration",
    "save_calibration_report",
]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .constants import CAN_TARGETS
from .metrics import (
    ACCEL_CLASSES,
    STEER_CLASSES,
    dacon_stage3_metrics,
    denormalize,
)


@dataclass(frozen=True)
class ProxyRule:
    """One diagnostic rule that maps continuous CAN signals to Stage-3-like labels.

    IMPORTANT:
    These thresholds are *not* DACON's hidden thresholds. They are only used to
    monitor whether continuous-CAN pretraining produces class-consistent motion
    predictions under several plausible dead-zones.
    """

    stop_speed_mps: float
    accel_deadzone_mps2: float
    steer_deadzone_deg: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, float]) -> "ProxyRule":
        return cls(
            stop_speed_mps=float(payload["stop_speed_mps"]),
            accel_deadzone_mps2=float(payload["accel_deadzone_mps2"]),
            steer_deadzone_deg=float(payload["steer_deadzone_deg"]),
        )


def assert_dacon_metric_contract() -> None:
    """Guard the official Stage 3 metric implementation against accidental drift.

    This does not implement a second competition metric. It only exercises
    dacon_stage3_metrics() from the locked metrics.py implementation.
    """

    result = dacon_stage3_metrics(
        ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"],
        ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"],
        ["LEFT", "STRAIGHT", "RIGHT", "LEFT"],
        ["LEFT", "STRAIGHT", "RIGHT", "RIGHT"],
    )
    assert abs(result["accel_macro_f1"] - 1.0) < 1e-12
    assert abs(result["steer_macro_f1"] - 1.0) < 1e-12
    assert result["steer_eval_frames"] == 3
    assert abs(result["stage3_score"] - 1.0) < 1e-12

    result = dacon_stage3_metrics(
        ["CONSTANT", "CONSTANT"],
        ["CONSTANT", "CONSTANT"],
        ["STRAIGHT", "STRAIGHT"],
        ["STRAIGHT", "STRAIGHT"],
    )
    assert abs(result["accel_macro_f1"] - 0.25) < 1e-12
    assert abs(result["steer_macro_f1"] - (1.0 / 3.0)) < 1e-12
    assert abs(result["stage3_score"] - (0.7 * 0.25 + 0.3 / 3.0)) < 1e-12

    result = dacon_stage3_metrics(
        ["CONSTANT", "STOPPED"],
        ["STOPPED", "STOPPED"],
        ["LEFT", "RIGHT"],
        ["RIGHT", "LEFT"],
    )
    assert result["steer_eval_frames"] == 1
    assert abs(result["steer_macro_f1"]) < 1e-12


def _proxy_labels(
    speed_mps: np.ndarray,
    accel_mps2: np.ndarray,
    steering_deg: np.ndarray,
    rule: ProxyRule,
) -> tuple[np.ndarray, np.ndarray]:
    """Map continuous signals to diagnostic Stage-3-like labels.

    Steering sign convention is used consistently for truth and prediction, so
    the Macro-F1 is invariant to a global LEFT/RIGHT sign swap. Per-class LEFT
    vs RIGHT values remain diagnostic until target-domain sign is confirmed.
    """

    speed = np.asarray(speed_mps, dtype=np.float64)
    accel = np.asarray(accel_mps2, dtype=np.float64)
    steer = np.asarray(steering_deg, dtype=np.float64)

    accel_labels = np.full(speed.shape, "CONSTANT", dtype=object)
    stopped = speed <= rule.stop_speed_mps
    moving = ~stopped
    accel_labels[stopped] = "STOPPED"
    accel_labels[moving & (accel > rule.accel_deadzone_mps2)] = "ACCELERATING"
    accel_labels[moving & (accel < -rule.accel_deadzone_mps2)] = "DECELERATING"

    steer_labels = np.full(speed.shape, "STRAIGHT", dtype=object)
    steer_labels[steer < -rule.steer_deadzone_deg] = "LEFT"
    steer_labels[steer > rule.steer_deadzone_deg] = "RIGHT"

    return accel_labels.astype(str), steer_labels.astype(str)


def _per_class_f1(
    truth_accel: np.ndarray,
    pred_accel: np.ndarray,
    truth_steer: np.ndarray,
    pred_steer: np.ndarray,
) -> dict[str, float]:
    accel_scores = f1_score(
        truth_accel,
        pred_accel,
        labels=list(ACCEL_CLASSES),
        average=None,
        zero_division=0,
    )

    moving = truth_accel != "STOPPED"
    if moving.any():
        steer_scores = f1_score(
            truth_steer[moving],
            pred_steer[moving],
            labels=list(STEER_CLASSES),
            average=None,
            zero_division=0,
        )
    else:
        steer_scores = np.zeros(len(STEER_CLASSES), dtype=np.float64)

    result: dict[str, float] = {}
    for name, value in zip(ACCEL_CLASSES, accel_scores, strict=True):
        result[f"f1_accel_{name}"] = float(value)
    for name, value in zip(STEER_CLASSES, steer_scores, strict=True):
        result[f"f1_steer_{name}"] = float(value)
    return result


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _accel_shape_diagnostics(
    truth_accel: np.ndarray,
    pred_accel: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth_accel, dtype=np.float64)
    pred = np.asarray(pred_accel, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[finite]
    pred = pred[finite]

    if truth.size == 0:
        return {}

    truth_std = float(np.std(truth))
    pred_std = float(np.std(pred))
    if truth.size >= 2 and truth_std > 1e-12:
        slope = float(np.polyfit(truth, pred, deg=1)[0])
    else:
        slope = 0.0

    return {
        "diag/accel/gt_std_mps2": truth_std,
        "diag/accel/pred_std_mps2": pred_std,
        "diag/accel/pred_to_gt_std_ratio": float(
            pred_std / max(truth_std, 1e-12)
        ),
        "diag/accel/correlation": _safe_corr(truth, pred),
        "diag/accel/pred_vs_gt_slope": slope,
        "diag/accel/pred_abs_p95_mps2": float(
            np.quantile(np.abs(pred), 0.95)
        ),
        "diag/accel/gt_abs_p95_mps2": float(
            np.quantile(np.abs(truth), 0.95)
        ),
    }


def _safe_binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if target.size == 0 or np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, score))


def _ordinal_aux_diagnostics(
    truth_accel_mps2: np.ndarray,
    logits: np.ndarray,
    thresholds_mps2: np.ndarray,
) -> dict[str, float]:
    """Diagnostic quality of the training-only acceleration ordinal head.

    These are NOT DACON metrics. They answer a narrower question: can the shared
    frozen-backbone temporal representation linearly separate positive and
    negative acceleration events at several continuous-CAN magnitude levels?
    """
    truth = np.asarray(truth_accel_mps2, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64)
    thresholds = np.asarray(thresholds_mps2, dtype=np.float64).reshape(-1)

    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError(
            "ordinal diagnostic logits must have shape [N, K, 2], got "
            f"{logits.shape}"
        )
    if logits.shape[0] != truth.shape[0] or logits.shape[1] != len(thresholds):
        raise ValueError(
            "ordinal diagnostic shape mismatch: "
            f"truth={truth.shape}, logits={logits.shape}, thresholds={thresholds.shape}"
        )

    result: dict[str, float] = {}
    accel_aucs: list[float] = []
    decel_aucs: list[float] = []
    accel_f1s: list[float] = []
    decel_f1s: list[float] = []
    three_state_f1s: list[float] = []

    for k, threshold in enumerate(thresholds):
        tag = f"{float(threshold):.2f}".replace(".", "p")
        decel_truth = truth < -float(threshold)
        accel_truth = truth > float(threshold)

        decel_score = logits[:, k, 0]
        accel_score = logits[:, k, 1]
        decel_pred = decel_score > 0.0
        accel_pred = accel_score > 0.0

        decel_f1 = float(
            f1_score(decel_truth, decel_pred, average="binary", zero_division=0)
        )
        accel_f1 = float(
            f1_score(accel_truth, accel_pred, average="binary", zero_division=0)
        )
        decel_auc = _safe_binary_auc(decel_truth, decel_score)
        accel_auc = _safe_binary_auc(accel_truth, accel_score)

        result[f"aux/ordinal/decel_f1_thr_{tag}"] = decel_f1
        result[f"aux/ordinal/accel_f1_thr_{tag}"] = accel_f1
        result[f"aux/ordinal/decel_auc_thr_{tag}"] = decel_auc
        result[f"aux/ordinal/accel_auc_thr_{tag}"] = accel_auc
        result[f"aux/ordinal/decel_positive_fraction_thr_{tag}"] = float(
            decel_truth.mean()
        )
        result[f"aux/ordinal/accel_positive_fraction_thr_{tag}"] = float(
            accel_truth.mean()
        )

        # A three-state diagnostic using only this auxiliary threshold pair.
        # If both directions fire, select the larger logit. Otherwise the frame
        # is CONSTANT. This is intentionally separate from DACON proxy scoring.
        truth_state = np.full(truth.shape, 1, dtype=np.int64)  # constant
        truth_state[decel_truth] = 0
        truth_state[accel_truth] = 2

        pred_state = np.full(truth.shape, 1, dtype=np.int64)
        only_decel = decel_pred & ~accel_pred
        only_accel = accel_pred & ~decel_pred
        both = decel_pred & accel_pred
        pred_state[only_decel] = 0
        pred_state[only_accel] = 2
        if both.any():
            pred_state[both] = np.where(
                decel_score[both] >= accel_score[both],
                0,
                2,
            )

        three_state = float(
            f1_score(
                truth_state,
                pred_state,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )
        result[f"aux/ordinal/three_state_macro_f1_thr_{tag}"] = three_state

        decel_f1s.append(decel_f1)
        accel_f1s.append(accel_f1)
        three_state_f1s.append(three_state)
        if np.isfinite(decel_auc):
            decel_aucs.append(decel_auc)
        if np.isfinite(accel_auc):
            accel_aucs.append(accel_auc)

    if decel_f1s:
        result["aux/ordinal/mean_decel_f1"] = float(np.mean(decel_f1s))
        result["aux/ordinal/mean_accel_f1"] = float(np.mean(accel_f1s))
        result["aux/ordinal/mean_three_state_macro_f1"] = float(
            np.mean(three_state_f1s)
        )
    if decel_aucs:
        result["aux/ordinal/mean_decel_auc"] = float(np.mean(decel_aucs))
    if accel_aucs:
        result["aux/ordinal/mean_accel_auc"] = float(np.mean(accel_aucs))

    return result


class Stage3ValidationAccumulator:
    """Dataset-level validation metrics for continuous-CAN pretraining.

    Regression MAE/RMSE is accumulated globally over all valid frames.
    Proxy Macro-F1 delegates its score semantics to dacon_stage3_metrics() from
    the locked metrics.py. Additional accel-shape diagnostics are continuous
    diagnostics only; they are not competition metrics.
    """

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, float]],
        proxy_rules: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.stats = {str(k): dict(v) for k, v in stats.items()}
        self.proxy_rules = {
            str(name): ProxyRule.from_mapping(payload)
            for name, payload in dict(proxy_rules or {}).items()
        }

        self._reg = {
            name: {"n": 0, "abs": 0.0, "sq": 0.0}
            for name in CAN_TARGETS
        }

        self._proxy_truth: dict[str, list[np.ndarray]] = {
            "speed_mps": [],
            "accel_from_speed_mps2": [],
            "steering_deg": [],
        }
        self._proxy_pred: dict[str, list[np.ndarray]] = {
            "speed_mps": [],
            "accel_from_speed_mps2": [],
            "steering_deg": [],
        }

        self._ordinal_truth_accel: list[np.ndarray] = []
        self._ordinal_logits: list[np.ndarray] = []
        self._ordinal_thresholds_mps2: np.ndarray | None = None

    def update(self, outputs, target, valid) -> None:
        target_np = target.detach().float().cpu().numpy()
        valid_np = valid.detach().cpu().numpy().astype(bool)

        denorm_truth: dict[str, np.ndarray] = {}
        denorm_pred: dict[str, np.ndarray] = {}

        for i, name in enumerate(CAN_TARGETS):
            pred_norm = outputs[name].detach().float().cpu().numpy()
            truth_norm = target_np[..., i]

            pred = denormalize(pred_norm, self.stats[name])
            truth = denormalize(truth_norm, self.stats[name])
            mask = (
                valid_np[..., i]
                & np.isfinite(pred)
                & np.isfinite(truth)
            )

            if mask.any():
                err = (
                    pred[mask].astype(np.float64)
                    - truth[mask].astype(np.float64)
                )
                state = self._reg[name]
                state["n"] += int(err.size)
                state["abs"] += float(np.abs(err).sum())
                state["sq"] += float(np.square(err).sum())

            denorm_truth[name] = truth
            denorm_pred[name] = pred

        # Optional v3-A training-only ordinal head diagnostics. These are kept
        # independent of proxy_rules and of the official metrics.py contract.
        if "accel_ordinal_logits" in outputs:
            ordinal = (
                outputs["accel_ordinal_logits"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            thresholds_tensor = outputs.get(
                "accel_ordinal_thresholds_mps2"
            )
            if thresholds_tensor is None:
                raise KeyError(
                    "accel_ordinal_logits present without "
                    "accel_ordinal_thresholds_mps2"
                )
            thresholds = (
                thresholds_tensor.detach().float().cpu().numpy().reshape(-1)
            )
            accel_idx = CAN_TARGETS.index("accel_from_speed_mps2")
            ordinal_valid = (
                valid_np[..., accel_idx]
                & np.isfinite(denorm_truth["accel_from_speed_mps2"])
                & np.isfinite(ordinal).all(axis=(-1, -2))
            )

            if ordinal.shape[:2] != target_np.shape[:2]:
                raise ValueError(
                    "ordinal logits B/T shape mismatch in validation: "
                    f"{ordinal.shape[:2]} vs {target_np.shape[:2]}"
                )
            if ordinal.shape[2] != len(thresholds) or ordinal.shape[-1] != 2:
                raise ValueError(
                    "ordinal logits threshold/direction shape mismatch: "
                    f"logits={ordinal.shape}, thresholds={thresholds.shape}"
                )

            if self._ordinal_thresholds_mps2 is None:
                self._ordinal_thresholds_mps2 = thresholds.copy()
            elif not np.allclose(
                self._ordinal_thresholds_mps2, thresholds, atol=1e-6, rtol=0.0
            ):
                raise ValueError(
                    "ordinal thresholds changed across validation batches"
                )

            if ordinal_valid.any():
                self._ordinal_truth_accel.append(
                    denorm_truth["accel_from_speed_mps2"][ordinal_valid]
                    .astype(np.float32, copy=False)
                )
                self._ordinal_logits.append(
                    ordinal[ordinal_valid].astype(np.float32, copy=False)
                )

        if not self.proxy_rules:
            return

        indices = {
            name: CAN_TARGETS.index(name)
            for name in (
                "speed_mps",
                "accel_from_speed_mps2",
                "steering_deg",
            )
        }

        common = np.ones(target_np.shape[:-1], dtype=bool)
        for name, idx in indices.items():
            common &= valid_np[..., idx]
            common &= np.isfinite(denorm_truth[name])
            common &= np.isfinite(denorm_pred[name])

        if common.any():
            for name in indices:
                self._proxy_truth[name].append(
                    denorm_truth[name][common].astype(np.float32, copy=False)
                )
                self._proxy_pred[name].append(
                    denorm_pred[name][common].astype(np.float32, copy=False)
                )

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}

        for name, state in self._reg.items():
            n = int(state["n"])
            if n <= 0:
                continue
            result[f"reg/{name}/mae"] = float(state["abs"] / n)
            result[f"reg/{name}/rmse"] = float(np.sqrt(state["sq"] / n))
            result[f"reg/{name}/n"] = float(n)

        if (
            self._ordinal_truth_accel
            and self._ordinal_logits
            and self._ordinal_thresholds_mps2 is not None
        ):
            result.update(
                _ordinal_aux_diagnostics(
                    np.concatenate(self._ordinal_truth_accel),
                    np.concatenate(self._ordinal_logits),
                    self._ordinal_thresholds_mps2,
                )
            )

        if not self.proxy_rules:
            return result

        if not self._proxy_truth["speed_mps"]:
            return result

        truth = {
            name: np.concatenate(values)
            for name, values in self._proxy_truth.items()
        }
        pred = {
            name: np.concatenate(values)
            for name, values in self._proxy_pred.items()
        }

        result.update(
            _accel_shape_diagnostics(
                truth["accel_from_speed_mps2"],
                pred["accel_from_speed_mps2"],
            )
        )

        stage3_scores: list[float] = []
        accel_scores: list[float] = []
        steer_scores: list[float] = []

        for rule_name, rule in self.proxy_rules.items():
            truth_accel, truth_steer = _proxy_labels(
                truth["speed_mps"],
                truth["accel_from_speed_mps2"],
                truth["steering_deg"],
                rule,
            )
            pred_accel, pred_steer = _proxy_labels(
                pred["speed_mps"],
                pred["accel_from_speed_mps2"],
                pred["steering_deg"],
                rule,
            )

            official_semantics = dacon_stage3_metrics(
                truth_accel,
                pred_accel,
                truth_steer,
                pred_steer,
            )

            prefix = f"proxy/{rule_name}"
            result[f"{prefix}/stage3_score"] = official_semantics["stage3_score"]
            result[f"{prefix}/accel_macro_f1"] = official_semantics[
                "accel_macro_f1"
            ]
            result[f"{prefix}/steer_macro_f1"] = official_semantics[
                "steer_macro_f1"
            ]
            result[f"{prefix}/steer_eval_frames"] = float(
                official_semantics["steer_eval_frames"]
            )
            result[f"{prefix}/eval_frames"] = float(len(truth_accel))

            dynamic = np.isin(
                truth_accel,
                ["ACCELERATING", "DECELERATING"],
            )
            if dynamic.any():
                result[f"{prefix}/dynamic_to_constant_rate"] = float(
                    np.mean(pred_accel[dynamic] == "CONSTANT")
                )
                result[f"{prefix}/dynamic_fraction"] = float(dynamic.mean())

            for key, value in _per_class_f1(
                truth_accel,
                pred_accel,
                truth_steer,
                pred_steer,
            ).items():
                result[f"{prefix}/{key}"] = value

            stage3_scores.append(float(official_semantics["stage3_score"]))
            accel_scores.append(float(official_semantics["accel_macro_f1"]))
            steer_scores.append(float(official_semantics["steer_macro_f1"]))

        result["proxy/robust_mean_stage3_score"] = float(
            np.mean(stage3_scores)
        )
        result["proxy/robust_min_stage3_score"] = float(
            np.min(stage3_scores)
        )
        result["proxy/robust_max_stage3_score"] = float(
            np.max(stage3_scores)
        )
        result["proxy/robust_mean_accel_macro_f1"] = float(
            np.mean(accel_scores)
        )
        result["proxy/robust_mean_steer_macro_f1"] = float(
            np.mean(steer_scores)
        )

        return result


__all__ = [
    "ProxyRule",
    "Stage3ValidationAccumulator",
    "assert_dacon_metric_contract",
]

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .constants import CAN_TARGETS
from .losses import can_multitask_loss


def _stat(stats: Mapping, name: str) -> tuple[float, float]:
    item = dict(stats[name])
    return float(item["mean"]), max(float(item["std"]), 1e-6)


def _denorm(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * float(std) + float(mean)


def _balanced_binary_bce(
    logits: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    pos = valid & positive
    neg = valid & ~positive
    terms = []
    if pos.any():
        terms.append(
            F.binary_cross_entropy_with_logits(
                logits[pos],
                torch.ones_like(logits[pos]),
            )
        )
    if neg.any():
        terms.append(
            F.binary_cross_entropy_with_logits(
                logits[neg],
                torch.zeros_like(logits[neg]),
            )
        )
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _balanced_multiclass_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    terms = []
    for cls in range(int(num_classes)):
        mask = valid & (target == cls)
        if mask.any():
            terms.append(
                F.cross_entropy(
                    logits[mask],
                    target[mask],
                    reduction="mean",
                )
            )
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _increasing_stop_loss(
    logits: torch.Tensor,
    speed_mps: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    terms = []
    for k, thr in enumerate(thresholds):
        terms.append(
            _balanced_binary_bce(
                logits[..., k],
                speed_mps <= float(thr),
                valid,
            )
        )
    bce = torch.stack(terms).mean() if terms else logits.sum() * 0.0

    # P(speed <= threshold) should increase as threshold increases.
    if logits.shape[-1] > 1 and valid.any():
        violation = F.relu(logits[..., :-1] - logits[..., 1:])
        mask = valid[..., None].expand_as(violation)
        mono = violation[mask].mean()
    else:
        mono = logits.sum() * 0.0
    return bce, mono


def _signed_threshold_loss(
    logits: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    # logits B T K 2: [negative event, positive event]
    terms = []
    for k, thr in enumerate(thresholds):
        neg = _balanced_binary_bce(
            logits[..., k, 0],
            value < -float(thr),
            valid,
        )
        pos = _balanced_binary_bce(
            logits[..., k, 1],
            value > +float(thr),
            valid,
        )
        terms.append(torch.stack([neg, pos]).mean())
    bce = torch.stack(terms).mean() if terms else logits.sum() * 0.0

    # More stringent magnitude thresholds must not become more probable.
    if logits.shape[2] > 1 and valid.any():
        violation = F.relu(
            logits[:, :, 1:, :] - logits[:, :, :-1, :]
        )
        mask = valid[:, :, None, None].expand_as(violation)
        mono = violation[mask].mean()
    else:
        mono = logits.sum() * 0.0
    return bce, mono


def _activity_ordinal_loss(
    logits: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    terms = []
    for k, thr in enumerate(thresholds):
        terms.append(
            _balanced_binary_bce(
                logits[..., k],
                value > float(thr),
                valid,
            )
        )
    bce = torch.stack(terms).mean() if terms else logits.sum() * 0.0

    if logits.shape[-1] > 1 and valid.any():
        violation = F.relu(logits[..., 1:] - logits[..., :-1])
        mask = valid[..., None].expand_as(violation)
        mono = violation[mask].mean()
    else:
        mono = logits.sum() * 0.0
    return bce, mono


def v5_multitask_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    aux: Mapping[str, torch.Tensor],
    config: Mapping,
    *,
    stats: Mapping,
) -> tuple[torch.Tensor, dict[str, float]]:
    """V5-A objective.

    The official DACON metric code is not touched.  All threshold-based terms
    below are training auxiliaries over several plausible physical thresholds.
    """
    cfg = dict(config)
    base_cfg = dict(cfg.get("base") or {})
    total, base_parts = can_multitask_loss(
        outputs,
        target,
        valid,
        base_cfg,
    )
    parts = {f"base/{k}": v for k, v in base_parts.items()}
    parts.pop("base/total", None)

    speed_idx = CAN_TARGETS.index("speed_mps")
    accel_idx = CAN_TARGETS.index("accel_from_speed_mps2")
    yaw_idx = CAN_TARGETS.index("yaw_rate_rps")

    speed_mean, speed_std = _stat(stats, "speed_mps")
    accel_mean, accel_std = _stat(stats, "accel_from_speed_mps2")
    yaw_mean, yaw_std = _stat(stats, "yaw_rate_rps")

    speed_phys = _denorm(
        target[..., speed_idx], speed_mean, speed_std
    )
    yaw_phys = _denorm(
        target[..., yaw_idx], yaw_mean, yaw_std
    )

    # STOP multi-threshold ordinal.
    stop_cfg = dict(cfg.get("stop_ordinal") or {})
    stop_weight = float(stop_cfg.get("weight", 0.0))
    if stop_weight > 0:
        thresholds = [float(x) for x in stop_cfg["thresholds_mps"]]
        bce, mono = _increasing_stop_loss(
            outputs["stop_ordinal_logits"],
            speed_phys,
            valid[..., speed_idx],
            thresholds,
        )
        mono_w = float(stop_cfg.get("monotonic_weight", 0.10))
        term = bce + mono_w * mono
        total = total + stop_weight * term
        parts["v5/stop_bce"] = float(bce.detach().cpu())
        parts["v5/stop_monotonic"] = float(mono.detach().cpu())

    # Explicit physical delta-speed head.
    delta_cfg = dict(cfg.get("delta_speed") or {})
    delta_weight = float(delta_cfg.get("weight", 0.0))
    if delta_weight > 0:
        pred_delta = outputs["delta_speed_mps"][:, 1:]
        target_delta = speed_phys[:, 1:] - speed_phys[:, :-1]
        delta_valid = (
            valid[:, 1:, speed_idx] & valid[:, :-1, speed_idx]
        )
        if delta_valid.any():
            delta_loss = F.smooth_l1_loss(
                pred_delta[delta_valid],
                target_delta[delta_valid],
                beta=float(delta_cfg.get("beta_mps", 0.05)),
            )
        else:
            delta_loss = pred_delta.sum() * 0.0
        total = total + delta_weight * delta_loss
        parts["v5/delta_speed_mps"] = float(delta_loss.detach().cpu())

        consistency_weight = float(
            delta_cfg.get("accel_consistency_weight", 0.0)
        )
        if consistency_weight > 0:
            pred_accel_phys = _denorm(
                outputs["accel_from_speed_mps2"],
                accel_mean,
                accel_std,
            )
            common = (
                delta_valid
                & valid[:, 1:, accel_idx]
            )
            if common.any():
                dt = float(delta_cfg.get("dt_s", 0.10))
                implied_accel = pred_delta / max(dt, 1e-6)
                consistency = F.smooth_l1_loss(
                    implied_accel[common],
                    pred_accel_phys[:, 1:][common],
                    beta=float(
                        delta_cfg.get(
                            "consistency_beta_mps2", 0.20
                        )
                    ),
                )
            else:
                consistency = pred_delta.sum() * 0.0
            total = total + consistency_weight * consistency
            parts["v5/delta_accel_consistency"] = float(
                consistency.detach().cpu()
            )

    # Shared yaw-turn ordinal supervision: compatible across comma2k19/A2D2.
    turn_cfg = dict(cfg.get("turn_ordinal") or {})
    turn_weight = float(turn_cfg.get("weight", 0.0))
    if turn_weight > 0:
        thresholds = [
            float(x) for x in turn_cfg["thresholds_rps"]
        ]
        bce, mono = _signed_threshold_loss(
            outputs["turn_ordinal_logits"],
            yaw_phys,
            valid[..., yaw_idx],
            thresholds,
        )
        mono_w = float(turn_cfg.get("monotonic_weight", 0.10))
        total = total + turn_weight * (bce + mono_w * mono)
        parts["v5/turn_bce"] = float(bce.detach().cpu())
        parts["v5/turn_monotonic"] = float(mono.detach().cpu())

    # Steering direction auxiliary. A2D2 uses its verified wheel-angle sign;
    # comma uses the existing steering target sign.
    steer_dir_cfg = dict(cfg.get("steer_direction") or {})
    steer_dir_weight = float(steer_dir_cfg.get("weight", 0.0))
    if steer_dir_weight > 0:
        cls = aux["steer_direction_class"].long()
        mask = aux["steer_direction_valid"].bool()
        loss = _balanced_multiclass_ce(
            outputs["steer_direction_logits"],
            cls,
            mask,
            num_classes=3,
        )
        total = total + steer_dir_weight * loss
        parts["v5/steer_direction_ce"] = float(loss.detach().cpu())

    steer_act_cfg = dict(cfg.get("steer_activity_ordinal") or {})
    steer_act_weight = float(steer_act_cfg.get("weight", 0.0))
    if steer_act_weight > 0:
        thresholds = [
            float(x) for x in steer_act_cfg["thresholds"]
        ]
        bce, mono = _activity_ordinal_loss(
            outputs["steer_activity_ordinal_logits"],
            aux["steer_activity"].float(),
            aux["steer_activity_valid"].bool(),
            thresholds,
        )
        mono_w = float(
            steer_act_cfg.get("monotonic_weight", 0.10)
        )
        total = total + steer_act_weight * (bce + mono_w * mono)
        parts["v5/steer_activity_bce"] = float(bce.detach().cpu())
        parts["v5/steer_activity_monotonic"] = float(
            mono.detach().cpu()
        )

    # A2D2-only action auxiliaries.
    brake_cfg = dict(cfg.get("brake_ordinal") or {})
    brake_weight = float(brake_cfg.get("weight", 0.0))
    if brake_weight > 0:
        thresholds = [float(x) for x in brake_cfg["thresholds_bar"]]
        bce, mono = _activity_ordinal_loss(
            outputs["brake_ordinal_logits"],
            aux["brake_pressure_bar"].float(),
            aux["brake_valid"].bool(),
            thresholds,
        )
        mono_w = float(brake_cfg.get("monotonic_weight", 0.10))
        total = total + brake_weight * (bce + mono_w * mono)
        parts["v5/brake_bce"] = float(bce.detach().cpu())
        parts["v5/brake_monotonic"] = float(mono.detach().cpu())

    throttle_cfg = dict(cfg.get("throttle_ordinal") or {})
    throttle_weight = float(throttle_cfg.get("weight", 0.0))
    if throttle_weight > 0:
        thresholds = [
            float(x) for x in throttle_cfg["thresholds_pct"]
        ]
        bce, mono = _activity_ordinal_loss(
            outputs["throttle_ordinal_logits"],
            aux["accelerator_pedal_pct"].float(),
            aux["accelerator_valid"].bool(),
            thresholds,
        )
        mono_w = float(
            throttle_cfg.get("monotonic_weight", 0.10)
        )
        total = total + throttle_weight * (bce + mono_w * mono)
        parts["v5/throttle_bce"] = float(bce.detach().cpu())
        parts["v5/throttle_monotonic"] = float(mono.detach().cpu())

    parts["total"] = float(total.detach().cpu())
    return total, parts


__all__ = ["v5_multitask_loss"]

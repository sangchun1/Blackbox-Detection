from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .constants import CAN_TARGETS


def masked_smooth_l1(pred, target, valid, beta: float = 0.5):
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(
        pred[valid],
        target[valid],
        beta=float(beta),
        reduction="mean",
    )


def _weighted_masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    if valid.sum() == 0:
        return pred.sum() * 0.0

    element = F.smooth_l1_loss(
        pred[valid],
        target[valid],
        beta=float(beta),
        reduction="none",
    )
    weight = sample_weight[valid].to(dtype=element.dtype)
    return (element * weight).sum() / weight.sum().clamp_min(1e-6)


def _parse_loss_config(payload: Mapping | None):
    """Support legacy v1 plus structured v2/v3/v4 Stage-3 loss configs."""
    cfg = dict(payload or {})

    if "weights" in cfg or "mode" in cfg:
        mode = str(cfg.get("mode", "baseline")).lower()
        weights = dict(cfg.get("weights") or {})
        accel_v2 = dict(cfg.get("accel_v2") or {})
        accel_v3 = dict(cfg.get("accel_v3") or {})
        accel_v4 = dict(cfg.get("accel_v4") or {})
        normalization = dict(cfg.get("normalization") or {})
    else:
        # v1 compatibility:
        # {speed_mps: 1.0, accel_from_speed_mps2: 1.5, ...}
        mode = "baseline"
        weights = cfg
        accel_v2 = {}
        accel_v3 = {}
        accel_v4 = {}
        normalization = {}

    return mode, weights, accel_v2, accel_v3, accel_v4, normalization


def _require_stat(normalization: Mapping, target: str) -> tuple[float, float]:
    item = dict(normalization.get(target) or {})
    if "mean" not in item or "std" not in item:
        raise KeyError(
            f"Structured Stage-3 loss requires normalization[{target!r}] "
            "with mean/std. The Stage-3 training notebook injects "
            "target_stats.json at runtime."
        )
    mean = float(item["mean"])
    std = max(float(item["std"]), 1e-6)
    return mean, std


def _denormalize_tensor(
    value: torch.Tensor,
    *,
    mean: float,
    std: float,
) -> torch.Tensor:
    return value * float(std) + float(mean)


def _accel_magnitude_weights(
    target_accel_normalized: torch.Tensor,
    *,
    accel_mean: float,
    accel_std: float,
    scale_mps2: float,
    gain: float,
    max_weight: float,
) -> torch.Tensor:
    target_phys = _denormalize_tensor(
        target_accel_normalized,
        mean=accel_mean,
        std=accel_std,
    )
    scale = max(float(scale_mps2), 1e-6)
    activity = (target_phys.abs() / scale).clamp(0.0, 1.0)
    weight = 1.0 + float(gain) * activity
    return weight.clamp(max=max(float(max_weight), 1.0))


def _multi_threshold_margin_loss(
    pred_accel_normalized: torch.Tensor,
    target_accel_normalized: torch.Tensor,
    valid: torch.Tensor,
    *,
    accel_mean: float,
    accel_std: float,
    thresholds_mps2: list[float],
    temperature_mps2: float,
) -> torch.Tensor:
    """Threshold-robust signed margin supervision on the scalar accel output.

    This does NOT encode DACON's hidden threshold. Several plausible margins are
    enforced simultaneously so the direct acceleration head cannot minimize the
    loss by collapsing all moving frames close to zero.
    """
    pred_phys = _denormalize_tensor(
        pred_accel_normalized,
        mean=accel_mean,
        std=accel_std,
    )
    target_phys = _denormalize_tensor(
        target_accel_normalized,
        mean=accel_mean,
        std=accel_std,
    )

    temperature = max(float(temperature_mps2), 1e-4)
    terms: list[torch.Tensor] = []

    for threshold in thresholds_mps2:
        threshold = float(threshold)
        pos = valid & (target_phys > threshold)
        neg = valid & (target_phys < -threshold)
        class_terms: list[torch.Tensor] = []

        if pos.any():
            # soft hinge: prediction should exceed +threshold
            value = (
                F.softplus((threshold - pred_phys[pos]) / temperature)
                * temperature
                / accel_std
            ).mean()
            class_terms.append(value)

        if neg.any():
            # soft hinge: prediction should be below -threshold
            value = (
                F.softplus((pred_phys[neg] + threshold) / temperature)
                * temperature
                / accel_std
            ).mean()
            class_terms.append(value)

        if class_terms:
            # Equalize positive/negative sign contributions at each threshold.
            terms.append(torch.stack(class_terms).mean())

    if not terms:
        return pred_accel_normalized.sum() * 0.0

    # Equalize the different diagnostic margins as well.
    return torch.stack(terms).mean()


def _speed_delta_loss(
    pred_speed_normalized: torch.Tensor,
    target_speed_normalized: torch.Tensor,
    valid_speed: torch.Tensor,
    *,
    beta_normalized: float,
) -> torch.Tensor:
    """Supervise frame-to-frame speed changes in normalized speed units.

    Using normalized deltas keeps this auxiliary term well-scaled at random
    initialization while still targeting the temporal variation that absolute
    speed regression can over-smooth.
    """
    pred_delta = pred_speed_normalized[:, 1:] - pred_speed_normalized[:, :-1]
    target_delta = target_speed_normalized[:, 1:] - target_speed_normalized[:, :-1]
    valid_delta = valid_speed[:, 1:] & valid_speed[:, :-1]

    if valid_delta.sum() == 0:
        return pred_speed_normalized.sum() * 0.0

    return F.smooth_l1_loss(
        pred_delta[valid_delta],
        target_delta[valid_delta],
        beta=float(beta_normalized),
        reduction="mean",
    )


def _speed_accel_consistency_loss(
    pred_speed_normalized: torch.Tensor,
    pred_accel_normalized: torch.Tensor,
    valid_speed: torch.Tensor,
    valid_accel: torch.Tensor,
    *,
    speed_delta_scale_normalized: float,
    accel_scale_normalized: float,
    beta: float,
) -> torch.Tensor:
    """Bounded sign/motion consistency between speed change and acceleration.

    A raw physical derivative can explode at random initialization because a
    small normalized speed difference is multiplied by speed_std / dt. Instead
    we compare bounded tanh motion codes. The direct GT-supervised speed-delta
    and acceleration losses provide magnitude anchors; this term only encourages
    their temporal direction to agree.
    """
    speed_delta = pred_speed_normalized[:, 1:] - pred_speed_normalized[:, :-1]
    accel_at_next = pred_accel_normalized[:, 1:]
    common = (
        valid_speed[:, 1:]
        & valid_speed[:, :-1]
        & valid_accel[:, 1:]
    )

    if common.sum() == 0:
        return pred_accel_normalized.sum() * 0.0

    speed_scale = max(float(speed_delta_scale_normalized), 1e-6)
    accel_scale = max(float(accel_scale_normalized), 1e-6)
    speed_motion = torch.tanh(speed_delta / speed_scale)
    accel_motion = torch.tanh(accel_at_next / accel_scale)

    return F.smooth_l1_loss(
        speed_motion[common],
        accel_motion[common],
        beta=float(beta),
        reduction="mean",
    )



def _balanced_binary_bce(
    logits: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Binary BCE with equal positive/negative contribution when both exist.

    The acceleration-threshold tasks are deliberately balanced here so a rare
    event cannot be solved by always predicting the negative class. This is
    training-only auxiliary supervision and is unrelated to the DACON metric.
    """
    pos = valid & positive
    neg = valid & ~positive
    terms: list[torch.Tensor] = []

    if pos.any():
        terms.append(
            F.binary_cross_entropy_with_logits(
                logits[pos],
                torch.ones_like(logits[pos]),
                reduction="mean",
            )
        )
    if neg.any():
        terms.append(
            F.binary_cross_entropy_with_logits(
                logits[neg],
                torch.zeros_like(logits[neg]),
                reduction="mean",
            )
        )

    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _accel_ordinal_aux_loss(
    logits: torch.Tensor,
    target_accel_normalized: torch.Tensor,
    valid: torch.Tensor,
    *,
    accel_mean: float,
    accel_std: float,
    thresholds_mps2: list[float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Cumulative threshold supervision for deceleration and acceleration.

    ``logits`` must be B x T x K x 2. The last dimension is:
      0: target acceleration < -threshold  (decelerating event)
      1: target acceleration > +threshold  (accelerating event)

    Each threshold/direction is trained as its own balanced binary task. Using
    several thresholds avoids treating any one guessed cutoff as DACON's hidden
    label rule. The monotonic penalty enforces that event probability cannot
    increase as the requested magnitude threshold becomes stricter.
    """
    if logits.ndim != 4 or logits.shape[-1] != 2:
        raise ValueError(
            "accel_ordinal_logits must have shape [B, T, K, 2], got "
            f"{tuple(logits.shape)}"
        )
    if logits.shape[:2] != target_accel_normalized.shape:
        raise ValueError(
            "ordinal logits and accel targets must share B/T dimensions: "
            f"{tuple(logits.shape[:2])} vs "
            f"{tuple(target_accel_normalized.shape)}"
        )
    if logits.shape[2] != len(thresholds_mps2):
        raise ValueError(
            "ordinal logit threshold count mismatch: "
            f"K={logits.shape[2]} vs {len(thresholds_mps2)} configured"
        )

    target_phys = _denormalize_tensor(
        target_accel_normalized,
        mean=accel_mean,
        std=accel_std,
    )

    threshold_terms: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}

    for k, threshold in enumerate(thresholds_mps2):
        threshold = float(threshold)
        tag = f"{threshold:.2f}".replace(".", "p")

        decel_loss = _balanced_binary_bce(
            logits[..., k, 0],
            target_phys < -threshold,
            valid,
        )
        accel_loss = _balanced_binary_bce(
            logits[..., k, 1],
            target_phys > threshold,
            valid,
        )

        details[f"decel_bce_thr_{tag}"] = decel_loss
        details[f"accel_bce_thr_{tag}"] = accel_loss
        threshold_terms.append(torch.stack([decel_loss, accel_loss]).mean())

    if threshold_terms:
        classification = torch.stack(threshold_terms).mean()
    else:
        classification = logits.sum() * 0.0

    # Cumulative ordinal constraint: P(|a| > larger threshold) should not be
    # greater than P(|a| > smaller threshold), separately per direction.
    if logits.shape[2] > 1 and valid.any():
        violations = F.relu(logits[:, :, 1:, :] - logits[:, :, :-1, :])
        valid_pairs = valid[:, :, None, None].expand_as(violations)
        monotonic = violations[valid_pairs].mean()
    else:
        monotonic = logits.sum() * 0.0

    return classification, monotonic, details

def can_multitask_loss(
    outputs: dict,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: Mapping | None = None,
):
    """Stage-3 CAN loss with backward-compatible v1/v2/v3/v4 modes.

    v4-A keeps the v3-A objectives but the model's primary acceleration output
    is a residual fusion of the scalar regression branch and ordinal branch.
    The fusion itself is supervised by the same continuous-CAN objectives; no
    DACON categorical labels or hidden DACON thresholds are introduced.
    """
    (
        mode,
        base_weights,
        accel_v2,
        accel_v3,
        accel_v4,
        normalization,
    ) = _parse_loss_config(weights)

    total = target.new_tensor(0.0)
    parts: dict[str, float] = {}

    # Always report the original plain SmoothL1 terms so v1/v2 curves remain
    # directly comparable even when v2 optimizes a magnitude-weighted accel term.
    plain_losses: dict[str, torch.Tensor] = {}
    for i, name in enumerate(CAN_TARGETS):
        plain = masked_smooth_l1(
            outputs[name],
            target[..., i],
            valid[..., i],
            beta=0.5,
        )
        plain_losses[name] = plain
        parts[name] = float(plain.detach().cpu())

    structured_accel_modes = {
        "accel_v2",
        "accel-v2",
        "accel_v3_ordinal",
        "accel-v3-ordinal",
        "accel_v4_fusion",
        "accel-v4-fusion",
    }
    if mode not in structured_accel_modes:
        for name in CAN_TARGETS:
            total = total + float(base_weights.get(name, 1.0)) * plain_losses[name]
        parts["total"] = float(total.detach().cpu())
        return total, parts

    speed_idx = CAN_TARGETS.index("speed_mps")
    accel_idx = CAN_TARGETS.index("accel_from_speed_mps2")

    speed_mean, speed_std = _require_stat(normalization, "speed_mps")
    accel_mean, accel_std = _require_stat(
        normalization,
        "accel_from_speed_mps2",
    )

    # Non-acceleration base tasks are unchanged from v1.
    for name in CAN_TARGETS:
        if name == "accel_from_speed_mps2":
            continue
        total = total + float(base_weights.get(name, 1.0)) * plain_losses[name]

    reg_cfg = dict(accel_v2.get("magnitude_weighted_regression") or {})
    magnitude_weights = _accel_magnitude_weights(
        target[..., accel_idx],
        accel_mean=accel_mean,
        accel_std=accel_std,
        scale_mps2=float(reg_cfg.get("scale_mps2", 0.30)),
        gain=float(reg_cfg.get("gain", 2.0)),
        max_weight=float(reg_cfg.get("max_weight", 3.0)),
    )
    weighted_accel = _weighted_masked_smooth_l1(
        outputs["accel_from_speed_mps2"],
        target[..., accel_idx],
        valid[..., accel_idx],
        magnitude_weights,
        beta=float(reg_cfg.get("beta", 0.5)),
    )
    total = total + float(
        base_weights.get("accel_from_speed_mps2", 1.0)
    ) * weighted_accel

    accel_valid = valid[..., accel_idx]
    if accel_valid.any():
        mean_weight = magnitude_weights[accel_valid].mean()
    else:
        mean_weight = magnitude_weights.mean() * 0.0

    parts["accel_v2/weighted_regression"] = float(
        weighted_accel.detach().cpu()
    )
    parts["accel_v2/mean_sample_weight"] = float(
        mean_weight.detach().cpu()
    )

    margin_cfg = dict(accel_v2.get("multi_threshold_margin") or {})
    margin_weight = float(margin_cfg.get("weight", 0.0))
    if margin_weight > 0:
        margin_loss = _multi_threshold_margin_loss(
            outputs["accel_from_speed_mps2"],
            target[..., accel_idx],
            accel_valid,
            accel_mean=accel_mean,
            accel_std=accel_std,
            thresholds_mps2=[
                float(x)
                for x in margin_cfg.get(
                    "thresholds_mps2",
                    [0.10, 0.20, 0.30, 0.50],
                )
            ],
            temperature_mps2=float(
                margin_cfg.get("temperature_mps2", 0.05)
            ),
        )
        total = total + margin_weight * margin_loss
        parts["accel_v2/multi_threshold_margin"] = float(
            margin_loss.detach().cpu()
        )

    delta_cfg = dict(accel_v2.get("speed_delta") or {})
    delta_weight = float(delta_cfg.get("weight", 0.0))
    if delta_weight > 0:
        delta_loss = _speed_delta_loss(
            outputs["speed_mps"],
            target[..., speed_idx],
            valid[..., speed_idx],
            beta_normalized=float(delta_cfg.get("beta_normalized", 0.01)),
        )
        total = total + delta_weight * delta_loss
        parts["accel_v2/speed_delta"] = float(delta_loss.detach().cpu())

    consistency_cfg = dict(accel_v2.get("speed_accel_consistency") or {})
    consistency_weight = float(consistency_cfg.get("weight", 0.0))
    if consistency_weight > 0:
        consistency_loss = _speed_accel_consistency_loss(
            outputs["speed_mps"],
            outputs["accel_from_speed_mps2"],
            valid[..., speed_idx],
            valid[..., accel_idx],
            speed_delta_scale_normalized=float(
                consistency_cfg.get("speed_delta_scale_normalized", 0.01)
            ),
            accel_scale_normalized=float(
                consistency_cfg.get("accel_scale_normalized", 0.5)
            ),
            beta=float(consistency_cfg.get("beta", 0.5)),
        )
        total = total + consistency_weight * consistency_loss
        parts["accel_v2/speed_accel_consistency"] = float(
            consistency_loss.detach().cpu()
        )

    if mode in {
        "accel_v3_ordinal",
        "accel-v3-ordinal",
        "accel_v4_fusion",
        "accel-v4-fusion",
    }:
        ordinal_cfg = dict(accel_v3.get("ordinal") or {})
        ordinal_weight = float(ordinal_cfg.get("weight", 0.0))
        monotonic_weight = float(ordinal_cfg.get("monotonic_weight", 0.0))
        thresholds = [
            float(x)
            for x in ordinal_cfg.get(
                "thresholds_mps2",
                [0.10, 0.20, 0.30, 0.50],
            )
        ]

        if ordinal_weight > 0:
            if "accel_ordinal_logits" not in outputs:
                raise KeyError(
                    "accel_v3_ordinal requires model output "
                    "'accel_ordinal_logits'"
                )

            model_thresholds = outputs.get(
                "accel_ordinal_thresholds_mps2"
            )
            if model_thresholds is not None:
                model_values = [
                    float(x)
                    for x in model_thresholds.detach().cpu().tolist()
                ]
                if len(model_values) != len(thresholds) or any(
                    abs(a - b) > 1e-6
                    for a, b in zip(model_values, thresholds, strict=True)
                ):
                    raise ValueError(
                        "model/loss ordinal thresholds differ: "
                        f"model={model_values}, loss={thresholds}"
                    )

            ordinal_bce, ordinal_monotonic, ordinal_details = (
                _accel_ordinal_aux_loss(
                    outputs["accel_ordinal_logits"],
                    target[..., accel_idx],
                    accel_valid,
                    accel_mean=accel_mean,
                    accel_std=accel_std,
                    thresholds_mps2=thresholds,
                )
            )
            ordinal_total = (
                ordinal_bce + monotonic_weight * ordinal_monotonic
            )
            total = total + ordinal_weight * ordinal_total

            parts["accel_v3/ordinal_bce"] = float(
                ordinal_bce.detach().cpu()
            )
            parts["accel_v3/ordinal_monotonic"] = float(
                ordinal_monotonic.detach().cpu()
            )
            parts["accel_v3/ordinal_total"] = float(
                ordinal_total.detach().cpu()
            )
            for key, value in ordinal_details.items():
                parts[f"accel_v3/{key}"] = float(value.detach().cpu())

    if mode in {"accel_v4_fusion", "accel-v4-fusion"}:
        if "accel_raw_from_speed_mps2" not in outputs:
            raise KeyError(
                "accel_v4_fusion requires model output "
                "'accel_raw_from_speed_mps2'"
            )

        raw_accel = outputs["accel_raw_from_speed_mps2"]
        fused_accel = outputs["accel_from_speed_mps2"]
        raw_plain = masked_smooth_l1(
            raw_accel,
            target[..., accel_idx],
            accel_valid,
            beta=0.5,
        )
        delta = fused_accel - raw_accel

        parts["accel_v4/raw_smooth_l1"] = float(raw_plain.detach().cpu())
        if accel_valid.any():
            parts["accel_v4/fused_minus_raw_abs_mean"] = float(
                delta[accel_valid].abs().mean().detach().cpu()
            )
        else:
            parts["accel_v4/fused_minus_raw_abs_mean"] = 0.0

        gate = outputs.get("accel_fusion_gate")
        if gate is not None:
            parts["accel_v4/fusion_gate"] = float(
                gate.detach().float().mean().cpu()
            )

        signed = outputs.get("accel_ordinal_signed_score")
        if signed is not None and accel_valid.any():
            parts["accel_v4/ordinal_signed_abs_mean"] = float(
                signed[accel_valid].abs().mean().detach().cpu()
            )

    parts["total"] = float(total.detach().cpu())
    return total, parts

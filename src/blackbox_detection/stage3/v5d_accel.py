from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .constants import CAN_TARGETS
from .v5_losses import v5_multitask_loss
from .v5b_finetune import (
    VJEPA21DenseCANV5B,
    split_v5b_optimizer_parameters,
)


def _positive_increasing(
    values: Sequence[float] | None,
    *,
    name: str,
) -> tuple[float, ...]:
    out = tuple(float(x) for x in (values or ()))
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if any(x <= 0 for x in out):
        raise ValueError(f"{name} must contain positive values: {out}")
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"{name} must be strictly increasing: {out}")
    return out


class VJEPA21DenseCANV5D(VJEPA21DenseCANV5B):
    """V5-B plus an acceleration decision auxiliary head.

    The production continuous outputs are intentionally left unchanged.  The new
    head sees physically meaningful predictions already produced by V5-B:
    speed, speed delta, fused/raw acceleration, explicit delta-speed, acceleration
    ordinal probabilities and STOP ordinal probabilities.

    It predicts DECEL / CONSTANT / ACCEL at several plausible acceleration
    thresholds.  This makes CONSTANT an explicit competitor instead of only the
    "both binary ordinal tasks are negative" state used by the original ordinal
    auxiliary head.

    The head is auxiliary during V5-D training.  Whether its probabilities are
    useful for final DACON inference is decided later on the untouched validation
    protocol; no hidden DACON threshold is assumed here.
    """

    def __init__(
        self,
        *args,
        accel_state_thresholds_mps2: Sequence[float],
        accel_state_hidden: int = 96,
        accel_state_dropout: float = 0.10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.accel_state_thresholds_mps2 = _positive_increasing(
            accel_state_thresholds_mps2,
            name="accel_state_thresholds_mps2",
        )

        ordinal_thresholds = tuple(
            float(x)
            for x in getattr(
                self.head,
                "accel_ordinal_thresholds_mps2",
                (),
            )
        )
        if not ordinal_thresholds:
            raise ValueError(
                "V5-D requires the existing acceleration ordinal head."
            )

        stop_thresholds = tuple(
            float(x)
            for x in getattr(self.head, "stop_thresholds_mps", ())
        )
        if not stop_thresholds:
            raise ValueError(
                "V5-D requires the existing STOP ordinal head."
            )

        # Features:
        #   speed, predicted speed delta, fused accel, raw accel,
        #   explicit delta-speed, 2*K accel ordinal probabilities,
        #   S stop ordinal probabilities.
        feature_dim = 5 + 2 * len(ordinal_thresholds) + len(stop_thresholds)
        hidden = int(accel_state_hidden)

        self.accel_state_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(accel_state_dropout)),
            nn.Linear(
                hidden,
                len(self.accel_state_thresholds_mps2) * 3,
            ),
        )

        # Small final-layer initialization preserves the V5-B representation at
        # warm start while avoiding a completely zero gradient to input features.
        final = self.accel_state_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(video)

        speed = outputs["speed_mps"]
        speed_delta = torch.cat(
            [
                torch.zeros_like(speed[:, :1]),
                speed[:, 1:] - speed[:, :-1],
            ],
            dim=1,
        )
        fused_accel = outputs["accel_from_speed_mps2"]
        raw_accel = outputs.get(
            "accel_raw_from_speed_mps2",
            fused_accel,
        )
        explicit_delta = outputs["delta_speed_mps"]

        ordinal_prob = torch.sigmoid(
            outputs["accel_ordinal_logits"]
        ).flatten(start_dim=2)
        stop_prob = torch.sigmoid(outputs["stop_ordinal_logits"])

        state_features = torch.cat(
            [
                speed.unsqueeze(-1),
                speed_delta.unsqueeze(-1),
                fused_accel.unsqueeze(-1),
                raw_accel.unsqueeze(-1),
                (explicit_delta / 0.10).unsqueeze(-1),
                ordinal_prob,
                stop_prob,
            ],
            dim=-1,
        )

        logits = self.accel_state_head(state_features).view(
            state_features.shape[0],
            state_features.shape[1],
            len(self.accel_state_thresholds_mps2),
            3,
        )
        outputs["accel_state_logits"] = logits
        outputs["accel_state_thresholds_mps2"] = logits.new_tensor(
            self.accel_state_thresholds_mps2,
            dtype=torch.float32,
        )
        return outputs


def split_v5d_optimizer_parameters(
    model: VJEPA21DenseCANV5D,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Split the new acceleration head from the already-trained V5-B head."""
    base = split_v5b_optimizer_parameters(model)

    old_head: list[tuple[str, nn.Parameter]] = []
    accel_state: list[tuple[str, nn.Parameter]] = []

    for name, parameter in base["head"]:
        if name.startswith("accel_state_head."):
            accel_state.append((name, parameter))
        else:
            old_head.append((name, parameter))

    if not old_head:
        raise RuntimeError("V5-D existing head parameter family is empty")
    if not accel_state:
        raise RuntimeError("V5-D acceleration-state head family is empty")

    return {
        "head": old_head,
        "accel_state_head": accel_state,
        "backbone_penultimate": base["backbone_penultimate"],
        "backbone_last": base["backbone_last"],
    }


def _balanced_state_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for cls in range(3):
        mask = valid & (target == cls)
        if mask.any():
            terms.append(
                F.cross_entropy(
                    logits[mask],
                    target[mask],
                    reduction="mean",
                    label_smoothing=float(label_smoothing),
                )
            )
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _accel_state_loss(
    logits: torch.Tensor,
    target_accel_mps2: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Sequence[float],
    *,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4 or logits.shape[-1] != 3:
        raise ValueError(
            "accel_state_logits must have shape [B,T,K,3], got "
            f"{tuple(logits.shape)}"
        )
    if logits.shape[2] != len(thresholds):
        raise ValueError(
            f"state threshold mismatch: {logits.shape[2]} vs {len(thresholds)}"
        )

    terms: list[torch.Tensor] = []
    for k, threshold in enumerate(thresholds):
        state = torch.ones_like(
            target_accel_mps2,
            dtype=torch.long,
        )
        state[target_accel_mps2 < -float(threshold)] = 0
        state[target_accel_mps2 > +float(threshold)] = 2
        terms.append(
            _balanced_state_ce(
                logits[..., k, :],
                state,
                valid,
                label_smoothing=label_smoothing,
            )
        )

    classification = torch.stack(terms).mean()

    # DECEL/ACCEL event probability should not increase for a stricter
    # magnitude threshold. CONSTANT is therefore encouraged to expand as the
    # threshold gets larger.
    if logits.shape[2] > 1 and valid.any():
        prob = torch.softmax(logits.float(), dim=-1)
        event = prob[..., (0, 2)]
        violation = F.relu(event[:, :, 1:] - event[:, :, :-1])
        mask = valid[:, :, None, None].expand_as(violation)
        monotonic = violation[mask].mean()
    else:
        monotonic = logits.sum() * 0.0

    return classification, monotonic


def _temporal_accel_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    horizons: Sequence[int],
    beta_normalized: float,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for raw_h in horizons:
        h = int(raw_h)
        if h < 1 or h >= pred.shape[1]:
            continue

        common = valid[:, h:] & valid[:, :-h]
        if not common.any():
            continue

        pred_gradient = (pred[:, h:] - pred[:, :-h]) / float(h)
        target_gradient = (target[:, h:] - target[:, :-h]) / float(h)
        terms.append(
            F.smooth_l1_loss(
                pred_gradient[common],
                target_gradient[common],
                beta=float(beta_normalized),
                reduction="mean",
            )
        )

    if not terms:
        return pred.sum() * 0.0
    return torch.stack(terms).mean()


def _integrated_accel_speed_loss(
    pred_accel_mps2: torch.Tensor,
    target_speed_mps: torch.Tensor,
    valid_speed: torch.Tensor,
    *,
    horizons: Sequence[int],
    dt_s: float,
    beta_mps: float,
) -> torch.Tensor:
    """Multi-horizon kinematic consistency: integral(accel) ~= delta speed.

    Ground-truth speed is used as the independent anchor; this is deliberately
    different from regressing the provided acceleration target again. Longer
    horizons suppress frame-to-frame derivative noise.
    """
    terms: list[torch.Tensor] = []
    dt = float(dt_s)

    for raw_h in horizons:
        h = int(raw_h)
        if h < 1 or h >= pred_accel_mps2.shape[1]:
            continue

        # T-1 interval-aligned acceleration values.  A length-h unfold gives
        # exactly T-h windows, matching speed[:, h:] - speed[:, :-h].
        intervals = pred_accel_mps2[:, :-1].unfold(
            dimension=1,
            size=h,
            step=1,
        )
        predicted_delta_speed = intervals.sum(dim=-1) * dt
        target_delta_speed = (
            target_speed_mps[:, h:] - target_speed_mps[:, :-h]
        )
        common = valid_speed[:, h:] & valid_speed[:, :-h]

        if common.any():
            terms.append(
                F.smooth_l1_loss(
                    predicted_delta_speed[common],
                    target_delta_speed[common],
                    beta=float(beta_mps),
                    reduction="mean",
                )
            )

    if not terms:
        return pred_accel_mps2.sum() * 0.0
    return torch.stack(terms).mean()


def v5d_multitask_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    aux: Mapping[str, torch.Tensor],
    config: Mapping,
    *,
    stats: Mapping,
) -> tuple[torch.Tensor, dict[str, float]]:
    """V5-D acceleration-focused objective.

    V5 loss remains intact.  V5-D adds:
      1) balanced 3-state acceleration decisions over several thresholds;
      2) temporal acceleration-shape supervision;
      3) multi-horizon acceleration integral -> speed-delta consistency.
    """
    total, base_parts = v5_multitask_loss(
        outputs,
        target,
        valid,
        aux,
        config,
        stats=stats,
    )
    parts = dict(base_parts)
    parts.pop("total", None)

    accel_idx = CAN_TARGETS.index("accel_from_speed_mps2")
    speed_idx = CAN_TARGETS.index("speed_mps")

    accel_stat = dict(stats["accel_from_speed_mps2"])
    speed_stat = dict(stats["speed_mps"])
    accel_mean = float(accel_stat["mean"])
    accel_std = max(float(accel_stat["std"]), 1e-6)
    speed_mean = float(speed_stat["mean"])
    speed_std = max(float(speed_stat["std"]), 1e-6)

    target_accel_phys = (
        target[..., accel_idx] * accel_std + accel_mean
    )
    target_speed_phys = (
        target[..., speed_idx] * speed_std + speed_mean
    )
    pred_accel_phys = (
        outputs["accel_from_speed_mps2"] * accel_std + accel_mean
    )

    cfg = dict(config)

    state_cfg = dict(cfg.get("accel_state") or {})
    state_weight = float(state_cfg.get("weight", 0.0))
    if state_weight > 0:
        thresholds = [
            float(x)
            for x in state_cfg["thresholds_mps2"]
        ]
        model_thresholds = [
            float(x)
            for x in outputs[
                "accel_state_thresholds_mps2"
            ].detach().cpu().tolist()
        ]
        if len(model_thresholds) != len(thresholds) or any(
            abs(a - b) > 1e-6
            for a, b in zip(model_thresholds, thresholds, strict=True)
        ):
            raise ValueError(
                "V5-D model/loss state thresholds differ: "
                f"{model_thresholds} vs {thresholds}"
            )

        ce, mono = _accel_state_loss(
            outputs["accel_state_logits"],
            target_accel_phys,
            valid[..., accel_idx],
            thresholds,
            label_smoothing=float(
                state_cfg.get("label_smoothing", 0.02)
            ),
        )
        mono_weight = float(
            state_cfg.get("monotonic_weight", 0.05)
        )
        total = total + state_weight * (ce + mono_weight * mono)
        parts["v5d/accel_state_ce"] = float(ce.detach().cpu())
        parts["v5d/accel_state_monotonic"] = float(
            mono.detach().cpu()
        )

    gradient_cfg = dict(
        cfg.get("accel_temporal_gradient") or {}
    )
    gradient_weight = float(gradient_cfg.get("weight", 0.0))
    if gradient_weight > 0:
        loss = _temporal_accel_gradient_loss(
            outputs["accel_from_speed_mps2"],
            target[..., accel_idx],
            valid[..., accel_idx],
            horizons=gradient_cfg.get("horizons", [1, 2, 4]),
            beta_normalized=float(
                gradient_cfg.get("beta_normalized", 0.10)
            ),
        )
        total = total + gradient_weight * loss
        parts["v5d/accel_temporal_gradient"] = float(
            loss.detach().cpu()
        )

    kine_cfg = dict(cfg.get("integrated_kinematics") or {})
    kine_weight = float(kine_cfg.get("weight", 0.0))
    if kine_weight > 0:
        loss = _integrated_accel_speed_loss(
            pred_accel_phys,
            target_speed_phys,
            valid[..., speed_idx],
            horizons=kine_cfg.get("horizons", [1, 2, 4]),
            dt_s=float(kine_cfg.get("dt_s", 0.10)),
            beta_mps=float(kine_cfg.get("beta_mps", 0.05)),
        )
        total = total + kine_weight * loss
        parts["v5d/integrated_kinematics"] = float(
            loss.detach().cpu()
        )

    parts["total"] = float(total.detach().cpu())
    return total, parts


__all__ = [
    "VJEPA21DenseCANV5D",
    "split_v5d_optimizer_parameters",
    "v5d_multitask_loss",
]

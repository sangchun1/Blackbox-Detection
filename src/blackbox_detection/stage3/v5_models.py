from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .constants import CAN_TARGETS
from .models import DenseTemporalCANHead


def _validate_positive_increasing(
    values: Sequence[float] | None,
    *,
    name: str,
) -> tuple[float, ...]:
    out = tuple(float(x) for x in (values or ()))
    if not out:
        return out
    if any(x <= 0 for x in out):
        raise ValueError(f"{name} must be > 0, got {out}")
    if tuple(sorted(out)) != out or len(set(out)) != len(out):
        raise ValueError(f"{name} must be strictly increasing, got {out}")
    return out


class SpatialMomentPooler(nn.Module):
    """Preserve spatial motion geometry while retaining v4 warm-start behavior.

    V4 averaged all HxW tokens.  V5 keeps that global mean *and* computes three
    signed moments from a 3x4 spatial grid:
      - left/right
      - top/bottom
      - center/edge

    A zero-initialized residual projection means the initial v5 representation
    is exactly the old global spatial mean, while gradients can immediately
    learn spatially asymmetric motion cues.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        grid_size: tuple[int, int] = (3, 4),
        gate_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.grid_h, self.grid_w = map(int, grid_size)

        ys = torch.linspace(-1.0, 1.0, self.grid_h)
        xs = torch.linspace(-1.0, 1.0, self.grid_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        radius = torch.sqrt(xx.square() + yy.square())
        center = -radius
        center = center - center.mean()

        def _normalize(v: torch.Tensor) -> torch.Tensor:
            return v / v.abs().sum().clamp_min(1e-6)

        self.register_buffer("x_weight", _normalize(xx), persistent=False)
        self.register_buffer("y_weight", _normalize(yy), persistent=False)
        self.register_buffer(
            "center_weight",
            _normalize(center),
            persistent=False,
        )

        self.spatial_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(gate_init / (1.0 - gate_init)))
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: B, Tt, Ht, Wt, D
        if tokens.ndim != 5:
            raise ValueError(
                f"SpatialMomentPooler expects [B,T,H,W,D], got {tokens.shape}"
            )

        b, t, h, w, d = tokens.shape
        x = (
            tokens.permute(0, 1, 4, 2, 3)
            .reshape(b * t, d, h, w)
        )
        grid = F.adaptive_avg_pool2d(
            x, (self.grid_h, self.grid_w)
        )
        grid = (
            grid.reshape(b, t, d, self.grid_h * self.grid_w)
            .permute(0, 1, 3, 2)
        )  # B T G D

        global_mean = tokens.mean(dim=(2, 3))

        x_moment = (
            grid
            * self.x_weight.to(dtype=grid.dtype)[None, None, :, None]
        ).sum(dim=2)
        y_moment = (
            grid
            * self.y_weight.to(dtype=grid.dtype)[None, None, :, None]
        ).sum(dim=2)
        center_moment = (
            grid
            * self.center_weight.to(dtype=grid.dtype)[None, None, :, None]
        ).sum(dim=2)

        spatial = self.spatial_proj(
            torch.cat([x_moment, y_moment, center_moment], dim=-1)
        )
        gate = torch.sigmoid(self.gate_logit).to(dtype=spatial.dtype)
        return global_mean + gate * spatial


class DenseTemporalCANHeadV5(DenseTemporalCANHead):
    def __init__(
        self,
        input_dim: int = 768,
        feature_dim: int = 384,
        hidden: int = 256,
        layers: int = 2,
        *,
        accel_ordinal_thresholds_mps2: Sequence[float] | None = None,
        accel_fusion_enabled: bool = True,
        accel_fusion_hidden: int = 64,
        accel_fusion_gate_init: float = 0.10,
        accel_fusion_detach_ordinal_inputs: bool = True,
        stop_thresholds_mps: Sequence[float] | None = None,
        turn_yaw_thresholds_rps: Sequence[float] | None = None,
        steer_activity_thresholds: Sequence[float] | None = None,
        brake_thresholds_bar: Sequence[float] | None = None,
        throttle_thresholds_pct: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden=hidden,
            layers=layers,
            accel_ordinal_thresholds_mps2=accel_ordinal_thresholds_mps2,
            accel_fusion_enabled=accel_fusion_enabled,
            accel_fusion_hidden=accel_fusion_hidden,
            accel_fusion_gate_init=accel_fusion_gate_init,
            accel_fusion_detach_ordinal_inputs=(
                accel_fusion_detach_ordinal_inputs
            ),
        )

        width = hidden * 2
        self.stop_thresholds_mps = _validate_positive_increasing(
            stop_thresholds_mps,
            name="stop_thresholds_mps",
        )
        self.turn_yaw_thresholds_rps = _validate_positive_increasing(
            turn_yaw_thresholds_rps,
            name="turn_yaw_thresholds_rps",
        )
        self.steer_activity_thresholds = _validate_positive_increasing(
            steer_activity_thresholds,
            name="steer_activity_thresholds",
        )
        self.brake_thresholds_bar = _validate_positive_increasing(
            brake_thresholds_bar,
            name="brake_thresholds_bar",
        )
        self.throttle_thresholds_pct = _validate_positive_increasing(
            throttle_thresholds_pct,
            name="throttle_thresholds_pct",
        )

        self.stop_ordinal_head = (
            nn.Linear(width, len(self.stop_thresholds_mps))
            if self.stop_thresholds_mps
            else None
        )
        self.delta_speed_head = nn.Linear(width, 1)

        self.turn_ordinal_head = (
            nn.Linear(width, len(self.turn_yaw_thresholds_rps) * 2)
            if self.turn_yaw_thresholds_rps
            else None
        )
        self.steer_direction_head = nn.Linear(width, 3)
        self.steer_activity_ordinal_head = (
            nn.Linear(width, len(self.steer_activity_thresholds))
            if self.steer_activity_thresholds
            else None
        )
        self.brake_ordinal_head = (
            nn.Linear(width, len(self.brake_thresholds_bar))
            if self.brake_thresholds_bar
            else None
        )
        self.throttle_ordinal_head = (
            nn.Linear(width, len(self.throttle_thresholds_pct))
            if self.throttle_thresholds_pct
            else None
        )

        # Frame 0 has no previous-frame delta. A zero start is a stable prior.
        nn.init.zeros_(self.delta_speed_head.weight)
        nn.init.zeros_(self.delta_speed_head.bias)

    def _temporal_hidden(self, z: torch.Tensor) -> torch.Tensor:
        d1 = torch.cat(
            [torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1
        )
        d2 = torch.cat(
            [torch.zeros_like(d1[:, :1]), d1[:, 1:] - d1[:, :-1]], dim=1
        )
        h = self.project(torch.cat([z, d1, d2], dim=-1))
        h, _ = self.temporal(h)
        return h

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._temporal_hidden(z)

        outputs = {
            name: head(h).squeeze(-1)
            for name, head in self.heads.items()
        }

        if self.accel_ordinal_head is not None:
            ordinal = self.accel_ordinal_head(h)
            ordinal = ordinal.view(
                h.shape[0],
                h.shape[1],
                len(self.accel_ordinal_thresholds_mps2),
                2,
            )
            outputs["accel_ordinal_logits"] = ordinal
            outputs["accel_ordinal_thresholds_mps2"] = ordinal.new_tensor(
                self.accel_ordinal_thresholds_mps2,
                dtype=torch.float32,
            )

            if self.accel_fusion_mlp is not None:
                raw_accel = outputs["accel_from_speed_mps2"]
                ordinal_probs = torch.sigmoid(ordinal)
                probs_for_fusion = (
                    ordinal_probs.detach()
                    if self.accel_fusion_detach_ordinal_inputs
                    else ordinal_probs
                )

                decel_prob = probs_for_fusion[..., 0]
                accel_prob = probs_for_fusion[..., 1]
                signed_score = (
                    accel_prob.mean(dim=-1)
                    - decel_prob.mean(dim=-1)
                )
                activity_score = 0.5 * (
                    accel_prob.mean(dim=-1)
                    + decel_prob.mean(dim=-1)
                )

                threshold_tensor = ordinal.new_tensor(
                    self.accel_ordinal_thresholds_mps2,
                    dtype=probs_for_fusion.dtype,
                )
                threshold_weight = (
                    threshold_tensor
                    / threshold_tensor.sum().clamp_min(1e-6)
                )
                signed_magnitude_score = (
                    (accel_prob - decel_prob) * threshold_weight
                ).sum(dim=-1)

                fusion_input = torch.cat(
                    [
                        raw_accel.unsqueeze(-1),
                        probs_for_fusion.flatten(start_dim=2),
                        signed_score.unsqueeze(-1),
                        signed_magnitude_score.unsqueeze(-1),
                        activity_score.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                residual = self.accel_fusion_mlp(
                    fusion_input
                ).squeeze(-1)
                gate = torch.sigmoid(
                    self.accel_fusion_gate_logit
                ).to(dtype=residual.dtype)
                fused_accel = raw_accel + gate * residual

                outputs["accel_raw_from_speed_mps2"] = raw_accel
                outputs["accel_fusion_residual"] = residual
                outputs["accel_fusion_gate"] = gate
                outputs["accel_ordinal_signed_score"] = signed_score
                outputs[
                    "accel_ordinal_signed_magnitude_score"
                ] = signed_magnitude_score
                outputs["accel_from_speed_mps2"] = fused_accel

        if self.stop_ordinal_head is not None:
            outputs["stop_ordinal_logits"] = self.stop_ordinal_head(h)
            outputs["stop_thresholds_mps"] = h.new_tensor(
                self.stop_thresholds_mps,
                dtype=torch.float32,
            )

        outputs["delta_speed_mps"] = self.delta_speed_head(h).squeeze(-1)

        if self.turn_ordinal_head is not None:
            x = self.turn_ordinal_head(h).view(
                h.shape[0],
                h.shape[1],
                len(self.turn_yaw_thresholds_rps),
                2,
            )
            outputs["turn_ordinal_logits"] = x
            outputs["turn_yaw_thresholds_rps"] = h.new_tensor(
                self.turn_yaw_thresholds_rps,
                dtype=torch.float32,
            )

        outputs["steer_direction_logits"] = self.steer_direction_head(h)

        if self.steer_activity_ordinal_head is not None:
            outputs[
                "steer_activity_ordinal_logits"
            ] = self.steer_activity_ordinal_head(h)
            outputs["steer_activity_thresholds"] = h.new_tensor(
                self.steer_activity_thresholds,
                dtype=torch.float32,
            )

        if self.brake_ordinal_head is not None:
            outputs["brake_ordinal_logits"] = self.brake_ordinal_head(h)
            outputs["brake_thresholds_bar"] = h.new_tensor(
                self.brake_thresholds_bar,
                dtype=torch.float32,
            )

        if self.throttle_ordinal_head is not None:
            outputs[
                "throttle_ordinal_logits"
            ] = self.throttle_ordinal_head(h)
            outputs["throttle_thresholds_pct"] = h.new_tensor(
                self.throttle_thresholds_pct,
                dtype=torch.float32,
            )

        return outputs


class VJEPA21DenseCANV5(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_size: int = 16,
        tubelet_size: int = 2,
        freeze_backbone: bool = True,
        feature_dim: int = 384,
        temporal_hidden: int = 256,
        temporal_layers: int = 2,
        spatial_grid: tuple[int, int] = (3, 4),
        spatial_gate_init: float = 0.10,
        accel_ordinal_thresholds_mps2: Sequence[float] | None = None,
        accel_fusion_enabled: bool = True,
        accel_fusion_hidden: int = 64,
        accel_fusion_gate_init: float = 0.10,
        accel_fusion_detach_ordinal_inputs: bool = True,
        stop_thresholds_mps: Sequence[float] | None = None,
        turn_yaw_thresholds_rps: Sequence[float] | None = None,
        steer_activity_thresholds: Sequence[float] | None = None,
        brake_thresholds_bar: Sequence[float] | None = None,
        throttle_thresholds_pct: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.freeze_backbone = bool(freeze_backbone)

        embed_dim = int(getattr(backbone, "embed_dim", 768))
        self.spatial_pool = SpatialMomentPooler(
            embed_dim,
            grid_size=spatial_grid,
            gate_init=spatial_gate_init,
        )
        self.head = DenseTemporalCANHeadV5(
            embed_dim,
            feature_dim,
            temporal_hidden,
            temporal_layers,
            accel_ordinal_thresholds_mps2=accel_ordinal_thresholds_mps2,
            accel_fusion_enabled=accel_fusion_enabled,
            accel_fusion_hidden=accel_fusion_hidden,
            accel_fusion_gate_init=accel_fusion_gate_init,
            accel_fusion_detach_ordinal_inputs=(
                accel_fusion_detach_ordinal_inputs
            ),
            stop_thresholds_mps=stop_thresholds_mps,
            turn_yaw_thresholds_rps=turn_yaw_thresholds_rps,
            steer_activity_thresholds=steer_activity_thresholds,
            brake_thresholds_bar=brake_thresholds_bar,
            throttle_thresholds_pct=throttle_thresholds_pct,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = video.shape
        ctx = (
            torch.no_grad()
            if self.freeze_backbone
            else torch.enable_grad()
        )
        with ctx:
            outputs = self.backbone(video)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        layers = [
            F.layer_norm(x.float(), (x.shape[-1],))
            for x in outputs
        ]
        tokens = torch.stack(layers, dim=0).mean(dim=0)

        t_tokens = t // self.tubelet_size
        h_tokens = h // self.patch_size
        w_tokens = w // self.patch_size
        expected = t_tokens * h_tokens * w_tokens
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"unexpected V-JEPA token count {tokens.shape[1]}, "
                f"expected {expected} for input {(t, h, w)}"
            )

        tokens = tokens.view(
            b, t_tokens, h_tokens, w_tokens, -1
        )
        z = self.spatial_pool(tokens)
        z = F.interpolate(
            z.transpose(1, 2),
            size=t,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return z

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self._encode(video))


def load_v4_head_warm_start(
    model: VJEPA21DenseCANV5,
    checkpoint_path: str | Path,
) -> dict:
    """Load only v4 head parameters into v5.

    The v5 backbone is constructed for 32 frames, so loading the entire v4
    checkpoint could hit positional/tubelet shape differences.  The continuous
    CAN head is length-agnostic and is copied parameter-by-parameter instead.
    """
    checkpoint_path = Path(checkpoint_path)
    obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = obj.get("model") if isinstance(obj, dict) else None
    if not isinstance(state, dict):
        raise KeyError(
            f"checkpoint has no model state_dict: {checkpoint_path}"
        )

    current = model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    skipped_shape: dict[str, tuple] = {}
    missing_in_v5: list[str] = []

    old_head_keys = [k for k in state if k.startswith("head.")]
    for key in old_head_keys:
        if key not in current:
            missing_in_v5.append(key)
            continue
        if tuple(state[key].shape) != tuple(current[key].shape):
            skipped_shape[key] = (
                tuple(state[key].shape),
                tuple(current[key].shape),
            )
            continue
        copied[key] = state[key]

    if missing_in_v5 or skipped_shape:
        raise RuntimeError(
            "v4 -> v5 head warm-start mismatch: "
            f"missing_in_v5={missing_in_v5}, "
            f"shape_mismatch={skipped_shape}"
        )

    incompatible = model.load_state_dict(copied, strict=False)

    # Missing keys are expected: official v5 backbone state already loaded,
    # spatial pool + new auxiliary heads are intentionally fresh.
    return {
        "copied_head_keys": sorted(copied),
        "copied_head_tensors": len(copied),
        "checkpoint_epoch": obj.get("epoch"),
        "checkpoint_best_score": obj.get("best_score"),
        "v5_missing_after_partial_load": list(incompatible.missing_keys),
        "unexpected_after_partial_load": list(incompatible.unexpected_keys),
    }


__all__ = [
    "SpatialMomentPooler",
    "DenseTemporalCANHeadV5",
    "VJEPA21DenseCANV5",
    "load_v4_head_warm_start",
]

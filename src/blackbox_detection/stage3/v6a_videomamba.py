from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .v5_models import DenseTemporalCANHeadV5, SpatialMomentPooler


def _positive_increasing(values: Sequence[float] | None, *, name: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in (values or ()))
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if any(x <= 0 for x in out):
        raise ValueError(f"{name} must contain positive values: {out}")
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"{name} must be strictly increasing: {out}")
    return out


@dataclass(frozen=True)
class VideoMambaFinetuneReport:
    depth: int
    trainable_layer_indices: tuple[int, ...]
    train_final_norm: bool
    trainable_backbone_params: int
    frozen_backbone_params: int


class VideoMambaDenseTokenAdapter(nn.Module):
    """Expose dense VideoMamba patch tokens as [B,T,H,W,C]."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._ft_layer_indices: tuple[int, ...] = ()
        self._train_final_norm = False
        self._checkpoint_trainable = False
        self._partial_ft_configured = False

    @property
    def embed_dim(self) -> int:
        return int(self.backbone.embed_dim)

    def _spatial_pos_embed(self, h_tokens: int, w_tokens: int, *, dtype, device):
        pos = self.backbone.pos_embed
        extra = pos[:, :1]
        patch = pos[:, 1:]
        n = int(patch.shape[1])
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise RuntimeError(f"VideoMamba checkpoint pos grid is not square: {n}")

        if side != int(h_tokens) or side != int(w_tokens):
            patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2)
            patch = F.interpolate(
                patch.float(),
                size=(int(h_tokens), int(w_tokens)),
                mode="bicubic",
                align_corners=False,
            )
            patch = (
                patch.permute(0, 2, 3, 1)
                .reshape(1, int(h_tokens) * int(w_tokens), -1)
                .to(dtype=pos.dtype)
            )
        return torch.cat([extra, patch], dim=1).to(device=device, dtype=dtype)

    def _temporal_pos_embed(self, t_tokens: int, *, dtype, device):
        pos = self.backbone.temporal_pos_embedding
        if int(pos.shape[1]) != int(t_tokens):
            pos = F.interpolate(
                pos.float().transpose(1, 2),
                size=int(t_tokens),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(dtype=self.backbone.temporal_pos_embedding.dtype)
        return pos.to(device=device, dtype=dtype)

    def configure_frozen_backbone(self) -> VideoMambaFinetuneReport:
        """Freeze the entire VideoMamba backbone for interface adaptation."""
        self.backbone.requires_grad_(False)
        self._ft_layer_indices = ()
        self._train_final_norm = False
        self._checkpoint_trainable = False
        self._partial_ft_configured = True
        self.train(self.training)

        frozen = sum(
            p.numel()
            for p in self.backbone.parameters()
            if not p.requires_grad
        )
        return VideoMambaFinetuneReport(
            depth=len(self.backbone.layers),
            trainable_layer_indices=(),
            train_final_norm=False,
            trainable_backbone_params=0,
            frozen_backbone_params=int(frozen),
        )

    def configure_partial_backbone(
        self,
        *,
        last_n_layers: int = 2,
        train_final_norm: bool = True,
        checkpoint_trainable: bool = False,
    ) -> VideoMambaFinetuneReport:
        layers = self.backbone.layers
        depth = len(layers)
        n = int(last_n_layers)
        if n < 1 or n > depth:
            raise ValueError(f"last_n_layers must be in [1,{depth}], got {n}")

        self.backbone.requires_grad_(False)
        indices = tuple(range(depth - n, depth))
        for idx in indices:
            layers[idx].requires_grad_(True)
        if train_final_norm:
            self.backbone.norm_f.requires_grad_(True)

        self._ft_layer_indices = indices
        self._train_final_norm = bool(train_final_norm)
        self._checkpoint_trainable = bool(checkpoint_trainable)
        self._partial_ft_configured = True
        self.train(self.training)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        return VideoMambaFinetuneReport(
            depth=depth,
            trainable_layer_indices=indices,
            train_final_norm=bool(train_final_norm),
            trainable_backbone_params=int(trainable),
            frozen_backbone_params=int(frozen),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._partial_ft_configured:
            self.backbone.eval()
            if mode:
                for idx in self._ft_layer_indices:
                    self.backbone.layers[idx].train(True)
                if self._train_final_norm:
                    self.backbone.norm_f.train(True)
        return self

    def _run_layers(self, hidden_states, residual):
        layers = self.backbone.layers
        first_trainable = (
            min(self._ft_layer_indices)
            if self._partial_ft_configured
            else len(layers)
        )

        if first_trainable > 0:
            with torch.no_grad():
                for idx in range(first_trainable):
                    hidden_states, residual = layers[idx](
                        hidden_states, residual, inference_params=None
                    )
            hidden_states = hidden_states.detach()
            if residual is not None:
                residual = residual.detach()

        for idx in range(first_trainable, len(layers)):
            hidden_states, residual = layers[idx](
                hidden_states,
                residual,
                inference_params=None,
                use_checkpoint=(
                    self.training and self._checkpoint_trainable
                ),
            )
        return hidden_states, residual

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(video.shape)}")

        x = self.backbone.patch_embed(video)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)

        spatial_pos = self._spatial_pos_embed(h, w, dtype=x.dtype, device=x.device)
        patch_pos = spatial_pos[:, 1:]
        cls = (
            self.backbone.cls_token
            + spatial_pos[:, :1].to(dtype=self.backbone.cls_token.dtype)
        ).expand(b, -1, -1).to(dtype=x.dtype)

        x = x + patch_pos
        x = x.reshape(b, t, h * w, c).permute(0, 2, 1, 3).reshape(b * h * w, t, c)
        x = x + self._temporal_pos_embed(t, dtype=x.dtype, device=x.device)
        x = x.reshape(b, h * w, t, c).permute(0, 2, 1, 3).reshape(b, t * h * w, c)

        hidden_states = self.backbone.pos_drop(torch.cat([cls, x], dim=1))
        residual = None
        hidden_states, residual = self._run_layers(hidden_states, residual)

        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.backbone.drop_path(hidden_states)
        hidden_states = self.backbone.norm_f(
            residual.to(dtype=self.backbone.norm_f.weight.dtype)
        )

        dense = hidden_states[:, 1:]
        expected = int(t * h * w)
        if dense.shape[1] != expected:
            raise RuntimeError(
                f"unexpected VideoMamba token count {dense.shape[1]}, expected {expected}"
            )
        return dense.reshape(b, t, h, w, c)


class VideoMambaDenseCANV6A(nn.Module):
    """VideoMamba backbone with the V5-D Stage3 decision stack."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int = 384,
        temporal_hidden: int = 256,
        temporal_layers: int = 2,
        spatial_grid: tuple[int, int] = (3, 4),
        spatial_gate_init: float = 0.10,
        accel_ordinal_thresholds_mps2: Sequence[float],
        accel_fusion_enabled: bool,
        accel_fusion_hidden: int,
        accel_fusion_gate_init: float,
        accel_fusion_detach_ordinal_inputs: bool,
        accel_state_thresholds_mps2: Sequence[float],
        accel_state_hidden: int = 96,
        accel_state_dropout: float = 0.10,
        stop_thresholds_mps: Sequence[float],
        turn_yaw_thresholds_rps: Sequence[float],
        steer_activity_thresholds: Sequence[float],
        brake_thresholds_bar: Sequence[float],
        throttle_thresholds_pct: Sequence[float],
    ) -> None:
        super().__init__()
        self.backbone = VideoMambaDenseTokenAdapter(backbone)
        embed_dim = int(self.backbone.embed_dim)

        self.spatial_pool = SpatialMomentPooler(
            embed_dim,
            grid_size=spatial_grid,
            gate_init=spatial_gate_init,
        )
        self.head = DenseTemporalCANHeadV5(
            input_dim=embed_dim,
            feature_dim=int(feature_dim),
            hidden=int(temporal_hidden),
            layers=int(temporal_layers),
            accel_ordinal_thresholds_mps2=accel_ordinal_thresholds_mps2,
            accel_fusion_enabled=bool(accel_fusion_enabled),
            accel_fusion_hidden=int(accel_fusion_hidden),
            accel_fusion_gate_init=float(accel_fusion_gate_init),
            accel_fusion_detach_ordinal_inputs=bool(accel_fusion_detach_ordinal_inputs),
            stop_thresholds_mps=stop_thresholds_mps,
            turn_yaw_thresholds_rps=turn_yaw_thresholds_rps,
            steer_activity_thresholds=steer_activity_thresholds,
            brake_thresholds_bar=brake_thresholds_bar,
            throttle_thresholds_pct=throttle_thresholds_pct,
        )

        self.accel_state_thresholds_mps2 = _positive_increasing(
            accel_state_thresholds_mps2,
            name="accel_state_thresholds_mps2",
        )

        ordinal_thresholds = tuple(float(x) for x in self.head.accel_ordinal_thresholds_mps2)
        stop_thresholds = tuple(float(x) for x in self.head.stop_thresholds_mps)
        state_feature_dim = 5 + 2 * len(ordinal_thresholds) + len(stop_thresholds)

        hidden = int(accel_state_hidden)
        self.accel_state_head = nn.Sequential(
            nn.LayerNorm(state_feature_dim),
            nn.Linear(state_feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(accel_state_dropout)),
            nn.Linear(hidden, len(self.accel_state_thresholds_mps2) * 3),
        )
        final = self.accel_state_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)

    def configure_frozen_backbone(self):
        return self.backbone.configure_frozen_backbone()

    def configure_partial_backbone(
        self,
        *,
        last_n_layers=2,
        train_final_norm=True,
        checkpoint_trainable=False,
    ):
        return self.backbone.configure_partial_backbone(
            last_n_layers=last_n_layers,
            train_final_norm=train_final_norm,
            checkpoint_trainable=checkpoint_trainable,
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.spatial_pool(self.backbone(video))
        outputs = self.head(z)

        speed = outputs["speed_mps"]
        speed_delta = torch.cat(
            [torch.zeros_like(speed[:, :1]), speed[:, 1:] - speed[:, :-1]],
            dim=1,
        )
        fused_accel = outputs["accel_from_speed_mps2"]
        raw_accel = outputs.get("accel_raw_from_speed_mps2", fused_accel)
        explicit_delta = outputs["delta_speed_mps"]
        ordinal_prob = torch.sigmoid(outputs["accel_ordinal_logits"]).flatten(start_dim=2)
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
            self.accel_state_thresholds_mps2, dtype=torch.float32
        )
        return outputs


def _checkpoint_state(checkpoint_path: str | Path):
    obj = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return dict(obj["model"]), obj
    if isinstance(obj, dict):
        return dict(obj), {}
    raise TypeError(f"unsupported checkpoint payload: {type(obj)!r}")


def load_v5d_compatible_warm_start(
    model: VideoMambaDenseCANV6A,
    checkpoint_path: str | Path,
) -> dict:
    """Copy V5-D tensors whose semantics and shapes are unchanged."""
    state, meta = _checkpoint_state(checkpoint_path)
    current = model.state_dict()
    fresh_prefixes = ("backbone.", "spatial_pool.", "head.project.")

    copied = {}
    skipped_fresh = []
    skipped_shape = {}
    missing = []

    for key, value in state.items():
        if key.startswith(fresh_prefixes):
            skipped_fresh.append(key)
            continue
        if key not in current:
            missing.append(key)
            continue
        if tuple(value.shape) != tuple(current[key].shape):
            skipped_shape[key] = {
                "source": tuple(value.shape),
                "target": tuple(current[key].shape),
            }
            continue
        copied[key] = value

    incompatible = model.load_state_dict(copied, strict=False)
    return {
        "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_best_score": meta.get("best_score"),
        "copied_tensors": len(copied),
        "copied_keys": sorted(copied),
        "skipped_fresh": sorted(skipped_fresh),
        "skipped_shape": skipped_shape,
        "missing_source_keys": sorted(missing),
        "missing_after_partial_load": sorted(incompatible.missing_keys),
        "unexpected_after_partial_load": sorted(incompatible.unexpected_keys),
    }


def split_v6a_optimizer_parameters(
    model: VideoMambaDenseCANV6A,
    *,
    require_backbone: bool = True,
):
    groups = {
        "interface": [],
        "head": [],
        "accel_state_head": [],
        "backbone_penultimate": [],
        "backbone_last": [],
        "backbone_final_norm": [],
    }

    layers = model.backbone.backbone.layers
    depth = len(layers)
    penultimate_prefix = f"backbone.backbone.layers.{depth - 2}."
    last_prefix = f"backbone.backbone.layers.{depth - 1}."
    norm_prefix = "backbone.backbone.norm_f."

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("spatial_pool.") or name.startswith("head.project."):
            groups["interface"].append((name, p))
        elif name.startswith("accel_state_head."):
            groups["accel_state_head"].append((name, p))
        elif name.startswith(penultimate_prefix):
            groups["backbone_penultimate"].append((name, p))
        elif name.startswith(last_prefix):
            groups["backbone_last"].append((name, p))
        elif name.startswith(norm_prefix):
            groups["backbone_final_norm"].append((name, p))
        elif name.startswith("head."):
            groups["head"].append((name, p))
        else:
            raise RuntimeError(f"unclassified trainable V6-A parameter: {name}")

    always_required = ("interface", "head", "accel_state_head")
    for key in always_required:
        if not groups[key]:
            raise RuntimeError(f"V6-A optimizer family is empty: {key}")

    backbone_keys = (
        "backbone_penultimate",
        "backbone_last",
        "backbone_final_norm",
    )
    if require_backbone:
        for key in backbone_keys:
            if not groups[key]:
                raise RuntimeError(
                    f"V6-A optimizer family is empty: {key}"
                )
    else:
        for key in backbone_keys:
            if groups[key]:
                raise RuntimeError(
                    "Stage-A frozen backbone unexpectedly has trainable "
                    f"parameters in {key}"
                )
    return groups


__all__ = [
    "VideoMambaFinetuneReport",
    "VideoMambaDenseTokenAdapter",
    "VideoMambaDenseCANV6A",
    "load_v5d_compatible_warm_start",
    "split_v6a_optimizer_parameters",
]

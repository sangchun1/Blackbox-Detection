from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .constants import CAN_TARGETS


class DenseTemporalCANHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        feature_dim: int = 384,
        hidden: int = 256,
        layers: int = 2,
        accel_ordinal_thresholds_mps2: Sequence[float] | None = None,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim * 3, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.1),
        )
        self.temporal = nn.GRU(
            feature_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden * 2, 1) for name in CAN_TARGETS}
        )

        thresholds = tuple(
            float(x) for x in (accel_ordinal_thresholds_mps2 or ())
        )
        if thresholds:
            if any(x <= 0 for x in thresholds):
                raise ValueError(
                    "accel ordinal thresholds must all be > 0, got "
                    f"{thresholds}"
                )
            if tuple(sorted(thresholds)) != thresholds:
                raise ValueError(
                    "accel ordinal thresholds must be strictly increasing, got "
                    f"{thresholds}"
                )
            if len(set(thresholds)) != len(thresholds):
                raise ValueError(
                    "accel ordinal thresholds must be unique, got "
                    f"{thresholds}"
                )

        self.accel_ordinal_thresholds_mps2 = thresholds
        self.accel_ordinal_head = (
            nn.Linear(hidden * 2, len(thresholds) * 2)
            if thresholds
            else None
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # z: B T D
        d1 = torch.cat(
            [torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1
        )
        d2 = torch.cat(
            [torch.zeros_like(d1[:, :1]), d1[:, 1:] - d1[:, :-1]], dim=1
        )
        h = self.project(torch.cat([z, d1, d2], dim=-1))
        h, _ = self.temporal(h)

        outputs = {
            name: head(h).squeeze(-1)
            for name, head in self.heads.items()
        }

        if self.accel_ordinal_head is not None:
            # B T K 2, where the last dimension is:
            #   0 = DECEL event: a < -threshold
            #   1 = ACCEL event: a > +threshold
            ordinal = self.accel_ordinal_head(h)
            ordinal = ordinal.view(
                h.shape[0],
                h.shape[1],
                len(self.accel_ordinal_thresholds_mps2),
                2,
            )
            outputs["accel_ordinal_logits"] = ordinal
            outputs["accel_ordinal_thresholds_mps2"] = ordinal.new_tensor(
                self.accel_ordinal_thresholds_mps2
            )

        return outputs


class VJEPA21DenseCAN(nn.Module):
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
        accel_ordinal_thresholds_mps2: Sequence[float] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.freeze_backbone = bool(freeze_backbone)
        embed_dim = int(getattr(backbone, "embed_dim", 768))
        self.head = DenseTemporalCANHead(
            embed_dim,
            feature_dim,
            temporal_hidden,
            temporal_layers,
            accel_ordinal_thresholds_mps2=accel_ordinal_thresholds_mps2,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        # video: B C T H W
        B, C, T, H, W = video.shape
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            outputs = self.backbone(video)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        # Normalize each tapped layer independently, then average. This avoids
        # a 4x feature dimension while retaining multi-level temporal features.
        layers = [
            F.layer_norm(x.float(), (x.shape[-1],))
            for x in outputs
        ]
        tokens = torch.stack(layers, dim=0).mean(dim=0)  # B N D

        t_tokens = T // self.tubelet_size
        h_tokens = H // self.patch_size
        w_tokens = W // self.patch_size
        expected = t_tokens * h_tokens * w_tokens
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"unexpected V-JEPA token count {tokens.shape[1]}, expected "
                f"{expected} for input {(T, H, W)}"
            )

        z = tokens.view(B, t_tokens, h_tokens * w_tokens, -1).mean(dim=2)
        # Return to 10-Hz frame resolution. Tubelets represent pairs of source
        # frames; linear interpolation keeps the v1/v2 baseline behavior.
        z = F.interpolate(
            z.transpose(1, 2),
            size=T,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return z

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self._encode(video))

from __future__ import annotations

from collections.abc import Sequence
import math

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
        accel_fusion_enabled: bool = False,
        accel_fusion_hidden: int = 64,
        accel_fusion_gate_init: float = 0.10,
        accel_fusion_detach_ordinal_inputs: bool = True,
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

        self.accel_fusion_enabled = bool(accel_fusion_enabled)
        self.accel_fusion_detach_ordinal_inputs = bool(
            accel_fusion_detach_ordinal_inputs
        )
        if self.accel_fusion_enabled and not thresholds:
            raise ValueError(
                "accel fusion requires non-empty ordinal thresholds"
            )

        if self.accel_fusion_enabled:
            fusion_hidden = max(int(accel_fusion_hidden), 8)
            # raw accel + 2*K probabilities + signed score +
            # threshold-weighted signed score + activity score.
            fusion_input_dim = 1 + 2 * len(thresholds) + 3
            self.accel_fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.GELU(),
                nn.LayerNorm(fusion_hidden),
                nn.Linear(fusion_hidden, 1),
            )
            # Exact warm-start preservation: the new residual starts at zero,
            # so a v3-A checkpoint produces identical scalar acceleration before
            # the first v4-A optimizer step.
            nn.init.zeros_(self.accel_fusion_mlp[-1].weight)
            nn.init.zeros_(self.accel_fusion_mlp[-1].bias)

            gate_init = min(max(float(accel_fusion_gate_init), 1e-4), 1.0 - 1e-4)
            self.accel_fusion_gate_logit = nn.Parameter(
                torch.tensor(math.log(gate_init / (1.0 - gate_init)))
            )
        else:
            self.accel_fusion_mlp = None
            self.register_parameter("accel_fusion_gate_logit", None)

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
            # Thresholds are metadata, not activations.  Keep them in FP32 even
            # under bf16 autocast; otherwise values such as 0.1 become
            # 0.10009765625 and the model/loss contract check falsely fails.
            outputs["accel_ordinal_thresholds_mps2"] = ordinal.new_tensor(
                self.accel_ordinal_thresholds_mps2,
                dtype=torch.float32,
            )

            if self.accel_fusion_mlp is not None:
                raw_accel = outputs["accel_from_speed_mps2"]

                ordinal_probs = torch.sigmoid(ordinal)
                if self.accel_fusion_detach_ordinal_inputs:
                    ordinal_probs_for_fusion = ordinal_probs.detach()
                else:
                    ordinal_probs_for_fusion = ordinal_probs

                decel_prob = ordinal_probs_for_fusion[..., 0]
                accel_prob = ordinal_probs_for_fusion[..., 1]
                signed_score = accel_prob.mean(dim=-1) - decel_prob.mean(dim=-1)
                activity_score = 0.5 * (
                    accel_prob.mean(dim=-1) + decel_prob.mean(dim=-1)
                )

                threshold_tensor = ordinal.new_tensor(
                    self.accel_ordinal_thresholds_mps2,
                    dtype=ordinal_probs_for_fusion.dtype,
                )
                threshold_weight = threshold_tensor / threshold_tensor.sum().clamp_min(1e-6)
                signed_magnitude_score = (
                    (accel_prob - decel_prob) * threshold_weight
                ).sum(dim=-1)

                fusion_input = torch.cat(
                    [
                        raw_accel.unsqueeze(-1),
                        ordinal_probs_for_fusion.flatten(start_dim=2),
                        signed_score.unsqueeze(-1),
                        signed_magnitude_score.unsqueeze(-1),
                        activity_score.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                residual = self.accel_fusion_mlp(fusion_input).squeeze(-1)
                gate = torch.sigmoid(self.accel_fusion_gate_logit).to(
                    dtype=residual.dtype
                )
                fused_accel = raw_accel + gate * residual

                outputs["accel_raw_from_speed_mps2"] = raw_accel
                outputs["accel_fusion_residual"] = residual
                outputs["accel_fusion_gate"] = gate
                outputs["accel_ordinal_signed_score"] = signed_score
                outputs["accel_ordinal_signed_magnitude_score"] = (
                    signed_magnitude_score
                )
                outputs["accel_from_speed_mps2"] = fused_accel

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
        accel_fusion_enabled: bool = False,
        accel_fusion_hidden: int = 64,
        accel_fusion_gate_init: float = 0.10,
        accel_fusion_detach_ordinal_inputs: bool = True,
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
            accel_fusion_enabled=accel_fusion_enabled,
            accel_fusion_hidden=accel_fusion_hidden,
            accel_fusion_gate_init=accel_fusion_gate_init,
            accel_fusion_detach_ordinal_inputs=(
                accel_fusion_detach_ordinal_inputs
            ),
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

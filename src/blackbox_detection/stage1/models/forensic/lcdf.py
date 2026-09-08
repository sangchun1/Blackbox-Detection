"""F5 - LC&DF-inspired dual-stream forensic model (main forensic candidate).

Reference
---------
P. Li, C. Chen, Y. Chernyshova, D. Nikolaev, S. Tan and V. Arlazarov,
"Disentangling Moire and Texture: Towards Robust Display-Recapture Detection
for Document Images", IEEE WIFS 2025.
Official code: ``github.com/chenlewis/LC-DF-For-DPAD``.

Taken from the paper / official code
------------------------------------
* The dual-stream decomposition: a **Local Chromaticity** stream and a
  **Discriminative Frequency** stream, fused into one forensic embedding.
* The LC representation is the per-pixel chromaticity z-score of ``IIC_sort``,
  reused verbatim from :class:`~...forensic.chromaticity.ChromaticityMap`.
* The DF stream input is an **FMAG-processed image**, not a spectrum. Verified
  from ``src/engine/FMAG.py``: the amplitude spectrum is split into moire peak
  bands and the remainder, the remainder is scaled by a learnable coefficient,
  the original phase is restored and the image is inverted back::

      A_new = A * mask_peaks + alpha * A * (1 - mask_peaks)
      img'  = | ifft2( ifftshift( A_new * exp(i * phase) ) ) |

* Automatic peak localisation: local maxima of the log amplitude, thresholded
  at ``0.6`` times the local maximum, DC neighbourhood and border excluded, and
  four-fold symmetric discs drawn at each peak's radial distance.
* ``alpha`` is a learnable, range-clamped parameter, as in
  ``trainer_FMAGgs.py``.
* Adapter-style parameter-efficient training: the pretrained encoders can stay
  frozen while the fusion and head train, which is what
  ``freeze_backbone()`` does here.

Adapted for Stage 1
-------------------
* **Not a reproduction.** The paper fuses the streams inside a Swin-B with
  bi-directional adapters and a frequency-domain moire-aware adapter injected
  per block. Here two independent encoders are fused once, which is far cheaper
  and keeps F3/F4 reusable. Their released "Masked Attention" (Otsu
  foreground/background) is bypassed in the official ``forward`` and is not
  ported.
* Peak localisation uses a max-pool local-maximum filter and the top-``k``
  strongest peaks so the mask can be built on GPU inside the training loop; the
  release calls ``scipy.ndimage.maximum_filter`` on CPU over all peaks.
* No disentanglement loss is added, matching the official training path, which
  uses only a class-balanced binary cross entropy: the disentanglement is
  architectural.
* 256x256 native-resolution video patches instead of resized 224 document
  images, and the old dependency stack of the reference code is not carried
  over; only the architecture idea is ported to this repository's environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..base import (
    ClassifierHead,
    Stage1Model,
    build_patch_encoder,
    encoder_blocks,
)
from .chromaticity import ChromaticityMap, ChromaticityNormalization
from .frequency import FrequencyRepresentation, SpectrumNormalization

DFInput = Literal["fmag", "spectrum"]
FusionMode = Literal["concat_mlp", "gated"]


class MoireAwareFMAG(nn.Module):
    """Frequency-domain moire-aware transform of the LC&DF release.

    Keeps the detected moire peak bands intact and attenuates the remaining
    amplitude by a learnable ``alpha``, then reconstructs an image with the
    original phase. The output is therefore an RGB image in which the periodic
    display-camera signature is relatively enhanced.

    Args:
        alpha: Initial attenuation coefficient of the non-peak amplitude.
        alpha_range: Clamp range of ``alpha``. The released trainer clamps it,
            but the numeric range is not set in the public files; the FMAG
            amplitude range ``(0.5, 0.8)`` is used here and is configurable.
        learnable_alpha: Optimise ``alpha`` jointly with the network.
        filter_size: Max-pool window used as the local-maximum filter.
        relative_threshold: Fraction of the local maximum a peak must reach.
        ring_radius: Radius of the discs drawn around each peak position.
        border_margin: Border width excluded from the peak search.
        min_radius: Minimum radial distance from DC for a peak to count.
        max_peaks: Number of strongest peaks kept per sample.
        eps: Numerical floor of the log magnitude.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.65,
        alpha_range: tuple[float, float] = (0.5, 0.8),
        learnable_alpha: bool = True,
        filter_size: int = 31,
        relative_threshold: float = 0.6,
        ring_radius: int = 10,
        border_margin: int = 15,
        min_radius: int = 5,
        max_peaks: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        low, high = alpha_range
        if not 0.0 <= low <= high:
            raise ValueError(f"Invalid alpha_range: {alpha_range}.")
        if filter_size % 2 == 0:
            raise ValueError(f"filter_size must be odd, got {filter_size}.")
        if max_peaks <= 0:
            raise ValueError(f"max_peaks must be positive, got {max_peaks}.")

        self.alpha_range = (float(low), float(high))
        self.filter_size = int(filter_size)
        self.relative_threshold = float(relative_threshold)
        self.ring_radius = int(ring_radius)
        self.border_margin = int(border_margin)
        self.min_radius = int(min_radius)
        self.max_peaks = int(max_peaks)
        self.eps = float(eps)

        initial = float(min(max(alpha, low), high))
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(initial, dtype=torch.float32))

    @property
    def out_channels(self) -> int:
        return 3

    @torch.no_grad()
    def peak_mask(self, amplitude: torch.Tensor) -> torch.Tensor:
        """Build the moire peak mask for a centred amplitude spectrum.

        Args:
            amplitude: ``(B, C, H, W)`` centred amplitude spectrum.

        Returns:
            ``(B, 1, H, W)`` mask in ``{0, 1}``, shared across channels because
            the release detects peaks on a single channel.
        """
        batch, _, height, width = amplitude.shape
        centre_y, centre_x = height // 2, width // 2

        magnitude = torch.log(amplitude.mean(dim=1, keepdim=True) + self.eps)
        local_max = F.max_pool2d(
            magnitude,
            kernel_size=self.filter_size,
            stride=1,
            padding=self.filter_size // 2,
        )
        candidates = (magnitude >= local_max) & (
            magnitude > self.relative_threshold * local_max
        )

        margin = self.border_margin
        if margin > 0:
            candidates[:, :, :margin, :] = False
            candidates[:, :, -margin:, :] = False
            candidates[:, :, :, :margin] = False
            candidates[:, :, :, -margin:] = False

        # Radial distance of every position from DC, used to reject the DC
        # neighbourhood and to place the symmetric discs.
        ys = torch.arange(height, device=amplitude.device).reshape(1, 1, -1, 1)
        xs = torch.arange(width, device=amplitude.device).reshape(1, 1, 1, -1)
        radius_map = torch.sqrt(
            (ys - centre_y).float() ** 2 + (xs - centre_x).float() ** 2
        )
        candidates &= radius_map > self.min_radius

        # Keep the strongest peaks per sample so the mask can be built on GPU.
        scores = torch.where(candidates, magnitude, torch.full_like(magnitude, -torch.inf))
        flat_scores = scores.reshape(batch, -1)
        top = min(self.max_peaks, flat_scores.shape[1])
        best_values, best_indices = flat_scores.topk(top, dim=1)
        valid = torch.isfinite(best_values)

        flat_radius = radius_map.reshape(1, -1).expand(batch, -1)
        peak_radii = torch.gather(flat_radius, 1, best_indices).round()

        mask = torch.zeros(batch, 1, height, width, device=amplitude.device)
        grid_y = torch.arange(height, device=amplitude.device).reshape(1, -1, 1).float()
        grid_x = torch.arange(width, device=amplitude.device).reshape(1, 1, -1).float()
        squared_ring = float(self.ring_radius) ** 2

        for slot in range(top):
            radii = peak_radii[:, slot].reshape(-1, 1, 1)
            slot_valid = valid[:, slot].reshape(-1, 1, 1)
            for offset_y, offset_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                centre_py = centre_y + offset_y * radii
                centre_px = centre_x + offset_x * radii
                distance = (grid_y - centre_py) ** 2 + (grid_x - centre_px) ** 2
                disc = (distance <= squared_ring) & slot_valid
                mask[:, 0] = torch.maximum(mask[:, 0], disc.float())

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the moire-aware frequency transform.

        Args:
            x: ``(B, 3, H, W)`` linear RGB in ``[0, 1]``.

        Returns:
            ``(B, 3, H, W)`` reconstructed image in ``[0, 1]``.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected a (B, 3, H, W) RGB tensor, got shape {tuple(x.shape)}.")

        spectrum = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
        amplitude = spectrum.abs()
        phase = torch.angle(spectrum)

        mask = self.peak_mask(amplitude)
        alpha = self.alpha.clamp(*self.alpha_range)

        # A_new = A * mask + alpha * A * (1 - mask)
        new_amplitude = amplitude * mask + alpha * amplitude * (1.0 - mask)
        restored = torch.fft.ifft2(
            torch.fft.ifftshift(
                new_amplitude * torch.exp(1j * phase.to(new_amplitude.dtype)),
                dim=(-2, -1),
            ),
            dim=(-2, -1),
        )
        return restored.abs().clamp(0.0, 1.0).to(x.dtype)


class DualStreamFusion(nn.Module):
    """Fuse the chromaticity and frequency features into one embedding.

    Args:
        lc_dim: Chromaticity feature dimension.
        df_dim: Frequency feature dimension.
        out_dim: Fused embedding dimension.
        mode: ``"concat_mlp"`` concatenates and projects; ``"gated"`` projects
            both streams to ``out_dim`` and mixes them with learned per-sample
            gates, which makes the stream contribution inspectable.
        dropout: Dropout inside the fusion MLP.
    """

    def __init__(
        self,
        lc_dim: int,
        df_dim: int,
        *,
        out_dim: int = 512,
        mode: FusionMode = "concat_mlp",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in ("concat_mlp", "gated"):
            raise ValueError(f"mode must be 'concat_mlp' or 'gated', got {mode!r}.")

        self.mode = mode
        self.out_dim = int(out_dim)

        if mode == "concat_mlp":
            self.project = nn.Sequential(
                nn.Linear(lc_dim + df_dim, self.out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.out_dim, self.out_dim),
            )
        else:
            self.lc_project = nn.Linear(lc_dim, self.out_dim)
            self.df_project = nn.Linear(df_dim, self.out_dim)
            self.gate = nn.Sequential(
                nn.Linear(lc_dim + df_dim, self.out_dim * 2),
                nn.Sigmoid(),
            )
            self.dropout = nn.Dropout(dropout)

    def forward(self, lc_features: torch.Tensor, df_features: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([lc_features, df_features], dim=1)
        if self.mode == "concat_mlp":
            return self.project(joint)

        gates = self.gate(joint)
        lc_gate, df_gate = gates.chunk(2, dim=1)
        fused = lc_gate * self.lc_project(lc_features) + df_gate * self.df_project(df_features)
        return self.dropout(fused)


class LCDFDualStreamClassifier(Stage1Model):
    """LC&DF-inspired dual-stream Stage 1 classifier.

    Args:
        num_classes: Output classes.
        chromaticity_normalize: LC representation variant.
        df_input: ``"fmag"`` uses :class:`MoireAwareFMAG` (the paper's DF stream
            input); ``"spectrum"`` reuses F4's log-amplitude representation.
        spectrum_normalize: Spectrum normalisation when ``df_input="spectrum"``.
        encoder: Encoder name for both streams.
        pretrained: Load ImageNet weights into both encoders.
        fusion: Fusion mode.
        fusion_dim: Fused embedding dimension.
        dropout: Dropout in fusion and head.
        patch_size: Expected patch size.
        learnable_alpha: Optimise the FMAG attenuation coefficient.
    """

    model_name: ClassVar[str] = "lcdf"
    input_kind: ClassVar[str] = "patch"

    def __init__(
        self,
        *,
        num_classes: int = 2,
        chromaticity_normalize: ChromaticityNormalization = "lcdf",
        df_input: DFInput = "fmag",
        spectrum_normalize: SpectrumNormalization = "standardize",
        encoder: str = "resnet18",
        pretrained: bool = True,
        fusion: FusionMode = "concat_mlp",
        fusion_dim: int = 512,
        dropout: float = 0.1,
        patch_size: int = 256,
        learnable_alpha: bool = True,
    ) -> None:
        super().__init__()
        if df_input not in ("fmag", "spectrum"):
            raise ValueError(f"df_input must be 'fmag' or 'spectrum', got {df_input!r}.")

        self.patch_size = int(patch_size)
        self.df_input = df_input

        # F3 module, reused as-is.
        self.chromaticity = ChromaticityMap(normalize=chromaticity_normalize)
        self.lc_encoder = build_patch_encoder(
            encoder, in_channels=self.chromaticity.out_channels, pretrained=pretrained
        )

        if df_input == "fmag":
            self.df_representation: nn.Module = MoireAwareFMAG(
                learnable_alpha=learnable_alpha
            )
        else:
            # F4 module, reused as-is.
            self.df_representation = FrequencyRepresentation(
                normalize=spectrum_normalize
            )
        self.df_encoder = build_patch_encoder(
            encoder,
            in_channels=int(self.df_representation.out_channels),
            pretrained=pretrained,
        )

        self.fusion = DualStreamFusion(
            int(self.lc_encoder.out_channels),
            int(self.df_encoder.out_channels),
            out_dim=fusion_dim,
            mode=fusion,
            dropout=dropout,
        )
        self._feature_dim = int(self.fusion.out_dim)
        self.head = ClassifierHead(self._feature_dim, num_classes, dropout=dropout)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def blocks(self) -> Sequence[nn.Module]:
        """Paired blocks of the two streams, shallow to deep.

        Each element groups the ``i``-th block of both encoders, so
        ``unfreeze_last_n_blocks(n)`` unfreezes the last ``n`` stages of both
        streams symmetrically.
        """
        lc_blocks = encoder_blocks(self.lc_encoder)
        df_blocks = encoder_blocks(self.df_encoder)
        return [
            nn.ModuleList([lc_block, df_block])
            for lc_block, df_block in zip(lc_blocks, df_blocks)
        ]

    def head_parameters(self) -> Iterable[nn.Parameter]:
        parameters = list(self.head.parameters()) + list(self.fusion.parameters())
        if isinstance(self.df_representation, MoireAwareFMAG) and isinstance(
            self.df_representation.alpha, nn.Parameter
        ):
            parameters.append(self.df_representation.alpha)
        return parameters

    def backbone_modules(self) -> Iterable[nn.Module]:
        return [self.lc_encoder, self.df_encoder]

    def stream_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return both stream embeddings before fusion, for diagnostics."""
        return {
            "lc": self.lc_encoder(self.chromaticity(x)),
            "df": self.df_encoder(self.df_representation(x)),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        streams = self.stream_features(x)
        return self.fusion(streams["lc"], streams["df"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def preprocessing(self) -> Mapping[str, Any]:
        return {
            "input_kind": "patch",
            "patch_size": self.patch_size,
            "input_range": "unit",
            "representation": (
                f"chromaticity({self.chromaticity.normalize}) + df({self.df_input})"
            ),
            "notes": "Linear RGB in [0, 1]; no resize before cropping.",
        }


__all__ = [
    "DFInput",
    "FusionMode",
    "MoireAwareFMAG",
    "DualStreamFusion",
    "LCDFDualStreamClassifier",
]

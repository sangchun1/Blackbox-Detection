"""F4 - FMAG / M2FM-inspired frequency (moire) branch.

References
----------
* C. Chen, B. Li, R. Cai, J. Zeng and J. Huang, "Distortion Model-Based
  Spectral Augmentation for Generalized Recaptured Document Detection",
  IEEE TIFS 2024. Official code: ``github.com/chenlewis/FHAG-with-BOIL``.
* C. Chen, Y. Li, B. Li, W. Yu, B. Chen, B. Li and J. Huang, "Moire Spectral
  Augmentation and Masked Frequency Modeling for Document Presentation Attack
  Detection", IEEE TDSC 2025. Official code:
  ``github.com/chenlewis/FMAG-with-MMFM``.

Taken from the papers / official code
-------------------------------------
* Screen recapture leaves a periodic display-camera sampling signature, so the
  amplitude spectrum is treated as the discriminative quantity while the phase
  is left untouched.
* The spectrum is computed **per colour channel** with ``fft2`` followed by
  ``fftshift``, so that DC sits at the centre and the moire peaks appear as a
  four-fold symmetric pattern around it (``RGB_fft`` in both releases).
* The FMAG augmentation that relatively enhances those peaks lives in
  :func:`blackbox_detection.stage1.transforms.fmag_spectral_augment`, verified
  against the released ``FMAG.py``.

Adapted for Stage 1
-------------------
* **This is not a reproduction.** In both official releases the network input is
  a spatial RGB patch, and the frequency domain is used only to *manipulate*
  the amplitude and invert back to an image. Feeding the log amplitude spectrum
  directly to an encoder, as done here, is our Stage 1 design choice.
* ``log(1 + |F|)`` compression plus per-sample standardisation is our addition;
  Paper A's optional spectrum input path uses linear magnitude with no
  normalisation, which trains poorly because the dynamic range spans many
  orders of magnitude.
* The masked frequency modelling (M2FM) self-supervised pretraining stage of
  the TDSC paper is **not implemented**. Calling a plain FFT-CNN a reproduction
  of M2FM would be wrong; this module is an FMAG/M2FM-inspired supervised
  frequency branch.
* Patches are 256x256 crops of native-resolution dashcam-domain video frames
  rather than 224 document crops.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import torch
from torch import nn

from ..base import (
    ClassifierHead,
    Stage1Model,
    build_patch_encoder,
    encoder_blocks,
)

SpectrumNormalization = Literal["standardize", "minmax", "none"]


class FrequencyRepresentation(nn.Module):
    """Centred log-amplitude spectrum of an RGB patch.

    Args:
        log_scale: Apply ``log(1 + |F|)`` compression.
        normalize: ``"standardize"`` removes the per-sample per-channel mean and
            divides by the standard deviation, ``"minmax"`` rescales to
            ``[0, 1]``, ``"none"`` leaves the values untouched.
        grayscale: Compute a single-channel spectrum from the luminance instead
            of one spectrum per colour channel. The official releases are
            per-channel, which is the default here.
        eps: Numerical floor of the normalisation.
    """

    def __init__(
        self,
        *,
        log_scale: bool = True,
        normalize: SpectrumNormalization = "standardize",
        grayscale: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if normalize not in ("standardize", "minmax", "none"):
            raise ValueError(
                f"normalize must be 'standardize', 'minmax' or 'none', got {normalize!r}."
            )
        self.log_scale = bool(log_scale)
        self.normalize = normalize
        self.grayscale = bool(grayscale)
        self.eps = float(eps)
        # ITU-R BT.601 luminance weights, used only when grayscale=True.
        self.register_buffer(
            "luma_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    @property
    def out_channels(self) -> int:
        return 1 if self.grayscale else 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the spectrum.

        Args:
            x: ``(B, 3, H, W)`` linear RGB in ``[0, 1]``.

        Returns:
            ``(B, out_channels, H, W)`` spectrum representation.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected a (B, 3, H, W) RGB tensor, got shape {tuple(x.shape)}.")

        signal = (x * self.luma_weights).sum(dim=1, keepdim=True) if self.grayscale else x
        # float32 FFT even under autocast: half-precision FFT is unsupported and
        # would silently lose the low-amplitude high-frequency detail we need.
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(signal.float(), dim=(-2, -1)), dim=(-2, -1)
        )
        amplitude = spectrum.abs()
        if self.log_scale:
            amplitude = torch.log1p(amplitude)

        if self.normalize == "standardize":
            mean = amplitude.mean(dim=(-2, -1), keepdim=True)
            std = amplitude.std(dim=(-2, -1), keepdim=True)
            amplitude = (amplitude - mean) / (std + self.eps)
        elif self.normalize == "minmax":
            flat = amplitude.flatten(2)
            minimum = flat.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
            maximum = flat.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
            amplitude = (amplitude - minimum) / (maximum - minimum + self.eps)

        return amplitude.to(x.dtype)


class FrequencyClassifier(Stage1Model):
    """FMAG/M2FM-inspired frequency-branch Stage 1 classifier.

    Args:
        num_classes: Output classes.
        encoder: Encoder name accepted by :func:`build_patch_encoder`.
            ``"small_cnn"`` is the default because ImageNet features are a weak
            prior for a spectrum.
        pretrained: Load ImageNet weights when a ResNet encoder is used.
        log_scale: Apply log compression to the amplitude.
        normalize: Spectrum normalisation variant.
        grayscale: Use a single luminance spectrum.
        dropout: Head dropout.
        patch_size: Expected patch size.
    """

    model_name: ClassVar[str] = "frequency"
    input_kind: ClassVar[str] = "patch"

    def __init__(
        self,
        *,
        num_classes: int = 2,
        encoder: str = "small_cnn",
        pretrained: bool = False,
        log_scale: bool = True,
        normalize: SpectrumNormalization = "standardize",
        grayscale: bool = False,
        dropout: float = 0.1,
        patch_size: int = 256,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.representation = FrequencyRepresentation(
            log_scale=log_scale, normalize=normalize, grayscale=grayscale
        )
        self.encoder = build_patch_encoder(
            encoder, in_channels=self.representation.out_channels, pretrained=pretrained
        )
        self._feature_dim = int(self.encoder.out_channels)
        self.head = ClassifierHead(self._feature_dim, num_classes, dropout=dropout)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def blocks(self) -> Sequence[nn.Module]:
        return encoder_blocks(self.encoder)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_modules(self) -> Iterable[nn.Module]:
        return [self.encoder]

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spectrum representation, useful for visual diagnostics."""
        return self.representation(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.representation(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def preprocessing(self) -> Mapping[str, Any]:
        return {
            "input_kind": "patch",
            "patch_size": self.patch_size,
            "input_range": "unit",
            "representation": (
                f"log_amplitude(log={self.representation.log_scale}, "
                f"norm={self.representation.normalize})"
            ),
            "notes": "Linear RGB in [0, 1]; no resize before cropping.",
        }


__all__ = [
    "SpectrumNormalization",
    "FrequencyRepresentation",
    "FrequencyClassifier",
]

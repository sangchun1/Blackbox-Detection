"""F3 - CMA-inspired chromaticity branch.

Reference
---------
C. Chen, et al., "CMA: A Chromaticity Map Adapter for Robust Detection of
Screen-Recapture Document Images", CVPR 2024.
Official code: ``github.com/chenlewis/Chromaticity-Map-Adapter-for-DPAD``.
The same chromaticity map is reused by the authors' LC&DF work
(``github.com/chenlewis/LC-DF-For-DPAD``, ``IIC_sort``).

Taken from the paper / official code
------------------------------------
The chromaticity map is **not** invented here. CMA Eq. (5) defines, per pixel
``x`` and channel ``c``::

    C_c(x) = N[ I_c(x) / sum_c I_c(x) ]

that is, normalised RGB (the chromaticity ``sigma_c`` of the inverse-intensity
chromaticity space of Tan et al., JOSA A 2004) followed by a z-score ``N[.]``.
The released implementations of both CMA (``PromptedTransformer.IIC``) and
LC&DF (``IIC_sort``) take that z-score **per pixel across the three
chromaticity channels**::

    total = R + G + B
    c_k   = k / total                       for k in {R, G, B}
    c_k'  = (c_k - mean_k(c)) / std_k(c)

The map has three channels, is computed at native patch resolution, and is
insensitive to document content, which is why it survives the content-domain
gap. LC&DF additionally rescales the map per sample to ``[0, 1]`` and applies
ImageNet normalisation before its encoder; that variant is available as
``normalize="lcdf"``.

Adapted for Stage 1
-------------------
* This is a **standalone forensic branch**, not a reproduction of CMA. CMA
  feeds the map through a linear adapter into VPT prompt tokens of a frozen
  ViT-B/16; here the map goes into one lightweight encoder and a binary head.
  (The public CMA release also leaves its adapter output unused in
  ``incorporate_prompt``, so it does not reproduce the paper's fusion either.)
* ``eps`` is added to the denominators. The CMA release has no epsilon, and
  because ``c_R + c_G + c_B = 1`` by construction the per-pixel mean is exactly
  ``1/3`` and the per-pixel std vanishes on achromatic pixels, which makes the
  unguarded formula blow up on the near-neutral pixels that dominate documents.
* Applied to 256x256 native-resolution video patches instead of 224 document
  crops, and the inputs are linear RGB in ``[0, 1]`` rather than an already
  ImageNet-normalised tensor (the CMA release is internally inconsistent about
  its input range).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import torch
from torch import nn

from ..base import (
    ClassifierHead,
    ImageNetNormalize,
    Stage1Model,
    build_patch_encoder,
    encoder_blocks,
)

ChromaticityNormalization = Literal["zscore", "lcdf", "none"]


class ChromaticityMap(nn.Module):
    """Chromaticity map of CMA Eq. (5).

    Args:
        normalize: ``"zscore"`` is CMA's ``N[.]`` as implemented in the official
            release (per pixel, across the three chromaticity channels).
            ``"lcdf"`` additionally applies LC&DF's per-sample min-max rescale
            to ``[0, 1]`` plus ImageNet normalisation. ``"none"`` returns the
            raw chromaticity triplet.
        eps: Numerical floor for both denominators.
        unbiased_std: Use the unbiased standard deviation, matching
            ``torch.std``'s default in the released implementations.
    """

    def __init__(
        self,
        *,
        normalize: ChromaticityNormalization = "zscore",
        eps: float = 1e-7,
        unbiased_std: bool = True,
    ) -> None:
        super().__init__()
        if normalize not in ("zscore", "lcdf", "none"):
            raise ValueError(
                f"normalize must be 'zscore', 'lcdf' or 'none', got {normalize!r}."
            )
        self.normalize = normalize
        self.eps = float(eps)
        self.unbiased_std = bool(unbiased_std)
        self.imagenet_norm = ImageNetNormalize() if normalize == "lcdf" else None

    @property
    def out_channels(self) -> int:
        return 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the chromaticity map.

        Args:
            x: ``(B, 3, H, W)`` linear RGB in ``[0, 1]``.

        Returns:
            ``(B, 3, H, W)`` chromaticity map.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected a (B, 3, H, W) RGB tensor, got shape {tuple(x.shape)}.")

        total = x.sum(dim=1, keepdim=True) + self.eps
        chromaticity = x / total
        if self.normalize == "none":
            return chromaticity

        mean = chromaticity.mean(dim=1, keepdim=True)
        std = chromaticity.std(dim=1, keepdim=True, unbiased=self.unbiased_std)
        normalised = (chromaticity - mean) / (std + self.eps)
        if self.normalize == "zscore":
            return normalised

        # LC&DF variant: per-sample min-max to [0, 1], then ImageNet statistics.
        flat = normalised.flatten(1)
        minimum = flat.min(dim=1).values.reshape(-1, 1, 1, 1)
        maximum = flat.max(dim=1).values.reshape(-1, 1, 1, 1)
        scaled = (normalised - minimum) / (maximum - minimum + self.eps)
        assert self.imagenet_norm is not None  # narrowed by __init__
        return self.imagenet_norm(scaled)


class ChromaticityClassifier(Stage1Model):
    """CMA-inspired chromaticity-only Stage 1 classifier.

    Args:
        num_classes: Output classes.
        normalize: Chromaticity normalisation variant, see :class:`ChromaticityMap`.
        encoder: Encoder name accepted by :func:`build_patch_encoder`.
        pretrained: Load ImageNet weights into the encoder.
        dropout: Head dropout.
        patch_size: Expected patch size.
    """

    model_name: ClassVar[str] = "chromaticity"
    input_kind: ClassVar[str] = "patch"

    def __init__(
        self,
        *,
        num_classes: int = 2,
        normalize: ChromaticityNormalization = "lcdf",
        encoder: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.1,
        patch_size: int = 256,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.representation = ChromaticityMap(normalize=normalize)
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

    def chromaticity(self, x: torch.Tensor) -> torch.Tensor:
        """Return the chromaticity map, useful for visual diagnostics."""
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
            "representation": f"chromaticity({self.representation.normalize})",
            "notes": "Linear RGB in [0, 1] is required; the map is scale sensitive.",
        }


__all__ = [
    "ChromaticityNormalization",
    "ChromaticityMap",
    "ChromaticityClassifier",
]

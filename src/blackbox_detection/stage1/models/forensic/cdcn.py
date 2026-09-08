"""F2 - Central Difference Convolution binary classifier.

Reference
---------
Z. Yu, C. Zhao, Z. Wang, Y. Qin, Z. Su, X. Li, F. Zhou and G. Zhao,
"Searching Central Difference Convolutional Networks for Face Anti-Spoofing",
CVPR 2020.

Taken from the paper
--------------------
* The Central Difference Convolution operator, which aggregates an intensity
  term and a local gradient term::

      CDC(p0) = theta * sum_pn w(pn) * (x(p0 + pn) - x(p0))
                + (1 - theta) * sum_pn w(pn) * x(p0 + pn)

  implemented, as in the authors' released code, by subtracting a
  ``1 x 1`` convolution with the spatially summed kernel from the ordinary
  convolution output::

      out = conv(x) - theta * conv_1x1(x, weight.sum(dim=(2, 3)))

* ``theta = 0.7`` as the default trade-off between intensity and gradient
  information.
* Stacking CDC layers so the network keeps fine-grained texture detail rather
  than semantic content.

Adapted for Stage 1
-------------------
* The paper's face-specific pseudo-depth supervision and its NAS-searched
  topology are **not** used. Depth maps do not exist for our data and the
  spoofing geometry of a face does not transfer to a dashcam scene.
* Only the CDC operator and the fine-grained feature-extraction idea are kept,
  wired into a compact binary ORIGINAL / RERECORDED classifier over 256x256
  native-resolution patches.
* Multi-scale depth fusion is replaced by global pooling plus a linear head.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from ..base import ClassifierHead, ImageNetNormalize, Stage1Model


class CDCConv2d(nn.Module):
    """Central Difference Convolution.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size; the difference term is only defined for
            kernels larger than 1.
        stride: Convolution stride.
        padding: Convolution padding; defaults to ``kernel_size // 2``.
        dilation: Convolution dilation.
        groups: Convolution groups.
        bias: Whether to use a bias term.
        theta: Weight of the central-difference (gradient) term. ``0`` reduces
            the layer to a plain convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.7,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError(
                f"CDCConv2d needs kernel_size >= 2 for the difference term, got {kernel_size}."
            )
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}.")

        self.theta = float(theta)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        if self.theta == 0.0:
            return out_normal

        # Sum of each kernel's spatial weights, applied as a 1x1 convolution.
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(
            x,
            kernel_diff,
            bias=None,
            stride=self.conv.stride,  # type: ignore[arg-type]
            padding=0,
            groups=self.conv.groups,
        )
        return out_normal - self.theta * out_diff

    def extra_repr(self) -> str:
        return f"theta={self.theta}"


class CDCBlock(nn.Module):
    """Two CDC convolutions with normalisation, then optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        theta: float = 0.7,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            CDCConv2d(in_channels, out_channels, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CDCConv2d(out_channels, out_channels, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.layers(x))


class CDCBinaryClassifier(Stage1Model):
    """CDC-based binary classifier for Stage 1.

    Args:
        num_classes: Output classes.
        theta: Central-difference weight of every CDC layer.
        stem_channels: Channels of the CDC stem.
        widths: Channel width of each CDC stage.
        dropout: Head dropout.
        patch_size: Expected patch size, reported by :meth:`preprocessing`.
        normalize_input: Apply ImageNet normalisation inside the model, since
            forensic transforms deliver linear RGB in ``[0, 1]``.
    """

    model_name: ClassVar[str] = "cdc"
    input_kind: ClassVar[str] = "patch"

    def __init__(
        self,
        *,
        num_classes: int = 2,
        theta: float = 0.7,
        stem_channels: int = 32,
        widths: Sequence[int] = (64, 128, 256),
        dropout: float = 0.1,
        patch_size: int = 256,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.normalize = ImageNetNormalize() if normalize_input else nn.Identity()

        self.stem = nn.Sequential(
            CDCConv2d(3, int(stem_channels), theta=theta),
            nn.BatchNorm2d(int(stem_channels)),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        channels = int(stem_channels)
        for width in widths:
            stages.append(CDCBlock(channels, int(width), theta=theta, downsample=True))
            channels = int(width)
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = channels
        self.head = ClassifierHead(self._feature_dim, num_classes, dropout=dropout)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def blocks(self) -> Sequence[nn.Module]:
        return list(self.stages)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_modules(self) -> Iterable[nn.Module]:
        return [self.stem, self.stages]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(self.normalize(x))
        for stage in self.stages:
            features = stage(features)
        return self.pool(features).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def preprocessing(self) -> Mapping[str, Any]:
        return {
            "input_kind": "patch",
            "patch_size": self.patch_size,
            "input_range": "unit",
            "notes": "Linear RGB in [0, 1]; ImageNet normalisation applied internally.",
        }


__all__ = ["CDCConv2d", "CDCBlock", "CDCBinaryClassifier"]

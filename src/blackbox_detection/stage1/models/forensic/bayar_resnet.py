"""F1 - Bayar constrained convolution with a ResNet18 frontend.

Reference
---------
B. Bayar and M. C. Stamm, "Constrained Convolutional Neural Networks: A New
Approach Towards General Purpose Image Manipulation Detection",
IEEE Transactions on Information Forensics and Security, 2018.

Taken from the paper
--------------------
* The constrained convolution itself: the first layer is forced to compute a
  local prediction residual by constraining every kernel to

  .. code-block:: text

      w(0, 0) = -1
      sum_{(m, n) != (0, 0)} w(m, n) = 1

  which suppresses image content and exposes the local pixel-value
  relationships introduced by processing.
* Placing that constrained layer first, directly on the input, and letting the
  rest of the network operate on the residual.
* Re-imposing the constraint after every gradient update (available here as
  ``constraint_mode="project"``).

Adapted for Stage 1
-------------------
* The paper's MISLnet convolutional stack is replaced by a ResNet18 frontend.
  This is a Stage 1 baseline inspired by the constrained-convolution idea, not
  a reproduction of the paper's architecture.
* Binary ORIGINAL / RERECORDED output on 256x256 native-resolution video
  patches, instead of the paper's manipulation-type classification on image
  patches.
* A BatchNorm layer follows the constrained convolution, because the residual
  has a much smaller dynamic range than the ImageNet statistics the ResNet
  trunk was pretrained with.
* ``constraint_mode="reparam"`` normalises the kernels inside ``forward``, so
  the constraint holds even when a training loop forgets the post-step
  projection. The paper's projection variant is still available.
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

ConstraintMode = Literal["reparam", "project"]


class BayarConv2d(nn.Module):
    """Constrained convolution of Bayar and Stamm (TIFS 2018).

    Args:
        in_channels: Input channels.
        out_channels: Number of constrained filters.
        kernel_size: Odd kernel size; the paper uses 5.
        stride: Convolution stride.
        padding: Convolution padding; defaults to ``kernel_size // 2``.
        constraint_mode: ``"reparam"`` normalises the kernel on every forward
            pass; ``"project"`` keeps the raw parameter and requires
            :meth:`apply_constraint` after each optimiser step, which is the
            paper's procedure.
        eps: Floor for the non-centre weight sum, guarding against division by
            zero when a kernel collapses.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        *,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int | None = None,
        constraint_mode: ConstraintMode = "reparam",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}.")
        if constraint_mode not in ("reparam", "project"):
            raise ValueError(
                f"constraint_mode must be 'reparam' or 'project', got {constraint_mode!r}."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(kernel_size // 2 if padding is None else padding)
        self.constraint_mode = constraint_mode
        self.eps = float(eps)

        self.weight = nn.Parameter(
            torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            * 0.01
        )

        centre_mask = torch.ones(self.kernel_size * self.kernel_size)
        centre_mask[self.kernel_size * self.kernel_size // 2] = 0.0
        self.register_buffer("non_centre_mask", centre_mask.reshape(1, 1, -1))

        centre_onehot = torch.zeros(self.kernel_size * self.kernel_size)
        centre_onehot[self.kernel_size * self.kernel_size // 2] = 1.0
        self.register_buffer("centre_onehot", centre_onehot.reshape(1, 1, -1))

        with torch.no_grad():
            self.weight.copy_(self._constrain(self.weight))

    def _constrain(self, weight: torch.Tensor) -> torch.Tensor:
        """Project a weight tensor onto the Bayar constraint set."""
        flat = weight.reshape(self.out_channels, self.in_channels, -1)
        non_centre = flat * self.non_centre_mask
        total = non_centre.sum(dim=-1, keepdim=True)
        safe_total = torch.where(
            total.abs() < self.eps, torch.full_like(total, self.eps), total
        )
        normalised = non_centre / safe_total
        constrained = normalised * self.non_centre_mask - self.centre_onehot
        return constrained.reshape_as(weight)

    @property
    def constrained_weight(self) -> torch.Tensor:
        """The kernel actually used by :meth:`forward`."""
        if self.constraint_mode == "reparam":
            return self._constrain(self.weight)
        return self.weight

    @torch.no_grad()
    def apply_constraint(self) -> None:
        """Re-impose the constraint on the stored parameter (paper's procedure)."""
        self.weight.copy_(self._constrain(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x,
            self.constrained_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, mode={self.constraint_mode}"
        )


class BayarResNet18Classifier(Stage1Model):
    """Bayar constrained convolution followed by a ResNet trunk.

    Args:
        num_classes: Output classes; Stage 1 uses 2.
        num_bayar_filters: Constrained filters in the first layer. Keeping this
            at 3 lets the ResNet stem reuse its pretrained RGB weights directly.
        kernel_size: Constrained kernel size.
        constraint_mode: See :class:`BayarConv2d`.
        encoder: ``"resnet18"`` by default; any name accepted by
            :func:`build_patch_encoder`.
        pretrained: Load ImageNet weights into the trunk.
        dropout: Head dropout.
        patch_size: Expected input patch size, reported by :meth:`preprocessing`.
    """

    model_name: ClassVar[str] = "bayar_resnet18"
    input_kind: ClassVar[str] = "patch"

    def __init__(
        self,
        *,
        num_classes: int = 2,
        num_bayar_filters: int = 3,
        kernel_size: int = 5,
        constraint_mode: ConstraintMode = "reparam",
        encoder: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.1,
        patch_size: int = 256,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)

        self.bayar = BayarConv2d(
            in_channels=3,
            out_channels=int(num_bayar_filters),
            kernel_size=kernel_size,
            constraint_mode=constraint_mode,
        )
        # Adaptation: the constrained residual has a far smaller dynamic range
        # than natural images, so it is standardised before the ResNet trunk.
        self.residual_norm = nn.BatchNorm2d(int(num_bayar_filters))
        self.encoder = build_patch_encoder(
            encoder, in_channels=int(num_bayar_filters), pretrained=pretrained
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
        return list(self.head.parameters()) + list(self.bayar.parameters()) + list(
            self.residual_norm.parameters()
        )

    def backbone_modules(self) -> Iterable[nn.Module]:
        return [self.encoder]

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the constrained-convolution residual, useful for diagnostics."""
        return self.bayar(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.residual_norm(self.bayar(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def on_after_optimizer_step(self) -> None:
        """Re-project the constrained kernel; a no-op in ``reparam`` mode."""
        if self.bayar.constraint_mode == "project":
            self.bayar.apply_constraint()

    def preprocessing(self) -> Mapping[str, Any]:
        return {
            "input_kind": "patch",
            "patch_size": self.patch_size,
            "input_range": "unit",
            "notes": "Linear RGB in [0, 1]; no resize before cropping.",
        }


__all__ = ["ConstraintMode", "BayarConv2d", "BayarResNet18Classifier"]

"""Shared interface and building blocks for Stage 1 models.

Every Stage 1 model, video or forensic, exposes the same surface so that the
trainer, evaluator, inference API and late fusion never branch on the model
type:

``forward(x) -> (B, num_classes)``
``extract_features(x) -> (B, feature_dim)``
``freeze_backbone()``
``unfreeze_last_n_blocks(n)``
``unfreeze_all()``
``preprocessing() -> Mapping``  (the input contract of the checkpoint)

``input_kind`` says which dataset feeds the model: ``"video"`` for
``Stage1VideoDataset`` clips and ``"patch"`` for ``Stage1ForensicDataset``
native-resolution patches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import torch
from torch import nn

from ..transforms import IMAGENET_MEAN, IMAGENET_STD

InputKind = Literal["video", "patch"]
FinetuneMode = Literal["head_only", "last_n", "full"]


class PretrainedWeightsError(RuntimeError):
    """Raised when pretrained weights cannot be loaded.

    Stage 1 never falls back to random initialisation silently: DLC-2021 is far
    too small to train these backbones from scratch, so a silent fallback would
    produce a quietly worthless run.
    """


def set_requires_grad(module: nn.Module | Iterable[nn.Parameter], flag: bool) -> None:
    """Set ``requires_grad`` on a module's parameters or on a parameter iterable."""
    if isinstance(module, nn.Module):
        for parameter in module.parameters():
            parameter.requires_grad_(flag)
        return
    for parameter in module:
        parameter.requires_grad_(flag)


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


class ImageNetNormalize(nn.Module):
    """Channel normalisation applied inside a model.

    Forensic transforms return linear RGB in ``[0, 1]`` so that chromaticity and
    spectral representations are computed from unmodified pixel values. Models
    that need standardised inputs apply this module themselves.
    """

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ClassifierHead(nn.Module):
    """Dropout plus a linear layer, the shared Stage 1 classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        *,
        dropout: float = 0.0,
        hidden_features: int | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if hidden_features:
            layers += [
                nn.Linear(in_features, hidden_features),
                nn.ReLU(inplace=True),
            ]
            in_features = hidden_features
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Stage1Model(nn.Module, ABC):
    """Base class implementing the shared Stage 1 model interface."""

    model_name: ClassVar[str] = "stage1_model"
    input_kind: ClassVar[InputKind] = "patch"

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension returned by :meth:`extract_features`."""

    @property
    @abstractmethod
    def blocks(self) -> Sequence[nn.Module]:
        """Backbone blocks ordered from input to output.

        Used by :meth:`unfreeze_last_n_blocks`. Return an empty sequence when a
        model has no meaningful block decomposition.
        """

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features of shape ``(B, feature_dim)``."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(B, num_classes)``."""

    @abstractmethod
    def head_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters that stay trainable in ``head_only`` mode."""

    @abstractmethod
    def backbone_modules(self) -> Iterable[nn.Module]:
        """Modules that constitute the pretrained backbone."""

    def preprocessing(self) -> Mapping[str, Any]:
        """Return the input contract of this model.

        Keys are model dependent; video models report ``mean``, ``std``,
        ``num_frames`` and ``input_size``, forensic models report
        ``patch_size`` and ``input_range``.
        """
        return {}

    def freeze_backbone(self) -> None:
        """Freeze every backbone module, leaving the head trainable."""
        for module in self.backbone_modules():
            set_requires_grad(module, False)
        set_requires_grad(self.head_parameters(), True)

    def unfreeze_all(self) -> None:
        """Make every parameter trainable."""
        set_requires_grad(self, True)

    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        """Freeze the backbone, then unfreeze its last ``num_blocks`` blocks.

        Args:
            num_blocks: Number of trailing blocks to unfreeze. ``0`` is
                equivalent to :meth:`freeze_backbone`.

        Raises:
            ValueError: If ``num_blocks`` is negative or exceeds the block count.
            NotImplementedError: If the model exposes no blocks.
        """
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}.")

        blocks = list(self.blocks)
        if not blocks:
            raise NotImplementedError(
                f"{type(self).__name__} does not expose blocks; use freeze_backbone() "
                "or unfreeze_all()."
            )
        if num_blocks > len(blocks):
            raise ValueError(
                f"num_blocks={num_blocks} exceeds the {len(blocks)} available blocks."
            )

        self.freeze_backbone()
        for block in blocks[len(blocks) - num_blocks :]:
            set_requires_grad(block, True)

    def apply_finetune_mode(self, mode: FinetuneMode, *, num_blocks: int = 0) -> None:
        """Apply one of the three supported fine-tuning modes.

        Args:
            mode: ``"head_only"``, ``"last_n"`` or ``"full"``.
            num_blocks: Number of trailing blocks for ``"last_n"``.
        """
        if mode == "head_only":
            self.freeze_backbone()
        elif mode == "last_n":
            if num_blocks <= 0:
                raise ValueError("mode='last_n' requires num_blocks > 0.")
            self.unfreeze_last_n_blocks(num_blocks)
        elif mode == "full":
            self.unfreeze_all()
        else:
            raise ValueError(
                f"Unknown finetune mode {mode!r}; expected 'head_only', 'last_n' or 'full'."
            )


# 2-D patch encoders ----------------------------------------------------------


class SmallPatchEncoder(nn.Module):
    """Compact convolutional encoder for forensic representations.

    Four stride-2 stages of ``Conv-BN-ReLU`` followed by global average
    pooling. Used where an ImageNet-pretrained backbone is a poor prior, for
    example on a frequency-magnitude map, whose statistics have nothing in
    common with natural images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        widths: Sequence[int] = (32, 64, 128, 256),
        norm: bool = True,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            layers: list[nn.Module] = [
                nn.Conv2d(channels, width, kernel_size=3, stride=2, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=not norm)
            )
            if norm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            stages.append(nn.Sequential(*layers))
            channels = width

        # Kept as separate stages, not one Sequential, so that partial
        # unfreezing works here exactly as it does for the ResNet/ViT encoders.
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = int(channels)

    @property
    def blocks(self) -> Sequence[nn.Module]:
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for stage in self.stages:
            features = stage(features)
        return self.pool(features).flatten(1)


def _adapt_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a conv layer for a different input channel count.

    Pretrained RGB weights are reused by averaging them across the input
    dimension and repeating the mean, which preserves the learnt spatial filter
    shapes instead of discarding them.
    """
    if conv.in_channels == in_channels:
        return conv

    adapted = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        dilation=conv.dilation,  # type: ignore[arg-type]
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        mean_weight = conv.weight.mean(dim=1, keepdim=True)
        adapted.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1) * (conv.in_channels / in_channels))
        if conv.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(conv.bias)
    return adapted


class ResNetPatchEncoder(nn.Module):
    """torchvision ResNet trunk returning pooled features.

    Exposes ``blocks`` so that partial unfreezing works the same way as for the
    transformer video backbones.
    """

    def __init__(
        self,
        *,
        arch: str = "resnet18",
        in_channels: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from torchvision import models as tv_models

        builders = {
            "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights),
            "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights),
            "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights),
        }
        if arch not in builders:
            raise ValueError(f"Unsupported ResNet arch {arch!r}; expected one of {sorted(builders)}.")

        builder, weights_enum = builders[arch]
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        backbone = builder(weights=weights)

        backbone.conv1 = _adapt_conv_in_channels(backbone.conv1, in_channels)
        self.out_channels = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    @property
    def blocks(self) -> Sequence[nn.Module]:
        return [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_patch_encoder(
    name: str,
    *,
    in_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Build a 2-D encoder for forensic branches.

    Args:
        name: ``"resnet18"``, ``"resnet34"``, ``"resnet50"`` or ``"small_cnn"``.
        in_channels: Number of input channels of the representation.
        pretrained: Load ImageNet weights (ignored by ``"small_cnn"``).

    Returns:
        A module with an ``out_channels`` attribute whose ``forward`` returns
        pooled ``(B, out_channels)`` features.
    """
    if name == "small_cnn":
        return SmallPatchEncoder(in_channels=in_channels)
    return ResNetPatchEncoder(arch=name, in_channels=in_channels, pretrained=pretrained)


def encoder_blocks(encoder: nn.Module) -> Sequence[nn.Module]:
    """Return an encoder's blocks, or an empty list when it has none."""
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        return []
    return list(blocks)


__all__ = [
    "InputKind",
    "FinetuneMode",
    "PretrainedWeightsError",
    "set_requires_grad",
    "count_parameters",
    "ImageNetNormalize",
    "ClassifierHead",
    "Stage1Model",
    "SmallPatchEncoder",
    "ResNetPatchEncoder",
    "build_patch_encoder",
    "encoder_blocks",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .v5_models import VJEPA21DenseCANV5


@dataclass(frozen=True)
class BackboneFinetuneReport:
    depth: int
    trainable_block_indices: tuple[int, ...]
    trainable_norm_indices: tuple[int, ...]
    trainable_backbone_params: int
    frozen_backbone_params: int


def _numel(parameters: Iterable[nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in parameters)


class VJEPA21DenseCANV5B(VJEPA21DenseCANV5):
    """V5 model with conservative partial V-JEPA fine-tuning.

    V-JEPA 2.1 ViT-B has 12 transformer blocks. V5-B freezes the patch
    embedding and blocks 0..9, and trains only the last N blocks (default 2)
    plus the output norm corresponding to the final hierarchical tap.

    The whole backbone is kept in eval mode except those explicitly trainable
    modules. This avoids changing frozen-block stochastic behavior while still
    allowing gradients through the selected blocks.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Base _encode must run with autograd, so this flag is False. Actual
        # requires_grad control is applied by configure_partial_backbone().
        kwargs["freeze_backbone"] = False
        super().__init__(*args, **kwargs)
        self._partial_ft_configured = False
        self._ft_block_indices: tuple[int, ...] = ()
        self._ft_norm_indices: tuple[int, ...] = ()

    def configure_partial_backbone(
        self,
        *,
        last_n_blocks: int = 2,
        train_final_tap_norm: bool = True,
    ) -> BackboneFinetuneReport:
        backbone = self.backbone
        if not hasattr(backbone, "blocks"):
            raise AttributeError(
                "V-JEPA backbone has no `blocks` ModuleList; "
                "cannot configure partial fine-tuning."
            )

        blocks = backbone.blocks
        depth = len(blocks)
        n = int(last_n_blocks)
        if n < 1 or n > depth:
            raise ValueError(
                f"last_n_blocks must be in [1, {depth}], got {n}"
            )

        # Freeze everything first, then opt in only the intended modules.
        backbone.requires_grad_(False)

        block_indices = tuple(range(depth - n, depth))
        for idx in block_indices:
            blocks[idx].requires_grad_(True)

        norm_indices: tuple[int, ...] = ()
        if train_final_tap_norm:
            if not hasattr(backbone, "norms_block"):
                raise AttributeError(
                    "V-JEPA backbone has no `norms_block`; "
                    "expected V-JEPA 2.1 VisionTransformer."
                )
            norms = backbone.norms_block
            if len(norms) < 1:
                raise RuntimeError("backbone.norms_block is empty")

            # With out_layers=[2,5,8,11], norms_block[-1] normalizes the
            # representation emitted after block 11. Since only the final tap
            # is affected by blocks 10/11, adapt only this norm.
            norm_idx = len(norms) - 1
            norms[norm_idx].requires_grad_(True)
            norm_indices = (norm_idx,)

        self._ft_block_indices = block_indices
        self._ft_norm_indices = norm_indices
        self._partial_ft_configured = True

        trainable = [p for p in backbone.parameters() if p.requires_grad]
        frozen = [p for p in backbone.parameters() if not p.requires_grad]

        # Put modules into the exact intended mode immediately.
        self.train(self.training)

        return BackboneFinetuneReport(
            depth=depth,
            trainable_block_indices=block_indices,
            trainable_norm_indices=norm_indices,
            trainable_backbone_params=_numel(trainable),
            frozen_backbone_params=_numel(frozen),
        )

    def train(self, mode: bool = True):
        # Use nn.Module.train directly via the parent implementation. The v5
        # parent does not force backbone.eval because freeze_backbone=False.
        super().train(mode)

        if self._partial_ft_configured:
            # Frozen V-JEPA modules stay deterministic.
            self.backbone.eval()

            if mode:
                for idx in self._ft_block_indices:
                    self.backbone.blocks[idx].train(True)
                for idx in self._ft_norm_indices:
                    self.backbone.norms_block[idx].train(True)

        return self

    def trainable_backbone_named_parameters(
        self,
    ) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, p)
            for name, p in self.backbone.named_parameters()
            if p.requires_grad
        ]


def split_v5b_optimizer_parameters(
    model: VJEPA21DenseCANV5B,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Return named parameter families for discriminative LR.

    Families:
      - head: everything outside the V-JEPA backbone
      - backbone_penultimate: second-to-last unfrozen transformer block
      - backbone_last: last transformer block + final tap norm
    """
    if not model._partial_ft_configured:
        raise RuntimeError(
            "configure_partial_backbone() must be called first"
        )

    indices = model._ft_block_indices
    if len(indices) != 2:
        raise ValueError(
            "v5-B optimizer policy expects exactly 2 unfrozen blocks; "
            f"got {indices}"
        )
    penultimate_idx, last_idx = indices

    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "head": [],
        "backbone_penultimate": [],
        "backbone_last": [],
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if not name.startswith("backbone."):
            groups["head"].append((name, p))
            continue

        penultimate_prefix = f"backbone.blocks.{penultimate_idx}."
        last_prefix = f"backbone.blocks.{last_idx}."
        final_norm_prefixes = tuple(
            f"backbone.norms_block.{idx}."
            for idx in model._ft_norm_indices
        )

        if name.startswith(penultimate_prefix):
            groups["backbone_penultimate"].append((name, p))
        elif name.startswith(last_prefix) or name.startswith(
            final_norm_prefixes
        ):
            groups["backbone_last"].append((name, p))
        else:
            raise RuntimeError(
                "Unexpected trainable backbone parameter outside intended "
                f"V5-B modules: {name}"
            )

    for key, items in groups.items():
        if not items:
            raise RuntimeError(f"optimizer family {key!r} is empty")

    return groups


__all__ = [
    "BackboneFinetuneReport",
    "VJEPA21DenseCANV5B",
    "split_v5b_optimizer_parameters",
]

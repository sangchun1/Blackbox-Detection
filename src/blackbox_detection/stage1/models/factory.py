"""Factory for Stage 1 models.

Notebooks and the inference API refer to models by name only, so no
model-specific construction code leaks into them and a new model becomes
available everywhere by registering it here.

======================  ===========================================  ==========
``model_name``          Class                                        Input kind
======================  ===========================================  ==========
``videomaev2_b``        :class:`~.videomaev2.VideoMAEv2Classifier`    video
``vjepa2_1_b``          :class:`~.vjepa2.VJEPA21Classifier`           video
``bayar_resnet18``      :class:`~.forensic.BayarResNet18Classifier`   patch
``cdc``                 :class:`~.forensic.CDCBinaryClassifier`       patch
``chromaticity``        :class:`~.forensic.ChromaticityClassifier`    patch
``frequency``           :class:`~.forensic.FrequencyClassifier`       patch
``lcdf``                :class:`~.forensic.LCDFDualStreamClassifier`  patch
======================  ===========================================  ==========
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import FinetuneMode, InputKind, Stage1Model
from .forensic import (
    BayarResNet18Classifier,
    CDCBinaryClassifier,
    ChromaticityClassifier,
    FrequencyClassifier,
    LCDFDualStreamClassifier,
)
from .videomaev2 import VideoMAEv2Classifier, VideoMAEv2Config
from .vjepa2 import VJEPA21Classifier, VJEPA21Config

VIDEO_MODEL_NAMES: tuple[str, ...] = ("videomaev2_b", "vjepa2_1_b")
FORENSIC_MODEL_NAMES: tuple[str, ...] = (
    "bayar_resnet18",
    "cdc",
    "chromaticity",
    "frequency",
    "lcdf",
)
STAGE1_MODEL_NAMES: tuple[str, ...] = (*VIDEO_MODEL_NAMES, *FORENSIC_MODEL_NAMES)

MODEL_INPUT_KINDS: Mapping[str, InputKind] = {
    **{name: "video" for name in VIDEO_MODEL_NAMES},
    **{name: "patch" for name in FORENSIC_MODEL_NAMES},
}

# Recommended experiment order. The implementation numbering (F1..F5) and the
# execution order differ on purpose: the screen-recapture-specific branches
# F3/F4/F5 are evaluated before the more generic F2.
RECOMMENDED_ORDER: tuple[str, ...] = (
    "videomaev2_b",
    "bayar_resnet18",
    "chromaticity",
    "frequency",
    "lcdf",
    "cdc",
    "vjepa2_1_b",
)


def _build_videomaev2(**kwargs: Any) -> Stage1Model:
    return VideoMAEv2Classifier(VideoMAEv2Config(**kwargs))


def _build_vjepa2_1(**kwargs: Any) -> Stage1Model:
    return VJEPA21Classifier(VJEPA21Config(**kwargs))


_BUILDERS: Mapping[str, Callable[..., Stage1Model]] = {
    "videomaev2_b": _build_videomaev2,
    "vjepa2_1_b": _build_vjepa2_1,
    "bayar_resnet18": BayarResNet18Classifier,
    "cdc": CDCBinaryClassifier,
    "chromaticity": ChromaticityClassifier,
    "frequency": FrequencyClassifier,
    "lcdf": LCDFDualStreamClassifier,
}


def model_input_kind(model_name: str) -> InputKind:
    """Return which Stage 1 dataset feeds ``model_name``."""
    try:
        return MODEL_INPUT_KINDS[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown Stage 1 model {model_name!r}. Available: {list(STAGE1_MODEL_NAMES)}"
        ) from error


def build_stage1_model(
    model_name: str,
    *,
    finetune_mode: FinetuneMode | None = None,
    unfreeze_last_n: int = 0,
    **kwargs: Any,
) -> Stage1Model:
    """Build a Stage 1 model by name.

    Args:
        model_name: One of :data:`STAGE1_MODEL_NAMES`.
        finetune_mode: Optionally apply ``"head_only"``, ``"last_n"`` or
            ``"full"`` right after construction.
        unfreeze_last_n: Number of trailing blocks for ``finetune_mode="last_n"``.
        **kwargs: Forwarded to the model or its config dataclass.

    Returns:
        The constructed model.

    Raises:
        ValueError: On an unknown model name or invalid keyword arguments.
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown Stage 1 model {model_name!r}. Available: {list(STAGE1_MODEL_NAMES)}"
        )

    try:
        model = _BUILDERS[model_name](**kwargs)
    except TypeError as error:
        raise ValueError(
            f"Invalid arguments for Stage 1 model {model_name!r}: {error}"
        ) from error

    if finetune_mode is not None:
        model.apply_finetune_mode(finetune_mode, num_blocks=unfreeze_last_n)
    return model


def build_from_config(config: Mapping[str, Any]) -> Stage1Model:
    """Build a model from a config fragment.

    Expected shape::

        model:
          name: lcdf
          finetune_mode: head_only
          unfreeze_last_n: 0
          params:
            encoder: resnet18
            pretrained: true

    Args:
        config: Mapping with ``name`` and optional ``params``,
            ``finetune_mode``, ``unfreeze_last_n``.

    Returns:
        The constructed model.
    """
    if "name" not in config:
        raise ValueError("Model config must contain a 'name' key.")

    unknown = set(config) - {"name", "params", "finetune_mode", "unfreeze_last_n"}
    if unknown:
        raise ValueError(f"Unknown model config keys: {sorted(unknown)}")

    return build_stage1_model(
        str(config["name"]),
        finetune_mode=config.get("finetune_mode"),
        unfreeze_last_n=int(config.get("unfreeze_last_n", 0) or 0),
        **dict(config.get("params") or {}),
    )


__all__ = [
    "VIDEO_MODEL_NAMES",
    "FORENSIC_MODEL_NAMES",
    "STAGE1_MODEL_NAMES",
    "MODEL_INPUT_KINDS",
    "RECOMMENDED_ORDER",
    "model_input_kind",
    "build_stage1_model",
    "build_from_config",
]

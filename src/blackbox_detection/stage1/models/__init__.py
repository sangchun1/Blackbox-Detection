"""Stage 1 models: video branch, forensic branch, and the shared factory."""

from __future__ import annotations

from .base import (
    ClassifierHead,
    FinetuneMode,
    ImageNetNormalize,
    InputKind,
    PretrainedWeightsError,
    ResNetPatchEncoder,
    SmallPatchEncoder,
    Stage1Model,
    build_patch_encoder,
    count_parameters,
    set_requires_grad,
)
from .factory import (
    FORENSIC_MODEL_NAMES,
    MODEL_INPUT_KINDS,
    RECOMMENDED_ORDER,
    STAGE1_MODEL_NAMES,
    VIDEO_MODEL_NAMES,
    build_from_config,
    build_stage1_model,
    model_input_kind,
)
from .forensic import (
    BayarConv2d,
    BayarResNet18Classifier,
    CDCBinaryClassifier,
    CDCConv2d,
    ChromaticityClassifier,
    ChromaticityMap,
    DualStreamFusion,
    FrequencyClassifier,
    FrequencyRepresentation,
    LCDFDualStreamClassifier,
    MoireAwareFMAG,
)
from .videomaev2 import VideoMAEv2Classifier, VideoMAEv2Config
from .vjepa2 import AttentiveProbeHead, VJEPA21Classifier, VJEPA21Config

__all__ = [
    # Interface and helpers
    "Stage1Model",
    "InputKind",
    "FinetuneMode",
    "PretrainedWeightsError",
    "ClassifierHead",
    "ImageNetNormalize",
    "SmallPatchEncoder",
    "ResNetPatchEncoder",
    "build_patch_encoder",
    "set_requires_grad",
    "count_parameters",
    # Factory
    "STAGE1_MODEL_NAMES",
    "VIDEO_MODEL_NAMES",
    "FORENSIC_MODEL_NAMES",
    "MODEL_INPUT_KINDS",
    "RECOMMENDED_ORDER",
    "build_stage1_model",
    "build_from_config",
    "model_input_kind",
    # Video branch
    "VideoMAEv2Config",
    "VideoMAEv2Classifier",
    "VJEPA21Config",
    "VJEPA21Classifier",
    "AttentiveProbeHead",
    # Forensic branch
    "BayarConv2d",
    "BayarResNet18Classifier",
    "CDCConv2d",
    "CDCBinaryClassifier",
    "ChromaticityMap",
    "ChromaticityClassifier",
    "FrequencyRepresentation",
    "FrequencyClassifier",
    "MoireAwareFMAG",
    "DualStreamFusion",
    "LCDFDualStreamClassifier",
]

"""Forensic Stage 1 branches operating on native-resolution patches.

Each module documents which parts come from its reference paper and which parts
are our Stage 1 adaptation.

======  ==================================  ====================================
Key     Model                               Reference
======  ==================================  ====================================
F1      :class:`BayarResNet18Classifier`    Bayar and Stamm, IEEE TIFS 2018
F2      :class:`CDCBinaryClassifier`        Yu et al., CVPR 2020
F3      :class:`ChromaticityClassifier`     Chen et al., CVPR 2024 (CMA)
F4      :class:`FrequencyClassifier`        Chen et al., TIFS 2024 / TDSC 2025
F5      :class:`LCDFDualStreamClassifier`   Li et al., IEEE WIFS 2025 (LC&DF)
======  ==================================  ====================================
"""

from __future__ import annotations

from .bayar_resnet import BayarConv2d, BayarResNet18Classifier
from .cdcn import CDCBinaryClassifier, CDCBlock, CDCConv2d
from .chromaticity import ChromaticityClassifier, ChromaticityMap
from .frequency import FrequencyClassifier, FrequencyRepresentation
from .lcdf import (
    DualStreamFusion,
    LCDFDualStreamClassifier,
    MoireAwareFMAG,
)

__all__ = [
    "BayarConv2d",
    "BayarResNet18Classifier",
    "CDCConv2d",
    "CDCBlock",
    "CDCBinaryClassifier",
    "ChromaticityMap",
    "ChromaticityClassifier",
    "FrequencyRepresentation",
    "FrequencyClassifier",
    "MoireAwareFMAG",
    "DualStreamFusion",
    "LCDFDualStreamClassifier",
]

"""V2 - V-JEPA 2.1-B video branch.

Checkpoint and architecture
---------------------------
V-JEPA 2.1 (released 2026-03-16) is the only V-JEPA 2.x generation that
includes a Base encoder. The official ViT-B/16 checkpoint is
``vjepa2_1_vitb_dist_vitG_384.pt`` (80M parameters, distilled from ViT-G,
``checkpoint_key="ema_encoder"``), and ``src/hub/backbones.py`` builds it as

.. code-block:: python

    vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

``transformers`` cannot load V-JEPA 2.1: its ``VJEPA2Config`` has no
``interpolate_rope`` / ``img_temporal_dim_size`` and the 2.1 encoder lives in a
different module tree (``app/vjepa_2_1/models/vision_transformer.py``). There is
no official ``facebook/vjepa2.1-*`` repository and no community ViT-B
conversion. This module therefore builds the encoder from a **local clone of
the official repository** (``facebookresearch/vjepa2``) and loads the official
``.pt`` from disk.

Design decisions this forces, all deliberate:

* ``source_root`` must point at that clone. There is no vendored copy of Meta's
  ViT, so nothing here can drift away from the official implementation.
* ``pretrained_path`` (a local file) is the default weight source, because the
  DACON submission environment may be offline. Downloading is opt-in.
  The upstream ``torch.hub`` path is additionally broken at the time of
  writing: ``src/hub/backbones.py`` ships with
  ``VJEPA_BASE_URL = "http://localhost:8300"`` left over from testing.
* Loading uses ``strict=True``, matching the official 2.1 loader, and any
  failure raises :class:`~.base.PretrainedWeightsError`. There is no silent
  fallback to random weights.

Higher spatial resolution is this backbone's advantage, so 384 is the default
input geometry. RoPE with ``interpolate_rope=True`` makes the frame count
flexible, so ``input_frames`` may be smaller than the 64 the checkpoint was
built with; both are configurable rather than hard-coded.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from torch import nn

from .base import ClassifierHead, PretrainedWeightsError, Stage1Model
from ..transforms import IMAGENET_MEAN, IMAGENET_STD

PoolingMode = Literal["attentive", "mean"]

VJEPA21_MODULE_PATH = "app.vjepa_2_1.models.vision_transformer"
VJEPA21_BASE_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
)
VJEPA21_BASE_CHECKPOINT_KEY = "ema_encoder"

# Encoder keyword arguments of ``vjepa2_1_vit_base_384`` in the official
# ``src/hub/backbones.py``. Only geometry is overridden from the config.
VJEPA21_ENCODER_KWARGS: Mapping[str, Any] = {
    "patch_size": 16,
    "tubelet_size": 2,
    "use_sdpa": True,
    "use_SiLU": False,
    "wide_SiLU": True,
    "uniform_power": False,
    "use_rope": True,
    "img_temporal_dim_size": 1,
    "interpolate_rope": True,
}


@dataclass
class VJEPA21Config:
    """Configuration of the V-JEPA 2.1-B branch.

    Attributes:
        source_root: Local clone of ``facebookresearch/vjepa2``, needed to build
            the official encoder.
        arch: Encoder factory name in the official module; ``"vit_base"`` is
            V-JEPA 2.1-B.
        checkpoint_path: Local official ``.pt`` file.
        checkpoint_url: Download URL used only when ``allow_download=True``.
        checkpoint_key: Key holding the encoder weights; ``"ema_encoder"`` for
            the distilled B/L checkpoints.
        allow_download: Permit fetching the checkpoint over the network.
        pretrained: Load pretrained weights at all.
        allow_random_init: Explicit opt-in to random initialisation.
        build_img_size: Spatial geometry the checkpoint was built with.
        build_num_frames: Frame count the checkpoint was built with.
        input_size: Runtime crop size.
        input_frames: Runtime frame count; RoPE interpolation allows fewer
            frames than ``build_num_frames``.
        pooling: ``"attentive"`` for an attentive probe, ``"mean"`` for mean
            pooling of the patch tokens.
        num_classes: Output classes.
        dropout: Head dropout.
        probe_num_heads: Attention heads of the attentive probe.
        strict_load: Use ``strict=True`` when loading the encoder weights.
        mean: Normalisation mean; the official processor uses ImageNet values.
        std: Normalisation std.
        extra_encoder_kwargs: Additional encoder keyword arguments.
    """

    source_root: str | Path | None = None
    arch: str = "vit_base"
    checkpoint_path: str | Path | None = None
    checkpoint_url: str = VJEPA21_BASE_CHECKPOINT_URL
    checkpoint_key: str = VJEPA21_BASE_CHECKPOINT_KEY
    allow_download: bool = False
    pretrained: bool = True
    allow_random_init: bool = False
    build_img_size: int = 384
    build_num_frames: int = 64
    input_size: int = 384
    input_frames: int = 16
    pooling: PoolingMode = "attentive"
    num_classes: int = 2
    dropout: float = 0.0
    probe_num_heads: int = 12
    strict_load: bool = True
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    extra_encoder_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pooling not in ("attentive", "mean"):
            raise ValueError(f"pooling must be 'attentive' or 'mean', got {self.pooling!r}.")
        if self.input_frames <= 0 or self.input_frames % 2 != 0:
            raise ValueError(
                f"input_frames must be a positive even number (tubelet_size=2), "
                f"got {self.input_frames}."
            )
        if self.input_size % 16 != 0:
            raise ValueError(
                f"input_size must be a multiple of the patch size 16, got {self.input_size}."
            )
        if not self.pretrained and not self.allow_random_init:
            raise PretrainedWeightsError(
                "pretrained=False would train V-JEPA 2.1-B from scratch. Set "
                "allow_random_init=True if this is really intended."
            )


def _import_official_module(source_root: str | Path | None) -> Any:
    """Import the official V-JEPA 2.1 vision transformer module.

    Args:
        source_root: Local clone of ``facebookresearch/vjepa2``. When ``None``,
            the module is expected to be importable already.

    Raises:
        PretrainedWeightsError: If the module cannot be imported.
    """
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            raise PretrainedWeightsError(
                f"V-JEPA 2.1 source_root does not exist: {root}. Clone "
                "https://github.com/facebookresearch/vjepa2 and point "
                "source_root at it."
            )
        if not (root / "app" / "vjepa_2_1").is_dir():
            raise PretrainedWeightsError(
                f"{root} does not look like a facebookresearch/vjepa2 clone: "
                "'app/vjepa_2_1' is missing."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    try:
        return importlib.import_module(VJEPA21_MODULE_PATH)
    except Exception as error:  # noqa: BLE001
        raise PretrainedWeightsError(
            f"Could not import {VJEPA21_MODULE_PATH!r}. V-JEPA 2.1 is not "
            "supported by transformers, so a local clone of "
            "facebookresearch/vjepa2 is required; pass its path as "
            f"source_root. Original error: {error}"
        ) from error


def _clean_encoder_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the ``module.``/``backbone.`` prefixes, as the official loader does."""
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        cleaned[new_key] = value
    return cleaned


def load_vjepa21_encoder(config: VJEPA21Config) -> tuple[nn.Module, dict[str, Any]]:
    """Build the official V-JEPA 2.1 encoder and load its weights.

    Returns:
        The encoder and a load report.

    Raises:
        PretrainedWeightsError: On any build or load failure.
    """
    module = _import_official_module(config.source_root)
    factory = getattr(module, config.arch, None)
    if factory is None:
        raise PretrainedWeightsError(
            f"{VJEPA21_MODULE_PATH} has no architecture {config.arch!r}."
        )

    encoder_kwargs: dict[str, Any] = {
        **VJEPA21_ENCODER_KWARGS,
        "img_size": (config.build_img_size, config.build_img_size),
        "num_frames": config.build_num_frames,
        **config.extra_encoder_kwargs,
    }
    encoder = factory(**encoder_kwargs)

    report: dict[str, Any] = {
        "arch": config.arch,
        "encoder_kwargs": {
            key: value for key, value in encoder_kwargs.items() if key != "norm_layer"
        },
    }

    if not config.pretrained:
        report["weights"] = {"source": "random init (explicitly allowed)"}
        return encoder, report

    if config.checkpoint_path is not None:
        checkpoint_path = Path(config.checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise PretrainedWeightsError(
                f"V-JEPA 2.1 checkpoint not found: {checkpoint_path}"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = str(checkpoint_path)
    elif config.allow_download:
        payload = torch.hub.load_state_dict_from_url(
            config.checkpoint_url, map_location="cpu"
        )
        source = config.checkpoint_url
    else:
        raise PretrainedWeightsError(
            "No V-JEPA 2.1 weights available: set checkpoint_path to a local "
            f"copy of {config.checkpoint_url} (recommended, and required for an "
            "offline submission environment) or set allow_download=True."
        )

    if isinstance(payload, Mapping) and config.checkpoint_key in payload:
        state_dict = payload[config.checkpoint_key]
    elif isinstance(payload, Mapping) and "encoder" in payload:
        state_dict = payload["encoder"]
    elif isinstance(payload, Mapping) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state_dict = payload
    else:
        available = sorted(payload.keys()) if isinstance(payload, Mapping) else []
        raise PretrainedWeightsError(
            f"Checkpoint {source} has no key {config.checkpoint_key!r}. "
            f"Available top-level keys: {available}"
        )

    cleaned = _clean_encoder_state_dict(state_dict)
    try:
        incompatible = encoder.load_state_dict(cleaned, strict=config.strict_load)
    except RuntimeError as error:
        raise PretrainedWeightsError(
            f"Failed to load {source} into {config.arch}: {error}"
        ) from error

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing and not config.strict_load:
        # Report but do not hide a partial load.
        report["warning"] = (
            f"{len(missing)} parameter(s) were not present in the checkpoint."
        )

    report["weights"] = {
        "source": source,
        "checkpoint_key": config.checkpoint_key,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return encoder, report


class AttentiveProbeHead(nn.Module):
    """Single-query attentive probe over patch tokens.

    Adapted from Meta's ``AttentiveClassifier`` (``src/models/attentive_pooler.py``),
    whose probe-based protocol is the reference way to evaluate frozen V-JEPA
    features. Reimplemented here as one learned query attending over the token
    sequence, which keeps this repository free of a hard dependency on the
    official package at training time.

    Args:
        embed_dim: Token dimension.
        num_classes: Output classes.
        num_heads: Attention heads.
        dropout: Dropout before the linear classifier.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 2,
        *,
        num_heads: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = ClassifierHead(embed_dim, num_classes, dropout=dropout)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the pooled ``(B, embed_dim)`` embedding."""
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(tokens))


class VJEPA21Classifier(Stage1Model):
    """V-JEPA 2.1-B encoder with an attentive probe or mean-pool head.

    Args:
        config: Branch configuration; defaults to :class:`VJEPA21Config`.
    """

    model_name: ClassVar[str] = "vjepa2_1_b"
    input_kind: ClassVar[str] = "video"

    def __init__(self, config: VJEPA21Config | None = None) -> None:
        super().__init__()
        self.config = config or VJEPA21Config()
        self.encoder, self.load_report = load_vjepa21_encoder(self.config)

        embed_dim = int(getattr(self.encoder, "embed_dim", 0))
        if embed_dim <= 0:
            raise PretrainedWeightsError(
                "Could not determine the V-JEPA 2.1 encoder embedding dimension."
            )
        self._feature_dim = embed_dim

        if self.config.pooling == "attentive":
            self.probe: nn.Module = AttentiveProbeHead(
                embed_dim,
                self.config.num_classes,
                num_heads=self.config.probe_num_heads,
                dropout=self.config.dropout,
            )
        else:
            self.probe = ClassifierHead(
                embed_dim, self.config.num_classes, dropout=self.config.dropout
            )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def blocks(self) -> Sequence[nn.Module]:
        blocks = getattr(self.encoder, "blocks", None)
        return list(blocks) if blocks is not None else []

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.probe.parameters())

    def backbone_modules(self) -> Iterable[nn.Module]:
        return [self.encoder]

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return ``(B, num_tokens, embed_dim)``."""
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(
                "Expected a (B, 3, T, H, W) clip tensor, got shape "
                f"{tuple(x.shape)}."
            )

        tokens = self.encoder(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(
                f"The V-JEPA 2.1 encoder returned {type(tokens).__name__}, "
                "expected a tensor of patch tokens."
            )
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected (B, N, D) tokens, got shape {tuple(tokens.shape)}."
            )
        return tokens

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)
        if isinstance(self.probe, AttentiveProbeHead):
            return self.probe.pool(tokens)
        return tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)
        if isinstance(self.probe, AttentiveProbeHead):
            return self.probe(tokens)
        return self.probe(tokens.mean(dim=1))

    def preprocessing(self) -> Mapping[str, Any]:
        return {
            "input_kind": "video",
            "num_frames": self.config.input_frames,
            "input_size": self.config.input_size,
            "mean": tuple(self.config.mean),
            "std": tuple(self.config.std),
            "channels_first": True,
            "source": self.config.checkpoint_url,
            "notes": (
                "Official V-JEPA 2.1 preprocessing: shorter side resized to "
                "crop_size * 256 / 224, centre crop, ImageNet normalisation. "
                "The released 2.1 encoders are 384-resolution."
            ),
        }


def build_vjepa2_1_b(**kwargs: Any) -> VJEPA21Classifier:
    """Build the V-JEPA 2.1-B branch from keyword arguments."""
    return VJEPA21Classifier(VJEPA21Config(**kwargs))


__all__ = [
    "VJEPA21_MODULE_PATH",
    "VJEPA21_BASE_CHECKPOINT_URL",
    "VJEPA21_BASE_CHECKPOINT_KEY",
    "VJEPA21_ENCODER_KWARGS",
    "PoolingMode",
    "VJEPA21Config",
    "AttentiveProbeHead",
    "VJEPA21Classifier",
    "load_vjepa21_encoder",
    "build_vjepa2_1_b",
]

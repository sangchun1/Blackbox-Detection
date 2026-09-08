"""V1 - VideoMAEv2-B video branch.

Checkpoint
----------
``transformers`` has no native VideoMAEv2 model, so the official
``OpenGVLab/VideoMAEv2-Base`` repository is used through
``trust_remote_code=True``. Its ``config.json`` declares
``patch_size=16, embed_dim=768, depth=12, num_heads=12, tubelet_size=2,
num_frames=16, img_size=224, use_mean_pooling=True, num_classes=0``, and the
matching ``preprocessor_config.json`` normalises with ImageNet statistics
(``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``). Those values,
not hand-picked ones, are what this module uses.

Two important details of that repository, both handled here:

* the remote code expects ``(B, C, T, H, W)`` while
  ``VideoMAEImageProcessor`` emits ``(B, T, C, H, W)``. Our
  ``Stage1VideoDataset`` already produces channel-first clips, which is what
  this wrapper feeds the model;
* the bare ``VideoMAEImageProcessor()`` class default is ``mean=std=0.5``,
  which is *not* what the checkpoint wants. The processor is therefore always
  built with ``from_pretrained`` and its statistics are asserted.

A local ``.pth`` from the official ``OpenGVLab/VideoMAE2`` weight release (for
example the distilled ``vit_b_k710_dl_from_giant.pth``) can be loaded on top of
the architecture with ``pretrained_path``. Key prefixes are reconciled
automatically and the match is reported; a failed load raises
:class:`~..base.PretrainedWeightsError` instead of leaving random weights in
place.

Sampling and augmentation live in :mod:`...sampling` and :mod:`...transforms`:
training uses a random temporal start with a stride drawn from ``{1, 2, 4}``,
validation a deterministic clip, and every augmentation parameter is drawn once
per clip.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from torch import nn

from .base import ClassifierHead, PretrainedWeightsError, Stage1Model

VideoMAEBackend = Literal["hf_remote", "hf_videomae"]

DEFAULT_VIDEOMAEV2_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"


@dataclass
class VideoMAEv2Config:
    """Configuration of the VideoMAEv2-B branch.

    Attributes:
        hf_model_id: Hugging Face repository providing the architecture (and
            the weights, unless ``pretrained_path`` is set).
        backend: ``"hf_remote"`` uses the OpenGVLab remote code and channel-first
            clips; ``"hf_videomae"`` uses the native ``transformers`` VideoMAE
            implementation and frame-first clips.
        pretrained_path: Optional local checkpoint or raw ``state_dict``.
        pretrained: Load pretrained weights at all. ``False`` is only valid
            together with ``allow_random_init=True``.
        allow_random_init: Explicit opt-in to random initialisation. Guards
            against a silently untrained backbone.
        local_files_only: Forbid network access when resolving the repository,
            for offline environments with a warm Hugging Face cache.
        num_frames: Frames per clip.
        input_size: Spatial crop size.
        num_classes: Output classes.
        dropout: Head dropout.
        mean: Normalisation mean; resolved from the checkpoint processor when
            ``None``.
        std: Normalisation std; resolved from the checkpoint processor when
            ``None``.
        min_load_ratio: Minimum fraction of the model's parameters that a
            local checkpoint must supply before the load is accepted.
        extra_model_kwargs: Extra keyword arguments forwarded to the loader.
    """

    hf_model_id: str = DEFAULT_VIDEOMAEV2_MODEL_ID
    backend: VideoMAEBackend = "hf_remote"
    pretrained_path: str | Path | None = None
    pretrained: bool = True
    allow_random_init: bool = False
    local_files_only: bool = False
    num_frames: int = 16
    input_size: int = 224
    num_classes: int = 2
    dropout: float = 0.0
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    min_load_ratio: float = 0.9
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in ("hf_remote", "hf_videomae"):
            raise ValueError(f"Unknown backend {self.backend!r}.")
        if self.num_frames <= 0 or self.num_frames % 2 != 0:
            raise ValueError(
                f"num_frames must be a positive even number (tubelet_size=2), "
                f"got {self.num_frames}."
            )
        if self.input_size % 16 != 0:
            raise ValueError(
                f"input_size must be a multiple of the patch size 16, got {self.input_size}."
            )
        if not self.pretrained and not self.allow_random_init:
            raise PretrainedWeightsError(
                "pretrained=False would train VideoMAEv2-B from scratch, which "
                "DLC-2021 is far too small for. Set allow_random_init=True if "
                "this is really intended."
            )


def _resolve_normalization(
    config: VideoMAEv2Config,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the checkpoint's official normalisation statistics."""
    if config.mean is not None and config.std is not None:
        return tuple(config.mean), tuple(config.std)  # type: ignore[return-value]

    try:
        from transformers import VideoMAEImageProcessor

        processor = VideoMAEImageProcessor.from_pretrained(
            config.hf_model_id, local_files_only=config.local_files_only
        )
    except Exception as error:  # noqa: BLE001 - re-raised with guidance
        raise PretrainedWeightsError(
            f"Could not load the image processor of {config.hf_model_id!r}, so the "
            "official normalisation statistics are unknown. Pass mean/std "
            f"explicitly if you know them. Original error: {error}"
        ) from error

    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    if len(mean) != 3 or len(std) != 3:
        raise PretrainedWeightsError(
            f"Unexpected normalisation statistics from {config.hf_model_id!r}: "
            f"mean={mean}, std={std}"
        )
    return mean, std  # type: ignore[return-value]


_PREFIX_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("", ""),
    ("module.", ""),
    ("backbone.", ""),
    ("encoder.", ""),
    ("", "model."),
    ("model.", ""),
    ("module.", "model."),
)


def _rewrite_keys(state_dict: Mapping[str, Any], strip: str, add: str) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix(strip) if strip else key
        rewritten[f"{add}{new_key}"] = value
    return rewritten


def load_local_weights(
    module: nn.Module,
    path: str | Path,
    *,
    min_load_ratio: float = 0.9,
    drop_prefixes: Sequence[str] = ("head.", "model.head."),
) -> dict[str, Any]:
    """Load an external checkpoint into ``module``, reconciling key prefixes.

    Official VideoMAEv2 weights are stored with the plain ``VisionTransformer``
    key names (``blocks.*``, ``patch_embed.proj.*``), while the Hugging Face
    wrapper nests them under ``model.``. Every plausible prefix rewrite is tried
    and the one matching the most parameters wins.

    Args:
        module: Target module.
        path: Checkpoint or raw ``state_dict`` file.
        min_load_ratio: Minimum fraction of the module's parameters the
            checkpoint must supply.
        drop_prefixes: Keys to discard, typically a classification head trained
            for a different label set.

    Returns:
        A report with the chosen prefix rewrite, matched key count, and the
        missing/unexpected key lists.

    Raises:
        PretrainedWeightsError: If the file is unusable or too few keys match.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise PretrainedWeightsError(f"Pretrained weights not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("model", "module", "state_dict", "model_state_dict", "ema_encoder"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping):
        raise PretrainedWeightsError(
            f"Could not find a state_dict inside {checkpoint_path}."
        )

    state_dict = {
        key: value
        for key, value in payload.items()
        if not any(key.startswith(prefix) for prefix in drop_prefixes)
    }

    target_keys = set(module.state_dict().keys())
    best: tuple[int, str, str, dict[str, Any]] = (-1, "", "", {})
    for strip, add in _PREFIX_CANDIDATES:
        candidate = _rewrite_keys(state_dict, strip, add)
        matched = len(target_keys & set(candidate.keys()))
        if matched > best[0]:
            best = (matched, strip, add, candidate)

    matched, strip, add, candidate = best
    ratio = matched / max(len(target_keys), 1)
    if ratio < min_load_ratio:
        raise PretrainedWeightsError(
            f"Only {matched}/{len(target_keys)} keys of {checkpoint_path.name} match "
            f"the model ({ratio:.1%} < {min_load_ratio:.1%}). Refusing to continue "
            "with a mostly random backbone. Check that the checkpoint matches the "
            "configured architecture."
        )

    incompatible = module.load_state_dict(candidate, strict=False)
    return {
        "path": str(checkpoint_path),
        "strip_prefix": strip,
        "add_prefix": add,
        "matched_keys": matched,
        "target_keys": len(target_keys),
        "load_ratio": ratio,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _build_hf_remote_backbone(config: VideoMAEv2Config) -> tuple[nn.Module, dict[str, Any]]:
    from transformers import AutoConfig, AutoModel

    try:
        hf_config = AutoConfig.from_pretrained(
            config.hf_model_id,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
        )
    except Exception as error:  # noqa: BLE001
        raise PretrainedWeightsError(
            f"Could not resolve {config.hf_model_id!r}. For an offline run the "
            "repository (including its remote code) must already be in the "
            f"Hugging Face cache. Original error: {error}"
        ) from error

    # Align the architecture with our clip geometry and drop any pretrained
    # classification head so the checkpoint's label set cannot leak in.
    model_config = getattr(hf_config, "model_config", None)
    if isinstance(model_config, dict):
        model_config.update(
            {
                "img_size": config.input_size,
                "num_frames": config.num_frames,
                "num_classes": 0,
            }
        )

    report: dict[str, Any] = {"source": config.hf_model_id, "backend": "hf_remote"}
    if config.pretrained_path is not None:
        model = AutoModel.from_config(
            hf_config, trust_remote_code=True, **config.extra_model_kwargs
        )
        report["weights"] = load_local_weights(
            model, config.pretrained_path, min_load_ratio=config.min_load_ratio
        )
    elif config.pretrained:
        try:
            model = AutoModel.from_pretrained(
                config.hf_model_id,
                config=hf_config,
                trust_remote_code=True,
                local_files_only=config.local_files_only,
                **config.extra_model_kwargs,
            )
        except Exception as error:  # noqa: BLE001
            raise PretrainedWeightsError(
                f"Could not load pretrained weights for {config.hf_model_id!r}: {error}"
            ) from error
        report["weights"] = {"source": config.hf_model_id}
    else:
        model = AutoModel.from_config(
            hf_config, trust_remote_code=True, **config.extra_model_kwargs
        )
        report["weights"] = {"source": "random init (explicitly allowed)"}

    return model, report


def _build_native_videomae_backbone(
    config: VideoMAEv2Config,
) -> tuple[nn.Module, dict[str, Any]]:
    from transformers import VideoMAEConfig, VideoMAEModel

    hf_config = VideoMAEConfig(
        image_size=config.input_size,
        num_frames=config.num_frames,
        patch_size=16,
        tubelet_size=2,
        use_mean_pooling=True,
    )
    report: dict[str, Any] = {"source": config.hf_model_id, "backend": "hf_videomae"}

    if config.pretrained_path is not None:
        model = VideoMAEModel(hf_config)
        report["weights"] = load_local_weights(
            model,
            config.pretrained_path,
            min_load_ratio=config.min_load_ratio,
            drop_prefixes=("classifier.", "fc_norm."),
        )
    elif config.pretrained:
        try:
            model = VideoMAEModel.from_pretrained(
                config.hf_model_id,
                local_files_only=config.local_files_only,
                **config.extra_model_kwargs,
            )
        except Exception as error:  # noqa: BLE001
            raise PretrainedWeightsError(
                f"Could not load {config.hf_model_id!r} with the native "
                "transformers VideoMAE implementation. VideoMAEv2 checkpoints "
                "generally need backend='hf_remote'. Original error: "
                f"{error}"
            ) from error
        report["weights"] = {"source": config.hf_model_id}
    else:
        model = VideoMAEModel(hf_config)
        report["weights"] = {"source": "random init (explicitly allowed)"}

    return model, report


class VideoMAEv2Classifier(Stage1Model):
    """VideoMAEv2-B with a linear Stage 1 head.

    Args:
        config: Branch configuration; defaults to :class:`VideoMAEv2Config`.
    """

    model_name: ClassVar[str] = "videomaev2_b"
    input_kind: ClassVar[str] = "video"

    def __init__(self, config: VideoMAEv2Config | None = None) -> None:
        super().__init__()
        self.config = config or VideoMAEv2Config()
        self.mean, self.std = _resolve_normalization(self.config)

        if self.config.backend == "hf_remote":
            self.backbone, self.load_report = _build_hf_remote_backbone(self.config)
            self._channels_first = True
        else:
            self.backbone, self.load_report = _build_native_videomae_backbone(self.config)
            self._channels_first = False

        self._inner = getattr(self.backbone, "model", self.backbone)
        self._feature_dim = self._infer_feature_dim()
        self.head = ClassifierHead(
            self._feature_dim, self.config.num_classes, dropout=self.config.dropout
        )

    def _infer_feature_dim(self) -> int:
        for attribute in ("embed_dim", "num_features"):
            value = getattr(self._inner, attribute, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        hidden = getattr(getattr(self.backbone, "config", None), "hidden_size", None)
        if isinstance(hidden, int) and hidden > 0:
            return int(hidden)
        raise PretrainedWeightsError(
            "Could not determine the feature dimension of the VideoMAEv2 backbone."
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def blocks(self) -> Sequence[nn.Module]:
        for attribute in ("blocks", "layer"):
            candidate = getattr(self._inner, attribute, None)
            if candidate is not None:
                return list(candidate)
        encoder = getattr(self._inner, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return list(encoder.layer)
        return []

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_modules(self) -> Iterable[nn.Module]:
        return [self.backbone]

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder ``(B, C, T, H, W)`` clips to the backbone's expected layout."""
        if x.ndim != 5:
            raise ValueError(
                f"Expected a (B, C, T, H, W) clip tensor, got shape {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise ValueError(
                "Expected 3 colour channels in dimension 1; the dataset returns "
                f"channel-first clips, got shape {tuple(x.shape)}."
            )
        return x if self._channels_first else x.permute(0, 2, 1, 3, 4).contiguous()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean-pooled clip embedding of shape ``(B, feature_dim)``."""
        inputs = self._prepare_input(x)

        forward_features = getattr(self._inner, "forward_features", None)
        if callable(forward_features):
            features = forward_features(inputs)
        else:
            features = self.backbone(inputs)

        if not isinstance(features, torch.Tensor):
            features = getattr(features, "last_hidden_state", None)
            if features is None:
                raise TypeError(
                    "The VideoMAEv2 backbone returned an object without "
                    "'last_hidden_state'; cannot extract features."
                )
        if features.ndim == 3:  # (B, num_tokens, dim) -> mean pooling
            features = features.mean(dim=1)
        if features.ndim != 2:
            raise ValueError(
                f"Expected (B, dim) features, got shape {tuple(features.shape)}."
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def preprocessing(self) -> Mapping[str, Any]:
        return {
            "input_kind": "video",
            "num_frames": self.config.num_frames,
            "input_size": self.config.input_size,
            "mean": self.mean,
            "std": self.std,
            "channels_first": self._channels_first,
            "source": self.config.hf_model_id,
            "notes": (
                "Official checkpoint preprocessing: ImageNet normalisation, "
                "16 frames, 224x224, tubelet_size=2."
            ),
        }


def build_videomaev2_b(**kwargs: Any) -> VideoMAEv2Classifier:
    """Build the VideoMAEv2-B branch from keyword arguments."""
    return VideoMAEv2Classifier(VideoMAEv2Config(**kwargs))


__all__ = [
    "DEFAULT_VIDEOMAEV2_MODEL_ID",
    "VideoMAEBackend",
    "VideoMAEv2Config",
    "VideoMAEv2Classifier",
    "build_videomaev2_b",
    "load_local_weights",
]

"""Stage 1 inference API.

This module is self-contained and does not depend on the removed ``manifest.py``.
It exposes the inference symbols re-exported by ``blackbox_detection.stage1``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch

from ..utils.checkpoint import load_checkpoint
from .dataset import (
    Stage1ForensicDataset,
    Stage1VideoDataset,
    build_dataloader,
    forensic_batch_adapter,
    video_batch_adapter,
)
from .evaluator import AggregationConfig, Stage1Evaluator, finalize_predictions
from .models import Stage1Model, build_stage1_model, model_input_kind
from .sampling import build_clip_sampler, build_forensic_samplers
from .transforms import ForensicPatchTransform, ValClipTransform


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
}


@dataclass
class LoadedStage1Model:
    """A checkpoint restored into a ready-to-run model."""

    model: Stage1Model
    model_name: str
    threshold: float
    preprocessing: Mapping[str, Any]
    aggregation: AggregationConfig
    metadata: Mapping[str, Any]

    @property
    def input_kind(self) -> str:
        return str(
            self.preprocessing.get(
                "input_kind",
                model_input_kind(self.model_name),
            )
        )


# Per-model overrides that keep checkpoint loading offline: the fine-tuned
# weights come from the checkpoint, so no pretrained weights are fetched.
_OFFLINE_OVERRIDES: Mapping[str, Mapping[str, Any]] = {
    "videomaev2_b": {
        "pretrained": False,
        "allow_random_init": True,
        "local_files_only": True,
    },
    "vjepa2_1_b": {
        "pretrained": False,
        "allow_random_init": True,
        "allow_download": False,
    },
    "bayar_resnet18": {"pretrained": False},
    "chromaticity": {"pretrained": False},
    "frequency": {"pretrained": False},
    "lcdf": {"pretrained": False},
    "cdc": {},
}

_OFFLINE_DROPPED_KEYS: Mapping[str, tuple[str, ...]] = {
    "videomaev2_b": ("pretrained_path",),
    "vjepa2_1_b": ("checkpoint_path",),
}


def _offline_model_params(
    model_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Strip anything that would trigger a download at inference time."""

    if model_name not in _OFFLINE_OVERRIDES:
        raise ValueError(f"Unknown Stage 1 model {model_name!r}.")

    offline = dict(params)
    for key in _OFFLINE_DROPPED_KEYS.get(model_name, ()):
        offline.pop(key, None)
    offline.update(_OFFLINE_OVERRIDES[model_name])
    return offline


def load_stage1_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = None,
    model_name: str | None = None,
    model_params: Mapping[str, Any] | None = None,
    threshold: float | None = None,
    strict: bool = True,
) -> LoadedStage1Model:
    """Rebuild a Stage 1 model from a training checkpoint."""

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    extra = dict(payload.get("extra") or {})
    stored_config = dict(payload.get("config") or {})
    stored_model_config = dict(stored_config.get("model_config") or {})

    name = model_name or extra.get("model_name") or stored_model_config.get("name")
    if not name:
        raise ValueError(
            f"{checkpoint_path} does not record its model name; "
            "pass model_name explicitly."
        )

    params = (
        dict(model_params)
        if model_params is not None
        else dict(stored_model_config.get("params") or {})
    )

    model = build_stage1_model(
        str(name),
        **_offline_model_params(str(name), params),
    )
    load_checkpoint(
        checkpoint_path,
        model=model,
        map_location=resolved_device,
        strict=strict,
        restore_rng_state=False,
    )
    model.to(resolved_device).eval()

    aggregation_payload = dict(extra.get("aggregation") or {})
    aggregation = (
        AggregationConfig(**aggregation_payload)
        if aggregation_payload
        else AggregationConfig()
    )

    preprocessing = dict(extra.get("preprocessing") or {}) or dict(
        model.preprocessing()
    )

    return LoadedStage1Model(
        model=model,
        model_name=str(name),
        threshold=float(
            threshold
            if threshold is not None
            else extra.get("best_threshold", 0.5)
        ),
        preprocessing=preprocessing,
        aggregation=aggregation,
        metadata=extra,
    )


def _iter_video_files(root: str | Path) -> list[Path]:
    """Return supported videos recursively in deterministic order."""

    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.as_posix(),
    )


def _probe_video_metadata(path: str | Path) -> dict[str, Any]:
    """Read lightweight container metadata with OpenCV.

    Failure to obtain one field does not make inference fail; the dataset can
    fall back to probing frame count itself.
    """

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {}

        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    metadata: dict[str, Any] = {}
    if num_frames > 0:
        metadata["num_frames"] = num_frames
    if fps > 0:
        metadata["fps"] = fps
    if width > 0:
        metadata["width"] = width
    if height > 0:
        metadata["height"] = height
    return metadata


def manifest_from_videos(
    videos: Sequence[str | Path] | str | Path,
    *,
    dataset: str = "inference",
    scene_type: str = "",
    probe_metadata: bool = True,
) -> pd.DataFrame:
    """Build the minimal manifest required by Stage 1 datasets.

    ``label`` is a placeholder because Stage1Dataset requires a known label
    column even though inference never uses the target for scoring.
    """

    if isinstance(videos, (str, Path)) and Path(videos).is_dir():
        paths = _iter_video_files(videos)
    elif isinstance(videos, (str, Path)):
        paths = [Path(videos)]
    else:
        paths = [Path(video) for video in videos]

    if not paths:
        raise ValueError("No videos found for inference.")

    resolved_paths = [path.expanduser().resolve() for path in paths]

    missing = [path for path in resolved_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} inference video(s) do not exist. "
            f"Examples: {missing[:3]}"
        )

    path_strings = [str(path) for path in resolved_paths]
    if len(set(path_strings)) != len(path_strings):
        raise ValueError(
            "The same inference video path was provided more than once."
        )

    stem_counts = Counter(path.stem for path in resolved_paths)
    rows: list[dict[str, Any]] = []

    for path in resolved_paths:
        stem = path.stem
        if stem_counts[stem] == 1:
            video_id = stem
        else:
            digest = hashlib.sha1(
                str(path).encode("utf-8")
            ).hexdigest()[:8]
            video_id = f"{stem}_{digest}"

        row: dict[str, Any] = {
            "video_path": str(path),
            "label": "ORIGINAL",  # placeholder required by dataset
            "video_id": video_id,
            "dataset": str(dataset),
            "source_video_id": "",
            "scene_type": str(scene_type),
        }

        if probe_metadata:
            row.update(_probe_video_metadata(path))

        rows.append(row)

    return pd.DataFrame(rows)


def _build_loader(
    loaded: LoadedStage1Model,
    manifest: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    num_clips: int,
    num_frames: int | None,
    num_patches: int,
    on_error: str,
) -> tuple[Any, Any]:
    """Build the deterministic inference loader and its batch adapter."""

    preprocessing = loaded.preprocessing

    if loaded.input_kind == "video":
        crop_size = int(preprocessing.get("input_size", 224))
        clip_frames = int(
            num_frames or preprocessing.get("num_frames", 16)
        )
        transform = ValClipTransform(
            crop_size=crop_size,
            mean=tuple(
                preprocessing.get(
                    "mean",
                    (0.485, 0.456, 0.406),
                )
            ),
            std=tuple(
                preprocessing.get(
                    "std",
                    (0.229, 0.224, 0.225),
                )
            ),
        )
        dataset = Stage1VideoDataset(
            manifest,
            clip_sampler=build_clip_sampler(
                train=False,
                num_frames=clip_frames,
                num_clips=num_clips,
            ),
            transform=transform,
            on_error=on_error,  # type: ignore[arg-type]
            deterministic=True,
        )
        adapter = video_batch_adapter()
    else:
        patch_size = int(preprocessing.get("patch_size", 256))
        frame_count = int(num_frames or 4)
        frame_sampler, patch_sampler = build_forensic_samplers(
            train=False,
            num_frames=frame_count,
            num_patches=num_patches,
            patch_size=patch_size,
        )
        dataset = Stage1ForensicDataset(
            manifest,
            frame_sampler=frame_sampler,
            patch_sampler=patch_sampler,
            transform=ForensicPatchTransform(train=False),
            patch_size=patch_size,
            on_error=on_error,  # type: ignore[arg-type]
            deterministic=True,
        )
        adapter = forensic_batch_adapter()

    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, adapter


def predict_stage1_proba(
    model_or_checkpoint: LoadedStage1Model | str | Path,
    videos: Sequence[str | Path] | str | Path | pd.DataFrame,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    num_clips: int = 1,
    num_frames: int | None = None,
    num_patches: int = 4,
    on_error: str = "zero",
    model_params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict video-level Stage 1 probabilities."""

    loaded = (
        model_or_checkpoint
        if isinstance(model_or_checkpoint, LoadedStage1Model)
        else load_stage1_model(
            model_or_checkpoint,
            device=device,
            model_params=model_params,
        )
    )

    manifest = (
        videos.copy()
        if isinstance(videos, pd.DataFrame)
        else manifest_from_videos(videos)
    )

    loader, adapter = _build_loader(
        loaded,
        manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        num_clips=num_clips,
        num_frames=num_frames,
        num_patches=num_patches,
        on_error=on_error,
    )

    evaluator = Stage1Evaluator(
        loaded.model,
        adapter,
        device=next(loaded.model.parameters()).device,
        aggregation=loaded.aggregation,
    )
    return evaluator.predict_videos(loader)


def predict_stage1(
    model_or_checkpoint: LoadedStage1Model | str | Path,
    videos: Sequence[str | Path] | str | Path | pd.DataFrame,
    *,
    threshold: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Predict official Stage 1 labels."""

    loaded = (
        model_or_checkpoint
        if isinstance(model_or_checkpoint, LoadedStage1Model)
        else load_stage1_model(
            model_or_checkpoint,
            device=kwargs.get("device"),
            model_params=kwargs.get("model_params"),
        )
    )

    probabilities = predict_stage1_proba(
        loaded,
        videos,
        **kwargs,
    )
    chosen = float(
        threshold if threshold is not None else loaded.threshold
    )
    return finalize_predictions(probabilities, chosen)


__all__ = [
    "LoadedStage1Model",
    "load_stage1_model",
    "manifest_from_videos",
    "predict_stage1_proba",
    "predict_stage1",
]

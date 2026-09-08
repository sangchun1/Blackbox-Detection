"""Stage 1 inference API.

Two functions form the contract that a submission pipeline can rely on:

``predict_stage1_proba``
    Video-level RERECORDED / ORIGINAL probabilities.
``predict_stage1``
    Official label strings ``"ORIGINAL"`` / ``"RERECORDED"``.
The official Baseline submission code is deliberately **not** touched. This
module only keeps the checkpoint format and the API semantics stable, so that
once a model is chosen it can be wired into the submission pipeline without
re-deriving anything.
Offline operation is a hard requirement: everything needed at inference time
(the fine-tuned weights, the threshold, the input geometry and the
normalisation statistics) is stored in the checkpoint, and models are rebuilt
with ``pretrained=False`` so no download is attempted.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from .manifest import (
    STAGE1_MANIFEST_COLUMNS,
    VIDEO_EXTENSIONS,
    iter_video_files,
    probe_video_metadata,
)
from .models import Stage1Model, build_stage1_model, model_input_kind
from .sampling import build_clip_sampler, build_forensic_samplers
from .transforms import ForensicPatchTransform, ValClipTransform

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
        return str(self.preprocessing.get("input_kind", model_input_kind(self.model_name)))
# Per-model overrides that keep checkpoint loading offline: the fine-tuned
# weights come from the checkpoint, so no pretrained weights are fetched.
# ``allow_random_init`` only exists on the video branches, whose guard against
# an accidentally untrained backbone must be waived here on purpose.
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
# Weight sources that are irrelevant once the checkpoint supplies the weights.
_OFFLINE_DROPPED_KEYS: Mapping[str, tuple[str, ...]] = {
    "videomaev2_b": ("pretrained_path",),
    "vjepa2_1_b": ("checkpoint_path",),
}


def _offline_model_params(model_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
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
    """Rebuild a Stage 1 model from a training checkpoint.
    Args:
        checkpoint_path: Path to ``best.pt``.
        device: Target device; resolved automatically when ``None``.
        model_name: Override the model key stored in the checkpoint.
        model_params: Override the stored model parameters. Needed for
            ``vjepa2_1_b``, whose architecture is built from a local clone of
            the official repository, so ``source_root`` must be supplied again.
        threshold: Override the stored decision threshold.
        strict: Require an exact ``state_dict`` match.
    Returns:
        A :class:`LoadedStage1Model`.
    Raises:
        ValueError: If the checkpoint does not identify its model.
    """
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    extra = dict(payload.get("extra") or {})
    stored_config = dict(payload.get("config") or {})
    stored_model_config = dict(stored_config.get("model_config") or {})
    name = model_name or extra.get("model_name") or stored_model_config.get("name")
    if not name:
        raise ValueError(
            f"{checkpoint_path} does not record its model name; pass model_name "
            "explicitly."
        )

    params = dict(model_params) if model_params is not None else dict(
        stored_model_config.get("params") or {}
    )
    model = build_stage1_model(str(name), **_offline_model_params(str(name), params))
    load_checkpoint(
        checkpoint_path,
        model=model,
        map_location=resolved_device,
        strict=strict,
        restore_rng_state=False,
    )
    model.to(resolved_device).eval()

    aggregation_payload = dict(extra.get("aggregation") or {})
    aggregation = AggregationConfig(**aggregation_payload) if aggregation_payload else AggregationConfig()
    preprocessing = dict(extra.get("preprocessing") or {}) or dict(model.preprocessing())
    return LoadedStage1Model(
        model=model,
        model_name=str(name),
        threshold=float(
            threshold if threshold is not None else extra.get("best_threshold", 0.5)
        ),
        preprocessing=preprocessing,
        aggregation=aggregation,
        metadata=extra,
    )

def manifest_from_videos(
    videos: Sequence[str | Path] | str | Path,
    *,
    dataset: str = "inference",
    scene_type: str = "",
    probe_metadata: bool = True,
) -> pd.DataFrame:
    """Build a minimal manifest for inference from paths or a directory.
    Labels are unknown at inference time and are filled with ``"ORIGINAL"``
    purely so the dataset's label mapping is satisfied. They are never used for
    scoring here.
    """
    if isinstance(videos, (str, Path)) and Path(videos).is_dir():
        paths = iter_video_files(videos, extensions=VIDEO_EXTENSIONS)
    elif isinstance(videos, (str, Path)):
        paths = [Path(videos)]
    else:
        paths = [Path(video) for video in videos]
    if not paths:
        raise ValueError("No videos found for inference.")

    resolved_paths = [Path(path).resolve() for path in paths]
    path_strings = [str(path) for path in resolved_paths]
    if len(set(path_strings)) != len(path_strings):
        raise ValueError("The same inference video path was provided more than once.")

    stem_counts = Counter(path.stem for path in resolved_paths)
    rows: list[dict[str, Any]] = []
    for path in resolved_paths:
        stem = path.stem
        if stem_counts[stem] == 1:
            video_id = stem
        else:
            digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
            video_id = f"{stem}_{digest}"

        row: dict[str, Any] = {
            "video_path": str(path),
            "label": "ORIGINAL",  # placeholder, unused
            "dataset": dataset,
            "video_id": video_id,
            "source_video_id": "",
            "scene_type": scene_type,
            "is_synthetic": False,
            "capture_device": "",
            "display_device": "",
        }
        if probe_metadata:
            row.update(probe_video_metadata(path, verify_decode=False).as_dict())
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in STAGE1_MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame

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
        clip_frames = int(num_frames or preprocessing.get("num_frames", 16))
        transform = ValClipTransform(
            crop_size=crop_size,
            mean=tuple(preprocessing.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(preprocessing.get("std", (0.229, 0.224, 0.225))),
        )
        dataset = Stage1VideoDataset(
            manifest,
            clip_sampler=build_clip_sampler(
                train=False, num_frames=clip_frames, num_clips=num_clips
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
    """Predict video-level Stage 1 probabilities.
    Args:
        model_or_checkpoint: A loaded model or a checkpoint path.
        videos: Video paths, a directory, or a manifest ``DataFrame``.
        device: Inference device.
        batch_size: Videos per batch.
        num_workers: DataLoader workers.
        num_clips: Clips per video (video branch); more clips means multi-clip
            inference.
        num_frames: Override the clip length or the forensic frame count.
        num_patches: Patches per frame (forensic branch).
        on_error: ``"zero"`` keeps going on a broken video, ``"raise"`` fails.
        model_params: Model parameters needed to rebuild the architecture, e.g.
            ``source_root`` for ``vjepa2_1_b``.
    Returns:
        One row per video with ``video_id``, ``prob_original``,
        ``prob_rerecorded``, ``num_units`` and ``num_frames``.
    """
    loaded = (
        model_or_checkpoint
        if isinstance(model_or_checkpoint, LoadedStage1Model)
        else load_stage1_model(
            model_or_checkpoint, device=device, model_params=model_params
        )
    )
    manifest = videos if isinstance(videos, pd.DataFrame) else manifest_from_videos(videos)
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
    """Predict official Stage 1 labels.
    Args:
        model_or_checkpoint: A loaded model or a checkpoint path.
        videos: Video paths, a directory, or a manifest ``DataFrame``.
        threshold: Decision threshold; the checkpoint's validated threshold is
            used when ``None``.
        **kwargs: Forwarded to :func:`predict_stage1_proba`.
    Returns:
        One row per video with ``video_id``, the probabilities and a
        ``prediction`` column holding ``"ORIGINAL"`` or ``"RERECORDED"``.
    """
    loaded = (
        model_or_checkpoint
        if isinstance(model_or_checkpoint, LoadedStage1Model)
        else load_stage1_model(
            model_or_checkpoint,
            device=kwargs.get("device"),
            model_params=kwargs.get("model_params"),
        )
    )
    probabilities = predict_stage1_proba(loaded, videos, **kwargs)
    chosen = float(threshold if threshold is not None else loaded.threshold)
    return finalize_predictions(probabilities, chosen)

__all__ = [
    "LoadedStage1Model",
    "load_stage1_model",
    "manifest_from_videos",
    "predict_stage1_proba",
    "predict_stage1",
]

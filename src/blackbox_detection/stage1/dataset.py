"""Manifest-driven Stage 1 datasets.

Both datasets consume the common Stage 1 manifest, so adding a data source
(CCD originals, CCD physical re-recordings) means adding manifest rows, not a
new ``Dataset`` class. No dataset-specific path logic lives here.

Two datasets exist because the two branches need fundamentally different
tensors, not because they come from different data:

``Stage1VideoDataset``
    Decodes a temporal clip, resizes/crops it and returns
    ``(num_clips, 3, T, H, W)``.
``Stage1ForensicDataset``
    Decodes whole **native-resolution** frames, crops
    ``patch_size`` x ``patch_size`` patches out of them without any resize, and
    returns ``(num_units, 3, P, P)`` where ``num_units = num_frames *
    num_patches``. Evaluation aggregates patch scores up to the video level.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.seed import DEFAULT_SEED, dataloader_seed_kwargs
from .manifest import STAGE1_LABEL_TO_INDEX
from .sampling import ClipSampler, FrameSampler, PatchSampler
from .transforms import pad_to_min_size

ErrorPolicy = Literal["raise", "zero"]
VideoTransform = Callable[[np.ndarray, np.random.Generator | None], torch.Tensor]
PatchTransform = Callable[[np.ndarray, np.random.Generator | None], torch.Tensor]


class VideoDecodeError(RuntimeError):
    """Raised when a video cannot be opened or a requested frame cannot decode."""


def _new_item_rng(deterministic: bool) -> np.random.Generator | None:
    """Return a per-item generator, or ``None`` for deterministic pipelines.
    The seed is drawn from the global NumPy RNG, which
    :func:`blackbox_detection.utils.seed.seed_worker` seeds deterministically
    per DataLoader worker and per epoch. Augmentation therefore varies across
    epochs while a run stays reproducible for a given seed.
    """
    if deterministic:
        return None
    return np.random.default_rng(int(np.random.randint(0, 2**32 - 1)))

def decode_frames(
    path: str | Path,
    frame_indices: Sequence[int] | np.ndarray,
    *,
    to_rgb: bool = True,
) -> np.ndarray:
    """Decode the requested frames at native resolution.

    Frames are read sequentially from the first requested index, which is far
    more reliable across containers than seeking to each frame. Repeated indices
    reuse the already decoded frame.
    Args:
        path: Video file path.
        frame_indices: Ascending frame indices.
        to_rgb: Convert BGR to RGB.

    Returns:
        ``(len(frame_indices), H, W, 3)`` uint8 array at native resolution.
    Raises:
        VideoDecodeError: If the file cannot be opened or no frame decodes.
    """
    video_path = Path(path)
    wanted = [int(index) for index in np.asarray(frame_indices).reshape(-1)]
    if not wanted:
        raise VideoDecodeError(f"No frame indices requested for {video_path}.")

    order = np.argsort(wanted, kind="stable")
    sorted_wanted = [wanted[position] for position in order]
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")

        decoded: list[np.ndarray | None] = [None] * len(sorted_wanted)
        capture.set(cv2.CAP_PROP_POS_FRAMES, sorted_wanted[0])
        position = sorted_wanted[0]
        last_frame: np.ndarray | None = None
        for slot, target in enumerate(sorted_wanted):
            if last_frame is not None and target < position:
                # Duplicate or clamped index: reuse the frame already decoded.
                decoded[slot] = last_frame
                continue

            frame = None
            ok = False
            while position <= target:
                ok, frame = capture.read()
                position += 1
                if not ok:
                    break
            if not ok or frame is None:
                decoded[slot] = last_frame
                continue

            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if to_rgb else frame
            decoded[slot] = last_frame

        if all(frame is None for frame in decoded):
            raise VideoDecodeError(f"Cannot decode any frame of {video_path}.")
        # Backfill leading failures with the first successfully decoded frame.
        first_valid = next(frame for frame in decoded if frame is not None)
        filled = [frame if frame is not None else first_valid for frame in decoded]
    finally:
        capture.release()

    restored: list[np.ndarray] = [np.empty(0)] * len(wanted)
    for slot, original_position in enumerate(order):
        restored[int(original_position)] = filled[slot]
    return np.stack(restored)

def crop_patch(frame: np.ndarray, top: int, left: int, patch_size: int) -> np.ndarray:
    """Crop one native-resolution patch, reflect-padding when the frame is small."""
    padded = pad_to_min_size(frame, patch_size)
    height, width = padded.shape[:2]
    top = int(min(max(top, 0), max(height - patch_size, 0)))
    left = int(min(max(left, 0), max(width - patch_size, 0)))
    return padded[top : top + patch_size, left : left + patch_size]

@dataclass(frozen=True)
class DatasetItemMeta:
    """Row-level metadata attached to every dataset item."""

    video_id: str
    dataset: str
    label_name: str
    row_index: int


class _Stage1BaseDataset(Dataset):
    """Shared manifest handling for both Stage 1 datasets."""
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        label_map: Mapping[str, int] = STAGE1_LABEL_TO_INDEX,
        on_error: ErrorPolicy = "zero",
        deterministic: bool = False,
    ) -> None:
        if len(manifest) == 0:
            raise ValueError("Cannot build a Stage 1 dataset from an empty manifest.")
        if on_error not in ("raise", "zero"):
            raise ValueError(f"on_error must be 'raise' or 'zero', got {on_error!r}.")
        required = {"video_path", "label", "video_id", "dataset"}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest is missing columns required by the dataset: {missing}")

        self.manifest = manifest.reset_index(drop=True)
        self.label_map = dict(label_map)
        self.on_error = on_error
        self.deterministic = bool(deterministic)
        unknown = sorted(set(self.manifest["label"]) - set(self.label_map))
        if unknown:
            raise ValueError(f"Manifest contains labels without a mapping: {unknown}")
        self._paths = self.manifest["video_path"].astype(str).tolist()
        self._labels = [self.label_map[label] for label in self.manifest["label"]]
        self._video_ids = self.manifest["video_id"].astype(str).tolist()
        self._datasets = self.manifest["dataset"].astype(str).tolist()
        self._label_names = self.manifest["label"].astype(str).tolist()
        self._num_frames_hint = (
            self.manifest["num_frames"].astype("int64").tolist()
            if "num_frames" in self.manifest.columns
            else [0] * len(self.manifest)
        )
    def __len__(self) -> int:
        return len(self.manifest)

    def _meta(self, index: int) -> DatasetItemMeta:
        return DatasetItemMeta(
            video_id=self._video_ids[index],
            dataset=self._datasets[index],
            label_name=self._label_names[index],
            row_index=index,
        )
    def _total_frames(self, index: int) -> int:
        """Frame count from the manifest, falling back to a container probe."""
        hint = int(self._num_frames_hint[index])
        if hint > 0:
            return hint

        capture = cv2.VideoCapture(self._paths[index])
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        return max(total, 1)
    def _handle_error(self, index: int, error: Exception) -> None:
        if self.on_error == "raise":
            raise VideoDecodeError(
                f"Failed to load {self._video_ids[index]} "
                f"({self._paths[index]}): {error}"
            ) from error


class Stage1VideoDataset(_Stage1BaseDataset):
    """Clip dataset for the video branch.
    Args:
        manifest: Stage 1 manifest (already restricted to one split).
        clip_sampler: Temporal sampler returning ``(num_clips, num_frames)``.
        transform: Callable mapping ``(T, H, W, 3)`` uint8 plus an optional
            generator to a normalised ``(3, T, H, W)`` tensor.
        label_map: Label-name to class-index mapping.
        on_error: ``"raise"`` to fail loudly, ``"zero"`` to emit a zero clip
            with ``valid=False`` so that a single broken file cannot kill a run.
        deterministic: Disable the per-item generator; use for validation.
    """
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        clip_sampler: ClipSampler,
        transform: VideoTransform,
        label_map: Mapping[str, int] = STAGE1_LABEL_TO_INDEX,
        on_error: ErrorPolicy = "zero",
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            manifest,
            label_map=label_map,
            on_error=on_error,
            deterministic=deterministic,
        )
        self.clip_sampler = clip_sampler
        self.transform = transform
    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self._meta(index)
        rng = _new_item_rng(self.deterministic)
        clips = np.atleast_2d(self.clip_sampler(self._total_frames(index), rng))
        tensors: list[torch.Tensor] = []
        unit_valid: list[bool] = []
        for clip_indices in clips:
            try:
                frames = decode_frames(self._paths[index], clip_indices)
                tensors.append(self.transform(frames, rng))
                unit_valid.append(True)
            except Exception as error:  # noqa: BLE001 - reported via `valid`
                self._handle_error(index, error)
                tensors.append(None)  # type: ignore[arg-type]
                unit_valid.append(False)
        reference = next((tensor for tensor in tensors if tensor is not None), None)
        if reference is None:
            # Nothing decoded: emit a zero clip whose shape matches the sampler.
            crop_size = int(getattr(self.transform, "crop_size", 224))
            reference = torch.zeros(
                3, int(clips.shape[1]), crop_size, crop_size, dtype=torch.float32
            )
            tensors = [reference for _ in tensors]
        else:
            tensors = [
                tensor if tensor is not None else torch.zeros_like(reference)
                for tensor in tensors
            ]
        return {
            "pixels": torch.stack(tensors),
            "label": torch.tensor(self._labels[index], dtype=torch.long),
            "valid": torch.tensor(bool(all(unit_valid))),
            "unit_valid": torch.tensor(unit_valid, dtype=torch.bool),
            "video_id": meta.video_id,
            "dataset": meta.dataset,
            "label_name": meta.label_name,
            "row_index": torch.tensor(meta.row_index, dtype=torch.long),
        }


class Stage1ForensicDataset(_Stage1BaseDataset):
    """Native-resolution patch dataset for the forensic branch.
    The pipeline order is fixed and deliberate::

        native-resolution decoded frame -> native-resolution crop -> P x P patch

    A frame is never resized before cropping.
    Args:
        manifest: Stage 1 manifest (already restricted to one split).
        frame_sampler: Frame index sampler.
        patch_sampler: Patch position sampler.
        transform: Callable mapping ``(P, P, 3)`` uint8 plus an optional
            generator to a ``(3, P, P)`` tensor in ``[0, 1]``.
        patch_size: Patch side length in native pixels.
        label_map: Label-name to class-index mapping.
        on_error: Error policy, see :class:`Stage1VideoDataset`.
        deterministic: Disable the per-item generator; use for validation.
    """
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        frame_sampler: FrameSampler,
        patch_sampler: PatchSampler,
        transform: PatchTransform,
        patch_size: int = 256,
        label_map: Mapping[str, int] = STAGE1_LABEL_TO_INDEX,
        on_error: ErrorPolicy = "zero",
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            manifest,
            label_map=label_map,
            on_error=on_error,
            deterministic=deterministic,
        )
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if int(getattr(patch_sampler, "patch_size", patch_size)) != int(patch_size):
            raise ValueError(
                "patch_sampler.patch_size "
                f"({getattr(patch_sampler, 'patch_size', None)}) must match "
                f"patch_size ({patch_size})."
            )
        self.frame_sampler = frame_sampler
        self.patch_sampler = patch_sampler
        self.transform = transform
        self.patch_size = int(patch_size)

    @property
    def num_units(self) -> int:
        """Patches returned per video item."""
        return int(self.frame_sampler.num_frames) * int(self.patch_sampler.num_patches)
    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self._meta(index)
        rng = _new_item_rng(self.deterministic)
        frame_indices = np.asarray(
            self.frame_sampler(self._total_frames(index), rng)
        ).reshape(-1)

        patches: list[torch.Tensor] = []
        frame_slots: list[int] = []
        patch_slots: list[int] = []
        unit_valid: list[bool] = []
        valid = True
        try:
            frames = decode_frames(self._paths[index], frame_indices)
        except Exception as error:  # noqa: BLE001 - reported via `valid`
            self._handle_error(index, error)
            valid = False
            frames = None
        if frames is not None:
            for frame_slot, frame in enumerate(frames):
                height, width = frame.shape[:2]
                positions = np.asarray(self.patch_sampler(height, width, rng))
                for patch_slot, (top, left) in enumerate(positions):
                    patch = crop_patch(frame, int(top), int(left), self.patch_size)
                    patches.append(self.transform(patch, rng))
                    frame_slots.append(int(frame_indices[frame_slot]))
                    patch_slots.append(patch_slot)
                    unit_valid.append(True)
        expected = self.num_units
        if len(patches) != expected:
            valid = False
            zero = torch.zeros(3, self.patch_size, self.patch_size, dtype=torch.float32)
            while len(patches) < expected:
                patches.append(zero.clone())
                frame_slots.append(-1)
                patch_slots.append(-1)
                unit_valid.append(False)
            patches = patches[:expected]
            frame_slots = frame_slots[:expected]
            patch_slots = patch_slots[:expected]
            unit_valid = unit_valid[:expected]
        return {
            "patches": torch.stack(patches),
            "label": torch.tensor(self._labels[index], dtype=torch.long),
            "valid": torch.tensor(bool(valid and all(unit_valid))),
            "unit_valid": torch.tensor(unit_valid, dtype=torch.bool),
            "frame_indices": torch.tensor(frame_slots, dtype=torch.long),
            "patch_indices": torch.tensor(patch_slots, dtype=torch.long),
            "video_id": meta.video_id,
            "dataset": meta.dataset,
            "label_name": meta.label_name,
            "row_index": torch.tensor(meta.row_index, dtype=torch.long),
        }

def stage1_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate Stage 1 items, stacking tensors and keeping string metadata as lists."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    collated: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = list(values)
    return collated

def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    seed: int = DEFAULT_SEED,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Build a seeded DataLoader with the project's reproducibility settings.
    ``pin_memory`` defaults to whether CUDA is available, so the same call works
    on a GPU box and on a CPU-only machine without a warning.
    """
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": (
            torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        ),
        "drop_last": bool(drop_last),
        "collate_fn": stage1_collate,
        **dataloader_seed_kwargs(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = (
            bool(persistent_workers) if persistent_workers is not None else True
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


__all__ = [
    "ErrorPolicy",
    "VideoTransform",
    "PatchTransform",
    "VideoDecodeError",
    "DatasetItemMeta",
    "decode_frames",
    "crop_patch",
    "Stage1VideoDataset",
    "Stage1ForensicDataset",
    "stage1_collate",
    "build_dataloader",
]
# Batch adapters --------------------------------------------------------------


@dataclass(frozen=True)
class AdaptedBatch:
    """One batch flattened to model-ready units.

    A "unit" is a clip for the video branch and a patch for the forensic
    branch. Both datasets return ``(B, num_units, ...)``, so the trainer and
    evaluator work on flattened units and keep ``video_index`` to aggregate
    back to video level.
    Attributes:
        inputs: ``(B * num_units, ...)`` model input.
        targets: ``(B * num_units,)`` class indices.
        video_index: ``(B * num_units,)`` index into the batch's video list.
        valid_mask: ``(B * num_units,)`` boolean mask. Invalid decode/padded
            units must not contribute to training loss or evaluation.
        num_units: Units per video in this batch.
        meta: Additional flattened per-unit tensors, e.g. frame/patch indices.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    video_index: torch.Tensor
    valid_mask: torch.Tensor
    num_units: int
    meta: Mapping[str, torch.Tensor]

@dataclass(frozen=True)
class Stage1BatchAdapter:
    """Flatten a Stage 1 batch into units without branching on the model type.

    Attributes:
        input_key: ``"pixels"`` for the video branch, ``"patches"`` for the
            forensic branch.
        meta_keys: Per-unit metadata tensors to flatten alongside the inputs.
    """

    input_key: str
    meta_keys: tuple[str, ...] = ()
    def unpack(
        self,
        batch: Mapping[str, Any],
        device: torch.device | str,
        *,
        non_blocking: bool = True,
    ) -> AdaptedBatch:
        if self.input_key not in batch:
            raise KeyError(
                f"Batch has no key {self.input_key!r}; available keys: {sorted(batch)}"
            )
        inputs = batch[self.input_key]
        if inputs.ndim < 3:
            raise ValueError(
                f"Expected batch[{self.input_key!r}] with shape (B, U, ...), got "
                f"{tuple(inputs.shape)}."
            )
        batch_size, num_units = int(inputs.shape[0]), int(inputs.shape[1])
        flat_inputs = inputs.flatten(0, 1).to(device, non_blocking=non_blocking)
        targets = (
            batch["label"].repeat_interleave(num_units).to(device, non_blocking=non_blocking)
        )
        video_index = torch.arange(batch_size).repeat_interleave(num_units)

        if "unit_valid" in batch:
            unit_valid = batch["unit_valid"]
            if tuple(unit_valid.shape[:2]) != (batch_size, num_units):
                raise ValueError(
                    "batch['unit_valid'] must have shape (B, U), got "
                    f"{tuple(unit_valid.shape)} for B={batch_size}, U={num_units}."
                )
            valid_mask = unit_valid.reshape(-1).bool()
        elif "valid" in batch:
            valid_mask = batch["valid"].bool().repeat_interleave(num_units)
        else:
            valid_mask = torch.ones(batch_size * num_units, dtype=torch.bool)
        valid_mask = valid_mask.to(device, non_blocking=non_blocking)

        meta: dict[str, torch.Tensor] = {}
        for key in self.meta_keys:
            if key in batch:
                meta[key] = batch[key].reshape(-1)

        return AdaptedBatch(
            inputs=flat_inputs,
            targets=targets,
            video_index=video_index,
            valid_mask=valid_mask,
            num_units=num_units,
            meta=meta,
        )

def video_batch_adapter() -> Stage1BatchAdapter:
    """Adapter for :class:`Stage1VideoDataset` batches."""
    return Stage1BatchAdapter(input_key="pixels")


def forensic_batch_adapter() -> Stage1BatchAdapter:
    """Adapter for :class:`Stage1ForensicDataset` batches."""
    return Stage1BatchAdapter(
        input_key="patches", meta_keys=("frame_indices", "patch_indices")
    )


__all__ += [
    "AdaptedBatch",
    "Stage1BatchAdapter",
    "video_batch_adapter",
    "forensic_batch_adapter",
]

"""Temporal and spatial samplers for Stage 1.

Video branch
------------
Training uses a dense clip with a random temporal start and a stride drawn from
a small set, which exposes the model to different temporal aliasing rates of the
display refresh. Validation uses a deterministic clip so that scores are
comparable across epochs and models. Multi-clip inference is expressed by the
same interface, so switching from 1 to N clips does not change calling code:
every sampler returns an array of shape ``(num_clips, num_frames)``.

Forensic branch
---------------
Patches are cropped from **native-resolution** frames. The samplers here only
produce frame indices and patch top-left coordinates; the dataset does the
decoding and cropping. Nothing resizes a frame before cropping, because a
resize destroys the moire / aliasing / sub-pixel traces the forensic branch is
supposed to detect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

DEFAULT_TRAIN_STRIDES: tuple[int, ...] = (1, 2, 4)


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _clip_indices(start: int, num_frames: int, stride: int, total_frames: int) -> np.ndarray:
    """Build one clip's frame indices, clamped to the last valid frame.

    Clamping (rather than wrapping) repeats the final frame for short videos,
    which avoids introducing an artificial temporal discontinuity that could
    look like a recapture artefact.
    """
    indices = start + np.arange(num_frames, dtype=np.int64) * stride
    return np.clip(indices, 0, max(total_frames - 1, 0))


class ClipSampler(Protocol):
    """Interface shared by every temporal sampler."""

    num_frames: int

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return frame indices of shape ``(num_clips, num_frames)``."""


@dataclass(frozen=True)
class DenseClipSampler:
    """Random dense clip for training.

    A stride is drawn uniformly from ``strides`` and the clip start is drawn
    uniformly from the valid range, so the model sees varying temporal sampling
    rates of the same video.

    Attributes:
        num_frames: Frames per clip.
        strides: Candidate temporal strides.
        num_clips: Clips returned per call.
    """

    num_frames: int = 16
    strides: tuple[int, ...] = DEFAULT_TRAIN_STRIDES
    num_clips: int = 1

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")
        if not self.strides or any(stride <= 0 for stride in self.strides):
            raise ValueError(f"strides must be positive integers, got {self.strides}.")
        if self.num_clips <= 0:
            raise ValueError(f"num_clips must be positive, got {self.num_clips}.")

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = _as_rng(rng)
        total = max(int(total_frames), 1)

        clips = np.empty((self.num_clips, self.num_frames), dtype=np.int64)
        for clip_index in range(self.num_clips):
            stride = int(generator.choice(np.asarray(self.strides, dtype=np.int64)))
            span = (self.num_frames - 1) * stride + 1
            max_start = max(total - span, 0)
            start = int(generator.integers(0, max_start + 1))
            clips[clip_index] = _clip_indices(start, self.num_frames, stride, total)
        return clips


@dataclass(frozen=True)
class DeterministicClipSampler:
    """Deterministic centre clip for validation.

    Attributes:
        num_frames: Frames per clip.
        stride: Fixed temporal stride.
        num_clips: Number of evenly spaced clips (1 = centre clip only).
    """

    num_frames: int = 16
    stride: int = 2
    num_clips: int = 1

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")
        if self.num_clips <= 0:
            raise ValueError(f"num_clips must be positive, got {self.num_clips}.")

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        del rng  # Deterministic by construction.
        total = max(int(total_frames), 1)
        span = (self.num_frames - 1) * self.stride + 1
        max_start = max(total - span, 0)

        clips = np.empty((self.num_clips, self.num_frames), dtype=np.int64)
        for clip_index in range(self.num_clips):
            # Evenly spaced clip centres, identical to the single centre clip
            # when num_clips == 1.
            position = (clip_index + 0.5) / self.num_clips
            start = int(round(position * max_start)) if self.num_clips > 1 else max_start // 2
            start = min(max(start, 0), max_start)
            clips[clip_index] = _clip_indices(start, self.num_frames, self.stride, total)
        return clips


@dataclass(frozen=True)
class MultiClipSampler:
    """Deterministic multi-clip sampler for test-time aggregation.

    Thin alias of :class:`DeterministicClipSampler` with a clip count, kept as a
    separate name so notebooks can express "multi-clip inference" explicitly.
    """

    num_frames: int = 16
    stride: int = 2
    num_clips: int = 3

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return DeterministicClipSampler(
            num_frames=self.num_frames,
            stride=self.stride,
            num_clips=self.num_clips,
        )(total_frames, rng)


# Forensic frame samplers -----------------------------------------------------


class FrameSampler(Protocol):
    """Interface shared by forensic frame samplers."""

    num_frames: int

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return a 1-D array of frame indices."""


@dataclass(frozen=True)
class RandomFrameSampler:
    """Random distinct frame indices for forensic training."""

    num_frames: int = 2

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = _as_rng(rng)
        total = max(int(total_frames), 1)
        if total <= self.num_frames:
            indices = np.arange(total, dtype=np.int64)
            if len(indices) < self.num_frames:
                pad = np.full(self.num_frames - len(indices), total - 1, dtype=np.int64)
                indices = np.concatenate([indices, pad])
            return np.sort(indices)
        indices = generator.choice(total, size=self.num_frames, replace=False)
        return np.sort(indices.astype(np.int64))


@dataclass(frozen=True)
class DeterministicFrameSampler:
    """Evenly spaced frame indices for forensic validation."""

    num_frames: int = 4

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")

    def __call__(
        self,
        total_frames: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        del rng
        total = max(int(total_frames), 1)
        positions = (np.arange(self.num_frames) + 0.5) / self.num_frames
        indices = np.floor(positions * total).astype(np.int64)
        return np.clip(indices, 0, total - 1)


# Forensic patch samplers -----------------------------------------------------


class PatchSampler(Protocol):
    """Interface shared by forensic patch samplers."""

    patch_size: int
    num_patches: int

    def __call__(
        self,
        height: int,
        width: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return patch top-left coordinates of shape ``(num_patches, 2)`` as ``(y, x)``."""


def _max_offset(size: int, patch_size: int) -> int:
    """Largest valid top-left offset; 0 when the frame is smaller than a patch.

    Frames smaller than the patch are handled by the dataset via reflection
    padding, never by upscaling, so that no interpolation is introduced.
    """
    return max(int(size) - int(patch_size), 0)


@dataclass(frozen=True)
class RandomPatchSampler:
    """Uniformly random native-resolution patch positions for training."""

    patch_size: int = 256
    num_patches: int = 4

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
        if self.num_patches <= 0:
            raise ValueError(f"num_patches must be positive, got {self.num_patches}.")

    def __call__(
        self,
        height: int,
        width: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = _as_rng(rng)
        max_y = _max_offset(height, self.patch_size)
        max_x = _max_offset(width, self.patch_size)
        ys = generator.integers(0, max_y + 1, size=self.num_patches)
        xs = generator.integers(0, max_x + 1, size=self.num_patches)
        return np.stack([ys, xs], axis=1).astype(np.int64)


@dataclass(frozen=True)
class DeterministicPatchSampler:
    """Deterministic grid of native-resolution patch positions for validation.

    Positions are the cell centres of a ``grid`` x ``grid`` layout, ordered
    row-major and truncated to ``num_patches``. The centre-most positions come
    first for odd grids, so reducing ``num_patches`` keeps the most informative
    crops.
    """

    patch_size: int = 256
    num_patches: int = 4
    grid: int = 2

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
        if self.num_patches <= 0:
            raise ValueError(f"num_patches must be positive, got {self.num_patches}.")
        if self.grid <= 0:
            raise ValueError(f"grid must be positive, got {self.grid}.")
        if self.num_patches > self.grid * self.grid:
            raise ValueError(
                f"num_patches={self.num_patches} exceeds grid capacity "
                f"{self.grid * self.grid}. Increase grid."
            )

    def __call__(
        self,
        height: int,
        width: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        del rng
        max_y = _max_offset(height, self.patch_size)
        max_x = _max_offset(width, self.patch_size)

        fractions = (np.arange(self.grid) + 0.5) / self.grid
        ys = np.round(fractions * max_y).astype(np.int64)
        xs = np.round(fractions * max_x).astype(np.int64)

        positions = [(int(y), int(x)) for y in ys for x in xs]
        centre = np.array([max_y / 2.0, max_x / 2.0])
        positions.sort(key=lambda pos: float(np.hypot(pos[0] - centre[0], pos[1] - centre[1])))
        return np.asarray(positions[: self.num_patches], dtype=np.int64)


def build_clip_sampler(
    *,
    train: bool,
    num_frames: int = 16,
    strides: Sequence[int] = DEFAULT_TRAIN_STRIDES,
    val_stride: int = 2,
    num_clips: int = 1,
) -> ClipSampler:
    """Return the train or validation clip sampler for one configuration."""
    if train:
        return DenseClipSampler(
            num_frames=num_frames,
            strides=tuple(int(stride) for stride in strides),
            num_clips=num_clips,
        )
    return DeterministicClipSampler(
        num_frames=num_frames,
        stride=int(val_stride),
        num_clips=num_clips,
    )


def build_forensic_samplers(
    *,
    train: bool,
    num_frames: int,
    num_patches: int,
    patch_size: int = 256,
    val_grid: int = 2,
) -> tuple[FrameSampler, PatchSampler]:
    """Return matching frame and patch samplers for the forensic branch."""
    if train:
        return (
            RandomFrameSampler(num_frames=num_frames),
            RandomPatchSampler(patch_size=patch_size, num_patches=num_patches),
        )
    return (
        DeterministicFrameSampler(num_frames=num_frames),
        DeterministicPatchSampler(
            patch_size=patch_size,
            num_patches=num_patches,
            grid=max(val_grid, int(np.ceil(np.sqrt(num_patches)))),
        ),
    )


__all__ = [
    "DEFAULT_TRAIN_STRIDES",
    "ClipSampler",
    "FrameSampler",
    "PatchSampler",
    "DenseClipSampler",
    "DeterministicClipSampler",
    "MultiClipSampler",
    "RandomFrameSampler",
    "DeterministicFrameSampler",
    "RandomPatchSampler",
    "DeterministicPatchSampler",
    "build_clip_sampler",
    "build_forensic_samplers",
]

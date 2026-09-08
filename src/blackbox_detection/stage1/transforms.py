"""Stage 1 augmentation and tensor conversion.

Video branch: clip-consistent augmentation
------------------------------------------
Every spatial and photometric parameter is drawn **once per clip** and applied
identically to all frames of that clip. Per-frame jitter would synthesise
brightness/colour flicker, which is exactly one of the cues the video branch is
supposed to learn from real re-recorded video, and would therefore teach the
model an artefact we created ourselves.

Forensic branch: no resizing
----------------------------
Forensic transforms operate on patches that were already cropped from
native-resolution frames and never resize, blur or colour-jitter them, because
those operations destroy moire, aliasing and sub-pixel traces.

Output contract
---------------
* Video transforms return ``(C, T, H, W)`` float tensors normalised with the
  mean/std of the backbone checkpoint, which the caller passes in.
* Forensic transforms return ``(C, H, W)`` float tensors in ``[0, 1]``. Each
  forensic model applies its own normalisation internally, so that
  chromaticity and spectrum representations are computed from linear RGB rather
  than from an already standardised tensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter

# ImageNet statistics, the normalisation used by both the official VideoMAEv2
# and the official V-JEPA 2 / 2.1 pipelines. Kept here only as a fallback; the
# authoritative values come from each model's ``preprocessing()``.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Official evaluation resize convention of both backbones: the shorter side is
# resized to ``crop_size * 256 / 224`` before centre cropping.
EVAL_RESIZE_RATIO = 256.0 / 224.0


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


@dataclass(frozen=True)
class ClipAugmentConfig:
    """Clip-consistent augmentation strength for the video branch.

    Defaults are intentionally mild. Strong RandAugment, CutMix, heavy
    RandomErasing and blur are excluded because they either destroy the
    recapture traces or create new temporal artefacts.

    Attributes:
        crop_size: Output spatial size.
        scale_range: Area fraction range of the random crop.
        ratio_range: Aspect-ratio range of the random crop.
        hflip_prob: Horizontal flip probability (one draw per clip).
        brightness: Maximum relative brightness change.
        contrast: Maximum relative contrast change.
        perspective_prob: Probability of applying a very mild perspective warp.
        perspective_scale: Corner displacement as a fraction of the crop size.
    """

    crop_size: int = 224
    scale_range: tuple[float, float] = (0.7, 1.0)
    ratio_range: tuple[float, float] = (0.9, 1.1111)
    hflip_prob: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1
    perspective_prob: float = 0.2
    perspective_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}.")
        low, high = self.scale_range
        if not 0.0 < low <= high <= 1.0:
            raise ValueError(f"Invalid scale_range: {self.scale_range}.")
        if not 0.0 < self.ratio_range[0] <= self.ratio_range[1]:
            raise ValueError(f"Invalid ratio_range: {self.ratio_range}.")


@dataclass(frozen=True)
class PatchAugmentConfig:
    """Augmentation strength for the forensic branch.

    Attributes:
        hflip_prob: Horizontal flip probability.
        vflip_prob: Vertical flip probability.
        rot90_prob: Probability of a multiple-of-90-degree rotation. Off by
            default: rotating changes the orientation of display-camera moire,
            which is a discriminative cue rather than a nuisance.
        spectral_augment_prob: Probability of applying
            :func:`fmag_spectral_augment`.
        spectral_augment_alpha_range: ``alpha`` range of the spectral
            perturbation.
        spectral_augment_beta_std: Standard deviation of the additive spectral
            noise.
        spectral_augment_keep_outside: Keep the unperturbed amplitude outside
            the peak bands in addition to the perturbed one.
    """

    hflip_prob: float = 0.5
    vflip_prob: float = 0.0
    rot90_prob: float = 0.0
    spectral_augment_prob: float = 0.0
    spectral_augment_alpha_range: tuple[float, float] = (0.5, 0.8)
    spectral_augment_beta_std: float = 1.0
    spectral_augment_keep_outside: bool = False


# Shared helpers --------------------------------------------------------------


def _to_float_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert ``(T, H, W, C)`` or ``(H, W, C)`` uint8 into a float tensor in [0, 1]."""
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
    if tensor.ndim == 4:  # (T, H, W, C) -> (C, T, H, W)
        return tensor.permute(3, 0, 1, 2).contiguous()
    if tensor.ndim == 3:  # (H, W, C) -> (C, H, W)
        return tensor.permute(2, 0, 1).contiguous()
    raise ValueError(f"Expected a 3-D or 4-D array, got shape {frames.shape}.")


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Normalise a ``(C, ...)`` tensor in place-safe fashion."""
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype).reshape(
        -1, *([1] * (tensor.ndim - 1))
    )
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype).reshape(
        -1, *([1] * (tensor.ndim - 1))
    )
    if torch.any(std_tensor == 0):
        raise ValueError(f"Normalisation std must be non-zero, got {std}.")
    return (tensor - mean_tensor) / std_tensor


def resize_shorter_side(frame: np.ndarray, target: int) -> np.ndarray:
    """Resize so that the shorter side equals ``target``, preserving aspect ratio."""
    height, width = frame.shape[:2]
    if min(height, width) == target:
        return frame

    scale = target / float(min(height, width))
    new_size = (max(target, int(round(width * scale))), max(target, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interpolation)


def center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Centre crop to ``size`` x ``size``, assuming both sides are large enough."""
    height, width = frame.shape[:2]
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    return frame[top : top + size, left : left + size]


def pad_to_min_size(frame: np.ndarray, size: int) -> np.ndarray:
    """Reflect-pad a frame so both sides are at least ``size``.

    Reflection padding is used instead of upscaling so that no interpolation is
    introduced into a forensic patch.
    """
    height, width = frame.shape[:2]
    pad_y = max(size - height, 0)
    pad_x = max(size - width, 0)
    if pad_y == 0 and pad_x == 0:
        return frame

    return cv2.copyMakeBorder(
        frame,
        pad_y // 2,
        pad_y - pad_y // 2,
        pad_x // 2,
        pad_x - pad_x // 2,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _sample_crop_box(
    height: int,
    width: int,
    config: ClipAugmentConfig,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Draw one crop box ``(top, left, crop_h, crop_w)`` shared by the whole clip."""
    area = float(height * width)
    for _ in range(10):
        target_area = area * float(rng.uniform(*config.scale_range))
        log_ratio = np.log(config.ratio_range)
        aspect = float(np.exp(rng.uniform(log_ratio[0], log_ratio[1])))

        crop_w = int(round(np.sqrt(target_area * aspect)))
        crop_h = int(round(np.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = int(rng.integers(0, height - crop_h + 1))
            left = int(rng.integers(0, width - crop_w + 1))
            return top, left, crop_h, crop_w

    side = min(height, width)
    return (height - side) // 2, (width - side) // 2, side, side


def _perspective_matrix(
    size: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one mild perspective matrix shared by the whole clip."""
    magnitude = scale * size
    source = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    offsets = rng.uniform(-magnitude, magnitude, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(source, source + offsets)


class ClipTransform:
    """Clip-consistent training augmentation for the video branch."""

    def __init__(
        self,
        *,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        config: ClipAugmentConfig | None = None,
    ) -> None:
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.config = config or ClipAugmentConfig()

    @property
    def crop_size(self) -> int:
        return self.config.crop_size

    def __call__(
        self,
        frames: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        """Augment one clip.

        Args:
            frames: ``(T, H, W, 3)`` uint8 RGB clip.
            rng: Generator used for the single per-clip parameter draw.

        Returns:
            Normalised ``(3, T, crop_size, crop_size)`` float tensor.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected a (T, H, W, 3) clip, got shape {frames.shape}.")

        generator = _as_rng(rng)
        config = self.config
        size = config.crop_size
        height, width = frames.shape[1:3]

        # One parameter draw for the entire clip.
        top, left, crop_h, crop_w = _sample_crop_box(height, width, config, generator)
        do_hflip = bool(generator.random() < config.hflip_prob)
        brightness = 1.0 + float(generator.uniform(-config.brightness, config.brightness))
        contrast = 1.0 + float(generator.uniform(-config.contrast, config.contrast))
        do_perspective = bool(generator.random() < config.perspective_prob)
        perspective = (
            _perspective_matrix(size, config.perspective_scale, generator)
            if do_perspective
            else None
        )
        interpolation = (
            cv2.INTER_AREA if crop_h >= size or crop_w >= size else cv2.INTER_LINEAR
        )

        processed = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            crop = frame[top : top + crop_h, left : left + crop_w]
            crop = cv2.resize(crop, (size, size), interpolation=interpolation)
            if do_hflip:
                crop = crop[:, ::-1]
            if perspective is not None:
                crop = cv2.warpPerspective(
                    crop,
                    perspective,
                    (size, size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            processed[index] = crop

        tensor = _to_float_tensor(processed)
        # Photometric adjustment with clip-constant factors.
        tensor = tensor.mul_(contrast).add_(brightness - 1.0).clamp_(0.0, 1.0)
        return normalize_tensor(tensor, self.mean, self.std)


class ValClipTransform:
    """Deterministic resize + centre crop for validation and inference."""

    def __init__(
        self,
        *,
        crop_size: int = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        resize_ratio: float = EVAL_RESIZE_RATIO,
    ) -> None:
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}.")
        self.crop_size = int(crop_size)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.short_side = int(round(self.crop_size * float(resize_ratio)))

    def __call__(
        self,
        frames: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        del rng  # Deterministic.
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected a (T, H, W, 3) clip, got shape {frames.shape}.")

        size = self.crop_size
        processed = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            resized = resize_shorter_side(frame, self.short_side)
            processed[index] = center_crop(resized, size)

        return normalize_tensor(_to_float_tensor(processed), self.mean, self.std)


# Forensic transforms ---------------------------------------------------------


def _peak_mask(
    amplitude: np.ndarray,
    *,
    filter_size: int = 30,
    relative_threshold: float = 0.6,
    ring_radius: int = 10,
    border_margin: int = 15,
    min_radius: int = 5,
) -> np.ndarray:
    """Locate moire peak bands in a centred amplitude spectrum.

    Verified against the official LC&DF implementation
    (``chenlewis/LC-DF-For-DPAD``, ``src/engine/FMAG.py``): local maxima of the
    log amplitude are found with a maximum filter, thresholded at
    ``relative_threshold`` times the local maximum, the DC neighbourhood and the
    border are excluded, and each surviving peak contributes a four-fold
    symmetric set of filled discs at its radial distance from the centre.

    Args:
        amplitude: Centred amplitude spectrum of one channel, ``(H, W)``.
        filter_size: Maximum-filter window size.
        relative_threshold: Fraction of the local maximum a peak must reach.
        ring_radius: Radius of the discs drawn around each peak position.
        border_margin: Border width excluded from peak search.
        min_radius: Minimum radial distance from DC for a peak to count.

    Returns:
        Float mask in ``{0, 1}`` with the same shape as ``amplitude``.
    """
    height, width = amplitude.shape
    centre_y, centre_x = height // 2, width // 2

    spectrum = np.log(amplitude + 1e-6)
    local_max = maximum_filter(spectrum, size=filter_size)
    candidates = (spectrum == local_max) & (spectrum > relative_threshold * local_max)

    candidates[:border_margin, :] = False
    candidates[-border_margin:, :] = False
    candidates[:, :border_margin] = False
    candidates[:, -border_margin:] = False

    mask = np.zeros((height, width), dtype=np.uint8)
    ys, xs = np.nonzero(candidates)
    for peak_y, peak_x in zip(ys.tolist(), xs.tolist()):
        radius = int(round(float(np.hypot(peak_x - centre_x, peak_y - centre_y))))
        if radius <= min_radius:
            continue
        for offset in (radius, -radius):
            cv2.circle(mask, (centre_x + offset, centre_y), ring_radius, 1, -1)
            cv2.circle(mask, (centre_x, centre_y + offset), ring_radius, 1, -1)

    return mask.astype(np.float32)


def fmag_spectral_augment(
    patch: np.ndarray,
    *,
    alpha_range: tuple[float, float] = (0.5, 0.8),
    beta_std: float = 1.0,
    keep_outside: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """FMAG-inspired moire spectral augmentation of one RGB patch.

    Taken from the papers / official code:
        * Amplitude-only, phase-preserving manipulation: forward FFT with
          ``fftshift``, perturb the amplitude, recombine with the original
          phase, inverse FFT and take the modulus.
        * Moire peak bands are located automatically and left intact while the
          remaining amplitudes are attenuated by ``alpha ~ U(0.5, 0.8)`` and
          jittered by ``beta ~ N(0, 1)``, which relatively enhances the moire
          peaks.
        * Per-channel processing.
        (Verified against ``chenlewis/FMAG-with-MMFM`` ``FMAG.py`` and
        ``chenlewis/LC-DF-For-DPAD`` ``src/engine/FMAG.py``. The released FMAG
        ``add_noise`` drops the outside-band passthrough that the earlier FHAG
        implementation keeps; ``keep_outside`` exposes both variants and
        defaults to the shipped FMAG behaviour.)

    Adapted for Stage 1:
        * The device-derived peak seed of the original FMAG (``raw_peak=95`` for
          their capture rig) is replaced by the automatic peak detection of the
          LC&DF release, because our recapture devices are unknown and the
          closed-form seed derivation is not published.
        * Applied to a 256x256 native-resolution patch instead of their 224
          document crop.

    Args:
        patch: ``(H, W, 3)`` uint8 RGB patch.
        alpha_range: Multiplicative range applied outside the peak bands.
        beta_std: Standard deviation of the additive amplitude noise.
        keep_outside: Also keep the unperturbed outside-band amplitude.
        rng: Random generator.

    Returns:
        A ``(H, W, 3)`` uint8 patch.
    """
    if patch.ndim != 3 or patch.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) patch, got shape {patch.shape}.")

    generator = _as_rng(rng)
    source = patch.astype(np.float32)
    output = np.empty_like(source)

    for channel in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(source[..., channel]))
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        mask = _peak_mask(amplitude)
        inverse_mask = 1.0 - mask

        alpha = generator.uniform(*alpha_range, size=amplitude.shape).astype(np.float32)
        beta = generator.normal(0.0, beta_std, size=amplitude.shape).astype(np.float32)
        outside = amplitude * inverse_mask
        perturbed = (alpha * outside + beta) * inverse_mask
        if keep_outside:
            perturbed = perturbed + outside

        new_amplitude = np.clip(amplitude * mask + perturbed, 0.0, None)
        restored = np.fft.ifft2(np.fft.ifftshift(new_amplitude * np.exp(1j * phase)))
        output[..., channel] = np.abs(restored)

    return np.clip(output, 0.0, 255.0).astype(np.uint8)


class ForensicPatchTransform:
    """Augmentation and tensor conversion for native-resolution forensic patches.

    Never resizes. Returns a ``(3, H, W)`` float tensor in ``[0, 1]``; each
    forensic model applies its own normalisation so that chromaticity and
    spectral representations are computed from linear RGB.
    """

    def __init__(
        self,
        *,
        train: bool,
        config: PatchAugmentConfig | None = None,
    ) -> None:
        self.train = bool(train)
        self.config = config or PatchAugmentConfig()

    def __call__(
        self,
        patch: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        if patch.ndim != 3 or patch.shape[-1] != 3:
            raise ValueError(f"Expected an (H, W, 3) patch, got shape {patch.shape}.")

        if not self.train:
            return _to_float_tensor(patch)

        generator = _as_rng(rng)
        config = self.config
        augmented = patch

        if config.spectral_augment_prob > 0.0 and generator.random() < config.spectral_augment_prob:
            augmented = fmag_spectral_augment(
                augmented,
                alpha_range=config.spectral_augment_alpha_range,
                beta_std=config.spectral_augment_beta_std,
                keep_outside=config.spectral_augment_keep_outside,
                rng=generator,
            )

        if generator.random() < config.hflip_prob:
            augmented = augmented[:, ::-1]
        if generator.random() < config.vflip_prob:
            augmented = augmented[::-1, :]
        if config.rot90_prob > 0.0 and generator.random() < config.rot90_prob:
            augmented = np.rot90(augmented, k=int(generator.integers(1, 4)))

        return _to_float_tensor(np.ascontiguousarray(augmented))


def build_video_transforms(
    *,
    crop_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    train_config: ClipAugmentConfig | None = None,
    resize_ratio: float = EVAL_RESIZE_RATIO,
) -> tuple[ClipTransform, ValClipTransform]:
    """Return matching train/validation video transforms for one backbone."""
    config = train_config or ClipAugmentConfig(crop_size=crop_size)
    if config.crop_size != crop_size:
        raise ValueError(
            f"train_config.crop_size={config.crop_size} does not match "
            f"crop_size={crop_size}."
        )

    return (
        ClipTransform(mean=mean, std=std, config=config),
        ValClipTransform(
            crop_size=crop_size, mean=mean, std=std, resize_ratio=resize_ratio
        ),
    )


def build_forensic_transforms(
    *,
    train_config: PatchAugmentConfig | None = None,
) -> tuple[ForensicPatchTransform, ForensicPatchTransform]:
    """Return matching train/validation forensic patch transforms."""
    return (
        ForensicPatchTransform(train=True, config=train_config),
        ForensicPatchTransform(train=False, config=train_config),
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "EVAL_RESIZE_RATIO",
    "ClipAugmentConfig",
    "PatchAugmentConfig",
    "normalize_tensor",
    "resize_shorter_side",
    "center_crop",
    "pad_to_min_size",
    "ClipTransform",
    "ValClipTransform",
    "fmag_spectral_augment",
    "ForensicPatchTransform",
    "build_video_transforms",
    "build_forensic_transforms",
]

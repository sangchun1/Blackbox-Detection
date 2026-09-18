from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


@dataclass(frozen=True)
class CommonNuisanceConfig:
    """Class-agnostic nuisances applied to both CCD ORIGINAL and CCD SYN-RR.

    The point is to prevent the classifier from learning trivial shortcuts such as
    "blur/compression/noise => rerecorded". These ranges are intentionally mild.
    """

    prob: float = 0.65
    resample_prob: float = 0.20
    resample_scale_min: float = 0.90
    resample_scale_max: float = 0.995
    gamma_prob: float = 0.25
    gamma_min: float = 0.96
    gamma_max: float = 1.04
    gain_min: float = 0.98
    gain_max: float = 1.02
    blur_prob: float = 0.20
    blur_sigma_min: float = 0.10
    blur_sigma_max: float = 0.55
    noise_prob: float = 0.20
    noise_std_min: float = 0.001
    noise_std_max: float = 0.005
    quantize_prob: float = 0.15
    quantize_levels_min: int = 96
    quantize_levels_max: int = 224


@dataclass(frozen=True)
class StrengthProfile:
    """One recapture severity profile.

    All values operate in normalized image coordinates or [0, 1] RGB space.
    """

    # virtual display / camera resampling
    resample_scale_min: float
    resample_scale_max: float
    perspective_prob: float
    perspective_scale_max: float

    # display / ISP colour response
    gamma_min: float
    gamma_max: float
    exposure_gain_min: float
    exposure_gain_max: float
    channel_gain_delta: float
    crosstalk_max: float

    # recapture-specific spectral/chromatic cues
    moire_prob: float
    moire_components_min: int
    moire_components_max: int
    moire_amp_min: float
    moire_amp_max: float
    moire_freq_min: float
    moire_freq_max: float
    moire_phase_speed_max: float
    moire_channel_phase_max: float

    # display refresh / rolling banding
    banding_prob: float
    band_amp_min: float
    band_amp_max: float
    band_freq_min: float
    band_freq_max: float
    band_phase_speed_max: float

    # optics / sensor / compression proxy
    blur_prob: float
    blur_sigma_min: float
    blur_sigma_max: float
    noise_prob: float
    noise_std_min: float
    noise_std_max: float
    recompress_prob: float
    recompress_scale_min: float
    recompress_scale_max: float
    recompress_levels_min: int
    recompress_levels_max: int


@dataclass(frozen=True)
class RecaptureSimConfig:
    """Configuration for RecaptureSimV1.

    V1 is deliberately conservative: weak/medium samples dominate, and the
    nuisance family shared by both classes is separated from recapture-specific
    cues. This is meant for domain adaptation, not for producing visually obvious
    fake videos.
    """

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    common: CommonNuisanceConfig = field(default_factory=CommonNuisanceConfig)
    severity_probs: tuple[float, float, float] = (0.45, 0.40, 0.15)

    weak: StrengthProfile = field(
        default_factory=lambda: StrengthProfile(
            resample_scale_min=0.88,
            resample_scale_max=0.985,
            perspective_prob=0.35,
            perspective_scale_max=0.012,
            gamma_min=0.94,
            gamma_max=1.07,
            exposure_gain_min=0.96,
            exposure_gain_max=1.04,
            channel_gain_delta=0.025,
            crosstalk_max=0.008,
            moire_prob=0.35,
            moire_components_min=1,
            moire_components_max=2,
            moire_amp_min=0.0025,
            moire_amp_max=0.010,
            moire_freq_min=0.020,
            moire_freq_max=0.085,
            moire_phase_speed_max=0.12,
            moire_channel_phase_max=0.22,
            banding_prob=0.20,
            band_amp_min=0.002,
            band_amp_max=0.009,
            band_freq_min=0.004,
            band_freq_max=0.018,
            band_phase_speed_max=0.25,
            blur_prob=0.40,
            blur_sigma_min=0.10,
            blur_sigma_max=0.55,
            noise_prob=0.30,
            noise_std_min=0.001,
            noise_std_max=0.004,
            recompress_prob=0.20,
            recompress_scale_min=0.88,
            recompress_scale_max=0.98,
            recompress_levels_min=112,
            recompress_levels_max=224,
        )
    )
    medium: StrengthProfile = field(
        default_factory=lambda: StrengthProfile(
            resample_scale_min=0.75,
            resample_scale_max=0.965,
            perspective_prob=0.60,
            perspective_scale_max=0.022,
            gamma_min=0.89,
            gamma_max=1.13,
            exposure_gain_min=0.92,
            exposure_gain_max=1.08,
            channel_gain_delta=0.050,
            crosstalk_max=0.015,
            moire_prob=0.58,
            moire_components_min=1,
            moire_components_max=3,
            moire_amp_min=0.004,
            moire_amp_max=0.020,
            moire_freq_min=0.025,
            moire_freq_max=0.130,
            moire_phase_speed_max=0.22,
            moire_channel_phase_max=0.40,
            banding_prob=0.35,
            band_amp_min=0.003,
            band_amp_max=0.016,
            band_freq_min=0.004,
            band_freq_max=0.024,
            band_phase_speed_max=0.45,
            blur_prob=0.58,
            blur_sigma_min=0.15,
            blur_sigma_max=0.85,
            noise_prob=0.45,
            noise_std_min=0.0015,
            noise_std_max=0.006,
            recompress_prob=0.33,
            recompress_scale_min=0.78,
            recompress_scale_max=0.96,
            recompress_levels_min=80,
            recompress_levels_max=192,
        )
    )
    strong: StrengthProfile = field(
        default_factory=lambda: StrengthProfile(
            resample_scale_min=0.62,
            resample_scale_max=0.93,
            perspective_prob=0.78,
            perspective_scale_max=0.032,
            gamma_min=0.84,
            gamma_max=1.20,
            exposure_gain_min=0.86,
            exposure_gain_max=1.14,
            channel_gain_delta=0.075,
            crosstalk_max=0.022,
            moire_prob=0.72,
            moire_components_min=2,
            moire_components_max=3,
            moire_amp_min=0.006,
            moire_amp_max=0.032,
            moire_freq_min=0.030,
            moire_freq_max=0.170,
            moire_phase_speed_max=0.32,
            moire_channel_phase_max=0.60,
            banding_prob=0.52,
            band_amp_min=0.005,
            band_amp_max=0.024,
            band_freq_min=0.005,
            band_freq_max=0.030,
            band_phase_speed_max=0.70,
            blur_prob=0.72,
            blur_sigma_min=0.25,
            blur_sigma_max=1.10,
            noise_prob=0.55,
            noise_std_min=0.002,
            noise_std_max=0.009,
            recompress_prob=0.45,
            recompress_scale_min=0.68,
            recompress_scale_max=0.92,
            recompress_levels_min=64,
            recompress_levels_max=160,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecaptureMode:
    apply_common: bool
    apply_recapture: bool
    strength: str | None = None


class RecaptureSimV1:
    """On-the-fly, temporally coherent screen-recapture approximation.

    Expected input is *already normalized* RGB video with one of these layouts:
      - C,T,H,W
      - N,C,T,H,W  (N clips)
      - T,C,H,W

    The simulator temporarily converts the tensor back to [0, 1], applies the
    synthetic camera/display pipeline, then re-applies the original normalization.
    It therefore plugs into Stage1VideoDataset without modifying the repository's
    decoder or video transforms.
    """

    def __init__(self, config: RecaptureSimConfig | None = None):
        self.config = config or RecaptureSimConfig()

    @staticmethod
    def _rand(generator: torch.Generator | None = None) -> float:
        return float(torch.rand((), generator=generator).item())

    @classmethod
    def _uniform(
        cls,
        lo: float,
        hi: float,
        generator: torch.Generator | None = None,
    ) -> float:
        if hi <= lo:
            return float(lo)
        return float(lo + (hi - lo) * cls._rand(generator))

    @classmethod
    def _randint(
        cls,
        lo: int,
        hi_inclusive: int,
        generator: torch.Generator | None = None,
    ) -> int:
        if hi_inclusive <= lo:
            return int(lo)
        return int(torch.randint(lo, hi_inclusive + 1, (), generator=generator).item())

    @staticmethod
    def _to_ncthw(x: torch.Tensor) -> tuple[torch.Tensor, str]:
        if x.ndim == 4:
            if x.shape[0] == 3:  # C,T,H,W
                return x.unsqueeze(0), "cthw"
            if x.shape[1] == 3:  # T,C,H,W
                return x.permute(1, 0, 2, 3).unsqueeze(0), "tchw"
        elif x.ndim == 5 and x.shape[1] == 3:  # N,C,T,H,W
            return x, "ncthw"
        raise ValueError(
            "RecaptureSimV1 expected C,T,H,W; T,C,H,W; or N,C,T,H,W, "
            f"got {tuple(x.shape)}"
        )

    @staticmethod
    def _restore_layout(x: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "cthw":
            return x[0]
        if layout == "tchw":
            return x[0].permute(1, 0, 2, 3)
        if layout == "ncthw":
            return x
        raise ValueError(layout)

    def _mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.new_tensor(self.config.mean).view(1, 3, 1, 1, 1)
        std = x.new_tensor(self.config.std).view(1, 3, 1, 1, 1)
        return mean, std

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._mean_std(x)
        return (x * std + mean).clamp(0.0, 1.0)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._mean_std(x)
        return (x.clamp(0.0, 1.0) - mean) / std

    @staticmethod
    def _flatten_frames(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        n, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
        return frames, n, t

    @staticmethod
    def _unflatten_frames(frames: torch.Tensor, n: int, t: int) -> torch.Tensor:
        _, c, h, w = frames.shape
        return frames.reshape(n, t, c, h, w).permute(0, 2, 1, 3, 4)

    def _resample_roundtrip(
        self,
        x: torch.Tensor,
        scale: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        frames, n, t = self._flatten_frames(x)
        h, w = frames.shape[-2:]
        target_h = max(32, int(round(h * scale)))
        target_w = max(32, int(round(w * scale)))

        modes = ("bilinear", "bicubic", "area")
        down_mode = modes[self._randint(0, len(modes) - 1, generator)]
        if down_mode == "area":
            small = F.interpolate(frames, size=(target_h, target_w), mode="area")
        else:
            small = F.interpolate(
                frames,
                size=(target_h, target_w),
                mode=down_mode,
                align_corners=False,
                antialias=True,
            )

        up_mode = "bilinear" if self._rand(generator) < 0.55 else "bicubic"
        restored = F.interpolate(
            small,
            size=(h, w),
            mode=up_mode,
            align_corners=False,
            antialias=True,
        )
        return self._unflatten_frames(restored, n, t)

    def _perspective(
        self,
        x: torch.Tensor,
        distortion_scale: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        frames, n, t = self._flatten_frames(x)
        h, w = frames.shape[-2:]
        max_dx = max(1, int(round(w * distortion_scale)))
        max_dy = max(1, int(round(h * distortion_scale)))
        pad = max(max_dx, max_dy) + 3

        # Reflect-pad before warping and crop back afterwards. This avoids the
        # synthetic-only black triangular borders that a naive perspective warp
        # would create after the 384x384 crop.
        frames = F.pad(frames, (pad, pad, pad, pad), mode="reflect")
        hp, wp = frames.shape[-2:]

        start = [[0, 0], [wp - 1, 0], [wp - 1, hp - 1], [0, hp - 1]]
        end = [
            [
                self._randint(0, max_dx, generator),
                self._randint(0, max_dy, generator),
            ],
            [
                wp - 1 - self._randint(0, max_dx, generator),
                self._randint(0, max_dy, generator),
            ],
            [
                wp - 1 - self._randint(0, max_dx, generator),
                hp - 1 - self._randint(0, max_dy, generator),
            ],
            [
                self._randint(0, max_dx, generator),
                hp - 1 - self._randint(0, max_dy, generator),
            ],
        ]
        warped = TVF.perspective(
            frames,
            startpoints=start,
            endpoints=end,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        warped = warped[..., pad : pad + h, pad : pad + w]
        return self._unflatten_frames(warped, n, t)

    def _display_response(
        self,
        x: torch.Tensor,
        profile: StrengthProfile,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        gamma = self._uniform(profile.gamma_min, profile.gamma_max, generator)
        exposure = self._uniform(
            profile.exposure_gain_min,
            profile.exposure_gain_max,
            generator,
        )
        x = x.clamp_min(1e-6).pow(gamma) * exposure

        delta = profile.channel_gain_delta
        gains = x.new_tensor(
            [
                self._uniform(1.0 - delta, 1.0 + delta, generator),
                self._uniform(1.0 - delta, 1.0 + delta, generator),
                self._uniform(1.0 - delta, 1.0 + delta, generator),
            ]
        ).view(1, 3, 1, 1, 1)
        x = x * gains

        # Mild RGB cross-talk. Diagonal stays at one; off-diagonal terms are small.
        ct = profile.crosstalk_max
        matrix = torch.eye(3, device=x.device, dtype=x.dtype)
        if ct > 0:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        matrix[i, j] = self._uniform(-ct, ct, generator)
        x = torch.einsum("ij,njthw->nithw", matrix, x)
        return x.clamp(0.0, 1.0)

    def _moire(
        self,
        x: torch.Tensor,
        profile: StrengthProfile,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        n, c, t, h, w = x.shape
        yy = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, 1, h, 1)
        xx = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, 1, w)
        tt = torch.arange(t, device=x.device, dtype=x.dtype).view(1, 1, t, 1, 1)

        pattern = torch.zeros((1, 3, t, h, w), device=x.device, dtype=x.dtype)
        components = self._randint(
            profile.moire_components_min,
            profile.moire_components_max,
            generator,
        )

        for _ in range(components):
            freq = self._uniform(profile.moire_freq_min, profile.moire_freq_max, generator)
            angle = self._uniform(0.0, math.pi, generator)
            fx = freq * math.cos(angle)
            fy = freq * math.sin(angle)
            amp = self._uniform(profile.moire_amp_min, profile.moire_amp_max, generator)
            base_phase = self._uniform(0.0, 2.0 * math.pi, generator)
            speed = self._uniform(
                -profile.moire_phase_speed_max,
                profile.moire_phase_speed_max,
                generator,
            )
            channel_offsets = x.new_tensor(
                [
                    self._uniform(-profile.moire_channel_phase_max, profile.moire_channel_phase_max, generator),
                    self._uniform(-profile.moire_channel_phase_max, profile.moire_channel_phase_max, generator),
                    self._uniform(-profile.moire_channel_phase_max, profile.moire_channel_phase_max, generator),
                ]
            ).view(1, 3, 1, 1, 1)

            phase = 2.0 * math.pi * (fx * xx + fy * yy) + base_phase + speed * tt
            pattern = pattern + amp * torch.sin(phase + channel_offsets)

        return (x + pattern).clamp(0.0, 1.0)

    def _banding(
        self,
        x: torch.Tensor,
        profile: StrengthProfile,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        _, _, t, h, _ = x.shape
        yy = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, 1, h, 1)
        tt = torch.arange(t, device=x.device, dtype=x.dtype).view(1, 1, t, 1, 1)
        freq = self._uniform(profile.band_freq_min, profile.band_freq_max, generator)
        amp = self._uniform(profile.band_amp_min, profile.band_amp_max, generator)
        phase0 = self._uniform(0.0, 2.0 * math.pi, generator)
        speed = self._uniform(
            -profile.band_phase_speed_max,
            profile.band_phase_speed_max,
            generator,
        )
        band = amp * torch.sin(2.0 * math.pi * freq * yy + phase0 + speed * tt)
        return (x * (1.0 + band)).clamp(0.0, 1.0)

    def _blur(
        self,
        x: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        frames, n, t = self._flatten_frames(x)
        kernel = max(3, int(round(sigma * 6.0)) | 1)
        kernel = min(kernel, 9)
        blurred = TVF.gaussian_blur(frames, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
        return self._unflatten_frames(blurred, n, t)

    def _noise(
        self,
        x: torch.Tensor,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        # torch.randn_like does not accept a generator on some torch versions.
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) * std
        return (x + noise).clamp(0.0, 1.0)

    def _quantize(
        self,
        x: torch.Tensor,
        levels: int,
    ) -> torch.Tensor:
        levels = max(8, int(levels))
        return (torch.round(x * (levels - 1)) / float(levels - 1)).clamp(0.0, 1.0)

    def _recompression_proxy(
        self,
        x: torch.Tensor,
        profile: StrengthProfile,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        scale = self._uniform(
            profile.recompress_scale_min,
            profile.recompress_scale_max,
            generator,
        )
        x = self._resample_roundtrip(x, scale=scale, generator=generator)
        levels = self._randint(
            profile.recompress_levels_min,
            profile.recompress_levels_max,
            generator,
        )
        return self._quantize(x, levels)

    def _apply_common(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        cfg = self.config.common
        if self._rand(generator) >= cfg.prob:
            return x

        if self._rand(generator) < cfg.resample_prob:
            scale = self._uniform(cfg.resample_scale_min, cfg.resample_scale_max, generator)
            x = self._resample_roundtrip(x, scale, generator)

        if self._rand(generator) < cfg.gamma_prob:
            gamma = self._uniform(cfg.gamma_min, cfg.gamma_max, generator)
            gain = self._uniform(cfg.gain_min, cfg.gain_max, generator)
            x = (x.clamp_min(1e-6).pow(gamma) * gain).clamp(0.0, 1.0)

        if self._rand(generator) < cfg.blur_prob:
            sigma = self._uniform(cfg.blur_sigma_min, cfg.blur_sigma_max, generator)
            x = self._blur(x, sigma)

        if self._rand(generator) < cfg.noise_prob:
            std = self._uniform(cfg.noise_std_min, cfg.noise_std_max, generator)
            x = self._noise(x, std, generator)

        if self._rand(generator) < cfg.quantize_prob:
            levels = self._randint(cfg.quantize_levels_min, cfg.quantize_levels_max, generator)
            x = self._quantize(x, levels)

        return x

    def _profile(self, strength: str) -> StrengthProfile:
        strength = str(strength).strip().lower()
        if strength == "weak":
            return self.config.weak
        if strength == "medium":
            return self.config.medium
        if strength == "strong":
            return self.config.strong
        raise ValueError(f"Unknown recapture strength: {strength!r}")

    def _sample_strength(self, generator: torch.Generator | None) -> str:
        probs = self.config.severity_probs
        if len(probs) != 3 or min(probs) < 0 or sum(probs) <= 0:
            raise ValueError(f"Invalid severity_probs: {probs}")
        total = float(sum(probs))
        u = self._rand(generator) * total
        if u < probs[0]:
            return "weak"
        if u < probs[0] + probs[1]:
            return "medium"
        return "strong"

    def _apply_recapture(
        self,
        x: torch.Tensor,
        strength: str,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        p = self._profile(strength)

        # Display response first; then camera/sampling stages.
        x = self._display_response(x, p, generator)

        scale = self._uniform(p.resample_scale_min, p.resample_scale_max, generator)
        x = self._resample_roundtrip(x, scale, generator)

        if self._rand(generator) < p.perspective_prob:
            distortion = self._uniform(0.003, p.perspective_scale_max, generator)
            x = self._perspective(x, distortion, generator)

        if self._rand(generator) < p.moire_prob:
            x = self._moire(x, p, generator)

        if self._rand(generator) < p.banding_prob:
            x = self._banding(x, p, generator)

        if self._rand(generator) < p.blur_prob:
            sigma = self._uniform(p.blur_sigma_min, p.blur_sigma_max, generator)
            x = self._blur(x, sigma)

        if self._rand(generator) < p.noise_prob:
            std = self._uniform(p.noise_std_min, p.noise_std_max, generator)
            x = self._noise(x, std, generator)

        if self._rand(generator) < p.recompress_prob:
            x = self._recompression_proxy(x, p, generator)

        return x.clamp(0.0, 1.0)

    def __call__(
        self,
        pixels: torch.Tensor,
        *,
        apply_common: bool = True,
        apply_recapture: bool = False,
        strength: str | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(pixels):
            raise TypeError(f"pixels must be a torch.Tensor, got {type(pixels)!r}")

        x, layout = self._to_ncthw(pixels)
        original_dtype = x.dtype
        x = x.float()
        x = self._denormalize(x)

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=x.device.type)
            generator.manual_seed(int(seed) & 0x7FFFFFFF_FFFFFFFF)

        if apply_common:
            x = self._apply_common(x, generator)

        if apply_recapture:
            strength = strength or self._sample_strength(generator)
            x = self._apply_recapture(x, strength, generator)

        x = self._normalize(x)
        x = self._restore_layout(x, layout)
        return x.to(dtype=original_dtype)


def stable_seed(text: str, base_seed: int = 42) -> int:
    payload = f"{int(base_seed)}::{text}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF_FFFFFFFF


class OnTheFlyRecaptureDataset(Dataset):
    """Wrap Stage1VideoDataset and conditionally synthesize CCD re-recordings.

    `mode_by_video_id` maps the manifest video_id to a RecaptureMode. The base
    dataset is responsible for decoding/sampling/cropping/normalization. This
    wrapper only changes the returned `pixels` tensor.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        mode_by_video_id: Mapping[str, RecaptureMode],
        simulator: RecaptureSimV1,
        *,
        deterministic: bool,
        base_seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.mode_by_video_id = dict(mode_by_video_id)
        self.simulator = simulator
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _extract_video_id(sample: Mapping[str, Any]) -> str:
        for key in ("video_id", "ID", "id"):
            if key in sample:
                value = sample[key]
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                return str(value)
        raise KeyError(
            "Base Stage1VideoDataset sample does not expose video_id/ID/id. "
            "The recapture wrapper needs a stable identifier to select CCD modes."
        )

    def __getitem__(self, index: int) -> Any:
        sample = self.base_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError(
                "OnTheFlyRecaptureDataset expects mapping samples from Stage1VideoDataset, "
                f"got {type(sample)!r}"
            )

        video_id = self._extract_video_id(sample)
        mode = self.mode_by_video_id.get(video_id)
        if mode is None:
            return sample

        # Respect decoder-invalid examples when the base dataset exposes a flag.
        if "valid" in sample and not bool(sample["valid"]):
            return sample

        if "pixels" not in sample:
            raise KeyError("Base sample is missing the 'pixels' tensor.")

        if self.deterministic:
            seed = stable_seed(video_id, self.base_seed)
        else:
            # The DataLoader worker seed controls torch's global CPU RNG, so this
            # produces a reproducible but epoch-varying stream when workers are seeded.
            seed = int(torch.randint(0, 2**31 - 1, ()).item())

        out = dict(sample)
        out["pixels"] = self.simulator(
            sample["pixels"],
            apply_common=mode.apply_common,
            apply_recapture=mode.apply_recapture,
            strength=mode.strength,
            seed=seed,
        )
        return out

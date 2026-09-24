from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .constants import CAN_TARGETS, IMAGENET_MEAN, IMAGENET_STD, TARGET_TO_VALID
from .schema import read_frame_table


def _load_manifest(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(value)


class MixedStage3CANDataset(Dataset):
    """Stage-3 v5 mixed-domain video/CAN dataset.

    Each source mapping requires:
      - name
      - manifest
      - processed_root

    Optional source-specific steering auxiliary settings:
      - steering_direction_deadzone_deg
      - steering_activity_scale_deg
      - steering_sign_multiplier

    Continuous CAN targets always use the canonical ``target_stats`` supplied by
    the caller.  For v5-A this should stay equal to the original comma2k19 train
    statistics so that A2D2 does not silently redefine the regression scale.
    """

    def __init__(
        self,
        sources: Sequence[Mapping],
        *,
        target_stats: str | Path | Mapping,
        clip_len: int = 32,
        window_stride: int = 16,
        input_size: tuple[int, int] = (288, 384),
        random_flip: bool = False,
        seed: int = 20260918,
        max_windows_by_source: Mapping[str, int | None] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("sources must be non-empty")

        self.clip_len = int(clip_len)
        self.window_stride = int(window_stride)
        self.input_h, self.input_w = map(int, input_size)
        self.random_flip = bool(random_flip)
        self.seed = int(seed)

        if isinstance(target_stats, (str, Path)):
            self.stats = json.loads(Path(target_stats).read_text(encoding="utf-8"))
        else:
            self.stats = dict(target_stats)

        self.source_cfg: dict[str, dict] = {}
        frames: list[pd.DataFrame] = []

        for source in sources:
            cfg = dict(source)
            name = str(cfg["name"])
            if name in self.source_cfg:
                raise ValueError(f"duplicate source name: {name}")

            root = Path(cfg["processed_root"])
            manifest = _load_manifest(cfg["manifest"])
            if manifest.empty:
                raise ValueError(f"empty manifest for source {name}")

            manifest = manifest.copy()
            manifest["source_name_v5"] = name
            manifest["processed_root_v5"] = str(root)
            manifest["source_row_v5"] = np.arange(len(manifest), dtype=np.int64)

            self.source_cfg[name] = {
                "steering_direction_deadzone_deg": float(
                    cfg.get("steering_direction_deadzone_deg", 2.0)
                ),
                "steering_activity_scale_deg": max(
                    float(cfg.get("steering_activity_scale_deg", 8.0)),
                    1e-3,
                ),
                "steering_sign_multiplier": float(
                    cfg.get("steering_sign_multiplier", 1.0)
                ),
            }
            frames.append(manifest)

        self.segments = pd.concat(frames, ignore_index=True, sort=False)

        windows_by_source: dict[str, list[tuple[int, int]]] = {
            name: [] for name in self.source_cfg
        }
        for seg_idx, row in enumerate(self.segments.itertuples(index=False)):
            n = int(row.num_frames)
            if n < self.clip_len:
                continue
            starts = list(range(0, n - self.clip_len + 1, self.window_stride))
            if starts[-1] != n - self.clip_len:
                starts.append(n - self.clip_len)
            windows_by_source[str(row.source_name_v5)].extend(
                (seg_idx, int(s)) for s in starts
            )

        max_map = dict(max_windows_by_source or {})
        rng = random.Random(self.seed)
        self.windows: list[tuple[int, int]] = []
        for source_name in self.source_cfg:
            items = windows_by_source[source_name]
            limit = max_map.get(source_name)
            if limit is not None and len(items) > int(limit):
                items = rng.sample(items, int(limit))
                items.sort()
            self.windows.extend(items)

        # Keep segment-major ordering.  This makes event-index construction
        # Drive-friendly because metadata for adjacent windows is reused.
        self.windows.sort(key=lambda x: (x[0], x[1]))

        self.mean = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32
        )[:, None, None, None]
        self.std = torch.tensor(
            IMAGENET_STD, dtype=torch.float32
        )[:, None, None, None]

    def __len__(self) -> int:
        return len(self.windows)

    def fingerprint(self) -> str:
        payload = {
            "clip_len": self.clip_len,
            "window_stride": self.window_stride,
            "segments": [
                {
                    "source": str(row.source_name_v5),
                    "route": str(row.route_id),
                    "segment": str(row.segment_id),
                    "frames": int(row.num_frames),
                    "video": str(row.video_relpath),
                    "meta": str(row.metadata_relpath),
                    "aux": (
                        None
                        if not hasattr(row, "aux_metadata_relpath")
                        or pd.isna(row.aux_metadata_relpath)
                        else str(row.aux_metadata_relpath)
                    ),
                }
                for row in self.segments.itertuples(index=False)
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha1(raw).hexdigest()

    def window_source(self, index: int) -> str:
        seg_idx, _ = self.windows[index]
        return str(self.segments.iloc[seg_idx]["source_name_v5"])

    @lru_cache(maxsize=256)
    def _metadata(self, fullpath: str) -> pd.DataFrame:
        return read_frame_table(fullpath)

    @lru_cache(maxsize=128)
    def _aux_npz(self, fullpath: str) -> dict[str, np.ndarray]:
        with np.load(fullpath, allow_pickle=False) as z:
            return {key: z[key].copy() for key in z.files}

    def _decode(self, path: Path, start: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames: list[np.ndarray] = []
        for _ in range(self.clip_len):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(
                rgb,
                (self.input_w, self.input_h),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(rgb)
        cap.release()
        if not frames:
            raise RuntimeError(f"cannot decode {path} from frame {start}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1].copy())
        return np.stack(frames, axis=0)

    def _source_aux(
        self,
        row: pd.Series,
        meta: pd.DataFrame,
        start: int,
    ) -> dict[str, np.ndarray]:
        n = len(meta)
        source_name = str(row["source_name_v5"])
        cfg = self.source_cfg[source_name]

        steer_value = meta["steering_deg"].to_numpy(dtype=np.float32).copy()
        steer_valid = (
            meta["valid_steer"].to_numpy(dtype=bool)
            & np.isfinite(steer_value)
        )

        brake = np.zeros(n, dtype=np.float32)
        brake_valid = np.zeros(n, dtype=bool)
        throttle = np.zeros(n, dtype=np.float32)
        throttle_valid = np.zeros(n, dtype=bool)

        aux_rel = row.get("aux_metadata_relpath", np.nan)
        if isinstance(aux_rel, str) and aux_rel and aux_rel.lower() != "nan":
            root = Path(str(row["processed_root_v5"]))
            aux = self._aux_npz(str(root / aux_rel))
            sl = slice(start, start + n)

            if "steering_wheel_deg" in aux:
                steer_value = np.asarray(
                    aux["steering_wheel_deg"][sl],
                    dtype=np.float32,
                ).copy()
                steer_valid = np.asarray(
                    aux.get(
                        "valid_steering_wheel",
                        np.isfinite(aux["steering_wheel_deg"]),
                    )[sl],
                    dtype=bool,
                ).copy()
                steer_valid &= np.isfinite(steer_value)

            if "brake_pressure_bar" in aux:
                brake = np.asarray(
                    aux["brake_pressure_bar"][sl],
                    dtype=np.float32,
                ).copy()
                brake_valid = np.asarray(
                    aux.get("valid_brake", np.isfinite(aux["brake_pressure_bar"]))[sl],
                    dtype=bool,
                ).copy()
                brake_valid &= np.isfinite(brake)

            if "accelerator_pedal_pct" in aux:
                throttle = np.asarray(
                    aux["accelerator_pedal_pct"][sl],
                    dtype=np.float32,
                ).copy()
                throttle_valid = np.asarray(
                    aux.get("valid_accelerator", np.isfinite(aux["accelerator_pedal_pct"]))[sl],
                    dtype=bool,
                ).copy()
                throttle_valid &= np.isfinite(throttle)

        steer_value *= float(cfg["steering_sign_multiplier"])
        deadzone = float(cfg["steering_direction_deadzone_deg"])
        steer_class = np.full(n, 1, dtype=np.int64)
        steer_class[steer_value < -deadzone] = 0
        steer_class[steer_value > +deadzone] = 2
        steer_activity = (
            np.abs(steer_value)
            / float(cfg["steering_activity_scale_deg"])
        ).astype(np.float32)

        return {
            "steer_direction_class": steer_class,
            "steer_direction_valid": steer_valid,
            "steer_activity": steer_activity,
            "steer_activity_valid": steer_valid.copy(),
            "brake_pressure_bar": brake,
            "brake_valid": brake_valid,
            "accelerator_pedal_pct": throttle,
            "accelerator_valid": throttle_valid,
        }

    def __getitem__(self, index: int) -> dict:
        seg_idx, start = self.windows[index]
        row = self.segments.iloc[seg_idx]
        root = Path(str(row["processed_root_v5"]))
        meta_full = self._metadata(str(root / row.metadata_relpath))
        meta = meta_full.iloc[start : start + self.clip_len].copy()
        frames = self._decode(root / row.video_relpath, start)

        aux = self._source_aux(row, meta, start)

        flipped = self.random_flip and (random.random() < 0.5)
        if flipped:
            frames = frames[:, :, ::-1, :].copy()
            for name in ("steering_deg", "steering_rate_dps", "yaw_rate_rps"):
                if name in meta:
                    meta.loc[:, name] = -meta[name].to_numpy()

            cls = aux["steer_direction_class"].copy()
            left = cls == 0
            right = cls == 2
            cls[left] = 2
            cls[right] = 0
            aux["steer_direction_class"] = cls

        x = (
            torch.from_numpy(frames)
            .permute(3, 0, 1, 2)
            .float()
            .div_(255.0)
        )
        x = (x - self.mean) / self.std

        targets = []
        valids = []
        for name in CAN_TARGETS:
            values = meta[name].to_numpy(dtype=np.float32)
            mask = (
                meta[TARGET_TO_VALID[name]].to_numpy(dtype=bool)
                & np.isfinite(values)
            )
            stat = self.stats[name]
            std = max(float(stat["std"]), 1e-6)
            values = (
                np.nan_to_num(values, nan=float(stat["mean"]))
                - float(stat["mean"])
            ) / std
            targets.append(values)
            valids.append(mask)

        target = torch.from_numpy(np.stack(targets, axis=-1)).float()
        valid = torch.from_numpy(np.stack(valids, axis=-1)).bool()

        return {
            "video": x,
            "target": target,
            "valid": valid,
            "aux": {
                key: torch.from_numpy(value)
                for key, value in aux.items()
            },
            "source_name": str(row["source_name_v5"]),
            "route_id": str(row.route_id),
            "segment_id": str(row.segment_id),
            "start": int(start),
            "flipped": flipped,
        }

    def event_tag(self, index: int) -> str:
        """Cheap metadata-only event label used by the sampler."""
        seg_idx, start = self.windows[index]
        row = self.segments.iloc[seg_idx]
        root = Path(str(row["processed_root_v5"]))
        meta = self._metadata(str(root / row.metadata_relpath)).iloc[
            start : start + self.clip_len
        ]

        speed = meta["speed_mps"].to_numpy(dtype=np.float32)
        accel = meta["accel_from_speed_mps2"].to_numpy(dtype=np.float32)
        yaw = meta["yaw_rate_rps"].to_numpy(dtype=np.float32)

        valid_speed = (
            meta["valid_speed"].to_numpy(dtype=bool) & np.isfinite(speed)
        )
        valid_accel = (
            meta["valid_accel_from_speed"].to_numpy(dtype=bool)
            & np.isfinite(accel)
        )
        valid_yaw = (
            meta["valid_yaw"].to_numpy(dtype=bool) & np.isfinite(yaw)
        )

        low_speed = (
            float(np.mean(speed[valid_speed] < 0.8))
            if valid_speed.any()
            else 0.0
        )
        accel_max = (
            float(np.max(accel[valid_accel]))
            if valid_accel.any()
            else 0.0
        )
        accel_min = (
            float(np.min(accel[valid_accel]))
            if valid_accel.any()
            else 0.0
        )
        turn_fraction = (
            float(np.mean(np.abs(yaw[valid_yaw]) > 0.03))
            if valid_yaw.any()
            else 0.0
        )

        reversal = False
        if valid_yaw.sum() >= 3:
            y = yaw.copy()
            active = valid_yaw & (np.abs(y) > 0.02)
            signs = np.sign(y[active])
            if len(signs) >= 2:
                reversal = bool(np.any(signs[1:] * signs[:-1] < 0))

        # A2D2 brake pressure strengthens the hard-decel event detector.
        hard_brake = False
        aux_rel = row.get("aux_metadata_relpath", np.nan)
        if isinstance(aux_rel, str) and aux_rel and aux_rel.lower() != "nan":
            aux = self._aux_npz(str(root / aux_rel))
            if "brake_pressure_bar" in aux:
                b = np.asarray(
                    aux["brake_pressure_bar"][
                        start : start + self.clip_len
                    ],
                    dtype=np.float32,
                )
                hard_brake = bool(np.nanmax(b) > 2.0)

        if reversal:
            return "reversal"
        if low_speed >= 0.20 and (accel_max > 0.25 or accel_min < -0.25):
            return "stop_start"
        if accel_min < -0.50 or hard_brake:
            return "hard_decel"
        if accel_max > 0.50:
            return "hard_accel"
        if turn_fraction >= 0.10:
            return "turn"
        return "cruise"


def _event_index_cache(
    dataset: MixedStage3CANDataset,
    cache_path: str | Path | None,
) -> tuple[list[str], list[str]]:
    fingerprint = dataset.fingerprint()

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as z:
                cached_fingerprint = str(z["fingerprint"].item())
                if cached_fingerprint == fingerprint:
                    tags = z["tags"].astype(str).tolist()
                    sources = z["sources"].astype(str).tolist()
                    if len(tags) == len(dataset):
                        print(
                            "event index cache: HIT | "
                            f"{cache_path} | windows={len(tags)}"
                        )
                        return tags, sources

    tags: list[str] = []
    sources: list[str] = []
    for i in tqdm(
        range(len(dataset)),
        desc="Build v5 event index",
        mininterval=0.5,
    ):
        tags.append(dataset.event_tag(i))
        sources.append(dataset.window_source(i))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            fingerprint=np.asarray(fingerprint),
            tags=np.asarray(tags),
            sources=np.asarray(sources),
        )
        tmp.replace(cache_path)
        print("event index cache: SAVED ->", cache_path)

    return tags, sources


def build_event_balanced_sampler(
    dataset: MixedStage3CANDataset,
    *,
    num_samples: int,
    event_multipliers: Mapping[str, float] | None = None,
    source_multipliers: Mapping[str, float] | None = None,
    inverse_frequency_power: float = 0.35,
    max_normalized_weight: float = 30.0,
    seed: int = 20260918,
    cache_path: str | Path | None = None,
) -> tuple[WeightedRandomSampler, dict]:
    """Weighted sampler for rare driving events + A2D2 domain exposure."""
    if len(dataset) == 0:
        raise ValueError("cannot sample an empty dataset")

    tags, sources = _event_index_cache(dataset, cache_path)

    event_mult = {
        "cruise": 0.70,
        "turn": 2.50,
        "hard_accel": 3.00,
        "hard_decel": 3.00,
        "stop_start": 4.00,
        "reversal": 3.00,
    }
    event_mult.update(
        {str(k): float(v) for k, v in dict(event_multipliers or {}).items()}
    )
    source_mult = {
        str(k): float(v)
        for k, v in dict(source_multipliers or {}).items()
    }

    event_counts = Counter(tags)
    source_counts = Counter(sources)
    power = float(inverse_frequency_power)

    weights = np.empty(len(dataset), dtype=np.float64)
    for i, (tag, source) in enumerate(zip(tags, sources, strict=True)):
        freq_term = max(float(event_counts[tag]), 1.0) ** (-power)
        weights[i] = (
            freq_term
            * float(event_mult.get(tag, 1.0))
            * float(source_mult.get(source, 1.0))
        )

    weights /= max(float(weights.mean()), 1e-12)
    weights = np.clip(
        weights,
        0.05,
        max(float(max_normalized_weight), 1.0),
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )

    total_weight = float(weights.sum())
    expected_source_share = {}
    for source in sorted(source_counts):
        mask = np.asarray([x == source for x in sources], dtype=bool)
        expected_source_share[source] = float(weights[mask].sum() / total_weight)

    expected_event_share = {}
    for tag in sorted(event_counts):
        mask = np.asarray([x == tag for x in tags], dtype=bool)
        expected_event_share[tag] = float(weights[mask].sum() / total_weight)

    report = {
        "dataset_fingerprint": dataset.fingerprint(),
        "num_windows": len(dataset),
        "num_samples_per_epoch": int(num_samples),
        "event_counts": dict(event_counts),
        "source_counts": dict(source_counts),
        "expected_source_share": expected_source_share,
        "expected_event_share": expected_event_share,
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
    }
    return sampler, report


__all__ = [
    "MixedStage3CANDataset",
    "build_event_balanced_sampler",
]

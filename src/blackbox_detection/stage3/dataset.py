from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import CAN_TARGETS, IMAGENET_MEAN, IMAGENET_STD, TARGET_TO_VALID
from .schema import read_frame_table


class Stage3CANDataset(Dataset):
    def __init__(
        self,
        segment_manifest: str | Path | pd.DataFrame,
        processed_root: str | Path,
        *,
        target_stats: str | Path | dict,
        clip_len: int = 16,
        window_stride: int = 16,
        input_size: tuple[int, int] = (288, 384),
        random_flip: bool = False,
        max_windows: int | None = None,
        seed: int = 20260918,
    ):
        if isinstance(segment_manifest, pd.DataFrame):
            self.segments = segment_manifest.copy()
        else:
            self.segments = pd.read_csv(segment_manifest)
        self.processed_root = Path(processed_root)
        self.clip_len = int(clip_len)
        self.window_stride = int(window_stride)
        self.input_h, self.input_w = map(int, input_size)
        self.random_flip = bool(random_flip)
        self.seed = int(seed)

        if isinstance(target_stats, (str, Path)):
            self.stats = json.loads(Path(target_stats).read_text())
        else:
            self.stats = target_stats

        windows = []
        for seg_idx, row in enumerate(self.segments.itertuples(index=False)):
            n = int(row.num_frames)
            if n < self.clip_len:
                continue
            starts = list(range(0, n - self.clip_len + 1, self.window_stride))
            if starts[-1] != n - self.clip_len:
                starts.append(n - self.clip_len)
            windows.extend((seg_idx, s) for s in starts)

        rng = random.Random(self.seed)
        if max_windows is not None and len(windows) > max_windows:
            windows = rng.sample(windows, int(max_windows))
        self.windows = windows
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None, None]
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None, None]

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=128)
    def _metadata(self, relpath: str) -> pd.DataFrame:
        return read_frame_table(self.processed_root / relpath)

    def _decode(self, path: Path, start: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        for _ in range(self.clip_len):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        cap.release()
        if not frames:
            raise RuntimeError(f"cannot decode {path} from frame {start}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1].copy())
        return np.stack(frames, axis=0)  # T H W C

    def __getitem__(self, index):
        seg_idx, start = self.windows[index]
        row = self.segments.iloc[seg_idx]
        meta = self._metadata(str(row.metadata_relpath)).iloc[start : start + self.clip_len].copy()
        frames = self._decode(self.processed_root / row.video_relpath, start)

        flipped = self.random_flip and (random.random() < 0.5)
        if flipped:
            frames = frames[:, :, ::-1, :].copy()
            for name in ("steering_deg", "steering_rate_dps", "yaw_rate_rps"):
                if name in meta:
                    meta.loc[:, name] = -meta[name].to_numpy()

        x = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std

        targets = []
        valid = []
        for name in CAN_TARGETS:
            values = meta[name].to_numpy(dtype=np.float32)
            mask = meta[TARGET_TO_VALID[name]].to_numpy(dtype=bool) & np.isfinite(values)
            stat = self.stats[name]
            std = max(float(stat["std"]), 1e-6)
            values = (np.nan_to_num(values, nan=float(stat["mean"])) - float(stat["mean"])) / std
            targets.append(values)
            valid.append(mask)

        target = torch.from_numpy(np.stack(targets, axis=-1)).float()  # T K
        valid_mask = torch.from_numpy(np.stack(valid, axis=-1)).bool()
        return {
            "video": x,
            "target": target,
            "valid": valid_mask,
            "route_id": str(row.route_id),
            "segment_id": str(row.segment_id),
            "start": int(start),
            "flipped": flipped,
        }

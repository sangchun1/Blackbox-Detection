from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .v5_dataset import MixedStage3CANDataset


class MixedStage3CANV5DDataset(MixedStage3CANDataset):
    """Acceleration-priority event taxonomy for V5-D sampling.

    V5-A/B used one mutually-exclusive event tag and checked reversal/turn
    before some moderate longitudinal events.  V5-D gives longitudinal dynamics
    priority and adds moderate + mixed acceleration categories so the sampler
    does not focus only on >0.5 m/s^2 extremes.
    """


    def _read_clip(
        self,
        path: Path,
        start: int,
        *,
        sequential: bool,
    ) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return []

        frames: list[np.ndarray] = []
        try:
            if sequential:
                for _ in range(int(start)):
                    ok, _ = cap.read()
                    if not ok:
                        return []
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))

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
        finally:
            cap.release()

        return frames

    def _decode(self, path: Path, start: int) -> np.ndarray:
        """Drive-safe MP4 decoding for V5-D."""
        path = Path(path)
        last_count = 0

        for attempt in range(3):
            frames = self._read_clip(path, start, sequential=False)
            last_count = len(frames)
            if last_count == self.clip_len:
                return np.stack(frames, axis=0)
            time.sleep(0.20 * (attempt + 1))

        for attempt in range(2):
            frames = self._read_clip(path, start, sequential=True)
            last_count = len(frames)
            if last_count == self.clip_len:
                return np.stack(frames, axis=0)
            time.sleep(0.40 * (attempt + 1))

        size = path.stat().st_size if path.is_file() else -1
        raise RuntimeError(
            "V5-D robust decode failed after retries: "
            f"path={path} start={start} clip_len={self.clip_len} "
            f"last_decoded={last_count} size_bytes={size}"
        )

    def event_tag(self, index: int) -> str:
        seg_idx, start = self.windows[index]
        row = self.segments.iloc[seg_idx]
        root = self._root_for_row(row)

        meta = self._metadata(
            str(root / row.metadata_relpath)
        ).iloc[start : start + self.clip_len]

        speed = meta["speed_mps"].to_numpy(dtype=np.float32)
        accel = meta["accel_from_speed_mps2"].to_numpy(
            dtype=np.float32
        )
        yaw = meta["yaw_rate_rps"].to_numpy(dtype=np.float32)

        valid_speed = (
            meta["valid_speed"].to_numpy(dtype=bool)
            & np.isfinite(speed)
        )
        valid_accel = (
            meta["valid_accel_from_speed"].to_numpy(dtype=bool)
            & np.isfinite(accel)
        )
        valid_yaw = (
            meta["valid_yaw"].to_numpy(dtype=bool)
            & np.isfinite(yaw)
        )

        low_speed_fraction = (
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
            active = valid_yaw & (np.abs(yaw) > 0.02)
            signs = np.sign(yaw[active])
            if len(signs) >= 2:
                reversal = bool(
                    np.any(signs[1:] * signs[:-1] < 0)
                )

        hard_brake = False
        aux_rel = row.get("aux_metadata_relpath", np.nan)
        if (
            isinstance(aux_rel, str)
            and aux_rel
            and aux_rel.lower() != "nan"
        ):
            aux = self._aux_npz(str(root / aux_rel))
            if "brake_pressure_bar" in aux:
                brake = np.asarray(
                    aux["brake_pressure_bar"][
                        start : start + self.clip_len
                    ],
                    dtype=np.float32,
                )
                if np.isfinite(brake).any():
                    hard_brake = bool(np.nanmax(brake) > 2.0)

        has_hard_accel = accel_max > 0.50
        has_hard_decel = accel_min < -0.50 or hard_brake
        has_mod_accel = accel_max > 0.15
        has_mod_decel = accel_min < -0.15

        # Longitudinal dynamics take priority in V5-D.
        if (
            low_speed_fraction >= 0.20
            and (accel_max > 0.20 or accel_min < -0.20)
        ):
            return "stop_start"
        if has_hard_accel and has_hard_decel:
            return "hard_mixed"
        if has_hard_decel:
            return "hard_decel"
        if has_hard_accel:
            return "hard_accel"
        if has_mod_accel and has_mod_decel:
            return "moderate_mixed"
        if has_mod_decel:
            return "moderate_decel"
        if has_mod_accel:
            return "moderate_accel"
        if reversal:
            return "reversal"
        if turn_fraction >= 0.10:
            return "turn"
        return "cruise"

    @staticmethod
    def _root_for_row(row):
        # Kept as a tiny helper so event_tag remains easy to audit.
        from pathlib import Path

        return Path(str(row["processed_root_v5"]))


__all__ = ["MixedStage3CANV5DDataset"]

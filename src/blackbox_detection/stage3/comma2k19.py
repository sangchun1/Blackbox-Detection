from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .schema import read_frame_table, write_frame_table
from .signals import acceleration_from_speed, derivative, interpolate_signal
from .video import transcode_every_other_frame


@dataclass(frozen=True)
class SegmentRef:
    archive: Path
    prefix: str
    route_id: str
    segment_id: str
    video_member: str

    @property
    def vehicle_id(self) -> str:
        if "|" in self.route_id:
            return self.route_id.split("|", 1)[0]
        if "__" in self.route_id:
            return self.route_id.split("__", 1)[0]
        return "unknown"

    @property
    def safe_route_id(self) -> str:
        return self.route_id.replace("|", "__").replace("/", "_")


@dataclass
class PrepareConfig:
    processed_root: Path
    width: int = 512
    height: int = 384
    source_fps: int = 20
    target_fps: int = 10
    crf: int = 23
    overwrite: bool = False


def find_archives(raw_root: str | Path) -> list[Path]:
    return sorted(Path(raw_root).rglob("*.zip"))


def _route_and_segment(parts: tuple[str, ...]) -> tuple[str, str]:
    route_idx = None
    for i, part in enumerate(parts):
        if "|" in part:
            route_idx = i
            break
    if route_idx is None:
        for i, part in enumerate(parts):
            if "--" in part and i + 1 < len(parts):
                route_idx = i
                break
    if route_idx is None or route_idx + 1 >= len(parts):
        raise ValueError(
            f"cannot parse route/segment from archive path: {'/'.join(parts)}"
        )
    return parts[route_idx], parts[route_idx + 1]


def discover_segments(archive: str | Path) -> list[SegmentRef]:
    archive = Path(archive)
    refs: list[SegmentRef] = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.lower().endswith("/video.hevc"):
                continue
            p = PurePosixPath(name)
            route_id, segment_id = _route_and_segment(p.parts)
            refs.append(
                SegmentRef(
                    archive=archive,
                    prefix=str(p.parent),
                    route_id=route_id,
                    segment_id=segment_id,
                    video_member=name,
                )
            )
    refs.sort(
        key=lambda r: (
            r.route_id,
            int(r.segment_id) if r.segment_id.isdigit() else r.segment_id,
        )
    )
    return refs


def _match_member(
    names: list[str],
    prefix: str,
    tokens: tuple[str, ...],
    basenames: tuple[str, ...],
) -> str | None:
    prefix_lower = prefix.lower().rstrip("/") + "/"
    candidates = []

    for name in names:
        low = name.lower()
        if not low.startswith(prefix_lower):
            continue
        if not all(tok.lower() in low for tok in tokens):
            continue

        base = PurePosixPath(low).name
        stem = PurePosixPath(base).stem
        if base in basenames or stem in basenames:
            candidates.append(name)

    if not candidates:
        return None

    candidates.sort(key=len)
    return candidates[0]


def _load_npy(zf: zipfile.ZipFile, member: str | None):
    if member is None:
        return None
    return np.load(BytesIO(zf.read(member)), allow_pickle=False)


def _load_pair(zf, names, prefix, group_tokens):
    t_member = _match_member(
        names,
        prefix,
        group_tokens,
        ("t", "time", "times", "timestamp", "timestamps"),
    )
    v_member = _match_member(
        names,
        prefix,
        group_tokens,
        ("value", "values", "v"),
    )
    if t_member is None or v_member is None:
        return None, None
    return _load_npy(zf, t_member), _load_npy(zf, v_member)


def _load_pair_any(zf, names, prefix, token_options):
    """Try multiple archive path aliases and return the first valid pair."""
    for tokens in token_options:
        t, v = _load_pair(zf, names, prefix, tokens)
        if t is not None and v is not None:
            return t, v
    return None, None


def _load_global(zf, names, prefix, basename):
    return _load_npy(
        zf,
        _match_member(names, prefix, ("global",), (basename,)),
    )


def load_segment_signals(
    zf: zipfile.ZipFile,
    ref: SegmentRef,
) -> dict[str, np.ndarray | None]:
    names = zf.namelist()

    frame_times = _load_global(zf, names, ref.prefix, "frame_times")
    if frame_times is None:
        raise FileNotFoundError(f"frame_times not found for {ref.prefix}")

    frame_velocities = _load_global(
        zf, names, ref.prefix, "frame_velocities"
    )

    # Actual Academic Torrents archive currently uses CAN/speed and
    # IMU/accelerometer. Keep documented/legacy aliases as fallbacks.
    speed_t, speed_v = _load_pair_any(
        zf,
        names,
        ref.prefix,
        [
            ("processed_log", "/can/speed/"),
            ("processed_log", "/can/car_speed/"),
        ],
    )

    steer_t, steer_v = _load_pair_any(
        zf,
        names,
        ref.prefix,
        [
            ("processed_log", "/can/steering_angle/"),
        ],
    )

    accel_t, accel_v = _load_pair_any(
        zf,
        names,
        ref.prefix,
        [
            ("processed_log", "/imu/accelerometer/"),
            ("processed_log", "/imu/acceleration/"),
        ],
    )

    gyro_t, gyro_v = _load_pair_any(
        zf,
        names,
        ref.prefix,
        [
            ("processed_log", "/imu/gyro/"),
            ("processed_log", "/imu/gyro_uncalibrated/"),
        ],
    )

    if speed_t is None or steer_t is None:
        raise FileNotFoundError(
            f"required CAN speed/steering signals missing in {ref.prefix}"
        )

    return {
        "frame_times": np.asarray(frame_times),
        "frame_velocities": (
            None if frame_velocities is None
            else np.asarray(frame_velocities)
        ),
        "speed_t": speed_t,
        "speed_v": speed_v,
        "steer_t": steer_t,
        "steer_v": steer_v,
        "accel_t": accel_t,
        "accel_v": accel_v,
        "gyro_t": gyro_t,
        "gyro_v": gyro_v,
    }


def build_frame_table(
    ref: SegmentRef,
    signals: dict,
    video_relpath: str,
) -> pd.DataFrame:
    frame_times = np.asarray(
        signals["frame_times"],
        dtype=np.float64,
    ).reshape(-1)

    source_indices = np.arange(
        0,
        len(frame_times),
        2,
        dtype=np.int64,
    )
    q = frame_times[source_indices]

    speed_all, valid_speed_all = interpolate_signal(
        frame_times,
        signals["speed_t"],
        signals["speed_v"],
    )
    steer_all, valid_steer_all = interpolate_signal(
        frame_times,
        signals["steer_t"],
        signals["steer_v"],
    )

    speed = np.asarray(speed_all).reshape(-1)[source_indices]
    steering = np.asarray(steer_all).reshape(-1)[source_indices]

    valid_speed = (
        valid_speed_all[source_indices]
        & np.isfinite(speed)
    )
    valid_steer = (
        valid_steer_all[source_indices]
        & np.isfinite(steering)
    )

    accel_from_speed, speed_smoothed = acceleration_from_speed(
        speed,
        q,
        smooth_window=11,
    )

    valid_accel_from_speed = valid_speed.copy()
    if len(valid_accel_from_speed) > 0:
        valid_accel_from_speed[[0, -1]] = False

    steering_rate = derivative(steering, q)

    accel_imu = np.full(len(q), np.nan, dtype=np.float64)
    valid_accel_imu = np.zeros(len(q), dtype=bool)

    if (
        signals["accel_t"] is not None
        and signals["accel_v"] is not None
    ):
        a_all, a_valid_all = interpolate_signal(
            frame_times,
            signals["accel_t"],
            signals["accel_v"],
        )
        a_all = np.asarray(a_all)

        # comma2k19 IMU order: [forward, right, down]
        if a_all.ndim > 1:
            a_all = a_all[:, 0]

        accel_imu = a_all[source_indices]
        valid_accel_imu = (
            a_valid_all[source_indices]
            & np.isfinite(accel_imu)
        )

    yaw = np.full(len(q), np.nan, dtype=np.float64)
    valid_yaw = np.zeros(len(q), dtype=bool)

    if (
        signals["gyro_t"] is not None
        and signals["gyro_v"] is not None
    ):
        g_all, g_valid_all = interpolate_signal(
            frame_times,
            signals["gyro_t"],
            signals["gyro_v"],
        )
        g_all = np.asarray(g_all)

        # [forward, right, down] -> yaw about down axis
        if g_all.ndim > 1 and g_all.shape[1] >= 3:
            g_all = g_all[:, 2]
        elif g_all.ndim > 1:
            g_all = g_all[:, -1]

        yaw = g_all[source_indices]
        valid_yaw = (
            g_valid_all[source_indices]
            & np.isfinite(yaw)
        )

    pose_speed = np.full(len(q), np.nan, dtype=np.float64)
    frame_velocities = signals.get("frame_velocities")

    if frame_velocities is not None:
        fv = np.asarray(frame_velocities, dtype=np.float64)
        if (
            fv.ndim == 2
            and fv.shape[0] >= len(frame_times)
            and fv.shape[1] >= 3
        ):
            pose_speed = np.linalg.norm(
                fv[: len(frame_times), :3],
                axis=1,
            )[source_indices]

    return pd.DataFrame(
        {
            "dataset": "comma2k19",
            "vehicle_id": ref.vehicle_id,
            "route_id": ref.route_id,
            "segment_id": ref.segment_id,
            "frame_index_10hz": np.arange(
                len(q),
                dtype=np.int32,
            ),
            "frame_index_source": source_indices.astype(np.int32),
            "timestamp": q,
            "speed_mps": speed_smoothed.astype(np.float32),
            "speed_pose_mps": pose_speed.astype(np.float32),
            "accel_from_speed_mps2": accel_from_speed.astype(np.float32),
            "accel_imu_forward_mps2": accel_imu.astype(np.float32),
            "steering_deg": steering.astype(np.float32),
            "steering_rate_dps": steering_rate.astype(np.float32),
            "yaw_rate_rps": yaw.astype(np.float32),
            "valid_speed": valid_speed,
            "valid_accel_from_speed": valid_accel_from_speed,
            "valid_accel_imu": valid_accel_imu,
            "valid_steer": valid_steer,
            "valid_yaw": valid_yaw,
            "video_relpath": video_relpath,
        }
    )


def prepare_segment(
    zf: zipfile.ZipFile,
    ref: SegmentRef,
    cfg: PrepareConfig,
) -> dict:
    processed_root = Path(cfg.processed_root)
    route = ref.safe_route_id

    video_out = (
        processed_root
        / "videos"
        / route
        / f"{ref.segment_id}.mp4"
    )
    meta_out = (
        processed_root
        / "metadata"
        / route
        / f"{ref.segment_id}.npz"
    )

    video_relpath = video_out.relative_to(processed_root).as_posix()
    meta_relpath = meta_out.relative_to(processed_root).as_posix()

    if video_out.exists() and meta_out.exists() and not cfg.overwrite:
        df = read_frame_table(
            meta_out,
            columns=[
                "route_id",
                "segment_id",
                "vehicle_id",
                "timestamp",
            ],
        )
        return {
            "dataset": "comma2k19",
            "archive": ref.archive.name,
            "vehicle_id": str(df.iloc[0]["vehicle_id"]),
            "route_id": str(df.iloc[0]["route_id"]),
            "segment_id": str(df.iloc[0]["segment_id"]),
            "num_frames": len(df),
            "duration_s": (
                float(
                    df["timestamp"].iloc[-1]
                    - df["timestamp"].iloc[0]
                )
                if len(df) > 1 else 0.0
            ),
            "video_relpath": video_relpath,
            "metadata_relpath": meta_relpath,
        }

    signals = load_segment_signals(zf, ref)

    with tempfile.TemporaryDirectory(
        prefix="comma2k19_"
    ) as tmp:
        tmp_hevc = Path(tmp) / "video.hevc"

        with zf.open(ref.video_member) as src, open(
            tmp_hevc,
            "wb",
        ) as dst:
            shutil.copyfileobj(
                src,
                dst,
                length=8 * 1024 * 1024,
            )

        transcode_every_other_frame(
            tmp_hevc,
            video_out,
            width=cfg.width,
            height=cfg.height,
            source_fps=cfg.source_fps,
            target_fps=cfg.target_fps,
            crf=cfg.crf,
        )

    df = build_frame_table(
        ref,
        signals,
        video_relpath,
    )
    write_frame_table(df, meta_out)

    return {
        "dataset": "comma2k19",
        "archive": ref.archive.name,
        "vehicle_id": ref.vehicle_id,
        "route_id": ref.route_id,
        "segment_id": ref.segment_id,
        "num_frames": len(df),
        "duration_s": (
            float(
                df["timestamp"].iloc[-1]
                - df["timestamp"].iloc[0]
            )
            if len(df) > 1 else 0.0
        ),
        "video_relpath": video_relpath,
        "metadata_relpath": meta_relpath,
    }


def prepare_archive(
    archive: str | Path,
    cfg: PrepareConfig,
    *,
    max_segments: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    archive = Path(archive)
    refs = discover_segments(archive)

    if max_segments is not None:
        refs = refs[:max_segments]

    rows = []

    with zipfile.ZipFile(archive) as zf:
        iterator = (
            tqdm(refs, desc=archive.name)
            if show_progress else refs
        )

        for ref in iterator:
            try:
                rows.append(
                    prepare_segment(zf, ref, cfg)
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": "comma2k19",
                        "archive": archive.name,
                        "vehicle_id": ref.vehicle_id,
                        "route_id": ref.route_id,
                        "segment_id": ref.segment_id,
                        "error": repr(exc),
                    }
                )

    return pd.DataFrame(rows)

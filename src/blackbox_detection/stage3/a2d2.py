from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .schema import write_frame_table
from .signals import acceleration_from_speed, derivative, interpolate_signal
from .video import require_ffmpeg

_FRAME_RE = re.compile(r"_(\d{9})\.(?:json|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class CameraRecord:
    source_index: int
    timestamp_us: int
    png_member: str


@dataclass
class A2D2PrepareConfig:
    processed_root: Path
    width: int = 512
    height: int = 384
    target_fps: int = 10
    segment_frames: int = 600
    jpeg_quality: int = 92
    crf: int = 23
    ffmpeg_preset: str = "veryfast"
    max_camera_skew_ms: float = 25.0
    overwrite: bool = False
    work_root: Path = Path("/content/a2d2_work")


def _signal(bus: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(bus[name]["values"], dtype=np.float64)
    return values[:, 0], values[:, 1]


def load_bus_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_bus(bus: dict) -> dict:
    required = [
        "vehicle_speed", "acceleration_x", "steering_angle_calculated",
        "steering_angle_calculated_sign", "angular_velocity_omega_z",
        "brake_pressure", "accelerator_pedal",
    ]
    missing = [k for k in required if k not in bus]
    if missing:
        raise ValueError(f"missing A2D2 bus signals: {missing}")

    t_speed, speed_kmh = _signal(bus, "vehicle_speed")
    t_acc, accel_x = _signal(bus, "acceleration_x")
    speed_mps = speed_kmh / 3.6
    speed_smooth = (
        pd.Series(speed_mps)
        .rolling(51, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    dvdt = np.gradient(speed_smooth, t_speed / 1e6)
    acc_at_speed = np.interp(t_speed, t_acc, accel_x)
    mask = np.isfinite(dvdt) & np.isfinite(acc_at_speed) & (speed_mps > 1.0)
    accel_corr = float(np.corrcoef(dvdt[mask], acc_at_speed[mask])[0, 1])

    t_steer, steer_mag = _signal(bus, "steering_angle_calculated")
    t_sign, steer_sign = _signal(bus, "steering_angle_calculated_sign")
    if not np.array_equal(t_steer, t_sign):
        raise ValueError("steering magnitude/sign timestamps differ")
    steer_signed = steer_mag * np.where(steer_sign > 0.5, -1.0, +1.0)
    t_yaw, yaw_dps = _signal(bus, "angular_velocity_omega_z")
    yaw_at_steer = np.interp(t_steer, t_yaw, yaw_dps)
    mask = (
        np.isfinite(steer_signed) & np.isfinite(yaw_at_steer)
        & (steer_mag > 2.0) & (np.abs(yaw_at_steer) > 0.5)
    )
    steer_yaw_corr = float(
        np.corrcoef(steer_signed[mask], yaw_at_steer[mask])[0, 1]
    )
    return {
        "accel_dvdt_corr": accel_corr,
        "steer_yaw_corr": steer_yaw_corr,
        "accel_sign": "+acceleration_x",
        "steering_sign_rule": "sign=1 -> negative; sign=0 -> positive",
    }


def scan_camera_records(camera_tar: str | Path) -> list[CameraRecord]:
    records: list[CameraRecord] = []
    with tarfile.open(camera_tar, "r:") as tf:
        for member in tqdm(tf, desc="scan camera JSON"):
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if "/camera/cam_front_center/" not in member.name:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            meta = json.loads(f.read())
            if meta.get("cam_name") != "front_center":
                continue
            m = _FRAME_RE.search(member.name)
            if m is None:
                raise ValueError(f"cannot parse frame index: {member.name}")
            parent = PurePosixPath(member.name).parent
            records.append(CameraRecord(
                source_index=int(m.group(1)),
                timestamp_us=int(meta["cam_tstamp"]),
                png_member=str(parent / meta["image_png"]),
            ))
    if not records:
        raise RuntimeError("no front-center camera JSON found")
    records.sort(key=lambda r: (r.timestamp_us, r.source_index))
    ts = np.asarray([r.timestamp_us for r in records], dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise RuntimeError("camera timestamps are not strictly increasing")
    return records


def select_camera_10hz(
    records: list[CameraRecord], target_fps: int = 10, max_skew_ms: float = 25.0
):
    src = np.asarray([r.timestamp_us for r in records], dtype=np.int64)
    step = int(round(1_000_000 / target_fps))
    target = np.arange(src[0], src[-1] + 1, step, dtype=np.int64)
    right = np.clip(np.searchsorted(src, target), 0, len(src) - 1)
    left = np.clip(right - 1, 0, len(src) - 1)
    choose_left = np.abs(src[left] - target) <= np.abs(src[right] - target)
    idx = np.where(choose_left, left, right)
    if len(np.unique(idx)) != len(idx):
        raise RuntimeError("duplicate source frame selected")
    skew = src[idx] - target
    if np.max(np.abs(skew)) > max_skew_ms * 1000:
        raise RuntimeError(
            f"camera skew too large: {np.max(np.abs(skew))/1000:.3f} ms"
        )
    return [records[int(i)] for i in idx], skew


def _interp(bus: dict, name: str, q_us: np.ndarray):
    t, v = _signal(bus, name)
    out, valid = interpolate_signal(q_us, t, v)
    return np.asarray(out).reshape(-1), valid


def build_frame_table(bus: dict, selected: list[CameraRecord], route_id: str,
                      video_relpaths: list[str], segment_ids: list[str],
                      frame_indices: list[int]):
    q_us = np.asarray([r.timestamp_us for r in selected], dtype=np.float64)
    q_s = q_us / 1e6

    speed_kmh, valid_speed = _interp(bus, "vehicle_speed", q_us)
    speed = speed_kmh / 3.6
    accel_from_speed, speed_smoothed = acceleration_from_speed(speed, q_s, 11)
    valid_afs = valid_speed & np.isfinite(accel_from_speed)
    if len(valid_afs):
        valid_afs[[0, -1]] = False

    accel_x, valid_accel = _interp(bus, "acceleration_x", q_us)
    yaw_dps, valid_yaw = _interp(bus, "angular_velocity_omega_z", q_us)
    yaw_rps = np.deg2rad(yaw_dps)

    steer_mag, valid_sm = _interp(bus, "steering_angle_calculated", q_us)
    steer_sign, valid_ss = _interp(bus, "steering_angle_calculated_sign", q_us)
    steering_wheel = steer_mag * np.where(steer_sign >= 0.5, -1.0, +1.0)
    steering_rate = derivative(steering_wheel, q_s)

    brake, valid_brake = _interp(bus, "brake_pressure", q_us)
    throttle, valid_throttle = _interp(bus, "accelerator_pedal", q_us)

    n = len(selected)
    df = pd.DataFrame({
        "dataset": "a2d2",
        "vehicle_id": "audi_a2d2",
        "route_id": route_id,
        "segment_id": np.asarray(segment_ids, dtype=str),
        "frame_index_10hz": np.asarray(frame_indices, dtype=np.int32),
        "frame_index_source": np.asarray([r.source_index for r in selected], dtype=np.int32),
        "timestamp": q_s,
        "speed_mps": speed_smoothed.astype(np.float32),
        "speed_pose_mps": np.full(n, np.nan, dtype=np.float32),
        "accel_from_speed_mps2": accel_from_speed.astype(np.float32),
        "accel_imu_forward_mps2": accel_x.astype(np.float32),
        "steering_deg": steering_wheel.astype(np.float32),
        "steering_rate_dps": steering_rate.astype(np.float32),
        "yaw_rate_rps": yaw_rps.astype(np.float32),
        "valid_speed": valid_speed & np.isfinite(speed_smoothed),
        "valid_accel_from_speed": valid_afs,
        "valid_accel_imu": valid_accel & np.isfinite(accel_x),
        # Steering wheel angle scale is not the same target as comma2k19.
        # Keep it for diagnostics, but do not feed it to the existing shared regression loss.
        "valid_steer": np.zeros(n, dtype=bool),
        "valid_yaw": valid_yaw & np.isfinite(yaw_rps),
        "video_relpath": np.asarray(video_relpaths, dtype=str),
    })
    aux = {
        "timestamp": q_s,
        "steering_wheel_deg": steering_wheel.astype(np.float32),
        "steering_magnitude_deg": np.abs(steering_wheel).astype(np.float32),
        "steering_direction_sign": np.sign(steering_wheel).astype(np.int8),
        "valid_steering_wheel": valid_sm & valid_ss & np.isfinite(steering_wheel),
        "yaw_rate_dps": yaw_dps.astype(np.float32),
        "acceleration_x_mps2": accel_x.astype(np.float32),
        "brake_pressure_bar": brake.astype(np.float32),
        "valid_brake": valid_brake & np.isfinite(brake),
        "accelerator_pedal_pct": throttle.astype(np.float32),
        "valid_accelerator": valid_throttle & np.isfinite(throttle),
    }
    return df, aux


def extract_selected(camera_tar: str | Path, selected: list[CameraRecord],
                     work_dir: str | Path, width: int, height: int,
                     jpeg_quality: int = 92):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wanted = {r.png_member: i for i, r in enumerate(selected)}
    found = np.zeros(len(selected), dtype=bool)
    with tarfile.open(camera_tar, "r:") as tf:
        for member in tqdm(tf, desc="extract selected PNG"):
            seq = wanted.get(member.name)
            if seq is None or not member.isfile():
                continue
            f = tf.extractfile(member)
            raw = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"decode failed: {member.name}")
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            out = work_dir / f"frame_{seq:09d}.jpg"
            if not cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise RuntimeError(f"write failed: {out}")
            found[seq] = True
    missing = np.flatnonzero(~found)
    if len(missing):
        raise RuntimeError(f"missing selected PNGs: {len(missing)}")


def encode_segment(work_dir: Path, output: Path, start: int, count: int,
                   fps: int, crf: int, preset: str):
    require_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", str(start),
        "-i", str(work_dir / "frame_%09d.jpg"),
        "-frames:v", str(count), "-an", "-c:v", "libx264",
        "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), "-movflags", "+faststart", str(output),
    ]
    subprocess.run(cmd, check=True)


def prepare_session(camera_tar: str | Path, bus_json: str | Path,
                    session_id: str, cfg: A2D2PrepareConfig) -> dict:
    processed = Path(cfg.processed_root)
    route_id = f"a2d2_{session_id}"
    bus = load_bus_json(bus_json)
    audit = audit_bus(bus)
    if audit["accel_dvdt_corr"] < 0.80:
        raise RuntimeError(f"accel audit failed: {audit}")
    if audit["steer_yaw_corr"] < 0.50:
        raise RuntimeError(f"steering audit failed: {audit}")

    records = scan_camera_records(camera_tar)
    selected, skew_us = select_camera_10hz(
        records, cfg.target_fps, cfg.max_camera_skew_ms
    )
    n = len(selected)
    n_segments = (n + cfg.segment_frames - 1) // cfg.segment_frames

    video_relpaths, segment_ids, frame_indices = [""] * n, [""] * n, [0] * n
    segments = []
    for si in range(n_segments):
        start = si * cfg.segment_frames
        end = min(n, start + cfg.segment_frames)
        seg_id = f"{si:03d}"
        video_out = processed / "videos" / route_id / f"{seg_id}.mp4"
        meta_out = processed / "metadata" / route_id / f"{seg_id}.npz"
        aux_out = processed / "aux_metadata" / route_id / f"{seg_id}.npz"
        rel = video_out.relative_to(processed).as_posix()
        for j, gi in enumerate(range(start, end)):
            video_relpaths[gi] = rel
            segment_ids[gi] = seg_id
            frame_indices[gi] = j
        segments.append((start, end, seg_id, video_out, meta_out, aux_out, rel))

    frame_table, aux = build_frame_table(
        bus, selected, route_id, video_relpaths, segment_ids, frame_indices
    )

    work = Path(cfg.work_root) / route_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        if cfg.overwrite or any(not s[3].exists() for s in segments):
            extract_selected(
                camera_tar, selected, work, cfg.width, cfg.height, cfg.jpeg_quality
            )

        rows = []
        for start, end, seg_id, video_out, meta_out, aux_out, rel in tqdm(
            segments, desc="write A2D2 segments"
        ):
            seg_df = frame_table.iloc[start:end].copy().reset_index(drop=True)
            if cfg.overwrite or not video_out.exists():
                encode_segment(
                    work, video_out, start, end - start,
                    cfg.target_fps, cfg.crf, cfg.ffmpeg_preset,
                )
            if cfg.overwrite or not meta_out.exists():
                write_frame_table(seg_df, meta_out)
            if cfg.overwrite or not aux_out.exists():
                aux_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    aux_out,
                    **{k: np.asarray(v[start:end]) for k, v in aux.items()},
                )
            rows.append({
                "dataset": "a2d2", "archive": Path(camera_tar).name,
                "vehicle_id": "audi_a2d2", "route_id": route_id,
                "segment_id": seg_id, "num_frames": end - start,
                "duration_s": float(seg_df["timestamp"].iloc[-1] - seg_df["timestamp"].iloc[0]) if len(seg_df) > 1 else 0.0,
                "video_relpath": rel,
                "metadata_relpath": meta_out.relative_to(processed).as_posix(),
                "aux_metadata_relpath": aux_out.relative_to(processed).as_posix(),
            })

        manifest = pd.DataFrame(rows)
        processed.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(processed / "manifest.csv", index=False)
        report = {
            "session_id": session_id,
            "camera_records": len(records),
            "selected_10hz_frames": len(selected),
            "segments": n_segments,
            "duration_hours": len(selected) / cfg.target_fps / 3600.0,
            "camera_skew_ms": {
                "mean_abs": float(np.mean(np.abs(skew_us)) / 1000),
                "p95_abs": float(np.percentile(np.abs(skew_us), 95) / 1000),
                "max_abs": float(np.max(np.abs(skew_us)) / 1000),
            },
            "bus_audit": audit,
            "valid_counts": {
                c: int(frame_table[c].sum()) for c in [
                    "valid_speed", "valid_accel_from_speed", "valid_accel_imu",
                    "valid_steer", "valid_yaw",
                ]
            },
        }
        (processed / "prepare_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    finally:
        if work.exists():
            shutil.rmtree(work)

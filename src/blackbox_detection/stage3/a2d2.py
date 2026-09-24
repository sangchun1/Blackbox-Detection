from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .schema import write_frame_table
from .signals import acceleration_from_speed, derivative, interpolate_signal
from .video import require_ffmpeg

_FRAME_RE = re.compile(r"_(\d{9})\.(?:json|png)$", re.IGNORECASE)
_INDEX_VERSION = 2


@dataclass(frozen=True)
class CameraRecord:
    source_index: int
    timestamp_us: int
    json_member: str
    png_member: str


@dataclass(frozen=True)
class PngMember:
    name: str
    data_offset: int
    size: int


@dataclass
class A2D2PrepareConfig:
    processed_root: Path
    width: int = 512
    height: int = 384
    target_fps: int = 10
    segment_frames: int = 600
    crf: int = 23
    ffmpeg_preset: str = "veryfast"
    max_camera_skew_ms: float = 25.0
    scan_checkpoint_every: int = 2000
    overwrite: bool = False
    work_root: Path = Path("/content/a2d2_work")


def _signal(bus: dict, name: str):
    arr = np.asarray(bus[name]["values"], dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def load_bus_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_bus(bus: dict) -> dict:
    required = [
        "vehicle_speed", "acceleration_x", "steering_angle_calculated",
        "steering_angle_calculated_sign", "angular_velocity_omega_z",
        "brake_pressure", "accelerator_pedal",
    ]
    missing = [x for x in required if x not in bus]
    if missing:
        raise ValueError(f"missing bus signals: {missing}")

    ts, v_kmh = _signal(bus, "vehicle_speed")
    ta, ax = _signal(bus, "acceleration_x")
    v = v_kmh / 3.6
    vs = pd.Series(v).rolling(51, center=True, min_periods=1).mean().to_numpy()
    dvdt = np.gradient(vs, ts / 1e6)
    ax_i = np.interp(ts, ta, ax)
    mask = np.isfinite(dvdt) & np.isfinite(ax_i) & (v > 1.0)
    accel_corr = float(np.corrcoef(dvdt[mask], ax_i[mask])[0, 1])

    tst, mag = _signal(bus, "steering_angle_calculated")
    tsg, sign = _signal(bus, "steering_angle_calculated_sign")
    if not np.array_equal(tst, tsg):
        raise ValueError("steering magnitude/sign timestamps differ")
    signed = mag * np.where(sign > 0.5, -1.0, +1.0)
    ty, yaw = _signal(bus, "angular_velocity_omega_z")
    yaw_i = np.interp(tst, ty, yaw)
    mask = np.isfinite(signed) & np.isfinite(yaw_i) & (mag > 2.0) & (np.abs(yaw_i) > 0.5)
    steer_corr = float(np.corrcoef(signed[mask], yaw_i[mask])[0, 1])

    return {
        "accel_dvdt_corr": accel_corr,
        "steer_yaw_corr": steer_corr,
        "accel_sign": "+acceleration_x",
        "steering_sign_rule": "sign=1 -> negative; sign=0 -> positive",
    }


def _sig(path: Path) -> dict:
    st = path.stat()
    return {"name": path.name, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _atomic_json(obj: dict, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, dst)


def _tar_name(h: bytes) -> str:
    dec = lambda b: b.rstrip(b"\0 ").decode("utf-8", errors="surrogateescape")
    name, prefix = dec(h[:100]), dec(h[345:500])
    return f"{prefix}/{name}" if prefix else name


def _round512(n: int) -> int:
    return ((n + 511) // 512) * 512


def _pax(data: bytes) -> dict[str, str]:
    out = {}
    pos = 0
    while pos < len(data):
        sp = data.find(b" ", pos)
        if sp < 0:
            break
        try:
            n = int(data[pos:sp])
        except ValueError:
            break
        rec = data[sp + 1:pos + n].rstrip(b"\n")
        if b"=" in rec:
            k, v = rec.split(b"=", 1)
            out[k.decode()] = v.decode("utf-8", errors="surrogateescape")
        pos += n
    return out


def _decode_index(obj: dict):
    records = [CameraRecord(**x) for x in obj["records"]]
    pngs = {
        k: PngMember(k, int(v[0]), int(v[1]))
        for k, v in obj["png_members"].items()
    }
    return records, pngs


def scan_camera_index_resume(camera_tar: str | Path, cache_dir: str | Path, checkpoint_every: int = 2000):
    camera_tar, cache_dir = Path(camera_tar), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "camera_index.json"
    ckpt_path = cache_dir / "camera_index.checkpoint.json"
    signature = _sig(camera_tar)

    if index_path.is_file():
        obj = json.loads(index_path.read_text(encoding="utf-8"))
        if obj.get("version") == _INDEX_VERSION and obj.get("complete") and obj.get("archive_signature") == signature:
            records, pngs = _decode_index(obj)
            print(f"camera index: REUSE ({len(records)} json, {len(pngs)} png)")
            return records, pngs, {"source": "final_index", "path": str(index_path)}

    offset = 0
    members = 0
    records: list[CameraRecord] = []
    pngs: dict[str, PngMember] = {}
    resumed = False

    if ckpt_path.is_file():
        obj = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if obj.get("version") == _INDEX_VERSION and obj.get("archive_signature") == signature:
            offset = int(obj["next_offset"])
            members = int(obj["members_scanned"])
            records, pngs = _decode_index(obj)
            resumed = True

    total = camera_tar.stat().st_size
    print(f"camera index: {'RESUME' if resumed else 'NEW'} | offset={offset/2**30:.2f}/{total/2**30:.2f} GiB | members={members}")
    last_ckpt = members
    longname = None
    pax_path = None

    with open(camera_tar, "rb", buffering=0) as f, tqdm(total=total, initial=offset, unit="B", unit_scale=True, desc="A2D2 TAR index") as bar:
        while offset + 512 <= total:
            f.seek(offset)
            h = f.read(512)
            if len(h) != 512:
                raise RuntimeError(f"short tar header at {offset}")
            if h == b"\0" * 512:
                break

            name = _tar_name(h)
            size = int(tarfile.nti(h[124:136]))
            typ = h[156:157] or b"\0"
            data_off = offset + 512
            next_off = data_off + _round512(size)

            if typ == b"x":
                f.seek(data_off)
                pax_path = _pax(f.read(size)).get("path", pax_path)
                bar.update(next_off - offset); offset = next_off; continue
            if typ == b"L":
                f.seek(data_off)
                longname = f.read(size).rstrip(b"\0\n").decode("utf-8", errors="surrogateescape")
                bar.update(next_off - offset); offset = next_off; continue
            if typ == b"g":
                bar.update(next_off - offset); offset = next_off; continue

            eff = longname or pax_path or name
            longname = pax_path = None
            members += 1
            front = "/camera/cam_front_center/" in eff
            regular = typ in (b"\0", b"0", b"7")

            if front and regular and eff.endswith(".png"):
                pngs[eff] = PngMember(eff, data_off, size)
            elif front and regular and eff.endswith(".json"):
                f.seek(data_off)
                meta = json.loads(f.read(size))
                if meta.get("cam_name") == "front_center":
                    m = _FRAME_RE.search(eff)
                    if not m:
                        raise ValueError(f"cannot parse frame index: {eff}")
                    png_member = str(PurePosixPath(eff).parent / meta["image_png"])
                    records.append(CameraRecord(int(m.group(1)), int(meta["cam_tstamp"]), eff, png_member))

            bar.update(next_off - offset)
            offset = next_off

            if members - last_ckpt >= checkpoint_every:
                _atomic_json({
                    "version": _INDEX_VERSION,
                    "archive_signature": signature,
                    "next_offset": offset,
                    "members_scanned": members,
                    "records": [asdict(x) for x in records],
                    "png_members": {k: [v.data_offset, v.size] for k, v in pngs.items()},
                }, ckpt_path)
                print(f"checkpoint: members={members}, offset={offset/2**30:.2f} GiB, json={len(records)}, png={len(pngs)}")
                last_ckpt = members

    records.sort(key=lambda x: (x.timestamp_us, x.source_index))
    if not records or not pngs:
        raise RuntimeError("camera index is empty")
    missing = [x.png_member for x in records if x.png_member not in pngs]
    if missing:
        raise RuntimeError(f"{len(missing)} indexed JSON records have no PNG")

    _atomic_json({
        "version": _INDEX_VERSION,
        "complete": True,
        "archive_signature": signature,
        "members_scanned": members,
        "records": [asdict(x) for x in records],
        "png_members": {k: [v.data_offset, v.size] for k, v in pngs.items()},
    }, index_path)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"camera index COMPLETE: {len(records)} json, {len(pngs)} png")
    return records, pngs, {"source": "scan", "resumed": resumed, "path": str(index_path)}


def select_camera_10hz(records: list[CameraRecord], target_fps: int = 10, max_skew_ms: float = 25.0):
    ts = np.asarray([x.timestamp_us for x in records], dtype=np.int64)
    step = int(round(1_000_000 / target_fps))
    target = np.arange(ts[0], ts[-1] + 1, step, dtype=np.int64)
    right = np.clip(np.searchsorted(ts, target), 0, len(ts) - 1)
    left = np.clip(right - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[left] - target) <= np.abs(ts[right] - target), left, right)
    if len(np.unique(idx)) != len(idx):
        raise RuntimeError("duplicate selected frames")
    selected = [records[int(i)] for i in idx]
    skew = ts[idx] - target
    if np.max(np.abs(skew)) > max_skew_ms * 1000:
        raise RuntimeError(f"camera skew too large: {np.max(np.abs(skew))/1000:.2f} ms")
    return selected, skew


def build_aligned_signals(bus: dict, selected: list[CameraRecord], vehicle_id: str, route_id: str):
    q_us = np.asarray([r.timestamp_us for r in selected], dtype=np.float64)
    q_s = q_us / 1e6

    t, v = _signal(bus, "vehicle_speed")
    speed, valid_speed = interpolate_signal(q_us, t, v / 3.6)
    speed = np.asarray(speed).reshape(-1)
    accel_from_speed, speed_sm = acceleration_from_speed(speed, q_s, smooth_window=11)
    valid_accel = valid_speed & np.isfinite(accel_from_speed)
    if len(valid_accel):
        valid_accel[[0, -1]] = False

    t, v = _signal(bus, "acceleration_x")
    accel_imu, valid_imu = interpolate_signal(q_us, t, v)
    accel_imu = np.asarray(accel_imu).reshape(-1)

    t, v = _signal(bus, "angular_velocity_omega_z")
    yaw_dps, valid_yaw = interpolate_signal(q_us, t, v)
    yaw_dps = np.asarray(yaw_dps).reshape(-1)
    yaw_rps = np.deg2rad(yaw_dps)

    ts, mag = _signal(bus, "steering_angle_calculated")
    tsg, sgn = _signal(bus, "steering_angle_calculated_sign")
    mag_q, valid_mag = interpolate_signal(q_us, ts, mag)
    sgn_q, valid_sgn = interpolate_signal(q_us, tsg, sgn)
    mag_q = np.asarray(mag_q).reshape(-1)
    sgn_q = np.asarray(sgn_q).reshape(-1)
    steer = mag_q * np.where(sgn_q >= 0.5, -1.0, +1.0)

    tb, br = _signal(bus, "brake_pressure")
    brake, valid_brake = interpolate_signal(q_us, tb, br)
    tt, th = _signal(bus, "accelerator_pedal")
    throttle, valid_th = interpolate_signal(q_us, tt, th)
    brake = np.asarray(brake).reshape(-1)
    throttle = np.asarray(throttle).reshape(-1)

    n = len(selected)
    df = pd.DataFrame({
        "dataset": "a2d2", "vehicle_id": vehicle_id, "route_id": route_id,
        "segment_id": np.full(n, "", dtype=object),
        "frame_index_10hz": np.zeros(n, dtype=np.int32),
        "frame_index_source": np.asarray([r.source_index for r in selected], dtype=np.int32),
        "timestamp": q_s,
        "speed_mps": speed_sm.astype(np.float32),
        "speed_pose_mps": np.full(n, np.nan, dtype=np.float32),
        "accel_from_speed_mps2": accel_from_speed.astype(np.float32),
        "accel_imu_forward_mps2": accel_imu.astype(np.float32),
        "steering_deg": steer.astype(np.float32),
        "steering_rate_dps": derivative(steer, q_s).astype(np.float32),
        "yaw_rate_rps": yaw_rps.astype(np.float32),
        "valid_speed": valid_speed & np.isfinite(speed_sm),
        "valid_accel_from_speed": valid_accel,
        "valid_accel_imu": valid_imu & np.isfinite(accel_imu),
        "valid_steer": np.zeros(n, dtype=bool),
        "valid_yaw": valid_yaw & np.isfinite(yaw_rps),
        "video_relpath": np.full(n, "", dtype=object),
    })
    aux = {
        "timestamp": q_s.astype(np.float64),
        "steering_wheel_deg": steer.astype(np.float32),
        "steering_magnitude_deg": np.abs(steer).astype(np.float32),
        "steering_direction_sign": np.sign(steer).astype(np.int8),
        "valid_steering_wheel": valid_mag & valid_sgn & np.isfinite(steer),
        "yaw_rate_dps": yaw_dps.astype(np.float32),
        "acceleration_x_mps2": accel_imu.astype(np.float32),
        "brake_pressure_bar": brake.astype(np.float32),
        "valid_brake": valid_brake & np.isfinite(brake),
        "accelerator_pedal_pct": throttle.astype(np.float32),
        "valid_accelerator": valid_th & np.isfinite(throttle),
    }
    return df, aux


def _quick_video(path: Path, expected_frames: int, fps: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        nf = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        vf = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return nf == expected_frames and abs(vf - fps) < 0.2


def _read_png(fp, m: PngMember):
    fp.seek(m.data_offset)
    data = fp.read(m.size)
    if len(data) != m.size:
        raise RuntimeError(f"short read: {m.name}")
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"decode failed: {m.name}")
    return im


def _encode(frames: list[np.ndarray], dst: Path, width: int, height: int, fps: int, crf: int, preset: str):
    require_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), "-movflags", "+faststart", str(dst),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fr in frames:
            p.stdin.write(fr.tobytes())
        p.stdin.close()
        rc = p.wait()
    except Exception:
        p.kill(); p.wait(); raise
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def _atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()
    shutil.copy2(src, part)
    os.replace(part, dst)


def prepare_session_resume(camera_tar: str | Path, bus_json: str | Path, session_id: str, cfg: A2D2PrepareConfig, vehicle_id: str = "audi_a2d2"):
    camera_tar, bus_json = Path(camera_tar), Path(bus_json)
    root = Path(cfg.processed_root)
    route = f"a2d2_{session_id}"
    cache = root / "cache" / route
    root.mkdir(parents=True, exist_ok=True)
    cfg.work_root.mkdir(parents=True, exist_ok=True)

    bus = load_bus_json(bus_json)
    bus_report = audit_bus(bus)
    if bus_report["accel_dvdt_corr"] < 0.80 or bus_report["steer_yaw_corr"] < 0.50:
        raise RuntimeError(f"bus audit failed: {bus_report}")

    records, pngs, index_report = scan_camera_index_resume(camera_tar, cache, cfg.scan_checkpoint_every)
    selected, skew = select_camera_10hz(records, cfg.target_fps, cfg.max_camera_skew_ms)
    table, aux = build_aligned_signals(bus, selected, vehicle_id, route)

    n = len(selected)
    ns = (n + cfg.segment_frames - 1) // cfg.segment_frames
    rows = []

    with open(camera_tar, "rb", buffering=0) as fp:
        for si in range(ns):
            start = si * cfg.segment_frames
            end = min(n, start + cfg.segment_frames)
            count = end - start
            sid = f"{si:03d}"
            video = root / "videos" / route / f"{sid}.mp4"
            meta = root / "metadata" / route / f"{sid}.npz"
            auxp = root / "aux_metadata" / route / f"{sid}.npz"
            done = root / "done" / route / f"{sid}.json"
            rel = video.relative_to(root).as_posix()

            reusable = (not cfg.overwrite and done.is_file() and meta.is_file() and auxp.is_file() and _quick_video(video, count, cfg.target_fps))
            if reusable:
                print(f"[{si+1:02d}/{ns:02d}] {sid}: SKIP")
            else:
                print(f"[{si+1:02d}/{ns:02d}] {sid}: processing {count} frames")
                jobs = []
                for local_i, rec in enumerate(selected[start:end]):
                    m = pngs[rec.png_member]
                    jobs.append((m.data_offset, local_i, m))
                jobs.sort(key=lambda x: x[0])
                frames = [None] * count
                for _, local_i, m in tqdm(jobs, desc=f"segment {sid} decode", leave=False):
                    im = _read_png(fp, m)
                    frames[local_i] = np.ascontiguousarray(cv2.resize(im, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA))

                with tempfile.TemporaryDirectory(prefix=f"a2d2_{sid}_", dir=str(cfg.work_root)) as td:
                    lv = Path(td) / f"{sid}.mp4"
                    _encode(frames, lv, cfg.width, cfg.height, cfg.target_fps, cfg.crf, cfg.ffmpeg_preset)
                    if not _quick_video(lv, count, cfg.target_fps):
                        raise RuntimeError(f"video validation failed: {sid}")
                    _atomic_copy(lv, video)

                    seg = table.iloc[start:end].copy().reset_index(drop=True)
                    seg["segment_id"] = sid
                    seg["frame_index_10hz"] = np.arange(count, dtype=np.int32)
                    seg["video_relpath"] = rel
                    lm = Path(td) / f"{sid}_meta.npz"
                    write_frame_table(seg, lm)
                    _atomic_copy(lm, meta)
                    la = Path(td) / f"{sid}_aux.npz"
                    np.savez_compressed(la, **{k: np.asarray(v[start:end]) for k, v in aux.items()})
                    _atomic_copy(la, auxp)

                _atomic_json({"segment_id": sid, "num_frames": count, "fps": cfg.target_fps}, done)
                print(f"[{si+1:02d}/{ns:02d}] {sid}: DONE")

            rows.append({
                "dataset": "a2d2", "archive": camera_tar.name, "vehicle_id": vehicle_id,
                "route_id": route, "segment_id": sid, "num_frames": count,
                "duration_s": (count - 1) / cfg.target_fps if count > 1 else 0.0,
                "video_relpath": rel,
                "metadata_relpath": meta.relative_to(root).as_posix(),
                "aux_metadata_relpath": auxp.relative_to(root).as_posix(),
            })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "manifest.csv", index=False)
    report = {
        "session_id": session_id,
        "index_report": index_report,
        "camera_records": len(records),
        "selected_10hz_frames": len(selected),
        "segments": ns,
        "duration_hours": len(selected) / cfg.target_fps / 3600,
        "camera_skew_ms": {"max_abs": float(np.max(np.abs(skew)) / 1000), "p95_abs": float(np.percentile(np.abs(skew), 95) / 1000)},
        "bus_audit": bus_report,
    }
    _atomic_json(report, root / "prepare_report.json")
    return report


prepare_session = prepare_session_resume

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from .calibration import Stage3Calibration, apply_calibration
from .constants import CAN_TARGETS, IMAGENET_MEAN, IMAGENET_STD
from .models import VJEPA21DenseCAN
from .vjepa21 import load_vjepa21_base_encoder
from ..utils.checkpoint import load_checkpoint


def find_stage3_videos(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    videos = sorted(
        {
            *data_dir.rglob("*.mp4"),
            *data_dir.rglob("*.MP4"),
        }
    )
    if not videos:
        raise FileNotFoundError(f"no Stage 3 .mp4 files found under {data_dir}")
    return videos


def infer_public_frame_mapping(labels_for_video: pd.DataFrame) -> tuple[int, int]:
    """Infer raw-public-video frame mapping from labels.csv reference columns.

    The released example labels include ``frame_index`` specifically to locate
    their labeled timestamps in the public source video. The hidden evaluation
    data does not use this mapping: hidden Stage 3 videos are already 10 Hz and
    decoded frame order is 1:1 with sample_index.
    """
    if "frame_index" not in labels_for_video:
        raise ValueError(
            "public calibration requires labels.csv frame_index; "
            "do not infer public FPS from broken container metadata"
        )
    x = labels_for_video["sample_index"].to_numpy(dtype=float)
    y = labels_for_video["frame_index"].to_numpy(dtype=float)
    if len(x) < 2:
        raise ValueError("need at least two public labels to infer frame mapping")

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    mask = np.abs(dx) > 0
    slopes = (dy[mask] / dx[mask]).astype(float)
    slope = float(np.median(slopes))
    stride = int(round(slope))
    if stride <= 0 or abs(slope - stride) > 1e-3:
        raise ValueError(
            f"public frame mapping is not an integer stride: median slope={slope}"
        )
    offsets = y - stride * x
    offset = int(round(float(np.median(offsets))))
    residual = np.max(np.abs(y - (stride * x + offset)))
    if residual > 1e-6:
        raise ValueError(
            f"public labels are not consistent with one frame mapping: "
            f"stride={stride}, offset={offset}, max_residual={residual}"
        )
    return stride, offset


def decode_sampled_video(
    video_path: str | Path,
    *,
    input_height: int,
    input_width: int,
    raw_stride: int = 1,
    raw_offset: int = 0,
) -> np.ndarray:
    """Sequentially decode and resize frames.

    Returns uint8 RGB frames with shape [T,H,W,C]. We intentionally do not trust
    container FPS/PTS metadata because the released public example has known
    metadata irregularities.
    """
    video_path = Path(video_path)
    raw_stride = max(int(raw_stride), 1)
    raw_offset = int(raw_offset)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    raw_index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if raw_index >= raw_offset and (raw_index - raw_offset) % raw_stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(
                rgb,
                (int(input_width), int(input_height)),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(rgb)
        raw_index += 1
    cap.release()

    if not frames:
        raise RuntimeError(
            f"decoded zero sampled frames: {video_path}, "
            f"stride={raw_stride}, offset={raw_offset}"
        )
    return np.stack(frames, axis=0)


def _window_starts(n: int, clip_len: int, stride: int) -> list[int]:
    if n <= clip_len:
        return [0]
    starts = list(range(0, n - clip_len + 1, max(int(stride), 1)))
    if starts[-1] != n - clip_len:
        starts.append(n - clip_len)
    return starts


def _denormalize(values: np.ndarray, stat: Mapping[str, float]) -> np.ndarray:
    return values * float(stat["std"]) + float(stat["mean"])


@torch.inference_mode()
def infer_resized_frames(
    model: torch.nn.Module,
    frames_rgb_uint8: np.ndarray,
    *,
    target_stats: Mapping[str, Mapping[str, float]],
    ordinal_thresholds_mps2: Sequence[float],
    device: str | torch.device = "cuda",
    clip_len: int = 16,
    window_stride: int = 16,
    batch_size: int = 2,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> pd.DataFrame:
    if frames_rgb_uint8.ndim != 4 or frames_rgb_uint8.shape[-1] != 3:
        raise ValueError(
            f"frames must have [T,H,W,3], got {frames_rgb_uint8.shape}"
        )
    n = int(frames_rgb_uint8.shape[0])
    device = torch.device(device)
    model.eval()
    model.to(device)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[None, :, None, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[None, :, None, None, None]

    starts = _window_starts(n, int(clip_len), int(window_stride))
    continuous_names = list(CAN_TARGETS)
    accum = {
        name: np.zeros(n, dtype=np.float64)
        for name in continuous_names
    }
    accum["accel_raw_from_speed_mps2"] = np.zeros(n, dtype=np.float64)

    thresholds = [float(x) for x in ordinal_thresholds_mps2]
    ordinal_accum = np.zeros((n, len(thresholds), 2), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for batch_start in range(0, len(starts), int(batch_size)):
        batch_starts = starts[batch_start : batch_start + int(batch_size)]
        clips: list[np.ndarray] = []
        valid_lengths: list[int] = []

        for start in batch_starts:
            end = min(start + int(clip_len), n)
            clip = frames_rgb_uint8[start:end]
            valid_len = len(clip)
            if valid_len < int(clip_len):
                pad = np.repeat(clip[-1:], int(clip_len) - valid_len, axis=0)
                clip = np.concatenate([clip, pad], axis=0)
            clips.append(clip)
            valid_lengths.append(valid_len)

        x = torch.from_numpy(np.stack(clips, axis=0))
        x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)
        x = (x - mean) / std
        x = x.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            out = model(x)

        ordinal = torch.sigmoid(out["accel_ordinal_logits"]).float().cpu().numpy()
        for local_idx, (start, valid_len) in enumerate(zip(batch_starts, valid_lengths)):
            sl = slice(start, start + valid_len)
            counts[sl] += 1.0

            for name in continuous_names:
                values = out[name][local_idx, :valid_len].float().cpu().numpy()
                values = _denormalize(values, target_stats[name])
                accum[name][sl] += values

            raw = (
                out.get("accel_raw_from_speed_mps2", out["accel_from_speed_mps2"])
                [local_idx, :valid_len]
                .float()
                .cpu()
                .numpy()
            )
            raw = _denormalize(raw, target_stats["accel_from_speed_mps2"])
            accum["accel_raw_from_speed_mps2"][sl] += raw
            ordinal_accum[sl] += ordinal[local_idx, :valid_len]

    if np.any(counts == 0):
        raise RuntimeError("some decoded frames received no model window prediction")

    for name in accum:
        accum[name] /= counts
    ordinal_accum /= counts[:, None, None]

    result = pd.DataFrame(
        {
            "sample_index": np.arange(n, dtype=int),
            "speed_mps": accum["speed_mps"],
            "accel_mps2": accum["accel_from_speed_mps2"],
            "accel_raw_mps2": accum["accel_raw_from_speed_mps2"],
            "steering_deg": accum["steering_deg"],
            "yaw_rate_rps": accum["yaw_rate_rps"],
        }
    )
    for k, threshold in enumerate(thresholds):
        tag = f"{threshold:.2f}".replace(".", "p")
        result[f"ordinal_decel_{tag}"] = ordinal_accum[:, k, 0]
        result[f"ordinal_accel_{tag}"] = ordinal_accum[:, k, 1]
    return result


def extract_video_features(
    model: torch.nn.Module,
    video_path: str | Path,
    *,
    video_id: str,
    target_stats: Mapping[str, Mapping[str, float]],
    ordinal_thresholds_mps2: Sequence[float],
    input_height: int = 288,
    input_width: int = 384,
    raw_stride: int = 1,
    raw_offset: int = 0,
    clip_len: int = 16,
    window_stride: int = 16,
    batch_size: int = 2,
    device: str | torch.device = "cuda",
) -> pd.DataFrame:
    frames = decode_sampled_video(
        video_path,
        input_height=input_height,
        input_width=input_width,
        raw_stride=raw_stride,
        raw_offset=raw_offset,
    )
    result = infer_resized_frames(
        model,
        frames,
        target_stats=target_stats,
        ordinal_thresholds_mps2=ordinal_thresholds_mps2,
        device=device,
        clip_len=clip_len,
        window_stride=window_stride,
        batch_size=batch_size,
    )
    result.insert(0, "ID", str(video_id))
    return result


def build_v4a_model_from_bundle(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict, dict]:
    """Build the frozen V-JEPA v4-A model from an offline submission bundle.

    Expected layout beneath ``model_dir/stage3``:
      - vjepa21b_can_accel_v4a.yaml
      - target_stats.json
      - v4a_best.pt
      - vjepa2_1_vitb_dist_vitG_384.pt
      - vjepa2/   (pinned local V-JEPA source tree)
      - stage3_calibration.json
    """
    stage3_dir = Path(model_dir) / "stage3"
    cfg = yaml.safe_load(
        (stage3_dir / "vjepa21b_can_accel_v4a.yaml").read_text(encoding="utf-8")
    )
    stats = json.loads(
        (stage3_dir / "target_stats.json").read_text(encoding="utf-8")
    )
    mc = cfg["model"]
    dc = cfg["data"]
    fusion_cfg = dict(mc.get("accel_fusion") or {})

    backbone = load_vjepa21_base_encoder(
        stage3_dir / "vjepa2",
        stage3_dir / "vjepa2_1_vitb_dist_vitG_384.pt",
        num_frames=int(dc["clip_len"]),
        out_layers=tuple(mc["out_layers"]),
        freeze=True,
    )
    model = VJEPA21DenseCAN(
        backbone,
        freeze_backbone=True,
        feature_dim=int(mc["feature_dim"]),
        temporal_hidden=int(mc["temporal_hidden"]),
        temporal_layers=int(mc["temporal_layers"]),
        accel_ordinal_thresholds_mps2=mc["accel_ordinal_thresholds_mps2"],
        accel_fusion_enabled=bool(fusion_cfg.get("enabled", True)),
        accel_fusion_hidden=int(fusion_cfg.get("hidden", 64)),
        accel_fusion_gate_init=float(fusion_cfg.get("gate_init", 0.10)),
        accel_fusion_detach_ordinal_inputs=bool(
            fusion_cfg.get("detach_ordinal_inputs", True)
        ),
    )
    metadata = load_checkpoint(
        stage3_dir / "v4a_best.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
        strict=True,
        restore_rng_state=False,
    )
    model.to(device).eval()
    return model, cfg, stats


def predict_stage3_v4a(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Submission-oriented Stage 3 predictor.

    Hidden Stage 3 videos are decoded sequentially at their provided 10-Hz frame
    grid; no FPS metadata is used. One output row is emitted per decoded frame.
    """
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, cfg, stats = build_v4a_model_from_bundle(model_dir, device=device)
    stage3_dir = Path(model_dir) / "stage3"
    calibration = Stage3Calibration.load(stage3_dir / "stage3_calibration.json")

    dc = cfg["data"]
    thresholds = [float(x) for x in cfg["model"]["accel_ordinal_thresholds_mps2"]]
    tables: list[pd.DataFrame] = []
    for video_path in find_stage3_videos(data_dir):
        features = extract_video_features(
            model,
            video_path,
            video_id=video_path.stem,
            target_stats=stats,
            ordinal_thresholds_mps2=thresholds,
            input_height=int(dc["input_height"]),
            input_width=int(dc["input_width"]),
            raw_stride=1,
            raw_offset=0,
            clip_len=int(dc["clip_len"]),
            window_stride=int(dc.get("window_stride", dc["clip_len"])),
            batch_size=2,
            device=device,
        )
        labels = apply_calibration(features, calibration)
        tables.append(labels)

    result = pd.concat(tables, ignore_index=True)
    return result[["ID", "sample_index", "accel_label", "steer_label"]]


__all__ = [
    "find_stage3_videos",
    "infer_public_frame_mapping",
    "decode_sampled_video",
    "infer_resized_frames",
    "extract_video_features",
    "build_v4a_model_from_bundle",
    "predict_stage3_v4a",
]

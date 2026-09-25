from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .metrics import (
    ACCEL_CLASSES,
    STEER_CLASSES,
    dacon_stage3_metrics,
)
from .proxy_metrics import ProxyRule

# Keep image normalization identical to the Stage-3 training/inference stack.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DecisionRule:
    stop_speed_mps: float
    accel_pos_mps2: float
    accel_neg_mps2: float
    accel_bias_mps2: float = 0.0
    steer_left_deg: float = 5.0
    steer_right_deg: float = 5.0
    steering_sign: int = 1
    steering_bias_deg: float = 0.0

    @classmethod
    def from_proxy(cls, rule: ProxyRule) -> "DecisionRule":
        d = float(rule.accel_deadzone_mps2)
        s = float(rule.steer_deadzone_deg)
        return cls(
            stop_speed_mps=float(rule.stop_speed_mps),
            accel_pos_mps2=d,
            accel_neg_mps2=d,
            accel_bias_mps2=0.0,
            steer_left_deg=s,
            steer_right_deg=s,
            steering_sign=1,
            steering_bias_deg=0.0,
        )


@dataclass(frozen=True)
class FusionConfig:
    stop_weight: float = 0.0
    accel_weight: float = 0.0
    steer_weight: float = 0.0
    turn_weight: float = 0.0
    stop_temperature_mps: float = 0.25
    accel_temperature_mps2: float = 0.10
    steer_temperature_deg: float = 2.0

    @classmethod
    def from_mapping(cls, payload: Mapping | None) -> "FusionConfig":
        p = dict(payload or {})
        return cls(
            stop_weight=float(p.get("stop_weight", 0.0)),
            accel_weight=float(p.get("accel_weight", 0.0)),
            steer_weight=float(p.get("steer_weight", 0.0)),
            turn_weight=float(p.get("turn_weight", 0.0)),
            stop_temperature_mps=float(
                p.get("stop_temperature_mps", 0.25)
            ),
            accel_temperature_mps2=float(
                p.get("accel_temperature_mps2", 0.10)
            ),
            steer_temperature_deg=float(
                p.get("steer_temperature_deg", 2.0)
            ),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "stop_weight": float(self.stop_weight),
            "accel_weight": float(self.accel_weight),
            "steer_weight": float(self.steer_weight),
            "turn_weight": float(self.turn_weight),
            "stop_temperature_mps": float(self.stop_temperature_mps),
            "accel_temperature_mps2": float(self.accel_temperature_mps2),
            "steer_temperature_deg": float(self.steer_temperature_deg),
        }


def threshold_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def window_starts(n: int, clip_len: int, stride: int) -> list[int]:
    n = int(n)
    clip_len = int(clip_len)
    stride = max(int(stride), 1)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    if n <= clip_len:
        return [0]
    starts = list(range(0, n - clip_len + 1, stride))
    if starts[-1] != n - clip_len:
        starts.append(n - clip_len)
    return starts


def center_weights(
    clip_len: int,
    *,
    floor: float = 0.25,
) -> np.ndarray:
    """Smooth overlap-add weights with nonzero boundaries.

    ``floor=1`` is uniform averaging. ``floor<1`` gives more trust to frames
    near the temporal center of a clip while never discarding edge frames.
    """
    clip_len = int(clip_len)
    floor = float(floor)
    if clip_len <= 0:
        raise ValueError("clip_len must be positive")
    if not (0.0 < floor <= 1.0):
        raise ValueError(f"floor must be in (0,1], got {floor}")

    x = (np.arange(clip_len, dtype=np.float64) + 0.5) / clip_len
    center = np.sin(np.pi * x) ** 2
    w = floor + (1.0 - floor) * center
    return w.astype(np.float32)


def decode_sampled_video(
    video_path: str | Path,
    *,
    input_height: int,
    input_width: int,
    raw_stride: int = 1,
    raw_offset: int = 0,
) -> np.ndarray:
    """Decode and resize a video to uint8 RGB [T,H,W,C].

    Hidden DACON Stage-3 videos are already on the 10-Hz sample grid, so use
    raw_stride=1 there. The released public example is 20-Hz source video and
    should use the mapping inferred from labels.csv (currently stride=2).
    """
    path = Path(video_path)
    stride = max(int(raw_stride), 1)
    offset = int(raw_offset)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {path}")

    frames: list[np.ndarray] = []
    raw_index = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if raw_index >= offset and (raw_index - offset) % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(
                    rgb,
                    (int(input_width), int(input_height)),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(rgb)
            raw_index += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(
            f"decoded zero sampled frames: {path}, stride={stride}, "
            f"offset={offset}"
        )
    return np.stack(frames, axis=0)


def _denorm(
    values: np.ndarray,
    stats: Mapping[str, Mapping[str, float]],
    name: str,
) -> np.ndarray:
    item = stats[name]
    return (
        values.astype(np.float32, copy=False) * float(item["std"])
        + float(item["mean"])
    )


def _empty_accumulator(
    n: int,
    *,
    stop_k: int,
    accel_k: int,
    turn_k: int,
) -> dict[str, np.ndarray]:
    return {
        "weight": np.zeros(n, dtype=np.float64),
        "speed_mps": np.zeros(n, dtype=np.float64),
        "accel_mps2": np.zeros(n, dtype=np.float64),
        "accel_raw_mps2": np.zeros(n, dtype=np.float64),
        "steering_deg": np.zeros(n, dtype=np.float64),
        "yaw_rate_rps": np.zeros(n, dtype=np.float64),
        "stop_prob": np.zeros((n, stop_k), dtype=np.float64),
        "accel_prob": np.zeros((n, accel_k, 2), dtype=np.float64),
        "steer_prob": np.zeros((n, 3), dtype=np.float64),
        "turn_prob": np.zeros((n, turn_k, 2), dtype=np.float64),
    }


@torch.inference_mode()
def infer_resized_frames_v5c(
    model: torch.nn.Module,
    frames_rgb_uint8: np.ndarray,
    *,
    target_stats: Mapping[str, Mapping[str, float]],
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
    device: str | torch.device = "cuda",
    clip_len: int = 32,
    window_stride: int = 16,
    batch_size: int = 2,
    center_floors: Sequence[float] = (0.25,),
    use_amp: bool = False,
) -> dict[float, pd.DataFrame]:
    """Run overlapping V5 inference and weighted overlap-add aggregation.

    One expensive model pass per window is shared across every requested
    ``center_floor``. Classifier heads are averaged in probability space.
    """
    if frames_rgb_uint8.ndim != 4 or frames_rgb_uint8.shape[-1] != 3:
        raise ValueError(
            f"frames must have [T,H,W,3], got {frames_rgb_uint8.shape}"
        )
    n = int(frames_rgb_uint8.shape[0])
    device = torch.device(device)
    model = model.to(device).eval()

    clip_len = int(clip_len)
    starts = window_starts(n, clip_len, int(window_stride))
    floors = tuple(dict.fromkeys(float(x) for x in center_floors))
    weight_vectors = {
        floor: center_weights(clip_len, floor=floor).astype(np.float64)
        for floor in floors
    }

    stop_thr = tuple(float(x) for x in stop_thresholds_mps)
    accel_thr = tuple(float(x) for x in accel_thresholds_mps2)
    turn_thr = tuple(float(x) for x in turn_thresholds_rps)

    accum = {
        floor: _empty_accumulator(
            n,
            stop_k=len(stop_thr),
            accel_k=len(accel_thr),
            turn_k=len(turn_thr),
        )
        for floor in floors
    }

    mean = torch.tensor(
        _IMAGENET_MEAN, dtype=torch.float32
    )[None, :, None, None, None]
    std = torch.tensor(
        _IMAGENET_STD, dtype=torch.float32
    )[None, :, None, None, None]

    for batch_start in range(0, len(starts), int(batch_size)):
        batch_starts = starts[batch_start : batch_start + int(batch_size)]
        clips: list[np.ndarray] = []
        valid_lengths: list[int] = []

        for start in batch_starts:
            end = min(start + clip_len, n)
            clip = frames_rgb_uint8[start:end]
            valid_len = len(clip)
            if valid_len <= 0:
                raise RuntimeError("empty clip generated")
            if valid_len < clip_len:
                pad = np.repeat(clip[-1:], clip_len - valid_len, axis=0)
                clip = np.concatenate([clip, pad], axis=0)
            clips.append(clip)
            valid_lengths.append(valid_len)

        x = (
            torch.from_numpy(np.stack(clips, axis=0))
            .permute(0, 4, 1, 2, 3)
            .float()
            .div_(255.0)
        )
        x = (x - mean) / std
        x = x.to(device, non_blocking=True)

        amp_dtype = torch.bfloat16
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            out = model(x)

        speed = _denorm(
            out["speed_mps"].float().cpu().numpy(),
            target_stats,
            "speed_mps",
        )
        accel = _denorm(
            out["accel_from_speed_mps2"].float().cpu().numpy(),
            target_stats,
            "accel_from_speed_mps2",
        )
        raw_norm = out.get(
            "accel_raw_from_speed_mps2",
            out["accel_from_speed_mps2"],
        )
        raw = _denorm(
            raw_norm.float().cpu().numpy(),
            target_stats,
            "accel_from_speed_mps2",
        )
        steering = _denorm(
            out["steering_deg"].float().cpu().numpy(),
            target_stats,
            "steering_deg",
        )
        yaw = _denorm(
            out["yaw_rate_rps"].float().cpu().numpy(),
            target_stats,
            "yaw_rate_rps",
        )

        stop_prob = torch.sigmoid(
            out["stop_ordinal_logits"]
        ).float().cpu().numpy()
        accel_prob = torch.sigmoid(
            out["accel_ordinal_logits"]
        ).float().cpu().numpy()
        steer_prob = torch.softmax(
            out["steer_direction_logits"].float(),
            dim=-1,
        ).cpu().numpy()
        turn_prob = torch.sigmoid(
            out["turn_ordinal_logits"]
        ).float().cpu().numpy()

        if stop_prob.shape[-1] != len(stop_thr):
            raise ValueError("stop threshold/output size mismatch")
        if accel_prob.shape[-2:] != (len(accel_thr), 2):
            raise ValueError("accel threshold/output size mismatch")
        if turn_prob.shape[-2:] != (len(turn_thr), 2):
            raise ValueError("turn threshold/output size mismatch")

        for j, (start, valid_len) in enumerate(
            zip(batch_starts, valid_lengths, strict=True)
        ):
            sl = slice(start, start + valid_len)
            for floor, a in accum.items():
                w = weight_vectors[floor][:valid_len]
                a["weight"][sl] += w
                a["speed_mps"][sl] += speed[j, :valid_len] * w
                a["accel_mps2"][sl] += accel[j, :valid_len] * w
                a["accel_raw_mps2"][sl] += raw[j, :valid_len] * w
                a["steering_deg"][sl] += steering[j, :valid_len] * w
                a["yaw_rate_rps"][sl] += yaw[j, :valid_len] * w
                a["stop_prob"][sl] += stop_prob[j, :valid_len] * w[:, None]
                a["accel_prob"][sl] += (
                    accel_prob[j, :valid_len] * w[:, None, None]
                )
                a["steer_prob"][sl] += (
                    steer_prob[j, :valid_len] * w[:, None]
                )
                a["turn_prob"][sl] += (
                    turn_prob[j, :valid_len] * w[:, None, None]
                )

    result: dict[float, pd.DataFrame] = {}
    for floor, a in accum.items():
        weight = a.pop("weight")
        if np.any(weight <= 0):
            missing = np.flatnonzero(weight <= 0)[:10].tolist()
            raise RuntimeError(
                f"some frames received zero overlap weight: {missing}"
            )

        denom1 = weight
        denom2 = weight[:, None]
        denom3 = weight[:, None, None]

        frame = pd.DataFrame(
            {
                "sample_index": np.arange(n, dtype=np.int64),
                "speed_mps": a["speed_mps"] / denom1,
                "accel_mps2": a["accel_mps2"] / denom1,
                "accel_raw_mps2": a["accel_raw_mps2"] / denom1,
                "steering_deg": a["steering_deg"] / denom1,
                "yaw_rate_rps": a["yaw_rate_rps"] / denom1,
            }
        )

        stop_avg = a["stop_prob"] / denom2
        accel_avg = a["accel_prob"] / denom3
        steer_avg = a["steer_prob"] / denom2
        turn_avg = a["turn_prob"] / denom3

        for k, thr in enumerate(stop_thr):
            frame[f"stop_p_{threshold_tag(thr)}"] = stop_avg[:, k]

        for k, thr in enumerate(accel_thr):
            tag = threshold_tag(thr)
            frame[f"accel_decel_p_{tag}"] = accel_avg[:, k, 0]
            frame[f"accel_accel_p_{tag}"] = accel_avg[:, k, 1]

        frame["steer_p_left"] = steer_avg[:, 0]
        frame["steer_p_straight"] = steer_avg[:, 1]
        frame["steer_p_right"] = steer_avg[:, 2]

        for k, thr in enumerate(turn_thr):
            tag = threshold_tag(thr)
            frame[f"turn_neg_p_{tag}"] = turn_avg[:, k, 0]
            frame[f"turn_pos_p_{tag}"] = turn_avg[:, k, 1]

        result[floor] = frame

    return result


def extract_video_features_v5c(
    model: torch.nn.Module,
    video_path: str | Path,
    *,
    target_stats: Mapping[str, Mapping[str, float]],
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
    input_height: int = 288,
    input_width: int = 384,
    raw_stride: int = 1,
    raw_offset: int = 0,
    clip_len: int = 32,
    window_stride: int = 16,
    batch_size: int = 2,
    center_floors: Sequence[float] = (0.25,),
    device: str | torch.device = "cuda",
    use_amp: bool = False,
) -> dict[float, pd.DataFrame]:
    frames = decode_sampled_video(
        video_path,
        input_height=input_height,
        input_width=input_width,
        raw_stride=raw_stride,
        raw_offset=raw_offset,
    )
    return infer_resized_frames_v5c(
        model,
        frames,
        target_stats=target_stats,
        stop_thresholds_mps=stop_thresholds_mps,
        accel_thresholds_mps2=accel_thresholds_mps2,
        turn_thresholds_rps=turn_thresholds_rps,
        device=device,
        clip_len=clip_len,
        window_stride=window_stride,
        batch_size=batch_size,
        center_floors=center_floors,
        use_amp=use_amp,
    )


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p) - np.log1p(-p)


def _interp_probability(
    thresholds: Sequence[float],
    matrix: np.ndarray,
    target_threshold: float,
) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(thresholds):
        raise ValueError(
            f"probability matrix mismatch: {matrix.shape}, "
            f"thresholds={len(thresholds)}"
        )
    t = float(target_threshold)
    if t <= thresholds[0]:
        return matrix[:, 0]
    if t >= thresholds[-1]:
        return matrix[:, -1]
    hi = int(np.searchsorted(thresholds, t, side="right"))
    lo = hi - 1
    alpha = (t - thresholds[lo]) / (thresholds[hi] - thresholds[lo])
    return (1.0 - alpha) * matrix[:, lo] + alpha * matrix[:, hi]


def _matrix_from_columns(
    frame: pd.DataFrame,
    prefix: str,
    thresholds: Sequence[float],
) -> np.ndarray:
    cols = [
        f"{prefix}{threshold_tag(t)}"
        for t in thresholds
    ]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return frame[cols].to_numpy(dtype=np.float64)


def _turn_mean_probability(
    frame: pd.DataFrame,
    thresholds: Sequence[float],
    direction: str,
) -> np.ndarray:
    prefix = "turn_neg_p_" if direction == "neg" else "turn_pos_p_"
    return _matrix_from_columns(frame, prefix, thresholds).mean(axis=1)


def labels_with_aux_fusion(
    frame: pd.DataFrame,
    *,
    rule: DecisionRule,
    fusion: FusionConfig | Mapping | None,
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
    accel_source: str = "fused",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert continuous + auxiliary predictions to Stage-3 labels.

    The continuous decision boundary is preserved exactly when every fusion
    weight is zero. Auxiliary heads contribute log-odds evidence around that
    boundary rather than replacing the physical continuous prediction.
    """
    f = (
        fusion
        if isinstance(fusion, FusionConfig)
        else FusionConfig.from_mapping(fusion)
    )

    speed = frame["speed_mps"].to_numpy(dtype=np.float64)

    if str(accel_source) == "raw":
        accel = frame["accel_raw_mps2"].to_numpy(dtype=np.float64)
    elif str(accel_source) == "fused":
        accel = frame["accel_mps2"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"unknown accel_source: {accel_source}")

    accel = accel + float(rule.accel_bias_mps2)
    steer = (
        int(rule.steering_sign)
        * frame["steering_deg"].to_numpy(dtype=np.float64)
        + float(rule.steering_bias_deg)
    )

    # STOP: continuous speed boundary + learned multi-threshold speed CDF.
    stop_temp = max(float(f.stop_temperature_mps), 1e-4)
    stop_z = (
        float(rule.stop_speed_mps) - speed
    ) / stop_temp

    if abs(float(f.stop_weight)) > 1e-12:
        stop_matrix = _matrix_from_columns(
            frame,
            "stop_p_",
            stop_thresholds_mps,
        )
        p_stop = _interp_probability(
            stop_thresholds_mps,
            stop_matrix,
            float(rule.stop_speed_mps),
        )
        stop_z = stop_z + float(f.stop_weight) * _logit(p_stop)

    stopped = stop_z > 0.0

    # Dynamic acceleration: compare positive/deceleration evidence against
    # CONSTANT reference score 0. Continuous thresholds stay exact at w=0.
    accel_temp = max(float(f.accel_temperature_mps2), 1e-4)
    z_acc = (
        accel - float(rule.accel_pos_mps2)
    ) / accel_temp
    z_dec = (
        -accel - float(rule.accel_neg_mps2)
    ) / accel_temp

    if abs(float(f.accel_weight)) > 1e-12:
        p_acc = _interp_probability(
            accel_thresholds_mps2,
            _matrix_from_columns(
                frame,
                "accel_accel_p_",
                accel_thresholds_mps2,
            ),
            float(rule.accel_pos_mps2),
        )
        p_dec = _interp_probability(
            accel_thresholds_mps2,
            _matrix_from_columns(
                frame,
                "accel_decel_p_",
                accel_thresholds_mps2,
            ),
            float(rule.accel_neg_mps2),
        )
        z_acc = z_acc + float(f.accel_weight) * _logit(p_acc)
        z_dec = z_dec + float(f.accel_weight) * _logit(p_dec)

    accel_labels = np.full(len(frame), "CONSTANT", dtype=object)
    accel_labels[stopped] = "STOPPED"
    moving = ~stopped
    choose_acc = moving & (z_acc > 0.0) & (z_acc >= z_dec)
    choose_dec = moving & (z_dec > 0.0) & (z_dec > z_acc)
    accel_labels[choose_acc] = "ACCELERATING"
    accel_labels[choose_dec] = "DECELERATING"

    # Steering: continuous LEFT/RIGHT margins against STRAIGHT score 0.
    steer_temp = max(float(f.steer_temperature_deg), 1e-4)
    z_left = (
        -steer - float(rule.steer_left_deg)
    ) / steer_temp
    z_right = (
        steer - float(rule.steer_right_deg)
    ) / steer_temp

    p_left = frame["steer_p_left"].to_numpy(dtype=np.float64)
    p_straight = frame["steer_p_straight"].to_numpy(dtype=np.float64)
    p_right = frame["steer_p_right"].to_numpy(dtype=np.float64)

    # Public calibration has steering_sign=-1. Auxiliary direction was trained
    # in the raw comma/A2D2 steering convention, so swap L/R under sign flip.
    if int(rule.steering_sign) < 0:
        p_left, p_right = p_right.copy(), p_left.copy()

    if abs(float(f.steer_weight)) > 1e-12:
        z_left = z_left + float(f.steer_weight) * (
            np.log(_clip_prob(p_left)) - np.log(_clip_prob(p_straight))
        )
        z_right = z_right + float(f.steer_weight) * (
            np.log(_clip_prob(p_right)) - np.log(_clip_prob(p_straight))
        )

    if abs(float(f.turn_weight)) > 1e-12:
        p_turn_left = _turn_mean_probability(
            frame, turn_thresholds_rps, "neg"
        )
        p_turn_right = _turn_mean_probability(
            frame, turn_thresholds_rps, "pos"
        )
        if int(rule.steering_sign) < 0:
            p_turn_left, p_turn_right = (
                p_turn_right.copy(),
                p_turn_left.copy(),
            )
        z_left = z_left + float(f.turn_weight) * _logit(p_turn_left)
        z_right = z_right + float(f.turn_weight) * _logit(p_turn_right)

    steer_labels = np.full(len(frame), "STRAIGHT", dtype=object)
    choose_left = (z_left > 0.0) & (z_left >= z_right)
    choose_right = (z_right > 0.0) & (z_right > z_left)
    steer_labels[choose_left] = "LEFT"
    steer_labels[choose_right] = "RIGHT"

    return accel_labels.astype(str), steer_labels.astype(str)


def _truth_proxy_labels(
    frame: pd.DataFrame,
    rule: ProxyRule,
) -> tuple[np.ndarray, np.ndarray]:
    speed = frame["gt_speed_mps"].to_numpy(dtype=np.float64)
    accel = frame["gt_accel_mps2"].to_numpy(dtype=np.float64)
    steer = frame["gt_steering_deg"].to_numpy(dtype=np.float64)

    accel_labels = np.full(len(frame), "CONSTANT", dtype=object)
    stopped = speed <= float(rule.stop_speed_mps)
    moving = ~stopped
    accel_labels[stopped] = "STOPPED"
    accel_labels[
        moving & (accel > float(rule.accel_deadzone_mps2))
    ] = "ACCELERATING"
    accel_labels[
        moving & (accel < -float(rule.accel_deadzone_mps2))
    ] = "DECELERATING"

    steer_labels = np.full(len(frame), "STRAIGHT", dtype=object)
    steer_labels[
        steer < -float(rule.steer_deadzone_deg)
    ] = "LEFT"
    steer_labels[
        steer > float(rule.steer_deadzone_deg)
    ] = "RIGHT"
    return accel_labels.astype(str), steer_labels.astype(str)


def _per_class(
    truth_accel: np.ndarray,
    pred_accel: np.ndarray,
    truth_steer: np.ndarray,
    pred_steer: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    a = f1_score(
        truth_accel,
        pred_accel,
        labels=list(ACCEL_CLASSES),
        average=None,
        zero_division=0,
    )
    moving = truth_accel != "STOPPED"
    if moving.any():
        s = f1_score(
            truth_steer[moving],
            pred_steer[moving],
            labels=list(STEER_CLASSES),
            average=None,
            zero_division=0,
        )
    else:
        s = np.zeros(len(STEER_CLASSES), dtype=np.float64)

    for name, value in zip(ACCEL_CLASSES, a, strict=True):
        result[f"f1_accel_{name}"] = float(value)
    for name, value in zip(STEER_CLASSES, s, strict=True):
        result[f"f1_steer_{name}"] = float(value)
    return result


def score_proxy_table(
    frame: pd.DataFrame,
    proxy_rules: Mapping[str, Mapping[str, float] | ProxyRule],
    *,
    fusion: FusionConfig | Mapping | None,
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
) -> dict[str, float]:
    required = {
        "gt_speed_mps",
        "gt_accel_mps2",
        "gt_steering_deg",
        "speed_mps",
        "accel_mps2",
        "accel_raw_mps2",
        "steering_deg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"proxy table missing columns: {missing}")

    valid = np.ones(len(frame), dtype=bool)
    for col in (
        "valid_speed",
        "valid_accel",
        "valid_steer",
    ):
        if col in frame.columns:
            valid &= frame[col].to_numpy(dtype=bool)

    finite_cols = list(required)
    for col in finite_cols:
        valid &= np.isfinite(frame[col].to_numpy(dtype=np.float64))

    work = frame.loc[valid].reset_index(drop=True)
    if work.empty:
        raise ValueError("no valid rows for proxy scoring")

    stage3_scores = []
    accel_scores = []
    steer_scores = []
    result: dict[str, float] = {}

    for name, raw_rule in proxy_rules.items():
        rule = (
            raw_rule
            if isinstance(raw_rule, ProxyRule)
            else ProxyRule.from_mapping(raw_rule)
        )
        truth_a, truth_s = _truth_proxy_labels(work, rule)
        pred_a, pred_s = labels_with_aux_fusion(
            work,
            rule=DecisionRule.from_proxy(rule),
            fusion=fusion,
            stop_thresholds_mps=stop_thresholds_mps,
            accel_thresholds_mps2=accel_thresholds_mps2,
            turn_thresholds_rps=turn_thresholds_rps,
            accel_source="fused",
        )
        metrics = dacon_stage3_metrics(
            truth_a,
            pred_a,
            truth_s,
            pred_s,
        )
        prefix = f"proxy/{name}"
        result[f"{prefix}/stage3_score"] = float(
            metrics["stage3_score"]
        )
        result[f"{prefix}/accel_macro_f1"] = float(
            metrics["accel_macro_f1"]
        )
        result[f"{prefix}/steer_macro_f1"] = float(
            metrics["steer_macro_f1"]
        )
        for k, v in _per_class(
            truth_a, pred_a, truth_s, pred_s
        ).items():
            result[f"{prefix}/{k}"] = float(v)

        stage3_scores.append(float(metrics["stage3_score"]))
        accel_scores.append(float(metrics["accel_macro_f1"]))
        steer_scores.append(float(metrics["steer_macro_f1"]))

    result["proxy/robust_mean_stage3_score"] = float(
        np.mean(stage3_scores)
    )
    result["proxy/robust_min_stage3_score"] = float(
        np.min(stage3_scores)
    )
    result["proxy/robust_mean_accel_macro_f1"] = float(
        np.mean(accel_scores)
    )
    result["proxy/robust_mean_steer_macro_f1"] = float(
        np.mean(steer_scores)
    )

    # Continuous shape diagnostics, useful when overlap improves smoothness but
    # not a particular thresholded proxy.
    gt_a = work["gt_accel_mps2"].to_numpy(dtype=np.float64)
    pr_a = work["accel_mps2"].to_numpy(dtype=np.float64)
    result["diag/accel/mae"] = float(np.mean(np.abs(pr_a - gt_a)))
    result["diag/accel/rmse"] = float(
        np.sqrt(np.mean(np.square(pr_a - gt_a)))
    )
    if np.std(gt_a) > 1e-12 and np.std(pr_a) > 1e-12:
        result["diag/accel/correlation"] = float(
            np.corrcoef(gt_a, pr_a)[0, 1]
        )
        result["diag/accel/pred_to_gt_std_ratio"] = float(
            np.std(pr_a) / np.std(gt_a)
        )
    else:
        result["diag/accel/correlation"] = 0.0
        result["diag/accel/pred_to_gt_std_ratio"] = 0.0

    return result


def grid_search_fusion(
    tune_frame: pd.DataFrame,
    proxy_rules: Mapping[str, Mapping[str, float] | ProxyRule],
    *,
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
    stop_weights: Sequence[float],
    accel_weights: Sequence[float],
    steer_weights: Sequence[float],
    turn_weights: Sequence[float],
    stop_temperature_mps: float = 0.25,
    accel_temperature_mps2: float = 0.10,
    steer_temperature_deg: float = 2.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sw, aw, stw, tw in product(
        stop_weights,
        accel_weights,
        steer_weights,
        turn_weights,
    ):
        cfg = FusionConfig(
            stop_weight=float(sw),
            accel_weight=float(aw),
            steer_weight=float(stw),
            turn_weight=float(tw),
            stop_temperature_mps=float(stop_temperature_mps),
            accel_temperature_mps2=float(accel_temperature_mps2),
            steer_temperature_deg=float(steer_temperature_deg),
        )
        score = score_proxy_table(
            tune_frame,
            proxy_rules,
            fusion=cfg,
            stop_thresholds_mps=stop_thresholds_mps,
            accel_thresholds_mps2=accel_thresholds_mps2,
            turn_thresholds_rps=turn_thresholds_rps,
        )
        rows.append(
            {
                **cfg.as_dict(),
                "robust_stage3": score[
                    "proxy/robust_mean_stage3_score"
                ],
                "robust_min_stage3": score[
                    "proxy/robust_min_stage3_score"
                ],
                "robust_accel": score[
                    "proxy/robust_mean_accel_macro_f1"
                ],
                "robust_steer": score[
                    "proxy/robust_mean_steer_macro_f1"
                ],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["robust_stage3", "robust_min_stage3"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def smooth_feature_table(
    frame: pd.DataFrame,
    *,
    accel_window: int,
    steer_window: int,
) -> pd.DataFrame:
    out = frame.copy()
    accel_cols = [
        c
        for c in out.columns
        if (
            c in {
                "speed_mps",
                "accel_mps2",
                "accel_raw_mps2",
            }
            or c.startswith("stop_p_")
            or c.startswith("accel_decel_p_")
            or c.startswith("accel_accel_p_")
        )
    ]
    steer_cols = [
        c
        for c in out.columns
        if (
            c == "steering_deg"
            or c.startswith("steer_p_")
            or c.startswith("turn_neg_p_")
            or c.startswith("turn_pos_p_")
        )
    ]
    for col in accel_cols:
        out[col] = centered_mean(
            out[col].to_numpy(dtype=np.float64),
            accel_window,
        )
    for col in steer_cols:
        out[col] = centered_mean(
            out[col].to_numpy(dtype=np.float64),
            steer_window,
        )
    return out


def apply_public_calibration_with_fusion(
    features: pd.DataFrame,
    calibration: Mapping,
    *,
    fusion: FusionConfig | Mapping | None,
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
) -> pd.DataFrame:
    """Apply the existing target-domain calibration plus V5 auxiliary fusion."""
    accel_cfg = dict(calibration["accel"])
    steer_cfg = dict(calibration["steer"])

    f = (
        fusion
        if isinstance(fusion, FusionConfig)
        else FusionConfig.from_mapping(fusion)
    )

    outputs: list[pd.DataFrame] = []
    if "ID" in features.columns:
        grouped = features.groupby("ID", sort=False)
    else:
        grouped = [(None, features)]

    for video_id, group in grouped:
        g = group.sort_values("sample_index").reset_index(drop=True)
        g = smooth_feature_table(
            g,
            accel_window=int(accel_cfg["smoothing_window"]),
            steer_window=int(steer_cfg["smoothing_window"]),
        )

        rule = DecisionRule(
            stop_speed_mps=float(accel_cfg["stop_speed_mps"]),
            accel_pos_mps2=float(
                accel_cfg["accel_deadzone_pos_mps2"]
            ),
            accel_neg_mps2=float(
                accel_cfg["accel_deadzone_neg_mps2"]
            ),
            accel_bias_mps2=float(
                accel_cfg["accel_bias_mps2"]
            ),
            steer_left_deg=float(steer_cfg["left_deadzone_deg"]),
            steer_right_deg=float(steer_cfg["right_deadzone_deg"]),
            steering_sign=int(steer_cfg["steering_sign"]),
            steering_bias_deg=float(steer_cfg["steering_bias_deg"]),
        )

        accel_labels, steer_labels = labels_with_aux_fusion(
            g,
            rule=rule,
            fusion=f,
            stop_thresholds_mps=stop_thresholds_mps,
            accel_thresholds_mps2=accel_thresholds_mps2,
            turn_thresholds_rps=turn_thresholds_rps,
            accel_source=str(accel_cfg["source"]),
        )
        result = pd.DataFrame(
            {
                "sample_index": g["sample_index"].to_numpy(dtype=int),
                "accel_label": accel_labels,
                "steer_label": steer_labels,
            }
        )
        if video_id is not None:
            result.insert(0, "ID", str(video_id))
        outputs.append(result)

    return pd.concat(outputs, ignore_index=True)


def score_public_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    calibration: Mapping,
    *,
    fusion: FusionConfig | Mapping | None,
    stop_thresholds_mps: Sequence[float],
    accel_thresholds_mps2: Sequence[float],
    turn_thresholds_rps: Sequence[float],
) -> dict[str, float]:
    pred = apply_public_calibration_with_fusion(
        features,
        calibration,
        fusion=fusion,
        stop_thresholds_mps=stop_thresholds_mps,
        accel_thresholds_mps2=accel_thresholds_mps2,
        turn_thresholds_rps=turn_thresholds_rps,
    )
    truth = labels[
        ["ID", "sample_index", "accel_label", "steer_label"]
    ].copy()
    merged = truth.merge(
        pred,
        on=["ID", "sample_index"],
        how="inner",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise ValueError(
            f"public labels alignment mismatch: {len(merged)} vs {len(truth)}"
        )
    return dacon_stage3_metrics(
        merged["accel_label_true"],
        merged["accel_label_pred"],
        merged["steer_label_true"],
        merged["steer_label_pred"],
    )


__all__ = [
    "DecisionRule",
    "FusionConfig",
    "apply_public_calibration_with_fusion",
    "center_weights",
    "decode_sampled_video",
    "extract_video_features_v5c",
    "grid_search_fusion",
    "infer_resized_frames_v5c",
    "labels_with_aux_fusion",
    "score_proxy_table",
    "score_public_labels",
    "threshold_tag",
    "window_starts",
]

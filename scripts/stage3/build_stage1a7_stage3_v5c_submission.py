from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

STAGE3_V5C_BLOCK = '# ---------------------------------------------------------------------------\n# Stage 3: V-JEPA 2.1-B v5-C overlap + auxiliary fusion\n# ---------------------------------------------------------------------------\n# Stage 1 / Stage 2 above are intentionally kept byte-identical to stage1_a7.\n#\n# Runtime assets:\n#   model/stage3/model.ts\n#   model/stage3/target_stats.json\n#   model/stage3/calibration.json\n#   model/stage3/v5c_selection.json\n#\n# Hidden Stage 3 videos are already on the 10-Hz evaluation frame grid.\n# Each video is processed independently, as required by the competition rules.\n\nS3_V5C_NUM_FRAMES = 32\nS3_V5C_STRIDE = 8\nS3_V5C_CENTER_FLOOR = 0.25\nS3_V5C_HEIGHT = 288\nS3_V5C_WIDTH = 384\nS3_V5C_BATCH = 2\n\nS3_V5C_STOP_THRESHOLDS = np.asarray(\n    [0.10, 0.30, 0.50, 1.00, 2.00],\n    dtype=np.float32,\n)\nS3_V5C_TURN_THRESHOLDS = np.asarray(\n    [0.01, 0.03, 0.05],\n    dtype=np.float32,\n)\n\nS3_V5C_MEAN = torch.tensor(\n    [0.485, 0.456, 0.406],\n    dtype=torch.float32,\n)[:, None, None, None]\nS3_V5C_STD = torch.tensor(\n    [0.229, 0.224, 0.225],\n    dtype=torch.float32,\n)[:, None, None, None]\n\n\ndef _stage3_v5c_resize_rgb(bgr: np.ndarray) -> np.ndarray:\n    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n    return cv2.resize(\n        rgb,\n        (S3_V5C_WIDTH, S3_V5C_HEIGHT),\n        interpolation=cv2.INTER_AREA,\n    )\n\n\ndef _stage3_v5c_decode(path: Path) -> np.ndarray:\n    cap = cv2.VideoCapture(str(path))\n    if not cap.isOpened():\n        cap.release()\n        raise ValueError(f"cannot open Stage 3 video: {path.name}")\n\n    frames = []\n    try:\n        while True:\n            ok, bgr = cap.read()\n            if not ok:\n                break\n            frames.append(_stage3_v5c_resize_rgb(bgr))\n    finally:\n        cap.release()\n\n    if not frames:\n        raise ValueError(f"cannot decode Stage 3 video: {path.name}")\n\n    return np.stack(frames, axis=0)\n\n\ndef _stage3_v5c_window_starts(n: int):\n    if n <= S3_V5C_NUM_FRAMES:\n        return [0]\n\n    last = n - S3_V5C_NUM_FRAMES\n    starts = list(range(0, last + 1, S3_V5C_STRIDE))\n    if starts[-1] != last:\n        starts.append(last)\n    return starts\n\n\ndef _stage3_v5c_center_weights():\n    x = (\n        np.arange(S3_V5C_NUM_FRAMES, dtype=np.float64) + 0.5\n    ) / S3_V5C_NUM_FRAMES\n    center = np.sin(np.pi * x) ** 2\n    return (\n        S3_V5C_CENTER_FLOOR\n        + (1.0 - S3_V5C_CENTER_FLOOR) * center\n    ).astype(np.float64)\n\n\ndef _stage3_v5c_denormalize(\n    values: np.ndarray,\n    stats: dict,\n    name: str,\n) -> np.ndarray:\n    item = stats[name]\n    return (\n        values.astype(np.float32, copy=False) * float(item["std"])\n        + float(item["mean"])\n    )\n\n\ndef _stage3_v5c_centered_mean(\n    values: np.ndarray,\n    window: int,\n) -> np.ndarray:\n    window = int(window)\n    if window <= 1:\n        return np.asarray(values, dtype=np.float64)\n    if window % 2 == 0:\n        raise ValueError(\n            f"Stage 3 smoothing window must be odd, got {window}"\n        )\n    return (\n        pd.Series(np.asarray(values, dtype=np.float64))\n        .rolling(window=window, center=True, min_periods=1)\n        .mean()\n        .to_numpy(dtype=np.float64)\n    )\n\n\ndef _stage3_v5c_clip_prob(values: np.ndarray) -> np.ndarray:\n    return np.clip(\n        np.asarray(values, dtype=np.float64),\n        1e-4,\n        1.0 - 1e-4,\n    )\n\n\ndef _stage3_v5c_logit(values: np.ndarray) -> np.ndarray:\n    p = _stage3_v5c_clip_prob(values)\n    return np.log(p) - np.log1p(-p)\n\n\ndef _stage3_v5c_interp_probability(\n    thresholds: np.ndarray,\n    matrix: np.ndarray,\n    target: float,\n) -> np.ndarray:\n    thresholds = np.asarray(thresholds, dtype=np.float64)\n    matrix = np.asarray(matrix, dtype=np.float64)\n    target = float(target)\n\n    if target <= thresholds[0]:\n        return matrix[:, 0]\n    if target >= thresholds[-1]:\n        return matrix[:, -1]\n\n    hi = int(np.searchsorted(thresholds, target, side="right"))\n    lo = hi - 1\n    alpha = (\n        (target - thresholds[lo])\n        / (thresholds[hi] - thresholds[lo])\n    )\n    return (\n        (1.0 - alpha) * matrix[:, lo]\n        + alpha * matrix[:, hi]\n    )\n\n\ndef _stage3_v5c_flush_batch(\n    model,\n    clips,\n    starts,\n    valid_lengths,\n    device,\n    stats,\n    center_weights,\n    accum,\n):\n    real_batch = len(clips)\n    if real_batch == 0:\n        return\n\n    while len(clips) < S3_V5C_BATCH:\n        clips.append(clips[-1].copy())\n        starts.append(starts[-1])\n        valid_lengths.append(0)\n\n    x = (\n        torch.from_numpy(np.stack(clips, axis=0))\n        .permute(0, 4, 1, 2, 3)\n        .float()\n        .div_(255.0)\n    )\n    x = (x - S3_V5C_MEAN[None]) / S3_V5C_STD[None]\n    x = x.to(\n        device=device,\n        dtype=torch.float32,\n        non_blocking=True,\n    )\n\n    # FP32 is deliberate. Earlier submission testing hit a mixed-dtype\n    # attention failure under autocast.\n    outputs = model(x)\n    if not isinstance(outputs, (tuple, list)) or len(outputs) != 7:\n        raise RuntimeError(\n            "Stage 3 v5-C model.ts must return "\n            "(speed, fused_accel, raw_accel, steering, "\n            "stop_prob, steer_prob, turn_prob)"\n        )\n\n    (\n        speed_n,\n        fused_n,\n        raw_n,\n        steering_n,\n        stop_prob,\n        steer_prob,\n        turn_prob,\n    ) = [\n        tensor.float().cpu().numpy()\n        for tensor in outputs\n    ]\n\n    speed = _stage3_v5c_denormalize(\n        speed_n, stats, "speed_mps"\n    )\n    fused = _stage3_v5c_denormalize(\n        fused_n, stats, "accel_from_speed_mps2"\n    )\n    raw = _stage3_v5c_denormalize(\n        raw_n, stats, "accel_from_speed_mps2"\n    )\n    steering = _stage3_v5c_denormalize(\n        steering_n, stats, "steering_deg"\n    )\n\n    for i in range(real_batch):\n        n = int(valid_lengths[i])\n        if n <= 0:\n            continue\n\n        start = int(starts[i])\n        sl = slice(start, start + n)\n        w = center_weights[:n]\n\n        accum["weight"][sl] += w\n        accum["speed"][sl] += speed[i, :n] * w\n        accum["fused"][sl] += fused[i, :n] * w\n        accum["raw"][sl] += raw[i, :n] * w\n        accum["steering"][sl] += steering[i, :n] * w\n        accum["stop"][sl] += stop_prob[i, :n] * w[:, None]\n        accum["steer"][sl] += steer_prob[i, :n] * w[:, None]\n        accum["turn"][sl] += turn_prob[i, :n] * w[:, None, None]\n\n\ndef _stage3_v5c_predict_features(\n    path: Path,\n    model,\n    device,\n    stats: dict,\n):\n    frames = _stage3_v5c_decode(path)\n    n = int(len(frames))\n    starts_all = _stage3_v5c_window_starts(n)\n    center_weights = _stage3_v5c_center_weights()\n\n    accum = {\n        "weight": np.zeros(n, dtype=np.float64),\n        "speed": np.zeros(n, dtype=np.float64),\n        "fused": np.zeros(n, dtype=np.float64),\n        "raw": np.zeros(n, dtype=np.float64),\n        "steering": np.zeros(n, dtype=np.float64),\n        "stop": np.zeros(\n            (n, len(S3_V5C_STOP_THRESHOLDS)),\n            dtype=np.float64,\n        ),\n        "steer": np.zeros((n, 3), dtype=np.float64),\n        "turn": np.zeros(\n            (n, len(S3_V5C_TURN_THRESHOLDS), 2),\n            dtype=np.float64,\n        ),\n    }\n\n    clips = []\n    starts = []\n    valid_lengths = []\n\n    for start in starts_all:\n        end = min(start + S3_V5C_NUM_FRAMES, n)\n        clip = frames[start:end]\n        valid = len(clip)\n\n        if valid < S3_V5C_NUM_FRAMES:\n            pad = np.repeat(\n                clip[-1:],\n                S3_V5C_NUM_FRAMES - valid,\n                axis=0,\n            )\n            clip = np.concatenate([clip, pad], axis=0)\n\n        clips.append(clip)\n        starts.append(start)\n        valid_lengths.append(valid)\n\n        if len(clips) == S3_V5C_BATCH:\n            _stage3_v5c_flush_batch(\n                model,\n                clips,\n                starts,\n                valid_lengths,\n                device,\n                stats,\n                center_weights,\n                accum,\n            )\n            clips = []\n            starts = []\n            valid_lengths = []\n\n    if clips:\n        _stage3_v5c_flush_batch(\n            model,\n            clips,\n            starts,\n            valid_lengths,\n            device,\n            stats,\n            center_weights,\n            accum,\n        )\n\n    weight = accum["weight"]\n    if np.any(weight <= 0):\n        missing = np.flatnonzero(weight <= 0)[:10].tolist()\n        raise RuntimeError(\n            f"Stage 3 frames received no window prediction: {missing}"\n        )\n\n    return {\n        "speed": accum["speed"] / weight,\n        "fused": accum["fused"] / weight,\n        "raw": accum["raw"] / weight,\n        "steering": accum["steering"] / weight,\n        "stop": accum["stop"] / weight[:, None],\n        "steer": accum["steer"] / weight[:, None],\n        "turn": accum["turn"] / weight[:, None, None],\n    }\n\n\ndef _stage3_v5c_apply(\n    features: dict,\n    calibration: dict,\n    selection: dict,\n):\n    accel_cfg = dict(calibration["accel"])\n    steer_cfg = dict(calibration["steer"])\n    fusion = dict(selection["fusion"])\n\n    overlap = dict(selection["overlap"])\n    if int(overlap["stride"]) != S3_V5C_STRIDE:\n        raise ValueError("v5-C selection stride mismatch")\n    if abs(\n        float(overlap["center_floor"]) - S3_V5C_CENTER_FLOOR\n    ) > 1e-12:\n        raise ValueError("v5-C selection center_floor mismatch")\n\n    accel_window = int(accel_cfg["smoothing_window"])\n    steer_window = int(steer_cfg["smoothing_window"])\n\n    speed = _stage3_v5c_centered_mean(\n        features["speed"], accel_window\n    )\n\n    source = str(accel_cfg["source"])\n    if source == "fused":\n        accel = features["fused"]\n    elif source == "raw":\n        accel = features["raw"]\n    else:\n        raise ValueError(f"unknown Stage 3 accel source: {source}")\n\n    accel = _stage3_v5c_centered_mean(\n        accel, accel_window\n    )\n    accel = accel + float(accel_cfg["accel_bias_mps2"])\n\n    stop_prob = np.column_stack(\n        [\n            _stage3_v5c_centered_mean(\n                features["stop"][:, k],\n                accel_window,\n            )\n            for k in range(features["stop"].shape[1])\n        ]\n    )\n\n    steering = _stage3_v5c_centered_mean(\n        features["steering"], steer_window\n    )\n    steering = (\n        int(steer_cfg["steering_sign"]) * steering\n        + float(steer_cfg["steering_bias_deg"])\n    )\n\n    steer_prob = np.column_stack(\n        [\n            _stage3_v5c_centered_mean(\n                features["steer"][:, k],\n                steer_window,\n            )\n            for k in range(3)\n        ]\n    )\n\n    turn_prob = np.empty_like(features["turn"], dtype=np.float64)\n    for k in range(features["turn"].shape[1]):\n        for d in range(2):\n            turn_prob[:, k, d] = _stage3_v5c_centered_mean(\n                features["turn"][:, k, d],\n                steer_window,\n            )\n\n    stop_temperature = max(\n        float(fusion["stop_temperature_mps"]),\n        1e-4,\n    )\n    stop_z = (\n        float(accel_cfg["stop_speed_mps"]) - speed\n    ) / stop_temperature\n\n    stop_weight = float(fusion["stop_weight"])\n    if abs(stop_weight) > 1e-12:\n        p_stop = _stage3_v5c_interp_probability(\n            S3_V5C_STOP_THRESHOLDS,\n            stop_prob,\n            float(accel_cfg["stop_speed_mps"]),\n        )\n        stop_z = (\n            stop_z\n            + stop_weight * _stage3_v5c_logit(p_stop)\n        )\n\n    stopped = stop_z > 0.0\n    moving = ~stopped\n\n    if abs(float(fusion["accel_weight"])) > 1e-12:\n        raise ValueError(\n            "This compact v5-C submission was validated with accel_weight=0"\n        )\n\n    accel_labels = np.full(\n        len(speed),\n        "CONSTANT",\n        dtype=object,\n    )\n    accel_labels[stopped] = "STOPPED"\n    accel_labels[\n        moving\n        & (\n            accel\n            > float(accel_cfg["accel_deadzone_pos_mps2"])\n        )\n    ] = "ACCELERATING"\n    accel_labels[\n        moving\n        & (\n            accel\n            < -float(accel_cfg["accel_deadzone_neg_mps2"])\n        )\n    ] = "DECELERATING"\n\n    steer_temperature = max(\n        float(fusion["steer_temperature_deg"]),\n        1e-4,\n    )\n    z_left = (\n        -steering - float(steer_cfg["left_deadzone_deg"])\n    ) / steer_temperature\n    z_right = (\n        steering - float(steer_cfg["right_deadzone_deg"])\n    ) / steer_temperature\n\n    p_left = steer_prob[:, 0].copy()\n    p_straight = steer_prob[:, 1].copy()\n    p_right = steer_prob[:, 2].copy()\n\n    if int(steer_cfg["steering_sign"]) < 0:\n        p_left, p_right = p_right, p_left\n\n    steer_weight = float(fusion["steer_weight"])\n    if abs(steer_weight) > 1e-12:\n        z_left = z_left + steer_weight * (\n            np.log(_stage3_v5c_clip_prob(p_left))\n            - np.log(_stage3_v5c_clip_prob(p_straight))\n        )\n        z_right = z_right + steer_weight * (\n            np.log(_stage3_v5c_clip_prob(p_right))\n            - np.log(_stage3_v5c_clip_prob(p_straight))\n        )\n\n    turn_weight = float(fusion["turn_weight"])\n    if abs(turn_weight) > 1e-12:\n        p_turn_left = turn_prob[:, :, 0].mean(axis=1)\n        p_turn_right = turn_prob[:, :, 1].mean(axis=1)\n\n        if int(steer_cfg["steering_sign"]) < 0:\n            p_turn_left, p_turn_right = (\n                p_turn_right,\n                p_turn_left,\n            )\n\n        z_left = (\n            z_left\n            + turn_weight * _stage3_v5c_logit(p_turn_left)\n        )\n        z_right = (\n            z_right\n            + turn_weight * _stage3_v5c_logit(p_turn_right)\n        )\n\n    steer_labels = np.full(\n        len(steering),\n        "STRAIGHT",\n        dtype=object,\n    )\n    choose_left = (z_left > 0.0) & (z_left >= z_right)\n    choose_right = (z_right > 0.0) & (z_right > z_left)\n    steer_labels[choose_left] = "LEFT"\n    steer_labels[choose_right] = "RIGHT"\n\n    return (\n        accel_labels.astype(str),\n        steer_labels.astype(str),\n    )\n\n\ndef predict_stage3(data_dir, model_dir):\n    import json\n\n    device = _device()\n    model_dir = Path(model_dir)\n\n    script_path = model_dir / "model.ts"\n    stats_path = model_dir / "target_stats.json"\n    calibration_path = model_dir / "calibration.json"\n    selection_path = model_dir / "v5c_selection.json"\n\n    for required in (\n        script_path,\n        stats_path,\n        calibration_path,\n        selection_path,\n    ):\n        if not required.is_file():\n            raise FileNotFoundError(\n                f"Stage 3 asset not found: {required}"\n            )\n\n    stats = json.loads(\n        stats_path.read_text(encoding="utf-8")\n    )\n    calibration = json.loads(\n        calibration_path.read_text(encoding="utf-8")\n    )\n    selection = json.loads(\n        selection_path.read_text(encoding="utf-8")\n    )\n\n    model = torch.jit.load(\n        str(script_path),\n        map_location=device,\n    )\n    model.eval()\n\n    videos = _video_paths(Path(data_dir) / "videos")\n    rows = []\n\n    with torch.inference_mode():\n        for path in videos:\n            features = _stage3_v5c_predict_features(\n                path,\n                model,\n                device,\n                stats,\n            )\n            accel_labels, steer_labels = _stage3_v5c_apply(\n                features,\n                calibration,\n                selection,\n            )\n\n            if len(accel_labels) != len(steer_labels):\n                raise RuntimeError(\n                    f"Stage 3 output length mismatch for {path.name}"\n                )\n\n            for sample_index, (accel_label, steer_label) in enumerate(\n                zip(accel_labels, steer_labels)\n            ):\n                rows.append(\n                    {\n                        "ID": path.stem,\n                        "sample_index": int(sample_index),\n                        "accel_label": accel_label,\n                        "steer_label": steer_label,\n                    }\n                )\n\n    del model\n    torch.cuda.empty_cache()\n\n    return pd.DataFrame(\n        rows,\n        columns=[\n            "ID",\n            "sample_index",\n            "accel_label",\n            "steer_label",\n        ],\n    )\n'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_submission_root(root: Path) -> Path:
    if (
        (root / "inference.py").is_file()
        and (root / "model").is_dir()
    ):
        return root

    dirs = [p for p in root.iterdir() if p.is_dir()]
    files = [p for p in root.iterdir() if p.is_file()]

    if len(dirs) == 1 and not files:
        candidate = dirs[0]
        if (
            (candidate / "inference.py").is_file()
            and (candidate / "model").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not resolve stage1_a7.zip root. "
        "Expected inference.py + model/."
    )


def write_submit_zip(source_root: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(
        output_zip,
        "w",
        allowZip64=True,
    ) as zf:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue

            arcname = path.relative_to(source_root).as_posix()
            compress_type = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in {".pt", ".pth", ".ts"}
                else zipfile.ZIP_DEFLATED
            )
            zf.write(
                path,
                arcname,
                compress_type=compress_type,
            )


try:
    from google.colab import drive
except ImportError as exc:
    raise RuntimeError(
        "This build script is intended for Google Colab."
    ) from exc

drive.mount("/content/drive", force_remount=False)

DRIVE_ROOT = Path("/content/drive/MyDrive/Blackbox-Detection")
THIS_FILE = Path(__file__).resolve()
REPO = THIS_FILE.parents[2]
BRANCH = "stage3-sangchun"

if not (REPO / ".git").is_dir():
    raise RuntimeError(f"Repository not found at {REPO}")

subprocess.run(
    ["git", "-C", str(REPO), "checkout", BRANCH],
    check=True,
)

dirty = subprocess.run(
    ["git", "-C", str(REPO), "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

if dirty:
    print("WARNING: repository has local changes; git pull skipped")
else:
    subprocess.run(
        ["git", "-C", str(REPO), "pull", "--ff-only", "origin", BRANCH],
        check=True,
    )

BUILD_EXTRAS = [
    "timm==1.0.15",
    "fvcore==0.1.5.post20221221",
    "iopath==0.1.10",
    "yacs==0.1.8",
    "einops==0.8.1",
    "easydict==1.13",
]
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade-strategy", "only-if-needed",
        *BUILD_EXTRAS,
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "--no-deps", "-e", str(REPO),
    ],
    check=True,
)

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch
import yaml
from torch import nn

from blackbox_detection.stage3.v5_models import VJEPA21DenseCANV5


BASE_A7_ZIP_OVERRIDE = None
a7_candidates = [
    DRIVE_ROOT / "submissions" / "stage1_a7.zip",
    DRIVE_ROOT / "stage1_a7.zip",
    Path("/content/stage1_a7.zip"),
]
if BASE_A7_ZIP_OVERRIDE is not None:
    a7_candidates.insert(0, Path(BASE_A7_ZIP_OVERRIDE))

BASE_A7_ZIP = next((p for p in a7_candidates if p.is_file()), None)
if BASE_A7_ZIP is None:
    raise FileNotFoundError("stage1_a7.zip not found")

SOURCE_RUN = "vjepa21b_can_v5b_last2_ft"
V5B_BEST = (
    DRIVE_ROOT / "outputs" / "stage3" / SOURCE_RUN / "best.pt"
)
V5B_CFG = REPO / "configs/stage3/vjepa21b_can_v5b.yaml"
TARGET_STATS = (
    DRIVE_ROOT / "manifests/stage3/v1/target_stats.json"
)
V5C_SELECTION = (
    DRIVE_ROOT
    / "outputs/stage3/vjepa21b_can_v5c_overlap_auxfusion/v5c_selection.json"
)
CALIBRATION = REPO / "calibrations/stage3_calibration_consensus.json"

for p in (
    V5B_BEST,
    V5B_CFG,
    TARGET_STATS,
    V5C_SELECTION,
    CALIBRATION,
):
    if not p.is_file():
        raise FileNotFoundError(p)

selection = json.loads(V5C_SELECTION.read_text(encoding="utf-8"))

EXPECTED_SELECTION = {
    "source_run": SOURCE_RUN,
    "source_checkpoint": "best.pt",
    "clip_len": 32,
    "overlap": {
        "stride": 8,
        "center_floor": 0.25,
    },
    "fusion": {
        "stop_weight": 0.75,
        "accel_weight": 0.0,
        "steer_weight": 0.3,
        "turn_weight": 0.3,
        "stop_temperature_mps": 0.25,
        "accel_temperature_mps2": 0.1,
        "steer_temperature_deg": 2.0,
    },
}

for key in ("source_run", "source_checkpoint", "clip_len"):
    if selection.get(key) != EXPECTED_SELECTION[key]:
        raise RuntimeError(
            f"v5-C selection mismatch for {key}: {selection.get(key)}"
        )
for key, value in EXPECTED_SELECTION["overlap"].items():
    if abs(float(selection["overlap"][key]) - float(value)) > 1e-12:
        raise RuntimeError(f"v5-C overlap selection mismatch: {key}")
for key, value in EXPECTED_SELECTION["fusion"].items():
    if abs(float(selection["fusion"][key]) - float(value)) > 1e-12:
        raise RuntimeError(f"v5-C fusion selection mismatch: {key}")
if not bool(selection.get("fusion_accepted", False)):
    raise RuntimeError("v5-C fusion was not accepted by validation")

OUT_DIR = DRIVE_ROOT / "submissions/stage1a7_stage3_v5c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Base A7 ZIP :", BASE_A7_ZIP)
print("v5-B best   :", V5B_BEST)
print("v5-C select :", V5C_SELECTION)
print("calibration :", CALIBRATION)
print("output dir  :", OUT_DIR)
print("v5-C selection lock: PASS")


EXTRACT_DIR = Path("/content/a7_base_extract_v5c")
BUILD_ROOT = Path("/content/stage1a7_stage3_v5c_build")
for p in (EXTRACT_DIR, BUILD_ROOT):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)

with zipfile.ZipFile(BASE_A7_ZIP) as zf:
    zf.extractall(EXTRACT_DIR)

A7_ROOT = resolve_submission_root(EXTRACT_DIR)
shutil.copytree(A7_ROOT, BUILD_ROOT, dirs_exist_ok=True)

BASE_INFERENCE = (
    A7_ROOT / "inference.py"
).read_text(encoding="utf-8")
STAGE3_SPLIT = (
    "# ---------------------------------------------------------------------------\n"
    "# Stage 3:"
)
if STAGE3_SPLIT not in BASE_INFERENCE:
    raise RuntimeError(
        "Could not locate Stage 3 boundary in stage1_a7 inference.py"
    )

BASE_PREFIX = BASE_INFERENCE.split(STAGE3_SPLIT, 1)[0]
NEW_INFERENCE = BASE_PREFIX + STAGE3_V5C_BLOCK

STAGE1_ROOT = A7_ROOT / "model/stage1"
STAGE2_ROOT = A7_ROOT / "model/stage2"

stage1_tree_before = {
    p.relative_to(STAGE1_ROOT).as_posix(): sha256(p)
    for p in sorted(STAGE1_ROOT.rglob("*"))
    if p.is_file()
}
stage2_tree_before = {
    p.relative_to(STAGE2_ROOT).as_posix(): sha256(p)
    for p in sorted(STAGE2_ROOT.rglob("*"))
    if p.is_file()
}
a7_requirements_sha = sha256(A7_ROOT / "requirements.txt")
print("Stage 1/2 source lock: PASS")


cfg = yaml.safe_load(V5B_CFG.read_text(encoding="utf-8"))
dc = cfg["data"]
mc = cfg["model"]
spatial_cfg = dict(mc["spatial_pool"])
fusion_cfg = dict(mc["accel_fusion"])

if int(dc["clip_len"]) != 32:
    raise RuntimeError(f"unexpected v5-B clip_len: {dc['clip_len']}")

VJEPA_REPO = Path("/content/vjepa2")
VJEPA_COMMIT = "45d025f636dfc58fc2426905fc4a1ab755b1c3e5"

if not (VJEPA_REPO / ".git").is_dir():
    subprocess.run(
        [
            "git", "clone", "-q",
            "https://github.com/facebookresearch/vjepa2.git",
            str(VJEPA_REPO),
        ],
        check=True,
    )

subprocess.run(
    ["git", "-C", str(VJEPA_REPO), "fetch", "--all", "--tags"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(VJEPA_REPO), "checkout", "-q", VJEPA_COMMIT],
    check=True,
)

actual_commit = subprocess.run(
    ["git", "-C", str(VJEPA_REPO), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if actual_commit != VJEPA_COMMIT:
    raise RuntimeError(f"V-JEPA commit mismatch: {actual_commit}")

backbone, predictor = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
    pretrained=False,
    num_frames=int(dc["clip_len"]),
    out_layers=list(mc["out_layers"]),
)
del predictor
backbone.requires_grad_(False)
backbone.eval()

model = VJEPA21DenseCANV5(
    backbone,
    freeze_backbone=True,
    feature_dim=int(mc["feature_dim"]),
    temporal_hidden=int(mc["temporal_hidden"]),
    temporal_layers=int(mc["temporal_layers"]),
    spatial_grid=tuple(spatial_cfg["grid"]),
    spatial_gate_init=float(spatial_cfg["gate_init"]),
    accel_ordinal_thresholds_mps2=mc["accel_ordinal_thresholds_mps2"],
    accel_fusion_enabled=bool(fusion_cfg["enabled"]),
    accel_fusion_hidden=int(fusion_cfg["hidden"]),
    accel_fusion_gate_init=float(fusion_cfg["gate_init"]),
    accel_fusion_detach_ordinal_inputs=bool(
        fusion_cfg["detach_ordinal_inputs"]
    ),
    stop_thresholds_mps=mc["stop_thresholds_mps"],
    turn_yaw_thresholds_rps=mc["turn_yaw_thresholds_rps"],
    steer_activity_thresholds=mc["steer_activity_thresholds"],
    brake_thresholds_bar=mc["brake_thresholds_bar"],
    throttle_thresholds_pct=mc["throttle_thresholds_pct"],
)

checkpoint = torch.load(
    V5B_BEST,
    map_location="cpu",
    weights_only=False,
)
state = checkpoint.get("model")
if not isinstance(state, dict):
    raise RuntimeError("v5-B checkpoint has no model state_dict")
if not any(k.startswith("backbone.") for k in state):
    raise RuntimeError("v5-B checkpoint has no backbone state")

incompatible = model.load_state_dict(state, strict=True)
if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(str(incompatible))

checkpoint_epoch = checkpoint.get("epoch")
del checkpoint, state

print("v5-B checkpoint strict load: PASS")
print("checkpoint epoch:", checkpoint_epoch)


class Stage3V5CExportWrapper(nn.Module):
    def __init__(self, source_model):
        super().__init__()
        self.model = source_model

    def forward(self, video):
        out = self.model(video)
        return (
            out["speed_mps"],
            out["accel_from_speed_mps2"],
            out["accel_raw_from_speed_mps2"],
            out["steering_deg"],
            torch.sigmoid(out["stop_ordinal_logits"]),
            torch.softmax(out["steer_direction_logits"], dim=-1),
            torch.sigmoid(out["turn_ordinal_logits"]),
        )


device = torch.device("cuda")
wrapper = Stage3V5CExportWrapper(model).to(device).eval()

example = torch.randn(
    2,
    3,
    32,
    int(dc["input_height"]),
    int(dc["input_width"]),
    device=device,
    dtype=torch.float32,
)

with torch.inference_mode():
    eager = tuple(x.detach().cpu() for x in wrapper(example))
    traced = torch.jit.trace(
        wrapper,
        example,
        strict=False,
        check_trace=False,
    )
    traced = torch.jit.freeze(traced.eval())
    traced_out = tuple(x.detach().cpu() for x in traced(example))

for a, b in zip(eager, traced_out, strict=True):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

expected_shapes = [
    (2, 32),
    (2, 32),
    (2, 32),
    (2, 32),
    (2, 32, len(mc["stop_thresholds_mps"])),
    (2, 32, 3),
    (2, 32, len(mc["turn_yaw_thresholds_rps"]), 2),
]
for i, (tensor, shape) in enumerate(zip(eager, expected_shapes, strict=True)):
    if tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(
            f"Unexpected v5-C output[{i}] shape: "
            f"{tuple(tensor.shape)} vs {shape}"
        )

LOCAL_STAGE3_MODEL = Path("/content/stage3_v5c_model.ts")
traced.save(str(LOCAL_STAGE3_MODEL))

reloaded = torch.jit.load(
    str(LOCAL_STAGE3_MODEL),
    map_location=device,
).eval()
with torch.inference_mode():
    reload_out = tuple(x.detach().cpu() for x in reloaded(example))
for a, b in zip(eager, reload_out, strict=True):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

print("Stage 3 v5-C TorchScript export: PASS")
print("model.ts MiB:", f"{LOCAL_STAGE3_MODEL.stat().st_size / 2**20:.1f}")

with torch.inference_mode():
    for _ in range(2):
        _ = reloaded(example)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    bench_repeats = 5
    for _ in range(bench_repeats):
        _ = reloaded(example)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

seconds_per_batch2 = elapsed / bench_repeats
starts_600 = list(range(0, 600 - 32 + 1, 8))
if starts_600[-1] != 600 - 32:
    starts_600.append(600 - 32)
batches_600 = math.ceil(len(starts_600) / 2)
print(
    "Build-GPU Stage3 model-only estimate / 600 frames:",
    f"{seconds_per_batch2 * batches_600:.1f}s",
    "(decode/postprocess excluded)",
)

del (
    reloaded,
    traced,
    wrapper,
    model,
    backbone,
    example,
    eager,
    traced_out,
    reload_out,
)
torch.cuda.empty_cache()


STAGE3_DIR = BUILD_ROOT / "model/stage3"
if STAGE3_DIR.exists():
    shutil.rmtree(STAGE3_DIR)
STAGE3_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy2(LOCAL_STAGE3_MODEL, STAGE3_DIR / "model.ts")
shutil.copy2(TARGET_STATS, STAGE3_DIR / "target_stats.json")
shutil.copy2(CALIBRATION, STAGE3_DIR / "calibration.json")
shutil.copy2(V5C_SELECTION, STAGE3_DIR / "v5c_selection.json")

(BUILD_ROOT / "inference.py").write_text(
    NEW_INFERENCE,
    encoding="utf-8",
)
shutil.copy2(
    A7_ROOT / "requirements.txt",
    BUILD_ROOT / "requirements.txt",
)


stage1_tree_after = {
    p.relative_to(BUILD_ROOT / "model/stage1").as_posix(): sha256(p)
    for p in sorted((BUILD_ROOT / "model/stage1").rglob("*"))
    if p.is_file()
}
stage2_tree_after = {
    p.relative_to(BUILD_ROOT / "model/stage2").as_posix(): sha256(p)
    for p in sorted((BUILD_ROOT / "model/stage2").rglob("*"))
    if p.is_file()
}

if stage1_tree_after != stage1_tree_before:
    raise RuntimeError("ABORT: Stage 1 files changed")
if stage2_tree_after != stage2_tree_before:
    raise RuntimeError("ABORT: Stage 2 files changed")
if sha256(BUILD_ROOT / "requirements.txt") != a7_requirements_sha:
    raise RuntimeError("ABORT: requirements.txt changed from A7")

final_inference = (
    BUILD_ROOT / "inference.py"
).read_text(encoding="utf-8")
if final_inference.split(STAGE3_SPLIT, 1)[0] != BASE_PREFIX:
    raise RuntimeError("ABORT: Stage 1/2 inference prefix changed")
ast.parse(final_inference)

print("Stage 1 files unchanged: PASS")
print("Stage 2 files unchanged: PASS")
print("Stage 1/2 inference prefix unchanged: PASS")
print("A7 requirements unchanged: PASS")


OUTPUT_ZIP = OUT_DIR / "submit_stage1a7_stage3_v5c.zip"
write_submit_zip(BUILD_ROOT, OUTPUT_ZIP)

required = {
    "inference.py",
    "requirements.txt",
    "model/stage1/model.ts",
    "model/stage2/best.pt",
    "model/stage2/resnet18-f37072fd.pth",
    "model/stage3/model.ts",
    "model/stage3/target_stats.json",
    "model/stage3/calibration.json",
    "model/stage3/v5c_selection.json",
}

with zipfile.ZipFile(OUTPUT_ZIP) as zf:
    names = set(zf.namelist())

    roots = {
        name.split("/", 1)[0]
        for name in names
        if name
    }
    invalid_roots = roots - {
        "model",
        "inference.py",
        "requirements.txt",
    }
    if invalid_roots:
        raise RuntimeError(
            f"invalid top-level entries: {sorted(invalid_roots)}"
        )

    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"missing files: {missing}")

    if hashlib.sha256(
        zf.read("model/stage1/model.ts")
    ).hexdigest() != stage1_tree_before["model.ts"]:
        raise RuntimeError("Stage 1 model.ts changed inside ZIP")

    inference_text = zf.read("inference.py").decode("utf-8")
    if inference_text.split(STAGE3_SPLIT, 1)[0] != BASE_PREFIX:
        raise RuntimeError("Stage 1/2 inference changed inside ZIP")
    ast.parse(inference_text)

    selection_in_zip = json.loads(
        zf.read("model/stage3/v5c_selection.json").decode("utf-8")
    )
    if selection_in_zip["overlap"] != selection["overlap"]:
        raise RuntimeError("v5-C selection changed inside ZIP")

zip_gib = OUTPUT_ZIP.stat().st_size / 2**30
if zip_gib > 10.0:
    raise RuntimeError(
        f"submit.zip exceeds DACON 10GB limit: {zip_gib:.3f} GiB"
    )

print("ZIP contract: PASS")
print("ZIP size GiB:", f"{zip_gib:.3f}")


s1 = torch.jit.load(
    str(BUILD_ROOT / "model/stage1/model.ts"),
    map_location=device,
).eval()
print("Stage 1 TorchScript load: PASS")
del s1
torch.cuda.empty_cache()

s3 = torch.jit.load(
    str(BUILD_ROOT / "model/stage3/model.ts"),
    map_location=device,
).eval()

dummy = torch.zeros(
    2, 3, 32, 288, 384,
    dtype=torch.float32,
    device=device,
)
with torch.inference_mode():
    output = s3(dummy)

if not isinstance(output, (tuple, list)) or len(output) != 7:
    raise RuntimeError("Unexpected Stage 3 v5-C TorchScript contract")
for tensor, shape in zip(output, expected_shapes, strict=True):
    if tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(
            f"Unexpected Stage 3 shape: {tuple(tensor.shape)} vs {shape}"
        )

print("Stage 3 v5-C TorchScript load/forward: PASS")

del s3, dummy, output
torch.cuda.empty_cache()


report = {
    "base_a7_zip": str(BASE_A7_ZIP),
    "stage1_files_sha256": stage1_tree_before,
    "stage2_files_sha256": stage2_tree_before,
    "a7_requirements_sha256": a7_requirements_sha,
    "stage1_inference_prefix_sha256": hashlib.sha256(
        BASE_PREFIX.encode("utf-8")
    ).hexdigest(),
    "vjepa_commit": VJEPA_COMMIT,
    "source_run": SOURCE_RUN,
    "source_checkpoint": str(V5B_BEST),
    "source_checkpoint_epoch": checkpoint_epoch,
    "v5c_selection": selection,
    "calibration": str(CALIBRATION),
    "stage3_model_ts_bytes": LOCAL_STAGE3_MODEL.stat().st_size,
    "submission_zip": str(OUTPUT_ZIP),
    "submission_zip_bytes": OUTPUT_ZIP.stat().st_size,
    "submission_zip_sha256": sha256(OUTPUT_ZIP),
}

REPORT_PATH = OUT_DIR / "build_report.json"
REPORT_PATH.write_text(
    json.dumps(report, indent=2, default=str),
    encoding="utf-8",
)

print("\nBUILD COMPLETE")
print("Submit       :", OUTPUT_ZIP)
print("SHA256       :", report["submission_zip_sha256"])
print("Build report :", REPORT_PATH)

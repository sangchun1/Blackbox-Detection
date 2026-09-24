from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Stage 3 inference implementation is embedded here so the builder does not
# depend on any file under the gitignored submissions/ directory.
STAGE3_V4_BLOCK = '# ---------------------------------------------------------------------------\n# Stage 3: V-JEPA 2.1-B v4-A continuous-CAN + calibrated categorical mapping\n# ---------------------------------------------------------------------------\n# Stage 1 / Stage 2 above are intentionally kept byte-identical to stage1_a7.\n#\n# Stage 3 assets:\n#   model/stage3/model.ts\n#   model/stage3/target_stats.json\n#   model/stage3/calibration.json\n#\n# The Stage 3 network is exported to TorchScript at build time. Therefore the\n# DACON evaluator does not need the facebookresearch/vjepa2 repository, timm,\n# einops, or any network download to construct this model.\n\nS3_V4_NUM_FRAMES = 16\nS3_V4_HEIGHT = 288\nS3_V4_WIDTH = 384\nS3_V4_BATCH = 2\n\nS3_V4_MEAN = torch.tensor(\n    [0.485, 0.456, 0.406],\n    dtype=torch.float32,\n)[:, None, None, None]\nS3_V4_STD = torch.tensor(\n    [0.229, 0.224, 0.225],\n    dtype=torch.float32,\n)[:, None, None, None]\n\n\ndef _stage3_v4_resize_rgb(bgr: np.ndarray) -> np.ndarray:\n    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n    return cv2.resize(\n        rgb,\n        (S3_V4_WIDTH, S3_V4_HEIGHT),\n        interpolation=cv2.INTER_AREA,\n    )\n\n\ndef _stage3_v4_denormalize(\n    values: np.ndarray,\n    stats: dict,\n    name: str,\n) -> np.ndarray:\n    stat = stats[name]\n    return (\n        values.astype(np.float32, copy=False) * float(stat["std"])\n        + float(stat["mean"])\n    )\n\n\ndef _stage3_v4_centered_mean(\n    values: np.ndarray,\n    window: int,\n) -> np.ndarray:\n    window = int(window)\n    if window <= 1:\n        return np.asarray(values, dtype=np.float32)\n    if window % 2 == 0:\n        raise ValueError(\n            f"Stage 3 smoothing window must be odd, got {window}"\n        )\n\n    return (\n        pd.Series(np.asarray(values, dtype=np.float32))\n        .rolling(\n            window=window,\n            center=True,\n            min_periods=1,\n        )\n        .mean()\n        .to_numpy(dtype=np.float32)\n    )\n\n\ndef _stage3_v4_flush_batch(\n    model,\n    clips,\n    valid_lengths,\n    device,\n    stats,\n    speed_parts,\n    fused_accel_parts,\n    raw_accel_parts,\n    steering_parts,\n):\n    """Run one fixed-size Stage 3 TorchScript batch.\n\n    model.ts is exported with batch size 2. The final incomplete batch is\n    padded with its last clip; padded predictions are discarded.\n    """\n    real_batch = len(clips)\n    if real_batch == 0:\n        return\n\n    while len(clips) < S3_V4_BATCH:\n        clips.append(clips[-1].copy())\n        valid_lengths.append(0)\n\n    batch_np = np.stack(clips, axis=0)  # B T H W C\n    x = (\n        torch.from_numpy(batch_np)\n        .permute(0, 4, 1, 2, 3)\n        .float()\n        .div_(255.0)\n    )\n    x = (x - S3_V4_MEAN[None]) / S3_V4_STD[None]\n    x = x.to(\n        device=device,\n        dtype=torch.float32,\n        non_blocking=True,\n    )\n\n    # Same safety policy as Stage 1 A7: scripted video attention runs in FP32.\n    # Do not wrap this TorchScript call in autocast.\n    outputs = model(x)\n    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:\n        raise RuntimeError(\n            "Stage 3 model.ts must return "\n            "(speed, fused_accel, raw_accel, steering)"\n        )\n\n    speed_n, fused_n, raw_n, steering_n = [\n        tensor.float().cpu().numpy()\n        for tensor in outputs\n    ]\n\n    speed = _stage3_v4_denormalize(\n        speed_n,\n        stats,\n        "speed_mps",\n    )\n    fused = _stage3_v4_denormalize(\n        fused_n,\n        stats,\n        "accel_from_speed_mps2",\n    )\n    raw = _stage3_v4_denormalize(\n        raw_n,\n        stats,\n        "accel_from_speed_mps2",\n    )\n    steering = _stage3_v4_denormalize(\n        steering_n,\n        stats,\n        "steering_deg",\n    )\n\n    for i in range(real_batch):\n        n = int(valid_lengths[i])\n        if n <= 0:\n            continue\n        speed_parts.append(speed[i, :n].copy())\n        fused_accel_parts.append(fused[i, :n].copy())\n        raw_accel_parts.append(raw[i, :n].copy())\n        steering_parts.append(steering[i, :n].copy())\n\n\ndef _stage3_v4_predict_continuous(\n    path: Path,\n    model,\n    device,\n    stats: dict,\n):\n    """Sequential 10-Hz inference with bounded CPU memory."""\n    capture = cv2.VideoCapture(str(path))\n    if not capture.isOpened():\n        capture.release()\n        raise ValueError(\n            f"cannot open Stage 3 video: {path.name}"\n        )\n\n    speed_parts = []\n    fused_accel_parts = []\n    raw_accel_parts = []\n    steering_parts = []\n\n    clips = []\n    valid_lengths = []\n    current = []\n\n    try:\n        while True:\n            ok, bgr = capture.read()\n            if not ok:\n                break\n\n            current.append(_stage3_v4_resize_rgb(bgr))\n\n            if len(current) == S3_V4_NUM_FRAMES:\n                clips.append(np.stack(current, axis=0))\n                valid_lengths.append(S3_V4_NUM_FRAMES)\n                current = []\n\n                if len(clips) == S3_V4_BATCH:\n                    _stage3_v4_flush_batch(\n                        model,\n                        clips,\n                        valid_lengths,\n                        device,\n                        stats,\n                        speed_parts,\n                        fused_accel_parts,\n                        raw_accel_parts,\n                        steering_parts,\n                    )\n                    clips = []\n                    valid_lengths = []\n\n        if current:\n            valid = len(current)\n            while len(current) < S3_V4_NUM_FRAMES:\n                current.append(current[-1].copy())\n\n            clips.append(np.stack(current, axis=0))\n            valid_lengths.append(valid)\n\n        if clips:\n            _stage3_v4_flush_batch(\n                model,\n                clips,\n                valid_lengths,\n                device,\n                stats,\n                speed_parts,\n                fused_accel_parts,\n                raw_accel_parts,\n                steering_parts,\n            )\n    finally:\n        capture.release()\n\n    if not speed_parts:\n        raise ValueError(\n            f"cannot decode Stage 3 video: {path.name}"\n        )\n\n    return (\n        np.concatenate(speed_parts),\n        np.concatenate(fused_accel_parts),\n        np.concatenate(raw_accel_parts),\n        np.concatenate(steering_parts),\n    )\n\n\ndef _stage3_v4_apply_calibration(\n    speed_mps: np.ndarray,\n    fused_accel_mps2: np.ndarray,\n    raw_accel_mps2: np.ndarray,\n    steering_deg: np.ndarray,\n    calibration: dict,\n):\n    accel_cfg = dict(calibration["accel"])\n    steer_cfg = dict(calibration["steer"])\n\n    # Both selected calibration candidates use ordinal_blend=0. The ordinal\n    # branch was useful during training, but direct ordinal blending was not\n    # selected by any LOVO fold.\n    ordinal_blend = float(\n        accel_cfg.get("ordinal_blend", 0.0)\n    )\n    if abs(ordinal_blend) > 1e-12:\n        raise ValueError(\n            "Compact Stage 3 TorchScript supports "\n            "ordinal_blend=0 only; got "\n            f"{ordinal_blend}"\n        )\n\n    accel_window = int(\n        accel_cfg["smoothing_window"]\n    )\n\n    # Match calibration.py: speed and acceleration use the same centered\n    # smoothing window before the STOP gate / dynamic thresholds.\n    speed = _stage3_v4_centered_mean(\n        speed_mps,\n        accel_window,\n    )\n\n    source = str(accel_cfg["source"])\n    if source == "fused":\n        accel = fused_accel_mps2\n    elif source == "raw":\n        accel = raw_accel_mps2\n    else:\n        raise ValueError(\n            f"unknown Stage 3 accel source: {source}"\n        )\n\n    accel = _stage3_v4_centered_mean(\n        accel,\n        accel_window,\n    )\n    accel = (\n        accel\n        + float(accel_cfg["accel_bias_mps2"])\n    )\n\n    accel_labels = np.full(\n        len(speed),\n        "CONSTANT",\n        dtype=object,\n    )\n\n    stopped = (\n        speed\n        <= float(accel_cfg["stop_speed_mps"])\n    )\n    moving = ~stopped\n\n    accel_labels[stopped] = "STOPPED"\n\n    accel_labels[\n        moving\n        & (\n            accel\n            > float(\n                accel_cfg[\n                    "accel_deadzone_pos_mps2"\n                ]\n            )\n        )\n    ] = "ACCELERATING"\n\n    accel_labels[\n        moving\n        & (\n            accel\n            < -float(\n                accel_cfg[\n                    "accel_deadzone_neg_mps2"\n                ]\n            )\n        )\n    ] = "DECELERATING"\n\n    steering = _stage3_v4_centered_mean(\n        steering_deg,\n        int(steer_cfg["smoothing_window"]),\n    )\n    steering = (\n        int(steer_cfg["steering_sign"])\n        * steering\n        + float(steer_cfg["steering_bias_deg"])\n    )\n\n    steer_labels = np.full(\n        len(steering),\n        "STRAIGHT",\n        dtype=object,\n    )\n\n    steer_labels[\n        steering\n        < -float(\n            steer_cfg["left_deadzone_deg"]\n        )\n    ] = "LEFT"\n\n    steer_labels[\n        steering\n        > float(\n            steer_cfg["right_deadzone_deg"]\n        )\n    ] = "RIGHT"\n\n    return (\n        accel_labels.astype(str),\n        steer_labels.astype(str),\n    )\n\n\ndef predict_stage3(data_dir, model_dir):\n    import json\n\n    device = _device()\n    model_dir = Path(model_dir)\n\n    script_path = model_dir / "model.ts"\n    stats_path = model_dir / "target_stats.json"\n    calibration_path = model_dir / "calibration.json"\n\n    for required in (\n        script_path,\n        stats_path,\n        calibration_path,\n    ):\n        if not required.is_file():\n            raise FileNotFoundError(\n                f"Stage 3 asset not found: {required}"\n            )\n\n    stats = json.loads(\n        stats_path.read_text(encoding="utf-8")\n    )\n    calibration = json.loads(\n        calibration_path.read_text(\n            encoding="utf-8"\n        )\n    )\n\n    model = torch.jit.load(\n        str(script_path),\n        map_location=device,\n    )\n    model.eval()\n\n    videos = _video_paths(\n        Path(data_dir) / "videos"\n    )\n    rows = []\n\n    with torch.inference_mode():\n        for path in videos:\n            (\n                speed,\n                fused_accel,\n                raw_accel,\n                steering,\n            ) = _stage3_v4_predict_continuous(\n                path,\n                model,\n                device,\n                stats,\n            )\n\n            (\n                accel_labels,\n                steer_labels,\n            ) = _stage3_v4_apply_calibration(\n                speed,\n                fused_accel,\n                raw_accel,\n                steering,\n                calibration,\n            )\n\n            if len(accel_labels) != len(steer_labels):\n                raise RuntimeError(\n                    "Stage 3 output length mismatch "\n                    f"for {path.name}: "\n                    f"{len(accel_labels)} vs "\n                    f"{len(steer_labels)}"\n                )\n\n            for sample_index, (\n                accel_label,\n                steer_label,\n            ) in enumerate(\n                zip(\n                    accel_labels,\n                    steer_labels,\n                )\n            ):\n                rows.append(\n                    {\n                        "ID": path.stem,\n                        "sample_index": int(\n                            sample_index\n                        ),\n                        "accel_label": accel_label,\n                        "steer_label": steer_label,\n                    }\n                )\n\n    del model\n    torch.cuda.empty_cache()\n\n    return pd.DataFrame(\n        rows,\n        columns=[\n            "ID",\n            "sample_index",\n            "accel_label",\n            "steer_label",\n        ],\n    )\n'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def resolve_submission_root(root: Path) -> Path:
    if (
        (root / "inference.py").is_file()
        and (root / "model").is_dir()
    ):
        return root

    dirs = [
        p for p in root.iterdir()
        if p.is_dir()
    ]
    files = [
        p for p in root.iterdir()
        if p.is_file()
    ]

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


def write_submit_zip(
    source_root: Path,
    output_zip: Path,
) -> None:
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(
        output_zip,
        "w",
        allowZip64=True,
    ) as zf:
        for path in sorted(
            source_root.rglob("*")
        ):
            if not path.is_file():
                continue

            arcname = (
                path.relative_to(source_root)
                .as_posix()
            )

            compress_type = (
                zipfile.ZIP_STORED
                if path.suffix.lower()
                in {".pt", ".pth", ".ts"}
                else zipfile.ZIP_DEFLATED
            )

            zf.write(
                path,
                arcname,
                compress_type=compress_type,
            )


# ============================================================
# Colab / repository setup
# ============================================================
try:
    from google.colab import drive
except ImportError as exc:
    raise RuntimeError(
        "This build script is intended for Google Colab."
    ) from exc

drive.mount(
    "/content/drive",
    force_remount=False,
)

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Blackbox-Detection"
)

# When this file lives in the repository:
#   scripts/stage3/build_stage1a7_stage3_submission.py
THIS_FILE = Path(__file__).resolve()
REPO = THIS_FILE.parents[2]

if not (REPO / ".git").is_dir():
    raise RuntimeError(
        f"Repository not found at {REPO}. "
        "Clone Blackbox-Detection first."
    )

BRANCH = "stage3-sangchun"

subprocess.run(
    [
        "git",
        "-C",
        str(REPO),
        "checkout",
        BRANCH,
    ],
    check=True,
)

dirty = subprocess.run(
    [
        "git",
        "-C",
        str(REPO),
        "status",
        "--porcelain",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

if dirty:
    print(
        "WARNING: repository has local changes; "
        "git pull skipped."
    )
else:
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "pull",
            "--ff-only",
            "origin",
            BRANCH,
        ],
        check=True,
    )

# Build-time only. These are NOT copied into the submission
# requirements.txt.
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
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade-strategy",
        "only-if-needed",
        *BUILD_EXTRAS,
    ],
    check=True,
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "-e",
        str(REPO),
    ],
    check=True,
)

if str(REPO / "src") not in sys.path:
    sys.path.insert(
        0,
        str(REPO / "src"),
    )

import torch
import yaml
from torch import nn

from blackbox_detection.stage3.models import (
    VJEPA21DenseCAN,
)


# ============================================================
# User-editable input path
# ============================================================
BASE_A7_ZIP_OVERRIDE = None

a7_candidates = [
    DRIVE_ROOT
    / "submissions"
    / "stage1_a7.zip",
    DRIVE_ROOT
    / "stage1_a7.zip",
    Path("/content/stage1_a7.zip"),
]

if BASE_A7_ZIP_OVERRIDE is not None:
    a7_candidates.insert(
        0,
        Path(BASE_A7_ZIP_OVERRIDE),
    )

BASE_A7_ZIP = next(
    (
        p for p in a7_candidates
        if p.is_file()
    ),
    None,
)

if BASE_A7_ZIP is None:
    raise FileNotFoundError(
        "stage1_a7.zip not found. "
        "Put it on Drive or set "
        "BASE_A7_ZIP_OVERRIDE."
    )


# ============================================================
# Stage 3 sources
# ============================================================
# No submission template files are read from the repository.
# `submissions/` is gitignored in this project. The final inference.py is built
# directly from the uploaded/Drive A7 inference.py + the embedded Stage 3 block.
V4_RUN = (
    "vjepa21b_can_v4a_ordinal_fusion"
)

V4_BEST = (
    DRIVE_ROOT
    / "outputs"
    / "stage3"
    / V4_RUN
    / "best.pt"
)

TARGET_STATS = (
    DRIVE_ROOT
    / "manifests"
    / "stage3"
    / "v1"
    / "target_stats.json"
)

V4_CFG = (
    REPO
    / "configs"
    / "stage3"
    / "vjepa21b_can_accel_v4a.yaml"
)

CAL_DIR = (
    DRIVE_ROOT
    / "outputs"
    / "stage3"
    / "dacon_stage3_calibration_v1"
)

CONSENSUS_JSON = (
    CAL_DIR
    / "stage3_calibration_consensus.json"
)
ALLFIT_JSON = (
    CAL_DIR
    / "stage3_calibration_allfit.json"
)

for p in (
    V4_BEST,
    TARGET_STATS,
    V4_CFG,
    CONSENSUS_JSON,
    ALLFIT_JSON,
):
    if not p.is_file():
        raise FileNotFoundError(p)

OUT_DIR = (
    DRIVE_ROOT
    / "submissions"
    / "stage1a7_stage3_v1"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Base A7 ZIP :", BASE_A7_ZIP)
print("v4-A best   :", V4_BEST)
print("output dir  :", OUT_DIR)


# ============================================================
# Extract A7 and lock Stage 1 / Stage 2
# ============================================================
EXTRACT_DIR = Path(
    "/content/a7_base_extract"
)
BUILD_ROOT = Path(
    "/content/stage1a7_stage3_build"
)

for p in (
    EXTRACT_DIR,
    BUILD_ROOT,
):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)

with zipfile.ZipFile(
    BASE_A7_ZIP
) as zf:
    zf.extractall(EXTRACT_DIR)

A7_ROOT = resolve_submission_root(
    EXTRACT_DIR
)

shutil.copytree(
    A7_ROOT,
    BUILD_ROOT,
    dirs_exist_ok=True,
)

BASE_INFERENCE = (
    A7_ROOT
    / "inference.py"
).read_text(
    encoding="utf-8"
)

STAGE3_SPLIT = (
    "# ---------------------------------------------------------------------------\n"
    "# Stage 3:"
)

if STAGE3_SPLIT not in BASE_INFERENCE:
    raise RuntimeError(
        "Could not locate Stage 3 boundary in stage1_a7 inference.py."
    )

BASE_PREFIX = BASE_INFERENCE.split(
    STAGE3_SPLIT,
    1,
)[0]

# Final inference is always derived from the actual A7 file, so everything
# before Stage 3 is byte-identical by construction.
NEW_INFERENCE = BASE_PREFIX + STAGE3_V4_BLOCK

if (
    NEW_INFERENCE.split(
        STAGE3_SPLIT,
        1,
    )[0]
    != BASE_PREFIX
):
    raise RuntimeError(
        "ABORT: generated inference.py changed the A7 Stage 1/2 prefix."
    )

STAGE1_ROOT = (
    A7_ROOT
    / "model"
    / "stage1"
)

stage1_tree_before = {
    p.relative_to(
        STAGE1_ROOT
    ).as_posix(): sha256(p)
    for p in sorted(
        STAGE1_ROOT.rglob("*")
    )
    if p.is_file()
}

STAGE2_ROOT = A7_ROOT / "model" / "stage2"
stage2_tree_before = {
    p.relative_to(
        STAGE2_ROOT
    ).as_posix(): sha256(p)
    for p in sorted(
        STAGE2_ROOT.rglob("*")
    )
    if p.is_file()
}

a7_requirements_sha = sha256(
    A7_ROOT / "requirements.txt"
)

print(
    "Stage 1/2 inference prefix: "
    "A7 SOURCE LOCKED"
)
print(
    "Stage 1 files locked:",
    stage1_tree_before,
)


# ============================================================
# Build V-JEPA v4-A architecture
# ============================================================
cfg = yaml.safe_load(
    V4_CFG.read_text(
        encoding="utf-8"
    )
)

dc = cfg["data"]
mc = cfg["model"]
fusion_cfg = dict(
    mc.get("accel_fusion")
    or {}
)

VJEPA_REPO = Path(
    "/content/vjepa2"
)
VJEPA_COMMIT = (
    "45d025f636dfc58fc2426905fc4a1ab755b1c3e5"
)

if not (
    VJEPA_REPO / ".git"
).is_dir():
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "https://github.com/facebookresearch/vjepa2.git",
            str(VJEPA_REPO),
        ],
        check=True,
    )

subprocess.run(
    [
        "git",
        "-C",
        str(VJEPA_REPO),
        "fetch",
        "--all",
        "--tags",
    ],
    check=True,
)

subprocess.run(
    [
        "git",
        "-C",
        str(VJEPA_REPO),
        "checkout",
        "-q",
        VJEPA_COMMIT,
    ],
    check=True,
)

actual_commit = subprocess.run(
    [
        "git",
        "-C",
        str(VJEPA_REPO),
        "rev-parse",
        "HEAD",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

if actual_commit != VJEPA_COMMIT:
    raise RuntimeError(
        "V-JEPA commit mismatch: "
        f"{actual_commit}"
    )

backbone, predictor = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
    pretrained=False,
    num_frames=int(
        dc["clip_len"]
    ),
    out_layers=list(
        mc["out_layers"]
    ),
)

del predictor

backbone.requires_grad_(
    False
)
backbone.eval()

model = VJEPA21DenseCAN(
    backbone,
    freeze_backbone=True,
    feature_dim=int(
        mc["feature_dim"]
    ),
    temporal_hidden=int(
        mc["temporal_hidden"]
    ),
    temporal_layers=int(
        mc["temporal_layers"]
    ),
    accel_ordinal_thresholds_mps2=(
        mc[
            "accel_ordinal_thresholds_mps2"
        ]
    ),
    accel_fusion_enabled=bool(
        fusion_cfg.get(
            "enabled",
            True,
        )
    ),
    accel_fusion_hidden=int(
        fusion_cfg.get(
            "hidden",
            64,
        )
    ),
    accel_fusion_gate_init=float(
        fusion_cfg.get(
            "gate_init",
            0.10,
        )
    ),
    accel_fusion_detach_ordinal_inputs=bool(
        fusion_cfg.get(
            "detach_ordinal_inputs",
            True,
        )
    ),
)

checkpoint = torch.load(
    V4_BEST,
    map_location="cpu",
    weights_only=False,
)

state = checkpoint.get(
    "model"
)

if not isinstance(
    state,
    dict,
):
    raise RuntimeError(
        "v4-A checkpoint has no "
        "model state_dict"
    )

if not any(
    key.startswith(
        "backbone."
    )
    for key in state
):
    raise RuntimeError(
        "v4-A checkpoint does not "
        "contain backbone weights"
    )

incompatible = model.load_state_dict(
    state,
    strict=True,
)

if (
    incompatible.missing_keys
    or incompatible.unexpected_keys
):
    raise RuntimeError(
        str(incompatible)
    )

del checkpoint, state


# ============================================================
# Export Stage 3 TorchScript
# ============================================================
class Stage3ExportWrapper(nn.Module):
    def __init__(
        self,
        source_model,
    ):
        super().__init__()
        self.model = source_model

    def forward(
        self,
        video,
    ):
        out = self.model(video)
        return (
            out["speed_mps"],
            out[
                "accel_from_speed_mps2"
            ],
            out[
                "accel_raw_from_speed_mps2"
            ],
            out["steering_deg"],
        )


device = torch.device("cuda")

wrapper = (
    Stage3ExportWrapper(model)
    .to(device)
    .eval()
)

# Fixed batch size 2. inference.py always pads
# the final Stage 3 batch to exactly 2 clips.
example = torch.randn(
    2,
    3,
    int(dc["clip_len"]),
    int(dc["input_height"]),
    int(dc["input_width"]),
    device=device,
    dtype=torch.float32,
)

with torch.inference_mode():
    eager = tuple(
        x.detach().cpu()
        for x in wrapper(example)
    )

    traced = torch.jit.trace(
        wrapper,
        example,
        strict=False,
        check_trace=False,
    )

    traced = torch.jit.freeze(
        traced.eval()
    )

    traced_out = tuple(
        x.detach().cpu()
        for x in traced(example)
    )

expected_shape = (
    2,
    int(dc["clip_len"]),
)

for i, (
    eager_tensor,
    traced_tensor,
) in enumerate(
    zip(
        eager,
        traced_out,
    )
):
    torch.testing.assert_close(
        eager_tensor,
        traced_tensor,
        rtol=1e-4,
        atol=1e-4,
    )

    if tuple(
        eager_tensor.shape
    ) != expected_shape:
        raise RuntimeError(
            "Unexpected Stage 3 "
            f"output[{i}] shape: "
            f"{tuple(eager_tensor.shape)}"
        )

LOCAL_STAGE3_MODEL = Path(
    "/content/stage3_v4a_model.ts"
)

traced.save(
    str(LOCAL_STAGE3_MODEL)
)

# Reload exactly as inference.py will.
reloaded = torch.jit.load(
    str(LOCAL_STAGE3_MODEL),
    map_location=device,
).eval()

with torch.inference_mode():
    reload_out = tuple(
        x.detach().cpu()
        for x in reloaded(example)
    )

for (
    eager_tensor,
    reload_tensor,
) in zip(
    eager,
    reload_out,
):
    torch.testing.assert_close(
        eager_tensor,
        reload_tensor,
        rtol=1e-4,
        atol=1e-4,
    )

print(
    "Stage 3 TorchScript export: PASS"
)
print(
    "model.ts MiB:",
    f"{LOCAL_STAGE3_MODEL.stat().st_size / 2**20:.1f}",
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


# ============================================================
# Replace Stage 3 only
# ============================================================
STAGE3_DIR = (
    BUILD_ROOT
    / "model"
    / "stage3"
)

if STAGE3_DIR.exists():
    shutil.rmtree(
        STAGE3_DIR
    )

STAGE3_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

shutil.copy2(
    LOCAL_STAGE3_MODEL,
    STAGE3_DIR / "model.ts",
)

shutil.copy2(
    TARGET_STATS,
    STAGE3_DIR / "target_stats.json",
)

# Generate inference.py directly from the actual A7 submission.
(BUILD_ROOT / "inference.py").write_text(
    NEW_INFERENCE,
    encoding="utf-8",
)

# Stage 3 is TorchScript-only at evaluation time and adds no runtime pip
# dependency. Preserve the exact A7 requirements.txt instead of replacing it.
A7_REQUIREMENTS = A7_ROOT / "requirements.txt"
if not A7_REQUIREMENTS.is_file():
    raise FileNotFoundError(
        f"A7 requirements.txt not found: {A7_REQUIREMENTS}"
    )
shutil.copy2(
    A7_REQUIREMENTS,
    BUILD_ROOT / "requirements.txt",
)


# ============================================================
# Re-check immutable Stage 1 / Stage 2
# ============================================================
stage1_tree_after = {
    p.relative_to(
        BUILD_ROOT
        / "model"
        / "stage1"
    ).as_posix(): sha256(p)
    for p in sorted(
        (
            BUILD_ROOT
            / "model"
            / "stage1"
        ).rglob("*")
    )
    if p.is_file()
}

if (
    stage1_tree_after
    != stage1_tree_before
):
    raise RuntimeError(
        "ABORT: Stage 1 model files "
        "changed during build"
    )

stage2_tree_after = {
    p.relative_to(
        BUILD_ROOT / "model" / "stage2"
    ).as_posix(): sha256(p)
    for p in sorted(
        (BUILD_ROOT / "model" / "stage2").rglob("*")
    )
    if p.is_file()
}
if stage2_tree_after != stage2_tree_before:
    raise RuntimeError(
        "ABORT: Stage 2 model files changed during build"
    )

if sha256(BUILD_ROOT / "requirements.txt") != a7_requirements_sha:
    raise RuntimeError(
        "ABORT: requirements.txt changed from the A7 submission"
    )

final_inference = (
    BUILD_ROOT
    / "inference.py"
).read_text(
    encoding="utf-8"
)

if (
    final_inference.split(
        STAGE3_SPLIT,
        1,
    )[0]
    != BASE_PREFIX
):
    raise RuntimeError(
        "ABORT: final inference.py "
        "Stage 1/2 prefix changed"
    )

ast.parse(
    final_inference
)

print(
    "Stage 1 files unchanged: PASS"
)
print(
    "Stage 1/2 inference unchanged: PASS"
)
print(
    "Stage 2 model files unchanged: PASS"
)
print(
    "A7 requirements.txt unchanged: PASS"
)


# ============================================================
# Build consensus + all-fit variants
# ============================================================
variants = {
    "consensus": CONSENSUS_JSON,
    "allfit": ALLFIT_JSON,
}

built = {}

for (
    variant,
    calibration_path,
) in variants.items():
    calibration = json.loads(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )

    ordinal_blend = float(
        calibration[
            "accel"
        ].get(
            "ordinal_blend",
            0.0,
        )
    )

    if abs(
        ordinal_blend
    ) > 1e-12:
        raise RuntimeError(
            f"{variant} requires "
            "ordinal_blend != 0, but the "
            "compact export intentionally "
            "omits ordinal logits."
        )

    shutil.copy2(
        calibration_path,
        STAGE3_DIR
        / "calibration.json",
    )

    output_zip = (
        OUT_DIR
        / (
            "submit_stage1a7_stage3_"
            f"{variant}.zip"
        )
    )

    write_submit_zip(
        BUILD_ROOT,
        output_zip,
    )

    built[
        variant
    ] = output_zip

    print(
        variant,
        "->",
        output_zip,
        f"{output_zip.stat().st_size / 2**30:.3f} GiB",
    )


# ============================================================
# Validate actual ZIP bytes
# ============================================================
required = {
    "inference.py",
    "requirements.txt",
    "model/stage1/model.ts",
    "model/stage2/best.pt",
    "model/stage2/resnet18-f37072fd.pth",
    "model/stage3/model.ts",
    "model/stage3/target_stats.json",
    "model/stage3/calibration.json",
}

for (
    variant,
    zpath,
) in built.items():
    with zipfile.ZipFile(
        zpath
    ) as zf:
        names = set(
            zf.namelist()
        )

        roots = {
            name.split(
                "/",
                1,
            )[0]
            for name in names
            if name
        }

        invalid_roots = (
            roots
            - {
                "model",
                "inference.py",
                "requirements.txt",
            }
        )

        if invalid_roots:
            raise RuntimeError(
                f"{variant}: invalid "
                "top-level entries: "
                f"{sorted(invalid_roots)}"
            )

        missing = sorted(
            required - names
        )
        if missing:
            raise RuntimeError(
                f"{variant}: missing "
                f"files: {missing}"
            )

        stage1_bytes = zf.read(
            "model/stage1/model.ts"
        )
        stage1_zip_sha = (
            hashlib.sha256(
                stage1_bytes
            ).hexdigest()
        )

        expected_s1_sha = (
            stage1_tree_before[
                "model.ts"
            ]
        )

        if (
            stage1_zip_sha
            != expected_s1_sha
        ):
            raise RuntimeError(
                f"{variant}: Stage 1 "
                "model.ts changed"
            )

        inference_text = zf.read(
            "inference.py"
        ).decode(
            "utf-8"
        )

        if (
            inference_text.split(
                STAGE3_SPLIT,
                1,
            )[0]
            != BASE_PREFIX
        ):
            raise RuntimeError(
                f"{variant}: Stage 1/2 "
                "inference changed"
            )

        ast.parse(
            inference_text
        )

        forbidden = [
            name
            for name in names
            if (
                name
                == "model/stage3/best.pt"
                or "vjepa2_1_vitb_dist_vitG_384.pt"
                in name
                or name.startswith(
                    "model/stage3/vjepa2/"
                )
            )
        ]

        if forbidden:
            raise RuntimeError(
                f"{variant}: redundant "
                "Stage 3 assets found: "
                f"{forbidden}"
            )

    print(
        "ZIP contract PASS:",
        variant,
    )


# ============================================================
# Offline load smoke
# ============================================================
s1 = torch.jit.load(
    str(
        BUILD_ROOT
        / "model"
        / "stage1"
        / "model.ts"
    ),
    map_location=device,
).eval()

print(
    "Stage 1 TorchScript load: PASS"
)

del s1
torch.cuda.empty_cache()

s3 = torch.jit.load(
    str(
        BUILD_ROOT
        / "model"
        / "stage3"
        / "model.ts"
    ),
    map_location=device,
).eval()

dummy = torch.zeros(
    2,
    3,
    16,
    288,
    384,
    dtype=torch.float32,
    device=device,
)

with torch.inference_mode():
    output = s3(dummy)

if (
    not isinstance(
        output,
        (tuple, list),
    )
    or len(output) != 4
):
    raise RuntimeError(
        "Unexpected Stage 3 "
        "TorchScript output contract"
    )

for tensor in output:
    if tuple(
        tensor.shape
    ) != (2, 16):
        raise RuntimeError(
            f"Unexpected Stage 3 "
            f"shape: {tensor.shape}"
        )

print(
    "Stage 3 TorchScript "
    "load/forward: PASS"
)

del s3, dummy, output
torch.cuda.empty_cache()


# ============================================================
# Build report
# ============================================================
report = {
    "base_a7_zip": str(
        BASE_A7_ZIP
    ),
    "stage1_files_sha256": (
        stage1_tree_before
    ),
    "stage1_inference_prefix_sha256": (
        hashlib.sha256(
            BASE_PREFIX.encode(
                "utf-8"
            )
        ).hexdigest()
    ),
    "stage2_files_sha256": (
        stage2_tree_before
    ),
    "a7_requirements_sha256": (
        a7_requirements_sha
    ),
    "vjepa_commit": (
        VJEPA_COMMIT
    ),
    "stage3_model_ts_bytes": (
        LOCAL_STAGE3_MODEL.stat().st_size
    ),
    "consensus_zip": str(
        built["consensus"]
    ),
    "allfit_zip": str(
        built["allfit"]
    ),
    "consensus_zip_bytes": (
        built["consensus"].stat().st_size
    ),
    "allfit_zip_bytes": (
        built["allfit"].stat().st_size
    ),
}

report_path = (
    OUT_DIR
    / "build_report.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "\nBUILD COMPLETE"
)
print(
    "First submit :",
    built["consensus"],
)
print(
    "A/B variant  :",
    built["allfit"],
)
print(
    "Build report :",
    report_path,
)

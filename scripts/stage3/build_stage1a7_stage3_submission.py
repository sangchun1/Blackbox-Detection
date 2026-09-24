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
TEMPLATE_DIR = (
    REPO
    / "submissions"
    / "stage1a7_stage3"
)
INFERENCE_TEMPLATE = (
    TEMPLATE_DIR / "inference.py"
)
REQUIREMENTS_TEMPLATE = (
    TEMPLATE_DIR / "requirements.txt"
)

for p in (
    INFERENCE_TEMPLATE,
    REQUIREMENTS_TEMPLATE,
):
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing supplied template: {p}"
        )

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

NEW_INFERENCE = (
    INFERENCE_TEMPLATE
).read_text(
    encoding="utf-8"
)

STAGE3_SPLIT = (
    "# ---------------------------------------------------------------------------\n"
    "# Stage 3:"
)

if (
    STAGE3_SPLIT not in BASE_INFERENCE
    or STAGE3_SPLIT not in NEW_INFERENCE
):
    raise RuntimeError(
        "Could not locate Stage 3 boundary."
    )

BASE_PREFIX = BASE_INFERENCE.split(
    STAGE3_SPLIT,
    1,
)[0]

NEW_PREFIX = NEW_INFERENCE.split(
    STAGE3_SPLIT,
    1,
)[0]

if BASE_PREFIX != NEW_PREFIX:
    raise RuntimeError(
        "ABORT: supplied inference.py "
        "changes Stage 1/2 code. "
        "The A7 prefix must be byte-identical."
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

print(
    "Stage 1/2 inference prefix: "
    "EXACT MATCH"
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

shutil.copy2(
    INFERENCE_TEMPLATE,
    BUILD_ROOT / "inference.py",
)

shutil.copy2(
    REQUIREMENTS_TEMPLATE,
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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FRAME_COLUMNS = (
    "dataset",
    "vehicle_id",
    "route_id",
    "segment_id",
    "frame_index_10hz",
    "frame_index_source",
    "timestamp",
    "speed_mps",
    "speed_pose_mps",
    "accel_from_speed_mps2",
    "accel_imu_forward_mps2",
    "steering_deg",
    "steering_rate_dps",
    "yaw_rate_rps",
    "valid_speed",
    "valid_accel_from_speed",
    "valid_accel_imu",
    "valid_steer",
    "valid_yaw",
    "video_relpath",
)


def validate_frame_table(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    missing = [c for c in FRAME_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        if strict:
            raise ValueError(errors[-1])
        return errors

    if len(df) == 0:
        errors.append("empty frame table")
    if not df["frame_index_10hz"].is_monotonic_increasing:
        errors.append("frame_index_10hz is not monotonic")
    if df["frame_index_10hz"].duplicated().any():
        errors.append("duplicate frame_index_10hz")
    if not df["timestamp"].is_monotonic_increasing:
        errors.append("timestamp is not monotonic")

    if strict and errors:
        raise ValueError("; ".join(errors))
    return errors


def write_frame_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Store one segment without pyarrow/fastparquet.

    The DACON evaluation image does not list a parquet engine among its
    preinstalled packages. Stage 3 therefore keeps per-segment metadata as
    compressed NumPy arrays and split manifests as CSV files so the training
    and inference stack stays close to the official server environment.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"frame table must use .npz, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_frame_table(df)

    arrays: dict[str, np.ndarray] = {}
    for column in FRAME_COLUMNS:
        series = df[column]
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            arrays[column] = series.astype(str).to_numpy(dtype=np.str_)
        else:
            arrays[column] = series.to_numpy()
    np.savez_compressed(path, **arrays)
    return path


def read_frame_table(path: str | Path, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path)
    wanted = list(columns) if columns is not None else list(FRAME_COLUMNS)
    with np.load(path, allow_pickle=False) as data:
        missing = [c for c in wanted if c not in data.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        frame = pd.DataFrame({column: data[column] for column in wanted})
    return frame

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CAN_TARGETS, TARGET_TO_VALID
from .schema import read_frame_table


def build_segment_manifest(processed_root: str | Path) -> pd.DataFrame:
    root = Path(processed_root)
    rows = []
    for meta in sorted((root / "metadata").rglob("*.npz")):
        df = read_frame_table(meta)
        if df.empty:
            continue
        first = df.iloc[0]
        rows.append({
            "dataset": first["dataset"],
            "vehicle_id": first["vehicle_id"],
            "route_id": first["route_id"],
            "segment_id": str(first["segment_id"]),
            "num_frames": len(df),
            "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) > 1 else 0.0,
            "video_relpath": first["video_relpath"],
            "metadata_relpath": meta.relative_to(root).as_posix(),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["vehicle_id", "route_id", "segment_id"]).reset_index(drop=True)
    return out


def _route_score(route_id: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{route_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_routes(manifest: pd.DataFrame, val_fraction: float = 0.15, seed: int = 20260918):
    """Deterministic route-level split, separately within each vehicle."""
    train_routes, val_routes = set(), set()
    for _, group in manifest[["vehicle_id", "route_id"]].drop_duplicates().groupby("vehicle_id"):
        routes = sorted(group["route_id"].astype(str).tolist())
        scored = sorted(((_route_score(r, seed), r) for r in routes))
        n_val = max(1, int(round(len(routes) * val_fraction))) if len(routes) > 1 else 0
        val = {r for _, r in scored[:n_val]}
        val_routes |= val
        train_routes |= set(routes) - val
    train = manifest[manifest["route_id"].astype(str).isin(train_routes)].reset_index(drop=True)
    val = manifest[manifest["route_id"].astype(str).isin(val_routes)].reset_index(drop=True)
    return train, val


def compute_target_stats(segment_manifest: pd.DataFrame, processed_root: str | Path) -> dict:
    root = Path(processed_root)
    state = {
        name: {"n": 0, "sum": 0.0, "sum2": 0.0, "min": float("inf"), "max": float("-inf")}
        for name in CAN_TARGETS
    }
    for row in segment_manifest.itertuples(index=False):
        df = read_frame_table(root / row.metadata_relpath)
        for name in CAN_TARGETS:
            valid_col = TARGET_TO_VALID[name]
            values = df[name].to_numpy(dtype=float)
            mask = df[valid_col].astype(bool).to_numpy() & np.isfinite(values)
            x = values[mask].astype(np.float64, copy=False)
            if len(x) == 0:
                continue
            s = state[name]
            s["n"] += int(len(x))
            s["sum"] += float(x.sum())
            s["sum2"] += float(np.square(x).sum())
            s["min"] = min(s["min"], float(x.min()))
            s["max"] = max(s["max"], float(x.max()))

    result = {}
    for name, s in state.items():
        if s["n"] == 0:
            result[name] = {"mean": 0.0, "std": 1.0, "n": 0}
            continue
        mean = s["sum"] / s["n"]
        var = max(s["sum2"] / s["n"] - mean * mean, 1e-12)
        result[name] = {
            "mean": mean,
            "std": float(np.sqrt(var)),
            "min": s["min"],
            "max": s["max"],
            "n": s["n"],
        }
    return result


def save_stats(stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path

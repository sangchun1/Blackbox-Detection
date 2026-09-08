"""Reproducible video-level splits for Stage 1.

Splitting rules
---------------
1. The split is always decided at **video level**, before any frame or patch is
   extracted. Frames/patches of one video therefore never appear in both train
   and validation.
2. The split is written to CSV and reused by every model, so that model
   comparison and late fusion operate on identical validation videos.
3. When ``source_video_id`` is available (future paired CCD data), whole groups
   move together, so that ``ccd_001_original`` and ``ccd_001_rr_phoneA`` can
   never land on opposite sides.

DLC-2021 currently has no verifiable pairing, so ``source_video_id`` is empty
and ``strategy="auto"`` degenerates to a stratified video-level split. No
artificial grouping is fabricated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.seed import DEFAULT_SEED
from .manifest import validate_manifest

SplitStrategy = Literal["auto", "stratified", "group"]

SPLIT_COLUMNS: tuple[str, ...] = (
    "video_id",
    "dataset",
    "label",
    "source_video_id",
    "group_id",
    "split",
)

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


@dataclass(frozen=True)
class SplitConfig:
    """Configuration of a Stage 1 video-level split.

    Attributes:
        val_size: Fraction of groups assigned to validation, in ``(0, 1)``.
        seed: Random seed; the same seed always yields the same split.
        strategy: ``"stratified"`` splits individual videos, ``"group"`` splits
            whole ``source_video_id`` groups, ``"auto"`` uses ``"group"`` when
            any non-empty ``source_video_id`` exists and ``"stratified"``
            otherwise.
        stratify_columns: Columns forming the stratification key.
        group_column: Column holding the shared source identifier.
    """

    val_size: float = 0.2
    seed: int = DEFAULT_SEED
    strategy: SplitStrategy = "auto"
    stratify_columns: tuple[str, ...] = ("dataset", "label")
    group_column: str = "source_video_id"

    def __post_init__(self) -> None:
        if not 0.0 < self.val_size < 1.0:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")
        if self.strategy not in ("auto", "stratified", "group"):
            raise ValueError(f"Unknown split strategy: {self.strategy!r}")


@dataclass(frozen=True)
class ValidationSubsetSpec:
    """Definition of a controlled validation subset (VAL-B style diagnostic).

    Attributes:
        name: Subset name used in evaluation reports.
        column: Diagnostic manifest column to filter on.
        minimum: Inclusive lower bound, or ``None``.
        maximum: Inclusive upper bound, or ``None``.
        allowed_values: Explicit allowed values, or ``None``.
        min_per_class: Minimum videos required per Stage 1 class for the subset
            to be considered usable.
        description: Human-readable purpose of the subset.
    """

    name: str
    column: str
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[Any, ...] | None = None
    min_per_class: int = 20
    description: str = ""


@dataclass
class ValidationSubsetResult:
    """Outcome of materialising a :class:`ValidationSubsetSpec`."""

    name: str
    video_ids: list[str] = field(default_factory=list)
    usable: bool = False
    reason: str = ""
    class_counts: dict[str, int] = field(default_factory=dict)


def _resolve_strategy(manifest: pd.DataFrame, config: SplitConfig) -> SplitStrategy:
    if config.strategy != "auto":
        return config.strategy

    if config.group_column not in manifest.columns:
        return "stratified"

    groups = manifest[config.group_column].astype(str).str.strip()
    return "group" if bool((groups != "").any()) else "stratified"


def _group_ids(manifest: pd.DataFrame, config: SplitConfig) -> pd.Series:
    """Group key per video: ``source_video_id`` when known, else ``video_id``.

    Falling back to ``video_id`` keeps group splitting well defined on mixed
    manifests (grouped CCD rows next to ungrouped DLC rows) without inventing
    a grouping for the ungrouped rows.
    """
    if config.group_column not in manifest.columns:
        return manifest["video_id"].astype(str)

    source = manifest[config.group_column].astype(str).str.strip()
    return source.where(source != "", manifest["video_id"].astype(str))


def make_video_level_split(
    manifest: pd.DataFrame,
    *,
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Create a reproducible train/validation split at video level.

    Args:
        manifest: Stage 1 manifest. Filter out broken videos beforehand.
        config: Split configuration; defaults to ``SplitConfig()``.

    Returns:
        A ``DataFrame`` with :data:`SPLIT_COLUMNS`, one row per video.

    Raises:
        ValueError: If a stratum cannot supply at least one train and one
            validation group, which would make the split degenerate.
    """
    split_config = config or SplitConfig()
    validate_manifest(manifest, check_paths_exist=False)

    strategy = _resolve_strategy(manifest, split_config)
    frame = manifest.copy()
    frame["group_id"] = _group_ids(frame, split_config)

    missing_strata = [
        column
        for column in split_config.stratify_columns
        if column not in frame.columns
    ]
    if missing_strata:
        raise ValueError(f"Manifest is missing stratify columns: {missing_strata}")

    if strategy == "group":
        units = (
            frame.groupby("group_id", sort=True)
            .agg({column: "first" for column in split_config.stratify_columns})
            .reset_index()
        )
        # A group whose members disagree on the stratification key would make
        # stratification ill-defined; check explicitly instead of silently
        # taking the first value.
        inconsistent = [
            column
            for column in split_config.stratify_columns
            if frame.groupby("group_id")[column].nunique().max() > 1
        ]
        if inconsistent:
            raise ValueError(
                "Group-aware splitting requires a consistent stratification key "
                f"inside each group, but these columns vary: {inconsistent}. "
                "Either fix the manifest or use strategy='stratified'."
            )
    else:
        units = frame[["group_id", *split_config.stratify_columns]].copy()

    stratum = units[list(split_config.stratify_columns)].astype(str).agg("|".join, axis=1)
    units = units.assign(_stratum=stratum)

    rng = np.random.RandomState(split_config.seed)
    val_groups: set[str] = set()

    for stratum_value, stratum_units in units.groupby("_stratum", sort=True):
        group_ids = np.sort(stratum_units["group_id"].to_numpy().astype(str))
        num_groups = len(group_ids)
        num_val = int(round(num_groups * split_config.val_size))
        num_val = max(1, min(num_groups - 1, num_val)) if num_groups > 1 else 0

        if num_val == 0:
            raise ValueError(
                f"Stratum {stratum_value!r} has only {num_groups} group(s); "
                "it cannot be split into train and validation. Reduce the "
                "stratification granularity or collect more data."
            )

        permutation = rng.permutation(num_groups)
        val_groups.update(group_ids[permutation[:num_val]].tolist())

    split_frame = frame[
        ["video_id", "dataset", "label", "group_id"]
        + (
            [split_config.group_column]
            if split_config.group_column in frame.columns
            else []
        )
    ].copy()
    if split_config.group_column not in split_frame.columns:
        split_frame[split_config.group_column] = ""

    split_frame["split"] = np.where(
        split_frame["group_id"].isin(val_groups), VAL_SPLIT, TRAIN_SPLIT
    )
    split_frame = split_frame.rename(columns={split_config.group_column: "source_video_id"})
    split_frame = split_frame[list(SPLIT_COLUMNS)]
    split_frame = split_frame.sort_values("video_id", kind="mergesort").reset_index(
        drop=True
    )

    assert_no_group_leakage(split_frame)
    return split_frame


def assert_no_group_leakage(split: pd.DataFrame) -> None:
    """Raise if any ``group_id`` appears in both train and validation."""
    counts = split.groupby("group_id")["split"].nunique()
    leaked = counts[counts > 1].index.tolist()
    if leaked:
        raise ValueError(
            f"{len(leaked)} group(s) appear in both splits: {leaked[:5]}. "
            "This would leak frames/patches of the same source video."
        )


def assert_no_video_leakage(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Raise if a ``video_id`` occurs in both train and validation frames."""
    overlap = sorted(set(train["video_id"]) & set(val["video_id"]))
    if overlap:
        raise ValueError(
            f"{len(overlap)} video_id(s) appear in both splits: {overlap[:5]}"
        )


def apply_split(
    manifest: pd.DataFrame,
    split: pd.DataFrame,
    *,
    require_full_coverage: bool = True,
) -> dict[str, pd.DataFrame]:
    """Attach split assignments to a manifest and return per-split manifests.

    Args:
        manifest: Stage 1 manifest.
        split: Split table from :func:`make_video_level_split`.
        require_full_coverage: Raise when a manifest video has no assignment.

    Returns:
        Mapping ``{"train": DataFrame, "val": DataFrame}``; the frames carry the
        manifest columns plus ``split`` and ``group_id``.
    """
    assignments = split[["video_id", "split", "group_id"]]
    merged = manifest.merge(assignments, on="video_id", how="left", validate="one_to_one")

    unassigned = merged.loc[merged["split"].isna(), "video_id"]
    if len(unassigned):
        if require_full_coverage:
            raise ValueError(
                f"{len(unassigned)} manifest video(s) are missing from the split "
                f"table. Examples: {unassigned.tolist()[:5]}. Regenerate the "
                "split after changing the manifest."
            )
        merged = merged.loc[merged["split"].notna()].reset_index(drop=True)

    train = merged.loc[merged["split"] == TRAIN_SPLIT].reset_index(drop=True)
    val = merged.loc[merged["split"] == VAL_SPLIT].reset_index(drop=True)
    assert_no_video_leakage(train, val)
    return {TRAIN_SPLIT: train, VAL_SPLIT: val}


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    """Counts of videos and groups per split, dataset and label."""
    summary = (
        split.groupby(["split", "dataset", "label"], dropna=False)
        .agg(num_videos=("video_id", "count"), num_groups=("group_id", "nunique"))
        .reset_index()
    )
    return summary


def build_validation_subsets(
    val_manifest: pd.DataFrame,
    specs: Sequence[ValidationSubsetSpec],
) -> dict[str, ValidationSubsetResult]:
    """Materialise controlled validation subsets for shortcut diagnostics.

    VAL-A is the full stratified validation set. Each spec here defines a
    VAL-B style controlled subset (for example 4K-only videos) used to check how
    much of the score depends on resolution/FPS shortcuts. A subset is marked
    unusable, rather than silently reported, when it does not hold enough
    videos per class.

    Args:
        val_manifest: Validation manifest (output of :func:`apply_split`).
        specs: Subset definitions.

    Returns:
        Mapping from subset name to :class:`ValidationSubsetResult`.
    """
    results: dict[str, ValidationSubsetResult] = {}

    for spec in specs:
        if spec.column not in val_manifest.columns:
            results[spec.name] = ValidationSubsetResult(
                name=spec.name,
                usable=False,
                reason=f"column {spec.column!r} is not in the manifest",
            )
            continue

        mask = pd.Series(True, index=val_manifest.index)
        values = val_manifest[spec.column]
        if spec.minimum is not None:
            mask &= pd.to_numeric(values, errors="coerce") >= spec.minimum
        if spec.maximum is not None:
            mask &= pd.to_numeric(values, errors="coerce") <= spec.maximum
        if spec.allowed_values is not None:
            mask &= values.isin(list(spec.allowed_values))

        subset = val_manifest.loc[mask.fillna(False)]
        class_counts = subset["label"].value_counts().to_dict()
        minimum_count = min(
            (int(class_counts.get(label, 0)) for label in val_manifest["label"].unique()),
            default=0,
        )
        usable = bool(len(subset)) and minimum_count >= spec.min_per_class

        results[spec.name] = ValidationSubsetResult(
            name=spec.name,
            video_ids=subset["video_id"].tolist(),
            usable=usable,
            reason=(
                ""
                if usable
                else (
                    f"only {minimum_count} video(s) in the smallest class, "
                    f"need >= {spec.min_per_class}"
                )
            ),
            class_counts={str(key): int(value) for key, value in class_counts.items()},
        )

    return results


def resolution_subset_spec(
    *,
    name: str = "val_b_4k",
    min_height: int = 2000,
    min_per_class: int = 20,
) -> ValidationSubsetSpec:
    """Convenience spec for a high-resolution-only controlled subset."""
    return ValidationSubsetSpec(
        name=name,
        column="height",
        minimum=float(min_height),
        min_per_class=min_per_class,
        description=(
            "Controlled subset restricted to high-resolution videos, used to "
            "check how much Stage 1 performance depends on a resolution "
            "shortcut rather than on recapture artefacts."
        ),
    )


def save_split(split: pd.DataFrame, path: str | Path) -> Path:
    """Write a split table to CSV."""
    missing = [column for column in SPLIT_COLUMNS if column not in split.columns]
    if missing:
        raise ValueError(f"Split table is missing columns: {missing}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split[list(SPLIT_COLUMNS)].to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def load_split(path: str | Path) -> pd.DataFrame:
    """Read a split table from CSV and re-check for group leakage."""
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split = pd.read_csv(split_path, encoding="utf-8")
    missing = [column for column in SPLIT_COLUMNS if column not in split.columns]
    if missing:
        raise ValueError(f"Split file {split_path} is missing columns: {missing}")

    for column in ("video_id", "dataset", "label", "source_video_id", "group_id", "split"):
        split[column] = split[column].fillna("").astype(str)

    assert_no_group_leakage(split)
    return split


def split_config_from_mapping(mapping: Mapping[str, Any]) -> SplitConfig:
    """Build a :class:`SplitConfig` from a YAML/dict fragment."""
    known = {field_name for field_name in SplitConfig.__dataclass_fields__}
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(f"Unknown split config keys: {sorted(unknown)}")

    payload = dict(mapping)
    if "stratify_columns" in payload:
        payload["stratify_columns"] = tuple(payload["stratify_columns"])
    return SplitConfig(**payload)


__all__ = [
    "SPLIT_COLUMNS",
    "TRAIN_SPLIT",
    "VAL_SPLIT",
    "SplitStrategy",
    "SplitConfig",
    "ValidationSubsetSpec",
    "ValidationSubsetResult",
    "make_video_level_split",
    "assert_no_group_leakage",
    "assert_no_video_leakage",
    "apply_split",
    "split_summary",
    "build_validation_subsets",
    "resolution_subset_spec",
    "save_split",
    "load_split",
    "split_config_from_mapping",
]

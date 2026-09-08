"""Stage 1 manifest construction.

Stage 1 classifies a video as ORIGINAL or RERECORDED. Several data sources feed
this stage over time (currently DLC-2021; later CCD originals and their physical
display-camera re-recordings), so every dataset is normalised into one common
manifest schema instead of giving each dataset its own ``Dataset`` class.

Schema
------
Required columns (:data:`STAGE1_MANIFEST_COLUMNS`)

``video_path``
    Absolute path to the video file, stored as a plain string.
``label``
    Official Stage 1 label, ``"ORIGINAL"`` or ``"RERECORDED"``.
``dataset``
    Source dataset key, e.g. ``"dlc2021"`` and later ``"ccd"``.
``video_id``
    Globally unique identifier of this video file.
``source_video_id``
    Identifier shared by every recording of the same underlying content.
    Empty string when unknown. This is what makes group-aware splitting
    possible once paired CCD re-recordings exist; it is never guessed.
``scene_type``
    Content domain, e.g. ``"document"`` (DLC-2021) or ``"driving"`` (CCD).
``is_synthetic``
    ``True`` only for digitally simulated re-recording, ``False`` for real
    display-camera captures.
``capture_device`` / ``display_device``
    Recapture chain metadata, empty string when unknown.

Diagnostic columns (:data:`STAGE1_DIAGNOSTIC_COLUMNS`) are filled by
:func:`probe_video_metadata` and exist for dataset analysis and split
diagnostics only. Resolution, FPS and codec are shortcut features for this task
and must never be fed to a classifier.

Unknown metadata stays empty. Nothing here infers pairs, devices or source
identity from file names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from ..utils.metrics import STAGE1_LABELS

# Manifest schema -------------------------------------------------------------

STAGE1_MANIFEST_COLUMNS: tuple[str, ...] = (
    "video_path",
    "label",
    "dataset",
    "video_id",
    "source_video_id",
    "scene_type",
    "is_synthetic",
    "capture_device",
    "display_device",
)

STAGE1_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "width",
    "height",
    "fps",
    "num_frames",
    "duration_sec",
    "codec",
    "file_size_bytes",
    "is_readable",
    "probe_error",
)

STAGE1_LABEL_TO_INDEX: Mapping[str, int] = {
    label: index for index, label in enumerate(STAGE1_LABELS)
}
STAGE1_INDEX_TO_LABEL: Mapping[int, str] = {
    index: label for label, index in STAGE1_LABEL_TO_INDEX.items()
}

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv", ".mpg", ".mpeg"}
)

# DLC-2021 ships four subsets. Only the real screen-recapture pair is used for
# Stage 1: ``or`` (original) and ``re`` (recaptured). ``cc``/``cg`` are excluded.
DLC2021_LABEL_DIRECTORIES: Mapping[str, str] = {
    "or": "ORIGINAL",
    "re": "RERECORDED",
}
DLC2021_IGNORED_DIRECTORIES: frozenset[str] = frozenset({"cc", "cg"})


@dataclass(frozen=True)
class VideoMetadata:
    """Container-level metadata used for diagnostics only."""

    width: int
    height: int
    fps: float
    num_frames: int
    duration_sec: float
    codec: str
    file_size_bytes: int
    is_readable: bool
    probe_error: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_sec": self.duration_sec,
            "codec": self.codec,
            "file_size_bytes": self.file_size_bytes,
            "is_readable": self.is_readable,
            "probe_error": self.probe_error,
        }


_UNKNOWN_METADATA = VideoMetadata(
    width=0,
    height=0,
    fps=0.0,
    num_frames=0,
    duration_sec=0.0,
    codec="",
    file_size_bytes=0,
    is_readable=False,
    probe_error="not probed",
)


# Metadata probing ------------------------------------------------------------


def _fourcc_to_str(fourcc: float) -> str:
    """Decode an OpenCV FOURCC value into its four-character string."""
    value = int(fourcc)
    if value <= 0:
        return ""
    try:
        return "".join(chr((value >> (8 * shift)) & 0xFF) for shift in range(4)).strip()
    except ValueError:  # pragma: no cover - defensive
        return ""


def probe_video_metadata(
    path: str | Path,
    *,
    verify_decode: bool = True,
) -> VideoMetadata:
    """Read container metadata and optionally verify that one frame decodes.

    Args:
        path: Video file path.
        verify_decode: Decode the first frame to detect broken files. The
            reported frame size is taken from the decoded frame when available,
            because some containers report wrong header values.

    Returns:
        A :class:`VideoMetadata` instance. Unreadable files are reported with
        ``is_readable=False`` and a non-empty ``probe_error`` instead of raising,
        so that a scan can report every broken video at once.
    """
    video_path = Path(path)
    file_size = video_path.stat().st_size if video_path.is_file() else 0

    if not video_path.is_file():
        return VideoMetadata(
            width=0,
            height=0,
            fps=0.0,
            num_frames=0,
            duration_sec=0.0,
            codec="",
            file_size_bytes=file_size,
            is_readable=False,
            probe_error="file not found",
        )

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return VideoMetadata(
                width=0,
                height=0,
                fps=0.0,
                num_frames=0,
                duration_sec=0.0,
                codec="",
                file_size_bytes=file_size,
                is_readable=False,
                probe_error="cv2.VideoCapture could not open the file",
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _fourcc_to_str(capture.get(cv2.CAP_PROP_FOURCC))

        error = ""
        if verify_decode:
            ok, frame = capture.read()
            if not ok or frame is None:
                error = "first frame could not be decoded"
            else:
                height, width = int(frame.shape[0]), int(frame.shape[1])

        if not error:
            if width <= 0 or height <= 0:
                error = "invalid frame size reported"
            elif num_frames <= 0:
                error = "invalid frame count reported"
    finally:
        capture.release()

    fps = fps if fps and fps > 0 else 0.0
    duration = float(num_frames) / fps if fps > 0 and num_frames > 0 else 0.0

    return VideoMetadata(
        width=max(width, 0),
        height=max(height, 0),
        fps=fps,
        num_frames=max(num_frames, 0),
        duration_sec=duration,
        codec=codec,
        file_size_bytes=file_size,
        is_readable=not error,
        probe_error=error,
    )


# Generic scanning ------------------------------------------------------------


def _slugify_relative_path(relative_path: Path) -> str:
    """Build a filesystem-independent identifier fragment from a relative path."""
    parts = [*relative_path.parent.parts, relative_path.stem]
    cleaned = [
        "".join(char if char.isalnum() else "_" for char in part).strip("_")
        for part in parts
        if part not in (".", "")
    ]
    return "_".join(part for part in cleaned if part)


def iter_video_files(
    directory: str | Path,
    *,
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
    recursive: bool = True,
) -> list[Path]:
    """Return video files under ``directory`` in a deterministic order."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    allowed = {ext.lower() for ext in extensions}
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in allowed
    ]
    # Sort on POSIX-style relative strings so Windows and Linux agree.
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def scan_video_directory(
    directory: str | Path,
    *,
    label: str,
    dataset: str,
    scene_type: str,
    video_id_prefix: str,
    source_video_id: str = "",
    is_synthetic: bool = False,
    capture_device: str = "",
    display_device: str = "",
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
    recursive: bool = True,
    probe_metadata: bool = True,
    verify_decode: bool = True,
    metadata_from_path: Callable[[Path], Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Scan one directory of same-label videos into Stage 1 manifest rows.

    This is the shared building block for every dataset scanner, including the
    future CCD original / CCD re-recording scanners.

    Args:
        directory: Directory containing the videos.
        label: Official Stage 1 label for every file found here.
        dataset: Source dataset key.
        scene_type: Content domain of the videos.
        video_id_prefix: Prefix for generated ``video_id`` values.
        source_video_id: Constant ``source_video_id`` for all rows. Leave empty
            when the pairing is unknown; do not invent groups.
        is_synthetic: Whether these videos are digitally simulated recaptures.
        capture_device: Camera used for recapture, empty when unknown.
        display_device: Display used for playback, empty when unknown.
        extensions: Accepted file suffixes.
        recursive: Whether to descend into subdirectories.
        probe_metadata: Fill the diagnostic columns.
        verify_decode: Decode one frame per file while probing.
        metadata_from_path: Optional hook returning manifest overrides for one
            file, e.g. to fill ``source_video_id``/``capture_device`` from a
            dataset-specific naming convention that is actually documented.

    Returns:
        A manifest ``DataFrame`` with the required and diagnostic columns.
    """
    if label not in STAGE1_LABEL_TO_INDEX:
        raise ValueError(
            f"label must be one of {tuple(STAGE1_LABELS)!r}, got {label!r}."
        )

    root = Path(directory)
    rows: list[dict[str, Any]] = []

    for path in iter_video_files(root, extensions=extensions, recursive=recursive):
        relative = path.relative_to(root)
        row: dict[str, Any] = {
            "video_path": str(path.resolve()),
            "label": label,
            "dataset": dataset,
            "video_id": f"{video_id_prefix}_{_slugify_relative_path(relative)}",
            "source_video_id": source_video_id,
            "scene_type": scene_type,
            "is_synthetic": bool(is_synthetic),
            "capture_device": capture_device,
            "display_device": display_device,
        }

        metadata = (
            probe_video_metadata(path, verify_decode=verify_decode)
            if probe_metadata
            else _UNKNOWN_METADATA
        )
        row.update(metadata.as_dict())

        if metadata_from_path is not None:
            overrides = dict(metadata_from_path(path))
            unknown = set(overrides) - set(STAGE1_MANIFEST_COLUMNS)
            if unknown:
                raise ValueError(
                    "metadata_from_path may only override manifest columns, "
                    f"got unknown keys: {sorted(unknown)}"
                )
            row.update(overrides)

        rows.append(row)

    return _to_manifest_frame(rows)


def _to_manifest_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [*STAGE1_MANIFEST_COLUMNS, *STAGE1_DIAGNOSTIC_COLUMNS]
    frame = pd.DataFrame(list(rows), columns=columns)
    return _coerce_manifest_dtypes(frame)


def _coerce_manifest_dtypes(manifest: pd.DataFrame) -> pd.DataFrame:
    frame = manifest.copy()

    string_columns = [
        "video_path",
        "label",
        "dataset",
        "video_id",
        "source_video_id",
        "scene_type",
        "capture_device",
        "display_device",
        "codec",
        "probe_error",
    ]
    for column in string_columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)

    for column in ("width", "height", "num_frames", "file_size_bytes"):
        if column in frame.columns:
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int64")
            )

    for column in ("fps", "duration_sec"):
        if column in frame.columns:
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
            )

    for column in ("is_synthetic", "is_readable"):
        if column in frame.columns:
            frame[column] = frame[column].fillna(False).astype(bool)

    return frame


# DLC-2021 --------------------------------------------------------------------


def scan_dlc2021(
    root: str | Path,
    *,
    dataset: str = "dlc2021",
    scene_type: str = "document",
    label_directories: Mapping[str, str] = DLC2021_LABEL_DIRECTORIES,
    video_id_prefix: str = "dlc",
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
    probe_metadata: bool = True,
    verify_decode: bool = True,
    metadata_from_path: Callable[[Path], Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Scan DLC-2021 ``or``/``re`` subsets into a Stage 1 manifest.

    ``cc`` and ``cg`` are ignored: only the real screen-recapture pair is used
    for Stage 1 pretraining.

    ``source_video_id``, ``capture_device`` and ``display_device`` are left
    empty because DLC-2021 does not expose a verifiable original/recapture
    pairing here. Guessing them would silently corrupt group-aware splitting.

    Args:
        root: DLC-2021 root directory containing the subset directories.
        dataset: Dataset key written into the manifest.
        scene_type: Content domain label; DLC-2021 is document-centric.
        label_directories: Mapping of subset directory name to Stage 1 label.
        video_id_prefix: Prefix of generated ``video_id`` values.
        extensions: Accepted video suffixes.
        probe_metadata: Fill the diagnostic columns.
        verify_decode: Decode one frame per file while probing.
        metadata_from_path: Optional hook to fill documented metadata.

    Returns:
        Validated Stage 1 manifest for DLC-2021.

    Raises:
        FileNotFoundError: If ``root`` or a required subset directory is missing.
    """
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DLC-2021 root not found: {dataset_root}")

    frames: list[pd.DataFrame] = []
    for directory_name, label in label_directories.items():
        subset_dir = dataset_root / directory_name
        if not subset_dir.is_dir():
            raise FileNotFoundError(
                f"DLC-2021 subset directory not found: {subset_dir}. "
                f"Expected subsets: {sorted(label_directories)}"
            )

        frames.append(
            scan_video_directory(
                subset_dir,
                label=label,
                dataset=dataset,
                scene_type=scene_type,
                video_id_prefix=f"{video_id_prefix}_{directory_name}",
                source_video_id="",
                is_synthetic=False,
                capture_device="",
                display_device="",
                extensions=extensions,
                recursive=True,
                probe_metadata=probe_metadata,
                verify_decode=verify_decode,
                metadata_from_path=metadata_from_path,
            )
        )

    manifest = merge_manifests(*frames)
    validate_manifest(manifest, check_paths_exist=False)
    return manifest


# Manifest operations ---------------------------------------------------------


def merge_manifests(*manifests: pd.DataFrame) -> pd.DataFrame:
    """Concatenate manifests and sort deterministically by ``video_id``.

    This is the extension point for adding datasets: build one manifest per
    source (DLC-2021 now, CCD originals and CCD re-recordings later) and merge.
    """
    if not manifests:
        raise ValueError("merge_manifests() requires at least one manifest.")

    frames = [frame for frame in manifests if len(frame) > 0]
    if not frames:
        return _to_manifest_frame([])

    columns = [*STAGE1_MANIFEST_COLUMNS, *STAGE1_DIAGNOSTIC_COLUMNS]
    merged = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in merged.columns:
            merged[column] = ""

    merged = merged[columns]
    merged = merged.sort_values("video_id", kind="mergesort").reset_index(drop=True)
    return _coerce_manifest_dtypes(merged)


def validate_manifest(
    manifest: pd.DataFrame,
    *,
    check_paths_exist: bool = False,
    require_readable: bool = False,
) -> None:
    """Validate the Stage 1 manifest schema.

    Args:
        manifest: Manifest to check.
        check_paths_exist: Verify that every ``video_path`` exists on disk.
        require_readable: Require ``is_readable`` to be true for every row.

    Raises:
        ValueError: On a missing column, unknown label, duplicate ``video_id``,
            missing file, or unreadable video when requested.
    """
    missing = [
        column for column in STAGE1_MANIFEST_COLUMNS if column not in manifest.columns
    ]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if len(manifest) == 0:
        raise ValueError("Manifest is empty.")

    unknown_labels = sorted(set(manifest["label"]) - set(STAGE1_LABELS))
    if unknown_labels:
        raise ValueError(
            f"Manifest contains labels outside {tuple(STAGE1_LABELS)!r}: "
            f"{unknown_labels}"
        )

    duplicated = manifest["video_id"][manifest["video_id"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            f"Manifest contains {len(duplicated)} duplicate video_id value(s). "
            f"Examples: {sorted(duplicated)[:5]}"
        )

    if check_paths_exist:
        missing_paths = [
            path for path in manifest["video_path"] if not Path(path).is_file()
        ]
        if missing_paths:
            raise ValueError(
                f"{len(missing_paths)} manifest video_path value(s) do not exist. "
                f"Examples: {missing_paths[:5]}"
            )

    if require_readable and "is_readable" in manifest.columns:
        broken = manifest.loc[~manifest["is_readable"].astype(bool), "video_id"]
        if len(broken):
            raise ValueError(
                f"{len(broken)} manifest row(s) are not readable. "
                f"Examples: {broken.tolist()[:5]}"
            )


def broken_videos(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return the rows whose video could not be probed or decoded."""
    if "is_readable" not in manifest.columns:
        return manifest.iloc[0:0]

    columns = ["video_id", "dataset", "label", "video_path", "probe_error"]
    available = [column for column in columns if column in manifest.columns]
    return manifest.loc[~manifest["is_readable"].astype(bool), available].reset_index(
        drop=True
    )


def drop_broken_videos(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return only the readable rows, keeping manifest ordering."""
    if "is_readable" not in manifest.columns:
        return manifest.reset_index(drop=True)
    return manifest.loc[manifest["is_readable"].astype(bool)].reset_index(drop=True)


def filter_manifest(
    manifest: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    scene_types: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    readable_only: bool = False,
) -> pd.DataFrame:
    """Subset a manifest by dataset, label, scene type or explicit ids."""
    frame = manifest
    if datasets is not None:
        frame = frame[frame["dataset"].isin(list(datasets))]
    if labels is not None:
        frame = frame[frame["label"].isin(list(labels))]
    if scene_types is not None:
        frame = frame[frame["scene_type"].isin(list(scene_types))]
    if video_ids is not None:
        frame = frame[frame["video_id"].isin(list(video_ids))]
    if readable_only and "is_readable" in frame.columns:
        frame = frame[frame["is_readable"].astype(bool)]
    return frame.reset_index(drop=True)


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset/label counts plus diagnostic ranges.

    Resolution and FPS are reported for analysis only; they are shortcut
    features for this task and must not be used as classifier inputs.
    """
    grouped = manifest.groupby(["dataset", "label"], dropna=False)
    summary = grouped.agg(
        num_videos=("video_id", "count"),
        num_broken=("is_readable", lambda values: int((~values.astype(bool)).sum())),
        num_source_groups=(
            "source_video_id",
            lambda values: int(values[values.astype(str) != ""].nunique()),
        ),
        min_height=("height", "min"),
        max_height=("height", "max"),
        median_fps=("fps", "median"),
        median_duration_sec=("duration_sec", "median"),
    )
    return summary.reset_index()


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    """Write a manifest to CSV, creating parent directories."""
    validate_manifest(manifest, check_paths_exist=False)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def load_manifest(
    path: str | Path,
    *,
    validate: bool = True,
    check_paths_exist: bool = False,
) -> pd.DataFrame:
    """Read a manifest CSV back with stable dtypes."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, encoding="utf-8")
    manifest = _coerce_manifest_dtypes(manifest)
    if validate:
        validate_manifest(manifest, check_paths_exist=check_paths_exist)
    return manifest


def add_manifest_rows(
    manifest: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Append explicit rows to a manifest and re-validate.

    Useful for registering a small number of videos by hand, and the mechanism
    by which CCD / CCD-rerecorded rows will be added later.
    """
    return merge_manifests(manifest, _to_manifest_frame(rows))


__all__ = [
    "STAGE1_MANIFEST_COLUMNS",
    "STAGE1_DIAGNOSTIC_COLUMNS",
    "STAGE1_LABEL_TO_INDEX",
    "STAGE1_INDEX_TO_LABEL",
    "VIDEO_EXTENSIONS",
    "DLC2021_LABEL_DIRECTORIES",
    "DLC2021_IGNORED_DIRECTORIES",
    "VideoMetadata",
    "probe_video_metadata",
    "iter_video_files",
    "scan_video_directory",
    "scan_dlc2021",
    "merge_manifests",
    "validate_manifest",
    "broken_videos",
    "drop_broken_videos",
    "filter_manifest",
    "manifest_summary",
    "save_manifest",
    "load_manifest",
    "add_manifest_rows",
]

"""Stage 1: ORIGINAL vs RERECORDED video classification.

Stage 1 decides whether a video was captured directly or replayed on a display
and filmed again. The discriminative evidence is not semantic, so two
complementary branches are trained independently and fused at probability
level:

* the **video branch** (:mod:`.models.videomaev2`, :mod:`.models.vjepa2`) learns
  spatio-temporal recapture behaviour: temporal flicker, time-varying moire,
  camera shake, global perspective;
* the **forensic branch** (:mod:`.models.forensic`) learns per-patch traces of
  the display-camera chain: moire, chromaticity distortion, aliasing,
  processing residual, frequency-domain signatures.

Data flow::

    manifest.py  ->  split.py  ->  sampling.py + transforms.py  ->  dataset.py
                                                                       |
                                                            models/factory.py
                                                                       |
                                            trainer.py  ->  evaluator.py
                                                                       |
                                                  fusion.py / inference.py

Current data is DLC-2021 only (``or`` -> ORIGINAL, ``re`` -> RERECORDED). The
manifest, split and dataset layers are dataset-agnostic on purpose: adding the
planned CCD originals and their physical re-recordings means adding manifest
rows and switching the split strategy to group-aware, not rewriting the
pipeline.
"""

from __future__ import annotations

from .dataset import (
    AdaptedBatch,
    Stage1BatchAdapter,
    Stage1ForensicDataset,
    Stage1VideoDataset,
    VideoDecodeError,
    build_dataloader,
    decode_frames,
    forensic_batch_adapter,
    stage1_collate,
    video_batch_adapter,
)
from .evaluator import (
    PREDICTION_COLUMNS,
    AggregationConfig,
    EvaluationResult,
    Stage1Evaluator,
    aggregate_unit_predictions,
    evaluate_predictions,
    load_predictions,
    per_class_f1,
    save_predictions,
    search_best_threshold,
)
from .fusion import (
    FusionResult,
    compare_models,
    fuse_probabilities,
    fused_predictions,
    load_prediction_tables,
    prediction_correlation,
    search_all_combinations,
    search_late_fusion,
)
from .inference import (
    LoadedStage1Model,
    load_stage1_model,
    manifest_from_videos,
    predict_stage1,
    predict_stage1_proba,
)
from .manifest import (
    STAGE1_DIAGNOSTIC_COLUMNS,
    STAGE1_INDEX_TO_LABEL,
    STAGE1_LABEL_TO_INDEX,
    STAGE1_MANIFEST_COLUMNS,
    broken_videos,
    drop_broken_videos,
    filter_manifest,
    load_manifest,
    manifest_summary,
    merge_manifests,
    probe_video_metadata,
    save_manifest,
    scan_dlc2021,
    scan_video_directory,
    validate_manifest,
)
from .models import (
    STAGE1_MODEL_NAMES,
    Stage1Model,
    build_stage1_model,
    model_input_kind,
)
from .sampling import (
    DeterministicClipSampler,
    DeterministicFrameSampler,
    DeterministicPatchSampler,
    DenseClipSampler,
    MultiClipSampler,
    RandomFrameSampler,
    RandomPatchSampler,
    build_clip_sampler,
    build_forensic_samplers,
)
from .split import (
    SplitConfig,
    ValidationSubsetSpec,
    apply_split,
    build_validation_subsets,
    load_split,
    make_video_level_split,
    resolution_subset_spec,
    save_split,
    split_summary,
)
from .trainer import Stage1Trainer, TrainConfig, TrainingOutcome
from .transforms import (
    ClipAugmentConfig,
    ClipTransform,
    ForensicPatchTransform,
    PatchAugmentConfig,
    ValClipTransform,
    build_forensic_transforms,
    build_video_transforms,
    fmag_spectral_augment,
)

__all__ = [
    # manifest
    "STAGE1_MANIFEST_COLUMNS",
    "STAGE1_DIAGNOSTIC_COLUMNS",
    "STAGE1_LABEL_TO_INDEX",
    "STAGE1_INDEX_TO_LABEL",
    "scan_dlc2021",
    "scan_video_directory",
    "merge_manifests",
    "validate_manifest",
    "broken_videos",
    "drop_broken_videos",
    "filter_manifest",
    "manifest_summary",
    "probe_video_metadata",
    "save_manifest",
    "load_manifest",
    # split
    "SplitConfig",
    "ValidationSubsetSpec",
    "make_video_level_split",
    "apply_split",
    "split_summary",
    "build_validation_subsets",
    "resolution_subset_spec",
    "save_split",
    "load_split",
    # sampling
    "DenseClipSampler",
    "DeterministicClipSampler",
    "MultiClipSampler",
    "RandomFrameSampler",
    "DeterministicFrameSampler",
    "RandomPatchSampler",
    "DeterministicPatchSampler",
    "build_clip_sampler",
    "build_forensic_samplers",
    # transforms
    "ClipAugmentConfig",
    "PatchAugmentConfig",
    "ClipTransform",
    "ValClipTransform",
    "ForensicPatchTransform",
    "fmag_spectral_augment",
    "build_video_transforms",
    "build_forensic_transforms",
    # dataset
    "Stage1VideoDataset",
    "Stage1ForensicDataset",
    "Stage1BatchAdapter",
    "AdaptedBatch",
    "video_batch_adapter",
    "forensic_batch_adapter",
    "stage1_collate",
    "build_dataloader",
    "decode_frames",
    "VideoDecodeError",
    # models
    "Stage1Model",
    "STAGE1_MODEL_NAMES",
    "build_stage1_model",
    "model_input_kind",
    # training / evaluation
    "TrainConfig",
    "TrainingOutcome",
    "Stage1Trainer",
    "AggregationConfig",
    "EvaluationResult",
    "Stage1Evaluator",
    "PREDICTION_COLUMNS",
    "per_class_f1",
    "search_best_threshold",
    "aggregate_unit_predictions",
    "evaluate_predictions",
    "save_predictions",
    "load_predictions",
    # fusion / inference
    "FusionResult",
    "load_prediction_tables",
    "compare_models",
    "prediction_correlation",
    "fuse_probabilities",
    "search_late_fusion",
    "search_all_combinations",
    "fused_predictions",
    "LoadedStage1Model",
    "load_stage1_model",
    "manifest_from_videos",
    "predict_stage1_proba",
    "predict_stage1",
]

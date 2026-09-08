# Stage 1 notebooks

Stage 1 classifies a video as `ORIGINAL` or `RERECORDED`. These notebooks only
orchestrate: every dataset, model, trainer and evaluator lives in
`src/blackbox_detection/stage1/`, and configuration lives in
`configs/stage1/`. Run each notebook top to bottom.

## Order

| notebook | phase | what it does |
| --- | --- | --- |
| `00_prepare_dlc2021.ipynb` | 0 | scan DLC-2021 `or`/`re`, build the common manifest, report broken videos and shortcut-risk distributions, create the fixed video-level split |
| `01_train_videomaev2_b.ipynb` | 1 | V1 VideoMAEv2-B video branch |
| `02_train_forensic.ipynb` | 2-6 | F1..F5 forensic branches; change `MODEL_NAME` and re-run |
| `03_train_vjepa2_1_b.ipynb` | 7 | V2 V-JEPA 2.1-B video branch, after 01 is validated |
| `04_compare_and_fuse.ipynb` | 8 | compare all models and search the late fusion, from saved predictions only |

Forensic execution order inside notebook 02 (implementation numbering and
experiment order differ on purpose - the screen-recapture-specific branches come
first):

```
bayar_resnet18 -> chromaticity -> frequency -> lcdf -> cdc
     F1               F3            F4          F5     F2
```

## Before the first run

```bash
pip install -e .
```

Then set the machine-specific paths:

* `configs/stage1/dlc2021.yaml` -> `dataset.root` (DLC-2021 root containing the
  subset directories),
* `configs/stage1/vjepa2_1_b.yaml` -> `model.params.source_root` (local clone of
  `facebookresearch/vjepa2`) and `model.params.checkpoint_path` (local copy of
  `vjepa2_1_vitb_dist_vitG_384.pt`). V-JEPA 2.1 is not supported by
  `transformers`, so both are required.

## Invariants these notebooks rely on

* The split is created **once** in notebook 00 and only loaded afterwards, so
  every model is scored on the same validation videos and predictions stay
  fusable.
* Splitting happens at **video level before any frame or patch extraction**.
  Frames or patches of one video never straddle train and validation.
* Resolution, FPS, duration and codec are diagnostics only, never classifier
  inputs.
* Scoring is at video level with
  `blackbox_detection.utils.metrics.stage1_score`; the decision threshold is
  searched on validation Macro-F1 rather than fixed at 0.5.
* Forensic patches are cropped from native-resolution frames. Nothing resizes a
  frame before cropping.

## Outputs

```
outputs/stage1/
├── manifests/dlc2021.csv
├── splits/dlc2021_seed42.csv
├── <model_name>/            # one per experiment
│   ├── best.pt              # + epoch, model_name, val Macro-F1, best threshold, model config
│   ├── latest.pt
│   ├── val_predictions.csv  # video_id, label, prob_original, prob_rerecorded, prediction, dataset
│   ├── history.csv
│   ├── summary.json
│   └── train.log
└── comparison/              # notebook 04
```

`outputs/` is git-ignored; `val_predictions.csv` is the artefact that makes
threshold tuning, model comparison, prediction-correlation analysis and late
fusion possible without retraining.

## Interpreting DLC-2021 scores

DLC-2021 is real screen-recapture data but document centric, while the DACON
target domain is dashcam video. A high Macro-F1 here can mean the model learned
document layout, text, borders, source resolution, FPS or one specific
display/camera pair. Do not discard a lower-scoring model on this number alone:
record Macro-F1, class-wise F1, the optimal threshold, overfitting behaviour,
VAL-A versus the controlled VAL-B subset, and prediction diversity, and re-decide
once the paired CCD re-recordings allow a driving-domain validation.

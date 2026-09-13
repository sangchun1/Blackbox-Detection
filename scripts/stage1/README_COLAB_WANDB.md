# Stage 1 Colab notebooks — actual DATASET CSV layout

These notebooks match the supplied Drive layout:

```text
/content/drive/MyDrive/Blackbox-Detection/
├── wandb_key.txt
├── DATASET/
│   ├── DLC-2021/
│   │   ├── dlc_split.csv
│   │   ├── dlc-2021_or.csv
│   │   ├── dlc-2021_re.csv
│   │   ├── or/
│   │   │   └── clips/
│   │   └── re/
│   │       └── clips/
│   └── CCD/
│       ├── ccd_split.csv
│       └── videos/
│           ├── Crash-1500/
│           └── Normal/
└── outputs/
    └── stage1/
```

There is **no `DATASET/stage1_splits/` assumption**.

## Which CSVs are used

Training notebooks use only:

- `DATASET/DLC-2021/dlc_split.csv`
- `DATASET/CCD/ccd_split.csv`

`dlc-2021_or.csv` and `dlc-2021_re.csv` are deliberately not used as the
fixed Stage 1 split.

The supplied `dlc_split.csv` has 690 DLC OR/RE rows with its own
`train` / `val` / `test` column. It does not contain video paths, so the
notebook indexes real video files recursively under `or/clips` and `re/clips`
and resolves each `clip_id`.

The supplied `ccd_split.csv` has 4,500 rows with `Crash` / `Normal` scene
classes and video paths. Until physical re-recordings exist, both scene classes
are converted to Stage 1 `ORIGINAL`.

## A-series vs B-series

At the top of notebook 01/02/03:

```python
TRAIN_DATA_MODE = "DLC"
```

runs the DLC-only A-series.

For the current CCD hard-negative ablation:

```python
TRAIN_DATA_MODE = "DLC_CCD_OR"
CCD_TRAIN_MAX = 200
```

uses a deterministic 200-video CCD train subset while preserving Crash:Normal
proportions. Set `CCD_TRAIN_MAX = None` only when you intentionally want all
3,150 CCD train videos.

Validation remains DLC OR/RE while CCD has no re-recorded pairs.

Outputs are separated automatically:

```text
outputs/stage1/dlc/<model>/
outputs/stage1/dlc_ccd_or_200/<model>/
outputs/stage1/dlc_ccd_or_all/<model>/
```

so A/B experiments do not overwrite each other.

Notebook 04 uses `RESULT_VARIANT` to select which family to compare/fuse.

## W&B

All notebooks read the key from:

```text
/content/drive/MyDrive/Blackbox-Detection/wandb_key.txt
```

and log to the project:

```text
blackbox-stage1
```

The API key is read but never printed.

## Recommended order

1. `01_train_videomaev2_b.ipynb`
2. `02_train_forensic.ipynb`
   - `bayar_resnet18`
   - `chromaticity`
   - `frequency`
   - `lcdf`
   - `cdc`
3. `03_train_vjepa2_1_b.ipynb`
4. `04_compare_and_fuse.ipynb`

First run A-series with `TRAIN_DATA_MODE="DLC"`. Then rerun selected strong
models with `TRAIN_DATA_MODE="DLC_CCD_OR"` to measure the effect of CCD
ORIGINAL hard negatives.

# Stage 1 notebooks

Stage 1 classifies each video as `ORIGINAL` or `RERECORDED`.

The repository code and the large dataset are intentionally separated:

```text
/content/Blackbox-Detection/                    # GitHub clone / code
/content/drive/MyDrive/Blackbox-Detection/
└── DATASET/                                    # Google Drive data only
    ├── DLC-2021/
    ├── CCD/
    │   └── ccd_split.csv                       # NOT a Stage 1 OR/RE split
    └── stage1_splits/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

`Blackbox_swshin` is not used.

## Notebook order

| notebook | purpose |
|---|---|
| `01_train_videomaev2_b.ipynb` | V1 VideoMAEv2-B video branch |
| `02_train_forensic.ipynb` | F1/F3/F4/F5/F2 forensic experiments |
| `03_train_vjepa2_1_b.ipynb` | V2 V-JEPA 2.1-B video branch |
| `04_compare_and_fuse.ipynb` | compare saved validation predictions and late-fuse models |

There is no Stage 1 data-preparation notebook. The team provides fixed `train.csv`, `val.csv`, and `test.csv`.

## CSV convention

Minimum required columns:

```csv
video_path,label
DLC-2021/or/.../001.mp4,ORIGINAL
DLC-2021/re/.../002.mp4,RERECORDED
```

Recommended optional columns are `video_id`, `dataset`, and (once paired CCD re-recordings exist) `source_video_id`.
If `video_id` or `dataset` is missing, notebooks fill them deterministically.

`video_path` should normally be relative to `DATASET/`. Absolute paths are also accepted.
The notebooks do not use `CCD/ccd_split.csv` as a Stage 1 label file because CCD Crash/Normal labels are not ORIGINAL/RERECORDED labels.

## Colab workflow

First clone/check out this repository under `/content/Blackbox-Detection` and install it:

```bash
git clone -b stage1-sangchun https://github.com/sangchun1/Blackbox-Detection.git /content/Blackbox-Detection
cd /content/Blackbox-Detection
pip install -e .
```

The notebooks then mount Google Drive and use only:

```text
/content/drive/MyDrive/Blackbox-Detection/DATASET
```

If the Drive directory changes, edit only `DATASET_ROOT` in the notebook path cell.

## Split usage

- `train.csv`: gradient updates
- `val.csv`: best epoch, threshold tuning, model comparison, late-fusion weights
- `test.csv`: final internal holdout only; training notebooks intentionally do not load it

All branches must use the exact same train/validation CSVs so that their saved `val_predictions.csv` files are directly comparable and fusable.

## Forensic order

```text
bayar_resnet18 -> chromaticity -> frequency -> lcdf -> cdc
      F1              F3             F4          F5     F2
```

Forensic patches are cropped from native-resolution decoded frames before any resize, and final scoring is video-level Macro-F1.

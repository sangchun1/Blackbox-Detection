# Stage 1 notebooks

Stage 1 classifies each video as `ORIGINAL` or `RERECORDED`.

## Runtime layout

The code repository and Google Drive dataset stay separate:

```text
/content/Blackbox-Detection/                    # GitHub clone / code

/content/drive/MyDrive/Blackbox-Detection/
└── DATASET/
    ├── DLC-2021/
    ├── CCD/
    │   └── videos/
    │       ├── Crash-1500/
    │       └── Normal/
    └── stage1_splits/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

Only `DATASET/` is used from Drive. `Blackbox_swshin` is ignored.

## Current CCD policy

The currently available CCD Crash/Normal videos are **not re-recorded**.

For Stage 1:
- DLC-2021 `or` -> `ORIGINAL`
- DLC-2021 `re` -> `RERECORDED`
- CCD Crash videos -> `ORIGINAL`
- CCD Normal videos -> `ORIGINAL`

`Crash` / `Normal` are scene labels, not Stage 1 labels. If needed, keep them in
an optional `scene_type` column.

CCD can be included in `train.csv` as driving-domain ORIGINAL hard negatives.
The notebooks use `train.csv` exactly as supplied; they do not silently resample
CCD. Therefore the data team controls the DLC:CCD ratio in the CSV.

Until physical CCD re-recordings exist, model selection and threshold tuning use
only the DLC-2021 portion of `val.csv`. This prevents ORIGINAL-only CCD from
distorting the primary Macro-F1. The full validation CSV is still checked for
train/val leakage and its composition is printed.

## CSV convention

Minimum columns:

```csv
video_path,label
DLC-2021/or/.../001.mp4,ORIGINAL
DLC-2021/re/.../002.mp4,RERECORDED
CCD/videos/Crash-1500/.../C_001.mp4,ORIGINAL
CCD/videos/Normal/.../N_001.mp4,ORIGINAL
```

Recommended optional columns:
- `video_id`
- `dataset` (`dlc2021` or `ccd`)
- `scene_type` (`crash`, `normal`, `document`)
- later, `source_video_id` for paired CCD ORIGINAL/RERECORDED leakage control

`video_path` should be relative to `DATASET/`. Absolute paths are also accepted.

## Notebook order

| notebook | purpose |
|---|---|
| `01_train_videomaev2_b.ipynb` | V1 VideoMAEv2-B |
| `02_train_forensic.ipynb` | F1/F3/F4/F5/F2 forensic experiments |
| `03_train_vjepa2_1_b.ipynb` | V2 V-JEPA 2.1-B |
| `04_compare_and_fuse.ipynb` | validation-prediction comparison and late fusion |

There is no data-preparation/split notebook. The data team owns the fixed
`train.csv`, `val.csv`, and `test.csv`.

## Split usage

- `train.csv`: gradient updates; may contain DLC OR/RE + current CCD ORIGINAL
- `val.csv`: full file is leakage-checked, but DLC rows are the current primary validation
- `test.csv`: final internal holdout only; training notebooks do not load it

When physical CCD re-recordings are ready, this policy should be revised so CCD
contains both ORIGINAL and RERECORDED pairs, with each source pair kept in the
same split.

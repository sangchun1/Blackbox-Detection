# Stage 1 path correction

Actual dataset layout used:

```text
DATASET/
├── DLC-2021/
│   ├── dlc_split.csv
│   ├── or/
│   │   ├── clips/
│   │   │   └── annotations/   # JSON only
│   │   └── clips_video/
│   │       └── <document folders>/...video files
│   └── re/
│       ├── clips/
│       │   └── annotations/   # JSON only
│       └── clips_video/
│           └── <document folders>/...video files
└── CCD/
    ├── ccd_split.csv
    └── videos/
        ├── Crash-1500/
        └── Normal/
```

01–03 now index DLC videos from `clips_video`, not `clips`.
The CCD loader is unchanged and continues to use `ccd_split.csv` video paths.
All Colab environment, W&B, experiment-variant, and persistent-output logic is unchanged.

# Stage 1 Colab notebooks — environment-safe version

This bundle keeps the actual dataset layout and W&B setup from the previous
version, and fixes the Colab scientific-stack corruption issue.

## Why `numpy.rec` failed

The previous setup cell imported NumPy/Pandas/Torch and then ran:

```bash
pip install -e .
```

The repository's `pyproject.toml` pins NumPy/SciPy/PyTorch. On a live Colab
kernel, pip can replace binary packages after the kernel has already loaded a
different version. The current process can then contain a mixture of old loaded
modules and newly installed files, producing errors such as:

```text
ModuleNotFoundError: No module named 'numpy.rec'
```

## New setup policy

The new notebooks:

1. mount Drive,
2. clone/update `stage1-sangchun`,
3. install only Stage 1 extra packages,
4. install this repository with `pip install --no-deps -e ...`,
5. only then import NumPy/SciPy/PyTorch,
6. run an explicit SciPy health check,
7. log in to W&B from `MyDrive/Blackbox-Detection/wandb_key.txt`.

This intentionally leaves Colab's preinstalled NumPy/SciPy/PyTorch stack alone.

## Important for the CURRENT broken Colab session

If you already executed the previous notebook and now see `numpy.rec`:

**Runtime -> Restart session**

Then open/run one of these corrected notebooks from the top.

A restart is required once because the old runtime already has a broken/mixed
NumPy state in memory. You do not need to disconnect/delete the whole runtime.

## Dataset layout

Unchanged:

```text
MyDrive/Blackbox-Detection/
├── wandb_key.txt
└── DATASET/
    ├── DLC-2021/
    │   └── dlc_split.csv
    └── CCD/
        └── ccd_split.csv
```

The notebooks do not expect `DATASET/stage1_splits/`.

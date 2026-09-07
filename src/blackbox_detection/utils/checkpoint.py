"""Checkpoint utilities for training and inference.

The checkpoint format stores enough state to resume training as faithfully as
possible, including model/optimizer/scheduler/scaler states and RNG states.

Only load checkpoints that you trust. ``torch.load(..., weights_only=False)``
is used because training checkpoints contain Python/NumPy RNG state in addition
to tensors.
"""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 1


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped by DataParallel/DDP-like wrappers."""
    return model.module if hasattr(model, "module") else model


def _get_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and CUDA RNG states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG states captured by :func:`_get_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch.set_rng_state(state["torch"])

    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        current_device_count = torch.cuda.device_count()

        # torch.cuda.set_rng_state_all expects one state per visible CUDA device.
        if len(cuda_states) == current_device_count:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            # A checkpoint may be resumed on a machine with a different number
            # of visible GPUs. Restore as many matching devices as possible.
            for device_idx, cuda_state in enumerate(cuda_states[:current_device_count]):
                torch.cuda.set_rng_state(cuda_state, device=device_idx)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Atomically save an object to reduce the chance of a corrupted checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    global_step: int | None = None,
    best_score: float | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    save_rng_state: bool = True,
) -> Path:
    """Save a complete training checkpoint.

    Parameters
    ----------
    path:
        Output checkpoint path, e.g. ``outputs/run_001/latest.pt``.
    model:
        Model to save. DataParallel/DDP wrappers are automatically unwrapped.
    epoch:
        Current epoch index or completed epoch number, according to the caller's
        convention. Keep this convention consistent across the project.
    optimizer, scheduler, scaler:
        Optional training states required for exact resume.
    global_step:
        Optional optimizer/global step.
    best_score:
        Best validation score observed so far.
    config:
        Experiment configuration.
    extra:
        Additional metadata to persist.
    save_rng_state:
        Whether to save Python/NumPy/PyTorch/CUDA RNG states.

    Returns
    -------
    pathlib.Path
        Saved checkpoint path.
    """
    checkpoint_path = Path(path)

    checkpoint: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": global_step,
        "best_score": best_score,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": dict(config) if config is not None else None,
        "extra": dict(extra) if extra is not None else None,
        "rng_state": _get_rng_state() if save_rng_state else None,
    }

    _atomic_torch_save(checkpoint, checkpoint_path)
    return checkpoint_path


def _normalize_state_dict_keys(
    state_dict: Mapping[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """Handle checkpoints saved with or without a ``module.`` prefix.

    This makes loading robust across plain modules and DataParallel/DDP-style
    checkpoints from external code.
    """
    state_dict = dict(state_dict)
    if not state_dict:
        return state_dict

    checkpoint_has_module = all(key.startswith("module.") for key in state_dict)
    model_keys = list(_unwrap_model(model).state_dict().keys())
    model_has_module = bool(model_keys) and all(
        key.startswith("module.") for key in model_keys
    )

    if checkpoint_has_module and not model_has_module:
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    if not checkpoint_has_module and model_has_module:
        return {f"module.{key}": value for key, value in state_dict.items()}

    return state_dict


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    strict: bool = True,
    restore_rng_state: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore all training states.

    Parameters
    ----------
    path:
        Checkpoint path.
    model:
        Model receiving the saved weights.
    optimizer, scheduler, scaler:
        Pass these when resuming training. Leave them as ``None`` for inference.
    map_location:
        Passed to ``torch.load``. ``"cpu"`` is the safest default.
    strict:
        Passed to ``model.load_state_dict``.
    restore_rng_state:
        Restore Python/NumPy/PyTorch/CUDA RNG states when available.

    Returns
    -------
    dict
        Metadata useful to continue training. Contains ``epoch``,
        ``global_step``, ``best_score``, ``config``, ``extra``, and
        missing/unexpected model keys.

    Notes
    -----
    This function uses ``weights_only=False`` because project checkpoints store
    optimizer state and Python/NumPy RNG state. Only load trusted checkpoints.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint mapping, got {type(checkpoint).__name__}."
        )

    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'model' state_dict. "
            "Use load_model_weights() for a standalone state_dict."
        )

    target_model = _unwrap_model(model)
    model_state = _normalize_state_dict_keys(checkpoint["model"], target_model)

    incompatible = target_model.load_state_dict(
        model_state,
        strict=strict,
    )

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    if restore_rng_state and checkpoint.get("rng_state") is not None:
        _set_rng_state(checkpoint["rng_state"])

    return {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_score": checkpoint.get("best_score"),
        "config": checkpoint.get("config"),
        "extra": checkpoint.get("extra"),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_model_weights(
    path: str | Path,
    *,
    model: nn.Module,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    strict: bool = True,
    state_dict_key: str | None = None,
) -> dict[str, list[str]]:
    """Load model weights from either a project checkpoint or a raw state_dict.

    Useful for inference, fine-tuning, and loading external pretrained weights.

    Parameters
    ----------
    path:
        Path to a checkpoint or raw PyTorch state_dict.
    model:
        Target model.
    map_location:
        Passed to ``torch.load``.
    strict:
        Passed to ``model.load_state_dict``.
    state_dict_key:
        Optional explicit key when an external checkpoint stores weights under
        a custom key such as ``"state_dict"`` or ``"model_state_dict"``.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Weights not found: {checkpoint_path}")

    obj = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if state_dict_key is not None:
        if not isinstance(obj, Mapping) or state_dict_key not in obj:
            raise KeyError(
                f"Key {state_dict_key!r} not found in checkpoint: {checkpoint_path}"
            )
        state_dict = obj[state_dict_key]
    elif isinstance(obj, Mapping) and "model" in obj:
        state_dict = obj["model"]
    elif isinstance(obj, Mapping) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, Mapping) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    else:
        state_dict = obj

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Could not identify a model state_dict in the supplied checkpoint."
        )

    target_model = _unwrap_model(model)
    state_dict = _normalize_state_dict_keys(state_dict, target_model)

    incompatible = target_model.load_state_dict(
        state_dict,
        strict=strict,
    )

    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def save_latest_and_best(
    output_dir: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    global_step: int | None = None,
    best_score: float | None = None,
    is_best: bool = False,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    latest_filename: str = "latest.pt",
    best_filename: str = "best.pt",
    save_rng_state: bool = True,
) -> dict[str, Path | None]:
    """Save ``latest.pt`` every time and ``best.pt`` when ``is_best`` is true."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "model": model,
        "epoch": epoch,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "global_step": global_step,
        "best_score": best_score,
        "config": config,
        "extra": extra,
        "save_rng_state": save_rng_state,
    }

    latest_path = save_checkpoint(
        output_path / latest_filename,
        **common_kwargs,
    )

    best_path: Path | None = None
    if is_best:
        best_path = save_checkpoint(
            output_path / best_filename,
            **common_kwargs,
        )

    return {
        "latest": latest_path,
        "best": best_path,
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_weights",
    "save_latest_and_best",
]

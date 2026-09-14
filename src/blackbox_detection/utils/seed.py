"""Reproducibility utilities for blackbox_detection.

Use ``seed_everything`` once near the start of a notebook/script, before the
first CUDA operation, and pass ``seed_worker`` + a seeded ``torch.Generator``
to every DataLoader.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

import cv2


DEFAULT_SEED = 42


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    deterministic: bool = True,
    strict: bool = True,
) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and OpenCV RNGs.

    Args:
        seed: Global random seed.
        deterministic: Enable deterministic PyTorch/CUDA behavior.
        strict: If True, raise when PyTorch encounters a known
            non-deterministic operation. If False, PyTorch only warns.

    Notes:
        - Call this before the first CUDA operation. ``CUBLAS_WORKSPACE_CONFIG``
          must be set before cuBLAS is initialized for deterministic CUDA ops.
        - Setting ``PYTHONHASHSEED`` at runtime does not retroactively change
          the hash seed of the already-running Python interpreter. It is still
          exported for child processes. In normal training code, the RNGs below
          are the important sources of randomness.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Required by PyTorch for deterministic cuBLAS behavior on CUDA >= 10.2.
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cv2 is not None:
        cv2.setRNGSeed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Reduce hardware-dependent numerical differences.
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

        torch.use_deterministic_algorithms(True, warn_only=not strict)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """Seed every DataLoader worker deterministically.

    Keep the worker seed inside signed 32-bit range because NumPy/OpenCV
    bindings in some Colab environments reject larger integer seeds.
    """
    del worker_id  # worker id is already encoded in torch.initial_seed()

    # 0 ~ 2^31-1: safe for Python, NumPy, PyTorch, and OpenCV.
    worker_seed = int(torch.initial_seed() & 0x7FFFFFFF)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    if cv2 is not None:
        cv2.setRNGSeed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Create a deterministic generator for DataLoader shuffle/base seeds."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def dataloader_seed_kwargs(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Return the seed-related kwargs that should be passed to every DataLoader.

    Example:
        >>> train_loader = DataLoader(
        ...     train_dataset,
        ...     shuffle=True,
        ...     num_workers=4,
        ...     **dataloader_seed_kwargs(42),
        ... )
    """
    return {
        "worker_init_fn": seed_worker,
        "generator": make_generator(seed),
    }

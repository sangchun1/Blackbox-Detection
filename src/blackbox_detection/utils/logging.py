"""Logging utilities for training and experiments.

Features
--------
- Console logging that does not break tqdm progress bars.
- Optional file logging.
- tqdm progress bars with explicit elapsed/remaining time (ETA).
- Weights & Biases initialization and metric logging.
- Safe repeated initialization in Jupyter/Colab without duplicate handlers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import wandb
from tqdm.auto import tqdm

T = TypeVar("T")

_DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm without breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str = "blackbox_detection",
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    file_mode: str = "a",
) -> logging.Logger:
    """Create a logger suitable for notebooks and tqdm loops.

    Repeated calls with the same logger name replace old handlers, which avoids
    duplicated log messages when a Jupyter/Colab cell is executed multiple times.

    Parameters
    ----------
    name:
        Logger name.
    level:
        Logging level, e.g. logging.INFO or "INFO".
    log_file:
        Optional file path. Parent directories are created automatically.
    file_mode:
        File opening mode. Usually "a" (append) or "w" (overwrite).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Jupyter/Colab cells may be re-run many times. Remove stale handlers first.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        fmt=_DEFAULT_LOG_FORMAT,
        datefmt=_DEFAULT_DATE_FORMAT,
    )

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            mode=file_mode,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = True,
    dynamic_ncols: bool = True,
    mininterval: float = 0.2,
    **kwargs: Any,
) -> tqdm:
    """Create a tqdm progress bar with explicit elapsed time and ETA.

    Examples
    --------
    >>> progress = create_progress_bar(train_loader, desc="Train 1/10")
    >>> for batch in progress:
    ...     ...
    ...     progress.set_postfix(loss=f"{loss:.4f}")
    """
    kwargs.setdefault("bar_format", _DEFAULT_BAR_FORMAT)

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=dynamic_ncols,
        mininterval=mininterval,
        **kwargs,
    )


def init_wandb(
    *,
    project: str,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    entity: str | None = None,
    group: str | None = None,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    run_id: str | None = None,
    resume: str | bool | None = "allow",
    mode: str | None = None,
    directory: str | Path | None = None,
    reinit: bool | str | None = True,
    **kwargs: Any,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a Weights & Biases run.

    Parameters are intentionally close to ``wandb.init`` so experiments can
    use one shared project convention without hiding W&B functionality.

    ``mode="disabled"`` can be used when W&B should be temporarily skipped.
    """
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=project,
        name=name,
        config=dict(config) if config is not None else None,
        entity=entity,
        group=group,
        tags=list(tags) if tags is not None else None,
        notes=notes,
        id=run_id,
        resume=resume,
        mode=mode,
        dir=str(directory) if directory is not None else None,
        reinit=reinit,
        **kwargs,
    )

    if run is None:
        raise RuntimeError("wandb.init() did not return a run.")

    return run


def log_metrics(
    metrics: Mapping[str, Any],
    *,
    step: int | None = None,
    commit: bool = True,
) -> None:
    """Log metrics to the currently active W&B run.

    Raises an error instead of silently dropping metrics when W&B has not been
    initialized. This helps catch experiment-logging mistakes early.
    """
    if wandb.run is None:
        raise RuntimeError(
            "No active W&B run. Call init_wandb() before log_metrics()."
        )

    wandb.log(dict(metrics), step=step, commit=commit)


def finish_wandb(*, exit_code: int | None = None) -> None:
    """Finish the active W&B run, if one exists."""
    if wandb.run is not None:
        wandb.finish(exit_code=exit_code)


__all__ = [
    "TqdmLoggingHandler",
    "setup_logger",
    "create_progress_bar",
    "init_wandb",
    "log_metrics",
    "finish_wandb",
]

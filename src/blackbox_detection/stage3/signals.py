from __future__ import annotations

import numpy as np


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def clean_timeseries(times, values):
    """Sort, remove invalid rows and collapse duplicate timestamps."""
    t = _as_float_array(times).reshape(-1)
    v = _as_float_array(values)
    if v.shape[0] != t.shape[0]:
        raise ValueError(f"time/value length mismatch: {t.shape[0]} vs {v.shape[0]}")

    finite_t = np.isfinite(t)
    if v.ndim == 1:
        finite_v = np.isfinite(v)
    else:
        finite_v = np.isfinite(v).all(axis=tuple(range(1, v.ndim)))
    keep = finite_t & finite_v
    t, v = t[keep], v[keep]
    if len(t) == 0:
        return t, v

    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]

    # Keep the last sample for duplicate timestamps.
    _, reverse_idx = np.unique(t[::-1], return_index=True)
    idx = np.sort(len(t) - 1 - reverse_idx)
    return t[idx], v[idx]


def interpolate_signal(query_times, signal_times, signal_values):
    """Linear interpolation without pretending extrapolated samples are valid.

    Returns `(values, valid_mask)`. Outside the source time interval values are
    edge-filled by numpy, but `valid_mask` is False and downstream loss must mask them.
    """
    q = _as_float_array(query_times).reshape(-1)
    t, v = clean_timeseries(signal_times, signal_values)
    if len(t) < 2:
        shape = (len(q),) + tuple(v.shape[1:])
        return np.full(shape, np.nan, dtype=np.float64), np.zeros(len(q), dtype=bool)

    valid = (q >= t[0]) & (q <= t[-1])
    if v.ndim == 1:
        out = np.interp(q, t, v)
    else:
        flat = v.reshape(v.shape[0], -1)
        cols = [np.interp(q, t, flat[:, i]) for i in range(flat.shape[1])]
        out = np.stack(cols, axis=-1).reshape((len(q),) + v.shape[1:])
    return out, valid


def smooth_1d(values: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    x = _as_float_array(values).reshape(-1)
    if len(x) < 5:
        return x.copy()
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(x, window_length=window, polyorder=min(polyorder, window - 2), mode="interp")
    except Exception:
        kernel = np.ones(window, dtype=np.float64) / window
        pad = window // 2
        return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")


def derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    x = _as_float_array(values).reshape(-1)
    t = _as_float_array(times).reshape(-1)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, t, edge_order=1)


def acceleration_from_speed(speed_mps: np.ndarray, times: np.ndarray, smooth_window: int = 11):
    smoothed = smooth_1d(speed_mps, window=smooth_window)
    return derivative(smoothed, times), smoothed

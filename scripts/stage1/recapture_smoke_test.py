from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from blackbox_detection.stage1.recapture import RecaptureSimConfig, RecaptureSimV1


def main() -> None:
    parser = argparse.ArgumentParser(description="RecaptureSimV1 smoke test")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]

    # Smooth synthetic image content is more representative than white noise for a transform smoke test.
    yy = torch.linspace(0, 1, args.size)[None, None, :, None]
    xx = torch.linspace(0, 1, args.size)[None, None, None, :]
    tt = torch.linspace(0, 1, args.frames)[None, :, None, None]
    rgb = torch.cat(
        [
            (0.25 + 0.55 * xx + 0.10 * tt).expand(1, args.frames, args.size, args.size),
            (0.20 + 0.45 * yy + 0.12 * tt).expand(1, args.frames, args.size, args.size),
            (0.30 + 0.25 * xx + 0.25 * yy).expand(1, args.frames, args.size, args.size),
        ],
        dim=0,
    ).clamp(0, 1)
    pixels = (rgb - mean) / std

    sim = RecaptureSimV1(RecaptureSimConfig())

    outputs = {}
    for strength in ("weak", "medium", "strong"):
        started = time.perf_counter()
        out = sim(
            pixels,
            apply_common=True,
            apply_recapture=True,
            strength=strength,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - started
        assert out.shape == pixels.shape
        assert torch.isfinite(out).all()
        outputs[strength] = out
        pixel_out = (out * std + mean).clamp(0, 1)
        mad = float((pixel_out - rgb).abs().mean())
        print(f"{strength:6s} | shape={tuple(out.shape)} | MAD={mad:.6f} | {elapsed:.3f}s")

    # Same seed must be numerically reproducible. Some CPU interpolation kernels
    # can differ by a few ulps across threaded calls, so use a tight tolerance.
    a = sim(pixels, apply_common=True, apply_recapture=True, strength="medium", seed=args.seed)
    b = sim(pixels, apply_common=True, apply_recapture=True, strength="medium", seed=args.seed)
    if not torch.allclose(a, b, atol=1e-5, rtol=0.0):
        raise RuntimeError("same seed did not produce reproducible output")

    # Different seeds should normally differ.
    c = sim(pixels, apply_common=True, apply_recapture=True, strength="medium", seed=args.seed + 1)
    if torch.equal(a, c):
        raise RuntimeError("different seeds unexpectedly produced identical output")

    print("[PASS] finite outputs, expected shapes, reproducible same-seed validation, variable different-seed augmentation")


if __name__ == "__main__":
    main()

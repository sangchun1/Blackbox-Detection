from __future__ import annotations

from pathlib import Path
import torch


def _clean_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        out[key] = value
    return out


def load_vjepa21_base_encoder(
    repo_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    num_frames: int = 16,
    out_layers=(2, 5, 8, 11),
    freeze: bool = True,
):
    """Build V-JEPA 2.1 ViT-B locally and load the official EMA encoder weights.

    This intentionally avoids the current PyTorch-Hub pretrained URL path.
    """
    repo_dir = Path(repo_dir)
    checkpoint_path = Path(checkpoint_path)
    if not repo_dir.exists():
        raise FileNotFoundError(f"V-JEPA repo not found: {repo_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    encoder, predictor = torch.hub.load(
        str(repo_dir),
        "vjepa2_1_vit_base_384",
        source="local",
        pretrained=False,
        num_frames=int(num_frames),
        out_layers=list(out_layers),
    )
    del predictor

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_encoder") or checkpoint.get("target_encoder") or checkpoint.get("encoder")
    if state is None:
        raise KeyError(f"no encoder weights in checkpoint keys: {list(checkpoint)}")
    result = encoder.load_state_dict(_clean_state_dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"V-JEPA state mismatch: {result}")

    if freeze:
        encoder.requires_grad_(False)
        encoder.eval()
    return encoder

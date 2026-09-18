from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import CAN_TARGETS


def masked_smooth_l1(pred, target, valid, beta: float = 0.5):
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid], beta=beta, reduction="mean")


def can_multitask_loss(outputs: dict, target: torch.Tensor, valid: torch.Tensor, weights: dict | None = None):
    weights = weights or {}
    total = target.new_tensor(0.0)
    parts = {}
    for i, name in enumerate(CAN_TARGETS):
        loss = masked_smooth_l1(outputs[name], target[..., i], valid[..., i])
        w = float(weights.get(name, 1.0))
        total = total + w * loss
        parts[name] = float(loss.detach().cpu())
    parts["total"] = float(total.detach().cpu())
    return total, parts

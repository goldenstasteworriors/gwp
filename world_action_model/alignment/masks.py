"""Losses for partially observed counterfactual robot masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Combine BCE and soft Dice while excluding unknown camera/view regions."""
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError(
            f"Mask logits, target, and valid shapes must match, got "
            f"{tuple(logits.shape)}, {tuple(target.shape)}, {tuple(valid.shape)}"
        )
    if logits.ndim != 4:
        raise ValueError(f"Expected [B, T, H, W] masks, got {tuple(logits.shape)}")
    valid = valid.to(dtype=logits.dtype)
    target = target.to(dtype=logits.dtype)
    valid_count = valid.sum()
    if valid_count <= 0:
        raise ValueError("Counterfactual mask validity region is empty")

    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum()
    bce = bce / valid_count

    probability = logits.sigmoid()
    reduce_dims = (-2, -1)
    intersection = (probability * target * valid).sum(dim=reduce_dims)
    denominator = ((probability + target) * valid).sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    frame_valid = valid.sum(dim=reduce_dims) > 0
    dice_loss = (1.0 - dice)[frame_valid].mean()
    return float(bce_weight) * bce + float(dice_weight) * dice_loss

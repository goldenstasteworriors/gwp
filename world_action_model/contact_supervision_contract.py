"""Lightweight fail-closed supervision checks for spec-only contact routes."""

from __future__ import annotations

import torch


def validate_positive_contact_supervision(target, mask, confidence=None) -> int:
    """Require at least one finite, genuinely positive masked target element."""

    if target is None or mask is None:
        raise ValueError("spec-only contact routes require target and explicit loss mask")
    target = torch.as_tensor(target)
    mask = torch.as_tensor(mask, device=target.device)
    if not torch.isfinite(target).all() or not torch.isfinite(mask).all():
        raise ValueError("contact target/mask must be finite")
    if bool((mask < 0).any()):
        raise ValueError("contact loss mask must be nonnegative")
    try:
        mask = torch.broadcast_to(mask, target.shape)
    except RuntimeError as exc:
        raise ValueError("contact loss mask cannot broadcast to target") from exc
    if confidence is None:
        confidence = torch.ones_like(target)
    else:
        confidence = torch.as_tensor(confidence, device=target.device)
        if not torch.isfinite(confidence).all() or bool((confidence < 0).any()):
            raise ValueError("contact confidence must be finite and nonnegative")
        try:
            confidence = torch.broadcast_to(confidence, target.shape)
        except RuntimeError as exc:
            raise ValueError("contact confidence cannot broadcast to target") from exc
    positive_count = int(((mask > 0) & (confidence > 0)).sum().item())
    effective_weight = (mask.to(torch.float64) * confidence.to(torch.float64)).sum()
    if positive_count <= 0 or not torch.isfinite(effective_weight) or effective_weight <= 0:
        raise ValueError("contact supervision mask has no positive effective weight")
    return positive_count

"""Lightweight fail-closed supervision checks for spec-only contact routes."""

from __future__ import annotations

import torch

from world_action_model.contact_transition_schema import (
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_TARGET_DIM,
)


CONTACT_GROUPED_SUPERVISION_RULE_VERSION = "contact_grouped_supervision_v1"


def grouped_contact_supervision_weights(mask, confidence=None) -> dict[str, torch.Tensor]:
    """Build the exact per-field weights consumed by grouped contact loss.

    Binary/tau fields are eligible independently.  A categorical field is
    eligible only when every dimension in its one-hot group has mask exactly
    one; partial category masks contribute no loss and no effective count.
    """

    mask = torch.as_tensor(mask)
    if mask.ndim < 1 or mask.shape[-1] != SEGMENT_TARGET_DIM:
        raise ValueError(
            f"contact mask must end in width {SEGMENT_TARGET_DIM}, got {tuple(mask.shape)}"
        )
    if not torch.isfinite(mask).all() or bool((mask < 0).any()):
        raise ValueError("contact loss mask must be finite and nonnegative")
    if confidence is None:
        confidence = torch.ones_like(mask)
    else:
        confidence = torch.as_tensor(confidence, device=mask.device)
        try:
            confidence = torch.broadcast_to(confidence, mask.shape)
        except RuntimeError as exc:
            raise ValueError("contact confidence cannot broadcast to loss mask") from exc
        if not torch.isfinite(confidence).all() or bool((confidence < 0).any()):
            raise ValueError("contact confidence must be finite and nonnegative")

    weights: dict[str, torch.Tensor] = {}
    for field in SEGMENT_BINARY_FIELDS + SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        weights[field] = (
            mask[..., field_slice] * confidence[..., field_slice]
        ).float()
    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        complete_group = (mask[..., field_slice] == 1).all(dim=-1)
        weights[field] = (
            complete_group.to(dtype=torch.float32)
            * confidence[..., field_slice].amin(dim=-1).float()
        )
    return weights


def _positive_grouped_weight_count(weights: dict[str, torch.Tensor]) -> int:
    return sum(int((weight > 0).sum().item()) for weight in weights.values())


def effective_contact_supervision_count(mask) -> int:
    """Count schema-valid grouped-loss items across all leading dimensions."""

    return _positive_grouped_weight_count(grouped_contact_supervision_weights(mask))


def validate_positive_contact_supervision(target, mask, confidence=None) -> int:
    """Require at least one item eligible under the grouped-loss schema."""

    if target is None or mask is None:
        raise ValueError("spec-only contact routes require target and explicit loss mask")
    target = torch.as_tensor(target)
    mask = torch.as_tensor(mask, device=target.device)
    if target.ndim < 1 or target.shape[-1] != SEGMENT_TARGET_DIM:
        raise ValueError(
            f"contact target must end in width {SEGMENT_TARGET_DIM}, got {tuple(target.shape)}"
        )
    if not torch.isfinite(target).all():
        raise ValueError("contact target must be finite")
    try:
        mask = torch.broadcast_to(mask, target.shape)
    except RuntimeError as exc:
        raise ValueError("contact loss mask cannot broadcast to target") from exc
    if confidence is not None:
        confidence = torch.as_tensor(confidence, device=target.device)
        try:
            confidence = torch.broadcast_to(confidence, target.shape)
        except RuntimeError as exc:
            raise ValueError("contact confidence cannot broadcast to target") from exc
    mask_count = effective_contact_supervision_count(mask)
    effective_count = _positive_grouped_weight_count(
        grouped_contact_supervision_weights(mask, confidence)
    )
    if mask_count <= 0 or effective_count <= 0:
        raise ValueError("contact supervision has no grouped-loss-eligible item")
    return mask_count

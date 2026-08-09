"""Schema and encoders for ordered contact-transition event sidecars.

The representation deliberately separates target values from supervision masks.
Weak annotations can therefore supervise timing and contact side without
inventing a finger site, object part, or 3-D surface normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "contact_transition_v1"

FIELD_WIDTHS = {
    "presence": 1,
    "side": 3,
    "site": 7,
    "object_role": 4,
    "part_slot": 8,
    "contact_mode": 6,
    "transition": 5,
    "object_effect": 7,
    "event_time_bin": 4,
    "duration_bin": 4,
    "local_region": 2,
    "normal_proxy": 3,
}

ENUM_VALUES = {
    "side": ("left", "right", "object_object"),
    "site": ("opposer", "primary", "aux1", "aux2", "aux3", "palm", "tool"),
    "object_role": ("target", "tool", "receptacle", "fixture"),
    "part_slot": tuple(str(index) for index in range(8)),
    "contact_mode": ("free", "stick", "slide", "roll", "impact", "unknown"),
    "transition": ("none", "onset", "hold", "switch_to_slide", "release"),
    "object_effect": ("none", "translate", "rotate", "articulate", "flow", "deform", "transfer"),
    "event_time_bin": tuple(str(index) for index in range(4)),
    "duration_bin": tuple(str(index) for index in range(4)),
}


def _build_slices() -> dict[str, slice]:
    offset = 0
    result = {}
    for name, width in FIELD_WIDTHS.items():
        result[name] = slice(offset, offset + width)
        offset += width
    return result


FIELD_SLICES = _build_slices()
TARGET_DIM = sum(FIELD_WIDTHS.values())


@dataclass(frozen=True)
class ContactEvent:
    """One ordered event in a future action window."""

    time_offset: int
    duration: int
    transition: str
    side: str | None = None
    site: str | None = None
    object_role: str | None = None
    part_slot: int | str | None = None
    contact_mode: str | None = None
    object_effect: str | None = None
    local_region: tuple[float, float] | None = None
    normal_proxy: tuple[float, float, float] | None = None
    confidence: float = 1.0


def quantize_nonnegative(value: int, scale: int, bins: int = 4) -> int:
    """Quantize a non-negative frame count into equal-width clipped bins."""

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    value = max(0, int(value))
    return min(bins - 1, (value * bins) // scale)


def _set_one_hot(
    target: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    row: int,
    field: str,
    value: int | str | None,
    weight: float,
) -> None:
    if value is None:
        return
    choices = ENUM_VALUES[field]
    value = str(value)
    if value not in choices:
        raise ValueError(f"invalid {field}={value!r}; expected one of {choices}")
    field_slice = FIELD_SLICES[field]
    target[row, field_slice] = 0.0
    target[row, field_slice.start + choices.index(value)] = 1.0
    mask[row, field_slice] = 1.0
    confidence[row, field_slice] = weight


def _set_continuous(
    target: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    row: int,
    field: str,
    value: Sequence[float] | None,
    weight: float,
) -> None:
    if value is None:
        return
    field_slice = FIELD_SLICES[field]
    array = np.asarray(value, dtype=np.float32)
    expected = FIELD_WIDTHS[field]
    if array.shape != (expected,):
        raise ValueError(f"{field} must have shape ({expected},), got {array.shape}")
    target[row, field_slice] = array
    mask[row, field_slice] = 1.0
    confidence[row, field_slice] = weight


def encode_ordered_events(
    events: Sequence[ContactEvent],
    *,
    num_queries: int,
    horizon_frames: int,
    negative_confidence: float = 0.1,
) -> Mapping[str, np.ndarray]:
    """Encode events and field-level uncertainty into fixed-size arrays."""

    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")
    if len(events) > num_queries:
        raise ValueError(f"{len(events)} events exceed num_queries={num_queries}")
    if not 0.0 <= negative_confidence <= 1.0:
        raise ValueError("negative_confidence must be in [0, 1]")

    target = np.zeros((num_queries, TARGET_DIM), dtype=np.float32)
    loss_mask = np.zeros_like(target)
    confidence = np.zeros_like(target)
    condition_mask = np.zeros((num_queries, 1), dtype=np.float32)

    # Every padded query is a known negative only in an annotated episode.
    presence_slice = FIELD_SLICES["presence"]
    loss_mask[:, presence_slice] = 1.0
    confidence[:, presence_slice] = negative_confidence

    ordered = sorted(
        events,
        key=lambda event: (
            int(event.time_offset),
            str(event.side or ""),
            str(event.site or ""),
            str(event.transition),
        ),
    )
    for row, event in enumerate(ordered):
        if not 0.0 <= event.confidence <= 1.0:
            raise ValueError(f"event confidence must be in [0, 1], got {event.confidence}")
        weight = float(event.confidence)
        target[row, presence_slice] = 1.0
        confidence[row, presence_slice] = weight
        condition_mask[row, 0] = 1.0

        _set_one_hot(target, loss_mask, confidence, row, "side", event.side, weight)
        _set_one_hot(target, loss_mask, confidence, row, "site", event.site, weight)
        _set_one_hot(target, loss_mask, confidence, row, "object_role", event.object_role, weight)
        _set_one_hot(target, loss_mask, confidence, row, "part_slot", event.part_slot, weight)
        _set_one_hot(target, loss_mask, confidence, row, "contact_mode", event.contact_mode, weight)
        _set_one_hot(target, loss_mask, confidence, row, "transition", event.transition, weight)
        _set_one_hot(target, loss_mask, confidence, row, "object_effect", event.object_effect, weight)
        _set_one_hot(
            target,
            loss_mask,
            confidence,
            row,
            "event_time_bin",
            quantize_nonnegative(event.time_offset, horizon_frames),
            weight,
        )
        _set_one_hot(
            target,
            loss_mask,
            confidence,
            row,
            "duration_bin",
            quantize_nonnegative(event.duration, horizon_frames),
            weight,
        )
        _set_continuous(
            target,
            loss_mask,
            confidence,
            row,
            "local_region",
            event.local_region,
            weight,
        )
        _set_continuous(
            target,
            loss_mask,
            confidence,
            row,
            "normal_proxy",
            event.normal_proxy,
            weight,
        )

    return {
        "contact_target": target,
        "contact_loss_mask": loss_mask,
        "contact_confidence": confidence,
        "contact_condition_mask": condition_mask,
    }

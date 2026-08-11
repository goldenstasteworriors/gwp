"""Schema and encoders for ordered contact-transition event sidecars.

The representation deliberately separates target values from supervision masks.
Weak annotations can therefore supervise timing and contact side without
inventing a finger site, object part, or 3-D surface normal.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "contact_transition_v1"
SEGMENT_SCHEMA_VERSION = "contact_segment_v1_1"

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


# v1.1 is deliberately an opt-in per-segment schema.  Keep the v1 event
# constants above as the public defaults so existing 54-D sidecars and
# checkpoints continue to load unchanged.
SEGMENT_FIELD_WIDTHS = {
    "presence": 1,
    "side": 3,
    "site": 6,
    "object_role": 4,
    "contact_mode": 4,
    "start_tau": 1,
    "end_tau": 1,
    "left_censored": 1,
    "right_censored": 1,
}

SEGMENT_ENUM_VALUES = {
    "side": ("left", "right", "object_object"),
    "site": (
        "opposition",
        "primary",
        "support",
        "palm_or_jaw",
        "tool_tip",
        "other_surface",
    ),
    "object_role": ("manipulated", "tool", "receptacle", "fixture"),
    "contact_mode": ("stick", "slide", "roll", "impact"),
}


def _build_slices_from_widths(widths: Mapping[str, int]) -> dict[str, slice]:
    offset = 0
    result = {}
    for name, width in widths.items():
        result[name] = slice(offset, offset + width)
        offset += width
    return result


SEGMENT_FIELD_SLICES = _build_slices_from_widths(SEGMENT_FIELD_WIDTHS)
SEGMENT_TARGET_DIM = sum(SEGMENT_FIELD_WIDTHS.values())

SEGMENT_BINARY_FIELDS = ("presence", "left_censored", "right_censored")
SEGMENT_CATEGORICAL_FIELDS = ("side", "site", "object_role", "contact_mode")
SEGMENT_CONTINUOUS_FIELDS = ("start_tau", "end_tau")

CONTACT_SEMANTIC_TRANSPORT_MODES = (
    "soft",
    "hard_st",
    "hard_detached",
)


EQUAL_BIT_CODEC_VERSION = "contact_equal_bit_unstructured_v1"
EQUAL_BIT_VALID_STATES = 48_673
EQUAL_BIT_NUM_BITS = 16
EQUAL_BIT_CODE_WIDTH = SEGMENT_TARGET_DIM
EQUAL_BIT_BIT_ORDER = "little_endian_lsb_first"
EQUAL_BIT_MIXING_LAYOUT = "row_major_int8_twos_complement"
HARD_SEGMENT_GRAMMAR_VERSION = "contact_segment_hard_grammar_b16_v1"
# Receipt-bound fixed Rademacher matrix.  It was generated once from the
# documented seed using SHA-256 counter blocks, checked to have row rank 16,
# and is now a literal so library/RNG changes cannot alter the codec.
EQUAL_BIT_MIXING_SEED = "contact-equal-bit-rademacher-v1"
EQUAL_BIT_MIXING_MATRIX = (
    (-1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1),
    (-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1),
    (1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1),
    (1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1),
    (1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1),
    (1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1),
    (-1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1),
    (-1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1),
    (1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1),
    (-1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1),
    (-1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1),
    (1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1),
    (1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1),
    (1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1),
    (1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1),
    (-1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1),
)


def _equal_bit_mixing_bytes() -> bytes:
    return np.asarray(EQUAL_BIT_MIXING_MATRIX, dtype=np.int8).tobytes(order="C")


EQUAL_BIT_MIXING_MATRIX_SHA256 = hashlib.sha256(
    _equal_bit_mixing_bytes()
).hexdigest()


def validated_equal_bit_mixing_matrix() -> np.ndarray:
    """Return a copy of the receipt-bound matrix after byte/rank validation."""

    matrix = np.asarray(EQUAL_BIT_MIXING_MATRIX, dtype=np.int8)
    if matrix.shape != (EQUAL_BIT_NUM_BITS, EQUAL_BIT_CODE_WIDTH):
        raise RuntimeError("equal-bit mixing matrix shape differs from its codec contract")
    if hashlib.sha256(matrix.tobytes(order="C")).hexdigest() != EQUAL_BIT_MIXING_MATRIX_SHA256:
        raise RuntimeError("equal-bit mixing matrix bytes differ from their fixed digest")
    if np.linalg.matrix_rank(matrix.astype(np.float64)) != EQUAL_BIT_NUM_BITS:
        raise RuntimeError("equal-bit mixing matrix is not full row rank")
    return matrix.copy()


def unstructured_capacity_metadata(
    K: int = EQUAL_BIT_VALID_STATES,
    Q: int = 12,
    width: int = EQUAL_BIT_CODE_WIDTH,
) -> Mapping[str, int | float | str | bool | list[list[int]]]:
    """Return the exact, machine-readable equal-bit codec receipt."""

    K, Q, width = int(K), int(Q), int(width)
    if K != EQUAL_BIT_VALID_STATES:
        raise ValueError(
            f"the v1 equal-bit codec fixes K={EQUAL_BIT_VALID_STATES}, got {K}"
        )
    if Q < 0:
        raise ValueError(f"Q must be non-negative, got {Q}")
    if width != EQUAL_BIT_CODE_WIDTH:
        raise ValueError(
            f"the v1 equal-bit codec fixes width={EQUAL_BIT_CODE_WIDTH}, got {width}"
        )
    matrix = validated_equal_bit_mixing_matrix()
    matrix_bytes = matrix.tobytes(order="C")
    matrix_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
    bits_per_query = math.log2(K)
    return {
        "codec_version": EQUAL_BIT_CODEC_VERSION,
        "finite_capacity": True,
        "valid_id_min": 0,
        "valid_id_max": K - 1,
        "invalid_id_min": K,
        "invalid_id_max": 2**EQUAL_BIT_NUM_BITS - 1,
        "invalid_id_canonical_value": 0,
        "states_per_query": K,
        "num_queries": Q,
        "bits_per_query": bits_per_query,
        "total_states": K**Q,
        "total_bits": bits_per_query * Q,
        "raw_bits": EQUAL_BIT_NUM_BITS,
        "code_width": width,
        "action_visible_bit_dimensions": EQUAL_BIT_NUM_BITS,
        "canonical_positive_zero_dimensions": width - EQUAL_BIT_NUM_BITS,
        "bit_order": EQUAL_BIT_BIT_ORDER,
        "mixing_shape": [EQUAL_BIT_NUM_BITS, EQUAL_BIT_CODE_WIDTH],
        "mixing_dtype": "int8",
        "mixing_layout": EQUAL_BIT_MIXING_LAYOUT,
        "mixing_seed_record_only": EQUAL_BIT_MIXING_SEED,
        "mixing_matrix": matrix.astype(int).tolist(),
        "mixing_matrix_bytes_hex": matrix_bytes.hex(),
        "mixing_matrix_sha256": matrix_sha256,
        "mixing_matrix_row_rank": EQUAL_BIT_NUM_BITS,
        "hard_packet_validator": "validate_unstructured_code_v1",
    }


def canonicalize_unstructured_ids(ids: np.ndarray | Sequence[int]) -> np.ndarray:
    """Canonicalize every integer outside ``[0, 48672]`` to id zero."""

    values = np.asarray(ids)
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError("equal-bit ids must use an integer dtype")
    wide = values.astype(np.int64, copy=False)
    return np.where(
        (wide >= 0) & (wide < EQUAL_BIT_VALID_STATES), wide, 0
    ).astype(np.uint16)


def unstructured_ids_to_code(ids: np.ndarray | Sequence[int]) -> np.ndarray:
    """Decode canonical ids to 16 little-endian bits followed by six ``+0``s."""

    canonical = canonicalize_unstructured_ids(ids)
    shifts = np.arange(EQUAL_BIT_NUM_BITS, dtype=np.uint16)
    bits = ((canonical[..., None] >> shifts) & 1).astype(np.float32)
    code = np.zeros(canonical.shape + (EQUAL_BIT_CODE_WIDTH,), dtype=np.float32)
    code[..., :EQUAL_BIT_NUM_BITS] = bits
    return code


def unstructured_code_to_ids(code: np.ndarray) -> np.ndarray:
    """Encode a canonical v1 code, rejecting non-binary or non-zero padding."""

    value = np.asarray(code)
    if value.ndim < 1 or value.shape[-1] != EQUAL_BIT_CODE_WIDTH:
        raise ValueError(
            f"equal-bit code must end in width {EQUAL_BIT_CODE_WIDTH}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("equal-bit code requires finite values")
    if value.dtype.kind != "f":
        raise TypeError("equal-bit code must use a floating dtype")
    if np.signbit(value[value == 0]).any():
        raise ValueError("equal-bit code requires canonical positive zero")
    if not np.logical_or(value[..., :EQUAL_BIT_NUM_BITS] == 0, value[..., :EQUAL_BIT_NUM_BITS] == 1).all():
        raise ValueError("equal-bit code prefix must contain only zero/one")
    if not (value[..., EQUAL_BIT_NUM_BITS:] == 0).all():
        raise ValueError("equal-bit code suffix must be canonical zero")
    weights = (1 << np.arange(EQUAL_BIT_NUM_BITS, dtype=np.uint32))
    raw = (value[..., :EQUAL_BIT_NUM_BITS].astype(np.uint32) * weights).sum(
        axis=-1, dtype=np.uint32
    )
    if (raw >= EQUAL_BIT_VALID_STATES).any():
        raise ValueError("equal-bit code contains a non-canonical invalid id")
    return raw.astype(np.uint16)


def validate_hard_contact_segment_code(
    code: np.ndarray,
    *,
    time_bins: int = 16,
) -> np.ndarray:
    """Validate exactly the finite hard semantic grammar emitted by transport.

    The accepted per-query domain is the single canonical absent packet plus
    every categorical product and valid time/censor state.  This is a decoder,
    not a logits re-transport check: every value must already be canonical.
    """

    value = np.asarray(code)
    if value.ndim < 1 or value.shape[-1] != SEGMENT_TARGET_DIM:
        raise ValueError(
            f"hard contact code must end in width {SEGMENT_TARGET_DIM}, got {value.shape}"
        )
    if value.dtype.kind != "f":
        raise TypeError("hard contact code must use a floating dtype")
    if not np.isfinite(value).all():
        raise ValueError("hard contact code requires finite values")
    if np.signbit(value[value == 0]).any():
        raise ValueError("hard contact code requires canonical positive zero")
    time_bins = int(time_bins)
    if time_bins != 16:
        raise ValueError("the receipt-bound hard grammar validator requires B=16")

    flat = value.reshape(-1, SEGMENT_TARGET_DIM)
    presence = flat[:, SEGMENT_FIELD_SLICES["presence"]].reshape(-1)
    if not np.logical_or(presence == 0, presence == 1).all():
        raise ValueError("hard contact presence must be binary")
    absent = presence == 0
    if absent.any() and not (flat[absent, 1:] == 0).all():
        raise ValueError("absent hard contact packets must be all positive zero")

    present = flat[presence == 1]
    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_value = present[:, SEGMENT_FIELD_SLICES[field]]
        if not np.logical_or(field_value == 0, field_value == 1).all() or not (
            field_value.sum(axis=-1) == 1
        ).all():
            raise ValueError(f"hard contact field {field!r} must be exactly one-hot")
    for field in ("left_censored", "right_censored"):
        field_value = present[:, SEGMENT_FIELD_SLICES[field]]
        if not np.logical_or(field_value == 0, field_value == 1).all():
            raise ValueError(f"hard contact field {field!r} must be binary")

    if present.size:
        lattice = (
            np.arange(time_bins, dtype=value.dtype)
            / np.asarray(time_bins - 1, dtype=value.dtype)
        )
        start = present[:, SEGMENT_FIELD_SLICES["start_tau"]].reshape(-1)
        end = present[:, SEGMENT_FIELD_SLICES["end_tau"]].reshape(-1)
        if not (start[:, None] == lattice[None, :]).any(axis=-1).all() or not (
            end[:, None] == lattice[None, :]
        ).any(axis=-1).all():
            raise ValueError("hard contact times must lie exactly on the B=16 lattice")
        left = present[:, SEGMENT_FIELD_SLICES["left_censored"]].reshape(-1).astype(bool)
        right = present[:, SEGMENT_FIELD_SLICES["right_censored"]].reshape(-1).astype(bool)
        if (left & (start != 0)).any():
            raise ValueError("left-censored hard contacts require start_tau=+0")
        if (right & (end != 1)).any():
            raise ValueError("right-censored hard contacts require end_tau=1")
        uncensored = ~(left | right)
        if (uncensored & (start > end)).any():
            raise ValueError("uncensored hard contacts require start_tau<=end_tau")
    return value


def contact_transport_capacity_metadata(
    *,
    mode: str,
    time_bins: int,
    num_queries: int,
) -> Mapping[str, int | float | str | bool | None]:
    """Return an auditable upper bound for the action-visible channel.

    Hard transports canonicalize every non-presence field to zero when a
    query is absent.  For a present segment, uncensored times satisfy
    ``start <= end``; left-censored segments fix start to zero;
    right-censored segments fix end to one.  The exact number of valid
    time/censor states is therefore

    ``time_bins*(time_bins+1)/2 + 2*time_bins + 1``.

    ``soft`` intentionally reports no auditable finite semantic alphabet
    bound.  This is a statement about semantic identifiability, not a claim
    that finite-precision machine tensors have mathematically infinite bits.
    """

    mode = str(mode)
    if mode not in CONTACT_SEMANTIC_TRANSPORT_MODES:
        raise ValueError(
            f"unsupported contact semantic transport mode {mode!r}; "
            f"expected one of {CONTACT_SEMANTIC_TRANSPORT_MODES}"
        )
    time_bins = int(time_bins)
    num_queries = int(num_queries)
    if time_bins < 2:
        raise ValueError(f"time_bins must be at least 2, got {time_bins}")
    if num_queries < 0:
        raise ValueError(f"num_queries must be non-negative, got {num_queries}")

    metadata: dict[str, int | float | str | bool | None] = {
        "mode": mode,
        "time_bins": time_bins,
        "num_queries": num_queries,
        "canonical_absent_state": True,
    }
    if mode == "soft":
        metadata.update(
            finite_capacity=False,
            strict_valid_time_censor_states=None,
            strict_valid_per_segment_states=None,
            strict_valid_per_segment_bits=None,
            strict_valid_total_states=None,
            strict_valid_total_bits=None,
            unconstrained_per_segment_states_upper_bound=None,
            unconstrained_per_segment_bits_upper_bound=None,
            unconstrained_total_bits_upper_bound=None,
        )
        return metadata

    categorical_states = 3 * 6 * 4 * 4
    valid_time_censor_states = time_bins * (time_bins + 1) // 2 + 2 * time_bins + 1
    strict_per_segment_states = 1 + categorical_states * valid_time_censor_states
    strict_per_segment_bits = math.log2(strict_per_segment_states)
    unconstrained_per_segment_states = 1 + categorical_states * 2 * 2 * time_bins**2
    unconstrained_per_segment_bits = math.log2(unconstrained_per_segment_states)
    metadata.update(
        finite_capacity=True,
        strict_valid_time_censor_states=valid_time_censor_states,
        strict_valid_per_segment_states=strict_per_segment_states,
        strict_valid_per_segment_bits=strict_per_segment_bits,
        strict_valid_total_states=strict_per_segment_states**num_queries,
        strict_valid_total_bits=strict_per_segment_bits * num_queries,
        unconstrained_per_segment_states_upper_bound=unconstrained_per_segment_states,
        unconstrained_per_segment_bits_upper_bound=unconstrained_per_segment_bits,
        unconstrained_total_bits_upper_bound=(
            unconstrained_per_segment_bits * num_queries
        ),
    )
    return metadata


@dataclass(frozen=True)
class ContactSegment:
    """One contact interval clipped to the current action horizon.

    ``start_frame`` and ``end_frame`` use half-open window coordinates in
    ``[0, horizon_frames]``.  Censor flags distinguish an observed boundary
    from a segment that began before, or continues after, the window.
    Unknown categorical fields stay unsupervised through their field mask.
    """

    start_frame: int
    end_frame: int
    side: str | None = None
    site: str | None = None
    object_role: str | None = None
    contact_mode: str | None = None
    left_censored: bool = False
    right_censored: bool = False
    confidence: float = 1.0


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


def _set_segment_one_hot(
    target: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    row: int,
    field: str,
    value: str | None,
    weight: float,
) -> None:
    if value is None:
        return
    choices = SEGMENT_ENUM_VALUES[field]
    if value not in choices:
        raise ValueError(f"invalid {field}={value!r}; expected one of {choices}")
    field_slice = SEGMENT_FIELD_SLICES[field]
    target[row, field_slice.start + choices.index(value)] = 1.0
    mask[row, field_slice] = 1.0
    confidence[row, field_slice] = weight


def encode_ordered_segments(
    segments: Sequence[ContactSegment],
    *,
    num_queries: int,
    horizon_frames: int,
    coverage_is_exhaustive: bool = True,
    negative_confidence: float = 0.1,
) -> Mapping[str, np.ndarray]:
    """Encode ordered v1.1 contact segments into 22-D query targets.

    Query order is a deterministic fallback for the first implementation.
    Object tracking IDs intentionally do not enter the cross-embodiment
    target.  A future Hungarian matcher may replace this ordering in the loss
    without changing the sidecar tensor contract.
    """

    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")
    if horizon_frames <= 0:
        raise ValueError(f"horizon_frames must be positive, got {horizon_frames}")
    if len(segments) > num_queries:
        raise ValueError(f"{len(segments)} segments exceed num_queries={num_queries}")
    if not 0.0 <= negative_confidence <= 1.0:
        raise ValueError("negative_confidence must be in [0, 1]")

    target = np.zeros((num_queries, SEGMENT_TARGET_DIM), dtype=np.float32)
    loss_mask = np.zeros_like(target)
    confidence = np.zeros_like(target)
    condition_mask = np.zeros_like(target)
    presence_slice = SEGMENT_FIELD_SLICES["presence"]
    if coverage_is_exhaustive:
        loss_mask[:, presence_slice] = 1.0
        confidence[:, presence_slice] = negative_confidence
        condition_mask[:, presence_slice] = 1.0

    ordered = sorted(
        segments,
        key=lambda segment: (
            int(segment.start_frame),
            int(segment.end_frame),
            str(segment.side or ""),
            str(segment.site or ""),
        ),
    )
    for row, segment in enumerate(ordered):
        if not 0.0 <= segment.confidence <= 1.0:
            raise ValueError(
                f"segment confidence must be in [0, 1], got {segment.confidence}"
            )
        if not 0 <= segment.start_frame <= segment.end_frame <= horizon_frames:
            raise ValueError(
                "segment bounds must satisfy 0 <= start <= end <= horizon, got "
                f"[{segment.start_frame}, {segment.end_frame}] for H={horizon_frames}"
            )
        weight = float(segment.confidence)
        target[row, presence_slice] = 1.0
        loss_mask[row, presence_slice] = 1.0
        confidence[row, presence_slice] = weight
        condition_mask[row, presence_slice] = 1.0

        for field, value in (
            ("side", segment.side),
            ("site", segment.site),
            ("object_role", segment.object_role),
            ("contact_mode", segment.contact_mode),
        ):
            _set_segment_one_hot(
                target, loss_mask, confidence, row, field, value, weight
            )

        for field, value in (
            ("start_tau", float(segment.start_frame) / horizon_frames),
            ("end_tau", float(segment.end_frame) / horizon_frames),
            ("left_censored", float(segment.left_censored)),
            ("right_censored", float(segment.right_censored)),
        ):
            field_slice = SEGMENT_FIELD_SLICES[field]
            target[row, field_slice] = value
            loss_mask[row, field_slice] = 1.0
            confidence[row, field_slice] = weight

        condition_mask[row] = loss_mask[row]

    return {
        "contact_target": target,
        "contact_loss_mask": loss_mask,
        "contact_confidence": confidence,
        "contact_condition_mask": condition_mask,
    }

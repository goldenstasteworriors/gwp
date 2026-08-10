"""Opt-in action/state contract for the asymmetric SONIC G1 hardware.

The legacy SONIC dataset uses 30 physical dimensions with two symmetric
six-channel Inspire hands.  This module defines a separate, deliberately
incompatible 25-D contract for a left RH56DFTP and a right one-active-DoF
OmniPicker.  Callers must opt in via :data:`ASYMMETRIC_SCHEMA_ID`; slicing the
legacy 30-D representation to 25 dimensions is never a valid conversion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np


MODEL_ACTION_DIM = 32
LEGACY_SONIC_ACTION_DIM = 30
ASYMMETRIC_ACTION_DIM = 25
ASYMMETRIC_SCHEMA_ID = "sonic_g1_rh56dftp6_omnipicker1_v1"
LEGACY_SCHEMA_ID = "sonic_g1_inspire6x2_wrist_delta_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

LEFT_WRIST_SLICE = slice(0, 9)
RIGHT_WRIST_SLICE = slice(9, 18)
RH56_SLICE = slice(18, 24)
OMNIPICKER_APERTURE_INDEX = 24

WRIST_ROT6D_IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

RH56_CHANNEL_NAMES = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)

_WRIST_STATE_NAMES = (
    "wrist_x",
    "wrist_y",
    "wrist_z",
    "wrist_r00",
    "wrist_r10",
    "wrist_r20",
    "wrist_r01",
    "wrist_r11",
    "wrist_r21",
)
_WRIST_ACTION_NAMES = tuple(
    name.replace("wrist_", "wrist_delta_") for name in _WRIST_STATE_NAMES
)

ASYMMETRIC_STATE_NAMES = (
    *(f"left_{name}" for name in _WRIST_STATE_NAMES),
    *(f"right_{name}" for name in _WRIST_STATE_NAMES),
    *(f"left_rh56_{name}" for name in RH56_CHANNEL_NAMES),
    "right_omnipicker_aperture",
)
ASYMMETRIC_ACTION_NAMES = (
    *(f"left_{name}" for name in _WRIST_ACTION_NAMES),
    *(f"right_{name}" for name in _WRIST_ACTION_NAMES),
    *(f"left_rh56_{name}_target" for name in RH56_CHANNEL_NAMES),
    "right_omnipicker_aperture_target",
)


def _feature(value, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != width:
        raise ValueError(f"{name} must have trailing dimension {width}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scalar_feature(value, name: str, leading_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == leading_shape:
        array = array[..., None]
    if array.shape != leading_shape + (1,):
        raise ValueError(
            f"{name} must have shape {leading_shape} or {leading_shape + (1,)}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def pack_asymmetric_physical(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    left_rh56: np.ndarray,
    right_omnipicker_aperture: np.ndarray | float,
    *,
    name: str = "asymmetric physical feature",
) -> np.ndarray:
    """Pack ``9 + 9 + 6 + 1`` values into the canonical 25-D layout."""

    left_wrist = _feature(left_wrist, 9, f"{name}.left_wrist")
    right_wrist = _feature(right_wrist, 9, f"{name}.right_wrist")
    left_rh56 = _feature(left_rh56, 6, f"{name}.left_rh56")
    leading = left_wrist.shape[:-1]
    if right_wrist.shape[:-1] != leading or left_rh56.shape[:-1] != leading:
        raise ValueError(
            f"{name} leading shapes differ: left_wrist={left_wrist.shape}, "
            f"right_wrist={right_wrist.shape}, left_rh56={left_rh56.shape}"
        )
    aperture = _scalar_feature(
        right_omnipicker_aperture,
        f"{name}.right_omnipicker_aperture",
        leading,
    )
    return np.concatenate((left_wrist, right_wrist, left_rh56, aperture), axis=-1)


def split_asymmetric_physical(value: np.ndarray) -> dict[str, np.ndarray]:
    """Return named views of a validated 25-D physical tensor."""

    value = _feature(value, ASYMMETRIC_ACTION_DIM, "asymmetric physical feature")
    return {
        "left_wrist": value[..., LEFT_WRIST_SLICE],
        "right_wrist": value[..., RIGHT_WRIST_SLICE],
        "left_rh56": value[..., RH56_SLICE],
        "right_omnipicker_aperture": value[..., OMNIPICKER_APERTURE_INDEX : OMNIPICKER_APERTURE_INDEX + 1],
    }


def pad_asymmetric_to_model(value: np.ndarray, model_dim: int = MODEL_ACTION_DIM) -> np.ndarray:
    """Zero-pad a 25-D tensor to the model width without changing physical values."""

    value = _feature(value, ASYMMETRIC_ACTION_DIM, "asymmetric physical feature")
    model_dim = int(model_dim)
    if model_dim < ASYMMETRIC_ACTION_DIM:
        raise ValueError(
            f"model_dim={model_dim} is smaller than asymmetric physical dim {ASYMMETRIC_ACTION_DIM}"
        )
    result = np.zeros(value.shape[:-1] + (model_dim,), dtype=np.float32)
    result[..., :ASYMMETRIC_ACTION_DIM] = value
    return result


def unpad_model_to_asymmetric(value: np.ndarray, model_dim: int = MODEL_ACTION_DIM) -> np.ndarray:
    """Validate model padding and recover the physical 25-D prefix."""

    value = _feature(value, int(model_dim), "model action/state")
    padding = value[..., ASYMMETRIC_ACTION_DIM:]
    if padding.size and not np.allclose(padding, 0.0, atol=1e-6, rtol=0.0):
        raise ValueError("model action/state has non-zero values in asymmetric padding dimensions")
    return value[..., :ASYMMETRIC_ACTION_DIM].copy()


def asymmetric_dim_mask(model_dim: int = MODEL_ACTION_DIM) -> np.ndarray:
    model_dim = int(model_dim)
    if model_dim < ASYMMETRIC_ACTION_DIM:
        raise ValueError(
            f"model_dim={model_dim} is smaller than asymmetric physical dim {ASYMMETRIC_ACTION_DIM}"
        )
    mask = np.zeros(model_dim, dtype=np.bool_)
    mask[:ASYMMETRIC_ACTION_DIM] = True
    return mask


def validate_asymmetric_norm_payload(
    payload: Mapping,
    *,
    model_dim: int = MODEL_ACTION_DIM,
    expected_hardware_contract_sha256: str | None = None,
    expected_calibration_capture_sha256: str | None = None,
) -> None:
    """Reject legacy or unlabelled normalization statistics on the 25-D path."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("asymmetric normalization stats require a metadata object")
    schema_id = metadata.get("action_schema_id")
    if schema_id != ASYMMETRIC_SCHEMA_ID:
        raise ValueError(
            f"normalization stats action_schema_id={schema_id!r}; expected {ASYMMETRIC_SCHEMA_ID!r}. "
            "Legacy 30-D SONIC stats cannot be used for asymmetric hardware."
        )
    hardware_digest = metadata.get("hardware_contract_sha256")
    calibration_digest = metadata.get("calibration_capture_sha256")
    if not isinstance(hardware_digest, str) or not _SHA256_RE.fullmatch(hardware_digest):
        raise ValueError("asymmetric normalization stats require hardware_contract_sha256")
    if not isinstance(calibration_digest, str) or not _SHA256_RE.fullmatch(calibration_digest):
        raise ValueError("asymmetric normalization stats require calibration_capture_sha256")
    if (
        expected_hardware_contract_sha256 is not None
        and hardware_digest != expected_hardware_contract_sha256
    ):
        raise ValueError("normalization stats were produced for a different hardware contract")
    if (
        expected_calibration_capture_sha256 is not None
        and calibration_digest != expected_calibration_capture_sha256
    ):
        raise ValueError("normalization stats were produced from a different calibration capture")
    stats = payload.get("norm_stats")
    if not isinstance(stats, Mapping):
        raise ValueError("normalization stats payload is missing norm_stats")
    for feature_name in ("observation.state", "action"):
        feature_stats = stats.get(feature_name)
        if not isinstance(feature_stats, Mapping):
            raise ValueError(f"normalization stats are missing {feature_name!r}")
        for statistic in ("q01", "q99"):
            values = feature_stats.get(statistic)
            if not isinstance(values, Sequence) or len(values) < int(model_dim):
                raise ValueError(
                    f"{feature_name}.{statistic} must contain at least {model_dim} values"
                )

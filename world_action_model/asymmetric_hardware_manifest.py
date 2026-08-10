"""Fail-closed hardware contract for the asymmetric 25-D action path.

This module intentionally has no JSON-Schema dependency.  The companion
``schemas/asymmetric25_hardware_manifest.schema.json`` documents the wire
format, while the validators below enforce the cross-field constraints needed
by conversion, normalization and deployment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

import numpy as np

from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    ASYMMETRIC_ACTION_NAMES,
    ASYMMETRIC_STATE_NAMES,
)


HARDWARE_MANIFEST_VERSION = "asymmetric25-hardware-contract/v1"
CALIBRATION_CAPTURE_VERSION = "asymmetric25-calibration-capture/v1"
CANONICAL_HAND_SEMANTICS = "open_fraction_0_closed_1_open"
RH56_CHANNEL_ORDER = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)
G1_BODY_JOINT_COUNT = 29
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class HardwareContractError(ValueError):
    """Raised when an asymmetric hardware contract is incomplete or unsafe."""


def _mapping(value, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise HardwareContractError(f"{path} must be an object")
    return value


def _sequence(value, path: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HardwareContractError(f"{path} must be an array")
    return value


def _string(value, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareContractError(f"{path} must be a non-empty string")
    return value.strip()


def _number(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareContractError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise HardwareContractError(f"{path} must be a finite number")
    return result


def _positive(value, path: str, *, allow_zero: bool = False) -> float:
    result = _number(value, path)
    invalid = result < 0.0 if allow_zero else result <= 0.0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise HardwareContractError(f"{path} must be {qualifier}")
    return result


def _boolean(value, path: str) -> bool:
    if not isinstance(value, bool):
        raise HardwareContractError(f"{path} must be boolean")
    return value


def _exact(value, expected, path: str) -> None:
    if value != expected:
        raise HardwareContractError(f"{path}={value!r}; expected {expected!r}")


def _validate_signal(signal: Mapping, path: str, *, command: bool) -> None:
    signal = _mapping(signal, path)
    _string(signal.get("name"), f"{path}.name")
    _string(signal.get("unit"), f"{path}.unit")
    closed = _number(signal.get("closed_value"), f"{path}.closed_value")
    opened = _number(signal.get("open_value"), f"{path}.open_value")
    low = _number(signal.get("valid_min"), f"{path}.valid_min")
    high = _number(signal.get("valid_max"), f"{path}.valid_max")
    if not low < high:
        raise HardwareContractError(f"{path}.valid_min must be less than valid_max")
    if closed == opened:
        raise HardwareContractError(f"{path} open_value and closed_value must differ")
    if not low <= min(closed, opened) <= max(closed, opened) <= high:
        raise HardwareContractError(
            f"{path} open/closed calibration values must lie inside the valid range"
        )
    if command:
        _exact(
            signal.get("semantics"),
            "absolute_position_target",
            f"{path}.semantics",
        )
        safety_low = _number(
            signal.get("safety_min_open_fraction"),
            f"{path}.safety_min_open_fraction",
        )
        safety_high = _number(
            signal.get("safety_max_open_fraction"),
            f"{path}.safety_max_open_fraction",
        )
        max_step = _positive(
            signal.get("max_step_open_fraction"),
            f"{path}.max_step_open_fraction",
        )
        if not 0.0 <= safety_low < safety_high <= 1.0:
            raise HardwareContractError(
                f"{path} safety range must satisfy 0 <= min < max <= 1"
            )
        if max_step > 1.0:
            raise HardwareContractError(f"{path}.max_step_open_fraction must be <= 1")


def _validate_hand_channel(channel: Mapping, path: str, expected_semantic: str) -> None:
    channel = _mapping(channel, path)
    _exact(channel.get("semantic"), expected_semantic, f"{path}.semantic")
    _validate_signal(channel.get("state"), f"{path}.state", command=False)
    _validate_signal(channel.get("command"), f"{path}.command", command=True)


def _validate_interface(interface: Mapping, path: str) -> None:
    interface = _mapping(interface, path)
    _string(interface.get("protocol"), f"{path}.protocol")
    _string(interface.get("state_endpoint"), f"{path}.state_endpoint")
    _string(interface.get("command_endpoint"), f"{path}.command_endpoint")
    _string(interface.get("driver_revision"), f"{path}.driver_revision")
    _string(interface.get("firmware_revision"), f"{path}.firmware_revision")


def _validate_calibration(calibration: Mapping) -> None:
    calibration = _mapping(calibration, "calibration")
    level = calibration.get("verification_level")
    if level not in {"documentation_only", "bench_verified", "robot_verified"}:
        raise HardwareContractError(
            "calibration.verification_level must be documentation_only, bench_verified, or robot_verified"
        )
    _string(calibration.get("capture_id"), "calibration.capture_id")
    digest = _string(calibration.get("capture_sha256"), "calibration.capture_sha256")
    if not _SHA256_RE.fullmatch(digest):
        raise HardwareContractError("calibration.capture_sha256 must be 64 lowercase hex characters")
    captured_at = _string(calibration.get("captured_at_utc"), "calibration.captured_at_utc")
    if not _UTC_RE.fullmatch(captured_at):
        raise HardwareContractError("calibration.captured_at_utc must be ISO-8601 UTC ending in Z")
    _string(calibration.get("source_revision"), "calibration.source_revision")
    _string(calibration.get("verified_by"), "calibration.verified_by")


def validate_hardware_manifest(
    manifest: Mapping,
    *,
    mode: Literal["schema", "training", "deployment"] = "schema",
) -> None:
    """Validate a complete asymmetric hardware contract.

    ``training`` and ``deployment`` additionally require their explicit
    readiness gates.  A template containing null placeholders therefore fails
    validation until real measurements and approval are supplied.
    """

    manifest = _mapping(manifest, "manifest")
    _exact(
        manifest.get("manifest_version"),
        HARDWARE_MANIFEST_VERSION,
        "manifest_version",
    )
    _exact(manifest.get("action_schema_id"), ASYMMETRIC_SCHEMA_ID, "action_schema_id")
    _string(manifest.get("contract_id"), "contract_id")
    _exact(
        manifest.get("canonical_hand_semantics"),
        CANONICAL_HAND_SEMANTICS,
        "canonical_hand_semantics",
    )

    readiness = _mapping(manifest.get("readiness"), "readiness")
    training_ready = _boolean(readiness.get("training"), "readiness.training")
    deployment_ready = _boolean(readiness.get("deployment"), "readiness.deployment")
    if mode == "training" and not training_ready:
        raise HardwareContractError("readiness.training must be true for asymmetric conversion")
    if mode == "deployment" and not deployment_ready:
        raise HardwareContractError("readiness.deployment must be true for inference")
    if training_ready or deployment_ready:
        _string(readiness.get("approved_by"), "readiness.approved_by")
        approved_at = _string(readiness.get("approved_at_utc"), "readiness.approved_at_utc")
        if not _UTC_RE.fullmatch(approved_at):
            raise HardwareContractError("readiness.approved_at_utc must be ISO-8601 UTC ending in Z")

    hardware = _mapping(manifest.get("hardware"), "hardware")
    left = _mapping(hardware.get("left_hand"), "hardware.left_hand")
    _exact(left.get("manufacturer"), "Inspire Robots", "hardware.left_hand.manufacturer")
    _exact(left.get("model"), "RH56DFTP", "hardware.left_hand.model")
    _exact(left.get("side"), "left", "hardware.left_hand.side")
    _exact(left.get("active_dof"), 6, "hardware.left_hand.active_dof")
    _string(left.get("serial_number"), "hardware.left_hand.serial_number")
    _string(left.get("state_feature"), "hardware.left_hand.state_feature")
    _string(left.get("command_feature"), "hardware.left_hand.command_feature")
    _validate_interface(left.get("interface"), "hardware.left_hand.interface")
    left_channels = _sequence(left.get("channels"), "hardware.left_hand.channels")
    if len(left_channels) != len(RH56_CHANNEL_ORDER):
        raise HardwareContractError("hardware.left_hand.channels must contain exactly six channels")
    for index, semantic in enumerate(RH56_CHANNEL_ORDER):
        _validate_hand_channel(
            left_channels[index],
            f"hardware.left_hand.channels[{index}]",
            semantic,
        )

    right = _mapping(hardware.get("right_gripper"), "hardware.right_gripper")
    _exact(right.get("manufacturer"), "AGIBOT", "hardware.right_gripper.manufacturer")
    _exact(right.get("model"), "OmniPicker", "hardware.right_gripper.model")
    _exact(right.get("side"), "right", "hardware.right_gripper.side")
    _exact(right.get("active_dof"), 1, "hardware.right_gripper.active_dof")
    _exact(right.get("jaw_count"), 2, "hardware.right_gripper.jaw_count")
    _string(right.get("serial_number"), "hardware.right_gripper.serial_number")
    _string(right.get("state_feature"), "hardware.right_gripper.state_feature")
    _string(right.get("command_feature"), "hardware.right_gripper.command_feature")
    _validate_interface(right.get("interface"), "hardware.right_gripper.interface")
    _validate_hand_channel(
        right.get("channel"),
        "hardware.right_gripper.channel",
        "aperture",
    )

    all_channels = list(left_channels) + [right["channel"]]
    state_names = [channel["state"]["name"] for channel in all_channels]
    command_names = [channel["command"]["name"] for channel in all_channels]
    if len(set(state_names)) != len(state_names):
        raise HardwareContractError("hand/gripper state channel names must be unique")
    if len(set(command_names)) != len(command_names):
        raise HardwareContractError("hand/gripper command channel names must be unique")
    right_text = " ".join(
        (
            right["channel"]["state"]["name"],
            right["channel"]["command"]["name"],
        )
    ).lower()
    if not any(marker in right_text for marker in ("omnipicker", "aperture", "jaw", "gripper")):
        raise HardwareContractError(
            "right-gripper channel names must explicitly identify OmniPicker/aperture/jaw/gripper"
        )
    if any(marker in right_text for marker in ("little", "ring", "middle", "index", "thumb")):
        raise HardwareContractError(
            "legacy right-hand finger channels cannot be declared as OmniPicker aperture"
        )

    wrists = _mapping(manifest.get("wrists"), "wrists")
    _string(wrists.get("state_feature"), "wrists.state_feature")
    _exact(
        wrists.get("state_representation"),
        "xyz_quat_wxyz",
        "wrists.state_representation",
    )
    _exact(
        wrists.get("action_representation"),
        "current_wbc_fk_local_delta_xyz_rotation_6d",
        "wrists.action_representation",
    )
    _exact(
        wrists.get("rotation_6d_layout"),
        "first_two_rotation_matrix_columns_column_major",
        "wrists.rotation_6d_layout",
    )
    _exact(
        wrists.get("relative_transform_convention"),
        "T_current_inverse_times_T_next",
        "wrists.relative_transform_convention",
    )
    for side in ("left", "right"):
        frame = _mapping(wrists.get(side), f"wrists.{side}")
        _string(frame.get("state_parent_frame"), f"wrists.{side}.state_parent_frame")
        _string(frame.get("state_child_frame"), f"wrists.{side}.state_child_frame")
        _string(frame.get("wbc_fk_frame"), f"wrists.{side}.wbc_fk_frame")
    translation_limit = _positive(
        wrists.get("translation_step_limit_m"),
        "wrists.translation_step_limit_m",
    )
    rotation_limit = _positive(
        wrists.get("rotation_step_limit_rad"),
        "wrists.rotation_step_limit_rad",
    )
    if translation_limit > 2.0:
        raise HardwareContractError("wrists.translation_step_limit_m is implausibly greater than 2 m")
    if rotation_limit > math.pi:
        raise HardwareContractError("wrists.rotation_step_limit_rad must be <= pi")

    wbc = _mapping(manifest.get("wbc"), "wbc")
    _string(wbc.get("state_feature"), "wbc.state_feature")
    _string(wbc.get("command_feature"), "wbc.command_feature")
    _exact(
        wbc.get("command_semantics"),
        "absolute_joint_position_target",
        "wbc.command_semantics",
    )
    for kind in ("state", "command"):
        field = f"{kind}_joint_names"
        body_names = _sequence(wbc.get(field), f"wbc.{field}")
        if len(body_names) != G1_BODY_JOINT_COUNT:
            raise HardwareContractError(
                f"wbc.{field} must contain {G1_BODY_JOINT_COUNT} ordered names"
            )
        body_names = [
            _string(name, f"wbc.{field}[{index}]")
            for index, name in enumerate(body_names)
        ]
        if len(set(body_names)) != len(body_names):
            raise HardwareContractError(f"wbc.{field} must be unique")

    timing = _mapping(manifest.get("timing"), "timing")
    _string(timing.get("timestamp_feature"), "timing.timestamp_feature")
    _exact(timing.get("unit"), "s", "timing.unit")
    _string(timing.get("clock_domain"), "timing.clock_domain")
    _exact(
        timing.get("timestamp_origin"),
        "hardware_or_synchronized_clock",
        "timing.timestamp_origin",
    )
    _exact(
        timing.get("state_sample_semantics"),
        "measured_at_timestamp",
        "timing.state_sample_semantics",
    )
    _exact(
        timing.get("action_sample_semantics"),
        "absolute_target_issued_at_timestamp",
        "timing.action_sample_semantics",
    )
    _exact(timing.get("action_index_offset"), 1, "timing.action_index_offset")
    _exact(
        timing.get("terminal_policy"),
        "repeat_last_hand_target_and_identity_wrist_delta",
        "timing.terminal_policy",
    )
    nominal_rate = _positive(timing.get("nominal_rate_hz"), "timing.nominal_rate_hz")
    if nominal_rate > 1000.0:
        raise HardwareContractError("timing.nominal_rate_hz is implausibly greater than 1000 Hz")
    _positive(timing.get("max_period_jitter_s"), "timing.max_period_jitter_s", allow_zero=True)
    measured_latency = _positive(
        timing.get("measured_command_latency_s"),
        "timing.measured_command_latency_s",
        allow_zero=True,
    )
    max_latency = _positive(
        timing.get("max_command_latency_s"),
        "timing.max_command_latency_s",
    )
    if measured_latency > max_latency:
        raise HardwareContractError(
            "timing.measured_command_latency_s exceeds max_command_latency_s"
        )
    _positive(
        timing.get("max_state_action_skew_s"),
        "timing.max_state_action_skew_s",
        allow_zero=True,
    )
    _positive(
        timing.get("max_camera_state_skew_s"),
        "timing.max_camera_state_skew_s",
        allow_zero=True,
    )
    _string(timing.get("latency_measurement_method"), "timing.latency_measurement_method")

    safety = _mapping(manifest.get("safety"), "safety")
    _string(safety.get("emergency_stop_endpoint"), "safety.emergency_stop_endpoint")
    _positive(safety.get("watchdog_timeout_s"), "safety.watchdog_timeout_s")
    invalid_limit = safety.get("max_consecutive_invalid_commands")
    if isinstance(invalid_limit, bool) or not isinstance(invalid_limit, int) or invalid_limit < 1:
        raise HardwareContractError("safety.max_consecutive_invalid_commands must be a positive integer")
    _validate_calibration(manifest.get("calibration"))
    verification_level = manifest["calibration"]["verification_level"]
    if mode == "training" and verification_level not in {"bench_verified", "robot_verified"}:
        raise HardwareContractError(
            "asymmetric conversion requires bench_verified or robot_verified calibration"
        )
    if mode == "deployment" and verification_level != "robot_verified":
        raise HardwareContractError(
            "asymmetric inference requires robot_verified calibration"
        )


def load_hardware_manifest(
    path: str | Path,
    *,
    mode: Literal["schema", "training", "deployment"] = "schema",
) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_hardware_manifest(payload, mode=mode)
    return payload


def hardware_contract_sha256(manifest: Mapping) -> str:
    """Return a deterministic digest over the complete validated contract."""

    validate_hardware_manifest(manifest, mode="schema")
    canonical = dict(manifest)
    canonical.pop("$schema", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hand_channels(manifest: Mapping) -> tuple[list[Mapping], Mapping]:
    hardware = manifest["hardware"]
    return list(hardware["left_hand"]["channels"]), hardware["right_gripper"]["channel"]


def validate_manifest_against_source_info(manifest: Mapping, info: Mapping) -> dict:
    """Bind a complete contract to raw collection metadata and return indices."""

    validate_hardware_manifest(manifest, mode="training")
    features = _mapping(info.get("features"), "dataset.features")
    left_channels, right_channel = hand_channels(manifest)
    wrists = manifest["wrists"]
    wbc = manifest["wbc"]
    left_hand = manifest["hardware"]["left_hand"]
    right_gripper = manifest["hardware"]["right_gripper"]
    state_feature = wbc["state_feature"]
    command_feature = wbc["command_feature"]
    if left_hand["state_feature"] != state_feature:
        raise HardwareContractError("left-hand state_feature must match wbc.state_feature")
    if left_hand["command_feature"] != command_feature:
        raise HardwareContractError("left-hand command_feature must match wbc.command_feature")
    if right_gripper["state_feature"] != state_feature:
        raise HardwareContractError("left and right state_feature must match")
    if right_gripper["command_feature"] != command_feature:
        raise HardwareContractError("right-gripper command_feature must match wbc.feature")

    def feature_names(key: str) -> list[str]:
        feature = _mapping(features.get(key), f"dataset.features[{key!r}]")
        names = _sequence(feature.get("names"), f"dataset.features[{key!r}].names")
        names = [_string(name, f"dataset.features[{key!r}].names") for name in names]
        shape = feature.get("shape")
        if shape != [len(names)] and shape != (len(names),):
            raise HardwareContractError(
                f"dataset feature {key!r} shape {shape!r} does not match {len(names)} names"
            )
        return names

    state_names = feature_names(state_feature)
    command_names = feature_names(command_feature)
    state_body_names = list(wbc["state_joint_names"])
    command_body_names = list(wbc["command_joint_names"])
    if state_names[:G1_BODY_JOINT_COUNT] != state_body_names:
        raise HardwareContractError(
            "dataset measured-state prefix does not exactly match manifest wbc.state_joint_names"
        )
    if command_names[:G1_BODY_JOINT_COUNT] != command_body_names:
        raise HardwareContractError(
            "dataset command prefix does not exactly match manifest wbc.command_joint_names"
        )
    eef_feature = _mapping(features.get(wrists["state_feature"]), "dataset wrist state feature")
    if eef_feature.get("shape") not in ([14], (14,)):
        raise HardwareContractError("wrist state feature must have shape [14] (xyz+quat per side)")
    timestamp_feature = timing_feature = manifest["timing"]["timestamp_feature"]
    if timestamp_feature not in features:
        raise HardwareContractError(f"dataset is missing timestamp feature {timing_feature!r}")
    declared_fps = _number(info.get("fps"), "dataset.fps")
    if not math.isclose(declared_fps, manifest["timing"]["nominal_rate_hz"], rel_tol=0.0, abs_tol=1e-6):
        raise HardwareContractError(
            f"dataset fps {declared_fps} does not match manifest nominal_rate_hz "
            f"{manifest['timing']['nominal_rate_hz']}"
        )

    requested_state = [channel["state"]["name"] for channel in left_channels]
    requested_state.append(right_channel["state"]["name"])
    requested_command = [channel["command"]["name"] for channel in left_channels]
    requested_command.append(right_channel["command"]["name"])
    missing_state = [name for name in requested_state if name not in state_names]
    missing_command = [name for name in requested_command if name not in command_names]
    if missing_state or missing_command:
        raise HardwareContractError(
            "dataset does not satisfy asymmetric hardware channels; "
            f"missing_state={missing_state} missing_command={missing_command}. "
            "Legacy dual-Inspire data cannot stand in for OmniPicker data."
        )
    return {
        "state_feature": state_feature,
        "command_feature": command_feature,
        "wrist_state_feature": wrists["state_feature"],
        "timestamp_feature": timestamp_feature,
        "left_state_indices": [state_names.index(name) for name in requested_state[:6]],
        "left_action_indices": [command_names.index(name) for name in requested_command[:6]],
        "right_state_index": state_names.index(requested_state[6]),
        "right_action_index": command_names.index(requested_command[6]),
        "manifest": manifest,
        "hardware_contract_sha256": hardware_contract_sha256(manifest),
    }


def canonicalize_hand_signal(
    values: np.ndarray,
    channels: Sequence[Mapping],
    *,
    signal: Literal["state", "command"],
    name: str,
) -> np.ndarray:
    """Convert raw hand/gripper channels to canonical open fraction."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != len(channels):
        raise HardwareContractError(
            f"{name} must have trailing dimension {len(channels)}, got {values.shape}"
        )
    result = np.empty_like(values, dtype=np.float64)
    for index, channel in enumerate(channels):
        spec = channel[signal]
        source = values[..., index]
        low = float(spec["valid_min"])
        high = float(spec["valid_max"])
        if not np.isfinite(source).all() or np.any(source < low) or np.any(source > high):
            observed_low = float(np.nanmin(source)) if source.size else math.nan
            observed_high = float(np.nanmax(source)) if source.size else math.nan
            raise HardwareContractError(
                f"{name}.{channel['semantic']} values [{observed_low}, {observed_high}] "
                f"fall outside declared source range [{low}, {high}]"
            )
        result[..., index] = (source - float(spec["closed_value"])) / (
            float(spec["open_value"]) - float(spec["closed_value"])
        )
    tolerance = 1e-6
    if np.any(result < -tolerance) or np.any(result > 1.0 + tolerance):
        raise HardwareContractError(f"{name} maps outside canonical open-fraction range [0, 1]")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def validate_episode_timing(timestamps: np.ndarray, manifest: Mapping, *, name: str) -> None:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0 or not np.isfinite(timestamps).all():
        raise HardwareContractError(f"{name} timestamps must be finite and non-empty")
    if timestamps.size == 1:
        return
    dt = np.diff(timestamps)
    if np.any(dt <= 0.0):
        raise HardwareContractError(f"{name} timestamps must be strictly increasing")
    expected = 1.0 / float(manifest["timing"]["nominal_rate_hz"])
    max_jitter = float(manifest["timing"]["max_period_jitter_s"])
    if np.any(np.abs(dt - expected) > max_jitter + 1e-12):
        raise HardwareContractError(
            f"{name} timestamp period exceeds declared jitter {max_jitter}s around {expected}s"
        )


def _rotation6d_angle(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    first = values[..., :3]
    second = values[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-6):
        raise HardwareContractError(f"{name} first rotation-6D column is degenerate")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-6):
        raise HardwareContractError(f"{name} rotation-6D columns are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    trace = first[..., 0] + second[..., 1] + third[..., 2]
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def validate_asymmetric_action(values: np.ndarray, manifest: Mapping, *, name: str) -> None:
    """Validate canonical action values against calibrated deployment limits."""

    validate_hardware_manifest(manifest, mode="schema")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in {1, 2, 3} or values.shape[-1] != ASYMMETRIC_ACTION_DIM:
        raise HardwareContractError(
            f"{name} must have shape [25], [T,25], or [B,T,25], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise HardwareContractError(f"{name} contains non-finite values")
    wrist_limits = manifest["wrists"]
    translation_limit = float(wrist_limits["translation_step_limit_m"])
    rotation_limit = float(wrist_limits["rotation_step_limit_rad"])
    for side, start in (("left", 0), ("right", 9)):
        translation = np.linalg.norm(values[..., start : start + 3], axis=-1)
        if np.any(translation > translation_limit + 1e-9):
            raise HardwareContractError(
                f"{name}.{side}_wrist translation exceeds {translation_limit} m/step"
            )
        angle = _rotation6d_angle(values[..., start + 3 : start + 9], f"{name}.{side}_wrist")
        if np.any(angle > rotation_limit + 1e-9):
            raise HardwareContractError(
                f"{name}.{side}_wrist rotation exceeds {rotation_limit} rad/step"
            )

    left_channels, right_channel = hand_channels(manifest)
    hand_values = values[..., 18:25]
    for index, channel in enumerate(left_channels + [right_channel]):
        command = channel["command"]
        low = float(command["safety_min_open_fraction"])
        high = float(command["safety_max_open_fraction"])
        target = hand_values[..., index]
        if np.any(target < low - 1e-9) or np.any(target > high + 1e-9):
            raise HardwareContractError(
                f"{name}.{channel['semantic']} target exceeds calibrated safety range [{low}, {high}]"
            )
        # Time is axis -2 in the original action tensor. Differencing before
        # selecting the channel makes [T,25] and [B,T,25] unambiguous; a lone
        # [25] command has no temporal step to check.
        if values.ndim >= 2 and values.shape[-2] > 1:
            step = np.abs(np.diff(hand_values, axis=-2)[..., index])
            if np.any(step > float(command["max_step_open_fraction"]) + 1e-9):
                raise HardwareContractError(
                    f"{name}.{channel['semantic']} exceeds max canonical command step"
                )


def validate_converted_dataset_info(info: Mapping, manifest: Mapping) -> None:
    """Validate a converted 25-D dataset and its embedded contract digest."""

    validate_hardware_manifest(manifest, mode="training")
    _exact(info.get("action_schema_id"), ASYMMETRIC_SCHEMA_ID, "dataset.action_schema_id")
    features = _mapping(info.get("features"), "dataset.features")
    expected = {
        "observation.state": list(ASYMMETRIC_STATE_NAMES),
        "action": list(ASYMMETRIC_ACTION_NAMES),
    }
    for key, names in expected.items():
        feature = _mapping(features.get(key), f"dataset.features[{key!r}]")
        if feature.get("shape") != [ASYMMETRIC_ACTION_DIM]:
            raise HardwareContractError(f"dataset {key} must have shape [25]")
        if feature.get("names") != names:
            raise HardwareContractError(f"dataset {key} names do not match canonical 25-D schema")
    expected_digest = hardware_contract_sha256(manifest)
    if info.get("hardware_contract_sha256") != expected_digest:
        raise HardwareContractError("dataset hardware contract digest is missing or inconsistent")


def validate_calibration_capture(capture: Mapping) -> None:
    """Validate the small calibration record referenced by a hardware manifest."""

    capture = _mapping(capture, "capture")
    _exact(capture.get("capture_version"), CALIBRATION_CAPTURE_VERSION, "capture_version")
    level = capture.get("verification_level")
    if level not in {"documentation_only", "bench_verified", "robot_verified"}:
        raise HardwareContractError(
            "verification_level must be documentation_only, bench_verified, or robot_verified"
        )
    _string(capture.get("capture_id"), "capture_id")
    captured_at = _string(capture.get("captured_at_utc"), "captured_at_utc")
    if not _UTC_RE.fullmatch(captured_at):
        raise HardwareContractError("captured_at_utc must be ISO-8601 UTC ending in Z")
    _string(capture.get("operator"), "operator")
    _string(capture.get("source_revision"), "source_revision")
    devices = _mapping(capture.get("devices"), "devices")
    _string(devices.get("left_rh56dftp_serial"), "devices.left_rh56dftp_serial")
    _string(devices.get("right_omnipicker_serial"), "devices.right_omnipicker_serial")
    measurements = _mapping(capture.get("measurements"), "measurements")
    # Results are structured measurement summaries, not prose attestations.
    # Binding below compares every effective manifest section against them.
    for key in ("hardware", "wrists", "wbc", "timing", "safety"):
        item = _mapping(measurements.get(key), f"measurements.{key}")
        _exact(item.get("verified"), True, f"measurements.{key}.verified")
        _string(item.get("method"), f"measurements.{key}.method")
        sample_count = item.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise HardwareContractError(
                f"measurements.{key}.sample_count must be a positive integer"
            )
        _mapping(item.get("result"), f"measurements.{key}.result")
    approval = _mapping(capture.get("approval"), "approval")
    _exact(approval.get("approved"), True, "approval.approved")
    _string(approval.get("approved_by"), "approval.approved_by")
    approved_at = _string(approval.get("approved_at_utc"), "approval.approved_at_utc")
    if not _UTC_RE.fullmatch(approved_at):
        raise HardwareContractError("approval.approved_at_utc must be ISO-8601 UTC ending in Z")


def calibration_capture_sha256(capture: Mapping) -> str:
    validate_calibration_capture(capture)
    encoded = json.dumps(
        capture,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_capture_binding(manifest: Mapping, capture: Mapping) -> None:
    """Ensure the manifest is backed by exactly this calibration record."""

    validate_hardware_manifest(manifest, mode="schema")
    validate_calibration_capture(capture)
    calibration = manifest["calibration"]
    if calibration["capture_id"] != capture["capture_id"]:
        raise HardwareContractError("manifest and calibration capture_id differ")
    digest = calibration_capture_sha256(capture)
    if calibration["capture_sha256"] != digest:
        raise HardwareContractError("manifest calibration.capture_sha256 does not match capture")
    if calibration["verification_level"] != capture["verification_level"]:
        raise HardwareContractError("manifest and capture verification levels differ")
    if calibration["source_revision"] != capture["source_revision"]:
        raise HardwareContractError("manifest and capture source revisions differ")
    if calibration["captured_at_utc"] != capture["captured_at_utc"]:
        raise HardwareContractError("manifest and capture timestamps differ")
    if calibration["verified_by"] != capture["approval"]["approved_by"]:
        raise HardwareContractError("manifest verifier differs from capture approver")
    devices = capture["devices"]
    if manifest["hardware"]["left_hand"]["serial_number"] != devices["left_rh56dftp_serial"]:
        raise HardwareContractError("manifest RH56DFTP serial does not match calibration capture")
    if manifest["hardware"]["right_gripper"]["serial_number"] != devices["right_omnipicker_serial"]:
        raise HardwareContractError("manifest OmniPicker serial does not match calibration capture")
    for key in ("hardware", "wrists", "wbc", "timing", "safety"):
        if capture["measurements"][key]["result"] != manifest[key]:
            raise HardwareContractError(
                f"calibration measurement summary for {key} does not exactly match manifest"
            )

"""Explicit conversion between LeRobot LIBERO and simulator action encodings."""

from __future__ import annotations

import numpy as np


POLICY_ACTION_ENCODING = (
    "lerobot_libero_delta_xyz_axis_angle_gripper_1_open_0_closed"
)
SIMULATOR_ACTION_ENCODING = (
    "libero_raw_delta_xyz_axis_angle_gripper_minus1_open_plus1_closed"
)
POLICY_TO_SIMULATOR_CONVERSION = (
    "clip_arm_to_minus1_plus1_and_set_sim_gripper_equal_1_minus_2_times_"
    "clipped_lerobot_gripper"
)


def lerobot_action_to_libero(action: np.ndarray) -> np.ndarray:
    """Convert one action or an action batch to raw LIBERO simulator semantics."""

    value = np.asarray(action)
    if value.ndim < 1 or value.shape[-1] < 7:
        raise ValueError(f"Expected [..., >=7] action, got {value.shape}")
    if not np.isfinite(value[..., :7]).all():
        raise ValueError("Cannot convert non-finite actions")
    converted = np.asarray(value[..., :7], dtype=np.float32).copy()
    converted[..., :6] = np.clip(converted[..., :6], -1.0, 1.0)
    lerobot_gripper = np.clip(converted[..., 6], 0.0, 1.0)
    converted[..., 6] = 1.0 - 2.0 * lerobot_gripper
    return converted

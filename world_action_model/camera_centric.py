"""Camera-frame wrist deltas for the SONIC 30-D WAM action schema.

The source wrist deltas are expressed in each wrist's *local* frame.  A
camera-frame action is a different label and must be constructed with the
per-frame wrist orientation and a camera-to-base rotation.  Translation of
the camera is deliberately absent: these are relative actions.
"""

from __future__ import annotations

import numpy as np


CAMERA_ACTION_SCHEMA_ID = "sonic_g1_dual_wrist_camera_delta30_v1"


def rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"rotation-6D must end in 6, got {value.shape}")
    first = value[..., :3]
    second = value[..., 3:6]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in 3x3, got {matrix.shape}")
    return matrix[..., :, :2].swapaxes(-2, -1).reshape(matrix.shape[:-2] + (6,))


def validate_rotation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera-to-base rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-3
    ):
        raise ValueError("camera-to-base rotation must belong to SO(3)")
    return matrix


def wrist_local_to_camera_action(
    state: np.ndarray, action: np.ndarray, camera_to_base: np.ndarray
) -> np.ndarray:
    """Convert [left wrist 9, right wrist 9, hand deltas 12] to camera frame."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    camera_to_base = validate_rotation(camera_to_base)
    if state.shape != action.shape or state.shape[-1] != 30:
        raise ValueError(f"expected matching 30-D state/action, got {state.shape}, {action.shape}")
    base_to_camera = camera_to_base.T
    result = action.copy()
    for start in (0, 9):
        wrist_to_base = rot6d_to_matrix(state[..., start + 3 : start + 9])
        local_rotation = rot6d_to_matrix(action[..., start + 3 : start + 9])
        camera_delta = base_to_camera @ wrist_to_base
        result[..., start : start + 3] = np.einsum(
            "...ij,...j->...i", camera_delta, action[..., start : start + 3]
        )
        result[..., start + 3 : start + 9] = matrix_to_rot6d(
            camera_delta @ local_rotation @ np.swapaxes(camera_delta, -1, -2)
        )
    return result.astype(np.float32)


def camera_to_wrist_local_action(
    state: np.ndarray, action: np.ndarray, camera_to_base: np.ndarray
) -> np.ndarray:
    """Convert a camera-frame prediction to executable wrist-local deltas."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    camera_to_base = validate_rotation(camera_to_base)
    if state.shape != action.shape or state.shape[-1] != 30:
        raise ValueError(f"expected matching 30-D state/action, got {state.shape}, {action.shape}")
    result = action.copy()
    for start in (0, 9):
        wrist_to_base = rot6d_to_matrix(state[..., start + 3 : start + 9])
        camera_to_wrist = np.swapaxes(wrist_to_base, -1, -2) @ camera_to_base
        camera_rotation = rot6d_to_matrix(action[..., start + 3 : start + 9])
        result[..., start : start + 3] = np.einsum(
            "...ij,...j->...i", camera_to_wrist, action[..., start : start + 3]
        )
        result[..., start + 3 : start + 9] = matrix_to_rot6d(
            camera_to_wrist @ camera_rotation @ np.swapaxes(camera_to_wrist, -1, -2)
        )
    return result.astype(np.float32)

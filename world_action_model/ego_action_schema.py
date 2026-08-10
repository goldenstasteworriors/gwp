"""Canonical 20-D action/state conversion for EgoHumanoid demonstrations.

The processed EgoHumanoid dataset stores an absolute dual-wrist pose as
``[xyz, qx, qy, qz, qw] x 2`` and a local-frame increment as
``[xyz, roll, pitch, yaw] x 2``.  WAM uses the continuous rotation-6D
representation already adopted by the SONIC wrist-delta pipeline.  Two final
dimensions retain the dataset's left/right binary hand-closure proxy.  The
upstream EgoHumanoid collection/deployment code defines it as ``0=open,
1=closed``; it is not an OmniPicker aperture or a dexterous joint vector.
"""

from __future__ import annotations

import numpy as np


EGO_WAM_ACTION_DIM = 20
EGO_CANONICAL_ACTION_SCHEMA_ID = "ego_dual_eef20_canonical_open_v1"
EGO_HAND_STATUS_SEMANTICS = "closure_fraction_0_open_1_closed"
CANONICAL_OPEN_FRACTION_SEMANTICS = "open_fraction_0_closed_1_open"
SUPPORTED_HAND_STATUS_SEMANTICS = frozenset(
    {EGO_HAND_STATUS_SEMANTICS, CANONICAL_OPEN_FRACTION_SEMANTICS}
)


def _as_float_array(value, *, trailing_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != trailing_dim:
        raise ValueError(
            f"{name} must have trailing dimension {trailing_dim}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def convert_hand_status_semantics(
    hand_status: np.ndarray,
    *,
    source_semantics: str,
    target_semantics: str,
) -> np.ndarray:
    """Explicitly convert between closure fraction and canonical open fraction.

    The conversion is deliberately opt-in.  Callers may not infer polarity
    from a column name or silently reuse the Ego binary proxy as a hardware
    aperture target.
    """

    hand_status = _as_float_array(
        hand_status,
        trailing_dim=2,
        name="hand status",
    )
    if source_semantics not in SUPPORTED_HAND_STATUS_SEMANTICS:
        raise ValueError(f"unsupported source hand-status semantics: {source_semantics!r}")
    if target_semantics not in SUPPORTED_HAND_STATUS_SEMANTICS:
        raise ValueError(f"unsupported target hand-status semantics: {target_semantics!r}")
    if np.any(hand_status < 0.0) or np.any(hand_status > 1.0):
        raise ValueError("hand status must lie in [0, 1]")
    if source_semantics == target_semantics:
        return hand_status.copy()
    return (1.0 - hand_status).astype(np.float32, copy=False)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Return the first two rotation-matrix columns in SONIC's ordering."""

    matrix = _as_float_array(matrix, trailing_dim=3, name="rotation matrix")
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {matrix.shape}")
    return matrix[..., :, :2].swapaxes(-2, -1).reshape(matrix.shape[:-2] + (6,))


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _as_float_array(
        quaternion,
        trailing_dim=4,
        name="xyzw quaternion",
    )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("xyzw quaternion contains a near-zero norm")
    x, y, z, w = np.moveaxis(quaternion / norm, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rows = (
        np.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), axis=-1),
        np.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), axis=-1),
        np.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), axis=-1),
    )
    return np.stack(rows, axis=-2).astype(np.float32, copy=False)


def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Invert SciPy's lowercase ``as_euler('xyz')`` convention."""

    rpy = _as_float_array(rpy, trailing_dim=3, name="xyz Euler rotation")
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rows = (
        np.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), axis=-1),
        np.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), axis=-1),
        np.stack((-sp, cp * sr, cp * cr), axis=-1),
    )
    return np.stack(rows, axis=-2).astype(np.float32, copy=False)


def dual_eef_pose_to_state(pose: np.ndarray, hand_status: np.ndarray) -> np.ndarray:
    """Convert absolute dual-wrist pose plus grip proxy to a 20-D state."""

    pose = _as_float_array(pose, trailing_dim=14, name="dual EEF pose")
    hand_status = _as_float_array(
        hand_status,
        trailing_dim=2,
        name="hand status",
    )
    if pose.shape[:-1] != hand_status.shape[:-1]:
        raise ValueError(
            f"pose/hand-status leading shapes differ: {pose.shape} vs {hand_status.shape}"
        )
    left = np.concatenate(
        (pose[..., :3], matrix_to_rot6d(quaternion_xyzw_to_matrix(pose[..., 3:7]))),
        axis=-1,
    )
    right = np.concatenate(
        (pose[..., 7:10], matrix_to_rot6d(quaternion_xyzw_to_matrix(pose[..., 10:14]))),
        axis=-1,
    )
    return np.concatenate((left, right, hand_status), axis=-1).astype(
        np.float32,
        copy=False,
    )


def dual_eef_delta_to_action(delta: np.ndarray, hand_status: np.ndarray) -> np.ndarray:
    """Convert local dual-wrist xyz/RPY increments plus grip proxy to 20-D."""

    delta = _as_float_array(delta, trailing_dim=12, name="dual EEF delta")
    hand_status = _as_float_array(
        hand_status,
        trailing_dim=2,
        name="hand status",
    )
    if delta.shape[:-1] != hand_status.shape[:-1]:
        raise ValueError(
            f"delta/hand-status leading shapes differ: {delta.shape} vs {hand_status.shape}"
        )
    left = np.concatenate(
        (delta[..., :3], matrix_to_rot6d(euler_xyz_to_matrix(delta[..., 3:6]))),
        axis=-1,
    )
    right = np.concatenate(
        (delta[..., 6:9], matrix_to_rot6d(euler_xyz_to_matrix(delta[..., 9:12]))),
        axis=-1,
    )
    return np.concatenate((left, right, hand_status), axis=-1).astype(
        np.float32,
        copy=False,
    )

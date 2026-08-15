#!/usr/bin/env python3
"""Standalone q-to-mask renderer with no LIBERO environment construction.

The runtime input is a robot-only compiled kinematic model, joint positions,
gripper positions, and an explicit camera calibration.  Only
``mujoco.mj_kinematics`` is used to evaluate the robot tree.  No benchmark
environment, task object, controller, contact, state transition, or dynamics
API is imported or called.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import mujoco
import numpy as np


ASSET_SCHEMA = "libero-standalone-panda-kinematic-asset-v1"
CALIBRATION_SCHEMA = "libero-fixed-camera-calibration-v1"
RASTERIZER_SEMANTICS = (
    "amodal_robot_visual_mesh_triangle_union_from_explicit_q_and_camera_calibration"
)
DYNAMICS_APIS = ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_forwardSkip")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def forbid_mujoco_dynamics() -> Iterator[dict[str, int]]:
    """Make accidental dynamics or full-forward calls fail at runtime."""

    counts = {name: 0 for name in DYNAMICS_APIS}
    originals = {name: getattr(mujoco, name) for name in DYNAMICS_APIS}

    def make_forbidden(name: str) -> Any:
        def forbidden(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            counts[name] += 1
            raise RuntimeError(
                f"Only standalone FK is allowed for robot masks; called {name}"
            )

        return forbidden

    for name in DYNAMICS_APIS:
        setattr(mujoco, name, make_forbidden(name))
    try:
        yield counts
    finally:
        for name, function in originals.items():
            setattr(mujoco, name, function)


class StandaloneRobotMaskRenderer:
    """Specified q + explicit camera -> robot-only FK -> CPU rasterization."""

    def __init__(self, asset_dir: Path, width: int, height: int) -> None:
        self.asset_dir = Path(asset_dir)
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rasterizer dimensions must be positive")

        manifest = _load_json(self.asset_dir / "manifest.json")
        calibration = _load_json(self.asset_dir / "camera_calibration.json")
        if manifest.get("schema_version") != ASSET_SCHEMA:
            raise ValueError(f"Unexpected standalone asset schema: {manifest}")
        if calibration.get("schema_version") != CALIBRATION_SCHEMA:
            raise ValueError(f"Unexpected camera calibration schema: {calibration}")
        required_false = (
            "benchmark_environment_imported",
            "benchmark_environment_constructed",
            "benchmark_state_read",
            "controller_used",
            "contact_used",
        )
        if any(manifest.get(key) is not False for key in required_false):
            raise ValueError("Standalone asset manifest contains an environment dependency")
        if manifest.get("fk_api") != "mujoco.mj_kinematics":
            raise ValueError("Standalone renderer requires the mj_kinematics-only contract")

        model_path = self.asset_dir / str(manifest["compiled_model_file"])
        if _sha256_file(model_path) != manifest["compiled_model_sha256"]:
            raise ValueError("Standalone robot model hash mismatch")
        if _sha256_file(self.asset_dir / "camera_calibration.json") != manifest[
            "camera_calibration_sha256"
        ]:
            raise ValueError("Standalone camera calibration hash mismatch")
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.manifest = manifest
        self.calibration = calibration

        self.arm_qpos_indexes = self._qpos_indexes(manifest["arm_joint_names"])
        self.gripper_qpos_indexes = self._qpos_indexes(
            manifest["gripper_joint_names"]
        )
        self.arm_joint_limits = np.asarray(manifest["arm_joint_limits"], dtype=np.float64)
        if self.arm_joint_limits.shape != (7, 2):
            raise ValueError("Expected seven arm joint limits")

        self.robot_geom_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_GEOM, manifest["robot_visual_geom_names"]
        )
        self.gripper_geom_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_GEOM, manifest["gripper_visual_geom_names"]
        )
        if any(
            int(self.model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_MESH)
            for index in self.robot_geom_ids
        ):
            raise ValueError("Standalone rasterizer only accepts visual mesh geoms")
        self.eef_site_id = self._named_id(
            mujoco.mjtObj.mjOBJ_SITE, str(manifest["eef_site_name"])
        )

        self.camera_position = np.asarray(
            calibration["camera_position_world"], dtype=np.float64
        )
        self.camera_to_world = np.asarray(
            calibration["camera_rotation_camera_to_world"], dtype=np.float64
        )
        self.camera_intrinsic = np.asarray(
            calibration["camera_intrinsic_at_reference_resolution"], dtype=np.float64
        )
        reference_width, reference_height = calibration["reference_resolution"]
        if self.camera_position.shape != (3,) or self.camera_to_world.shape != (3, 3):
            raise ValueError("Malformed camera extrinsic calibration")
        if self.camera_intrinsic.shape != (3, 3):
            raise ValueError("Malformed camera intrinsic calibration")
        self.camera_intrinsic = self.camera_intrinsic.copy()
        self.camera_intrinsic[0, :] *= self.width / float(reference_width)
        self.camera_intrinsic[1, :] *= self.height / float(reference_height)

        mujoco.mj_kinematics(self.model, self.data)

    def _named_id(self, object_type: Any, name: str) -> int:
        index = int(mujoco.mj_name2id(self.model, object_type, name))
        if index < 0:
            raise ValueError(f"Name absent from standalone model: {name}")
        return index

    def _named_ids(self, object_type: Any, names: list[str]) -> np.ndarray:
        return np.asarray(
            [self._named_id(object_type, str(name)) for name in names], dtype=np.int64
        )

    def _qpos_indexes(self, names: list[str]) -> np.ndarray:
        joint_ids = self._named_ids(mujoco.mjtObj.mjOBJ_JOINT, names)
        indexes = np.asarray(self.model.jnt_qposadr[joint_ids], dtype=np.int64)
        if len(indexes) != len(set(int(value) for value in indexes)):
            raise ValueError("Joint qpos addresses are not unique")
        return indexes

    def set_configuration(
        self, arm_qpos: np.ndarray, gripper_qpos: np.ndarray
    ) -> None:
        arm_qpos = np.asarray(arm_qpos, dtype=np.float64)
        gripper_qpos = np.asarray(gripper_qpos, dtype=np.float64)
        if arm_qpos.shape != (7,):
            raise ValueError(f"Expected arm q shape (7,), got {arm_qpos.shape}")
        if gripper_qpos.shape != (2,):
            raise ValueError(f"Expected gripper q shape (2,), got {gripper_qpos.shape}")
        self.data.qpos[self.arm_qpos_indexes] = np.clip(
            arm_qpos, self.arm_joint_limits[:, 0], self.arm_joint_limits[:, 1]
        )
        self.data.qpos[self.gripper_qpos_indexes] = gripper_qpos
        mujoco.mj_kinematics(self.model, self.data)

    def _project_world_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_points = (
            np.asarray(points, dtype=np.float64) - self.camera_position
        ) @ self.camera_to_world
        depth = -camera_points[:, 2]
        normalized_x = camera_points[:, 0] / depth
        normalized_y = camera_points[:, 1] / depth
        u_raw = (
            self.camera_intrinsic[0, 0] * normalized_x + self.camera_intrinsic[0, 2]
        )
        v_raw = (
            self.camera_intrinsic[1, 1] * normalized_y + self.camera_intrinsic[1, 2]
        )
        orientation = self.calibration["dataset_image_orientation"]
        if orientation != "lerobot_libero_visualization_flip_height_width":
            raise ValueError(f"Unsupported image orientation: {orientation}")
        pixels = np.stack(
            [(self.width - 1) - u_raw, (self.height - 1) - v_raw], axis=-1
        )
        return pixels, depth

    def _geom_triangles_pixels(self, geom_id: int) -> list[np.ndarray]:
        mesh_id = int(self.model.geom_dataid[geom_id])
        vertex_start = int(self.model.mesh_vertadr[mesh_id])
        vertex_count = int(self.model.mesh_vertnum[mesh_id])
        face_start = int(self.model.mesh_faceadr[mesh_id])
        face_count = int(self.model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            self.model.mesh_vert[vertex_start : vertex_start + vertex_count],
            dtype=np.float64,
        )
        faces = np.asarray(
            self.model.mesh_face[face_start : face_start + face_count], dtype=np.int64
        )
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        world_vertices = vertices @ rotation.T + translation
        pixels, depth = self._project_world_points(world_vertices)
        valid_faces = np.all(depth[faces] > 1e-5, axis=1)
        triangles = np.rint(pixels[faces[valid_faces]]).astype(np.int32)
        return [triangle for triangle in triangles]

    def render_masks(self) -> dict[str, np.ndarray]:
        robot_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        gripper_mask = np.zeros_like(robot_mask)
        robot_triangles: list[np.ndarray] = []
        gripper_triangles: list[np.ndarray] = []
        gripper_ids = set(int(value) for value in self.gripper_geom_ids)
        for geom_id in self.robot_geom_ids:
            triangles = self._geom_triangles_pixels(int(geom_id))
            robot_triangles.extend(triangles)
            if int(geom_id) in gripper_ids:
                gripper_triangles.extend(triangles)
        if robot_triangles:
            cv2.fillPoly(robot_mask, robot_triangles, color=1)
        if gripper_triangles:
            cv2.fillPoly(gripper_mask, gripper_triangles, color=1)
        gripper_mask &= robot_mask
        return {
            "robot_mask": robot_mask,
            "gripper_mask": gripper_mask,
            "arm_qpos": np.asarray(self.data.qpos[self.arm_qpos_indexes], dtype=np.float32),
            "gripper_qpos": np.asarray(
                self.data.qpos[self.gripper_qpos_indexes], dtype=np.float32
            ),
            "eef_position": np.asarray(
                self.data.site_xpos[self.eef_site_id], dtype=np.float32
            ),
            "eef_rotation": np.asarray(
                self.data.site_xmat[self.eef_site_id], dtype=np.float32
            ).reshape(3, 3),
            "camera_position": self.camera_position.astype(np.float32),
            "camera_rotation": self.camera_to_world.astype(np.float32),
            "camera_intrinsic": self.camera_intrinsic.astype(np.float32),
            "camera_fovy": np.asarray(
                self.calibration["camera_fovy_degrees"], dtype=np.float32
            ),
        }

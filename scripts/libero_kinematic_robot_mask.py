#!/usr/bin/env python3
"""Pure-kinematic LIBERO robot-mask rendering utilities.

This module deliberately forbids MuJoCo dynamics.  A robot configuration is
written directly into ``qpos``; ``mj_forward`` performs forward kinematics;
and a CPU triangle rasterizer projects only the robot visual meshes.  Task
objects, contacts, rewards, and environment transitions are not consulted.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Iterator

import cv2
import numpy as np


ROBOT_INSTANCES = {"MountedPanda0", "PandaGripper0"}
GRIPPER_INSTANCE = "PandaGripper0"
RASTERIZER_SEMANTICS = (
    "amodal_robot_visual_mesh_triangle_union_from_q_and_calibrated_camera"
)
KINEMATIC_TRANSITION = (
    "damped_least_squares_inverse_kinematics_without_mj_step_or_env_step"
)


def import_mujoco_without_render_context() -> tuple[Any, Any, Any]:
    """Import MuJoCo/robosuite without requiring an EGL device.

    Robosuite 1.4 forces ``MUJOCO_GL=egl`` while importing its bindings when
    GPU rendering is enabled in its installed macros, even when the requested
    environment has both renderers disabled.  This kinematic path never
    creates a GL context.  Preloading MuJoCo with its lazy GLFW backend and
    using robosuite's accepted ``glx`` sentinel during the binding import
    prevents that unnecessary EGL initialization.  No package or driver is
    modified.
    """

    original_gl = os.environ.get("MUJOCO_GL")
    try:
        if "mujoco" not in sys.modules:
            os.environ["MUJOCO_GL"] = "glfw"
        mujoco = importlib.import_module("mujoco")
        os.environ["MUJOCO_GL"] = "glx"
        base = importlib.import_module("robosuite.environments.base")
        binding = importlib.import_module("robosuite.utils.binding_utils")
    finally:
        if original_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = original_gl
    return mujoco, base.MujocoEnv, binding.MjSim


@contextmanager
def forbid_mujoco_dynamics() -> Iterator[dict[str, int]]:
    """Raise immediately if code inside the context attempts ``mj_step``.

    Patching both entry points makes the prohibition executable rather than a
    manifest-only claim.  ``mj_forward`` remains available because it is the
    FK operation required by this renderer.
    """

    mujoco, MujocoEnv, MjSim = import_mujoco_without_render_context()

    counts = {"mj_step": 0, "sim_step": 0, "env_step": 0}
    original_mj_step = mujoco.mj_step
    original_sim_step = MjSim.step
    original_env_step = MujocoEnv.step

    def forbidden_mj_step(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        counts["mj_step"] += 1
        raise RuntimeError("Dynamics are forbidden for kinematic mask rendering: mj_step")

    def forbidden_sim_step(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        counts["sim_step"] += 1
        raise RuntimeError("Dynamics are forbidden for kinematic mask rendering: sim.step")

    def forbidden_env_step(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        counts["env_step"] += 1
        raise RuntimeError("Dynamics are forbidden for kinematic mask rendering: env.step")

    mujoco.mj_step = forbidden_mj_step
    MjSim.step = forbidden_sim_step
    MujocoEnv.step = forbidden_env_step
    try:
        yield counts
    finally:
        mujoco.mj_step = original_mj_step
        MjSim.step = original_sim_step
        MujocoEnv.step = original_env_step


def axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Match robosuite OSC's three-dimensional orientation error."""

    return 0.5 * sum(
        np.cross(current[:, axis], desired[:, axis]) for axis in range(3)
    )


def create_kinematic_libero_environment(
    suite: Any,
    libero_task_index: int,
) -> Any:
    """Load task XML assets without creating a renderer or stepping physics."""

    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    task = suite.get_task(libero_task_index)
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    return ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        initialization_noise=None,
        hard_reset=False,
    )


class KinematicRobotMaskRenderer:
    """Direct q -> FK -> robot-only mesh rasterization for a Panda robot."""

    def __init__(
        self,
        environment: Any,
        camera_name: str,
        width: int,
        height: int,
        damping: float = 0.05,
        max_iterations: int = 30,
        position_tolerance: float = 1e-5,
        orientation_tolerance: float = 1e-4,
        max_joint_update: float = 0.12,
        translation_execution_gain: float = 0.2,
        rotation_execution_gain: float = 0.2,
    ) -> None:
        self.environment = environment
        self.sim = environment.sim
        self.robot = environment.robots[0]
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.max_joint_update = float(max_joint_update)
        self.translation_execution_gain = float(translation_execution_gain)
        self.rotation_execution_gain = float(rotation_execution_gain)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rasterizer dimensions must be positive")
        if not 0.0 < self.translation_execution_gain <= 1.0:
            raise ValueError("translation_execution_gain must be in (0, 1]")
        if not 0.0 < self.rotation_execution_gain <= 1.0:
            raise ValueError("rotation_execution_gain must be in (0, 1]")
        self.arm_qpos_indexes = np.asarray(
            self.robot._ref_joint_pos_indexes, dtype=np.int64
        )
        self.arm_qvel_indexes = np.asarray(
            self.robot._ref_joint_vel_indexes, dtype=np.int64
        )
        self.gripper_qpos_indexes = np.asarray(
            self.robot._ref_gripper_joint_pos_indexes, dtype=np.int64
        )
        self.arm_joint_ids = np.asarray(
            [self.sim.model.joint_name2id(name) for name in self.robot.robot_joints],
            dtype=np.int64,
        )
        self.arm_joint_limits = np.asarray(
            self.sim.model.jnt_range[self.arm_joint_ids], dtype=np.float64
        )
        self.eef_site_name = self.robot.gripper.important_sites["grip_site"]
        self.eef_site_id = self.sim.model.site_name2id(self.eef_site_name)
        self.camera_id = self.sim.model.camera_name2id(camera_name)

        mapping = environment.env.model.geom_ids_to_instances
        self.robot_geom_ids = np.asarray(
            sorted(
                int(geom_id)
                for geom_id, instance in mapping.items()
                if instance in ROBOT_INSTANCES
                and int(self.sim.model.geom_group[int(geom_id)]) == 1
            ),
            dtype=np.int64,
        )
        self.gripper_geom_ids = np.asarray(
            sorted(
                int(geom_id)
                for geom_id, instance in mapping.items()
                if instance == GRIPPER_INSTANCE
                and int(self.sim.model.geom_group[int(geom_id)]) == 1
            ),
            dtype=np.int64,
        )
        if not len(self.robot_geom_ids) or not len(self.gripper_geom_ids):
            raise RuntimeError("Could not identify Panda visual mesh geometries")
        if any(int(self.sim.model.geom_type[index]) != 7 for index in self.robot_geom_ids):
            raise RuntimeError("Pure rasterizer currently requires all visual geoms to be meshes")

        gripper_joint_ids = np.asarray(
            [
                self.sim.model.joint_name2id(name)
                for name in self.robot.gripper_joints
            ],
            dtype=np.int64,
        )
        gripper_limits = np.asarray(
            self.sim.model.jnt_range[gripper_joint_ids], dtype=np.float64
        )
        # Panda finger joints have opposite signs.  The endpoints with the
        # larger absolute opening are the canonical open configuration.
        self.gripper_open_qpos = np.asarray(
            [gripper_limits[0, 1], gripper_limits[1, 0]], dtype=np.float64
        )
        self.gripper_closed_qpos = np.asarray(
            [gripper_limits[0, 0], gripper_limits[1, 1]], dtype=np.float64
        )
        self.sim.forward()

    def set_configuration(
        self,
        arm_qpos: np.ndarray,
        gripper_qpos: np.ndarray,
    ) -> None:
        arm_qpos = np.asarray(arm_qpos, dtype=np.float64)
        gripper_qpos = np.asarray(gripper_qpos, dtype=np.float64)
        if arm_qpos.shape != (len(self.arm_qpos_indexes),):
            raise ValueError(f"Expected arm q shape (7,), got {arm_qpos.shape}")
        if gripper_qpos.shape != (len(self.gripper_qpos_indexes),):
            raise ValueError(f"Expected gripper q shape (2,), got {gripper_qpos.shape}")
        self.sim.data.qpos[self.arm_qpos_indexes] = np.clip(
            arm_qpos, self.arm_joint_limits[:, 0], self.arm_joint_limits[:, 1]
        )
        self.sim.data.qpos[self.gripper_qpos_indexes] = gripper_qpos
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()

    def gripper_qpos_from_policy_state(self, policy_gripper: float) -> np.ndarray:
        alpha = float(np.clip(policy_gripper, 0.0, 1.0))
        return (
            alpha * self.gripper_open_qpos
            + (1.0 - alpha) * self.gripper_closed_qpos
        )

    def current_arm_qpos(self) -> np.ndarray:
        return np.asarray(
            self.sim.data.qpos[self.arm_qpos_indexes], dtype=np.float64
        ).copy()

    def current_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(self.sim.data.site_xpos[self.eef_site_id], dtype=np.float64)
        rotation = np.asarray(
            self.sim.data.site_xmat[self.eef_site_id], dtype=np.float64
        ).reshape(3, 3)
        return position.copy(), rotation.copy()

    def integrate_policy_action(self, action: np.ndarray) -> dict[str, Any]:
        """Map one normalized OSC pose command to q using only differential IK."""

        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")
        clipped = np.clip(action[:6], -1.0, 1.0)
        start_position, start_rotation = self.current_eef_pose()
        desired_position = (
            start_position
            + clipped[:3] * 0.05 * self.translation_execution_gain
        )
        desired_rotation = axis_angle_to_matrix(
            clipped[3:6] * 0.5 * self.rotation_execution_gain
        ) @ start_rotation

        position_error_norm = float("inf")
        orientation_error_norm = float("inf")
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            position, rotation = self.current_eef_pose()
            position_delta = desired_position - position
            orientation_delta = orientation_error(desired_rotation, rotation)
            position_error_norm = float(np.linalg.norm(position_delta))
            orientation_error_norm = float(np.linalg.norm(orientation_delta))
            if (
                position_error_norm <= self.position_tolerance
                and orientation_error_norm <= self.orientation_tolerance
            ):
                break
            jacobian_position = np.asarray(
                self.sim.data.get_site_jacp(self.eef_site_name), dtype=np.float64
            ).reshape(3, -1)[:, self.arm_qvel_indexes]
            jacobian_rotation = np.asarray(
                self.sim.data.get_site_jacr(self.eef_site_name), dtype=np.float64
            ).reshape(3, -1)[:, self.arm_qvel_indexes]
            jacobian = np.vstack([jacobian_position, jacobian_rotation])
            error = np.concatenate([position_delta, orientation_delta])
            regularized = jacobian @ jacobian.T + np.eye(6) * self.damping**2
            joint_update = jacobian.T @ np.linalg.solve(regularized, error)
            update_norm = float(np.linalg.norm(joint_update))
            if update_norm > self.max_joint_update:
                joint_update *= self.max_joint_update / update_norm
            next_q = np.clip(
                self.current_arm_qpos() + joint_update,
                self.arm_joint_limits[:, 0],
                self.arm_joint_limits[:, 1],
            )
            self.sim.data.qpos[self.arm_qpos_indexes] = next_q
            self.sim.data.qvel[:] = 0.0
            self.sim.forward()

        gripper_qpos = self.gripper_qpos_from_policy_state(float(action[-1]))
        self.sim.data.qpos[self.gripper_qpos_indexes] = gripper_qpos
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()
        return {
            "iterations": iterations,
            "position_error": position_error_norm,
            "orientation_error": orientation_error_norm,
            "converged": bool(
                position_error_norm <= self.position_tolerance
                and orientation_error_norm <= self.orientation_tolerance
            ),
        }

    def _project_world_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_position = np.asarray(
            self.sim.data.cam_xpos[self.camera_id], dtype=np.float64
        )
        camera_to_world = np.asarray(
            self.sim.data.cam_xmat[self.camera_id], dtype=np.float64
        ).reshape(3, 3)
        camera_points = (np.asarray(points, dtype=np.float64) - camera_position) @ camera_to_world
        depth = -camera_points[:, 2]
        focal = 0.5 * self.height / np.tan(
            np.deg2rad(float(self.sim.model.cam_fovy[self.camera_id])) * 0.5
        )
        # MuJoCo read_pixels is OpenGL-bottom-up.  LIBERO / LeRobot then flips
        # both H and W; encode that final dataset convention directly here.
        u_raw = self.width * 0.5 + focal * camera_points[:, 0] / depth
        v_raw = self.height * 0.5 + focal * camera_points[:, 1] / depth
        pixels = np.stack(
            [(self.width - 1) - u_raw, (self.height - 1) - v_raw], axis=-1
        )
        return pixels, depth

    def _geom_triangles_pixels(self, geom_id: int) -> list[np.ndarray]:
        model = self.sim.model
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + vertex_count],
            dtype=np.float64,
        )
        faces = np.asarray(
            model.mesh_face[face_start : face_start + face_count], dtype=np.int64
        )
        rotation = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=np.float64)
        world_vertices = vertices @ rotation.T + translation
        pixels, depth = self._project_world_points(world_vertices)
        valid_faces = np.all(depth[faces] > 1e-5, axis=1)
        triangles = np.rint(pixels[faces[valid_faces]]).astype(np.int32)
        if not len(triangles):
            return []
        return [triangle for triangle in triangles]

    def render_masks(self) -> dict[str, np.ndarray]:
        """Rasterize amodal robot and gripper silhouettes on the CPU."""

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
        position, rotation = self.current_eef_pose()
        return {
            "robot_mask": robot_mask,
            "gripper_mask": gripper_mask,
            "arm_qpos": self.current_arm_qpos().astype(np.float32),
            "gripper_qpos": np.asarray(
                self.sim.data.qpos[self.gripper_qpos_indexes], dtype=np.float32
            ).copy(),
            "eef_position": position.astype(np.float32),
            "eef_rotation": rotation.astype(np.float32),
            "camera_position": np.asarray(
                self.sim.data.cam_xpos[self.camera_id], dtype=np.float32
            ).copy(),
            "camera_rotation": np.asarray(
                self.sim.data.cam_xmat[self.camera_id], dtype=np.float32
            ).reshape(3, 3).copy(),
            "camera_fovy": np.asarray(
                self.sim.model.cam_fovy[self.camera_id], dtype=np.float32
            ),
        }

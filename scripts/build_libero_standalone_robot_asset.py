#!/usr/bin/env python3
"""Build a robot-only Panda FK asset and explicit fixed-camera calibration.

This is a static model-construction utility.  It imports robot / gripper MJCF
definitions but never imports or constructs a LIBERO benchmark environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from libero_standalone_robot_mask import ASSET_SCHEMA, CALIBRATION_SCHEMA


EXPECTED_PANDA_ARM_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--robot-base-position", type=float, nargs=3, required=True)
    parser.add_argument("--camera-position", type=float, nargs=3, required=True)
    parser.add_argument("--camera-quaternion-wxyz", type=float, nargs=4, required=True)
    parser.add_argument("--camera-fovy-degrees", type=float, required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--reference-width", type=int, default=256)
    parser.add_argument("--reference-height", type=int, default=256)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_joint_limits(model: mujoco.MjModel, names: list[str]) -> list[list[float]]:
    result: list[list[float]] = []
    for name in names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0:
            raise ValueError(f"Joint absent from compiled model: {name}")
        result.append([float(value) for value in model.jnt_range[joint_id]])
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.reference_width <= 0 or args.reference_height <= 0:
        raise ValueError("Reference resolution must be positive")
    args.output_dir.mkdir(parents=True)

    # These imports expose static robosuite MJCF model definitions only.
    # LIBERO and its benchmark environment modules remain unimported.
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.mounts import mount_factory
    from robosuite.models.robots.manipulators.panda_robot import Panda

    robot = Panda(idn=0)
    robot.add_mount(mount_factory(robot.default_mount, idn=0))
    gripper = gripper_factory(robot.default_gripper, idn=0)
    robot.add_gripper(gripper)
    robot.set_base_xpos(np.asarray(args.robot_base_position, dtype=np.float64))

    # Arena models normally contribute this compiler setting during merge.
    # A robot-only model must declare it explicitly; otherwise MuJoCo's default
    # degree convention would reinterpret Panda's radian joint ranges.
    compiler = robot.root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(robot.root, "compiler")
    compiler.set("angle", "radian")

    xml = robot.get_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    model_path = args.output_dir / "robot_kinematic_model.mjb"
    mujoco.mj_saveModel(model, str(model_path), None)
    xml_path = args.output_dir / "robot_kinematic_model.xml"
    xml_path.write_text(xml, encoding="utf-8")

    arm_joint_names = [f"robot0_joint{index}" for index in range(1, 8)]
    gripper_joint_names = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
    arm_joint_limits = np.asarray(
        named_joint_limits(model, arm_joint_names), dtype=np.float64
    )
    if not np.allclose(arm_joint_limits, EXPECTED_PANDA_ARM_LIMITS, atol=1e-7):
        raise ValueError(
            "Compiled Panda joint limits are not in radians: "
            f"{arm_joint_limits.tolist()}"
        )
    robot_visual_geom_names = list(robot.visual_geoms) + list(gripper.visual_geoms)
    gripper_visual_geom_names = list(gripper.visual_geoms)
    if len(robot_visual_geom_names) != len(set(robot_visual_geom_names)):
        raise ValueError("Duplicate visual geom name in robot-only model")

    quaternion = np.asarray(args.camera_quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion)
    rotation = rotation.reshape(3, 3)
    focal = 0.5 * args.reference_height / np.tan(
        0.5 * np.deg2rad(args.camera_fovy_degrees)
    )
    intrinsic = [
        [float(focal), 0.0, args.reference_width * 0.5],
        [0.0, float(focal), args.reference_height * 0.5],
        [0.0, 0.0, 1.0],
    ]
    calibration = {
        "schema_version": CALIBRATION_SCHEMA,
        "profile_id": args.profile_id,
        "source": "explicit_fixed_benchmark_camera_calibration",
        "camera_name": args.camera_name,
        "camera_position_world": [float(value) for value in args.camera_position],
        "camera_quaternion_wxyz": [float(value) for value in quaternion],
        "camera_rotation_camera_to_world": rotation.tolist(),
        "camera_fovy_degrees": float(args.camera_fovy_degrees),
        "camera_intrinsic_at_reference_resolution": intrinsic,
        "reference_resolution": [args.reference_width, args.reference_height],
        "dataset_image_orientation": "lerobot_libero_visualization_flip_height_width",
    }
    calibration_path = args.output_dir / "camera_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    eef_site_name = "gripper0_grip_site"
    if int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site_name)) < 0:
        raise ValueError(f"EEF site absent from compiled model: {eef_site_name}")
    manifest = {
        "schema_version": ASSET_SCHEMA,
        "profile_id": args.profile_id,
        "mapping": "specified q + gripper q + explicit camera calibration -> standalone robot FK + mesh rasterization",
        "compiled_model_file": model_path.name,
        "compiled_model_sha256": sha256_file(model_path),
        "source_xml_file": xml_path.name,
        "source_xml_sha256": sha256_bytes(xml.encode("utf-8")),
        "camera_calibration_file": calibration_path.name,
        "camera_calibration_sha256": sha256_file(calibration_path),
        "robot_base_position": [float(value) for value in args.robot_base_position],
        "arm_joint_names": arm_joint_names,
        "gripper_joint_names": gripper_joint_names,
        "arm_joint_limits": arm_joint_limits.tolist(),
        "robot_visual_geom_names": robot_visual_geom_names,
        "gripper_visual_geom_names": gripper_visual_geom_names,
        "eef_site_name": eef_site_name,
        "fk_api": "mujoco.mj_kinematics",
        "benchmark_environment_imported": False,
        "benchmark_environment_constructed": False,
        "benchmark_state_read": False,
        "controller_used": False,
        "contact_used": False,
        "dynamics_used": False,
        "mujoco_version": mujoco.__version__,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

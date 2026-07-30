#!/usr/bin/env python3
"""Visualize a GWP SONIC wrist-delta rollout with DiT4DiT's MuJoCo stack.

The rollout stores 30-D dual-arm wrist/hand deltas.  Each prediction chunk is
integrated from the matching SONIC ``action.wbc`` frame, converted to the two
7-DoF arm targets with DiT4DiT's damped-least-squares MuJoCo IK, and rendered
with DiT4DiT's G1/Inspire pose and video helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if "--viewer" in sys.argv and os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
    os.environ.pop("MUJOCO_GL", None)

import imageio.v2 as imageio
import mujoco
import numpy as np


DEFAULT_DIT4DIT_ROOT = Path("/home/ykj/project/WM/DiT4DiT")
DEFAULT_DATASET = Path(
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/grab_red_bottle_test"
)
DEFAULT_MODEL = Path(
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/"
    "decoupled_wbc/control/robot_model/model_data/g1/scene_29dof_inspire.xml"
)
LEFT_ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True, help="GWP rollout.npz")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--source", choices=("pred", "gt"), default="pred")
    parser.add_argument("--dit4dit-root", type=Path, default=DEFAULT_DIT4DIT_ROOT)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--position-tolerance", type=float, default=2e-5)
    parser.add_argument("--orientation-tolerance", type=float, default=2e-4)
    return parser


def load_dit4dit_helpers(root: Path) -> SimpleNamespace:
    root = root.expanduser().resolve()
    if not (root / "examples").is_dir():
        raise FileNotFoundError(f"DiT4DiT checkout not found: {root}")
    sys.path.insert(0, str(root))
    try:
        from examples.PipetteRightOnly.convert_wrist_delta_to_joint_chunks import (
            BODY_NAMES,
            RightArmIK,
            rotation_6d_to_matrix,
        )
        from examples.Real_G1.joint_angle_deploy.render_collected_episode_mujoco import (
            joint_qpos_indices,
            load_episode,
            overlay,
            set_pose,
            validate_body_limits,
        )
    except ImportError as error:
        raise ImportError(f"cannot import DiT4DiT MuJoCo helpers from {root}: {error}") from error
    return SimpleNamespace(
        BODY_NAMES=BODY_NAMES,
        RightArmIK=RightArmIK,
        rotation_6d_to_matrix=rotation_6d_to_matrix,
        joint_qpos_indices=joint_qpos_indices,
        load_episode=load_episode,
        overlay=overlay,
        set_pose=set_pose,
        validate_body_limits=validate_body_limits,
    )


def make_left_arm_ik(base_class, body_names: list[str]):
    """Reuse DiT4DiT's RightArmIK.solve with left-arm model indices."""

    class LeftArmIK(base_class):
        def __init__(self, model_path: Path, args: argparse.Namespace):
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.data = mujoco.MjData(self.model)
            self.args = args
            self.wrist_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
            )
            self.pelvis_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
            )
            self.body_qpos = np.array([self._qpos_address(name) for name in body_names])
            self.arm_qpos = np.array([self._qpos_address(name) for name in LEFT_ARM_NAMES])
            self.arm_dofs = np.array([self._dof_address(name) for name in LEFT_ARM_NAMES])
            joint_ids = np.array(
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in LEFT_ARM_NAMES
                ]
            )
            if self.wrist_body < 0 or self.pelvis_body < 0 or np.any(joint_ids < 0):
                raise ValueError("left arm, wrist, or pelvis is missing from the MuJoCo model")
            self.lower = self.model.jnt_range[joint_ids, 0] + 1e-6
            self.upper = self.model.jnt_range[joint_ids, 1] - 1e-6

    return LeftArmIK


def load_rollout(path: Path, source: str) -> tuple[np.ndarray, np.ndarray, int]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as record:
        missing = {"starts", source, "state"}.difference(record.files)
        if missing:
            raise ValueError(f"rollout is missing arrays: {sorted(missing)}")
        starts = np.asarray(record["starts"], dtype=np.int64)
        chunks = np.asarray(record[source], dtype=np.float64)
        state_length = len(record["state"])
    if starts.ndim != 1 or chunks.ndim != 3 or chunks.shape[:1] != starts.shape:
        raise ValueError(f"invalid starts/chunks shapes: {starts.shape}, {chunks.shape}")
    if chunks.shape[2] != 30 or not np.isfinite(chunks).all():
        raise ValueError(f"{source} must contain finite [chunks,horizon,30], got {chunks.shape}")
    if len(starts) == 0 or starts[0] < 0 or np.any(np.diff(starts) <= 0):
        raise ValueError("starts must be a non-empty, strictly increasing array")
    return starts, chunks, state_length


def integrate_rollout(
    helpers: SimpleNamespace,
    args: argparse.Namespace,
    source_actions: np.ndarray,
    starts: np.ndarray,
    chunks: np.ndarray,
) -> tuple[np.ndarray, dict]:
    model_path = args.model_xml.expanduser().resolve()
    LeftArmIK = make_left_arm_ik(helpers.RightArmIK, helpers.BODY_NAMES)
    left_ik = LeftArmIK(model_path, args)
    right_ik = helpers.RightArmIK(model_path, args)

    last_frame = min(
        len(source_actions) - 1,
        max(int(start) + chunks.shape[1] for start in starts),
    )
    trajectory = source_actions[: last_frame + 1].copy()
    position_residuals: list[float] = []
    orientation_residuals: list[float] = []
    iteration_counts: list[int] = []
    clipped_hand_values = 0

    for chunk_index, start_value in enumerate(starts):
        start = int(start_value)
        if start >= len(source_actions) - 1:
            continue
        trajectory[start] = source_actions[start]
        left_ik.set_source_state(source_actions[start])
        right_ik.set_source_state(source_actions[start])
        left_position, left_rotation = left_ik.wrist_pose_in_pelvis()
        right_position, right_rotation = right_ik.wrist_pose_in_pelvis()
        left_pelvis_position, left_pelvis_rotation = left_ik.pelvis_pose()
        right_pelvis_position, right_pelvis_rotation = right_ik.pelvis_pose()
        left_hand = source_actions[start, 29:35].copy()
        right_hand = source_actions[start, 35:41].copy()
        horizon = min(len(chunks[chunk_index]), len(trajectory) - start - 1)

        for offset, action in enumerate(chunks[chunk_index, :horizon], start=1):
            left_position = left_position + left_rotation @ action[0:3]
            right_position = right_position + right_rotation @ action[9:12]
            # GWP stores the first two matrix columns; DiT4DiT stores rows.
            left_rotation = left_rotation @ helpers.rotation_6d_to_matrix(action[3:9]).T
            right_rotation = right_rotation @ helpers.rotation_6d_to_matrix(action[12:18]).T

            left_joints, left_pos_error, left_rot_error, left_iterations = left_ik.solve(
                left_pelvis_position + left_pelvis_rotation @ left_position,
                left_pelvis_rotation @ left_rotation,
            )
            right_joints, right_pos_error, right_rot_error, right_iterations = right_ik.solve(
                right_pelvis_position + right_pelvis_rotation @ right_position,
                right_pelvis_rotation @ right_rotation,
            )
            raw_left_hand = left_hand + action[18:24]
            raw_right_hand = right_hand + action[24:30]
            clipped_hand_values += int(
                np.count_nonzero((raw_left_hand < 0.0) | (raw_left_hand > 1.0))
                + np.count_nonzero((raw_right_hand < 0.0) | (raw_right_hand > 1.0))
            )
            left_hand = np.clip(raw_left_hand, 0.0, 1.0)
            right_hand = np.clip(raw_right_hand, 0.0, 1.0)

            frame_index = start + offset
            sample = source_actions[frame_index].copy()
            sample[15:22] = left_joints
            sample[22:29] = right_joints
            sample[29:35] = left_hand
            sample[35:41] = right_hand
            trajectory[frame_index] = sample
            position_residuals.extend((left_pos_error, right_pos_error))
            orientation_residuals.extend((left_rot_error, right_rot_error))
            iteration_counts.extend((left_iterations, right_iterations))

        print(
            f"[IK] chunk {chunk_index + 1}/{len(starts)} start={start} horizon={horizon}",
            flush=True,
        )

    position_array = np.asarray(position_residuals, dtype=np.float64)
    orientation_array = np.asarray(orientation_residuals, dtype=np.float64)
    iterations_array = np.asarray(iteration_counts, dtype=np.int64)
    metrics = {
        "frames": len(trajectory),
        "chunks": len(starts),
        "ik_position_residual_m": {
            "mean": float(position_array.mean()),
            "p99": float(np.quantile(position_array, 0.99)),
            "max": float(position_array.max()),
        },
        "ik_orientation_residual_rad": {
            "mean": float(orientation_array.mean()),
            "p99": float(np.quantile(orientation_array, 0.99)),
            "max": float(orientation_array.max()),
        },
        "ik_iterations": {
            "mean": float(iterations_array.mean()),
            "max": int(iterations_array.max()),
        },
        "clipped_hand_values": clipped_hand_values,
    }
    return trajectory, metrics


def render(
    helpers: SimpleNamespace,
    args: argparse.Namespace,
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    dataset_fps: float,
    starts: np.ndarray,
    metrics: dict,
) -> Path | None:
    model = mujoco.MjModel.from_xml_path(str(args.model_xml.expanduser().resolve()))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    body_indices, left_hand_indices, right_hand_indices = helpers.joint_qpos_indices(model)
    helpers.validate_body_limits(model, body_indices, trajectory[:, :29])

    relative_timestamps = timestamps[: len(trajectory)] - timestamps[0]
    duration = float(relative_timestamps[-1]) / args.speed

    def set_frame(frame_index: int) -> None:
        helpers.set_pose(
            data,
            trajectory[frame_index],
            body_indices,
            left_hand_indices,
            right_hand_indices,
            True,
        )
        mujoco.mj_forward(model, data)

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance = 2.15
            viewer.cam.azimuth = 150
            viewer.cam.elevation = -12
            started = time.monotonic()
            while viewer.is_running():
                elapsed = (time.monotonic() - started) * args.speed
                frame_index = min(
                    int(np.searchsorted(relative_timestamps, elapsed)), len(trajectory) - 1
                )
                set_frame(frame_index)
                viewer.sync()
                if frame_index == len(trajectory) - 1:
                    break
                time.sleep(1.0 / args.fps)
        return None

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.rollout.expanduser().resolve().with_name(f"mujoco_{args.source}.mp4")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 2.15
    camera.azimuth = 150
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8)
    render_frames = round(duration * args.fps) + 1
    try:
        for render_index in range(render_frames):
            recorded_time = render_index / args.fps * args.speed
            frame_index = min(
                int(np.searchsorted(relative_timestamps, recorded_time)), len(trajectory) - 1
            )
            set_frame(frame_index)
            renderer.update_scene(data, camera=camera)
            chunk_index = max(0, int(np.searchsorted(starts, frame_index, side="right") - 1))
            frame = helpers.overlay(
                renderer.render(),
                [
                    f"GWP {args.source.upper()} | episode {args.episode:06d} | chunk {chunk_index + 1}/{len(starts)}",
                    f"t={recorded_time:5.2f}/{relative_timestamps[-1]:5.2f}s  frame={frame_index + 1}/{len(trajectory)}",
                    f"IK max residual: {metrics['ik_position_residual_m']['max']:.4f} m / "
                    f"{metrics['ik_orientation_residual_rad']['max']:.4f} rad",
                ],
            )
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()
    print(f"[render] wrote {render_frames} frames to {output}")
    return output


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode < 0 or args.max_iterations <= 0:
        raise SystemExit("episode must be non-negative and max-iterations must be positive")
    if min(
        args.speed,
        args.fps,
        args.width,
        args.height,
        args.damping,
        args.max_step,
        args.position_tolerance,
        args.orientation_tolerance,
    ) <= 0:
        raise SystemExit("timing, image-size, and IK parameters must be positive")

    helpers = load_dit4dit_helpers(args.dit4dit_root)
    source_actions, timestamps, dataset_fps, parquet = helpers.load_episode(
        args.dataset.expanduser().resolve(), args.episode, "action"
    )
    starts, chunks, state_length = load_rollout(args.rollout, args.source)
    if state_length != len(source_actions):
        raise ValueError(
            f"rollout/source episode length mismatch: {state_length} != {len(source_actions)}"
        )
    if starts[-1] >= len(source_actions):
        raise ValueError(f"last chunk starts outside episode: {starts[-1]} >= {len(source_actions)}")

    print(f"rollout: {args.rollout.expanduser().resolve()}")
    print(f"source parquet: {parquet}")
    print(f"action source: {args.source}, chunks={len(starts)}, horizon={chunks.shape[1]}")
    trajectory, metrics = integrate_rollout(helpers, args, source_actions, starts, chunks)
    output = render(helpers, args, trajectory, timestamps, dataset_fps, starts, metrics)
    summary = {
        "rollout": str(args.rollout.expanduser().resolve()),
        "source_parquet": str(parquet),
        "dit4dit_root": str(args.dit4dit_root.expanduser().resolve()),
        "model_xml": str(args.model_xml.expanduser().resolve()),
        "episode": args.episode,
        "action_source": args.source,
        "dataset_fps": dataset_fps,
        "output": str(output) if output is not None else None,
        **metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if output is not None:
        with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

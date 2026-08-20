#!/usr/bin/env python3
"""Render a recursively integrated left-arm/left-Inspire rollout in MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
elif os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
    os.environ.pop("MUJOCO_GL", None)

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_MODEL = Path(
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/"
    "decoupled_wbc/control/robot_model/model_data/g1/scene_29dof_inspire.xml"
)
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
LEFT_HAND_JOINTS = (
    "left_hand_little_joint",
    "left_hand_ring_joint",
    "left_hand_middle_joint",
    "left_hand_index_joint",
    "left_hand_thumb_rotate_joint",
    "left_hand_thumb_bend_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--source-fps", type=float, default=50.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--pred-only", action="store_true")
    parser.add_argument("--show-right-hand", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as record:
        required = {"predicted_state", "reference_state", "frame_indices"}
        missing = required.difference(record.files)
        if missing:
            raise ValueError(f"rollout is missing arrays: {sorted(missing)}")
        predicted = np.asarray(record["predicted_state"], dtype=np.float64)
        reference = np.asarray(record["reference_state"], dtype=np.float64)
        frame_indices = np.asarray(record["frame_indices"], dtype=np.int64)
    if predicted.ndim != 2 or predicted.shape[1] != 13 or reference.shape != predicted.shape:
        raise ValueError(f"expected matching finite [T,13] trajectories, got {predicted.shape}/{reference.shape}")
    if frame_indices.shape != (len(predicted),) or len(predicted) < 2:
        raise ValueError(f"invalid frame indices: {frame_indices.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("rollout contains non-finite states")
    return predicted, reference, frame_indices


def load_arm_semantics(path: Path) -> str:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return "unknown_action_representation"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    semantics = manifest.get("rollout_semantics", {}).get("left_arm_channels_0_7")
    return str(semantics or "unknown_action_representation")


def qpos_indices(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        missing = [name for name, joint_id in zip(names, joint_ids) if joint_id < 0]
        raise ValueError(f"joints missing from MuJoCo model: {missing}")
    return model.jnt_qposadr[joint_ids].copy(), model.jnt_range[joint_ids, 0].copy(), model.jnt_range[joint_ids, 1].copy()


def inspire_dds_to_mujoco(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.array(
        [
            (1.0 - values[0]) * 1.7,
            (1.0 - values[1]) * 1.7,
            (1.0 - values[2]) * 1.7,
            (1.0 - values[3]) * 1.7,
            (1.0 - values[5]) * 1.4 - 0.1,
            (1.0 - values[4]) * 0.5,
        ],
        dtype=np.float64,
    )


def hide_right_inspire(model: mujoco.MjModel) -> int:
    hidden = 0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("right_hand_"):
            model.geom_rgba[geom_id, 3] = 0.0
            hidden += 1
    return hidden


def set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: np.ndarray,
    arm_qpos: np.ndarray,
    arm_low: np.ndarray,
    arm_high: np.ndarray,
    hand_qpos: np.ndarray,
) -> tuple[int, int]:
    arm = np.asarray(state[:7], dtype=np.float64)
    hand = np.asarray(state[7:13], dtype=np.float64)
    arm_clipped = int(np.count_nonzero((arm < arm_low) | (arm > arm_high)))
    hand_clipped = int(np.count_nonzero((hand < 0.0) | (hand > 1.0)))
    data.qpos[arm_qpos] = np.clip(arm, arm_low, arm_high)
    data.qpos[hand_qpos] = inspire_dds_to_mujoco(hand)
    mujoco.mj_forward(model, data)
    return arm_clipped, hand_clipped


def interpolate(states: np.ndarray, position: float) -> np.ndarray:
    left = min(int(np.floor(position)), len(states) - 1)
    right = min(left + 1, len(states) - 1)
    alpha = float(position - left)
    return states[left] * (1.0 - alpha) + states[right] * alpha


def overlay(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, image.width - 12, 124), fill=(8, 8, 8))
    draw.text((24, 22), title, fill=(90, 220, 255))
    for index, line in enumerate(lines):
        draw.text((24, 48 + 23 * index), line, fill=(235, 235, 235))
    return np.asarray(image)


def configure_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 2.15
    camera.azimuth = 150
    camera.elevation = -12
    return camera


def main() -> None:
    args = parse_args()
    if min(args.fps, args.source_fps, args.speed, args.panel_width, args.height) <= 0:
        raise SystemExit("FPS, speed, width and height must be positive")
    rollout = args.rollout.expanduser().resolve()
    model_xml = args.model_xml.expanduser().resolve()
    predicted, reference, frame_indices = load_rollout(rollout)
    arm_semantics = load_arm_semantics(rollout)
    if not model_xml.is_file():
        raise FileNotFoundError(model_xml)

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    model.vis.global_.offwidth = args.panel_width
    model.vis.global_.offheight = args.height
    hidden_right_geoms = 0 if args.show_right_hand else hide_right_inspire(model)
    arm_qpos, arm_low, arm_high = qpos_indices(model, LEFT_ARM_JOINTS)
    hand_qpos, _, _ = qpos_indices(model, LEFT_HAND_JOINTS)
    predicted_data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    camera = configure_camera()

    duration = (len(predicted) - 1) / args.source_fps / args.speed
    output = (
        args.output.expanduser().resolve()
        if args.output
        else rollout.with_name(f"{rollout.stem}_mujoco.mp4")
    )
    print(f"rollout: {rollout}")
    print(f"model: {model_xml}")
    print(f"frames: {len(predicted)}, duration: {duration:.2f}s at {args.speed:g}x")
    print(f"left-arm action semantics: {arm_semantics}")
    print(f"right Inspire geoms hidden: {hidden_right_geoms}")

    if args.viewer:
        from mujoco import viewer as mj_viewer

        try:
            with mj_viewer.launch_passive(model, predicted_data) as viewer:
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
                while viewer.is_running():
                    started = time.monotonic()
                    for index, state in enumerate(predicted):
                        if not viewer.is_running():
                            break
                        arm_clip, hand_clip = set_state(
                            model, predicted_data, state, arm_qpos, arm_low, arm_high, hand_qpos
                        )
                        viewer.sync()
                        if index % 50 == 0 or arm_clip or hand_clip:
                            print(
                                f"frame={frame_indices[index]} arm_clipped={arm_clip} "
                                f"hand_clipped={hand_clip}"
                            )
                        time.sleep(max(0.0, 1.0 / args.source_fps / args.speed))
                    if not args.loop:
                        break
                    elapsed = time.monotonic() - started
                    print(f"loop completed in {elapsed:.2f}s")
        except KeyboardInterrupt:
            print("viewer interrupted")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    predicted_renderer = mujoco.Renderer(model, height=args.height, width=args.panel_width)
    reference_renderer = (
        None
        if args.pred_only
        else mujoco.Renderer(model, height=args.height, width=args.panel_width)
    )
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8)
    render_frames = round(duration * args.fps) + 1
    total_arm_clips = 0
    total_hand_clips = 0
    try:
        for render_index in range(render_frames):
            source_position = min(
                render_index / args.fps * args.speed * args.source_fps,
                len(predicted) - 1,
            )
            predicted_state = interpolate(predicted, source_position)
            arm_clip, hand_clip = set_state(
                model,
                predicted_data,
                predicted_state,
                arm_qpos,
                arm_low,
                arm_high,
                hand_qpos,
            )
            total_arm_clips += arm_clip
            total_hand_clips += hand_clip
            predicted_renderer.update_scene(predicted_data, camera=camera)
            source_index = min(int(round(source_position)), len(predicted) - 1)
            pred_frame = overlay(
                predicted_renderer.render(),
                "MODEL | recursive predicted state",
                [
                    f"dataset frame={frame_indices[source_index]}  t={source_position / args.source_fps:.2f}s",
                    f"arm={arm_semantics}; hand=absolute",
                    f"range clips: arm={arm_clip} hand={hand_clip}",
                ],
            )
            if reference_renderer is None:
                writer.append_data(pred_frame)
                continue
            reference_state = interpolate(reference, source_position)
            set_state(
                model,
                reference_data,
                reference_state,
                arm_qpos,
                arm_low,
                arm_high,
                hand_qpos,
            )
            reference_renderer.update_scene(reference_data, camera=camera)
            ref_frame = overlay(
                reference_renderer.render(),
                "REFERENCE | visualization only",
                [
                    f"dataset frame={frame_indices[source_index]}  t={source_position / args.source_fps:.2f}s",
                    "not fed to policy after the initial state",
                    "reference RGB is fed to the policy each step",
                ],
            )
            writer.append_data(np.concatenate([pred_frame, ref_frame], axis=1))
    finally:
        writer.close()
        predicted_renderer.close()
        if reference_renderer is not None:
            reference_renderer.close()

    summary = {
        "rollout": str(rollout),
        "video": str(output),
        "model_xml": str(model_xml),
        "frames": len(predicted),
        "render_frames": render_frames,
        "duration_seconds": duration,
        "right_inspire_hidden": not args.show_right_hand,
        "left_arm_action_semantics": arm_semantics,
        "arm_limit_clip_events": total_arm_clips,
        "hand_range_clip_events": total_hand_clips,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

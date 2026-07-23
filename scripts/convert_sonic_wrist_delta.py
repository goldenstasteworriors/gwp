#!/usr/bin/env python3
"""Convert SONIC G1 LeRobot data to the 30-D wrist/hand action space.

The converted per-frame action is the transition from the WBC target at frame
``t`` to the WBC target at frame ``t + 1``:

    [left wrist local translation (3), left wrist relative rotation-6D (6),
     right wrist local translation (3), right wrist relative rotation-6D (6),
     left hand joint delta (6), right hand joint delta (6)]

All action dimensions are already relative, so the WAM delta mask must be all
False.  The final frame in each episode has a zero action.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


STATE_NAMES = [
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "left_wrist_r00",
    "left_wrist_r10",
    "left_wrist_r20",
    "left_wrist_r01",
    "left_wrist_r11",
    "left_wrist_r21",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "right_wrist_r00",
    "right_wrist_r10",
    "right_wrist_r20",
    "right_wrist_r01",
    "right_wrist_r11",
    "right_wrist_r21",
    "left_hand_little",
    "left_hand_ring",
    "left_hand_middle",
    "left_hand_index",
    "left_hand_thumb_bend",
    "left_hand_thumb_rotate",
    "right_hand_little",
    "right_hand_ring",
    "right_hand_middle",
    "right_hand_index",
    "right_hand_thumb_bend",
    "right_hand_thumb_rotate",
]

ACTION_NAMES = [
    name.replace("wrist_", "wrist_delta_") if "wrist_" in name else f"{name}_delta"
    for name in STATE_NAMES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sonic-root",
        required=True,
        type=Path,
        help="GR00T-WholeBodyControl checkout containing gear_sonic.",
    )
    return parser.parse_args()


def rot6d(matrix: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns, matching SONIC's convention."""
    return matrix[:, :2].T.reshape(6)


def load_robot_model(sonic_root: Path):
    sys.path.insert(0, str(sonic_root))
    from gear_sonic.data.features_sonic_vla import get_g1_robot_model

    return get_g1_robot_model()


def wrist_poses(robot_model, actions: np.ndarray, source_names: list[str]):
    model_names = list(robot_model.joint_names)
    source_lookup = {name: i for i, name in enumerate(source_names)}
    body_pairs = [
        (model_names.index(name), source_lookup[name])
        for name in source_names[:29]
        if name in model_names
    ]
    if len(body_pairs) != 29:
        raise ValueError(f"Expected 29 mapped body joints, found {len(body_pairs)}")

    positions = np.empty((len(actions), 2, 3), dtype=np.float32)
    rotations = np.empty((len(actions), 2, 3, 3), dtype=np.float32)
    frame_names = robot_model.supplemental_info.hand_frame_names
    q = np.asarray(robot_model.q_zero, dtype=np.float64).copy()
    for row, action in enumerate(actions):
        for model_idx, source_idx in body_pairs:
            q[model_idx] = action[source_idx]
        robot_model.cache_forward_kinematics(q)
        for side_idx, side in enumerate(("left", "right")):
            placement = robot_model.frame_placement(frame_names[side])
            positions[row, side_idx] = placement.translation
            rotations[row, side_idx] = placement.rotation
    return positions, rotations


def convert_episode(
    source: Path,
    destination: Path,
    robot_model,
    source_names: list[str],
) -> int:
    frame = pd.read_parquet(source)
    observed_eef = np.stack(frame["observation.eef_state"]).astype(np.float64)
    observed_state = np.stack(frame["observation.state"]).astype(np.float64)
    wbc_action = np.stack(frame["action.wbc"]).astype(np.float64)

    state = np.empty((len(frame), 30), dtype=np.float32)
    for side, (pos_slice, quat_slice, out_start) in enumerate(
        ((slice(0, 3), slice(3, 7), 0), (slice(7, 10), slice(10, 14), 9))
    ):
        del side
        state[:, out_start : out_start + 3] = observed_eef[:, pos_slice]
        quat_wxyz = observed_eef[:, quat_slice]
        matrices = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        state[:, out_start + 3 : out_start + 9] = np.stack([rot6d(r) for r in matrices])
    state[:, 18:24] = observed_state[:, 29:35]
    state[:, 24:30] = observed_state[:, 35:41]

    target_pos, target_rot = wrist_poses(robot_model, wbc_action, source_names)
    action = np.zeros((len(frame), 30), dtype=np.float32)
    identity_rot6d = rot6d(np.eye(3, dtype=np.float32))
    action[:, 3:9] = identity_rot6d
    action[:, 12:18] = identity_rot6d
    if len(frame) > 1:
        for t in range(len(frame) - 1):
            for side, out_start in ((0, 0), (1, 9)):
                current_r = target_rot[t, side]
                relative_r = current_r.T @ target_rot[t + 1, side]
                relative_p = current_r.T @ (target_pos[t + 1, side] - target_pos[t, side])
                action[t, out_start : out_start + 3] = relative_p
                action[t, out_start + 3 : out_start + 9] = rot6d(relative_r)
            action[t, 18:24] = wbc_action[t + 1, 29:35] - wbc_action[t, 29:35]
            action[t, 24:30] = wbc_action[t + 1, 35:41] - wbc_action[t, 35:41]

    keep = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    converted = frame[keep].copy()
    converted.insert(0, "action", list(action))
    converted.insert(0, "observation.state", list(state))
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(destination, index=False)
    return len(frame)


def main() -> None:
    args = parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    sonic_root = args.sonic_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    info_path = source_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_names = list(info["features"]["action.wbc"]["names"])
    if len(source_names) != 41:
        raise ValueError(f"Expected 41-D Inspire action, found {len(source_names)}")

    robot_model = load_robot_model(sonic_root)
    shutil.copytree(source_root / "meta", output_root / "meta")
    shutil.copytree(source_root / "videos", output_root / "videos")

    total_frames = 0
    parquet_paths = sorted((source_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files under {source_root / 'data'}")
    for episode_idx, source in enumerate(parquet_paths):
        relative = source.relative_to(source_root)
        total_frames += convert_episode(
            source,
            output_root / relative,
            robot_model,
            source_names,
        )
        if (episode_idx + 1) % 10 == 0 or episode_idx + 1 == len(parquet_paths):
            print(f"Converted {episode_idx + 1}/{len(parquet_paths)} episodes")

    info["robot_type"] = "sonic_g1"
    info["features"] = {
        "observation.images.ego_view": info["features"]["observation.images.ego_view"],
        "observation.state": {
            "dtype": "float32",
            "shape": [30],
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [30],
            "names": ACTION_NAMES,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info["total_frames"] = total_frames
    (output_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n",
        encoding="utf-8",
    )
    modality = {
        "state": {"wrist_hand_state": {"start": 0, "end": 30}},
        "action": {"wrist_hand_delta": {"start": 0, "end": 30}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (output_root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=4) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_root}: episodes={len(parquet_paths)} frames={total_frames}")


if __name__ == "__main__":
    main()

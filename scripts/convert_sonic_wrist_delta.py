#!/usr/bin/env python3
"""Convert SONIC G1 LeRobot data to a selected wrist/hand action space.

The converted per-frame action is the transition from the WBC target at frame
``t`` to the WBC target at frame ``t + 1``:

    [left wrist local translation (3), left wrist relative rotation-6D (6),
     right wrist local translation (3), right wrist relative rotation-6D (6),
     left hand joint delta (6), right hand joint delta (6)]

The default ``legacy30`` schema preserves the existing two-Inspire-hand dataset
exactly.  The opt-in ``asymmetric25`` schema is a distinct hardware contract:
left/right wrist ``9+9``, left RH56DFTP targets ``6`` and right OmniPicker
aperture target ``1``.  It requires an explicit hardware manifest and named
source channels, so legacy right-hand joints cannot be silently truncated into
an OmniPicker command.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_ACTION_NAMES,
    ASYMMETRIC_SCHEMA_ID,
    ASYMMETRIC_STATE_NAMES,
    LEGACY_SCHEMA_ID,
    pack_asymmetric_physical,
)
from world_action_model.asymmetric_hardware_manifest import (
    calibration_capture_sha256,
    canonicalize_hand_signal,
    hand_channels,
    load_hardware_manifest,
    validate_asymmetric_action,
    validate_episode_timing,
    validate_manifest_against_source_info,
    validate_manifest_capture_binding,
)


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
    parser.add_argument(
        "--action-schema",
        choices=("legacy30", "asymmetric25"),
        default="legacy30",
        help="Defaults to the unchanged 30-D two-Inspire-hand representation.",
    )
    parser.add_argument(
        "--hardware-manifest",
        type=Path,
        default=None,
        help="Required for asymmetric25; declares named RH56DFTP and OmniPicker source channels.",
    )
    parser.add_argument(
        "--calibration-capture",
        type=Path,
        default=None,
        help="Required for asymmetric25; proves calibration digest, devices and verification level.",
    )
    return parser.parse_args()


def _load_asymmetric_manifest(
    path: Path | None,
    calibration_capture_path: Path | None,
    info: dict,
) -> dict:
    if path is None:
        raise ValueError("--hardware-manifest is required with --action-schema asymmetric25")
    if calibration_capture_path is None:
        raise ValueError("--calibration-capture is required with --action-schema asymmetric25")
    manifest = load_hardware_manifest(path, mode="training")
    capture = json.loads(calibration_capture_path.resolve().read_text(encoding="utf-8"))
    validate_manifest_capture_binding(manifest, capture)
    layout = validate_manifest_against_source_info(manifest, info)
    layout["calibration_capture_sha256"] = calibration_capture_sha256(capture)
    return layout


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
    *,
    action_schema: str = "legacy30",
    asymmetric_layout: dict | None = None,
) -> int:
    if action_schema not in {"legacy30", "asymmetric25"}:
        raise ValueError(f"unsupported action_schema={action_schema!r}")
    frame = pd.read_parquet(source)
    wrist_key = (
        "observation.eef_state"
        if asymmetric_layout is None
        else asymmetric_layout["wrist_state_feature"]
    )
    observed_eef = np.stack(frame[wrist_key]).astype(np.float64)
    state_key = "observation.state" if asymmetric_layout is None else asymmetric_layout["state_feature"]
    command_key = "action.wbc" if asymmetric_layout is None else asymmetric_layout["command_feature"]
    observed_state = np.stack(frame[state_key]).astype(np.float64)
    wbc_action = np.stack(frame[command_key]).astype(np.float64)
    if len(frame) == 0:
        raise ValueError(f"empty episode: {source}")

    physical_dim = 30 if action_schema == "legacy30" else ASYMMETRIC_ACTION_DIM
    state = np.empty((len(frame), physical_dim), dtype=np.float32)
    for side, (pos_slice, quat_slice, out_start) in enumerate(
        ((slice(0, 3), slice(3, 7), 0), (slice(7, 10), slice(10, 14), 9))
    ):
        del side
        state[:, out_start : out_start + 3] = observed_eef[:, pos_slice]
        quat_wxyz = observed_eef[:, quat_slice]
        matrices = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        state[:, out_start + 3 : out_start + 9] = np.stack([rot6d(r) for r in matrices])
    if action_schema == "legacy30":
        state[:, 18:24] = observed_state[:, 29:35]
        state[:, 24:30] = observed_state[:, 35:41]
    else:
        if asymmetric_layout is None:
            raise ValueError("asymmetric25 conversion requires a validated hardware layout")
        manifest = asymmetric_layout["manifest"]
        validate_episode_timing(
            frame[asymmetric_layout["timestamp_feature"]].to_numpy(),
            manifest,
            name=source.name,
        )
        left_channels, right_channel = hand_channels(manifest)
        left_state = canonicalize_hand_signal(
            observed_state[:, asymmetric_layout["left_state_indices"]],
            left_channels,
            signal="state",
            name="left RH56 state",
        )
        right_state = canonicalize_hand_signal(
            observed_state[:, [asymmetric_layout["right_state_index"]]],
            [right_channel],
            signal="state",
            name="right OmniPicker state",
        )
        state = pack_asymmetric_physical(
            state[:, :9],
            state[:, 9:18],
            left_state,
            right_state,
            name="asymmetric state",
        )

    target_pos, target_rot = wrist_poses(robot_model, wbc_action, source_names)
    action = np.zeros((len(frame), physical_dim), dtype=np.float32)
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
            if action_schema == "legacy30":
                action[t, 18:24] = wbc_action[t + 1, 29:35] - wbc_action[t, 29:35]
                action[t, 24:30] = wbc_action[t + 1, 35:41] - wbc_action[t, 35:41]

    if action_schema == "asymmetric25":
        assert asymmetric_layout is not None
        manifest = asymmetric_layout["manifest"]
        left_channels, right_channel = hand_channels(manifest)
        # Wrist commands stay local relative transforms. Hand commands are
        # absolute normalized actuator/aperture targets, matching the human
        # grip-target semantics instead of inventing a six-channel right hand.
        target_indices = np.minimum(np.arange(len(frame)) + 1, len(frame) - 1)
        left_target = canonicalize_hand_signal(
            wbc_action[target_indices][:, asymmetric_layout["left_action_indices"]],
            left_channels,
            signal="command",
            name="left RH56 command",
        )
        right_target = canonicalize_hand_signal(
            wbc_action[target_indices][:, [asymmetric_layout["right_action_index"]]],
            [right_channel],
            signal="command",
            name="right OmniPicker command",
        )
        action = pack_asymmetric_physical(
            action[:, :9],
            action[:, 9:18],
            left_target,
            right_target,
            name="asymmetric action",
        )
        validate_asymmetric_action(action, manifest, name=f"{source.name} action")

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
    asymmetric_layout = (
        _load_asymmetric_manifest(args.hardware_manifest, args.calibration_capture, info)
        if args.action_schema == "asymmetric25"
        else None
    )
    command_key = "action.wbc" if asymmetric_layout is None else asymmetric_layout["command_feature"]
    source_names = list(info["features"][command_key]["names"])
    if args.action_schema == "legacy30" and len(source_names) != 41:
        raise ValueError(f"Expected 41-D Inspire action, found {len(source_names)}")

    robot_model = load_robot_model(sonic_root)
    (output_root / "meta").mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    video_key = "observation.images.ego_view"
    (output_root / "videos" / video_key / "chunk-000").mkdir(parents=True)

    tasks = [
        json.loads(line)
        for line in (source_root / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    task_frame = pd.DataFrame(
        {"task_index": [item["task_index"] for item in tasks]},
        index=pd.Index([item["task"] for item in tasks]),
    )
    task_frame.to_parquet(output_root / "meta" / "tasks.parquet")
    source_episodes = {
        item["episode_index"]: item
        for item in (
            json.loads(line)
            for line in (source_root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
    }

    total_frames = 0
    episode_rows = []
    parquet_paths = sorted((source_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files under {source_root / 'data'}")
    for episode_idx, source in enumerate(parquet_paths):
        frame_count = convert_episode(
            source,
            output_root / "data" / "chunk-000" / f"file-{episode_idx:03d}.parquet",
            robot_model,
            source_names,
            action_schema=args.action_schema,
            asymmetric_layout=asymmetric_layout,
        )
        source_video = next(
            (source_root / "videos").glob(
                f"*/{video_key}/episode_{episode_idx:06d}.mp4"
            )
        )
        shutil.copy2(
            source_video,
            output_root
            / "videos"
            / video_key
            / "chunk-000"
            / f"file-{episode_idx:03d}.mp4",
        )
        episode_info = source_episodes[episode_idx]
        episode_rows.append(
            {
                "episode_index": episode_idx,
                "tasks": np.asarray(episode_info["tasks"], dtype=object),
                "length": frame_count,
                "data/chunk_index": 0,
                "data/file_index": episode_idx,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + frame_count,
                f"videos/{video_key}/chunk_index": 0,
                f"videos/{video_key}/file_index": episode_idx,
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": frame_count / info["fps"],
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        total_frames += frame_count
        if (episode_idx + 1) % 10 == 0 or episode_idx + 1 == len(parquet_paths):
            print(f"Converted {episode_idx + 1}/{len(parquet_paths)} episodes")

    info["robot_type"] = "sonic_g1"
    info["codebase_version"] = "v3.1"
    info["action_schema_id"] = (
        LEGACY_SCHEMA_ID if args.action_schema == "legacy30" else ASYMMETRIC_SCHEMA_ID
    )
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    output_dim = 30 if args.action_schema == "legacy30" else ASYMMETRIC_ACTION_DIM
    output_state_names = STATE_NAMES if args.action_schema == "legacy30" else list(ASYMMETRIC_STATE_NAMES)
    output_action_names = ACTION_NAMES if args.action_schema == "legacy30" else list(ASYMMETRIC_ACTION_NAMES)
    info["features"] = {
        "observation.images.ego_view": info["features"]["observation.images.ego_view"],
        "observation.state": {
            "dtype": "float32",
            "shape": [output_dim],
            "names": output_state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [output_dim],
            "names": output_action_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info["total_frames"] = total_frames
    info["total_episodes"] = len(parquet_paths)
    info["total_chunks"] = 1
    info["total_videos"] = len(parquet_paths)
    info["chunks_size"] = max(len(parquet_paths), 1)
    if asymmetric_layout is not None:
        info["asymmetric_hardware_manifest"] = asymmetric_layout["manifest"]
        info["hardware_contract_sha256"] = asymmetric_layout["hardware_contract_sha256"]
        info["calibration_capture_sha256"] = asymmetric_layout[
            "calibration_capture_sha256"
        ]
    (output_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n",
        encoding="utf-8",
    )
    action_modality_name = "wrist_hand_delta" if args.action_schema == "legacy30" else "wrist_hand_action"
    modality = {
        "state": {"wrist_hand_state": {"start": 0, "end": output_dim}},
        "action": {action_modality_name: {"start": 0, "end": output_dim}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (output_root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=4) + "\n",
        encoding="utf-8",
    )
    episodes_dir = output_root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pd.DataFrame(episode_rows).to_parquet(
        episodes_dir / "file-000.parquet",
        index=False,
    )
    print(f"Wrote {output_root}: episodes={len(parquet_paths)} frames={total_frames}")


if __name__ == "__main__":
    main()

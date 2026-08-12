#!/usr/bin/env python3
"""Convert SONIC G1 recordings to absolute arm/hand joint targets.

``--side both`` emits left arm (7), right arm (7), left Inspire hand (6),
right Inspire hand (6). ``--side left`` emits only left arm (7) and left
Inspire hand (6). ``action.wbc`` is kept as an absolute target rather than
converted to a delta.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_SCHEMA_IDS = {
    "both": "sonic_g1_dual_arm_inspire6_absolute_joint_v1",
    "left": "sonic_g1_left_arm_inspire6_absolute_joint_v1",
}
VIDEO_KEY = "observation.images.ego_view"
LEFT_ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_HAND_NAMES = (
    "left_hand_little",
    "left_hand_ring",
    "left_hand_middle",
    "left_hand_index",
    "left_hand_thumb_bend",
    "left_hand_thumb_rotate",
)
RIGHT_HAND_NAMES = (
    "right_hand_little",
    "right_hand_ring",
    "right_hand_middle",
    "right_hand_index",
    "right_hand_thumb_bend",
    "right_hand_thumb_rotate",
)
JOINT_NAMES_BY_SIDE = {
    "both": LEFT_ARM_NAMES + RIGHT_ARM_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES,
    "left": LEFT_ARM_NAMES + LEFT_HAND_NAMES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--side", choices=("both", "left"), default="both")
    return parser.parse_args()


def _feature_indices(info: dict, key: str, joint_names: tuple[str, ...]) -> list[int]:
    names = list(info["features"][key]["names"])
    missing = [name for name in joint_names if name not in names]
    if missing:
        raise ValueError(f"{key} is missing required joints: {missing}")
    return [names.index(name) for name in joint_names]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    joint_names = JOINT_NAMES_BY_SIDE[args.side]
    output_dim = len(joint_names)
    state_indices = _feature_indices(info, "observation.state", joint_names)
    action_indices = _feature_indices(info, "action.wbc", joint_names)
    tasks = _read_jsonl(source_root / "meta" / "tasks.jsonl")
    source_episodes = {
        int(row["episode_index"]): row
        for row in _read_jsonl(source_root / "meta" / "episodes.jsonl")
    }

    data_dir = output_root / "data" / "chunk-000"
    video_dir = output_root / "videos" / VIDEO_KEY / "chunk-000"
    episodes_dir = output_root / "meta" / "episodes" / "chunk-000"
    for directory in (data_dir, video_dir, episodes_dir):
        directory.mkdir(parents=True, exist_ok=False)

    task_frame = pd.DataFrame(
        {"task_index": [row["task_index"] for row in tasks]},
        index=pd.Index([row["task"] for row in tasks]),
    )
    task_frame.to_parquet(output_root / "meta" / "tasks.parquet")

    total_frames = 0
    episode_rows = []
    right_hand_min = np.full(6, np.inf, dtype=np.float64) if args.side == "both" else None
    right_hand_max = np.full(6, -np.inf, dtype=np.float64) if args.side == "both" else None
    sources = sorted((source_root / "data").glob("*/*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No parquet episodes under {source_root / 'data'}")

    for output_index, source in enumerate(sources):
        frame = pd.read_parquet(source)
        episode_values = np.asarray(frame["episode_index"], dtype=np.int64)
        unique_episode = np.unique(episode_values)
        if unique_episode.size != 1:
            raise ValueError(f"Expected one episode in {source}, got {unique_episode.tolist()}")
        episode_index = int(unique_episode[0])
        if episode_index != output_index:
            raise ValueError(
                f"Episodes must be contiguous from zero; file {source.name} has {episode_index}, "
                f"expected {output_index}"
            )

        observed = np.stack(frame["observation.state"]).astype(np.float32)
        commanded = np.stack(frame["action.wbc"]).astype(np.float32)
        state = observed[:, state_indices]
        action = commanded[:, action_indices]
        if args.side == "both":
            right_hand_min = np.minimum(right_hand_min, action[:, -6:].min(axis=0))
            right_hand_max = np.maximum(right_hand_max, action[:, -6:].max(axis=0))

        converted = frame[
            ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        ].copy()
        converted.insert(0, "action", list(action))
        converted.insert(0, "observation.state", list(state))
        converted.to_parquet(data_dir / f"file-{episode_index:03d}.parquet", index=False)

        source_video = source_root / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        shutil.copy2(source_video, video_dir / f"file-{episode_index:03d}.mp4")

        length = len(frame)
        episode_info = source_episodes[episode_index]
        if int(episode_info["length"]) != length:
            raise ValueError(f"Episode {episode_index} length metadata does not match parquet")
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": np.asarray(episode_info["tasks"], dtype=object),
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + length,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": episode_index,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{VIDEO_KEY}/to_timestamp": length / float(info["fps"]),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        total_frames += length

    info["robot_type"] = "sonic_g1"
    info["codebase_version"] = "v3.1"
    info["action_schema_id"] = ACTION_SCHEMA_IDS[args.side]
    info["action_semantics"] = "absolute_joint_target"
    info["controlled_side"] = args.side
    if args.side == "both":
        info["right_hand_supervision"] = {
            "semantics": "recorded_default_open_pose",
            "names": list(RIGHT_HAND_NAMES),
            "min": right_hand_min.tolist(),
            "max": right_hand_max.tolist(),
        }
    else:
        info.pop("right_hand_supervision", None)
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"] = {
        VIDEO_KEY: info["features"][VIDEO_KEY],
        "observation.state": {"dtype": "float32", "shape": [output_dim], "names": list(joint_names)},
        "action": {"dtype": "float32", "shape": [output_dim], "names": list(joint_names)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info["total_frames"] = total_frames
    info["total_episodes"] = len(sources)
    info["total_chunks"] = 1
    info["total_videos"] = len(sources)
    info["chunks_size"] = max(len(sources), 1)
    (output_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )
    modality = {
        "state": {"arm_hand_joint_state": {"start": 0, "end": output_dim}},
        "action": {"arm_hand_joint_target": {"start": 0, "end": output_dim}},
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (output_root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=4) + "\n", encoding="utf-8"
    )
    pd.DataFrame(episode_rows).to_parquet(episodes_dir / "file-000.parquet", index=False)
    suffix = (
        f" right_hand_range={(right_hand_max - right_hand_min).tolist()}"
        if args.side == "both"
        else ""
    )
    print(
        f"Wrote {output_root}: episodes={len(sources)} frames={total_frames} "
        f"side={args.side} dim={output_dim}{suffix}"
    )


if __name__ == "__main__":
    main()

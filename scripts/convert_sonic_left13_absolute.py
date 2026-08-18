#!/usr/bin/env python3
"""Convert SONIC G1 LeRobot v2.1 data to left-arm/hand absolute 13-D v3.1."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_little",
    "left_hand_ring",
    "left_hand_middle",
    "left_hand_index",
    "left_hand_thumb_bend",
    "left_hand_thumb_rotate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def convert_episode(
    source: Path,
    destination: Path,
    state_indices: list[int],
    action_indices: list[int],
    episode_index: int,
    dataset_from_index: int,
) -> int:
    frame = pd.read_parquet(source)
    state_full = np.stack(frame["observation.state"]).astype(np.float32)
    action_full = np.stack(frame["action.wbc"]).astype(np.float32)
    state = state_full[:, state_indices]
    action = action_full[:, action_indices]
    if state.shape[1] != 13 or action.shape[1] != 13:
        raise ValueError(
            f"Expected 13-D state/action, got state={state.shape} action={action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"Non-finite state/action values in {source}")

    keep = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    missing = [name for name in keep if name not in frame]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")
    converted = frame[keep].copy()
    converted["frame_index"] = np.arange(len(frame), dtype=np.int64)
    converted["episode_index"] = episode_index
    converted["index"] = np.arange(
        dataset_from_index,
        dataset_from_index + len(frame),
        dtype=np.int64,
    )
    converted.insert(0, "action", list(action))
    converted.insert(0, "observation.state", list(state))
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(destination, index=False)
    return len(frame)


def main() -> None:
    args = parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    info = json.loads((source_root / "meta" / "info.json").read_text())
    state_names = list(info["features"]["observation.state"]["names"])
    action_names = list(info["features"]["action.wbc"]["names"])
    state_indices = [state_names.index(name) for name in ACTION_NAMES]
    action_indices = [action_names.index(name) for name in ACTION_NAMES]

    tasks = read_jsonl(source_root / "meta" / "tasks.jsonl")
    if not tasks:
        raise ValueError("Dataset has no tasks")
    task_frame = pd.DataFrame(
        {"task_index": [item["task_index"] for item in tasks]},
        index=pd.Index([item["task"] for item in tasks]),
    )

    source_episodes = {
        item["episode_index"]: item
        for item in read_jsonl(source_root / "meta" / "episodes.jsonl")
    }
    parquet_paths = sorted((source_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {source_root / 'data'}")

    (output_root / "meta").mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    video_key = "observation.images.ego_view"
    video_output_dir = output_root / "videos" / video_key / "chunk-000"
    video_output_dir.mkdir(parents=True)
    task_frame.to_parquet(output_root / "meta" / "tasks.parquet")

    total_frames = 0
    episode_rows = []
    for episode_index, source in enumerate(parquet_paths):
        frame_count = convert_episode(
            source,
            output_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet",
            state_indices,
            action_indices,
            episode_index,
            total_frames,
        )
        source_video_matches = list(
            (source_root / "videos").glob(
                f"*/{video_key}/episode_{episode_index:06d}.mp4"
            )
        )
        if len(source_video_matches) != 1:
            raise FileNotFoundError(
                f"Expected one video for episode {episode_index}, got {source_video_matches}"
            )
        shutil.copy2(
            source_video_matches[0],
            video_output_dir / f"file-{episode_index:03d}.mp4",
        )

        source_episode = source_episodes[episode_index]
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": np.asarray(source_episode["tasks"], dtype=object),
                "length": frame_count,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + frame_count,
                f"videos/{video_key}/chunk_index": 0,
                f"videos/{video_key}/file_index": episode_index,
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": frame_count / info["fps"],
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        total_frames += frame_count
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(parquet_paths):
            print(f"Converted {episode_index + 1}/{len(parquet_paths)} episodes")

    info["robot_type"] = "sonic_g1"
    info["codebase_version"] = "v3.1"
    info["total_episodes"] = len(parquet_paths)
    info["total_frames"] = total_frames
    info["total_videos"] = len(parquet_paths)
    info["total_chunks"] = 1
    info["chunks_size"] = max(len(parquet_paths), 1)
    info["splits"] = {"train": f"0:{len(parquet_paths)}"}
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    info["features"] = {
        video_key: info["features"][video_key],
        "observation.state": {
            "dtype": "float32",
            "shape": [13],
            "names": ACTION_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [13],
            "names": ACTION_NAMES,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info["action_schema_id"] = "sonic_g1_left_arm_inspire6_absolute_joint_v1"
    info["action_semantics"] = "absolute_joint_target"
    info["controlled_side"] = "left"
    info["source_episode_filter"] = {
        "min_episode_index_inclusive": 0,
        "selected_source_episode_indices": list(range(len(parquet_paths))),
    }
    (output_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n",
        encoding="utf-8",
    )

    modality = {
        "state": {"arm_hand_joint_state": {"start": 0, "end": 13}},
        "action": {"arm_hand_joint_target": {"start": 0, "end": 13}},
        "video": {"ego_view": {"original_key": video_key}},
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
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
    print(
        f"Wrote {output_root}: episodes={len(parquet_paths)} frames={total_frames}"
    )


if __name__ == "__main__":
    main()

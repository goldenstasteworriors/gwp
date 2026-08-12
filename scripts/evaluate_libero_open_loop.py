#!/usr/bin/env python3
"""Evaluate 48-step LIBERO action chunks directly from LeRobot episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIGA_MODELS = PROJECT_ROOT / "third_party" / "giga-models"
if str(GIGA_MODELS) not in sys.path:
    sys.path.insert(0, str(GIGA_MODELS))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

from giga_models.sockets import RobotInferenceClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11421)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_prompt(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("t5_embedding", next(iter(value.values())))
    return torch.as_tensor(value).float()[:64]


def select_episodes(frames: pd.DataFrame, explicit: list[int] | None, per_task: int) -> list[int]:
    episode_tasks = frames.groupby("episode_index", sort=True)["task_index"].first()
    if explicit:
        missing = sorted(set(explicit) - set(episode_tasks.index))
        if missing:
            raise ValueError(f"Unknown episode indices: {missing}")
        return sorted(set(explicit))
    selected: list[int] = []
    for _, group in episode_tasks.groupby(episode_tasks):
        selected.extend(group.index[:per_task].to_list())
    return sorted(int(index) for index in selected)


def decode_selected_frames(video_path: Path, frame_indices: dict[int, int]) -> dict[int, np.ndarray]:
    by_frame = {frame_index: episode for episode, frame_index in frame_indices.items()}
    result: dict[int, np.ndarray] = {}
    last_needed = max(by_frame)
    with av.open(str(video_path)) as container:
        for decoded_index, frame in enumerate(container.decode(video=0)):
            if decoded_index in by_frame:
                result[by_frame[decoded_index]] = frame.to_ndarray(format="rgb24")
            if decoded_index >= last_needed:
                break
    missing = sorted(set(frame_indices) - set(result))
    if missing:
        raise RuntimeError(f"Failed to decode first frames for episodes {missing}")
    return result


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = sorted(args.dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data under {args.dataset_root}")
    columns = ["index", "episode_index", "frame_index", "task_index", "observation.state", "action"]
    frames = pd.concat([pd.read_parquet(path, columns=columns) for path in parquet_paths], ignore_index=True)
    episodes = select_episodes(frames, args.episodes, args.episodes_per_task)
    grouped = {
        episode: frames[frames["episode_index"] == episode].sort_values("frame_index")
        for episode in episodes
    }
    first_indices = {episode: int(group.iloc[0]["index"]) for episode, group in grouped.items()}
    video_paths = sorted(args.dataset_root.glob("videos/observation.images.image/chunk-*/file-*.mp4"))
    if len(video_paths) != 1:
        raise NotImplementedError("This evaluator currently expects the packed LIBERO v3 single video file")
    images = decode_selected_frames(video_paths[0], first_indices)

    client = RobotInferenceClient(host=args.host, port=args.port)
    sent_tasks: set[int] = set()
    task_prompt_episode = frames.groupby("task_index")["episode_index"].min().to_dict()
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        group = grouped[episode]
        task_id = int(group.iloc[0]["task_index"])
        actions = np.stack(group["action"].to_numpy()).astype(np.float32)[:, :7]
        if len(actions) < 48:
            actions = np.concatenate([actions, np.repeat(actions[-1:], 48 - len(actions), axis=0)])
        target = actions[:48]
        request = {
            "image": images[episode],
            "state": np.asarray(group.iloc[0]["observation.state"], dtype=np.float32),
            "prompt_key": f"libero_spatial:{task_id}",
            "task_id": task_id,
            "episode_index": episode,
            "chunk_index": 0,
        }
        if task_id not in sent_tasks:
            prompt_episode = int(task_prompt_episode[task_id])
            request["prompt_embedding"] = load_prompt(
                args.dataset_root / "t5_embedding" / f"episode_{prompt_episode:06d}.pt"
            )
            sent_tasks.add(task_id)
        predicted = np.asarray(client.inference(request), dtype=np.float32)[:48, :7]
        error = predicted - target
        episode_dir = args.output_dir / f"episode_{episode:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.save(episode_dir / "predicted_actions.npy", predicted)
        row = {
            "episode_index": episode,
            "task_id": task_id,
            "mse": float(np.mean(error**2)),
            "translation_mse": float(np.mean(error[:, :3] ** 2)),
            "rotation_mse": float(np.mean(error[:, 3:6] ** 2)),
            "gripper_mse": float(np.mean(error[:, 6] ** 2)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    task_ids = sorted({row["task_id"] for row in rows})
    summary = {
        "num_episodes": len(rows),
        "mse": float(np.mean([row["mse"] for row in rows])),
        "translation_mse": float(np.mean([row["translation_mse"] for row in rows])),
        "rotation_mse": float(np.mean([row["rotation_mse"] for row in rows])),
        "gripper_mse": float(np.mean([row["gripper_mse"] for row in rows])),
        "mse_by_task": {
            str(task_id): float(np.mean([row["mse"] for row in rows if row["task_id"] == task_id]))
            for task_id in task_ids
        },
        "episodes": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()

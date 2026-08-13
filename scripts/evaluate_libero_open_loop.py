#!/usr/bin/env python3
"""Evaluate 48-step LIBERO action chunks directly from LeRobot episodes."""

from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_AGENT_VIEW_KEY = "observation.images.image"
DEFAULT_WRIST_VIEW_KEY = "observation.images.wrist_image"
ACTION_HORIZONS = (1, 12, 24, 36, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", required=True, help="LIBERO suite name used to namespace prompt caches")
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument("--episodes", type=int, nargs="+")
    episode_group.add_argument(
        "--episodes-file",
        type=Path,
        help="Text file containing one episode index per line.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11421)
    parser.add_argument("--num-views", type=int, choices=(1, 2), default=1)
    parser.add_argument("--agent-view-key", default=DEFAULT_AGENT_VIEW_KEY)
    parser.add_argument("--wrist-view-key", default=DEFAULT_WRIST_VIEW_KEY)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-num-inference-steps", type=int, required=True)
    parser.add_argument(
        "--policy-model-dtype", choices=("bf16", "fp16", "fp32"), required=True
    )
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_episode_file(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        episodes = [int(token) for token in path.read_text(encoding="utf-8").split()]
    except ValueError as error:
        raise ValueError(f"Episode file must contain only integers: {path}") from error
    if not episodes:
        raise ValueError(f"Episode file is empty: {path}")
    if len(episodes) != len(set(episodes)):
        raise ValueError(f"Episode file contains duplicate indices: {path}")
    return episodes


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
    """Decode selected local frame indices from one packed video shard."""

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


def decode_episode_first_frames(
    dataset_root: Path,
    episodes: list[int],
    fps: float,
    view_key: str = DEFAULT_AGENT_VIEW_KEY,
) -> dict[int, np.ndarray]:
    """Resolve each episode through LeRobot v3 video shard metadata."""

    metadata_paths = sorted(dataset_root.glob("meta/episodes/chunk-*/*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta/episodes'}")
    columns = [
        "episode_index",
        f"videos/{view_key}/chunk_index",
        f"videos/{view_key}/file_index",
        f"videos/{view_key}/from_timestamp",
    ]
    metadata = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in metadata_paths], ignore_index=True
    ).set_index("episode_index")
    requests_by_shard: dict[tuple[int, int], dict[int, int]] = {}
    for episode in episodes:
        if episode not in metadata.index:
            raise ValueError(f"No video metadata for episode {episode}")
        row = metadata.loc[episode]
        chunk = int(row[f"videos/{view_key}/chunk_index"])
        file_index = int(row[f"videos/{view_key}/file_index"])
        local_frame = int(round(float(row[f"videos/{view_key}/from_timestamp"]) * fps))
        requests_by_shard.setdefault((chunk, file_index), {})[episode] = local_frame

    images: dict[int, np.ndarray] = {}
    for (chunk, file_index), frame_indices in requests_by_shard.items():
        video_path = (
            dataset_root
            / "videos"
            / view_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        images.update(decode_selected_frames(video_path, frame_indices))
    return images


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


def action_error_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if predicted.shape != target.shape or predicted.shape != (48, 7):
        raise ValueError(
            f"Expected predicted/target actions with shape (48, 7), got "
            f"{predicted.shape} and {target.shape}"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Predicted and target actions must be finite")
    error = predicted - target
    squared = error**2
    absolute = np.abs(error)
    clipped = np.clip(predicted, -1.0, 1.0)
    clipped_error = clipped - target
    gripper_predicted = predicted[:, 6] >= 0.0
    gripper_target = target[:, 6] >= 0.0
    return {
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(absolute)),
        "translation_mse": float(np.mean(squared[:, :3])),
        "translation_mae": float(np.mean(absolute[:, :3])),
        "rotation_mse": float(np.mean(squared[:, 3:6])),
        "rotation_mae": float(np.mean(absolute[:, 3:6])),
        "gripper_mse": float(np.mean(squared[:, 6])),
        "gripper_mae": float(np.mean(absolute[:, 6])),
        "gripper_sign_accuracy": float(np.mean(gripper_predicted == gripper_target)),
        "out_of_range_element_fraction": float(np.mean(np.abs(predicted) > 1.0)),
        "out_of_range_step_fraction": float(np.mean(np.any(np.abs(predicted) > 1.0, axis=1))),
        "clipped_mse": float(np.mean(clipped_error**2)),
        "per_dimension_mse": np.mean(squared, axis=0).tolist(),
        "per_dimension_mae": np.mean(absolute, axis=0).tolist(),
        "prefix_mse_by_horizon": {
            str(horizon): float(np.mean(squared[:horizon]))
            for horizon in ACTION_HORIZONS
        },
        "prefix_gripper_sign_accuracy_by_horizon": {
            str(horizon): float(
                np.mean(gripper_predicted[:horizon] == gripper_target[:horizon])
            )
            for horizon in ACTION_HORIZONS
        },
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = sorted(args.dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data under {args.dataset_root}")
    columns = ["index", "episode_index", "frame_index", "task_index", "observation.state", "action"]
    frames = pd.concat([pd.read_parquet(path, columns=columns) for path in parquet_paths], ignore_index=True)
    explicit_episodes = (
        read_episode_file(args.episodes_file)
        if args.episodes_file is not None
        else args.episodes
    )
    episodes = select_episodes(frames, explicit_episodes, args.episodes_per_task)
    grouped = {
        episode: frames[frames["episode_index"] == episode].sort_values("frame_index")
        for episode in episodes
    }
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    agent_images = decode_episode_first_frames(
        args.dataset_root,
        episodes,
        float(info["fps"]),
        args.agent_view_key,
    )
    wrist_images = None
    if args.num_views == 2:
        wrist_images = decode_episode_first_frames(
            args.dataset_root,
            episodes,
            float(info["fps"]),
            args.wrist_view_key,
        )

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
            "image": agent_images[episode],
            "state": np.asarray(group.iloc[0]["observation.state"], dtype=np.float32),
            "prompt_key": f"{args.suite}:{task_id}",
            "task_id": task_id,
            "episode_index": episode,
            "chunk_index": 0,
        }
        if wrist_images is not None:
            request["wrist_image"] = wrist_images[episode]
        if task_id not in sent_tasks:
            prompt_episode = int(task_prompt_episode[task_id])
            request["prompt_embedding"] = load_prompt(
                args.dataset_root / "t5_embedding" / f"episode_{prompt_episode:06d}.pt"
            )
            sent_tasks.add(task_id)
        predicted = np.asarray(client.inference(request), dtype=np.float32)[:48, :7]
        if predicted.shape != (48, 7):
            raise ValueError(
                f"Policy returned {predicted.shape} for episode {episode}; expected (48, 7)"
            )
        episode_dir = args.output_dir / f"episode_{episode:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        predicted_path = episode_dir / "predicted_actions.npy"
        target_path = episode_dir / "target_actions.npy"
        np.save(predicted_path, predicted)
        np.save(target_path, target)
        row = {
            "episode_index": episode,
            "task_id": task_id,
            "available_target_steps": min(len(group), 48),
            "predicted_actions_file": str(predicted_path.relative_to(args.output_dir)),
            "predicted_actions_sha256": sha256_file(predicted_path),
            "target_actions_file": str(target_path.relative_to(args.output_dir)),
            "target_actions_sha256": sha256_file(target_path),
            **action_error_metrics(predicted, target),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    task_ids = sorted({row["task_id"] for row in rows})
    scalar_metric_keys = (
        "mse",
        "mae",
        "translation_mse",
        "translation_mae",
        "rotation_mse",
        "rotation_mae",
        "gripper_mse",
        "gripper_mae",
        "gripper_sign_accuracy",
        "out_of_range_element_fraction",
        "out_of_range_step_fraction",
        "clipped_mse",
    )
    summary = {
        "schema_version": "libero-open-loop-v2",
        "num_episodes": len(rows),
        "num_views": args.num_views,
        "view_keys": (
            [args.agent_view_key]
            if args.num_views == 1
            else [args.agent_view_key, args.wrist_view_key]
        ),
        "episode_selection_file": (
            str(args.episodes_file.resolve()) if args.episodes_file is not None else None
        ),
        "episode_selection_sha256": (
            sha256_file(args.episodes_file) if args.episodes_file is not None else None
        ),
        "protocol": {
            "dataset_root": str(args.dataset_root.resolve()),
            "suite": args.suite,
            "num_views": int(args.num_views),
            "view_keys": (
                [args.agent_view_key]
                if args.num_views == 1
                else [args.agent_view_key, args.wrist_view_key]
            ),
            "policy_checkpoint": str(args.policy_checkpoint.resolve()),
            "policy_num_inference_steps": args.policy_num_inference_steps,
            "policy_model_dtype": args.policy_model_dtype,
            "policy_seed": args.policy_seed,
        },
        **{key: mean_metric(rows, key) for key in scalar_metric_keys},
        "per_dimension_mse": np.mean(
            np.asarray([row["per_dimension_mse"] for row in rows]), axis=0
        ).tolist(),
        "per_dimension_mae": np.mean(
            np.asarray([row["per_dimension_mae"] for row in rows]), axis=0
        ).tolist(),
        "prefix_mse_by_horizon": {
            str(horizon): float(
                np.mean(
                    [row["prefix_mse_by_horizon"][str(horizon)] for row in rows]
                )
            )
            for horizon in ACTION_HORIZONS
        },
        "prefix_gripper_sign_accuracy_by_horizon": {
            str(horizon): float(
                np.mean(
                    [
                        row["prefix_gripper_sign_accuracy_by_horizon"][str(horizon)]
                        for row in rows
                    ]
                )
            )
            for horizon in ACTION_HORIZONS
        },
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

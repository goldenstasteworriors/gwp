#!/usr/bin/env python3
"""Run closed-loop LIBERO rollouts against a GWP socket server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--rollouts-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11421)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def task_prompt_files(dataset_root: Path) -> dict[int, Path]:
    paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    frames = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "task_index"]) for path in paths],
        ignore_index=True,
    )
    first_episodes = frames.groupby("task_index")["episode_index"].min().to_dict()
    return {
        int(task_id): dataset_root / "t5_embedding" / f"episode_{int(episode):06d}.pt"
        for task_id, episode in first_episodes.items()
    }


def load_prompt(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("t5_embedding", next(iter(value.values())))
    value = torch.as_tensor(value).float()
    return value[:64]


def policy_state(observation: dict[str, Any]) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    robot = observation["robot_state"]
    position = np.asarray(robot["eef"]["pos"], dtype=np.float32)
    axis_angle = np.asarray(quat2axisangle(robot["eef"]["quat"]), dtype=np.float32)
    gripper = np.asarray(robot["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([position, axis_angle, gripper], axis=0)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = task_prompt_files(args.dataset_root)
    client = RobotInferenceClient(host=args.host, port=args.port)

    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    rows: list[dict[str, Any]] = []
    prompt_sent: set[int] = set()
    for task_id in args.task_ids:
        if task_id < 0 or task_id >= suite.n_tasks:
            raise ValueError(f"Invalid task id {task_id}; suite has {suite.n_tasks} tasks")
        prompt = load_prompt(prompts[task_id])
        for rollout in range(args.rollouts_per_task):
            env = LiberoEnv(
                suite,
                task_id,
                args.suite,
                episode_length=args.max_steps,
                camera_name="agentview_image",
                obs_type="pixels_agent_pos",
                init_states=True,
                episode_index=rollout,
                n_envs=1,
                num_steps_wait=10,
            )
            observation, _ = env.reset(seed=args.seed + rollout)
            executed: list[np.ndarray] = []
            success = False
            step = 0
            chunk = 0
            while step < args.max_steps and not success:
                request = {
                    "image": observation["pixels"]["image"],
                    "state": policy_state(observation),
                    "prompt_key": f"{args.suite}:{task_id}",
                    "task_id": task_id,
                    "episode_index": rollout,
                    "chunk_index": chunk,
                }
                if task_id not in prompt_sent:
                    request["prompt_embedding"] = prompt
                    prompt_sent.add(task_id)
                action_chunk = np.asarray(client.inference(request), dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                    raise ValueError(f"Policy returned malformed actions: {action_chunk.shape}")
                for action in action_chunk[: args.replan_steps, :7]:
                    observation, reward, terminated, truncated, _ = env.step(
                        np.clip(action, -1.0, 1.0)
                    )
                    executed.append(action.copy())
                    step += 1
                    success = success or bool(reward > 0.5)
                    if success or terminated or truncated or step >= args.max_steps:
                        break
                chunk += 1

            rollout_dir = args.output_dir / f"task_{task_id:02d}" / f"rollout_{rollout:02d}"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            np.save(rollout_dir / "executed_actions.npy", np.asarray(executed, dtype=np.float32))
            row = {
                "suite": args.suite,
                "task_id": task_id,
                "task_description": env.task_description,
                "rollout": rollout,
                "success": success,
                "steps": step,
                "chunks": chunk,
            }
            (rollout_dir / "result.json").write_text(json.dumps(to_jsonable(row), indent=2) + "\n")
            rows.append(row)
            print(json.dumps(to_jsonable(row)), flush=True)
            env.close()

    by_task = {
        str(task_id): float(np.mean([row["success"] for row in rows if row["task_id"] == task_id]))
        for task_id in args.task_ids
    }
    summary = {
        "suite": args.suite,
        "num_rollouts": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "success_rate_by_task": by_task,
        "rollouts": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n")
    print(json.dumps(to_jsonable({key: value for key, value in summary.items() if key != "rollouts"}), indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()

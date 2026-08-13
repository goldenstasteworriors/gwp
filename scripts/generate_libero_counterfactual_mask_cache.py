#!/usr/bin/env python3
"""Render policy-action robot masks without requiring a future RGB target.

Each input ``predicted_actions.npy`` is executed from the matching LIBERO
demonstration initial state.  The resulting simulator robot masks are valid
embodiment-consequence targets for those predicted actions; demonstration
future frames and demonstration actions are intentionally never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "libero-counterfactual-robot-mask-v1"
EPISODE_PATTERN = re.compile(r"episode_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Directory containing episode_XXXXXX/predicted_actions.npy files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 12, 24, 36, 48])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def discover_predictions(root: Path, episodes: list[int] | None) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    for path in sorted(root.glob("episode_*/predicted_actions.npy")):
        match = EPISODE_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        episode_index = int(match.group(1))
        if episode_index in discovered:
            raise ValueError(f"Duplicate prediction for episode {episode_index}")
        discovered[episode_index] = path
    if not discovered:
        raise FileNotFoundError(f"No episode_*/predicted_actions.npy files under {root}")
    if episodes is None:
        return discovered
    requested = sorted(set(episodes))
    missing = [episode for episode in requested if episode not in discovered]
    if missing:
        raise FileNotFoundError(f"Missing predictions for episodes: {missing}")
    return {episode: discovered[episode] for episode in requested}


def load_episode_index(dataset_root: Path) -> dict[int, tuple[int, int]]:
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    frames = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "task_index"]) for path in parquet_paths],
        ignore_index=True,
    )
    episode_tasks = frames.groupby("episode_index", sort=True)["task_index"].first()
    result: dict[int, tuple[int, int]] = {}
    for task_index, task_episodes in episode_tasks.groupby(episode_tasks, sort=True):
        for task_episode_rank, episode_index in enumerate(task_episodes.index.to_list()):
            result[int(episode_index)] = (int(task_index), task_episode_rank)
    return result


def load_actions(path: Path, max_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.load(path, allow_pickle=False)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected [T, >=7] actions in {path}, got {actions.shape}")
    actions = np.asarray(actions[:, :7], dtype=np.float32)
    if len(actions) < max_horizon:
        raise ValueError(f"{path} has {len(actions)} actions, need at least {max_horizon}")
    if not np.isfinite(actions).all():
        raise ValueError(f"Non-finite action values in {path}")
    return actions, np.clip(actions, -1.0, 1.0)


def robot_geom_ids(env: Any) -> tuple[np.ndarray, np.ndarray]:
    mapping = env._env.env.model.geom_ids_to_instances
    robot_instances = {"MountedPanda0", "PandaGripper0"}
    robot = sorted(int(geom_id) for geom_id, instance in mapping.items() if instance in robot_instances)
    gripper = sorted(int(geom_id) for geom_id, instance in mapping.items() if instance == "PandaGripper0")
    if not robot or not gripper:
        raise RuntimeError("Could not identify Panda robot geometries")
    return np.asarray(robot), np.asarray(gripper)


def capture_state(
    env: Any,
    obs: dict[str, Any],
    robot_ids: np.ndarray,
    gripper_ids: np.ndarray,
    camera_name: str,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    sim = env._env.sim
    segmentation = np.flipud(
        np.asarray(
            sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
                segmentation=True,
            )
        )
    )
    geom_ids = segmentation[..., 1]
    camera_id = sim.model.camera_name2id(camera_name)
    return {
        "robot_mask": np.isin(geom_ids, robot_ids).astype(np.uint8),
        "gripper_mask": np.isin(geom_ids, gripper_ids).astype(np.uint8),
        "qpos": np.asarray(sim.data.qpos, dtype=np.float32).copy(),
        "eef_position": np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32),
        "eef_rotation": np.asarray(obs["robot_state"]["eef"]["mat"], dtype=np.float32).reshape(3, 3),
        "camera_position": np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float32).copy(),
        "camera_rotation": np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float32).reshape(3, 3).copy(),
        "camera_fovy": np.asarray(sim.model.cam_fovy[camera_id], dtype=np.float32),
    }


def stack_captures(captures: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = captures[0].keys()
    return {key: np.stack([capture[key] for capture in captures], axis=0) for key in keys}


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or horizons[0] < 0:
        raise ValueError("Horizons must be non-negative")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Render width and height must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    predictions = discover_predictions(args.predictions_root, args.episodes)
    episode_index = load_episode_index(args.dataset_root)
    unknown = sorted(set(predictions) - set(episode_index))
    if unknown:
        raise ValueError(f"Prediction episodes absent from dataset: {unknown}")

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    dataset_info_sha256 = sha256_file(info_path)
    args.output_dir.mkdir(parents=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir()

    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    rows: list[dict[str, Any]] = []
    for episode, prediction_path in predictions.items():
        task_index, task_episode_rank = episode_index[episode]
        raw_actions, executed_actions = load_actions(prediction_path, max(horizons))
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_index,
            task_suite_name=args.suite,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image",
            init_states=True,
            episode_index=task_episode_rank,
            n_envs=1,
            num_steps_wait=args.num_steps_wait,
        )
        try:
            obs, _ = env.reset(seed=args.seed)
            robot_ids, gripper_ids = robot_geom_ids(env)
            captures_by_horizon = {
                0: capture_state(
                    env,
                    obs,
                    robot_ids,
                    gripper_ids,
                    args.camera_name,
                    args.width,
                    args.height,
                )
            }
            for step in range(1, max(horizons) + 1):
                obs, *_ = env.step(executed_actions[step - 1])
                if step in horizons:
                    captures_by_horizon[step] = capture_state(
                        env,
                        obs,
                        robot_ids,
                        gripper_ids,
                        args.camera_name,
                        args.width,
                        args.height,
                    )
        finally:
            env.close()

        arrays = stack_captures([captures_by_horizon[horizon] for horizon in horizons])
        cache_path = episodes_dir / f"episode_{episode:06d}.npz"
        np.savez_compressed(
            cache_path,
            horizons=np.asarray(horizons, dtype=np.int32),
            raw_actions=raw_actions,
            executed_actions=executed_actions,
            **arrays,
        )
        row = {
            "episode_index": episode,
            "task_index": task_index,
            "task_episode_rank": task_episode_rank,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "cache_file": str(cache_path.relative_to(args.output_dir)),
            "cache_sha256": sha256_file(cache_path),
            "robot_pixels": [int(mask.sum()) for mask in arrays["robot_mask"]],
        }
        rows.append(row)
        print(json.dumps(jsonable(row)), flush=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_semantics": "robot embodiment after executing the matching policy-predicted action prefix",
        "rgb_supervision": None,
        "suite": args.suite,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_info_sha256": dataset_info_sha256,
        "predictions_root": str(args.predictions_root.resolve()),
        "horizons": horizons,
        "camera_name": args.camera_name,
        "image_size": [args.height, args.width],
        "num_steps_wait": args.num_steps_wait,
        "seed": args.seed,
        "num_episodes": len(rows),
        "episodes": rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()

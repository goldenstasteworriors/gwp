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
import os
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION = "libero-counterfactual-robot-mask-v3"
IMAGE_ORIENTATION = "lerobot_libero_visualization_flip_height_width"
EPISODE_PATTERN = re.compile(r"episode_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--raw-demos-root",
        type=Path,
        required=True,
        help=(
            "Directory containing the authoritative LIBERO <task>_demo.hdf5 files. "
            "The LeRobot episode is uniquely matched by its complete action sequence."
        ),
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Directory containing episode_XXXXXX/predicted_actions.npy files.",
    )
    parser.add_argument(
        "--policy-id",
        help="Checkpoint or run identifier used to produce the predicted actions.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument("--episodes", type=int, nargs="+")
    episode_group.add_argument(
        "--episodes-file",
        type=Path,
        help="Text file containing one episode index per line.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 12, 24, 36, 48])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-match-atol", type=float, default=1e-6)
    parser.add_argument("--source-state-atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.tobytes())
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


def configure_libero_path(dataset_root: Path) -> Path:
    """Resolve a non-interactive LIBERO config before importing LIBERO."""

    configured = os.environ.get("LIBERO_CONFIG_PATH")
    if configured:
        config_path = Path(configured).expanduser().resolve() / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"LIBERO_CONFIG_PATH does not contain config.yaml: {config_path}"
            )
        return config_path
    resolved = dataset_root.resolve()
    for parent in (resolved, *resolved.parents):
        config_path = parent / "libero_config" / "config.yaml"
        if config_path.is_file():
            os.environ["LIBERO_CONFIG_PATH"] = str(config_path.parent)
            return config_path
    raise FileNotFoundError(
        f"Could not discover libero_config/config.yaml from {dataset_root}"
    )


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


def load_episode_languages(dataset_root: Path) -> dict[int, str]:
    metadata_paths = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"No episode metadata under {dataset_root}")
    metadata = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "tasks"]) for path in metadata_paths],
        ignore_index=True,
    )
    result: dict[int, str] = {}
    for _, row in metadata.iterrows():
        episode = int(row["episode_index"])
        tasks = list(row["tasks"])
        if len(tasks) != 1 or not str(tasks[0]).strip():
            raise ValueError(f"Episode {episode} must have exactly one language task: {tasks}")
        result[episode] = str(tasks[0]).strip()
    return result


def load_episode_records(dataset_root: Path) -> dict[int, dict[str, Any]]:
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    frames = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "episode_index",
                    "frame_index",
                    "task_index",
                    "observation.state",
                    "action",
                ],
            )
            for path in parquet_paths
        ],
        ignore_index=True,
    )
    episode_languages = load_episode_languages(dataset_root)
    result: dict[int, dict[str, Any]] = {}
    for episode_index, episode_frames in frames.groupby("episode_index", sort=True):
        episode_frames = episode_frames.sort_values("frame_index")
        if int(episode_frames.iloc[0]["frame_index"]) != 0:
            raise ValueError(f"Episode {episode_index} has no frame 0")
        task_indices = episode_frames["task_index"].unique()
        if len(task_indices) != 1:
            raise ValueError(f"Episode {episode_index} spans multiple tasks")
        result[int(episode_index)] = {
            "task_index": int(task_indices[0]),
            "task_language": episode_languages[int(episode_index)],
            "source_observation_state": np.asarray(
                episode_frames.iloc[0]["observation.state"], dtype=np.float32
            ),
            "demonstration_actions": np.stack(episode_frames["action"].to_numpy()).astype(
                np.float32
            ),
        }
    return result


def filter_raw_demo_actions(actions: np.ndarray) -> np.ndarray:
    """Apply the no-op filter while preserving simulator action semantics."""

    kept: list[np.ndarray] = []
    for action in np.asarray(actions, dtype=np.float64):
        previous = kept[-1] if kept else None
        is_noop = np.linalg.norm(action[:-1]) < 1e-4 and (
            previous is None or action[-1] == previous[-1]
        )
        if not is_noop:
            kept.append(action.copy())
    if not kept:
        raise ValueError("Raw LIBERO demonstration contains only no-op actions")
    return np.stack(kept).astype(np.float32)


def normalize_raw_demo_actions(actions: np.ndarray) -> np.ndarray:
    """Match the no-op filtering and gripper encoding used by the LeRobot conversion."""

    normalized = filter_raw_demo_actions(actions)
    # Raw LIBERO uses {-1=open, +1=closed}; this LeRobot conversion stores
    # {1=open, 0=closed}.
    normalized[:, -1] = (1.0 - normalized[:, -1]) * 0.5
    return normalized


def load_raw_demo_candidates(path: Path, expected_language: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    candidates: list[dict[str, Any]] = []
    with h5py.File(path, "r") as raw_file:
        if "data" not in raw_file:
            raise ValueError(f"Raw LIBERO HDF5 has no data group: {path}")
        problem_info = json.loads(str(raw_file["data"].attrs["problem_info"]))
        raw_language = str(problem_info.get("language_instruction", "")).strip()
        if raw_language != expected_language:
            raise ValueError(
                f"Raw LIBERO HDF5 language {raw_language!r} does not match dataset "
                f"language {expected_language!r}: {path}"
            )
        for demo_name, demo in raw_file["data"].items():
            if "actions" not in demo or "states" not in demo or len(demo["states"]) == 0:
                raise ValueError(f"Incomplete raw LIBERO demo {path}:{demo_name}")
            candidates.append(
                {
                    "demo_name": str(demo_name),
                    "actions": normalize_raw_demo_actions(demo["actions"][()]),
                    "initial_state": np.asarray(demo["states"][0], dtype=np.float64),
                }
            )
    if not candidates:
        raise ValueError(f"Raw LIBERO HDF5 contains no demonstrations: {path}")
    return candidates


def match_raw_demo(
    episode: int,
    episode_actions: np.ndarray,
    candidates: list[dict[str, Any]],
    atol: float,
) -> tuple[dict[str, Any], float]:
    matches: list[tuple[dict[str, Any], float]] = []
    for candidate in candidates:
        candidate_actions = candidate["actions"]
        if candidate_actions.shape != episode_actions.shape:
            continue
        max_abs = float(np.max(np.abs(candidate_actions - episode_actions)))
        if max_abs <= atol:
            matches.append((candidate, max_abs))
    if len(matches) != 1:
        labels = [f"{candidate['demo_name']}:{error:.3g}" for candidate, error in matches]
        raise ValueError(
            f"Episode {episode} must uniquely match one raw LIBERO demo by its full action "
            f"sequence; found {len(matches)} matches within atol={atol}: {labels}"
        )
    return matches[0]


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion.shape}")
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float64)
    return quaternion[:3] / denominator * (2.0 * np.arccos(w))


def flatten_observation_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64),
            quaternion_to_axis_angle(obs["robot_state"]["eef"]["quat"]),
            np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float64),
        ]
    ).astype(np.float32)


def suite_task_index_by_language(suite: Any) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for task_index in range(len(suite.tasks)):
        language = str(suite.get_task(task_index).language).strip()
        if language in mapping:
            raise ValueError(f"Duplicate LIBERO suite task language: {language!r}")
        mapping[language] = task_index
    return mapping


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
    # LeRobot's LIBERO recordings use LiberoEnv.render(), which rotates the
    # raw robosuite/OpenGL observation by 180 degrees (flip H and W).  Apply
    # the identical visualization convention to geometry targets so masks
    # and the RGB tokens supervised by GWP share one pixel coordinate frame.
    segmentation = np.flip(
        np.asarray(
            sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
                segmentation=True,
            )
        ),
        axis=(0, 1),
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
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be non-negative")
    if args.action_match_atol <= 0 or args.source_state_atol <= 0:
        raise ValueError("Action/state matching tolerances must be positive")
    if not args.raw_demos_root.is_dir():
        raise FileNotFoundError(args.raw_demos_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    explicit_episodes = (
        read_episode_file(args.episodes_file)
        if args.episodes_file is not None
        else args.episodes
    )
    predictions = discover_predictions(args.predictions_root, explicit_episodes)
    episode_records = load_episode_records(args.dataset_root)
    unknown = sorted(set(predictions) - set(episode_records))
    if unknown:
        raise ValueError(f"Prediction episodes absent from dataset: {unknown}")

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    dataset_info_sha256 = sha256_file(info_path)
    libero_config_path = configure_libero_path(args.dataset_root)
    from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

    suite = _get_suite(args.suite)
    suite_language_index = suite_task_index_by_language(suite)
    raw_sources: dict[int, dict[str, Any]] = {}
    matched_demos: dict[int, tuple[dict[str, Any], float]] = {}
    used_demo_names: dict[int, set[str]] = {}
    for episode in predictions:
        source = episode_records[episode]
        task_index = int(source["task_index"])
        task_language = str(source["task_language"])
        if task_language not in suite_language_index:
            raise ValueError(
                f"Dataset task language is absent from suite {args.suite}: {task_language!r}"
            )
        libero_task_index = suite_language_index[task_language]
        if task_index not in raw_sources:
            task = suite.get_task(libero_task_index)
            raw_path = args.raw_demos_root / f"{task.name}_demo.hdf5"
            raw_sources[task_index] = {
                "task_index": task_index,
                "libero_task_index": libero_task_index,
                "task_language": task_language,
                "task_name": task.name,
                "path": str(raw_path.resolve()),
                "sha256": sha256_file(raw_path),
                "candidates": load_raw_demo_candidates(raw_path, task_language),
            }
        elif (
            int(raw_sources[task_index]["libero_task_index"]) != libero_task_index
            or raw_sources[task_index]["task_language"] != task_language
        ):
            raise ValueError(f"Dataset task_index {task_index} maps to multiple task languages")
        matched_demo, match_error = match_raw_demo(
            episode,
            source["demonstration_actions"],
            raw_sources[task_index]["candidates"],
            args.action_match_atol,
        )
        demo_name = str(matched_demo["demo_name"])
        if demo_name in used_demo_names.setdefault(task_index, set()):
            raise ValueError(
                f"Raw demo {task_index}:{demo_name} matched more than one selected episode"
            )
        used_demo_names[task_index].add(demo_name)
        matched_demos[episode] = (matched_demo, match_error)

    args.output_dir.mkdir(parents=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir()

    rows: list[dict[str, Any]] = []
    for episode, prediction_path in predictions.items():
        source = episode_records[episode]
        task_index = int(source["task_index"])
        task_language = str(source["task_language"])
        source_observation_state = source["source_observation_state"]
        matched_demo, action_match_max_abs = matched_demos[episode]
        raw_initial_state = matched_demo["initial_state"]
        raw_source = raw_sources[task_index]
        libero_task_index = int(raw_source["libero_task_index"])
        raw_actions, executed_actions = load_actions(prediction_path, max(horizons))
        env = LiberoEnv(
            task_suite=suite,
            task_id=libero_task_index,
            task_suite_name=args.suite,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image",
            init_states=False,
            n_envs=1,
            num_steps_wait=0,
        )
        try:
            env.reset(seed=args.seed)
            raw_obs = env._env.set_init_state(raw_initial_state)
            for _ in range(args.num_steps_wait):
                raw_obs, _, _, _ = env._env.step(get_libero_dummy_action())
            obs = env._format_raw_obs(raw_obs)
            rendered_initial_observation_state = flatten_observation_state(obs)
            source_state_delta = rendered_initial_observation_state - source_observation_state
            source_state_max_abs = float(np.max(np.abs(source_state_delta)))
            source_state_l2 = float(np.linalg.norm(source_state_delta))
            if source_state_max_abs > args.source_state_atol:
                raise ValueError(
                    f"Episode {episode} raw demo {matched_demo['demo_name']} does not reproduce "
                    f"LeRobot frame-0 state: max_abs={source_state_max_abs:.9g}, "
                    f"l2={source_state_l2:.9g}, atol={args.source_state_atol}"
                )
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
            raw_initial_state=raw_initial_state,
            source_observation_state=source_observation_state,
            rendered_initial_observation_state=rendered_initial_observation_state,
            **arrays,
        )
        row = {
            "episode_index": episode,
            "source_frame_index": 0,
            "action_chunk_index": 0,
            "task_index": task_index,
            "libero_task_index": libero_task_index,
            "task_language": task_language,
            "raw_demo_name": matched_demo["demo_name"],
            "raw_hdf5_path": str(raw_source["path"]),
            "raw_hdf5_sha256": raw_source["sha256"],
            "raw_initial_state_sha256": sha256_array(raw_initial_state),
            "raw_demo_action_match_max_abs": action_match_max_abs,
            "source_state_max_abs": source_state_max_abs,
            "source_state_l2": source_state_l2,
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
        "raw_demos_root": str(args.raw_demos_root.resolve()),
        "raw_demo_files": [
            {
                key: value
                for key, value in raw_sources[task_index].items()
                if key
                in {
                    "task_index",
                    "libero_task_index",
                    "task_language",
                    "task_name",
                    "path",
                    "sha256",
                }
            }
            for task_index in sorted(raw_sources)
        ],
        "libero_config_path": str(libero_config_path),
        "libero_config_sha256": sha256_file(libero_config_path),
        "predictions_root": str(args.predictions_root.resolve()),
        "policy_id": args.policy_id,
        "episode_selection_file": (
            str(args.episodes_file.resolve()) if args.episodes_file is not None else None
        ),
        "episode_selection_sha256": (
            sha256_file(args.episodes_file) if args.episodes_file is not None else None
        ),
        "source_observation_semantics": (
            "Authoritative raw LIBERO demonstration states[0], uniquely matched to the "
            "LeRobot episode by its full converted action sequence, followed by "
            "num_steps_wait no-op steps; every target horizon is measured after executing "
            "that many policy-predicted actions"
        ),
        "raw_demo_matching": "unique_full_action_sequence_after_noop_filter_and_gripper_conversion",
        "action_match_atol": args.action_match_atol,
        "source_state_atol": args.source_state_atol,
        "source_frame_index": 0,
        "action_chunk_index": 0,
        "horizons": horizons,
        "camera_name": args.camera_name,
        "image_orientation": IMAGE_ORIENTATION,
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

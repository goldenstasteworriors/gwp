#!/usr/bin/env python3
"""Render LIBERO robot masks from q without constructing a benchmark environment.

The input contract is intentionally q-space, not action-space.  For every
episode this tool consumes ``predicted_joint_qpos.npy`` and
``predicted_gripper_qpos.npy`` and performs only FK plus robot-mesh
rasterization against a robot-only compiled model and explicit camera
calibration.  It never imports or constructs a LIBERO task environment,
restores a demonstration state, executes an action, evaluates contacts, or
advances simulator dynamics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_libero_counterfactual_mask_cache import (
    jsonable,
    load_episode_languages,
    sha256_file,
)
from libero_standalone_robot_mask import (
    RASTERIZER_SEMANTICS,
    StandaloneRobotMaskRenderer,
    forbid_mujoco_dynamics,
)


SCHEMA_VERSION = "libero-joint-standalone-kinematic-robot-mask-v2"
IMAGE_ORIENTATION = "lerobot_libero_visualization_flip_height_width"
EPISODE_PATTERN = re.compile(r"episode_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--joint-trajectories-root", type=Path, required=True)
    parser.add_argument("--joint-predictor-id", required=True)
    parser.add_argument("--robot-asset-dir", type=Path, required=True)
    parser.add_argument(
        "--q-source",
        choices=(
            "external_joint_predictor_outputs_only",
            "logged_demonstration_joint_trajectory",
        ),
        default="external_joint_predictor_outputs_only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--episodes-file", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 12, 24, 36, 48])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    return parser.parse_args()


def read_episode_file(path: Path) -> list[int]:
    episodes = [int(token) for token in path.read_text(encoding="utf-8").split()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError(f"Episode file must be non-empty and unique: {path}")
    return episodes


def discover_joint_trajectories(
    root: Path, requested: list[int] | None
) -> dict[int, tuple[Path, Path]]:
    records: dict[int, tuple[Path, Path]] = {}
    for arm_path in sorted(root.glob("episode_*/predicted_joint_qpos.npy")):
        match = EPISODE_PATTERN.fullmatch(arm_path.parent.name)
        if match is None:
            continue
        episode = int(match.group(1))
        gripper_path = arm_path.parent / "predicted_gripper_qpos.npy"
        if not gripper_path.is_file():
            raise FileNotFoundError(gripper_path)
        if episode in records:
            raise ValueError(f"Duplicate joint trajectory for episode {episode}")
        records[episode] = (arm_path, gripper_path)
    if requested is not None:
        missing = sorted(set(requested) - set(records))
        if missing:
            raise FileNotFoundError(f"Missing joint trajectories: {missing}")
        records = {episode: records[episode] for episode in requested}
    if not records:
        raise FileNotFoundError(f"No predicted joint trajectories under {root}")
    return records


def load_episode_tasks(
    dataset_root: Path, episodes: set[int]
) -> dict[int, dict[str, Any]]:
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No dataset parquet files under {dataset_root}")
    frames = pd.concat(
        [
            pd.read_parquet(path, columns=["episode_index", "frame_index", "task_index"])
            for path in parquet_paths
        ],
        ignore_index=True,
    )
    frames = frames[frames["episode_index"].isin(episodes)]
    languages = load_episode_languages(dataset_root)
    result: dict[int, dict[str, Any]] = {}
    for episode, episode_frames in frames.groupby("episode_index", sort=True):
        task_indices = episode_frames["task_index"].unique()
        if len(task_indices) != 1:
            raise ValueError(f"Episode {episode} spans multiple tasks")
        result[int(episode)] = {
            "task_index": int(task_indices[0]),
            "task_language": languages[int(episode)],
        }
    missing = sorted(episodes - set(result))
    if missing:
        raise ValueError(f"Joint-trajectory episodes absent from dataset: {missing}")
    return result


def sample_horizons(
    path: Path, expected_width: int, horizons: list[int]
) -> tuple[np.ndarray, str]:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(f"Expected [T,{expected_width}] in {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite joint trajectory: {path}")
    if len(values) == len(horizons):
        return values.copy(), "one_row_per_declared_horizon"
    if len(values) <= max(horizons):
        raise ValueError(f"{path} has {len(values)} rows, need index {max(horizons)}")
    return values[np.asarray(horizons, dtype=np.int64)].copy(), "dense_rows_indexed_by_horizon"


def stack_captures(captures: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([capture[key] for capture in captures], axis=0)
        for key in captures[0]
    }


def main() -> None:
    args = parse_args()
    if args.episodes is not None and args.episodes_file is not None:
        raise ValueError("Use only one of --episodes and --episodes-file")
    horizons = sorted(set(int(value) for value in args.horizons))
    if not horizons or horizons[0] < 0:
        raise ValueError("Horizons must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    requested = (
        read_episode_file(args.episodes_file)
        if args.episodes_file is not None
        else args.episodes
    )
    trajectories = discover_joint_trajectories(
        args.joint_trajectories_root, requested
    )
    episode_tasks = load_episode_tasks(args.dataset_root, set(trajectories))
    dataset_info_path = args.dataset_root / "meta/info.json"
    asset_manifest_path = args.robot_asset_dir / "manifest.json"
    calibration_path = args.robot_asset_dir / "camera_calibration.json"
    if not asset_manifest_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError(f"Incomplete standalone robot asset: {args.robot_asset_dir}")
    renderer = StandaloneRobotMaskRenderer(
        args.robot_asset_dir, args.width, args.height
    )
    args.output_dir.mkdir(parents=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir()
    rows: list[dict[str, Any]] = []
    trajectory_layouts: set[str] = set()

    with forbid_mujoco_dynamics() as dynamics_counts:
        for episode, (arm_path, gripper_path) in trajectories.items():
            task = episode_tasks[episode]
            language = str(task["task_language"])
            arm_qpos, arm_layout = sample_horizons(arm_path, 7, horizons)
            gripper_qpos, gripper_layout = sample_horizons(
                gripper_path, 2, horizons
            )
            if arm_layout != gripper_layout:
                raise ValueError(
                    f"Arm/gripper trajectory layouts differ for episode {episode}"
                )
            trajectory_layouts.add(arm_layout)
            captures: list[dict[str, np.ndarray]] = []
            for arm_q, gripper_q in zip(arm_qpos, gripper_qpos):
                renderer.set_configuration(arm_q, gripper_q)
                captures.append(renderer.render_masks())
            arrays = stack_captures(captures)
            cache_path = episodes_dir / f"episode_{episode:06d}.npz"
            np.savez_compressed(
                cache_path,
                horizons=np.asarray(horizons, dtype=np.int32),
                predicted_joint_qpos=arm_qpos,
                predicted_gripper_qpos=gripper_qpos,
                **arrays,
            )
            row = {
                "episode_index": episode,
                "source_frame_index": 0,
                "action_chunk_index": 0,
                "task_index": int(task["task_index"]),
                "libero_task_index": int(task["task_index"]),
                "task_language": language,
                "joint_qpos_path": str(arm_path.resolve()),
                "joint_qpos_sha256": sha256_file(arm_path),
                "gripper_qpos_path": str(gripper_path.resolve()),
                "gripper_qpos_sha256": sha256_file(gripper_path),
                "trajectory_layout": arm_layout,
                "cache_file": str(cache_path.relative_to(args.output_dir)),
                "cache_sha256": sha256_file(cache_path),
                "robot_pixels": [int(mask.sum()) for mask in arrays["robot_mask"]],
                "gripper_pixels": [int(mask.sum()) for mask in arrays["gripper_mask"]],
            }
            rows.append(row)
            print(json.dumps(jsonable(row)), flush=True)

    if any(value != 0 for value in dynamics_counts.values()):
        raise RuntimeError(f"Forbidden dynamics call attempted: {dynamics_counts}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_mapping": "specified q + gripper q + explicit camera calibration -> standalone robot FK -> mesh rasterization -> robot mask",
        "q_source": args.q_source,
        "action_input": None,
        "simulator_future_state_input": None,
        "raw_demonstration_future_state_input": None,
        "rgb_supervision": None,
        "benchmark_environment_imported": False,
        "benchmark_environment_constructed": False,
        "benchmark_state_read": False,
        "controller_used": False,
        "contact_used": False,
        "fk_api": "mujoco.mj_kinematics",
        "dynamics": "forbidden",
        "dynamics_attempted_calls": dynamics_counts,
        "rasterizer_semantics": RASTERIZER_SEMANTICS,
        "robot_visibility": "amodal_robot_only_no_task_object_occlusion",
        "suite": args.suite,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_info_sha256": sha256_file(dataset_info_path),
        "standalone_robot_asset_dir": str(args.robot_asset_dir.resolve()),
        "standalone_robot_asset_manifest_sha256": sha256_file(asset_manifest_path),
        "camera_calibration_sha256": sha256_file(calibration_path),
        "joint_trajectories_root": str(args.joint_trajectories_root.resolve()),
        "joint_predictor_id": args.joint_predictor_id,
        "trajectory_layouts": sorted(trajectory_layouts),
        "horizons": horizons,
        "camera_profile_id": renderer.calibration["profile_id"],
        "camera_name": renderer.calibration["camera_name"],
        "image_size": [args.height, args.width],
        "image_orientation": IMAGE_ORIENTATION,
        "num_episodes": len(rows),
        "episodes": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

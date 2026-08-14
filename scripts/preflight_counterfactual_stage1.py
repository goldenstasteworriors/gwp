#!/usr/bin/env python3
"""Audit stage-one counterfactual caches, split integrity, and disk capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from generate_libero_counterfactual_mask_cache import (
    load_episode_languages,
    normalize_raw_demo_actions,
    sha256_array,
)
from libero_action_encoding import (
    POLICY_ACTION_ENCODING,
    POLICY_TO_SIMULATOR_CONVERSION,
    SIMULATOR_ACTION_ENCODING,
    lerobot_action_to_libero,
)


SPLIT_SCHEMA = "libero-counterfactual-split-v1"
CACHE_SCHEMA = "libero-counterfactual-robot-mask-v5"
EXPECTED_IMAGE_ORIENTATION = "lerobot_libero_visualization_flip_height_width"
EXPECTED_HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--heldout-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matched-output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--transient-gib", type=float, default=6.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_episode_file(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes = [int(token) for token in path.read_text(encoding="utf-8").split()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError(f"Episode list must be non-empty and unique: {path}")
    return episodes


def require_shape(name: str, value: np.ndarray, shape: tuple[int, ...], path: Path) -> None:
    if value.shape != shape:
        raise ValueError(f"{path}: {name} shape {value.shape}, expected {shape}")


def path_is_relative_to(path: Path, root: Path) -> bool:
    """Python 3.8-compatible equivalent of ``Path.is_relative_to``."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def filesystem_id(path: Path) -> int:
    """Return the filesystem id, including across OverlayFS lower/upper inodes."""

    return int(os.statvfs(path).f_fsid)


def load_source_episodes(
    dataset_root: Path,
    requested_episodes: set[int],
) -> dict[int, dict[str, Any]]:
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
    records: dict[int, dict[str, Any]] = {}
    for episode, episode_frames in frames[frames["episode_index"].isin(requested_episodes)].groupby(
        "episode_index", sort=True
    ):
        episode_frames = episode_frames.sort_values("frame_index")
        records[int(episode)] = {
            "task_index": int(episode_frames.iloc[0]["task_index"]),
            "task_language": episode_languages[int(episode)],
            "source_observation_state": np.asarray(
                episode_frames.iloc[0]["observation.state"], dtype=np.float32
            ),
            "demonstration_actions": np.stack(episode_frames["action"].to_numpy()).astype(
                np.float32
            ),
        }
    missing = sorted(requested_episodes - set(records))
    if missing:
        raise ValueError(f"Selected episodes absent from source dataset: {missing}")
    return records


def audit_cache(
    root: Path,
    expected_episodes: list[int],
    selection_file: Path,
    checkpoint: Path,
    split_manifest: dict[str, Any],
    source_episodes: dict[int, dict[str, Any]],
    raw_file_hash_cache: dict[Path, str],
    raw_demo_cache: dict[tuple[Path, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"Unexpected cache schema in {manifest_path}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError(f"Counterfactual cache must not contain RGB targets: {manifest_path}")
    if manifest.get("image_orientation") != EXPECTED_IMAGE_ORIENTATION:
        raise ValueError(f"Cache/training image orientation mismatch: {manifest_path}")
    if [int(value) for value in manifest.get("horizons", [])] != EXPECTED_HORIZONS:
        raise ValueError(f"Unexpected horizons in {manifest_path}")
    if Path(manifest["dataset_root"]).resolve() != Path(split_manifest["dataset_root"]).resolve():
        raise ValueError(f"Cache/split dataset root mismatch: {manifest_path}")
    if manifest.get("dataset_info_sha256") != split_manifest.get("dataset_info_sha256"):
        raise ValueError(f"Cache/split dataset hash mismatch: {manifest_path}")
    if (
        manifest.get("raw_demo_matching")
        != "unique_full_action_sequence_after_noop_filter_and_gripper_conversion"
    ):
        raise ValueError(f"Cache does not use authoritative raw-demo matching: {manifest_path}")
    action_match_atol = float(manifest.get("action_match_atol", -1.0))
    source_state_atol = float(manifest.get("source_state_atol", -1.0))
    if not 0.0 < action_match_atol <= 1e-5:
        raise ValueError(f"Unsafe raw-demo action tolerance in {manifest_path}")
    if not 0.0 < source_state_atol <= 1e-4:
        raise ValueError(f"Unsafe source-state tolerance in {manifest_path}")
    raw_demos_root = Path(str(manifest.get("raw_demos_root"))).resolve()
    if not raw_demos_root.is_dir():
        raise FileNotFoundError(f"Raw LIBERO demonstrations are missing: {raw_demos_root}")
    raw_files_by_task: dict[int, dict[str, Any]] = {}
    for raw_record in manifest.get("raw_demo_files", []):
        task_index = int(raw_record["task_index"])
        if task_index in raw_files_by_task:
            raise ValueError(f"Duplicate raw-demo task {task_index} in {manifest_path}")
        raw_path = Path(str(raw_record["path"])).resolve()
        if not path_is_relative_to(raw_path, raw_demos_root):
            raise ValueError(f"Raw demo file is outside raw_demos_root: {raw_path}")
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if raw_path not in raw_file_hash_cache:
            raw_file_hash_cache[raw_path] = sha256_file(raw_path)
        raw_sha256 = raw_file_hash_cache[raw_path]
        if raw_sha256 != raw_record.get("sha256"):
            raise ValueError(f"Raw demo HDF5 changed after cache generation: {raw_path}")
        raw_files_by_task[task_index] = raw_record
    libero_config_path = Path(str(manifest.get("libero_config_path"))).resolve()
    if not libero_config_path.is_file():
        raise FileNotFoundError(f"Cache LIBERO config is missing: {libero_config_path}")
    if manifest.get("libero_config_sha256") != sha256_file(libero_config_path):
        raise ValueError(f"Cache LIBERO config changed: {manifest_path}")
    if Path(str(manifest.get("policy_id"))).resolve() != checkpoint.resolve():
        raise ValueError(f"Cache policy_id does not match checkpoint: {manifest_path}")
    if manifest.get("episode_selection_sha256") != sha256_file(selection_file):
        raise ValueError(f"Episode selection hash mismatch: {manifest_path}")
    expected_action_semantics = {
        "policy_action_encoding": POLICY_ACTION_ENCODING,
        "simulator_action_encoding": SIMULATOR_ACTION_ENCODING,
        "policy_to_simulator_conversion": POLICY_TO_SIMULATOR_CONVERSION,
        "policy_action_array": "policy_actions_lerobot",
        "simulator_action_array": "simulator_actions",
    }
    for key, expected in expected_action_semantics.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Cache has unexpected {key}: {manifest_path}")

    records = list(manifest.get("episodes", []))
    got_episodes = [int(record["episode_index"]) for record in records]
    if got_episodes != expected_episodes:
        raise ValueError(f"Cache episode order/content mismatch: {manifest_path}")
    if int(manifest.get("num_episodes", -1)) != len(records):
        raise ValueError(f"Cache episode count mismatch: {manifest_path}")

    robot_pixels_by_horizon = np.zeros(5, dtype=np.int64)
    gripper_pixels_by_horizon = np.zeros(5, dtype=np.int64)
    converted_values = 0
    action_values = 0
    for record in records:
        episode = int(record["episode_index"])
        source_episode = source_episodes[episode]
        task_index = int(record["task_index"])
        if task_index != int(source_episode["task_index"]):
            raise ValueError(f"Cache/source task mismatch for episode {episode}")
        if record.get("task_language") != source_episode["task_language"]:
            raise ValueError(f"Cache/source task language mismatch for episode {episode}")
        if task_index not in raw_files_by_task:
            raise ValueError(f"No raw HDF5 provenance for task {task_index}")
        raw_metadata = raw_files_by_task[task_index]
        if raw_metadata.get("task_language") != source_episode["task_language"]:
            raise ValueError(f"Raw/source task language mismatch for episode {episode}")
        if int(record["libero_task_index"]) != int(raw_metadata["libero_task_index"]):
            raise ValueError(f"Record/raw LIBERO task index mismatch for episode {episode}")
        raw_hdf5_path = Path(record["raw_hdf5_path"]).resolve()
        if raw_hdf5_path != Path(str(raw_metadata["path"])).resolve():
            raise ValueError(f"Record/manifest raw HDF5 mismatch for episode {episode}")
        if record.get("raw_hdf5_sha256") != raw_metadata.get("sha256"):
            raise ValueError(f"Record raw HDF5 hash mismatch for episode {episode}")
        raw_demo_name = str(record["raw_demo_name"])
        raw_demo_key = (raw_hdf5_path, raw_demo_name)
        if raw_demo_key not in raw_demo_cache:
            with h5py.File(raw_hdf5_path, "r") as raw_file:
                problem_info = json.loads(str(raw_file["data"].attrs["problem_info"]))
                if str(problem_info.get("language_instruction", "")).strip() != str(
                    raw_metadata["task_language"]
                ).strip():
                    raise ValueError(f"Raw HDF5/task language mismatch: {raw_hdf5_path}")
                expected_filename = f"{raw_metadata['task_name']}_demo.hdf5"
                if raw_hdf5_path.name != expected_filename:
                    raise ValueError(
                        f"Raw HDF5 filename {raw_hdf5_path.name!r} != {expected_filename!r}"
                    )
                demo_path = f"data/{raw_demo_name}"
                if demo_path not in raw_file:
                    raise ValueError(f"Missing raw demo {raw_hdf5_path}:{demo_path}")
                raw_demo = raw_file[demo_path]
                raw_demo_cache[raw_demo_key] = (
                    np.asarray(raw_demo["states"][0], dtype=np.float64),
                    normalize_raw_demo_actions(raw_demo["actions"][()]),
                )
        authoritative_raw_state, authoritative_demo_actions = raw_demo_cache[raw_demo_key]
        source_demo_actions = source_episode["demonstration_actions"]
        if authoritative_demo_actions.shape != source_demo_actions.shape:
            raise ValueError(f"Raw/source action length mismatch for episode {episode}")
        computed_action_match_max_abs = float(
            np.max(np.abs(authoritative_demo_actions - source_demo_actions))
        )
        if computed_action_match_max_abs > action_match_atol:
            raise ValueError(f"Raw/source action mismatch for episode {episode}")
        if not np.isclose(
            computed_action_match_max_abs,
            float(record["raw_demo_action_match_max_abs"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Raw action-match summary mismatch for episode {episode}")
        if int(record.get("source_frame_index", -1)) != 0:
            raise ValueError(f"Nonzero cache source frame: {record}")
        if int(record.get("action_chunk_index", -1)) != 0:
            raise ValueError(f"Nonzero action chunk index: {record}")
        prediction_path = Path(record["prediction_path"])
        cache_path = root / record["cache_file"]
        if sha256_file(prediction_path) != record["prediction_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        if sha256_file(cache_path) != record["cache_sha256"]:
            raise ValueError(f"Cache hash mismatch: {cache_path}")

        with np.load(cache_path, allow_pickle=False) as arrays:
            required = {
                "horizons",
                "policy_actions_lerobot",
                "simulator_actions",
                "raw_initial_state",
                "source_observation_state",
                "rendered_initial_observation_state",
                "robot_mask",
                "gripper_mask",
                "qpos",
                "eef_position",
                "eef_rotation",
                "camera_position",
                "camera_rotation",
                "camera_fovy",
            }
            missing = sorted(required - set(arrays.files))
            if missing:
                raise ValueError(f"{cache_path}: missing arrays {missing}")
            horizons = arrays["horizons"]
            policy_actions = arrays["policy_actions_lerobot"]
            simulator_actions = arrays["simulator_actions"]
            raw_initial_state = arrays["raw_initial_state"]
            source_observation_state = arrays["source_observation_state"]
            rendered_initial_observation_state = arrays[
                "rendered_initial_observation_state"
            ]
            robot_mask = arrays["robot_mask"]
            gripper_mask = arrays["gripper_mask"]
            require_shape("horizons", horizons, (5,), cache_path)
            require_shape(
                "policy_actions_lerobot", policy_actions, (48, 7), cache_path
            )
            require_shape("simulator_actions", simulator_actions, (48, 7), cache_path)
            prediction_actions = np.load(prediction_path, allow_pickle=False)
            if prediction_actions.ndim == 3 and prediction_actions.shape[0] == 1:
                prediction_actions = prediction_actions[0]
            if prediction_actions.ndim != 2 or prediction_actions.shape[0] < 48:
                raise ValueError(
                    f"{prediction_path}: expected at least [48,D] policy actions, "
                    f"got {prediction_actions.shape}"
                )
            prediction_actions = np.asarray(
                prediction_actions[:48, :7], dtype=np.float32
            )
            if not np.array_equal(policy_actions, prediction_actions):
                raise ValueError(
                    f"{cache_path}: policy_actions_lerobot does not match the "
                    "hashed policy prediction"
                )
            require_shape(
                "raw_initial_state",
                raw_initial_state,
                authoritative_raw_state.shape,
                cache_path,
            )
            require_shape("source_observation_state", source_observation_state, (8,), cache_path)
            require_shape(
                "rendered_initial_observation_state",
                rendered_initial_observation_state,
                (8,),
                cache_path,
            )
            require_shape("robot_mask", robot_mask, (5, 256, 256), cache_path)
            require_shape("gripper_mask", gripper_mask, (5, 256, 256), cache_path)
            if horizons.tolist() != EXPECTED_HORIZONS:
                raise ValueError(f"{cache_path}: wrong horizons")
            expected_simulator_actions = lerobot_action_to_libero(policy_actions)
            if not np.array_equal(simulator_actions, expected_simulator_actions):
                raise ValueError(
                    f"{cache_path}: simulator_actions has wrong simulator encoding"
                )
            if not np.array_equal(raw_initial_state, authoritative_raw_state):
                raise ValueError(f"{cache_path}: raw initial state changed")
            if sha256_array(raw_initial_state) != record.get("raw_initial_state_sha256"):
                raise ValueError(f"{cache_path}: raw initial state hash mismatch")
            if not np.array_equal(
                source_observation_state, source_episode["source_observation_state"]
            ):
                raise ValueError(f"{cache_path}: source frame-0 state mismatch")
            source_state_delta = rendered_initial_observation_state - source_observation_state
            source_state_max_abs = float(np.max(np.abs(source_state_delta)))
            source_state_l2 = float(np.linalg.norm(source_state_delta))
            if source_state_max_abs > source_state_atol:
                raise ValueError(f"{cache_path}: rendered initial state misses source frame 0")
            if not np.isclose(
                source_state_max_abs,
                float(record["source_state_max_abs"]),
                rtol=0.0,
                atol=1e-12,
            ) or not np.isclose(
                source_state_l2,
                float(record["source_state_l2"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{cache_path}: source-state summary mismatch")
            if not np.allclose(
                arrays["eef_position"][0],
                rendered_initial_observation_state[:3],
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(f"{cache_path}: horizon-0 EEF/state mismatch")
            if not np.logical_or(robot_mask == 0, robot_mask == 1).all():
                raise ValueError(f"{cache_path}: robot mask is not binary")
            if not np.logical_or(gripper_mask == 0, gripper_mask == 1).all():
                raise ValueError(f"{cache_path}: gripper mask is not binary")
            if np.logical_and(gripper_mask > 0, robot_mask == 0).any():
                raise ValueError(f"{cache_path}: gripper mask is not a subset of robot mask")
            for key in required - {"horizons", "robot_mask", "gripper_mask"}:
                if not np.isfinite(arrays[key]).all():
                    raise ValueError(f"{cache_path}: non-finite values in {key}")
            robot_pixels = robot_mask.sum(axis=(-2, -1), dtype=np.int64)
            if robot_pixels.tolist() != [int(value) for value in record["robot_pixels"]]:
                raise ValueError(f"{cache_path}: robot pixel summary mismatch")
            robot_pixels_by_horizon += robot_pixels
            gripper_pixels_by_horizon += gripper_mask.sum(axis=(-2, -1), dtype=np.int64)
            converted_values += int(
                np.count_nonzero(policy_actions != simulator_actions)
            )
            action_values += int(policy_actions.size)

    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "num_episodes": len(records),
        "robot_pixels_mean": (robot_pixels_by_horizon / len(records)).tolist(),
        "gripper_pixels_mean": (gripper_pixels_by_horizon / len(records)).tolist(),
        "action_values_changed_by_sim_conversion_fraction": float(
            converted_values / action_values
        ),
        "metadata": {
            key: manifest.get(key)
            for key in (
                "suite",
                "policy_id",
                "camera_name",
                "image_orientation",
                "libero_config_sha256",
                "raw_demo_matching",
                "raw_demo_files",
                "action_match_atol",
                "source_state_atol",
                "policy_action_encoding",
                "simulator_action_encoding",
                "policy_to_simulator_conversion",
                "policy_action_array",
                "simulator_action_array",
                "image_size",
                "num_steps_wait",
            )
        },
        "raw_demos_root": str(raw_demos_root),
        "raw_demo_files": [str(Path(str(value["path"])).resolve()) for value in raw_files_by_task.values()],
    }


def checkpoint_weight_bytes(checkpoint: Path) -> int:
    files = sorted(checkpoint.glob("*.bin")) + sorted(checkpoint.glob("*.safetensors"))
    total = sum(path.stat().st_size for path in files if path.is_file())
    if total <= 0:
        raise FileNotFoundError(f"No model weights in checkpoint: {checkpoint}")
    return total


def main() -> None:
    args = parse_args()
    if args.reserve_gib <= 0 or args.transient_gib <= 0:
        raise ValueError("--reserve-gib and --transient-gib must be positive")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite preflight report: {args.report}")
    if not args.project_root.is_dir() or not (args.project_root / ".git").exists():
        raise FileNotFoundError(f"Not a Git project: {args.project_root}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    for output in (args.matched_output, args.full_output):
        if output.exists():
            raise FileExistsError(f"Refusing existing stage-one output: {output}")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"Stage-one output parent does not exist: {output.parent}")
    if filesystem_id(args.matched_output.parent) != filesystem_id(
        args.full_output.parent
    ):
        raise ValueError("Matched and full outputs must use the same mounted filesystem")
    output_filesystem = filesystem_id(args.matched_output.parent)
    for source in (args.checkpoint, args.train_cache, args.heldout_cache):
        if filesystem_id(source) != output_filesystem:
            raise ValueError(f"Large stage-one source/output crosses filesystems: {source}")

    split_manifest_path = args.split_dir / "manifest.json"
    split_manifest = load_json(split_manifest_path)
    if split_manifest.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError(f"Unexpected split schema: {split_manifest_path}")
    train_file = args.split_dir / split_manifest["train_episodes_file"]
    heldout_file = args.split_dir / split_manifest["heldout_episodes_file"]
    train_episodes = read_episode_file(train_file)
    heldout_episodes = read_episode_file(heldout_file)
    if train_episodes != [int(value) for value in split_manifest["train_episodes"]]:
        raise ValueError("Train episode file does not match split manifest")
    if heldout_episodes != [int(value) for value in split_manifest["heldout_episodes"]]:
        raise ValueError("Held-out episode file does not match split manifest")
    overlap = sorted(set(train_episodes) & set(heldout_episodes))
    if overlap:
        raise ValueError(f"Train/held-out split overlap: {overlap}")
    selected_lengths = [
        int(split_manifest["episode_lengths"][str(episode)])
        for episode in train_episodes + heldout_episodes
    ]
    if min(selected_lengths) < 49:
        raise ValueError("Every selected episode must cover frame 0 plus a 48-step chunk")

    source_episodes = load_source_episodes(
        Path(split_manifest["dataset_root"]).resolve(),
        set(train_episodes + heldout_episodes),
    )
    raw_file_hash_cache: dict[Path, str] = {}
    raw_demo_cache: dict[tuple[Path, str], tuple[np.ndarray, np.ndarray]] = {}
    train_audit = audit_cache(
        args.train_cache,
        train_episodes,
        train_file,
        args.checkpoint,
        split_manifest,
        source_episodes,
        raw_file_hash_cache,
        raw_demo_cache,
    )
    heldout_audit = audit_cache(
        args.heldout_cache,
        heldout_episodes,
        heldout_file,
        args.checkpoint,
        split_manifest,
        source_episodes,
        raw_file_hash_cache,
        raw_demo_cache,
    )
    if train_audit["metadata"] != heldout_audit["metadata"]:
        raise ValueError("Train and held-out cache metadata differ")
    for raw_file in train_audit["raw_demo_files"] + heldout_audit["raw_demo_files"]:
        if filesystem_id(Path(raw_file)) != output_filesystem:
            raise ValueError(f"Raw demo/cache/output crosses filesystems: {raw_file}")

    source_weight_bytes = checkpoint_weight_bytes(args.checkpoint)
    # Two BF16 checkpoints are approximately one FP32 source checkpoint.  Keep
    # an explicit reserve for logs, temporary save overhead, and filesystem health.
    reserve_bytes = int(args.reserve_gib * 1024**3)
    transient_bytes = int(args.transient_gib * 1024**3)
    required_free_bytes = source_weight_bytes + reserve_bytes + transient_bytes
    free_bytes = shutil.disk_usage(args.matched_output.parent).free
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            f"Insufficient disk: free={free_bytes / 1024**3:.2f} GiB, "
            f"required={required_free_bytes / 1024**3:.2f} GiB"
        )

    git_commit = subprocess.check_output(
        ["git", "-C", str(args.project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_status = subprocess.check_output(
        ["git", "-C", str(args.project_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if tracked_status:
        raise RuntimeError(f"Tracked project files are dirty:\n{tracked_status}")

    report = {
        "status": "passed",
        "project_root": str(args.project_root.resolve()),
        "git_commit": git_commit,
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "num_train": len(train_episodes),
        "num_heldout": len(heldout_episodes),
        "episode_overlap": 0,
        "checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_weight_gib": source_weight_bytes / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "required_free_gib": required_free_bytes / 1024**3,
        "reserve_gib": args.reserve_gib,
        "transient_gib": args.transient_gib,
        "matched_output": str(args.matched_output.resolve()),
        "full_output": str(args.full_output.resolve()),
        "train_cache": train_audit,
        "heldout_cache": heldout_audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(args.report, flush=True)


if __name__ == "__main__":
    main()

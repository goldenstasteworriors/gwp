#!/usr/bin/env python3
"""Build logged LIBERO action-to-joint supervision without a simulator.

The output contains only action, arm-q, and gripper-q sequences recorded in
the authoritative HDF5 demonstrations.  It imports no LIBERO / MuJoCo runtime,
does not replay actions, and is intended solely to train and evaluate a
deployment-time action-to-joint predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from generate_libero_counterfactual_mask_cache import (
    load_episode_records,
    load_raw_demo_candidates,
    match_raw_demo,
    normalize_raw_demo_actions,
)


SCHEMA_VERSION = "libero-logged-action-joint-trajectory-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-demos-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--train-episodes-file", type=Path, required=True)
    parser.add_argument("--heldout-episodes-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-match-atol", type=float, default=1e-6)
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


def read_episode_file(path: Path) -> list[int]:
    values = [int(token) for token in path.read_text(encoding="utf-8").split()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Episode file must be non-empty and unique: {path}")
    return values


def kept_action_indices(actions: np.ndarray) -> np.ndarray:
    kept: list[int] = []
    previous: np.ndarray | None = None
    for index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        is_noop = np.linalg.norm(action[:-1]) < 1e-4 and (
            previous is None or action[-1] == previous[-1]
        )
        if not is_noop:
            kept.append(index)
            previous = action.copy()
    if not kept:
        raise ValueError("Raw demonstration contains only no-op actions")
    return np.asarray(kept, dtype=np.int64)


def task_name_from_language(raw_root: Path, language: str) -> tuple[str, Path]:
    matches: list[tuple[str, Path]] = []
    for path in sorted(raw_root.glob("*_demo.hdf5")):
        with h5py.File(path, "r") as raw_file:
            problem_info = json.loads(str(raw_file["data"].attrs["problem_info"]))
        if str(problem_info.get("language_instruction", "")).strip() == language:
            matches.append((path.name[: -len("_demo.hdf5")], path))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one raw HDF5 for language {language!r}, found {len(matches)}"
        )
    return matches[0]


def build_split(
    name: str,
    episodes: list[int],
    episode_records: dict[int, dict[str, Any]],
    raw_demos_root: Path,
    output_root: Path,
    action_match_atol: float,
) -> dict[str, Any]:
    split_root = output_root / name
    episodes_root = split_root / "episodes"
    episodes_root.mkdir(parents=True)
    raw_candidate_cache: dict[Path, list[dict[str, Any]]] = {}
    raw_metadata_cache: dict[str, tuple[str, Path]] = {}
    raw_hash_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []

    for episode in episodes:
        if episode not in episode_records:
            raise ValueError(f"Episode {episode} is absent from LeRobot dataset")
        source = episode_records[episode]
        language = str(source["task_language"])
        if language not in raw_metadata_cache:
            raw_metadata_cache[language] = task_name_from_language(
                raw_demos_root, language
            )
        task_name, raw_path = raw_metadata_cache[language]
        if raw_path not in raw_candidate_cache:
            raw_candidate_cache[raw_path] = load_raw_demo_candidates(
                raw_path, language
            )
        if raw_path not in raw_hash_cache:
            raw_hash_cache[raw_path] = sha256_file(raw_path)
        matched, match_error = match_raw_demo(
            episode,
            source["demonstration_actions"],
            raw_candidate_cache[raw_path],
            action_match_atol,
        )
        demo_name = str(matched["demo_name"])
        with h5py.File(raw_path, "r") as raw_file:
            demo = raw_file[f"data/{demo_name}"]
            raw_actions = np.asarray(demo["actions"], dtype=np.float32)
            arm_qpos_raw = np.asarray(
                demo["obs/joint_states"], dtype=np.float32
            )
            gripper_qpos_raw = np.asarray(
                demo["obs/gripper_states"], dtype=np.float32
            )
        if not (
            len(raw_actions) == len(arm_qpos_raw) == len(gripper_qpos_raw)
        ):
            raise ValueError(f"Raw action/q length mismatch: {raw_path}:{demo_name}")
        kept = kept_action_indices(raw_actions)
        policy_actions = normalize_raw_demo_actions(raw_actions)
        arm_qpos = arm_qpos_raw[kept]
        gripper_qpos = gripper_qpos_raw[kept]
        if not np.array_equal(policy_actions, source["demonstration_actions"]):
            max_abs = float(
                np.max(np.abs(policy_actions - source["demonstration_actions"]))
            )
            if max_abs > action_match_atol:
                raise ValueError(
                    f"Filtered action/source mismatch episode {episode}: {max_abs}"
                )
        if not (
            len(policy_actions) == len(arm_qpos) == len(gripper_qpos)
        ):
            raise RuntimeError(f"Filtered action/q length mismatch episode {episode}")
        if len(policy_actions) <= 48:
            raise ValueError(f"Episode {episode} has no full 48-step training window")
        if not all(
            np.isfinite(value).all()
            for value in (policy_actions, arm_qpos, gripper_qpos)
        ):
            raise ValueError(f"Non-finite action/q values episode {episode}")

        cache_path = episodes_root / f"episode_{episode:06d}.npz"
        np.savez_compressed(
            cache_path,
            policy_actions_lerobot=policy_actions.astype(np.float32),
            arm_qpos=arm_qpos.astype(np.float32),
            gripper_qpos=gripper_qpos.astype(np.float32),
            raw_kept_indices=kept.astype(np.int32),
        )
        row = {
            "episode_index": episode,
            "task_index": int(source["task_index"]),
            "task_language": language,
            "task_name": task_name,
            "raw_hdf5_path": str(raw_path.resolve()),
            "raw_hdf5_sha256": raw_hash_cache[raw_path],
            "raw_demo_name": demo_name,
            "raw_demo_action_match_max_abs": float(match_error),
            "num_steps": int(len(policy_actions)),
            "num_full_48_windows": int(len(policy_actions) - 48),
            "policy_actions_sha256": sha256_array(policy_actions),
            "arm_qpos_sha256": sha256_array(arm_qpos),
            "gripper_qpos_sha256": sha256_array(gripper_qpos),
            "cache_file": str(cache_path.relative_to(split_root)),
            "cache_sha256": sha256_file(cache_path),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": name,
        "supervision_semantics": (
            "logged expert action and synchronously recorded robot joint trajectory"
        ),
        "counterfactual_supervision": False,
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "rgb_supervision": None,
        "action_encoding": "LeRobot LIBERO: arm OSC normalized; gripper 1=open,0=closed",
        "arm_qpos_units": "radians",
        "gripper_qpos_units": "meters_with_opposite_signed_finger_joints",
        "num_episodes": len(rows),
        "num_full_48_windows": int(
            sum(row["num_full_48_windows"] for row in rows)
        ),
        "episodes": rows,
    }
    manifest_path = split_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "root": str(split_root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "num_episodes": len(rows),
        "num_full_48_windows": manifest["num_full_48_windows"],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not args.dataset_root.is_dir() or not args.raw_demos_root.is_dir():
        raise FileNotFoundError("Dataset/raw-demo root is missing")
    if not 0.0 < args.action_match_atol <= 1e-5:
        raise ValueError("--action-match-atol must be in (0,1e-5]")
    train_episodes = read_episode_file(args.train_episodes_file)
    heldout_episodes = read_episode_file(args.heldout_episodes_file)
    overlap = sorted(set(train_episodes) & set(heldout_episodes))
    if overlap:
        raise ValueError(f"Train/heldout overlap: {overlap}")
    requested = set(train_episodes) | set(heldout_episodes)
    all_records = load_episode_records(args.dataset_root)
    episode_records = {episode: all_records[episode] for episode in requested}

    args.output_dir.mkdir(parents=True)
    train_summary = build_split(
        "train",
        train_episodes,
        episode_records,
        args.raw_demos_root,
        args.output_dir,
        args.action_match_atol,
    )
    heldout_summary = build_split(
        "heldout",
        heldout_episodes,
        episode_records,
        args.raw_demos_root,
        args.output_dir,
        args.action_match_atol,
    )
    dataset_info = args.dataset_root / "meta/info.json"
    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite": args.suite,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_info_sha256": sha256_file(dataset_info),
        "raw_demos_root": str(args.raw_demos_root.resolve()),
        "train_episodes_file": str(args.train_episodes_file.resolve()),
        "train_episodes_sha256": sha256_file(args.train_episodes_file),
        "heldout_episodes_file": str(args.heldout_episodes_file.resolve()),
        "heldout_episodes_sha256": sha256_file(args.heldout_episodes_file),
        "episode_overlap": overlap,
        "action_match_atol": args.action_match_atol,
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "train": train_summary,
        "heldout": heldout_summary,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

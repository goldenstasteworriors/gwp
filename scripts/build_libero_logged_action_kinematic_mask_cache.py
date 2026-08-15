#!/usr/bin/env python3
"""Bind logged expert actions to simultaneous direct-q kinematic masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


JOINT_DATA_SCHEMA = "libero-logged-action-joint-trajectory-v1"
JOINT_MASK_SCHEMA = "libero-joint-standalone-kinematic-robot-mask-v2"
OUTPUT_SCHEMA = "libero-logged-action-standalone-mask-v2"
EXPECTED_HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-dataset-split", type=Path, required=True)
    parser.add_argument("--joint-mask-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def index_records(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {
        int(record["episode_index"]): record for record in manifest.get("episodes", [])
    }
    if len(records) != int(manifest.get("num_episodes", -1)) or not records:
        raise ValueError("Invalid episode records")
    return records


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    joint_manifest_path = args.joint_dataset_split / "manifest.json"
    mask_manifest_path = args.joint_mask_cache / "manifest.json"
    joint = load_json(joint_manifest_path)
    masks = load_json(mask_manifest_path)
    if joint.get("schema_version") != JOINT_DATA_SCHEMA:
        raise ValueError("Unexpected logged action/q schema")
    if joint.get("simulator_runtime_imported") is not False or int(
        joint.get("simulator_step_calls", -1)
    ) != 0:
        raise ValueError("Logged action/q dataset queried a simulator")
    if masks.get("schema_version") != JOINT_MASK_SCHEMA:
        raise ValueError("Unexpected direct-q mask schema")
    if masks.get("q_source") != "logged_demonstration_joint_trajectory":
        raise ValueError("Expert mask must use simultaneous logged q")
    calls = masks.get("dynamics_attempted_calls", {})
    if masks.get("dynamics") != "forbidden" or set(calls) != {
        "mj_step",
        "mj_step1",
        "mj_step2",
        "mj_forward",
        "mj_forwardSkip",
    } or any(int(value) != 0 for value in calls.values()):
        raise ValueError("Direct-q mask cache attempted dynamics")
    for key in (
        "benchmark_environment_imported",
        "benchmark_environment_constructed",
        "benchmark_state_read",
        "controller_used",
        "contact_used",
    ):
        if masks.get(key) is not False:
            raise ValueError(f"Standalone direct-q mask violates {key}")
    if masks.get("fk_api") != "mujoco.mj_kinematics":
        raise ValueError("Direct-q mask did not use standalone FK")

    joint_records = index_records(joint)
    mask_records = index_records(masks)
    if set(joint_records) != set(mask_records):
        raise ValueError("Logged action/q and mask episode sets differ")
    args.output_dir.mkdir(parents=True)
    episodes_root = args.output_dir / "episodes"
    episodes_root.mkdir()
    rows: list[dict[str, Any]] = []
    for episode, joint_record in joint_records.items():
        mask_record = mask_records[episode]
        for key in ("task_index", "task_language"):
            if joint_record.get(key) != mask_record.get(key):
                raise ValueError(f"Episode {episode} provenance differs for {key}")
        joint_path = args.joint_dataset_split / str(joint_record["cache_file"])
        mask_path = args.joint_mask_cache / str(mask_record["cache_file"])
        if sha256_file(joint_path) != joint_record["cache_sha256"]:
            raise ValueError(f"Logged action/q cache changed: {joint_path}")
        if sha256_file(mask_path) != mask_record["cache_sha256"]:
            raise ValueError(f"Direct-q mask cache changed: {mask_path}")
        with np.load(joint_path, allow_pickle=False) as arrays:
            action = np.asarray(arrays["policy_actions_lerobot"][:48], dtype=np.float32)
        if action.shape != (48, 7):
            raise ValueError(f"Wrong expert action shape episode {episode}")
        with np.load(mask_path, allow_pickle=False) as arrays:
            selected = {
                key: np.asarray(arrays[key])
                for key in (
                    "horizons",
                    "predicted_joint_qpos",
                    "predicted_gripper_qpos",
                    "robot_mask",
                    "gripper_mask",
                    "eef_position",
                    "eef_rotation",
                    "camera_position",
                    "camera_rotation",
                    "camera_intrinsic",
                    "camera_fovy",
                )
            }
        if selected["horizons"].tolist() != EXPECTED_HORIZONS:
            raise ValueError(f"Wrong mask horizons episode {episode}")
        output_path = episodes_root / f"episode_{episode:06d}.npz"
        np.savez_compressed(output_path, expert_actions_lerobot=action, **selected)
        row = {
            "episode_index": episode,
            "source_frame_index": 0,
            "action_chunk_index": 0,
            "task_index": int(joint_record["task_index"]),
            "libero_task_index": int(mask_record["libero_task_index"]),
            "task_language": str(joint_record["task_language"]),
            "raw_demo_name": str(joint_record["raw_demo_name"]),
            "raw_hdf5_path": str(joint_record["raw_hdf5_path"]),
            "raw_hdf5_sha256": str(joint_record["raw_hdf5_sha256"]),
            "logged_action_q_cache_sha256": joint_record["cache_sha256"],
            "direct_q_mask_cache_sha256": mask_record["cache_sha256"],
            "cache_file": str(output_path.relative_to(args.output_dir)),
            "cache_sha256": sha256_file(output_path),
        }
        rows.append(row)

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "dataset_root": masks["dataset_root"],
        "dataset_info_sha256": masks["dataset_info_sha256"],
        "suite": masks["suite"],
        "horizons": EXPECTED_HORIZONS,
        "camera_name": masks["camera_name"],
        "image_size": masks["image_size"],
        "image_orientation": masks["image_orientation"],
        "rgb_supervision": None,
        "action_array": "expert_actions_lerobot",
        "q_source": "logged demonstration observations simultaneous with real video",
        "target_mapping": masks["target_mapping"],
        "robot_visibility": masks["robot_visibility"],
        "dynamics": "forbidden",
        "dynamics_attempted_calls": calls,
        "benchmark_environment_imported": False,
        "benchmark_environment_constructed": False,
        "benchmark_state_read": False,
        "controller_used": False,
        "contact_used": False,
        "fk_api": "mujoco.mj_kinematics",
        "joint_dataset_split": str(args.joint_dataset_split.resolve()),
        "joint_dataset_manifest_sha256": sha256_file(joint_manifest_path),
        "joint_mask_cache": str(args.joint_mask_cache.resolve()),
        "joint_mask_manifest_sha256": sha256_file(mask_manifest_path),
        "num_episodes": len(rows),
        "episodes": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"num_episodes": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build training sidecars for support-constrained action consequence masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


INFERENCE_SCHEMA = "libero-support-constrained-action-to-joint-inference-v1"
MASK_SCHEMA = "libero-joint-kinematic-robot-mask-v1"
SOURCE_POLICY_SCHEMA = "libero-policy-action-kinematic-mask-v1"
OUTPUT_SCHEMA = "libero-support-constrained-action-kinematic-mask-v1"
HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-inference", type=Path, required=True)
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
        int(record["episode_index"]): record
        for record in manifest.get("episodes", [])
    }
    if not records or len(records) != int(manifest.get("num_episodes", -1)):
        raise ValueError("Invalid episode records")
    return records


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    inference_path = args.joint_inference / "manifest.json"
    mask_path = args.joint_mask_cache / "manifest.json"
    inference = load_json(inference_path)
    masks = load_json(mask_path)
    if inference.get("schema_version") != INFERENCE_SCHEMA:
        raise ValueError("Unexpected support-constrained q inference schema")
    if inference.get("simulator_runtime_imported") is not False or int(
        inference.get("simulator_step_calls", -1)
    ) != 0:
        raise ValueError("Support-constrained q inference queried a simulator")
    if masks.get("schema_version") != MASK_SCHEMA:
        raise ValueError("Unexpected direct-q mask schema")
    if Path(str(masks.get("joint_trajectories_root"))).resolve() != args.joint_inference.resolve():
        raise ValueError("Mask cache refers to another q inference")
    calls = masks.get("dynamics_attempted_calls", {})
    if masks.get("dynamics") != "forbidden" or set(calls) != {
        "env_step",
        "sim_step",
        "mj_step",
    } or any(int(value) != 0 for value in calls.values()):
        raise ValueError("Direct-q mask cache attempted dynamics")

    source_policy_root = Path(str(inference["policy_cache"]))
    source_policy_path = source_policy_root / "manifest.json"
    source_policy = load_json(source_policy_path)
    if source_policy.get("schema_version") != SOURCE_POLICY_SCHEMA:
        raise ValueError("Unexpected source policy schema")
    if sha256_file(source_policy_path) != inference["policy_cache_manifest_sha256"]:
        raise ValueError("Source policy changed after q inference")
    inference_records = index_records(inference)
    mask_records = index_records(masks)
    if list(inference_records) != list(mask_records):
        raise ValueError("Inference/mask episode order differs")

    args.output_dir.mkdir(parents=True)
    episodes_root = args.output_dir / "episodes"
    episodes_root.mkdir()
    rows: list[dict[str, Any]] = []
    for episode, inference_record in inference_records.items():
        mask_record = mask_records[episode]
        action_path = args.joint_inference / str(inference_record["condition_action_path"])
        direct_mask_path = args.joint_mask_cache / str(mask_record["cache_file"])
        if sha256_file(action_path) != inference_record["condition_action_sha256"]:
            raise ValueError(f"Condition action changed: {action_path}")
        if sha256_file(direct_mask_path) != mask_record["cache_sha256"]:
            raise ValueError(f"Direct-q mask changed: {direct_mask_path}")
        action = np.asarray(np.load(action_path, allow_pickle=False), dtype=np.float32)
        if action.shape != (48, 7):
            raise ValueError(f"Wrong condition action shape episode {episode}")
        with np.load(direct_mask_path, allow_pickle=False) as arrays:
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
                    "camera_fovy",
                )
            }
        if selected["horizons"].tolist() != HORIZONS:
            raise ValueError(f"Wrong horizons episode {episode}")
        output_path = episodes_root / f"episode_{episode:06d}.npz"
        np.savez_compressed(output_path, policy_actions_lerobot=action, **selected)
        rows.append(
            {
                "episode_index": episode,
                "source_frame_index": 0,
                "action_chunk_index": 0,
                "task_index": int(inference_record["task_index"]),
                "libero_task_index": int(mask_record["libero_task_index"]),
                "task_language": str(inference_record["task_language"]),
                "maximum_supported_alpha": float(
                    inference_record["maximum_supported_alpha"]
                ),
                "condition_action_sha256": inference_record[
                    "condition_action_sha256"
                ],
                "joint_inference_arm_sha256": inference_record["arm_sha256"],
                "joint_inference_gripper_sha256": inference_record[
                    "gripper_sha256"
                ],
                "direct_q_mask_cache_sha256": mask_record["cache_sha256"],
                "cache_file": str(output_path.relative_to(args.output_dir)),
                "cache_sha256": sha256_file(output_path),
            }
        )

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "split": inference["split"],
        "dataset_root": masks["dataset_root"],
        "dataset_info_sha256": masks["dataset_info_sha256"],
        "suite": masks["suite"],
        "policy_id": source_policy["policy_id"],
        "policy_action_array": "policy_actions_lerobot",
        "condition_action_semantics": inference["action_semantics"],
        "alpha_selection": inference["alpha_selection"],
        "alpha_counts": inference["alpha_counts"],
        "action_chunk_index": 0,
        "horizons": HORIZONS,
        "camera_name": masks["camera_name"],
        "image_size": masks["image_size"],
        "image_orientation": masks["image_orientation"],
        "rgb_supervision": None,
        "q_source": "frozen action-to-q predictor on support-approved interpolated action",
        "target_mapping": masks["target_mapping"],
        "robot_visibility": masks["robot_visibility"],
        "dynamics": "forbidden",
        "dynamics_attempted_calls": calls,
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "legacy_rollout_masks_read": False,
        "legacy_rollout_qpos_read": False,
        "simulator_actions_read": False,
        "joint_inference": str(args.joint_inference.resolve()),
        "joint_inference_manifest_sha256": sha256_file(inference_path),
        "joint_mask_cache": str(args.joint_mask_cache.resolve()),
        "joint_mask_manifest_sha256": sha256_file(mask_path),
        "input_support_report": inference["input_support_report"],
        "input_support_report_sha256": inference["input_support_report_sha256"],
        "num_episodes": len(rows),
        "episodes": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"num_episodes": len(rows), "alpha_counts": output["alpha_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

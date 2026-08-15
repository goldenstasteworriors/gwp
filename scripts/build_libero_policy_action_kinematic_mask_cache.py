#!/usr/bin/env python3
"""Bind policy actions to simulator-free predicted-q robot masks.

The legacy v5 cache is used only as a hash-bound source of
``policy_actions_lerobot`` and episode provenance.  All q and mask targets are
read from the direct FK + robot-mesh rasterizer output.  Legacy simulator
actions, rollout qpos, segmentation masks, and RGB are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POLICY_SCHEMA = "libero-counterfactual-robot-mask-v5"
JOINT_INFERENCE_SCHEMA = "libero-policy-action-to-joint-inference-v1"
JOINT_MASK_SCHEMA = "libero-joint-kinematic-robot-mask-v1"
GEOMETRY_GATE_SCHEMA = "libero-action-to-joint-geometry-audit-v1"
OUTPUT_SCHEMA = "libero-policy-action-kinematic-mask-v1"
EXPECTED_HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-cache", type=Path, required=True)
    parser.add_argument("--joint-inference", type=Path, required=True)
    parser.add_argument("--joint-mask-cache", type=Path, required=True)
    parser.add_argument("--geometry-gate-report", type=Path, required=True)
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


def indexed_records(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {
        int(record["episode_index"]): record for record in manifest.get("episodes", [])
    }
    if len(records) != int(manifest.get("num_episodes", -1)) or not records:
        raise ValueError("Invalid or duplicate episode records")
    return records


def require_zero_dynamics(manifest: dict[str, Any]) -> None:
    if manifest.get("dynamics") != "forbidden":
        raise ValueError("Joint masks do not forbid dynamics")
    calls = manifest.get("dynamics_attempted_calls", {})
    if set(calls) != {"env_step", "sim_step", "mj_step"} or any(
        int(value) != 0 for value in calls.values()
    ):
        raise ValueError(f"Joint-mask dynamics attempt detected: {calls}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    policy_manifest_path = args.policy_cache / "manifest.json"
    inference_manifest_path = args.joint_inference / "manifest.json"
    mask_manifest_path = args.joint_mask_cache / "manifest.json"
    policy = load_json(policy_manifest_path)
    inference = load_json(inference_manifest_path)
    masks = load_json(mask_manifest_path)
    geometry = load_json(args.geometry_gate_report)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("Policy action source must be the explicit v5 schema")
    if policy.get("policy_action_array") != "policy_actions_lerobot":
        raise ValueError("Ambiguous policy action array")
    if inference.get("schema_version") != JOINT_INFERENCE_SCHEMA:
        raise ValueError("Unexpected joint-inference schema")
    if inference.get("policy_cache_manifest_sha256") != sha256_file(policy_manifest_path):
        raise ValueError("Joint inference/policy cache mismatch")
    if inference.get("legacy_rollout_masks_read") is not False or inference.get(
        "legacy_rollout_qpos_read"
    ) is not False or inference.get("simulator_actions_read") is not False:
        raise ValueError("Joint inference read a prohibited legacy field")
    if inference.get("simulator_runtime_imported") is not False or int(
        inference.get("simulator_step_calls", -1)
    ) != 0:
        raise ValueError("Joint inference queried simulator runtime")
    if masks.get("schema_version") != JOINT_MASK_SCHEMA:
        raise ValueError("Unexpected direct-q mask schema")
    if masks.get("q_source") != "external_joint_predictor_outputs_only":
        raise ValueError("Policy mask q must come from the frozen predictor")
    if Path(str(masks.get("joint_trajectories_root"))).resolve() != args.joint_inference.resolve():
        raise ValueError("Joint-mask cache refers to another q inference")
    require_zero_dynamics(masks)
    if geometry.get("schema_version") != GEOMETRY_GATE_SCHEMA:
        raise ValueError("Unexpected geometry gate schema")
    gate = geometry.get("engineering_gate", {})
    if gate.get("status") != "passed" or gate.get("decision") != (
        "eligible_for_frozen_gwp_alignment_adapter"
    ):
        raise ValueError("Action-to-joint geometry gate did not pass")
    if inference.get("geometry_gate_report_sha256") != sha256_file(
        args.geometry_gate_report
    ):
        raise ValueError("Joint inference/geometry gate mismatch")

    policy_records = indexed_records(policy)
    inference_records = indexed_records(inference)
    mask_records = indexed_records(masks)
    if set(policy_records) != set(inference_records) or set(policy_records) != set(mask_records):
        raise ValueError("Policy/q/mask episode sets differ")

    args.output_dir.mkdir(parents=True)
    episodes_root = args.output_dir / "episodes"
    episodes_root.mkdir()
    rows: list[dict[str, Any]] = []
    for episode, policy_record in policy_records.items():
        inference_record = inference_records[episode]
        mask_record = mask_records[episode]
        for key in ("task_index", "task_language"):
            if policy_record.get(key) != inference_record.get(key) or policy_record.get(
                key
            ) != mask_record.get(key):
                raise ValueError(f"Episode {episode} provenance differs for {key}")
        prediction_path = Path(str(policy_record["prediction_path"]))
        if sha256_file(prediction_path) != policy_record["prediction_sha256"]:
            raise ValueError(f"Policy prediction changed: {prediction_path}")
        policy_cache_path = args.policy_cache / str(policy_record["cache_file"])
        mask_cache_path = args.joint_mask_cache / str(mask_record["cache_file"])
        if sha256_file(policy_cache_path) != policy_record["cache_sha256"]:
            raise ValueError(f"Policy action cache changed: {policy_cache_path}")
        if sha256_file(mask_cache_path) != mask_record["cache_sha256"]:
            raise ValueError(f"Direct-q mask cache changed: {mask_cache_path}")
        with np.load(policy_cache_path, allow_pickle=False) as arrays:
            action = np.asarray(arrays["policy_actions_lerobot"], dtype=np.float32)
        source_prediction = np.asarray(np.load(prediction_path, allow_pickle=False))
        if source_prediction.ndim == 3 and source_prediction.shape[0] == 1:
            source_prediction = source_prediction[0]
        if action.shape != (48, 7) or not np.array_equal(action, source_prediction[:48, :7]):
            raise ValueError(f"Policy action mismatch episode {episode}")
        with np.load(mask_cache_path, allow_pickle=False) as arrays:
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
        if selected["horizons"].tolist() != EXPECTED_HORIZONS:
            raise ValueError(f"Wrong horizons episode {episode}")
        if selected["robot_mask"].shape != (5, 256, 256) or selected[
            "gripper_mask"
        ].shape != (5, 256, 256):
            raise ValueError(f"Wrong mask shape episode {episode}")
        if np.logical_and(selected["gripper_mask"] > 0, selected["robot_mask"] == 0).any():
            raise ValueError(f"Gripper mask is not a robot subset episode {episode}")

        output_path = episodes_root / f"episode_{episode:06d}.npz"
        np.savez_compressed(output_path, policy_actions_lerobot=action, **selected)
        row = {
            key: policy_record[key]
            for key in (
                "episode_index",
                "task_index",
                "libero_task_index",
                "task_language",
                "raw_demo_name",
                "raw_hdf5_path",
                "raw_hdf5_sha256",
                "raw_initial_state_sha256",
                "prediction_path",
                "prediction_sha256",
            )
            if key in policy_record
        }
        row.update(
            {
                "source_frame_index": 0,
                "action_chunk_index": 0,
                "joint_inference_arm_sha256": inference_record["arm_sha256"],
                "joint_inference_gripper_sha256": inference_record["gripper_sha256"],
                "direct_q_mask_cache_sha256": mask_record["cache_sha256"],
                "cache_file": str(output_path.relative_to(args.output_dir)),
                "cache_sha256": sha256_file(output_path),
                "robot_pixels": [
                    int(value) for value in selected["robot_mask"].sum(axis=(-2, -1))
                ],
                "gripper_pixels": [
                    int(value) for value in selected["gripper_mask"].sum(axis=(-2, -1))
                ],
            }
        )
        rows.append(row)

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "dataset_root": masks["dataset_root"],
        "dataset_info_sha256": masks["dataset_info_sha256"],
        "suite": masks["suite"],
        "policy_id": policy["policy_id"],
        "policy_action_array": "policy_actions_lerobot",
        "action_chunk_index": 0,
        "horizons": EXPECTED_HORIZONS,
        "camera_name": masks["camera_name"],
        "image_size": masks["image_size"],
        "image_orientation": masks["image_orientation"],
        "rgb_supervision": None,
        "target_mapping": masks["target_mapping"],
        "q_source": "frozen action-to-joint predictor from policy actions plus logged q0",
        "robot_visibility": masks["robot_visibility"],
        "dynamics": "forbidden",
        "dynamics_attempted_calls": masks["dynamics_attempted_calls"],
        "geometry_gate_report": str(args.geometry_gate_report.resolve()),
        "geometry_gate_report_sha256": sha256_file(args.geometry_gate_report),
        "policy_cache": str(args.policy_cache.resolve()),
        "policy_cache_manifest_sha256": sha256_file(policy_manifest_path),
        "policy_cache_arrays_read": ["policy_actions_lerobot"],
        "legacy_rollout_masks_read": False,
        "legacy_rollout_qpos_read": False,
        "simulator_actions_read": False,
        "joint_inference": str(args.joint_inference.resolve()),
        "joint_inference_manifest_sha256": sha256_file(inference_manifest_path),
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

#!/usr/bin/env python3
"""Infer future q for policy-predicted LIBERO action chunks without simulation.

Only ``policy_actions_lerobot`` and the logged current q0 are consumed.  Legacy
rollout masks, simulator actions, future simulator qpos, and RGB targets are
explicitly ignored.  Outputs follow the direct-q mask renderer contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_libero_action_to_joint_predictor import (
    ActionToJointGRU,
    REPORT_SCHEMA,
)


POLICY_CACHE_SCHEMA = "libero-counterfactual-robot-mask-v5"
JOINT_DATA_SCHEMA = "libero-logged-action-joint-trajectory-v1"
GEOMETRY_GATE_SCHEMA = "libero-action-to-joint-geometry-audit-v1"
OUTPUT_SCHEMA = "libero-policy-action-to-joint-inference-v1"
PANDA_ARM_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float32,
)
PANDA_GRIPPER_LIMITS = np.asarray([[0.0, 0.04], [-0.04, 0.0]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-output", type=Path, required=True)
    parser.add_argument("--geometry-gate-report", type=Path, required=True)
    parser.add_argument("--joint-dataset-split", type=Path, required=True)
    parser.add_argument("--policy-cache", type=Path, required=True)
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


def load_q0(split_root: Path, record: dict[str, Any]) -> np.ndarray:
    path = split_root / str(record["cache_file"])
    if sha256_file(path) != record["cache_sha256"]:
        raise ValueError(f"Joint cache changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        arm = np.asarray(arrays["arm_qpos"][0], dtype=np.float32)
        gripper = np.asarray(arrays["gripper_qpos"][0], dtype=np.float32)
    return np.concatenate([arm, gripper], axis=0)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    predictor_report_path = args.predictor_output / "report.json"
    predictor_report = load_json(predictor_report_path)
    if predictor_report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unexpected action-to-joint predictor schema")
    checkpoint_path = Path(str(predictor_report["checkpoint"]))
    if sha256_file(checkpoint_path) != predictor_report["checkpoint_sha256"]:
        raise ValueError("Action-to-joint checkpoint changed")
    geometry_gate = load_json(args.geometry_gate_report)
    if geometry_gate.get("schema_version") != GEOMETRY_GATE_SCHEMA:
        raise ValueError("Unexpected action-to-joint geometry gate schema")
    if geometry_gate.get("fk_api") != "mujoco.mj_kinematics" or any(
        geometry_gate.get(key) is not False
        for key in (
            "benchmark_environment_imported",
            "benchmark_environment_constructed",
            "benchmark_state_read",
            "controller_used",
            "contact_used",
        )
    ):
        raise ValueError("Action-to-joint geometry gate is not standalone FK-only")
    gate = geometry_gate.get("engineering_gate", {})
    if gate.get("status") != "passed" or gate.get("decision") != (
        "eligible_for_frozen_gwp_alignment_adapter"
    ):
        raise ValueError("Action-to-joint geometry gate did not pass")
    if geometry_gate.get("predictor_report_sha256") != sha256_file(
        predictor_report_path
    ):
        raise ValueError("Geometry gate/predictor report mismatch")

    joint_manifest_path = args.joint_dataset_split / "manifest.json"
    joint_manifest = load_json(joint_manifest_path)
    if joint_manifest.get("schema_version") != JOINT_DATA_SCHEMA:
        raise ValueError("Unexpected logged joint dataset schema")
    if joint_manifest.get("simulator_runtime_imported") is not False:
        raise ValueError("Logged joint dataset imported simulator runtime")
    joint_records = {
        int(record["episode_index"]): record
        for record in joint_manifest["episodes"]
    }
    policy_manifest_path = args.policy_cache / "manifest.json"
    policy_manifest = load_json(policy_manifest_path)
    if policy_manifest.get("schema_version") != POLICY_CACHE_SCHEMA:
        raise ValueError("Policy cache must use explicit v5 action semantics")
    if policy_manifest.get("policy_action_array") != "policy_actions_lerobot":
        raise ValueError("Policy cache action array is ambiguous")
    policy_records = list(policy_manifest["episodes"])
    if set(joint_records) != {
        int(record["episode_index"]) for record in policy_records
    }:
        raise ValueError("Policy cache and logged-q split episodes differ")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unexpected predictor checkpoint schema")
    model = ActionToJointGRU(
        int(checkpoint["hidden_dim"]), int(checkpoint["num_layers"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    q_mean = np.asarray(checkpoint["q_mean"], dtype=np.float32)
    q_std = np.asarray(checkpoint["q_std"], dtype=np.float32)

    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    arm_clipped = 0
    gripper_clipped = 0
    total_arm = 0
    total_gripper = 0
    with torch.no_grad():
        for policy_record in policy_records:
            episode = int(policy_record["episode_index"])
            joint_record = joint_records[episode]
            prediction_path = Path(str(policy_record["prediction_path"]))
            if sha256_file(prediction_path) != policy_record["prediction_sha256"]:
                raise ValueError(f"Policy prediction changed: {prediction_path}")
            policy_cache_path = args.policy_cache / str(policy_record["cache_file"])
            if sha256_file(policy_cache_path) != policy_record["cache_sha256"]:
                raise ValueError(f"Policy cache changed: {policy_cache_path}")
            # Deliberately read only the policy action array.  The v5 file's
            # simulator actions, masks, and rollout qpos remain inaccessible to
            # this inference path.
            with np.load(policy_cache_path, allow_pickle=False) as arrays:
                actions = np.asarray(
                    arrays["policy_actions_lerobot"][:48, :7], dtype=np.float32
                )
            source_prediction = np.asarray(
                np.load(prediction_path, allow_pickle=False), dtype=np.float32
            )
            if source_prediction.ndim == 3 and source_prediction.shape[0] == 1:
                source_prediction = source_prediction[0]
            if not np.array_equal(actions, source_prediction[:48, :7]):
                raise ValueError(f"Policy action array mismatch episode {episode}")
            q0 = load_q0(args.joint_dataset_split, joint_record)
            q0_normalized = torch.from_numpy(((q0 - q_mean) / q_std)[None])
            action_tensor = torch.from_numpy(actions[None])
            predicted = model(q0_normalized, action_tensor).numpy()[0]
            predicted = predicted * q_std + q_mean
            dense = np.concatenate([q0[None], predicted], axis=0).astype(np.float32)
            unclipped = dense.copy()
            dense[:, :7] = np.clip(
                dense[:, :7], PANDA_ARM_LIMITS[:, 0], PANDA_ARM_LIMITS[:, 1]
            )
            dense[:, 7:] = np.clip(
                dense[:, 7:],
                PANDA_GRIPPER_LIMITS[:, 0],
                PANDA_GRIPPER_LIMITS[:, 1],
            )
            arm_clipped += int(np.count_nonzero(dense[:, :7] != unclipped[:, :7]))
            gripper_clipped += int(
                np.count_nonzero(dense[:, 7:] != unclipped[:, 7:])
            )
            total_arm += int(dense[:, :7].size)
            total_gripper += int(dense[:, 7:].size)

            episode_root = args.output_dir / f"episode_{episode:06d}"
            episode_root.mkdir()
            arm_path = episode_root / "predicted_joint_qpos.npy"
            gripper_path = episode_root / "predicted_gripper_qpos.npy"
            np.save(arm_path, dense[:, :7])
            np.save(gripper_path, dense[:, 7:])
            rows.append(
                {
                    "episode_index": episode,
                    "task_index": int(joint_record["task_index"]),
                    "task_language": str(joint_record["task_language"]),
                    "source_frame_index": 0,
                    "policy_prediction_path": str(prediction_path.resolve()),
                    "policy_prediction_sha256": sha256_file(prediction_path),
                    "logged_q0_sha256": hashlib.sha256(q0.tobytes()).hexdigest(),
                    "arm_path": str(arm_path.relative_to(args.output_dir)),
                    "arm_sha256": sha256_file(arm_path),
                    "gripper_path": str(gripper_path.relative_to(args.output_dir)),
                    "gripper_sha256": sha256_file(gripper_path),
                }
            )
    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "policy_actions_mapped_to_future_q_without_simulation",
        "predictor_output": str(args.predictor_output.resolve()),
        "predictor_report_sha256": sha256_file(predictor_report_path),
        "predictor_checkpoint_sha256": sha256_file(checkpoint_path),
        "geometry_gate_report": str(args.geometry_gate_report.resolve()),
        "geometry_gate_report_sha256": sha256_file(args.geometry_gate_report),
        "joint_dataset_split": str(args.joint_dataset_split.resolve()),
        "joint_dataset_manifest_sha256": sha256_file(joint_manifest_path),
        "policy_cache": str(args.policy_cache.resolve()),
        "policy_cache_manifest_sha256": sha256_file(policy_manifest_path),
        "policy_cache_arrays_read": ["policy_actions_lerobot"],
        "legacy_rollout_masks_read": False,
        "legacy_rollout_qpos_read": False,
        "simulator_actions_read": False,
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "input": "policy action chunk plus logged current q0",
        "output": "dense q0 plus 48 predicted post-action joint states",
        "arm_joint_limits": PANDA_ARM_LIMITS.tolist(),
        "gripper_joint_limits": PANDA_GRIPPER_LIMITS.tolist(),
        "arm_clipped_fraction": float(arm_clipped / total_arm),
        "gripper_clipped_fraction": float(gripper_clipped / total_gripper),
        "num_episodes": len(rows),
        "episodes": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "num_episodes": len(rows),
        "arm_clipped_fraction": manifest["arm_clipped_fraction"],
        "gripper_clipped_fraction": manifest["gripper_clipped_fraction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

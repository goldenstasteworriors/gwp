#!/usr/bin/env python3
"""Infer q for the largest support-approved expert/policy action interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_libero_action_to_joint_predictor import ActionToJointGRU, REPORT_SCHEMA


JOINT_SCHEMA = "libero-logged-action-joint-trajectory-v1"
POLICY_SCHEMA = "libero-policy-action-standalone-mask-v2"
SUPPORT_SCHEMA = "libero-action-to-joint-input-support-audit-v1"
GEOMETRY_SCHEMA = "libero-action-to-joint-geometry-audit-v1"
OUTPUT_SCHEMA = "libero-support-constrained-action-to-joint-inference-v1"
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
    parser.add_argument("--input-support-report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "heldout"), required=True)
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


def load_joint_values(root: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = root / str(record["cache_file"])
    if sha256_file(path) != record["cache_sha256"]:
        raise ValueError(f"Joint cache changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        expert = np.asarray(arrays["policy_actions_lerobot"][:48], dtype=np.float32)
        q0 = np.concatenate(
            [
                np.asarray(arrays["arm_qpos"][0], dtype=np.float32),
                np.asarray(arrays["gripper_qpos"][0], dtype=np.float32),
            ]
        )
    if expert.shape != (48, 7) or q0.shape != (9,):
        raise ValueError(f"Unexpected expert action/q0 shape: {path}")
    return expert, q0


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    predictor_report_path = args.predictor_output / "report.json"
    predictor_report = load_json(predictor_report_path)
    if predictor_report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unexpected predictor schema")
    checkpoint_path = Path(str(predictor_report["checkpoint"]))
    if sha256_file(checkpoint_path) != predictor_report["checkpoint_sha256"]:
        raise ValueError("Predictor checkpoint changed")
    geometry = load_json(args.geometry_gate_report)
    if geometry.get("schema_version") != GEOMETRY_SCHEMA or geometry.get(
        "engineering_gate", {}
    ).get("status") != "passed":
        raise ValueError("Action-to-q geometry gate did not pass")
    if geometry.get("fk_api") != "mujoco.mj_kinematics" or any(
        geometry.get(key) is not False
        for key in (
            "benchmark_environment_imported",
            "benchmark_environment_constructed",
            "benchmark_state_read",
            "controller_used",
            "contact_used",
        )
    ):
        raise ValueError("Action-to-q geometry gate is not standalone FK-only")
    if geometry.get("predictor_report_sha256") != sha256_file(predictor_report_path):
        raise ValueError("Geometry gate/predictor mismatch")

    support = load_json(args.input_support_report)
    if support.get("schema_version") != SUPPORT_SCHEMA:
        raise ValueError("Unexpected input-support schema")
    if support.get("status") != "failed" or support.get("decision") != (
        "filter_or_retrain_action_to_q_before_alignment"
    ):
        raise ValueError("This adapter is only for a failed full-policy support gate")
    alphas = {
        int(row["episode_index"]): row["maximum_supported_alpha"]
        for row in support["maximum_supported_alpha"][args.split]
    }
    if any(value is None for value in alphas.values()):
        raise ValueError("At least one episode has no support-approved alpha")
    allowed_alphas = {
        float(value) for value in support["interpolation_alphas_fixed_before_evaluation"]
    }
    if not all(float(value) in allowed_alphas for value in alphas.values()):
        raise ValueError("Support report selected an undeclared alpha")

    joint_manifest_path = args.joint_dataset_split / "manifest.json"
    joint_manifest = load_json(joint_manifest_path)
    if joint_manifest.get("schema_version") != JOINT_SCHEMA:
        raise ValueError("Unexpected joint split schema")
    if joint_manifest.get("split") != args.split:
        raise ValueError("Joint split/name mismatch")
    joint_records = {
        int(record["episode_index"]): record
        for record in joint_manifest.get("episodes", [])
    }
    policy_manifest_path = args.policy_cache / "manifest.json"
    policy_manifest = load_json(policy_manifest_path)
    if policy_manifest.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("Unexpected policy cache schema")
    expected_policy_hash = support[f"policy_{args.split}_manifest_sha256"]
    if sha256_file(policy_manifest_path) != expected_policy_hash:
        raise ValueError("Policy cache changed after input-support gate")
    policy_records = {
        int(record["episode_index"]): record
        for record in policy_manifest.get("episodes", [])
    }
    if set(joint_records) != set(policy_records) or set(joint_records) != set(alphas):
        raise ValueError("Joint/policy/support episode sets differ")

    # This checkpoint was produced by this repository and contains NumPy q
    # statistics in addition to tensors, so PyTorch >=2.6's tensor-only
    # default is intentionally disabled after the SHA256/provenance checks.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ActionToJointGRU(
        int(checkpoint["hidden_dim"]), int(checkpoint["num_layers"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    q_mean = np.asarray(checkpoint["q_mean"], dtype=np.float32)
    q_std = np.asarray(checkpoint["q_std"], dtype=np.float32)

    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    alpha_counts: dict[str, int] = {}
    arm_clipped = gripper_clipped = total_arm = total_gripper = 0
    with torch.no_grad():
        for episode, joint_record in joint_records.items():
            policy_record = policy_records[episode]
            expert_actions, q0 = load_joint_values(args.joint_dataset_split, joint_record)
            policy_path = args.policy_cache / str(policy_record["cache_file"])
            if sha256_file(policy_path) != policy_record["cache_sha256"]:
                raise ValueError(f"Policy cache changed: {policy_path}")
            with np.load(policy_path, allow_pickle=False) as arrays:
                policy_actions = np.asarray(
                    arrays["policy_actions_lerobot"], dtype=np.float32
                )
                policy_q0 = np.concatenate(
                    [
                        np.asarray(arrays["predicted_joint_qpos"][0], dtype=np.float32),
                        np.asarray(
                            arrays["predicted_gripper_qpos"][0], dtype=np.float32
                        ),
                    ]
                )
            if not np.array_equal(q0, policy_q0):
                raise ValueError(f"Policy/expert q0 differs episode {episode}")
            alpha = float(alphas[episode])
            condition_actions = (
                (1.0 - alpha) * expert_actions + alpha * policy_actions
            ).astype(np.float32)
            q0_normalized = torch.from_numpy(((q0 - q_mean) / q_std)[None])
            predicted = model(
                q0_normalized, torch.from_numpy(condition_actions[None])
            ).numpy()[0]
            predicted = predicted * q_std + q_mean
            dense = np.concatenate([q0[None], predicted], axis=0).astype(np.float32)
            unclipped = dense.copy()
            dense[:, :7] = np.clip(
                dense[:, :7], PANDA_ARM_LIMITS[:, 0], PANDA_ARM_LIMITS[:, 1]
            )
            dense[:, 7:] = np.clip(
                dense[:, 7:], PANDA_GRIPPER_LIMITS[:, 0], PANDA_GRIPPER_LIMITS[:, 1]
            )
            arm_clipped += int(np.count_nonzero(dense[:, :7] != unclipped[:, :7]))
            gripper_clipped += int(np.count_nonzero(dense[:, 7:] != unclipped[:, 7:]))
            total_arm += int(dense[:, :7].size)
            total_gripper += int(dense[:, 7:].size)

            episode_root = args.output_dir / f"episode_{episode:06d}"
            episode_root.mkdir()
            arm_path = episode_root / "predicted_joint_qpos.npy"
            gripper_path = episode_root / "predicted_gripper_qpos.npy"
            action_path = episode_root / "condition_actions_lerobot.npy"
            np.save(arm_path, dense[:, :7])
            np.save(gripper_path, dense[:, 7:])
            np.save(action_path, condition_actions)
            alpha_counts[f"{alpha:.2f}"] = alpha_counts.get(f"{alpha:.2f}", 0) + 1
            rows.append(
                {
                    "episode_index": episode,
                    "task_index": int(joint_record["task_index"]),
                    "task_language": str(joint_record["task_language"]),
                    "source_frame_index": 0,
                    "maximum_supported_alpha": alpha,
                    "arm_path": str(arm_path.relative_to(args.output_dir)),
                    "arm_sha256": sha256_file(arm_path),
                    "gripper_path": str(gripper_path.relative_to(args.output_dir)),
                    "gripper_sha256": sha256_file(gripper_path),
                    "condition_action_path": str(action_path.relative_to(args.output_dir)),
                    "condition_action_sha256": sha256_file(action_path),
                }
            )
    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "split": args.split,
        "status": "support_constrained_actions_mapped_to_q_without_simulation",
        "action_semantics": "(1-alpha)*logged expert + alpha*baseline policy",
        "alpha_selection": "largest predeclared value passing input-support gate",
        "alpha_counts": alpha_counts,
        "predictor_output": str(args.predictor_output.resolve()),
        "predictor_report_sha256": sha256_file(predictor_report_path),
        "predictor_checkpoint_sha256": sha256_file(checkpoint_path),
        "geometry_gate_report": str(args.geometry_gate_report.resolve()),
        "geometry_gate_report_sha256": sha256_file(args.geometry_gate_report),
        "input_support_report": str(args.input_support_report.resolve()),
        "input_support_report_sha256": sha256_file(args.input_support_report),
        "joint_dataset_split": str(args.joint_dataset_split.resolve()),
        "joint_dataset_manifest_sha256": sha256_file(joint_manifest_path),
        "policy_cache": str(args.policy_cache.resolve()),
        "policy_cache_manifest_sha256": sha256_file(policy_manifest_path),
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "arm_clipped_fraction": float(arm_clipped / total_arm),
        "gripper_clipped_fraction": float(gripper_clipped / total_gripper),
        "num_episodes": len(rows),
        "episodes": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "alpha_counts": alpha_counts,
                "arm_clipped_fraction": manifest["arm_clipped_fraction"],
                "gripper_clipped_fraction": manifest["gripper_clipped_fraction"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

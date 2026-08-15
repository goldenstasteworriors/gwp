#!/usr/bin/env python3
"""Audit registration and non-triviality of kinematic alignment targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPERT_SCHEMA = "libero-logged-action-kinematic-mask-v1"
POLICY_SCHEMAS = {
    "libero-policy-action-kinematic-mask-v1",
    "libero-support-constrained-action-kinematic-mask-v1",
}
REPORT_SCHEMA = "libero-kinematic-alignment-target-audit-v1"
HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-cache", type=Path, required=True)
    parser.add_argument("--policy-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def validate_manifest(
    root: Path, schema: str | set[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / "manifest.json"
    manifest = load_json(path)
    accepted = {schema} if isinstance(schema, str) else schema
    if manifest.get("schema_version") not in accepted:
        raise ValueError(f"Unexpected schema: {path}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError("Alignment cache contains RGB supervision")
    calls = manifest.get("dynamics_attempted_calls", {})
    if manifest.get("dynamics") != "forbidden" or set(calls) != {
        "env_step",
        "sim_step",
        "mj_step",
    } or any(int(value) != 0 for value in calls.values()):
        raise ValueError(f"Dynamics call in {path}: {calls}")
    if manifest.get("horizons") != HORIZONS:
        raise ValueError(f"Unexpected horizons: {path}")
    records = list(manifest.get("episodes", []))
    if not records or len(records) != int(manifest.get("num_episodes", -1)):
        raise ValueError(f"Invalid records: {path}")
    for record in records:
        cache_path = root / str(record["cache_file"])
        if sha256_file(cache_path) != record["cache_sha256"]:
            raise ValueError(f"Cache changed: {cache_path}")
    return manifest, records


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Metrics must be finite and non-empty")
    return {
        "mean": float(array.mean()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    expert, expert_records = validate_manifest(args.expert_cache, EXPERT_SCHEMA)
    policy, policy_records = validate_manifest(args.policy_cache, POLICY_SCHEMAS)
    for key in ("dataset_root", "dataset_info_sha256", "suite", "camera_name", "image_size"):
        if expert.get(key) != policy.get(key):
            raise ValueError(f"Expert/policy metadata differs for {key}")
    expert_by_episode = {int(record["episode_index"]): record for record in expert_records}
    policy_by_episode = {int(record["episode_index"]): record for record in policy_records}
    if list(expert_by_episode) != list(policy_by_episode):
        raise ValueError("Expert/policy episode order differs")

    metrics: dict[int, dict[str, list[float]]] = {
        horizon: {
            "expert_policy_robot_mask_iou": [],
            "expert_policy_gripper_mask_iou": [],
            "expert_policy_arm_q_mae_rad": [],
            "expert_policy_gripper_q_mae": [],
            "expert_policy_eef_distance_m": [],
            "policy_initial_robot_mask_iou": [],
            "expert_initial_robot_mask_iou": [],
        }
        for horizon in HORIZONS
    }
    initial_exact = []
    for episode, expert_record in expert_by_episode.items():
        policy_record = policy_by_episode[episode]
        with np.load(
            args.expert_cache / str(expert_record["cache_file"]), allow_pickle=False
        ) as arrays:
            expert_values = {key: np.asarray(arrays[key]) for key in (
                "robot_mask", "gripper_mask", "predicted_joint_qpos",
                "predicted_gripper_qpos", "eef_position"
            )}
        with np.load(
            args.policy_cache / str(policy_record["cache_file"]), allow_pickle=False
        ) as arrays:
            policy_values = {key: np.asarray(arrays[key]) for key in (
                "robot_mask", "gripper_mask", "predicted_joint_qpos",
                "predicted_gripper_qpos", "eef_position"
            )}
        exact = all(
            np.array_equal(expert_values[key][0], policy_values[key][0])
            for key in ("robot_mask", "gripper_mask", "predicted_joint_qpos", "predicted_gripper_qpos")
        )
        initial_exact.append(exact)
        for index, horizon in enumerate(HORIZONS):
            row = metrics[horizon]
            row["expert_policy_robot_mask_iou"].append(mask_iou(
                expert_values["robot_mask"][index], policy_values["robot_mask"][index]
            ))
            row["expert_policy_gripper_mask_iou"].append(mask_iou(
                expert_values["gripper_mask"][index], policy_values["gripper_mask"][index]
            ))
            row["expert_policy_arm_q_mae_rad"].append(float(np.abs(
                expert_values["predicted_joint_qpos"][index]
                - policy_values["predicted_joint_qpos"][index]
            ).mean()))
            row["expert_policy_gripper_q_mae"].append(float(np.abs(
                expert_values["predicted_gripper_qpos"][index]
                - policy_values["predicted_gripper_qpos"][index]
            ).mean()))
            row["expert_policy_eef_distance_m"].append(float(np.linalg.norm(
                expert_values["eef_position"][index] - policy_values["eef_position"][index]
            )))
            row["policy_initial_robot_mask_iou"].append(mask_iou(
                policy_values["robot_mask"][0], policy_values["robot_mask"][index]
            ))
            row["expert_initial_robot_mask_iou"].append(mask_iou(
                expert_values["robot_mask"][0], expert_values["robot_mask"][index]
            ))

    summary = {
        str(horizon): {name: summarize(values) for name, values in row.items()}
        for horizon, row in metrics.items()
    }
    h48 = summary["48"]
    checks = {
        "all_initial_q_and_masks_exact": all(initial_exact),
        "future_policy_target_moves_from_initial": (
            h48["policy_initial_robot_mask_iou"]["mean"] < 0.90
        ),
        "expert_and_policy_future_are_nonidentical": (
            h48["expert_policy_robot_mask_iou"]["mean"] < 0.95
            and h48["expert_policy_eef_distance_m"]["mean"] > 0.01
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "decision": (
            "ready_for_frozen_latent_probe_and_action_conditioning_gap_gate"
            if all(checks.values())
            else "do_not_train_alignment"
        ),
        "expert_cache": str(args.expert_cache.resolve()),
        "expert_manifest_sha256": sha256_file(args.expert_cache / "manifest.json"),
        "policy_cache": str(args.policy_cache.resolve()),
        "policy_manifest_sha256": sha256_file(args.policy_cache / "manifest.json"),
        "num_episodes": len(expert_records),
        "target_semantics": (
            "same logged q0; future policy action -> frozen q predictor -> direct FK "
            "robot mesh mask, compared only for registration and non-triviality"
        ),
        "dynamics_attempted_calls": {"env_step": 0, "sim_step": 0, "mj_step": 0},
        "checks": checks,
        "by_horizon": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()

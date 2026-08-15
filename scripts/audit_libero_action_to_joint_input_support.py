#!/usr/bin/env python3
"""Gate policy action chunks against logged action-to-q training support.

This is an input-space support diagnostic, not a claim of counterfactual q
ground truth.  It embeds q0 plus temporal action statistics using only logged
data, calibrates k-nearest-neighbour distance on held-out expert chunks, and
marks policy chunks that are too far outside that support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


JOINT_SCHEMA = "libero-logged-action-joint-trajectory-v1"
POLICY_SCHEMA = "libero-policy-action-kinematic-mask-v1"
REPORT_SCHEMA = "libero-action-to-joint-input-support-audit-v1"
ACTION_HORIZON = 48
PCA_DIM = 32
K_NEIGHBORS = 5
CALIBRATION_QUANTILE = 95.0
THRESHOLD_MULTIPLIER = 1.5
MIN_POLICY_ACCEPTED_FRACTION = 0.80
SAMPLE_INDICES = np.asarray([0, 5, 11, 23, 35, 47], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-dataset-root", type=Path, required=True)
    parser.add_argument("--policy-train-cache", type=Path, required=True)
    parser.add_argument("--policy-heldout-cache", type=Path, required=True)
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


def action_features(q0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if q0.shape[-1] != 9 or actions.shape[-2:] != (ACTION_HORIZON, 7):
        raise ValueError(f"Unexpected q0/action shape: {q0.shape}, {actions.shape}")
    sampled = actions[..., SAMPLE_INDICES, :].reshape(*actions.shape[:-2], -1)
    statistics = np.concatenate(
        [
            actions.mean(axis=-2),
            actions.std(axis=-2),
            actions.min(axis=-2),
            actions.max(axis=-2),
            np.abs(np.diff(actions, axis=-2)).mean(axis=-2),
        ],
        axis=-1,
    )
    return np.concatenate([q0, sampled, statistics], axis=-1).astype(np.float32)


def load_joint_split(
    root: Path, *, all_windows: bool
) -> tuple[np.ndarray, list[dict[str, int]]]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != JOINT_SCHEMA:
        raise ValueError(f"Unexpected joint schema: {manifest_path}")
    if manifest.get("simulator_runtime_imported") is not False or int(
        manifest.get("simulator_step_calls", -1)
    ) != 0:
        raise ValueError("Joint dataset queried simulator runtime")
    features: list[np.ndarray] = []
    rows: list[dict[str, int]] = []
    for record in manifest.get("episodes", []):
        cache_path = root / str(record["cache_file"])
        if sha256_file(cache_path) != record["cache_sha256"]:
            raise ValueError(f"Joint cache changed: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as arrays:
            actions = np.asarray(arrays["policy_actions_lerobot"], dtype=np.float32)
            arm = np.asarray(arrays["arm_qpos"], dtype=np.float32)
            gripper = np.asarray(arrays["gripper_qpos"], dtype=np.float32)
        q = np.concatenate([arm, gripper], axis=-1)
        starts = range(len(actions) - ACTION_HORIZON) if all_windows else range(1)
        for start in starts:
            features.append(
                action_features(q[start], actions[start : start + ACTION_HORIZON])
            )
            rows.append(
                {
                    "episode_index": int(record["episode_index"]),
                    "task_index": int(record["task_index"]),
                    "source_frame_index": int(start),
                }
            )
    return np.stack(features), rows


def load_policy_features(root: Path) -> tuple[np.ndarray, list[dict[str, int]]]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"Unexpected policy schema: {manifest_path}")
    calls = manifest.get("dynamics_attempted_calls", {})
    if manifest.get("dynamics") != "forbidden" or any(
        int(value) != 0 for value in calls.values()
    ):
        raise ValueError("Policy kinematic cache attempted dynamics")
    features: list[np.ndarray] = []
    rows: list[dict[str, int]] = []
    for record in manifest.get("episodes", []):
        cache_path = root / str(record["cache_file"])
        if sha256_file(cache_path) != record["cache_sha256"]:
            raise ValueError(f"Policy cache changed: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as arrays:
            actions = np.asarray(arrays["policy_actions_lerobot"], dtype=np.float32)
            arm = np.asarray(arrays["predicted_joint_qpos"][0], dtype=np.float32)
            gripper = np.asarray(
                arrays["predicted_gripper_qpos"][0], dtype=np.float32
            )
        features.append(action_features(np.concatenate([arm, gripper]), actions))
        rows.append(
            {
                "episode_index": int(record["episode_index"]),
                "task_index": int(record["task_index"]),
                "source_frame_index": int(record.get("source_frame_index", 0)),
            }
        )
    return np.stack(features), rows


def transform_features(
    train: np.ndarray, *queries: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    normalized = (train - mean) / std
    _, singular_values, right = np.linalg.svd(normalized, full_matrices=False)
    dimension = min(PCA_DIM, right.shape[0])
    basis = right[:dimension].T
    projected_train = (normalized @ basis).astype(np.float32)
    projected_queries = [
        (((query - mean) / std) @ basis).astype(np.float32) for query in queries
    ]
    explained = singular_values**2
    metadata = {
        "raw_feature_dim": int(train.shape[1]),
        "pca_dim": int(dimension),
        "pca_explained_variance_fraction": float(
            explained[:dimension].sum() / explained.sum()
        ),
    }
    return projected_train, projected_queries, metadata


def knn_distance(bank: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    if len(bank) < k:
        raise ValueError("Feature bank is smaller than k")
    bank_norm = np.sum(bank * bank, axis=1)
    results: list[np.ndarray] = []
    for start in range(0, len(query), 128):
        batch = query[start : start + 128]
        distance2 = (
            np.sum(batch * batch, axis=1, keepdims=True)
            + bank_norm[None]
            - 2.0 * batch @ bank.T
        )
        distance2 = np.maximum(distance2, 0.0)
        nearest = np.partition(distance2, kth=k - 1, axis=1)[:, :k]
        results.append(np.sqrt(nearest).mean(axis=1) / np.sqrt(bank.shape[1]))
    return np.concatenate(results).astype(np.float64)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def policy_result(
    distances: np.ndarray, rows: list[dict[str, int]], threshold: float
) -> dict[str, Any]:
    accepted = distances <= threshold
    return {
        "distance": summary(distances),
        "threshold": threshold,
        "accepted_fraction": float(accepted.mean()),
        "num_accepted": int(accepted.sum()),
        "num_total": len(rows),
        "episodes": [
            {
                **row,
                "support_distance": float(distance),
                "accepted": bool(is_accepted),
            }
            for row, distance, is_accepted in zip(rows, distances, accepted)
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    root_manifest_path = args.joint_dataset_root / "manifest.json"
    root_manifest = load_json(root_manifest_path)
    if root_manifest.get("schema_version") != JOINT_SCHEMA:
        raise ValueError("Unexpected joint dataset root schema")
    train_features, train_rows = load_joint_split(
        args.joint_dataset_root / "train", all_windows=True
    )
    heldout_features, heldout_rows = load_joint_split(
        args.joint_dataset_root / "heldout", all_windows=False
    )
    policy_train_features, policy_train_rows = load_policy_features(
        args.policy_train_cache
    )
    policy_heldout_features, policy_heldout_rows = load_policy_features(
        args.policy_heldout_cache
    )
    bank, queries, transform = transform_features(
        train_features,
        heldout_features,
        policy_train_features,
        policy_heldout_features,
    )
    heldout, policy_train, policy_heldout = [
        knn_distance(bank, query, K_NEIGHBORS) for query in queries
    ]
    calibration_p95 = float(np.percentile(heldout, CALIBRATION_QUANTILE))
    threshold = calibration_p95 * THRESHOLD_MULTIPLIER
    train_result = policy_result(policy_train, policy_train_rows, threshold)
    heldout_result = policy_result(policy_heldout, policy_heldout_rows, threshold)
    checks = {
        "policy_train_accepted_fraction": (
            train_result["accepted_fraction"] >= MIN_POLICY_ACCEPTED_FRACTION
        ),
        "policy_heldout_accepted_fraction": (
            heldout_result["accepted_fraction"] >= MIN_POLICY_ACCEPTED_FRACTION
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "decision": (
            "policy_actions_supported_for_frozen_action_to_q_alignment"
            if all(checks.values())
            else "filter_or_retrain_action_to_q_before_alignment"
        ),
        "metric_semantics": (
            "standardized q0 plus temporal action feature PCA kNN distance to "
            "logged expert training windows"
        ),
        "joint_dataset_root": str(args.joint_dataset_root.resolve()),
        "joint_dataset_manifest_sha256": sha256_file(root_manifest_path),
        "policy_train_cache": str(args.policy_train_cache.resolve()),
        "policy_train_manifest_sha256": sha256_file(
            args.policy_train_cache / "manifest.json"
        ),
        "policy_heldout_cache": str(args.policy_heldout_cache.resolve()),
        "policy_heldout_manifest_sha256": sha256_file(
            args.policy_heldout_cache / "manifest.json"
        ),
        "num_training_windows": len(train_rows),
        "num_calibration_episodes": len(heldout_rows),
        "feature_transform": transform,
        "k_neighbors": K_NEIGHBORS,
        "calibration_quantile": CALIBRATION_QUANTILE,
        "threshold_multiplier_fixed_before_policy_evaluation": THRESHOLD_MULTIPLIER,
        "minimum_policy_accepted_fraction": MIN_POLICY_ACCEPTED_FRACTION,
        "heldout_expert_calibration_distance": summary(heldout),
        "support_threshold": threshold,
        "policy_train": train_result,
        "policy_heldout": heldout_result,
        "checks": checks,
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "support_threshold": threshold,
                "policy_train_accepted_fraction": train_result["accepted_fraction"],
                "policy_heldout_accepted_fraction": heldout_result[
                    "accepted_fraction"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

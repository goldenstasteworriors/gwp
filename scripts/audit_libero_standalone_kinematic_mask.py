#!/usr/bin/env python3
"""Audit standalone q-to-mask outputs without constructing a benchmark env.

An optional legacy cache is accepted only as a one-time registration reference;
its masks are never copied into the standalone cache or exposed as a training
target.  Both sides must contain exactly the same specified q trajectories.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "libero-standalone-kinematic-mask-audit-v1"
STANDALONE_SCHEMA = "libero-joint-standalone-kinematic-robot-mask-v2"
REFERENCE_SCHEMA = "libero-joint-kinematic-robot-mask-v1"
DYNAMICS_APIS = {"mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_forwardSkip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone-cache", type=Path, required=True)
    parser.add_argument("--legacy-registration-reference", type=Path, required=True)
    parser.add_argument("--renderer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def index_records(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {
        int(record["episode_index"]): record for record in manifest.get("episodes", [])
    }
    if not records or len(records) != int(manifest.get("num_episodes", -1)):
        raise ValueError("Invalid or duplicate episode records")
    return records


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite values")
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


def audit_renderer_imports(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    prohibited = sorted(imported_roots.intersection({"libero", "robosuite", "gym"}))
    if prohibited:
        raise ValueError(f"Standalone renderer imports benchmark runtime: {prohibited}")
    if "mj_kinematics" not in source:
        raise ValueError("Standalone renderer does not call an explicit FK API")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "imported_roots": sorted(imported_roots),
        "prohibited_imports": prohibited,
        "fk_api_present": True,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    standalone_path = args.standalone_cache / "manifest.json"
    reference_path = args.legacy_registration_reference / "manifest.json"
    standalone = load_json(standalone_path)
    reference = load_json(reference_path)
    if standalone.get("schema_version") != STANDALONE_SCHEMA:
        raise ValueError("Unexpected standalone cache schema")
    if reference.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError("Unexpected legacy registration-reference schema")
    for key in (
        "benchmark_environment_imported",
        "benchmark_environment_constructed",
        "benchmark_state_read",
        "controller_used",
        "contact_used",
    ):
        if standalone.get(key) is not False:
            raise ValueError(f"Standalone manifest violates {key}")
    calls = standalone.get("dynamics_attempted_calls", {})
    if set(calls) != DYNAMICS_APIS or any(int(value) != 0 for value in calls.values()):
        raise ValueError(f"Standalone renderer attempted forbidden API: {calls}")
    if standalone.get("fk_api") != "mujoco.mj_kinematics":
        raise ValueError("Standalone cache did not use mj_kinematics-only FK")

    standalone_records = index_records(standalone)
    reference_records = index_records(reference)
    if list(standalone_records) != list(reference_records):
        raise ValueError("Standalone/reference episode order differs")
    robot_ious: list[float] = []
    gripper_ious: list[float] = []
    q_exact: list[bool] = []
    for episode, record in standalone_records.items():
        reference_record = reference_records[episode]
        standalone_cache = args.standalone_cache / str(record["cache_file"])
        reference_cache = args.legacy_registration_reference / str(
            reference_record["cache_file"]
        )
        if sha256_file(standalone_cache) != record["cache_sha256"]:
            raise ValueError(f"Standalone cache changed: {standalone_cache}")
        if sha256_file(reference_cache) != reference_record["cache_sha256"]:
            raise ValueError(f"Registration reference changed: {reference_cache}")
        with np.load(standalone_cache, allow_pickle=False) as arrays:
            standalone_values = {
                key: np.asarray(arrays[key])
                for key in (
                    "predicted_joint_qpos",
                    "predicted_gripper_qpos",
                    "robot_mask",
                    "gripper_mask",
                )
            }
        with np.load(reference_cache, allow_pickle=False) as arrays:
            reference_values = {
                key: np.asarray(arrays[key])
                for key in (
                    "predicted_joint_qpos",
                    "predicted_gripper_qpos",
                    "robot_mask",
                    "gripper_mask",
                )
            }
        q_exact.append(
            np.array_equal(
                standalone_values["predicted_joint_qpos"],
                reference_values["predicted_joint_qpos"],
            )
            and np.array_equal(
                standalone_values["predicted_gripper_qpos"],
                reference_values["predicted_gripper_qpos"],
            )
        )
        for standalone_mask, reference_mask in zip(
            standalone_values["robot_mask"], reference_values["robot_mask"]
        ):
            robot_ious.append(mask_iou(standalone_mask, reference_mask))
        for standalone_mask, reference_mask in zip(
            standalone_values["gripper_mask"], reference_values["gripper_mask"]
        ):
            gripper_ious.append(mask_iou(standalone_mask, reference_mask))

    robot_summary = summarize(robot_ious)
    gripper_summary = summarize(gripper_ious)
    checks = {
        "renderer_has_no_benchmark_runtime_import": True,
        "benchmark_environment_not_constructed": True,
        "all_dynamics_and_full_forward_calls_zero": True,
        "all_specified_q_exactly_equal_to_registration_reference": all(q_exact),
        "robot_registration_mean_at_least_0_85": robot_summary["mean"] >= 0.85,
        "robot_registration_p05_at_least_0_80": robot_summary["p05"] >= 0.80,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "benchmark_environment_constructed": False,
        "reference_role": "one_time_registration_only_never_training_target",
        "standalone_cache": str(args.standalone_cache.resolve()),
        "standalone_manifest_sha256": sha256_file(standalone_path),
        "legacy_registration_reference": str(
            args.legacy_registration_reference.resolve()
        ),
        "legacy_registration_reference_manifest_sha256": sha256_file(reference_path),
        "renderer_source_audit": audit_renderer_imports(args.renderer_source),
        "dynamics_prohibition": {"attempted_calls": calls},
        "renderer_registration": {
            "robot_mask_iou": robot_summary,
            "gripper_mask_iou": gripper_summary,
        },
        "num_episodes": len(standalone_records),
        "num_frames": len(robot_ious),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit learned action-to-joint predictions in EEF and robot-mask space.

All masks are rendered from predicted or logged q with a standalone robot-only
FK asset and explicit camera calibration.  No LIBERO task environment is
constructed.  Logged future q is an audit oracle only and is never copied into
the predictor output or a counterfactual training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from libero_standalone_robot_mask import (
    RASTERIZER_SEMANTICS,
    StandaloneRobotMaskRenderer,
    forbid_mujoco_dynamics,
)


PREDICTOR_SCHEMA = "libero-action-to-joint-predictor-v1"
DATA_SCHEMA = "libero-logged-action-joint-trajectory-v1"
REPORT_SCHEMA = "libero-action-to-joint-geometry-audit-v1"
HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-output", type=Path, required=True)
    parser.add_argument("--joint-dataset", type=Path, required=True)
    parser.add_argument("--fixed-ik-report", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--robot-asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
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


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left > 0, right > 0).sum())
    union = int(np.logical_or(left > 0, right > 0).sum())
    return 1.0 if union == 0 else float(intersection / union)


def rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left) @ np.asarray(right).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def load_logged_q(
    heldout_root: Path, record: dict[str, Any]
) -> np.ndarray:
    path = heldout_root / str(record["cache_file"])
    if sha256_file(path) != record["cache_sha256"]:
        raise ValueError(f"Logged q cache changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        arm = np.asarray(arrays["arm_qpos"], dtype=np.float32)
        gripper = np.asarray(arrays["gripper_qpos"], dtype=np.float32)
    return np.concatenate([arm, gripper], axis=-1)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictor_report_path = args.predictor_output / "report.json"
    predictor = load_json(predictor_report_path)
    if predictor.get("schema_version") != PREDICTOR_SCHEMA:
        raise ValueError("Unexpected predictor report schema")
    if predictor.get("simulator_runtime_imported") is not False:
        raise ValueError("Predictor imported a simulator runtime")
    if int(predictor.get("simulator_step_calls", -1)) != 0:
        raise ValueError("Predictor used simulator steps")
    dataset_manifest_path = args.joint_dataset / "manifest.json"
    dataset = load_json(dataset_manifest_path)
    if dataset.get("schema_version") != DATA_SCHEMA:
        raise ValueError("Unexpected joint dataset schema")
    if predictor.get("dataset_manifest_sha256") != sha256_file(dataset_manifest_path):
        raise ValueError("Predictor/joint-dataset manifest mismatch")
    heldout_root = args.joint_dataset / "heldout"
    heldout_manifest = load_json(heldout_root / "manifest.json")
    heldout_records = list(heldout_manifest["episodes"])
    records_by_episode = {
        int(record["episode_index"]): record for record in heldout_records
    }
    prediction_rows = list(predictor["predictions"])
    if set(records_by_episode) != {
        int(row["episode_index"]) for row in prediction_rows
    }:
        raise ValueError("Predictor output and held-out episodes differ")

    fixed_ik = load_json(args.fixed_ik_report)
    if fixed_ik.get("schema_version") != "libero-kinematic-mask-pipeline-audit-v1":
        raise ValueError("Unexpected fixed-IK audit schema")
    attempted = fixed_ik.get("dynamics_prohibition", {}).get("attempted_calls")
    if attempted != {"env_step": 0, "mj_step": 0, "sim_step": 0}:
        raise ValueError("Fixed-IK comparator did not prove zero dynamics calls")
    fixed_h48 = fixed_ik["ik_action_to_q"]["by_horizon"]["48"]

    renderer = StandaloneRobotMaskRenderer(
        args.robot_asset_dir, args.width, args.height
    )
    metric_names = (
        "arm_q_mae_rad",
        "gripper_q_mae",
        "eef_position_error_m",
        "eef_rotation_error_deg",
        "robot_mask_iou",
        "gripper_mask_iou",
    )
    learned_metrics = {
        horizon: {name: [] for name in metric_names} for horizon in HORIZONS
    }
    copy_metrics = {
        horizon: {name: [] for name in metric_names} for horizon in HORIZONS
    }
    episode_rows: list[dict[str, Any]] = []

    with forbid_mujoco_dynamics() as dynamics_counts:
        for prediction_row in prediction_rows:
            episode = int(prediction_row["episode_index"])
            record = records_by_episode[episode]
            arm_path = args.predictor_output / str(prediction_row["arm_path"])
            gripper_path = args.predictor_output / str(prediction_row["gripper_path"])
            if sha256_file(arm_path) != prediction_row["arm_sha256"]:
                raise ValueError(f"Predicted arm q changed: {arm_path}")
            if sha256_file(gripper_path) != prediction_row["gripper_sha256"]:
                raise ValueError(f"Predicted gripper q changed: {gripper_path}")
            predicted_q = np.concatenate(
                [
                    np.load(arm_path, allow_pickle=False),
                    np.load(gripper_path, allow_pickle=False),
                ],
                axis=-1,
            ).astype(np.float32)
            logged_q = load_logged_q(heldout_root, record)
            if predicted_q.shape != (49, 9) or len(logged_q) <= 48:
                raise ValueError(f"Unexpected predicted/logged q episode {episode}")
            q0_copy = np.repeat(logged_q[0:1], 49, axis=0)
            episode_metric: dict[str, Any] = {}
            for horizon in HORIZONS:
                renderer.set_configuration(logged_q[horizon, :7], logged_q[horizon, 7:])
                oracle = renderer.render_masks()
                horizon_result: dict[str, Any] = {}
                for label, q_value, target_metrics in (
                    ("learned", predicted_q[horizon], learned_metrics),
                    ("copy_q0", q0_copy[horizon], copy_metrics),
                ):
                    renderer.set_configuration(q_value[:7], q_value[7:])
                    rendered = renderer.render_masks()
                    values = {
                        "arm_q_mae_rad": float(
                            np.mean(np.abs(q_value[:7] - logged_q[horizon, :7]))
                        ),
                        "gripper_q_mae": float(
                            np.mean(np.abs(q_value[7:] - logged_q[horizon, 7:]))
                        ),
                        "eef_position_error_m": float(
                            np.linalg.norm(rendered["eef_position"] - oracle["eef_position"])
                        ),
                        "eef_rotation_error_deg": rotation_error_degrees(
                            rendered["eef_rotation"], oracle["eef_rotation"]
                        ),
                        "robot_mask_iou": mask_iou(
                            rendered["robot_mask"], oracle["robot_mask"]
                        ),
                        "gripper_mask_iou": mask_iou(
                            rendered["gripper_mask"], oracle["gripper_mask"]
                        ),
                    }
                    for name, value in values.items():
                        target_metrics[horizon][name].append(value)
                    horizon_result[label] = values
                episode_metric[str(horizon)] = horizon_result
            episode_rows.append(
                {
                    "episode_index": episode,
                    "task_index": int(record["task_index"]),
                    "libero_task_index": int(record["task_index"]),
                    "metrics": episode_metric,
                }
            )

    if dynamics_counts != {
        "mj_step": 0,
        "mj_step1": 0,
        "mj_step2": 0,
        "mj_forward": 0,
        "mj_forwardSkip": 0,
    }:
        raise RuntimeError(f"Forbidden dynamics call attempted: {dynamics_counts}")
    learned_summary = {
        str(horizon): {
            name: summarize(values)
            for name, values in learned_metrics[horizon].items()
        }
        for horizon in HORIZONS
    }
    copy_summary = {
        str(horizon): {
            name: summarize(values)
            for name, values in copy_metrics[horizon].items()
        }
        for horizon in HORIZONS
    }
    thresholds = {
        "horizon_12_robot_iou_mean_min": 0.85,
        "horizon_48_robot_iou_mean_min": 0.80,
        "horizon_48_robot_iou_p05_min": 0.65,
        "horizon_48_eef_error_mean_max_m": 0.03,
        "horizon_48_eef_error_p95_max_m": 0.05,
        "horizon_48_arm_q_mae_mean_max_rad": 0.08,
    }
    h12 = learned_summary["12"]
    h48 = learned_summary["48"]
    absolute_gate = bool(
        h12["robot_mask_iou"]["mean"]
        >= thresholds["horizon_12_robot_iou_mean_min"]
        and h48["robot_mask_iou"]["mean"]
        >= thresholds["horizon_48_robot_iou_mean_min"]
        and h48["robot_mask_iou"]["p05"]
        >= thresholds["horizon_48_robot_iou_p05_min"]
        and h48["eef_position_error_m"]["mean"]
        <= thresholds["horizon_48_eef_error_mean_max_m"]
        and h48["eef_position_error_m"]["p95"]
        <= thresholds["horizon_48_eef_error_p95_max_m"]
        and h48["arm_q_mae_rad"]["mean"]
        <= thresholds["horizon_48_arm_q_mae_mean_max_rad"]
    )
    # The q metric is named arm_q_mae_rad in this report and joint_mae_rad in
    # the fixed-IK report; compare it explicitly outside the expression above.
    beats_fixed_ik = bool(
        h48["robot_mask_iou"]["mean"]
        > float(fixed_h48["robot_mask_iou"]["mean"])
        and h48["eef_position_error_m"]["mean"]
        < float(fixed_h48["eef_position_error_m"]["mean"])
        and h48["arm_q_mae_rad"]["mean"]
        < float(fixed_h48["joint_mae_rad"]["mean"])
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "engineering_gate_evaluated",
        "predictor_output": str(args.predictor_output.resolve()),
        "predictor_report_sha256": sha256_file(predictor_report_path),
        "joint_dataset": str(args.joint_dataset.resolve()),
        "joint_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "fixed_ik_report": str(args.fixed_ik_report.resolve()),
        "fixed_ik_report_sha256": sha256_file(args.fixed_ik_report),
        "num_episodes": len(episode_rows),
        "horizons": HORIZONS,
        "target_mapping": (
            "predicted q + gripper q + explicit camera calibration -> standalone FK -> robot-only rasterization"
        ),
        "benchmark_environment_imported": False,
        "benchmark_environment_constructed": False,
        "benchmark_state_read": False,
        "controller_used": False,
        "contact_used": False,
        "fk_api": "mujoco.mj_kinematics",
        "rasterizer_semantics": RASTERIZER_SEMANTICS,
        "logged_future_q_role": "heldout_audit_oracle_only",
        "dynamics_prohibition": {
            "attempted_calls": dynamics_counts,
            "all_dynamics_and_full_forward_apis": "runtime_monkeypatch_raises",
        },
        "learned": learned_summary,
        "copy_q0": copy_summary,
        "fixed_ik_horizon_48": fixed_h48,
        "engineering_gate": {
            "thresholds_defined_before_full_geometry_audit": thresholds,
            "absolute_gate_passed": absolute_gate,
            "beats_fixed_ik": beats_fixed_ik,
            "status": "passed" if absolute_gate and beats_fixed_ik else "failed",
            "decision": (
                "eligible_for_frozen_gwp_alignment_adapter"
                if absolute_gate and beats_fixed_ik
                else "do_not_attach_to_gwp"
            ),
        },
        "episodes": episode_rows,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["learned"], sort_keys=True))
    print(json.dumps(report["engineering_gate"], sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit a dynamics-free LIBERO action -> q -> FK -> mask pipeline.

The audit has two independent gates:

1. Renderer registration: render the exact q stored by an older simulator
   rollout and compare the CPU robot-only silhouette with MuJoCo segmentation.
2. Kinematic action conversion: feed demonstration actions through differential
   IK and compare the resulting q / EEF / silhouette with recorded q.  Recorded
   future q is used only as an audit oracle and is never emitted as supervision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from generate_libero_counterfactual_mask_cache import normalize_raw_demo_actions
from libero_kinematic_robot_mask import (
    KINEMATIC_TRANSITION,
    RASTERIZER_SEMANTICS,
    KinematicRobotMaskRenderer,
    create_kinematic_libero_environment,
    forbid_mujoco_dynamics,
)


SCHEMA_VERSION = "libero-kinematic-mask-pipeline-audit-v1"
LEGACY_SCHEMA = "libero-counterfactual-robot-mask-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--translation-execution-gain", type=float, default=0.2)
    parser.add_argument("--rotation-execution-gain", type=float, default=0.2)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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
    if not len(array):
        raise ValueError("Cannot summarize an empty metric")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.legacy_cache / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != LEGACY_SCHEMA:
        raise ValueError(f"Expected legacy v5 cache: {manifest_path}")
    records = list(manifest.get("episodes", []))
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        records = records[: args.max_episodes]
    if not records:
        raise ValueError("No episodes selected")
    expected_horizons = [int(value) for value in manifest["horizons"]]
    if expected_horizons != [0, 12, 24, 36, 48]:
        raise ValueError(f"Unexpected legacy horizons: {expected_horizons}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[str(manifest["suite"])]()
    renderers: dict[int, tuple[Any, KinematicRobotMaskRenderer]] = {}
    registration_robot_iou: list[float] = []
    registration_gripper_iou: list[float] = []
    ik_joint_mae: dict[int, list[float]] = {value: [] for value in expected_horizons}
    ik_gripper_mae: dict[int, list[float]] = {value: [] for value in expected_horizons}
    ik_eef_position: dict[int, list[float]] = {value: [] for value in expected_horizons}
    ik_eef_rotation: dict[int, list[float]] = {value: [] for value in expected_horizons}
    ik_robot_iou: dict[int, list[float]] = {value: [] for value in expected_horizons}
    ik_gripper_iou: dict[int, list[float]] = {value: [] for value in expected_horizons}
    convergence: list[bool] = []
    episode_summaries: list[dict[str, Any]] = []

    with forbid_mujoco_dynamics() as dynamics_counts:
        try:
            for record in records:
                task_index = int(record["libero_task_index"])
                if task_index not in renderers:
                    environment = create_kinematic_libero_environment(suite, task_index)
                    renderers[task_index] = (
                        environment,
                        KinematicRobotMaskRenderer(
                            environment,
                            args.camera_name,
                            args.width,
                            args.height,
                            damping=args.damping,
                            max_iterations=args.max_iterations,
                            translation_execution_gain=args.translation_execution_gain,
                            rotation_execution_gain=args.rotation_execution_gain,
                        ),
                    )
                _, renderer = renderers[task_index]
                cache_path = args.legacy_cache / str(record["cache_file"])
                with np.load(cache_path, allow_pickle=False) as cache:
                    legacy_qpos = cache["qpos"].copy()
                    legacy_robot_masks = cache["robot_mask"].copy()
                    legacy_gripper_masks = cache["gripper_mask"].copy()

                exact_q_ious: list[float] = []
                for horizon_index in range(len(expected_horizons)):
                    full_qpos = legacy_qpos[horizon_index]
                    renderer.set_configuration(
                        full_qpos[renderer.arm_qpos_indexes],
                        full_qpos[renderer.gripper_qpos_indexes],
                    )
                    rendered = renderer.render_masks()
                    robot_iou = mask_iou(
                        rendered["robot_mask"], legacy_robot_masks[horizon_index]
                    )
                    gripper_iou = mask_iou(
                        rendered["gripper_mask"], legacy_gripper_masks[horizon_index]
                    )
                    registration_robot_iou.append(robot_iou)
                    registration_gripper_iou.append(gripper_iou)
                    exact_q_ious.append(robot_iou)

                raw_path = Path(str(record["raw_hdf5_path"]))
                with h5py.File(raw_path, "r") as raw_file:
                    demo = raw_file[f"data/{record['raw_demo_name']}"]
                    raw_actions = np.asarray(demo["actions"], dtype=np.float32)
                    joint_states = np.asarray(
                        demo["obs/joint_states"], dtype=np.float64
                    )
                    gripper_states = np.asarray(
                        demo["obs/gripper_states"], dtype=np.float64
                    )
                kept = kept_action_indices(raw_actions)
                policy_actions = normalize_raw_demo_actions(raw_actions)
                if len(kept) != len(policy_actions):
                    raise RuntimeError("No-op index and action filters disagree")
                if len(kept) <= max(expected_horizons):
                    raise ValueError(
                        f"Demo is too short for audit horizon: {raw_path}:{record['raw_demo_name']}"
                    )

                renderer.set_configuration(
                    joint_states[kept[0]], gripper_states[kept[0]]
                )
                action_diagnostics: list[dict[str, Any]] = []
                horizon_metrics: dict[str, Any] = {}
                for step in range(max(expected_horizons) + 1):
                    if step in expected_horizons:
                        predicted = renderer.render_masks()
                        predicted_arm_q = predicted["arm_qpos"].copy()
                        predicted_gripper_q = predicted["gripper_qpos"].copy()
                        renderer.set_configuration(
                            joint_states[kept[step]], gripper_states[kept[step]]
                        )
                        oracle = renderer.render_masks()
                        joint_mae = float(
                            np.mean(np.abs(predicted_arm_q - oracle["arm_qpos"]))
                        )
                        gripper_mae = float(
                            np.mean(
                                np.abs(predicted_gripper_q - oracle["gripper_qpos"])
                            )
                        )
                        eef_position_error = float(
                            np.linalg.norm(
                                predicted["eef_position"] - oracle["eef_position"]
                            )
                        )
                        eef_rotation_error = rotation_error_degrees(
                            predicted["eef_rotation"], oracle["eef_rotation"]
                        )
                        robot_iou = mask_iou(
                            predicted["robot_mask"], oracle["robot_mask"]
                        )
                        gripper_iou = mask_iou(
                            predicted["gripper_mask"], oracle["gripper_mask"]
                        )
                        ik_joint_mae[step].append(joint_mae)
                        ik_gripper_mae[step].append(gripper_mae)
                        ik_eef_position[step].append(eef_position_error)
                        ik_eef_rotation[step].append(eef_rotation_error)
                        ik_robot_iou[step].append(robot_iou)
                        ik_gripper_iou[step].append(gripper_iou)
                        horizon_metrics[str(step)] = {
                            "joint_mae_rad": joint_mae,
                            "gripper_mae": gripper_mae,
                            "eef_position_error_m": eef_position_error,
                            "eef_rotation_error_deg": eef_rotation_error,
                            "robot_mask_iou": robot_iou,
                            "gripper_mask_iou": gripper_iou,
                        }
                        renderer.set_configuration(
                            predicted_arm_q, predicted_gripper_q
                        )
                    if step < max(expected_horizons):
                        diagnostic = renderer.integrate_policy_action(
                            policy_actions[step, :7]
                        )
                        convergence.append(bool(diagnostic["converged"]))
                        action_diagnostics.append(diagnostic)

                episode_summaries.append(
                    {
                        "episode_index": int(record["episode_index"]),
                        "libero_task_index": task_index,
                        "raw_demo_name": str(record["raw_demo_name"]),
                        "renderer_registration_robot_iou": exact_q_ious,
                        "ik_horizons": horizon_metrics,
                        "ik_converged_fraction": float(
                            np.mean([value["converged"] for value in action_diagnostics])
                        ),
                    }
                )
        finally:
            for environment, _ in renderers.values():
                environment.close()

    if any(value != 0 for value in dynamics_counts.values()):
        raise RuntimeError(f"A forbidden dynamics call was attempted: {dynamics_counts}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "audit_complete_no_training_gate_decision_encoded",
        "legacy_cache": str(args.legacy_cache.resolve()),
        "legacy_cache_schema": LEGACY_SCHEMA,
        "num_episodes": len(records),
        "horizons": expected_horizons,
        "target_mapping": "q + gripper_q + camera_calibration -> FK -> mesh rasterization -> robot mask",
        "rasterizer_semantics": RASTERIZER_SEMANTICS,
        "kinematic_transition": KINEMATIC_TRANSITION,
        "future_recorded_q_role": "audit_oracle_only_never_emitted_as_training_supervision",
        "dynamics_prohibition": {
            "mj_step": "runtime_monkeypatch_raises",
            "sim_step": "runtime_monkeypatch_raises",
            "env_step": "runtime_monkeypatch_raises",
            "attempted_calls": dynamics_counts,
        },
        "renderer_registration": {
            "robot_mask_iou": summarize(registration_robot_iou),
            "gripper_mask_iou": summarize(registration_gripper_iou),
            "interpretation": (
                "CPU masks are amodal robot-only silhouettes; legacy MuJoCo masks may "
                "be object-occluded, so this is a registration check rather than identity."
            ),
        },
        "ik_action_to_q": {
            "converged_fraction": float(np.mean(convergence)),
            "by_horizon": {
                str(horizon): {
                    "joint_mae_rad": summarize(ik_joint_mae[horizon]),
                    "gripper_mae": summarize(ik_gripper_mae[horizon]),
                    "eef_position_error_m": summarize(ik_eef_position[horizon]),
                    "eef_rotation_error_deg": summarize(ik_eef_rotation[horizon]),
                    "robot_mask_iou": summarize(ik_robot_iou[horizon]),
                    "gripper_mask_iou": summarize(ik_gripper_iou[horizon]),
                }
                for horizon in expected_horizons
            },
        },
        "configuration": {
            "camera_name": args.camera_name,
            "width": args.width,
            "height": args.height,
            "damping": args.damping,
            "max_iterations": args.max_iterations,
            "translation_execution_gain": args.translation_execution_gain,
            "rotation_execution_gain": args.rotation_execution_gain,
        },
        "episodes": episode_summaries,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["renderer_registration"], sort_keys=True))
    print(json.dumps(report["ik_action_to_q"], sort_keys=True))


if __name__ == "__main__":
    main()

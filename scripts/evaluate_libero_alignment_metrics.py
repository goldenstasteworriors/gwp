#!/usr/bin/env python3
"""Evaluate control-grounded action/video alignment in LIBERO.

This tool rolls two action sequences from the same simulator initial state. It
compares their end-effector effects and robot occupancy, while estimating camera
jitter from non-robot background pixels. It can evaluate a learned policy dump
(``--predicted-actions``) or controlled action perturbations used to validate the
metrics before expensive policy training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_action_model.alignment.geometry import (  # noqa: E402
    binary_mask_metrics,
    project_world_points,
    register_and_compare_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12, 24, 36, 48])
    parser.add_argument(
        "--perturbation",
        choices=["identity", "noise", "scale", "delay", "zero"],
        default="noise",
    )
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--action-scale", type=float, default=0.65)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--predicted-actions", type=Path)
    parser.add_argument("--camera-jitter-px", type=float, default=3.0)
    parser.add_argument("--camera-jitter-deg", type=float, default=0.6)
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="No-op settling steps before rollout; 10 matches the official LeRobot LIBERO evaluator.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def json_number(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_number(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_number(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_episode(dataset_root: Path, episode_index: int) -> tuple[pd.DataFrame, int, int]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    columns = ["episode_index", "frame_index", "task_index", "observation.state", "action"]
    frames = pd.concat([pd.read_parquet(path, columns=columns) for path in parquet_paths], ignore_index=True)
    episode_tasks = frames.groupby("episode_index", sort=True)["task_index"].first()
    if episode_index not in episode_tasks.index:
        raise ValueError(f"Episode {episode_index} is absent; dataset has {info['total_episodes']} episodes")
    task_index = int(episode_tasks.loc[episode_index])
    same_task = episode_tasks[episode_tasks == task_index].index.to_list()
    task_episode_rank = same_task.index(episode_index)
    episode = frames[frames["episode_index"] == episode_index].sort_values("frame_index")
    return episode, task_index, task_episode_rank


def make_predicted_actions(actions: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    if args.predicted_actions is not None:
        if args.predicted_actions.suffix == ".npy":
            predicted = np.load(args.predicted_actions)
        else:
            predicted = np.asarray(json.loads(args.predicted_actions.read_text()))
        if predicted.ndim == 3:
            predicted = predicted[0]
        if predicted.shape[-1] < 7:
            raise ValueError(f"Predicted actions need at least 7 dimensions, got {predicted.shape}")
        return np.asarray(predicted[:, :7], dtype=np.float32)
    if args.perturbation == "identity":
        return actions.copy()
    if args.perturbation == "zero":
        predicted = np.zeros_like(actions)
        predicted[:, 6] = actions[:, 6]
        return predicted
    if args.perturbation == "scale":
        predicted = actions.copy()
        predicted[:, :6] *= args.action_scale
        return predicted
    if args.perturbation == "delay":
        predicted = np.zeros_like(actions)
        delay = max(0, int(args.delay))
        if delay < len(actions):
            predicted[delay:] = actions[: len(actions) - delay]
        predicted[:delay, 6] = actions[0, 6]
        return predicted
    predicted = actions.copy()
    predicted[:, :6] += rng.normal(0.0, args.noise_std, size=predicted[:, :6].shape)
    return np.clip(predicted, -1.0, 1.0)


def robot_geom_ids(env: Any) -> tuple[list[int], list[int]]:
    mapping = env._env.env.model.geom_ids_to_instances
    robot_instances = {"MountedPanda0", "PandaGripper0"}
    robot = sorted(int(geom_id) for geom_id, instance in mapping.items() if instance in robot_instances)
    gripper = sorted(int(geom_id) for geom_id, instance in mapping.items() if instance == "PandaGripper0")
    if not robot or not gripper:
        raise RuntimeError("Could not identify Panda robot geometries")
    return robot, gripper


def centroid(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if not x.size:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([x.mean(), y.mean()], dtype=np.float64)


def render_state(env: Any, obs: dict[str, Any], robot_ids: list[int], gripper_ids: list[int]) -> dict[str, Any]:
    sim = env._env.sim
    rgb = np.flipud(np.asarray(sim.render(width=256, height=256, camera_name="agentview"))).copy()
    segmentation = np.flipud(
        np.asarray(sim.render(width=256, height=256, camera_name="agentview", segmentation=True))
    ).copy()
    geom_ids = segmentation[..., 1]
    camera_id = sim.model.camera_name2id("agentview")
    camera_position = sim.data.cam_xpos[camera_id].copy()
    camera_rotation = sim.data.cam_xmat[camera_id].reshape(3, 3).copy()
    eef_position = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)
    eef_pixel, valid = project_world_points(
        eef_position,
        camera_position,
        camera_rotation,
        float(sim.model.cam_fovy[camera_id]),
        rgb.shape[:2],
    )
    robot_mask = np.isin(geom_ids, robot_ids)
    gripper_mask = np.isin(geom_ids, gripper_ids)
    return {
        "rgb": rgb,
        "robot_mask": robot_mask,
        "gripper_mask": gripper_mask,
        "gripper_centroid": centroid(gripper_mask),
        "eef_position": eef_position,
        "eef_pixel": eef_pixel,
        "eef_visible": bool(valid),
    }


def apply_camera_jitter(
    rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    max_translation: float,
    max_rotation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    tx, ty = rng.uniform(-max_translation, max_translation, size=2)
    angle = float(rng.uniform(-max_rotation, max_rotation))
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    affine[:, 2] += [tx, ty]
    homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    jittered_rgb = cv2.warpPerspective(
        rgb, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    jittered_mask = cv2.warpPerspective(
        mask.astype(np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    return jittered_rgb, jittered_mask, homography


def transform_point(point: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float64)
    transformed = homography @ homogeneous
    return transformed[:2] / transformed[2]


def vector_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = float(np.linalg.norm(predicted - target))
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(target))
    cosine = float(np.dot(predicted, target) / denominator) if denominator > 1e-8 else float("nan")
    return {"endpoint_error_px": error, "cosine": cosine}


def overlay_image(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    registered_source_mask: np.ndarray,
    target_eef: np.ndarray,
    source_eef: np.ndarray,
) -> np.ndarray:
    overlay = target_rgb.astype(np.float32).copy()
    overlay[target_mask] = 0.55 * overlay[target_mask] + 0.45 * np.array([0, 255, 0])
    overlay[registered_source_mask] = 0.55 * overlay[registered_source_mask] + 0.45 * np.array([255, 0, 0])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.circle(overlay, tuple(np.rint(target_eef).astype(int)), 4, (0, 255, 0), -1)
    cv2.circle(overlay, tuple(np.rint(source_eef).astype(int)), 4, (255, 0, 0), -1)
    return overlay


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    episode, task_index, task_episode_rank = load_episode(args.dataset_root, args.episode_index)
    actions = np.stack(episode["action"].to_numpy()).astype(np.float32)[:, :7]
    predicted_actions = make_predicted_actions(actions, args, rng)
    max_horizon = min(max(args.horizons), len(actions), len(predicted_actions))
    horizons = sorted({0, *(h for h in args.horizons if 0 < h <= max_horizon)})

    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    common = dict(
        task_suite=suite,
        task_id=task_index,
        task_suite_name=args.suite,
        obs_type="pixels_agent_pos",
        camera_name="agentview_image",
        init_states=True,
        episode_index=task_episode_rank,
        n_envs=1,
        num_steps_wait=args.num_steps_wait,
    )
    gt_env = LiberoEnv(**common)
    pred_env = LiberoEnv(**common)
    gt_obs, _ = gt_env.reset(seed=args.seed)
    pred_obs, _ = pred_env.reset(seed=args.seed)
    gt_robot_ids, gt_gripper_ids = robot_geom_ids(gt_env)
    pred_robot_ids, pred_gripper_ids = robot_geom_ids(pred_env)
    gt_states = {0: render_state(gt_env, gt_obs, gt_robot_ids, gt_gripper_ids)}
    pred_states = {0: render_state(pred_env, pred_obs, pred_robot_ids, pred_gripper_ids)}

    for step in range(1, max_horizon + 1):
        gt_obs, *_ = gt_env.step(np.clip(actions[step - 1], -1.0, 1.0))
        pred_obs, *_ = pred_env.step(np.clip(predicted_actions[step - 1], -1.0, 1.0))
        if step in horizons:
            gt_states[step] = render_state(gt_env, gt_obs, gt_robot_ids, gt_gripper_ids)
            pred_states[step] = render_state(pred_env, pred_obs, pred_robot_ids, pred_gripper_ids)

    results: list[dict[str, Any]] = []
    for horizon in horizons:
        gt = gt_states[horizon]
        pred = pred_states[horizon]
        jittered_rgb, jittered_mask, jitter_h = apply_camera_jitter(
            pred["rgb"],
            pred["robot_mask"],
            rng,
            args.camera_jitter_px,
            args.camera_jitter_deg,
        )
        raw_metrics = binary_mask_metrics(jittered_mask, gt["robot_mask"])
        registered_metrics, registration_h, registration = register_and_compare_masks(
            jittered_rgb, jittered_mask, gt["rgb"], gt["robot_mask"]
        )
        registered_mask = cv2.warpPerspective(
            jittered_mask.astype(np.uint8),
            registration_h,
            (gt["rgb"].shape[1], gt["rgb"].shape[0]),
            flags=cv2.INTER_NEAREST,
        ) > 0
        jittered_eef = transform_point(pred["eef_pixel"], jitter_h)
        registered_eef = transform_point(jittered_eef, registration_h)
        eef_rollout_error = float(np.linalg.norm(registered_eef - gt["eef_pixel"]))

        gt_projected_effect = gt["eef_pixel"] - gt_states[0]["eef_pixel"]
        gt_observed_effect = gt["gripper_centroid"] - gt_states[0]["gripper_centroid"]
        projection_vs_video = vector_metrics(gt_projected_effect, gt_observed_effect)
        rollout_effect = vector_metrics(
            registered_eef - gt_states[0]["eef_pixel"],
            gt_projected_effect,
        )
        row = {
            "horizon": horizon,
            "raw_mask": raw_metrics,
            "registered_mask": registered_metrics,
            "registration": registration,
            "eef_rollout_error_px": eef_rollout_error,
            "gt_eef_projection_vs_gripper_video": projection_vs_video,
            "predicted_vs_gt_eef_effect": rollout_effect,
            "gt_eef_position": gt["eef_position"],
            "predicted_eef_position": pred["eef_position"],
            "gt_eef_pixel": gt["eef_pixel"],
            "registered_predicted_eef_pixel": registered_eef,
        }
        results.append(row)
        overlay = overlay_image(
            gt["rgb"], gt["robot_mask"], registered_mask, gt["eef_pixel"], registered_eef
        )
        cv2.imwrite(str(args.output_dir / f"horizon_{horizon:03d}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    nonzero = [row for row in results if row["horizon"] > 0]
    aggregate = {
        "registered_mask_iou_mean": float(np.mean([row["registered_mask"]["iou"] for row in nonzero])),
        "raw_mask_iou_mean": float(np.mean([row["raw_mask"]["iou"] for row in nonzero])),
        "eef_rollout_error_px_mean": float(np.mean([row["eef_rollout_error_px"] for row in nonzero])),
        "projection_vs_gripper_endpoint_error_px_mean": float(
            np.mean([row["gt_eef_projection_vs_gripper_video"]["endpoint_error_px"] for row in nonzero])
        ),
    }
    report = {
        "suite": args.suite,
        "episode_index": args.episode_index,
        "task_index": task_index,
        "task_episode_rank": task_episode_rank,
        "action_source": str(args.predicted_actions) if args.predicted_actions else args.perturbation,
        "camera_jitter_px": args.camera_jitter_px,
        "camera_jitter_deg": args.camera_jitter_deg,
        "num_steps_wait": args.num_steps_wait,
        "aggregate": aggregate,
        "horizons": results,
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(json_number(report), indent=2) + "\n")
    print(json.dumps(json_number(report["aggregate"]), indent=2))
    print(report_path)
    gt_env.close()
    pred_env.close()


if __name__ == "__main__":
    main()

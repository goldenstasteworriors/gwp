#!/usr/bin/env python3
"""Evaluate an exported SONIC policy on held-out early/late and history interventions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "third_party" / "giga-datasets",
    PROJECT_ROOT / "third_party" / "giga-models",
    PROJECT_ROOT / "third_party" / "giga-train",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset
from world_action_model.inference import SonicHistoryPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--max-samples-per-episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def _mean(values) -> float | None:
    return float(np.mean(values)) if values else None


def _trajectory_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    """Measure within-chunk motion, including high-frequency shake at dataset FPS."""
    predicted_motion = float(np.linalg.norm(predicted[-1] - predicted[0]))
    target_motion = float(np.linalg.norm(target[-1] - target[0]))
    result: dict[str, float | None] = {
        "horizon_movement_ratio": (
            None if target_motion <= 1e-4 else predicted_motion / target_motion
        ),
        "horizon_stall": float(target_motion > 1e-4 and predicted_motion < 0.2 * target_motion),
        "velocity_mse": None,
        "acceleration_mse": None,
        "predicted_jitter_energy": None,
        "target_jitter_energy": None,
        "excess_jitter_energy": None,
    }
    if predicted.shape[0] >= 2:
        predicted_velocity = np.diff(predicted, axis=0)
        target_velocity = np.diff(target, axis=0)
        result["velocity_mse"] = float(np.mean((predicted_velocity - target_velocity) ** 2))
    if predicted.shape[0] >= 3:
        predicted_acceleration = np.diff(predicted, n=2, axis=0)
        target_acceleration = np.diff(target, n=2, axis=0)
        predicted_energy = float(np.mean(predicted_acceleration**2))
        target_energy = float(np.mean(target_acceleration**2))
        result.update(
            {
                "acceleration_mse": float(
                    np.mean((predicted_acceleration - target_acceleration) ** 2)
                ),
                "predicted_jitter_energy": predicted_energy,
                "target_jitter_energy": target_energy,
                "excess_jitter_energy": max(0.0, predicted_energy - target_energy),
            }
        )
    return result


def _summary(records: list[dict], prefix: str) -> dict:
    keys = (
        "first_step_mse",
        "horizon_mse",
        "terminal_mse",
        "movement_ratio",
        "stall",
        "horizon_movement_ratio",
        "horizon_stall",
        "velocity_mse",
        "acceleration_mse",
        "predicted_jitter_energy",
        "target_jitter_energy",
        "excess_jitter_energy",
        "hold_horizon_mse",
        "hold_terminal_mse",
        "hold_horizon_stall",
        "horizon_mse_vs_hold_ratio",
        "latency_s",
        "repeat_first_step_mse",
        "reverse_first_step_mse",
    )
    result = {"samples": len(records)}
    for key in keys:
        result[f"{prefix}{key}"] = _mean([row[key] for row in records if row.get(key) is not None])
    correct = result.get(f"{prefix}first_step_mse")
    repeat = result.get(f"{prefix}repeat_first_step_mse")
    reverse = result.get(f"{prefix}reverse_first_step_mse")
    result[f"{prefix}repeat_history_degradation"] = (
        None if correct is None or repeat is None else repeat - correct
    )
    result[f"{prefix}reverse_history_degradation"] = (
        None if correct is None or reverse is None else reverse - correct
    )
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = args.output or manifest_path.parent / "benchmark.json"
    output = output.expanduser().resolve()

    dataset_root = Path(manifest["evaluation_dataset"]).expanduser().resolve()
    episodes = [int(value) for value in manifest.get("evaluation_episodes", [20, 21, 22, 23])]
    image_key = str(manifest.get("image_key", "observation.images.ego_view"))
    image_offsets = [int(value) for value in manifest["image_frame_offsets"]]
    state_offsets = [int(value) for value in manifest.get("state_history_offsets", [0])]
    action_horizon = int(manifest["action_horizon"])
    physical_dim = int(manifest["physical_action_dim"])
    visual_history_frames = int(manifest.get("visual_history_frames", 1))
    fps = float(manifest.get("fps", 50.0))

    dataset = WAMLeRobotDataset(
        data_path=str(dataset_root),
        delta_info={"action": action_horizon},
        delta_frames={image_key: image_offsets, "observation.state": state_offsets},
        video_backend="pyav",
        t5_load_from="dir",
        t5_cfg={"t5_embedding_dir": str(dataset_root / "t5_embedding")},
        robotype="sonic_g1",
        episodes=episodes,
    )
    dataset.open()
    policy = SonicHistoryPolicy(manifest_path, device=args.device)

    episode_counts = defaultdict(int)
    episode_last_frames = defaultdict(lambda: -1)
    sampled_frames = defaultdict(list)
    previous = {}
    jitter_rows = []
    records = []
    for index in range(len(dataset)):
        raw = dataset._get_data(index)
        episode = int(torch.as_tensor(raw["episode_index"]).item())
        frame = int(torch.as_tensor(raw["frame_index"]).item())
        episode_last_frames[episode] = max(episode_last_frames[episode], frame)
        if frame % args.sample_stride != 0:
            continue
        if args.max_samples_per_episode and episode_counts[episode] >= args.max_samples_per_episode:
            continue
        episode_counts[episode] += 1
        sampled_frames[episode].append(frame)

        images = torch.as_tensor(raw[image_key])[:visual_history_frames]
        states = torch.as_tensor(raw["observation.state"], dtype=torch.float32)
        target = torch.as_tensor(raw["action"], dtype=torch.float32)[:, :physical_dim].numpy()
        is_pad = torch.as_tensor(
            raw.get("action_is_pad", torch.zeros(target.shape[0], dtype=torch.bool))
        ).flatten().bool().numpy()
        valid_steps = int((~is_pad[: target.shape[0]]).sum())
        valid_steps = max(1, min(valid_steps, target.shape[0]))
        progress = float(raw.get("episode_progress", 0.0))

        started = time.perf_counter()
        predicted = policy.predict(images, states, progress, seed=args.seed + frame)
        latency = time.perf_counter() - started
        predicted = predicted[:valid_steps]
        valid_target = target[:valid_steps]
        current_state = states[-1, :physical_dim].numpy()
        gt_motion = float(np.linalg.norm(valid_target[0] - current_state))
        pred_motion = float(np.linalg.norm(predicted[0] - current_state))
        movement_ratio = None if gt_motion <= 1e-4 else pred_motion / gt_motion

        row = {
            "episode": episode,
            "frame": frame,
            "time_s": frame / fps,
            "split": "post5" if frame / fps >= 5.0 else "pre5",
            "first_step_mse": float(np.mean((predicted[0] - valid_target[0]) ** 2)),
            "horizon_mse": float(np.mean((predicted - valid_target) ** 2)),
            "terminal_mse": float(np.mean((predicted[-1] - valid_target[-1]) ** 2)),
            "movement_ratio": movement_ratio,
            "stall": float(gt_motion > 1e-4 and pred_motion < 0.2 * gt_motion),
            "latency_s": latency,
            "repeat_first_step_mse": None,
            "reverse_first_step_mse": None,
        }
        row.update(_trajectory_metrics(predicted, valid_target))
        hold_prediction = np.broadcast_to(current_state, valid_target.shape)
        hold_metrics = _trajectory_metrics(hold_prediction, valid_target)
        hold_horizon_mse = float(np.mean((hold_prediction - valid_target) ** 2))
        row.update(
            {
                "hold_horizon_mse": hold_horizon_mse,
                "hold_terminal_mse": float(
                    np.mean((hold_prediction[-1] - valid_target[-1]) ** 2)
                ),
                "hold_horizon_stall": hold_metrics["horizon_stall"],
                "horizon_mse_vs_hold_ratio": (
                    None
                    if hold_horizon_mse <= 1e-8
                    else row["horizon_mse"] / hold_horizon_mse
                ),
            }
        )

        if row["split"] == "post5" and (len(state_offsets) > 1 or visual_history_frames > 1):
            repeated_states = states[-1:].expand_as(states)
            current_image = images[visual_history_frames - 1]
            repeated_images = current_image[None].expand_as(images)
            repeated = policy.predict(
                repeated_images,
                repeated_states,
                progress,
                seed=args.seed + frame,
            )[0]
            reversed_prediction = policy.predict(
                torch.flip(images, dims=(0,)),
                torch.flip(states, dims=(0,)),
                progress,
                seed=args.seed + frame,
            )[0]
            row["repeat_first_step_mse"] = float(np.mean((repeated - valid_target[0]) ** 2))
            row["reverse_first_step_mse"] = float(
                np.mean((reversed_prediction - valid_target[0]) ** 2)
            )

        if episode in previous:
            prev_pred, prev_target = previous[episode]
            jitter_rows.append(
                {
                    "split": row["split"],
                    "jitter_mse": float(
                        np.mean(((predicted[0] - prev_pred) - (valid_target[0] - prev_target)) ** 2)
                    ),
                }
            )
        previous[episode] = (predicted[0].copy(), valid_target[0].copy())
        records.append(row)
        print(
            f"episode={episode} frame={frame} split={row['split']} "
            f"first_mse={row['first_step_mse']:.6g} latency={latency:.3f}s",
            flush=True,
        )

    pre = [row for row in records if row["split"] == "pre5"]
    post = [row for row in records if row["split"] == "post5"]
    summary = {
        "manifest": str(manifest_path),
        "evaluation_dataset": str(dataset_root),
        "episodes": episodes,
        "sample_stride": args.sample_stride,
        "max_samples_per_episode": args.max_samples_per_episode,
        "episode_coverage": {
            str(episode): {
                "last_observed_frame": episode_last_frames[episode],
                "last_observed_time_s": episode_last_frames[episode] / fps,
                "last_sampled_frame": max(sampled_frames[episode], default=None),
                "last_sampled_time_s": (
                    max(sampled_frames[episode]) / fps if sampled_frames[episode] else None
                ),
                "tail_gap_frames": (
                    episode_last_frames[episode] - max(sampled_frames[episode])
                    if sampled_frames[episode]
                    else None
                ),
                "samples": len(sampled_frames[episode]),
            }
            for episode in episodes
        },
        "metric_notes": {
            "jitter_mse": "Prediction-transition error between sparsely sampled benchmark calls.",
            "acceleration_mse": "Within-chunk 50 Hz second-difference error against ground truth.",
            "excess_jitter_energy": "Positive excess predicted second-difference energy over ground truth.",
            "horizon_stall": "Predicted end-to-end chunk motion is below 20% of nontrivial ground-truth motion.",
            "hold_horizon_mse": "MSE of a reference policy that holds the current joint state for the full chunk.",
            "horizon_mse_vs_hold_ratio": "Policy horizon MSE divided by hold-policy MSE; below one beats stopping.",
        },
        "all": _summary(records, ""),
        "pre5": _summary(pre, ""),
        "post5": _summary(post, ""),
        "pre5_jitter_mse": _mean(
            [row["jitter_mse"] for row in jitter_rows if row["split"] == "pre5"]
        ),
        "post5_jitter_mse": _mean(
            [row["jitter_mse"] for row in jitter_rows if row["split"] == "post5"]
        ),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved benchmark to {output}")


if __name__ == "__main__":
    main()

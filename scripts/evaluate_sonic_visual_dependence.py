#!/usr/bin/env python3
"""Paired held-out test of a SONIC policy's dependence on its image input.

For every anchor, action/state/text/noise stay fixed while the conditioning
image is either correct, taken from a distant time in the same episode, or
taken from another held-out episode at approximately the same progress.
"""

from __future__ import annotations

import argparse
import json
import math
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

from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset  # noqa: E402
from world_action_model.inference import SonicHistoryPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--episodes",
        default="",
        help="Comma-separated evaluation episodes; defaults to the manifest list.",
    )
    parser.add_argument("--samples-per-episode", type=int, default=24)
    parser.add_argument("--executed-prefix-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def stable_seed(base_seed: int, episode: int, frame: int) -> int:
    value = (
        int(base_seed) * 1_000_003
        + int(episode) * 97_409
        + (int(frame) + 100_000) * 65_537
    )
    return value % (2**63 - 1)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def paired_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    sem = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sem": sem,
        "two_sem": 2.0 * sem,
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def mse_groups(predicted: np.ndarray, target: np.ndarray, prefix: int) -> dict[str, float]:
    prefix = max(1, min(int(prefix), predicted.shape[0]))
    error = (predicted - target) ** 2
    return {
        "prefix_mse": float(error[:prefix].mean()),
        "prefix_arm_mse": float(error[:prefix, :7].mean()),
        "prefix_hand_mse": float(error[:prefix, 7:13].mean()),
        "horizon_mse": float(error.mean()),
    }


def summarize(records: list[dict]) -> dict:
    modes = ("normal", "same_episode_wrong_time", "cross_episode")
    result: dict[str, object] = {}
    for mode in modes:
        result[mode] = {
            key: mean([row[mode][key] for row in records])
            for key in ("prefix_mse", "prefix_arm_mse", "prefix_hand_mse", "horizon_mse")
        }
    normal = result["normal"]
    for mode in modes[1:]:
        deltas = [row[mode]["prefix_mse"] - row["normal"]["prefix_mse"] for row in records]
        shifts = [row[mode]["prefix_prediction_shift_mse"] for row in records]
        mode_summary = result[mode]
        mode_summary["prefix_mse_degradation"] = paired_summary(deltas)
        mode_summary["prefix_mse_relative_degradation_percent"] = (
            100.0 * mean(deltas) / normal["prefix_mse"]
        )
        mode_summary["prefix_prediction_shift_mse"] = mean(shifts)
    result["per_episode"] = {}
    for episode in sorted({int(row["episode"]) for row in records}):
        rows = [row for row in records if int(row["episode"]) == episode]
        result["per_episode"][str(episode)] = {
            mode: {
                "prefix_mse": mean([row[mode]["prefix_mse"] for row in rows]),
                "degradation_vs_normal": mean(
                    [row[mode]["prefix_mse"] - row["normal"]["prefix_mse"] for row in rows]
                ),
            }
            for mode in modes
        }
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    dataset_root = Path(manifest["evaluation_dataset"]).expanduser().resolve()
    episodes = (
        [int(value) for value in args.episodes.split(",") if value.strip()]
        if args.episodes
        else [int(value) for value in manifest["evaluation_episodes"]]
    )
    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for cross-episode intervention")
    image_key = str(manifest.get("image_key", "observation.images.ego_view"))
    image_offsets = [int(value) for value in manifest["image_frame_offsets"]]
    state_offsets = [int(value) for value in manifest.get("state_history_offsets", [0])]
    horizon = int(manifest["action_horizon"])
    physical_dim = int(manifest["physical_action_dim"])
    visual_history = int(manifest.get("visual_history_frames", 1))

    dataset = WAMLeRobotDataset(
        data_path=str(dataset_root),
        delta_info={"action": horizon},
        delta_frames={image_key: image_offsets, "observation.state": state_offsets},
        video_backend="pyav",
        t5_load_from="dir",
        t5_cfg={"t5_embedding_dir": str(dataset_root / "t5_embedding")},
        robotype="sonic_g1",
        episodes=episodes,
    )
    dataset.open()

    # Read anchor metadata without decoding every video frame.  FastLeRobot's
    # filtered HF table has the same row order as dataset[index].
    hf_dataset = dataset.dataset.hf_dataset
    episode_column = hf_dataset["episode_index"]
    frame_column = hf_dataset["frame_index"]
    if len(episode_column) != len(dataset) or len(frame_column) != len(dataset):
        raise RuntimeError("Filtered metadata length differs from evaluation dataset")
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, (episode, frame) in enumerate(zip(episode_column, frame_column)):
        candidates[int(episode)].append((index, int(frame)))
    for values in candidates.values():
        values.sort(key=lambda value: value[1])

    anchors: list[tuple[int, int, int, int]] = []
    for episode in episodes:
        values = candidates[episode]
        count = min(args.samples_per_episode, len(values))
        positions = np.unique(np.linspace(0, len(values) - 1, num=count, dtype=np.int64))
        anchors.extend((episode, int(pos), values[int(pos)][0], values[int(pos)][1]) for pos in positions)

    policy = SonicHistoryPolicy(manifest_path, device=args.device)
    records = []
    started_all = time.perf_counter()
    for number, (episode, position, index, frame) in enumerate(anchors, start=1):
        values = candidates[episode]
        wrong_position = (position + max(1, len(values) // 2)) % len(values)
        wrong_index, wrong_frame = values[wrong_position]

        donor_episode = episodes[(episodes.index(episode) + 1) % len(episodes)]
        donor_values = candidates[donor_episode]
        relative_position = position / max(1, len(values) - 1)
        donor_position = int(round(relative_position * max(0, len(donor_values) - 1)))
        donor_index, donor_frame = donor_values[donor_position]

        raw = dataset._get_data(index)
        wrong_raw = dataset._get_data(wrong_index)
        donor_raw = dataset._get_data(donor_index)
        images = torch.as_tensor(raw[image_key])[:visual_history]
        wrong_images = torch.as_tensor(wrong_raw[image_key])[:visual_history]
        donor_images = torch.as_tensor(donor_raw[image_key])[:visual_history]
        states = torch.as_tensor(raw["observation.state"], dtype=torch.float32)
        target = torch.as_tensor(raw["action"], dtype=torch.float32)[:, :physical_dim].numpy()
        is_pad = torch.as_tensor(
            raw.get("action_is_pad", torch.zeros(target.shape[0], dtype=torch.bool))
        ).flatten().bool().numpy()
        valid_steps = max(1, min(int((~is_pad[: target.shape[0]]).sum()), target.shape[0]))
        target = target[:valid_steps]
        progress = float(raw.get("episode_progress", 0.0))
        seed = stable_seed(args.seed, episode, frame)

        predictions = {
            "normal": policy.predict(images, states, progress, seed=seed)[:valid_steps],
            "same_episode_wrong_time": policy.predict(
                wrong_images, states, progress, seed=seed
            )[:valid_steps],
            "cross_episode": policy.predict(donor_images, states, progress, seed=seed)[:valid_steps],
        }
        normal_prediction = predictions["normal"]
        row = {
            "episode": episode,
            "frame": frame,
            "same_episode_donor_frame": wrong_frame,
            "cross_episode_donor_episode": donor_episode,
            "cross_episode_donor_frame": donor_frame,
            "valid_steps": valid_steps,
        }
        for mode, prediction in predictions.items():
            metrics = mse_groups(prediction, target, args.executed_prefix_steps)
            metrics["prefix_prediction_shift_mse"] = float(
                np.mean(
                    (prediction[: args.executed_prefix_steps] - normal_prediction[: args.executed_prefix_steps])
                    ** 2
                )
            )
            row[mode] = metrics
        records.append(row)
        print(
            f"[{number}/{len(anchors)}] episode={episode} frame={frame} "
            f"normal={row['normal']['prefix_mse']:.7g} "
            f"wrong_time={row['same_episode_wrong_time']['prefix_mse']:.7g} "
            f"cross_episode={row['cross_episode']['prefix_mse']:.7g}",
            flush=True,
        )

    payload = {
        "format_version": 1,
        "manifest": str(manifest_path),
        "checkpoint": manifest["transformer"],
        "evaluation_dataset": str(dataset_root),
        "episodes": episodes,
        "samples_per_episode": args.samples_per_episode,
        "executed_prefix_steps": args.executed_prefix_steps,
        "seed": args.seed,
        "intervention_contract": {
            "fixed": ["action_target", "state", "text", "diffusion_noise", "diffusion_timestep"],
            "same_episode_wrong_time": "image from the opposite half of the same episode",
            "cross_episode": "image from next held-out episode at matched normalized progress",
        },
        "elapsed_seconds": time.perf_counter() - started_all,
        "summary": summarize(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved visual-dependence benchmark to {output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired comparison of LIBERO open-loop action-chunk diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260813
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
LOWER_IS_BETTER = (
    "mse",
    "mae",
    "translation_mse",
    "translation_mae",
    "rotation_mse",
    "rotation_mae",
    "gripper_mse",
    "gripper_mae",
    "clipped_mse",
    "out_of_range_element_fraction",
    "out_of_range_step_fraction",
)
HIGHER_IS_BETTER = ("gripper_sign_accuracy",)
REQUIRED_PROTOCOL_FIELDS = {
    "dataset_root",
    "suite",
    "num_views",
    "view_keys",
    "policy_checkpoint",
    "policy_num_inference_steps",
    "policy_model_dtype",
    "policy_seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    if summary.get("schema_version") != "libero-open-loop-v2":
        raise ValueError(f"Expected libero-open-loop-v2 summary: {path}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Summary contains no episodes: {path}")
    if int(summary.get("num_episodes", -1)) != len(episodes):
        raise ValueError(f"Summary episode count is inconsistent: {path}")
    return summary


def validate_protocol(summary: dict[str, Any], side: str) -> dict[str, Any]:
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{side} summary is missing its protocol")
    missing = sorted(
        key
        for key in REQUIRED_PROTOCOL_FIELDS
        if key not in protocol or protocol[key] is None
    )
    if missing:
        raise ValueError(f"{side} protocol is incomplete: {missing}")
    return protocol


def index_episodes(summary: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed = {}
    for row in summary["episodes"]:
        if not isinstance(row, dict):
            raise ValueError("Every episode record must be a JSON object")
        key = int(row["task_id"]), int(row["episode_index"])
        if key in indexed:
            raise ValueError(f"Duplicate episode key: {key}")
        for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER):
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Non-finite {metric} for episode {key}")
        if not isinstance(row.get("target_actions_sha256"), str):
            raise ValueError(f"Missing target action hash for episode {key}")
        indexed[key] = row
    return indexed


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def task_stratified_bootstrap(
    values_by_task: dict[int, list[float]], *, seed: int, num_resamples: int
) -> dict[str, float | int]:
    if num_resamples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not values_by_task or any(not values for values in values_by_task.values()):
        raise ValueError("Every task must contain paired values")
    generator = random.Random(seed)
    bootstrap_means = []
    for _ in range(num_resamples):
        task_means = []
        for values in values_by_task.values():
            task_means.append(
                sum(values[generator.randrange(len(values))] for _ in values) / len(values)
            )
        bootstrap_means.append(sum(task_means) / len(task_means))
    bootstrap_means.sort()
    task_means = [sum(values) / len(values) for values in values_by_task.values()]
    return {
        "seed": seed,
        "num_resamples": num_resamples,
        "num_tasks": len(values_by_task),
        "mean": sum(task_means) / len(task_means),
        "ci95_low": percentile(bootstrap_means, 0.025),
        "ci95_high": percentile(bootstrap_means, 0.975),
    }


def compare(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    seed: int,
    num_resamples: int,
) -> dict[str, Any]:
    left_protocol = validate_protocol(left, "Left")
    right_protocol = validate_protocol(right, "Right")
    left_scientific = {
        key: value for key, value in left_protocol.items() if key != "policy_checkpoint"
    }
    right_scientific = {
        key: value for key, value in right_protocol.items() if key != "policy_checkpoint"
    }
    if left_scientific != right_scientific:
        raise ValueError("Open-loop evaluation protocols differ")
    for key in ("episode_selection_sha256", "episode_selection_file"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Open-loop episode selection differs for {key}")

    left_rows = index_episodes(left)
    right_rows = index_episodes(right)
    if set(left_rows) != set(right_rows):
        raise ValueError("Open-loop episode keys differ")

    pairs = []
    metric_values: dict[str, dict[int, list[float]]] = {
        metric: defaultdict(list) for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER)
    }
    for task_id, episode_index in sorted(left_rows):
        left_row = left_rows[(task_id, episode_index)]
        right_row = right_rows[(task_id, episode_index)]
        if left_row["target_actions_sha256"] != right_row["target_actions_sha256"]:
            raise ValueError(f"Expert target differs for episode {episode_index}")
        effects = {}
        for metric in LOWER_IS_BETTER:
            effect = float(right_row[metric]) - float(left_row[metric])
            effects[metric] = effect
            metric_values[metric][task_id].append(effect)
        for metric in HIGHER_IS_BETTER:
            effect = float(left_row[metric]) - float(right_row[metric])
            effects[metric] = effect
            metric_values[metric][task_id].append(effect)
        pairs.append(
            {
                "task_id": task_id,
                "episode_index": episode_index,
                "target_actions_sha256": left_row["target_actions_sha256"],
                "positive_means_left_is_better": effects,
            }
        )

    effects = {}
    for metric_index, metric in enumerate((*LOWER_IS_BETTER, *HIGHER_IS_BETTER)):
        effects[metric] = {
            "direction": (
                "right_minus_left; positive favors left"
                if metric in LOWER_IS_BETTER
                else "left_minus_right; positive favors left"
            ),
            **task_stratified_bootstrap(
                metric_values[metric],
                seed=seed + metric_index,
                num_resamples=num_resamples,
            ),
        }
    return {
        "scientific_protocol": left_scientific,
        "policy_checkpoints": {
            "left": left_protocol["policy_checkpoint"],
            "right": right_protocol["policy_checkpoint"],
        },
        "num_pairs": len(pairs),
        "paired_effect_bootstrap_95ci": effects,
        "paired_episodes": pairs,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output}")
    if args.left_label == args.right_label:
        raise ValueError("left/right labels must differ")
    left_path = args.left.resolve()
    right_path = args.right.resolve()
    report = {
        "schema_version": "libero-open-loop-paired-comparison-v1",
        "left": {
            "label": args.left_label,
            "summary": str(left_path),
            "sha256": sha256_file(left_path),
        },
        "right": {
            "label": args.right_label,
            "summary": str(right_path),
            "sha256": sha256_file(right_path),
        },
        **compare(
            load_summary(left_path),
            load_summary(right_path),
            seed=args.seed,
            num_resamples=args.bootstrap_samples,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in report.items() if key != "paired_episodes"}
    print(json.dumps(compact, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

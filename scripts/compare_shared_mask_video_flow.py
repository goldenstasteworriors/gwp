#!/usr/bin/env python3
"""Paired comparison of held-out shared mask-video flow evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


EVALUATION_SCHEMA = "shared-mask-video-flow-evaluation-v1"
SCHEMA_VERSION = "shared-mask-video-flow-paired-comparison-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError(f"Unsupported evaluation schema: {path}")
    if int(value.get("num_samples", -1)) <= 0:
        raise ValueError(f"Evaluation has no samples: {path}")
    rows = value.get("per_sample")
    if not isinstance(rows, list) or len(rows) != int(value["num_samples"]):
        raise ValueError(f"Evaluation must contain complete per-sample rows: {path}")
    return value


def validate_common(left: dict[str, Any], right: dict[str, Any]) -> None:
    required_equal = (
        "training_comparison_id",
        "vae",
        "dataset",
        "cache",
        "cache_manifest_sha256",
        "policy_action_array",
        "simulator_action_array",
        "norm_stats",
        "num_samples",
        "num_views",
        "view_keys",
        "model_dtype",
        "timesteps",
        "seed",
        "shuffle_semantics",
    )
    for key in required_equal:
        if left.get(key) != right.get(key):
            raise ValueError(f"Evaluation protocols differ for {key}")
    if left.get("checkpoint") == right.get("checkpoint"):
        raise ValueError("Paired evaluations use the same checkpoint")
    if left.get("training_variant") == right.get("training_variant"):
        raise ValueError("Paired evaluations use the same training variant")


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
        raise ValueError("bootstrap sample count must be positive")
    if not values_by_task or any(not values for values in values_by_task.values()):
        raise ValueError("Every task must contain paired effects")
    generator = random.Random(seed)
    means: list[float] = []
    task_ids = sorted(values_by_task)
    for _ in range(num_resamples):
        task_means = []
        for task_id in task_ids:
            values = values_by_task[task_id]
            sampled = [values[generator.randrange(len(values))] for _ in values]
            task_means.append(sum(sampled) / len(sampled))
        means.append(sum(task_means) / len(task_means))
    means.sort()
    observed_task_means = [
        sum(values_by_task[task_id]) / len(values_by_task[task_id])
        for task_id in task_ids
    ]
    return {
        "num_tasks": len(task_ids),
        "num_resamples": num_resamples,
        "seed": seed,
        "mean": sum(observed_task_means) / len(observed_task_means),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def index_rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in report["per_sample"]:
        episode = int(row["episode_index"])
        if episode in indexed:
            raise ValueError(f"Duplicate episode {episode}")
        indexed[episode] = row
    return indexed


def collect_effects(
    left_rows: dict[int, dict[str, Any]],
    right_rows: dict[int, dict[str, Any]],
    timesteps: list[int],
    value: Callable[[dict[str, float]], float],
) -> tuple[dict[int, list[float]], dict[str, dict[int, list[float]]]]:
    overall: dict[int, list[float]] = defaultdict(list)
    by_timestep: dict[str, dict[int, list[float]]] = {
        str(step): defaultdict(list) for step in timesteps
    }
    for episode in sorted(left_rows):
        left = left_rows[episode]
        right = right_rows[episode]
        if int(left["task_index"]) != int(right["task_index"]):
            raise ValueError(f"Task mismatch for episode {episode}")
        if int(left["shuffle_index"]) != int(right["shuffle_index"]):
            raise ValueError(f"Shuffle mismatch for episode {episode}")
        task = int(left["task_index"])
        for step in timesteps:
            key = str(step)
            left_values = left["timesteps"][key]
            right_values = right["timesteps"][key]
            if not math.isclose(
                float(left_values["zero_flow_mse"]),
                float(right_values["zero_flow_mse"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Zero-flow target mismatch for episode {episode}")
            effect = value(right_values) - value(left_values)
            if not math.isfinite(effect):
                raise ValueError(f"Non-finite paired effect for episode {episode}")
            overall[task].append(effect)
            by_timestep[key][task].append(effect)
    return overall, by_timestep


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    left = load_report(args.left)
    right = load_report(args.right)
    validate_common(left, right)
    left_rows = index_rows(left)
    right_rows = index_rows(right)
    if set(left_rows) != set(right_rows):
        raise ValueError("Evaluation episode sets differ")
    timesteps = [int(value) for value in left["timesteps"]]
    definitions: dict[str, Callable[[dict[str, float]], float]] = {
        "correct_action_mse": lambda row: float(row["correct_action_mse"]),
        "shuffled_action_mse": lambda row: float(row["shuffled_action_mse"]),
        "action_sensitivity_advantage": lambda row: float(
            row["shuffled_action_mse"] - row["correct_action_mse"]
        ),
    }
    effects: dict[str, Any] = {}
    for offset, (name, function) in enumerate(definitions.items()):
        overall, by_timestep = collect_effects(
            left_rows, right_rows, timesteps, function
        )
        effects[name] = {
            "direction": "right_minus_left",
            "overall": task_stratified_bootstrap(
                overall,
                seed=args.seed + offset * 100,
                num_resamples=args.bootstrap_samples,
            ),
            "by_timestep": {
                step: task_stratified_bootstrap(
                    values,
                    seed=args.seed + offset * 100 + index + 1,
                    num_resamples=args.bootstrap_samples,
                )
                for index, (step, values) in enumerate(by_timestep.items())
            },
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "left": {
            "label": args.left_label,
            "report": str(args.left.resolve()),
            "sha256": sha256_file(args.left),
            "variant": left["training_variant"],
            "checkpoint": left["checkpoint"],
        },
        "right": {
            "label": args.right_label,
            "report": str(args.right.resolve()),
            "sha256": sha256_file(args.right),
            "variant": right["training_variant"],
            "checkpoint": right["checkpoint"],
        },
        "protocol": {
            key: left[key]
            for key in (
                "training_comparison_id",
                "cache",
                "cache_manifest_sha256",
                "policy_action_array",
                "simulator_action_array",
                "num_samples",
                "timesteps",
                "seed",
                "shuffle_semantics",
            )
        },
        "paired_effects": effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

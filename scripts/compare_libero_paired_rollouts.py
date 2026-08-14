#!/usr/bin/env python3
"""Compare two LIBERO evaluations on identical task/init-state rollouts."""

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
REQUIRED_V2_PROTOCOL_FIELDS = {
    "dataset_root",
    "task_ids",
    "rollouts_per_task",
    "max_steps",
    "replan_steps",
    "num_views",
    "seed",
    "policy_checkpoint",
    "policy_num_inference_steps",
    "policy_model_dtype",
    "policy_seed",
}
REQUIRED_V3_PROTOCOL_FIELDS = REQUIRED_V2_PROTOCOL_FIELDS | {"task_prompt_mapping"}
REQUIRED_V4_PROTOCOL_FIELDS = REQUIRED_V3_PROTOCOL_FIELDS | {
    "policy_action_encoding",
    "simulator_action_encoding",
    "policy_to_simulator_conversion",
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rollouts"), list):
        raise ValueError(f"Expected a suite-level LIBERO summary with rollouts: {path}")
    return value


def rollout_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_id"]), int(row["rollout"])


def indexed_rollouts(summary: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in summary["rollouts"]:
        if not isinstance(row, dict):
            raise ValueError("Every rollout record must be a JSON object")
        key = rollout_key(row)
        if key in rows:
            raise ValueError(f"Duplicate rollout key: {key}")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"Rollout success must be boolean: {key}")
        rows[key] = row
    if not rows:
        raise ValueError("Evaluation contains no rollouts")
    if int(summary.get("num_rollouts", len(rows))) != len(rows):
        raise ValueError("Summary num_rollouts does not match rollout records")
    successes = sum(bool(row["success"]) for row in rows.values())
    declared_rate = float(summary.get("success_rate", successes / len(rows)))
    if not math.isclose(declared_rate, successes / len(rows), abs_tol=1e-12):
        raise ValueError("Summary success_rate does not match rollout records")
    declared_suite = summary.get("suite")
    if declared_suite is not None and {key[0] for key in rows} != {str(declared_suite)}:
        raise ValueError("Summary suite does not match rollout records")
    return rows


def validated_protocol(
    summary: dict[str, Any],
    rows: dict[tuple[str, int, int], dict[str, Any]],
    side: str,
) -> dict[str, Any] | None:
    protocol = summary.get("protocol")
    if protocol is not None and not isinstance(protocol, dict):
        raise ValueError(f"{side} protocol must be a JSON object when present")
    schema_version = summary.get("schema_version")
    if schema_version in {
        "libero-closed-loop-v2",
        "libero-closed-loop-v3",
        "libero-closed-loop-v4",
    }:
        if not isinstance(protocol, dict):
            raise ValueError(f"{side} {schema_version} summary is missing its protocol")
        if schema_version == "libero-closed-loop-v4":
            required_fields = REQUIRED_V4_PROTOCOL_FIELDS
        elif schema_version == "libero-closed-loop-v3":
            required_fields = REQUIRED_V3_PROTOCOL_FIELDS
        else:
            required_fields = REQUIRED_V2_PROTOCOL_FIELDS
        missing = sorted(
            key
            for key in required_fields
            if key not in protocol or protocol[key] is None
        )
        if missing:
            raise ValueError(f"{side} {schema_version} protocol is incomplete: {missing}")
        task_ids = sorted({key[1] for key in rows})
        if [int(value) for value in protocol["task_ids"]] != task_ids:
            raise ValueError(f"{side} protocol task_ids do not match rollout records")
        expected_per_task = int(protocol["rollouts_per_task"])
        task_counts: dict[tuple[str, int], int] = defaultdict(int)
        for suite, task_id, _ in rows:
            task_counts[(suite, task_id)] += 1
        if any(count != expected_per_task for count in task_counts.values()):
            raise ValueError(
                f"{side} protocol rollouts_per_task does not match rollout records"
            )
        if schema_version in {"libero-closed-loop-v3", "libero-closed-loop-v4"}:
            prompt_mapping = protocol["task_prompt_mapping"]
            if not isinstance(prompt_mapping, dict):
                raise ValueError(f"{side} task_prompt_mapping must be a JSON object")
            if sorted(int(key) for key in prompt_mapping) != task_ids:
                raise ValueError(
                    f"{side} task_prompt_mapping keys do not match rollout task ids"
                )
            for task_id in task_ids:
                entry = prompt_mapping[str(task_id)]
                if not isinstance(entry, dict):
                    raise ValueError(f"{side} prompt mapping {task_id} must be an object")
                expected_language = {
                    str(row["task_description"])
                    for key, row in rows.items()
                    if key[1] == task_id
                }
                if expected_language != {str(entry.get("task_language", ""))}:
                    raise ValueError(
                        f"{side} prompt language differs from rollout task {task_id}"
                    )
    return protocol


def exact_mcnemar_pvalue(left_wins: int, right_wins: int) -> float:
    discordant = left_wins + right_wins
    if discordant == 0:
        return 1.0
    tail = min(left_wins, right_wins)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def task_stratified_bootstrap(
    paired: list[dict[str, Any]], *, seed: int, num_resamples: int
) -> dict[str, float | int]:
    if num_resamples <= 0:
        raise ValueError("bootstrap samples must be positive")
    by_task: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in paired:
        by_task[(row["suite"], row["task_id"])].append(
            float(row["left_success"]) - float(row["right_success"])
        )
    generator = random.Random(seed)
    bootstrap_means = []
    for _ in range(num_resamples):
        task_means = []
        for values in by_task.values():
            task_means.append(
                sum(values[generator.randrange(len(values))] for _ in values) / len(values)
            )
        bootstrap_means.append(sum(task_means) / len(task_means))
    bootstrap_means.sort()
    observed_task_means = [sum(values) / len(values) for values in by_task.values()]
    return {
        "seed": seed,
        "num_resamples": num_resamples,
        "num_tasks": len(by_task),
        "mean_delta": sum(observed_task_means) / len(observed_task_means),
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
    left_rows = indexed_rollouts(left)
    right_rows = indexed_rollouts(right)
    if set(left_rows) != set(right_rows):
        missing_left = sorted(set(right_rows) - set(left_rows))
        missing_right = sorted(set(left_rows) - set(right_rows))
        raise ValueError(
            f"Rollout keys differ; missing_left={missing_left[:10]}, "
            f"missing_right={missing_right[:10]}"
        )

    left_protocol = validated_protocol(left, left_rows, "Left")
    right_protocol = validated_protocol(right, right_rows, "Right")
    left_scientific_protocol = (
        {key: value for key, value in left_protocol.items() if key != "policy_checkpoint"}
        if isinstance(left_protocol, dict)
        else None
    )
    right_scientific_protocol = (
        {key: value for key, value in right_protocol.items() if key != "policy_checkpoint"}
        if isinstance(right_protocol, dict)
        else None
    )
    if (
        left_scientific_protocol is not None
        and right_scientific_protocol is not None
        and left_scientific_protocol != right_scientific_protocol
    ):
        raise ValueError("Closed-loop evaluation protocols differ")
    if left_scientific_protocol is not None and right_scientific_protocol is not None:
        protocol_status = "verified_equal"
    elif left_scientific_protocol is None and right_scientific_protocol is None:
        protocol_status = "missing_in_both_legacy_summaries"
    else:
        protocol_status = "missing_in_one_legacy_summary"

    paired = []
    for key in sorted(left_rows):
        left_row = left_rows[key]
        right_row = right_rows[key]
        if left_row.get("task_description") != right_row.get("task_description"):
            raise ValueError(f"Task description differs for rollout {key}")
        paired.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "rollout": key[2],
                "left_success": bool(left_row["success"]),
                "right_success": bool(right_row["success"]),
            }
        )

    left_successes = sum(row["left_success"] for row in paired)
    right_successes = sum(row["right_success"] for row in paired)
    left_wins = sum(row["left_success"] and not row["right_success"] for row in paired)
    right_wins = sum(row["right_success"] and not row["left_success"] for row in paired)
    both_success = sum(row["left_success"] and row["right_success"] for row in paired)
    both_failure = len(paired) - left_wins - right_wins - both_success

    by_task = {}
    for suite, task_id in sorted({(row["suite"], row["task_id"]) for row in paired}):
        subset = [
            row for row in paired if row["suite"] == suite and row["task_id"] == task_id
        ]
        left_rate = sum(row["left_success"] for row in subset) / len(subset)
        right_rate = sum(row["right_success"] for row in subset) / len(subset)
        by_task[f"{suite}:{task_id}"] = {
            "num_pairs": len(subset),
            "left_success_rate": left_rate,
            "right_success_rate": right_rate,
            "delta": left_rate - right_rate,
        }

    return {
        "protocol_status": protocol_status,
        "scientific_protocol": (
            left_scientific_protocol
            if left_scientific_protocol == right_scientific_protocol
            else None
        ),
        "policy_checkpoints": {
            "left": left_protocol.get("policy_checkpoint")
            if isinstance(left_protocol, dict)
            else None,
            "right": right_protocol.get("policy_checkpoint")
            if isinstance(right_protocol, dict)
            else None,
        },
        "num_pairs": len(paired),
        "left_success_rate": left_successes / len(paired),
        "right_success_rate": right_successes / len(paired),
        "success_rate_delta": (left_successes - right_successes) / len(paired),
        "paired_outcomes": {
            "left_wins": left_wins,
            "right_wins": right_wins,
            "both_success": both_success,
            "both_failure": both_failure,
            "discordant": left_wins + right_wins,
        },
        "exact_mcnemar_two_sided_p": exact_mcnemar_pvalue(left_wins, right_wins),
        "task_stratified_paired_bootstrap_95ci": task_stratified_bootstrap(
            paired, seed=seed, num_resamples=num_resamples
        ),
        "by_task": by_task,
        "paired_rollouts": paired,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output}")
    if args.left_label == args.right_label:
        raise ValueError("left/right labels must differ")
    left_path = args.left.resolve()
    right_path = args.right.resolve()
    result = compare(
        load_summary(left_path),
        load_summary(right_path),
        seed=args.seed,
        num_resamples=args.bootstrap_samples,
    )
    report = {
        "schema_version": "libero-paired-rollout-comparison-v1",
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
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact_report = {
        key: value for key, value in report.items() if key != "paired_rollouts"
    }
    print(json.dumps(compact_report, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

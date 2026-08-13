#!/usr/bin/env python3
"""Audit paired stage-one runs and recompute the held-out mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


HEAD_HASH_PATTERN = re.compile(
    r"Counterfactual mask head initial SHA256:\s*([0-9a-f]{64})"
)
GATE_KEYS = ("dynamic_iou", "new_occupancy_iou", "gripper_region_recall")
EXPECTED_VARIANT_ENV = {
    "matched": {
        "GWP_COUNTERFACTUAL_MASK_WEIGHT": "0.0",
        "GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL": "1",
    },
    "full": {
        "GWP_COUNTERFACTUAL_MASK_WEIGHT": "0.1",
        "GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL": "0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--matched-output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--matched-eval", type=Path, required=True)
    parser.add_argument("--full-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_report_path(report: dict[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Preflight report has no valid {key!r}")
    return Path(value).resolve()


def find_checkpoint(output_root: Path) -> Path:
    candidates = sorted(output_root.glob("models/checkpoint_*_step_200/transformer"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one step-200 transformer in {output_root}, got {candidates}"
        )
    checkpoint = candidates[0].resolve()
    config = load_json(checkpoint / "config.json")
    if int(config.get("counterfactual_mask_frames_per_latent", 0)) != 4:
        raise ValueError(f"Mask head is not enabled in {checkpoint / 'config.json'}")
    weight_files = sorted(checkpoint.glob("*.bin")) + sorted(
        checkpoint.glob("*.safetensors")
    )
    weight_bytes = sum(path.stat().st_size for path in weight_files if path.is_file())
    if weight_bytes <= 0:
        raise FileNotFoundError(f"No transformer weights in {checkpoint}")
    parent = checkpoint.parent
    forbidden = list(parent.glob("optimizer*.bin")) + list(parent.glob("optimizer_*"))
    if (parent / "pytorch_model").exists():
        forbidden.append(parent / "pytorch_model")
    if forbidden:
        raise ValueError(f"Compact checkpoint retains optimizer state: {forbidden}")
    return checkpoint


def read_initial_head_hash(output_root: Path) -> str:
    log_paths = sorted((output_root / "logs").glob("train_*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No training logs under {output_root / 'logs'}")
    hashes: set[str] = set()
    for log_path in log_paths:
        hashes.update(HEAD_HASH_PATTERN.findall(log_path.read_text(errors="replace")))
    if len(hashes) != 1:
        raise ValueError(
            f"Expected one unique initial mask-head hash in {output_root}, got {hashes}"
        )
    return next(iter(hashes))


def validate_launch(
    output_root: Path,
    variant: str,
    report: dict[str, Any],
    report_path: Path,
) -> tuple[dict[str, Any], Path, str]:
    expected_root = resolve_report_path(
        report, "matched_output" if variant == "matched" else "full_output"
    )
    if output_root.resolve() != expected_root:
        raise ValueError(f"{variant} output does not match preflight report")
    if (output_root / "stage1_exit_code.txt").read_text().strip() != "0":
        raise RuntimeError(f"{variant} training did not exit successfully")
    manifest = load_json(output_root / "stage1_launch_manifest.json")
    if manifest.get("status") != "complete" or int(manifest.get("exit_code", -1)) != 0:
        raise RuntimeError(f"{variant} launch manifest is not complete")
    if manifest.get("variant") != variant:
        raise ValueError(f"Wrong variant in {output_root}")
    common = manifest.get("common")
    environment = manifest.get("environment")
    if not isinstance(common, dict) or not isinstance(environment, dict):
        raise ValueError(f"Malformed launch manifest: {output_root}")
    if common.get("preflight_report_sha256") != sha256_file(report_path):
        raise ValueError(f"{variant} launch/preflight hash mismatch")
    if common.get("git_commit") != report.get("git_commit"):
        raise ValueError(f"{variant} launch/preflight Git commit mismatch")
    for key, expected in EXPECTED_VARIANT_ENV[variant].items():
        if environment.get(key) != expected:
            raise ValueError(f"{variant} has unexpected {key}={environment.get(key)!r}")
    checkpoint = find_checkpoint(output_root)
    initial_hash = read_initial_head_hash(output_root)
    return manifest, checkpoint, initial_hash


def validate_pair(
    matched: dict[str, Any],
    full: dict[str, Any],
    matched_hash: str,
    full_hash: str,
) -> None:
    if matched.get("comparison_id") != full.get("comparison_id"):
        raise ValueError("Matched/full comparison_id differs")
    if matched.get("common") != full.get("common"):
        raise ValueError("Matched/full common launch settings differ")
    if matched_hash != full_hash:
        raise ValueError("Matched/full mask-head initialization differs")

    matched_env = dict(matched["environment"])
    full_env = dict(full["environment"])
    allowed_differences = {
        "GWP05_PROJECT_DIR",
        "GWP_COUNTERFACTUAL_MASK_WEIGHT",
        "GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL",
    }
    differing = {
        key
        for key in set(matched_env) | set(full_env)
        if matched_env.get(key) != full_env.get(key)
    }
    if differing != allowed_differences:
        raise ValueError(
            f"Matched/full environment differences are {sorted(differing)}, "
            f"expected {sorted(allowed_differences)}"
        )


def finite_metric(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Non-finite {context}.{key}: {value!r}")
    return float(value)


def validate_evaluation(
    evaluation_path: Path,
    checkpoint: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    evaluation = load_json(evaluation_path)
    if Path(str(evaluation.get("checkpoint"))).resolve() != checkpoint:
        raise ValueError(f"Evaluation/checkpoint mismatch: {evaluation_path}")
    expected_cache = Path(str(report["heldout_cache"]["root"])).resolve()
    expected_mean_cache = Path(str(report["train_cache"]["root"])).resolve()
    if Path(str(evaluation.get("cache"))).resolve() != expected_cache:
        raise ValueError(f"Evaluation/held-out cache mismatch: {evaluation_path}")
    if Path(str(evaluation.get("mean_mask_cache"))).resolve() != expected_mean_cache:
        raise ValueError(f"Evaluation/mean-mask cache mismatch: {evaluation_path}")
    if int(evaluation.get("evaluation_episode_count", -1)) != 50:
        raise ValueError(f"Evaluation must contain 50 held-out episodes: {evaluation_path}")
    if int(evaluation.get("mean_mask_episode_count", -1)) != 200:
        raise ValueError(f"Mean-mask prior must contain 200 train episodes: {evaluation_path}")
    if evaluation.get("shuffle_semantics") != "cyclic derangement within task":
        raise ValueError(f"Taskwise derangement is missing: {evaluation_path}")

    aggregate = evaluation.get("aggregate")
    required_modes = {
        "correct_action",
        "shuffled_action",
        "copy_initial",
        "per_horizon_mean_mask",
    }
    if not isinstance(aggregate, dict) or set(aggregate) != required_modes:
        raise ValueError(f"Unexpected evaluation modes in {evaluation_path}")
    for mode in required_modes:
        if int(aggregate[mode].get("num_samples", -1)) != 50:
            raise ValueError(f"{mode} does not contain 50 samples")
        mean = aggregate[mode].get("mean", {})
        for key in GATE_KEYS:
            finite_metric(mean, key, f"{mode}.mean")
    sensitivity = evaluation.get("correct_minus_shuffled")
    if not isinstance(sensitivity, dict):
        raise ValueError(f"Missing correct-minus-shuffled: {evaluation_path}")
    for key in GATE_KEYS:
        finite_metric(sensitivity, key, "correct_minus_shuffled")
    per_sample = evaluation.get("per_sample")
    if not isinstance(per_sample, list) or len(per_sample) != 50:
        raise ValueError(f"Evaluation must retain 50 per-sample records: {evaluation_path}")
    if [row.get("index") for row in per_sample] != list(range(50)):
        raise ValueError(f"Evaluation per-sample order is invalid: {evaluation_path}")
    for index, row in enumerate(per_sample):
        for mode in ("correct_action", "shuffled_action"):
            result = row.get(mode)
            if not isinstance(result, dict) or not isinstance(result.get("mean"), dict):
                raise ValueError(
                    f"Missing per-sample {mode} result at index {index}: {evaluation_path}"
                )
            for key in GATE_KEYS:
                finite_metric(result["mean"], key, f"per_sample[{index}].{mode}.mean")
    return evaluation


def recompute_gate(
    full: dict[str, Any],
    matched: dict[str, Any],
) -> dict[str, Any]:
    full_aggregate = full["aggregate"]
    correct = full_aggregate["correct_action"]["mean"]
    shuffled = full_aggregate["shuffled_action"]["mean"]
    copy_initial = full_aggregate["copy_initial"]["mean"]
    mean_mask = full_aggregate["per_horizon_mean_mask"]["mean"]
    matched_correct = matched["aggregate"]["correct_action"]["mean"]
    sensitivity = full["correct_minus_shuffled"]
    matched_sensitivity = matched["correct_minus_shuffled"]

    checks: dict[str, bool] = {}
    for key in GATE_KEYS:
        checks[f"correct_minus_shuffled.{key}"] = float(sensitivity[key]) > 0.0
        checks[f"correct_beats_copy_initial.{key}"] = (
            float(correct[key]) > float(copy_initial[key])
        )
        checks[f"correct_beats_mean_mask.{key}"] = (
            float(correct[key]) > float(mean_mask[key])
        )
        checks[f"correct_beats_matched_correct.{key}"] = (
            float(correct[key]) > float(matched_correct[key])
        )
        checks[f"sensitivity_beats_matched_sensitivity.{key}"] = (
            float(sensitivity[key]) > float(matched_sensitivity[key])
        )
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "full_correct": {key: float(correct[key]) for key in GATE_KEYS},
        "full_shuffled": {key: float(shuffled[key]) for key in GATE_KEYS},
        "full_copy_initial": {key: float(copy_initial[key]) for key in GATE_KEYS},
        "full_mean_mask": {key: float(mean_mask[key]) for key in GATE_KEYS},
        "correct_minus_shuffled": {
            key: float(sensitivity[key]) for key in GATE_KEYS
        },
        "matched_correct_minus_shuffled": {
            key: float(matched_sensitivity[key]) for key in GATE_KEYS
        },
        "full_minus_matched_correct": {
            key: float(correct[key]) - float(matched_correct[key]) for key in GATE_KEYS
        },
        "full_minus_matched_sensitivity": {
            key: float(sensitivity[key]) - float(matched_sensitivity[key])
            for key in GATE_KEYS
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output}")
    report_path = args.preflight_report.resolve()
    report = load_json(report_path)
    if report.get("status") != "passed":
        raise ValueError("Preflight report did not pass")

    matched_manifest, matched_checkpoint, matched_hash = validate_launch(
        args.matched_output.resolve(), "matched", report, report_path
    )
    full_manifest, full_checkpoint, full_hash = validate_launch(
        args.full_output.resolve(), "full", report, report_path
    )
    validate_pair(matched_manifest, full_manifest, matched_hash, full_hash)
    matched_eval = validate_evaluation(
        args.matched_eval.resolve(), matched_checkpoint, report
    )
    full_eval = validate_evaluation(args.full_eval.resolve(), full_checkpoint, report)
    for key in ("seed", "threshold", "mask_resolution", "shuffle_semantics"):
        if matched_eval.get(key) != full_eval.get(key):
            raise ValueError(f"Matched/full evaluation setting differs for {key}")

    gate = recompute_gate(full_eval, matched_eval)
    audit = {
        "audit_status": "passed",
        "comparison_id": matched_manifest["comparison_id"],
        "initial_mask_head_sha256": matched_hash,
        "matched_checkpoint": str(matched_checkpoint),
        "full_checkpoint": str(full_checkpoint),
        "mechanism_gate": gate,
        "control_evaluation_allowed": gate["status"] == "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

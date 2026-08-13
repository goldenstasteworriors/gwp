#!/usr/bin/env python3
"""Audit baseline/matched/full paired control evaluations after the mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from compare_libero_open_loop import (  # noqa: E402
    compare as recompute_open_comparison,
    load_summary as load_open_summary,
)
from compare_libero_paired_rollouts import (  # noqa: E402
    compare as recompute_closed_comparison,
    load_summary as load_closed_summary,
)


BUNDLE_COMMON_KEYS = (
    "git_commit",
    "vae",
    "vae_config_sha256",
    "norm_stats",
    "norm_stats_sha256",
    "dataset_root",
    "dataset_info_sha256",
    "suite",
    "episodes_file",
    "episodes_file_sha256",
    "rollouts_per_task",
    "max_steps",
    "replan_steps",
    "num_inference_steps",
    "model_dtype",
    "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--matched-bundle", type=Path, required=True)
    parser.add_argument("--full-bundle", type=Path, required=True)
    parser.add_argument("--full-vs-matched-open", type=Path, required=True)
    parser.add_argument("--full-vs-baseline-open", type=Path, required=True)
    parser.add_argument("--full-vs-matched-closed", type=Path, required=True)
    parser.add_argument("--full-vs-baseline-closed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_bundle(path: Path) -> dict[str, Any]:
    bundle = load_json(path)
    if bundle.get("schema_version") != "libero-policy-evaluation-bundle-v1":
        raise ValueError(f"Unexpected bundle schema: {path}")
    if bundle.get("status") != "complete":
        raise RuntimeError(f"Evaluation bundle is not complete: {path}")
    checkpoint = bundle.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("root"):
        raise ValueError(f"Bundle checkpoint provenance is incomplete: {path}")
    if not checkpoint.get("weight_files") or int(checkpoint.get("weight_bytes", 0)) <= 0:
        raise ValueError(f"Bundle checkpoint weights are incomplete: {path}")
    for kind in ("open_loop", "closed_loop"):
        summary = Path(str(bundle[f"{kind}_summary"])).resolve()
        if sha256_file(summary) != bundle[f"{kind}_summary_sha256"]:
            raise ValueError(f"Bundle {kind} summary hash mismatch: {path}")
    return bundle


def validate_bundle_pair(bundles: dict[str, dict[str, Any]]) -> None:
    reference = bundles["full"]
    for name, bundle in bundles.items():
        for key in BUNDLE_COMMON_KEYS:
            if bundle.get(key) != reference.get(key):
                raise ValueError(f"Bundle {name} differs from full for {key}")
    checkpoints = [
        str(bundle["checkpoint"]["root"]) for bundle in bundles.values()
    ]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("Baseline/matched/full bundles must use distinct checkpoints")


def validate_comparison_side(
    comparison: dict[str, Any],
    side: str,
    bundle: dict[str, Any],
    summary_kind: str,
) -> None:
    record = comparison.get(side)
    if not isinstance(record, dict):
        raise ValueError(f"Comparison is missing {side} provenance")
    summary = Path(str(bundle[f"{summary_kind}_summary"])).resolve()
    if Path(str(record.get("summary"))).resolve() != summary:
        raise ValueError(f"Comparison {side} points to the wrong summary")
    if record.get("sha256") != bundle[f"{summary_kind}_summary_sha256"]:
        raise ValueError(f"Comparison {side} summary hash mismatch")


def validate_open_comparison(
    path: Path,
    full: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    comparison = load_json(path)
    if comparison.get("schema_version") != "libero-open-loop-paired-comparison-v1":
        raise ValueError(f"Unexpected open-loop comparison schema: {path}")
    validate_comparison_side(comparison, "left", full, "open_loop")
    validate_comparison_side(comparison, "right", right, "open_loop")
    if int(comparison.get("num_pairs", -1)) != 50:
        raise ValueError(f"Open-loop comparison must contain 50 pairs: {path}")
    effects = comparison.get("paired_effect_bootstrap_95ci")
    if not isinstance(effects, dict) or "mse" not in effects:
        raise ValueError(f"Open-loop comparison has no paired effects: {path}")
    mse_effect = effects["mse"]
    expected = recompute_open_comparison(
        load_open_summary(Path(str(full["open_loop_summary"]))),
        load_open_summary(Path(str(right["open_loop_summary"]))),
        seed=int(mse_effect["seed"]),
        num_resamples=int(mse_effect["num_resamples"]),
    )
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise ValueError(f"Open-loop comparison does not independently recompute: {path}")
    return comparison


def finite_float(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Non-finite {context}.{key}: {value!r}")
    return float(value)


def validate_closed_comparison(
    path: Path,
    full: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    comparison = load_json(path)
    if comparison.get("schema_version") != "libero-paired-rollout-comparison-v1":
        raise ValueError(f"Unexpected closed-loop comparison schema: {path}")
    validate_comparison_side(comparison, "left", full, "closed_loop")
    validate_comparison_side(comparison, "right", right, "closed_loop")
    if comparison.get("protocol_status") != "verified_equal":
        raise ValueError(f"Closed-loop protocol was not verified: {path}")
    if int(comparison.get("num_pairs", -1)) != 100:
        raise ValueError(f"Closed-loop comparison must contain 100 pairs: {path}")
    outcomes = comparison.get("paired_outcomes")
    interval = comparison.get("task_stratified_paired_bootstrap_95ci")
    if not isinstance(outcomes, dict) or not isinstance(interval, dict):
        raise ValueError(f"Closed-loop comparison is incomplete: {path}")
    if sum(
        int(outcomes[key])
        for key in ("left_wins", "right_wins", "both_success", "both_failure")
    ) != 100:
        raise ValueError(f"Closed-loop paired outcome counts do not sum to 100: {path}")
    for key in ("success_rate_delta", "exact_mcnemar_two_sided_p"):
        finite_float(comparison, key, str(path))
    for key in ("mean_delta", "ci95_low", "ci95_high"):
        finite_float(interval, key, f"{path}.bootstrap")
    expected = recompute_closed_comparison(
        load_closed_summary(Path(str(full["closed_loop_summary"]))),
        load_closed_summary(Path(str(right["closed_loop_summary"]))),
        seed=int(interval["seed"]),
        num_resamples=int(interval["num_resamples"]),
    )
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise ValueError(f"Closed-loop comparison does not independently recompute: {path}")
    return comparison


def control_effect(comparison: dict[str, Any]) -> dict[str, Any]:
    outcomes = comparison["paired_outcomes"]
    interval = comparison["task_stratified_paired_bootstrap_95ci"]
    delta = float(comparison["success_rate_delta"])
    left_wins = int(outcomes["left_wins"])
    right_wins = int(outcomes["right_wins"])
    p_value = float(comparison["exact_mcnemar_two_sided_p"])
    ci_low = float(interval["ci95_low"])
    ci_high = float(interval["ci95_high"])
    return {
        "success_rate_delta": delta,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "exact_mcnemar_two_sided_p": p_value,
        "task_stratified_ci95_low": ci_low,
        "task_stratified_ci95_high": ci_high,
        "positive_effect": delta > 0.0 and left_wins > right_wins,
        "statistically_supported_positive_effect": (
            delta > 0.0 and left_wins > right_wins and p_value < 0.05 and ci_low > 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    pair_audit = load_json(args.pair_audit.resolve())
    if pair_audit.get("audit_status") != "passed":
        raise ValueError("Stage-one pair audit did not pass")
    if pair_audit.get("mechanism_gate", {}).get("status") != "passed":
        raise ValueError("Stage-one mechanism gate did not pass")
    if pair_audit.get("control_evaluation_allowed") is not True:
        raise ValueError("Stage-one pair audit did not authorize control evaluation")

    bundles = {
        "baseline": validate_bundle(args.baseline_bundle.resolve()),
        "matched": validate_bundle(args.matched_bundle.resolve()),
        "full": validate_bundle(args.full_bundle.resolve()),
    }
    validate_bundle_pair(bundles)
    expected_checkpoints = {
        "baseline": args.baseline_checkpoint.resolve(),
        "matched": Path(str(pair_audit["matched_checkpoint"])).resolve(),
        "full": Path(str(pair_audit["full_checkpoint"])).resolve(),
    }
    for name, expected in expected_checkpoints.items():
        actual = Path(str(bundles[name]["checkpoint"]["root"])).resolve()
        if actual != expected:
            raise ValueError(
                f"{name} control bundle checkpoint {actual} does not match {expected}"
            )
    open_full_matched = validate_open_comparison(
        args.full_vs_matched_open.resolve(), bundles["full"], bundles["matched"]
    )
    open_full_baseline = validate_open_comparison(
        args.full_vs_baseline_open.resolve(), bundles["full"], bundles["baseline"]
    )
    closed_full_matched = validate_closed_comparison(
        args.full_vs_matched_closed.resolve(), bundles["full"], bundles["matched"]
    )
    closed_full_baseline = validate_closed_comparison(
        args.full_vs_baseline_closed.resolve(), bundles["full"], bundles["baseline"]
    )

    causal_effect = control_effect(closed_full_matched)
    baseline_effect = control_effect(closed_full_baseline)
    audit = {
        "schema_version": "counterfactual-control-evaluation-audit-v1",
        "audit_status": "passed",
        "pair_audit": str(args.pair_audit.resolve()),
        "pair_audit_sha256": sha256_file(args.pair_audit.resolve()),
        "git_commit": bundles["full"]["git_commit"],
        "checkpoints": {
            name: bundle["checkpoint"]["root"] for name, bundle in bundles.items()
        },
        "bundle_manifest_sha256": {
            "baseline": sha256_file(args.baseline_bundle.resolve()),
            "matched": sha256_file(args.matched_bundle.resolve()),
            "full": sha256_file(args.full_bundle.resolve()),
        },
        "causal_full_vs_matched": causal_effect,
        "utility_full_vs_baseline": baseline_effect,
        "head_only_and_gt_action_controls_allowed": causal_effect["positive_effect"],
        "open_loop_diagnostic_files": {
            "full_vs_matched": {
                "path": str(args.full_vs_matched_open.resolve()),
                "sha256": sha256_file(args.full_vs_matched_open.resolve()),
            },
            "full_vs_baseline": {
                "path": str(args.full_vs_baseline_open.resolve()),
                "sha256": sha256_file(args.full_vs_baseline_open.resolve()),
            },
        },
        "closed_loop_comparison_files": {
            "full_vs_matched": {
                "path": str(args.full_vs_matched_closed.resolve()),
                "sha256": sha256_file(args.full_vs_matched_closed.resolve()),
            },
            "full_vs_baseline": {
                "path": str(args.full_vs_baseline_closed.resolve()),
                "sha256": sha256_file(args.full_vs_baseline_closed.resolve()),
            },
        },
        "open_loop_primary_mse_effects": {
            "full_vs_matched": open_full_matched["paired_effect_bootstrap_95ci"]["mse"],
            "full_vs_baseline": open_full_baseline["paired_effect_bootstrap_95ci"]["mse"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

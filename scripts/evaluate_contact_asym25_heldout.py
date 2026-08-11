#!/usr/bin/env python3
"""Prepare fixed windows or run production asymmetric25 held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_asym25_eval_runtime import (  # noqa: E402
    ProductionHeldoutRuntime,
)
from world_action_model.contact_asym25_heldout_eval import (  # noqa: E402
    ARTIFACT_RECEIPT_VERSION,
    CHECKPOINT_STEPS,
    EVALUATION_SEEDS,
    RESULT_VERSION,
    HeldoutEvaluationError,
    build_code_receipt,
    build_environment_receipt,
    build_fixed_window_manifest,
    checkpoint_entries_by_key,
    load_and_replay_preflight_receipt,
    load_checkpoint_manifest,
    load_completed_group_split,
    load_fixed_window_manifest,
    public_tree_receipt,
    reduce_metric_rows,
    score_heldout_window,
    snapshot_artifact_tree,
    verify_artifact_tree_unchanged,
    write_per_window_csv,
    write_strict_json,
)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed production evaluator for contact/asymmetric25 WAM. "
            "No default path writes under /tmp."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-windows",
        help="Create deterministic val/test windows for the receipt seed before training.",
    )
    prepare.add_argument("--preflight-receipt", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate step 0/100/200 for the receipt-selected split seed."
    )
    evaluate.add_argument("--preflight-receipt", type=Path, required=True)
    evaluate.add_argument("--window-manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint-manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    evaluate.add_argument("--inference-steps", type=int, default=10)
    evaluate.add_argument("--overwrite", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate",
        help="Offline-check and aggregate exactly one report for each production seed.",
    )
    aggregate.add_argument(
        "--result", type=Path, action="append", required=True,
        help="Repeat exactly three times for split seeds 6666, 7777, and 8888.",
    )
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--overwrite", action="store_true")

    fingerprint = subparsers.add_parser(
        "fingerprint-tree",
        help="Print the evaluator's deterministic tree receipt for a local bundle.",
    )
    fingerprint.add_argument("--path", type=Path, required=True)
    return parser


def _load_source_contract(preflight_path: Path):
    preflight, preflight_sha256 = load_and_replay_preflight_receipt(preflight_path)
    split_path = Path(preflight["split_manifest_path"])
    if split_path != Path(preflight["artifact_paths"]["split_manifest"]):
        raise HeldoutEvaluationError("preflight split path aliases differ")
    split_rows, assignments = load_completed_group_split(
        split_path,
        expected_sha256=preflight["split_manifest_sha256"],
        expected_episode_lengths=preflight["episode_lengths"],
    )
    return preflight, preflight_sha256, split_path, split_rows, assignments


def prepare_windows(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve(strict=False)
    if output == Path("/tmp") or Path("/tmp") in output.parents:
        raise HeldoutEvaluationError("window manifests must not be written under /tmp")
    preflight, preflight_sha256, _, _, assignments = _load_source_contract(
        args.preflight_receipt
    )
    manifest = build_fixed_window_manifest(
        preflight_receipt=preflight,
        preflight_receipt_sha256=preflight_sha256,
        split_assignments=assignments,
    )
    digest = write_strict_json(output, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "window_count": len(manifest["windows"]),
            },
            sort_keys=True,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir == Path("/tmp") or Path("/tmp") in output_dir.parents:
        raise HeldoutEvaluationError("evaluation outputs must not be written under /tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "contact_asym25_heldout_per_window.csv"
    result_path = output_dir / "contact_asym25_heldout_metrics.json"
    hash_path = output_dir / "contact_asym25_heldout_metrics.json.sha256"
    if hash_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite result hash: {hash_path}")

    preflight, preflight_sha256, split_path, _, assignments = _load_source_contract(
        args.preflight_receipt
    )
    windows, windows_sha256 = load_fixed_window_manifest(
        args.window_manifest,
        preflight_receipt=preflight,
        preflight_receipt_sha256=preflight_sha256,
        split_assignments=assignments,
    )
    dataset_snapshot = snapshot_artifact_tree(preflight["data_root"])
    checkpoint_manifest, checkpoint_manifest_sha256, artifact_snapshots = (
        load_checkpoint_manifest(
            args.checkpoint_manifest,
            preflight_receipt=preflight,
            preflight_receipt_sha256=preflight_sha256,
            split_assignments=assignments,
            dataset_tree_sha256=dataset_snapshot["tree_sha256"],
            heldout_window_manifest_sha256=windows_sha256,
        )
    )
    code_receipt = build_code_receipt(REPO_ROOT)
    environment_receipt = build_environment_receipt(
        device=args.device,
        dtype=args.dtype,
        inference_steps=args.inference_steps,
    )
    entries = checkpoint_entries_by_key(checkpoint_manifest)
    selected_seed = preflight["split_seed"]
    selected_windows = [
        window for window in windows["windows"] if window["seed"] == selected_seed
    ]
    if len(selected_windows) != 512:
        raise HeldoutEvaluationError(
            "the receipt seed must expose 256 val and 256 test fixed windows"
        )
    runtime = ProductionHeldoutRuntime(
        preflight_receipt=preflight,
        base_model_path=checkpoint_manifest["base_model"]["path"],
        device=args.device,
        dtype=args.dtype,
        inference_steps=args.inference_steps,
    )
    rows = []
    try:
        for step in CHECKPOINT_STEPS:
            entry = entries[(selected_seed, step)]
            runtime.load_checkpoint(entry["checkpoint_path"])
            for window in selected_windows:
                transformed = runtime.load_window(window)
                prediction = runtime.predict(
                    transformed,
                    action_noise_seed=window["action_noise_seed"],
                )
                rows.append(
                    score_heldout_window(
                        window=window,
                        split_seed=selected_seed,
                        checkpoint_step=step,
                        checkpoint_sha256=entry["checkpoint_bundle_sha256"],
                        action_prediction=prediction["action_prediction"],
                        action_target=transformed["action"].cpu().numpy(),
                        action_dim_mask=transformed["action_dim_mask"].cpu().numpy(),
                        action_time_mask=transformed[
                            "action_time_mask"
                        ].cpu().numpy(),
                        action_denorm_scale=transformed[
                            "action_denorm_scale"
                        ].cpu().numpy(),
                        contact_logits=prediction["contact_logits"],
                        contact_hard_semantic=prediction["contact_hard_semantic"],
                        contact_target=transformed["contact_target"].cpu().numpy(),
                        contact_loss_mask=transformed[
                            "contact_loss_mask"
                        ].cpu().numpy(),
                        contact_confidence=transformed[
                            "contact_confidence"
                        ].cpu().numpy(),
                        contact_condition_mask=transformed[
                            "contact_condition_mask"
                        ].cpu().numpy(),
                    )
                )
            runtime.unload_checkpoint()
    finally:
        runtime.close()

    expected_rows = len(CHECKPOINT_STEPS) * 512
    if len(rows) != expected_rows:
        raise HeldoutEvaluationError(
            f"evaluation emitted {len(rows)} rows; expected {expected_rows}"
        )
    verify_artifact_tree_unchanged(dataset_snapshot)
    for snapshot in artifact_snapshots.values():
        verify_artifact_tree_unchanged(snapshot)

    results = []
    for step in CHECKPOINT_STEPS:
        for heldout_split in ("val", "test"):
            selected = [
                row
                for row in rows
                if row["split_seed"] == selected_seed
                and row["checkpoint_step"] == step
                and row["split"] == heldout_split
            ]
            if len(selected) != 256:
                raise HeldoutEvaluationError(
                    "seed/step/split result does not contain exactly 256 windows"
                )
            results.append(
                {
                    "split_seed": selected_seed,
                    "checkpoint_step": step,
                    "heldout_split": heldout_split,
                    "metrics": reduce_metric_rows(selected),
                }
            )
    per_window_sha256 = write_per_window_csv(
        csv_path, rows, overwrite=args.overwrite
    )
    checkpoint_artifacts = [
        public_tree_receipt(snapshot)
        for _, snapshot in sorted(artifact_snapshots.items())
    ]
    result = {
        "result_version": RESULT_VERSION,
        "status": "passed",
        "scientific_claim_automatic": False,
        "source_receipt": {
            "production_preflight_receipt": {
                "path_record_only": str(args.preflight_receipt.expanduser().absolute()),
                "sha256": preflight_sha256,
            },
            "split_manifest": {
                "path_record_only": str(split_path),
                "sha256": preflight["split_manifest_sha256"],
            },
            "fixed_window_manifest": {
                "path_record_only": str(args.window_manifest.expanduser().absolute()),
                "sha256": windows_sha256,
            },
            "dataset_tree": public_tree_receipt(dataset_snapshot),
        },
        "checkpoint_receipt": {
            "manifest": {
                "path_record_only": str(
                    args.checkpoint_manifest.expanduser().absolute()
                ),
                "sha256": checkpoint_manifest_sha256,
            },
            "verified_artifact_trees": checkpoint_artifacts,
        },
        "code_receipt": code_receipt,
        "environment_receipt": environment_receipt,
        "evaluation_contract": {
            "augmentation_enabled": False,
            "eval_crop_mode": "center",
            "windows_per_val_seed": 256,
            "windows_per_test_seed": 256,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "split_seed": selected_seed,
            "action_target_access": "metrics_only_after_prediction",
            "oracle_contact_condition": False,
        },
        "results_by_seed_step_split": results,
        "per_window_csv": {
            "path_record_only": str(csv_path),
            "sha256": per_window_sha256,
            "row_count": len(rows),
        },
    }
    result_sha256 = write_strict_json(
        result_path, result, overwrite=args.overwrite
    )
    if hash_path.exists() and args.overwrite:
        hash_path.unlink()
    hash_path.write_text(
        f"{result_sha256}  {result_path.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": result_sha256,
                "per_window_csv": str(csv_path),
                "per_window_sha256": per_window_sha256,
            },
            sort_keys=True,
        )
    )


def aggregate(args: argparse.Namespace) -> None:
    from world_action_model.contact_asym25_heldout_eval import (
        AGGREGATE_RESULT_VERSION,
        load_evaluation_result_bundle,
    )

    output_path = args.output.expanduser().resolve(strict=False)
    if output_path == Path("/tmp") or Path("/tmp") in output_path.parents:
        raise HeldoutEvaluationError("aggregate reports must not be written under /tmp")
    if len(args.result) != len(EVALUATION_SEEDS):
        raise HeldoutEvaluationError(
            "aggregate requires exactly three --result paths"
        )
    bundles = [load_evaluation_result_bundle(path) for path in args.result]
    by_seed = {}
    for bundle in bundles:
        seed = bundle["preflight_receipt"]["split_seed"]
        if seed in by_seed:
            raise HeldoutEvaluationError(f"duplicate aggregate split seed: {seed}")
        by_seed[seed] = bundle
    if set(by_seed) != set(EVALUATION_SEEDS):
        raise HeldoutEvaluationError(
            "aggregate reports must cover exactly seeds 6666, 7777, and 8888"
        )
    dataset_ids = {
        bundle["preflight_receipt"]["dataset_id"] for bundle in bundles
    }
    split_hashes = {
        bundle["preflight_receipt"]["split_manifest_sha256"]
        for bundle in bundles
    }
    dataset_tree_hashes = {
        bundle["result"]["source_receipt"]["dataset_tree"]["tree_sha256"]
        for bundle in bundles
    }
    code_commits = {
        bundle["result"]["code_receipt"]["git_commit"] for bundle in bundles
    }
    base_model_hashes = {
        bundle["checkpoint_manifest"]["base_model"]["bundle_sha256"]
        for bundle in bundles
    }
    runtime_contracts = {
        (
            bundle["result"]["environment_receipt"].get("model_dtype"),
            bundle["result"]["environment_receipt"].get("inference_steps"),
            bundle["result"]["environment_receipt"].get(
                "deterministic_algorithms"
            ),
            bundle["result"]["environment_receipt"].get(
                "augmentation_enabled"
            ),
            bundle["result"]["environment_receipt"].get("eval_crop_mode"),
        )
        for bundle in bundles
    }
    if (
        len(dataset_ids) != 1
        or len(split_hashes) != 1
        or len(dataset_tree_hashes) != 1
        or len(code_commits) != 1
        or len(base_model_hashes) != 1
        or len(runtime_contracts) != 1
    ):
        raise HeldoutEvaluationError(
            "aggregate reports differ in dataset/split/source/code/base/runtime identity"
        )
    aggregated = []
    for step in CHECKPOINT_STEPS:
        for heldout_split in ("val", "test"):
            selected = [
                row
                for bundle in bundles
                for row in bundle["rows"]
                if row["checkpoint_step"] == step
                and row["split"] == heldout_split
            ]
            if len(selected) != len(EVALUATION_SEEDS) * 256:
                raise HeldoutEvaluationError(
                    "aggregate step/split does not contain 256 windows per seed"
                )
            aggregated.append(
                {
                    "checkpoint_step": step,
                    "heldout_split": heldout_split,
                    "metrics": reduce_metric_rows(selected),
                }
            )
    output = {
        "aggregate_result_version": AGGREGATE_RESULT_VERSION,
        "status": "passed",
        "scientific_claim_automatic": False,
        "dataset_id": next(iter(dataset_ids)),
        "split_manifest_sha256": next(iter(split_hashes)),
        "dataset_tree_sha256": next(iter(dataset_tree_hashes)),
        "evaluator_git_commit": next(iter(code_commits)),
        "base_model_bundle_sha256": next(iter(base_model_hashes)),
        "split_seeds": list(EVALUATION_SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "source_reports": [
            {
                "split_seed": seed,
                "path_record_only": str(by_seed[seed]["path"]),
                "sha256": by_seed[seed]["result_sha256"],
                "production_preflight_receipt_sha256": by_seed[seed][
                    "preflight_receipt_sha256"
                ],
                "fixed_window_manifest_sha256": by_seed[seed][
                    "window_manifest_sha256"
                ],
                "per_window_csv_sha256": by_seed[seed]["per_window_csv_sha256"],
            }
            for seed in EVALUATION_SEEDS
        ],
        "aggregate_across_seeds": aggregated,
    }
    digest = write_strict_json(output_path, output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


def fingerprint_tree(args: argparse.Namespace) -> None:
    receipt = public_tree_receipt(snapshot_artifact_tree(args.path))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    args = _base_parser().parse_args()
    if args.command == "prepare-windows":
        prepare_windows(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "fingerprint-tree":
        fingerprint_tree(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

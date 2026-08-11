#!/usr/bin/env python3
"""Run or offline-replay the causal-main held-out evaluator v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_action_model.contact_asym25_heldout_eval import (  # noqa: E402
    ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
    HeldoutEvaluationError,
    build_environment_receipt,
    build_fixed_window_manifest,
    contact_grammar_validity,
    load_and_replay_preflight_receipt,
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
from world_action_model.contact_causal_checkpoint import (  # noqa: E402
    validate_checkpoint_manifest_v3,
)
from world_action_model.contact_causal_heldout_eval import (  # noqa: E402
    CAUSAL_EVALUATION_CONTRACT,
    build_aggregate_v5,
    validate_result_v3,
)
from world_action_model.contact_causal_preflight import (  # noqa: E402
    validate_preflight_receipt_v2,
)
from world_action_model.contact_causal_production import (  # noqa: E402
    CAUSAL_ROUTES,
    CHECKPOINT_STEPS,
    EVALUATOR_VERSION,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    RESULT_VERSION,
    CausalProductionError,
    ensure_not_project_tmp,
    sha256_json,
    strict_json_file,
)
from world_action_model.contact_transition_schema import (  # noqa: E402
    EQUAL_BIT_VALID_STATES,
    unstructured_code_to_ids,
    validate_hard_contact_segment_code,
)


def _source_contract(causal_path: Path, hardened_path: Path):
    causal, causal_sha, _ = strict_json_file(causal_path, name="causal preflight v2")
    validate_preflight_receipt_v2(causal)
    try:
        hardened, hardened_sha = load_and_replay_preflight_receipt(hardened_path)
        _, assignments = load_completed_group_split(
            hardened["split_manifest_path"],
            expected_sha256=hardened["split_manifest_sha256"],
            expected_episode_lengths=hardened["episode_lengths"],
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("hardened v2 held-out source replay failed") from exc
    training = causal["training_data_receipt"]
    mirrors = {
        "dataset_id": "dataset_id",
        "split_seed": "split_seed",
        "dataset_info_sha256": "dataset_info_sha256",
        "hardware_contract_sha256": "hardware_contract_sha256",
        "calibration_capture_sha256": "calibration_capture_sha256",
        "train_only_norm_sha256": "norm_stats_sha256",
        "sidecar_bundle_sha256": "sidecar_bundle_sha256",
        "split_manifest_sha256": "split_manifest_sha256",
    }
    if any(training[new] != hardened.get(old) for new, old in mirrors.items()):
        raise CausalProductionError("causal preflight differs from mature replay")
    return causal, causal_sha, hardened, hardened_sha, assignments


def _digest_array(digest: Any, window_id: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    for payload in (
        window_id.encode("utf-8"), str(array.dtype).encode("ascii"),
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def prepare_windows(args: argparse.Namespace) -> None:
    output = ensure_not_project_tmp(args.output, project_root=REPOSITORY_ROOT)
    try:
        hardened, hardened_sha = load_and_replay_preflight_receipt(
            args.hardened_preflight_receipt
        )
        _, assignments = load_completed_group_split(
            hardened["split_manifest_path"],
            expected_sha256=hardened["split_manifest_sha256"],
            expected_episode_lengths=hardened["episode_lengths"],
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("cannot prepare mature fixed windows") from exc
    manifest = build_fixed_window_manifest(
        preflight_receipt=hardened,
        preflight_receipt_sha256=hardened_sha,
        split_assignments=assignments,
    )
    digest = write_strict_json(output, manifest)
    print(json.dumps({"output": str(output), "sha256": digest}, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    output_dir = ensure_not_project_tmp(args.output_dir, project_root=REPOSITORY_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    causal, causal_sha, hardened, hardened_sha, assignments = _source_contract(
        args.causal_preflight_receipt, args.hardened_preflight_receipt
    )
    try:
        windows, windows_sha = load_fixed_window_manifest(
            args.window_manifest,
            preflight_receipt=hardened,
            preflight_receipt_sha256=hardened_sha,
            split_assignments=assignments,
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("fixed held-out windows replay failed") from exc
    manifest, manifest_sha, _ = strict_json_file(
        args.checkpoint_manifest, name="causal checkpoint manifest v3"
    )
    manifest, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
    seed = causal["training_data_receipt"]["split_seed"]
    route = manifest["causal_route_id"]
    if manifest["seed"] != seed or manifest["causal_route_id"] != route:
        raise CausalProductionError("checkpoint manifest differs from causal preflight")
    try:
        base_model_snapshot = snapshot_artifact_tree(args.base_model)
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("base-model artifact tree replay failed") from exc
    environment_receipt = build_environment_receipt(
        device=args.device,
        dtype="bfloat16",
        inference_steps=args.inference_steps,
    )
    selected_windows = [window for window in windows["windows"] if window["seed"] == seed]
    if len(selected_windows) != 512:
        raise CausalProductionError("causal evaluator needs 256 val + 256 test windows")

    from world_action_model.contact_causal_eval_runtime import (
        CausalProductionHeldoutRuntime,
    )

    runtime = CausalProductionHeldoutRuntime(
        preflight_receipt=hardened,
        base_model_path=args.base_model,
        device=args.device,
        dtype="bfloat16",
        inference_steps=args.inference_steps,
    )
    rows = []
    step_digests = {}
    packet_values: dict[int, list[np.ndarray]] = {step: [] for step in CHECKPOINT_STEPS}
    try:
        for step in CHECKPOINT_STEPS:
            receipt, _ = accepted[step]
            checkpoint_dir = Path(receipt["checkpoint_file"]["path_record_only"]).parent
            runtime.load_checkpoint(str(checkpoint_dir))
            if runtime.causal_route != route:
                raise CausalProductionError("runtime checkpoint route differs from manifest")
            digests = {
                "action": hashlib.sha256(),
                "logits": hashlib.sha256(),
                "packet": hashlib.sha256(),
            }
            for window in selected_windows:
                transformed = runtime.load_window(window)
                prediction = runtime.predict(
                    transformed, action_noise_seed=window["action_noise_seed"]
                )
                _digest_array(digests["action"], window["window_id"], prediction["action_prediction"])
                _digest_array(digests["logits"], window["window_id"], prediction["contact_logits"])
                packet = prediction["route_packet"]
                if route == "aux_only_direct_q12":
                    if packet is not None:
                        raise CausalProductionError("direct evaluator observed a packet")
                else:
                    if packet is None:
                        raise CausalProductionError("strict evaluator omitted route packet")
                    _digest_array(digests["packet"], window["window_id"], packet)
                    packet_values[step].append(packet)
                rows.append(
                    score_heldout_window(
                        window=window,
                        split_seed=seed,
                        checkpoint_step=step,
                        checkpoint_sha256=receipt["checkpoint_artifact_hashes"]["whole_bundle_sha256"],
                        action_prediction=prediction["action_prediction"],
                        action_target=transformed["action"].cpu().numpy(),
                        action_dim_mask=transformed["action_dim_mask"].cpu().numpy(),
                        action_time_mask=transformed["action_time_mask"].cpu().numpy(),
                        action_denorm_scale=transformed["action_denorm_scale"].cpu().numpy(),
                        action_schema_contract=ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
                        grip_active_future=None,
                        contact_logits=prediction["contact_logits"],
                        contact_hard_semantic=prediction["contact_hard_semantic"],
                        contact_target=transformed["contact_target"].cpu().numpy(),
                        contact_loss_mask=transformed["contact_loss_mask"].cpu().numpy(),
                        contact_confidence=transformed["contact_confidence"].cpu().numpy(),
                        contact_condition_mask=transformed["contact_condition_mask"].cpu().numpy(),
                    )
                )
            step_digests[step] = {name: digest.hexdigest() for name, digest in digests.items()}
            runtime.unload_checkpoint()
    finally:
        runtime.close()
    try:
        verify_artifact_tree_unchanged(base_model_snapshot)
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("base-model artifact changed during evaluation") from exc
    if len(rows) != 1536:
        raise CausalProductionError("causal evaluator emitted an incomplete row matrix")
    csv_path = output_dir / "contact_causal_heldout_per_window.csv"
    csv_sha = write_per_window_csv(csv_path, rows, overwrite=args.overwrite)
    action_contract_sha = sha256_json(ASYMMETRIC25_ACTION_SCHEMA_CONTRACT)
    contact_contract_sha = sha256_json(
        {"source": "shared_semantic_aux_decoder_logits", "q": 12, "width": 22, "b": 16}
    )
    checkpoint_outputs = []
    manifest_entries = {row["step"]: row for row in manifest["checkpoints"]}
    for step in CHECKPOINT_STEPS:
        reduced = {
            split: reduce_metric_rows(
                [row for row in rows if row["checkpoint_step"] == step and row["split"] == split]
            )
            for split in ("val", "test")
        }
        if route == "aux_only_direct_q12":
            packet_sha = None
            packet_metrics = {
                "kind": "none_direct_reference",
                "semantic_field_metrics_allowed": False,
            }
        elif route == "equal_bit_unstructured_hard_st":
            codes = np.stack(packet_values[step])
            ids = unstructured_code_to_ids(codes)
            counts = np.bincount(ids.reshape(-1).astype(np.int64))
            probabilities = counts[counts > 0] / float(ids.size)
            packet_sha = step_digests[step]["packet"]
            packet_metrics = {
                "kind": "unstructured_code",
                "codec_validity": 1.0,
                "utilization": float(np.unique(ids).size / EQUAL_BIT_VALID_STATES),
                "entropy": float(-(probabilities * np.log2(probabilities)).sum()),
                "invalid_to_zero_rate": 0.0,
                "semantic_field_metrics_allowed": False,
            }
        else:
            packets = np.stack(packet_values[step])
            validate_hard_contact_segment_code(packets, time_bins=16)
            grammar = contact_grammar_validity(packets)
            packet_sha = step_digests[step]["packet"]
            packet_metrics = {
                "kind": "semantic_grammar",
                "grammar_validity": float(grammar.mean()),
                "field_accuracy": float(
                    sum(
                        reduced[split]["contact"]["segment_exact"]["hard_b16_fully_supervised_accuracy"]
                        for split in ("val", "test")
                    ) / 2.0
                ),
                "semantic_field_metrics_allowed": True,
            }
        route_output = {
            "evaluator_version": EVALUATOR_VERSION,
            "matrix_id": MATRIX_ID,
            "causal_route_id": route,
            "contact_aux_prediction_source": "shared_semantic_aux_decoder_logits",
            "contact_target_model_input": False,
            "contact_condition_mask_model_input": False,
            "oracle_used": False,
            "packet_kind": {
                "aux_only_direct_q12": "none_direct_reference",
                "equal_bit_unstructured_hard_st": "unstructured_code",
                "hard_st_semantic": "semantic_grammar",
            }[route],
            "packet_present": route != "aux_only_direct_q12",
            "packet_sha256": packet_sha,
            "semantic_aux_logits_sha256": step_digests[step]["logits"],
            "action_prediction_sha256": step_digests[step]["action"],
            "action_metrics_contract_sha256": action_contract_sha,
            "packet_metrics": packet_metrics,
        }
        checkpoint_outputs.append(
            {
                "checkpoint_step": step,
                "checkpoint_acceptance_path_record_only": manifest_entries[step]["independent_acceptance_path"],
                "checkpoint_acceptance_sha256": manifest_entries[step]["independent_acceptance_sha256"],
                "fixed_window_set_sha256": windows_sha,
                "route_output": route_output,
                "action_metrics": {split: reduced[split]["action"] for split in ("val", "test")},
                "contact_aux_metrics": {split: reduced[split]["contact"] for split in ("val", "test")},
            }
        )
    result = {
        "result_version": RESULT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "seed": seed,
        "causal_route_id": route,
        "causal_preflight_receipt": {"path_record_only": str(args.causal_preflight_receipt.absolute()), "sha256": causal_sha},
        "hardened_preflight_receipt": {"path_record_only": str(args.hardened_preflight_receipt.absolute()), "sha256": hardened_sha},
        "checkpoint_manifest": {"path_record_only": str(args.checkpoint_manifest.absolute()), "sha256": manifest_sha},
        "base_model_artifact_tree": public_tree_receipt(base_model_snapshot),
        "fixed_window_manifest": {"path_record_only": str(args.window_manifest.absolute()), "sha256": windows_sha},
        "per_window_csv": {"path_record_only": str(csv_path.absolute()), "sha256": csv_sha, "row_count": len(rows)},
        "evaluator_source_receipt_sha256": causal["current_head_source_tree_sha256"],
        "precursor_receipt_bytes_sha256": manifest["precursor_receipt_bytes_sha256"],
        "action_metrics_contract_sha256": action_contract_sha,
        "contact_aux_metrics_contract_sha256": contact_contract_sha,
        "evaluation_contract": CAUSAL_EVALUATION_CONTRACT,
        "environment_receipt": environment_receipt,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "primary_checkpoint_step": 200,
        "best_checkpoint_selection_allowed": False,
        "checkpoint_outputs": checkpoint_outputs,
        "oracle_used": False,
        "benefit_claimed": False,
    }
    validate_result_v3(result, verify_files=True)
    result_path = output_dir / "contact_causal_heldout_result.json"
    result_sha = write_strict_json(result_path, result, overwrite=args.overwrite)
    print(json.dumps({"result": str(result_path), "sha256": result_sha}, sort_keys=True))


def aggregate(args: argparse.Namespace) -> None:
    if len(args.result) != 9:
        raise CausalProductionError("causal aggregate requires exactly nine results")
    paths = {}
    evidence = {}
    for path in args.result:
        result, _, _ = strict_json_file(path, name="causal held-out result")
        validate_result_v3(result, verify_files=True)
        key = f"{result['seed']}/{result['causal_route_id']}"
        if key in paths:
            raise CausalProductionError("duplicate causal aggregate seed/route")
        paths[key] = path
        manifest, _, _ = strict_json_file(
            result["checkpoint_manifest"]["path_record_only"],
            name="causal aggregate checkpoint manifest",
        )
        _, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
        for step, pair in accepted.items():
            evidence[(result["seed"], result["causal_route_id"], step)] = pair
    ordered_paths = {
        f"{seed}/{route}": paths[f"{seed}/{route}"]
        for seed in (6666, 7777, 8888)
        for route in CAUSAL_ROUTES
    }
    output = build_aggregate_v5(
        result_path_by_seed_route=ordered_paths, evidence=evidence, verify_files=True
    )
    target = ensure_not_project_tmp(args.output, project_root=REPOSITORY_ROOT)
    digest = write_strict_json(target, output, overwrite=args.overwrite)
    print(json.dumps({"aggregate": str(target), "sha256": digest}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Causal-main held-out evaluator v2")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-windows")
    prepare.add_argument("--hardened-preflight-receipt", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--causal-preflight-receipt", type=Path, required=True)
    run.add_argument("--hardened-preflight-receipt", type=Path, required=True)
    run.add_argument("--window-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-manifest", type=Path, required=True)
    run.add_argument("--base-model", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--inference-steps", type=int, default=10)
    run.add_argument("--overwrite", action="store_true")
    merge = commands.add_parser("aggregate")
    merge.add_argument("--result", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    {"prepare-windows": prepare_windows, "evaluate": evaluate, "aggregate": aggregate}[
        args.command
    ](args)


if __name__ == "__main__":
    main()

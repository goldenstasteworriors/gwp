#!/usr/bin/env python3
"""Run or offline-replay the causal-main held-out evaluator v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
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
    ROUTE_OUTPUT_EVIDENCE_FORMAT,
    _recompute_route_packet_metrics,
    validate_result_v3,
    write_route_output_evidence,
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

    cache_parent = output_dir.parent / ".contact_causal_eval_private_hf_cache"
    cache_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_stat = cache_parent.lstat()
    if (
        cache_parent.is_symlink()
        or not cache_parent.is_dir()
        or cache_stat.st_uid != os.geteuid()
        or stat.S_IMODE(cache_stat.st_mode) != 0o700
    ):
        raise CausalProductionError(
            "causal evaluator private HF cache parent must be owned mode 0700"
        )
    evaluator_nonce = "causal-eval-" + hashlib.sha256(
        (
            manifest["training_run_nonce"] + ":" + route + ":" + str(seed)
        ).encode("utf-8")
    ).hexdigest()

    runtime = CausalProductionHeldoutRuntime(
        preflight_receipt=hardened,
        base_model_path=args.base_model,
        device=args.device,
        dtype="bfloat16",
        inference_steps=args.inference_steps,
        verified_private_cache_parent=str(cache_parent),
        verified_run_nonce=evaluator_nonce,
    )
    rows = []
    step_digests = {}
    evidence_steps = []
    evidence_window_ids = []
    evidence_actions = []
    evidence_action_targets = []
    evidence_action_dim_masks = []
    evidence_action_time_masks = []
    evidence_action_denorm_scales = []
    evidence_logits = []
    evidence_hard_semantics = []
    evidence_contact_targets = []
    evidence_contact_loss_masks = []
    evidence_contact_confidences = []
    evidence_contact_condition_masks = []
    evidence_packets = []
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
                inference_packet = runtime.load_inference_window(window)
                prediction = runtime.predict(
                    inference_packet, action_noise_seed=window["action_noise_seed"]
                )
                targets = runtime.load_metric_targets(window)
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
                    evidence_packets.append(packet)
                evidence_steps.append(step)
                evidence_window_ids.append(window["window_id"])
                evidence_actions.append(prediction["action_prediction"])
                evidence_action_targets.append(targets["action"].cpu().numpy())
                evidence_action_dim_masks.append(
                    targets["action_dim_mask"].cpu().numpy()
                )
                evidence_action_time_masks.append(
                    targets["action_time_mask"].cpu().numpy()
                )
                evidence_action_denorm_scales.append(
                    targets["action_denorm_scale"].cpu().numpy()
                )
                evidence_logits.append(prediction["contact_logits"])
                evidence_hard_semantics.append(
                    prediction["contact_hard_semantic"]
                )
                evidence_contact_targets.append(
                    targets["contact_target"].cpu().numpy()
                )
                evidence_contact_loss_masks.append(
                    targets["contact_loss_mask"].cpu().numpy()
                )
                evidence_contact_confidences.append(
                    targets["contact_confidence"].cpu().numpy()
                )
                evidence_contact_condition_masks.append(
                    targets["contact_condition_mask"].cpu().numpy()
                )
                rows.append(
                    score_heldout_window(
                        window=window,
                        split_seed=seed,
                        checkpoint_step=step,
                        checkpoint_sha256=receipt["checkpoint_artifact_hashes"]["whole_bundle_sha256"],
                        action_prediction=prediction["action_prediction"],
                        action_target=targets["action"].cpu().numpy(),
                        action_dim_mask=targets["action_dim_mask"].cpu().numpy(),
                        action_time_mask=targets["action_time_mask"].cpu().numpy(),
                        action_denorm_scale=targets["action_denorm_scale"].cpu().numpy(),
                        action_schema_contract=ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
                        grip_active_future=None,
                        contact_logits=prediction["contact_logits"],
                        contact_hard_semantic=prediction["contact_hard_semantic"],
                        contact_target=targets["contact_target"].cpu().numpy(),
                        contact_loss_mask=targets["contact_loss_mask"].cpu().numpy(),
                        contact_confidence=targets["contact_confidence"].cpu().numpy(),
                        contact_condition_mask=targets["contact_condition_mask"].cpu().numpy(),
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
    route_evidence_path = output_dir / "contact_causal_route_outputs.npz"
    action_evidence = np.stack(evidence_actions).astype(np.float32, copy=False)
    logits_evidence = np.stack(evidence_logits).astype(np.float32, copy=False)
    route_packet_evidence = (
        np.empty((len(evidence_steps), 0, 22), dtype=np.float32)
        if route == "aux_only_direct_q12"
        else np.stack(evidence_packets).astype(np.float32, copy=False)
    )
    route_evidence_sha = write_route_output_evidence(
        route_evidence_path,
        route=route,
        checkpoint_steps=evidence_steps,
        window_ids=evidence_window_ids,
        action_predictions=action_evidence,
        action_targets=np.stack(evidence_action_targets),
        action_dim_masks=np.stack(evidence_action_dim_masks),
        action_time_masks=np.stack(evidence_action_time_masks),
        action_denorm_scales=np.stack(evidence_action_denorm_scales),
        semantic_aux_logits=logits_evidence,
        contact_hard_semantics=np.stack(evidence_hard_semantics),
        contact_targets=np.stack(evidence_contact_targets),
        contact_loss_masks=np.stack(evidence_contact_loss_masks),
        contact_confidences=np.stack(evidence_contact_confidences),
        contact_condition_masks=np.stack(evidence_contact_condition_masks),
        route_packets=route_packet_evidence,
        overwrite=args.overwrite,
    )
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
        else:
            packet_sha = step_digests[step]["packet"]
            step_mask = np.asarray(evidence_steps, dtype=np.int64) == step
            packet_metrics = _recompute_route_packet_metrics(
                route=route,
                logits=logits_evidence[step_mask],
                packets=route_packet_evidence[step_mask],
                reduced_contact_by_split={
                    split: reduced[split]["contact"] for split in ("val", "test")
                },
            )
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
        "route_output_evidence": {
            "path_record_only": str(route_evidence_path.absolute()),
            "sha256": route_evidence_sha,
            "row_count": len(evidence_steps),
            "format": ROUTE_OUTPUT_EVIDENCE_FORMAT,
        },
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

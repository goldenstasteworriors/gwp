#!/usr/bin/env python3
"""Independently replay and accept one causal-main checkpoint.

This command is intentionally separate from the trainer.  It reopens every
bound artifact, verifies current committed code and the Robot v3 external
review, safely reloads the tensor mapping, then writes one O_EXCL acceptance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_action_model.contact_causal_gate_bootstrap import (  # noqa: E402
    ROBOT_BOOTSTRAP_CODE_PATHS,
    _verify_head_code,
    validate_robot_route_acceptance_v1,
)
from world_action_model.contact_causal_preflight import (  # noqa: E402
    build_robot_runtime_source_receipts_v1,
    replay_persisted_hardened_contact_asym25_v2,
    replay_robot_gate_evidence_v3,
    validate_preflight_receipt_v2,
    validate_robot_gate_review_v2,
    validate_robot_gate_v3,
    validate_training_data_receipt_v2,
)
from world_action_model.contact_causal_coverage import (  # noqa: E402
    validate_recomputed_dataset_supervision_coverage,
)
from world_action_model.contact_causal_matched_control import (  # noqa: E402
    compare_causal_records,
    load_trace_jsonl,
    replay_current_run_trace_prefix_v1,
    validate_collector_report_v3,
)
from world_action_model.contact_ego_causal_precursor import (  # noqa: E402
    validate_precursor_receipt,
)
from world_action_model.contact_causal_checkpoint import (  # noqa: E402
    inspect_causal_checkpoint_bundle,
    load_safe_weight_mapping,
)
from world_action_model.contact_causal_production import (  # noqa: E402
    ACCEPTANCE_VERSION,
    CAUSAL_ROUTES,
    CausalProductionError,
    MATRIX_ID,
    current_git_head,
    current_head_source_tree_receipt,
    ensure_not_project_tmp,
    sha256_json,
    strict_json_file,
    validate_acceptance_v3,
    validate_deadline,
    validate_training_receipt_v5,
    named_tensor_digest,
    verify_checkpoint_file,
    verify_acceptance_bound_artifacts_v3,
    write_canonical_json_exclusive,
)


def _identity_set(*groups: object) -> set[str]:
    flattened: list[str] = []
    for group in groups:
        if isinstance(group, dict):
            flattened.extend(group.values())
        elif isinstance(group, (list, tuple)):
            flattened.extend(group)
        else:
            flattened.append(group)  # type: ignore[arg-type]
    if any(not isinstance(item, str) for item in flattened):
        raise CausalProductionError("review identity namespace is malformed")
    if len(flattened) != len(set(flattened)):
        raise CausalProductionError("review/training identity namespace overlaps")
    return set(flattened)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a causal checkpoint and emit external v3 acceptance."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--robot-gate", type=Path, required=True)
    parser.add_argument("--gate-reviewer-acceptance", type=Path, required=True)
    parser.add_argument("--route-training-acceptance", type=Path, required=True)
    parser.add_argument("--precursor-receipt", type=Path, required=True)
    parser.add_argument("--precursor-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hardened-preflight-receipt", type=Path, required=True)
    parser.add_argument("--production-preflight-receipt", type=Path, required=True)
    parser.add_argument("--training-data-receipt", type=Path, required=True)
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--t5-embedding-dir", type=Path, required=True)
    parser.add_argument("--matched-control-report", type=Path, required=True)
    for causal_route in CAUSAL_ROUTES:
        parser.add_argument(
            f"--trace-{causal_route.replace('_', '-')}", type=Path, required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-identity", required=True)
    parser.add_argument("--reviewer-attestation-nonce", required=True)
    parser.add_argument("--acceptance-run-id", required=True)
    parser.add_argument("--acceptance-run-nonce", required=True)
    parser.add_argument("--evidence-valid-until-utc", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()

    repository_root = args.repository_root.expanduser().absolute()
    checkpoint_dir = ensure_not_project_tmp(
        args.checkpoint_dir, project_root=repository_root
    )
    output = ensure_not_project_tmp(args.output, project_root=repository_root)
    gate_path = ensure_not_project_tmp(args.robot_gate, project_root=repository_root)
    review_path = ensure_not_project_tmp(
        args.gate_reviewer_acceptance, project_root=repository_root
    )
    if checkpoint_dir == output.parent or checkpoint_dir in output.parents:
        raise CausalProductionError(
            "checkpoint acceptance must be external to the checkpoint directory"
        )
    validate_deadline(args.evidence_valid_until_utc)

    gate, gate_sha256, _ = strict_json_file(gate_path, name="causal Robot v3 gate")
    validate_robot_gate_v3(gate, gate_path=gate_path, load_external_review=True)
    gate_review, gate_review_sha256, _ = strict_json_file(
        review_path, name="causal Robot v3 external review"
    )
    validate_robot_gate_review_v2(gate_review, gate=gate)
    review_entry = gate["evidence_reports"]["gate_reviewer_acceptance"]
    expected_review_path = Path(review_entry["path"])
    if not expected_review_path.is_absolute():
        expected_review_path = gate_path.parent / expected_review_path
    if (
        expected_review_path.absolute() != review_path
        or review_entry["sha256"] != gate_review_sha256
    ):
        raise CausalProductionError("explicit gate-reviewer file differs from gate")
    requested_expiry = datetime.fromisoformat(
        args.evidence_valid_until_utc[:-1] + "+00:00"
    )
    gate_expiry = datetime.fromisoformat(
        gate["evidence_valid_until_utc"][:-1] + "+00:00"
    )
    review_expiry = datetime.fromisoformat(
        gate_review["evidence_valid_until_utc"][:-1] + "+00:00"
    )
    if requested_expiry > min(gate_expiry, review_expiry):
        raise CausalProductionError(
            "checkpoint acceptance cannot outlive gate/reviewer evidence"
        )

    receipt_path = checkpoint_dir / "checkpoint_training_receipt.json"
    receipt, receipt_file_sha256, _ = strict_json_file(
        receipt_path, name="causal checkpoint training receipt v5"
    )
    validate_training_receipt_v5(receipt)
    if receipt_file_sha256 != sha256_json(receipt):
        raise CausalProductionError("training receipt bytes are not canonical/stable")
    verify_checkpoint_file(receipt)
    route = receipt["route"]["causal_route_id"]
    if (
        receipt["output_path_record_only"] != str(checkpoint_dir)
        or Path(receipt["checkpoint_file"]["path_record_only"]).parent
        != checkpoint_dir
        or receipt["evidence"]["training_gate_sha256"] != gate_sha256
        or receipt["evidence"]["external_reviewer_acceptance_sha256"]
        != gate_review_sha256
        or receipt["run_identity"]["training_run_id"]
        != gate["training_run_id_by_route"][route]
        or receipt["run_identity"]["training_run_nonce"]
        != gate["training_run_nonce_by_route"][route]
        or receipt["run_identity"]["external_acceptance_sha256"]
        != gate["external_acceptance_sha256_by_route"][route]
    ):
        raise CausalProductionError("checkpoint receipt differs from Robot gate")

    route_acceptance_path = args.route_training_acceptance.expanduser().absolute()
    route_acceptance, route_acceptance_sha256, _ = strict_json_file(
        route_acceptance_path, name="causal route training acceptance"
    )
    expected_route_acceptance_path = Path(
        gate["external_acceptance_path_by_route"][route]
    )
    if not expected_route_acceptance_path.is_absolute():
        expected_route_acceptance_path = gate_path.parent / expected_route_acceptance_path
    if (
        route_acceptance_path != expected_route_acceptance_path.absolute()
        or route_acceptance_sha256
        != gate["external_acceptance_sha256_by_route"][route]
    ):
        raise CausalProductionError("explicit route acceptance differs from gate")
    validate_robot_route_acceptance_v1(
        route_acceptance, gate=gate, route=route, repository_root=repository_root
    )

    precursor_path = args.precursor_receipt.expanduser().absolute()
    precursor_receipt, precursor_sha256, precursor_bytes = strict_json_file(
        precursor_path, name="Ego causal precursor receipt"
    )
    validate_precursor_receipt(precursor_receipt)
    precursor_dir = args.precursor_checkpoint_dir.expanduser().absolute()
    precursor_bundle = inspect_causal_checkpoint_bundle(precursor_dir)
    precursor_mapping, _ = load_safe_weight_mapping(
        precursor_dir / precursor_bundle["weight_file"]
    )
    precursor_parameter_names = precursor_receipt["parameter_partition"][
        "all_parameter_names"
    ]
    if (
        precursor_sha256 != receipt["precursor"]["receipt_bytes_sha256"]
        or precursor_bundle["bundle_manifest"]["bundle_sha256"]
        != receipt["precursor"]["bundle_sha256"]
        or not set(precursor_parameter_names).issubset(precursor_mapping)
        or named_tensor_digest(
            [(name, precursor_mapping[name]) for name in precursor_parameter_names]
        ) != receipt["precursor"]["parameter_tensor_sha256"]
    ):
        raise CausalProductionError("precursor receipt/bundle/tensor bytes differ")

    hardened_path = args.hardened_preflight_receipt.expanduser().absolute()
    hardened, hardened_sha256, _ = strict_json_file(
        hardened_path, name="hardened contact-asym25 v2 preflight receipt"
    )
    if (
        hardened_sha256 != receipt["evidence"]["hardened_preflight_receipt_sha256"]
        or hardened_sha256
        != gate["artifact_fingerprints"]["hardened_preflight_receipt_sha256"]
    ):
        raise CausalProductionError("hardened v2 preflight bytes differ")
    replay_persisted_hardened_contact_asym25_v2(hardened)
    runtime_sources = build_robot_runtime_source_receipts_v1(
        hardened_preflight_receipt=hardened,
        t5_embedding_dir=args.t5_embedding_dir,
    )
    if (
        runtime_sources["training_source_runtime_receipt_sha256"]
        != gate["artifact_fingerprints"]["dataset_runtime_source_receipt_sha256"]
        or runtime_sources["t5_runtime_source_receipt_sha256"]
        != gate["artifact_fingerprints"]["t5_runtime_source_receipt_sha256"]
        or receipt["stable_contract"]["dataset_runtime_source_receipt_sha256"]
        != runtime_sources["training_source_runtime_receipt_sha256"]
        or receipt["stable_contract"]["t5_runtime_source_receipt_sha256"]
        != runtime_sources["t5_runtime_source_receipt_sha256"]
    ):
        raise CausalProductionError("checkpoint runtime dataset/T5 bytes differ")

    training_data_path = args.training_data_receipt.expanduser().absolute()
    training_data, training_data_sha256, _ = strict_json_file(
        training_data_path, name="causal training-data receipt"
    )
    validate_training_data_receipt_v2(
        training_data, gate=gate, gate_sha256=gate_sha256
    )
    preflight_path = args.production_preflight_receipt.expanduser().absolute()
    preflight, preflight_sha256, _ = strict_json_file(
        preflight_path, name="causal production preflight receipt"
    )
    validate_preflight_receipt_v2(
        preflight, gate=gate, gate_sha256=gate_sha256,
        training_data_receipt=training_data,
    )
    coverage_path = args.coverage_receipt.expanduser().absolute()
    coverage, coverage_sha256, _ = strict_json_file(
        coverage_path, name="causal contact coverage receipt"
    )
    validate_recomputed_dataset_supervision_coverage(
        coverage,
        sidecar_dir=args.sidecar_dir,
        expected_meta_sha256=gate["artifact_fingerprints"]["sidecar_meta_sha256"],
        expected_episode_sha256=hardened["episode_npz_sha256"],
        dataset_id=hardened["dataset_id"], split_seed=hardened["split_seed"],
        split_episode_ids={
            "train": hardened["train_episode_ids"],
            "val": hardened["validation_episode_ids"],
            "test": hardened["heldout_episode_ids"],
        },
    )
    if (
        training_data_sha256 != receipt["evidence"]["training_data_receipt_sha256"]
        or preflight_sha256 != receipt["evidence"]["production_preflight_receipt_sha256"]
        or coverage_sha256 != receipt["evidence"]["contact_coverage_receipt_sha256"]
        or preflight["training_data_receipt_sha256"] != training_data_sha256
        or training_data["dataset_contact_coverage_receipt_sha256"] != coverage_sha256
    ):
        raise CausalProductionError("checkpoint data/preflight/coverage chain differs")

    replay_robot_gate_evidence_v3(gate, gate_path=gate_path)
    trace_paths = {
        causal_route: getattr(args, f"trace_{causal_route}").expanduser().absolute()
        for causal_route in CAUSAL_ROUTES
    }
    records = {
        causal_route: load_trace_jsonl(path, route=causal_route)
        for causal_route, path in trace_paths.items()
    }
    recomputed_report = compare_causal_records(
        records, required_seed=receipt["seed"]
    )
    report_path = args.matched_control_report.expanduser().absolute()
    report, report_sha256, _ = strict_json_file(
        report_path, name="current-run causal matched-control report"
    )
    validate_collector_report_v3(report)
    if report != recomputed_report:
        raise CausalProductionError("current-run matched-control report replay differs")
    trace_prefix = receipt["checkpoint_file"]["current_run_trace_prefix_receipt"]
    if (
        Path(trace_prefix["trace_path_record_only"]).absolute() != trace_paths[route]
        or replay_current_run_trace_prefix_v1(trace_prefix) != trace_prefix
    ):
        raise CausalProductionError("checkpoint current-run trace prefix differs")

    head = current_git_head(repository_root)
    if head != gate["expected_git_commit"] or head != receipt["stable_contract"]["git_commit"]:
        raise CausalProductionError("checkpoint review does not run at bound current HEAD")
    source_receipt = current_head_source_tree_receipt(
        repository_root, expected_commit=head
    )
    source_receipt_sha256 = sha256_json(source_receipt)
    if (
        source_receipt_sha256
        != gate["artifact_fingerprints"]["current_head_source_tree_sha256"]
        or source_receipt_sha256
        != receipt["evidence"]["current_head_source_receipt_sha256"]
        or receipt["stable_contract"]["source_tree_sha256"]
        != source_receipt_sha256
    ):
        raise CausalProductionError("current HEAD source receipt binding differs")
    _verify_head_code(
        repository_root=repository_root,
        expected_commit=head,
        expected_code_sha256=gate_review["reviewed_code_blob_sha256"],
        logical_paths=ROBOT_BOOTSTRAP_CODE_PATHS,
    )

    new_identities = [
        args.reviewer_identity,
        args.reviewer_attestation_nonce,
        args.acceptance_run_id,
        args.acceptance_run_nonce,
    ]
    route_reviewer_identities = []
    for route_name in CAUSAL_ROUTES:
        route_path = Path(gate["external_acceptance_path_by_route"][route_name])
        if not route_path.is_absolute():
            route_path = gate_path.parent / route_path
        route_value, route_digest, _ = strict_json_file(
            route_path, name=f"causal route acceptance identity {route_name}"
        )
        if route_digest != gate["external_acceptance_sha256_by_route"][route_name]:
            raise CausalProductionError("route acceptance identity bytes differ")
        validate_robot_route_acceptance_v1(
            route_value, gate=gate, route=route_name, repository_root=repository_root
        )
        route_reviewer_identities.extend(
            route_value[name]
            for name in (
                "reviewer_identity", "reviewer_attestation_nonce",
                "acceptance_run_id", "acceptance_run_nonce",
            )
        )
    _identity_set(
        new_identities,
        gate["training_run_id_by_route"],
        gate["training_run_nonce_by_route"],
        [
            gate_review["reviewer_identity"],
            gate_review["reviewer_attestation_nonce"],
            gate_review["acceptance_run_id"],
            gate_review["acceptance_run_nonce"],
        ],
        route_reviewer_identities,
    )
    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    acceptance = {
        "acceptance_version": ACCEPTANCE_VERSION,
        "status": "accepted",
        "accepted": True,
        "matrix_id": MATRIX_ID,
        "causal_route_id": route,
        "seed": receipt["seed"],
        "checkpoint_step": receipt["checkpoint_step"],
        "training_receipt_sha256": receipt_file_sha256,
        "checkpoint_file_sha256": receipt["checkpoint_file"]["sha256"],
        "checkpoint_tensor_manifest_sha256": receipt["checkpoint_file"][
            "tensor_manifest_sha256"
        ],
        "checkpoint_path_record_only": receipt["checkpoint_file"][
            "path_record_only"
        ],
        "robot_gate_path_record_only": str(gate_path),
        "robot_gate_sha256": gate_sha256,
        "gate_reviewer_acceptance_path_record_only": str(review_path),
        "gate_reviewer_acceptance_sha256": gate_review_sha256,
        "route_training_acceptance_path_record_only": str(route_acceptance_path),
        "route_training_acceptance_sha256": route_acceptance_sha256,
        "precursor_receipt_path_record_only": str(precursor_path),
        "precursor_receipt_bytes_sha256": precursor_sha256,
        "precursor_checkpoint_dir_record_only": str(precursor_dir),
        "precursor_bundle_sha256": precursor_bundle["bundle_manifest"][
            "bundle_sha256"
        ],
        "hardened_preflight_path_record_only": str(hardened_path),
        "hardened_preflight_receipt_sha256": hardened_sha256,
        "production_preflight_path_record_only": str(preflight_path),
        "production_preflight_receipt_sha256": preflight_sha256,
        "training_data_path_record_only": str(training_data_path),
        "training_data_receipt_sha256": training_data_sha256,
        "coverage_path_record_only": str(coverage_path),
        "contact_coverage_receipt_sha256": coverage_sha256,
        "sidecar_dir_record_only": str(args.sidecar_dir.expanduser().absolute()),
        "t5_embedding_dir_record_only": str(
            args.t5_embedding_dir.expanduser().absolute()
        ),
        "matched_control_report_path_record_only": str(report_path),
        "matched_control_report_sha256": report_sha256,
        "matched_control_trace_manifest_sha256": report[
            "trace_manifest_sha256"
        ],
        "matched_control_trace_path_by_route": {
            causal_route: str(path) for causal_route, path in trace_paths.items()
        },
        "current_run_trace_prefix_receipt_sha256": trace_prefix[
            "receipt_sha256"
        ],
        "current_head_source_receipt_sha256": source_receipt_sha256,
        "reviewer_identity": args.reviewer_identity,
        "reviewer_attestation_nonce": args.reviewer_attestation_nonce,
        "acceptance_run_id": args.acceptance_run_id,
        "acceptance_run_nonce": args.acceptance_run_nonce,
        "reviewed_current_head_sha256": head,
        "reviewed_source_tree_sha256": source_receipt_sha256,
        "reviewed_at_utc": reviewed_at,
        "evidence_valid_until_utc": args.evidence_valid_until_utc,
        "oracle_used": False,
    }
    validate_acceptance_v3(acceptance, receipt=receipt)
    verify_acceptance_bound_artifacts_v3(
        acceptance, receipt=receipt, repository_root=repository_root
    )
    digest = write_canonical_json_exclusive(output, acceptance)
    print(f"{digest}  {output}")


if __name__ == "__main__":
    main()

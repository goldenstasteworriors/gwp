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
)
from world_action_model.contact_causal_preflight import (  # noqa: E402
    validate_robot_gate_review_v2,
    validate_robot_gate_v3,
)
from world_action_model.contact_causal_production import (  # noqa: E402
    ACCEPTANCE_VERSION,
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
    verify_checkpoint_file,
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
        "robot_gate_sha256": gate_sha256,
        "gate_reviewer_acceptance_sha256": gate_review_sha256,
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
    digest = write_canonical_json_exclusive(output, acceptance)
    print(f"{digest}  {output}")


if __name__ == "__main__":
    main()

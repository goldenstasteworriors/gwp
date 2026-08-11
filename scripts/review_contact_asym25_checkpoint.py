#!/usr/bin/env python3
"""Create an external reviewer acceptance for one production checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_asym25_checkpoint_acceptance import (  # noqa: E402
    ACCEPTANCE_VERSION,
    CheckpointAcceptanceError,
    strict_json_file,
    validate_acceptance,
    validate_bundle_manifest,
    validate_training_receipt,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently rehash one checkpoint and emit reviewer-authored "
            "acceptance. The output must be outside the checkpoint directory."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-identity", required=True)
    parser.add_argument("--reviewer-attestation-nonce", required=True)
    parser.add_argument("--acceptance-run-id", required=True)
    parser.add_argument("--acceptance-run-nonce", required=True)
    parser.add_argument(
        "--current-run-matched-report",
        type=Path,
        required=True,
        help="Passed report produced from all current-run matched variant traces.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint_dir.expanduser().absolute()
    output = args.output.expanduser().absolute()
    if output == Path("/tmp") or Path("/tmp") in output.parents:
        raise CheckpointAcceptanceError("reviewer acceptance must not use /tmp")
    if checkpoint == output.parent or checkpoint in output.parents:
        raise CheckpointAcceptanceError(
            "reviewer acceptance must be external to the checkpoint directory"
        )
    bundle, bundle_file_sha = strict_json_file(
        checkpoint / "checkpoint_bundle_manifest.json", "checkpoint bundle manifest"
    )
    validate_bundle_manifest(bundle, checkpoint)
    receipt, receipt_sha = strict_json_file(
        checkpoint / "checkpoint_training_receipt.json", "checkpoint training receipt"
    )
    validate_training_receipt(receipt)
    if (
        receipt["checkpoint_bundle_manifest_sha256"] != bundle_file_sha
        or receipt["checkpoint_bundle_sha256"] != bundle["bundle_sha256"]
    ):
        raise CheckpointAcceptanceError("training receipt bundle binding differs")
    matched_report_path = args.current_run_matched_report.expanduser().absolute()
    if checkpoint == matched_report_path.parent or checkpoint in matched_report_path.parents:
        raise CheckpointAcceptanceError(
            "current-run matched report must be external to checkpoint"
        )
    _, matched_report_sha256 = strict_json_file(
        matched_report_path, "current-run matched-control report"
    )
    acceptance = {
        "acceptance_version": ACCEPTANCE_VERSION,
        "status": "accepted",
        "reviewer_identity": args.reviewer_identity,
        "reviewer_attestation_nonce": args.reviewer_attestation_nonce,
        "acceptance_run_id": args.acceptance_run_id,
        "acceptance_run_nonce": args.acceptance_run_nonce,
        "checkpoint_path_record_only": str(checkpoint),
        "checkpoint_step": receipt["checkpoint_step"],
        "split_seed": receipt["split_seed"],
        "training_run_nonce": receipt["training_run_nonce"],
        "training_run_id": receipt["training_run_id"],
        "current_run_matched_report_path_record_only": str(matched_report_path),
        "current_run_matched_report_sha256": matched_report_sha256,
        "checkpoint_bundle_manifest_sha256": bundle_file_sha,
        "checkpoint_bundle_sha256": bundle["bundle_sha256"],
        "checkpoint_training_receipt_sha256": receipt_sha,
    }
    validate_acceptance(
        acceptance,
        checkpoint_dir=checkpoint,
        bundle_manifest=bundle,
        bundle_manifest_sha256=bundle_file_sha,
        training_receipt=receipt,
        training_receipt_sha256=receipt_sha,
    )
    digest = write_json_exclusive(output, acceptance)
    print(f"{digest}  {output}")


if __name__ == "__main__":
    main()

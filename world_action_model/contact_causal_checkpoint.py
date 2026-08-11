"""Checkpoint manifest/acceptance v3 helpers for causal-main Stage2 runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from world_action_model.contact_causal_production import (
    ACCEPTANCE_VERSION,
    CHECKPOINT_MANIFEST_VERSION,
    CHECKPOINT_STEPS,
    CausalProductionError,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    PRECURSOR_ID,
    TRAINING_RECEIPT_VERSION,
    ensure_exact_causal_route,
    sha256_json,
    strict_json_file,
    validate_acceptance_v3,
    validate_training_receipt_v5,
    verify_checkpoint_file,
)


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def validate_checkpoint_manifest_v3(
    value: object, *, verify_files: bool = True
) -> tuple[dict[str, Any], dict[int, tuple[dict[str, Any], dict[str, Any]]]]:
    manifest = _exact(
        value,
        {
            "manifest_version",
            "matrix_schema_version",
            "matrix_id",
            "precursor_id",
            "precursor_receipt_bytes_sha256",
            "precursor_bundle_sha256",
            "seed",
            "causal_route_id",
            "training_run_id",
            "training_run_nonce",
            "primary_checkpoint_step",
            "best_checkpoint_selection_allowed",
            "checkpoints",
        },
        "causal checkpoint manifest",
    )
    route = ensure_exact_causal_route(manifest["causal_route_id"])
    if (
        manifest["manifest_version"] != CHECKPOINT_MANIFEST_VERSION
        or manifest["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or manifest["matrix_id"] != MATRIX_ID
        or manifest["precursor_id"] != PRECURSOR_ID
        or manifest["primary_checkpoint_step"] != 200
        or manifest["best_checkpoint_selection_allowed"] is not False
    ):
        raise CausalProductionError("causal checkpoint manifest identity differs")
    _sha(manifest["precursor_receipt_bytes_sha256"], "manifest precursor receipt")
    _sha(manifest["precursor_bundle_sha256"], "manifest precursor bundle")
    entries = manifest["checkpoints"]
    if not isinstance(entries, list) or len(entries) != 3:
        raise CausalProductionError("causal checkpoint manifest needs step 0/100/200")
    accepted: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw_entry in entries:
        entry = _exact(
            raw_entry,
            {
                "step",
                "training_receipt_path",
                "training_receipt_sha256",
                "independent_acceptance_path",
                "independent_acceptance_sha256",
                "checkpoint_file_path",
                "checkpoint_file_sha256",
            },
            "causal checkpoint manifest entry",
        )
        step = entry["step"]
        if isinstance(step, bool) or step not in CHECKPOINT_STEPS or step in accepted:
            raise CausalProductionError("causal checkpoint step is invalid/duplicate")
        for name in (
            "training_receipt_path",
            "independent_acceptance_path",
            "checkpoint_file_path",
        ):
            if not isinstance(entry[name], str) or not Path(entry[name]).is_absolute():
                raise CausalProductionError("causal checkpoint evidence paths must be absolute")
        receipt, receipt_digest, _ = strict_json_file(
            entry["training_receipt_path"], name=f"causal training receipt step {step}"
        )
        validate_training_receipt_v5(receipt)
        acceptance, acceptance_digest, _ = strict_json_file(
            entry["independent_acceptance_path"],
            name=f"causal independent acceptance step {step}",
        )
        validate_acceptance_v3(acceptance, receipt=receipt)
        if (
            receipt_digest != entry["training_receipt_sha256"]
            or acceptance_digest != entry["independent_acceptance_sha256"]
            or receipt["checkpoint_file"]["path_record_only"]
            != entry["checkpoint_file_path"]
            or receipt["checkpoint_file"]["sha256"]
            != entry["checkpoint_file_sha256"]
            or receipt["checkpoint_step"] != step
            or receipt["seed"] != manifest["seed"]
            or receipt["route"]["causal_route_id"] != route
            or receipt["run_identity"]["training_run_id"]
            != manifest["training_run_id"]
            or receipt["run_identity"]["training_run_nonce"]
            != manifest["training_run_nonce"]
            or receipt["precursor"]["receipt_bytes_sha256"]
            != manifest["precursor_receipt_bytes_sha256"]
            or receipt["precursor"]["bundle_sha256"]
            != manifest["precursor_bundle_sha256"]
        ):
            raise CausalProductionError("causal checkpoint manifest entry binding differs")
        if verify_files:
            verify_checkpoint_file(receipt)
        accepted[step] = (receipt, acceptance)
    if set(accepted) != set(CHECKPOINT_STEPS):
        raise CausalProductionError("causal checkpoint step set differs")
    rows = [accepted[step][0] for step in CHECKPOINT_STEPS]
    if (
        rows[0]["parent_checkpoint_weight_sha256"]
        != rows[0]["precursor"]["parameter_tensor_sha256"]
        or rows[1]["parent_checkpoint_weight_sha256"]
        != rows[0]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]
        or rows[2]["parent_checkpoint_weight_sha256"]
        != rows[1]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]
    ):
        raise CausalProductionError("causal checkpoint parent chain differs")
    return manifest, accepted


def build_checkpoint_manifest_v3(
    *,
    receipts_and_acceptances: Mapping[
        int, tuple[str | Path, Mapping[str, Any], str | Path, Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    if set(receipts_and_acceptances) != set(CHECKPOINT_STEPS):
        raise CausalProductionError("checkpoint manifest builder needs step 0/100/200")
    first = receipts_and_acceptances[0][1]
    validate_training_receipt_v5(first)
    entries = []
    for step in CHECKPOINT_STEPS:
        receipt_path, receipt, acceptance_path, acceptance = receipts_and_acceptances[step]
        validate_training_receipt_v5(receipt)
        validate_acceptance_v3(acceptance, receipt=receipt)
        entries.append(
            {
                "step": step,
                "training_receipt_path": str(Path(receipt_path).absolute()),
                "training_receipt_sha256": sha256_json(receipt),
                "independent_acceptance_path": str(Path(acceptance_path).absolute()),
                "independent_acceptance_sha256": sha256_json(acceptance),
                "checkpoint_file_path": receipt["checkpoint_file"]["path_record_only"],
                "checkpoint_file_sha256": receipt["checkpoint_file"]["sha256"],
            }
        )
    return {
        "manifest_version": CHECKPOINT_MANIFEST_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "precursor_id": PRECURSOR_ID,
        "precursor_receipt_bytes_sha256": first["precursor"]["receipt_bytes_sha256"],
        "precursor_bundle_sha256": first["precursor"]["bundle_sha256"],
        "seed": first["seed"],
        "causal_route_id": first["route"]["causal_route_id"],
        "training_run_id": first["run_identity"]["training_run_id"],
        "training_run_nonce": first["run_identity"]["training_run_nonce"],
        "primary_checkpoint_step": 200,
        "best_checkpoint_selection_allowed": False,
        "checkpoints": entries,
    }

"""Robot Stage2 causal-main gate v3 and preflight/data receipt contracts.

The historical Robot gate v2 remains owned by
``contact_asym25_preflight.py``.  This module accepts only the causal-main
matrix and never auto-detects or translates a legacy gate.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
from typing import Any

from world_action_model.contact_causal_production import (
    CAUSAL_ROUTES,
    COLLECTOR_VERSION,
    CausalProductionError,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    PREFLIGHT_RECEIPT_VERSION,
    PRECURSOR_ID,
    ROBOT_GATE_REVIEW_VERSION,
    ROBOT_GATE_VERSION,
    SEEDS,
    SUPERVISION_VERSION,
    TRAINING_DATA_RECEIPT_VERSION,
    canonical_json_bytes,
    ensure_exact_causal_route,
    sha256_json,
    strict_json_file,
    validate_dataset_supervision_coverage,
    validate_deadline,
)
from world_action_model.verified_io import hash_verified_regular_file


REQUIRED_GATE_FLAGS = (
    "human_adjudicated_robot_sidecar_passed",
    "dual_annotation_agreement_passed",
    "group_metadata_complete",
    "asymmetric25_real_target_data_passed",
    "hardware_contract_verified",
    "train_only_norm_verified",
    "dataset_positive_contact_coverage_passed",
    "oracle_sufficiency_passed",
    "precursor_byte_identity_passed",
    "matched_control_hash_collector_passed",
    "current_head_source_binding_passed",
    "allow_server_training",
    "allow_5b_training",
)
FINGERPRINT_KEYS = (
    "matrix_manifest_sha256",
    "current_head_source_tree_sha256",
    "precursor_receipt_bytes_sha256",
    "precursor_bundle_sha256",
    "hardware_contract_sha256",
    "calibration_capture_sha256",
    "dataset_tree_sha256",
    "dataset_info_sha256",
    "episode_metadata_sha256",
    "train_only_norm_sha256",
    "sidecar_meta_sha256",
    "sidecar_bundle_sha256",
    "split_manifest_sha256",
    "coverage_csv_sha256",
    "segments_csv_sha256",
    "inventory_csv_sha256",
    "build_config_sha256",
    "dataset_contact_coverage_receipt_sha256",
    "oracle_sufficiency_receipt_sha256",
    "matched_control_report_sha256",
    "matched_control_trace_manifest_sha256",
    "route_contract_bundle_sha256",
)
EVIDENCE_REPORT_KEYS = (
    "annotation_agreement",
    "hardware_verification",
    "dataset_contact_coverage",
    "oracle_sufficiency",
    "ego_precursor",
    "matched_control_hash_collector",
    "gate_reviewer_acceptance",
)


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def _identity(value: object, name: str) -> str:
    import re

    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", value
    ):
        raise CausalProductionError(f"{name} must be a 16-128 character identity")
    return value


def _validate_identity_maps(gate: Mapping[str, Any]) -> None:
    identities: set[str] = set()
    for name in ("training_run_id_by_route", "training_run_nonce_by_route"):
        mapping = gate[name]
        if not isinstance(mapping, dict) or list(mapping) != list(CAUSAL_ROUTES):
            raise CausalProductionError(f"{name} route set/order differs")
        for route, value in mapping.items():
            _identity(value, f"{name} {route}")
            if value in identities:
                raise CausalProductionError("training IDs/nonces must be globally unique")
            identities.add(value)
    acceptances = gate["external_acceptance_sha256_by_route"]
    if not isinstance(acceptances, dict) or list(acceptances) != list(CAUSAL_ROUTES):
        raise CausalProductionError("external acceptance route set/order differs")
    for route, value in acceptances.items():
        _sha(value, f"external acceptance {route}")
    if len(set(acceptances.values())) != len(CAUSAL_ROUTES):
        raise CausalProductionError("external acceptance digests must be route-unique")


def validate_robot_gate_review_v2(value: object, *, gate: Mapping[str, Any]) -> dict[str, Any]:
    validate_robot_gate_v3(gate, load_external_review=False)
    review = _exact(
        value,
        {
            "acceptance_version",
            "status",
            "accepted",
            "matrix_schema_version",
            "matrix_id",
            "route_set",
            "reviewer_identity",
            "reviewer_attestation_nonce",
            "acceptance_run_id",
            "acceptance_run_nonce",
            "evidence_valid_until_utc",
            "reviewed_gate_contract_sha256",
            "current_head_source_tree_sha256",
            "hardware_contract_sha256",
            "dataset_tree_sha256",
            "train_only_norm_sha256",
            "sidecar_bundle_sha256",
            "precursor_receipt_bytes_sha256",
            "matched_control_report_sha256",
            "oracle_sufficiency_receipt_sha256",
            "oracle_is_main_matrix_route",
        },
        "causal Robot gate review",
    )
    gate_without_review = dict(gate)
    reports = dict(gate_without_review["evidence_reports"])
    reports["gate_reviewer_acceptance"] = {"path": None, "sha256": None}
    gate_without_review["evidence_reports"] = reports
    expected_gate_digest = sha256_json(gate_without_review)
    if (
        review["acceptance_version"] != ROBOT_GATE_REVIEW_VERSION
        or review["status"] != "accepted"
        or review["accepted"] is not True
        or review["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or review["matrix_id"] != MATRIX_ID
        or review["route_set"] != list(CAUSAL_ROUTES)
        or review["reviewed_gate_contract_sha256"] != expected_gate_digest
        or review["current_head_source_tree_sha256"]
        != gate["artifact_fingerprints"]["current_head_source_tree_sha256"]
        or review["hardware_contract_sha256"]
        != gate["artifact_fingerprints"]["hardware_contract_sha256"]
        or review["dataset_tree_sha256"]
        != gate["artifact_fingerprints"]["dataset_tree_sha256"]
        or review["train_only_norm_sha256"]
        != gate["artifact_fingerprints"]["train_only_norm_sha256"]
        or review["sidecar_bundle_sha256"]
        != gate["artifact_fingerprints"]["sidecar_bundle_sha256"]
        or review["precursor_receipt_bytes_sha256"]
        != gate["artifact_fingerprints"]["precursor_receipt_bytes_sha256"]
        or review["matched_control_report_sha256"]
        != gate["artifact_fingerprints"]["matched_control_report_sha256"]
        or review["oracle_sufficiency_receipt_sha256"]
        != gate["artifact_fingerprints"]["oracle_sufficiency_receipt_sha256"]
        or review["oracle_is_main_matrix_route"] is not False
    ):
        raise CausalProductionError("causal Robot gate review binding differs")
    review_ids = [
        review["reviewer_identity"],
        review["reviewer_attestation_nonce"],
        review["acceptance_run_id"],
        review["acceptance_run_nonce"],
    ]
    for index, item in enumerate(review_ids):
        _identity(item, f"gate review identity {index}")
    gate_ids = set(gate["training_run_id_by_route"].values()) | set(
        gate["training_run_nonce_by_route"].values()
    )
    if len(set(review_ids)) != 4 or set(review_ids) & gate_ids:
        raise CausalProductionError("gate reviewer identity namespace overlaps training")
    validate_deadline(review["evidence_valid_until_utc"])
    return review


def validate_robot_gate_v3(
    value: object,
    *,
    gate_path: str | Path | None = None,
    load_external_review: bool = True,
) -> dict[str, Any]:
    gate = _exact(
        value,
        {
            "gate_version",
            "matrix_schema_version",
            "matrix_id",
            "precursor_id",
            "route_set",
            "collector_version",
            "supervision_version",
            "expected_git_commit",
            "evidence_valid_until_utc",
            *REQUIRED_GATE_FLAGS,
            "artifact_fingerprints",
            "evidence_reports",
            "training_run_id_by_route",
            "training_run_nonce_by_route",
            "external_acceptance_sha256_by_route",
            "reviewed_by",
            "reviewed_at_utc",
            "note",
        },
        "causal Robot training gate",
    )
    if (
        gate["gate_version"] != ROBOT_GATE_VERSION
        or gate["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or gate["matrix_id"] != MATRIX_ID
        or gate["precursor_id"] != PRECURSOR_ID
        or gate["route_set"] != list(CAUSAL_ROUTES)
        or gate["collector_version"] != COLLECTOR_VERSION
        or gate["supervision_version"] != SUPERVISION_VERSION
    ):
        raise CausalProductionError("causal Robot gate identity differs")
    import re

    if not isinstance(gate["expected_git_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", gate["expected_git_commit"]
    ):
        raise CausalProductionError("causal Robot gate requires a full current HEAD")
    failed = [name for name in REQUIRED_GATE_FLAGS if gate[name] is not True]
    if failed:
        raise CausalProductionError(f"causal Robot gate flags are not explicitly true: {failed}")
    validate_deadline(gate["evidence_valid_until_utc"])
    _identity(gate["reviewed_by"], "causal Robot gate reviewer")
    if not isinstance(gate["reviewed_at_utc"], str) or not gate["reviewed_at_utc"].endswith("Z"):
        raise CausalProductionError("causal Robot reviewed_at_utc must be UTC")
    _validate_identity_maps(gate)
    fingerprints = _exact(
        gate["artifact_fingerprints"], set(FINGERPRINT_KEYS), "causal Robot fingerprints"
    )
    for name in FINGERPRINT_KEYS:
        _sha(fingerprints[name], f"causal Robot fingerprint {name}")
    reports = _exact(gate["evidence_reports"], set(EVIDENCE_REPORT_KEYS), "causal Robot evidence")
    for name, entry in reports.items():
        entry = _exact(entry, {"path", "sha256"}, f"causal Robot evidence {name}")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise CausalProductionError(f"causal Robot evidence path missing: {name}")
        _sha(entry["sha256"], f"causal Robot evidence {name}")
    if reports["matched_control_hash_collector"]["sha256"] != fingerprints["matched_control_report_sha256"]:
        raise CausalProductionError("gate matched-control report mirror differs")
    if reports["oracle_sufficiency"]["sha256"] != fingerprints["oracle_sufficiency_receipt_sha256"]:
        raise CausalProductionError("gate oracle receipt mirror differs")
    if reports["dataset_contact_coverage"]["sha256"] != fingerprints["dataset_contact_coverage_receipt_sha256"]:
        raise CausalProductionError("gate dataset coverage mirror differs")
    if load_external_review:
        review_entry = reports["gate_reviewer_acceptance"]
        review_path = Path(review_entry["path"])
        if not review_path.is_absolute():
            if gate_path is None:
                raise CausalProductionError("relative gate review path requires gate path")
            review_path = Path(gate_path).absolute().parent / review_path
        review, digest, _ = strict_json_file(review_path, name="causal Robot gate review")
        if digest != review_entry["sha256"]:
            raise CausalProductionError("causal Robot gate review bytes changed")
        validate_robot_gate_review_v2(review, gate=gate)
    return gate


def build_training_data_receipt_v2(
    *,
    gate: Mapping[str, Any],
    gate_sha256: str,
    seed: int,
    dataset_id: str,
    train_episode_ids: list[int],
    validation_episode_ids: list[int],
    heldout_episode_ids: list[int],
    ordered_window_manifest_sha256: str,
    coverage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validate_robot_gate_v3(gate, load_external_review=False)
    validate_dataset_supervision_coverage(coverage_receipt)
    if seed not in SEEDS or coverage_receipt["split_seed"] != seed:
        raise CausalProductionError("training-data receipt seed differs")
    episode_sets = [train_episode_ids, validation_episode_ids, heldout_episode_ids]
    if any(
        not isinstance(values, list)
        or not values
        or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
        or len(set(values)) != len(values)
        for values in episode_sets
    ):
        raise CausalProductionError("training-data episode sets must be non-empty integer lists")
    if set(train_episode_ids) & set(validation_episode_ids) or set(train_episode_ids) & set(heldout_episode_ids) or set(validation_episode_ids) & set(heldout_episode_ids):
        raise CausalProductionError("training-data split sets overlap")
    for value, name in (
        (gate_sha256, "training gate"),
        (ordered_window_manifest_sha256, "ordered window manifest"),
    ):
        _sha(value, name)
    fingerprints = gate["artifact_fingerprints"]
    receipt = {
        "receipt_version": TRAINING_DATA_RECEIPT_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "dataset_id": dataset_id,
        "hardware_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "split_seed": seed,
        "train_episode_ids": train_episode_ids,
        "validation_episode_ids": validation_episode_ids,
        "heldout_episode_ids": heldout_episode_ids,
        "ordered_window_manifest_sha256": ordered_window_manifest_sha256,
        "dataset_tree_sha256": fingerprints["dataset_tree_sha256"],
        "dataset_info_sha256": fingerprints["dataset_info_sha256"],
        "episode_metadata_sha256": fingerprints["episode_metadata_sha256"],
        "hardware_contract_sha256": fingerprints["hardware_contract_sha256"],
        "calibration_capture_sha256": fingerprints["calibration_capture_sha256"],
        "train_only_norm_sha256": fingerprints["train_only_norm_sha256"],
        "sidecar_bundle_sha256": fingerprints["sidecar_bundle_sha256"],
        "split_manifest_sha256": fingerprints["split_manifest_sha256"],
        "dataset_contact_coverage_receipt": dict(coverage_receipt),
        "dataset_contact_coverage_receipt_sha256": sha256_json(coverage_receipt),
        "training_gate_sha256": gate_sha256,
        "precursor_receipt_bytes_sha256": fingerprints["precursor_receipt_bytes_sha256"],
        "precursor_bundle_sha256": fingerprints["precursor_bundle_sha256"],
        "human_adjudicated_robot_native_labels": True,
        "hand_active_used_as_contact_ground_truth": False,
        "grip_used_as_contact_ground_truth": False,
        "train_only_normalization": True,
    }
    validate_training_data_receipt_v2(receipt, gate=gate, gate_sha256=gate_sha256)
    return receipt


def validate_training_data_receipt_v2(
    value: object,
    *,
    gate: Mapping[str, Any] | None = None,
    gate_sha256: str | None = None,
) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "receipt_version",
            "matrix_schema_version",
            "matrix_id",
            "dataset_id",
            "hardware_schema_id",
            "split_seed",
            "train_episode_ids",
            "validation_episode_ids",
            "heldout_episode_ids",
            "ordered_window_manifest_sha256",
            "dataset_tree_sha256",
            "dataset_info_sha256",
            "episode_metadata_sha256",
            "hardware_contract_sha256",
            "calibration_capture_sha256",
            "train_only_norm_sha256",
            "sidecar_bundle_sha256",
            "split_manifest_sha256",
            "dataset_contact_coverage_receipt",
            "dataset_contact_coverage_receipt_sha256",
            "training_gate_sha256",
            "precursor_receipt_bytes_sha256",
            "precursor_bundle_sha256",
            "human_adjudicated_robot_native_labels",
            "hand_active_used_as_contact_ground_truth",
            "grip_used_as_contact_ground_truth",
            "train_only_normalization",
        },
        "causal Robot training-data receipt",
    )
    if (
        receipt["receipt_version"] != TRAINING_DATA_RECEIPT_VERSION
        or receipt["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or receipt["matrix_id"] != MATRIX_ID
        or receipt["hardware_schema_id"]
        != "sonic_g1_rh56dftp6_omnipicker1_v1"
        or receipt["split_seed"] not in SEEDS
        or isinstance(receipt["split_seed"], bool)
        or not isinstance(receipt["dataset_id"], str)
        or not receipt["dataset_id"]
        or receipt["human_adjudicated_robot_native_labels"] is not True
        or receipt["hand_active_used_as_contact_ground_truth"] is not False
        or receipt["grip_used_as_contact_ground_truth"] is not False
        or receipt["train_only_normalization"] is not True
    ):
        raise CausalProductionError("causal Robot training-data identity differs")
    episode_sets = [
        receipt["train_episode_ids"],
        receipt["validation_episode_ids"],
        receipt["heldout_episode_ids"],
    ]
    if any(
        not isinstance(values, list)
        or not values
        or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
        or len(set(values)) != len(values)
        for values in episode_sets
    ):
        raise CausalProductionError("training-data episode sets are invalid")
    if (
        set(episode_sets[0]) & set(episode_sets[1])
        or set(episode_sets[0]) & set(episode_sets[2])
        or set(episode_sets[1]) & set(episode_sets[2])
    ):
        raise CausalProductionError("training-data split sets overlap")
    for name in (
        "ordered_window_manifest_sha256",
        "dataset_tree_sha256",
        "dataset_info_sha256",
        "episode_metadata_sha256",
        "hardware_contract_sha256",
        "calibration_capture_sha256",
        "train_only_norm_sha256",
        "sidecar_bundle_sha256",
        "split_manifest_sha256",
        "dataset_contact_coverage_receipt_sha256",
        "training_gate_sha256",
        "precursor_receipt_bytes_sha256",
        "precursor_bundle_sha256",
    ):
        _sha(receipt[name], f"causal training-data {name}")
    coverage = validate_dataset_supervision_coverage(
        receipt["dataset_contact_coverage_receipt"]
    )
    if (
        sha256_json(coverage)
        != receipt["dataset_contact_coverage_receipt_sha256"]
        or coverage["split_seed"] != receipt["split_seed"]
    ):
        raise CausalProductionError("training-data contact coverage binding differs")
    if (gate is None) != (gate_sha256 is None):
        raise CausalProductionError("gate and gate digest must be supplied together")
    if gate is not None:
        validate_robot_gate_v3(gate, load_external_review=False)
        _sha(gate_sha256, "causal training-data gate")
        fingerprints = gate["artifact_fingerprints"]
        expected = {
            "dataset_tree_sha256": fingerprints["dataset_tree_sha256"],
            "dataset_info_sha256": fingerprints["dataset_info_sha256"],
            "episode_metadata_sha256": fingerprints["episode_metadata_sha256"],
            "hardware_contract_sha256": fingerprints["hardware_contract_sha256"],
            "calibration_capture_sha256": fingerprints["calibration_capture_sha256"],
            "train_only_norm_sha256": fingerprints["train_only_norm_sha256"],
            "sidecar_bundle_sha256": fingerprints["sidecar_bundle_sha256"],
            "split_manifest_sha256": fingerprints["split_manifest_sha256"],
            "dataset_contact_coverage_receipt_sha256": fingerprints[
                "dataset_contact_coverage_receipt_sha256"
            ],
            "training_gate_sha256": gate_sha256,
            "precursor_receipt_bytes_sha256": fingerprints[
                "precursor_receipt_bytes_sha256"
            ],
            "precursor_bundle_sha256": fingerprints["precursor_bundle_sha256"],
        }
        if any(receipt[name] != expected_value for name, expected_value in expected.items()):
            raise CausalProductionError("training-data receipt differs from gate evidence")
    return receipt


def build_preflight_receipt_v2(
    *,
    gate: Mapping[str, Any],
    gate_sha256: str,
    route: str,
    training_data_receipt: Mapping[str, Any],
    current_head: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    validate_robot_gate_v3(gate, load_external_review=False)
    route = ensure_exact_causal_route(route)
    _sha(gate_sha256, "training gate")
    _sha(source_tree_sha256, "current source tree")
    import re

    if not isinstance(current_head, str) or not re.fullmatch(r"[0-9a-f]{40}", current_head):
        raise CausalProductionError("preflight must bind a full current HEAD")
    if current_head != gate["expected_git_commit"]:
        raise CausalProductionError("preflight current HEAD differs from gate")
    if source_tree_sha256 != gate["artifact_fingerprints"]["current_head_source_tree_sha256"]:
        raise CausalProductionError("preflight source tree differs from gate")
    if training_data_receipt.get("receipt_version") != TRAINING_DATA_RECEIPT_VERSION:
        raise CausalProductionError("preflight training-data receipt version differs")
    if training_data_receipt.get("training_gate_sha256") != gate_sha256:
        raise CausalProductionError("preflight training-data gate binding differs")
    receipt = {
        "receipt_version": PREFLIGHT_RECEIPT_VERSION,
        "status": "artifact_consistency_passed_with_explicit_external_gates",
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "causal_route_id": route,
        "route_set": list(CAUSAL_ROUTES),
        "precursor_id": PRECURSOR_ID,
        "git_commit": current_head,
        "current_head_source_tree_sha256": source_tree_sha256,
        "training_gate_sha256": gate_sha256,
        "training_data_receipt": dict(training_data_receipt),
        "training_data_receipt_sha256": sha256_json(training_data_receipt),
        "matched_control_report_sha256": gate["artifact_fingerprints"]["matched_control_report_sha256"],
        "oracle_sufficiency_receipt_sha256": gate["artifact_fingerprints"]["oracle_sufficiency_receipt_sha256"],
        "contact_condition_mode": "predicted",
        "oracle_used": False,
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }
    validate_preflight_receipt_v2(
        receipt,
        gate=gate,
        gate_sha256=gate_sha256,
        training_data_receipt=training_data_receipt,
    )
    return receipt


def validate_preflight_receipt_v2(
    value: object,
    *,
    gate: Mapping[str, Any] | None = None,
    gate_sha256: str | None = None,
    training_data_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "receipt_version",
            "status",
            "matrix_schema_version",
            "matrix_id",
            "causal_route_id",
            "route_set",
            "precursor_id",
            "git_commit",
            "current_head_source_tree_sha256",
            "training_gate_sha256",
            "training_data_receipt",
            "training_data_receipt_sha256",
            "matched_control_report_sha256",
            "oracle_sufficiency_receipt_sha256",
            "contact_condition_mode",
            "oracle_used",
            "production_data_admission",
            "allow_5b_training",
            "allow_server_training",
            "benefit_claimed",
        },
        "causal Robot preflight receipt",
    )
    import re

    route = ensure_exact_causal_route(receipt["causal_route_id"])
    if (
        receipt["receipt_version"] != PREFLIGHT_RECEIPT_VERSION
        or receipt["status"]
        != "artifact_consistency_passed_with_explicit_external_gates"
        or receipt["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or receipt["matrix_id"] != MATRIX_ID
        or receipt["route_set"] != list(CAUSAL_ROUTES)
        or receipt["precursor_id"] != PRECURSOR_ID
        or not isinstance(receipt["git_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", receipt["git_commit"])
        or receipt["contact_condition_mode"] != "predicted"
        or receipt["oracle_used"] is not False
        or receipt["production_data_admission"] is not False
        or receipt["allow_5b_training"] is not False
        or receipt["allow_server_training"] is not False
        or receipt["benefit_claimed"] is not False
    ):
        raise CausalProductionError("causal Robot preflight identity differs")
    del route
    for name in (
        "current_head_source_tree_sha256",
        "training_gate_sha256",
        "training_data_receipt_sha256",
        "matched_control_report_sha256",
        "oracle_sufficiency_receipt_sha256",
    ):
        _sha(receipt[name], f"causal preflight {name}")
    embedded = validate_training_data_receipt_v2(receipt["training_data_receipt"])
    if sha256_json(embedded) != receipt["training_data_receipt_sha256"]:
        raise CausalProductionError("preflight embedded training-data bytes differ")
    if training_data_receipt is not None and dict(training_data_receipt) != embedded:
        raise CausalProductionError("preflight training-data argument differs")
    if (gate is None) != (gate_sha256 is None):
        raise CausalProductionError("preflight gate and digest must be supplied together")
    if gate is not None:
        validate_robot_gate_v3(gate, load_external_review=False)
        validate_training_data_receipt_v2(
            embedded, gate=gate, gate_sha256=gate_sha256
        )
        expected = {
            "git_commit": gate["expected_git_commit"],
            "current_head_source_tree_sha256": gate["artifact_fingerprints"][
                "current_head_source_tree_sha256"
            ],
            "training_gate_sha256": gate_sha256,
            "matched_control_report_sha256": gate["artifact_fingerprints"][
                "matched_control_report_sha256"
            ],
            "oracle_sufficiency_receipt_sha256": gate["artifact_fingerprints"][
                "oracle_sufficiency_receipt_sha256"
            ],
        }
        if any(receipt[name] != expected_value for name, expected_value in expected.items()):
            raise CausalProductionError("preflight receipt differs from Robot gate")
    return receipt


def verify_gate_fingerprint_paths(
    gate: Mapping[str, Any], path_by_fingerprint: Mapping[str, str | Path]
) -> None:
    """Re-hash explicitly supplied files; directories need their own tree receipt."""

    validate_robot_gate_v3(gate, load_external_review=False)
    expected = gate["artifact_fingerprints"]
    if not isinstance(path_by_fingerprint, Mapping) or not path_by_fingerprint:
        raise CausalProductionError("preflight fingerprint path mapping is empty")
    for name, raw_path in path_by_fingerprint.items():
        if name not in expected:
            raise CausalProductionError(f"unknown gate fingerprint path: {name}")
        try:
            digest, _ = hash_verified_regular_file(raw_path)
        except (OSError, ValueError) as exc:
            raise CausalProductionError(f"cannot verify gate artifact: {name}") from exc
        if digest != expected[name]:
            raise CausalProductionError(f"gate artifact bytes changed: {name}")


def load_robot_gate_from_environment() -> tuple[dict[str, Any], str, Path]:
    path_text = os.environ.get("GWP_CONTACT_CAUSAL_TRAINING_GATE", "").strip()
    if not path_text:
        raise CausalProductionError("causal Robot config requires an explicit v3 gate path")
    path = Path(path_text).expanduser().absolute()
    gate, digest, _ = strict_json_file(path, name="causal Robot training gate")
    validate_robot_gate_v3(gate, gate_path=path, load_external_review=True)
    route = ensure_exact_causal_route(os.environ.get("GWP_CONTACT_CAUSAL_ROUTE", "").strip())
    if os.environ.get("GWP_CONTACT_CAUSAL_MATRIX_ID", "").strip() != MATRIX_ID:
        raise CausalProductionError("runtime causal matrix ID differs")
    if os.environ.get("GWP_MATCHED_CONTROL_VARIANT_ID", route).strip() != route:
        raise CausalProductionError("matched-control variant must equal the exact causal route")
    run_id = os.environ.get("GWP_CONTACT_TRAINING_RUN_ID", "").strip()
    run_nonce = os.environ.get("GWP_CONTACT_TRAINING_RUN_NONCE", "").strip()
    if (
        run_id != gate["training_run_id_by_route"][route]
        or run_nonce != gate["training_run_nonce_by_route"][route]
    ):
        raise CausalProductionError("runtime run identity differs from causal Robot gate")
    return gate, digest, path

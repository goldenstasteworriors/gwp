"""Stdlib-only admission bootstrap for the op42 Ego v7 and Robot v3 gates.

This module is intentionally safe to import before NumPy, Torch, datasets, or
trainer code.  It can only reject a launch.  The heavy preflights must replay
the exact bytes returned here before constructing a training config.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping

from world_action_model.verified_io import (
    read_verified_bytes,
    strict_json_loads,
    verified_head_code_manifest,
)


EGO_GATE_VERSION = "ego-discrete-contact-stage1-training-gate/v7"
EGO_REVIEW_VERSION = "ego-contact-planner-gate-independent-acceptance/v1"
ROBOT_GATE_VERSION = "contact-asym25-training-gate/v3"
ROBOT_REVIEW_VERSION = "contact-asym25-gate-reviewer-acceptance/v2"
ROBOT_ROUTE_ACCEPTANCE_VERSION = "contact-causal-route-training-acceptance/v1"
MATRIX_ID = "contact-asym25-causal-main-q12-b16-stage2/v1"
MATRIX_SCHEMA_VERSION = "contact-causal-production-matrix/v1"
PRECURSOR_ID = "ego-contact-planner-precursor-q12/v1"
CAUSAL_ROUTES = (
    "aux_only_direct_q12",
    "equal_bit_unstructured_hard_st",
    "hard_st_semantic",
)
EGO_REQUIRED_FLAGS = (
    "human_adjudicated_contact_labels_passed", "dual_annotation_agreement_passed",
    "group_metadata_complete", "dataset_source_binding_passed",
    "t5_independent_replay_passed", "route_neutral_objective_reviewed",
    "action_path_excluded_from_scientific_io", "parameter_freeze_receipt_passed",
    "fixed_save_step_preregistered", "allow_server_training", "allow_5b_training",
)
ROBOT_REQUIRED_FLAGS = (
    "human_adjudicated_robot_sidecar_passed", "dual_annotation_agreement_passed",
    "group_metadata_complete", "asymmetric25_real_target_data_passed",
    "hardware_contract_verified", "train_only_norm_verified",
    "dataset_positive_contact_coverage_passed", "oracle_sufficiency_passed",
    "precursor_byte_identity_passed", "matched_control_hash_collector_passed",
    "current_head_source_binding_passed", "allow_server_training",
    "allow_5b_training",
)
ROBOT_FINGERPRINT_KEYS = (
    "matrix_manifest_sha256", "current_head_source_tree_sha256",
    "precursor_receipt_bytes_sha256", "precursor_bundle_sha256",
    "hardware_contract_sha256", "calibration_capture_sha256",
    "dataset_tree_sha256", "dataset_info_sha256",
    "episode_metadata_sha256", "train_only_norm_sha256",
    "sidecar_meta_sha256", "sidecar_bundle_sha256",
    "split_manifest_sha256", "coverage_csv_sha256",
    "segments_csv_sha256", "inventory_csv_sha256", "build_config_sha256",
    "dataset_contact_coverage_receipt_sha256",
    "oracle_sufficiency_receipt_sha256", "matched_control_report_sha256",
    "matched_control_trace_manifest_sha256", "route_contract_bundle_sha256",
    "runtime_environment_sha256", "dataset_runtime_source_receipt_sha256",
    "t5_runtime_source_receipt_sha256", "hardened_preflight_receipt_sha256",
)

EGO_BOOTSTRAP_CODE_PATHS = (
    "configs/giga_world_policy_0_5_ego_contact_causal_precursor.py",
    "world_action_model/contact_causal_gate_bootstrap.py",
    "world_action_model/contact_ego_causal_precursor.py",
)
ROBOT_BOOTSTRAP_CODE_PATHS = (
    "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_causal_matrix_finetune.py",
    "world_action_model/contact_causal_gate_bootstrap.py",
    "world_action_model/contact_causal_preflight.py",
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)


class CausalGateBootstrapError(ValueError):
    pass


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalGateBootstrapError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not _SHA_RE.fullmatch(value)
        or value == "0" * 64
    ):
        raise CausalGateBootstrapError(f"{name} must be a nonzero lowercase SHA-256")
    return value


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise CausalGateBootstrapError(f"{name} must be a 16-128 character identity")
    return value


def _utc(value: object, name: str, *, require_future: bool) -> str:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        raise CausalGateBootstrapError(f"{name} must be canonical UTC with Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CausalGateBootstrapError(f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise CausalGateBootstrapError(f"{name} is not UTC")
    if require_future and parsed <= datetime.now(timezone.utc):
        raise CausalGateBootstrapError(f"{name} has expired")
    return value


def _utc_value(value: object, name: str, *, require_future: bool) -> datetime:
    _utc(value, name, require_future=require_future)
    return datetime.fromisoformat(str(value)[:-1] + "+00:00")


def _canonical_sha(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CausalGateBootstrapError("non-canonical gate value") from exc
    return hashlib.sha256(payload).hexdigest()


def _load(path: str | Path, name: str) -> tuple[dict[str, Any], str, bytes, Path]:
    selected = Path(path).expanduser().absolute()
    try:
        payload, digest, _ = read_verified_bytes(selected)
        value = strict_json_loads(payload, name=name)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CausalGateBootstrapError(f"cannot read verified {name} bytes") from exc
    if not isinstance(value, dict):
        raise CausalGateBootstrapError(f"{name} must be an object")
    return value, digest, payload, selected


def _resolve_external(gate_path: Path, entry: object, name: str):
    entry = _exact(entry, {"path", "sha256"}, f"{name} entry")
    _sha(entry["sha256"], f"{name} expected digest")
    if not isinstance(entry["path"], str) or not entry["path"]:
        raise CausalGateBootstrapError(f"{name} path is missing")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = gate_path.parent / path
    value, digest, payload, path = _load(path, name)
    if digest != entry["sha256"]:
        raise CausalGateBootstrapError(f"{name} verified bytes differ")
    return value, digest, payload, path


def _verify_head_code(
    *, repository_root: str | Path, expected_commit: object,
    expected_code_sha256: object, logical_paths: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(expected_commit, str) or not _COMMIT_RE.fullmatch(expected_commit):
        raise CausalGateBootstrapError("gate must bind a full current HEAD")
    expected_code_sha256 = _exact(
        expected_code_sha256, set(logical_paths), "reviewed HEAD code blob map"
    )
    for path, digest in expected_code_sha256.items():
        _sha(digest, f"reviewed HEAD code blob {path}")
    try:
        commit, manifest = verified_head_code_manifest(
            repository_root, logical_paths, expected_commit=expected_commit
        )
    except (OSError, ValueError) as exc:
        raise CausalGateBootstrapError("runtime code does not equal reviewed HEAD blobs") from exc
    actual = {row["logical_path"]: row["head_blob_sha256"] for row in manifest}
    if actual != expected_code_sha256:
        raise CausalGateBootstrapError("reviewed code blob map differs from current HEAD/runtime")
    return {"git_commit": commit, "code_blob_sha256": actual}


def _gate_contract_sha(gate: Mapping[str, Any], *, review_report_name: str) -> str:
    normalized = dict(gate)
    reports = dict(normalized["evidence_reports"])
    reports[review_report_name] = {"path": None, "sha256": None}
    normalized["evidence_reports"] = reports
    return _canonical_sha(normalized)


def _route_acceptance_gate_contract_sha(gate: Mapping[str, Any]) -> str:
    """Preregistered gate identity without either circular acceptance digest."""

    normalized = dict(gate)
    reports = dict(normalized["evidence_reports"])
    reports["gate_reviewer_acceptance"] = {"path": None, "sha256": None}
    normalized["evidence_reports"] = reports
    normalized["external_acceptance_path_by_route"] = {
        route: None for route in CAUSAL_ROUTES
    }
    normalized["external_acceptance_sha256_by_route"] = {
        route: None for route in CAUSAL_ROUTES
    }
    return _canonical_sha(normalized)


def validate_robot_route_acceptance_v1(
    value: object,
    *,
    gate: Mapping[str, Any],
    route: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    acceptance = _exact(
        value,
        {
            "acceptance_version", "status", "accepted",
            "matrix_schema_version", "matrix_id", "causal_route_id",
            "training_run_id", "training_run_nonce", "reviewer_identity",
            "reviewer_attestation_nonce", "acceptance_run_id",
            "acceptance_run_nonce", "evidence_valid_until_utc",
            "reviewed_at_utc", "reviewed_gate_preregistered_contract_sha256",
            "reviewed_git_commit", "reviewed_code_blob_sha256",
            "runtime_environment_sha256", "reviewed_artifact_fingerprints",
            "reviewed_evidence_report_sha256", "oracle_used",
            "trainer_self_signed", "production_data_admission",
            "allow_5b_training", "allow_server_training", "benefit_claimed",
        },
        f"Robot route acceptance {route}",
    )
    if (
        route not in CAUSAL_ROUTES
        or acceptance["acceptance_version"] != ROBOT_ROUTE_ACCEPTANCE_VERSION
        or acceptance["status"] != "accepted"
        or acceptance["accepted"] is not True
        or acceptance["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or acceptance["matrix_id"] != MATRIX_ID
        or acceptance["causal_route_id"] != route
        or acceptance["training_run_id"] != gate["training_run_id_by_route"][route]
        or acceptance["training_run_nonce"]
        != gate["training_run_nonce_by_route"][route]
        or acceptance["evidence_valid_until_utc"]
        != gate["evidence_valid_until_utc"]
        or acceptance["reviewed_gate_preregistered_contract_sha256"]
        != _route_acceptance_gate_contract_sha(gate)
        or acceptance["reviewed_git_commit"] != gate["expected_git_commit"]
        or acceptance["runtime_environment_sha256"]
        != gate["artifact_fingerprints"]["runtime_environment_sha256"]
        or acceptance["reviewed_artifact_fingerprints"]
        != gate["artifact_fingerprints"]
        or acceptance["oracle_used"] is not False
        or acceptance["trainer_self_signed"] is not False
        or acceptance["production_data_admission"] is not True
        or acceptance["allow_5b_training"] is not True
        or acceptance["allow_server_training"] is not True
        or acceptance["benefit_claimed"] is not False
    ):
        raise CausalGateBootstrapError("Robot route acceptance binding differs")
    expected_reports = {
        name: entry["sha256"]
        for name, entry in gate["evidence_reports"].items()
        if name != "gate_reviewer_acceptance"
    }
    if acceptance["reviewed_evidence_report_sha256"] != expected_reports:
        raise CausalGateBootstrapError("Robot route acceptance evidence map differs")
    identities = [
        acceptance[name]
        for name in (
            "reviewer_identity", "reviewer_attestation_nonce",
            "acceptance_run_id", "acceptance_run_nonce",
        )
    ]
    for index, identity in enumerate(identities):
        _identity(identity, f"Robot route reviewer identity {index}")
    if len(set(identities)) != 4:
        raise CausalGateBootstrapError("Robot route reviewer identities overlap")
    expiry = _utc_value(
        acceptance["evidence_valid_until_utc"],
        "Robot route acceptance expiry", require_future=True,
    )
    reviewed_at = _utc_value(
        acceptance["reviewed_at_utc"],
        "Robot route acceptance timestamp", require_future=False,
    )
    if reviewed_at > datetime.now(timezone.utc) or reviewed_at >= expiry:
        raise CausalGateBootstrapError("Robot route acceptance timestamp differs")
    if acceptance["runtime_environment_sha256"] != runtime_environment_sha256_stdlib():
        raise CausalGateBootstrapError("Robot route acceptance runtime differs")
    _verify_head_code(
        repository_root=repository_root,
        expected_commit=acceptance["reviewed_git_commit"],
        expected_code_sha256=acceptance["reviewed_code_blob_sha256"],
        logical_paths=ROBOT_BOOTSTRAP_CODE_PATHS,
    )
    return acceptance


def runtime_environment_sha256_stdlib() -> str:
    packages = {}
    for name in (
        "accelerate", "av", "datasets", "decord", "diffusers", "h5py",
        "huggingface-hub", "lerobot", "numpy", "pandas", "pyarrow",
        "safetensors", "torch", "torchvision", "transformers",
    ):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return _canonical_sha(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).absolute()),
            "platform": platform.platform(),
            "package_versions": packages,
        }
    )


def validate_ego_precursor_review_v1(
    value: object, *, gate: Mapping[str, Any], gate_bytes_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    review = _exact(
        value,
        {
            "acceptance_version", "status", "accepted", "precursor_id",
            "reviewer_identity", "reviewer_attestation_nonce",
            "acceptance_run_id", "acceptance_run_nonce",
            "training_run_id", "training_run_nonce",
            "evidence_valid_until_utc", "reviewed_at_utc",
            "reviewed_gate_contract_sha256",
            "reviewed_git_commit", "reviewed_code_blob_sha256",
            "current_head_source_tree_sha256",
            "runtime_environment_sha256", "dataset_fingerprint_sha256",
            "dataset_source_tree_sha256", "episode_metadata_sha256",
            "t5_source_receipt_sha256",
            "t5_independent_replay_attestation_sha256", "norm_stats_sha256",
            "hdf5_bundle_sha256", "sidecar_meta_sha256",
            "sidecar_bundle_sha256", "coverage_csv_sha256",
            "segments_csv_sha256", "inventory_csv_sha256",
            "build_config_sha256", "split_manifest_sha256",
            "root_checkpoint_bundle_sha256", "precursor_contract_sha256",
            "human_adjudicated_contact_only", "action_used_as_contact_proxy",
            "hand_active_used_as_contact_proxy", "other_proxy_used_as_contact",
        },
        "Ego precursor independent review",
    )
    if (
        review["acceptance_version"] != EGO_REVIEW_VERSION
        or review["status"] != "accepted"
        or review["accepted"] is not True
        or review["precursor_id"] != PRECURSOR_ID
        or review["reviewed_gate_contract_sha256"]
        != _gate_contract_sha(gate, review_report_name="external_reviewer_acceptance")
        or review["reviewed_git_commit"] != gate["expected_git_commit"]
        or review["training_run_id"] != gate["training_run_id"]
        or review["training_run_nonce"] != gate["training_run_nonce"]
        or review["evidence_valid_until_utc"] != gate["evidence_valid_until_utc"]
        or review["human_adjudicated_contact_only"] is not True
        or review["action_used_as_contact_proxy"] is not False
        or review["hand_active_used_as_contact_proxy"] is not False
        or review["other_proxy_used_as_contact"] is not False
    ):
        raise CausalGateBootstrapError("Ego independent review binding differs")
    expiry = _utc_value(
        review["evidence_valid_until_utc"], "Ego review expiry", require_future=True
    )
    reviewed_at = _utc_value(
        review["reviewed_at_utc"], "Ego review timestamp", require_future=False
    )
    if reviewed_at > datetime.now(timezone.utc) or reviewed_at >= expiry:
        raise CausalGateBootstrapError("Ego review timestamp is future/after expiry")
    identities = [
        review[name] for name in (
            "reviewer_identity", "reviewer_attestation_nonce",
            "acceptance_run_id", "acceptance_run_nonce",
            "training_run_id", "training_run_nonce",
        )
    ]
    for index, identity in enumerate(identities):
        _identity(identity, f"Ego review identity {index}")
    if len(set(identities)) != len(identities):
        raise CausalGateBootstrapError("trainer/reviewer identities and nonces must be distinct")
    fingerprint_mirrors = {
        name: review[name]
        for name in (
            "runtime_environment_sha256", "dataset_fingerprint_sha256",
            "current_head_source_tree_sha256",
            "dataset_source_tree_sha256", "episode_metadata_sha256",
            "t5_source_receipt_sha256", "t5_independent_replay_attestation_sha256",
            "norm_stats_sha256", "hdf5_bundle_sha256", "sidecar_meta_sha256",
            "sidecar_bundle_sha256", "coverage_csv_sha256", "segments_csv_sha256",
            "inventory_csv_sha256", "build_config_sha256", "split_manifest_sha256",
            "root_checkpoint_bundle_sha256", "precursor_contract_sha256",
        )
    }
    for name, digest in fingerprint_mirrors.items():
        _sha(digest, f"Ego reviewed {name}")
        if gate["artifact_fingerprints"].get(name) != digest:
            raise CausalGateBootstrapError(f"Ego review fingerprint differs: {name}")
    if review["runtime_environment_sha256"] != runtime_environment_sha256_stdlib():
        raise CausalGateBootstrapError("Ego reviewed runtime differs from current runtime")
    _sha(review["reviewed_gate_contract_sha256"], "Ego reviewed gate contract")
    _sha(gate_bytes_sha256, "Ego verified gate bytes")
    _verify_head_code(
        repository_root=repository_root,
        expected_commit=review["reviewed_git_commit"],
        expected_code_sha256=review["reviewed_code_blob_sha256"],
        logical_paths=EGO_BOOTSTRAP_CODE_PATHS,
    )
    return review


def bootstrap_ego_v7_stdlib(
    *, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    gate_text = os.environ.get("GWP_EGO_CAUSAL_PRECURSOR_GATE", "").strip()
    run_id = os.environ.get("GWP_EGO_CAUSAL_PRECURSOR_TRAINING_RUN_ID", "").strip()
    nonce = os.environ.get("GWP_EGO_CAUSAL_PRECURSOR_TRAINING_NONCE", "").strip()
    if not gate_text or not run_id or not nonce:
        raise CausalGateBootstrapError("Ego v7 bootstrap requires gate, run ID, and nonce")
    gate, gate_sha, gate_bytes, gate_path = _load(gate_text, "Ego v7 precursor gate")
    gate = _exact(
        gate,
        {
            "gate_version", "precursor_id", "evidence_valid_until_utc",
            "expected_git_commit", "training_run_id", "training_run_nonce",
            *EGO_REQUIRED_FLAGS, "fixed_save_step", "objective_contract",
            "artifact_fingerprints", "episode_npz_sha256", "evidence_reports",
            "reviewed_by", "reviewed_at_utc", "note",
        },
        "Ego v7 precursor gate",
    )
    if (
        gate.get("gate_version") != EGO_GATE_VERSION
        or gate.get("precursor_id") != PRECURSOR_ID
        or gate.get("training_run_id") != run_id
        or gate.get("training_run_nonce") != nonce
    ):
        raise CausalGateBootstrapError("Ego v7 gate identity differs")
    failed = [name for name in EGO_REQUIRED_FLAGS if gate[name] is not True]
    if failed:
        raise CausalGateBootstrapError(f"Ego v7 gate flags are not true: {failed}")
    if not isinstance(gate["artifact_fingerprints"], dict):
        raise CausalGateBootstrapError("Ego v7 fingerprints are missing")
    for name, digest in gate["artifact_fingerprints"].items():
        _sha(digest, f"Ego v7 fingerprint {name}")
    _identity(run_id, "Ego training run ID")
    _identity(nonce, "Ego training run nonce")
    _identity(gate["reviewed_by"], "Ego gate reviewer identity")
    gate_expiry = _utc_value(
        gate.get("evidence_valid_until_utc"), "Ego v7 gate expiry", require_future=True
    )
    gate_reviewed_at = _utc_value(
        gate["reviewed_at_utc"], "Ego gate review timestamp", require_future=False
    )
    if (
        len({run_id, nonce, gate["reviewed_by"]}) != 3
        or gate_reviewed_at > datetime.now(timezone.utc)
        or gate_reviewed_at >= gate_expiry
    ):
        raise CausalGateBootstrapError("Ego gate identity/timestamp namespace differs")
    reports = gate.get("evidence_reports")
    if not isinstance(reports, dict) or "external_reviewer_acceptance" not in reports:
        raise CausalGateBootstrapError("Ego v7 gate lacks independent reviewer acceptance")
    review, review_sha, review_bytes, review_path = _resolve_external(
        gate_path, reports["external_reviewer_acceptance"],
        "Ego precursor independent reviewer acceptance",
    )
    root = Path(repository_root or Path(__file__).resolve().parents[1]).absolute()
    validated_review = validate_ego_precursor_review_v1(
        review, gate=gate, gate_bytes_sha256=gate_sha, repository_root=root
    )
    review_ids = {
        validated_review[name]
        for name in (
            "reviewer_identity", "reviewer_attestation_nonce",
            "acceptance_run_id", "acceptance_run_nonce",
        )
    }
    if gate["reviewed_by"] in review_ids:
        raise CausalGateBootstrapError("Ego gate/external reviewer identities overlap")
    return {
        "bootstrap_version": "ego-causal-precursor-stdlib-bootstrap/v1",
        "gate": gate, "gate_sha256": gate_sha, "gate_bytes": gate_bytes,
        "gate_path": str(gate_path), "review": review,
        "review_sha256": review_sha, "review_bytes": review_bytes,
        "review_path": str(review_path),
    }


def bootstrap_robot_v3_stdlib(
    *, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    gate_text = os.environ.get("GWP_CONTACT_CAUSAL_TRAINING_GATE", "").strip()
    route = os.environ.get("GWP_CONTACT_CAUSAL_ROUTE", "").strip()
    run_id = os.environ.get("GWP_CONTACT_TRAINING_RUN_ID", "").strip()
    nonce = os.environ.get("GWP_CONTACT_TRAINING_RUN_NONCE", "").strip()
    if not gate_text or route not in CAUSAL_ROUTES or not run_id or not nonce:
        raise CausalGateBootstrapError("Robot v3 bootstrap identity/path is incomplete")
    gate, gate_sha, gate_bytes, gate_path = _load(gate_text, "Robot v3 training gate")
    gate = _exact(
        gate,
        {
            "gate_version", "matrix_schema_version", "matrix_id", "precursor_id",
            "route_set", "collector_version", "supervision_version",
            "expected_git_commit", "evidence_valid_until_utc", *ROBOT_REQUIRED_FLAGS,
            "artifact_fingerprints", "evidence_reports",
            "training_run_id_by_route", "training_run_nonce_by_route",
            "external_acceptance_path_by_route",
            "external_acceptance_sha256_by_route", "reviewed_by",
            "reviewed_at_utc", "note",
        },
        "Robot v3 training gate",
    )
    if (
        gate.get("gate_version") != ROBOT_GATE_VERSION
        or gate.get("matrix_schema_version") != MATRIX_SCHEMA_VERSION
        or gate.get("matrix_id") != MATRIX_ID
        or gate.get("precursor_id") != PRECURSOR_ID
        or gate.get("route_set") != list(CAUSAL_ROUTES)
        or gate.get("collector_version") != "matched-control-hash-collector/v3"
        or gate.get("supervision_version")
        != "contact_grouped_supervision_v2_zero_safe"
        or not isinstance(gate.get("training_run_id_by_route"), dict)
        or not isinstance(gate.get("training_run_nonce_by_route"), dict)
        or gate["training_run_id_by_route"].get(route) != run_id
        or gate["training_run_nonce_by_route"].get(route) != nonce
    ):
        raise CausalGateBootstrapError("Robot v3 gate/run identity differs")
    failed = [name for name in ROBOT_REQUIRED_FLAGS if gate.get(name) is not True]
    if failed:
        raise CausalGateBootstrapError(f"Robot v3 gate flags are not true: {failed}")
    if (
        not isinstance(gate["artifact_fingerprints"], dict)
        or set(gate["artifact_fingerprints"]) != set(ROBOT_FINGERPRINT_KEYS)
    ):
        raise CausalGateBootstrapError("Robot v3 fingerprints are missing/differ")
    for name, digest in gate["artifact_fingerprints"].items():
        _sha(digest, f"Robot v3 fingerprint {name}")
    route_maps = (
        gate["training_run_id_by_route"], gate["training_run_nonce_by_route"],
        gate["external_acceptance_path_by_route"],
        gate["external_acceptance_sha256_by_route"],
    )
    if any(
        not isinstance(mapping, dict) or list(mapping) != list(CAUSAL_ROUTES)
        for mapping in route_maps
    ):
        raise CausalGateBootstrapError("Robot v3 route identity maps differ")
    all_run_identities = list(route_maps[0].values()) + list(route_maps[1].values())
    for index, identity in enumerate(all_run_identities):
        _identity(identity, f"Robot v3 route identity {index}")
    if len(set(all_run_identities)) != len(all_run_identities):
        raise CausalGateBootstrapError("Robot v3 route identities/nonces overlap")
    for route_name, path_text in route_maps[2].items():
        if not isinstance(path_text, str) or not path_text:
            raise CausalGateBootstrapError(
                f"Robot v3 route acceptance path is missing: {route_name}"
            )
    for route_name, digest in route_maps[3].items():
        _sha(digest, f"Robot v3 external acceptance {route_name}")
    if len(set(route_maps[3].values())) != len(CAUSAL_ROUTES):
        raise CausalGateBootstrapError("Robot v3 external acceptance SHAs overlap")
    _identity(run_id, "Robot training run ID")
    _identity(nonce, "Robot training run nonce")
    _identity(gate["reviewed_by"], "Robot gate reviewer identity")
    gate_expiry = _utc_value(
        gate.get("evidence_valid_until_utc"), "Robot v3 gate expiry", require_future=True
    )
    gate_reviewed_at = _utc_value(
        gate["reviewed_at_utc"], "Robot gate review timestamp", require_future=False
    )
    if (
        gate["reviewed_by"] in set(all_run_identities)
        or gate_reviewed_at > datetime.now(timezone.utc)
        or gate_reviewed_at >= gate_expiry
    ):
        raise CausalGateBootstrapError("Robot gate identity/timestamp namespace differs")
    reports = gate.get("evidence_reports")
    if not isinstance(reports, dict) or "gate_reviewer_acceptance" not in reports:
        raise CausalGateBootstrapError("Robot v3 gate lacks external acceptance")
    review, review_sha, review_bytes, review_path = _resolve_external(
        gate_path, reports["gate_reviewer_acceptance"], "Robot v3 reviewer acceptance"
    )
    review = _exact(
        review,
        {
            "acceptance_version", "status", "accepted", "matrix_schema_version",
            "matrix_id", "route_set", "reviewer_identity",
            "reviewer_attestation_nonce", "acceptance_run_id",
            "acceptance_run_nonce", "evidence_valid_until_utc", "reviewed_at_utc",
            "reviewed_gate_contract_sha256", "reviewed_code_blob_sha256",
            "current_head_source_tree_sha256", "hardware_contract_sha256",
            "dataset_tree_sha256", "train_only_norm_sha256",
            "sidecar_bundle_sha256", "precursor_receipt_bytes_sha256",
            "matched_control_report_sha256", "oracle_sufficiency_receipt_sha256",
            "runtime_environment_sha256",
            "dataset_runtime_source_receipt_sha256",
            "t5_runtime_source_receipt_sha256",
            "hardened_preflight_receipt_sha256",
            "oracle_is_main_matrix_route",
        },
        "Robot v3 external review",
    )
    if (
        review.get("acceptance_version") != ROBOT_REVIEW_VERSION
        or review.get("status") != "accepted"
        or review.get("accepted") is not True
        or review.get("matrix_id") != MATRIX_ID
        or review.get("matrix_schema_version") != MATRIX_SCHEMA_VERSION
        or review.get("route_set") != list(CAUSAL_ROUTES)
        or review.get("reviewed_gate_contract_sha256")
        != _gate_contract_sha(gate, review_report_name="gate_reviewer_acceptance")
        or review.get("evidence_valid_until_utc") != gate["evidence_valid_until_utc"]
        or review.get("oracle_is_main_matrix_route") is not False
    ):
        raise CausalGateBootstrapError("Robot external review binding differs")
    expiry = _utc_value(
        review["evidence_valid_until_utc"], "Robot review expiry", require_future=True
    )
    reviewed_at = _utc_value(
        review["reviewed_at_utc"], "Robot review timestamp", require_future=False
    )
    if reviewed_at > datetime.now(timezone.utc) or reviewed_at >= expiry:
        raise CausalGateBootstrapError("Robot review timestamp is future/after expiry")
    review_identities = [
        review[name] for name in (
            "reviewer_identity", "reviewer_attestation_nonce",
            "acceptance_run_id", "acceptance_run_nonce",
        )
    ]
    for index, identity in enumerate(review_identities):
        _identity(identity, f"Robot reviewer identity {index}")
    gate_identities = (
        set(route_maps[0].values()) | set(route_maps[1].values())
        | {gate["reviewed_by"]}
    )
    if len(set(review_identities)) != 4 or gate_identities & set(review_identities):
        raise CausalGateBootstrapError("Robot trainer/reviewer identity namespace overlaps")
    fingerprint_mirrors = {
        "current_head_source_tree_sha256": "current_head_source_tree_sha256",
        "hardware_contract_sha256": "hardware_contract_sha256",
        "dataset_tree_sha256": "dataset_tree_sha256",
        "train_only_norm_sha256": "train_only_norm_sha256",
        "sidecar_bundle_sha256": "sidecar_bundle_sha256",
        "precursor_receipt_bytes_sha256": "precursor_receipt_bytes_sha256",
        "matched_control_report_sha256": "matched_control_report_sha256",
        "oracle_sufficiency_receipt_sha256": "oracle_sufficiency_receipt_sha256",
        "runtime_environment_sha256": "runtime_environment_sha256",
        "dataset_runtime_source_receipt_sha256": "dataset_runtime_source_receipt_sha256",
        "t5_runtime_source_receipt_sha256": "t5_runtime_source_receipt_sha256",
        "hardened_preflight_receipt_sha256": "hardened_preflight_receipt_sha256",
    }
    for review_name, gate_name in fingerprint_mirrors.items():
        digest = _sha(review[review_name], f"Robot reviewed {review_name}")
        if digest != gate["artifact_fingerprints"].get(gate_name):
            raise CausalGateBootstrapError(f"Robot reviewed fingerprint differs: {review_name}")
    if review["runtime_environment_sha256"] != runtime_environment_sha256_stdlib():
        raise CausalGateBootstrapError("Robot reviewed runtime environment differs")
    root = Path(repository_root or Path(__file__).resolve().parents[1]).absolute()
    expected_blobs = review.get("reviewed_code_blob_sha256")
    _verify_head_code(
        repository_root=root, expected_commit=gate.get("expected_git_commit"),
        expected_code_sha256=expected_blobs,
        logical_paths=ROBOT_BOOTSTRAP_CODE_PATHS,
    )
    route_acceptances: dict[str, Any] = {}
    route_acceptance_bytes: dict[str, bytes] = {}
    route_acceptance_paths: dict[str, str] = {}
    external_reviewer_identities: set[str] = set(review_identities) | gate_identities
    for route_name in CAUSAL_ROUTES:
        path = Path(gate["external_acceptance_path_by_route"][route_name])
        if not path.is_absolute():
            path = gate_path.parent / path
        value, digest, payload, selected_path = _load(
            path, f"Robot route acceptance {route_name}"
        )
        if digest != gate["external_acceptance_sha256_by_route"][route_name]:
            raise CausalGateBootstrapError(
                f"Robot route acceptance bytes changed: {route_name}"
            )
        validated = validate_robot_route_acceptance_v1(
            value, gate=gate, route=route_name, repository_root=root
        )
        route_ids = {
            validated[name]
            for name in (
                "reviewer_identity", "reviewer_attestation_nonce",
                "acceptance_run_id", "acceptance_run_nonce",
            )
        }
        if route_ids & external_reviewer_identities:
            raise CausalGateBootstrapError(
                "Robot route acceptance identities are not globally unique"
            )
        external_reviewer_identities.update(route_ids)
        route_acceptances[route_name] = validated
        route_acceptance_bytes[route_name] = payload
        route_acceptance_paths[route_name] = str(selected_path)
    return {
        "bootstrap_version": "contact-asym25-causal-stdlib-bootstrap/v1",
        "gate": gate, "gate_sha256": gate_sha, "gate_bytes": gate_bytes,
        "gate_path": str(gate_path), "review": review,
        "review_sha256": review_sha, "review_bytes": review_bytes,
        "review_path": str(review_path), "route": route,
        "route_acceptances": route_acceptances,
        "route_acceptance_bytes": route_acceptance_bytes,
        "route_acceptance_paths": route_acceptance_paths,
    }


def assert_stdlib_bootstrap_imports() -> None:
    """Formal negative helper: importing this module must not load heavy runtimes."""

    import sys

    forbidden = sorted(
        name for name in sys.modules
        if name == "torch" or name.startswith("torch.")
        or name == "numpy" or name.startswith("numpy.")
    )
    if forbidden:
        raise CausalGateBootstrapError(
            f"stdlib bootstrap imported a forbidden heavy module: {forbidden[0]}"
        )

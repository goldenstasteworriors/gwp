"""Route-neutral Ego contact-planner precursor evidence contract.

The precursor is not a fourth causal route.  It trains only the visual WAM
and contact planner/decoder from human-adjudicated contact labels.  The action
path is frozen and absent from the scientific objective inputs/outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
from typing import Any

from world_action_model.contact_causal_production import (
    CausalProductionError,
    EGO_PRECURSOR_GATE_VERSION,
    PRECURSOR_ID,
    canonical_json_bytes,
    sha256_json,
    strict_json_file,
    validate_deadline,
)
from world_action_model.verified_io import read_verified_bytes


PRECURSOR_CONTRACT_VERSION = "ego-contact-planner-precursor-contract/v1"
PRECURSOR_RECEIPT_VERSION = "ego-contact-planner-precursor-receipt/v1"
PRECURSOR_BUNDLE_VERSION = "ego-contact-planner-precursor-bundle/v1"
CONTACT_LABEL_SCHEMA = "contact_segment_v1_1"

GATE_FLAGS = (
    "human_adjudicated_contact_labels_passed",
    "dual_annotation_agreement_passed",
    "group_metadata_complete",
    "dataset_source_binding_passed",
    "t5_independent_replay_passed",
    "route_neutral_objective_reviewed",
    "action_path_excluded_from_scientific_io",
    "parameter_freeze_receipt_passed",
    "fixed_save_step_preregistered",
    "allow_server_training",
    "allow_5b_training",
)
FINGERPRINT_KEYS = (
    "runtime_environment_sha256",
    "dataset_fingerprint_sha256",
    "dataset_source_tree_sha256",
    "episode_metadata_sha256",
    "t5_source_receipt_sha256",
    "t5_independent_replay_attestation_sha256",
    "norm_stats_sha256",
    "hdf5_bundle_sha256",
    "sidecar_meta_sha256",
    "sidecar_bundle_sha256",
    "coverage_csv_sha256",
    "segments_csv_sha256",
    "inventory_csv_sha256",
    "build_config_sha256",
    "split_manifest_sha256",
    "root_checkpoint_bundle_sha256",
    "current_head_source_tree_sha256",
    "precursor_contract_sha256",
)
EVIDENCE_REPORT_KEYS = (
    "annotation_agreement",
    "group_metadata_receipt",
    "t5_independent_replay",
    "parameter_freeze_review",
    "fixed_save_step_registration",
    "external_reviewer_acceptance",
)


def precursor_objective_contract(*, fixed_save_step: int) -> dict[str, Any]:
    if isinstance(fixed_save_step, bool) or not isinstance(fixed_save_step, int) or fixed_save_step <= 0:
        raise CausalProductionError("precursor fixed save step must be a positive integer")
    payload = {
        "contract_version": PRECURSOR_CONTRACT_VERSION,
        "precursor_id": PRECURSOR_ID,
        "planner_queries": 12,
        "planner_width": 22,
        "contact_schema_version": CONTACT_LABEL_SCHEMA,
        "contact_label_origin": "human_adjudicated",
        "contact_review_status": "reviewed",
        "visual_wam_loss_weight": 1.0,
        "contact_loss_weight": 0.25,
        "action_loss_weight": 0.0,
        "action_input_used": False,
        "action_target_used": False,
        "action_output_produced": False,
        "action_path_scientific_input_output": False,
        "hand_active_used_as_contact_ground_truth": False,
        "grip_used_as_contact_ground_truth": False,
        "oracle_used": False,
        "route_specific_transport_used": False,
        "route_specific_action_cache_used": False,
        "fixed_save_step": fixed_save_step,
        "checkpoint_selection_by_stage2_result_allowed": False,
    }
    return {**payload, "receipt_sha256": sha256_json(payload)}


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def _identity(value: object, name: str) -> str:
    import re

    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", value
    ):
        raise CausalProductionError(f"{name} must be a 16-128 character identity")
    return value


def validate_precursor_gate(value: object) -> dict[str, Any]:
    gate = _exact(
        value,
        {
            "gate_version",
            "precursor_id",
            "evidence_valid_until_utc",
            "expected_git_commit",
            "training_run_id",
            "training_run_nonce",
            *GATE_FLAGS,
            "fixed_save_step",
            "objective_contract",
            "artifact_fingerprints",
            "episode_npz_sha256",
            "evidence_reports",
            "reviewed_by",
            "reviewed_at_utc",
            "note",
        },
        "Ego causal precursor gate",
    )
    if gate["gate_version"] != EGO_PRECURSOR_GATE_VERSION or gate["precursor_id"] != PRECURSOR_ID:
        raise CausalProductionError("Ego precursor gate version/ID differs")
    failed = [name for name in GATE_FLAGS if gate[name] is not True]
    if failed:
        raise CausalProductionError(f"Ego precursor gate flags are not explicitly true: {failed}")
    validate_deadline(gate["evidence_valid_until_utc"])
    import re

    if not isinstance(gate["expected_git_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", gate["expected_git_commit"]
    ):
        raise CausalProductionError("Ego precursor gate requires a full current HEAD")
    _identity(gate["training_run_id"], "precursor training run ID")
    _identity(gate["training_run_nonce"], "precursor training nonce")
    _identity(gate["reviewed_by"], "precursor reviewer")
    if len(
        {gate["training_run_id"], gate["training_run_nonce"], gate["reviewed_by"]}
    ) != 3:
        raise CausalProductionError("precursor trainer/reviewer identities overlap")
    if not isinstance(gate["reviewed_at_utc"], str) or not gate["reviewed_at_utc"].endswith("Z"):
        raise CausalProductionError("precursor reviewed_at_utc must be explicit UTC")
    expected_objective = precursor_objective_contract(fixed_save_step=gate["fixed_save_step"])
    if gate["objective_contract"] != expected_objective:
        raise CausalProductionError("precursor route-neutral objective contract differs")
    fingerprints = _exact(
        gate["artifact_fingerprints"], set(FINGERPRINT_KEYS), "precursor fingerprints"
    )
    for name in FINGERPRINT_KEYS:
        _sha(fingerprints[name], f"precursor fingerprint {name}")
    episode_hashes = gate["episode_npz_sha256"]
    if (
        not isinstance(episode_hashes, dict)
        or not episode_hashes
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"episode_[0-9]{6}\.npz", name)
            or _sha(digest, f"precursor sidecar {name}") != digest
            for name, digest in episode_hashes.items()
        )
    ):
        raise CausalProductionError("precursor sidecar episode SHA map is invalid")
    if fingerprints["precursor_contract_sha256"] != expected_objective["receipt_sha256"]:
        raise CausalProductionError("precursor objective fingerprint differs")
    reports = _exact(gate["evidence_reports"], set(EVIDENCE_REPORT_KEYS), "precursor evidence reports")
    for name, entry in reports.items():
        entry = _exact(entry, {"path", "sha256"}, f"precursor evidence {name}")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise CausalProductionError("precursor evidence path is missing")
        _sha(entry["sha256"], f"precursor evidence {name}")
    return gate


def validate_parameter_partition(value: object) -> dict[str, Any]:
    partition = _exact(
        value,
        {
            "all_parameter_names",
            "trainable_parameter_names",
            "frozen_parameter_names",
            "transfer_parameter_names",
            "action_path_parameter_names",
            "contact_condition_encoder_parameter_names",
            "all_parameter_schema_sha256",
            "trainable_parameter_names_sha256",
            "frozen_parameter_names_sha256",
            "transfer_parameter_names_sha256",
            "action_path_parameter_names_sha256",
        },
        "precursor parameter partition",
    )
    list_fields = (
        "all_parameter_names",
        "trainable_parameter_names",
        "frozen_parameter_names",
        "transfer_parameter_names",
        "action_path_parameter_names",
        "contact_condition_encoder_parameter_names",
    )
    for name in list_fields:
        values = partition[name]
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(not isinstance(item, str) or not item for item in values)
        ):
            raise CausalProductionError(f"precursor {name} must be sorted and unique")
    all_names = set(partition["all_parameter_names"])
    trainable = set(partition["trainable_parameter_names"])
    frozen = set(partition["frozen_parameter_names"])
    transfer = set(partition["transfer_parameter_names"])
    action = set(partition["action_path_parameter_names"])
    condition = set(partition["contact_condition_encoder_parameter_names"])
    if trainable & frozen or trainable | frozen != all_names:
        raise CausalProductionError("precursor trainable/frozen partition is incomplete")
    if not transfer <= all_names or not action <= frozen or not condition <= frozen:
        raise CausalProductionError("precursor action/condition path must be frozen")
    planner_prefixes = (
        "contact_query_tokens",
        "contact_decoder.",
        "contact_planner_",
    )
    if not any(name.startswith(planner_prefixes) for name in trainable):
        raise CausalProductionError("precursor planner/decoder parameter set is not trainable")
    mirrors = {
        "all_parameter_schema_sha256": partition["all_parameter_names"],
        "trainable_parameter_names_sha256": partition["trainable_parameter_names"],
        "frozen_parameter_names_sha256": partition["frozen_parameter_names"],
        "transfer_parameter_names_sha256": partition["transfer_parameter_names"],
        "action_path_parameter_names_sha256": partition["action_path_parameter_names"],
    }
    for digest_name, names in mirrors.items():
        if partition[digest_name] != sha256_json(names):
            raise CausalProductionError(f"precursor parameter digest differs: {digest_name}")
    return partition


def freeze_precursor_action_path(model: Any) -> dict[str, Any]:
    """Freeze the explicit action path before optimizer construction."""

    named = list(model.named_parameters())
    if not named:
        raise CausalProductionError("precursor model has no named parameters")
    action_names = []
    condition_names = []
    for name, parameter in named:
        is_condition = name.startswith("contact_condition_encoder.")
        is_action = (
            name.startswith("action_condition_embedder.")
            or name.startswith("action_encoder.")
            or name.startswith("action_decoder.")
            or name == "action_scale_shift_table"
            or ".action_expert." in name
            or is_condition
        )
        if is_action:
            parameter.requires_grad_(False)
            action_names.append(name)
        if is_condition:
            condition_names.append(name)
    all_names = sorted(name for name, _ in named)
    trainable = sorted(name for name, parameter in named if parameter.requires_grad)
    frozen = sorted(name for name, parameter in named if not parameter.requires_grad)
    transfer = list(all_names)
    partition = {
        "all_parameter_names": all_names,
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "transfer_parameter_names": transfer,
        "action_path_parameter_names": sorted(action_names),
        "contact_condition_encoder_parameter_names": sorted(condition_names),
        "all_parameter_schema_sha256": sha256_json(all_names),
        "trainable_parameter_names_sha256": sha256_json(trainable),
        "frozen_parameter_names_sha256": sha256_json(frozen),
        "transfer_parameter_names_sha256": sha256_json(transfer),
        "action_path_parameter_names_sha256": sha256_json(sorted(action_names)),
    }
    return validate_parameter_partition(partition)


def validate_precursor_receipt(value: object) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "receipt_version",
            "precursor_id",
            "gate_sha256",
            "objective_contract",
            "fixed_save_step",
            "root_checkpoint_bundle_sha256",
            "precursor_checkpoint_bundle_sha256",
            "root_parameter_tensor_sha256",
            "precursor_parameter_tensor_sha256",
            "frozen_root_tensor_sha256",
            "frozen_precursor_tensor_sha256",
            "parameter_partition",
            "dataset_source_tree_sha256",
            "sidecar_bundle_sha256",
            "training_code_head",
            "training_code_source_tree_sha256",
            "training_run_id",
            "training_run_nonce",
            "reviewer_acceptance_sha256",
            "action_path_scientific_input_output",
            "route_neutral",
            "hand_active_used_as_contact_ground_truth",
            "grip_used_as_contact_ground_truth",
            "scientific_admission",
        },
        "Ego precursor receipt",
    )
    if (
        receipt["receipt_version"] != PRECURSOR_RECEIPT_VERSION
        or receipt["precursor_id"] != PRECURSOR_ID
        or receipt["objective_contract"]
        != precursor_objective_contract(fixed_save_step=receipt["fixed_save_step"])
        or receipt["frozen_root_tensor_sha256"] != receipt["frozen_precursor_tensor_sha256"]
        or receipt["action_path_scientific_input_output"] is not False
        or receipt["route_neutral"] is not True
        or receipt["hand_active_used_as_contact_ground_truth"] is not False
        or receipt["grip_used_as_contact_ground_truth"] is not False
    ):
        raise CausalProductionError("Ego precursor receipt objective/frozen-route contract differs")
    validate_parameter_partition(receipt["parameter_partition"])
    for name, value in receipt.items():
        if name.endswith("_sha256"):
            _sha(value, f"precursor receipt {name}")
    import re

    if not isinstance(receipt["training_code_head"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", receipt["training_code_head"]
    ):
        raise CausalProductionError("precursor receipt must bind a full current HEAD")
    _identity(receipt["training_run_id"], "precursor run ID")
    _identity(receipt["training_run_nonce"], "precursor run nonce")
    if receipt["training_run_id"] == receipt["training_run_nonce"]:
        raise CausalProductionError("precursor run ID and nonce must be distinct")
    if receipt["scientific_admission"] != {
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }:
        raise CausalProductionError("precursor receipt cannot self-authorize science")
    return receipt


def build_live_precursor_receipt(
    *, model: Any, checkpoint_dir: str | Path,
    parameter_partition: Mapping[str, Any], objective_contract: Mapping[str, Any],
    evidence_base: Mapping[str, Any], root_parameter_tensor_sha256: str,
    frozen_root_tensor_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the route-neutral precursor receipt from live/saved tensors."""

    from world_action_model.contact_causal_checkpoint import (
        inspect_causal_checkpoint_bundle,
        validate_weight_mapping_against_topology,
    )
    from world_action_model.contact_causal_production import (
        model_parameter_and_buffer_digests,
        named_tensor_digest,
    )

    partition = validate_parameter_partition(parameter_partition)
    expected_evidence = {
        "gate_sha256", "root_checkpoint_bundle_sha256",
        "dataset_source_tree_sha256", "sidecar_bundle_sha256",
        "training_code_head", "training_code_source_tree_sha256",
        "training_run_id", "training_run_nonce", "reviewer_acceptance_sha256",
    }
    evidence = _exact(evidence_base, expected_evidence, "precursor evidence base")
    for name, value in evidence.items():
        if name.endswith("_sha256"):
            _sha(value, f"precursor evidence {name}")
    _identity(evidence["training_run_id"], "precursor evidence run ID")
    _identity(evidence["training_run_nonce"], "precursor evidence run nonce")
    if evidence["training_run_id"] == evidence["training_run_nonce"]:
        raise CausalProductionError("precursor evidence run identity overlaps")
    import re

    if not isinstance(evidence["training_code_head"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", evidence["training_code_head"]
    ):
        raise CausalProductionError("precursor evidence must bind a full HEAD")
    if objective_contract != precursor_objective_contract(
        fixed_save_step=objective_contract.get("fixed_save_step")
    ):
        raise CausalProductionError("live precursor objective differs")
    parameter_sha256, buffer_sha256 = model_parameter_and_buffer_digests(model)
    named_parameters = dict(model.named_parameters())
    frozen_current_sha256 = named_tensor_digest(
        [(name, named_parameters[name]) for name in partition["frozen_parameter_names"]]
    )
    inspected = inspect_causal_checkpoint_bundle(checkpoint_dir)
    parameter_schema = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(named_parameters.items())
    ]
    persistent_names = set(model.state_dict())
    buffer_schema = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(model.named_buffers())
        if name in persistent_names
    ]
    validate_weight_mapping_against_topology(
        Path(checkpoint_dir) / inspected["weight_file"],
        {
            "persistent_parameter_schema": parameter_schema,
            "persistent_buffer_schema": buffer_schema,
            "parameter_tensor_sha256": parameter_sha256,
            "buffer_tensor_sha256": buffer_sha256,
        },
    )
    receipt = {
        "receipt_version": PRECURSOR_RECEIPT_VERSION,
        "precursor_id": PRECURSOR_ID,
        "gate_sha256": evidence["gate_sha256"],
        "objective_contract": dict(objective_contract),
        "fixed_save_step": objective_contract["fixed_save_step"],
        "root_checkpoint_bundle_sha256": evidence["root_checkpoint_bundle_sha256"],
        "precursor_checkpoint_bundle_sha256": inspected["bundle_manifest"][
            "bundle_sha256"
        ],
        "root_parameter_tensor_sha256": _sha(
            root_parameter_tensor_sha256, "root parameter tensors"
        ),
        "precursor_parameter_tensor_sha256": parameter_sha256,
        "frozen_root_tensor_sha256": _sha(
            frozen_root_tensor_sha256, "root frozen tensors"
        ),
        "frozen_precursor_tensor_sha256": frozen_current_sha256,
        "parameter_partition": dict(partition),
        "dataset_source_tree_sha256": evidence["dataset_source_tree_sha256"],
        "sidecar_bundle_sha256": evidence["sidecar_bundle_sha256"],
        "training_code_head": evidence["training_code_head"],
        "training_code_source_tree_sha256": evidence[
            "training_code_source_tree_sha256"
        ],
        "training_run_id": evidence["training_run_id"],
        "training_run_nonce": evidence["training_run_nonce"],
        "reviewer_acceptance_sha256": evidence["reviewer_acceptance_sha256"],
        "action_path_scientific_input_output": False,
        "route_neutral": True,
        "hand_active_used_as_contact_ground_truth": False,
        "grip_used_as_contact_ground_truth": False,
        "scientific_admission": {
            "production_data_admission": False,
            "allow_5b_training": False,
            "allow_server_training": False,
            "benefit_claimed": False,
        },
    }
    return validate_precursor_receipt(receipt), inspected["bundle_manifest"]


def precursor_stage2_binding(receipt: Mapping[str, Any], *, receipt_bytes: bytes) -> dict[str, Any]:
    validate_precursor_receipt(receipt)
    canonical = canonical_json_bytes(receipt)
    if receipt_bytes != canonical:
        raise CausalProductionError("precursor receipt bytes must be canonical and immutable")
    return {
        "precursor_id": PRECURSOR_ID,
        "receipt_bytes_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "bundle_sha256": receipt["precursor_checkpoint_bundle_sha256"],
        "parameter_tensor_sha256": receipt["precursor_parameter_tensor_sha256"],
        "action_path_scientific_input_output": False,
        "route_neutral": True,
    }


def load_precursor_gate_from_environment(
    *, bootstrap_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = os.environ.get("GWP_EGO_CAUSAL_PRECURSOR_GATE", "").strip()
    run_id = os.environ.get("GWP_EGO_CAUSAL_PRECURSOR_TRAINING_RUN_ID", "").strip()
    nonce = os.environ.get("GWP_EGO_CAUSAL_PRECURSOR_TRAINING_NONCE", "").strip()
    if not path or not run_id or not nonce:
        raise CausalProductionError(
            "Ego precursor config requires explicit gate path, run ID, and training nonce"
        )
    gate, digest, payload = strict_json_file(path, name="Ego causal precursor gate")
    if bootstrap_receipt is not None:
        if (
            bootstrap_receipt.get("bootstrap_version")
            != "ego-causal-precursor-stdlib-bootstrap/v1"
            or bootstrap_receipt.get("gate_bytes") != payload
            or bootstrap_receipt.get("gate_sha256") != digest
            or bootstrap_receipt.get("gate") != gate
        ):
            raise CausalProductionError(
                "heavy Ego preflight did not replay the stdlib-verified gate bytes"
            )
        review_entry = gate.get("evidence_reports", {}).get(
            "external_reviewer_acceptance"
        )
        if (
            not isinstance(review_entry, dict)
            or bootstrap_receipt.get("review_sha256") != review_entry.get("sha256")
        ):
            raise CausalProductionError(
                "heavy Ego preflight lost the stdlib-verified reviewer bytes"
            )
        review_path = Path(bootstrap_receipt.get("review_path", ""))
        try:
            replayed_review_bytes, replayed_review_sha, _ = read_verified_bytes(
                review_path, expected_sha256=bootstrap_receipt["review_sha256"]
            )
        except (OSError, ValueError) as exc:
            raise CausalProductionError("cannot replay Ego reviewer acceptance bytes") from exc
        if (
            replayed_review_bytes != bootstrap_receipt.get("review_bytes")
            or replayed_review_sha != bootstrap_receipt.get("review_sha256")
        ):
            raise CausalProductionError("Ego reviewer acceptance changed after bootstrap")
    validate_precursor_gate(gate)
    if gate["training_run_id"] != run_id or gate["training_run_nonce"] != nonce:
        raise CausalProductionError("runtime precursor identity differs from reviewed gate")
    return gate


def replay_hardened_ego_stage1_from_environment(
    *, precursor_gate: Mapping[str, Any], precursor_gate_sha256: str,
) -> dict[str, Any]:
    """Replay the unchanged Ego Stage1 artifact/label preflight for v7."""

    from world_action_model.contact_ego_stage1_preflight import (
        ContactEgoStage1PreflightError,
        preflight_from_environment,
    )

    if (
        os.environ.get("GWP_EGO_TRAINING_RUN_NONCE", "").strip()
        != precursor_gate["training_run_nonce"]
    ):
        raise CausalProductionError(
            "Ego v7 nonce must equal the hardened Stage1 replay nonce"
        )

    try:
        receipt = preflight_from_environment()
    except ContactEgoStage1PreflightError as exc:
        raise CausalProductionError("hardened Ego Stage1 replay failed") from exc
    fingerprints = precursor_gate["artifact_fingerprints"]
    mirrors = {
        "dataset_fingerprint_sha256": "dataset_fingerprint_sha256",
        "dataset_source_tree_sha256": "source_receipt_sha256",
        "episode_metadata_sha256": "episode_metadata_sha256",
        "t5_source_receipt_sha256": "t5_source_receipt_sha256",
        "t5_independent_replay_attestation_sha256": (
            "t5_independent_replay_attestation_sha256"
        ),
        "norm_stats_sha256": "norm_stats_sha256",
        "hdf5_bundle_sha256": "hdf5_bundle_sha256",
        "sidecar_meta_sha256": "sidecar_meta_sha256",
        "sidecar_bundle_sha256": "sidecar_bundle_sha256",
        "coverage_csv_sha256": "coverage_csv_sha256",
        "segments_csv_sha256": "segments_csv_sha256",
        "inventory_csv_sha256": "inventory_csv_sha256",
        "build_config_sha256": "build_config_sha256",
        "split_manifest_sha256": "split_manifest_sha256",
    }
    if any(
        receipt.get(receipt_name) != fingerprints[gate_name]
        for gate_name, receipt_name in mirrors.items()
    ):
        raise CausalProductionError("Ego v7 gate differs from hardened Stage1 replay")
    if (
        receipt.get("episode_npz_sha256") != precursor_gate["episode_npz_sha256"]
        or receipt.get("training_run_nonce") != precursor_gate["training_run_nonce"]
        or receipt.get("contact_schema_version") != CONTACT_LABEL_SCHEMA
    ):
        raise CausalProductionError("Ego v7 sidecar/nonce differs from hardened replay")
    root_checkpoint = os.environ.get("GWP05_TRANSFORMER_PRETRAINED", "").strip()
    if not root_checkpoint:
        raise CausalProductionError("Ego v7 requires explicit precursor root checkpoint")
    from world_action_model.contact_asym25_heldout_eval import snapshot_artifact_tree

    root_snapshot = snapshot_artifact_tree(root_checkpoint)
    if (
        root_snapshot.get("tree_sha256")
        != fingerprints["root_checkpoint_bundle_sha256"]
    ):
        raise CausalProductionError("Ego precursor source checkpoint bytes changed")
    return {
        "receipt_version": "ego-causal-precursor-hardened-source-replay/v1",
        "precursor_gate_sha256": _sha(precursor_gate_sha256, "Ego precursor gate"),
        "legacy_stage1_preflight_receipt": receipt,
        "legacy_stage1_preflight_receipt_sha256": sha256_json(receipt),
        "root_checkpoint_source_tree": {
            key: root_snapshot[key]
            for key in ("path_record_only", "tree_sha256", "file_count", "total_size_bytes")
        },
    }

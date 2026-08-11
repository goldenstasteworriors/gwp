"""Versioned, fail-closed contracts for the Stage2 causal-main matrix.

This module is deliberately independent from the historical transport
mechanism production chain.  In particular it does not extend any legacy
variant registry and never treats ``hard_st`` as an alias for
``hard_st_semantic``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from world_action_model.contact_route_contract import (
    CONTACT_CAUSAL_ROUTE_CONTRACT_VERSION,
    ContactCausalRoute,
    contact_route_metadata,
)
from world_action_model.contact_transition_schema import (
    EQUAL_BIT_CODEC_VERSION,
    EQUAL_BIT_MIXING_MATRIX_SHA256,
    EQUAL_BIT_VALID_STATES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    contact_transport_capacity_metadata,
    unstructured_capacity_metadata,
)
from world_action_model.verified_io import read_verified_bytes, strict_json_loads


MATRIX_SCHEMA_VERSION = "contact-causal-production-matrix/v1"
MATRIX_ID = "contact-asym25-causal-main-q12-b16-stage2/v1"
MATRIX_REPORT_VERSION = "contact-asym25-causal-production-matrix/v2"
PRECURSOR_ID = "ego-contact-planner-precursor-q12/v1"
EGO_PRECURSOR_GATE_VERSION = "ego-discrete-contact-stage1-training-gate/v7"
ROBOT_GATE_VERSION = "contact-asym25-training-gate/v3"
ROBOT_GATE_REVIEW_VERSION = "contact-asym25-gate-reviewer-acceptance/v2"
PREFLIGHT_RECEIPT_VERSION = "contact-asym25-production-preflight-receipt/v2"
TRAINING_DATA_RECEIPT_VERSION = "contact-asym25-training-data-receipt/v2"
COLLECTOR_VERSION = "matched-control-hash-collector/v3"
SUPERVISION_VERSION = "contact_grouped_supervision_v2_zero_safe"
TRAINING_RECEIPT_VERSION = "contact-asym25-checkpoint-training-receipt/v5"
ACCEPTANCE_VERSION = "contact-asym25-checkpoint-independent-acceptance/v3"
CHECKPOINT_MANIFEST_VERSION = "contact-asym25-heldout-checkpoints/v3"
EVALUATOR_VERSION = "contact-asym25-heldout-evaluator/v2"
RESULT_VERSION = "contact-asym25-heldout-result/v3"
AGGREGATE_VERSION = "contact-asym25-heldout-aggregate/v5"

CAUSAL_ROUTES = (
    "aux_only_direct_q12",
    "equal_bit_unstructured_hard_st",
    "hard_st_semantic",
)
LEGACY_MECHANISM_ROUTES = ("soft_semantic", "hard_st", "hard_detached")
SEEDS = (6666, 7777, 8888)
CHECKPOINT_STEPS = (0, 100, 200)
STATES_PER_QUERY = 48_673
TOTAL_Q12_BITS = 186.85000894308362
SOURCE_TREE_RECEIPT_VERSION = "contact-causal-current-head-source-tree/v1"
SOURCE_LOGICAL_PATHS = (
    "configs/contact_causal_production_matrix.v1.json",
    "configs/giga_world_policy_0_5_ego_contact_causal_precursor.py",
    "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_causal_matrix_finetune.py",
    "world_action_model/contact_causal_production.py",
    "world_action_model/contact_causal_matched_control.py",
    "world_action_model/contact_causal_preflight.py",
    "world_action_model/contact_causal_checkpoint.py",
    "world_action_model/contact_causal_heldout_eval.py",
    "world_action_model/contact_ego_causal_precursor.py",
    "world_action_model/contact_route_contract.py",
    "world_action_model/contact_transition_schema.py",
    "world_action_model/contact_supervision_contract.py",
    "world_action_model/models/transformer_wa_casual_mot.py",
    "world_action_model/trainer/wa_casual_trainer_mot.py",
)

ROUTE_LOCAL_JSON_POINTERS = (
    "/route/causal_route_id",
    "/route/route_contract_sha256",
    "/route/action_attends_reference",
    "/route/action_uses_packet",
    "/route/action_visible_capacity",
    "/route/cache_policy",
    "/route/transport_receipt_sha256",
    "/model_config/contact_causal_route",
    "/model_config/transformer/contact_causal_route",
    "/model_config/transformer/contact_bottleneck",
    "/model_config/transformer/contact_semantic_bottleneck",
    "/run_identity/variant_id",
    "/run_identity/training_run_id",
    "/run_identity/training_run_nonce",
    "/run_identity/external_acceptance_sha256",
    "/output_path_record_only",
    "/measured_route_costs/forward_flops_proxy",
    "/measured_route_costs/backward_flops_proxy",
    "/measured_route_costs/active_gradient_parameter_count",
    "/measured_route_costs/peak_memory_bytes",
    "/measured_route_costs/step_time_ms",
    "/measured_route_costs/action_only_latency_ms",
    "/measured_route_costs/transport_flops_proxy",
    "/gradient_reachability_receipt/active_parameter_names_sha256",
    "/gradient_reachability_receipt/inactive_parameter_names_sha256",
    "/gradient_reachability_receipt/action_loss_reaches_planner",
    "/gradient_reachability_receipt/unused_parameter_names",
    "/parameter_topology/optimizer_state_names_sha256",
    "/parameter_topology/unused_parameter_names",
    "/parameter_topology/unused_parameter_names_sha256",
    "/checkpoint_artifact_hashes/config_sha256",
    "/checkpoint_artifact_hashes/whole_bundle_sha256",
    "/checkpoint_artifact_hashes/route_receipt_sha256",
    "/checkpoint_file/path_record_only",
)

CROSS_SEED_JSON_POINTERS = (
    "/seed",
    "/run_identity/training_run_id",
    "/run_identity/training_run_nonce",
    "/run_identity/external_acceptance_sha256",
    "/output_path_record_only",
    "/stable_contract/split_manifest_sha256",
    "/stable_contract/train_episode_ids_sha256",
    "/stable_contract/validation_episode_ids_sha256",
    "/stable_contract/heldout_episode_ids_sha256",
    "/stable_contract/train_only_norm_sha256",
    "/stable_contract/ordered_window_manifest_sha256",
    "/stable_contract/window_seed",
    "/stable_contract/noise_seed",
    "/stable_contract/training_seed",
    "/evidence/training_gate_sha256",
    "/evidence/production_preflight_receipt_sha256",
    "/evidence/training_data_receipt_sha256",
    "/evidence/matched_control_receipt_sha256",
    "/evidence/external_reviewer_acceptance_sha256",
    "/checkpoint_file/path_record_only",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)


class CausalProductionError(ValueError):
    """Raised when causal-main production evidence is incomplete or mixed."""


def canonical_json_bytes(value: object) -> bytes:
    _reject_nonfinite_recursive(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_nonfinite_recursive(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CausalProductionError("non-finite JSON number is forbidden")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite_recursive(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_nonfinite_recursive(child)


def strict_json_file(path: str | Path, *, name: str) -> tuple[dict[str, Any], str, bytes]:
    try:
        payload, digest, _ = read_verified_bytes(path)
        value = strict_json_loads(payload, name=name)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CausalProductionError(f"invalid strict JSON: {name}") from exc
    if not isinstance(value, dict):
        raise CausalProductionError(f"{name} must be a JSON object")
    _reject_nonfinite_recursive(value)
    return value, digest, payload


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise CausalProductionError(f"{name} must be a unique 16-128 character identity")
    return value


def _native_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalProductionError(f"{name} must be a native integer, not boolean")
    if minimum is not None and value < minimum:
        raise CausalProductionError(f"{name} is below its minimum")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalProductionError(f"{name} must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise CausalProductionError(f"{name} must be finite")
    return result


def validate_deadline(value: object, *, now: datetime | None = None) -> str:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        raise CausalProductionError("evidence expiry must be canonical UTC with Z suffix")
    try:
        deadline = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CausalProductionError("invalid evidence expiry") from exc
    current = now or datetime.now(timezone.utc)
    if deadline.tzinfo is None or deadline <= current:
        raise CausalProductionError("evidence has expired")
    return value


def route_packet_kind(route: str) -> str:
    if route == "aux_only_direct_q12":
        return "none_direct_reference"
    if route == "equal_bit_unstructured_hard_st":
        return "unstructured_code"
    if route == "hard_st_semantic":
        return "semantic_grammar"
    raise CausalProductionError(f"unknown causal-main route: {route!r}")


def build_production_route_receipt(route: str) -> dict[str, Any]:
    if route not in CAUSAL_ROUTES:
        raise CausalProductionError("causal route set is exact; aliases are forbidden")
    op38 = dict(
        contact_route_metadata(
            ContactCausalRoute(route), num_queries=12, time_bins=16
        )
    )
    if route == "aux_only_direct_q12":
        capacity = {
            "kind": "unmatched_high_bandwidth_continuous_reference",
            "equal_capacity_claimed": False,
            "equal_backward_flops_claimed": False,
            "exact_compute_matched_claimed": False,
        }
    elif route == "equal_bit_unstructured_hard_st":
        details = dict(unstructured_capacity_metadata())
        capacity = {
            "kind": "equal_bit_unstructured",
            "states_per_query": details["states_per_query"],
            "num_queries": details["num_queries"],
            "total_bits": details["total_bits"],
            "action_visible_width": details["code_width"],
            "details": details,
        }
    else:
        details = dict(
            contact_transport_capacity_metadata(mode="hard_st", time_bins=16, num_queries=12)
        )
        capacity = {
            "kind": "finite_semantic_grammar",
            "states_per_query": details["strict_valid_per_segment_states"],
            "num_queries": details["num_queries"],
            "total_bits": details["strict_valid_total_bits"],
            "action_visible_width": 22,
            "details": details,
        }
    payload = {
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "causal_route_id": route,
        "op38_route_contract_version": CONTACT_CAUSAL_ROUTE_CONTRACT_VERSION,
        "op38_route_receipt": op38,
        "op38_route_receipt_sha256": op38["receipt_sha256"],
        "planner_queries": 12,
        "planner_width": 22,
        "transport_mode": "hard_st",
        "planner_parameters_and_forward_aux_matched": True,
        "contact_loss_weight": 0.25,
        "supervision_version": SUPERVISION_VERSION,
        "action_attends_reference": route == "aux_only_direct_q12",
        "action_uses_packet": route != "aux_only_direct_q12",
        "packet_kind": route_packet_kind(route),
        "action_visible_capacity": capacity,
        "shared_action_condition_encoder": (
            None if route == "aux_only_direct_q12" else "22_to_action_width"
        ),
        "action_loss_reaches_planner": route != "aux_only_direct_q12",
        "direct_capacity_matched": False,
        "direct_backward_flops_matched": False,
        "oracle_allowed": False,
        "contact_condition_mode": "predicted",
        "contact_target_model_input": False,
        "contact_condition_mask_model_input": False,
        "hand_active_is_contact_ground_truth": False,
        "production_admission_source": "robot_gate_v3_plus_external_review",
    }
    if route != "aux_only_direct_q12":
        if (
            capacity.get("states_per_query") != STATES_PER_QUERY
            or capacity.get("total_bits") != TOTAL_Q12_BITS
        ):
            raise CausalProductionError("strict route capacity differs from Q12/B16 contract")
    return {**payload, "receipt_sha256": sha256_json(payload)}


def validate_matrix_manifest(value: object) -> dict[str, Any]:
    manifest = _exact(
        value,
        {
            "schema_version",
            "matrix_id",
            "stage",
            "precursor_id",
            "route_contract_version",
            "equal_bit_codec_version",
            "equal_bit_mixing_matrix_sha256",
            "routes",
            "route_receipts",
            "stable_json_pointer_policy",
            "route_local_json_pointers",
            "cross_seed_json_pointers",
            "versions",
            "source_logical_paths",
            "scientific_admission",
        },
        "causal production matrix manifest",
    )
    if (
        manifest["schema_version"] != MATRIX_SCHEMA_VERSION
        or manifest["matrix_id"] != MATRIX_ID
        or manifest["stage"] != "robot_stage2_only"
        or manifest["precursor_id"] != PRECURSOR_ID
        or manifest["route_contract_version"] != CONTACT_CAUSAL_ROUTE_CONTRACT_VERSION
        or manifest["equal_bit_codec_version"] != EQUAL_BIT_CODEC_VERSION
        or manifest["equal_bit_mixing_matrix_sha256"] != EQUAL_BIT_MIXING_MATRIX_SHA256
        or manifest["routes"] != list(CAUSAL_ROUTES)
        or manifest["stable_json_pointer_policy"]
        != "replace_only_exact_allowlisted_leaves_future_fields_stable_by_default"
        or manifest["route_local_json_pointers"] != list(ROUTE_LOCAL_JSON_POINTERS)
        or manifest["cross_seed_json_pointers"] != list(CROSS_SEED_JSON_POINTERS)
    ):
        raise CausalProductionError("causal production matrix identity differs")
    receipts = manifest["route_receipts"]
    if not isinstance(receipts, dict) or list(receipts) != list(CAUSAL_ROUTES):
        raise CausalProductionError("matrix route receipt set/order differs")
    for route in CAUSAL_ROUTES:
        if receipts[route] != build_production_route_receipt(route):
            raise CausalProductionError(f"matrix route receipt differs: {route}")
    expected_versions = {
        "ego_precursor_gate": EGO_PRECURSOR_GATE_VERSION,
        "robot_gate": ROBOT_GATE_VERSION,
        "robot_gate_review": ROBOT_GATE_REVIEW_VERSION,
        "preflight_receipt": PREFLIGHT_RECEIPT_VERSION,
        "training_data_receipt": TRAINING_DATA_RECEIPT_VERSION,
        "matched_collector": COLLECTOR_VERSION,
        "supervision": SUPERVISION_VERSION,
        "checkpoint_training_receipt": TRAINING_RECEIPT_VERSION,
        "checkpoint_acceptance": ACCEPTANCE_VERSION,
        "checkpoint_manifest": CHECKPOINT_MANIFEST_VERSION,
        "heldout_evaluator": EVALUATOR_VERSION,
        "heldout_result": RESULT_VERSION,
        "aggregate": AGGREGATE_VERSION,
        "matrix_report": MATRIX_REPORT_VERSION,
    }
    if manifest["versions"] != expected_versions:
        raise CausalProductionError("matrix evidence-chain versions differ")
    if (
        manifest["source_logical_paths"] != list(SOURCE_LOGICAL_PATHS)
        or not manifest["source_logical_paths"]
        or len(set(manifest["source_logical_paths"]))
        != len(manifest["source_logical_paths"])
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in manifest["source_logical_paths"]
        )
    ):
        raise CausalProductionError("matrix source logical paths are unsafe")
    if manifest["scientific_admission"] != {
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
        "oracle_is_separate_upper_bound": True,
    }:
        raise CausalProductionError("matrix manifest must remain scientifically unadmitted")
    return manifest


def _replace_pointer(root: dict[str, Any], pointer: str, replacement: str) -> None:
    if not pointer.startswith("/") or "~" in pointer:
        raise CausalProductionError(f"unsupported JSON pointer: {pointer}")
    parts = pointer[1:].split("/")
    cursor: Any = root
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise CausalProductionError(f"allowlisted JSON pointer is absent: {pointer}")
        cursor = cursor[part]
    leaf = parts[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise CausalProductionError(f"allowlisted JSON pointer is absent: {pointer}")
    cursor[leaf] = replacement


def canonicalize_exact_pointers(
    value: Mapping[str, Any], pointers: Sequence[str], *, namespace: str
) -> dict[str, Any]:
    try:
        normalized = deepcopy(json.loads(canonical_json_bytes(value).decode("utf-8")))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CausalProductionError("contract is not strict canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise CausalProductionError("normalized contract must be an object")
    if len(set(pointers)) != len(pointers):
        raise CausalProductionError("normalization allowlist contains duplicates")
    for pointer in pointers:
        _replace_pointer(normalized, pointer, f"__{namespace}:{pointer}__")
    return normalized


def compare_same_seed_route_contracts(values: Sequence[Mapping[str, Any]]) -> str:
    if len(values) != 3:
        raise CausalProductionError("same-seed comparison requires exactly three routes")
    routes = [value.get("route", {}).get("causal_route_id") for value in values]
    if set(routes) != set(CAUSAL_ROUTES) or len(set(routes)) != 3:
        raise CausalProductionError("same-seed route set is incomplete or contains aliases")
    normalized = [
        canonicalize_exact_pointers(
            value, ROUTE_LOCAL_JSON_POINTERS, namespace="route_local"
        )
        for value in values
    ]
    digests = {sha256_json(value) for value in normalized}
    if len(digests) != 1:
        raise CausalProductionError("same-seed routes differ outside exact route allowlist")
    return next(iter(digests))


def compare_cross_seed_contracts(values: Sequence[Mapping[str, Any]]) -> str:
    if len(values) != 3:
        raise CausalProductionError("cross-seed comparison requires exactly three seeds")
    if {value.get("seed") for value in values} != set(SEEDS):
        raise CausalProductionError("cross-seed set differs from 6666/7777/8888")
    normalized = [
        canonicalize_exact_pointers(
            value, CROSS_SEED_JSON_POINTERS, namespace="seed_derived"
        )
        for value in values
    ]
    digests = {sha256_json(value) for value in normalized}
    if len(digests) != 1:
        raise CausalProductionError("seeds differ outside exact seed-derived allowlist")
    return next(iter(digests))


def validate_dataset_supervision_coverage(value: object) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "receipt_version",
            "dataset_id",
            "split_seed",
            "contact_schema_version",
            "label_origin",
            "review_status",
            "hand_active_used_as_contact_ground_truth",
            "grip_used_as_contact_ground_truth",
            "total_windows",
            "supervised_windows",
            "zero_supervision_windows",
            "total_grouped_eligible_items",
            "positive_contact_event_count",
            "field_eligible_counts",
            "phase_coverage",
            "coverage_source_sha256",
            "sidecar_bundle_sha256",
        },
        "dataset supervision coverage receipt",
    )
    if (
        receipt["receipt_version"] != "contact-dataset-supervision-coverage/v2"
        or receipt["split_seed"] not in SEEDS
        or receipt["contact_schema_version"] != SEGMENT_SCHEMA_VERSION
        or receipt["label_origin"] != "human_adjudicated_robot_native"
        or receipt["review_status"] != "reviewed"
        or receipt["hand_active_used_as_contact_ground_truth"] is not False
        or receipt["grip_used_as_contact_ground_truth"] is not False
    ):
        raise CausalProductionError("dataset coverage does not prove reviewed robot-native labels")
    for field in (
        "total_windows",
        "supervised_windows",
        "zero_supervision_windows",
        "total_grouped_eligible_items",
        "positive_contact_event_count",
    ):
        _native_int(receipt[field], field, minimum=0)
    if (
        receipt["total_windows"] <= 0
        or receipt["supervised_windows"] <= 0
        or receipt["total_grouped_eligible_items"] <= 0
        or receipt["positive_contact_event_count"] <= 0
        or receipt["supervised_windows"] + receipt["zero_supervision_windows"]
        != receipt["total_windows"]
    ):
        raise CausalProductionError("dataset-level positive contact coverage is insufficient")
    if not isinstance(receipt["field_eligible_counts"], dict) or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in receipt["field_eligible_counts"].values()
    ):
        raise CausalProductionError("field coverage counts are invalid")
    if not isinstance(receipt["phase_coverage"], dict) or not receipt["phase_coverage"]:
        raise CausalProductionError("phase coverage is missing")
    _sha(receipt["coverage_source_sha256"], "coverage source")
    _sha(receipt["sidecar_bundle_sha256"], "sidecar bundle")
    return receipt


def validate_parameter_topology(value: object, *, route: str, step: int) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "persistent_parameter_schema_sha256",
            "persistent_buffer_schema_sha256",
            "parameter_tensor_sha256",
            "buffer_tensor_sha256",
            "step0_parameter_tensor_sha256",
            "optimizer_group_names_sha256",
            "optimizer_hyperparameters_sha256",
            "optimizer_state_names_sha256",
            "scheduler_state_sha256",
            "equal_bit_mixing_persistent",
            "unused_parameter_names",
            "unused_parameter_names_sha256",
            "unused_parameters_have_grad",
            "unused_parameters_have_optimizer_state",
            "unused_parameters_changed_from_step0",
        },
        "parameter topology receipt",
    )
    for field in (
        "persistent_parameter_schema_sha256",
        "persistent_buffer_schema_sha256",
        "parameter_tensor_sha256",
        "buffer_tensor_sha256",
        "step0_parameter_tensor_sha256",
        "optimizer_group_names_sha256",
        "optimizer_hyperparameters_sha256",
        "optimizer_state_names_sha256",
        "scheduler_state_sha256",
        "unused_parameter_names_sha256",
    ):
        _sha(receipt[field], field)
    if receipt["equal_bit_mixing_persistent"] is not False:
        raise CausalProductionError("equal-bit mixing matrix must be nonpersistent")
    unused = receipt["unused_parameter_names"]
    if not isinstance(unused, list) or any(not isinstance(name, str) or not name for name in unused):
        raise CausalProductionError("unused parameter allowlist is invalid")
    if unused != sorted(set(unused)) or sha256_json(unused) != receipt["unused_parameter_names_sha256"]:
        raise CausalProductionError("unused parameter allowlist digest differs")
    if route == "aux_only_direct_q12":
        if not unused or any(
            not name.startswith("contact_condition_encoder.") for name in unused
        ):
            raise CausalProductionError("direct unused set must be exactly contact_condition_encoder.*")
        if (
            receipt["unused_parameters_have_grad"] is not False
            or receipt["unused_parameters_have_optimizer_state"] is not False
            or receipt["unused_parameters_changed_from_step0"] is not False
        ):
            raise CausalProductionError("direct unused parameters acquired grad/state/drift")
    elif unused:
        raise CausalProductionError("strict routes must not declare route-local unused parameters")
    if step == 0 and receipt["parameter_tensor_sha256"] != receipt["step0_parameter_tensor_sha256"]:
        raise CausalProductionError("step-0 parameter tensor digest differs from itself")
    return receipt


def validate_training_receipt_v5(value: object) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "receipt_version",
            "matrix_schema_version",
            "matrix_id",
            "seed",
            "checkpoint_step",
            "route",
            "run_identity",
            "output_path_record_only",
            "model_config",
            "stable_contract",
            "training_contract",
            "precursor",
            "evidence",
            "topology",
            "parameter_topology",
            "measured_route_costs",
            "gradient_reachability_receipt",
            "checkpoint_artifact_hashes",
            "checkpoint_file",
            "parent_checkpoint_weight_sha256",
            "scientific_admission",
        },
        "causal checkpoint training receipt",
    )
    route_entry = _exact(
        receipt["route"],
        {
            "causal_route_id",
            "route_contract_sha256",
            "action_attends_reference",
            "action_uses_packet",
            "action_visible_capacity",
            "cache_policy",
            "transport_receipt_sha256",
        },
        "checkpoint route",
    )
    route = route_entry["causal_route_id"]
    expected_route = build_production_route_receipt(route)
    if (
        receipt["receipt_version"] != TRAINING_RECEIPT_VERSION
        or receipt["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or receipt["matrix_id"] != MATRIX_ID
        or receipt["seed"] not in SEEDS
        or receipt["checkpoint_step"] not in CHECKPOINT_STEPS
        or route_entry["route_contract_sha256"] != expected_route["receipt_sha256"]
        or route_entry["action_attends_reference"] != expected_route["action_attends_reference"]
        or route_entry["action_uses_packet"] != expected_route["action_uses_packet"]
        or route_entry["action_visible_capacity"] != expected_route["action_visible_capacity"]
        or route_entry["transport_receipt_sha256"]
        != expected_route["op38_route_receipt_sha256"]
    ):
        raise CausalProductionError("checkpoint matrix/route identity differs")
    _native_int(receipt["seed"], "seed")
    _native_int(receipt["checkpoint_step"], "checkpoint step", minimum=0)
    run = _exact(
        receipt["run_identity"],
        {
            "variant_id",
            "training_run_id",
            "training_run_nonce",
            "external_acceptance_sha256",
        },
        "run identity",
    )
    if run["variant_id"] != route:
        raise CausalProductionError("variant ID must equal the exact causal route")
    _identity(run["training_run_id"], "training run ID")
    _identity(run["training_run_nonce"], "training run nonce")
    if run["training_run_id"] == run["training_run_nonce"]:
        raise CausalProductionError("training run ID and nonce must be distinct")
    _sha(run["external_acceptance_sha256"], "external acceptance")
    model = _exact(
        receipt["model_config"],
        {"contact_causal_route", "transformer"},
        "checkpoint model config",
    )
    transformer = _exact(
        model["transformer"],
        {
            "contact_causal_route",
            "contact_bottleneck",
            "contact_semantic_bottleneck",
            "contact_semantic_transport_mode",
            "num_contact_tokens",
            "contact_target_dim",
            "contact_time_bins",
        },
        "checkpoint transformer config",
    )
    direct = route == "aux_only_direct_q12"
    if (
        model["contact_causal_route"] != route
        or transformer["contact_causal_route"] != route
        or transformer["contact_bottleneck"] is not (not direct)
        or transformer["contact_semantic_bottleneck"] is not (not direct)
        or transformer["contact_semantic_transport_mode"] != "hard_st"
        or transformer["num_contact_tokens"] != 12
        or transformer["contact_target_dim"] != 22
        or transformer["contact_time_bins"] != 16
    ):
        raise CausalProductionError("checkpoint model route config differs")
    stable = receipt["stable_contract"]
    required_stable = {
        "git_commit",
        "source_tree_sha256",
        "matrix_manifest_sha256",
        "precursor_bundle_sha256",
        "base_weights_sha256",
        "initial_parameter_tensor_sha256",
        "persistent_parameter_schema_sha256",
        "dataset_id",
        "dataset_tree_sha256",
        "dataset_info_sha256",
        "hardware_schema_id",
        "hardware_contract_sha256",
        "calibration_capture_sha256",
        "split_manifest_sha256",
        "train_episode_ids_sha256",
        "validation_episode_ids_sha256",
        "heldout_episode_ids_sha256",
        "sidecar_bundle_sha256",
        "contact_label_coverage_sha256",
        "contact_target_stream_sha256",
        "contact_mask_stream_sha256",
        "contact_confidence_stream_sha256",
        "action_target_stream_sha256",
        "action_mask_stream_sha256",
        "prompt_t5_stream_sha256",
        "train_only_norm_sha256",
        "transform_config_sha256",
        "augmentation_trace_sha256",
        "ordered_window_manifest_sha256",
        "visual_timestep_stream_sha256",
        "action_timestep_stream_sha256",
        "visual_noise_stream_sha256",
        "action_noise_stream_sha256",
        "pre_forward_rng_stream_sha256",
        "dropout_config_sha256",
        "optimizer_config_sha256",
        "optimizer_parameter_groups_sha256",
        "optimizer_step0_state_sha256",
        "scheduler_config_sha256",
        "scheduler_step0_state_sha256",
        "precision",
        "world_size",
        "rank_set",
        "window_seed",
        "noise_seed",
        "training_seed",
    }
    _exact(stable, required_stable, "stable contract")
    if not isinstance(stable["git_commit"], str) or not _GIT_RE.fullmatch(stable["git_commit"]):
        raise CausalProductionError("stable contract requires current full Git HEAD")
    for field, value in stable.items():
        if field.endswith("_sha256"):
            _sha(value, f"stable {field}")
    if (
        stable["hardware_schema_id"] != "sonic_g1_rh56dftp6_omnipicker1_v1"
        or stable["precision"] != "bfloat16"
        or stable["world_size"] <= 0
        or stable["rank_set"] != list(range(stable["world_size"]))
        or any(isinstance(stable[name], bool) or not isinstance(stable[name], int) for name in ("window_seed", "noise_seed", "training_seed"))
    ):
        raise CausalProductionError("stable hardware/precision/topology/seed contract differs")
    training = _exact(
        receipt["training_contract"],
        {
            "contact_schema_version",
            "supervision_version",
            "contact_loss_weight",
            "contact_condition_mode",
            "contact_target_model_input",
            "contact_condition_mask_model_input",
            "oracle_used",
            "hand_active_used_as_contact_ground_truth",
            "grip_used_as_contact_ground_truth",
            "zero_supervision_batch_policy",
            "dataset_level_coverage_required",
            "checkpoint_steps",
            "primary_checkpoint_step",
            "best_checkpoint_selection_allowed",
        },
        "training contract",
    )
    if (
        training["contact_schema_version"] != SEGMENT_SCHEMA_VERSION
        or training["supervision_version"] != SUPERVISION_VERSION
        or _finite_number(training["contact_loss_weight"], "contact loss weight") != 0.25
        or training["contact_condition_mode"] != "predicted"
        or training["contact_target_model_input"] is not False
        or training["contact_condition_mask_model_input"] is not False
        or training["oracle_used"] is not False
        or training["hand_active_used_as_contact_ground_truth"] is not False
        or training["grip_used_as_contact_ground_truth"] is not False
        or training["zero_supervision_batch_policy"] != "canonical_positive_zero_no_planner_gradient"
        or training["dataset_level_coverage_required"] is not True
        or training["checkpoint_steps"] != list(CHECKPOINT_STEPS)
        or training["primary_checkpoint_step"] != 200
        or training["best_checkpoint_selection_allowed"] is not False
    ):
        raise CausalProductionError("training supervision/oracle contract differs")
    precursor = _exact(
        receipt["precursor"],
        {
            "precursor_id",
            "receipt_bytes_sha256",
            "bundle_sha256",
            "parameter_tensor_sha256",
            "action_path_scientific_input_output",
            "route_neutral",
        },
        "precursor binding",
    )
    if (
        precursor["precursor_id"] != PRECURSOR_ID
        or precursor["action_path_scientific_input_output"] is not False
        or precursor["route_neutral"] is not True
        or stable["precursor_bundle_sha256"] != precursor["bundle_sha256"]
        or stable["initial_parameter_tensor_sha256"] != precursor["parameter_tensor_sha256"]
    ):
        raise CausalProductionError("route-neutral precursor binding differs")
    for field in ("receipt_bytes_sha256", "bundle_sha256", "parameter_tensor_sha256"):
        _sha(precursor[field], f"precursor {field}")
    evidence = _exact(
        receipt["evidence"],
        {
            "training_gate_sha256",
            "production_preflight_receipt_sha256",
            "training_data_receipt_sha256",
            "matched_control_receipt_sha256",
            "external_reviewer_acceptance_sha256",
            "contact_coverage_receipt_sha256",
            "trace_manifest_sha256",
            "current_head_source_receipt_sha256",
        },
        "checkpoint evidence",
    )
    for field, value in evidence.items():
        _sha(value, f"evidence {field}")
    topology = _exact(receipt["topology"], {"world_size", "rank_set"}, "runtime topology")
    if topology != {"world_size": stable["world_size"], "rank_set": stable["rank_set"]}:
        raise CausalProductionError("runtime topology differs from stable contract")
    validate_parameter_topology(
        receipt["parameter_topology"], route=route, step=receipt["checkpoint_step"]
    )
    costs = _exact(
        receipt["measured_route_costs"],
        {
            "forward_flops_proxy",
            "backward_flops_proxy",
            "active_gradient_parameter_count",
            "peak_memory_bytes",
            "step_time_ms",
            "action_only_latency_ms",
            "transport_flops_proxy",
        },
        "measured route costs",
    )
    for field, value in costs.items():
        _finite_number(value, f"measured route cost {field}")
    reachability = _exact(
        receipt["gradient_reachability_receipt"],
        {
            "active_parameter_names_sha256",
            "inactive_parameter_names_sha256",
            "action_loss_reaches_planner",
            "unused_parameter_names",
        },
        "gradient reachability receipt",
    )
    _sha(reachability["active_parameter_names_sha256"], "active parameter names")
    _sha(reachability["inactive_parameter_names_sha256"], "inactive parameter names")
    if (
        reachability["action_loss_reaches_planner"] is not (not direct)
        or reachability["unused_parameter_names"]
        != receipt["parameter_topology"]["unused_parameter_names"]
    ):
        raise CausalProductionError("gradient reachability differs from route contract")
    artifact_hashes = _exact(
        receipt["checkpoint_artifact_hashes"],
        {
            "weight_tensor_sha256",
            "buffer_tensor_sha256",
            "config_sha256",
            "whole_bundle_sha256",
            "route_receipt_sha256",
        },
        "checkpoint artifact hashes",
    )
    for field, value in artifact_hashes.items():
        _sha(value, f"checkpoint artifact {field}")
    if (
        artifact_hashes["weight_tensor_sha256"]
        != receipt["parameter_topology"]["parameter_tensor_sha256"]
        or artifact_hashes["buffer_tensor_sha256"]
        != receipt["parameter_topology"]["buffer_tensor_sha256"]
        or artifact_hashes["route_receipt_sha256"] != expected_route["receipt_sha256"]
    ):
        raise CausalProductionError("checkpoint artifact hash mirrors differ")
    checkpoint_file = _exact(
        receipt["checkpoint_file"],
        {"path_record_only", "sha256", "size_bytes"},
        "checkpoint file receipt",
    )
    _sha(checkpoint_file["sha256"], "checkpoint file")
    _native_int(checkpoint_file["size_bytes"], "checkpoint size", minimum=1)
    if not isinstance(checkpoint_file["path_record_only"], str) or not Path(checkpoint_file["path_record_only"]).is_absolute():
        raise CausalProductionError("checkpoint file path must be absolute")
    _sha(receipt["parent_checkpoint_weight_sha256"], "parent checkpoint weight")
    if receipt["scientific_admission"] != {
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }:
        raise CausalProductionError("checkpoint receipt must not self-authorize science")
    return receipt


def verify_checkpoint_file(receipt: Mapping[str, Any]) -> None:
    entry = receipt["checkpoint_file"]
    path = Path(entry["path_record_only"])
    try:
        payload, digest, stat_record = read_verified_bytes(
            path, expected_sha256=entry["sha256"]
        )
    except (OSError, ValueError) as exc:
        raise CausalProductionError("checkpoint file bytes are unavailable or changed") from exc
    if not payload or len(payload) != entry["size_bytes"] or stat_record["size_bytes"] <= 0 or digest != entry["sha256"]:
        raise CausalProductionError("checkpoint file must be real and non-empty")


def named_tensor_digest(rows: Sequence[tuple[str, Any]]) -> str:
    """Hash exact named tensor dtype/shape/bytes without pickle metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(rows, key=lambda item: item[0]):
        if not isinstance(name, str) or not name:
            raise CausalProductionError("named tensor digest requires non-empty names")
        try:
            value = tensor.detach().to(device="cpu").contiguous()
            raw = value.view(__import__("torch").uint8).numpy().tobytes(order="C")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise CausalProductionError("named tensor digest received a non-tensor") from exc
        for payload in (
            name.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
            raw,
        ):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def model_parameter_and_buffer_digests(model: Any) -> tuple[str, str]:
    return (
        named_tensor_digest(list(model.named_parameters())),
        named_tensor_digest(list(model.named_buffers())),
    )


def build_checkpoint_training_receipt_v5(
    *,
    base_receipt: Mapping[str, Any],
    checkpoint_step: int,
    checkpoint_file_path: str | Path,
    parameter_tensor_sha256: str,
    buffer_tensor_sha256: str,
    config_sha256: str,
    whole_bundle_sha256: str,
    parent_checkpoint_weight_sha256: str,
) -> dict[str, Any]:
    """Fill only checkpoint-local leaves of a reviewed v5 base receipt."""

    receipt = deepcopy(dict(base_receipt))
    if checkpoint_step not in CHECKPOINT_STEPS or isinstance(checkpoint_step, bool):
        raise CausalProductionError("causal checkpoint step must be 0/100/200")
    path = Path(checkpoint_file_path).expanduser().absolute()
    try:
        file_sha256, stat_record = __import__(
            "world_action_model.verified_io", fromlist=["hash_verified_regular_file"]
        ).hash_verified_regular_file(path)
    except (OSError, ValueError) as exc:
        raise CausalProductionError("cannot hash causal checkpoint weight file") from exc
    if stat_record["size_bytes"] <= 0:
        raise CausalProductionError("causal checkpoint weight file is empty")
    for value, name in (
        (parameter_tensor_sha256, "parameter tensor"),
        (buffer_tensor_sha256, "buffer tensor"),
        (config_sha256, "checkpoint config"),
        (whole_bundle_sha256, "whole checkpoint bundle"),
        (parent_checkpoint_weight_sha256, "parent checkpoint weight"),
    ):
        _sha(value, name)
    receipt["checkpoint_step"] = checkpoint_step
    receipt["output_path_record_only"] = str(path.parent)
    receipt["checkpoint_file"] = {
        "path_record_only": str(path),
        "sha256": file_sha256,
        "size_bytes": stat_record["size_bytes"],
    }
    receipt["parent_checkpoint_weight_sha256"] = parent_checkpoint_weight_sha256
    receipt["parameter_topology"] = dict(receipt["parameter_topology"])
    receipt["parameter_topology"].update(
        parameter_tensor_sha256=parameter_tensor_sha256,
        buffer_tensor_sha256=buffer_tensor_sha256,
    )
    receipt["checkpoint_artifact_hashes"] = dict(
        receipt["checkpoint_artifact_hashes"]
    )
    receipt["checkpoint_artifact_hashes"].update(
        weight_tensor_sha256=parameter_tensor_sha256,
        buffer_tensor_sha256=buffer_tensor_sha256,
        config_sha256=config_sha256,
        whole_bundle_sha256=whole_bundle_sha256,
        route_receipt_sha256=receipt["route"]["route_contract_sha256"],
    )
    validate_training_receipt_v5(receipt)
    return receipt


def write_canonical_json_exclusive(path: str | Path, value: object) -> str:
    """Write canonical JSON with exclusive/no-follow semantics."""

    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while persisting canonical evidence")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(payload).hexdigest()


def validate_acceptance_v3(value: object, *, receipt: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = _exact(
        value,
        {
            "acceptance_version",
            "status",
            "accepted",
            "matrix_id",
            "causal_route_id",
            "seed",
            "checkpoint_step",
            "training_receipt_sha256",
            "checkpoint_file_sha256",
            "reviewer_identity",
            "reviewer_attestation_nonce",
            "acceptance_run_id",
            "acceptance_run_nonce",
            "reviewed_current_head_sha256",
            "reviewed_source_tree_sha256",
            "evidence_valid_until_utc",
            "oracle_used",
        },
        "causal checkpoint independent acceptance",
    )
    validate_training_receipt_v5(receipt)
    if (
        acceptance["acceptance_version"] != ACCEPTANCE_VERSION
        or acceptance["status"] != "accepted"
        or acceptance["accepted"] is not True
        or acceptance["matrix_id"] != MATRIX_ID
        or acceptance["causal_route_id"] != receipt["route"]["causal_route_id"]
        or acceptance["seed"] != receipt["seed"]
        or acceptance["checkpoint_step"] != receipt["checkpoint_step"]
        or acceptance["training_receipt_sha256"] != sha256_json(receipt)
        or acceptance["checkpoint_file_sha256"] != receipt["checkpoint_file"]["sha256"]
        or acceptance["reviewed_current_head_sha256"] != receipt["stable_contract"]["git_commit"]
        or acceptance["reviewed_source_tree_sha256"] != receipt["stable_contract"]["source_tree_sha256"]
        or acceptance["oracle_used"] is not False
    ):
        raise CausalProductionError("independent checkpoint acceptance binding differs")
    identities = [
        acceptance["reviewer_identity"],
        acceptance["reviewer_attestation_nonce"],
        acceptance["acceptance_run_id"],
        acceptance["acceptance_run_nonce"],
        receipt["run_identity"]["training_run_id"],
        receipt["run_identity"]["training_run_nonce"],
    ]
    for index, item in enumerate(identities):
        _identity(item, f"acceptance identity {index}")
    if len(set(identities)) != len(identities):
        raise CausalProductionError("reviewer/acceptance/training identities overlap")
    validate_deadline(acceptance["evidence_valid_until_utc"])
    return acceptance


def validate_27_checkpoint_matrix(
    evidence: Mapping[
        tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    expected = {
        (seed, route, step)
        for seed in SEEDS
        for route in CAUSAL_ROUTES
        for step in CHECKPOINT_STEPS
    }
    if set(evidence) != expected:
        raise CausalProductionError("causal matrix requires exactly 27 seed/route/step entries")
    all_identities: set[str] = set()
    acceptance_hashes: set[str] = set()
    precursor_receipt_bytes: set[str] = set()
    precursor_bundles: set[str] = set()
    step0_weights: set[str] = set()
    common_by_seed: dict[str, str] = {}
    common_by_route: dict[str, str] = {}
    representatives: dict[tuple[int, str], Mapping[str, Any]] = {}
    for key in sorted(expected):
        seed, route, step = key
        receipt, acceptance = evidence[key]
        validate_training_receipt_v5(receipt)
        validate_acceptance_v3(acceptance, receipt=receipt)
        if (
            receipt["seed"] != seed
            or receipt["route"]["causal_route_id"] != route
            or receipt["checkpoint_step"] != step
        ):
            raise CausalProductionError("matrix key differs from embedded receipt identity")
        if verify_files:
            verify_checkpoint_file(receipt)
        receipt_ids = {
            receipt["run_identity"]["training_run_id"],
            receipt["run_identity"]["training_run_nonce"],
        }
        acceptance_ids = {
            acceptance["reviewer_identity"],
            acceptance["reviewer_attestation_nonce"],
            acceptance["acceptance_run_id"],
            acceptance["acceptance_run_nonce"],
        }
        if step == 0:
            if all_identities.intersection(receipt_ids):
                raise CausalProductionError("training run nonce or identity is globally reused")
            all_identities.update(receipt_ids)
            step0_weights.add(receipt["checkpoint_artifact_hashes"]["weight_tensor_sha256"])
        if all_identities.intersection(acceptance_ids):
            raise CausalProductionError("reviewer nonce or acceptance identity is globally reused")
        all_identities.update(acceptance_ids)
        acceptance_hashes.add(sha256_json(acceptance))
        precursor_receipt_bytes.add(receipt["precursor"]["receipt_bytes_sha256"])
        precursor_bundles.add(receipt["precursor"]["bundle_sha256"])
        representatives.setdefault((seed, route), receipt)
    if len(acceptance_hashes) != 27:
        raise CausalProductionError("all 27 independent acceptances must be byte-distinct")
    if len(precursor_receipt_bytes) != 1 or len(precursor_bundles) != 1:
        raise CausalProductionError("all nine runs must bind byte-identical precursor evidence")
    if len(step0_weights) != 1:
        raise CausalProductionError("all nine step-0 parameter tensor digests must match")
    for seed in SEEDS:
        same_seed = [representatives[(seed, route)] for route in CAUSAL_ROUTES]
        common_by_seed[str(seed)] = compare_same_seed_route_contracts(same_seed)
        strict = [
            representatives[(seed, "equal_bit_unstructured_hard_st")],
            representatives[(seed, "hard_st_semantic")],
        ]
        left, right = strict
        for field in (
            "persistent_parameter_schema_sha256",
            "optimizer_group_names_sha256",
            "optimizer_hyperparameters_sha256",
        ):
            if left["parameter_topology"][field] != right["parameter_topology"][field]:
                raise CausalProductionError("equal-bit and semantic parameter/optimizer topology differs")
    for route in CAUSAL_ROUTES:
        cross_seed = [representatives[(seed, route)] for seed in SEEDS]
        common_by_route[route] = compare_cross_seed_contracts(cross_seed)
    for seed in SEEDS:
        for route in CAUSAL_ROUTES:
            rows = [evidence[(seed, route, step)][0] for step in CHECKPOINT_STEPS]
            if rows[0]["parent_checkpoint_weight_sha256"] != rows[0]["precursor"]["parameter_tensor_sha256"]:
                raise CausalProductionError("step-0 parent must be the precursor tensor digest")
            if rows[1]["parent_checkpoint_weight_sha256"] != rows[0]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]:
                raise CausalProductionError("step-100 parent chain differs")
            if rows[2]["parent_checkpoint_weight_sha256"] != rows[1]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]:
                raise CausalProductionError("step-200 parent chain differs")
    return {
        "matrix_report_version": MATRIX_REPORT_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "routes": list(CAUSAL_ROUTES),
        "seeds": list(SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_acceptance_count": 27,
        "precursor_receipt_bytes_sha256": next(iter(precursor_receipt_bytes)),
        "precursor_bundle_sha256": next(iter(precursor_bundles)),
        "common_step0_weight_tensor_sha256": next(iter(step0_weights)),
        "same_seed_common_contract_sha256": common_by_seed,
        "cross_seed_common_contract_sha256": common_by_route,
        "independent_acceptance_sha256": sorted(acceptance_hashes),
        "primary_checkpoint_step": 200,
        "benefit_claimed": False,
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
    }


def current_git_head(repository_root: str | Path) -> str:
    """Return a full HEAD without silently accepting dirty/source drift."""

    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repository_root).absolute()), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CausalProductionError("cannot resolve current Git HEAD") from exc
    head = result.stdout.strip()
    if not _GIT_RE.fullmatch(head):
        raise CausalProductionError("current Git HEAD is not a full commit")
    return head


def current_head_source_tree_receipt(
    repository_root: str | Path, *, expected_commit: str
) -> dict[str, Any]:
    """Bind the exact committed/runtime bytes of every causal production source."""

    from world_action_model.verified_io import verified_head_code_manifest

    try:
        commit, rows = verified_head_code_manifest(
            repository_root,
            SOURCE_LOGICAL_PATHS,
            expected_commit=expected_commit,
        )
    except (OSError, ValueError) as exc:
        raise CausalProductionError(
            "causal production runtime source differs from current HEAD"
        ) from exc
    receipt = {
        "receipt_version": SOURCE_TREE_RECEIPT_VERSION,
        "git_commit": commit,
        "source_logical_paths": list(SOURCE_LOGICAL_PATHS),
        "source_files": rows,
    }
    canonical_json_bytes(receipt)
    return receipt


def ensure_exact_causal_route(route: object) -> str:
    if not isinstance(route, str) or route not in CAUSAL_ROUTES:
        raise CausalProductionError(
            f"route must be exactly one of {CAUSAL_ROUTES}; aliases and legacy routes are forbidden"
        )
    return route


def ensure_not_project_tmp(path: str | Path, *, project_root: str | Path) -> Path:
    target = Path(path).expanduser().absolute()
    root = Path(project_root).expanduser().absolute()
    if target == Path("/tmp") or Path("/tmp") in target.parents:
        raise CausalProductionError("causal production artifacts must not use /tmp")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise CausalProductionError("causal production artifact must stay under project root") from exc
    return target

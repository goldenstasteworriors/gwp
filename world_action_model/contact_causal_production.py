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
DIRECT_UNUSED_PARAMETER_NAMES = (
    "contact_condition_encoder.bias",
    "contact_condition_encoder.weight",
)
SOURCE_TREE_RECEIPT_VERSION = "contact-causal-current-head-source-tree/v1"
SOURCE_LOGICAL_PATHS = (
    "configs/contact_causal_production_matrix.v1.json",
    "configs/giga_world_policy_0_5_ego_contact_pretrain.py",
    "configs/giga_world_policy_0_5_ego_contact_causal_precursor.py",
    "configs/giga_world_policy_0_5_sonic_g1_finetune.py",
    "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_finetune.py",
    "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_causal_matrix_finetune.py",
    "world_action_model/asymmetric_action_schema.py",
    "world_action_model/asymmetric_hardware_manifest.py",
    "world_action_model/contact_asym25_eval_runtime.py",
    "world_action_model/contact_asym25_heldout_eval.py",
    "world_action_model/contact_asym25_preflight.py",
    "world_action_model/contact_causal_production.py",
    "world_action_model/contact_causal_gate_bootstrap.py",
    "world_action_model/contact_causal_coverage.py",
    "world_action_model/contact_causal_matched_control.py",
    "world_action_model/contact_causal_preflight.py",
    "world_action_model/contact_causal_checkpoint.py",
    "world_action_model/contact_causal_eval_runtime.py",
    "world_action_model/contact_causal_heldout_eval.py",
    "world_action_model/contact_ego_causal_precursor.py",
    "world_action_model/contact_ego_stage1_preflight.py",
    "world_action_model/contact_route_contract.py",
    "world_action_model/contact_transition_schema.py",
    "world_action_model/contact_supervision_contract.py",
    "world_action_model/datasets/wam_lerobot_dataset.py",
    "world_action_model/models/transformer_wa_casual_mot.py",
    "world_action_model/transforms/wa_transforms_lerobot_pretrain.py",
    "world_action_model/trainer/wa_casual_trainer_mot.py",
    "world_action_model/verified_io.py",
    "third_party/giga-train/giga_train/trainers/trainer.py",
    "third_party/giga-datasets/giga_datasets/datasets/lerobot_dataset.py",
    "scripts/compare_contact_causal_matrix_hashes.py",
    "scripts/review_contact_causal_checkpoint.py",
    "scripts/evaluate_contact_causal_heldout.py",
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
    "/diagnostic_not_admission/forward_flops_proxy",
    "/diagnostic_not_admission/backward_flops_proxy",
    "/diagnostic_not_admission/peak_memory_bytes",
    "/diagnostic_not_admission/step_time_ms",
    "/diagnostic_not_admission/action_only_latency_ms",
    "/diagnostic_not_admission/transport_flops_proxy",
    "/gradient_reachability_receipt/active_parameter_names_sha256",
    "/gradient_reachability_receipt/active_parameter_names",
    "/gradient_reachability_receipt/inactive_parameter_names_sha256",
    "/gradient_reachability_receipt/inactive_parameter_names",
    "/gradient_reachability_receipt/observed_gradient_parameter_names",
    "/gradient_reachability_receipt/observed_gradient_parameter_names_sha256",
    "/gradient_reachability_receipt/action_loss_gradient_parameter_names",
    "/gradient_reachability_receipt/action_loss_gradient_parameter_names_sha256",
    "/gradient_reachability_receipt/action_loss_reaches_planner",
    "/gradient_reachability_receipt/active_gradient_parameter_count",
    "/gradient_reachability_receipt/unused_parameter_names",
    "/checkpoint_artifact_hashes/config_sha256",
    "/checkpoint_artifact_hashes/weight_tensor_sha256",
    "/checkpoint_artifact_hashes/whole_bundle_sha256",
    "/checkpoint_artifact_hashes/route_receipt_sha256",
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
    "/stable_contract/contact_label_coverage_sha256",
    "/stable_contract/contact_target_stream_sha256",
    "/stable_contract/contact_mask_stream_sha256",
    "/stable_contract/contact_confidence_stream_sha256",
    "/stable_contract/action_target_stream_sha256",
    "/stable_contract/action_mask_stream_sha256",
    "/stable_contract/state_input_stream_sha256",
    "/stable_contract/state_mask_stream_sha256",
    "/stable_contract/embodiment_id_stream_sha256",
    "/stable_contract/prompt_t5_stream_sha256",
    "/stable_contract/ordered_window_manifest_sha256",
    "/stable_contract/augmentation_trace_sha256",
    "/stable_contract/visual_timestep_stream_sha256",
    "/stable_contract/action_timestep_stream_sha256",
    "/stable_contract/visual_noise_stream_sha256",
    "/stable_contract/action_noise_stream_sha256",
    "/stable_contract/pre_forward_rng_stream_sha256",
    "/stable_contract/window_seed",
    "/stable_contract/noise_seed",
    "/stable_contract/training_seed",
    "/evidence/training_gate_sha256",
    "/evidence/production_preflight_receipt_sha256",
    "/evidence/hardened_preflight_receipt_sha256",
    "/evidence/training_data_receipt_sha256",
    "/evidence/matched_control_receipt_sha256",
    "/evidence/external_reviewer_acceptance_sha256",
    "/evidence/contact_coverage_receipt_sha256",
    "/evidence/trace_manifest_sha256",
    "/checkpoint_file/path_record_only",
    "/checkpoint_file/current_run_trace_prefix_receipt/seed",
    "/checkpoint_file/current_run_trace_prefix_receipt/run_id",
    "/checkpoint_file/current_run_trace_prefix_receipt/run_nonce",
    "/checkpoint_file/current_run_trace_prefix_receipt/record_sha256_by_key",
    "/checkpoint_file/current_run_trace_prefix_receipt/trace_path_record_only",
    "/checkpoint_file/current_run_trace_prefix_receipt/receipt_sha256",
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
    if (
        not isinstance(value, str)
        or not _SHA256_RE.fullmatch(value)
        or value == "0" * 64
    ):
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
    comparison_values = []
    for value in values:
        scientific_inputs = deepcopy(dict(value))
        for outcome_field in (
            "parameter_topology", "checkpoint_file",
            "parent_checkpoint_weight_sha256",
        ):
            scientific_inputs.pop(outcome_field)
        comparison_values.append(scientific_inputs)
    normalized = [
        canonicalize_exact_pointers(
            value, ROUTE_LOCAL_JSON_POINTERS, namespace="route_local"
        )
        for value in comparison_values
    ]
    digests = {sha256_json(value) for value in normalized}
    if len(digests) != 1:
        differing = sorted(
            key
            for key in normalized[0]
            if len({sha256_json(value[key]) for value in normalized}) != 1
        )
        raise CausalProductionError(
            "same-seed routes differ outside exact route allowlist: "
            + ",".join(differing)
        )
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
        differing = sorted(
            key
            for key in normalized[0]
            if len({sha256_json(value[key]) for value in normalized}) != 1
        )
        raise CausalProductionError(
            "seeds differ outside exact seed-derived allowlist: "
            + ",".join(differing)
        )
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
    grouped_fields = {
        "presence", "left_censored", "right_censored", "side", "site",
        "object_role", "contact_mode", "start_tau", "end_tau",
    }
    if (
        not isinstance(receipt["field_eligible_counts"], dict)
        or set(receipt["field_eligible_counts"]) != grouped_fields
        or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in receipt["field_eligible_counts"].values()
        )
        or any(count <= 0 for count in receipt["field_eligible_counts"].values())
        or sum(receipt["field_eligible_counts"].values())
        != receipt["total_grouped_eligible_items"]
    ):
        raise CausalProductionError("field coverage counts are invalid")
    if (
        not isinstance(receipt["phase_coverage"], dict)
        or set(receipt["phase_coverage"]) != {"onset", "hold", "release"}
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in receipt["phase_coverage"].values()
        )
    ):
        raise CausalProductionError("phase coverage is missing")
    _sha(receipt["coverage_source_sha256"], "coverage source")
    _sha(receipt["sidecar_bundle_sha256"], "sidecar bundle")
    return receipt


def validate_parameter_topology(value: object, *, route: str, step: int) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "persistent_parameter_schema",
            "persistent_parameter_schema_sha256",
            "persistent_buffer_schema",
            "persistent_buffer_schema_sha256",
            "parameter_tensor_sha256",
            "buffer_tensor_sha256",
            "step0_parameter_tensor_sha256",
            "optimizer_group_parameter_names",
            "optimizer_group_names_sha256",
            "optimizer_group_hyperparameters",
            "optimizer_hyperparameters_sha256",
            "optimizer_state_parameter_names",
            "optimizer_state_names_sha256",
            "optimizer_state_receipt",
            "optimizer_state_sha256",
            "scheduler_state_receipt",
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
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "unused_parameter_names_sha256",
    ):
        _sha(receipt[field], field)
    parameter_schema = receipt["persistent_parameter_schema"]
    buffer_schema = receipt["persistent_buffer_schema"]
    for schema, name in (
        (parameter_schema, "parameter"),
        (buffer_schema, "buffer"),
    ):
        if not isinstance(schema, list):
            raise CausalProductionError(f"persistent {name} schema must be a list")
        names = []
        for row in schema:
            row = _exact(row, {"name", "shape", "dtype"}, f"persistent {name} row")
            if (
                not isinstance(row["name"], str)
                or not row["name"]
                or not isinstance(row["shape"], list)
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in row["shape"])
                or not isinstance(row["dtype"], str)
                or not row["dtype"].startswith("torch.")
            ):
                raise CausalProductionError(f"persistent {name} schema row differs")
            names.append(row["name"])
        if names != sorted(set(names)):
            raise CausalProductionError(f"persistent {name} schema names differ")
    if (
        sha256_json(parameter_schema)
        != receipt["persistent_parameter_schema_sha256"]
        or sha256_json(buffer_schema)
        != receipt["persistent_buffer_schema_sha256"]
    ):
        raise CausalProductionError("persistent parameter/buffer schema digest differs")
    groups = receipt["optimizer_group_parameter_names"]
    hyperparameters = receipt["optimizer_group_hyperparameters"]
    parameter_name_set = {row["name"] for row in parameter_schema}
    flattened_group_names = (
        [name for group in groups for name in group]
        if isinstance(groups, list) and all(isinstance(group, list) for group in groups)
        else []
    )
    if (
        not isinstance(groups, list)
        or not groups
        or any(
            not isinstance(group, list)
            or not group
            or any(not isinstance(name, str) or name not in parameter_name_set for name in group)
            for group in groups
        )
        or len(flattened_group_names) != len(set(flattened_group_names))
        or set(flattened_group_names) != parameter_name_set
        or not isinstance(hyperparameters, list)
        or len(hyperparameters) != len(groups)
    ):
        raise CausalProductionError("optimizer group receipt differs")
    for index, row in enumerate(hyperparameters):
        row = _exact(
            row,
            {"optimizer_index", "group_index", "hyperparameters"},
            "optimizer group hyperparameters",
        )
        _native_int(row["optimizer_index"], "optimizer index", minimum=0)
        _native_int(row["group_index"], "optimizer group index", minimum=0)
        if not isinstance(row["hyperparameters"], dict):
            raise CausalProductionError("optimizer group hyperparameters must be a mapping")
    group_indices = [
        (row["optimizer_index"], row["group_index"])
        for row in hyperparameters
    ]
    if (
        len(group_indices) != len(set(group_indices))
        or group_indices != sorted(group_indices)
        or sorted(set(index for index, _ in group_indices))
        != list(range(max(index for index, _ in group_indices) + 1))
        or any(
            sorted(group for index, group in group_indices if index == optimizer_index)
            != list(
                range(
                    sum(index == optimizer_index for index, _ in group_indices)
                )
            )
            for optimizer_index in set(index for index, _ in group_indices)
        )
    ):
        raise CausalProductionError("optimizer group indices/order differ")
    if (
        sha256_json(groups) != receipt["optimizer_group_names_sha256"]
        or sha256_json(hyperparameters)
        != receipt["optimizer_hyperparameters_sha256"]
    ):
        raise CausalProductionError("optimizer group/hyperparameter digest differs")
    state_names = receipt["optimizer_state_parameter_names"]
    if (
        not isinstance(state_names, list)
        or state_names != sorted(set(state_names))
        or any(name not in {row["name"] for row in parameter_schema} for name in state_names)
        or sha256_json(state_names) != receipt["optimizer_state_names_sha256"]
    ):
        raise CausalProductionError("optimizer state parameter names differ")
    if (
        not isinstance(receipt["optimizer_state_receipt"], list)
        or sha256_json(receipt["optimizer_state_receipt"])
        != receipt["optimizer_state_sha256"]
    ):
        raise CausalProductionError("optimizer state receipt digest differs")
    state_receipt_names = []
    state_receipt_keys = []
    parameter_optimizer_index = {
        parameter_name: hyperparameters[group_offset]["optimizer_index"]
        for group_offset, group in enumerate(groups)
        for parameter_name in group
    }
    for row in receipt["optimizer_state_receipt"]:
        row = _exact(
            row,
            {"optimizer_index", "parameter_name", "state"},
            "optimizer state row",
        )
        _native_int(row["optimizer_index"], "optimizer state index", minimum=0)
        if row["parameter_name"] not in parameter_name_set:
            raise CausalProductionError("optimizer state row references unknown parameter")
        if row["optimizer_index"] != parameter_optimizer_index[row["parameter_name"]]:
            raise CausalProductionError(
                "optimizer state row differs from parameter-group ownership"
            )
        state_receipt_names.append(row["parameter_name"])
        state_receipt_keys.append((row["optimizer_index"], row["parameter_name"]))
    if (
        len(state_receipt_keys) != len(set(state_receipt_keys))
        or sorted(state_receipt_names) != state_names
    ):
        raise CausalProductionError("optimizer state rows/names differ")
    if (
        not isinstance(receipt["scheduler_state_receipt"], list)
        or not receipt["scheduler_state_receipt"]
    ):
        raise CausalProductionError("scheduler state receipt must be a list")
    for scheduler_index, row in enumerate(receipt["scheduler_state_receipt"]):
        row = _exact(row, {"scheduler_index", "state"}, "scheduler state row")
        if row["scheduler_index"] != scheduler_index:
            raise CausalProductionError("scheduler state index/order differs")
    if sha256_json(receipt["scheduler_state_receipt"]) != receipt["scheduler_state_sha256"]:
        raise CausalProductionError("scheduler state receipt digest differs")
    if receipt["equal_bit_mixing_persistent"] is not False:
        raise CausalProductionError("equal-bit mixing matrix must be nonpersistent")
    unused = receipt["unused_parameter_names"]
    if not isinstance(unused, list) or any(not isinstance(name, str) or not name for name in unused):
        raise CausalProductionError("unused parameter allowlist is invalid")
    if unused != sorted(set(unused)) or sha256_json(unused) != receipt["unused_parameter_names_sha256"]:
        raise CausalProductionError("unused parameter allowlist digest differs")
    direct_unused = set(DIRECT_UNUSED_PARAMETER_NAMES)
    if step == 0:
        expected_step0_unused = (
            list(DIRECT_UNUSED_PARAMETER_NAMES)
            if route == "aux_only_direct_q12"
            else []
        )
        if unused != expected_step0_unused:
            raise CausalProductionError("step-0 route-unused set differs")
    elif route == "aux_only_direct_q12":
        if not direct_unused.issubset(unused):
            raise CausalProductionError(
                "direct route did not retain its exact 22-to-action inactive set"
            )
    elif direct_unused & set(unused):
        raise CausalProductionError(
            "strict route declared its 22-to-action encoder inactive"
        )
    if (
        receipt["unused_parameters_have_grad"] is not False
        or receipt["unused_parameters_have_optimizer_state"] is not False
        or receipt["unused_parameters_changed_from_step0"] is not False
    ):
        raise CausalProductionError(
            "inactive parameters acquired grad/state or route-unused drift"
        )
    if step > 0:
        expected_state_names = sorted(parameter_name_set - set(unused))
        if state_names != expected_state_names:
            raise CausalProductionError(
                "post-step optimizer state must cover every active parameter and only the exact direct-unused omission"
            )
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
            "diagnostic_not_admission",
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
        or route_entry["cache_policy"]
        != expected_route["op38_route_receipt"]["cache_policy"]
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
        "persistent_buffer_schema_sha256",
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
        "state_input_stream_sha256",
        "state_mask_stream_sha256",
        "embodiment_id_stream_sha256",
        "prompt_t5_stream_sha256",
        "runtime_environment_sha256",
        "dataset_runtime_source_receipt_sha256",
        "t5_runtime_source_receipt_sha256",
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
        or isinstance(stable["world_size"], bool)
        or not isinstance(stable["world_size"], int)
        or stable["world_size"] <= 0
        or stable["rank_set"] != list(range(stable["world_size"]))
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in stable["rank_set"])
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
            "hardened_preflight_receipt_sha256",
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
    parameter_topology = receipt["parameter_topology"]
    if (
        stable["initial_parameter_tensor_sha256"]
        != parameter_topology["step0_parameter_tensor_sha256"]
        or stable["persistent_parameter_schema_sha256"]
        != parameter_topology["persistent_parameter_schema_sha256"]
        or stable["persistent_buffer_schema_sha256"]
        != parameter_topology["persistent_buffer_schema_sha256"]
    ):
        raise CausalProductionError("stable model topology mirrors differ")
    if receipt["checkpoint_step"] == 0 and (
        stable["optimizer_parameter_groups_sha256"]
        != parameter_topology["optimizer_group_names_sha256"]
        or stable["optimizer_config_sha256"]
        != parameter_topology["optimizer_hyperparameters_sha256"]
        or stable["optimizer_step0_state_sha256"]
        != parameter_topology["optimizer_state_sha256"]
        or stable["scheduler_step0_state_sha256"]
        != parameter_topology["scheduler_state_sha256"]
    ):
        raise CausalProductionError("stable step-0 optimizer/scheduler mirrors differ")
    diagnostics = _exact(
        receipt["diagnostic_not_admission"],
        {
            "admission_role",
            "forward_flops_proxy",
            "backward_flops_proxy",
            "peak_memory_bytes",
            "step_time_ms",
            "action_only_latency_ms",
            "transport_flops_proxy",
        },
        "route diagnostics",
    )
    if diagnostics["admission_role"] != "diagnostic_not_admission":
        raise CausalProductionError("route performance values must not gate admission")
    for field, value in diagnostics.items():
        if field != "admission_role":
            _finite_number(value, f"route diagnostic {field}")
    reachability = _exact(
        receipt["gradient_reachability_receipt"],
        {
            "active_parameter_names",
            "active_parameter_names_sha256",
            "inactive_parameter_names",
            "inactive_parameter_names_sha256",
            "observed_gradient_parameter_names",
            "observed_gradient_parameter_names_sha256",
            "action_loss_gradient_parameter_names",
            "action_loss_gradient_parameter_names_sha256",
            "action_loss_reaches_planner",
            "active_gradient_parameter_count",
            "unused_parameter_names",
        },
        "gradient reachability receipt",
    )
    _sha(reachability["active_parameter_names_sha256"], "active parameter names")
    _sha(reachability["inactive_parameter_names_sha256"], "inactive parameter names")
    _sha(reachability["observed_gradient_parameter_names_sha256"], "observed gradient parameter names")
    _sha(
        reachability["action_loss_gradient_parameter_names_sha256"],
        "action-loss gradient parameter names",
    )
    active = reachability["active_parameter_names"]
    inactive = reachability["inactive_parameter_names"]
    observed = reachability["observed_gradient_parameter_names"]
    action_gradient = reachability["action_loss_gradient_parameter_names"]
    _native_int(
        reachability["active_gradient_parameter_count"],
        "active gradient parameter count", minimum=0,
    )
    parameter_names = [row["name"] for row in receipt["parameter_topology"]["persistent_parameter_schema"]]
    planner_names = sorted(name for name in parameter_names if name.startswith("contact_"))
    if (
        any(
            not isinstance(names, list) or names != sorted(set(names))
            for names in (active, inactive, observed, action_gradient)
        )
        or set(active) & set(inactive)
        or sorted(active + inactive) != parameter_names
        or inactive != receipt["parameter_topology"]["unused_parameter_names"]
        or any(name not in active for name in observed)
        or sha256_json(active) != reachability["active_parameter_names_sha256"]
        or sha256_json(inactive) != reachability["inactive_parameter_names_sha256"]
        or sha256_json(observed) != reachability["observed_gradient_parameter_names_sha256"]
        or sha256_json(action_gradient)
        != reachability["action_loss_gradient_parameter_names_sha256"]
        or (receipt["checkpoint_step"] == 0 and observed)
        or (receipt["checkpoint_step"] > 0 and observed != active)
    ):
        raise CausalProductionError("gradient reachability/observed set differs from model topology")
    if reachability["active_gradient_parameter_count"] != len(active):
        raise CausalProductionError(
            "active-gradient parameter count differs from the live active set"
        )
    expected_no_action_gradient = direct or receipt["checkpoint_step"] == 0
    if (
        any(name not in planner_names or name not in active for name in action_gradient)
        or (expected_no_action_gradient and action_gradient)
        or (not expected_no_action_gradient and not action_gradient)
        or reachability["action_loss_reaches_planner"] is not bool(action_gradient)
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
        {
            "path_record_only",
            "sha256",
            "size_bytes",
            "serialization",
            "tensor_manifest",
            "tensor_manifest_sha256",
            "bundle_manifest",
            "current_run_trace_prefix_receipt",
        },
        "checkpoint file receipt",
    )
    _sha(checkpoint_file["sha256"], "checkpoint file")
    _sha(checkpoint_file["tensor_manifest_sha256"], "checkpoint tensor manifest")
    from world_action_model.contact_causal_matched_control import (
        validate_current_run_trace_prefix_receipt_v1,
    )

    trace_prefix = validate_current_run_trace_prefix_receipt_v1(
        checkpoint_file["current_run_trace_prefix_receipt"]
    )
    if (
        trace_prefix["causal_route_id"] != route
        or trace_prefix["seed"] != receipt["seed"]
        or trace_prefix["run_id"] != run["training_run_id"]
        or trace_prefix["run_nonce"] != run["training_run_nonce"]
        or trace_prefix["through_checkpoint_step"] != receipt["checkpoint_step"]
        or trace_prefix["rank"] != 0
        or trace_prefix["world_size"] != stable["world_size"]
    ):
        raise CausalProductionError("checkpoint current-run trace binding differs")
    _native_int(checkpoint_file["size_bytes"], "checkpoint size", minimum=1)
    if not isinstance(checkpoint_file["path_record_only"], str) or not Path(checkpoint_file["path_record_only"]).is_absolute():
        raise CausalProductionError("checkpoint file path must be absolute")
    from world_action_model.contact_causal_checkpoint import (
        SAFE_WEIGHT_FILES,
        validate_safe_tensor_manifest,
    )

    tensor_manifest = validate_safe_tensor_manifest(
        checkpoint_file["tensor_manifest"]
    )
    if (
        checkpoint_file["serialization"] != tensor_manifest["serialization"]
        or checkpoint_file["serialization"]
        != SAFE_WEIGHT_FILES.get(Path(checkpoint_file["path_record_only"]).name)
        or sha256_json(tensor_manifest)
        != checkpoint_file["tensor_manifest_sha256"]
        or not isinstance(checkpoint_file["bundle_manifest"], dict)
        or checkpoint_file["bundle_manifest"].get("bundle_sha256")
        != artifact_hashes["whole_bundle_sha256"]
        or checkpoint_file["bundle_manifest"].get("weight_file")
        != Path(checkpoint_file["path_record_only"]).name
    ):
        raise CausalProductionError("checkpoint safe tensor/bundle mirrors differ")
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
    from world_action_model.contact_causal_checkpoint import (
        inspect_causal_checkpoint_bundle,
        validate_weight_mapping_against_topology,
    )

    inspected = inspect_causal_checkpoint_bundle(
        Path(entry["path_record_only"]).parent
    )
    validate_weight_mapping_against_topology(
        entry["path_record_only"], receipt["parameter_topology"]
    )
    validate_checkpoint_transformer_semantics(
        inspected["config"], route=receipt["route"]["causal_route_id"]
    )
    if (
        inspected["weight_file_sha256"] != entry["sha256"]
        or inspected["weight_file_size_bytes"] != entry["size_bytes"]
        or inspected["tensor_manifest"] != entry["tensor_manifest"]
        or inspected["tensor_manifest_sha256"]
        != entry["tensor_manifest_sha256"]
        or inspected["bundle_manifest"] != entry["bundle_manifest"]
        or inspected["config_sha256"]
        != receipt["checkpoint_artifact_hashes"]["config_sha256"]
        or inspected["bundle_manifest"]["bundle_sha256"]
        != receipt["checkpoint_artifact_hashes"]["whole_bundle_sha256"]
        or inspected["config"].get("contact_causal_route")
        != receipt["route"]["causal_route_id"]
        or inspected["config"].get("production_matrix_id") != MATRIX_ID
    ):
        raise CausalProductionError(
            "checkpoint safe tensor/config/bundle bytes changed"
        )


def checkpoint_transformer_semantic_config(route: str) -> dict[str, Any]:
    """Return the exact constructor semantics shared by all admitted checkpoints."""

    route = ensure_exact_causal_route(route)
    direct = route == "aux_only_direct_q12"
    return {
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": 48,
        "num_attention_heads": 24,
        "num_layers": 30,
        "out_channels": 48,
        "patch_size": [1, 2, 2],
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
        "action_expert_dim": 1024,
        "action_ffn_dim": 4096,
        "in_action_channels": 32,
        "out_action_channels": 32,
        "num_embodiments": 2,
        "num_contact_tokens": 12,
        "contact_target_dim": 22,
        "contact_condition_dim": 22,
        "contact_bottleneck": not direct,
        "contact_schema_version": SEGMENT_SCHEMA_VERSION,
        "contact_semantic_bottleneck": not direct,
        "contact_planner_num_heads": 8,
        "contact_semantic_transport_mode": "hard_st",
        "contact_time_bins": 16,
        "contact_causal_route": route,
        "contact_planner_precursor": False,
        "production_matrix_id": MATRIX_ID,
    }


def validate_checkpoint_transformer_semantics(
    config: Mapping[str, Any], *, route: str
) -> None:
    """Bind every constructor field that changes production forward semantics."""

    expected = checkpoint_transformer_semantic_config(route)
    if any(config.get(name) != value for name, value in expected.items()):
        differing = sorted(
            name for name, value in expected.items() if config.get(name) != value
        )
        raise CausalProductionError(
            "checkpoint transformer semantic config differs: " + ",".join(differing)
        )


def named_tensor_digest(rows: Sequence[tuple[str, Any]]) -> str:
    """Hash exact named tensor dtype/shape/bytes without pickle metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(rows, key=lambda item: item[0]):
        if not isinstance(name, str) or not name:
            raise CausalProductionError("named tensor digest requires non-empty names")
        try:
            value = tensor.detach().to(device="cpu").contiguous()
            raw = (
                value.reshape(-1)
                .view(__import__("torch").uint8)
                .numpy()
                .tobytes(order="C")
            )
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
    persistent_names = set(model.state_dict())
    return (
        named_tensor_digest(list(model.named_parameters())),
        named_tensor_digest(
            [
                (name, tensor)
                for name, tensor in model.named_buffers()
                if name in persistent_names
            ]
        ),
    )


def _runtime_state_receipt(value: Any) -> Any:
    """Convert optimizer/scheduler state to strict JSON without pickle trust."""

    try:
        import torch
    except ImportError as exc:
        raise CausalProductionError("torch is required for live training-state receipts") from exc
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes_sha256": named_tensor_digest([("state", tensor)]),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CausalProductionError("optimizer/scheduler state is non-finite")
        return value
    if isinstance(value, Mapping):
        rows = []
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise CausalProductionError("runtime state mapping key is not string/integer")
            rows.append({"key": str(key), "value": _runtime_state_receipt(value[key])})
        return {"kind": "mapping", "entries": rows}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [_runtime_state_receipt(item) for item in value],
        }
    raise CausalProductionError(
        f"unsupported optimizer/scheduler state type: {type(value).__name__}"
    )


def build_live_training_state_receipts(
    *,
    model: Any,
    optimizers: Sequence[Any],
    schedulers: Sequence[Any],
    route: str,
    checkpoint_step: int,
    step0_parameter_tensor_sha256: str,
    observed_gradient_parameter_names: Sequence[str],
    action_loss_gradient_parameter_names: Sequence[str],
    unused_step0_tensor_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute v5 topology/optimizer/scheduler/reachability from live objects."""

    route = ensure_exact_causal_route(route)
    if checkpoint_step not in CHECKPOINT_STEPS or isinstance(checkpoint_step, bool):
        raise CausalProductionError("live receipt checkpoint step differs")
    named_parameters = dict(model.named_parameters())
    state_dict_names = set(model.state_dict())
    named_buffers = {
        name: tensor
        for name, tensor in model.named_buffers()
        if name in state_dict_names
    }
    if not named_parameters:
        raise CausalProductionError("live causal model has no parameters")
    parameter_schema = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(named_parameters.items())
    ]
    buffer_schema = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(named_buffers.items())
    ]
    direct_route_unused = sorted(
        name
        for name in named_parameters
        if name.startswith("contact_condition_encoder.")
    )
    if direct_route_unused != list(DIRECT_UNUSED_PARAMETER_NAMES):
        raise CausalProductionError(
            "live causal model route-unused parameter schema differs"
        )
    observed = sorted(set(observed_gradient_parameter_names))
    if checkpoint_step == 0:
        unused = (
            direct_route_unused if route == "aux_only_direct_q12" else []
        )
    else:
        unused = sorted(set(named_parameters) - set(observed))
    if route == "aux_only_direct_q12":
        if not set(direct_route_unused).issubset(unused):
            raise CausalProductionError("live direct unused parameter set differs")
    elif set(direct_route_unused) & set(unused):
        raise CausalProductionError(
            "strict route did not activate its 22-to-action encoder"
        )
    parameter_names = sorted(named_parameters)
    active = sorted(set(parameter_names) - set(unused))
    action_gradient = sorted(set(action_loss_gradient_parameter_names))
    if any(name not in parameter_names for name in observed):
        raise CausalProductionError("observed gradient name is absent from live model")
    if any(name not in parameter_names for name in action_gradient):
        raise CausalProductionError("action-loss gradient name is absent from live model")
    by_identity = {id(parameter): name for name, parameter in named_parameters.items()}
    group_names = []
    group_hyperparameters = []
    state_names = []
    optimizer_state_receipt = []
    for optimizer_index, optimizer in enumerate(optimizers):
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        for group_index, group in enumerate(raw_optimizer.param_groups):
            names = []
            for parameter in group["params"]:
                try:
                    names.append(by_identity[id(parameter)])
                except KeyError as exc:
                    raise CausalProductionError("optimizer contains a parameter outside the model") from exc
            group_names.append(names)
            hyper = {
                key: _runtime_state_receipt(item)
                for key, item in sorted(group.items())
                if key != "params"
            }
            group_hyperparameters.append(
                {
                    "optimizer_index": optimizer_index,
                    "group_index": group_index,
                    "hyperparameters": hyper,
                }
            )
        for parameter, state in raw_optimizer.state.items():
            try:
                name = by_identity[id(parameter)]
            except KeyError as exc:
                raise CausalProductionError("optimizer state references an unknown parameter") from exc
            state_names.append(name)
            optimizer_state_receipt.append(
                {
                    "optimizer_index": optimizer_index,
                    "parameter_name": name,
                    "state": _runtime_state_receipt(state),
                }
            )
    state_names = sorted(state_names)
    optimizer_state_receipt.sort(
        key=lambda row: (row["optimizer_index"], row["parameter_name"])
    )
    scheduler_state_receipt = []
    for index, scheduler in enumerate(schedulers):
        state = scheduler.state_dict()
        scheduler_state_receipt.append(
            {"scheduler_index": index, "state": _runtime_state_receipt(state)}
        )
    parameter_sha256, buffer_sha256 = model_parameter_and_buffer_digests(model)
    direct_unused_current_sha256 = named_tensor_digest(
        [(name, named_parameters[name]) for name in direct_route_unused]
    )
    topology = {
        "persistent_parameter_schema": parameter_schema,
        "persistent_parameter_schema_sha256": sha256_json(parameter_schema),
        "persistent_buffer_schema": buffer_schema,
        "persistent_buffer_schema_sha256": sha256_json(buffer_schema),
        "parameter_tensor_sha256": parameter_sha256,
        "buffer_tensor_sha256": buffer_sha256,
        "step0_parameter_tensor_sha256": step0_parameter_tensor_sha256,
        "optimizer_group_parameter_names": group_names,
        "optimizer_group_names_sha256": sha256_json(group_names),
        "optimizer_group_hyperparameters": group_hyperparameters,
        "optimizer_hyperparameters_sha256": sha256_json(group_hyperparameters),
        "optimizer_state_parameter_names": state_names,
        "optimizer_state_names_sha256": sha256_json(state_names),
        "optimizer_state_receipt": optimizer_state_receipt,
        "optimizer_state_sha256": sha256_json(optimizer_state_receipt),
        "scheduler_state_receipt": scheduler_state_receipt,
        "scheduler_state_sha256": sha256_json(scheduler_state_receipt),
        "equal_bit_mixing_persistent": "equal_bit_mixing_matrix" in state_dict_names,
        "unused_parameter_names": unused,
        "unused_parameter_names_sha256": sha256_json(unused),
        "unused_parameters_have_grad": any(
            named_parameters[name].grad is not None for name in unused
        ),
        "unused_parameters_have_optimizer_state": bool(set(unused) & set(state_names)),
        "unused_parameters_changed_from_step0": (
            route == "aux_only_direct_q12"
            and direct_unused_current_sha256 != unused_step0_tensor_sha256
        ),
    }
    reachability = {
        "active_parameter_names": active,
        "active_parameter_names_sha256": sha256_json(active),
        "inactive_parameter_names": unused,
        "inactive_parameter_names_sha256": sha256_json(unused),
        "observed_gradient_parameter_names": observed,
        "observed_gradient_parameter_names_sha256": sha256_json(observed),
        "action_loss_gradient_parameter_names": action_gradient,
        "action_loss_gradient_parameter_names_sha256": sha256_json(action_gradient),
        "action_loss_reaches_planner": bool(action_gradient),
        "active_gradient_parameter_count": len(active),
        "unused_parameter_names": unused,
    }
    validate_parameter_topology(topology, route=route, step=checkpoint_step)
    return topology, reachability


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
    parameter_topology: Mapping[str, Any],
    gradient_reachability_receipt: Mapping[str, Any],
    current_run_trace_prefix_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill only checkpoint-local leaves of a reviewed v5 base receipt."""

    receipt = deepcopy(dict(base_receipt))
    if checkpoint_step not in CHECKPOINT_STEPS or isinstance(checkpoint_step, bool):
        raise CausalProductionError("causal checkpoint step must be 0/100/200")
    path = Path(checkpoint_file_path).expanduser().absolute()
    from world_action_model.contact_causal_checkpoint import (
        inspect_causal_checkpoint_bundle,
        validate_weight_mapping_against_topology,
    )

    inspected = inspect_causal_checkpoint_bundle(path.parent)
    file_sha256 = inspected["weight_file_sha256"]
    file_size_bytes = inspected["weight_file_size_bytes"]
    for value, name in (
        (parameter_tensor_sha256, "parameter tensor"),
        (buffer_tensor_sha256, "buffer tensor"),
        (config_sha256, "checkpoint config"),
        (whole_bundle_sha256, "whole checkpoint bundle"),
        (parent_checkpoint_weight_sha256, "parent checkpoint weight"),
    ):
        _sha(value, name)
    if (
        config_sha256 != inspected["config_sha256"]
        or whole_bundle_sha256
        != inspected["bundle_manifest"]["bundle_sha256"]
        or path.name != inspected["weight_file"]
    ):
        raise CausalProductionError(
            "runtime checkpoint config/bundle differs from safe inspection"
        )
    validate_weight_mapping_against_topology(path, parameter_topology)
    receipt["checkpoint_step"] = checkpoint_step
    receipt["output_path_record_only"] = str(path.parent)
    receipt["checkpoint_file"] = {
        "path_record_only": str(path),
        "sha256": file_sha256,
        "size_bytes": file_size_bytes,
        "serialization": inspected["tensor_manifest"]["serialization"],
        "tensor_manifest": inspected["tensor_manifest"],
        "tensor_manifest_sha256": inspected["tensor_manifest_sha256"],
        "bundle_manifest": inspected["bundle_manifest"],
        "current_run_trace_prefix_receipt": deepcopy(
            dict(current_run_trace_prefix_receipt)
        ),
    }
    receipt["parent_checkpoint_weight_sha256"] = parent_checkpoint_weight_sha256
    receipt["parameter_topology"] = deepcopy(dict(parameter_topology))
    receipt["gradient_reachability_receipt"] = deepcopy(
        dict(gradient_reachability_receipt)
    )
    if (
        receipt["parameter_topology"].get("parameter_tensor_sha256")
        != parameter_tensor_sha256
        or receipt["parameter_topology"].get("buffer_tensor_sha256")
        != buffer_tensor_sha256
    ):
        raise CausalProductionError("live topology tensor digests differ from checkpoint")
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
            "checkpoint_tensor_manifest_sha256",
            "checkpoint_path_record_only",
            "robot_gate_path_record_only",
            "robot_gate_sha256",
            "gate_reviewer_acceptance_path_record_only",
            "gate_reviewer_acceptance_sha256",
            "route_training_acceptance_path_record_only",
            "route_training_acceptance_sha256",
            "precursor_receipt_path_record_only",
            "precursor_receipt_bytes_sha256",
            "precursor_checkpoint_dir_record_only",
            "precursor_bundle_sha256",
            "hardened_preflight_path_record_only",
            "hardened_preflight_receipt_sha256",
            "production_preflight_path_record_only",
            "production_preflight_receipt_sha256",
            "training_data_path_record_only",
            "training_data_receipt_sha256",
            "coverage_path_record_only",
            "contact_coverage_receipt_sha256",
            "sidecar_dir_record_only",
            "t5_embedding_dir_record_only",
            "matched_control_report_path_record_only",
            "matched_control_report_sha256",
            "matched_control_trace_manifest_sha256",
            "matched_control_trace_path_by_route",
            "current_run_trace_prefix_receipt_sha256",
            "current_head_source_receipt_sha256",
            "reviewer_identity",
            "reviewer_attestation_nonce",
            "acceptance_run_id",
            "acceptance_run_nonce",
            "reviewed_current_head_sha256",
            "reviewed_source_tree_sha256",
            "reviewed_at_utc",
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
        or acceptance["checkpoint_tensor_manifest_sha256"]
        != receipt["checkpoint_file"]["tensor_manifest_sha256"]
        or acceptance["checkpoint_path_record_only"]
        != receipt["checkpoint_file"]["path_record_only"]
        or acceptance["robot_gate_sha256"]
        != receipt["evidence"]["training_gate_sha256"]
        or acceptance["gate_reviewer_acceptance_sha256"]
        != receipt["evidence"]["external_reviewer_acceptance_sha256"]
        or acceptance["route_training_acceptance_sha256"]
        != receipt["run_identity"]["external_acceptance_sha256"]
        or acceptance["precursor_receipt_bytes_sha256"]
        != receipt["precursor"]["receipt_bytes_sha256"]
        or acceptance["precursor_bundle_sha256"]
        != receipt["precursor"]["bundle_sha256"]
        or acceptance["hardened_preflight_receipt_sha256"]
        != receipt["evidence"]["hardened_preflight_receipt_sha256"]
        or acceptance["production_preflight_receipt_sha256"]
        != receipt["evidence"]["production_preflight_receipt_sha256"]
        or acceptance["training_data_receipt_sha256"]
        != receipt["evidence"]["training_data_receipt_sha256"]
        or acceptance["contact_coverage_receipt_sha256"]
        != receipt["evidence"]["contact_coverage_receipt_sha256"]
        or acceptance["current_run_trace_prefix_receipt_sha256"]
        != receipt["checkpoint_file"]["current_run_trace_prefix_receipt"][
            "receipt_sha256"
        ]
        or acceptance["matched_control_report_sha256"]
        != receipt["evidence"]["matched_control_receipt_sha256"]
        or acceptance["matched_control_trace_manifest_sha256"]
        != receipt["evidence"]["trace_manifest_sha256"]
        or acceptance["current_head_source_receipt_sha256"]
        != receipt["evidence"]["current_head_source_receipt_sha256"]
        or acceptance["reviewed_current_head_sha256"] != receipt["stable_contract"]["git_commit"]
        or acceptance["reviewed_source_tree_sha256"] != receipt["stable_contract"]["source_tree_sha256"]
        or acceptance["oracle_used"] is not False
    ):
        raise CausalProductionError("independent checkpoint acceptance binding differs")
    for field in (
        "checkpoint_tensor_manifest_sha256",
        "robot_gate_sha256",
        "gate_reviewer_acceptance_sha256",
        "current_head_source_receipt_sha256",
        "route_training_acceptance_sha256",
        "precursor_receipt_bytes_sha256",
        "precursor_bundle_sha256",
        "hardened_preflight_receipt_sha256",
        "production_preflight_receipt_sha256",
        "training_data_receipt_sha256",
        "contact_coverage_receipt_sha256",
        "matched_control_report_sha256",
        "matched_control_trace_manifest_sha256",
        "current_run_trace_prefix_receipt_sha256",
    ):
        _sha(acceptance[field], f"acceptance {field}")
    path_fields = (
        "checkpoint_path_record_only", "robot_gate_path_record_only",
        "gate_reviewer_acceptance_path_record_only",
        "route_training_acceptance_path_record_only",
        "precursor_receipt_path_record_only", "precursor_checkpoint_dir_record_only",
        "hardened_preflight_path_record_only", "production_preflight_path_record_only",
        "training_data_path_record_only", "coverage_path_record_only",
        "sidecar_dir_record_only", "t5_embedding_dir_record_only",
        "matched_control_report_path_record_only",
    )
    if any(
        not isinstance(acceptance[name], str)
        or not Path(acceptance[name]).is_absolute()
        for name in path_fields
    ):
        raise CausalProductionError("acceptance evidence paths must be absolute")
    trace_paths = acceptance["matched_control_trace_path_by_route"]
    if (
        not isinstance(trace_paths, dict)
        or list(trace_paths) != list(CAUSAL_ROUTES)
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in trace_paths.values())
    ):
        raise CausalProductionError("acceptance matched trace path map differs")
    if not isinstance(acceptance["reviewed_at_utc"], str) or not _UTC_RE.fullmatch(
        acceptance["reviewed_at_utc"]
    ):
        raise CausalProductionError("acceptance reviewed_at_utc must be canonical UTC")
    try:
        reviewed_at = datetime.fromisoformat(
            acceptance["reviewed_at_utc"][:-1] + "+00:00"
        )
        expiry = datetime.fromisoformat(
            acceptance["evidence_valid_until_utc"][:-1] + "+00:00"
        )
    except ValueError as exc:
        raise CausalProductionError("acceptance UTC timestamp is invalid") from exc
    if (
        reviewed_at.tzinfo is None
        or reviewed_at > datetime.now(timezone.utc)
        or reviewed_at >= expiry
    ):
        raise CausalProductionError("acceptance review time must precede expiry")
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


def verify_acceptance_bound_artifacts_v3(
    acceptance: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    repository_root: str | Path,
) -> None:
    """Reopen and semantically replay every external file named by v3 acceptance."""

    acceptance = validate_acceptance_v3(acceptance, receipt=receipt)
    root = Path(repository_root).expanduser().absolute()
    gate, gate_sha, _ = strict_json_file(
        acceptance["robot_gate_path_record_only"], name="accepted Robot gate"
    )
    from world_action_model.contact_causal_preflight import (
        build_robot_runtime_source_receipts_v1,
        replay_robot_gate_evidence_v3,
        replay_persisted_hardened_contact_asym25_v2,
        validate_preflight_receipt_v2,
        validate_robot_gate_review_v2,
        validate_robot_gate_v3,
        validate_training_data_receipt_v2,
    )
    from world_action_model.contact_causal_gate_bootstrap import (
        validate_robot_route_acceptance_v1,
    )
    from world_action_model.contact_causal_coverage import (
        validate_recomputed_dataset_supervision_coverage,
    )
    from world_action_model.contact_causal_matched_control import (
        compare_causal_records,
        load_trace_jsonl,
        replay_current_run_trace_prefix_v1,
        validate_collector_report_v3,
    )
    from world_action_model.contact_ego_causal_precursor import (
        validate_precursor_receipt,
    )
    from world_action_model.contact_causal_checkpoint import (
        inspect_causal_checkpoint_bundle,
        load_safe_weight_mapping,
    )

    if gate_sha != acceptance["robot_gate_sha256"]:
        raise CausalProductionError("accepted Robot gate bytes changed")
    validate_robot_gate_v3(
        gate, gate_path=acceptance["robot_gate_path_record_only"],
        load_external_review=True,
    )
    replay_robot_gate_evidence_v3(
        gate, gate_path=acceptance["robot_gate_path_record_only"]
    )
    gate_review, gate_review_sha, _ = strict_json_file(
        acceptance["gate_reviewer_acceptance_path_record_only"],
        name="accepted Robot gate review",
    )
    if gate_review_sha != acceptance["gate_reviewer_acceptance_sha256"]:
        raise CausalProductionError("accepted gate review bytes changed")
    validate_robot_gate_review_v2(gate_review, gate=gate)
    route = acceptance["causal_route_id"]
    route_acceptance, route_acceptance_sha, _ = strict_json_file(
        acceptance["route_training_acceptance_path_record_only"],
        name="accepted route training acceptance",
    )
    if route_acceptance_sha != acceptance["route_training_acceptance_sha256"]:
        raise CausalProductionError("accepted route training bytes changed")
    validate_robot_route_acceptance_v1(
        route_acceptance, gate=gate, route=route, repository_root=root
    )
    if (
        route_acceptance["training_run_id"]
        != receipt["run_identity"]["training_run_id"]
        or route_acceptance["training_run_nonce"]
        != receipt["run_identity"]["training_run_nonce"]
    ):
        raise CausalProductionError(
            "accepted route training identity differs from current run"
        )

    precursor, precursor_sha, _ = strict_json_file(
        acceptance["precursor_receipt_path_record_only"],
        name="accepted precursor receipt",
    )
    validate_precursor_receipt(precursor)
    precursor_bundle = inspect_causal_checkpoint_bundle(
        acceptance["precursor_checkpoint_dir_record_only"]
    )
    precursor_mapping, _ = load_safe_weight_mapping(
        Path(acceptance["precursor_checkpoint_dir_record_only"])
        / precursor_bundle["weight_file"]
    )
    parameter_names = precursor["parameter_partition"]["all_parameter_names"]
    if (
        precursor_sha != acceptance["precursor_receipt_bytes_sha256"]
        or precursor_sha
        != gate["artifact_fingerprints"]["precursor_receipt_bytes_sha256"]
        or precursor_bundle["bundle_manifest"]["bundle_sha256"]
        != acceptance["precursor_bundle_sha256"]
        or precursor_bundle["bundle_manifest"]["bundle_sha256"]
        != gate["artifact_fingerprints"]["precursor_bundle_sha256"]
        or not set(parameter_names).issubset(precursor_mapping)
        or named_tensor_digest(
            [(name, precursor_mapping[name]) for name in parameter_names]
        ) != receipt["precursor"]["parameter_tensor_sha256"]
    ):
        raise CausalProductionError("accepted precursor bytes changed")

    hardened, hardened_sha, _ = strict_json_file(
        acceptance["hardened_preflight_path_record_only"],
        name="accepted hardened preflight",
    )
    training_data, training_sha, _ = strict_json_file(
        acceptance["training_data_path_record_only"],
        name="accepted training-data receipt",
    )
    preflight, preflight_sha, _ = strict_json_file(
        acceptance["production_preflight_path_record_only"],
        name="accepted production preflight",
    )
    coverage, coverage_sha, _ = strict_json_file(
        acceptance["coverage_path_record_only"], name="accepted coverage receipt"
    )
    if (
        hardened_sha != acceptance["hardened_preflight_receipt_sha256"]
        or training_sha != acceptance["training_data_receipt_sha256"]
        or preflight_sha != acceptance["production_preflight_receipt_sha256"]
        or coverage_sha != acceptance["contact_coverage_receipt_sha256"]
    ):
        raise CausalProductionError("accepted data receipt bytes changed")
    replay_persisted_hardened_contact_asym25_v2(hardened)
    validate_training_data_receipt_v2(
        training_data, gate=gate, gate_sha256=gate_sha
    )
    validate_preflight_receipt_v2(
        preflight, gate=gate, gate_sha256=gate_sha,
        training_data_receipt=training_data,
    )
    validate_recomputed_dataset_supervision_coverage(
        coverage,
        sidecar_dir=acceptance["sidecar_dir_record_only"],
        expected_meta_sha256=gate["artifact_fingerprints"]["sidecar_meta_sha256"],
        expected_episode_sha256=hardened["episode_npz_sha256"],
        dataset_id=hardened["dataset_id"], split_seed=hardened["split_seed"],
        split_episode_ids={
            "train": hardened["train_episode_ids"],
            "val": hardened["validation_episode_ids"],
            "test": hardened["heldout_episode_ids"],
        },
    )
    runtime = build_robot_runtime_source_receipts_v1(
        hardened_preflight_receipt=hardened,
        t5_embedding_dir=acceptance["t5_embedding_dir_record_only"],
    )
    if (
        runtime["training_source_runtime_receipt_sha256"]
        != receipt["stable_contract"]["dataset_runtime_source_receipt_sha256"]
        or runtime["t5_runtime_source_receipt_sha256"]
        != receipt["stable_contract"]["t5_runtime_source_receipt_sha256"]
    ):
        raise CausalProductionError("accepted runtime source bytes changed")

    traces = {
        route_name: load_trace_jsonl(path, route=route_name)
        for route_name, path in acceptance["matched_control_trace_path_by_route"].items()
    }
    report, report_sha, _ = strict_json_file(
        acceptance["matched_control_report_path_record_only"],
        name="accepted current-run matched-control report",
    )
    validate_collector_report_v3(report)
    if (
        report_sha != acceptance["matched_control_report_sha256"]
        or report["trace_manifest_sha256"]
        != acceptance["matched_control_trace_manifest_sha256"]
        or report_sha != receipt["evidence"]["matched_control_receipt_sha256"]
        or report["trace_manifest_sha256"]
        != receipt["evidence"]["trace_manifest_sha256"]
        or report != compare_causal_records(traces, required_seed=receipt["seed"])
    ):
        raise CausalProductionError("accepted current-run matched-control replay changed")
    prefix = receipt["checkpoint_file"]["current_run_trace_prefix_receipt"]
    if (
        Path(prefix["trace_path_record_only"]).absolute()
        != Path(acceptance["matched_control_trace_path_by_route"][route]).absolute()
        or replay_current_run_trace_prefix_v1(prefix) != prefix
    ):
        raise CausalProductionError("accepted current-run trace prefix changed")
    source = current_head_source_tree_receipt(
        root, expected_commit=receipt["stable_contract"]["git_commit"]
    )
    if sha256_json(source) != acceptance["current_head_source_receipt_sha256"]:
        raise CausalProductionError("accepted current HEAD source bytes changed")


def validate_27_checkpoint_matrix(
    evidence: Mapping[
        tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    *,
    verify_files: bool = True,
    verify_external_evidence: bool = False,
    repository_root: str | Path | None = None,
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
    externally_replayed_gates: set[str] = set()
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
        if verify_external_evidence:
            verify_acceptance_bound_artifacts_v3(
                acceptance,
                receipt=receipt,
                repository_root=(
                    Path(__file__).resolve().parents[1]
                    if repository_root is None
                    else repository_root
                ),
            )
            gate_path = acceptance["robot_gate_path_record_only"]
            gate, gate_sha256, _ = strict_json_file(
                gate_path, name="matrix Robot gate identity replay"
            )
            if gate_sha256 not in externally_replayed_gates:
                gate_review, _, _ = strict_json_file(
                    acceptance["gate_reviewer_acceptance_path_record_only"],
                    name="matrix Robot gate reviewer identity replay",
                )
                external_identities = {gate["reviewed_by"]}
                external_identities.update(
                    gate_review[name]
                    for name in (
                        "reviewer_identity",
                        "reviewer_attestation_nonce",
                        "acceptance_run_id",
                        "acceptance_run_nonce",
                    )
                )
                for route_name in CAUSAL_ROUTES:
                    route_acceptance_path = Path(
                        gate["external_acceptance_path_by_route"][route_name]
                    )
                    if not route_acceptance_path.is_absolute():
                        route_acceptance_path = (
                            Path(gate_path).absolute().parent
                            / route_acceptance_path
                        )
                    route_acceptance, _, _ = strict_json_file(
                        route_acceptance_path,
                        name=(
                            "matrix Robot route reviewer identity replay "
                            f"{route_name}"
                        ),
                    )
                    external_identities.update(
                        route_acceptance[name]
                        for name in (
                            "reviewer_identity",
                            "reviewer_attestation_nonce",
                            "acceptance_run_id",
                            "acceptance_run_nonce",
                        )
                    )
                if len(external_identities) != 17:
                    raise CausalProductionError(
                        "external reviewer identities are not gate-globally unique"
                    )
                if all_identities.intersection(external_identities):
                    raise CausalProductionError(
                        "external reviewer identity is globally reused"
                    )
                all_identities.update(external_identities)
                externally_replayed_gates.add(gate_sha256)
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
            first = rows[0]
            for row in rows[1:]:
                for field in (
                    "run_identity",
                    "model_config",
                    "stable_contract",
                    "training_contract",
                    "precursor",
                    "evidence",
                    "topology",
                    "route",
                ):
                    if row[field] != first[field]:
                        raise CausalProductionError(
                            f"run step contract drifted outside checkpoint-local state: {field}"
                        )
            if rows[0]["parent_checkpoint_weight_sha256"] != rows[0]["precursor"]["parameter_tensor_sha256"]:
                raise CausalProductionError("step-0 parent must be the precursor tensor digest")
            if rows[1]["parent_checkpoint_weight_sha256"] != rows[0]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]:
                raise CausalProductionError("step-100 parent chain differs")
            if rows[2]["parent_checkpoint_weight_sha256"] != rows[1]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]:
                raise CausalProductionError("step-200 parent chain differs")
    for seed in SEEDS:
        for step in CHECKPOINT_STEPS:
            same_step = [evidence[(seed, route, step)][0] for route in CAUSAL_ROUTES]
            compare_same_seed_route_contracts(same_step)
            reference = same_step[0]
            for row in same_step[1:]:
                for field in (
                    "stable_contract",
                    "training_contract",
                    "precursor",
                    "evidence",
                    "topology",
                ):
                    if row[field] != reference[field]:
                        raise CausalProductionError(
                            f"same-seed/step matched contract differs: {field}"
                        )
            shared_topology_fields = (
                "persistent_parameter_schema",
                "persistent_parameter_schema_sha256",
                "persistent_buffer_schema",
                "persistent_buffer_schema_sha256",
                "step0_parameter_tensor_sha256",
                "optimizer_group_parameter_names",
                "optimizer_group_names_sha256",
                "optimizer_group_hyperparameters",
                "optimizer_hyperparameters_sha256",
                "scheduler_state_receipt",
                "scheduler_state_sha256",
            )
            for row in same_step[1:]:
                for field in shared_topology_fields:
                    if row["parameter_topology"][field] != reference[
                        "parameter_topology"
                    ][field]:
                        raise CausalProductionError(
                            "same-seed route model/optimizer/scheduler topology differs"
                        )
            if step == 0:
                for row in same_step[1:]:
                    for field in (
                        "optimizer_state_parameter_names",
                        "optimizer_state_names_sha256",
                        "optimizer_state_receipt",
                        "optimizer_state_sha256",
                    ):
                        if row["parameter_topology"][field] != reference[
                            "parameter_topology"
                        ][field]:
                            raise CausalProductionError(
                                "same-seed step-0 optimizer state differs across routes"
                            )
            equal = same_step[1]
            semantic = same_step[2]
            direct = same_step[0]
            for field in (
                "persistent_parameter_schema",
                "persistent_parameter_schema_sha256",
                "persistent_buffer_schema",
                "persistent_buffer_schema_sha256",
                "optimizer_group_parameter_names",
                "optimizer_group_names_sha256",
                "optimizer_group_hyperparameters",
                "optimizer_hyperparameters_sha256",
                "optimizer_state_parameter_names",
                "optimizer_state_names_sha256",
                "scheduler_state_receipt",
                "scheduler_state_sha256",
            ):
                if equal["parameter_topology"][field] != semantic["parameter_topology"][field]:
                    raise CausalProductionError(
                        "equal-bit and semantic active optimizer/scheduler state differs"
                    )
            for field in (
                "active_parameter_names",
                "active_parameter_names_sha256",
                "inactive_parameter_names",
                "inactive_parameter_names_sha256",
                "observed_gradient_parameter_names",
                "observed_gradient_parameter_names_sha256",
                "action_loss_gradient_parameter_names",
                "action_loss_gradient_parameter_names_sha256",
                "action_loss_reaches_planner",
                "active_gradient_parameter_count",
                "unused_parameter_names",
            ):
                if equal["gradient_reachability_receipt"][field] != semantic["gradient_reachability_receipt"][field]:
                    raise CausalProductionError(
                        "equal-bit and semantic gradient reachability differs"
                    )
            direct_reachability = direct["gradient_reachability_receipt"]
            strict_reachability = equal["gradient_reachability_receipt"]
            direct_only = set(DIRECT_UNUSED_PARAMETER_NAMES)
            if (
                set(direct_reachability["inactive_parameter_names"])
                != set(strict_reachability["inactive_parameter_names"])
                | direct_only
                or set(direct_reachability["active_parameter_names"])
                != set(strict_reachability["active_parameter_names"])
                - direct_only
            ):
                raise CausalProductionError(
                    "direct active/inactive difference exceeds the exact 22-to-action allowlist"
                )
            if step > 0 and (
                set(direct_reachability["observed_gradient_parameter_names"])
                != set(strict_reachability["observed_gradient_parameter_names"])
                - direct_only
                or set(
                    direct["parameter_topology"][
                        "optimizer_state_parameter_names"
                    ]
                )
                != set(
                    equal["parameter_topology"][
                        "optimizer_state_parameter_names"
                    ]
                )
                - direct_only
            ):
                raise CausalProductionError(
                    "direct optimizer/gradient population difference exceeds the exact allowlist"
                )
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

"""Route-aware held-out result v3 and 27-checkpoint aggregate v5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from world_action_model.contact_causal_production import (
    AGGREGATE_VERSION,
    CAUSAL_ROUTES,
    CHECKPOINT_STEPS,
    CausalProductionError,
    EVALUATOR_VERSION,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    RESULT_VERSION,
    SEEDS,
    route_packet_kind,
    sha256_json,
    validate_27_checkpoint_matrix,
)


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def validate_route_evaluator_output(value: object, *, route: str) -> dict[str, Any]:
    output = _exact(
        value,
        {
            "evaluator_version",
            "matrix_id",
            "causal_route_id",
            "contact_aux_prediction_source",
            "contact_target_model_input",
            "contact_condition_mask_model_input",
            "oracle_used",
            "packet_kind",
            "packet_present",
            "packet_sha256",
            "semantic_aux_logits_sha256",
            "action_prediction_sha256",
            "action_metrics_contract_sha256",
            "packet_metrics",
        },
        "causal route evaluator output",
    )
    expected_kind = route_packet_kind(route)
    if (
        output["evaluator_version"] != EVALUATOR_VERSION
        or output["matrix_id"] != MATRIX_ID
        or output["causal_route_id"] != route
        or output["contact_aux_prediction_source"]
        != "shared_semantic_aux_decoder_logits"
        or output["contact_target_model_input"] is not False
        or output["contact_condition_mask_model_input"] is not False
        or output["oracle_used"] is not False
        or output["packet_kind"] != expected_kind
        or output["packet_present"] is not (route != "aux_only_direct_q12")
    ):
        raise CausalProductionError("route-aware evaluator output semantics differ")
    for field in (
        "semantic_aux_logits_sha256",
        "action_prediction_sha256",
        "action_metrics_contract_sha256",
    ):
        _sha(output[field], f"evaluator output {field}")
    if route == "aux_only_direct_q12":
        if output["packet_sha256"] is not None or output["packet_metrics"] != {
            "kind": "none_direct_reference",
            "semantic_field_metrics_allowed": False,
        }:
            raise CausalProductionError("direct evaluator must not fabricate a packet")
    elif route == "equal_bit_unstructured_hard_st":
        _sha(output["packet_sha256"], "equal-bit packet")
        metrics = _exact(
            output["packet_metrics"],
            {
                "kind",
                "codec_validity",
                "utilization",
                "entropy",
                "invalid_to_zero_rate",
                "semantic_field_metrics_allowed",
            },
            "equal-bit packet metrics",
        )
        if metrics["kind"] != "unstructured_code" or metrics["semantic_field_metrics_allowed"] is not False:
            raise CausalProductionError("equal-bit code must not be interpreted as contact semantics")
    else:
        _sha(output["packet_sha256"], "semantic packet")
        metrics = _exact(
            output["packet_metrics"],
            {
                "kind",
                "grammar_validity",
                "field_accuracy",
                "semantic_field_metrics_allowed",
            },
            "semantic grammar packet metrics",
        )
        if metrics["kind"] != "semantic_grammar" or metrics["semantic_field_metrics_allowed"] is not True:
            raise CausalProductionError("semantic packet metrics differ")
    return output


def validate_result_v3(value: object) -> dict[str, Any]:
    result = _exact(
        value,
        {
            "result_version",
            "evaluator_version",
            "matrix_schema_version",
            "matrix_id",
            "seed",
            "causal_route_id",
            "checkpoint_manifest_sha256",
            "precursor_receipt_bytes_sha256",
            "fixed_window_manifest_sha256",
            "action_metrics_contract_sha256",
            "contact_aux_metrics_contract_sha256",
            "checkpoint_steps",
            "primary_checkpoint_step",
            "best_checkpoint_selection_allowed",
            "checkpoint_outputs",
            "oracle_used",
            "benefit_claimed",
        },
        "causal held-out result",
    )
    route = result["causal_route_id"]
    if (
        result["result_version"] != RESULT_VERSION
        or result["evaluator_version"] != EVALUATOR_VERSION
        or result["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or result["matrix_id"] != MATRIX_ID
        or result["seed"] not in SEEDS
        or route not in CAUSAL_ROUTES
        or result["checkpoint_steps"] != list(CHECKPOINT_STEPS)
        or result["primary_checkpoint_step"] != 200
        or result["best_checkpoint_selection_allowed"] is not False
        or result["oracle_used"] is not False
        or result["benefit_claimed"] is not False
    ):
        raise CausalProductionError("causal held-out result identity differs")
    for field in (
        "checkpoint_manifest_sha256",
        "precursor_receipt_bytes_sha256",
        "fixed_window_manifest_sha256",
        "action_metrics_contract_sha256",
        "contact_aux_metrics_contract_sha256",
    ):
        _sha(result[field], f"causal result {field}")
    outputs = result["checkpoint_outputs"]
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise CausalProductionError("causal result needs exactly three checkpoint outputs")
    seen = set()
    for row in outputs:
        row = _exact(
            row,
            {
                "checkpoint_step",
                "checkpoint_acceptance_sha256",
                "fixed_window_set_sha256",
                "route_output",
                "action_metrics",
                "contact_aux_metrics",
            },
            "causal checkpoint result row",
        )
        step = row["checkpoint_step"]
        if isinstance(step, bool) or step not in CHECKPOINT_STEPS or step in seen:
            raise CausalProductionError("causal result checkpoint step is invalid/duplicate")
        seen.add(step)
        _sha(row["checkpoint_acceptance_sha256"], "result checkpoint acceptance")
        if row["fixed_window_set_sha256"] != result["fixed_window_manifest_sha256"]:
            raise CausalProductionError("causal result fixed windows differ by checkpoint")
        validate_route_evaluator_output(row["route_output"], route=route)
        if not isinstance(row["action_metrics"], dict) or not isinstance(row["contact_aux_metrics"], dict):
            raise CausalProductionError("causal result metric bundles must be mappings")
    return result


def validate_aggregate_v5(
    value: object,
    *,
    evidence: Mapping[tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]],
    verify_files: bool = True,
) -> dict[str, Any]:
    aggregate = _exact(
        value,
        {
            "aggregate_version",
            "matrix_schema_version",
            "matrix_id",
            "seeds",
            "causal_routes",
            "checkpoint_steps",
            "checkpoint_acceptance_count",
            "result_sha256_by_seed_route",
            "matrix_contract",
            "primary_checkpoint_step",
            "oracle_included",
            "production_data_admission",
            "allow_5b_training",
            "allow_server_training",
            "benefit_claimed",
        },
        "causal held-out aggregate",
    )
    matrix_contract = validate_27_checkpoint_matrix(
        evidence, verify_files=verify_files
    )
    if (
        aggregate["aggregate_version"] != AGGREGATE_VERSION
        or aggregate["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or aggregate["matrix_id"] != MATRIX_ID
        or aggregate["seeds"] != list(SEEDS)
        or aggregate["causal_routes"] != list(CAUSAL_ROUTES)
        or aggregate["checkpoint_steps"] != list(CHECKPOINT_STEPS)
        or aggregate["checkpoint_acceptance_count"] != 27
        or aggregate["matrix_contract"] != matrix_contract
        or aggregate["primary_checkpoint_step"] != 200
        or aggregate["oracle_included"] is not False
        or aggregate["production_data_admission"] is not False
        or aggregate["allow_5b_training"] is not False
        or aggregate["allow_server_training"] is not False
        or aggregate["benefit_claimed"] is not False
    ):
        raise CausalProductionError("causal aggregate identity/admission differs")
    results = aggregate["result_sha256_by_seed_route"]
    expected_keys = [f"{seed}/{route}" for seed in SEEDS for route in CAUSAL_ROUTES]
    if not isinstance(results, dict) or list(results) != expected_keys:
        raise CausalProductionError("causal aggregate result set/order differs")
    for key, digest in results.items():
        _sha(digest, f"causal aggregate result {key}")
    return aggregate


def build_aggregate_v5(
    *,
    result_sha256_by_seed_route: Mapping[str, str],
    evidence: Mapping[tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]],
    verify_files: bool = True,
) -> dict[str, Any]:
    matrix_contract = validate_27_checkpoint_matrix(
        evidence, verify_files=verify_files
    )
    return {
        "aggregate_version": AGGREGATE_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "seeds": list(SEEDS),
        "causal_routes": list(CAUSAL_ROUTES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_acceptance_count": 27,
        "result_sha256_by_seed_route": dict(result_sha256_by_seed_route),
        "matrix_contract": matrix_contract,
        "primary_checkpoint_step": 200,
        "oracle_included": False,
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }

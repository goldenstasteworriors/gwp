"""Fail-closed contracts for spec-only contact causal-route controls.

The two new controls are implemented for CPU auditing only.  Nothing in this
module changes the production matched-control runner admission set.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Mapping

from world_action_model.contact_transition_schema import (
    EQUAL_BIT_CODEC_VERSION,
    EQUAL_BIT_MIXING_MATRIX_SHA256,
    EQUAL_BIT_VALID_STATES,
    HARD_SEGMENT_GRAMMAR_VERSION,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    contact_transport_capacity_metadata,
    unstructured_capacity_metadata,
)


CONTACT_CAUSAL_ROUTE_CONTRACT_VERSION = "contact_causal_route_contract_v3"
CONTACT_PACKET_RECEIPT_VERSION = "contact_action_packet_receipt_v1"
SPEC_ONLY_IMPLEMENTATION_STATUS = "implemented_cpu_route_unadmitted"


class ContactCausalRoute(str, Enum):
    SOFT_SEMANTIC = "soft_semantic"
    HARD_ST_SEMANTIC = "hard_st_semantic"
    HARD_DETACHED_SEMANTIC = "hard_detached_semantic"
    AUX_ONLY_DIRECT_Q12 = "aux_only_direct_q12"
    EQUAL_BIT_UNSTRUCTURED_HARD_ST = "equal_bit_unstructured_hard_st"


SPEC_ONLY_TRAINER_ROUTES = frozenset(
    {
        ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
        ContactCausalRoute.HARD_ST_SEMANTIC,
    }
)


_SEMANTIC_ROUTE_BY_TRANSPORT = {
    "soft": ContactCausalRoute.SOFT_SEMANTIC,
    "hard_st": ContactCausalRoute.HARD_ST_SEMANTIC,
    "hard_detached": ContactCausalRoute.HARD_DETACHED_SEMANTIC,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def contact_packet_receipt(
    *,
    route_receipt: Mapping[str, Any],
    packet_bytes: bytes,
    packet_dtype: str,
    packet_shape: tuple[int, ...] | list[int],
) -> Mapping[str, Any]:
    """Bind canonical packet bytes to the model's immutable route contract."""

    route_receipt_sha256 = route_receipt.get("receipt_sha256")
    if not isinstance(route_receipt_sha256, str) or len(route_receipt_sha256) != 64:
        raise ValueError("packet receipt requires a canonical route receipt")
    payload = {
        "receipt_version": CONTACT_PACKET_RECEIPT_VERSION,
        "route": route_receipt.get("route"),
        "route_contract_version": route_receipt.get("contract_version"),
        "route_receipt_sha256": route_receipt_sha256,
        "packet_dtype": str(packet_dtype),
        "packet_shape": [int(value) for value in packet_shape],
        "packet_bytes_sha256": hashlib.sha256(packet_bytes).hexdigest(),
    }
    return {**payload, "receipt_sha256": sha256_json(payload)}


def validate_spec_only_trainer_route_config(
    model_config: Mapping[str, Any],
) -> ContactCausalRoute | None:
    """Fail closed on split-brain spec route and auxiliary-loss settings."""

    transformer_config = model_config.get("transformer", {})
    top_raw = model_config.get("contact_causal_route")
    transformer_raw = transformer_config.get("contact_causal_route")
    top_present = top_raw is not None and str(top_raw).strip() != ""
    transformer_present = transformer_raw is not None and str(transformer_raw).strip() != ""
    if not top_present and not transformer_present:
        return None
    if not top_present or not transformer_present:
        raise ValueError(
            "spec-only contact route must be explicit at both models and transformer levels"
        )
    try:
        top_route = ContactCausalRoute(str(top_raw).strip())
    except ValueError as exc:
        raise ValueError(f"unknown top-level contact route {top_raw!r}") from exc
    transformer_route = resolve_contact_causal_route(
        transformer_raw,
        contact_semantic_bottleneck=bool(
            transformer_config.get("contact_semantic_bottleneck", False)
        ),
        contact_semantic_transport_mode=str(
            transformer_config.get("contact_semantic_transport_mode", "soft")
        ),
    )
    if top_route is not transformer_route or top_route not in SPEC_ONLY_TRAINER_ROUTES:
        if top_route is not transformer_route:
            raise ValueError(
                "top-level contact route differs from the transformer resolved route"
            )
        return None
    try:
        loss_weight = float(model_config.get("contact_loss_weight", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("spec-only contact loss weight must be finite and positive") from exc
    if not math.isfinite(loss_weight) or loss_weight <= 0.0:
        raise ValueError("spec-only contact loss weight must be finite and positive")
    if not bool(model_config.get("require_contact_targets", False)):
        raise ValueError("spec-only contact routes require contact targets")
    if not bool(model_config.get("require_positive_contact_supervision", False)):
        raise ValueError("spec-only contact routes require positive masked supervision")
    if str(model_config.get("contact_schema_version", "")) != SEGMENT_SCHEMA_VERSION:
        raise ValueError("spec-only contact routes require v1.1 segment targets")
    if top_route in {
        ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
    } and str(model_config.get("contact_condition_mode", "predicted")) != "predicted":
        raise ValueError(f"route {top_route.value!r} rejects oracle conditioning")
    return top_route


def resolve_contact_causal_route(
    route: str | ContactCausalRoute | None,
    *,
    contact_semantic_bottleneck: bool,
    contact_semantic_transport_mode: str,
) -> ContactCausalRoute | None:
    """Resolve an explicit route or map the unchanged legacy semantic route."""

    if route is None or str(route).strip() == "":
        if not contact_semantic_bottleneck:
            return None
        try:
            return _SEMANTIC_ROUTE_BY_TRANSPORT[str(contact_semantic_transport_mode)]
        except KeyError as exc:
            raise ValueError(
                "legacy semantic bottleneck has an unsupported transport mode"
            ) from exc
    try:
        resolved = (
            route
            if isinstance(route, ContactCausalRoute)
            else ContactCausalRoute(str(route))
        )
    except ValueError as exc:
        expected = tuple(item.value for item in ContactCausalRoute)
        raise ValueError(
            f"unsupported contact_causal_route={route!r}; expected one of {expected}"
        ) from exc
    expected_transport = {
        ContactCausalRoute.SOFT_SEMANTIC: "soft",
        ContactCausalRoute.HARD_ST_SEMANTIC: "hard_st",
        ContactCausalRoute.HARD_DETACHED_SEMANTIC: "hard_detached",
        ContactCausalRoute.AUX_ONLY_DIRECT_Q12: "hard_st",
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST: "hard_st",
    }[resolved]
    if str(contact_semantic_transport_mode) != expected_transport:
        raise ValueError(
            f"route {resolved.value!r} requires contact_semantic_transport_mode="
            f"{expected_transport!r}, got {contact_semantic_transport_mode!r}"
        )
    return resolved


def validate_contact_route_model_config(
    route: ContactCausalRoute | None,
    *,
    num_contact_tokens: int,
    contact_target_dim: int,
    contact_condition_dim: int,
    contact_bottleneck: bool,
    contact_schema_version: str,
    contact_semantic_bottleneck: bool,
    contact_time_bins: int,
) -> None:
    """Reject contradictory legacy booleans before any forward execution."""

    if route is None:
        return
    if contact_schema_version != SEGMENT_SCHEMA_VERSION:
        raise ValueError(
            f"causal routes require contact_schema_version={SEGMENT_SCHEMA_VERSION!r}"
        )
    if (
        contact_target_dim != SEGMENT_TARGET_DIM
        or contact_condition_dim != SEGMENT_TARGET_DIM
    ):
        raise ValueError(
            f"causal routes require target/condition width {SEGMENT_TARGET_DIM}"
        )
    if num_contact_tokens <= 0:
        raise ValueError("causal routes require num_contact_tokens > 0")

    direct = route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12
    if direct:
        if num_contact_tokens != 12:
            raise ValueError("aux_only_direct_q12 requires exactly 12 planner queries")
        if contact_bottleneck or contact_semantic_bottleneck:
            raise ValueError(
                "aux_only_direct_q12 requires both bottleneck booleans false; "
                "the standalone planner remains enabled by the route enum"
            )
        return

    if not contact_bottleneck or not contact_semantic_bottleneck:
        raise ValueError(
            f"route {route.value!r} requires contact_bottleneck=True and "
            "contact_semantic_bottleneck=True"
        )
    if route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
        if num_contact_tokens != 12 or contact_time_bins != 16:
            raise ValueError(
                "equal_bit_unstructured_hard_st is receipt-bound to Q=12/B=16"
            )
    if route is ContactCausalRoute.HARD_ST_SEMANTIC and contact_time_bins != 16:
        raise ValueError("explicit hard_st_semantic is receipt-bound to B=16")


def route_uses_standalone_planner(route: ContactCausalRoute | None) -> bool:
    return route is not None


def route_action_uses_packet(route: ContactCausalRoute | None) -> bool:
    return route is not None and route is not ContactCausalRoute.AUX_ONLY_DIRECT_Q12


def route_action_attends_reference(route: ContactCausalRoute | None) -> bool:
    return route is None or route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12


def route_rejects_oracle(route: ContactCausalRoute | None) -> bool:
    return route in {
        ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
    }


def contact_route_metadata(
    route: ContactCausalRoute,
    *,
    num_queries: int,
    time_bins: int,
) -> Mapping[str, Any]:
    """Create a canonical route receipt and its SHA-256 (never Python hash)."""

    direct = route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12
    equal_bit = route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
    transport = {
        ContactCausalRoute.SOFT_SEMANTIC: "soft",
        ContactCausalRoute.HARD_ST_SEMANTIC: "hard_st",
        ContactCausalRoute.HARD_DETACHED_SEMANTIC: "hard_detached",
        ContactCausalRoute.AUX_ONLY_DIRECT_Q12: "hard_st",
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST: "hard_st",
    }[route]
    if equal_bit:
        capacity: Mapping[str, Any] | None = unstructured_capacity_metadata(
            K=EQUAL_BIT_VALID_STATES,
            Q=num_queries,
            width=SEGMENT_TARGET_DIM,
        )
    elif direct:
        capacity = None
    else:
        capacity = contact_transport_capacity_metadata(
            mode=transport,
            time_bins=time_bins,
            num_queries=num_queries,
        )
    finite_semantic_packet = bool(
        not direct
        and not equal_bit
        and capacity is not None
        and capacity.get("finite_capacity", False)
    )
    payload = {
        "contract_version": CONTACT_CAUSAL_ROUTE_CONTRACT_VERSION,
        "implementation_status": SPEC_ONLY_IMPLEMENTATION_STATUS,
        "production_runner_admitted": False,
        "route": route.value,
        "planner_queries": num_queries,
        "planner_transport_mode": transport,
        "planner_parameters_and_forward_active": True,
        "matches_planner_compute_not_action_visible_bit_capacity": direct,
        "contact_aux_prediction_active": True,
        "action_attends_reference": direct,
        "action_visible_packet": (
            "none_direct_reference"
            if direct
            else "canonical_equal_bit_16_plus_6_positive_zero"
            if equal_bit
            else "canonical_contact_semantic_22d"
        ),
        "action_readable_sources": (
            ["state", "instruction_text", "noisy_action", "current_reference"]
            if direct
            else ["state", "instruction_text", "noisy_action", "route_packet"]
        ),
        "action_reads_raw_planner_logits": False,
        "action_reads_planner_hidden": False,
        "action_reads_contact_ground_truth": False,
        "action_reads_future_reference": False,
        "strict_image_cross_attention_policy": (
            "reject_nonempty" if not direct else "legacy_direct_allowed"
        ),
        "oracle_policy": (
            "reject" if route_rejects_oracle(route) else "semantic_field_merge_only"
        ),
        "cache_policy": (
            "direct_reference_kv_without_planner_logits_hidden_or_semantics"
            if direct
            else "state_and_canonical_code_kv_without_reference_or_raw_planner_values"
            if equal_bit
            else "legacy_semantic_packet_kv_without_reference_aux_logits_output_only"
        ),
        "capacity": capacity,
        "action_visible_packet_finite_semantic_grammar": finite_semantic_packet,
        "hard_semantic_grammar_version": (
            HARD_SEGMENT_GRAMMAR_VERSION
            if transport in {"hard_st", "hard_detached"}
            and time_bins == 16
            and not direct
            and not equal_bit
            else None
        ),
        "equal_bit_codec_version": EQUAL_BIT_CODEC_VERSION if equal_bit else None,
        "equal_bit_mixing_matrix_sha256": (
            EQUAL_BIT_MIXING_MATRIX_SHA256 if equal_bit else None
        ),
        "cache_packet_receipt_version": (
            CONTACT_PACKET_RECEIPT_VERSION if not direct else None
        ),
    }
    return {**payload, "receipt_sha256": sha256_json(payload)}

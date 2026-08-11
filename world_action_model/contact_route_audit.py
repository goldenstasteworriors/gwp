"""CPU-only causal-route audit and same-noise sensitivity harness.

This module verifies information-flow mechanics.  It does not evaluate task
benefit, physical feasibility, simulator behavior, or production admission.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import subprocess
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from world_action_model.contact_route_contract import (
    ContactCausalRoute,
    canonical_json_bytes,
    contact_packet_receipt,
    contact_route_metadata,
    resolve_contact_causal_route,
    sha256_json,
    validate_contact_route_model_config,
    validate_spec_only_trainer_route_config,
)
from world_action_model.contact_transition_schema import (
    EQUAL_BIT_MIXING_MATRIX,
    EQUAL_BIT_MIXING_MATRIX_SHA256,
    EQUAL_BIT_NUM_BITS,
    EQUAL_BIT_VALID_STATES,
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_TARGET_DIM,
    canonicalize_unstructured_ids,
    contact_transport_capacity_metadata,
    unstructured_capacity_metadata,
    unstructured_code_to_ids,
    unstructured_ids_to_code,
    validate_hard_contact_segment_code,
    validated_equal_bit_mixing_matrix,
)
from world_action_model.matched_control_hashes import IMPLEMENTED_RUNNER_VARIANTS
from world_action_model.models.transformer_wa_casual_mot import (
    CasualWorldActionTransformer_MoT,
    build_mot_attention_mask,
)
from world_action_model.contact_supervision_contract import (
    CONTACT_GROUPED_SUPERVISION_RULE_VERSION,
    effective_contact_supervision_count,
    grouped_contact_supervision_weights,
    validate_positive_contact_supervision,
)


AUDIT_VERSION = "contact_causal_route_cpu_audit_v3"
AUDIT_SEED = 38_202_608_11
CLAMP_TOLERANCE = 1e-6
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_GOLDEN_BASELINE_COMMIT = "d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88"
LEGACY_GOLDEN = {
    "soft": {
        "packet": "69bb119555dd526f252abc65988791c69983c720d0393f3e4bba3a0ddca58415",
        "action": "52dc091fa61b5deb3a26b321bea74b16cef6ecdcb8be066dc6ba9ab12ae09887",
        "contact": "8578b1f80772362b0fcf8f22bb4e03221c120433e734e6738f49e86c9850d7a8",
    },
    "hard_st": {
        "packet": "730347520db5eb60b36e6c1aecb42bb2692eef893d68f3685ee4db2edea0a6ae",
        "action": "ac3589fbe3c0d0c744772be16e29054ab759e1c974a447d37874b27befa2b3e6",
        "contact": "edc21ab0c4f418bb5d6a55059805f26b8005e010ef681d79c6ec09c80553c583",
    },
    "hard_detached": {
        "packet": "730347520db5eb60b36e6c1aecb42bb2692eef893d68f3685ee4db2edea0a6ae",
        "action": "9744ffebe401030fcefaee26dc38ea212e7646b7e44027d09e1420d225c873eb",
        "contact": "d2d479f3b703addaf4a4eff25b4718827e2347f37998fd82b4c91688633ca043",
    },
}
LEGACY_STATE_DICT_KEYS_SHA256 = "b5988dd88b5e77fd927e451efa6142a4671fb1d2fa05ec5ed803f3e3339df69f"


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    payload = (
        str(tensor.dtype).encode("ascii")
        + repr(tuple(tensor.shape)).encode("ascii")
        + tensor.view(torch.uint8).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


def _tiny_model(route: ContactCausalRoute, *, transport: str = "hard_st"):
    direct = route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12
    return CasualWorldActionTransformer_MoT(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=32,
        num_layers=1,
        action_expert_dim=32,
        action_ffn_dim=64,
        in_action_channels=4,
        out_action_channels=4,
        num_embodiments=1,
        num_contact_tokens=12,
        contact_target_dim=SEGMENT_TARGET_DIM,
        contact_condition_dim=SEGMENT_TARGET_DIM,
        contact_bottleneck=not direct,
        contact_schema_version="contact_segment_v1_1",
        contact_semantic_bottleneck=not direct,
        contact_semantic_transport_mode=transport,
        contact_time_bins=16,
        contact_causal_route=route.value,
    )


def _legacy_semantic_model(
    transport: str, explicit_route: bool, *, time_bins: int = 16
):
    route_by_transport = {
        "soft": ContactCausalRoute.SOFT_SEMANTIC.value,
        "hard_st": ContactCausalRoute.HARD_ST_SEMANTIC.value,
        "hard_detached": ContactCausalRoute.HARD_DETACHED_SEMANTIC.value,
    }
    kwargs = {}
    if explicit_route:
        kwargs["contact_causal_route"] = route_by_transport[transport]
    return CasualWorldActionTransformer_MoT(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=32,
        num_layers=1,
        action_expert_dim=32,
        action_ffn_dim=64,
        in_action_channels=4,
        out_action_channels=4,
        num_embodiments=1,
        num_contact_tokens=12,
        contact_target_dim=SEGMENT_TARGET_DIM,
        contact_condition_dim=SEGMENT_TARGET_DIM,
        contact_bottleneck=True,
        contact_schema_version="contact_segment_v1_1",
        contact_semantic_bottleneck=True,
        contact_semantic_transport_mode=transport,
        contact_time_bins=time_bins,
        **kwargs,
    )


def _inputs(seed: int = AUDIT_SEED) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "ref_latents": torch.randn(1, 4, 1, 2, 2, generator=generator),
        "future_latents": torch.randn(1, 4, 1, 2, 2, generator=generator),
        "encoder_hidden_states": torch.randn(1, 2, 8, generator=generator),
        "state": torch.randn(1, 2, 4, generator=generator),
        "action": torch.randn(1, 3, 4, generator=generator),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }


def _action_only(model, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model._forward_train_action_only_no_video(
        ref_latents=inputs["ref_latents"],
        noisy_latents=None,
        timestep=torch.zeros(1, 6),
        encoder_hidden_states=inputs["encoder_hidden_states"],
        state=inputs["state"],
        action=inputs["action"],
        embodiment_id=inputs["embodiment_id"],
        return_dict=True,
    )


def _action_with_future(model, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model._forward_train(
        ref_latents=inputs["ref_latents"],
        noisy_latents=inputs["future_latents"],
        timestep=torch.zeros(1, 7),
        encoder_hidden_states=inputs["encoder_hidden_states"],
        state=inputs["state"],
        action=inputs["action"],
        embodiment_id=inputs["embodiment_id"],
        return_dict=True,
    )


def _bind_fixed_packet(
    model: CasualWorldActionTransformer_MoT,
    packet: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    def fixed_plan(self, state, ref_visual_hidden_states, encoder_hidden_states, **kwargs):
        del ref_visual_hidden_states, encoder_hidden_states, kwargs
        batch_size = state.shape[0]
        return (
            logits.to(device=state.device, dtype=state.dtype).expand(batch_size, -1, -1),
            packet.to(device=state.device, dtype=state.dtype).expand(batch_size, -1, -1),
        )

    model.plan_contact_route = types.MethodType(fixed_plan, model)


def _semantic_packets() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.zeros(1, 12, SEGMENT_TARGET_DIM)
    second = first.clone()
    second[..., 0, SEGMENT_FIELD_SLICES["presence"]] = 1.0
    for field in ("side", "site", "object_role", "contact_mode"):
        second[..., 0, SEGMENT_FIELD_SLICES[field].start] = 1.0
    second[..., 0, SEGMENT_FIELD_SLICES["start_tau"]] = 0.0
    second[..., 0, SEGMENT_FIELD_SLICES["end_tau"]] = 1.0
    return first, second


def _equal_packets() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.from_numpy(
        unstructured_ids_to_code(np.zeros((1, 12), dtype=np.uint16))
    )
    ids = np.zeros((1, 12), dtype=np.uint16)
    ids[0, 0] = 1
    second = torch.from_numpy(unstructured_ids_to_code(ids))
    return first, second


def _all_hard_semantic_states_b16() -> np.ndarray:
    """Enumerate the exact one-query hard semantic grammar (48,673 states)."""

    timing_states = []
    for start in range(16):
        for end in range(start, 16):
            timing_states.append((start, end, 0, 0))
    timing_states.extend((0, end, 1, 0) for end in range(16))
    timing_states.extend((start, 15, 0, 1) for start in range(16))
    timing_states.append((0, 15, 1, 1))
    states = np.zeros((EQUAL_BIT_VALID_STATES, SEGMENT_TARGET_DIM), dtype=np.float32)
    cursor = 1
    for side in range(3):
        for site in range(6):
            for role in range(4):
                for mode in range(4):
                    for start, end, left, right in timing_states:
                        row = states[cursor]
                        row[SEGMENT_FIELD_SLICES["presence"]] = 1
                        row[SEGMENT_FIELD_SLICES["side"].start + side] = 1
                        row[SEGMENT_FIELD_SLICES["site"].start + site] = 1
                        row[SEGMENT_FIELD_SLICES["object_role"].start + role] = 1
                        row[SEGMENT_FIELD_SLICES["contact_mode"].start + mode] = 1
                        row[SEGMENT_FIELD_SLICES["start_tau"]] = start / 15.0
                        row[SEGMENT_FIELD_SLICES["end_tau"]] = end / 15.0
                        row[SEGMENT_FIELD_SLICES["left_censored"]] = left
                        row[SEGMENT_FIELD_SLICES["right_censored"]] = right
                        cursor += 1
    if cursor != EQUAL_BIT_VALID_STATES:
        raise AssertionError("hard semantic grammar enumeration count differs")
    return states


def _cache_domain_audit(
    model: CasualWorldActionTransformer_MoT,
    cache: dict[str, Any],
    route: ContactCausalRoute,
) -> dict[str, bool]:
    packet_key = (
        "contact_code"
        if route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
        else "contact_semantic"
    )
    original = cache[packet_key]
    checks: dict[str, bool] = {}

    def reject(name: str, packet: torch.Tensor, *, self_consistent: bool = False) -> None:
        probe = dict(cache)
        probe[packet_key] = packet
        if self_consistent:
            probe["contact_packet_receipt"] = contact_packet_receipt(
                route_receipt=model.contact_route_metadata,
                packet_bytes=model._canonical_packet_bytes(packet),
                packet_dtype=str(packet.dtype),
                packet_shape=tuple(packet.shape),
            )
        try:
            model._validate_contact_action_only_cache(probe)
        except RuntimeError:
            checks[name] = True
        else:
            raise AssertionError(f"cache domain accepted negative case {name}")

    reject("wrong_shape_rejected", original[..., :-1])
    reject("wrong_dtype_rejected", original.to(torch.float64))
    bad = original.clone()
    bad.reshape(-1)[0] = float("nan")
    reject("nonfinite_rejected", bad)
    bad = original.clone()
    zero_index = torch.nonzero(bad == 0, as_tuple=False)[0]
    bad[tuple(zero_index)] = -0.0
    reject("signed_negative_zero_rejected", bad)

    if route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
        bad = original.clone()
        bad[0, 0, 0] = 0.5
        reject("fractional_bit_rejected", bad)
        bad = original.clone()
        bad[0, 0, EQUAL_BIT_NUM_BITS] = 1
        reject("nonzero_padding_rejected", bad)
        bad = original.clone()
        raw_invalid = EQUAL_BIT_VALID_STATES
        bad[0, 0, :EQUAL_BIT_NUM_BITS] = torch.tensor(
            [(raw_invalid >> bit) & 1 for bit in range(EQUAL_BIT_NUM_BITS)],
            dtype=bad.dtype,
        )
        reject("invalid_little_endian_id_rejected", bad)
        valid_changed = _equal_packets()[1].to(dtype=original.dtype)
    else:
        valid = _semantic_packets()[1].to(dtype=original.dtype)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["presence"]] = 0.5
        reject("nonbinary_presence_rejected", bad)
        bad = original.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["side"].start] = 1
        reject("noncanonical_absent_rejected", bad)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["side"]] = 0
        reject("categorical_not_one_hot_rejected", bad)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["side"].start + 1] = 1
        reject("categorical_multi_hot_rejected", bad)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["start_tau"]] = 0.1
        reject("off_lattice_tau_rejected", bad)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["start_tau"]] = 1
        bad[0, 0, SEGMENT_FIELD_SLICES["end_tau"]] = 0
        reject("unordered_uncensored_tau_rejected", bad)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["left_censored"]] = 1
        bad[0, 0, SEGMENT_FIELD_SLICES["start_tau"]] = 1
        reject("left_censor_tau_mismatch_rejected", bad)
        bad = valid.clone()
        bad[0, 0, SEGMENT_FIELD_SLICES["right_censored"]] = 1
        bad[0, 0, SEGMENT_FIELD_SLICES["end_tau"]] = 0
        reject("right_censor_tau_mismatch_rejected", bad)
        valid_changed = valid

    changed = dict(cache)
    changed["contact_packet_receipt"] = dict(cache["contact_packet_receipt"])
    changed["contact_packet_receipt"]["packet_bytes_sha256"] = "0" * 64
    try:
        model._validate_contact_action_only_cache(changed)
    except RuntimeError:
        checks["receipt_payload_tamper_rejected"] = True
    else:
        raise AssertionError("cache accepted a tampered packet receipt")
    reject(
        "self_consistent_replacement_rejected_by_model_receipt",
        valid_changed,
        self_consistent=True,
    )
    return checks


def _semantic_cache_resolution_audit(
    inputs: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Exercise hard grammar validation for implicit and detached routes."""

    cases = (
        ("implicit_hard_st", "hard_st", False),
        ("implicit_hard_detached", "hard_detached", False),
        ("explicit_hard_detached", "hard_detached", True),
    )
    result: dict[str, dict[str, Any]] = {}
    canonical_packet = _semantic_packets()[1]
    for index, (name, transport, explicit) in enumerate(cases):
        torch.manual_seed(AUDIT_SEED + 350 + index)
        model = _legacy_semantic_model(transport, explicit_route=explicit).eval()
        model._enable_action_only_prefix_cache = True
        model.reset_action_only_prefix_cache()
        _bind_fixed_packet(
            model,
            canonical_packet,
            torch.zeros(1, 12, SEGMENT_TARGET_DIM),
        )
        call_kwargs = dict(
            action_only=True,
            no_video_tokens=True,
            ref_latents=inputs["ref_latents"],
            noisy_latents=None,
            timestep=torch.zeros(1, 6),
            encoder_hidden_states=inputs["encoder_hidden_states"],
            state=inputs["state"],
            action=inputs["action"],
            embodiment_id=inputs["embodiment_id"],
            return_dict=True,
        )
        with torch.no_grad():
            first = model(**call_kwargs)["action_pred"]
            second = model(**call_kwargs)["action_pred"]
        cache = model._action_only_prefix_cache
        model._validate_contact_action_only_cache(cache)
        if "contact_semantic" not in cache or "contact_packet_receipt" not in cache:
            raise AssertionError(f"{name} did not cache a canonical semantic packet")
        if not torch.equal(cache["contact_semantic"], canonical_packet):
            raise AssertionError(f"{name} cache changed canonical packet bytes")
        invalid = cache["contact_semantic"].clone()
        invalid[0, 0, SEGMENT_FIELD_SLICES["start_tau"]] = 0.25
        probe = dict(cache)
        probe["contact_semantic"] = invalid
        probe["contact_packet_receipt"] = contact_packet_receipt(
            route_receipt=model.contact_route_metadata,
            packet_bytes=model._canonical_packet_bytes(invalid),
            packet_dtype=str(invalid.dtype),
            packet_shape=tuple(invalid.shape),
        )
        try:
            model._validate_contact_action_only_cache(probe)
        except RuntimeError:
            continuous_0_25_rejected = True
        else:
            raise AssertionError(f"{name} accepted off-lattice continuous value 0.25")
        result[name] = {
            "resolved_route": model.contact_causal_route.value,
            "route_explicit": model.contact_causal_route_explicit,
            "transport": model.contact_semantic_transport_mode,
            "receipt_declares_finite_semantic_packet": bool(
                model.contact_route_metadata[
                    "action_visible_packet_finite_semantic_grammar"
                ]
            ),
            "canonical_packet_cached_and_accepted": True,
            "cached_reuse_action_bitwise_equal": bool(torch.equal(first, second)),
            "continuous_0_25_rejected": continuous_0_25_rejected,
        }
    return result


def _soft_cache_continuous_audit(
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove that the continuous soft semantic cache remains continuous."""

    torch.manual_seed(AUDIT_SEED + 359)
    model = _legacy_semantic_model("soft", explicit_route=False).eval()
    model._enable_action_only_prefix_cache = True
    model.reset_action_only_prefix_cache()
    continuous_packet = torch.full((1, 12, SEGMENT_TARGET_DIM), 0.25)
    _bind_fixed_packet(
        model,
        continuous_packet,
        torch.zeros(1, 12, SEGMENT_TARGET_DIM),
    )
    call_kwargs = dict(
        action_only=True,
        no_video_tokens=True,
        ref_latents=inputs["ref_latents"],
        noisy_latents=None,
        timestep=torch.zeros(1, 6),
        encoder_hidden_states=inputs["encoder_hidden_states"],
        state=inputs["state"],
        action=inputs["action"],
        embodiment_id=inputs["embodiment_id"],
        return_dict=True,
    )
    with torch.no_grad():
        first = model(**call_kwargs)["action_pred"]
        second = model(**call_kwargs)["action_pred"]
    cache = model._action_only_prefix_cache
    model._validate_contact_action_only_cache(cache)
    if not torch.equal(cache["contact_semantic"], continuous_packet):
        raise AssertionError("soft cache changed its continuous semantic packet")
    return {
        "resolved_route": model.contact_causal_route.value,
        "route_explicit": model.contact_causal_route_explicit,
        "receipt_declares_finite_semantic_packet": bool(
            model.contact_route_metadata[
                "action_visible_packet_finite_semantic_grammar"
            ]
        ),
        "continuous_0_25_cached_and_accepted": True,
        "cached_reuse_action_bitwise_equal": bool(torch.equal(first, second)),
    }


def _non_b16_hard_cache_fail_closed_audit(
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model = _legacy_semantic_model(
        "hard_st", explicit_route=False, time_bins=8
    ).eval()
    model._enable_action_only_prefix_cache = True
    model.reset_action_only_prefix_cache()
    _bind_fixed_packet(
        model,
        _semantic_packets()[1],
        torch.zeros(1, 12, SEGMENT_TARGET_DIM),
    )
    try:
        with torch.no_grad():
            model(
                action_only=True,
                no_video_tokens=True,
                ref_latents=inputs["ref_latents"],
                noisy_latents=None,
                timestep=torch.zeros(1, 6),
                encoder_hidden_states=inputs["encoder_hidden_states"],
                state=inputs["state"],
                action=inputs["action"],
                embodiment_id=inputs["embodiment_id"],
                return_dict=True,
            )
    except RuntimeError:
        rejected = True
    else:
        raise AssertionError("implicit finite hard B8 cache bypassed the B16 grammar")
    return {
        "resolved_route": model.contact_causal_route.value,
        "route_explicit": model.contact_causal_route_explicit,
        "time_bins": model.contact_time_bins,
        "receipt_declares_finite_semantic_packet": bool(
            model.contact_route_metadata[
                "action_visible_packet_finite_semantic_grammar"
            ]
        ),
        "cache_rejected": rejected,
        "cache_only_negative_does_not_invoke_ordinary_forward": True,
    }


def _matrix_checkpoint_audit() -> dict[str, bool]:
    model = _tiny_model(ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST)
    state = model.state_dict()
    if "equal_bit_mixing_matrix" in state:
        raise AssertionError("equal-bit matrix leaked into state_dict")
    injected = dict(state)
    injected["equal_bit_mixing_matrix"] = torch.zeros(16, 22)
    try:
        model.load_state_dict(injected, strict=True)
    except RuntimeError:
        strict_key_rejected = True
    else:
        raise AssertionError("strict state load accepted a mutable matrix key")
    with torch.no_grad():
        model.equal_bit_mixing_matrix[0, 0] *= -1
    try:
        model.transport_unstructured_contact_logits(torch.zeros(1, 12, 22))
    except RuntimeError:
        mutation_rejected = True
    else:
        raise AssertionError("runtime matrix mutation did not invalidate transport")
    with torch.no_grad():
        model.equal_bit_mixing_matrix.copy_(
            torch.from_numpy(validated_equal_bit_mixing_matrix())
        )
    model.transport_unstructured_contact_logits(torch.zeros(1, 12, 22))
    model.contact_route_metadata = dict(model.contact_route_metadata)
    model.contact_route_metadata["equal_bit_mixing_matrix_sha256"] = "0" * 64
    try:
        model.transport_unstructured_contact_logits(torch.zeros(1, 12, 22))
    except RuntimeError:
        route_metadata_mutation_rejected = True
    else:
        raise AssertionError("runtime route receipt mutation did not invalidate transport")
    model.contact_route_metadata = contact_route_metadata(
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
        num_queries=12,
        time_bins=16,
    )
    model.transport_unstructured_contact_logits(torch.zeros(1, 12, 22))
    return {
        "nonpersistent_state_dict": True,
        "strict_mutable_matrix_key_rejected": strict_key_rejected,
        "runtime_matrix_mutation_rejected": mutation_rejected,
        "source_reset_validated": True,
        "runtime_route_metadata_mutation_rejected": route_metadata_mutation_rejected,
        "source_route_metadata_reset_validated": True,
    }


def _trainer_fail_closed_audit() -> dict[str, bool | int]:
    route = ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST.value
    base = {
        "contact_causal_route": route,
        "contact_loss_weight": 0.25,
        "contact_schema_version": "contact_segment_v1_1",
        "contact_condition_mode": "predicted",
        "require_contact_targets": True,
        "require_positive_contact_supervision": True,
        "transformer": {
            "contact_causal_route": route,
            "contact_semantic_bottleneck": True,
            "contact_semantic_transport_mode": "hard_st",
        },
    }
    if validate_spec_only_trainer_route_config(base) is not ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
        raise AssertionError("valid spec trainer route did not resolve")
    checks: dict[str, bool | int] = {"matched_route_accepted": True}

    cases = {}
    case = {**base, "contact_causal_route": ContactCausalRoute.HARD_ST_SEMANTIC.value}
    cases["route_mismatch_rejected"] = case
    case = dict(base)
    case.pop("contact_causal_route")
    cases["missing_top_route_rejected"] = case
    case = {**base, "transformer": dict(base["transformer"])}
    case["transformer"].pop("contact_causal_route")
    cases["missing_transformer_route_rejected"] = case
    for label, weight in (("zero_weight_rejected", 0.0), ("nan_weight_rejected", float("nan"))):
        cases[label] = {**base, "contact_loss_weight": weight}
    for label, config in cases.items():
        try:
            validate_spec_only_trainer_route_config(config)
        except ValueError:
            checks[label] = True
        else:
            raise AssertionError(f"trainer contract accepted negative case {label}")
    target = torch.zeros(1, 12, SEGMENT_TARGET_DIM)
    try:
        validate_positive_contact_supervision(target, torch.zeros_like(target))
    except ValueError:
        checks["all_zero_mask_rejected"] = True
    else:
        raise AssertionError("all-zero contact mask was audited as positive")
    partial_category = torch.zeros_like(target)
    partial_category[..., SEGMENT_FIELD_SLICES["side"].start] = 1
    if effective_contact_supervision_count(partial_category) != 0:
        raise AssertionError("partial one-hot group was counted as eligible")
    try:
        validate_positive_contact_supervision(target, partial_category)
    except ValueError:
        checks["partial_categorical_group_rejected"] = True
    else:
        raise AssertionError("partial one-hot group was admitted as positive supervision")

    presence_mask = torch.zeros_like(target)
    presence_mask[..., SEGMENT_FIELD_SLICES["presence"]] = 1
    checks["presence_positive_count"] = validate_positive_contact_supervision(
        target, presence_mask
    )
    complete_group = torch.zeros_like(target)
    complete_group[..., SEGMENT_FIELD_SLICES["side"]] = 1
    target_with_side = target.clone()
    target_with_side[..., SEGMENT_FIELD_SLICES["side"].start] = 1
    checks["complete_categorical_group_positive_count"] = (
        validate_positive_contact_supervision(target_with_side, complete_group)
    )
    tau_mask = torch.zeros_like(target)
    tau_mask[..., SEGMENT_FIELD_SLICES["start_tau"]] = 1
    censor_mask = torch.zeros_like(target)
    censor_mask[..., SEGMENT_FIELD_SLICES["left_censored"]] = 1
    checks["tau_single_item_positive_count"] = effective_contact_supervision_count(
        tau_mask
    )
    checks["censor_single_item_positive_count"] = effective_contact_supervision_count(
        censor_mask
    )
    try:
        validate_positive_contact_supervision(
            target,
            presence_mask,
            confidence=torch.zeros_like(target),
        )
    except ValueError:
        checks["zero_confidence_effective_supervision_rejected"] = True
    else:
        raise AssertionError("zero-confidence supervision was admitted")
    full_mask = torch.ones_like(target)
    full_weights = grouped_contact_supervision_weights(full_mask)
    checks["full_mask_schema_count"] = effective_contact_supervision_count(full_mask)
    checks["full_mask_weight_field_count"] = len(full_weights)
    btq_presence = torch.zeros(2, 3, 12, SEGMENT_TARGET_DIM)
    btq_presence[..., SEGMENT_FIELD_SLICES["presence"]] = 1
    checks["btq_presence_positive_count"] = effective_contact_supervision_count(
        btq_presence
    )
    checks["supervision_rule_version_present"] = bool(
        CONTACT_GROUPED_SUPERVISION_RULE_VERSION
    )
    return checks


def _grouped_loss_supervision_ast_audit() -> dict[str, Any]:
    """Ensure grouped loss consumes the same shared schema-weight builder."""

    trainer_path = REPOSITORY_ROOT / "world_action_model/trainer/wa_casual_trainer_mot.py"
    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    function = functions.get("grouped_contact_segment_loss")
    if function is None:
        raise AssertionError("grouped contact loss function is missing")
    shared_builder_calls = sum(
        1
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "grouped_contact_supervision_weights"
    )
    shared_weight_reads = sum(
        1
        for node in ast.walk(function)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "supervision_weights"
    )
    if shared_builder_calls != 1 or shared_weight_reads != 3:
        raise AssertionError("grouped loss no longer uses one shared field-weight builder")
    selected = [
        functions["apply_contact_query_assignment"],
        functions["grouped_contact_segment_loss"],
    ]
    namespace = {
        "F": F,
        "torch": torch,
        "SEGMENT_BINARY_FIELDS": SEGMENT_BINARY_FIELDS,
        "SEGMENT_CATEGORICAL_FIELDS": SEGMENT_CATEGORICAL_FIELDS,
        "SEGMENT_CONTINUOUS_FIELDS": SEGMENT_CONTINUOUS_FIELDS,
        "SEGMENT_FIELD_SLICES": SEGMENT_FIELD_SLICES,
        "SEGMENT_TARGET_DIM": SEGMENT_TARGET_DIM,
        "grouped_contact_supervision_weights": grouped_contact_supervision_weights,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(trainer_path),
            "exec",
        ),
        namespace,
    )
    loss_fn = namespace["grouped_contact_segment_loss"]
    prediction = torch.zeros(1, 2, SEGMENT_TARGET_DIM)
    target = torch.zeros_like(prediction)
    target[..., SEGMENT_FIELD_SLICES["side"].start] = 1
    partial = torch.zeros_like(prediction)
    partial[..., SEGMENT_FIELD_SLICES["side"].start] = 1
    presence = torch.zeros_like(prediction)
    presence[..., SEGMENT_FIELD_SLICES["presence"]] = 1
    complete_group = torch.zeros_like(prediction)
    complete_group[..., SEGMENT_FIELD_SLICES["side"]] = 1
    partial_loss = float(loss_fn(prediction, target, mask=partial))
    presence_loss = float(loss_fn(prediction, target, mask=presence))
    complete_group_loss = float(loss_fn(prediction, target, mask=complete_group))
    if partial_loss != 0.0 or presence_loss <= 0.0 or complete_group_loss <= 0.0:
        raise AssertionError("grouped loss numerical eligibility differs from shared count")
    return {
        "rule_version": CONTACT_GROUPED_SUPERVISION_RULE_VERSION,
        "shared_builder_call_count": shared_builder_calls,
        "shared_weight_read_count": shared_weight_reads,
        "partial_category_loss": partial_loss,
        "presence_loss_positive": presence_loss > 0.0,
        "complete_category_group_loss_positive": complete_group_loss > 0.0,
        "passed": True,
    }


def _production_non_admission_audit() -> dict[str, Any]:
    expected = ("soft_semantic", "hard_st", "hard_detached")
    exact_runner_set = tuple(IMPLEMENTED_RUNNER_VARIANTS) == expected
    protected_paths = (
        "world_action_model/matched_control_hashes.py",
        "world_action_model/matched_control_config.py",
        "world_action_model/contact_asym25_preflight.py",
        "world_action_model/contact_asym25_checkpoint_acceptance.py",
        "scripts/preflight_contact_asym25_finetune.py",
        "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py",
        "configs/contact_asym25_training_gate.template.json",
        "configs/contact_asym25_gate_reviewer_acceptance.template.json",
    )
    result = subprocess.run(
        ["git", "diff", "--quiet", LEGACY_GOLDEN_BASELINE_COMMIT, "--", *protected_paths],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    gate_zero_diff = result.returncode == 0
    if not exact_runner_set or not gate_zero_diff:
        raise AssertionError("spec routes changed production runner admission/gate files")
    return {
        "production_runner_exact_set": list(expected),
        "production_runner_exact_set_unchanged": exact_runner_set,
        "production_gate_protected_paths": list(protected_paths),
        "production_gate_zero_diff_from_baseline": gate_zero_diff,
        "report_is_not_admission": True,
        "production_admitted": False,
    }


def _mask_audit() -> dict[str, Any]:
    direct = build_mot_attention_mask(
        2, 3, 1, 1, 1, 2, torch.device("cpu"), torch.float32,
        num_contact_tokens=0,
        action_attends_ref=True,
        state_attends_ref=True,
        contact_attends_ref=True,
    )[0, 0]
    strict = build_mot_attention_mask(
        2, 3, 1, 1, 1, 2, torch.device("cpu"), torch.float32,
        num_contact_tokens=12,
        action_attends_ref=False,
        state_attends_ref=False,
        contact_attends_ref=False,
    )[0, 0]
    direct_action_row = 2
    direct_ref_col = 2 + 3
    strict_action_row = 2 + 12
    strict_ref_col = 2 + 12 + 3
    strict_future_col = strict_ref_col + 1
    checks = {
        "direct_action_reads_reference": bool(direct[direct_action_row, direct_ref_col] == 0),
        "strict_action_rejects_reference": bool(torch.isneginf(strict[strict_action_row, strict_ref_col])),
        "strict_action_rejects_future": bool(torch.isneginf(strict[strict_action_row, strict_future_col])),
        "strict_action_reads_packet": bool(strict[strict_action_row, 2] == 0),
        "strict_state_rejects_reference": bool(torch.isneginf(strict[0, strict_ref_col])),
        "strict_packet_rejects_reference": bool(torch.isneginf(strict[2, strict_ref_col])),
    }
    if not all(checks.values()):
        raise AssertionError(f"attention route audit failed: {checks}")
    return checks


def _contract_fail_closed_audit() -> dict[str, bool]:
    checks = {}
    try:
        resolve_contact_causal_route(
            "boolean_composed_unknown_route",
            contact_semantic_bottleneck=True,
            contact_semantic_transport_mode="hard_st",
        )
    except ValueError:
        checks["unknown_route_rejected"] = True
    else:
        raise AssertionError("unknown route enum was accepted")
    invalid_cases = {
        "aux_contradictory_bottleneck_rejected": dict(
            route=ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
            num_contact_tokens=12,
            contact_bottleneck=True,
            contact_semantic_bottleneck=False,
            contact_time_bins=16,
        ),
        "equal_wrong_q_rejected": dict(
            route=ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
            num_contact_tokens=11,
            contact_bottleneck=True,
            contact_semantic_bottleneck=True,
            contact_time_bins=16,
        ),
        "equal_wrong_bins_rejected": dict(
            route=ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
            num_contact_tokens=12,
            contact_bottleneck=True,
            contact_semantic_bottleneck=True,
            contact_time_bins=8,
        ),
    }
    for name, case in invalid_cases.items():
        try:
            validate_contact_route_model_config(
                contact_target_dim=22,
                contact_condition_dim=22,
                contact_schema_version="contact_segment_v1_1",
                **case,
            )
        except ValueError:
            checks[name] = True
        else:
            raise AssertionError(f"fail-closed case was accepted: {name}")
    return checks


def _codec_audit(equal_model) -> dict[str, Any]:
    metadata = dict(unstructured_capacity_metadata())
    semantic = dict(
        contact_transport_capacity_metadata(mode="hard_st", time_bins=16, num_queries=12)
    )
    matrix = validated_equal_bit_mixing_matrix()
    if np.linalg.matrix_rank(matrix.astype(np.float64)) != EQUAL_BIT_NUM_BITS:
        raise AssertionError("fixed equal-bit mixing matrix is not full row rank")
    matrix_sha = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    if matrix_sha != EQUAL_BIT_MIXING_MATRIX_SHA256:
        raise AssertionError("fixed equal-bit mixing matrix digest differs")

    all_raw_ids = np.arange(2**EQUAL_BIT_NUM_BITS, dtype=np.int64)
    canonical = canonicalize_unstructured_ids(all_raw_ids)
    codes = unstructured_ids_to_code(all_raw_ids)
    row_bytes = codes.view(np.dtype((np.void, codes.dtype.itemsize * codes.shape[-1])))
    unique_forward_states = int(np.unique(row_bytes).size)
    legal_ids = np.arange(EQUAL_BIT_VALID_STATES, dtype=np.uint16)
    legal_roundtrip = unstructured_code_to_ids(unstructured_ids_to_code(legal_ids))
    signed_zero_code = unstructured_ids_to_code([0])
    signed_zero_code[0, -1] = -0.0
    try:
        unstructured_code_to_ids(signed_zero_code)
    except ValueError:
        pure_decoder_signed_zero_rejected = True
    else:
        raise AssertionError("pure equal-bit decoder accepted signed negative zero")
    hard_semantic_states = _all_hard_semantic_states_b16()
    validate_hard_contact_segment_code(hard_semantic_states, time_bins=16)
    hard_row_bytes = hard_semantic_states.view(
        np.dtype((np.void, hard_semantic_states.dtype.itemsize * SEGMENT_TARGET_DIM))
    )
    hard_unique_states = int(np.unique(hard_row_bytes).size)
    illegal = codes[EQUAL_BIT_VALID_STATES:]
    zero_code = unstructured_ids_to_code([0])[0]
    raw_bit_matrix = (
        (all_raw_ids[:, None] >> np.arange(EQUAL_BIT_NUM_BITS)) & 1
    ).astype(np.float32)
    target_signs = torch.from_numpy(raw_bit_matrix * 2.0 - 1.0)
    torch_matrix = torch.tensor(EQUAL_BIT_MIXING_MATRIX, dtype=torch.float32)
    realizing_logits = target_signs @ torch.linalg.pinv(torch_matrix.transpose(0, 1))
    model_codes = []
    with torch.no_grad():
        for chunk in realizing_logits.split(4096):
            model_codes.append(
                equal_model.transport_unstructured_contact_logits(chunk).numpy()
            )
    model_codes_array = np.concatenate(model_codes, axis=0)
    checks = {
        "matrix_full_row_rank": int(np.linalg.matrix_rank(matrix.astype(np.float64))),
        "matrix_sha256": matrix_sha,
        "unique_forward_byte_states_per_query": unique_forward_states,
        "legal_id_roundtrip": bool(np.array_equal(legal_roundtrip, legal_ids)),
        "all_invalid_ids_canonicalize_to_zero": bool(
            (canonical[EQUAL_BIT_VALID_STATES:] == 0).all()
            and (illegal == zero_code).all()
        ),
        "all_zero_values_positive": bool(not np.signbit(codes[codes == 0]).any()),
        "padding_six_dimensions_zero": bool((codes[..., EQUAL_BIT_NUM_BITS:] == 0).all()),
        "pure_decoder_signed_zero_rejected": pure_decoder_signed_zero_rejected,
        "hard_semantic_grammar_unique_states_per_query": hard_unique_states,
        "hard_semantic_grammar_all_zero_values_positive": bool(
            not np.signbit(hard_semantic_states[hard_semantic_states == 0]).any()
        ),
        "exhaustive_model_codec_matches_id_codec": bool(
            np.array_equal(model_codes_array, codes)
        ),
        "exhaustive_model_codec_byte_exact": bool(
            model_codes_array.tobytes(order="C") == codes.tobytes(order="C")
        ),
        "capacity_bits_match_semantic_b16": abs(
            metadata["total_bits"] - semantic["strict_valid_total_bits"]
        ),
    }
    if unique_forward_states != EQUAL_BIT_VALID_STATES:
        raise AssertionError("equal-bit forward alphabet is not exactly 48,673 states")
    if hard_unique_states != EQUAL_BIT_VALID_STATES:
        raise AssertionError("hard semantic grammar is not exactly 48,673 states")
    if not checks["exhaustive_model_codec_matches_id_codec"] or not checks[
        "exhaustive_model_codec_byte_exact"
    ]:
        raise AssertionError("model equal-bit transport differs from exhaustive id codec")
    if not checks["legal_id_roundtrip"] or not checks["all_invalid_ids_canonicalize_to_zero"]:
        raise AssertionError("equal-bit id canonicalization/roundtrip failed")
    if checks["capacity_bits_match_semantic_b16"] > 1e-9:
        raise AssertionError("equal-bit and semantic B16 capacities differ")
    if not checks["all_zero_values_positive"] or not checks["padding_six_dimensions_zero"]:
        raise AssertionError("equal-bit zero canonicalization failed")

    logits = torch.randn(2, 12, SEGMENT_TARGET_DIM, requires_grad=True)
    packet = equal_model.transport_unstructured_contact_logits(logits)
    packet.sum().backward()
    gradient = logits.grad
    checks.update(
        hard_st_gradient_finite=bool(gradient is not None and torch.isfinite(gradient).all()),
        hard_st_gradient_nonzero=bool(gradient is not None and (gradient != 0).any()),
        forward_values_binary=bool(torch.logical_or(packet == 0, packet == 1).all()),
        forward_padding_positive_zero=bool(
            not np.signbit(
                packet.detach().numpy()[packet.detach().numpy() == 0]
            ).any()
        ),
    )
    for bad in (float("nan"), float("inf"), float("-inf")):
        invalid = torch.zeros(1, 12, SEGMENT_TARGET_DIM)
        invalid[0, 0, 0] = bad
        try:
            equal_model.transport_unstructured_contact_logits(invalid)
        except ValueError:
            continue
        raise AssertionError("equal-bit transport accepted NaN/Inf")
    checks["nan_inf_rejected"] = True
    if not all(
        checks[name]
        for name in (
            "hard_st_gradient_finite",
            "hard_st_gradient_nonzero",
            "forward_values_binary",
            "forward_padding_positive_zero",
        )
    ):
        raise AssertionError("equal-bit hard-ST audit failed")
    return {"metadata": metadata, "checks": checks}


def _legacy_compatibility_audit(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    result = {}
    semantic_probe = torch.sigmoid(torch.linspace(-4, 4, 12 * 22)).reshape(1, 12, 22)
    for index, transport in enumerate(("soft", "hard_st", "hard_detached")):
        torch.manual_seed(AUDIT_SEED + index)
        legacy = _legacy_semantic_model(transport, explicit_route=False).eval()
        torch.manual_seed(AUDIT_SEED + index)
        explicit = _legacy_semantic_model(transport, explicit_route=True).eval()
        legacy_keys = tuple(legacy.state_dict())
        explicit_keys = tuple(explicit.state_dict())
        if legacy_keys != explicit_keys:
            raise AssertionError(f"{transport} state_dict keys changed")
        legacy_packet = legacy.transport_contact_semantic(semantic_probe)
        explicit_packet = explicit.transport_contact_semantic(semantic_probe)
        if not torch.equal(legacy_packet, explicit_packet):
            raise AssertionError(f"{transport} transport bytes changed")
        with torch.no_grad():
            legacy_output = _action_only(legacy, inputs)
            explicit_output = _action_only(explicit, inputs)
        if not torch.equal(legacy_output["action_pred"], explicit_output["action_pred"]):
            raise AssertionError(f"{transport} action output changed")
        if not torch.equal(legacy_output["contact_pred"], explicit_output["contact_pred"]):
            raise AssertionError(f"{transport} contact output changed")
        observed = {
            "packet": _tensor_sha256(legacy_packet),
            "action": _tensor_sha256(legacy_output["action_pred"]),
            "contact": _tensor_sha256(legacy_output["contact_pred"]),
        }
        key_sha = hashlib.sha256("\n".join(legacy_keys).encode("utf-8")).hexdigest()
        if observed != LEGACY_GOLDEN[transport] or key_sha != LEGACY_STATE_DICT_KEYS_SHA256:
            raise AssertionError(
                f"{transport} differs from baseline {LEGACY_GOLDEN_BASELINE_COMMIT}"
            )
        result[transport] = {
            "golden_baseline_commit": LEGACY_GOLDEN_BASELINE_COMMIT,
            "state_dict_key_count": len(legacy_keys),
            "state_dict_keys_sha256": key_sha,
            "packet_sha256": observed["packet"],
            "action_sha256": observed["action"],
            "contact_sha256": observed["contact"],
            "bitwise_equal": True,
        }
        invalid = torch.zeros(1, 12, SEGMENT_TARGET_DIM)
        invalid[0, 0, 0] = float("nan")
        try:
            legacy.transport_contact_semantic(invalid)
        except ValueError:
            result[transport]["nan_rejected"] = True
        else:
            raise AssertionError(f"{transport} semantic transport accepted NaN")
    if result["hard_st"]["packet_sha256"] != result["hard_detached"]["packet_sha256"]:
        raise AssertionError("legacy hard-ST and detached forward packet bytes differ")
    result["hard_forward_packets_bitwise_equal"] = True
    hard_st_model = _legacy_semantic_model("hard_st", explicit_route=False)
    hard_st_probe = semantic_probe.clone().requires_grad_(True)
    hard_st_model.transport_contact_semantic(hard_st_probe).sum().backward()
    if hard_st_probe.grad is None or not torch.isfinite(hard_st_probe.grad).all():
        raise AssertionError("legacy hard-ST surrogate gradient is not finite")
    detached_model = _legacy_semantic_model("hard_detached", explicit_route=False)
    detached_packet = detached_model.transport_contact_semantic(
        semantic_probe.clone().requires_grad_(True)
    )
    if detached_packet.requires_grad:
        raise AssertionError("legacy hard-detached packet retained an action gradient route")
    result["hard_st_gradient_finite"] = True
    result["hard_detached_gradient_blocked"] = True
    return result


def _ast_route_audit() -> dict[str, Any]:
    source = inspect.getsource(CasualWorldActionTransformer_MoT.encode_action_tokens)
    tree = ast.parse(source.lstrip())
    function = tree.body[0]
    parameter_names = {argument.arg for argument in function.args.args}
    identifier_names = {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }
    forbidden = {
        "planner_hidden",
        "contact_logits",
        "ref_visual_hidden_states",
        "ref_latents",
        "future_latents",
    }
    leaked_parameters = sorted(parameter_names.intersection(forbidden))
    leaked_identifiers = sorted(identifier_names.intersection(forbidden))
    if leaked_parameters or leaked_identifiers:
        raise AssertionError("raw planner/reference identifier enters encode_action_tokens")
    passed = not leaked_parameters and not leaked_identifiers
    return {
        "function": "CasualWorldActionTransformer_MoT.encode_action_tokens",
        "parameter_names": sorted(parameter_names),
        "forbidden_parameter_names": sorted(forbidden),
        "forbidden_identifiers_present": [],
        "passed": passed,
    }


def _direct_route_audit(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    torch.manual_seed(AUDIT_SEED + 100)
    model = _tiny_model(ContactCausalRoute.AUX_ONLY_DIRECT_Q12).eval()
    with torch.no_grad():
        before = _action_only(model, inputs)
        for name, parameter in model.named_parameters():
            if name.startswith("contact_"):
                parameter.add_(torch.randn_like(parameter) * 3.0)
        after = _action_only(model, inputs)
    action_delta = float((before["action_pred"] - after["action_pred"]).abs().max())
    contact_delta = float((before["contact_pred"] - after["contact_pred"]).abs().max())
    if action_delta != 0.0 or contact_delta == 0.0:
        raise AssertionError("direct action is not strictly planner-invariant")

    model.zero_grad(set_to_none=True)
    grad_output = _action_only(model, inputs)
    grad_output["action_pred"].float().square().mean().backward()
    planner_grad_nonzero = []
    for name, parameter in model.named_parameters():
        if name.startswith("contact_") and parameter.grad is not None and bool((parameter.grad != 0).any()):
            planner_grad_nonzero.append(name)
    if planner_grad_nonzero:
        raise AssertionError("direct action loss reaches planner parameters")

    model.eval()
    model._enable_action_only_prefix_cache = True
    model.reset_action_only_prefix_cache()
    with torch.no_grad():
        uncached = _action_only(model, inputs)["action_pred"]
        cached = model(
            action_only=True,
            no_video_tokens=True,
            ref_latents=inputs["ref_latents"],
            noisy_latents=None,
            timestep=torch.zeros(1, 6),
            encoder_hidden_states=inputs["encoder_hidden_states"],
            state=inputs["state"],
            action=inputs["action"],
            embodiment_id=inputs["embodiment_id"],
            return_dict=True,
        )["action_pred"]
    cache_delta = float((uncached - cached).abs().max())
    cache = model._action_only_prefix_cache
    if cache_delta > CLAMP_TOLERANCE:
        raise AssertionError("direct cached/uncached action differs")
    if {"contact_semantic", "contact_logits", "contact_code"}.intersection(cache):
        raise AssertionError("direct cache leaked planner values")
    tampered_cache = dict(cache)
    tampered_cache["contact_logits"] = torch.zeros(1)
    try:
        model._validate_contact_action_only_cache(tampered_cache)
    except RuntimeError:
        cache_tamper_rejected = True
    else:
        raise AssertionError("direct route accepted planner data in its cache")
    try:
        with torch.no_grad():
            model(
                action_only=True,
                ref_latents=inputs["ref_latents"],
                noisy_latents=None,
                timestep=torch.zeros(1, 6),
                encoder_hidden_states=inputs["encoder_hidden_states"],
                state=inputs["state"],
                action=inputs["action"],
                embodiment_id=inputs["embodiment_id"],
                contact_condition=torch.zeros(1, 12, 22),
                contact_condition_mask=torch.ones(1, 12, 22),
            )
    except ValueError:
        oracle_rejected = True
    else:
        raise AssertionError("direct route accepted oracle/condition input")
    return {
        "planner_perturb_action_max_abs": action_delta,
        "planner_perturb_contact_aux_max_abs": contact_delta,
        "action_loss_planner_nonzero_grad_parameters": planner_grad_nonzero,
        "cached_uncached_action_max_abs": cache_delta,
        "cache_keys": sorted(cache),
        "cache_tamper_rejected": cache_tamper_rejected,
        "oracle_rejected": oracle_rejected,
        "action_sha256": _tensor_sha256(before["action_pred"]),
    }


def _strict_route_audit(
    route: ContactCausalRoute,
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    torch.manual_seed(AUDIT_SEED + (200 if route is ContactCausalRoute.HARD_ST_SEMANTIC else 300))
    model = _tiny_model(route).train()
    gradient_output = _action_only(model, inputs)
    gradient_output["action_pred"].float().square().mean().backward()
    planner_action_gradient_names = []
    for name, parameter in model.named_parameters():
        if name.startswith("contact_") and parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"{route.value} action gradient is non-finite: {name}")
            if bool((parameter.grad != 0).any()):
                planner_action_gradient_names.append(name)
    if not planner_action_gradient_names:
        raise AssertionError(f"{route.value} action loss does not reach its planner")
    model.zero_grad(set_to_none=True)
    model.eval()
    packet_a, packet_b = (
        _semantic_packets()
        if route is ContactCausalRoute.HARD_ST_SEMANTIC
        else _equal_packets()
    )
    logits = torch.zeros(1, 12, SEGMENT_TARGET_DIM)
    _bind_fixed_packet(model, packet_a, logits)
    changed_ref = dict(inputs)
    changed_ref["ref_latents"] = inputs["ref_latents"] + 100.0
    changed_future = dict(inputs)
    changed_future["future_latents"] = inputs["future_latents"] - 100.0
    with torch.no_grad():
        action_a = _action_only(model, inputs)["action_pred"]
        action_changed_ref = _action_only(model, changed_ref)["action_pred"]
        future_a = _action_with_future(model, inputs)["action_pred"]
        future_changed = _action_with_future(model, changed_future)["action_pred"]
        _bind_fixed_packet(model, packet_b, logits)
        action_b = _action_only(model, inputs)["action_pred"]
    ref_delta = float((action_a - action_changed_ref).abs().max())
    future_delta = float((future_a - future_changed).abs().max())
    intervention_delta = float((action_a - action_b).abs().max())
    if ref_delta > CLAMP_TOLERANCE or future_delta > CLAMP_TOLERANCE:
        raise AssertionError(f"{route.value} has a strict reference/future bypass")
    if intervention_delta == 0.0:
        raise AssertionError(f"{route.value} does not expose its packet to action")

    model._enable_action_only_prefix_cache = True
    model.reset_action_only_prefix_cache()
    _bind_fixed_packet(model, packet_a, logits)
    with torch.no_grad():
        uncached = _action_only(model, inputs)["action_pred"]
        cached = model(
            action_only=True,
            no_video_tokens=True,
            ref_latents=inputs["ref_latents"],
            noisy_latents=None,
            timestep=torch.zeros(1, 6),
            encoder_hidden_states=inputs["encoder_hidden_states"],
            state=inputs["state"],
            action=inputs["action"],
            embodiment_id=inputs["embodiment_id"],
            return_dict=True,
        )["action_pred"]
    cache_delta = float((uncached - cached).abs().max())
    if cache_delta > CLAMP_TOLERANCE:
        raise AssertionError(f"{route.value} cached/uncached action differs")
    cache = model._action_only_prefix_cache
    expected_prefix_tokens = inputs["state"].shape[1] + 12
    cache_sequence_lengths = [int(block["key"].shape[1]) for block in cache["blocks"]]
    if cache_sequence_lengths != [expected_prefix_tokens] * len(cache_sequence_lengths):
        raise AssertionError(f"{route.value} cache retained reference KV")
    if route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST and (
        "contact_code" not in cache
        or {"contact_semantic", "contact_logits"}.intersection(cache)
    ):
        raise AssertionError("equal-bit cache contains a raw planner bypass")
    cache_domain = _cache_domain_audit(model, cache, route)
    tampered_cache = dict(cache)
    tampered_cache["contact_route_receipt_sha256"] = "0" * 64
    try:
        model._validate_contact_action_only_cache(tampered_cache)
    except RuntimeError:
        cache_tamper_rejected = True
    else:
        raise AssertionError(f"{route.value} accepted a mismatched cache receipt")

    try:
        model._forward_train_action_only_no_video(
            ref_latents=inputs["ref_latents"],
            noisy_latents=None,
            timestep=torch.zeros(1, 6),
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_hidden_states_image=torch.zeros(1, 1, 8),
            state=inputs["state"],
            action=inputs["action"],
            embodiment_id=inputs["embodiment_id"],
            return_dict=True,
        )
    except ValueError:
        image_condition_rejected = True
    else:
        raise AssertionError("strict route accepted image cross-attention input")

    if route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
        oracle_model = _tiny_model(route).eval()
        ref_tokens = oracle_model.encode_visual_tokens(inputs["ref_latents"])[0]
        try:
            oracle_model.plan_contact_route(
                inputs["state"],
                ref_tokens,
                inputs["encoder_hidden_states"],
                contact_condition=torch.zeros(1, 12, 22),
                contact_condition_mask=torch.ones(1, 12, 22),
            )
        except ValueError:
            oracle_rejected = True
        else:
            raise AssertionError("equal-bit route accepted oracle action input")
    else:
        oracle_rejected = False
    return {
        "fixed_packet_changed_reference_action_max_abs": ref_delta,
        "fixed_packet_changed_future_action_max_abs": future_delta,
        "packet_intervention_action_max_abs": intervention_delta,
        "action_loss_planner_finite_nonzero_gradient_parameters": len(
            planner_action_gradient_names
        ),
        "packet_a_sha256": _tensor_sha256(packet_a),
        "packet_b_sha256": _tensor_sha256(packet_b),
        "same_noise_action_a_sha256": _tensor_sha256(action_a),
        "same_noise_action_b_sha256": _tensor_sha256(action_b),
        "cached_uncached_action_max_abs": cache_delta,
        "cache_keys": sorted(cache),
        "cache_sequence_lengths": cache_sequence_lengths,
        "cache_tamper_rejected": cache_tamper_rejected,
        "cache_domain_fail_closed": cache_domain,
        "strict_image_condition_rejected": image_condition_rejected,
        "oracle_rejected": oracle_rejected,
        "interpretation": "route_sensitivity_only_not_physical_feasibility",
    }


def _reduce_invariant_results(report: dict[str, Any]) -> dict[str, bool]:
    """Reduce every auditable contract section to named pass/fail invariants."""

    direct = report[ContactCausalRoute.AUX_ONLY_DIRECT_Q12.value]
    semantic = report[ContactCausalRoute.HARD_ST_SEMANTIC.value]
    equal = report[ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST.value]
    codec = report["codec"]["checks"]
    legacy = report["legacy_compatibility"]
    trainer = report["trainer_fail_closed"]
    grouped_loss_ast = report["grouped_loss_supervision_ast"]
    semantic_cache_resolution = report["semantic_cache_resolution"]
    soft_cache_continuous = report["soft_cache_continuous"]
    non_b16_hard_cache = report["non_b16_hard_cache_fail_closed"]
    production = report["non_admission"]
    route_receipts_canonical = all(
        receipt["receipt_sha256"]
        == sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        for receipt in report["route_receipts"].values()
    )
    results = {
        "contract_negative_cases": all(report["contract_fail_closed"].values()),
        "strict_attention_mask": all(report["attention_mask"].values()),
        "equal_codec_exact_capacity": (
            codec["unique_forward_byte_states_per_query"] == EQUAL_BIT_VALID_STATES
            and codec["hard_semantic_grammar_unique_states_per_query"]
            == EQUAL_BIT_VALID_STATES
            and codec["capacity_bits_match_semantic_b16"] <= 1e-9
        ),
        "equal_codec_domain_and_gradient": all(
            bool(codec[name])
            for name in (
                "legal_id_roundtrip",
                "all_invalid_ids_canonicalize_to_zero",
                "all_zero_values_positive",
                "padding_six_dimensions_zero",
                "pure_decoder_signed_zero_rejected",
                "hard_semantic_grammar_all_zero_values_positive",
                "exhaustive_model_codec_matches_id_codec",
                "exhaustive_model_codec_byte_exact",
                "hard_st_gradient_finite",
                "hard_st_gradient_nonzero",
                "forward_values_binary",
                "forward_padding_positive_zero",
                "nan_inf_rejected",
            )
        ),
        "matrix_checkpoint_runtime_bound": all(report["matrix_checkpoint"].values()),
        "legacy_bitwise_and_state_dict": (
            all(legacy[name]["bitwise_equal"] for name in ("soft", "hard_st", "hard_detached"))
            and all(
                legacy[name]["state_dict_keys_sha256"] == LEGACY_STATE_DICT_KEYS_SHA256
                for name in ("soft", "hard_st", "hard_detached")
            )
            and legacy["hard_forward_packets_bitwise_equal"]
            and legacy["hard_st_gradient_finite"]
            and legacy["hard_detached_gradient_blocked"]
        ),
        "no_raw_parameter_route": bool(report["ast_parameter_route"]["passed"]),
        "direct_planner_action_invariant": (
            direct["planner_perturb_action_max_abs"] == 0.0
            and direct["planner_perturb_contact_aux_max_abs"] > 0.0
            and not direct["action_loss_planner_nonzero_grad_parameters"]
            and direct["cached_uncached_action_max_abs"] <= CLAMP_TOLERANCE
            and direct["cache_tamper_rejected"]
            and direct["oracle_rejected"]
        ),
        "hard_semantic_strict_route": (
            semantic["fixed_packet_changed_reference_action_max_abs"] <= CLAMP_TOLERANCE
            and semantic["fixed_packet_changed_future_action_max_abs"] <= CLAMP_TOLERANCE
            and semantic["packet_intervention_action_max_abs"] > 0.0
            and semantic["action_loss_planner_finite_nonzero_gradient_parameters"] > 0
            and semantic["cached_uncached_action_max_abs"] <= CLAMP_TOLERANCE
            and semantic["cache_tamper_rejected"]
            and semantic["strict_image_condition_rejected"]
            and all(semantic["cache_domain_fail_closed"].values())
        ),
        "equal_bit_strict_route": (
            equal["fixed_packet_changed_reference_action_max_abs"] <= CLAMP_TOLERANCE
            and equal["fixed_packet_changed_future_action_max_abs"] <= CLAMP_TOLERANCE
            and equal["packet_intervention_action_max_abs"] > 0.0
            and equal["action_loss_planner_finite_nonzero_gradient_parameters"] > 0
            and equal["cached_uncached_action_max_abs"] <= CLAMP_TOLERANCE
            and equal["cache_tamper_rejected"]
            and equal["strict_image_condition_rejected"]
            and equal["oracle_rejected"]
            and all(equal["cache_domain_fail_closed"].values())
        ),
        "trainer_route_weight_and_mask_fail_closed": (
            all(
                bool(trainer[name])
                for name in (
                    "matched_route_accepted",
                    "route_mismatch_rejected",
                    "missing_top_route_rejected",
                    "missing_transformer_route_rejected",
                    "zero_weight_rejected",
                    "nan_weight_rejected",
                    "all_zero_mask_rejected",
                    "partial_categorical_group_rejected",
                    "zero_confidence_effective_supervision_rejected",
                    "supervision_rule_version_present",
                )
            )
            and trainer["presence_positive_count"] == 12
            and trainer["complete_categorical_group_positive_count"] == 12
            and trainer["tau_single_item_positive_count"] == 12
            and trainer["censor_single_item_positive_count"] == 12
            and trainer["full_mask_schema_count"] == 108
            and trainer["full_mask_weight_field_count"] == 9
            and trainer["btq_presence_positive_count"] == 72
            and grouped_loss_ast["passed"]
            and grouped_loss_ast["shared_builder_call_count"] == 1
            and grouped_loss_ast["shared_weight_read_count"] == 3
            and grouped_loss_ast["partial_category_loss"] == 0.0
            and grouped_loss_ast["presence_loss_positive"]
            and grouped_loss_ast["complete_category_group_loss_positive"]
            and grouped_loss_ast["rule_version"]
            == CONTACT_GROUPED_SUPERVISION_RULE_VERSION
        ),
        "all_resolved_hard_semantic_caches_use_b16_grammar": all(
            case["receipt_declares_finite_semantic_packet"]
            and case["canonical_packet_cached_and_accepted"]
            and case["cached_reuse_action_bitwise_equal"]
            and case["continuous_0_25_rejected"]
            for case in semantic_cache_resolution.values()
        )
        and set(semantic_cache_resolution)
        == {"implicit_hard_st", "implicit_hard_detached", "explicit_hard_detached"}
        and not semantic_cache_resolution["implicit_hard_st"]["route_explicit"]
        and not semantic_cache_resolution["implicit_hard_detached"]["route_explicit"]
        and semantic_cache_resolution["explicit_hard_detached"]["route_explicit"],
        "soft_semantic_cache_remains_continuous": (
            soft_cache_continuous["resolved_route"]
            == ContactCausalRoute.SOFT_SEMANTIC.value
            and not soft_cache_continuous["route_explicit"]
            and not soft_cache_continuous[
                "receipt_declares_finite_semantic_packet"
            ]
            and soft_cache_continuous["continuous_0_25_cached_and_accepted"]
            and soft_cache_continuous["cached_reuse_action_bitwise_equal"]
        ),
        "finite_hard_non_b16_cache_fails_closed": (
            non_b16_hard_cache["resolved_route"]
            == ContactCausalRoute.HARD_ST_SEMANTIC.value
            and not non_b16_hard_cache["route_explicit"]
            and non_b16_hard_cache["time_bins"] == 8
            and non_b16_hard_cache[
                "receipt_declares_finite_semantic_packet"
            ]
            and non_b16_hard_cache["cache_rejected"]
            and non_b16_hard_cache[
                "cache_only_negative_does_not_invoke_ordinary_forward"
            ]
        ),
        "production_exact_set_gate_zero_diff": (
            production["production_runner_exact_set_unchanged"]
            and production["production_gate_zero_diff_from_baseline"]
            and production["report_is_not_admission"]
            and not production["production_admitted"]
        ),
        "route_receipt_binds_actual_equal_codec_matrix": (
            report["route_receipts"][
                ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST.value
            ]["equal_bit_mixing_matrix_sha256"]
            == EQUAL_BIT_MIXING_MATRIX_SHA256
            and report["route_receipts"][
                ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST.value
            ]["capacity"]["mixing_matrix_sha256"]
            == EQUAL_BIT_MIXING_MATRIX_SHA256
        ),
        "config_and_route_receipt_hashes_recomputed": (
            report["config_receipt_sha256"] == sha256_json(report["config_receipt"])
            and route_receipts_canonical
        ),
    }
    return {name: bool(value) for name, value in results.items()}


def run_contact_route_cpu_audit() -> dict[str, Any]:
    """Run all audits on CPU and return a strict-JSON-compatible report."""

    if torch.cuda.is_initialized():
        raise RuntimeError("contact route audit refuses an initialized CUDA runtime")
    torch.set_num_threads(1)
    torch.manual_seed(AUDIT_SEED)
    inputs = _inputs()
    route_receipts = {
        route.value: dict(contact_route_metadata(route, num_queries=12, time_bins=16))
        for route in (
            ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
            ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
            ContactCausalRoute.HARD_ST_SEMANTIC,
        )
    }
    equal_model = _tiny_model(ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST)
    config_receipt = {
        "audit_version": AUDIT_VERSION,
        "seed": AUDIT_SEED,
        "device": "cpu",
        "dtype": "float32",
        "clamp_tolerance": CLAMP_TOLERANCE,
        "routes": sorted(route_receipts),
        "source_file_sha256": {
            relative_path: _file_sha256(relative_path)
            for relative_path in (
                "configs/contact_transport_ablation_matrix.json",
                "configs/giga_world_policy_0_5_sonic_g1_contact_causal_controls.py",
                "world_action_model/contact_transition_schema.py",
                "world_action_model/contact_route_contract.py",
                "world_action_model/contact_supervision_contract.py",
                "world_action_model/contact_route_audit.py",
                "world_action_model/models/transformer_wa_casual_mot.py",
                "world_action_model/trainer/wa_casual_trainer_mot.py",
                "scripts/audit_contact_route_contract.py",
            )
        },
    }
    report = {
        "audit_version": AUDIT_VERSION,
        "scope": "cpu_route_mechanics_only_no_benefit_or_physics_claim",
        "admission_scope": "audit_report_is_evidence_only_never_production_admission",
        "config_receipt": config_receipt,
        "config_receipt_sha256": sha256_json(config_receipt),
        "route_receipts": route_receipts,
        "route_receipt_sha256": {
            route: receipt["receipt_sha256"] for route, receipt in route_receipts.items()
        },
        "contract_fail_closed": _contract_fail_closed_audit(),
        "attention_mask": _mask_audit(),
        "codec": _codec_audit(equal_model),
        "matrix_checkpoint": _matrix_checkpoint_audit(),
        "trainer_fail_closed": _trainer_fail_closed_audit(),
        "grouped_loss_supervision_ast": _grouped_loss_supervision_ast_audit(),
        "semantic_cache_resolution": _semantic_cache_resolution_audit(inputs),
        "soft_cache_continuous": _soft_cache_continuous_audit(inputs),
        "non_b16_hard_cache_fail_closed": (
            _non_b16_hard_cache_fail_closed_audit(inputs)
        ),
        "non_admission": _production_non_admission_audit(),
        "legacy_compatibility": _legacy_compatibility_audit(inputs),
        "ast_parameter_route": _ast_route_audit(),
        "aux_only_direct_q12": _direct_route_audit(inputs),
        "hard_st_semantic": _strict_route_audit(
            ContactCausalRoute.HARD_ST_SEMANTIC, inputs
        ),
        "equal_bit_unstructured_hard_st": _strict_route_audit(
            ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST, inputs
        ),
    }
    report["invariant_results"] = _reduce_invariant_results(report)
    report["passed"] = all(report["invariant_results"].values())
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report

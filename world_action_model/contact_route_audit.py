"""CPU-only causal-route audit and same-noise sensitivity harness.

This module verifies information-flow mechanics.  It does not evaluate task
benefit, physical feasibility, simulator behavior, or production admission.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import types
from typing import Any

import numpy as np
import torch

from world_action_model.contact_route_contract import (
    ContactCausalRoute,
    canonical_json_bytes,
    contact_route_metadata,
    resolve_contact_causal_route,
    sha256_json,
    validate_contact_route_model_config,
)
from world_action_model.contact_transition_schema import (
    EQUAL_BIT_MIXING_MATRIX,
    EQUAL_BIT_MIXING_MATRIX_SHA256,
    EQUAL_BIT_NUM_BITS,
    EQUAL_BIT_VALID_STATES,
    SEGMENT_FIELD_SLICES,
    SEGMENT_TARGET_DIM,
    canonicalize_unstructured_ids,
    contact_transport_capacity_metadata,
    unstructured_capacity_metadata,
    unstructured_code_to_ids,
    unstructured_ids_to_code,
)
from world_action_model.models.transformer_wa_casual_mot import (
    CasualWorldActionTransformer_MoT,
    build_mot_attention_mask,
)


AUDIT_VERSION = "contact_causal_route_cpu_audit_v1"
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


def _legacy_semantic_model(transport: str, explicit_route: bool):
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
        contact_time_bins=16,
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
    matrix = np.asarray(EQUAL_BIT_MIXING_MATRIX, dtype=np.int8)
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
    return {
        "function": "CasualWorldActionTransformer_MoT.encode_action_tokens",
        "parameter_names": sorted(parameter_names),
        "forbidden_parameter_names": sorted(forbidden),
        "forbidden_identifiers_present": [],
        "passed": True,
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
        "strict_image_condition_rejected": image_condition_rejected,
        "oracle_rejected": oracle_rejected,
        "interpretation": "route_sensitivity_only_not_physical_feasibility",
    }


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
        "passed": True,
        "config_receipt": config_receipt,
        "config_receipt_sha256": sha256_json(config_receipt),
        "route_receipts": route_receipts,
        "route_receipt_sha256": {
            route: receipt["receipt_sha256"] for route, receipt in route_receipts.items()
        },
        "contract_fail_closed": _contract_fail_closed_audit(),
        "attention_mask": _mask_audit(),
        "codec": _codec_audit(equal_model),
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
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report

"""CPU-only synthetic acceptance for the causal-main production plumbing."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import torch

from world_action_model.contact_causal_heldout_eval import (
    validate_aggregate_v5,
    validate_result_v3,
    validate_route_evaluator_output,
)
from world_action_model.contact_causal_checkpoint import (
    validate_checkpoint_manifest_v3,
)
from world_action_model.contact_causal_matched_control import (
    build_trace_record,
    compare_causal_records,
    validate_collector_config_v3,
)
from world_action_model.contact_causal_preflight import (
    validate_robot_gate_review_v2,
    validate_robot_gate_v3,
)
from world_action_model.contact_causal_production import (
    CAUSAL_ROUTES,
    CHECKPOINT_STEPS,
    CausalProductionError,
    EVALUATOR_VERSION,
    LEGACY_MECHANISM_ROUTES,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    PRECURSOR_ID,
    SEEDS,
    SUPERVISION_VERSION,
    build_production_route_receipt,
    compare_same_seed_route_contracts,
    ensure_exact_causal_route,
    route_packet_kind,
    sha256_json,
    strict_json_file,
    validate_27_checkpoint_matrix,
    validate_acceptance_v3,
    validate_matrix_manifest,
    validate_training_receipt_v5,
)
from world_action_model.contact_ego_causal_precursor import (
    freeze_precursor_action_path,
    validate_precursor_gate,
    validate_precursor_receipt,
)
from world_action_model.contact_route_contract import ContactCausalRoute
from world_action_model.contact_supervision_contract import (
    canonical_zero_contact_loss,
    validate_positive_contact_supervision,
    validate_zero_safe_contact_supervision,
)
from world_action_model.contact_transition_schema import (
    contact_transport_capacity_metadata,
    unstructured_capacity_metadata,
)
from world_action_model.matched_control_hashes import IMPLEMENTED_RUNNER_VARIANTS
from world_action_model.models.transformer_wa_casual_mot import (
    CasualWorldActionTransformer_MoT,
)
from world_action_model.verified_io import strict_json_loads


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _assert_legacy_protected_bytes(root: Path) -> dict[str, str]:
    protected = (
        "configs/contact_transport_ablation_matrix.json",
        "configs/ego_stage1_training_gate.template.json",
        "configs/contact_asym25_training_gate.template.json",
        "configs/contact_asym25_gate_reviewer_acceptance.template.json",
        "configs/matched_control_hash_collector.template.json",
        "world_action_model/ego_stage1_gate_bootstrap.py",
        "world_action_model/contact_asym25_preflight.py",
        "world_action_model/matched_control_hashes.py",
        "world_action_model/matched_control_config.py",
        "world_action_model/contact_asym25_checkpoint_acceptance.py",
        "world_action_model/contact_asym25_heldout_eval.py",
        "scripts/compare_matched_control_hashes.py",
        "scripts/preflight_contact_asym25_finetune.py",
        "scripts/review_contact_asym25_checkpoint.py",
        "scripts/evaluate_contact_asym25_heldout.py",
    )
    result = {}
    baseline = "e212afec9db5198a89930c7ff896e3e49a0c8249"
    for logical_path in protected:
        head_payload = _git_bytes(root, "show", f"{baseline}:{logical_path}")
        runtime_payload = (root / logical_path).read_bytes()
        if runtime_payload != head_payload:
            raise AssertionError(f"legacy protected bytes changed: {logical_path}")
        result[logical_path] = hashlib.sha256(runtime_payload).hexdigest()
    if tuple(IMPLEMENTED_RUNNER_VARIANTS) != LEGACY_MECHANISM_ROUTES:
        raise AssertionError("legacy collector route registry changed")
    return result


def _strict_json_negatives() -> None:
    invalid = (
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1e999}',
    )
    for payload in invalid:
        try:
            strict_json_loads(payload, name="op42 strict negative")
        except ValueError:
            continue
        raise AssertionError(f"strict JSON accepted invalid payload: {payload!r}")


def _false_template_negatives(root: Path) -> None:
    template_cases = (
        (
            "ego_contact_causal_precursor_gate.v7.template.json",
            "false Ego v7 template",
            validate_precursor_gate,
        ),
        (
            "ego_contact_causal_precursor_receipt.v1.template.json",
            "false Ego precursor receipt template",
            validate_precursor_receipt,
        ),
        (
            "contact_asym25_causal_training_gate.v3.template.json",
            "false Robot v3 template",
            lambda item: validate_robot_gate_v3(
                item, load_external_review=False
            ),
        ),
        (
            "contact_asym25_causal_gate_reviewer_acceptance.v2.template.json",
            "false Robot reviewer template",
            lambda item: validate_robot_gate_review_v2(item, gate={}),
        ),
        (
            "contact_causal_matched_control_collector.v3.template.json",
            "false collector v3 template",
            validate_collector_config_v3,
        ),
        (
            "contact_causal_checkpoint_acceptance.v3.template.json",
            "false checkpoint acceptance template",
            lambda item: validate_acceptance_v3(item, receipt={}),
        ),
        (
            "contact_causal_checkpoint_training_receipt.v5.template.json",
            "false checkpoint training receipt template",
            validate_training_receipt_v5,
        ),
        (
            "contact_causal_checkpoint_manifest.v3.template.json",
            "false checkpoint manifest template",
            lambda item: validate_checkpoint_manifest_v3(
                item, verify_files=False
            ),
        ),
        (
            "contact_causal_evaluator_output.v2.template.json",
            "false evaluator output template",
            lambda item: validate_route_evaluator_output(
                item, route="hard_st_semantic"
            ),
        ),
        (
            "contact_causal_heldout_result.v3.template.json",
            "false held-out result template",
            validate_result_v3,
        ),
        (
            "contact_causal_heldout_aggregate.v5.template.json",
            "false held-out aggregate template",
            lambda item: validate_aggregate_v5(
                item, evidence={}, verify_files=False
            ),
        ),
    )
    for filename, label, validator in template_cases:
        value, _, _ = strict_json_file(
            root / "configs" / filename,
            name=label,
        )
        try:
            validator(value)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("false/null causal template authorized production")
    old_robot, _, _ = strict_json_file(
        root / "configs/contact_asym25_training_gate.template.json",
        name="legacy Robot gate template",
    )
    mixed = deepcopy(old_robot)
    mixed["matrix_id"] = MATRIX_ID
    try:
        validate_robot_gate_v3(mixed, load_external_review=False)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("new validator accepted a mixed legacy gate")
    for alias in (*LEGACY_MECHANISM_ROUTES, "hard-st", "semantic", ""):
        try:
            ensure_exact_causal_route(alias)
        except CausalProductionError:
            continue
        raise AssertionError(f"causal registry accepted alias/legacy route {alias!r}")


def _capacity_and_direct_claims() -> None:
    semantic = contact_transport_capacity_metadata(
        mode="hard_st", time_bins=16, num_queries=12
    )
    equal = unstructured_capacity_metadata()
    if (
        semantic["strict_valid_per_segment_states"] != 48_673
        or equal["states_per_query"] != 48_673
        or semantic["strict_valid_total_bits"] != 186.85000894308362
        or equal["total_bits"] != 186.85000894308362
    ):
        raise AssertionError("equal/semantic Q12/B16 capacity differs")
    direct = build_production_route_receipt("aux_only_direct_q12")
    capacity = direct["action_visible_capacity"]
    if (
        direct["planner_parameters_and_forward_aux_matched"] is not True
        or direct["direct_capacity_matched"] is not False
        or direct["direct_backward_flops_matched"] is not False
        or capacity["equal_capacity_claimed"] is not False
        or capacity["equal_backward_flops_claimed"] is not False
        or capacity["exact_compute_matched_claimed"] is not False
    ):
        raise AssertionError("direct route overclaims capacity/compute matching")


def _actual_tiny_model_topology_and_precursor() -> None:
    from world_action_model.contact_route_audit import _inputs, _tiny_model

    routes = (
        ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
        ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
        ContactCausalRoute.HARD_ST_SEMANTIC,
    )
    models = [_tiny_model(route) for route in routes]
    schemas = [
        {
            name: (tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in model.state_dict().items()
        }
        for model in models
    ]
    if schemas[0] != schemas[1] or schemas[1] != schemas[2]:
        raise AssertionError("three causal routes do not have isomorphic persistent state")
    if "equal_bit_mixing_matrix" in schemas[1]:
        raise AssertionError("equal-bit source matrix became persistent")
    common_state = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(common_state, strict=True)
        if any(
            not torch.equal(common_state[name], model.state_dict()[name])
            for name in common_state
        ):
            raise AssertionError("three routes failed exact common step-0 weight load")
    precursor = CasualWorldActionTransformer_MoT(
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
        contact_target_dim=22,
        contact_condition_dim=22,
        contact_bottleneck=False,
        contact_schema_version="contact_segment_v1_1",
        contact_semantic_bottleneck=False,
        contact_semantic_transport_mode="soft",
        contact_time_bins=16,
        contact_causal_route=None,
        contact_planner_precursor=True,
    ).train()
    precursor_schema = {
        name: (tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in precursor.state_dict().items()
    }
    if precursor_schema != schemas[0]:
        raise AssertionError("precursor persistent state differs from all Stage2 routes")
    partition = freeze_precursor_action_path(precursor)
    if not partition["contact_condition_encoder_parameter_names"]:
        raise AssertionError("precursor did not freeze contact-condition encoder")
    inputs = _inputs()
    video, action, contact = precursor(
        contact_precursor=True,
        ref_latents=inputs["ref_latents"],
        noisy_latents=inputs["future_latents"],
        timestep=torch.zeros(1, 4),
        encoder_hidden_states=inputs["encoder_hidden_states"],
        state=inputs["state"],
        embodiment_id=inputs["embodiment_id"],
        return_dict=False,
    )
    if action is not None or video is None or tuple(contact.shape) != (1, 12, 22):
        raise AssertionError("precursor forward constructed an action output or lost planner output")


def _tiny_weight_payload(*, seed: int, route: str, step: int) -> bytes:
    buffer = io.BytesIO()
    values = torch.tensor(
        [0.0, 1.0, 2.0]
        if step == 0
        else [float(seed), float(CAUSAL_ROUTES.index(route)), float(step)],
        dtype=torch.float32,
    )
    torch.save(
        {"tiny_weight": values},
        buffer,
        _use_new_zipfile_serialization=False,
    )
    return buffer.getvalue()


def _zero_safe_supervision() -> None:
    prediction = torch.tensor([[[-2.0] * 22]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction)
    confidence = torch.zeros_like(prediction)
    receipt = validate_zero_safe_contact_supervision(target, mask, confidence)
    if not receipt["zero_supervision"] or receipt["unknown_encoded_as_negative"]:
        raise AssertionError("zero-safe supervision rewrote unknown as negative")
    loss = canonical_zero_contact_loss(prediction)
    if loss.item() != 0.0 or math.copysign(1.0, loss.item()) != 1.0:
        raise AssertionError("zero-safe contact loss is not canonical +0")
    loss.backward()
    if prediction.grad is None or not torch.equal(
        prediction.grad, torch.zeros_like(prediction.grad)
    ):
        raise AssertionError("zero-safe contact loss reached planner gradients")
    try:
        validate_positive_contact_supervision(target, mask, confidence)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy v1 positive-supervision rule changed")


def _collector_v3_acceptance() -> str:
    target = torch.zeros(1, 1, 22)
    mask = torch.zeros_like(target)
    confidence = torch.zeros_like(target)
    action = torch.zeros(1, 2, 4)
    records = {}
    stable = {
        "dataset": _h("collector-dataset"),
        "hardware": _h("collector-hardware"),
        "future_fields_are_stable": True,
    }
    for route in CAUSAL_ROUTES:
        records[route] = [
            build_trace_record(
                route=route,
                seed=6666,
                global_step=0,
                micro_step=0,
                rank=0,
                world_size=1,
                run_id=f"collector-run-id-{route}",
                run_nonce=f"collector-run-nonce-{route}",
                stable_contract=stable,
                route_local_contract={"causal_route_id": route},
                ordered_sample_ids=[{"episode": 1, "frame": 2}],
                raw_batch=[_h("raw-batch")],
                augmentation_parameters=[{"crop": [0, 0, 1, 1]}],
                visual_timestep=torch.tensor([10]),
                action_timestep=torch.tensor([20]),
                visual_noise=torch.zeros(1, 1),
                action_noise=torch.zeros_like(action),
                contact_target=target,
                contact_loss_mask=mask,
                contact_confidence=confidence,
                action_target=action,
                action_loss_mask=torch.ones(1, 2),
                prompt_t5=torch.zeros(1, 2, 3),
                pre_forward_rng={"cpu": [1, 2, 3]},
                initial_parameter_tensors_sha256=_h("initial-tensors"),
                optimizer_parameter_groups={"names": ["all"]},
                optimizer_step0_state={},
                scheduler_step0_state={"step": 0},
                precursor_receipt_bytes_sha256=_h("precursor-bytes"),
            )
        ]
    report = compare_causal_records(records, required_seed=6666)
    if report["all_hashes_equal_across_routes"] is not True:
        raise AssertionError("causal collector did not match exact synthetic controls")
    drift = deepcopy(records)
    drift_route = CAUSAL_ROUTES[1]
    drift_record = deepcopy(drift[drift_route][0])
    drift_record["stable_contract"]["unknown_future_field"] = "drift"
    drift_record["stable_contract_sha256"] = sha256_json(
        drift_record["stable_contract"]
    )
    drift[drift_route] = [drift_record]
    try:
        compare_causal_records(drift, required_seed=6666)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("collector future stable field drift escaped comparison")
    return sha256_json(report)


def _unused_names(route: str) -> list[str]:
    if route == "aux_only_direct_q12":
        return [
            "contact_condition_encoder.bias",
            "contact_condition_encoder.weight",
        ]
    return []


def _stable_contract(seed: int, *, head: str, step0_sha: str) -> dict[str, Any]:
    return {
        "git_commit": head,
        "source_tree_sha256": _h("source-tree"),
        "matrix_manifest_sha256": _h("matrix-manifest"),
        "precursor_bundle_sha256": _h("precursor-bundle"),
        "base_weights_sha256": step0_sha,
        "initial_parameter_tensor_sha256": step0_sha,
        "persistent_parameter_schema_sha256": _h("parameter-schema"),
        "dataset_id": "synthetic-canonical25-reviewed-target-hardware",
        "dataset_tree_sha256": _h("dataset-tree"),
        "dataset_info_sha256": _h("dataset-info"),
        "hardware_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "hardware_contract_sha256": _h("target-hardware"),
        "calibration_capture_sha256": _h("calibration-capture"),
        "split_manifest_sha256": _h(f"split-{seed}"),
        "train_episode_ids_sha256": _h(f"train-ids-{seed}"),
        "validation_episode_ids_sha256": _h(f"val-ids-{seed}"),
        "heldout_episode_ids_sha256": _h(f"test-ids-{seed}"),
        "sidecar_bundle_sha256": _h("reviewed-robot-sidecar"),
        "contact_label_coverage_sha256": _h("contact-coverage"),
        "contact_target_stream_sha256": _h("contact-target-stream"),
        "contact_mask_stream_sha256": _h("contact-mask-stream"),
        "contact_confidence_stream_sha256": _h("contact-confidence-stream"),
        "action_target_stream_sha256": _h("action-target-stream"),
        "action_mask_stream_sha256": _h("action-mask-stream"),
        "prompt_t5_stream_sha256": _h("prompt-t5-stream"),
        "train_only_norm_sha256": _h(f"train-only-norm-{seed}"),
        "transform_config_sha256": _h("transform-config"),
        "augmentation_trace_sha256": _h("augmentation-trace"),
        "ordered_window_manifest_sha256": _h(f"windows-{seed}"),
        "visual_timestep_stream_sha256": _h("visual-timesteps"),
        "action_timestep_stream_sha256": _h("action-timesteps"),
        "visual_noise_stream_sha256": _h("visual-noise"),
        "action_noise_stream_sha256": _h("action-noise"),
        "pre_forward_rng_stream_sha256": _h("pre-forward-rng"),
        "dropout_config_sha256": _h("dropout-disabled"),
        "optimizer_config_sha256": _h("adamw-config"),
        "optimizer_parameter_groups_sha256": _h("optimizer-groups"),
        "optimizer_step0_state_sha256": _h("optimizer-empty-step0"),
        "scheduler_config_sha256": _h("scheduler-config"),
        "scheduler_step0_state_sha256": _h("scheduler-step0"),
        "precision": "bfloat16",
        "world_size": 1,
        "rank_set": [0],
        "window_seed": seed + 10,
        "noise_seed": seed + 20,
        "training_seed": seed,
    }


def _make_receipt(
    *,
    root: Path,
    seed: int,
    route: str,
    step: int,
    head: str,
    step0_sha: str,
    checkpoint_sha: str,
    parent_sha: str,
) -> dict[str, Any]:
    route_receipt = build_production_route_receipt(route)
    unused = _unused_names(route)
    direct = route == "aux_only_direct_q12"
    run_label = f"seed{seed}-{route}"
    checkpoint_path = root / f"seed{seed}" / route / f"step{step}" / "tiny_weights.bin"
    parameter_topology = {
        "persistent_parameter_schema_sha256": _h("parameter-schema"),
        "persistent_buffer_schema_sha256": _h("buffer-schema"),
        "parameter_tensor_sha256": checkpoint_sha,
        "buffer_tensor_sha256": _h("buffer-tensors"),
        "step0_parameter_tensor_sha256": step0_sha,
        "optimizer_group_names_sha256": _h("optimizer-groups"),
        "optimizer_hyperparameters_sha256": _h("adamw-config"),
        "optimizer_state_names_sha256": _h(
            "optimizer-state-direct-missing-condition" if direct else "optimizer-state-strict"
        ),
        "scheduler_state_sha256": _h("scheduler-step0"),
        "equal_bit_mixing_persistent": False,
        "unused_parameter_names": unused,
        "unused_parameter_names_sha256": sha256_json(unused),
        "unused_parameters_have_grad": False,
        "unused_parameters_have_optimizer_state": False,
        "unused_parameters_changed_from_step0": False,
    }
    return {
        "receipt_version": "contact-asym25-checkpoint-training-receipt/v5",
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "seed": seed,
        "checkpoint_step": step,
        "route": {
            "causal_route_id": route,
            "route_contract_sha256": route_receipt["receipt_sha256"],
            "action_attends_reference": route_receipt["action_attends_reference"],
            "action_uses_packet": route_receipt["action_uses_packet"],
            "action_visible_capacity": route_receipt["action_visible_capacity"],
            "cache_policy": route_receipt["op38_route_receipt"]["cache_policy"],
            "transport_receipt_sha256": route_receipt["op38_route_receipt_sha256"],
        },
        "run_identity": {
            "variant_id": route,
            "training_run_id": f"training-run-id-{run_label}",
            "training_run_nonce": f"training-run-nonce-{run_label}",
            "external_acceptance_sha256": _h(f"external-review-{run_label}"),
        },
        "output_path_record_only": str(checkpoint_path.parent),
        "model_config": {
            "contact_causal_route": route,
            "transformer": {
                "contact_causal_route": route,
                "contact_bottleneck": not direct,
                "contact_semantic_bottleneck": not direct,
                "contact_semantic_transport_mode": "hard_st",
                "num_contact_tokens": 12,
                "contact_target_dim": 22,
                "contact_time_bins": 16,
            },
        },
        "stable_contract": _stable_contract(seed, head=head, step0_sha=step0_sha),
        "training_contract": {
            "contact_schema_version": "contact_segment_v1_1",
            "supervision_version": SUPERVISION_VERSION,
            "contact_loss_weight": 0.25,
            "contact_condition_mode": "predicted",
            "contact_target_model_input": False,
            "contact_condition_mask_model_input": False,
            "oracle_used": False,
            "hand_active_used_as_contact_ground_truth": False,
            "grip_used_as_contact_ground_truth": False,
            "zero_supervision_batch_policy": "canonical_positive_zero_no_planner_gradient",
            "dataset_level_coverage_required": True,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "primary_checkpoint_step": 200,
            "best_checkpoint_selection_allowed": False,
        },
        "precursor": {
            "precursor_id": PRECURSOR_ID,
            "receipt_bytes_sha256": _h("byte-identical-precursor-receipt"),
            "bundle_sha256": _h("precursor-bundle"),
            "parameter_tensor_sha256": step0_sha,
            "action_path_scientific_input_output": False,
            "route_neutral": True,
        },
        "evidence": {
            "training_gate_sha256": _h(f"training-gate-{seed}"),
            "production_preflight_receipt_sha256": _h(f"preflight-{seed}"),
            "training_data_receipt_sha256": _h(f"training-data-{seed}"),
            "matched_control_receipt_sha256": _h(f"matched-control-{seed}"),
            "external_reviewer_acceptance_sha256": _h(f"gate-review-{seed}"),
            "contact_coverage_receipt_sha256": _h("contact-coverage"),
            "trace_manifest_sha256": _h("trace-manifest"),
            "current_head_source_receipt_sha256": _h("source-tree"),
        },
        "topology": {"world_size": 1, "rank_set": [0]},
        "parameter_topology": parameter_topology,
        "measured_route_costs": {
            "forward_flops_proxy": 100.0 + CAUSAL_ROUTES.index(route),
            "backward_flops_proxy": 200.0 + CAUSAL_ROUTES.index(route),
            "active_gradient_parameter_count": 1000 + CAUSAL_ROUTES.index(route),
            "peak_memory_bytes": 4096 + CAUSAL_ROUTES.index(route),
            "step_time_ms": 10.0 + CAUSAL_ROUTES.index(route),
            "action_only_latency_ms": 2.0 + CAUSAL_ROUTES.index(route),
            "transport_flops_proxy": 5.0 + CAUSAL_ROUTES.index(route),
        },
        "gradient_reachability_receipt": {
            "active_parameter_names_sha256": _h(f"active-{route}"),
            "inactive_parameter_names_sha256": _h(f"inactive-{route}"),
            "action_loss_reaches_planner": not direct,
            "unused_parameter_names": unused,
        },
        "checkpoint_artifact_hashes": {
            "weight_tensor_sha256": checkpoint_sha,
            "buffer_tensor_sha256": _h("buffer-tensors"),
            "config_sha256": _h(f"config-{route}"),
            "whole_bundle_sha256": _h(f"bundle-{route}-step{step}"),
            "route_receipt_sha256": route_receipt["receipt_sha256"],
        },
        "checkpoint_file": {
            "path_record_only": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "parent_checkpoint_weight_sha256": parent_sha,
        "scientific_admission": {
            "production_data_admission": False,
            "allow_5b_training": False,
            "allow_server_training": False,
            "benefit_claimed": False,
        },
    }


def _make_acceptance(receipt: dict[str, Any]) -> dict[str, Any]:
    seed = receipt["seed"]
    route = receipt["route"]["causal_route_id"]
    step = receipt["checkpoint_step"]
    label = f"seed{seed}-{route}-step{step}"
    return {
        "acceptance_version": "contact-asym25-checkpoint-independent-acceptance/v3",
        "status": "accepted",
        "accepted": True,
        "matrix_id": MATRIX_ID,
        "causal_route_id": route,
        "seed": seed,
        "checkpoint_step": step,
        "training_receipt_sha256": sha256_json(receipt),
        "checkpoint_file_sha256": receipt["checkpoint_file"]["sha256"],
        "reviewer_identity": f"reviewer-identity-{label}",
        "reviewer_attestation_nonce": f"reviewer-attestation-{label}",
        "acceptance_run_id": f"acceptance-run-id-{label}",
        "acceptance_run_nonce": f"acceptance-run-nonce-{label}",
        "reviewed_current_head_sha256": receipt["stable_contract"]["git_commit"],
        "reviewed_source_tree_sha256": receipt["stable_contract"]["source_tree_sha256"],
        "evidence_valid_until_utc": "2099-12-31T23:59:59Z",
        "oracle_used": False,
    }


def _route_outputs() -> None:
    for route in CAUSAL_ROUTES:
        kind = route_packet_kind(route)
        if route == "aux_only_direct_q12":
            packet_sha = None
            packet_metrics = {
                "kind": kind,
                "semantic_field_metrics_allowed": False,
            }
        elif route == "equal_bit_unstructured_hard_st":
            packet_sha = _h("equal-packet")
            packet_metrics = {
                "kind": kind,
                "codec_validity": 1.0,
                "utilization": 0.5,
                "entropy": 1.0,
                "invalid_to_zero_rate": 0.0,
                "semantic_field_metrics_allowed": False,
            }
        else:
            packet_sha = _h("semantic-packet")
            packet_metrics = {
                "kind": kind,
                "grammar_validity": 1.0,
                "field_accuracy": 0.5,
                "semantic_field_metrics_allowed": True,
            }
        output = {
            "evaluator_version": EVALUATOR_VERSION,
            "matrix_id": MATRIX_ID,
            "causal_route_id": route,
            "contact_aux_prediction_source": "shared_semantic_aux_decoder_logits",
            "contact_target_model_input": False,
            "contact_condition_mask_model_input": False,
            "oracle_used": False,
            "packet_kind": kind,
            "packet_present": route != "aux_only_direct_q12",
            "packet_sha256": packet_sha,
            "semantic_aux_logits_sha256": _h("semantic-aux-logits"),
            "action_prediction_sha256": _h("action-prediction"),
            "action_metrics_contract_sha256": _h("action-metrics"),
            "packet_metrics": packet_metrics,
        }
        validate_route_evaluator_output(output, route=route)
    leaked = deepcopy(output)
    leaked["contact_target_model_input"] = True
    try:
        validate_route_evaluator_output(leaked, route="hard_st_semantic")
    except CausalProductionError:
        pass
    else:
        raise AssertionError("causal evaluator accepted GT leakage")


def run_cpu_acceptance(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).expanduser().absolute()
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="op42_cpu_acceptance_", dir=outputs))
    try:
        protected = _assert_legacy_protected_bytes(root)
        _strict_json_negatives()
        _false_template_negatives(root)
        _capacity_and_direct_claims()
        _actual_tiny_model_topology_and_precursor()
        _zero_safe_supervision()
        collector_report_sha256 = _collector_v3_acceptance()
        _route_outputs()
        from world_action_model.contact_route_audit import (
            run_contact_route_cpu_audit,
        )

        op38_report = run_contact_route_cpu_audit()
        if op38_report["passed"] is not True:
            raise AssertionError("protected op38 route audit no longer passes")
        matrix, _, _ = strict_json_file(
            root / "configs/contact_causal_production_matrix.v1.json",
            name="causal production matrix",
        )
        validate_matrix_manifest(matrix)
        head = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
        step0_payload = _tiny_weight_payload(
            seed=SEEDS[0], route=CAUSAL_ROUTES[0], step=0
        )
        step0_sha = hashlib.sha256(step0_payload).hexdigest()
        evidence = {}
        for seed in SEEDS:
            for route in CAUSAL_ROUTES:
                parent = step0_sha
                for step in CHECKPOINT_STEPS:
                    payload = (
                        step0_payload
                        if step == 0
                        else _tiny_weight_payload(seed=seed, route=route, step=step)
                    )
                    checkpoint_path = (
                        workspace
                        / f"seed{seed}"
                        / route
                        / f"step{step}"
                        / "tiny_weights.bin"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.write_bytes(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                    receipt = _make_receipt(
                        root=workspace,
                        seed=seed,
                        route=route,
                        step=step,
                        head=head,
                        step0_sha=step0_sha,
                        checkpoint_sha=digest,
                        parent_sha=(step0_sha if step == 0 else parent),
                    )
                    acceptance = _make_acceptance(receipt)
                    validate_training_receipt_v5(receipt)
                    validate_acceptance_v3(acceptance, receipt=receipt)
                    evidence[(seed, route, step)] = (receipt, acceptance)
                    parent = digest
        report = validate_27_checkpoint_matrix(evidence, verify_files=True)
        drift = deepcopy(evidence[(6666, CAUSAL_ROUTES[0], 0)][0])
        drift["stable_contract"]["optimizer_config_sha256"] = _h("drift")
        same_seed = [
            drift,
            evidence[(6666, CAUSAL_ROUTES[1], 0)][0],
            evidence[(6666, CAUSAL_ROUTES[2], 0)][0],
        ]
        try:
            compare_same_seed_route_contracts(same_seed)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("stable optimizer drift escaped route allowlist")
        precursor_drift = deepcopy(evidence)
        drift_receipt, drift_acceptance = precursor_drift[(6666, CAUSAL_ROUTES[0], 0)]
        drift_receipt = deepcopy(drift_receipt)
        drift_receipt["precursor"]["receipt_bytes_sha256"] = _h("different precursor bytes")
        drift_acceptance = _make_acceptance(drift_receipt)
        precursor_drift[(6666, CAUSAL_ROUTES[0], 0)] = (
            drift_receipt,
            drift_acceptance,
        )
        try:
            validate_27_checkpoint_matrix(precursor_drift, verify_files=False)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("precursor byte drift escaped 27-checkpoint validation")
        alias_receipt = deepcopy(evidence[(6666, CAUSAL_ROUTES[2], 0)][0])
        alias_receipt["route"]["causal_route_id"] = "hard_st"
        try:
            validate_training_receipt_v5(alias_receipt)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("legacy hard_st alias entered causal receipt")
        bool_receipt = deepcopy(evidence[(6666, CAUSAL_ROUTES[2], 0)][0])
        bool_receipt["checkpoint_step"] = False
        try:
            validate_training_receipt_v5(bool_receipt)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("boolean passed as causal checkpoint step")
        return {
            "acceptance_version": "contact-causal-production-cpu-acceptance/v1",
            "status": "passed",
            "device": "cpu",
            "cuda_visible_devices_required": "",
            "legacy_protected_file_count": len(protected),
            "legacy_protected_sha256": sha256_json(protected),
            "protected_op38_route_audit_sha256": op38_report["report_sha256"],
            "causal_collector_v3_report_sha256": collector_report_sha256,
            "synthetic_checkpoint_file_count": 27,
            "synthetic_checkpoint_files_nonempty": True,
            "synthetic_checkpoint_files_are_loadable_torch_weights": True,
            "matrix_report": report,
            "negative_cases": [
                "strict_json_duplicate_nan_inf_overflow",
                "false_null_templates",
                "old_new_schema_mix",
                "legacy_route_alias",
                "stable_optimizer_drift",
                "precursor_receipt_byte_drift",
                "boolean_integer_alias",
                "GT_oracle_evaluator_leakage",
                "zero_supervision_unknown_not_negative",
            ],
            "artifacts_cleaned_after_return": True,
            "production_data_admission": False,
            "allow_5b_training": False,
            "allow_server_training": False,
            "benefit_claimed": False,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=False)

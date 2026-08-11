"""CPU-only synthetic acceptance for the causal-main production plumbing."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import numpy as np
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import torch

from world_action_model.contact_causal_heldout_eval import (
    FORBIDDEN_INFERENCE_TARGET_KEYS,
    INFERENCE_PACKET_KEYS,
    ROUTE_OUTPUT_EVIDENCE_FORMAT,
    _array_stream_sha256,
    _load_route_output_evidence,
    _recompute_route_packet_metrics,
    _validate_finite_metric_tree,
    validate_aggregate_v5,
    validate_result_v3,
    validate_route_evaluator_output,
    validate_target_free_inference_packet,
    write_route_output_evidence,
)
from world_action_model.contact_causal_checkpoint import (
    inspect_causal_checkpoint_bundle,
    load_safe_weight_mapping,
    validate_checkpoint_manifest_v3,
)
from world_action_model.contact_causal_matched_control import (
    build_trace_record,
    build_current_run_trace_prefix_receipt_v1,
    build_synthetic_stable_contract_v3,
    capture_pre_forward_rng_v3,
    compare_causal_records,
    live_optimizer_scheduler_receipts_v3,
    runtime_control_config_receipts_v3,
    validate_collector_config_v3,
    validate_collector_runtime_bindings_v3,
)
from world_action_model.contact_causal_preflight import (
    validate_robot_gate_review_v2,
    validate_robot_gate_v3,
)
from world_action_model.contact_causal_gate_bootstrap import (
    validate_robot_route_acceptance_v1,
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
    checkpoint_transformer_semantic_config,
    build_live_training_state_receipts,
    compare_same_seed_route_contracts,
    ensure_exact_causal_route,
    route_packet_kind,
    named_tensor_digest,
    sha256_json,
    strict_json_file,
    validate_27_checkpoint_matrix,
    validate_acceptance_v3,
    validate_matrix_manifest,
    validate_training_receipt_v5,
    verify_checkpoint_file,
)
from world_action_model.contact_ego_causal_precursor import (
    build_live_precursor_receipt,
    freeze_precursor_action_path,
    precursor_objective_contract,
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
from world_action_model.matched_control_hashes import (
    IMPLEMENTED_RUNNER_VARIANTS,
    canonical_sha256,
)
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
    def reject_route_acceptance_template(item: object) -> None:
        try:
            validate_robot_route_acceptance_v1(
                item, gate={}, route=CAUSAL_ROUTES[0], repository_root=root
            )
        except ValueError as exc:
            raise CausalProductionError("false route acceptance rejected") from exc

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
            "contact_causal_route_training_acceptance.v1.template.json",
            "false route training acceptance template",
            reject_route_acceptance_template,
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


def _actual_tiny_model_topology_and_precursor(workspace: Path, head: str) -> None:
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
    root_parameter_sha256 = named_tensor_digest(list(precursor.named_parameters()))
    frozen_root_sha256 = named_tensor_digest(
        [
            (name, parameter)
            for name, parameter in precursor.named_parameters()
            if name in set(partition["frozen_parameter_names"])
        ]
    )
    precursor_dir = workspace / "tiny-precursor" / "transformer"
    precursor_dir.mkdir(parents=True, exist_ok=False)
    (precursor_dir / "config.json").write_bytes(
        json.dumps(
            {"_class_name": "SyntheticRouteNeutralPrecursor"},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    )
    (precursor_dir / "diffusion_pytorch_model.bin").write_bytes(
        _tiny_weight_payload(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in precursor.state_dict().items()
            }
        )
    )
    objective = precursor_objective_contract(fixed_save_step=100)
    receipt, bundle = build_live_precursor_receipt(
        model=precursor,
        checkpoint_dir=precursor_dir,
        parameter_partition=partition,
        objective_contract=objective,
        evidence_base={
            "gate_sha256": _h("precursor-gate"),
            "root_checkpoint_bundle_sha256": _h("precursor-root-bundle"),
            "dataset_source_tree_sha256": _h("precursor-dataset-source"),
            "sidecar_bundle_sha256": _h("precursor-sidecar"),
            "training_code_head": head,
            "training_code_source_tree_sha256": _h("precursor-source"),
            "training_run_id": "precursor-training-run-id",
            "training_run_nonce": "precursor-training-run-nonce",
            "reviewer_acceptance_sha256": _h("precursor-review"),
        },
        root_parameter_tensor_sha256=root_parameter_sha256,
        frozen_root_tensor_sha256=frozen_root_sha256,
    )
    validate_precursor_receipt(receipt)
    if bundle["file_count"] != 2:
        raise AssertionError("precursor bundle is not config + one safe weight file")


def _trainer_first_batch_three_route_acceptance() -> dict[str, Any]:
    """Run the real causal trainer forward/backward with the real tiny model."""

    import sys
    import types

    trainer_package = types.ModuleType("world_action_model.trainer")
    trainer_package.__path__ = [
        str(Path(__file__).parent / "trainer")
    ]
    sys.modules["world_action_model.trainer"] = trainer_package
    giga_train = types.ModuleType("giga_train")
    giga_train.ModuleDict = torch.nn.ModuleDict
    giga_train.Trainer = object
    sys.modules["giga_train"] = giga_train
    giga_models = types.ModuleType("giga_models")
    giga_models_utils = types.ModuleType("giga_models.utils")
    giga_models.utils = giga_models_utils
    sys.modules["giga_models"] = giga_models
    sys.modules["giga_models.utils"] = giga_models_utils

    from world_action_model.contact_supervision_contract import (
        CONTACT_GROUPED_SUPERVISION_ZERO_SAFE_VERSION,
    )
    from world_action_model.contact_transition_schema import (
        SEGMENT_SCHEMA_VERSION,
        SEGMENT_TARGET_DIM,
    )
    from world_action_model.trainer.wa_casual_trainer_mot import (
        CasualWATrainerMoT,
    )

    class _NamedModel:
        def __init__(self, transformer):
            self.transformer = transformer
            self.forward_conditions = []

        def __call__(self, name, **kwargs):
            if name != "transformer":
                raise AssertionError("trainer requested an unexpected model")
            self.forward_conditions.append(
                (
                    kwargs.get("contact_condition"),
                    kwargs.get("contact_condition_mask"),
                )
            )
            return self.transformer(**kwargs)

    class _Accelerator:
        @staticmethod
        def unwrap_model(value):
            return value

    def _model(route: str) -> CasualWorldActionTransformer_MoT:
        direct = route == "aux_only_direct_q12"
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
            contact_schema_version=SEGMENT_SCHEMA_VERSION,
            contact_semantic_bottleneck=not direct,
            contact_semantic_transport_mode="hard_st",
            contact_time_bins=16,
            contact_causal_route=route,
            production_matrix_id=MATRIX_ID,
        ).train()

    report = {}
    for route_index, route in enumerate(CAUSAL_ROUTES):
        torch.manual_seed(42_000 + route_index)
        model = _model(route)
        trainer = CasualWATrainerMoT.__new__(CasualWATrainerMoT)
        named_model = _NamedModel(model)
        trainer.model = named_model
        trainer.models = [{"transformer": model}]
        trainer.accelerator = _Accelerator()
        trainer.dtype = torch.float32
        trainer.ego_contact_precursor_objective = None
        trainer.expand_timesteps = True
        trainer.visual_flow_shift = 5.0
        trainer.action_flow_shift = 5.0
        trainer.state_repeats = 1
        trainer.action_repeats = 1
        trainer.matched_control_collector = None
        trainer.cur_step = 1
        trainer.production_matrix_id = MATRIX_ID
        trainer.production_causal_route_id = route
        trainer.contact_causal_route = route
        trainer.contact_condition_mode = "predicted"
        trainer.contact_schema_version = SEGMENT_SCHEMA_VERSION
        trainer.contact_supervision_rule_version = (
            CONTACT_GROUPED_SUPERVISION_ZERO_SAFE_VERSION
        )
        trainer.require_positive_contact_supervision = False
        trainer.require_contact_targets = True
        trainer.contact_loss_type = "bce"
        trainer.contact_loss_weight = 1.0
        trainer._causal_action_gradient_probe_completed = False
        trainer._causal_action_gradient_names = set()

        def _forward_vae(images):
            return torch.zeros(
                images.shape[0], 4, images.shape[1], 2, 2, dtype=torch.float32
            )

        def _get_timestep_and_sigma(batch_size, ndim, _flow_shift):
            return (
                torch.full((batch_size,), 500.0, dtype=torch.float32),
                torch.full(
                    (batch_size,) + (1,) * (ndim - 1),
                    0.5,
                    dtype=torch.float32,
                ),
            )

        trainer.forward_vae = _forward_vae
        trainer.get_timestep_and_sigma = _get_timestep_and_sigma
        target = torch.zeros(1, 12, SEGMENT_TARGET_DIM)
        mask = torch.ones_like(target)
        confidence = torch.ones_like(target)
        batch = {
            "images": torch.zeros(1, 2, 3, 2, 2),
            "ref_images": torch.zeros(1, 1, 3, 2, 2),
            "prompt_embeds": torch.zeros(1, 2, 8),
            "action": torch.zeros(1, 3, 4),
            "state": torch.zeros(1, 2, 4),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
            "action_dim_mask": torch.ones(4, dtype=torch.bool),
            "action_loss_mask": torch.ones(3, dtype=torch.bool),
            "contact_target": target,
            "contact_loss_mask": mask,
            "contact_confidence": confidence,
        }
        rejected = dict(batch)
        rejected["contact_condition_mask"] = torch.ones_like(target)
        try:
            trainer.forward_step(rejected)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"causal trainer accepted a model-side condition mask: {route}"
            )

        step0_sha = named_tensor_digest(list(model.named_parameters()))
        direct_unused = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("contact_condition_encoder.")
        ]
        unused_step0_sha = named_tensor_digest(direct_unused)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        optimizer.zero_grad(set_to_none=True)
        losses = trainer.forward_step(batch)
        if set(losses) != {"visual_loss", "action_loss", "contact_loss"}:
            raise AssertionError(f"causal first batch loss schema differs: {route}")
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise AssertionError(f"causal first batch produced a nonfinite loss: {route}")
        total.backward()
        observed = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        )
        optimizer.step()
        scheduler.step()
        parameter_names_by_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        optimizer_state_names = {
            parameter_names_by_id[id(parameter)] for parameter in optimizer.state
        }
        expected_optimizer_state_names = set(observed)
        if optimizer_state_names != expected_optimizer_state_names:
            raise AssertionError(
                f"causal live optimizer state coverage differs for {route}: "
                f"missing={sorted(expected_optimizer_state_names - optimizer_state_names)} "
                f"extra={sorted(optimizer_state_names - expected_optimizer_state_names)}"
            )
        topology, reachability = build_live_training_state_receipts(
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
            route=route,
            checkpoint_step=100,
            step0_parameter_tensor_sha256=step0_sha,
            observed_gradient_parameter_names=observed,
            action_loss_gradient_parameter_names=sorted(
                trainer._causal_action_gradient_names
            ),
            unused_step0_tensor_sha256=unused_step0_sha,
        )
        if any(
            condition is not None
            for pair in named_model.forward_conditions
            for condition in pair
        ):
            raise AssertionError(f"causal model received a condition input: {route}")
        report[route] = {
            "losses": sorted(losses),
            "action_loss_gradient_parameter_count": len(
                trainer._causal_action_gradient_names
            ),
            "live_optimizer_state_parameter_count": len(
                topology["optimizer_state_parameter_names"]
            ),
            "active_parameter_count": reachability[
                "active_gradient_parameter_count"
            ],
            "model_condition_inputs_all_none": True,
            "condition_mask_negative_rejected": True,
        }
    return {
        "status": "passed",
        "actual_trainer_module": str(
            Path(__import__(
                "world_action_model.trainer.wa_casual_trainer_mot",
                fromlist=["__file__"],
            ).__file__).absolute()
        ),
        "stub_scope": {
            "giga_train": "base Trainer/ModuleDict import only; forward_step is real",
            "giga_models": "unused get_models import only",
            "world_action_model.trainer": "package_init_only",
        },
        "routes": report,
    }


class _TinyCausalCheckpointModel(torch.nn.Module):
    """Small isomorphic state used to exercise the real receipt builders on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.contact_planner_tiny_weight = torch.nn.Parameter(
            torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        )
        self.contact_condition_encoder = torch.nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            self.contact_condition_encoder.weight.fill_(3.0)
            self.contact_condition_encoder.bias.fill_(4.0)

    @property
    def tiny_weight(self) -> torch.nn.Parameter:
        return self.contact_planner_tiny_weight


def _tiny_training_state(
    *, route: str, step: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any], str]:
    """Produce loadable weights and receipts from actual model/optimizer state."""

    model = _TinyCausalCheckpointModel()
    initial_parameter_sha256 = named_tensor_digest(list(model.named_parameters()))
    unused_initial_sha256 = named_tensor_digest(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("contact_condition_encoder.")
        ]
    )
    direct = route == "aux_only_direct_q12"
    optimized = [parameter for _, parameter in model.named_parameters()]
    optimizer = torch.optim.AdamW(optimized, lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    observed: set[str] = set()
    for _ in range(step // 100):
        optimizer.zero_grad(set_to_none=True)
        loss = model.tiny_weight.square().sum()
        if not direct:
            loss = loss + model.contact_condition_encoder.weight.square().sum()
            loss = loss + model.contact_condition_encoder.bias.square().sum()
        loss.backward()
        observed.update(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        )
        optimizer.step()
        scheduler.step()
    topology, reachability = build_live_training_state_receipts(
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        route=route,
        checkpoint_step=step,
        step0_parameter_tensor_sha256=initial_parameter_sha256,
        observed_gradient_parameter_names=sorted(observed),
        action_loss_gradient_parameter_names=(
            []
            if direct or step == 0
            else sorted(
                name
                for name, _ in model.named_parameters()
                if name.startswith("contact_")
            )
        ),
        unused_step0_tensor_sha256=unused_initial_sha256,
    )
    mapping = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    return mapping, topology, reachability, initial_parameter_sha256


def _tiny_weight_payload(mapping: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(mapping, buffer, _use_new_zipfile_serialization=False)
    return buffer.getvalue()


def _safe_checkpoint_negative_acceptance(
    workspace: Path, receipt: dict[str, Any]
) -> None:
    negative_root = workspace / "unsafe-checkpoint-negatives"
    negative_root.mkdir(parents=True, exist_ok=False)
    config_payload = json.dumps(
        {
            "contact_causal_route": "hard_st_semantic",
            "production_matrix_id": MATRIX_ID,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    nested_buffer = io.BytesIO()
    torch.save(
        {"nested": {"tensor": torch.ones(1)}},
        nested_buffer,
        _use_new_zipfile_serialization=False,
    )
    for name, payload in (
        ("random", b"README/random bytes are not model weights"),
        ("nested", nested_buffer.getvalue()),
    ):
        root = negative_root / name
        root.mkdir()
        (root / "config.json").write_bytes(config_payload)
        (root / "diffusion_pytorch_model.bin").write_bytes(payload)
        try:
            inspect_causal_checkpoint_bundle(root)
        except CausalProductionError:
            pass
        else:
            raise AssertionError(f"unsafe checkpoint payload was accepted: {name}")
    readme_root = negative_root / "readme-only"
    readme_root.mkdir()
    (readme_root / "config.json").write_bytes(config_payload)
    (readme_root / "README.md").write_text("not weights\n", encoding="utf-8")
    try:
        inspect_causal_checkpoint_bundle(readme_root)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("README-only checkpoint bundle was accepted")
    mixed_root = negative_root / "mixed-layout"
    shutil.copytree(Path(receipt["checkpoint_file"]["path_record_only"]).parent, mixed_root)
    (mixed_root / "diffusion_pytorch_model.safetensors").write_bytes(b"not-safe")
    try:
        inspect_causal_checkpoint_bundle(mixed_root)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("multiple weight layouts were accepted")
    manifest_drift = deepcopy(receipt)
    manifest_drift["checkpoint_file"]["tensor_manifest"]["tensors"][0][
        "bytes_sha256"
    ] = _h("forged tensor bytes")
    try:
        validate_training_receipt_v5(manifest_drift)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("forged per-tensor digest was accepted")


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
    action_mask = torch.ones(1, 2)
    prompt_t5 = torch.zeros(1, 2, 3)
    state_input = torch.zeros(1, 1, 3)
    state_mask = None
    embodiment_id = torch.zeros(1, dtype=torch.long)
    actual_action_mask = {
        "dimension_mask": None,
        "time_mask": action_mask,
        "dimension_weight": None,
    }
    model = _TinyCausalCheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    live = live_optimizer_scheduler_receipts_v3(
        model=model, optimizers=[optimizer], schedulers=[scheduler]
    )
    rng = capture_pre_forward_rng_v3()
    trace_key = "step0/micro0/rank0"

    def stream_digest(value: object) -> str:
        return sha256_json({trace_key: canonical_sha256(value)})

    records = {}
    stable = {
        "stable_contract_version": "contact-causal-matched-stable-contract/v3",
        "code": {
            "git_commit": "a" * 40,
            "source_tree_sha256": _h("collector-source-tree"),
            "collector_source_sha256": _h("collector-source"),
        },
        "config": {
            "matrix_manifest_sha256": _h("collector-matrix"),
            "control_config_sha256": _h("collector-control"),
            "transform_config_sha256": _h("collector-transform"),
            "optimizer_config_sha256": _h("collector-optimizer"),
            "scheduler_config_sha256": _h("collector-scheduler"),
        },
        "dataset": {
            "dataset_id": "synthetic-canonical25-reviewed-target-hardware",
            "dataset_tree_sha256": _h("collector-dataset"),
            "dataset_info_sha256": _h("collector-dataset-info"),
            "episode_metadata_sha256": _h("collector-episode-meta"),
            "split_manifest_sha256": _h("collector-split"),
            "ordered_window_manifest_sha256": _h("collector-windows"),
        },
        "hardware": {
            "hardware_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
            "hardware_contract_sha256": _h("collector-hardware"),
            "calibration_capture_sha256": _h("collector-capture"),
        },
        "normalization": {
            "train_only_norm_sha256": _h("collector-normalization")
        },
        "runtime": {
            "runtime_environment_sha256": _h("runtime-environment"),
            "dataset_runtime_source_receipt_sha256": _h("dataset-runtime-source"),
            "t5_runtime_source_receipt_sha256": _h("t5-runtime-source"),
        },
        "precursor": {
            "precursor_receipt_bytes_sha256": _h("precursor-bytes"),
            "precursor_bundle_sha256": _h("precursor-bundle"),
            "initial_parameter_tensors_sha256": _h("initial-tensors"),
        },
        "labels": {
            "sidecar_meta_sha256": _h("sidecar-meta"),
            "sidecar_bundle_sha256": _h("sidecar-bundle"),
            "contact_coverage_sha256": _h("coverage"),
            "contact_target_stream_sha256": stream_digest(target),
            "contact_mask_stream_sha256": stream_digest(mask),
            "contact_confidence_stream_sha256": stream_digest(confidence),
            "action_target_stream_sha256": stream_digest(action),
            "action_mask_stream_sha256": stream_digest(actual_action_mask),
            "state_input_stream_sha256": stream_digest(state_input),
            "state_mask_stream_sha256": stream_digest(state_mask),
            "embodiment_id_stream_sha256": stream_digest(embodiment_id),
            "prompt_t5_stream_sha256": stream_digest(prompt_t5),
        },
        "topology": {"world_size": 1, "rank_set": [0]},
        "source_receipts": {
            "training_data_receipt_sha256": _h("training-data"),
            "preflight_receipt_sha256": _h("preflight"),
            "coverage_receipt_sha256": _h("coverage-receipt"),
            "current_head_source_receipt_sha256": _h("source-receipt"),
        },
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
                route_local_contract=build_production_route_receipt(route),
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
                action_loss_mask=actual_action_mask,
                state_input=state_input,
                state_input_mask=state_mask,
                embodiment_id=embodiment_id,
                prompt_t5=prompt_t5,
                pre_forward_rng=rng,
                initial_parameter_tensors_sha256=_h("initial-tensors"),
                optimizer_parameter_groups=live["optimizer_parameter_groups"],
                optimizer_step0_state=live["optimizer_step0_state"],
                scheduler_step0_state=live["scheduler_step0_state"],
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


def _collector_runtime_binding_acceptance(root: Path, head: str) -> None:
    route = CAUSAL_ROUTES[0]
    training_config = {
        "dataloaders": {"train": {"transform": {"matched_control_capture": True}}},
        "train": {"seed": 6666, "mixed_precision": "bf16"},
        "optimizers": {"type": "SyntheticAdamW", "lr": 1e-3},
        "schedulers": {"type": "SyntheticConstant"},
        "models": {
            "contact_causal_route": route,
            "production_causal_route_id": route,
            "transformer": {
                "contact_causal_route": route,
                "contact_bottleneck": False,
                "contact_semantic_bottleneck": False,
            },
        },
    }
    collector_source = hashlib.sha256(
        (root / "world_action_model/contact_causal_matched_control.py").read_bytes()
    ).hexdigest()
    source_receipt = {
        "receipt_version": "synthetic-current-head-source-tree/v1",
        "git_commit": head,
        "source_logical_paths": [
            "world_action_model/contact_causal_matched_control.py"
        ],
        "source_files": [
            {
                "logical_path": "world_action_model/contact_causal_matched_control.py",
                "runtime_sha256": collector_source,
            }
        ],
    }
    training_data = {
        "dataset_id": "synthetic-canonical25-reviewed-target-hardware",
        "dataset_tree_sha256": _h("binding-dataset-tree"),
        "dataset_info_sha256": _h("binding-dataset-info"),
        "episode_metadata_sha256": _h("binding-episodes"),
        "split_manifest_sha256": _h("binding-split"),
        "ordered_window_manifest_sha256": _h("binding-windows"),
        "hardware_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "hardware_contract_sha256": _h("binding-hardware"),
        "calibration_capture_sha256": _h("binding-capture"),
        "train_only_norm_sha256": _h("binding-norm"),
        "sidecar_bundle_sha256": _h("binding-sidecar"),
    }
    training_sha = _h("binding-training-data")
    preflight_sha = _h("binding-preflight")
    coverage_sha = _h("binding-coverage")
    matrix_sha = _h("binding-matrix")
    sidecar_meta_sha = _h("binding-sidecar-meta")
    precursor = {
        "receipt_bytes_sha256": _h("binding-precursor-receipt"),
        "bundle_sha256": _h("binding-precursor-bundle"),
        "parameter_tensor_sha256": _h("binding-precursor-parameters"),
    }
    stable = build_synthetic_stable_contract_v3()
    stable["code"] = {
        "git_commit": head,
        "source_tree_sha256": sha256_json(source_receipt),
        "collector_source_sha256": collector_source,
    }
    stable["config"] = {
        "matrix_manifest_sha256": matrix_sha,
        **runtime_control_config_receipts_v3(training_config, route=route),
    }
    stable["dataset"] = {
        name: training_data[name]
        for name in (
            "dataset_id", "dataset_tree_sha256", "dataset_info_sha256",
            "episode_metadata_sha256", "split_manifest_sha256",
            "ordered_window_manifest_sha256",
        )
    }
    stable["hardware"] = {
        name: training_data[name]
        for name in (
            "hardware_schema_id", "hardware_contract_sha256",
            "calibration_capture_sha256",
        )
    }
    stable["normalization"] = {
        "train_only_norm_sha256": training_data["train_only_norm_sha256"]
    }
    stable["precursor"] = {
        "precursor_receipt_bytes_sha256": precursor["receipt_bytes_sha256"],
        "precursor_bundle_sha256": precursor["bundle_sha256"],
        "initial_parameter_tensors_sha256": precursor[
            "parameter_tensor_sha256"
        ],
    }
    stable["labels"].update(
        sidecar_meta_sha256=sidecar_meta_sha,
        sidecar_bundle_sha256=training_data["sidecar_bundle_sha256"],
        contact_coverage_sha256=coverage_sha,
    )
    stable["source_receipts"] = {
        "training_data_receipt_sha256": training_sha,
        "preflight_receipt_sha256": preflight_sha,
        "coverage_receipt_sha256": coverage_sha,
        "current_head_source_receipt_sha256": sha256_json(source_receipt),
    }
    live_model = _TinyCausalCheckpointModel()
    live_optimizer = torch.optim.AdamW(live_model.parameters(), lr=1e-3)
    live_scheduler = torch.optim.lr_scheduler.LambdaLR(
        live_optimizer, lr_lambda=lambda _: 1.0
    )
    live = live_optimizer_scheduler_receipts_v3(
        model=live_model,
        optimizers=[live_optimizer],
        schedulers=[live_scheduler],
    )
    collector = {
        "collector_version": "matched-control-hash-collector/v3",
        "enabled": True,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "causal_route_id": route,
        "seed": 6666,
        "run_id": "synthetic-runtime-binding-run-id",
        "run_nonce": "synthetic-runtime-binding-nonce",
        "output_dir": None,
        "write_report": False,
        "stable_contract": stable,
        "route_local_contract": build_production_route_receipt(route),
        "initial_parameter_tensors_sha256": precursor[
            "parameter_tensor_sha256"
        ],
        **live,
        "precursor_receipt_bytes_sha256": precursor["receipt_bytes_sha256"],
        "current_head_source_tree_sha256": sha256_json(source_receipt),
        "dataset_contact_coverage_receipt_sha256": coverage_sha,
        "all_hashes_equal_across_routes": False,
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }
    kwargs = {
        "training_config": training_config,
        "matrix_manifest_sha256": matrix_sha,
        "training_data_receipt": training_data,
        "training_data_receipt_sha256": training_sha,
        "preflight_receipt_sha256": preflight_sha,
        "coverage_receipt_sha256": coverage_sha,
        "precursor_binding": precursor,
        "sidecar_meta_sha256": sidecar_meta_sha,
        "source_tree_receipt": source_receipt,
        "runtime_environment_sha256": stable["runtime"][
            "runtime_environment_sha256"
        ],
        "dataset_runtime_source_receipt_sha256": stable["runtime"][
            "dataset_runtime_source_receipt_sha256"
        ],
        "t5_runtime_source_receipt_sha256": stable["runtime"][
            "t5_runtime_source_receipt_sha256"
        ],
    }
    validate_collector_runtime_bindings_v3(collector, **kwargs)
    drift = deepcopy(collector)
    drift["stable_contract"]["dataset"]["dataset_tree_sha256"] = _h(
        "binding-drift"
    )
    try:
        validate_collector_runtime_bindings_v3(drift, **kwargs)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("collector live dataset drift escaped recomputation")


def _stable_contract(
    seed: int,
    *,
    head: str,
    step0_sha: str,
    parameter_schema_sha256: str,
    buffer_schema_sha256: str,
    optimizer_group_names_sha256: str,
    optimizer_hyperparameters_sha256: str,
    optimizer_step0_state_sha256: str,
    scheduler_step0_state_sha256: str,
) -> dict[str, Any]:
    return {
        "git_commit": head,
        "source_tree_sha256": _h("source-tree"),
        "matrix_manifest_sha256": _h("matrix-manifest"),
        "precursor_bundle_sha256": _h("precursor-bundle"),
        "base_weights_sha256": step0_sha,
        "initial_parameter_tensor_sha256": step0_sha,
        "persistent_parameter_schema_sha256": parameter_schema_sha256,
        "persistent_buffer_schema_sha256": buffer_schema_sha256,
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
        "state_input_stream_sha256": _h("state-input-stream"),
        "state_mask_stream_sha256": _h("state-mask-stream"),
        "embodiment_id_stream_sha256": _h("embodiment-id-stream"),
        "prompt_t5_stream_sha256": _h("prompt-t5-stream"),
        "runtime_environment_sha256": _h("runtime-environment"),
        "dataset_runtime_source_receipt_sha256": _h("dataset-runtime-source"),
        "t5_runtime_source_receipt_sha256": _h("t5-runtime-source"),
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
        "optimizer_config_sha256": optimizer_hyperparameters_sha256,
        "optimizer_parameter_groups_sha256": optimizer_group_names_sha256,
        "optimizer_step0_state_sha256": optimizer_step0_state_sha256,
        "scheduler_config_sha256": _h("scheduler-config"),
        "scheduler_step0_state_sha256": scheduler_step0_state_sha256,
        "precision": "bfloat16",
        "world_size": 1,
        "rank_set": [0],
        "window_seed": seed + 10,
        "noise_seed": seed + 20,
        "training_seed": seed,
    }


def _make_receipt(
    *,
    seed: int,
    route: str,
    step: int,
    head: str,
    step0_sha: str,
    checkpoint_path: Path,
    parent_sha: str,
    parameter_topology: dict[str, Any],
    gradient_reachability_receipt: dict[str, Any],
    step0_optimizer_group_names_sha256: str,
    step0_optimizer_hyperparameters_sha256: str,
    step0_optimizer_state_sha256: str,
    step0_scheduler_state_sha256: str,
) -> dict[str, Any]:
    route_receipt = build_production_route_receipt(route)
    direct = route == "aux_only_direct_q12"
    run_label = f"seed{seed}-{route}"
    inspected = inspect_causal_checkpoint_bundle(checkpoint_path.parent)
    checkpoint_sha = inspected["weight_file_sha256"]
    parameter_sha256 = parameter_topology["parameter_tensor_sha256"]
    buffer_sha256 = parameter_topology["buffer_tensor_sha256"]
    trace_prefix_unsigned = {
        "receipt_version": "contact-causal-current-run-trace-prefix/v1",
        "causal_route_id": route,
        "seed": seed,
        "run_id": f"training-run-id-{run_label}",
        "run_nonce": f"training-run-nonce-{run_label}",
        "rank": 0,
        "world_size": 1,
        "through_checkpoint_step": step,
        "record_count": 1,
        "record_sha256_by_key": {
            f"step{step}/micro0/rank0": _h(
                f"synthetic-current-trace-{seed}-{route}-{step}"
            )
        },
        "trace_path_record_only": str(
            (checkpoint_path.parent / "synthetic-current-run-trace.jsonl").absolute()
        ),
    }
    trace_prefix = {
        **trace_prefix_unsigned,
        "receipt_sha256": sha256_json(trace_prefix_unsigned),
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
        "stable_contract": _stable_contract(
            seed,
            head=head,
            step0_sha=step0_sha,
            parameter_schema_sha256=parameter_topology[
                "persistent_parameter_schema_sha256"
            ],
            buffer_schema_sha256=parameter_topology[
                "persistent_buffer_schema_sha256"
            ],
            optimizer_group_names_sha256=step0_optimizer_group_names_sha256,
            optimizer_hyperparameters_sha256=(
                step0_optimizer_hyperparameters_sha256
            ),
            optimizer_step0_state_sha256=step0_optimizer_state_sha256,
            scheduler_step0_state_sha256=step0_scheduler_state_sha256,
        ),
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
            "hardened_preflight_receipt_sha256": _h(f"hardened-preflight-{seed}"),
            "training_data_receipt_sha256": _h(f"training-data-{seed}"),
            "matched_control_receipt_sha256": _h(f"matched-control-{seed}"),
            "external_reviewer_acceptance_sha256": _h(f"gate-review-{seed}"),
            "contact_coverage_receipt_sha256": _h("contact-coverage"),
            "trace_manifest_sha256": _h("trace-manifest"),
            "current_head_source_receipt_sha256": _h("source-tree"),
        },
        "topology": {"world_size": 1, "rank_set": [0]},
        "parameter_topology": parameter_topology,
        "diagnostic_not_admission": {
            "admission_role": "diagnostic_not_admission",
            "forward_flops_proxy": 100.0 + CAUSAL_ROUTES.index(route),
            "backward_flops_proxy": 200.0 + CAUSAL_ROUTES.index(route),
            "peak_memory_bytes": 4096 + CAUSAL_ROUTES.index(route),
            "step_time_ms": 10.0 + CAUSAL_ROUTES.index(route),
            "action_only_latency_ms": 2.0 + CAUSAL_ROUTES.index(route),
            "transport_flops_proxy": 5.0 + CAUSAL_ROUTES.index(route),
        },
        "gradient_reachability_receipt": gradient_reachability_receipt,
        "checkpoint_artifact_hashes": {
            "weight_tensor_sha256": parameter_sha256,
            "buffer_tensor_sha256": buffer_sha256,
            "config_sha256": inspected["config_sha256"],
            "whole_bundle_sha256": inspected["bundle_manifest"]["bundle_sha256"],
            "route_receipt_sha256": route_receipt["receipt_sha256"],
        },
        "checkpoint_file": {
            "path_record_only": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "serialization": inspected["tensor_manifest"]["serialization"],
            "tensor_manifest": inspected["tensor_manifest"],
            "tensor_manifest_sha256": inspected["tensor_manifest_sha256"],
            "bundle_manifest": inspected["bundle_manifest"],
            "current_run_trace_prefix_receipt": trace_prefix,
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
    fixture_root = Path(receipt["checkpoint_file"]["path_record_only"]).parent
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
        "checkpoint_tensor_manifest_sha256": receipt["checkpoint_file"][
            "tensor_manifest_sha256"
        ],
        "checkpoint_path_record_only": receipt["checkpoint_file"][
            "path_record_only"
        ],
        "robot_gate_path_record_only": str(fixture_root / "robot-gate.json"),
        "robot_gate_sha256": receipt["evidence"]["training_gate_sha256"],
        "gate_reviewer_acceptance_path_record_only": str(
            fixture_root / "gate-review.json"
        ),
        "gate_reviewer_acceptance_sha256": receipt["evidence"][
            "external_reviewer_acceptance_sha256"
        ],
        "route_training_acceptance_path_record_only": str(
            fixture_root / "route-acceptance.json"
        ),
        "route_training_acceptance_sha256": receipt["run_identity"][
            "external_acceptance_sha256"
        ],
        "precursor_receipt_path_record_only": str(fixture_root / "precursor.json"),
        "precursor_receipt_bytes_sha256": receipt["precursor"]["receipt_bytes_sha256"],
        "precursor_checkpoint_dir_record_only": str(fixture_root / "precursor"),
        "precursor_bundle_sha256": receipt["precursor"]["bundle_sha256"],
        "hardened_preflight_path_record_only": str(fixture_root / "hardened.json"),
        "hardened_preflight_receipt_sha256": receipt["evidence"][
            "hardened_preflight_receipt_sha256"
        ],
        "production_preflight_path_record_only": str(fixture_root / "preflight.json"),
        "production_preflight_receipt_sha256": receipt["evidence"][
            "production_preflight_receipt_sha256"
        ],
        "training_data_path_record_only": str(fixture_root / "training-data.json"),
        "training_data_receipt_sha256": receipt["evidence"][
            "training_data_receipt_sha256"
        ],
        "coverage_path_record_only": str(fixture_root / "coverage.json"),
        "contact_coverage_receipt_sha256": receipt["evidence"][
            "contact_coverage_receipt_sha256"
        ],
        "sidecar_dir_record_only": str(fixture_root / "sidecars"),
        "t5_embedding_dir_record_only": str(fixture_root / "t5_embedding"),
        "matched_control_report_path_record_only": str(
            fixture_root / "matched-control-report.json"
        ),
        "matched_control_report_sha256": receipt["evidence"][
            "matched_control_receipt_sha256"
        ],
        "matched_control_trace_manifest_sha256": receipt["evidence"][
            "trace_manifest_sha256"
        ],
        "matched_control_trace_path_by_route": {
            item: str(fixture_root / f"{item}.jsonl") for item in CAUSAL_ROUTES
        },
        "current_run_trace_prefix_receipt_sha256": receipt["checkpoint_file"][
            "current_run_trace_prefix_receipt"
        ]["receipt_sha256"],
        "current_head_source_receipt_sha256": receipt["evidence"][
            "current_head_source_receipt_sha256"
        ],
        "reviewer_identity": f"reviewer-identity-{label}",
        "reviewer_attestation_nonce": f"reviewer-attestation-{label}",
        "acceptance_run_id": f"acceptance-run-id-{label}",
        "acceptance_run_nonce": f"acceptance-run-nonce-{label}",
        "reviewed_current_head_sha256": receipt["stable_contract"]["git_commit"],
        "reviewed_source_tree_sha256": receipt["stable_contract"]["source_tree_sha256"],
        "reviewed_at_utc": "2026-08-11T00:00:00Z",
        "evidence_valid_until_utc": "2099-12-31T23:59:59Z",
        "oracle_used": False,
    }


def _route_outputs(workspace: Path) -> None:
    _validate_finite_metric_tree(
        {
            "grammar_validity": {
                "query_count": 12 * 512,
                "validity": 1.0,
            },
            "field_accuracy": {
                "supervised_query_count": 128,
                "accuracy": 0.5,
            },
        },
        path="contact_aux_metrics",
    )
    for invalid_metric in (
        {"grammar_validity": {"query_count": -1}},
        {"grammar_validity": {"query_count": 1.5}},
        {"grammar_validity": {"validity": 1.00001}},
        {"field_accuracy": {"accuracy": float("nan")}},
    ):
        try:
            _validate_finite_metric_tree(
                invalid_metric, path="contact_aux_metrics"
            )
        except CausalProductionError:
            pass
        else:
            raise AssertionError("causal metric tree accepted invalid numeric domain")

    target_free = {key: object() for key in INFERENCE_PACKET_KEYS}
    target_free["window_id"] = "synthetic-target-free-window"
    validate_target_free_inference_packet(target_free)
    for target_key in FORBIDDEN_INFERENCE_TARGET_KEYS:
        leaked = dict(target_free)
        leaked[target_key] = object()
        try:
            validate_target_free_inference_packet(leaked)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"target-free inference accepted target key {target_key}"
            )

    equal_output = None
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
        if route == "equal_bit_unstructured_hard_st":
            equal_output = deepcopy(output)
    leaked = deepcopy(output)
    leaked["contact_target_model_input"] = True
    try:
        validate_route_evaluator_output(leaked, route="hard_st_semantic")
    except CausalProductionError:
        pass
    else:
        raise AssertionError("causal evaluator accepted GT leakage")
    if equal_output is None:
        raise AssertionError("equal-bit evaluator output was not exercised")
    for invalid_value in (float("nan"), 1.00001):
        invalid = deepcopy(equal_output)
        invalid["packet_metrics"]["codec_validity"] = invalid_value
        try:
            validate_route_evaluator_output(
                invalid, route="equal_bit_unstructured_hard_st"
            )
        except CausalProductionError:
            pass
        else:
            raise AssertionError("causal evaluator accepted invalid codec metric")

    row_count = 3 * 2 * 256
    steps = np.repeat(np.asarray(CHECKPOINT_STEPS, dtype=np.int64), 512)
    window_ids = [f"window-{index % 512:03d}" for index in range(row_count)]
    actions = np.zeros((row_count, 48, 32), dtype=np.float32)
    action_targets = np.zeros_like(actions)
    action_dim_masks = np.zeros((row_count, 32), dtype=np.bool_)
    action_dim_masks[:, :25] = True
    action_time_masks = np.ones((row_count, 48), dtype=np.bool_)
    action_denorm_scales = np.ones((row_count, 32), dtype=np.float32)
    logits = np.zeros((row_count, 12, 22), dtype=np.float32)
    hard = np.zeros_like(logits)
    contact_targets = np.zeros_like(logits)
    contact_masks = np.zeros_like(logits)
    contact_confidences = np.zeros_like(logits)
    contact_condition_masks = np.zeros_like(logits)
    reduced = {
        split: {
            "segment_exact": {"hard_b16_fully_supervised_accuracy": 0.5}
        }
        for split in ("val", "test")
    }
    for route in CAUSAL_ROUTES:
        packets = (
            np.empty((row_count, 0, 22), dtype=np.float32)
            if route == "aux_only_direct_q12"
            else np.zeros((row_count, 12, 22), dtype=np.float32)
        )
        path = workspace / f"route-output-{route}.npz"
        digest = write_route_output_evidence(
            path,
            route=route,
            checkpoint_steps=steps,
            window_ids=window_ids,
            action_predictions=actions,
            action_targets=action_targets,
            action_dim_masks=action_dim_masks,
            action_time_masks=action_time_masks,
            action_denorm_scales=action_denorm_scales,
            semantic_aux_logits=logits,
            contact_hard_semantics=hard,
            contact_targets=contact_targets,
            contact_loss_masks=contact_masks,
            contact_confidences=contact_confidences,
            contact_condition_masks=contact_condition_masks,
            route_packets=packets,
        )
        entry = {
            "path_record_only": str(path.absolute()),
            "sha256": digest,
            "row_count": row_count,
            "format": ROUTE_OUTPUT_EVIDENCE_FORMAT,
        }
        replayed = _load_route_output_evidence(entry, route=route)
        first = replayed["checkpoint_steps"] == 0
        first_ids = replayed["window_ids"][first].tolist()
        replayed_action_sha = _array_stream_sha256(
            first_ids, replayed["action_predictions"][first]
        )
        if len(replayed_action_sha) != 64:
            raise AssertionError("CPU replayed action digest is invalid")
        metrics = _recompute_route_packet_metrics(
            route=route,
            logits=replayed["semantic_aux_logits"][first],
            packets=replayed["route_packets"][first],
            reduced_contact_by_split=reduced,
        )
        if route == "equal_bit_unstructured_hard_st" and (
            metrics["codec_validity"] != 1.0
            or metrics["invalid_to_zero_rate"] != 1.0
            or not 0.0 <= metrics["utilization"] <= 1.0
            or not math.isfinite(metrics["entropy"])
        ):
            raise AssertionError("equal-bit raw metric replay differs")
        if route == "equal_bit_unstructured_hard_st":
            packet_tamper = replayed["route_packets"][first].copy()
            packet_tamper[0, 0, 0] = 1.0
            try:
                _recompute_route_packet_metrics(
                    route=route,
                    logits=replayed["semantic_aux_logits"][first],
                    packets=packet_tamper,
                    reduced_contact_by_split=reduced,
                )
            except CausalProductionError:
                pass
            else:
                raise AssertionError("equal-bit packet/logit mismatch was accepted")
        if route == "hard_st_semantic" and (
            metrics["grammar_validity"] != 1.0
            or metrics["field_accuracy"] != 0.5
        ):
            raise AssertionError("semantic raw metric replay differs")
        tampered = dict(entry)
        tampered["sha256"] = _h(f"tampered-{route}")
        try:
            _load_route_output_evidence(tampered, route=route)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("tampered route-output evidence was accepted")


def _arbitrary_hex_aggregate_negative(
    *, workspace: Path, evidence: dict, matrix_contract: dict
) -> None:
    result_entries = {
        f"{seed}/{route}": {
            "path_record_only": str(
                (workspace / "nonexistent-results" / f"{seed}-{route}.json").absolute()
            ),
            "sha256": _h(f"arbitrary-result-{seed}-{route}"),
            "per_window_csv_sha256": _h(f"arbitrary-csv-{seed}-{route}"),
            "route_output_evidence_sha256": _h(
                f"arbitrary-route-output-{seed}-{route}"
            ),
        }
        for seed in SEEDS
        for route in CAUSAL_ROUTES
    }
    metrics = {
        f"{route}/{step}/{split}": {
            "action": {"synthetic": 0.0},
            "contact_aux": {"synthetic": 0.0},
        }
        for route in CAUSAL_ROUTES
        for step in CHECKPOINT_STEPS
        for split in ("val", "test")
    }
    aggregate = {
        "aggregate_version": "contact-asym25-heldout-aggregate/v5",
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "seeds": list(SEEDS),
        "causal_routes": list(CAUSAL_ROUTES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_acceptance_count": 27,
        "result_file_by_seed_route": result_entries,
        "aggregate_metrics_by_route_step_split": metrics,
        "matrix_contract": matrix_contract,
        "primary_checkpoint_step": 200,
        "oracle_included": False,
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }
    try:
        validate_aggregate_v5(aggregate, evidence=evidence, verify_files=True)
    except (CausalProductionError, OSError):
        return
    raise AssertionError("arbitrary SHA-256 strings authorized causal aggregate")


def _full_evaluator_acceptance_subprocess(
    *, workspace: Path, root: Path, evidence: dict
) -> dict[str, Any]:
    input_path = workspace / "full-evaluator-input.json"
    output_path = workspace / "full-evaluator-report.json"
    serialized = {
        "workspace": str(workspace / "full-evaluator-workspace"),
        "evidence": [
            {
                "seed": seed,
                "route": route,
                "step": step,
                "receipt": evidence[(seed, route, step)][0],
                "acceptance": evidence[(seed, route, step)][1],
            }
            for seed in SEEDS
            for route in CAUSAL_ROUTES
            for step in CHECKPOINT_STEPS
        ],
    }
    input_path.write_bytes(
        json.dumps(
            serialized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
            "conda",
            "run",
            "-n",
            "labvla",
            "python",
            "-B",
            "-m",
            "world_action_model.contact_causal_eval_synthetic_fixture",
            "--serialized-full-acceptance",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--repository-root",
            str(root),
        ]
    try:
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise AssertionError(
            f"full evaluator labvla subprocess failed: {detail}"
        ) from exc
    report, _, _ = strict_json_file(
        output_path, name="full evaluator CPU acceptance"
    )
    if report != {
        **report,
        "acceptance_version": "contact-causal-full-evaluator-cpu-acceptance/v1",
        "status": "passed",
        "result_count": 9,
        "checkpoint_acceptance_count": 27,
        "verify_files": True,
        "bound_acceptance_replay_count": 27,
        "tampered_route_output_rejected": True,
    }:
        raise AssertionError("full evaluator subprocess report differs")
    return report


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
        head = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
        _actual_tiny_model_topology_and_precursor(workspace, head)
        trainer_first_batch_report = _trainer_first_batch_three_route_acceptance()
        _zero_safe_supervision()
        collector_report_sha256 = _collector_v3_acceptance()
        _collector_runtime_binding_acceptance(root, head)
        _route_outputs(workspace)
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
        step0_mapping, step0_topology, _, step0_sha = _tiny_training_state(
            route=CAUSAL_ROUTES[0], step=0
        )
        step0_payload = _tiny_weight_payload(step0_mapping)
        evidence = {}
        for seed in SEEDS:
            for route in CAUSAL_ROUTES:
                parent = step0_sha
                for step in CHECKPOINT_STEPS:
                    mapping, parameter_topology, reachability, initial_sha = (
                        _tiny_training_state(route=route, step=step)
                    )
                    if initial_sha != step0_sha:
                        raise AssertionError("tiny initial parameter bytes drifted")
                    payload = step0_payload if step == 0 else _tiny_weight_payload(mapping)
                    checkpoint_path = (
                        workspace
                        / f"seed{seed}"
                        / route
                        / f"step{step}"
                        / "diffusion_pytorch_model.bin"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    config = {
                        "_class_name": "SyntheticCausalCheckpoint",
                        **checkpoint_transformer_semantic_config(route),
                    }
                    (checkpoint_path.parent / "config.json").write_bytes(
                        json.dumps(
                            config,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                    checkpoint_path.write_bytes(payload)
                    loaded, serialization = load_safe_weight_mapping(checkpoint_path)
                    if serialization != "pytorch_weights_only" or set(loaded) != set(mapping):
                        raise AssertionError("tiny checkpoint did not safely reload")
                    receipt = _make_receipt(
                        seed=seed,
                        route=route,
                        step=step,
                        head=head,
                        step0_sha=step0_sha,
                        checkpoint_path=checkpoint_path,
                        parent_sha=(step0_sha if step == 0 else parent),
                        parameter_topology=parameter_topology,
                        gradient_reachability_receipt=reachability,
                        step0_optimizer_group_names_sha256=step0_topology[
                            "optimizer_group_names_sha256"
                        ],
                        step0_optimizer_hyperparameters_sha256=step0_topology[
                            "optimizer_hyperparameters_sha256"
                        ],
                        step0_optimizer_state_sha256=step0_topology[
                            "optimizer_state_sha256"
                        ],
                        step0_scheduler_state_sha256=step0_topology[
                            "scheduler_state_sha256"
                        ],
                    )
                    acceptance = _make_acceptance(receipt)
                    validate_training_receipt_v5(receipt)
                    validate_acceptance_v3(acceptance, receipt=receipt)
                    evidence[(seed, route, step)] = (receipt, acceptance)
                    parent = parameter_topology["parameter_tensor_sha256"]
        report = validate_27_checkpoint_matrix(evidence, verify_files=True)
        full_evaluator_report = _full_evaluator_acceptance_subprocess(
            workspace=workspace, root=root, evidence=evidence
        )
        _arbitrary_hex_aggregate_negative(
            workspace=workspace, evidence=evidence, matrix_contract=report
        )
        _safe_checkpoint_negative_acceptance(
            workspace, evidence[(6666, "hard_st_semantic", 0)][0]
        )
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
        step_identity_drift = deepcopy(evidence)
        drift_receipt, drift_acceptance = step_identity_drift[
            (6666, "hard_st_semantic", 100)
        ]
        drift_receipt = deepcopy(drift_receipt)
        drift_receipt["run_identity"]["training_run_id"] = (
            "training-run-id-step-drift-hard-st-semantic"
        )
        drift_acceptance = _make_acceptance(drift_receipt)
        step_identity_drift[(6666, "hard_st_semantic", 100)] = (
            drift_receipt,
            drift_acceptance,
        )
        try:
            validate_27_checkpoint_matrix(step_identity_drift, verify_files=False)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("training run identity drifted across checkpoint steps")
        reviewer_replay = deepcopy(evidence)
        replay_receipt, replay_acceptance = reviewer_replay[
            (6666, "hard_st_semantic", 100)
        ]
        replay_acceptance = deepcopy(replay_acceptance)
        source_acceptance = evidence[(6666, "hard_st_semantic", 0)][1]
        for name in (
            "reviewer_identity",
            "reviewer_attestation_nonce",
            "acceptance_run_id",
            "acceptance_run_nonce",
        ):
            replay_acceptance[name] = source_acceptance[name]
        reviewer_replay[(6666, "hard_st_semantic", 100)] = (
            replay_receipt,
            replay_acceptance,
        )
        try:
            validate_27_checkpoint_matrix(reviewer_replay, verify_files=False)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("reviewer identity/nonce replay escaped matrix validation")
        direct_allowlist_drift = deepcopy(
            evidence[(6666, "aux_only_direct_q12", 0)][0]
        )
        direct_allowlist_drift["parameter_topology"]["unused_parameter_names"] = [
            "contact_condition_encoder.weight"
        ]
        direct_allowlist_drift["parameter_topology"][
            "unused_parameter_names_sha256"
        ] = sha256_json(["contact_condition_encoder.weight"])
        try:
            validate_training_receipt_v5(direct_allowlist_drift)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("direct unused-parameter subset was accepted")
        cache_policy_drift = deepcopy(
            evidence[(6666, "hard_st_semantic", 0)][0]
        )
        cache_policy_drift["route"]["cache_policy"] = "forged-cache-policy"
        try:
            validate_training_receipt_v5(cache_policy_drift)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("forged route cache policy was accepted")
        optimizer_group_omission = deepcopy(
            evidence[(6666, "hard_st_semantic", 0)][0]
        )
        groups = optimizer_group_omission["parameter_topology"][
            "optimizer_group_parameter_names"
        ]
        groups[0].pop()
        optimizer_group_omission["parameter_topology"][
            "optimizer_group_names_sha256"
        ] = sha256_json(groups)
        try:
            validate_training_receipt_v5(optimizer_group_omission)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("optimizer parameter omission was accepted")
        optimizer_state_omission = deepcopy(
            evidence[(6666, "hard_st_semantic", 100)][0]
        )
        topology = optimizer_state_omission["parameter_topology"]
        omitted_name = topology["optimizer_state_parameter_names"].pop()
        topology["optimizer_state_receipt"] = [
            row
            for row in topology["optimizer_state_receipt"]
            if row["parameter_name"] != omitted_name
        ]
        topology["optimizer_state_names_sha256"] = sha256_json(
            topology["optimizer_state_parameter_names"]
        )
        topology["optimizer_state_sha256"] = sha256_json(
            topology["optimizer_state_receipt"]
        )
        try:
            validate_training_receipt_v5(optimizer_state_omission)
        except CausalProductionError:
            pass
        else:
            raise AssertionError("active optimizer state omission was accepted")
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
            "trainer_first_batch_three_route": trainer_first_batch_report,
            "synthetic_checkpoint_file_count": 27,
            "synthetic_checkpoint_files_nonempty": True,
            "synthetic_checkpoint_files_are_loadable_torch_weights": True,
            "matrix_report": report,
            "full_evaluator_report": full_evaluator_report,
            "negative_cases": [
                "strict_json_duplicate_nan_inf_overflow",
                "false_null_templates",
                "old_new_schema_mix",
                "legacy_route_alias",
                "stable_optimizer_drift",
                "precursor_receipt_byte_drift",
                "safe_weight_random_nested_readme_mixed_layout",
                "per_tensor_manifest_digest_drift",
                "training_run_identity_cross_step_drift",
                "reviewer_nonce_replay",
                "direct_unused_parameter_subset",
                "route_cache_policy_drift",
                "optimizer_parameter_group_omission",
                "active_optimizer_state_omission",
                "boolean_integer_alias",
                "GT_oracle_evaluator_leakage",
                "target_key_in_exact_inference_packet",
                "route_output_evidence_sha_tamper",
                "equal_bit_packet_logits_codec_tamper",
                "nonfinite_or_out_of_range_evaluator_metrics",
                "arbitrary_hex_aggregate_artifact_replay",
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

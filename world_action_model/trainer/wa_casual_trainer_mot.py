import functools
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from giga_models import utils as gm_utils
from world_action_model.contact_transition_schema import (
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.contact_route_contract import (
    ContactCausalRoute,
    validate_spec_only_trainer_route_config,
)
from world_action_model.contact_supervision_contract import (
    CONTACT_GROUPED_SUPERVISION_RULE_VERSION,
    CONTACT_GROUPED_SUPERVISION_ZERO_SAFE_VERSION,
    canonical_zero_contact_loss,
    grouped_contact_supervision_weights,
    validate_positive_contact_supervision,
    validate_zero_safe_contact_supervision,
)
from world_action_model.matched_control_hashes import MatchedControlCollector
from world_action_model.models import CasualWorldActionTransformer_MoT
from world_action_model.trainer.wa_casual_trainer_pretrain import (
    _as_dim_mask,
    _as_time_mask,
    masked_mse,
)
from giga_train import ModuleDict, Trainer


def masked_contact_loss(
    prediction,
    target,
    mask=None,
    confidence=None,
    loss_type="bce",
):
    """Confidence-weighted loss for ordered contact-event query targets."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"contact target shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if loss_type == "bce":
        element_loss = F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )
    elif loss_type == "smooth_l1":
        element_loss = F.smooth_l1_loss(
            prediction.float(), target.float(), reduction="none"
        )
    else:
        raise ValueError(
            f"unsupported contact_loss_type={loss_type!r}; expected 'bce' or 'smooth_l1'"
        )

    weight = torch.ones_like(element_loss)
    for name, value in (("contact_loss_mask", mask), ("contact_confidence", confidence)):
        if value is None:
            continue
        value = value.to(device=prediction.device, dtype=element_loss.dtype)
        while value.ndim < element_loss.ndim:
            value = value.unsqueeze(-1)
        try:
            value = torch.broadcast_to(value, element_loss.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} cannot broadcast to "
                f"contact prediction {tuple(element_loss.shape)}"
            ) from exc
        weight = weight * value

    denominator = weight.sum().clamp_min(1.0)
    return (element_loss * weight).sum() / denominator


def apply_contact_query_assignment(
    prediction,
    target,
    mask,
    confidence,
    matcher=None,
):
    """Return query-aligned targets, leaving a clean Hungarian hook.

    v1.1 sidecars currently use deterministic segment ordering.  A future
    Hungarian matcher can be supplied as a callable with this same signature
    and must return ``(target, mask, confidence)`` aligned to prediction query
    order.  Keeping assignment outside the grouped loss avoids baking an
    ordering assumption into the loss implementation.
    """

    if matcher is None:
        return target, mask, confidence
    aligned = matcher(prediction, target, mask, confidence)
    if not isinstance(aligned, tuple) or len(aligned) != 3:
        raise TypeError("contact matcher must return (target, mask, confidence)")
    return aligned


def grouped_contact_segment_loss(
    prediction,
    target,
    mask=None,
    confidence=None,
    matcher=None,
):
    """Field-balanced v1.1 BCE/CE/Smooth-L1 loss on raw planner logits."""

    if prediction.shape != target.shape or prediction.shape[-1] != SEGMENT_TARGET_DIM:
        raise ValueError(
            "v1.1 contact shape mismatch: "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}, "
            f"expected last dim {SEGMENT_TARGET_DIM}"
        )
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if mask is None:
        mask = torch.ones_like(target)
    else:
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)
        mask = torch.broadcast_to(mask, target.shape)
    if confidence is None:
        confidence = torch.ones_like(target)
    else:
        confidence = confidence.to(device=prediction.device, dtype=prediction.dtype)
        confidence = torch.broadcast_to(confidence, target.shape)

    target, mask, confidence = apply_contact_query_assignment(
        prediction, target, mask, confidence, matcher
    )
    supervision_weights = grouped_contact_supervision_weights(mask, confidence)
    components = []
    active_components = []

    def weighted_mean(values, weights):
        denominator = weights.sum()
        value = (values * weights).sum() / denominator.clamp_min(1.0)
        return value, (denominator > 0).to(dtype=value.dtype)

    for field in SEGMENT_BINARY_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        values = F.binary_cross_entropy_with_logits(
            prediction[..., field_slice].float(),
            target[..., field_slice].float(),
            reduction="none",
        )
        weights = supervision_weights[field]
        component, active = weighted_mean(values, weights)
        components.append(component)
        active_components.append(active)

    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        field_target = target[..., field_slice].float()
        # Categorical supervision is valid only when the complete one-hot
        # group is known.  Partial category masks are treated as unknown.
        field_weight = supervision_weights[field]
        values = -(field_target * F.log_softmax(
            prediction[..., field_slice].float(), dim=-1
        )).sum(dim=-1)
        component, active = weighted_mean(values, field_weight)
        components.append(component)
        active_components.append(active)

    for field in SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        values = F.smooth_l1_loss(
            prediction[..., field_slice].float().sigmoid(),
            target[..., field_slice].float(),
            reduction="none",
        )
        weights = supervision_weights[field]
        component, active = weighted_mean(values, weights)
        components.append(component)
        active_components.append(active)

    values = torch.stack(components)
    active = torch.stack(active_components)
    return (values * active).sum() / active.sum().clamp_min(1.0)


def _as_action_loss_weight(
    weight: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize per-dimension action weights to a broadcastable [B,T,D]."""

    weight = torch.as_tensor(weight, device=device)
    if weight.ndim == 1:
        weight = weight[None, None, :]
    elif weight.ndim == 2:
        if tuple(weight.shape) == (seq_len, dim):
            weight = weight[None, :, :]
        elif tuple(weight.shape) == (batch_size, dim):
            weight = weight[:, None, :]
        else:
            raise ValueError(
                "rank-2 action_loss_weight must be [T,D] or [B,D], got "
                f"{tuple(weight.shape)}"
            )
    elif weight.ndim != 3:
        raise ValueError(
            f"action_loss_weight must have 1/2/3 dims, got {tuple(weight.shape)}"
        )
    expected = (batch_size, seq_len, dim)
    try:
        return torch.broadcast_to(weight, expected)
    except RuntimeError as exc:
        raise ValueError(
            f"action_loss_weight shape {tuple(weight.shape)} cannot broadcast to {expected}"
        ) from exc


def masked_weighted_action_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim_mask: torch.Tensor | None,
    time_mask: torch.Tensor | None,
    action_loss_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve ``masked_mse`` exactly unless explicit dimension weights exist."""

    if action_loss_weight is None:
        return masked_mse(pred, target, dim_mask=dim_mask, time_mask=time_mask)

    sq = (pred - target).pow(2)
    weights = _as_action_loss_weight(
        action_loss_weight,
        batch_size=pred.shape[0],
        seq_len=pred.shape[1],
        dim=pred.shape[2],
        device=pred.device,
    ).to(dtype=sq.dtype)
    if dim_mask is not None:
        weights = weights * dim_mask.to(dtype=sq.dtype, device=sq.device)

    # Normalize dimensions independently per token, matching masked_mse when
    # all explicit weights are one while allowing the ego grip channels to be
    # downweighted to 0.25.
    per_token = (sq * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    if time_mask is None:
        valid_token = (weights.sum(dim=-1) > 0).to(dtype=per_token.dtype)
        return (per_token * valid_token).sum() / valid_token.sum().clamp_min(1.0)
    time_mask = time_mask.to(dtype=per_token.dtype, device=per_token.device)
    valid_token = (weights.sum(dim=-1) > 0).to(dtype=per_token.dtype)
    time_weight = time_mask * valid_token
    return (per_token * time_weight).sum() / time_weight.sum().clamp_min(1.0)


class CasualWATrainerMoT(Trainer):
    """WAM pretraining trainer for the Mixture-of-Transformers policy."""

    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        self.visual_flow_shift = float(model_config.visual_flow_shift)
        self.action_flow_shift = float(model_config.action_flow_shift)
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.num_embodiments = int(model_config.get("num_embodiments", 1))
        self.contact_loss_weight = float(model_config.get("contact_loss_weight", 0.0))
        self.contact_loss_type = str(model_config.get("contact_loss_type", "bce"))
        self.contact_schema_version = str(
            model_config.get("contact_schema_version", "contact_transition_v1")
        )
        self.contact_condition_mode = str(
            model_config.get("contact_condition_mode", "predicted")
        )
        self.require_contact_targets = bool(
            model_config.get("require_contact_targets", False)
        )
        self.require_positive_contact_supervision = bool(
            model_config.get("require_positive_contact_supervision", False)
        )
        self.contact_supervision_rule_version = str(
            model_config.get(
                "contact_supervision_rule_version",
                CONTACT_GROUPED_SUPERVISION_RULE_VERSION,
            )
        )
        self.production_matrix_id = str(
            model_config.get("production_matrix_id", "") or ""
        )
        self.production_causal_route_id = str(
            model_config.get("production_causal_route_id", "") or ""
        )
        self.ego_contact_precursor_objective = model_config.get(
            "ego_contact_precursor_objective"
        )
        self.ego_contact_precursor_evidence_base = model_config.get(
            "ego_contact_precursor_evidence_base"
        )
        transformer_config = model_config.get("transformer", {})
        self.contact_causal_route = str(
            model_config.get(
                "contact_causal_route",
                transformer_config.get("contact_causal_route", ""),
            )
            or ""
        )
        if self.production_matrix_id:
            from world_action_model.contact_causal_production import (
                MATRIX_ID,
                SUPERVISION_VERSION,
                ensure_exact_causal_route,
            )

            if self.production_matrix_id != MATRIX_ID:
                raise ValueError("production causal matrix ID differs")
            selected_causal_route = ensure_exact_causal_route(
                self.production_causal_route_id
            )
            if self.contact_causal_route != selected_causal_route:
                raise ValueError("production causal route differs from model route")
            if self.contact_supervision_rule_version != SUPERVISION_VERSION:
                raise ValueError("production causal route requires zero-safe supervision v2")
            if self.require_positive_contact_supervision:
                raise ValueError(
                    "zero-safe production routes must not require positive supervision per batch"
                )
            resolved_spec_route = None
        else:
            resolved_spec_route = validate_spec_only_trainer_route_config(model_config)
        if resolved_spec_route is not None:
            self.contact_causal_route = resolved_spec_route.value
        self.current_training_data_receipt = model_config.get(
            "current_training_data_receipt"
        )
        self.current_training_data_receipt_sha256 = model_config.get(
            "current_training_data_receipt_sha256"
        )
        self.current_training_source_snapshot = model_config.get(
            "current_training_source_snapshot"
        )
        self.production_checkpoint_steps = tuple(
            int(value) for value in model_config.get("production_checkpoint_steps", [])
        )
        self.production_training_run_nonce = model_config.get(
            "production_training_run_nonce"
        )
        self.production_training_run_id = model_config.get(
            "production_training_run_id"
        )
        self.production_experiment_family_id = model_config.get(
            "production_experiment_family_id"
        )
        self.production_variant_id = model_config.get("production_variant_id")
        self.production_transport_variant_id = model_config.get(
            "production_transport_variant_id"
        )
        self.production_matched_control_receipt_sha256 = model_config.get(
            "production_matched_control_receipt_sha256"
        )
        self.production_training_precision = model_config.get(
            "production_training_precision"
        )
        self.production_action_schema_contract = model_config.get(
            "production_action_schema_contract"
        )
        self.production_run_contract = model_config.get("production_run_contract")
        self.production_run_contract_sha256 = model_config.get(
            "production_run_contract_sha256"
        )
        self.production_runtime_environment_receipt = model_config.get(
            "production_runtime_environment_receipt"
        )
        self.production_optimizer_config_sha256 = model_config.get(
            "production_optimizer_config_sha256"
        )
        self.production_scheduler_config_sha256 = model_config.get(
            "production_scheduler_config_sha256"
        )
        self.production_training_config_sha256 = model_config.get(
            "production_training_config_sha256"
        )
        self.production_train_window_receipt_sha256 = model_config.get(
            "production_train_window_receipt_sha256"
        )
        self.production_base_model_bundle_sha256 = model_config.get(
            "production_base_model_bundle_sha256"
        )
        self.causal_training_data_receipt = model_config.get(
            "causal_training_data_receipt"
        )
        self.causal_training_data_receipt_sha256 = model_config.get(
            "causal_training_data_receipt_sha256"
        )
        self.causal_preflight_receipt = model_config.get(
            "causal_preflight_receipt"
        )
        self.causal_preflight_receipt_sha256 = model_config.get(
            "causal_preflight_receipt_sha256"
        )
        self.causal_precursor_binding = model_config.get(
            "causal_precursor_binding"
        )
        self.causal_checkpoint_receipt_base = model_config.get(
            "causal_checkpoint_receipt_base"
        )
        self._causal_checkpoint_weight_sha256 = {}
        self._causal_initial_parameter_sha256 = None
        matched_control_config = model_config.get("matched_control_collector")
        self.production_matched_control_external_acceptance_sha256 = (
            matched_control_config.get("external_acceptance_sha256")
            if isinstance(matched_control_config, dict)
            else None
        )
        self._production_checkpoint_bundle_sha256 = {}
        self._production_initial_weights_snapshot = None
        if (self.current_training_data_receipt is None) != (
            self.current_training_data_receipt_sha256 is None
        ):
            raise ValueError(
                "production checkpoint binding requires both current receipt and SHA"
            )
        causal_values = (
            self.causal_training_data_receipt,
            self.causal_training_data_receipt_sha256,
            self.causal_preflight_receipt,
            self.causal_preflight_receipt_sha256,
            self.causal_precursor_binding,
            self.causal_checkpoint_receipt_base,
        )
        if self.production_matrix_id:
            if any(value is None for value in causal_values):
                raise ValueError("causal production checkpoint evidence bindings are incomplete")
            if self.production_checkpoint_steps != (0, 100, 200):
                raise ValueError("causal checkpoint steps must be exactly (0, 100, 200)")
            from world_action_model.contact_causal_production import (
                PREFLIGHT_RECEIPT_VERSION,
                TRAINING_DATA_RECEIPT_VERSION,
                sha256_json as causal_sha256_json,
                validate_training_receipt_v5,
            )

            if (
                not isinstance(self.causal_training_data_receipt, dict)
                or self.causal_training_data_receipt.get("receipt_version")
                != TRAINING_DATA_RECEIPT_VERSION
                or causal_sha256_json(self.causal_training_data_receipt)
                != self.causal_training_data_receipt_sha256
                or not isinstance(self.causal_preflight_receipt, dict)
                or self.causal_preflight_receipt.get("receipt_version")
                != PREFLIGHT_RECEIPT_VERSION
                or causal_sha256_json(self.causal_preflight_receipt)
                != self.causal_preflight_receipt_sha256
                or self.causal_preflight_receipt.get("route_set")
                != [
                    "aux_only_direct_q12",
                    "equal_bit_unstructured_hard_st",
                    "hard_st_semantic",
                ]
                or self.causal_preflight_receipt.get("training_data_receipt_sha256")
                != self.causal_training_data_receipt_sha256
                or not isinstance(self.causal_checkpoint_receipt_base, dict)
                or self.causal_checkpoint_receipt_base.get("receipt_version")
                != "contact-asym25-checkpoint-training-receipt/v5"
            ):
                raise ValueError("causal production preflight/training-data/base receipt differs")
            from world_action_model.contact_causal_preflight import (
                validate_preflight_receipt_v2,
                validate_training_data_receipt_v2,
            )

            validate_training_data_receipt_v2(self.causal_training_data_receipt)
            validate_preflight_receipt_v2(
                self.causal_preflight_receipt,
                training_data_receipt=self.causal_training_data_receipt,
            )
            validate_training_receipt_v5(self.causal_checkpoint_receipt_base)
            base = self.causal_checkpoint_receipt_base
            if (
                base["checkpoint_step"] != 0
                or base["matrix_id"] != self.production_matrix_id
                or base["route"]["causal_route_id"]
                != self.production_causal_route_id
                or base["seed"] != self.causal_training_data_receipt["split_seed"]
                or base["precursor"] != self.causal_precursor_binding
                or base["evidence"]["production_preflight_receipt_sha256"]
                != self.causal_preflight_receipt_sha256
                or base["evidence"].get("hardened_preflight_receipt_sha256")
                is None
                or base["evidence"]["training_data_receipt_sha256"]
                != self.causal_training_data_receipt_sha256
                or base["stable_contract"]["dataset_tree_sha256"]
                != self.causal_training_data_receipt["dataset_tree_sha256"]
                or base["stable_contract"]["hardware_contract_sha256"]
                != self.causal_training_data_receipt["hardware_contract_sha256"]
                or base["stable_contract"]["train_only_norm_sha256"]
                != self.causal_training_data_receipt["train_only_norm_sha256"]
                or base["topology"]
                != {
                    "world_size": self.num_processes,
                    "rank_set": list(range(self.num_processes)),
                }
            ):
                raise ValueError("causal checkpoint v5 base binding differs")
        elif any(value is not None for value in causal_values):
            raise ValueError("causal checkpoint bindings require the causal production matrix")
        if self.current_training_data_receipt is not None:
            if not isinstance(self.current_training_data_receipt, dict):
                raise ValueError("current training-data receipt must be a JSON object")
            receipt_payload = json.dumps(
                self.current_training_data_receipt,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            actual_receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
            if actual_receipt_sha256 != self.current_training_data_receipt_sha256:
                raise ValueError("current training-data receipt fingerprint differs")
            if self.current_training_data_receipt.get("receipt_version") != (
                "contact-asym25-current-training-data-receipt/v1"
            ):
                raise ValueError("current training-data receipt version mismatch")
            if self.production_checkpoint_steps != (0, 100, 200):
                raise ValueError(
                    "production checkpoint steps must be exactly (0, 100, 200)"
                )
            for name, value in (
                ("production_training_run_nonce", self.production_training_run_nonce),
                ("production_training_run_id", self.production_training_run_id),
                (
                    "production_experiment_family_id",
                    self.production_experiment_family_id,
                ),
                ("production_variant_id", self.production_variant_id),
            ):
                if not isinstance(value, str) or len(value) < 16:
                    raise ValueError(f"{name} must contain at least 16 characters")
            if self.production_transport_variant_id not in {
                "soft_semantic",
                "hard_st",
                "hard_detached",
            }:
                raise ValueError(
                    "production_transport_variant_id is not an implemented transport"
                )
            if self.production_training_precision != "bfloat16":
                raise ValueError("production training precision must be bfloat16")
            if (
                not isinstance(
                    self.production_matched_control_receipt_sha256, str
                )
                or len(self.production_matched_control_receipt_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.production_matched_control_receipt_sha256
                )
            ):
                raise ValueError(
                    "production matched-control receipt must be lowercase SHA-256"
                )
            if not isinstance(self.production_action_schema_contract, dict):
                raise ValueError("production action schema contract is required")
            from world_action_model.contact_asym25_checkpoint_acceptance import (
                PRODUCTION_RUN_CONTRACT_KEYS,
                sha256_json,
            )

            if (
                not isinstance(self.production_run_contract, dict)
                or set(self.production_run_contract) != PRODUCTION_RUN_CONTRACT_KEYS
                or sha256_json(self.production_run_contract)
                != self.production_run_contract_sha256
                or not isinstance(self.production_runtime_environment_receipt, dict)
                or not self.production_runtime_environment_receipt
                or sha256_json(self.production_runtime_environment_receipt)
                != self.production_run_contract["runtime_environment_receipt_sha256"]
            ):
                raise ValueError("production run/environment contract differs")
            training_config_receipt = self.production_run_contract.get(
                "training_config_receipt"
            )
            if (
                not isinstance(training_config_receipt, dict)
                or not training_config_receipt
                or isinstance(training_config_receipt.get("seed"), bool)
                or training_config_receipt.get("seed")
                != self.current_training_data_receipt.get("split_seed")
                or sha256_json(training_config_receipt)
                != self.production_training_config_sha256
            ):
                raise ValueError("production training config receipt differs")
            stable_training_config = dict(training_config_receipt)
            stable_training_config["seed"] = "__CONTACT_ASYM25_SPLIT_SEED__"
            if sha256_json(stable_training_config) != self.production_run_contract[
                "stable_training_config_sha256"
            ]:
                raise ValueError("stable production training config differs")
            contract_mirrors = {
                "optimizer_config_sha256": self.production_optimizer_config_sha256,
                "scheduler_config_sha256": self.production_scheduler_config_sha256,
                "training_config_sha256": self.production_training_config_sha256,
                "train_window_receipt_sha256": self.production_train_window_receipt_sha256,
                "base_model_bundle_sha256": self.production_base_model_bundle_sha256,
            }
            if any(
                self.production_run_contract[name] != value
                for name, value in contract_mirrors.items()
            ):
                raise ValueError("production checkpoint control contract differs")
            if (
                self.production_run_contract["training_code_receipt_sha256"]
                != self.current_training_data_receipt.get(
                    "training_code_receipt_sha256"
                )
                or self.production_run_contract["dataset_source_tree_sha256"]
                != self.current_training_data_receipt.get(
                    "dataset_source_tree_receipt", {}
                ).get("tree_sha256")
                or self.production_run_contract["preflight_receipt_sha256"]
                != self.current_training_data_receipt.get(
                    "production_preflight_receipt_sha256"
                )
                or self.production_run_contract["precision"]
                != self.production_training_precision
                or self.production_run_contract["action_schema_contract_sha256"]
                != sha256_json(self.production_action_schema_contract)
            ):
                raise ValueError(
                    "production run contract differs from live training inputs"
                )
            for name, value in (
                ("production_run_contract_sha256", self.production_run_contract_sha256),
                (
                    "production_matched_control_external_acceptance_sha256",
                    self.production_matched_control_external_acceptance_sha256,
                ),
                *contract_mirrors.items(),
            ):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"{name} must be lowercase SHA-256")
            source_receipt = self.current_training_data_receipt.get(
                "dataset_source_tree_receipt"
            )
            snapshot = self.current_training_source_snapshot
            if (
                not isinstance(source_receipt, dict)
                or not isinstance(snapshot, dict)
                or set(snapshot)
                != {
                    "path_record_only",
                    "tree_sha256",
                    "file_count",
                    "total_size_bytes",
                    "_runtime_rows",
                }
                or {
                    key: snapshot[key]
                    for key in (
                        "path_record_only",
                        "tree_sha256",
                        "file_count",
                        "total_size_bytes",
                    )
                }
                != source_receipt
                or not isinstance(snapshot["_runtime_rows"], list)
                or len(snapshot["_runtime_rows"]) != snapshot["file_count"]
            ):
                raise ValueError(
                    "current training source snapshot differs from its receipt"
                )
        elif self.current_training_source_snapshot is not None:
            raise ValueError(
                "current training source snapshot requires a production receipt"
            )
        self.matched_control_collector = None
        matched_control_config = model_config.get("matched_control_collector")
        if matched_control_config is not None and bool(
            matched_control_config.get("enabled", False)
        ):
            collector_class = MatchedControlCollector
            if matched_control_config.get("collector_version") == (
                "matched-control-hash-collector/v3"
            ):
                from world_action_model.contact_causal_matched_control import (
                    CausalMatchedControlCollector,
                )

                collector_class = CausalMatchedControlCollector
            self.matched_control_collector = collector_class(
                matched_control_config,
                rank=self.process_index,
                world_size=self.num_processes,
                trainer_seed=self.seed,
                project_dir=self.project_dir,
            )
        if self.contact_condition_mode not in {"predicted", "oracle"}:
            raise ValueError(
                "contact_condition_mode must be 'predicted' or 'oracle', got "
                f"{self.contact_condition_mode!r}"
            )
        if self.contact_causal_route in {
            route.value
            for route in (
                ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
                ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
                ContactCausalRoute.HARD_ST_SEMANTIC,
            )
        }:
            if self.contact_condition_mode != "predicted":
                if self.production_matrix_id or self.contact_causal_route != ContactCausalRoute.HARD_ST_SEMANTIC.value:
                    raise ValueError(
                        f"route {self.contact_causal_route!r} rejects oracle conditioning"
                    )
            if self.production_matrix_id:
                if not self.require_contact_targets or self.require_positive_contact_supervision:
                    raise ValueError(
                        f"route {self.contact_causal_route!r} requires zero-safe required "
                        "targets without a per-batch positive-supervision requirement"
                    )
            elif not self.require_contact_targets or not self.require_positive_contact_supervision:
                raise ValueError(
                    f"route {self.contact_causal_route!r} requires a positive grouped "
                    "contact auxiliary loss and positively masked required targets"
                )
            if self.contact_schema_version != SEGMENT_SCHEMA_VERSION:
                raise ValueError(
                    f"route {self.contact_causal_route!r} requires v1.1 segment targets"
                )
        self.contact_route_loss_receipt = {
            "route": self.contact_causal_route or "legacy",
            "action_loss": "flow_mse",
            "visual_loss": "flow_mse",
            "contact_aux_loss": (
                "grouped_contact_segment_loss"
                if self.contact_schema_version == SEGMENT_SCHEMA_VERSION
                and self.contact_loss_weight > 0.0
                else "disabled_or_legacy"
            ),
            "effect_or_counterfactual_physics_loss": False,
            "contact_supervision_rule_version": (
                self.contact_supervision_rule_version
                if self.contact_schema_version == SEGMENT_SCHEMA_VERSION
                else None
            ),
        }

        vae_pretrained = model_config.get("vae_pretrained", os.path.join(pretrained, "vae"))
        vae_dtype = model_config.get("vae_dtype", torch.float32)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        transformer_pretrained = model_config.get("transformer_pretrained", os.path.join(pretrained, "transformer"))
        if self.current_training_data_receipt is not None:
            from world_action_model.contact_asym25_heldout_eval import (
                snapshot_artifact_tree,
            )

            self._production_initial_weights_snapshot = snapshot_artifact_tree(
                transformer_pretrained
            )
            if (
                self._production_initial_weights_snapshot["tree_sha256"]
                != self.production_base_model_bundle_sha256
            ):
                raise ValueError(
                    "runtime initial weights differ from the reviewed base bundle"
                )
        transformer_cfg = dict(model_config.get("transformer"))
        if self.current_training_data_receipt is not None:
            transformer_cfg.update(
                current_training_data_receipt=self.current_training_data_receipt,
                current_training_data_receipt_sha256=(
                    self.current_training_data_receipt_sha256
                ),
                production_checkpoint_steps=self.production_checkpoint_steps,
            )
        transformer = CasualWorldActionTransformer_MoT(**transformer_cfg)
        transformer = load_pretrained_weights(
            transformer,
            transformer_pretrained,
            skip_action_expert=model_config.get("skip_action_expert", False),
            strict_load=model_config.get("strict_load", False),
        )
        if self._production_initial_weights_snapshot is not None:
            from world_action_model.contact_asym25_heldout_eval import (
                verify_artifact_tree_unchanged,
            )

            verify_artifact_tree_unchanged(
                self._production_initial_weights_snapshot
            )
        transformer.to(self.device)
        self.ego_contact_precursor_parameter_partition = None
        if self.ego_contact_precursor_objective is not None:
            from world_action_model.contact_ego_causal_precursor import (
                precursor_objective_contract,
                freeze_precursor_action_path,
            )

            if not isinstance(self.ego_contact_precursor_objective, dict):
                raise ValueError("Ego precursor objective must be a canonical mapping")
            fixed_step = self.ego_contact_precursor_objective.get("fixed_save_step")
            if self.ego_contact_precursor_objective != precursor_objective_contract(
                fixed_save_step=fixed_step
            ):
                raise ValueError("Ego precursor objective contract differs")
            if not bool(getattr(transformer, "contact_planner_precursor", False)):
                raise ValueError("Ego precursor requires model-side precursor planner mode")
            self.ego_contact_precursor_parameter_partition = (
                freeze_precursor_action_path(transformer)
            )
            from world_action_model.contact_causal_production import (
                model_parameter_and_buffer_digests,
                named_tensor_digest,
            )

            self._ego_precursor_root_parameter_sha256 = (
                model_parameter_and_buffer_digests(transformer)[0]
            )
            named_parameters = dict(transformer.named_parameters())
            self._ego_precursor_frozen_root_sha256 = named_tensor_digest(
                [
                    (name, named_parameters[name])
                    for name in self.ego_contact_precursor_parameter_partition[
                        "frozen_parameter_names"
                    ]
                ]
            )
        if model_config.get("enable_gradient_checkpointing", True):
            transformer.enable_gradient_checkpointing()

        model = dict(transformer=transformer)
        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        if self.production_matrix_id:
            from world_action_model.contact_causal_production import (
                DIRECT_UNUSED_PARAMETER_NAMES,
                model_parameter_and_buffer_digests,
                named_tensor_digest,
            )

            parameter_sha256, _ = model_parameter_and_buffer_digests(transformer)
            self._causal_initial_parameter_sha256 = parameter_sha256
            if (
                parameter_sha256
                != self.causal_precursor_binding.get("parameter_tensor_sha256")
                or parameter_sha256
                != self.causal_checkpoint_receipt_base.get(
                    "parameter_topology", {}
                ).get("step0_parameter_tensor_sha256")
            ):
                raise ValueError("causal Stage2 initial tensors differ from the precursor")
            self._causal_seen_gradient_names = set()
            self._causal_action_gradient_names = set()
            self._causal_action_gradient_probe_completed = False
            for parameter_name, parameter in transformer.named_parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda gradient, name=parameter_name: (
                            self._causal_seen_gradient_names.add(name) or gradient
                        )
                    )
            expected_unused = (
                list(DIRECT_UNUSED_PARAMETER_NAMES)
                if self.production_causal_route_id == "aux_only_direct_q12"
                else []
            )
            actual_unused = sorted(
                name
                for name, _ in transformer.named_parameters()
                if name.startswith("contact_condition_encoder.")
            )
            if expected_unused and actual_unused != expected_unused:
                raise ValueError("causal direct model-derived unused set differs")
            self._causal_unused_step0_sha256 = named_tensor_digest(
                [
                    (name, dict(transformer.named_parameters())[name])
                    for name in expected_unused
                ]
            )
        model = ModuleDict(model)
        model.train()
        return model

    def _production_checkpoint_dir(self, step: int) -> Path:
        epoch = int(0 if step == 0 else self.cur_epoch)
        return Path(self.model_dir) / f"checkpoint_epoch_{epoch}_step_{step}"

    def _write_production_checkpoint_receipt(self, output_dir: Path, step: int) -> None:
        if (
            self.current_training_data_receipt is None
            and not self.production_matrix_id
        ) or not self.is_main_process:
            return
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"production checkpoint directory was not written: {output_dir}"
            )
        if self.production_matrix_id:
            self._write_causal_checkpoint_receipt(output_dir, step)
            return
        from world_action_model.contact_asym25_checkpoint_acceptance import (
            CROSS_SEED_SCIENTIFIC_CONTRACT,
            build_bundle_manifest,
            write_json_exclusive,
        )

        if self._production_initial_weights_snapshot is None:
            raise ValueError("production initial-weights snapshot is unavailable")
        artifact_dir = output_dir / "transformer"
        if not artifact_dir.is_dir():
            raise FileNotFoundError("production transformer checkpoint was not written")
        bundle_manifest = build_bundle_manifest(artifact_dir)
        bundle_manifest_file_sha256 = write_json_exclusive(
            artifact_dir / "checkpoint_bundle_manifest.json", bundle_manifest
        )
        initial_sha256 = self._production_initial_weights_snapshot["tree_sha256"]
        if step == 0:
            parent_sha256 = initial_sha256
        else:
            previous_step = self.production_checkpoint_steps[
                self.production_checkpoint_steps.index(step) - 1
            ]
            try:
                parent_sha256 = self._production_checkpoint_bundle_sha256[
                    previous_step
                ]
            except KeyError as exc:
                raise ValueError(
                    "production checkpoint parent was not emitted in this run"
                ) from exc
        current = self.current_training_data_receipt
        payload = {
            "receipt_version": "contact-asym25-checkpoint-training-receipt/v4",
            "cross_seed_scientific_contract": CROSS_SEED_SCIENTIFIC_CONTRACT,
            "checkpoint_step": int(step),
            "checkpoint_bundle_manifest_sha256": bundle_manifest_file_sha256,
            "checkpoint_bundle_sha256": bundle_manifest["bundle_sha256"],
            "parent_checkpoint_bundle_sha256": parent_sha256,
            "initial_weights_bundle_sha256": initial_sha256,
            "base_model_bundle_sha256": self.production_base_model_bundle_sha256,
            "training_run_nonce": self.production_training_run_nonce,
            "training_run_id": self.production_training_run_id,
            "experiment_family_id": self.production_experiment_family_id,
            "variant_id": self.production_variant_id,
            "transport_variant_id": self.production_transport_variant_id,
            "matched_control_receipt_sha256": (
                self.production_matched_control_receipt_sha256
            ),
            "matched_control_external_acceptance_sha256": (
                self.production_matched_control_external_acceptance_sha256
            ),
            "production_run_contract": self.production_run_contract,
            "production_run_contract_sha256": self.production_run_contract_sha256,
            "runtime_environment_receipt": (
                self.production_runtime_environment_receipt
            ),
            "runtime_environment_receipt_sha256": self.production_run_contract[
                "runtime_environment_receipt_sha256"
            ],
            "optimizer_config_sha256": self.production_optimizer_config_sha256,
            "scheduler_config_sha256": self.production_scheduler_config_sha256,
            "training_config_sha256": self.production_training_config_sha256,
            "training_config_receipt": self.production_run_contract[
                "training_config_receipt"
            ],
            "stable_training_config_sha256": self.production_run_contract[
                "stable_training_config_sha256"
            ],
            "control_config_sha256": self.production_run_contract[
                "control_config_sha256"
            ],
            "train_window_receipt_sha256": (
                self.production_train_window_receipt_sha256
            ),
            "split_seed": current["split_seed"],
            "training_precision": self.production_training_precision,
            "contact_query_count": 12,
            "contact_target_dim": SEGMENT_TARGET_DIM,
            "contact_time_bins": 16,
            "action_schema_contract": self.production_action_schema_contract,
            "production_preflight_receipt_sha256": current[
                "production_preflight_receipt_sha256"
            ],
            "heldout_window_manifest_sha256": current[
                "heldout_window_manifest_sha256"
            ],
            "training_gate_sha256": current["training_gate_sha256"],
            "current_training_data_receipt": current,
            "current_training_data_receipt_sha256": (
                self.current_training_data_receipt_sha256
            ),
            "training_code_receipt_sha256": current[
                "training_code_receipt_sha256"
            ],
            "dataset_source_tree_sha256": current[
                "dataset_source_tree_receipt"
            ]["tree_sha256"],
            "initial_weights_tree_receipt": {
                key: self._production_initial_weights_snapshot[key]
                for key in (
                    "path_record_only",
                    "tree_sha256",
                    "file_count",
                    "total_size_bytes",
                )
            },
        }
        write_json_exclusive(
            artifact_dir / "checkpoint_training_receipt.json", payload
        )
        self._production_checkpoint_bundle_sha256[step] = bundle_manifest[
            "bundle_sha256"
        ]

    def _write_causal_checkpoint_receipt(self, output_dir: Path, step: int) -> None:
        from world_action_model.contact_asym25_checkpoint_acceptance import (
            build_bundle_manifest,
            write_json_exclusive,
        )
        from world_action_model.contact_causal_production import (
            build_live_training_state_receipts,
            build_checkpoint_training_receipt_v5,
            model_parameter_and_buffer_digests,
            write_canonical_json_exclusive,
        )
        from world_action_model.verified_io import hash_verified_regular_file

        artifact_dir = output_dir / "transformer"
        if not artifact_dir.is_dir():
            raise FileNotFoundError("causal transformer checkpoint was not written")
        bundle_manifest = build_bundle_manifest(artifact_dir)
        write_json_exclusive(
            artifact_dir / "checkpoint_bundle_manifest.json", bundle_manifest
        )
        weight_path = artifact_dir / bundle_manifest["weight_file"]
        config_sha256, _ = hash_verified_regular_file(artifact_dir / "config.json")
        unwrapped = self.accelerator.unwrap_model(self.models[0])
        transformer = unwrapped["transformer"]
        parameter_sha256, buffer_sha256 = model_parameter_and_buffer_digests(
            transformer
        )
        topology, reachability = build_live_training_state_receipts(
            model=transformer,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            route=self.production_causal_route_id,
            checkpoint_step=step,
            step0_parameter_tensor_sha256=self._causal_initial_parameter_sha256,
            observed_gradient_parameter_names=(
                [] if step == 0 else sorted(self._causal_seen_gradient_names)
            ),
            action_loss_gradient_parameter_names=(
                [] if step == 0 else sorted(self._causal_action_gradient_names)
            ),
            unused_step0_tensor_sha256=self._causal_unused_step0_sha256,
        )
        if self.matched_control_collector is None:
            raise ValueError("causal checkpoint requires the live collector")
        trace_prefix = self.matched_control_collector.current_run_trace_prefix_receipt(
            through_step=step
        )
        if step == 0:
            parent_sha256 = self.causal_precursor_binding["parameter_tensor_sha256"]
        else:
            prior = self.production_checkpoint_steps[
                self.production_checkpoint_steps.index(step) - 1
            ]
            if prior not in self._causal_checkpoint_weight_sha256:
                raise ValueError("causal checkpoint parent was not emitted in this run")
            parent_sha256 = self._causal_checkpoint_weight_sha256[prior]
        receipt = build_checkpoint_training_receipt_v5(
            base_receipt=self.causal_checkpoint_receipt_base,
            checkpoint_step=step,
            checkpoint_file_path=weight_path,
            parameter_tensor_sha256=parameter_sha256,
            buffer_tensor_sha256=buffer_sha256,
            config_sha256=config_sha256,
            whole_bundle_sha256=bundle_manifest["bundle_sha256"],
            parent_checkpoint_weight_sha256=parent_sha256,
            parameter_topology=topology,
            gradient_reachability_receipt=reachability,
            current_run_trace_prefix_receipt=trace_prefix,
        )
        write_canonical_json_exclusive(
            artifact_dir / "checkpoint_training_receipt.json", receipt
        )
        self._causal_checkpoint_weight_sha256[step] = parameter_sha256

    def _save_production_initial_checkpoint(self) -> None:
        if self.current_training_data_receipt is None and not self.production_matrix_id:
            return
        if self.cur_step != 0:
            raise ValueError("production step-0 checkpoint must precede optimization")
        self._verify_production_training_source()
        if self.production_matrix_id and self.matched_control_collector is not None:
            unwrapped = self.accelerator.unwrap_model(self.models[0])
            self.matched_control_collector.bind_live_training_state(
                model=unwrapped["transformer"],
                optimizers=self.optimizers,
                schedulers=self.schedulers,
            )
        output_dir = self._production_checkpoint_dir(0)
        if self.is_main_process:
            if output_dir.exists():
                raise FileExistsError(
                    f"refusing to overwrite production step-0 checkpoint: {output_dir}"
                )
            output_dir.mkdir(parents=True, exist_ok=False)
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.get_state_dict(self.models[0])
        self.save_model_hook(self.models, [state_dict], str(output_dir))
        self.accelerator.wait_for_everyone()
        self._write_production_checkpoint_receipt(output_dir, 0)
        self.accelerator.wait_for_everyone()

    def train(self) -> None:
        if self.current_training_data_receipt is not None or self.production_matrix_id:
            self._save_production_initial_checkpoint()
        try:
            super().train()
        finally:
            if self.matched_control_collector is not None:
                self.matched_control_collector.close()

    def _verify_production_training_source(self) -> None:
        if self.current_training_source_snapshot is None:
            return
        from world_action_model.contact_asym25_heldout_eval import (
            verify_artifact_tree_unchanged,
        )

        verify_artifact_tree_unchanged(self.current_training_source_snapshot)

    def save_checkpoint_step(self) -> None:
        if self._by_epoch and self.checkpoint_interval < self.max_epochs:
            checkpoint_interval = int(self.checkpoint_interval * self.epoch_size)
        else:
            checkpoint_interval = int(self.checkpoint_interval)
        should_save = (
            self.cur_step % checkpoint_interval == 0 or self.cur_step == self.max_steps
        )
        is_production_evidence_step = (
            (self.current_training_data_receipt is not None or self.production_matrix_id)
            and self.cur_step in self.production_checkpoint_steps
        )
        if should_save and is_production_evidence_step:
            self._verify_production_training_source()
        if should_save and (
            is_production_evidence_step
            or self.ego_contact_precursor_objective is not None
        ):
            guarded_output = Path(self.model_dir) / (
                f"checkpoint_epoch_{self.cur_epoch}_step_{self.cur_step}"
            )
            if self.is_main_process and guarded_output.exists():
                raise FileExistsError(
                    f"refusing base-trainer checkpoint overwrite: {guarded_output}"
                )
        super().save_checkpoint_step()
        if (
            should_save
            and self.ego_contact_precursor_objective is not None
            and self.cur_step
            == self.ego_contact_precursor_objective["fixed_save_step"]
        ):
            output_dir = Path(self.model_dir) / (
                f"checkpoint_epoch_{self.cur_epoch}_step_{self.cur_step}"
            )
            self.accelerator.wait_for_everyone()
            if self.is_main_process:
                from world_action_model.contact_causal_production import (
                    write_canonical_json_exclusive,
                )
                from world_action_model.contact_ego_causal_precursor import (
                    build_live_precursor_receipt,
                )

                receipt, bundle_manifest = build_live_precursor_receipt(
                    model=self.accelerator.unwrap_model(self.models[0])[
                        "transformer"
                    ],
                    checkpoint_dir=output_dir / "transformer",
                    parameter_partition=self.ego_contact_precursor_parameter_partition,
                    objective_contract=self.ego_contact_precursor_objective,
                    evidence_base=self.ego_contact_precursor_evidence_base,
                    root_parameter_tensor_sha256=(
                        self._ego_precursor_root_parameter_sha256
                    ),
                    frozen_root_tensor_sha256=(
                        self._ego_precursor_frozen_root_sha256
                    ),
                )
                write_canonical_json_exclusive(
                    output_dir / "ego_contact_causal_precursor_bundle_manifest.json",
                    bundle_manifest,
                )
                write_canonical_json_exclusive(
                    output_dir / "ego_contact_causal_precursor_receipt.json",
                    receipt,
                )
            self.accelerator.wait_for_everyone()
        if should_save and is_production_evidence_step:
            output_dir = self._production_checkpoint_dir(self.cur_step)
            self.accelerator.wait_for_everyone()
            self._write_production_checkpoint_receipt(output_dir, self.cur_step)
            self.accelerator.wait_for_everyone()

    def forward_step(self, batch_dict):
        if self.ego_contact_precursor_objective is not None:
            return self._forward_ego_contact_precursor_step(batch_dict)
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        bs = images.shape[0]
        prompt_embeds = batch_dict["prompt_embeds"]
        action = batch_dict["action"]
        state = batch_dict["state"]
        embodiment_id = batch_dict["embodiment_id"]

        visual_timestep, visual_sigma = self.get_timestep_and_sigma(bs, images.ndim, self.visual_flow_shift)
        action_timestep, action_sigma = self.get_timestep_and_sigma(bs, action.ndim, self.action_flow_shift)

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * visual_sigma + visual_latents * (1 - visual_sigma)

        dim_mask = None
        if "action_dim_mask" in batch_dict:
            dim_mask = _as_dim_mask(
                batch_dict["action_dim_mask"],
                batch_size=bs,
                seq_len=action.shape[1],
                dim=action.shape[2],
                device=action.device,
            )

        # Missing embodiment dimensions must not inject random flow noise into
        # the shared action stream.  Mask both endpoints before interpolation,
        # then use the same mask for the objective below.
        flow_mask = 1.0 if dim_mask is None else dim_mask.to(dtype=action.dtype, device=action.device)
        action_for_flow = action * flow_mask
        action_noise = torch.randn_like(action) * flow_mask
        action_target = (action_noise - action_for_flow) * flow_mask
        noisy_action = action_noise * action_sigma + action_for_flow * (1 - action_sigma)
        prompt_embeds = prompt_embeds.to(self.dtype)
        if "ref_images" not in batch_dict:
            raise ValueError("CasualWATrainerMoT requires ref_images in batch_dict")

        if not self.expand_timesteps:
            ref_images = batch_dict["ref_images"]
            ref_latents = self.forward_vae(ref_images)
            num_frames = images.shape[1]
            batch_size = ref_latents.shape[0]
            latent_height = ref_latents.shape[-2]
            latent_width = ref_latents.shape[-1]
            mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2).to(ref_latents.device)
            condition = torch.concat([mask_lat_size, ref_latents], dim=1)
            insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
        else:
            num_latent_frames = visual_latents.shape[2]
            latent_height = visual_latents.shape[-2]
            latent_width = visual_latents.shape[-1]
            ref_images = batch_dict["ref_images"][:, :1]
            ref_latents = self.forward_vae(ref_images)
            first_frame_mask = torch.ones(
                bs,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=visual_latents.dtype,
                device=visual_latents.device,
            )
            first_frame_mask[:, :, 0] = 0
            insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens
        timestep = torch.zeros(
            bs,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        timestep[:, num_state_tokens + num_clean_latent_tokens : num_state_tokens + num_clean_latent_tokens + num_action_tokens] = (
            action_timestep[:, None]
        )
        timestep[:, num_state_tokens + num_action_tokens + num_clean_latent_tokens :] = visual_timestep[:, None]

        contact_target = batch_dict.get("contact_target")
        contact_condition = batch_dict.get("contact_condition")
        contact_condition_mask = batch_dict.get("contact_condition_mask")
        if self.contact_supervision_rule_version == CONTACT_GROUPED_SUPERVISION_ZERO_SAFE_VERSION:
            supervision_receipt = validate_zero_safe_contact_supervision(
                contact_target,
                batch_dict.get("contact_loss_mask"),
                batch_dict.get("contact_confidence"),
            )
            if supervision_receipt["zero_supervision"]:
                self._last_zero_safe_supervision_receipt = supervision_receipt
        if self.require_positive_contact_supervision:
            validate_positive_contact_supervision(
                contact_target,
                batch_dict.get("contact_loss_mask"),
                batch_dict.get("contact_confidence"),
            )
        if self.production_matrix_id and (
            contact_condition is not None or contact_condition_mask is not None
        ):
            raise ValueError(
                f"route {self.contact_causal_route!r} rejects model-side contact/oracle inputs; "
                "contact_target may enter only the grouped auxiliary loss"
            )
        if self.contact_condition_mode == "oracle":
            if contact_target is None:
                raise ValueError(
                    "oracle contact conditioning requires contact_target in the batch"
                )
            contact_condition = contact_target
            if contact_condition_mask is None:
                contact_condition_mask = batch_dict.get("contact_loss_mask")
                if (
                    self.contact_schema_version != SEGMENT_SCHEMA_VERSION
                    and contact_condition_mask is not None
                    and contact_condition_mask.ndim == 3
                ):
                    contact_condition_mask = contact_condition_mask.amax(dim=-1)

        time_mask = None
        if "action_loss_mask" in batch_dict:
            raw_time_mask = batch_dict["action_loss_mask"]
            if self.action_repeats > 1:
                raw_time_mask = raw_time_mask.repeat(1, self.action_repeats)
            time_mask = _as_time_mask(
                raw_time_mask,
                batch_size=bs,
                seq_len=action.shape[1],
                device=action.device,
            )
        action_loss_weight = batch_dict.get("action_loss_weight")
        if action_loss_weight is not None and self.action_repeats > 1:
            action_loss_weight = action_loss_weight.repeat(
                1, self.action_repeats, 1
            )
        if self.matched_control_collector is not None:
            from world_action_model.contact_causal_matched_control import (
                capture_pre_forward_rng_v3,
            )

            self.matched_control_collector.record(
                global_step=self.cur_step,
                batch_dict=batch_dict,
                visual_timestep=visual_timestep,
                action_timestep=action_timestep,
                visual_noise=visual_noise,
                action_noise=action_noise,
                action_target=action_target,
                action_loss_mask={
                    "dimension_mask": dim_mask,
                    "time_mask": time_mask,
                    "dimension_weight": action_loss_weight,
                },
                state_input=state,
                state_input_mask=None,
                embodiment_id=embodiment_id,
                prompt_t5=prompt_embeds,
                pre_forward_rng=capture_pre_forward_rng_v3(),
            )

        model_output = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )
        if len(model_output) == 3:
            visual_pred, action_pred, contact_pred = model_output
        else:
            visual_pred, action_pred = model_output
            contact_pred = None

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        action_loss = masked_weighted_action_mse(
            action_pred.float(),
            action_target.float(),
            dim_mask=dim_mask,
            time_mask=time_mask,
            action_loss_weight=action_loss_weight,
        )
        if self.production_matrix_id and not self._causal_action_gradient_probe_completed:
            unwrapped = self.accelerator.unwrap_model(self.models[0])["transformer"]
            planner_parameters = [
                (name, parameter)
                for name, parameter in unwrapped.named_parameters()
                if name.startswith("contact_") and parameter.requires_grad
            ]
            if not planner_parameters:
                raise ValueError("causal production model has no planner parameters")
            gradients = torch.autograd.grad(
                action_loss,
                [parameter for _, parameter in planner_parameters],
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            self._causal_action_gradient_names = {
                name
                for (name, _), gradient in zip(
                    planner_parameters, gradients, strict=True
                )
                if gradient is not None
            }
            expected_names = (
                set()
                if self.production_causal_route_id == "aux_only_direct_q12"
                else {name for name, _ in planner_parameters}
            )
            if self._causal_action_gradient_names != expected_names:
                raise ValueError(
                    "live action-loss-to-planner gradient reachability differs from the causal route"
                )
            self._causal_action_gradient_probe_completed = True
        losses = {
            "visual_loss": visual_loss,
            "action_loss": action_loss,
        }
        if contact_pred is not None and contact_target is not None:
            if self.contact_schema_version == SEGMENT_SCHEMA_VERSION:
                zero_safe = (
                    self.contact_supervision_rule_version
                    == CONTACT_GROUPED_SUPERVISION_ZERO_SAFE_VERSION
                    and validate_zero_safe_contact_supervision(
                        contact_target,
                        batch_dict.get("contact_loss_mask"),
                        batch_dict.get("contact_confidence"),
                    )["zero_supervision"]
                )
                contact_loss = (
                    canonical_zero_contact_loss(contact_pred)
                    if zero_safe
                    else grouped_contact_segment_loss(
                        contact_pred,
                        contact_target,
                        mask=batch_dict.get("contact_loss_mask"),
                        confidence=batch_dict.get("contact_confidence"),
                    )
                )
            else:
                contact_loss = masked_contact_loss(
                    contact_pred,
                    contact_target,
                    mask=batch_dict.get("contact_loss_mask"),
                    confidence=batch_dict.get("contact_confidence"),
                    loss_type=self.contact_loss_type,
                )
            losses["contact_loss"] = self.contact_loss_weight * contact_loss
        elif self.require_contact_targets:
            missing = "contact_target" if contact_target is None else "contact prediction head"
            raise ValueError(f"required {missing} is missing")
        return losses

    def _forward_ego_contact_precursor_step(self, batch_dict):
        """Visual/contact-only Stage1 objective with no action scientific I/O."""

        transformer = functools.partial(self.model, "transformer")
        forbidden = {"contact_condition", "contact_condition_mask"}.intersection(
            batch_dict
        )
        if forbidden:
            raise ValueError(
                f"route-neutral Ego precursor rejects model-side contact inputs: {sorted(forbidden)}"
            )
        images = batch_dict["images"]
        ref_images = batch_dict["ref_images"]
        prompt_embeds = batch_dict["prompt_embeds"].to(self.dtype)
        state = batch_dict["state"]
        embodiment_id = batch_dict["embodiment_id"]
        contact_target = batch_dict.get("contact_target")
        contact_mask = batch_dict.get("contact_loss_mask")
        contact_confidence = batch_dict.get("contact_confidence")
        supervision = validate_zero_safe_contact_supervision(
            contact_target, contact_mask, contact_confidence
        )
        bs = images.shape[0]
        visual_timestep, visual_sigma = self.get_timestep_and_sigma(
            bs, images.ndim, self.visual_flow_shift
        )
        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * visual_sigma + visual_latents * (1 - visual_sigma)
        if not self.expand_timesteps:
            ref_latents_full = self.forward_vae(ref_images)
            num_frames = images.shape[1]
            latent_height = ref_latents_full.shape[-2]
            latent_width = ref_latents_full.shape[-1]
            mask_lat_size = torch.ones(
                bs, 1, num_frames, latent_height, latent_width
            )
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
            )
            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(
                bs,
                -1,
                self.vae_scale_factor_temporal,
                latent_height,
                latent_width,
            ).transpose(1, 2).to(ref_latents_full.device)
            condition = torch.concat([mask_lat_size, ref_latents_full], dim=1)
            insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
        else:
            num_latent_frames = visual_latents.shape[2]
            latent_height = visual_latents.shape[-2]
            latent_width = visual_latents.shape[-1]
            ref_latents_full = self.forward_vae(ref_images[:, :1])
            first_frame_mask = torch.ones(
                bs,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=visual_latents.dtype,
                device=visual_latents.device,
            )
            first_frame_mask[:, :, 0] = 0
            insert_noisy_latents = (
                (1 - first_frame_mask) * ref_latents_full
                + first_frame_mask * noisy_latents
            )
        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        state = state.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = (
            first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        )
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens
        timestep = torch.zeros(
            bs,
            state.shape[1] + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        timestep[:, state.shape[1] + num_clean_latent_tokens :] = visual_timestep[
            :, None
        ]
        visual_pred, action_pred, contact_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            state=state,
            embodiment_id=embodiment_id,
            contact_precursor=True,
        )
        if action_pred is not None:
            raise RuntimeError("route-neutral precursor unexpectedly produced an action output")
        visual_loss = (
            (visual_pred.float() - visual_target.float()) * first_frame_mask
        ).pow(2).mean()
        contact_loss = (
            canonical_zero_contact_loss(contact_pred)
            if supervision["zero_supervision"]
            else grouped_contact_segment_loss(
                contact_pred,
                contact_target,
                mask=contact_mask,
                confidence=contact_confidence,
            )
        )
        self._last_zero_safe_supervision_receipt = supervision
        return {
            "visual_loss": visual_loss,
            "contact_loss": self.contact_loss_weight * contact_loss,
        }

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b t c h w -> b c t h w")
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim, flow_shift):
        sigma = torch.rand(batch_size).to(self.device)
        sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

def load_pretrained_weights(model, pretrained_path, skip_action_expert=False, strict_load=False):
    from safetensors.torch import load_file

    pretrained_path = gm_utils.get_model_path(pretrained_path)
    if os.path.isdir(pretrained_path):
        weight_files = sorted(
            os.path.join(pretrained_path, f)
            for f in os.listdir(pretrained_path)
            if f.endswith((".safetensors", ".bin"))
        )
    else:
        weight_files = [pretrained_path]

    state_dicts = {}
    for weight_file in weight_files:
        print(f"Loading weights from {weight_file}")
        if weight_file.endswith(".bin"):
            checkpoint = torch.load(weight_file, map_location="cpu")
        else:
            checkpoint = load_file(weight_file)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dicts.update(checkpoint)

    is_native_mot = any(".visual_expert." in key or ".action_expert." in key for key in state_dicts)
    if not strict_load and is_native_mot and hasattr(model, "load_from_mot_pretrained_state_dict"):
        loaded, skipped, unexpected = model.load_from_mot_pretrained_state_dict(
            state_dicts,
            skip_action_expert=skip_action_expert,
        )
        if unexpected:
            print(f"[WARNING] Unexpected keys in MoT checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
        if skipped:
            print(f"[WARNING] Skipped keys during MoT load: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        print(f"Loaded {len(loaded)} tensors from native MoT checkpoint.")
        return model

    if not strict_load and hasattr(model, "load_from_wan_pretrained_state_dict"):
        loaded, skipped, unexpected, missing = model.load_from_wan_pretrained_state_dict(
            state_dicts,
            skip_action_expert=skip_action_expert,
        )
        if unexpected:
            print(f"[WARNING] Unexpected keys in Wan checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
        if skipped:
            print(f"[WARNING] Skipped keys during MoT remap: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        if missing:
            print(f"[WARNING] Missing keys after MoT remap: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print(f"Loaded {len(loaded)} tensors from Wan2.2 into MoT.")
        return model

    missing_keys, unexpected_keys = model.load_state_dict(state_dicts, strict=False)
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys in state_dict: {unexpected_keys}")
    if missing_keys:
        print(f"[WARNING] Missing keys in state_dict: {missing_keys}")
    return model

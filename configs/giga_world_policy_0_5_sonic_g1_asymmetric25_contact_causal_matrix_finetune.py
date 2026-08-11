"""Production-gated Robot Stage2 causal-main matrix config.

This config performs only the new v3/v1 bootstrap.  It does not import the
historical contact mechanism production config and it forbids legacy aliases.
"""

import copy
import os
from pathlib import Path

from world_action_model.contact_causal_preflight import (
    load_robot_gate_from_environment,
    validate_preflight_receipt_v2,
    validate_training_data_receipt_v2,
)
from world_action_model.contact_causal_production import (
    MATRIX_ID,
    SUPERVISION_VERSION,
    build_production_route_receipt,
    current_head_source_tree_receipt,
    ensure_exact_causal_route,
    strict_json_file,
    validate_dataset_supervision_coverage,
    validate_matrix_manifest,
    sha256_json,
    validate_training_receipt_v5,
)
from world_action_model.contact_ego_causal_precursor import (
    precursor_stage2_binding,
    validate_precursor_receipt,
)
from world_action_model.contact_causal_matched_control import (
    validate_collector_config_v3,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)


gate, gate_sha256, gate_path = load_robot_gate_from_environment()
route = ensure_exact_causal_route(os.environ["GWP_CONTACT_CAUSAL_ROUTE"].strip())


def _required_json_environment(name, description):
    path_text = os.environ.get(name, "").strip()
    if not path_text:
        raise ValueError(f"causal Robot config requires {name}")
    path = Path(path_text).expanduser().absolute()
    value, digest, payload = strict_json_file(path, name=description)
    return path, value, digest, payload


matrix_path, matrix_manifest, matrix_sha256, _ = _required_json_environment(
    "GWP_CONTACT_CAUSAL_MATRIX_MANIFEST", "causal production matrix"
)
validate_matrix_manifest(matrix_manifest)
if matrix_sha256 != gate["artifact_fingerprints"]["matrix_manifest_sha256"]:
    raise ValueError("causal production matrix bytes differ from the v3 gate")
source_tree_receipt = current_head_source_tree_receipt(
    Path(__file__).resolve().parents[1],
    expected_commit=gate["expected_git_commit"],
)
source_tree_sha256 = sha256_json(source_tree_receipt)
if source_tree_sha256 != gate["artifact_fingerprints"]["current_head_source_tree_sha256"]:
    raise ValueError("runtime causal source tree differs from the v3 gate")

precursor_path, precursor_receipt, precursor_sha256, precursor_bytes = (
    _required_json_environment(
        "GWP_EGO_CAUSAL_PRECURSOR_RECEIPT", "Ego causal precursor receipt"
    )
)
validate_precursor_receipt(precursor_receipt)
precursor_binding = precursor_stage2_binding(
    precursor_receipt, receipt_bytes=precursor_bytes
)
if (
    precursor_sha256 != gate["artifact_fingerprints"]["precursor_receipt_bytes_sha256"]
    or precursor_binding["bundle_sha256"]
    != gate["artifact_fingerprints"]["precursor_bundle_sha256"]
):
    raise ValueError("Ego precursor bytes/bundle differ from the v3 gate")

coverage_path, coverage_receipt, coverage_sha256, _ = _required_json_environment(
    "GWP_CONTACT_CAUSAL_DATASET_COVERAGE_RECEIPT",
    "Robot dataset contact coverage receipt",
)
validate_dataset_supervision_coverage(coverage_receipt)
if coverage_sha256 != gate["artifact_fingerprints"]["dataset_contact_coverage_receipt_sha256"]:
    raise ValueError("Robot contact coverage receipt differs from the v3 gate")

training_data_path, training_data_receipt, training_data_sha256, _ = (
    _required_json_environment(
        "GWP_CONTACT_CAUSAL_TRAINING_DATA_RECEIPT",
        "causal Robot training-data receipt",
    )
)
validate_training_data_receipt_v2(
    training_data_receipt, gate=gate, gate_sha256=gate_sha256
)
if training_data_receipt["dataset_contact_coverage_receipt_sha256"] != coverage_sha256:
    raise ValueError("causal Robot training-data receipt differs")

preflight_path, preflight_receipt, preflight_sha256, _ = _required_json_environment(
    "GWP_CONTACT_CAUSAL_PREFLIGHT_RECEIPT", "causal Robot preflight receipt"
)
validate_preflight_receipt_v2(
    preflight_receipt,
    gate=gate,
    gate_sha256=gate_sha256,
    training_data_receipt=training_data_receipt,
)
if (
    preflight_receipt["causal_route_id"] != route
    or preflight_receipt["training_data_receipt_sha256"] != training_data_sha256
):
    raise ValueError("causal Robot preflight receipt differs")

checkpoint_base_path, checkpoint_receipt_base, _, _ = _required_json_environment(
    "GWP_CONTACT_CAUSAL_CHECKPOINT_RECEIPT_BASE",
    "causal checkpoint training-receipt v5 base",
)
validate_training_receipt_v5(checkpoint_receipt_base)
if (
    checkpoint_receipt_base["checkpoint_step"] != 0
    or checkpoint_receipt_base["seed"] != training_data_receipt["split_seed"]
    or checkpoint_receipt_base["route"]["causal_route_id"] != route
    or checkpoint_receipt_base["run_identity"]["training_run_id"]
    != gate["training_run_id_by_route"][route]
    or checkpoint_receipt_base["run_identity"]["training_run_nonce"]
    != gate["training_run_nonce_by_route"][route]
    or checkpoint_receipt_base["run_identity"]["external_acceptance_sha256"]
    != gate["external_acceptance_sha256_by_route"][route]
    or checkpoint_receipt_base["precursor"] != precursor_binding
    or checkpoint_receipt_base["evidence"]["training_gate_sha256"] != gate_sha256
    or checkpoint_receipt_base["evidence"]["production_preflight_receipt_sha256"]
    != preflight_sha256
    or checkpoint_receipt_base["evidence"]["training_data_receipt_sha256"]
    != training_data_sha256
    or checkpoint_receipt_base["evidence"]["contact_coverage_receipt_sha256"]
    != coverage_sha256
    or checkpoint_receipt_base["stable_contract"]["git_commit"]
    != gate["expected_git_commit"]
    or checkpoint_receipt_base["stable_contract"]["source_tree_sha256"]
    != gate["artifact_fingerprints"]["current_head_source_tree_sha256"]
    or checkpoint_receipt_base["stable_contract"]["dataset_tree_sha256"]
    != training_data_receipt["dataset_tree_sha256"]
    or checkpoint_receipt_base["stable_contract"]["hardware_contract_sha256"]
    != training_data_receipt["hardware_contract_sha256"]
    or checkpoint_receipt_base["stable_contract"]["train_only_norm_sha256"]
    != training_data_receipt["train_only_norm_sha256"]
):
    raise ValueError("causal checkpoint v5 base receipt differs from reviewed inputs")

collector_path, collector_config, _, _ = _required_json_environment(
    "GWP_CONTACT_CAUSAL_COLLECTOR_CONFIG", "causal matched-control collector config"
)
validate_collector_config_v3(collector_config)
if (
    collector_config["causal_route_id"] != route
    or collector_config["seed"] != training_data_receipt["split_seed"]
    or collector_config["run_id"] != gate["training_run_id_by_route"][route]
    or collector_config["run_nonce"] != gate["training_run_nonce_by_route"][route]
    or collector_config["precursor_receipt_bytes_sha256"] != precursor_sha256
    or collector_config["initial_parameter_tensors_sha256"]
    != precursor_binding["parameter_tensor_sha256"]
    or collector_config["current_head_source_tree_sha256"]
    != gate["artifact_fingerprints"]["current_head_source_tree_sha256"]
    or collector_config["dataset_contact_coverage_receipt_sha256"]
    != coverage_sha256
    or collector_config["route_local_contract"]
    != build_production_route_receipt(route)
):
    raise ValueError("causal matched-control collector config differs")

from configs.giga_world_policy_0_5_sonic_g1_asymmetric25_finetune import (  # noqa: E402
    config as base_config,
)


config = copy.deepcopy(base_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR", f"outputs/contact_causal_main_{route}"
)
for dataset in config["dataloaders"]["train"]["data_or_config"]:
    dataset.update(
        episodes=training_data_receipt["train_episode_ids"],
        contact_sidecar_dir=os.environ["GWP_CONTACT_SIDECAR_DIR"],
        contact_sidecar_pattern="episode_{episode_index:06d}.npz",
        contact_sidecar_strict=True,
        contact_sidecar_require_reviewed_v1_1=True,
        contact_label_origin="human_adjudicated_robot_native",
        hand_active_contact_ground_truth=False,
        grip_contact_ground_truth=False,
    )

direct = route == "aux_only_direct_q12"
config["models"]["transformer"].update(
    num_contact_tokens=12,
    contact_target_dim=SEGMENT_TARGET_DIM,
    contact_condition_dim=SEGMENT_TARGET_DIM,
    contact_bottleneck=not direct,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_semantic_bottleneck=not direct,
    contact_semantic_transport_mode="hard_st",
    contact_time_bins=16,
    contact_causal_route=route,
    contact_planner_precursor=False,
    production_matrix_id=MATRIX_ID,
)
config["models"].update(
    contact_causal_route=route,
    production_matrix_id=MATRIX_ID,
    production_causal_route_id=route,
    contact_loss_weight=0.25,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_condition_mode="predicted",
    require_contact_targets=True,
    require_positive_contact_supervision=False,
    contact_supervision_rule_version=SUPERVISION_VERSION,
    production_checkpoint_steps=[0, 100, 200],
    causal_training_data_receipt=training_data_receipt,
    causal_training_data_receipt_sha256=training_data_sha256,
    causal_preflight_receipt=preflight_receipt,
    causal_preflight_receipt_sha256=preflight_sha256,
    causal_precursor_binding=precursor_binding,
    causal_checkpoint_receipt_base=checkpoint_receipt_base,
    matched_control_collector=collector_config,
)
config["train"].update(
    seed=training_data_receipt["split_seed"],
    checkpoint_interval=100,
    checkpoint_total_limit=-1,
)
if config["train"].get("mixed_precision") != "bf16" or int(
    config["train"].get("max_steps", 0)
) < 200:
    raise ValueError("causal production training requires bf16 and at least 200 steps")

config["contact_causal_main_integration"] = {
    "matrix_schema_version": matrix_manifest["schema_version"],
    "matrix_id": MATRIX_ID,
    "causal_route_id": route,
    "route_receipt": build_production_route_receipt(route),
    "precursor_binding": precursor_binding,
    "gate_path_record_only": str(gate_path),
    "matrix_path_record_only": str(matrix_path),
    "preflight_path_record_only": str(preflight_path),
    "training_data_path_record_only": str(training_data_path),
    "coverage_path_record_only": str(coverage_path),
    "collector_config_path_record_only": str(collector_path),
    "checkpoint_receipt_base_path_record_only": str(checkpoint_base_path),
    "contact_condition_mode": "predicted",
    "oracle_used": False,
    "production_data_admission": False,
    "allow_5b_training": False,
    "allow_server_training": False,
    "benefit_claimed": False,
}

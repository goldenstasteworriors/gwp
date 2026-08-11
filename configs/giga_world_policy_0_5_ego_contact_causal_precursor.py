"""Production-gated, route-neutral Ego contact-planner precursor config.

The v7 gate is checked before importing the heavyweight base config.  The
default all-false template cannot authorize import or training.
"""

import copy
import os

from world_action_model.contact_causal_production import SUPERVISION_VERSION
from world_action_model.contact_ego_causal_precursor import (
    PRECURSOR_ID,
    load_precursor_gate_from_environment,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)


precursor_gate = load_precursor_gate_from_environment()
objective = precursor_gate["objective_contract"]

from configs.giga_world_policy_0_5_ego_contact_pretrain import (  # noqa: E402
    config as base_config,
)


config = copy.deepcopy(base_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR", "outputs/gwp0_5_ego_contact_planner_precursor_q12"
)
config["models"]["transformer"].update(
    num_contact_tokens=12,
    contact_target_dim=SEGMENT_TARGET_DIM,
    contact_condition_dim=SEGMENT_TARGET_DIM,
    contact_bottleneck=False,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_semantic_bottleneck=False,
    contact_semantic_transport_mode="soft",
    contact_time_bins=16,
    contact_causal_route=None,
    contact_planner_precursor=True,
)
config["models"].update(
    contact_causal_route=None,
    contact_loss_weight=0.25,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_condition_mode="predicted",
    require_contact_targets=True,
    require_positive_contact_supervision=False,
    contact_supervision_rule_version=SUPERVISION_VERSION,
    ego_contact_precursor_objective=objective,
)
config["train"].update(
    max_steps=objective["fixed_save_step"],
    checkpoint_interval=objective["fixed_save_step"],
    checkpoint_total_limit=-1,
)
for dataset in config["dataloaders"]["train"]["data_or_config"]:
    dataset.update(
        contact_sidecar_strict=True,
        contact_sidecar_require_reviewed_v1_1=True,
        contact_label_origin="human_adjudicated",
        hand_active_contact_ground_truth=False,
        grip_contact_ground_truth=False,
        action_fields_record_only_not_scientific_io=True,
    )

config["ego_contact_causal_precursor"] = {
    "precursor_id": PRECURSOR_ID,
    "gate_version": precursor_gate["gate_version"],
    "objective_contract": objective,
    "action_path_frozen_before_optimizer": True,
    "action_path_scientific_input_output": False,
    "hand_active_is_contact_ground_truth": False,
    "grip_is_contact_ground_truth": False,
    "route_specific_transport_used": False,
    "route_specific_action_cache_used": False,
    "fixed_save_step": objective["fixed_save_step"],
    "production_data_admission": False,
    "allow_5b_training": False,
    "allow_server_training": False,
    "benefit_claimed": False,
}

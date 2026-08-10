"""Opt-in finite-capacity contact transport ablation.

This config is intentionally gated: it selects the 22-D segment schema and a
hard transport, but does not relax sidecar or asymmetric-25D data validation.
Do not launch 5B training until the gates in
``configs/contact_transport_ablation_matrix.json`` are satisfied.
"""

import copy
import os

from configs.giga_world_policy_0_5_sonic_g1_contact_pilot import config as base_config
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    contact_transport_capacity_metadata,
)


config = copy.deepcopy(base_config)

transport_mode = os.environ.get("GWP_CONTACT_TRANSPORT_MODE", "hard_st")
time_bins = int(os.environ.get("GWP_CONTACT_TIME_BINS", "16"))
num_contact_tokens = int(os.environ.get("GWP_CONTACT_TOKENS", "12"))

config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR",
    f"outputs/gwp0_5_sonic_g1_contact_{transport_mode}_b{time_bins}",
)
config["models"]["transformer"].update(
    num_contact_tokens=num_contact_tokens,
    contact_target_dim=SEGMENT_TARGET_DIM,
    contact_condition_dim=SEGMENT_TARGET_DIM,
    contact_bottleneck=True,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_semantic_bottleneck=True,
    contact_semantic_transport_mode=transport_mode,
    contact_time_bins=time_bins,
)
config["models"].update(
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    require_contact_targets=True,
)
config["contact_transport_metadata"] = dict(
    contact_transport_capacity_metadata(
        mode=transport_mode,
        time_bins=time_bins,
        num_queries=num_contact_tokens,
    )
)
config["contact_transport_metadata"].update(
    schema_version=SEGMENT_SCHEMA_VERSION,
    labels_gate="required",
    asymmetric25_gate="required_before_5b_robot_finetune",
)

"""Spec-only CPU-audited causal route controls; never production-admitted.

Select exactly one canonical variant with ``GWP_CONTACT_CAUSAL_ROUTE``.  This
config deliberately does not wire the production matched-control collector or
relax any training gate.
"""

import copy
import os

from configs.giga_world_policy_0_5_sonic_g1_finetune import config as base_config
from world_action_model.contact_route_contract import (
    ContactCausalRoute,
    contact_route_metadata,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)


_route_value = os.environ.get("GWP_CONTACT_CAUSAL_ROUTE", "").strip()
_allowed_routes = {
    ContactCausalRoute.AUX_ONLY_DIRECT_Q12,
    ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST,
    ContactCausalRoute.HARD_ST_SEMANTIC,
}
try:
    selected_route = ContactCausalRoute(_route_value)
except ValueError as exc:
    raise ValueError(
        "GWP_CONTACT_CAUSAL_ROUTE must explicitly select exactly one of "
        f"{tuple(route.value for route in sorted(_allowed_routes, key=lambda item: item.value))}"
    ) from exc
if selected_route not in _allowed_routes:
    raise ValueError(
        f"route {selected_route.value!r} is not part of the spec-only causal control matrix"
    )

config = copy.deepcopy(base_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR",
    f"outputs/spec_only_contact_causal_{selected_route.value}",
)

is_direct = selected_route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12
config["models"]["transformer"].update(
    num_contact_tokens=12,
    contact_target_dim=SEGMENT_TARGET_DIM,
    contact_condition_dim=SEGMENT_TARGET_DIM,
    contact_bottleneck=not is_direct,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_semantic_bottleneck=not is_direct,
    contact_semantic_transport_mode="hard_st",
    contact_time_bins=16,
    contact_causal_route=selected_route.value,
)
config["models"].update(
    contact_causal_route=selected_route.value,
    contact_loss_weight=0.25,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_condition_mode="predicted",
    require_contact_targets=True,
)
config["contact_causal_route_metadata"] = dict(
    contact_route_metadata(selected_route, num_queries=12, time_bins=16)
)
config["scientific_admission"] = {
    "implementation_status": "implemented_cpu_route_unadmitted",
    "allow_5b_training": False,
    "allow_server_training": False,
    "production_gate_v2_modified": False,
    "benefit_claimed": False,
}

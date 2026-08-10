"""Production-gated asymmetric25 + finite contact semantic finetune config.

Importing this config executes the artifact-bound preflight before the base
training config or dataloader can be constructed.  The repository's all-false
gate template intentionally makes the default path a hard No-Go.
"""

import copy
import os

from world_action_model.contact_asym25_preflight import (
    ContactAsym25PreflightError,
    preflight_from_environment,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    contact_transport_capacity_metadata,
)
from world_action_model.ego_action_schema import EGO_HAND_STATUS_SEMANTICS


_FIXED_ENV = {
    "GWP_CONTACT_TOKENS": "12",
    "GWP_CONTACT_TARGET_DIM": "22",
    "GWP_CONTACT_SCHEMA_VERSION": "contact_segment_v1_1",
    "GWP_CONTACT_TRANSPORT_MODE": "hard_st",
    "GWP_CONTACT_TIME_BINS": "16",
    "GWP_CONTACT_STRICT_BOTTLENECK": "1",
    "GWP_CONTACT_SEMANTIC_BOTTLENECK": "1",
    "GWP_CONTACT_SIDECAR_STRICT": "1",
    "GWP_CONTACT_REQUIRE_TARGETS": "1",
}
for _name, _expected in _FIXED_ENV.items():
    _actual = os.environ.get(_name)
    if _actual is not None and _actual.strip() != _expected:
        raise ContactAsym25PreflightError(
            f"production config fixes {_name}={_expected!r}; got override {_actual!r}"
        )

# This runs before importing the heavyweight base config.  All artifact and
# external review gates must be explicit and fingerprint-bound.
preflight_receipt = preflight_from_environment()

from configs.giga_world_policy_0_5_sonic_g1_asymmetric25_finetune import (  # noqa: E402
    config as base_config,
)


config = copy.deepcopy(base_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR",
    "outputs/gwp0_5_sonic_g1_asymmetric25_contact_hard_st_b16",
)

for dataset_config in config["dataloaders"]["train"]["data_or_config"]:
    if os.path.realpath(dataset_config["data_path"]) != preflight_receipt["data_root"]:
        raise ContactAsym25PreflightError(
            "base dataloader data_path differs from the preflighted asymmetric25 root"
        )
    dataset_config.update(
        contact_sidecar_dir=os.environ["GWP_CONTACT_SIDECAR_DIR"],
        contact_sidecar_pattern="episode_{episode_index:06d}.npz",
        contact_sidecar_cache_size=int(
            os.environ.get("GWP_CONTACT_SIDECAR_CACHE_SIZE", "8")
        ),
        contact_sidecar_strict=True,
        contact_sidecar_require_reviewed_v1_1=True,
        contact_sidecar_expected_meta_sha256=preflight_receipt[
            "sidecar_meta_sha256"
        ],
        contact_sidecar_expected_episode_sha256=preflight_receipt[
            "episode_npz_sha256"
        ],
    )

transform = config["dataloaders"]["train"]["transform"]
norm_paths = transform.get("norm_path")
if (
    not isinstance(norm_paths, list)
    or len(norm_paths) != 1
    or os.path.realpath(norm_paths[0]["path"]) != preflight_receipt["norm_stats_path"]
):
    raise ContactAsym25PreflightError(
        "base transform norm path differs from the preflighted normalization stats"
    )

config["models"]["transformer"].update(
    num_contact_tokens=12,
    contact_target_dim=SEGMENT_TARGET_DIM,
    contact_condition_dim=SEGMENT_TARGET_DIM,
    contact_bottleneck=True,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_semantic_bottleneck=True,
    contact_semantic_transport_mode="hard_st",
    contact_time_bins=16,
)
config["models"].update(
    contact_loss_weight=0.25,
    contact_schema_version=SEGMENT_SCHEMA_VERSION,
    contact_condition_mode="predicted",
    require_contact_targets=True,
)

config["contact_asym25_integration"] = {
    "preflight_receipt": preflight_receipt,
    "training_gate_path": str(
        os.path.realpath(os.environ["GWP_CONTACT_ASYM25_TRAINING_GATE"])
    ),
    "action_visible_route": (
        "state/ref/text planner -> canonical hard 22D -> 22-to-1024 encoder -> "
        "action; action has no direct reference or raw-planner-hidden route"
    ),
    "action_only_inference": (
        "prefix cache stores transported hard semantics; no contact labels are "
        "required at inference"
    ),
    "transport_capacity": dict(
        contact_transport_capacity_metadata(
            mode="hard_st",
            time_bins=16,
            num_queries=12,
        )
    ),
    "fixed_contract": {
        "num_contact_tokens": 12,
        "contact_target_dim": 22,
        "transport_mode": "hard_st",
        "time_bins": 16,
        "strict_semantic_bottleneck": True,
    },
    "cross_embodiment_ego_grip_semantics_required": EGO_HAND_STATUS_SEMANTICS,
    "scientific_gate_claimed": False,
    "physical_deployment": (
        "NO_GO: inference emits canonical25D only; this repository has no "
        "OmniPicker driver or calibrated canonical-to-device command mapper"
    ),
}

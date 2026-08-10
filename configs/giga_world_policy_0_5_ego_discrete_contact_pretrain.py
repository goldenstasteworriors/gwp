"""Production-gated Ego Stage1 with a hard discrete contact bottleneck.

The preflight runs before importing the permissive legacy/smoke config.  Fixed
semantic and transport choices below are not environment-tunable.
"""

import os


# Reject missing paths, false/stale gates, nonce mismatch, and an absent T5
# directory before importing NumPy, Torch, dataset, or model modules.  This
# lightweight pass never authorizes a launch; the full preflight below repeats
# these checks and verifies all artifact bytes.
from world_action_model.ego_stage1_gate_bootstrap import (  # noqa: E402
    minimal_ego_stage1_admission_precheck_from_environment,
)

# When the collector is enabled, reject launcher topology before importing the
# full preflight stack or inspecting T5. Keep this import inside the opt-in
# branch so the disabled collector does not read rank/world environment state.
if os.environ.get("GWP_MATCHED_CONTROL_OUTPUT_DIR", "").strip():
    from world_action_model.ego_stage1_gate_bootstrap import (
        minimal_matched_control_topology_precheck_from_environment,
    )

    minimal_matched_control_topology_precheck_from_environment()

minimal_ego_stage1_admission_precheck_from_environment()

import copy  # noqa: E402

from world_action_model.contact_ego_stage1_preflight import (  # noqa: E402
    ContactEgoStage1PreflightError,
    preflight_from_environment,
)
from world_action_model.contact_transition_schema import (  # noqa: E402
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    contact_transport_capacity_metadata,
)
from world_action_model.ego_action_schema import (  # noqa: E402
    EGO_CANONICAL_ACTION_SCHEMA_ID,
    EGO_HAND_STATUS_SEMANTICS,
)
from world_action_model.matched_control_config import (  # noqa: E402
    configure_matched_control_from_environment,
)


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
    "GWP_CONTACT_CONDITION_MODE": "predicted",
    "GWP_EGO_EPISODES": "",
    "GWP_T5_LOAD_FROM": "dir",
}
for _name, _expected in _FIXED_ENV.items():
    _actual = os.environ.get(_name)
    if _actual is not None and _actual.strip() != _expected:
        raise ContactEgoStage1PreflightError(
            f"production Ego Stage1 fixes {_name}={_expected!r}; got {_actual!r}"
        )

preflight_receipt = preflight_from_environment()

from configs.giga_world_policy_0_5_ego_contact_pretrain import (  # noqa: E402
    config as legacy_config,
)


config = copy.deepcopy(legacy_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR", "outputs/gwp0_5_ego_discrete_contact_stage1_hard_st_b16"
)
datasets = config["dataloaders"]["train"]["data_or_config"]
if len(datasets) != 1 or os.path.realpath(datasets[0]["data_path"]) != preflight_receipt["data_root"]:
    raise ContactEgoStage1PreflightError("legacy dataloader root differs from preflighted Ego root")
dataset = datasets[0]
dataset.pop("episodes", None)
if (
    dataset.get("t5_load_from") != "dir"
    or os.path.realpath(dataset.get("t5_cfg", {}).get("t5_embedding_dir", ""))
    != preflight_receipt["t5_embedding_dir"]
):
    raise ContactEgoStage1PreflightError(
        "legacy dataloader T5 source differs from the preflighted source"
    )
dataset.update(
    training_source_runtime_receipt=preflight_receipt[
        "training_source_runtime_receipt_record_only"
    ],
    t5_runtime_source_receipt=preflight_receipt[
        "t5_runtime_source_receipt_record_only"
    ],
    verified_private_cache_parent=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".gwp_stage1_private_hf_cache")
    ),
    verified_training_run_nonce=preflight_receipt["training_run_nonce"],
    contact_sidecar_dir=os.environ["GWP_CONTACT_SIDECAR_DIR"],
    contact_sidecar_pattern="episode_{episode_index:06d}.npz",
    contact_sidecar_strict=True,
    contact_sidecar_require_reviewed_v1_1=True,
    contact_sidecar_expected_meta_sha256=preflight_receipt["sidecar_meta_sha256"],
    contact_sidecar_expected_episode_sha256=preflight_receipt["episode_npz_sha256"],
)

transform = config["dataloaders"]["train"]["transform"]
norm_specs = transform.get("norm_path")
if (
    not isinstance(norm_specs, list)
    or len(norm_specs) != 1
    or os.path.realpath(norm_specs[0]["path"]) != preflight_receipt["norm_stats_path"]
):
    raise ContactEgoStage1PreflightError("transform norm differs from preflighted Ego norm")
norm_specs[0].update(
    sha256=preflight_receipt["norm_stats_sha256"],
    runtime_stat_record_only=preflight_receipt["norm_runtime_stat_record_only"],
)
transform.update(
    num_frames=48,
    fps=20,
    robotype_to_embodiment_id={"ego_human": 1},
    robotype_default_embodiment_id=1,
    model_action_dim=32,
    delta_mask_by_embodiment_id={"1": [False] * 20},
    ego_grip_semantics=EGO_HAND_STATUS_SEMANTICS,
    ego_grip_proxy_weight=0.25,
    require_exact_effective_dim=True,
    expected_action_schema_id=EGO_CANONICAL_ACTION_SCHEMA_ID,
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

# The single output-directory opt-in controls the collector. If enabled, its
# identity is locked to the exact portable identity produced by the preflight.
configure_matched_control_from_environment(
    config,
    variant_id="hard_st",
    expected_dataset_id=preflight_receipt["dataset_id"],
    **preflight_receipt["matched_control_expectations"],
)

config["ego_discrete_contact_stage1"] = {
    "preflight_receipt": preflight_receipt,
    "training_gate_path": os.path.realpath(os.environ["GWP_EGO_STAGE1_TRAINING_GATE"]),
    "fixed_contract": {
        "fps": 20,
        "frame_clock": "processed_action_20hz",
        "horizon_frames": 48,
        "num_contact_tokens": 12,
        "contact_target_dim": 22,
        "transport_mode": "hard_st",
        "time_bins": 16,
        "strict_semantic_bottleneck": True,
        "contact_condition_mode": "predicted",
        "source_action_dim": 20,
        "model_action_dim": 32,
    },
    "action_visible_route": (
        "state/ref/text planner -> canonical hard 22D -> 22-to-1024 encoder -> action; "
        "no direct reference or raw planner-hidden route"
    ),
    "action_only_inference": (
        "prefix cache stores predicted transported hard semantics; reviewed labels are training-only"
    ),
    "transport_capacity": dict(contact_transport_capacity_metadata(
        mode="hard_st", time_bins=16, num_queries=12
    )),
    "scientific_gate_claimed": False,
}

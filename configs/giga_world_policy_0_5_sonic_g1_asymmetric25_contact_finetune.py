"""Production-gated asymmetric25 + finite contact semantic finetune config.

Importing this config executes the artifact-bound preflight before the base
training config or dataloader can be constructed.  The repository's all-false
gate template intentionally makes the default path a hard No-Go.
"""

import copy
import os
from pathlib import Path

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
from world_action_model.contact_asym25_heldout_eval import (
    ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
    build_current_training_data_receipt,
    build_training_code_receipt,
    load_and_replay_preflight_receipt,
    load_completed_group_split,
    load_fixed_window_manifest,
    public_tree_receipt,
    snapshot_artifact_tree,
)


def _required_run_identity(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if len(value) < 16:
        raise ContactAsym25PreflightError(
            f"production training requires {name} with at least 16 characters"
        )
    return value


def _required_sha256(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContactAsym25PreflightError(
            f"production training requires lowercase SHA-256 in {name}"
        )
    return value


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

_persisted_preflight_path = os.environ.get(
    "GWP_CONTACT_ASYM25_PREFLIGHT_RECEIPT", ""
).strip()
_heldout_window_path = os.environ.get(
    "GWP_CONTACT_HELDOUT_WINDOW_MANIFEST", ""
).strip()
if not _persisted_preflight_path or not _heldout_window_path:
    raise ContactAsym25PreflightError(
        "production training requires GWP_CONTACT_ASYM25_PREFLIGHT_RECEIPT and "
        "GWP_CONTACT_HELDOUT_WINDOW_MANIFEST"
    )
_persisted_preflight, _persisted_preflight_sha256 = (
    load_and_replay_preflight_receipt(_persisted_preflight_path)
)
if _persisted_preflight != preflight_receipt:
    raise ContactAsym25PreflightError(
        "persisted preflight receipt differs from the config-import replay"
    )
_, _split_assignments = load_completed_group_split(
    preflight_receipt["split_manifest_path"],
    expected_sha256=preflight_receipt["split_manifest_sha256"],
    expected_episode_lengths=preflight_receipt["episode_lengths"],
)
_heldout_window_manifest, _heldout_window_sha256 = load_fixed_window_manifest(
    _heldout_window_path,
    preflight_receipt=preflight_receipt,
    preflight_receipt_sha256=_persisted_preflight_sha256,
    split_assignments=_split_assignments,
)
_repository_root = Path(__file__).resolve().parents[1]
_training_code_receipt = build_training_code_receipt(_repository_root)
_dataset_source_tree = snapshot_artifact_tree(preflight_receipt["data_root"])
current_training_data_receipt, current_training_data_receipt_sha256 = (
    build_current_training_data_receipt(
        preflight_receipt=preflight_receipt,
        preflight_receipt_sha256=_persisted_preflight_sha256,
        heldout_window_manifest_sha256=_heldout_window_sha256,
        training_code_receipt=_training_code_receipt,
        dataset_source_tree_receipt=public_tree_receipt(_dataset_source_tree),
    )
)

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
        episodes=preflight_receipt["train_episode_ids"],
        force_cache_sync=True,
        production_episode_filter_receipt=preflight_receipt[
            "training_data_receipt"
        ],
        production_episode_filter_receipt_sha256=preflight_receipt[
            "training_data_receipt_sha256"
        ],
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
_train_norm_receipt = preflight_receipt.get("train_norm_source_receipt")
if (
    not isinstance(_train_norm_receipt, dict)
    or _train_norm_receipt.get("split_seed") != preflight_receipt["split_seed"]
    or _train_norm_receipt.get("train_episode_ids")
    != preflight_receipt["train_episode_ids"]
    or _train_norm_receipt.get("production_split_receipt_sha256")
    != preflight_receipt.get("production_split_receipt_sha256")
):
    raise ContactAsym25PreflightError(
        "production transform norm receipt differs from its split seed/train episodes"
    )
# Re-hash the exact preflighted train-only norm bytes when each worker constructs
# its transform.  Legacy configs omit this opt-in SHA and retain their old path.
norm_paths[0]["sha256"] = preflight_receipt["norm_stats_sha256"]

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
    current_training_data_receipt=current_training_data_receipt,
    current_training_data_receipt_sha256=current_training_data_receipt_sha256,
    current_training_source_snapshot=_dataset_source_tree,
    production_checkpoint_steps=[0, 100, 200],
    production_training_run_nonce=_required_run_identity(
        "GWP_CONTACT_TRAINING_RUN_NONCE"
    ),
    production_experiment_family_id=_required_run_identity(
        "GWP_CONTACT_EXPERIMENT_FAMILY_ID"
    ),
    production_variant_id=_required_run_identity("GWP_CONTACT_VARIANT_ID"),
    production_matched_control_receipt_sha256=_required_sha256(
        "GWP_CONTACT_MATCHED_CONTROL_RECEIPT_SHA256"
    ),
    production_training_precision="bfloat16",
    production_action_schema_contract=ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
)

if int(config["train"].get("max_steps", 0)) < 200:
    raise ContactAsym25PreflightError(
        "production held-out contract requires max_steps >= 200"
    )
config["train"].update(
    checkpoint_interval=100,
    checkpoint_total_limit=-1,
)
if config["train"].get("mixed_precision") != "bf16":
    raise ContactAsym25PreflightError("production training requires bf16 precision")

config["contact_asym25_integration"] = {
    "preflight_receipt": preflight_receipt,
    "training_gate_path": str(
        os.path.realpath(os.environ["GWP_CONTACT_ASYM25_TRAINING_GATE"])
    ),
    "split_seed": preflight_receipt["split_seed"],
    "train_episode_ids": preflight_receipt["train_episode_ids"],
    "validation_episode_ids": preflight_receipt["validation_episode_ids"],
    "heldout_episode_ids": preflight_receipt["heldout_episode_ids"],
    "heldout_window_manifest": str(Path(_heldout_window_path).resolve()),
    "heldout_window_manifest_sha256": _heldout_window_sha256,
    "current_training_data_receipt_sha256": current_training_data_receipt_sha256,
    "production_split_receipt_sha256": preflight_receipt[
        "production_split_receipt_sha256"
    ],
    "train_norm_source_receipt_sha256": preflight_receipt[
        "train_norm_source_receipt_sha256"
    ],
    "norm_source_tree_receipt": preflight_receipt["norm_source_tree_receipt"],
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

"""Production-gated, route-neutral Ego contact-planner precursor config.

The v7 gate is checked before importing the heavyweight base config.  The
default all-false template cannot authorize import or training.
"""

import os
from pathlib import Path
import sys


_repository_root = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for _current, _directory_names, _file_names in os.walk(
    _repository_root, followlinks=False
):
    _relative = Path(_current).relative_to(_repository_root)
    _directory_names[:] = [
        _name
        for _name in _directory_names
        if not (_relative == Path(".") and _name in {".git", "outputs"})
    ]
    if "__pycache__" in _directory_names or any(
        _name.endswith(".pyc") for _name in _file_names
    ):
        raise RuntimeError("production Ego config forbids repository bytecode caches")

from world_action_model.contact_causal_gate_bootstrap import (
    bootstrap_ego_v7_stdlib,
)


# This reject-only pass verifies the external reviewer file and current HEAD
# code blobs before importing NumPy/Torch transitively through the full gate.
_stdlib_bootstrap = bootstrap_ego_v7_stdlib()

import copy  # noqa: E402

from world_action_model.contact_causal_production import (  # noqa: E402
    SUPERVISION_VERSION,
    current_head_source_tree_receipt,
    sha256_json,
)
from world_action_model.contact_ego_causal_precursor import (
    PRECURSOR_ID,
    load_precursor_gate_from_environment,
    replay_hardened_ego_stage1_from_environment,
)  # noqa: E402
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)  # noqa: E402


precursor_gate = load_precursor_gate_from_environment(
    bootstrap_receipt=_stdlib_bootstrap
)
_hardened_source_replay = replay_hardened_ego_stage1_from_environment(
    precursor_gate=precursor_gate,
    precursor_gate_sha256=_stdlib_bootstrap["gate_sha256"],
)
_hardened_receipt = _hardened_source_replay[
    "legacy_stage1_preflight_receipt"
]
objective = precursor_gate["objective_contract"]
_source_tree_receipt = current_head_source_tree_receipt(
    Path(__file__).resolve().parents[1],
    expected_commit=precursor_gate["expected_git_commit"],
)
if (
    sha256_json(_source_tree_receipt)
    != precursor_gate["artifact_fingerprints"]["current_head_source_tree_sha256"]
):
    raise ValueError("Ego precursor runtime source tree differs from the v7 gate")

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
    ego_contact_precursor_evidence_base={
        "gate_sha256": _stdlib_bootstrap["gate_sha256"],
        "root_checkpoint_bundle_sha256": precursor_gate["artifact_fingerprints"][
            "root_checkpoint_bundle_sha256"
        ],
        "dataset_source_tree_sha256": precursor_gate["artifact_fingerprints"][
            "dataset_source_tree_sha256"
        ],
        "sidecar_bundle_sha256": precursor_gate["artifact_fingerprints"][
            "sidecar_bundle_sha256"
        ],
        "training_code_head": precursor_gate["expected_git_commit"],
        "training_code_source_tree_sha256": precursor_gate["artifact_fingerprints"][
            "current_head_source_tree_sha256"
        ],
        "training_run_id": precursor_gate["training_run_id"],
        "training_run_nonce": precursor_gate["training_run_nonce"],
        "reviewer_acceptance_sha256": _stdlib_bootstrap["review_sha256"],
    },
)
config["train"].update(
    max_steps=objective["fixed_save_step"],
    checkpoint_interval=objective["fixed_save_step"],
    checkpoint_total_limit=-1,
    checkpoint_weights_only=True,
    checkpoint_safe_serialization=True,
)
_ego_datasets = config["dataloaders"]["train"]["data_or_config"]
if len(_ego_datasets) != 1:
    raise ValueError("Ego causal precursor requires exactly one verified dataset")
for dataset in _ego_datasets:
    if os.path.realpath(dataset["data_path"]) != _hardened_receipt["data_root"]:
        raise ValueError("Ego causal precursor dataset root differs from hardened evidence")
    if (
        dataset.get("t5_load_from") != "dir"
        or os.path.realpath(dataset.get("t5_cfg", {}).get("t5_embedding_dir", ""))
        != _hardened_receipt["t5_embedding_dir"]
    ):
        raise ValueError("Ego causal precursor T5 root differs from hardened evidence")
    if os.path.realpath(dataset.get("contact_sidecar_dir", "")) != os.path.realpath(
        _hardened_receipt["sidecar_dir"]
    ):
        raise ValueError(
            "Ego causal precursor sidecar root differs from hardened evidence"
        )
    dataset.pop("episodes", None)
    dataset.update(
        force_cache_sync=False,
        training_source_runtime_receipt=_hardened_receipt[
            "training_source_runtime_receipt_record_only"
        ],
        t5_runtime_source_receipt=_hardened_receipt[
            "t5_runtime_source_receipt_record_only"
        ],
        verified_private_cache_parent=str(
            (Path(__file__).resolve().parents[1] / ".gwp_op42_ego_private_hf_cache")
        ),
        verified_training_run_nonce=precursor_gate["training_run_nonce"],
        contact_sidecar_strict=True,
        contact_sidecar_require_reviewed_v1_1=True,
        contact_sidecar_expected_meta_sha256=_hardened_receipt[
            "sidecar_meta_sha256"
        ],
        contact_sidecar_expected_episode_sha256=_hardened_receipt[
            "episode_npz_sha256"
        ],
        contact_label_origin="human_adjudicated",
        contact_human_adjudicated_only=True,
        hand_active_contact_ground_truth=False,
        grip_contact_ground_truth=False,
        action_contact_ground_truth=False,
        other_proxy_contact_ground_truth=False,
        action_fields_record_only_not_scientific_io=True,
    )

config["ego_contact_causal_precursor"] = {
    "precursor_id": PRECURSOR_ID,
    "gate_version": precursor_gate["gate_version"],
    "objective_contract": objective,
    "hardened_ego_stage1_source_replay": _hardened_source_replay,
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

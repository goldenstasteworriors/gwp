"""Route-aware held-out result v3 and 27-checkpoint aggregate v5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    MODEL_ACTION_DIM,
)
from world_action_model.contact_causal_checkpoint import (
    validate_checkpoint_manifest_v3,
)
from world_action_model.contact_causal_preflight import (
    validate_preflight_receipt_v2,
)

from world_action_model.contact_causal_production import (
    AGGREGATE_VERSION,
    CAUSAL_ROUTES,
    CHECKPOINT_STEPS,
    CausalProductionError,
    EVALUATOR_VERSION,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    RESULT_VERSION,
    SEEDS,
    route_packet_kind,
    current_head_source_tree_receipt,
    sha256_json,
    strict_json_file,
    validate_27_checkpoint_matrix,
)


CAUSAL_EVALUATION_CONTRACT = {
    "mature_evaluator_version": "contact-asym25-heldout-evaluator/v1",
    "windows_per_val_seed": 256,
    "windows_per_test_seed": 256,
    "horizon_frames": 48,
    "image_frame_offsets": [0, 12, 24, 36, 48],
    "full_horizon_no_padding": True,
    "augmentation_enabled": False,
    "eval_crop_mode": "center",
    "model_dtype": "bfloat16",
    "batch_size": 1,
    "contact_query_count": 12,
    "contact_target_dim": 22,
    "contact_time_bins": 16,
    "action_schema_contract": {
        "schema_id": ASYMMETRIC_SCHEMA_ID,
        "model_action_dim": MODEL_ACTION_DIM,
        "physical_action_dim": ASYMMETRIC_ACTION_DIM,
        "groups": {
            "overall": list(range(25)),
            "wrist": list(range(18)),
            "grip": list(range(18, 25)),
            "left_overall": list(range(9)) + list(range(18, 24)),
            "left_wrist": list(range(9)),
            "left_grip": list(range(18, 24)),
            "right_overall": list(range(9, 18)) + [24],
            "right_wrist": list(range(9, 18)),
            "right_grip": [24],
        },
        "grip_active_mask_source": "all_physical_grip_dimensions",
    },
    "active_grip_action_dimensions": list(range(18, 25)),
    "action_target_access": "metrics_only_after_prediction",
    "contact_target_access": "additive_metrics_only_after_prediction",
    "oracle_contact_condition": False,
}


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalProductionError(f"{name} must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise CausalProductionError(f"{name} must be finite")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise CausalProductionError(f"{name} must be in [0,1]")
    return result


def _validate_finite_metric_tree(value: object, *, path: str = "metrics") -> None:
    if isinstance(value, Mapping):
        if not value:
            raise CausalProductionError(f"{path} must not be empty")
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise CausalProductionError(f"{path} metric key is invalid")
            _validate_finite_metric_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        if not value:
            raise CausalProductionError(f"{path} metric list must not be empty")
        for index, child in enumerate(value):
            _validate_finite_metric_tree(child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool):
        return
    lower = path.lower()
    unit = any(
        token in lower
        for token in (
            "accuracy", "fraction", "coverage", "precision", "recall",
            "f1", "validity", "utilization", "rate",
        )
    )
    number = _finite(value, path, unit_interval=unit)
    if any(token in lower for token in ("count", "sum", "mse", "mae", "nll", "brier", "entropy")) and number < 0:
        raise CausalProductionError(f"{path} must be non-negative")


def validate_route_evaluator_output(value: object, *, route: str) -> dict[str, Any]:
    output = _exact(
        value,
        {
            "evaluator_version",
            "matrix_id",
            "causal_route_id",
            "contact_aux_prediction_source",
            "contact_target_model_input",
            "contact_condition_mask_model_input",
            "oracle_used",
            "packet_kind",
            "packet_present",
            "packet_sha256",
            "semantic_aux_logits_sha256",
            "action_prediction_sha256",
            "action_metrics_contract_sha256",
            "packet_metrics",
        },
        "causal route evaluator output",
    )
    expected_kind = route_packet_kind(route)
    if (
        output["evaluator_version"] != EVALUATOR_VERSION
        or output["matrix_id"] != MATRIX_ID
        or output["causal_route_id"] != route
        or output["contact_aux_prediction_source"]
        != "shared_semantic_aux_decoder_logits"
        or output["contact_target_model_input"] is not False
        or output["contact_condition_mask_model_input"] is not False
        or output["oracle_used"] is not False
        or output["packet_kind"] != expected_kind
        or output["packet_present"] is not (route != "aux_only_direct_q12")
    ):
        raise CausalProductionError("route-aware evaluator output semantics differ")
    for field in (
        "semantic_aux_logits_sha256",
        "action_prediction_sha256",
        "action_metrics_contract_sha256",
    ):
        _sha(output[field], f"evaluator output {field}")
    if route == "aux_only_direct_q12":
        if output["packet_sha256"] is not None or output["packet_metrics"] != {
            "kind": "none_direct_reference",
            "semantic_field_metrics_allowed": False,
        }:
            raise CausalProductionError("direct evaluator must not fabricate a packet")
    elif route == "equal_bit_unstructured_hard_st":
        _sha(output["packet_sha256"], "equal-bit packet")
        metrics = _exact(
            output["packet_metrics"],
            {
                "kind",
                "codec_validity",
                "utilization",
                "entropy",
                "invalid_to_zero_rate",
                "semantic_field_metrics_allowed",
            },
            "equal-bit packet metrics",
        )
        if metrics["kind"] != "unstructured_code" or metrics["semantic_field_metrics_allowed"] is not False:
            raise CausalProductionError("equal-bit code must not be interpreted as contact semantics")
        _finite(metrics["codec_validity"], "equal-bit codec validity", unit_interval=True)
        _finite(metrics["utilization"], "equal-bit utilization", unit_interval=True)
        _finite(metrics["entropy"], "equal-bit entropy")
        _finite(
            metrics["invalid_to_zero_rate"],
            "equal-bit invalid-to-zero rate",
            unit_interval=True,
        )
    else:
        _sha(output["packet_sha256"], "semantic packet")
        metrics = _exact(
            output["packet_metrics"],
            {
                "kind",
                "grammar_validity",
                "field_accuracy",
                "semantic_field_metrics_allowed",
            },
            "semantic grammar packet metrics",
        )
        if metrics["kind"] != "semantic_grammar" or metrics["semantic_field_metrics_allowed"] is not True:
            raise CausalProductionError("semantic packet metrics differ")
        _finite(metrics["grammar_validity"], "semantic grammar validity", unit_interval=True)
        _finite(metrics["field_accuracy"], "semantic field accuracy", unit_interval=True)
    return output


def _artifact_entry(value: object, name: str) -> dict[str, Any]:
    entry = _exact(value, {"path_record_only", "sha256"}, name)
    if not isinstance(entry["path_record_only"], str) or not Path(
        entry["path_record_only"]
    ).is_absolute():
        raise CausalProductionError(f"{name} path must be absolute")
    _sha(entry["sha256"], f"{name} digest")
    return entry


def _tree_entry(value: object, name: str) -> dict[str, Any]:
    entry = _exact(
        value,
        {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
        name,
    )
    if (
        not isinstance(entry["path_record_only"], str)
        or not Path(entry["path_record_only"]).is_absolute()
        or isinstance(entry["file_count"], bool)
        or not isinstance(entry["file_count"], int)
        or entry["file_count"] <= 0
        or isinstance(entry["total_size_bytes"], bool)
        or not isinstance(entry["total_size_bytes"], int)
        or entry["total_size_bytes"] <= 0
    ):
        raise CausalProductionError(f"{name} tree receipt differs")
    _sha(entry["tree_sha256"], f"{name} tree digest")
    return entry


def validate_result_v3(
    value: object, *, verify_files: bool = True
) -> dict[str, Any]:
    result = _exact(
        value,
        {
            "result_version",
            "evaluator_version",
            "matrix_schema_version",
            "matrix_id",
            "seed",
            "causal_route_id",
            "causal_preflight_receipt",
            "hardened_preflight_receipt",
            "checkpoint_manifest",
            "base_model_artifact_tree",
            "fixed_window_manifest",
            "per_window_csv",
            "evaluator_source_receipt_sha256",
            "precursor_receipt_bytes_sha256",
            "action_metrics_contract_sha256",
            "contact_aux_metrics_contract_sha256",
            "evaluation_contract",
            "environment_receipt",
            "checkpoint_steps",
            "primary_checkpoint_step",
            "best_checkpoint_selection_allowed",
            "checkpoint_outputs",
            "oracle_used",
            "benefit_claimed",
        },
        "causal held-out result",
    )
    route = result["causal_route_id"]
    if (
        result["result_version"] != RESULT_VERSION
        or result["evaluator_version"] != EVALUATOR_VERSION
        or result["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or result["matrix_id"] != MATRIX_ID
        or result["seed"] not in SEEDS
        or route not in CAUSAL_ROUTES
        or result["checkpoint_steps"] != list(CHECKPOINT_STEPS)
        or result["primary_checkpoint_step"] != 200
        or result["best_checkpoint_selection_allowed"] is not False
        or result["oracle_used"] is not False
        or result["benefit_claimed"] is not False
    ):
        raise CausalProductionError("causal held-out result identity differs")
    for field in (
        "evaluator_source_receipt_sha256",
        "precursor_receipt_bytes_sha256",
        "action_metrics_contract_sha256",
        "contact_aux_metrics_contract_sha256",
    ):
        _sha(result[field], f"causal result {field}")
    if result["evaluation_contract"] != CAUSAL_EVALUATION_CONTRACT:
        raise CausalProductionError("causal evaluator fixed production contract differs")
    environment = _exact(
        result["environment_receipt"],
        {
            "python", "platform", "packages", "cuda_record_only",
            "evaluation_device", "model_dtype", "inference_steps",
            "deterministic_algorithms", "augmentation_enabled",
            "eval_crop_mode", "batch_size",
        },
        "causal evaluator environment",
    )
    packages = _exact(
        environment["packages"],
        {"numpy", "torch", "diffusers", "pandas"},
        "causal evaluator package versions",
    )
    cuda = _exact(
        environment["cuda_record_only"],
        {"torch_cuda_version", "cudnn_version", "device_count"},
        "causal evaluator CUDA record",
    )
    device = environment["evaluation_device"]
    if (
        not isinstance(environment["python"], str)
        or not isinstance(environment["platform"], str)
        or any(value is not None and not isinstance(value, str) for value in packages.values())
        or cuda["torch_cuda_version"] is None
        or not isinstance(cuda["torch_cuda_version"], str)
        or isinstance(cuda["cudnn_version"], bool)
        or not isinstance(cuda["cudnn_version"], int)
        or cuda["cudnn_version"] <= 0
        or isinstance(cuda["device_count"], bool)
        or not isinstance(cuda["device_count"], int)
        or cuda["device_count"] <= 0
        or not isinstance(device, str)
        or (
            device != "cuda"
            and not (
                device.startswith("cuda:")
                and device[5:].isdigit()
            )
        )
        or environment["model_dtype"] != "bfloat16"
        or isinstance(environment["inference_steps"], bool)
        or not isinstance(environment["inference_steps"], int)
        or environment["inference_steps"] <= 0
        or environment["deterministic_algorithms"] is not True
        or environment["augmentation_enabled"] is not False
        or environment["eval_crop_mode"] != "center"
        or environment["batch_size"] != 1
    ):
        raise CausalProductionError("causal evaluator environment contract differs")
    causal_preflight_entry = _artifact_entry(
        result["causal_preflight_receipt"], "causal preflight receipt"
    )
    hardened_entry = _artifact_entry(
        result["hardened_preflight_receipt"], "hardened preflight receipt"
    )
    checkpoint_manifest_entry = _artifact_entry(
        result["checkpoint_manifest"], "causal checkpoint manifest"
    )
    base_model_entry = _tree_entry(
        result["base_model_artifact_tree"], "causal evaluator base model"
    )
    windows_entry = _artifact_entry(
        result["fixed_window_manifest"], "fixed held-out windows"
    )
    csv_entry = _exact(
        result["per_window_csv"],
        {"path_record_only", "sha256", "row_count"},
        "causal per-window CSV",
    )
    if (
        not isinstance(csv_entry["path_record_only"], str)
        or not Path(csv_entry["path_record_only"]).is_absolute()
        or isinstance(csv_entry["row_count"], bool)
        or csv_entry["row_count"] != 3 * 2 * 256
    ):
        raise CausalProductionError("causal per-window CSV identity/count differs")
    _sha(csv_entry["sha256"], "causal per-window CSV")
    outputs = result["checkpoint_outputs"]
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise CausalProductionError("causal result needs exactly three checkpoint outputs")
    seen = set()
    for row in outputs:
        row = _exact(
            row,
            {
                "checkpoint_step",
                "checkpoint_acceptance_path_record_only",
                "checkpoint_acceptance_sha256",
                "fixed_window_set_sha256",
                "route_output",
                "action_metrics",
                "contact_aux_metrics",
            },
            "causal checkpoint result row",
        )
        step = row["checkpoint_step"]
        if isinstance(step, bool) or step not in CHECKPOINT_STEPS or step in seen:
            raise CausalProductionError("causal result checkpoint step is invalid/duplicate")
        seen.add(step)
        _sha(row["checkpoint_acceptance_sha256"], "result checkpoint acceptance")
        if (
            not isinstance(row["checkpoint_acceptance_path_record_only"], str)
            or not Path(row["checkpoint_acceptance_path_record_only"]).is_absolute()
            or row["fixed_window_set_sha256"] != windows_entry["sha256"]
        ):
            raise CausalProductionError("causal result fixed windows differ by checkpoint")
        validate_route_evaluator_output(row["route_output"], route=route)
        if (
            row["route_output"]["action_metrics_contract_sha256"]
            != result["action_metrics_contract_sha256"]
        ):
            raise CausalProductionError("causal result action metric contracts differ")
        if not isinstance(row["action_metrics"], dict) or not isinstance(row["contact_aux_metrics"], dict):
            raise CausalProductionError("causal result metric bundles must be mappings")
        _validate_finite_metric_tree(row["action_metrics"], path="action_metrics")
        _validate_finite_metric_tree(
            row["contact_aux_metrics"], path="contact_aux_metrics"
        )
    if seen != set(CHECKPOINT_STEPS):
        raise CausalProductionError("causal result checkpoint step set differs")
    if not verify_files:
        return result
    try:
        from world_action_model.contact_asym25_heldout_eval import (
            CHECKPOINT_STEPS as mature_checkpoint_steps,
            HORIZON_FRAMES as mature_horizon_frames,
            IMAGE_FRAME_OFFSETS as mature_image_frame_offsets,
            NUM_CONTACT_QUERIES as mature_contact_queries,
            TIME_BINS as mature_time_bins,
            WINDOWS_PER_SPLIT_PER_SEED as mature_windows_per_split,
            load_and_replay_preflight_receipt,
            load_completed_group_split,
            load_fixed_window_manifest,
            read_per_window_csv,
            reduce_metric_rows,
            public_tree_receipt,
            snapshot_artifact_tree,
        )
    except ImportError as exc:
        raise CausalProductionError(
            "mature held-out evaluator dependencies are unavailable"
        ) from exc
    if (
        tuple(mature_checkpoint_steps) != CHECKPOINT_STEPS
        or mature_windows_per_split != 256
        or mature_horizon_frames != 48
        or tuple(mature_image_frame_offsets) != (0, 12, 24, 36, 48)
        or mature_contact_queries != 12
        or mature_time_bins != 16
    ):
        raise CausalProductionError("mature held-out constants changed")
    causal_preflight, causal_preflight_sha256, _ = strict_json_file(
        causal_preflight_entry["path_record_only"], name="causal evaluator preflight"
    )
    validate_preflight_receipt_v2(causal_preflight)
    if causal_preflight_sha256 != causal_preflight_entry["sha256"]:
        raise CausalProductionError("causal evaluator preflight bytes changed")
    try:
        hardened, hardened_sha256 = load_and_replay_preflight_receipt(
            hardened_entry["path_record_only"]
        )
    except (ValueError, OSError) as exc:
        raise CausalProductionError("hardened held-out preflight replay failed") from exc
    if hardened_sha256 != hardened_entry["sha256"]:
        raise CausalProductionError("hardened held-out preflight bytes changed")
    training_data = causal_preflight["training_data_receipt"]
    mirrors = {
        "dataset_id": "dataset_id",
        "split_seed": "split_seed",
        "dataset_info_sha256": "dataset_info_sha256",
        "hardware_contract_sha256": "hardware_contract_sha256",
        "calibration_capture_sha256": "calibration_capture_sha256",
        "train_only_norm_sha256": "norm_stats_sha256",
        "sidecar_bundle_sha256": "sidecar_bundle_sha256",
        "split_manifest_sha256": "split_manifest_sha256",
    }
    if any(
        training_data[new_name] != hardened.get(old_name)
        for new_name, old_name in mirrors.items()
    ):
        raise CausalProductionError("causal evaluator differs from hardened v2 replay")
    if (
        result["seed"] != training_data["split_seed"]
        or result["evaluator_source_receipt_sha256"]
        != causal_preflight["current_head_source_tree_sha256"]
    ):
        raise CausalProductionError("causal evaluator preflight identity differs")
    source_receipt = current_head_source_tree_receipt(
        Path(__file__).resolve().parents[1],
        expected_commit=causal_preflight["git_commit"],
    )
    if sha256_json(source_receipt) != result["evaluator_source_receipt_sha256"]:
        raise CausalProductionError("causal evaluator runtime differs from current HEAD")
    split_path = Path(hardened["split_manifest_path"])
    try:
        _, assignments = load_completed_group_split(
            split_path,
            expected_sha256=hardened["split_manifest_sha256"],
            expected_episode_lengths=hardened["episode_lengths"],
        )
        windows, windows_sha256 = load_fixed_window_manifest(
            windows_entry["path_record_only"],
            preflight_receipt=hardened,
            preflight_receipt_sha256=hardened_sha256,
            split_assignments=assignments,
        )
    except (ValueError, OSError) as exc:
        raise CausalProductionError("mature fixed held-out windows replay failed") from exc
    if windows_sha256 != windows_entry["sha256"]:
        raise CausalProductionError("causal fixed-window bytes changed")
    try:
        current_base_model = public_tree_receipt(
            snapshot_artifact_tree(base_model_entry["path_record_only"])
        )
    except (ValueError, OSError) as exc:
        raise CausalProductionError("causal base-model artifact replay failed") from exc
    if current_base_model != base_model_entry:
        raise CausalProductionError("causal base-model artifact tree changed")
    assigned = assignments[result["seed"]]
    expected_sets = {
        split: sorted(episode for episode, value in assigned.items() if value == split)
        for split in ("train", "val", "test")
    }
    if (
        expected_sets["train"] != sorted(training_data["train_episode_ids"])
        or expected_sets["val"] != sorted(training_data["validation_episode_ids"])
        or expected_sets["test"] != sorted(training_data["heldout_episode_ids"])
    ):
        raise CausalProductionError("causal evaluator split leaks/differs from hardened v2")
    manifest, manifest_sha256, _ = strict_json_file(
        checkpoint_manifest_entry["path_record_only"],
        name="causal evaluator checkpoint manifest",
    )
    if manifest_sha256 != checkpoint_manifest_entry["sha256"]:
        raise CausalProductionError("causal checkpoint manifest bytes changed")
    manifest, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
    if (
        manifest["seed"] != result["seed"]
        or manifest["causal_route_id"] != route
        or manifest["precursor_receipt_bytes_sha256"]
        != result["precursor_receipt_bytes_sha256"]
    ):
        raise CausalProductionError("causal result checkpoint manifest binding differs")
    try:
        rows = read_per_window_csv(
            csv_entry["path_record_only"], expected_sha256=csv_entry["sha256"]
        )
    except (ValueError, OSError) as exc:
        raise CausalProductionError("causal additive per-window CSV failed replay") from exc
    if len(rows) != csv_entry["row_count"]:
        raise CausalProductionError("causal per-window CSV row count changed")
    window_identities = {
        (
            window["window_id"], window["split"], window["episode_id"],
            window["frame_index"], window["global_index"],
            window["action_noise_seed"],
        )
        for window in windows["windows"]
    }
    output_by_step = {row["checkpoint_step"]: row for row in outputs}
    for step in CHECKPOINT_STEPS:
        receipt, acceptance = accepted[step]
        output = output_by_step[step]
        if (
            output["checkpoint_acceptance_path_record_only"]
            != manifest["checkpoints"][CHECKPOINT_STEPS.index(step)][
                "independent_acceptance_path"
            ]
            or output["checkpoint_acceptance_sha256"] != sha256_json(acceptance)
        ):
            raise CausalProductionError("causal result independent acceptance changed")
        selected = [row for row in rows if row["checkpoint_step"] == step]
        identities = {
            (
                row["window_id"], row["split"], row["episode_id"],
                row["frame_index"], row["global_index"], row["action_noise_seed"],
            )
            for row in selected
        }
        if (
            len(selected) != 512
            or identities != window_identities
            or any(row["split_seed"] != result["seed"] for row in selected)
            or any(
                row["checkpoint_bundle_sha256"]
                != receipt["checkpoint_artifact_hashes"]["whole_bundle_sha256"]
                for row in selected
            )
        ):
            raise CausalProductionError("causal result window/checkpoint binding differs")
        recomputed_action = {}
        recomputed_contact = {}
        for split in ("val", "test"):
            split_rows = [row for row in selected if row["split"] == split]
            if len(split_rows) != 256:
                raise CausalProductionError("causal result split does not have 256 windows")
            try:
                reduced = reduce_metric_rows(split_rows)
            except (ValueError, OSError) as exc:
                raise CausalProductionError("causal additive metric reduction failed") from exc
            recomputed_action[split] = reduced["action"]
            recomputed_contact[split] = reduced["contact"]
        if (
            output["action_metrics"] != recomputed_action
            or output["contact_aux_metrics"] != recomputed_contact
        ):
            raise CausalProductionError("causal result metrics differ from additive CSV")
    return result


def validate_aggregate_v5(
    value: object,
    *,
    evidence: Mapping[tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]],
    verify_files: bool = True,
) -> dict[str, Any]:
    if verify_files:
        try:
            from world_action_model.contact_asym25_heldout_eval import (
                read_per_window_csv,
                reduce_metric_rows,
            )
        except ImportError as exc:
            raise CausalProductionError(
                "mature aggregate evaluator dependencies are unavailable"
            ) from exc
    aggregate = _exact(
        value,
        {
            "aggregate_version",
            "matrix_schema_version",
            "matrix_id",
            "seeds",
            "causal_routes",
            "checkpoint_steps",
            "checkpoint_acceptance_count",
            "result_file_by_seed_route",
            "aggregate_metrics_by_route_step_split",
            "matrix_contract",
            "primary_checkpoint_step",
            "oracle_included",
            "production_data_admission",
            "allow_5b_training",
            "allow_server_training",
            "benefit_claimed",
        },
        "causal held-out aggregate",
    )
    matrix_contract = validate_27_checkpoint_matrix(
        evidence, verify_files=verify_files
    )
    if (
        aggregate["aggregate_version"] != AGGREGATE_VERSION
        or aggregate["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or aggregate["matrix_id"] != MATRIX_ID
        or aggregate["seeds"] != list(SEEDS)
        or aggregate["causal_routes"] != list(CAUSAL_ROUTES)
        or aggregate["checkpoint_steps"] != list(CHECKPOINT_STEPS)
        or aggregate["checkpoint_acceptance_count"] != 27
        or aggregate["matrix_contract"] != matrix_contract
        or aggregate["primary_checkpoint_step"] != 200
        or aggregate["oracle_included"] is not False
        or aggregate["production_data_admission"] is not False
        or aggregate["allow_5b_training"] is not False
        or aggregate["allow_server_training"] is not False
        or aggregate["benefit_claimed"] is not False
    ):
        raise CausalProductionError("causal aggregate identity/admission differs")
    results = aggregate["result_file_by_seed_route"]
    expected_keys = [f"{seed}/{route}" for seed in SEEDS for route in CAUSAL_ROUTES]
    if not isinstance(results, dict) or list(results) != expected_keys:
        raise CausalProductionError("causal aggregate result set/order differs")
    result_rows: dict[str, list[dict[str, Any]]] = {}
    result_objects: dict[str, dict[str, Any]] = {}
    for key, raw_entry in results.items():
        entry = _exact(
            raw_entry,
            {"path_record_only", "sha256", "per_window_csv_sha256"},
            f"causal aggregate result {key}",
        )
        if not isinstance(entry["path_record_only"], str) or not Path(
            entry["path_record_only"]
        ).is_absolute():
            raise CausalProductionError("causal aggregate result path must be absolute")
        _sha(entry["sha256"], f"causal aggregate result {key}")
        _sha(entry["per_window_csv_sha256"], f"causal aggregate CSV {key}")
        if not verify_files:
            continue
        result, digest, _ = strict_json_file(
            entry["path_record_only"], name=f"causal held-out result {key}"
        )
        if digest != entry["sha256"]:
            raise CausalProductionError("causal aggregate result bytes changed")
        validate_result_v3(result, verify_files=True)
        seed_text, route = key.split("/", 1)
        if (
            result["seed"] != int(seed_text)
            or result["causal_route_id"] != route
            or result["per_window_csv"]["sha256"]
            != entry["per_window_csv_sha256"]
        ):
            raise CausalProductionError("causal aggregate result key/binding differs")
        try:
            rows = read_per_window_csv(
                result["per_window_csv"]["path_record_only"],
                expected_sha256=entry["per_window_csv_sha256"],
            )
        except (ValueError, OSError) as exc:
            raise CausalProductionError("causal aggregate CSV replay failed") from exc
        result_rows[key] = rows
        result_objects[key] = result
        manifest, _, _ = strict_json_file(
            result["checkpoint_manifest"]["path_record_only"],
            name=f"causal aggregate checkpoint manifest {key}",
        )
        _, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
        for step in CHECKPOINT_STEPS:
            if accepted[step] != evidence[(int(seed_text), route, step)]:
                raise CausalProductionError(
                    "causal aggregate evidence differs from result checkpoint manifest"
                )
    metrics = aggregate["aggregate_metrics_by_route_step_split"]
    expected_metric_keys = [
        f"{route}/{step}/{split}"
        for route in CAUSAL_ROUTES
        for step in CHECKPOINT_STEPS
        for split in ("val", "test")
    ]
    if not isinstance(metrics, dict) or list(metrics) != expected_metric_keys:
        raise CausalProductionError("causal aggregate metric matrix differs")
    for key, metric in metrics.items():
        metric = _exact(metric, {"action", "contact_aux"}, f"aggregate metric {key}")
        _validate_finite_metric_tree(metric, path=f"aggregate.{key}")
    if verify_files:
        for seed in SEEDS:
            same_seed = [result_objects[f"{seed}/{route}"] for route in CAUSAL_ROUTES]
            if len({row["hardened_preflight_receipt"]["sha256"] for row in same_seed}) != 1:
                raise CausalProductionError("same-seed routes use different hardened preflight")
            if len({row["fixed_window_manifest"]["sha256"] for row in same_seed}) != 1:
                raise CausalProductionError("same-seed routes use different held-out windows")
            if len({row["precursor_receipt_bytes_sha256"] for row in same_seed}) != 1:
                raise CausalProductionError("same-seed routes use different precursor bytes")
            if len({sha256_json(row["evaluation_contract"]) for row in same_seed}) != 1:
                raise CausalProductionError("same-seed evaluator contracts differ")
            if len({sha256_json(row["environment_receipt"]) for row in same_seed}) != 1:
                raise CausalProductionError("same-seed evaluator environments differ")
        all_results = list(result_objects.values())
        if len({sha256_json(row["environment_receipt"]) for row in all_results}) != 1:
            raise CausalProductionError("nine evaluator environments differ")
        if len({sha256_json(row["base_model_artifact_tree"]) for row in all_results}) != 1:
            raise CausalProductionError("nine evaluator base-model trees differ")
        recomputed = {}
        for route in CAUSAL_ROUTES:
            for step in CHECKPOINT_STEPS:
                for split in ("val", "test"):
                    selected = [
                        row
                        for seed in SEEDS
                        for row in result_rows[f"{seed}/{route}"]
                        if row["checkpoint_step"] == step and row["split"] == split
                    ]
                    if len(selected) != 3 * 256:
                        raise CausalProductionError("aggregate metric source row count differs")
                    try:
                        reduced = reduce_metric_rows(selected)
                    except (ValueError, OSError) as exc:
                        raise CausalProductionError("aggregate additive reduction failed") from exc
                    recomputed[f"{route}/{step}/{split}"] = {
                        "action": reduced["action"],
                        "contact_aux": reduced["contact"],
                    }
        if metrics != recomputed:
            raise CausalProductionError("aggregate metrics differ from nine hashed CSVs")
    return aggregate


def build_aggregate_v5(
    *,
    result_path_by_seed_route: Mapping[str, str | Path],
    evidence: Mapping[tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]],
    verify_files: bool = True,
) -> dict[str, Any]:
    try:
        from world_action_model.contact_asym25_heldout_eval import (
            read_per_window_csv,
            reduce_metric_rows,
        )
    except ImportError as exc:
        raise CausalProductionError(
            "mature aggregate evaluator dependencies are unavailable"
        ) from exc
    matrix_contract = validate_27_checkpoint_matrix(
        evidence, verify_files=verify_files
    )
    expected_keys = [f"{seed}/{route}" for seed in SEEDS for route in CAUSAL_ROUTES]
    if list(result_path_by_seed_route) != expected_keys:
        raise CausalProductionError("aggregate builder requires ordered nine-result matrix")
    entries = {}
    rows_by_key = {}
    for key, path in result_path_by_seed_route.items():
        result, digest, _ = strict_json_file(path, name=f"causal held-out result {key}")
        validate_result_v3(result, verify_files=True)
        entries[key] = {
            "path_record_only": str(Path(path).expanduser().absolute()),
            "sha256": digest,
            "per_window_csv_sha256": result["per_window_csv"]["sha256"],
        }
        try:
            rows_by_key[key] = read_per_window_csv(
                result["per_window_csv"]["path_record_only"],
                expected_sha256=result["per_window_csv"]["sha256"],
            )
        except (ValueError, OSError) as exc:
            raise CausalProductionError("aggregate builder CSV replay failed") from exc
    metrics = {}
    for route in CAUSAL_ROUTES:
        for step in CHECKPOINT_STEPS:
            for split in ("val", "test"):
                selected = [
                    row
                    for seed in SEEDS
                    for row in rows_by_key[f"{seed}/{route}"]
                    if row["checkpoint_step"] == step and row["split"] == split
                ]
                reduced = reduce_metric_rows(selected)
                metrics[f"{route}/{step}/{split}"] = {
                    "action": reduced["action"],
                    "contact_aux": reduced["contact"],
                }
    aggregate = {
        "aggregate_version": AGGREGATE_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "seeds": list(SEEDS),
        "causal_routes": list(CAUSAL_ROUTES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_acceptance_count": 27,
        "result_file_by_seed_route": entries,
        "aggregate_metrics_by_route_step_split": metrics,
        "matrix_contract": matrix_contract,
        "primary_checkpoint_step": 200,
        "oracle_included": False,
        "production_data_admission": False,
        "allow_5b_training": False,
        "allow_server_training": False,
        "benefit_claimed": False,
    }
    validate_aggregate_v5(aggregate, evidence=evidence, verify_files=verify_files)
    return aggregate

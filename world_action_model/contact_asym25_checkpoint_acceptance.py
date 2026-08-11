"""Strict checkpoint-bundle and independent acceptance contracts.

The trainer may create a bundle manifest and a training receipt, but it never
creates the reviewer acceptance consumed by held-out evaluation.  This keeps
artifact production and acceptance in separate processes/output locations.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any


BUNDLE_MANIFEST_VERSION = "contact-asym25-checkpoint-bundle/v2"
TRAINING_RECEIPT_VERSION = "contact-asym25-checkpoint-training-receipt/v3"
ACCEPTANCE_VERSION = "contact-asym25-checkpoint-independent-acceptance/v2"
PRODUCTION_STEPS = (0, 100, 200)
PRODUCTION_RUN_CONTRACT_KEYS = {
    "initial_weights_bundle_sha256",
    "base_model_bundle_sha256",
    "training_code_receipt_sha256",
    "dataset_source_tree_sha256",
    "runtime_environment_receipt_sha256",
    "optimizer_config_sha256",
    "scheduler_config_sha256",
    "training_config_sha256",
    "training_config_receipt",
    "stable_training_config_sha256",
    "control_config_sha256",
    "preflight_receipt_sha256",
    "train_window_receipt_sha256",
    "precision",
    "action_schema_contract_sha256",
    "action_schema_id",
    "contact_schema_version",
    "contact_query_count",
    "contact_target_dim",
    "contact_time_bins",
    "dataset_id",
    "dataset_info_sha256",
    "hardware_contract_sha256",
    "norm_stats_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXCLUDED_EVIDENCE_FILES = frozenset(
    {
        "checkpoint_bundle_manifest.json",
        "checkpoint_training_receipt.json",
        "checkpoint_reviewer_acceptance.json",
        "current_training_data_receipt.json",
    }
)


class CheckpointAcceptanceError(ValueError):
    """Raised when checkpoint evidence is incomplete or mutable."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CheckpointAcceptanceError(f"{name} must be lowercase SHA-256")
    return value


def _require_exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CheckpointAcceptanceError(f"{name} schema mismatch")
    return value


def strict_json_file(path: str | Path, name: str) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().absolute()
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointAcceptanceError(f"{name} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            payload = b""
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                payload += chunk
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise CheckpointAcceptanceError(f"cannot read {name}: {path}") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise CheckpointAcceptanceError(f"{name} changed while being read")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
        _reject_nonfinite_numbers(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointAcceptanceError(f"invalid strict JSON in {name}") from exc
    if not isinstance(value, dict):
        raise CheckpointAcceptanceError(f"{name} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_nonfinite_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite/overflow JSON number is forbidden")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite_numbers(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite_numbers(child)


def _hash_regular_file(path: Path) -> tuple[str, int]:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise CheckpointAcceptanceError(f"checkpoint entry is not regular: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise CheckpointAcceptanceError(f"checkpoint entry changed while hashing: {path}")
    return digest.hexdigest(), size


def build_bundle_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir).expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise CheckpointAcceptanceError("checkpoint path must be a real directory")
    rows: list[dict[str, Any]] = []
    for current_root, directories, files in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directories:
            candidate = current / name
            if candidate.is_symlink():
                raise CheckpointAcceptanceError(
                    f"checkpoint bundle contains symlink directory: {candidate}"
                )
        for name in files:
            relative = (current / name).relative_to(root).as_posix()
            if relative in _EXCLUDED_EVIDENCE_FILES:
                continue
            digest, size = _hash_regular_file(current / name)
            rows.append(
                {"logical_path": relative, "sha256": digest, "size_bytes": size}
            )
    rows.sort(key=lambda row: row["logical_path"])
    if not rows or not any(row["logical_path"] == "config.json" for row in rows):
        raise CheckpointAcceptanceError("checkpoint bundle must contain config.json")
    core = {
        "files": rows,
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
    }
    return {
        "manifest_version": BUNDLE_MANIFEST_VERSION,
        **core,
        "bundle_sha256": sha256_json(core),
    }


def validate_bundle_manifest(
    value: object, checkpoint_dir: str | Path
) -> dict[str, Any]:
    manifest = _require_exact(
        value,
        {
            "manifest_version",
            "files",
            "file_count",
            "total_size_bytes",
            "bundle_sha256",
        },
        "checkpoint bundle manifest",
    )
    if manifest["manifest_version"] != BUNDLE_MANIFEST_VERSION:
        raise CheckpointAcceptanceError("checkpoint bundle manifest version mismatch")
    actual = build_bundle_manifest(checkpoint_dir)
    if manifest != actual:
        raise CheckpointAcceptanceError("checkpoint bundle bytes differ from manifest")
    return manifest


def validate_training_receipt(value: object) -> dict[str, Any]:
    receipt = _require_exact(
        value,
        {
            "receipt_version",
            "checkpoint_step",
            "checkpoint_bundle_manifest_sha256",
            "checkpoint_bundle_sha256",
            "parent_checkpoint_bundle_sha256",
            "initial_weights_bundle_sha256",
            "base_model_bundle_sha256",
            "training_run_nonce",
            "training_run_id",
            "experiment_family_id",
            "variant_id",
            "transport_variant_id",
            "matched_control_receipt_sha256",
            "matched_control_external_acceptance_sha256",
            "production_run_contract",
            "production_run_contract_sha256",
            "runtime_environment_receipt",
            "runtime_environment_receipt_sha256",
            "optimizer_config_sha256",
            "scheduler_config_sha256",
            "training_config_sha256",
            "training_config_receipt",
            "stable_training_config_sha256",
            "control_config_sha256",
            "train_window_receipt_sha256",
            "split_seed",
            "training_precision",
            "contact_query_count",
            "contact_target_dim",
            "contact_time_bins",
            "action_schema_contract",
            "production_preflight_receipt_sha256",
            "heldout_window_manifest_sha256",
            "training_gate_sha256",
            "current_training_data_receipt",
            "current_training_data_receipt_sha256",
            "training_code_receipt_sha256",
            "dataset_source_tree_sha256",
            "initial_weights_tree_receipt",
        },
        "checkpoint training receipt",
    )
    if receipt["receipt_version"] != TRAINING_RECEIPT_VERSION:
        raise CheckpointAcceptanceError("checkpoint training receipt version mismatch")
    for field in (
        "checkpoint_bundle_manifest_sha256",
        "checkpoint_bundle_sha256",
        "parent_checkpoint_bundle_sha256",
        "initial_weights_bundle_sha256",
        "base_model_bundle_sha256",
        "production_preflight_receipt_sha256",
        "heldout_window_manifest_sha256",
        "training_gate_sha256",
        "current_training_data_receipt_sha256",
        "training_code_receipt_sha256",
        "dataset_source_tree_sha256",
        "matched_control_receipt_sha256",
        "matched_control_external_acceptance_sha256",
        "production_run_contract_sha256",
        "runtime_environment_receipt_sha256",
        "optimizer_config_sha256",
        "scheduler_config_sha256",
        "training_config_sha256",
        "stable_training_config_sha256",
        "control_config_sha256",
        "train_window_receipt_sha256",
    ):
        _require_sha256(receipt[field], field)
    if (
        isinstance(receipt["checkpoint_step"], bool)
        or not isinstance(receipt["checkpoint_step"], int)
        or receipt["checkpoint_step"] not in PRODUCTION_STEPS
        or receipt["split_seed"] not in {6666, 7777, 8888}
        or receipt["training_precision"] != "bfloat16"
        or receipt["contact_query_count"] != 12
        or receipt["contact_target_dim"] != 22
        or receipt["contact_time_bins"] != 16
    ):
        raise CheckpointAcceptanceError("checkpoint production shape/precision differs")
    for field in (
        "training_run_nonce",
        "training_run_id",
        "experiment_family_id",
        "variant_id",
    ):
        if not isinstance(receipt[field], str) or len(receipt[field]) < 16:
            raise CheckpointAcceptanceError(f"invalid checkpoint {field}")
    if receipt["transport_variant_id"] not in {
        "soft_semantic",
        "hard_st",
        "hard_detached",
    }:
        raise CheckpointAcceptanceError("invalid checkpoint transport_variant_id")
    schema = _require_exact(
        receipt["action_schema_contract"],
        {
            "schema_id",
            "model_action_dim",
            "physical_action_dim",
            "groups",
            "grip_active_mask_source",
        },
        "checkpoint action schema contract",
    )
    if not isinstance(schema["groups"], dict) or not schema["groups"]:
        raise CheckpointAcceptanceError("checkpoint action groups are empty")
    run_contract = _require_exact(
        receipt["production_run_contract"],
        PRODUCTION_RUN_CONTRACT_KEYS,
        "checkpoint production run contract",
    )
    runtime_environment = receipt["runtime_environment_receipt"]
    training_config = receipt["training_config_receipt"]
    if not isinstance(runtime_environment, dict) or not runtime_environment:
        raise CheckpointAcceptanceError("runtime environment receipt is empty")
    if (
        not isinstance(training_config, dict)
        or not training_config
        or isinstance(training_config.get("seed"), bool)
        or training_config.get("seed") != receipt["split_seed"]
        or sha256_json(training_config) != receipt["training_config_sha256"]
    ):
        raise CheckpointAcceptanceError("checkpoint training config receipt differs")
    normalized_training_config = dict(training_config)
    normalized_training_config["seed"] = "__CONTACT_ASYM25_SPLIT_SEED__"
    if sha256_json(normalized_training_config) != receipt[
        "stable_training_config_sha256"
    ]:
        raise CheckpointAcceptanceError(
            "checkpoint stable training config digest differs"
        )
    if (
        sha256_json(run_contract) != receipt["production_run_contract_sha256"]
        or sha256_json(runtime_environment)
        != receipt["runtime_environment_receipt_sha256"]
    ):
        raise CheckpointAcceptanceError("checkpoint runtime/run contract digest differs")
    for field in PRODUCTION_RUN_CONTRACT_KEYS:
        if field.endswith("_sha256"):
            _require_sha256(run_contract[field], f"run contract {field}")
    if (
        run_contract["precision"] != "bfloat16"
        or run_contract["action_schema_id"] != schema["schema_id"]
        or run_contract["contact_schema_version"] != "contact_segment_v1_1"
        or run_contract["contact_query_count"] != 12
        or run_contract["contact_target_dim"] != 22
        or run_contract["contact_time_bins"] != 16
        or not isinstance(run_contract["dataset_id"], str)
        or not run_contract["dataset_id"].strip()
    ):
        raise CheckpointAcceptanceError("checkpoint production run contract differs")
    duplicate_contract_fields = {
        "initial_weights_bundle_sha256": "initial_weights_bundle_sha256",
        "base_model_bundle_sha256": "base_model_bundle_sha256",
        "training_code_receipt_sha256": "training_code_receipt_sha256",
        "dataset_source_tree_sha256": "dataset_source_tree_sha256",
        "runtime_environment_receipt_sha256": "runtime_environment_receipt_sha256",
        "optimizer_config_sha256": "optimizer_config_sha256",
        "scheduler_config_sha256": "scheduler_config_sha256",
        "training_config_sha256": "training_config_sha256",
        "stable_training_config_sha256": "stable_training_config_sha256",
        "control_config_sha256": "control_config_sha256",
        "production_preflight_receipt_sha256": "preflight_receipt_sha256",
        "train_window_receipt_sha256": "train_window_receipt_sha256",
        "training_precision": "precision",
        "contact_query_count": "contact_query_count",
        "contact_target_dim": "contact_target_dim",
        "contact_time_bins": "contact_time_bins",
    }
    if any(
        receipt[receipt_field] != run_contract[contract_field]
        for receipt_field, contract_field in duplicate_contract_fields.items()
    ):
        raise CheckpointAcceptanceError("checkpoint run contract mirror differs")
    if run_contract["training_config_receipt"] != training_config:
        raise CheckpointAcceptanceError(
            "checkpoint run contract training config differs"
        )
    if (
        receipt["base_model_bundle_sha256"]
        != receipt["initial_weights_bundle_sha256"]
        or run_contract["action_schema_contract_sha256"] != sha256_json(schema)
    ):
        raise CheckpointAcceptanceError("checkpoint base/action contract differs")
    if sha256_json(receipt["current_training_data_receipt"]) != receipt[
        "current_training_data_receipt_sha256"
    ]:
        raise CheckpointAcceptanceError("embedded current training receipt differs")
    initial = _require_exact(
        receipt["initial_weights_tree_receipt"],
        {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
        "initial weights tree receipt",
    )
    if initial["tree_sha256"] != receipt["initial_weights_bundle_sha256"]:
        raise CheckpointAcceptanceError("initial weights tree binding differs")
    if (
        not isinstance(initial["path_record_only"], str)
        or not Path(initial["path_record_only"]).is_absolute()
        or isinstance(initial["file_count"], bool)
        or not isinstance(initial["file_count"], int)
        or initial["file_count"] <= 0
        or isinstance(initial["total_size_bytes"], bool)
        or not isinstance(initial["total_size_bytes"], int)
        or initial["total_size_bytes"] <= 0
    ):
        raise CheckpointAcceptanceError("initial weights tree receipt is invalid")
    return receipt


def validate_current_run_matched_report(
    path: str | Path,
    *,
    expected_sha256: str,
    training_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the current run's rank traces instead of trusting its report."""

    report_path = Path(path).expanduser().absolute()
    if report_path == Path("/tmp") or Path("/tmp") in report_path.parents:
        raise CheckpointAcceptanceError("current-run matched report must not use /tmp")
    try:
        if report_path.resolve(strict=True) != report_path:
            raise CheckpointAcceptanceError(
                "current-run matched report path must be canonical"
            )
    except OSError as exc:
        raise CheckpointAcceptanceError(
            "current-run matched report path cannot be resolved"
        ) from exc
    report, report_sha256 = strict_json_file(
        report_path, "current-run matched-control report"
    )
    if report_sha256 != _require_sha256(
        expected_sha256, "current-run matched-control report digest"
    ):
        raise CheckpointAcceptanceError(
            "current-run matched-control report digest differs"
        )
    trace_files = report.get("trace_files")
    if not isinstance(trace_files, dict) or set(trace_files) != {
        "soft_semantic",
        "hard_st",
        "hard_detached",
    }:
        raise CheckpointAcceptanceError(
            "current-run matched-control trace set differs"
        )
    variants: dict[str, list[Path]] = {}
    for variant, rows in trace_files.items():
        if not isinstance(rows, list) or not rows:
            raise CheckpointAcceptanceError(
                f"current-run matched-control trace list is empty for {variant}"
            )
        paths: list[Path] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise CheckpointAcceptanceError(
                    "current-run matched-control trace receipt schema mismatch"
                )
            _require_sha256(row["sha256"], "current-run trace digest")
            if not isinstance(row["path"], str) or not row["path"]:
                raise CheckpointAcceptanceError(
                    "current-run matched-control trace path is invalid"
                )
            candidate = report_path.parent / row["path"]
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise CheckpointAcceptanceError(
                    "current-run matched-control trace path cannot be resolved"
                ) from exc
            paths.append(resolved)
        variants[variant] = paths

    try:
        from world_action_model.matched_control_hashes import compare_variant_traces

        replayed = compare_variant_traces(
            variants,
            output_report=report_path,
            required_seeds=(training_receipt["split_seed"],),
            allow_distinct_variant_run_identities=True,
            write_report=False,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        raise CheckpointAcceptanceError(
            "current-run matched-control replay failed"
        ) from exc
    if replayed != report:
        raise CheckpointAcceptanceError(
            "current-run matched-control report differs from independent replay"
        )

    selected = training_receipt["transport_variant_id"]
    current = training_receipt["current_training_data_receipt"]
    code_receipt = current.get("training_code_receipt")
    if not isinstance(code_receipt, dict):
        raise CheckpointAcceptanceError(
            "current-run matched-control report lacks training code provenance"
        )
    expected = {
        "status": "passed",
        "dry_run_passed": True,
        "all_hashes_equal": True,
        "all_hashes_equal_across_variants": True,
        "dataset_id": training_receipt["production_run_contract"]["dataset_id"],
        "git_commit": code_receipt.get("git_commit"),
        "training_data_receipt_sha256": training_receipt[
            "current_training_data_receipt_sha256"
        ],
        "heldout_window_manifest_sha256": training_receipt[
            "heldout_window_manifest_sha256"
        ],
        "production_split_seed": training_receipt["split_seed"],
        "production_preflight_receipt_sha256": training_receipt[
            "production_preflight_receipt_sha256"
        ],
        "run_contract": training_receipt["production_run_contract"],
        "run_contract_sha256": training_receipt["production_run_contract_sha256"],
        "control_config_sha256": training_receipt["control_config_sha256"],
    }
    if any(report.get(field) != value for field, value in expected.items()):
        raise CheckpointAcceptanceError(
            "current-run matched-control report differs from checkpoint contract"
        )
    if (
        report.get("run_id_by_variant", {}).get(selected)
        != training_receipt["training_run_id"]
        or report.get("run_nonce_by_variant", {}).get(selected)
        != training_receipt["training_run_nonce"]
        or report.get("external_acceptance_sha256_by_variant", {}).get(selected)
        != training_receipt["matched_control_external_acceptance_sha256"]
        or report.get("transport_override_by_variant", {}).get(selected)
        != {
            "variant_id": selected,
            "contact_semantic_transport_mode": {
                "soft_semantic": "soft",
                "hard_st": "hard_st",
                "hard_detached": "hard_detached",
            }[selected],
        }
    ):
        raise CheckpointAcceptanceError(
            "current-run matched-control selected variant binding differs"
        )
    return report


def validate_acceptance_claim(
    value: object,
    *,
    training_receipt: Mapping[str, Any],
    training_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate reviewer-authored content without trusting producer adjacency."""

    acceptance = _require_exact(
        value,
        {
            "acceptance_version",
            "status",
            "reviewer_identity",
            "reviewer_attestation_nonce",
            "acceptance_run_id",
            "acceptance_run_nonce",
            "checkpoint_path_record_only",
            "checkpoint_step",
            "split_seed",
            "training_run_nonce",
            "training_run_id",
            "current_run_matched_report_path_record_only",
            "current_run_matched_report_sha256",
            "checkpoint_bundle_manifest_sha256",
            "checkpoint_bundle_sha256",
            "checkpoint_training_receipt_sha256",
        },
        "independent checkpoint acceptance",
    )
    if (
        acceptance["acceptance_version"] != ACCEPTANCE_VERSION
        or acceptance["status"] != "accepted"
        or isinstance(acceptance["checkpoint_step"], bool)
        or not isinstance(acceptance["checkpoint_step"], int)
        or acceptance["checkpoint_step"] != training_receipt["checkpoint_step"]
        or acceptance["split_seed"] != training_receipt["split_seed"]
        or acceptance["training_run_nonce"] != training_receipt["training_run_nonce"]
        or acceptance["training_run_id"] != training_receipt["training_run_id"]
        or acceptance["checkpoint_bundle_manifest_sha256"]
        != training_receipt["checkpoint_bundle_manifest_sha256"]
        or acceptance["checkpoint_bundle_sha256"]
        != training_receipt["checkpoint_bundle_sha256"]
        or acceptance["checkpoint_training_receipt_sha256"]
        != training_receipt_sha256
    ):
        raise CheckpointAcceptanceError("independent checkpoint acceptance binding differs")
    for field in (
        "reviewer_identity",
        "reviewer_attestation_nonce",
        "acceptance_run_id",
        "acceptance_run_nonce",
    ):
        if not isinstance(acceptance[field], str) or len(acceptance[field]) < 16:
            raise CheckpointAcceptanceError(f"invalid independent {field}")
    report_path = acceptance["current_run_matched_report_path_record_only"]
    if not isinstance(report_path, str) or not Path(report_path).is_absolute():
        raise CheckpointAcceptanceError(
            "current-run matched-control report path must be absolute"
        )
    validate_current_run_matched_report(
        report_path,
        expected_sha256=acceptance["current_run_matched_report_sha256"],
        training_receipt=training_receipt,
    )
    checkpoint_path = Path(acceptance["checkpoint_path_record_only"])
    matched_report_path = Path(report_path)
    if checkpoint_path == matched_report_path.parent or checkpoint_path in matched_report_path.parents:
        raise CheckpointAcceptanceError(
            "current-run matched-control report must be external to checkpoint"
        )
    return acceptance


def validate_acceptance(
    value: object,
    *,
    checkpoint_dir: str | Path,
    bundle_manifest: Mapping[str, Any],
    bundle_manifest_sha256: str,
    training_receipt: Mapping[str, Any],
    training_receipt_sha256: str,
) -> dict[str, Any]:
    acceptance = validate_acceptance_claim(
        value,
        training_receipt=training_receipt,
        training_receipt_sha256=training_receipt_sha256,
    )
    if (
        acceptance["checkpoint_path_record_only"]
        != str(Path(checkpoint_dir).expanduser().absolute())
        or acceptance["checkpoint_bundle_manifest_sha256"]
        != bundle_manifest_sha256
        or acceptance["checkpoint_bundle_sha256"]
        != bundle_manifest["bundle_sha256"]
        or acceptance["checkpoint_training_receipt_sha256"]
        != training_receipt_sha256
    ):
        raise CheckpointAcceptanceError("independent checkpoint acceptance binding differs")
    return acceptance


def write_json_exclusive(path: str | Path, value: object) -> str:
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

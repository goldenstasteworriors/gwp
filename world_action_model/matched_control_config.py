"""Opt-in config wiring for matched-control WAM dry runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import world_action_model.matched_control_hashes as matched_control_hashes
from world_action_model.matched_control_hashes import (
    ALLOWED_SEEDS,
    COLLECTOR_VERSION,
    HASH_SCHEMA,
    IMPLEMENTED_RUNNER_VARIANTS,
    canonical_sha256,
)


CONFIG_HASH_SCOPE = {
    "control": (
        "dataset_id, ordered data roots, dataloader batch/worker settings, complete "
        "pre-trace transform config, norm file bytes, and train accumulation/precision"
    ),
    "variant": (
        "control hash plus variant_id and complete model config before collector wiring"
    ),
}

_RUNTIME_RECEIPT_CONFIG_KEYS = {
    "training_source_runtime_receipt",
    "t5_runtime_source_receipt",
    "runtime_stat_record_only",
    "verified_private_cache_parent",
    "verified_training_run_nonce",
}


def _portable_config_hash_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _portable_config_hash_value(item)
            for key, item in value.items()
            if key not in _RUNTIME_RECEIPT_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_portable_config_hash_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable_config_hash_value(item) for item in value)
    return value


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required when matched-control capture is enabled")
    return value


def _required_ascii_decimal_environment(name: str) -> int:
    value = _required_environment(name)
    if not re.fullmatch(r"[0-9]+", value, flags=re.ASCII):
        raise ValueError(f"{name} must be a canonical non-negative base-10 integer")
    return int(value)


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_git_commit(start: str | os.PathLike[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(start).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot resolve current Git commit for matched controls") from exc
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("current matched-control Git commit is not a full SHA-1")
    return commit


def _validate_expected_evidence_binding(
    *,
    variant_id: str,
    expected_git_commit: str | None,
    expected_run_id: str | None,
    expected_run_nonce: str | None,
    expected_report_sha256: str | None,
    expected_trace_files: Mapping[str, list[Mapping[str, str]]] | None,
    expected_trace_manifest_sha256: str | None,
    expected_global_hashes_sha256: str | None,
    expected_hash_schema_sha256: str | None,
    expected_control_config_sha256: str | None,
    expected_variant_config_sha256: Mapping[str, str] | None,
    expected_world_size: int | None,
    expected_rank_set: list[int] | None,
    expected_collector_version: str | None,
    expected_collector_source_sha256: str | None,
) -> dict[str, Any] | None:
    values = (
        expected_git_commit,
        expected_run_id,
        expected_run_nonce,
        expected_report_sha256,
        expected_trace_files,
        expected_trace_manifest_sha256,
        expected_global_hashes_sha256,
        expected_hash_schema_sha256,
        expected_control_config_sha256,
        expected_variant_config_sha256,
        expected_world_size,
        expected_rank_set,
        expected_collector_version,
        expected_collector_source_sha256,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("matched-control evidence expectations must be supplied as one complete set")
    assert expected_git_commit is not None
    assert expected_run_id is not None
    assert expected_run_nonce is not None
    assert expected_report_sha256 is not None
    assert expected_trace_files is not None
    assert expected_trace_manifest_sha256 is not None
    assert expected_global_hashes_sha256 is not None
    assert expected_hash_schema_sha256 is not None
    assert expected_control_config_sha256 is not None
    assert expected_variant_config_sha256 is not None
    assert expected_world_size is not None
    assert expected_rank_set is not None
    assert expected_collector_version is not None
    assert expected_collector_source_sha256 is not None
    if not isinstance(expected_git_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", expected_git_commit
    ):
        raise ValueError("expected_git_commit must be a full 40-hex Git SHA")
    if not isinstance(expected_run_id, str) or not expected_run_id.strip():
        raise ValueError("expected_run_id must be non-empty")
    if not isinstance(expected_run_nonce, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", expected_run_nonce
    ):
        raise ValueError("expected_run_nonce must be an explicit 16-128 character audit nonce")
    for name, value in (
        ("expected_report_sha256", expected_report_sha256),
        ("expected_trace_manifest_sha256", expected_trace_manifest_sha256),
        ("expected_global_hashes_sha256", expected_global_hashes_sha256),
        ("expected_hash_schema_sha256", expected_hash_schema_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be a SHA256")
    if not isinstance(expected_control_config_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_control_config_sha256
    ):
        raise ValueError("expected_control_config_sha256 must be a SHA256")
    variant_hashes = dict(expected_variant_config_sha256)
    if (
        set(variant_hashes) != set(IMPLEMENTED_RUNNER_VARIANTS)
        or variant_id not in IMPLEMENTED_RUNNER_VARIANTS
        or any(not isinstance(name, str) or not name for name in variant_hashes)
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in variant_hashes.values()
        )
    ):
        raise ValueError("expected_variant_config_sha256 is not a complete variant mapping")
    if not isinstance(expected_trace_files, Mapping):
        raise ValueError("expected_trace_files must be a variant mapping")
    trace_files = {}
    for name, entries in expected_trace_files.items():
        if not isinstance(name, str) or not isinstance(entries, list):
            raise ValueError("expected_trace_files has an invalid variant entry")
        if any(not isinstance(entry, Mapping) for entry in entries):
            raise ValueError("expected_trace_files entries must be mappings")
        trace_files[name] = [dict(entry) for entry in entries]
    if set(trace_files) != set(IMPLEMENTED_RUNNER_VARIANTS):
        raise ValueError("expected_trace_files variants differ from the implemented runner set")
    seen_trace_paths = set()
    for variant, entries in trace_files.items():
        if not entries:
            raise ValueError(f"expected_trace_files is empty for {variant}")
        for entry in entries:
            if set(entry) != {"path", "sha256"}:
                raise ValueError("expected trace entries require exactly path and sha256")
            path = entry["path"]
            digest = entry["sha256"]
            if not isinstance(path, str) or not path or path in seen_trace_paths:
                raise ValueError("expected trace paths must be non-empty and globally unique")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("expected trace SHA must be a SHA256")
            seen_trace_paths.add(path)
    if canonical_sha256(trace_files) != expected_trace_manifest_sha256:
        raise ValueError("expected trace manifest SHA does not match expected_trace_files")
    if canonical_sha256(HASH_SCHEMA) != expected_hash_schema_sha256:
        raise ValueError("expected hash schema differs from the installed collector")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size <= 0
    ):
        raise ValueError("expected_world_size must be a positive integer")
    if (
        not isinstance(expected_rank_set, list)
        or any(
            isinstance(rank, bool) or not isinstance(rank, int)
            for rank in expected_rank_set
        )
        or expected_rank_set != list(range(expected_world_size))
    ):
        raise ValueError("expected_rank_set must equal range(expected_world_size)")
    if expected_collector_version != COLLECTOR_VERSION:
        raise ValueError("expected_collector_version differs from the installed collector")
    if not isinstance(expected_collector_source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_collector_source_sha256
    ):
        raise ValueError("expected_collector_source_sha256 must be a SHA256")
    source_path = Path(matched_control_hashes.__file__).resolve()
    if _file_sha256(source_path) != expected_collector_source_sha256:
        raise ValueError("installed matched-control collector source SHA differs from the gate")
    if _current_git_commit(source_path.parent) != expected_git_commit:
        raise ValueError("current Git commit differs from the matched-control gate")
    return {
        "expected_git_commit": expected_git_commit,
        "expected_run_id": expected_run_id,
        "expected_run_nonce": expected_run_nonce,
        "expected_report_sha256": expected_report_sha256,
        "expected_trace_files": trace_files,
        "expected_trace_manifest_sha256": expected_trace_manifest_sha256,
        "expected_global_hashes_sha256": expected_global_hashes_sha256,
        "expected_hash_schema_sha256": expected_hash_schema_sha256,
        "expected_world_size": expected_world_size,
        "expected_rank_set": list(expected_rank_set),
        "expected_collector_version": expected_collector_version,
        "expected_collector_source_sha256": expected_collector_source_sha256,
        "expected_control_config_sha256": expected_control_config_sha256,
        "expected_variant_config_sha256": variant_hashes,
    }


def _runtime_topology_from_environment(
    *, expected_world_size: int | None, expected_rank_set: list[int] | None
) -> dict[str, int]:
    """Read/compare launcher identity before config or filesystem inspection."""

    rank = _required_ascii_decimal_environment("RANK")
    world_size = _required_ascii_decimal_environment("WORLD_SIZE")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid runtime topology rank={rank}, world_size={world_size}")
    if expected_world_size is not None or expected_rank_set is not None:
        if (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
            or expected_world_size <= 0
            or not isinstance(expected_rank_set, list)
            or any(
                isinstance(expected_rank, bool) or not isinstance(expected_rank, int)
                for expected_rank in expected_rank_set
            )
            or expected_rank_set != list(range(expected_world_size))
        ):
            raise ValueError("external rank/world topology is invalid or incomplete")
        if world_size != expected_world_size or rank not in expected_rank_set:
            raise ValueError("actual runtime rank/world differs from external expectations")
    return {"rank": rank, "world_size": world_size}


def configure_matched_control_from_environment(
    config: dict[str, Any],
    *,
    variant_id: str,
    expected_dataset_id: str | None = None,
    expected_git_commit: str | None = None,
    expected_run_id: str | None = None,
    expected_run_nonce: str | None = None,
    expected_report_sha256: str | None = None,
    expected_trace_files: Mapping[str, list[Mapping[str, str]]] | None = None,
    expected_trace_manifest_sha256: str | None = None,
    expected_global_hashes_sha256: str | None = None,
    expected_hash_schema_sha256: str | None = None,
    expected_control_config_sha256: str | None = None,
    expected_variant_config_sha256: Mapping[str, str] | None = None,
    expected_world_size: int | None = None,
    expected_rank_set: list[int] | None = None,
    expected_collector_version: str | None = None,
    expected_collector_source_sha256: str | None = None,
) -> bool:
    """Wire dataset/transform/trainer tracing only when an output is requested.

    ``GWP_MATCHED_CONTROL_OUTPUT_DIR`` is the single opt-in switch.  The caller
    must set it to a directory below ``config['project_dir']``; no default path
    is created by the legacy route.
    """

    output_dir = os.environ.get("GWP_MATCHED_CONTROL_OUTPUT_DIR", "").strip()
    if not output_dir:
        return False
    runtime_topology = _runtime_topology_from_environment(
        expected_world_size=expected_world_size,
        expected_rank_set=expected_rank_set,
    )
    if variant_id not in IMPLEMENTED_RUNNER_VARIANTS:
        raise ValueError(
            f"variant_id is not an implemented matched-control runner: {variant_id!r}"
        )
    evidence_binding = _validate_expected_evidence_binding(
        variant_id=variant_id,
        expected_git_commit=expected_git_commit,
        expected_run_id=expected_run_id,
        expected_run_nonce=expected_run_nonce,
        expected_report_sha256=expected_report_sha256,
        expected_trace_files=expected_trace_files,
        expected_trace_manifest_sha256=expected_trace_manifest_sha256,
        expected_global_hashes_sha256=expected_global_hashes_sha256,
        expected_hash_schema_sha256=expected_hash_schema_sha256,
        expected_control_config_sha256=expected_control_config_sha256,
        expected_variant_config_sha256=expected_variant_config_sha256,
        expected_world_size=expected_world_size,
        expected_rank_set=expected_rank_set,
        expected_collector_version=expected_collector_version,
        expected_collector_source_sha256=expected_collector_source_sha256,
    )
    dataset_id = _required_environment("GWP_MATCHED_CONTROL_DATASET_ID")
    if expected_dataset_id is not None:
        expected_dataset_id = str(expected_dataset_id).strip()
        if not expected_dataset_id:
            raise ValueError("expected_dataset_id must be non-empty when provided")
        if dataset_id != expected_dataset_id:
            raise ValueError(
                "GWP_MATCHED_CONTROL_DATASET_ID differs from the preflight-bound "
                "expected_dataset_id"
            )
    run_id = _required_environment("GWP_MATCHED_CONTROL_RUN_ID")
    run_nonce = _required_environment("GWP_MATCHED_CONTROL_RUN_NONCE")
    if evidence_binding is not None and (
        run_id != evidence_binding["expected_run_id"]
        or run_nonce != evidence_binding["expected_run_nonce"]
    ):
        raise ValueError("runtime run_id/nonce differs from external expectations")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", run_nonce):
        raise ValueError("GWP_MATCHED_CONTROL_RUN_NONCE has an invalid audit nonce format")
    selected_variant = os.environ.get(
        "GWP_MATCHED_CONTROL_VARIANT_ID", variant_id
    ).strip()
    if selected_variant != variant_id:
        raise ValueError(
            "GWP_MATCHED_CONTROL_VARIANT_ID differs from the effective config variant"
        )
    seed = _required_ascii_decimal_environment("GWP_MATCHED_CONTROL_SEED")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"GWP_MATCHED_CONTROL_SEED must be one of {ALLOWED_SEEDS}")
    current_training_receipt_sha256 = config["models"].get(
        "current_training_data_receipt_sha256"
    )
    current_training_receipt = config["models"].get(
        "current_training_data_receipt"
    )
    heldout_window_manifest_sha256 = None
    if current_training_receipt_sha256 is not None:
        heldout_window_manifest_sha256 = current_training_receipt.get(
            "heldout_window_manifest_sha256"
        ) if isinstance(current_training_receipt, Mapping) else None
        if (
            not isinstance(current_training_receipt_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", current_training_receipt_sha256)
            or not isinstance(current_training_receipt, Mapping)
            or current_training_receipt.get("split_seed") != seed
            or not isinstance(heldout_window_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", heldout_window_manifest_sha256)
        ):
            raise ValueError(
                "matched-control seed/current training-data receipt binding differs"
            )

    dataset_configs = config["dataloaders"]["train"]["data_or_config"]
    transform = config["dataloaders"]["train"]["transform"]
    norm_entries = transform.get("norm_path", [])
    norm_fingerprints = []
    for entry in norm_entries:
        path = entry["path"] if isinstance(entry, dict) else entry
        norm_fingerprints.append(
            {"path": os.path.realpath(str(path)), "sha256": _file_sha256(path)}
        )

    control_payload = {
        "dataset_id": dataset_id,
        "data_roots": [
            os.path.realpath(str(dataset_config["data_path"]))
            for dataset_config in dataset_configs
        ],
        "dataloader": {
            "batch_size_per_gpu": config["dataloaders"]["train"].get(
                "batch_size_per_gpu"
            ),
            "num_workers": config["dataloaders"]["train"].get("num_workers"),
            "sampler": config["dataloaders"]["train"].get("sampler"),
            "batch_sampler": config["dataloaders"]["train"].get("batch_sampler"),
        },
        "dataset_configs": _portable_config_hash_value(dataset_configs),
        "transform": _portable_config_hash_value(transform),
        "norm_files": norm_fingerprints,
        "train_controls": {
            "gradient_accumulation_steps": config["train"].get(
                "gradient_accumulation_steps", 1
            ),
            "mixed_precision": config["train"].get("mixed_precision"),
        },
    }
    control_sha256 = canonical_sha256(control_payload)
    variant_sha256 = canonical_sha256(
        {
            "control_config_sha256": control_sha256,
            "variant_id": variant_id,
            "models": config["models"],
        }
    )
    if evidence_binding is not None:
        if control_sha256 != evidence_binding["expected_control_config_sha256"]:
            raise ValueError("effective control config hash differs from the matched-control gate")
        if (
            variant_sha256
            != evidence_binding["expected_variant_config_sha256"][variant_id]
        ):
            raise ValueError("effective variant config hash differs from the matched-control gate")

    for dataset_config in dataset_configs:
        dataset_config.update(
            matched_control_identity=True,
            matched_control_dataset_id=dataset_id,
        )
    transform["matched_control_capture"] = True
    config["train"]["seed"] = seed
    config["models"]["matched_control_collector"] = {
        "enabled": True,
        "output_dir": output_dir,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "variant_id": variant_id,
        "seed": seed,
        "variant_config_sha256": variant_sha256,
        "control_config_sha256": control_sha256,
        "config_hash_scope": CONFIG_HASH_SCOPE,
        "runtime_rank": runtime_topology["rank"],
        "runtime_world_size": runtime_topology["world_size"],
        "training_data_receipt_sha256": current_training_receipt_sha256,
        "heldout_window_manifest_sha256": heldout_window_manifest_sha256,
        "production_split_seed": (
            None if current_training_receipt is None
            else current_training_receipt.get("split_seed")
        ),
    }
    if evidence_binding is not None:
        config["models"]["matched_control_collector"].update(evidence_binding)
    return True

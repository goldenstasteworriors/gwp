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


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required when matched-control capture is enabled")
    return value


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
    expected_control_config_sha256: str | None,
    expected_variant_config_sha256: Mapping[str, str] | None,
    expected_world_size: int | None,
    expected_rank_set: list[int] | None,
    expected_collector_version: str | None,
    expected_collector_source_sha256: str | None,
) -> dict[str, Any] | None:
    values = (
        expected_git_commit,
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
    assert expected_control_config_sha256 is not None
    assert expected_variant_config_sha256 is not None
    assert expected_world_size is not None
    assert expected_rank_set is not None
    assert expected_collector_version is not None
    assert expected_collector_source_sha256 is not None
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("expected_git_commit must be a full 40-hex Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_control_config_sha256):
        raise ValueError("expected_control_config_sha256 must be a SHA256")
    variant_hashes = dict(expected_variant_config_sha256)
    if (
        len(variant_hashes) < 2
        or variant_id not in variant_hashes
        or any(not isinstance(name, str) or not name for name in variant_hashes)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in variant_hashes.values())
    ):
        raise ValueError("expected_variant_config_sha256 is not a complete variant mapping")
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
    if not re.fullmatch(r"[0-9a-f]{64}", expected_collector_source_sha256):
        raise ValueError("expected_collector_source_sha256 must be a SHA256")
    source_path = Path(matched_control_hashes.__file__).resolve()
    if _file_sha256(source_path) != expected_collector_source_sha256:
        raise ValueError("installed matched-control collector source SHA differs from the gate")
    if _current_git_commit(source_path.parent) != expected_git_commit:
        raise ValueError("current Git commit differs from the matched-control gate")
    return {
        "expected_git_commit": expected_git_commit,
        "expected_world_size": expected_world_size,
        "expected_rank_set": list(expected_rank_set),
        "expected_collector_version": expected_collector_version,
        "expected_collector_source_sha256": expected_collector_source_sha256,
        "expected_control_config_sha256": expected_control_config_sha256,
        "expected_variant_config_sha256": variant_hashes,
    }


def configure_matched_control_from_environment(
    config: dict[str, Any],
    *,
    variant_id: str,
    expected_dataset_id: str | None = None,
    expected_git_commit: str | None = None,
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
    evidence_binding = _validate_expected_evidence_binding(
        variant_id=variant_id,
        expected_git_commit=expected_git_commit,
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
    selected_variant = os.environ.get(
        "GWP_MATCHED_CONTROL_VARIANT_ID", variant_id
    ).strip()
    if selected_variant != variant_id:
        raise ValueError(
            "GWP_MATCHED_CONTROL_VARIANT_ID differs from the effective config variant"
        )
    seed = int(_required_environment("GWP_MATCHED_CONTROL_SEED"))
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"GWP_MATCHED_CONTROL_SEED must be one of {ALLOWED_SEEDS}")

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
        "dataset_configs": dataset_configs,
        "transform": transform,
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
        "variant_id": variant_id,
        "seed": seed,
        "variant_config_sha256": variant_sha256,
        "control_config_sha256": control_sha256,
        "config_hash_scope": CONFIG_HASH_SCOPE,
    }
    if evidence_binding is not None:
        config["models"]["matched_control_collector"].update(evidence_binding)
    return True

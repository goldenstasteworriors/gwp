"""Opt-in config wiring for matched-control WAM dry runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from world_action_model.matched_control_hashes import ALLOWED_SEEDS, canonical_sha256


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


def configure_matched_control_from_environment(
    config: dict[str, Any], *, variant_id: str, expected_dataset_id: str | None = None
) -> bool:
    """Wire dataset/transform/trainer tracing only when an output is requested.

    ``GWP_MATCHED_CONTROL_OUTPUT_DIR`` is the single opt-in switch.  The caller
    must set it to a directory below ``config['project_dir']``; no default path
    is created by the legacy route.
    """

    output_dir = os.environ.get("GWP_MATCHED_CONTROL_OUTPUT_DIR", "").strip()
    if not output_dir:
        return False
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
    return True

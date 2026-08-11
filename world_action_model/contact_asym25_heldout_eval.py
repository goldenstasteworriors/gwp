"""Fail-closed held-out evaluation contracts for production asymmetric25 WAM.

The module deliberately separates artifact validation, deterministic window
selection, model execution, and metric reduction.  Heavy model imports live in
``contact_asym25_eval_runtime`` so manifest preparation and report auditing can
run on a CPU-only host.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from typing import Any

import numpy as np

from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    LEFT_WRIST_SLICE,
    MODEL_ACTION_DIM,
    OMNIPICKER_APERTURE_INDEX,
    RH56_SLICE,
    RIGHT_WRIST_SLICE,
)
from world_action_model.contact_asym25_preflight import (
    PREFLIGHT_RECEIPT_VERSION,
    sha256_file,
    validate_contact_asym25_training_preflight,
    validate_contact_sidecar_arrays,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.verified_io import (
    hash_verified_regular_file,
    stat_identity,
    strict_json_loads,
    verified_head_code_manifest,
)


EVALUATOR_VERSION = "contact-asym25-heldout-evaluator/v1"
WINDOW_MANIFEST_VERSION = "contact-asym25-heldout-windows/v1"
CHECKPOINT_MANIFEST_VERSION = "contact-asym25-heldout-checkpoints/v1"
CURRENT_TRAINING_DATA_RECEIPT_VERSION = "contact-asym25-current-training-data-receipt/v1"
RESULT_VERSION = "contact-asym25-heldout-result/v1"
AGGREGATE_RESULT_VERSION = "contact-asym25-heldout-aggregate/v1"
ARTIFACT_RECEIPT_VERSION = "contact-asym25-heldout-artifacts/v1"

EVALUATION_SEEDS = (6666, 7777, 8888)
CHECKPOINT_STEPS = (0, 100, 200)
WINDOWS_PER_SPLIT_PER_SEED = 256
HORIZON_FRAMES = 48
IMAGE_FRAME_OFFSETS = (0, 12, 24, 36, 48)
TIME_BINS = 16
HARD_TIME_GRID_ATOL = 1.0 / 512.0 + 1e-7
NUM_CONTACT_QUERIES = 12
SPLIT_COLUMNS = {
    6666: "seed_6666_split",
    7777: "seed_7777_split",
    8888: "seed_8888_split",
}
SPLIT_HEADER = [
    "episode_id",
    "length_frames",
    "seed_6666_split",
    "seed_7777_split",
    "seed_8888_split",
    "split_unit",
    "split_purpose",
    "ood_eligible",
    "strategy_claim_allowed",
    "metadata_status",
]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRAINING_CODE_LOGICAL_PATHS = (
    "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py",
    "configs/giga_world_policy_0_5_sonic_g1_asymmetric25_finetune.py",
    "configs/giga_world_policy_0_5_sonic_g1_finetune.py",
    "scripts/contact_labels/contact_label_io_v1_1.py",
    "world_action_model/contact_asym25_preflight.py",
    "world_action_model/contact_asym25_heldout_eval.py",
    "world_action_model/production_split_contract.py",
    "world_action_model/asymmetric_action_schema.py",
    "world_action_model/asymmetric_hardware_manifest.py",
    "world_action_model/contact_transition_schema.py",
    "world_action_model/ego_action_schema.py",
    "world_action_model/verified_io.py",
    "world_action_model/datasets/wam_lerobot_dataset.py",
    "world_action_model/transforms/wa_transforms_lerobot_pretrain.py",
    "world_action_model/trainer/wa_casual_trainer_mot.py",
    "world_action_model/models/transformer_wa_casual_mot.py",
    "world_action_model/matched_control_config.py",
    "world_action_model/matched_control_hashes.py",
)


class HeldoutEvaluationError(ValueError):
    """Raised when a production evaluation invariant is not proven."""


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise HeldoutEvaluationError(
            f"{name} schema mismatch: expected {sorted(expected)}, got {actual}"
        )
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise HeldoutEvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_json(path: str | Path, name: str) -> tuple[dict, str]:
    path = Path(path).expanduser().absolute()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise HeldoutEvaluationError(f"cannot read {name}: {path}") from exc
    value = strict_json_loads(payload, name=name)
    if not isinstance(value, dict):
        raise HeldoutEvaluationError(f"{name} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HeldoutEvaluationError("object is not strict canonical JSON") from exc


def write_strict_json(path: str | Path, value: object, *, overwrite: bool = False) -> str:
    path = Path(path).expanduser().absolute()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale partial output exists: {partial}")
    try:
        partial.write_bytes(payload)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return hashlib.sha256(payload).hexdigest()


def load_and_replay_preflight_receipt(path: str | Path) -> tuple[dict, str]:
    """Re-run every production preflight check and compare the exact receipt."""

    receipt, receipt_sha256 = _strict_json(path, "production preflight receipt")
    if receipt.get("receipt_version") != PREFLIGHT_RECEIPT_VERSION:
        raise HeldoutEvaluationError("production preflight receipt version mismatch")
    artifact_paths = _require_exact_keys(
        receipt.get("artifact_paths"),
        {
            "data_root",
            "norm_stats",
            "hardware_manifest",
            "calibration_capture",
            "sidecar_dir",
            "coverage_csv",
            "segments_csv",
            "inventory_csv",
            "build_config",
            "split_manifest",
            "training_gate",
        },
        "preflight artifact_paths",
    )
    if any(not isinstance(value, str) or not Path(value).is_absolute() for value in artifact_paths.values()):
        raise HeldoutEvaluationError("all preflight artifact paths must be absolute")
    replayed = validate_contact_asym25_training_preflight(
        data_root=artifact_paths["data_root"],
        norm_stats_path=artifact_paths["norm_stats"],
        hardware_manifest_path=artifact_paths["hardware_manifest"],
        calibration_capture_path=artifact_paths["calibration_capture"],
        sidecar_dir=artifact_paths["sidecar_dir"],
        coverage_csv=artifact_paths["coverage_csv"],
        segments_csv=artifact_paths["segments_csv"],
        inventory_csv=artifact_paths["inventory_csv"],
        build_config_path=artifact_paths["build_config"],
        split_manifest=artifact_paths["split_manifest"],
        training_gate_path=artifact_paths["training_gate"],
        split_seed=receipt.get("split_seed"),
    )
    if replayed != receipt:
        raise HeldoutEvaluationError(
            "persisted production preflight receipt differs from a fresh replay"
        )
    fixed = {
        "status": "artifact_consistency_passed_with_explicit_external_gates",
        "action_schema_id": ASYMMETRIC_SCHEMA_ID,
        "contact_schema_version": SEGMENT_SCHEMA_VERSION,
        "contact_target_dim": SEGMENT_TARGET_DIM,
        "num_contact_tokens": NUM_CONTACT_QUERIES,
        "transport_mode": "hard_st",
        "time_bins": TIME_BINS,
        "horizon_frames": HORIZON_FRAMES,
    }
    for key, expected in fixed.items():
        if receipt.get(key) != expected:
            raise HeldoutEvaluationError(
                f"preflight receipt {key}={receipt.get(key)!r}; expected {expected!r}"
            )
    if receipt.get("scientific_gate_claimed") is not False:
        raise HeldoutEvaluationError("preflight receipt must not self-claim scientific validity")
    return receipt, receipt_sha256


def load_completed_group_split(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_episode_lengths: Mapping[str, int],
) -> tuple[list[dict[str, str]], dict[int, dict[int, str]]]:
    path = Path(path).expanduser().absolute()
    if sha256_file(path) != _require_sha256(expected_sha256, "split manifest digest"):
        raise HeldoutEvaluationError("group split manifest fingerprint differs")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SPLIT_HEADER:
                raise HeldoutEvaluationError("group split manifest header mismatch")
            rows = list(reader)
    except OSError as exc:
        raise HeldoutEvaluationError(f"cannot read group split manifest: {path}") from exc
    if not rows:
        raise HeldoutEvaluationError("group split manifest is empty")
    expected_lengths = {int(key): int(value) for key, value in expected_episode_lengths.items()}
    assignments: dict[int, dict[int, str]] = {seed: {} for seed in EVALUATION_SEEDS}
    seen = set()
    for row_number, row in enumerate(rows, 2):
        try:
            episode = int(row["episode_id"])
            length = int(row["length_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HeldoutEvaluationError(f"invalid split row {row_number}") from exc
        if episode in seen or expected_lengths.get(episode) != length:
            raise HeldoutEvaluationError(
                f"split row {row_number} duplicates an episode or changes its length"
            )
        seen.add(episode)
        if (
            row["split_unit"] != "group"
            or row["split_purpose"] != "production_strategy_evaluation"
            or row["ood_eligible"] != "true"
            or row["strategy_claim_allowed"] != "true"
            or row["metadata_status"] != "complete"
        ):
            raise HeldoutEvaluationError(
                f"split row {row_number} is diagnostic/incomplete and cannot support claims"
            )
        for seed, column in SPLIT_COLUMNS.items():
            split = row[column]
            if split not in {"train", "val", "test"}:
                raise HeldoutEvaluationError(f"invalid {column} in split row {row_number}")
            assignments[seed][episode] = split
    if seen != set(expected_lengths):
        raise HeldoutEvaluationError("group split does not cover the preflighted episode set")
    for seed in EVALUATION_SEEDS:
        if set(assignments[seed].values()) != {"train", "val", "test"}:
            raise HeldoutEvaluationError(f"split seed {seed} lacks train/val/test episodes")
    return rows, assignments


def _window_noise_seed(seed: int, episode: int, frame: int) -> int:
    payload = f"{WINDOW_MANIFEST_VERSION}\0{seed}\0{episode}\0{frame}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def build_fixed_window_manifest(
    *,
    preflight_receipt: Mapping[str, Any],
    preflight_receipt_sha256: str,
    split_assignments: Mapping[int, Mapping[int, str]],
) -> dict:
    """Select 256 full-horizon val and test windows for the receipt seed."""

    episode_lengths = {
        int(key): int(value)
        for key, value in preflight_receipt["episode_lengths"].items()
    }
    episode_offsets = {}
    offset = 0
    for episode, length in sorted(episode_lengths.items()):
        episode_offsets[episode] = offset
        offset += length
    selected_seed = preflight_receipt.get("split_seed")
    if selected_seed not in EVALUATION_SEEDS:
        raise HeldoutEvaluationError("preflight does not select a production split seed")
    if selected_seed not in split_assignments:
        raise HeldoutEvaluationError("selected split seed is absent from the split manifest")
    windows = []
    for seed in (selected_seed,):
        for heldout_split in ("val", "test"):
            candidates = []
            for episode, split in sorted(split_assignments[seed].items()):
                if split != heldout_split:
                    continue
                length = episode_lengths[episode]
                last_start = min(
                    length - HORIZON_FRAMES,
                    length - 1 - max(IMAGE_FRAME_OFFSETS),
                )
                for frame in range(max(0, last_start + 1)):
                    rank_payload = (
                        f"{WINDOW_MANIFEST_VERSION}\0sha256-rank/v1\0{seed}\0"
                        f"{heldout_split}\0{episode}\0{frame}"
                    ).encode("ascii")
                    candidates.append((hashlib.sha256(rank_payload).digest(), episode, frame))
            candidates.sort()
            if len(candidates) < WINDOWS_PER_SPLIT_PER_SEED:
                raise HeldoutEvaluationError(
                    f"split seed {seed} {heldout_split} has only {len(candidates)} "
                    f"full windows; {WINDOWS_PER_SPLIT_PER_SEED} are required"
                )
            prefix = "v" if heldout_split == "val" else "t"
            for window_index, (_, episode, frame) in enumerate(
                candidates[:WINDOWS_PER_SPLIT_PER_SEED]
            ):
                windows.append(
                    {
                        "seed": seed,
                        "window_index": window_index,
                        "window_id": (
                            f"s{seed}-{prefix}{window_index:03d}-"
                            f"e{episode:06d}-f{frame:06d}"
                        ),
                        "split_column": SPLIT_COLUMNS[seed],
                        "split": heldout_split,
                        "episode_id": episode,
                        "frame_index": frame,
                        "global_index": episode_offsets[episode] + frame,
                        "action_noise_seed": _window_noise_seed(
                            seed, episode, frame
                        ),
                    }
                )
    return {
        "manifest_version": WINDOW_MANIFEST_VERSION,
        "dataset_id": preflight_receipt["dataset_id"],
        "production_preflight_receipt_sha256": _require_sha256(
            preflight_receipt_sha256, "preflight receipt digest"
        ),
        "split_manifest_sha256": preflight_receipt["split_manifest_sha256"],
        "split_purpose": "production_strategy_evaluation",
        "selection_algorithm": "sha256_rank_without_replacement/v1",
        "seeds": [selected_seed],
        "windows_per_split_per_seed": WINDOWS_PER_SPLIT_PER_SEED,
        "horizon_frames": HORIZON_FRAMES,
        "image_frame_offsets": list(IMAGE_FRAME_OFFSETS),
        "augmentation_enabled": False,
        "eval_crop_mode": "center",
        "windows": windows,
    }


def validate_fixed_window_manifest(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
    preflight_receipt_sha256: str,
    split_assignments: Mapping[int, Mapping[int, str]],
) -> dict:
    manifest = _require_exact_keys(
        value,
        {
            "manifest_version",
            "dataset_id",
            "production_preflight_receipt_sha256",
            "split_manifest_sha256",
            "split_purpose",
            "selection_algorithm",
            "seeds",
            "windows_per_split_per_seed",
            "horizon_frames",
            "image_frame_offsets",
            "augmentation_enabled",
            "eval_crop_mode",
            "windows",
        },
        "fixed window manifest",
    )
    expected_scalars = {
        "manifest_version": WINDOW_MANIFEST_VERSION,
        "dataset_id": preflight_receipt["dataset_id"],
        "production_preflight_receipt_sha256": preflight_receipt_sha256,
        "split_manifest_sha256": preflight_receipt["split_manifest_sha256"],
        "split_purpose": "production_strategy_evaluation",
        "selection_algorithm": "sha256_rank_without_replacement/v1",
        "seeds": [preflight_receipt["split_seed"]],
        "windows_per_split_per_seed": WINDOWS_PER_SPLIT_PER_SEED,
        "horizon_frames": HORIZON_FRAMES,
        "image_frame_offsets": list(IMAGE_FRAME_OFFSETS),
        "augmentation_enabled": False,
        "eval_crop_mode": "center",
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise HeldoutEvaluationError(f"fixed window manifest {key} differs")
    expected_windows = build_fixed_window_manifest(
        preflight_receipt=preflight_receipt,
        preflight_receipt_sha256=preflight_receipt_sha256,
        split_assignments=split_assignments,
    )["windows"]
    if manifest["windows"] != expected_windows:
        raise HeldoutEvaluationError(
            "fixed windows differ from deterministic held-out selection"
        )
    return manifest


def load_fixed_window_manifest(
    path: str | Path,
    **validation_kwargs: Any,
) -> tuple[dict, str]:
    value, digest = _strict_json(path, "fixed window manifest")
    return validate_fixed_window_manifest(value, **validation_kwargs), digest


def _tree_rows(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise HeldoutEvaluationError(f"artifact tree must be a real directory: {root}")
    rows = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            entry = current_path / name
            mode = os.lstat(entry).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise HeldoutEvaluationError(f"artifact tree has unsafe directory entry: {entry}")
        for name in sorted(file_names):
            entry = current_path / name
            mode = os.lstat(entry).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise HeldoutEvaluationError(f"artifact tree has unsafe file entry: {entry}")
            digest, stat_receipt = hash_verified_regular_file(entry)
            rows.append(
                {
                    "logical_path": entry.relative_to(root).as_posix(),
                    "sha256": digest,
                    "size_bytes": stat_receipt["size_bytes"],
                    "stat": stat_receipt,
                }
            )
    if not rows:
        raise HeldoutEvaluationError(f"artifact tree is empty: {root}")
    rows.sort(key=lambda item: item["logical_path"])
    return rows


def snapshot_artifact_tree(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().absolute()
    rows = _tree_rows(root)
    digest_payload = [
        {
            "logical_path": row["logical_path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in rows
    ]
    digest = hashlib.sha256(_canonical_json_bytes(digest_payload)).hexdigest()
    return {
        "path_record_only": str(root),
        "tree_sha256": digest,
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "_runtime_rows": rows,
    }


def verify_artifact_tree_unchanged(snapshot: Mapping[str, Any]) -> None:
    root = Path(snapshot["path_record_only"])
    expected_rows = snapshot["_runtime_rows"]
    actual_paths = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            entry = current_path / name
            if stat.S_ISLNK(os.lstat(entry).st_mode):
                raise HeldoutEvaluationError(f"artifact tree gained a symlink: {entry}")
        for name in file_names:
            actual_paths.append((current_path / name).relative_to(root).as_posix())
    expected_paths = [row["logical_path"] for row in expected_rows]
    if sorted(actual_paths) != expected_paths:
        raise HeldoutEvaluationError("artifact tree file set changed during evaluation")
    for row in expected_rows:
        entry = root / row["logical_path"]
        try:
            actual = stat_identity(os.lstat(entry))
        except OSError as exc:
            raise HeldoutEvaluationError("artifact tree changed during evaluation") from exc
        if actual != row["stat"]:
            raise HeldoutEvaluationError(
                f"artifact source changed during evaluation: {entry}"
            )


def public_tree_receipt(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path_record_only": snapshot["path_record_only"],
        "tree_sha256": snapshot["tree_sha256"],
        "file_count": snapshot["file_count"],
        "total_size_bytes": snapshot["total_size_bytes"],
    }


def _episode_sets(assignments: Mapping[int, str]) -> dict[str, list[int]]:
    return {
        split: sorted(episode for episode, value in assignments.items() if value == split)
        for split in ("train", "val", "test")
    }


def _validate_historical_training_code_receipt(receipt: Mapping[str, Any]) -> None:
    """Prove every recorded training source row against its pinned Git commit."""

    receipt = _require_exact_keys(
        dict(receipt), {"git_commit", "verified_head_files"},
        "checkpoint training code receipt",
    )
    commit = receipt["git_commit"]
    rows = receipt["verified_head_files"]
    if (
        not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or not isinstance(rows, list)
        or len(rows) != len(TRAINING_CODE_LOGICAL_PATHS)
    ):
        raise HeldoutEvaluationError("checkpoint training code receipt identity differs")
    root = Path(__file__).resolve().parents[1]

    def git_bytes(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HeldoutEvaluationError(
                "cannot prove checkpoint training code from its Git commit"
            ) from exc
        return completed.stdout

    resolved_commit = git_bytes("rev-parse", "--verify", f"{commit}^{{commit}}").decode(
        "ascii"
    ).strip()
    if resolved_commit != commit:
        raise HeldoutEvaluationError("checkpoint training Git commit is unavailable")
    for logical_path, raw_row in zip(TRAINING_CODE_LOGICAL_PATHS, rows, strict=True):
        row = _require_exact_keys(
            raw_row,
            {
                "logical_path",
                "git_blob_sha1",
                "head_blob_sha256",
                "head_blob_size_bytes",
                "runtime_sha256",
                "runtime_size_bytes",
            },
            f"checkpoint training code row {logical_path}",
        )
        if row["logical_path"] != logical_path:
            raise HeldoutEvaluationError("checkpoint training code path order differs")
        blob_name = f"{commit}:{logical_path}"
        blob_sha1 = git_bytes("rev-parse", "--verify", blob_name).decode(
            "ascii"
        ).strip()
        payload = git_bytes("cat-file", "blob", blob_name)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if (
            blob_sha1 != row["git_blob_sha1"]
            or payload_sha256 != row["head_blob_sha256"]
            or payload_sha256 != row["runtime_sha256"]
            or len(payload) != row["head_blob_size_bytes"]
            or len(payload) != row["runtime_size_bytes"]
        ):
            raise HeldoutEvaluationError(
                f"checkpoint training code row differs from Git: {logical_path}"
            )


def validate_checkpoint_manifest(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
    preflight_receipt_sha256: str,
    split_assignments: Mapping[int, Mapping[int, str]],
    dataset_tree_sha256: str,
    heldout_window_manifest_sha256: str,
) -> tuple[dict, dict[str, dict[str, Any]]]:
    manifest = _require_exact_keys(
        value,
        {
            "manifest_version",
            "dataset_id",
            "production_preflight_receipt_sha256",
            "split_manifest_sha256",
            "heldout_window_manifest_sha256",
            "base_model",
            "checkpoints",
        },
        "checkpoint manifest",
    )
    if (
        manifest["manifest_version"] != CHECKPOINT_MANIFEST_VERSION
        or manifest["dataset_id"] != preflight_receipt["dataset_id"]
        or manifest["production_preflight_receipt_sha256"] != preflight_receipt_sha256
        or manifest["split_manifest_sha256"] != preflight_receipt["split_manifest_sha256"]
        or manifest["heldout_window_manifest_sha256"]
        != heldout_window_manifest_sha256
    ):
        raise HeldoutEvaluationError("checkpoint manifest source binding differs")
    base_model = _require_exact_keys(
        manifest["base_model"], {"path", "bundle_sha256"}, "base model entry"
    )
    if not isinstance(base_model["path"], str) or not Path(base_model["path"]).is_absolute():
        raise HeldoutEvaluationError("base model path must be absolute")
    _require_sha256(base_model["bundle_sha256"], "base model bundle digest")
    base_snapshot = snapshot_artifact_tree(base_model["path"])
    if base_snapshot["tree_sha256"] != base_model["bundle_sha256"]:
        raise HeldoutEvaluationError("base model bundle fingerprint differs")

    entries = manifest["checkpoints"]
    selected_seed = preflight_receipt.get("split_seed")
    if selected_seed not in EVALUATION_SEEDS:
        raise HeldoutEvaluationError("preflight does not select a production split seed")
    if not isinstance(entries, list) or len(entries) != len(CHECKPOINT_STEPS):
        raise HeldoutEvaluationError(
            "checkpoint manifest must contain exactly step 0/100/200 for the receipt seed"
        )
    snapshots: dict[str, dict[str, Any]] = {str(Path(base_model["path"]).absolute()): base_snapshot}
    seen = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_exact_keys(
            raw_entry,
            {
                "split_seed",
                "step",
                "checkpoint_path",
                "checkpoint_bundle_sha256",
                "current_training_data_receipt_path",
                "current_training_data_receipt_sha256",
            },
            f"checkpoint entry {index}",
        )
        key = (entry["split_seed"], entry["step"])
        if (
            key in seen
            or entry["split_seed"] != selected_seed
            or entry["step"] not in CHECKPOINT_STEPS
        ):
            raise HeldoutEvaluationError(f"invalid/duplicate checkpoint seed-step {key}")
        seen.add(key)
        if (
            not isinstance(entry["checkpoint_path"], str)
            or not Path(entry["checkpoint_path"]).is_absolute()
            or not isinstance(entry["current_training_data_receipt_path"], str)
            or not Path(entry["current_training_data_receipt_path"]).is_absolute()
        ):
            raise HeldoutEvaluationError(
                f"checkpoint/binding paths must be absolute for {key}"
            )
        checkpoint_path = str(Path(entry["checkpoint_path"]).expanduser().absolute())
        expected_binding_path = (
            Path(checkpoint_path).parent / "current_training_data_receipt.json"
        )
        if Path(entry["current_training_data_receipt_path"]) != expected_binding_path:
            raise HeldoutEvaluationError(
                f"checkpoint training-data binding is not adjacent for {key}"
            )
        snapshot = snapshots.get(checkpoint_path)
        if snapshot is None:
            snapshot = snapshot_artifact_tree(checkpoint_path)
            snapshots[checkpoint_path] = snapshot
        if snapshot["tree_sha256"] != _require_sha256(
            entry["checkpoint_bundle_sha256"], "checkpoint bundle digest"
        ):
            raise HeldoutEvaluationError(f"checkpoint bundle fingerprint differs for {key}")
        binding, binding_digest = _strict_json(
            entry["current_training_data_receipt_path"],
            f"checkpoint training-data binding {key}",
        )
        if binding_digest != _require_sha256(
            entry["current_training_data_receipt_sha256"],
            "checkpoint training-data binding digest",
        ):
            raise HeldoutEvaluationError(
                f"checkpoint training-data binding fingerprint differs for {key}"
            )
        binding = _require_exact_keys(
            binding,
            {
                "receipt_version",
                "checkpoint_step",
                "current_training_data_receipt",
                "current_training_data_receipt_sha256",
            },
            f"checkpoint training-data binding {key}",
        )
        current = binding["current_training_data_receipt"]
        current_digest = hashlib.sha256(_canonical_json_bytes(current)).hexdigest()
        if (
            binding["receipt_version"]
            != "contact-asym25-checkpoint-data-binding/v1"
            or binding["checkpoint_step"] != entry["step"]
            or binding["current_training_data_receipt_sha256"] != current_digest
        ):
            raise HeldoutEvaluationError(f"checkpoint binding content differs for {key}")
        current = _require_exact_keys(
            current,
            {
                "receipt_version",
                "base_training_data_receipt",
                "base_training_data_receipt_sha256",
                "production_preflight_receipt_sha256",
                "heldout_window_manifest_sha256",
                "training_gate_sha256",
                "split_seed",
                "train_episode_ids",
                "validation_episode_ids",
                "heldout_episode_ids",
                "training_code_receipt",
                "training_code_receipt_sha256",
                "dataset_source_tree_receipt",
            },
            f"current training-data receipt {key}",
        )
        episode_sets = _episode_sets(split_assignments[entry["split_seed"]])
        fixed_values = {
            "receipt_version": CURRENT_TRAINING_DATA_RECEIPT_VERSION,
            "base_training_data_receipt": preflight_receipt[
                "training_data_receipt"
            ],
            "base_training_data_receipt_sha256": preflight_receipt[
                "training_data_receipt_sha256"
            ],
            "production_preflight_receipt_sha256": preflight_receipt_sha256,
            "heldout_window_manifest_sha256": heldout_window_manifest_sha256,
            "training_gate_sha256": preflight_receipt["training_gate_sha256"],
            "split_seed": entry["split_seed"],
            "train_episode_ids": episode_sets["train"],
            "validation_episode_ids": episode_sets["val"],
            "heldout_episode_ids": episode_sets["test"],
        }
        for name, expected in fixed_values.items():
            if current.get(name) != expected:
                raise HeldoutEvaluationError(
                    f"checkpoint current training-data receipt {key} does not prove {name}"
                )
        source_tree = _require_exact_keys(
            current["dataset_source_tree_receipt"],
            {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
            f"checkpoint dataset source receipt {key}",
        )
        if source_tree["tree_sha256"] != dataset_tree_sha256:
            raise HeldoutEvaluationError(
                f"checkpoint {key} was trained from a different dataset source tree"
            )
        code_receipt = _require_exact_keys(
            current["training_code_receipt"],
            {"git_commit", "verified_head_files"},
            f"checkpoint training code receipt {key}",
        )
        if (
            hashlib.sha256(_canonical_json_bytes(code_receipt)).hexdigest()
            != current["training_code_receipt_sha256"]
            or not re.fullmatch(r"[0-9a-f]{40}", str(code_receipt["git_commit"]))
            or not isinstance(code_receipt["verified_head_files"], list)
            or not code_receipt["verified_head_files"]
        ):
            raise HeldoutEvaluationError(f"checkpoint training code receipt {key} is invalid")
        _validate_historical_training_code_receipt(code_receipt)
        config_path = Path(checkpoint_path) / "config.json"
        checkpoint_config, _ = _strict_json(
            config_path, f"checkpoint transformer config {key}"
        )
        if (
            checkpoint_config.get("current_training_data_receipt") != current
            or checkpoint_config.get("current_training_data_receipt_sha256")
            != current_digest
            or checkpoint_config.get("production_checkpoint_steps")
            != list(CHECKPOINT_STEPS)
        ):
            raise HeldoutEvaluationError(
                f"checkpoint transformer metadata is not bound to current receipt {key}"
            )
    if seen != {(selected_seed, step) for step in CHECKPOINT_STEPS}:
        raise HeldoutEvaluationError("checkpoint manifest seed/step matrix is incomplete")
    return manifest, snapshots


def load_checkpoint_manifest(
    path: str | Path, **validation_kwargs: Any
) -> tuple[dict, str, dict[str, dict[str, Any]]]:
    value, digest = _strict_json(path, "held-out checkpoint manifest")
    manifest, snapshots = validate_checkpoint_manifest(
        value, **validation_kwargs
    )
    return manifest, digest, snapshots


ACTION_GROUPS = {
    "overall": tuple(range(ASYMMETRIC_ACTION_DIM)),
    "wrist": tuple(range(LEFT_WRIST_SLICE.start, RIGHT_WRIST_SLICE.stop)),
    "grip": tuple(range(RH56_SLICE.start, OMNIPICKER_APERTURE_INDEX + 1)),
    "left_overall": tuple(range(LEFT_WRIST_SLICE.start, LEFT_WRIST_SLICE.stop))
    + tuple(range(RH56_SLICE.start, RH56_SLICE.stop)),
    "left_wrist": tuple(range(LEFT_WRIST_SLICE.start, LEFT_WRIST_SLICE.stop)),
    "left_grip": tuple(range(RH56_SLICE.start, RH56_SLICE.stop)),
    "right_overall": tuple(range(RIGHT_WRIST_SLICE.start, RIGHT_WRIST_SLICE.stop))
    + (OMNIPICKER_APERTURE_INDEX,),
    "right_wrist": tuple(range(RIGHT_WRIST_SLICE.start, RIGHT_WRIST_SLICE.stop)),
    "right_grip": (OMNIPICKER_APERTURE_INDEX,),
}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def activate_contact_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM) or not np.isfinite(logits).all():
        raise HeldoutEvaluationError(
            f"contact logits must be finite [{NUM_CONTACT_QUERIES},{SEGMENT_TARGET_DIM}]"
        )
    semantic = np.zeros_like(logits)
    for field in SEGMENT_BINARY_FIELDS + SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        semantic[..., field_slice] = _sigmoid(logits[..., field_slice])
    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        group = logits[..., field_slice]
        group = group - group.max(axis=-1, keepdims=True)
        exp_group = np.exp(group)
        semantic[..., field_slice] = exp_group / exp_group.sum(axis=-1, keepdims=True)
    return semantic


def contact_grammar_validity(hard: np.ndarray) -> np.ndarray:
    hard = np.asarray(hard)
    if hard.shape != (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM) or not np.isfinite(hard).all():
        return np.zeros(NUM_CONTACT_QUERIES, dtype=bool)
    valid = np.ones(NUM_CONTACT_QUERIES, dtype=bool)
    presence = hard[..., SEGMENT_FIELD_SLICES["presence"]].squeeze(-1)
    valid &= (presence == 0.0) | (presence == 1.0)
    for field in ("left_censored", "right_censored"):
        value = hard[..., SEGMENT_FIELD_SLICES[field]].squeeze(-1)
        valid &= (value == 0.0) | (value == 1.0)
    for field in SEGMENT_CATEGORICAL_FIELDS:
        group = hard[..., SEGMENT_FIELD_SLICES[field]]
        present_valid = np.all((group == 0.0) | (group == 1.0), axis=-1) & (
            group.sum(axis=-1) == 1.0
        )
        absent_valid = np.all(group == 0.0, axis=-1)
        valid &= np.where(presence == 1.0, present_valid, absent_valid)
    step = 1.0 / float(TIME_BINS - 1)
    start = hard[..., SEGMENT_FIELD_SLICES["start_tau"]].squeeze(-1)
    end = hard[..., SEGMENT_FIELD_SLICES["end_tau"]].squeeze(-1)
    left = hard[..., SEGMENT_FIELD_SLICES["left_censored"]].squeeze(-1)
    right = hard[..., SEGMENT_FIELD_SLICES["right_censored"]].squeeze(-1)
    for value in (start, end):
        valid &= (value >= 0.0) & (value <= 1.0)
        nearest = np.round(value / step) * step
        valid &= np.isclose(
            value, nearest, atol=HARD_TIME_GRID_ATOL, rtol=0.0
        )
    valid &= np.where(presence == 1.0, start <= end, (start == 0.0) & (end == 0.0))
    valid &= np.where((presence == 1.0) & (left == 1.0), start == 0.0, True)
    valid &= np.where((presence == 1.0) & (right == 1.0), end == 1.0, True)
    for field in SEGMENT_CATEGORICAL_FIELDS + SEGMENT_CONTINUOUS_FIELDS + (
        "left_censored",
        "right_censored",
    ):
        group = hard[..., SEGMENT_FIELD_SLICES[field]]
        valid &= np.where(presence[:, None] == 0.0, group == 0.0, True).all(axis=-1)
    return valid


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _canonical_hard_time(value: np.ndarray) -> np.ndarray:
    """Recover exact B16 values after float16/bfloat16 action transport."""

    return np.clip(
        np.round(np.asarray(value, dtype=np.float64) * (TIME_BINS - 1)),
        0,
        TIME_BINS - 1,
    ) / float(TIME_BINS - 1)


def _as_finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise HeldoutEvaluationError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array.astype(np.float64, copy=False)


def score_heldout_window(
    *,
    window: Mapping[str, Any],
    split_seed: int,
    checkpoint_step: int,
    checkpoint_sha256: str,
    action_prediction: object,
    action_target: object,
    action_dim_mask: object,
    action_time_mask: object,
    action_denorm_scale: object,
    contact_logits: object,
    contact_hard_semantic: object,
    contact_target: object,
    contact_loss_mask: object,
    contact_confidence: object,
    contact_condition_mask: object,
) -> dict[str, Any]:
    """Return additive per-window statistics; never invent metrics for missing labels."""

    if split_seed != window["seed"]:
        raise HeldoutEvaluationError("checkpoint split seed differs from window split seed")
    pred = _as_finite_array(action_prediction, (HORIZON_FRAMES, MODEL_ACTION_DIM), "action prediction")
    target = _as_finite_array(action_target, (HORIZON_FRAMES, MODEL_ACTION_DIM), "action target")
    dim_mask = np.asarray(action_dim_mask, dtype=bool)
    time_mask = np.asarray(action_time_mask, dtype=bool)
    expected_dim_mask = np.zeros(MODEL_ACTION_DIM, dtype=bool)
    expected_dim_mask[:ASYMMETRIC_ACTION_DIM] = True
    if not np.array_equal(dim_mask, expected_dim_mask):
        raise HeldoutEvaluationError("action_dim_mask is not canonical asymmetric25 active-only")
    if time_mask.shape != (HORIZON_FRAMES,) or not bool(time_mask.all()):
        raise HeldoutEvaluationError("fixed held-out windows must have 48 active action frames")
    scale = _as_finite_array(action_denorm_scale, (MODEL_ACTION_DIM,), "action denorm scale")
    if np.any(scale[:ASYMMETRIC_ACTION_DIM] <= 0.0):
        raise HeldoutEvaluationError("active action denormalization scales must be positive")
    if np.any(scale[ASYMMETRIC_ACTION_DIM:] != 1.0):
        raise HeldoutEvaluationError("inactive action padding scales must be canonical one")
    sq_norm = (pred - target) ** 2
    sq_physical = ((pred - target) * scale[None, :]) ** 2

    row: dict[str, Any] = {
        "window_id": window["window_id"],
        "split_seed": split_seed,
        "checkpoint_step": checkpoint_step,
        "checkpoint_bundle_sha256": _require_sha256(
            checkpoint_sha256, "checkpoint bundle digest"
        ),
        "episode_id": window["episode_id"],
        "split": window["split"],
        "frame_index": window["frame_index"],
        "global_index": window["global_index"],
        "action_noise_seed": window["action_noise_seed"],
    }
    for space, values in (("normalized", sq_norm), ("physical", sq_physical)):
        for group, dimensions in ACTION_GROUPS.items():
            group_values = values[:, dimensions]
            count = int(group_values.size)
            sse = float(group_values.sum(dtype=np.float64))
            row[f"action_{space}_{group}_sse"] = sse
            row[f"action_{space}_{group}_count"] = count
            row[f"action_{space}_{group}_mse"] = sse / count

    target_contact = _as_finite_array(
        contact_target, (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM), "contact target"
    ).astype(np.float32)
    loss_mask = _as_finite_array(
        contact_loss_mask, (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM), "contact loss mask"
    ).astype(np.float32)
    confidence = _as_finite_array(
        contact_confidence, (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM), "contact confidence"
    ).astype(np.float32)
    condition_mask = _as_finite_array(
        contact_condition_mask,
        (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM),
        "contact condition mask",
    ).astype(np.float32)
    validate_contact_sidecar_arrays(
        {
            "contact_target": target_contact[None],
            "contact_loss_mask": loss_mask[None],
            "contact_confidence": confidence[None],
            "contact_condition_mask": condition_mask[None],
        },
        frame_count=1,
    )
    logits = _as_finite_array(
        contact_logits, (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM), "raw contact logits"
    )
    hard = _as_finite_array(
        contact_hard_semantic,
        (NUM_CONTACT_QUERIES, SEGMENT_TARGET_DIM),
        "action-visible hard contact semantic",
    )
    grammar = contact_grammar_validity(hard)
    row["contact_grammar_valid_count"] = int(grammar.sum())
    row["contact_grammar_query_count"] = int(grammar.size)
    if not bool(grammar.all()):
        raise HeldoutEvaluationError(
            "hard_st action-visible contact output violates the finite grammar"
        )
    soft = activate_contact_logits(logits)
    presence_slice = SEGMENT_FIELD_SLICES["presence"]
    gt_presence = target_contact[..., presence_slice].squeeze(-1).astype(np.int64)
    presence_mask = loss_mask[..., presence_slice].squeeze(-1) == 1.0
    hard_presence = hard[..., presence_slice].squeeze(-1).astype(np.int64)
    soft_presence = soft[..., presence_slice].squeeze(-1)
    row["contact_presence_tp"] = int(np.sum(presence_mask & (gt_presence == 1) & (hard_presence == 1)))
    row["contact_presence_fp"] = int(np.sum(presence_mask & (gt_presence == 0) & (hard_presence == 1)))
    row["contact_presence_fn"] = int(np.sum(presence_mask & (gt_presence == 1) & (hard_presence == 0)))
    row["contact_presence_tn"] = int(np.sum(presence_mask & (gt_presence == 0) & (hard_presence == 0)))
    row["contact_presence_brier_sum"] = float(
        np.sum(((soft_presence - gt_presence) ** 2)[presence_mask], dtype=np.float64)
    )
    row["contact_presence_brier_count"] = int(presence_mask.sum())

    present_supervised = presence_mask & (gt_presence == 1)
    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        group_mask = present_supervised & np.all(loss_mask[..., field_slice] == 1.0, axis=-1)
        gt_index = target_contact[..., field_slice].argmax(axis=-1)
        hard_index = hard[..., field_slice].argmax(axis=-1)
        row[f"contact_{field}_correct"] = int(np.sum(group_mask & (gt_index == hard_index)))
        row[f"contact_{field}_count"] = int(group_mask.sum())
        selected_probability = soft[..., field_slice][np.arange(NUM_CONTACT_QUERIES), gt_index]
        row[f"contact_{field}_nll_sum"] = float(
            -np.log(np.clip(selected_probability[group_mask], 1e-12, 1.0)).sum(dtype=np.float64)
        )
        row[f"contact_{field}_nll_count"] = int(group_mask.sum())
        row[f"missing_{field}_eligible_count"] = int(present_supervised.sum())
        row[f"missing_{field}_supervised_count"] = int(group_mask.sum())

    for field in SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        field_mask = present_supervised & (loss_mask[..., field_slice].squeeze(-1) == 1.0)
        gt_value = target_contact[..., field_slice].squeeze(-1)
        hard_value = _canonical_hard_time(
            hard[..., field_slice].squeeze(-1)
        )
        soft_value = soft[..., field_slice].squeeze(-1)
        row[f"contact_{field}_hard_abs_error_sum"] = float(
            np.abs(hard_value - gt_value)[field_mask].sum(dtype=np.float64)
        )
        row[f"contact_{field}_soft_abs_error_sum"] = float(
            np.abs(soft_value - gt_value)[field_mask].sum(dtype=np.float64)
        )
        row[f"contact_{field}_count"] = int(field_mask.sum())
        row[f"missing_{field}_eligible_count"] = int(present_supervised.sum())
        row[f"missing_{field}_supervised_count"] = int(field_mask.sum())

    for field in ("left_censored", "right_censored"):
        field_slice = SEGMENT_FIELD_SLICES[field]
        field_mask = present_supervised & (loss_mask[..., field_slice].squeeze(-1) == 1.0)
        gt_value = target_contact[..., field_slice].squeeze(-1)
        hard_value = hard[..., field_slice].squeeze(-1)
        row[f"contact_{field}_correct"] = int(np.sum(field_mask & (gt_value == hard_value)))
        row[f"contact_{field}_count"] = int(field_mask.sum())
        row[f"missing_{field}_eligible_count"] = int(present_supervised.sum())
        row[f"missing_{field}_supervised_count"] = int(field_mask.sum())

    fully_labeled = present_supervised & np.all(loss_mask == 1.0, axis=-1)
    exact = hard_presence == gt_presence
    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        exact &= hard[..., field_slice].argmax(axis=-1) == target_contact[..., field_slice].argmax(axis=-1)
    for field in ("left_censored", "right_censored"):
        field_slice = SEGMENT_FIELD_SLICES[field]
        exact &= hard[..., field_slice].squeeze(-1) == target_contact[..., field_slice].squeeze(-1)
    tau_tolerance = 0.5 / float(TIME_BINS - 1) + 1e-7
    for field in SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        exact &= (
            np.abs(
                _canonical_hard_time(hard[..., field_slice].squeeze(-1))
                - target_contact[..., field_slice].squeeze(-1)
            )
            <= tau_tolerance
        )
    row["contact_segment_exact_correct"] = int(np.sum(fully_labeled & exact))
    row["contact_segment_exact_count"] = int(fully_labeled.sum())
    row["missing_presence_eligible_count"] = NUM_CONTACT_QUERIES
    row["missing_presence_supervised_count"] = int(presence_mask.sum())
    row["contact_supervised_element_count"] = int(loss_mask.sum())
    row["contact_total_element_count"] = int(loss_mask.size)
    row["contact_positive_segment_count"] = int(present_supervised.sum())
    return row


def per_window_csv_header() -> list[str]:
    header = [
        "window_id",
        "split_seed",
        "checkpoint_step",
        "checkpoint_bundle_sha256",
        "split",
        "episode_id",
        "frame_index",
        "global_index",
        "action_noise_seed",
    ]
    for space in ("normalized", "physical"):
        for group in ACTION_GROUPS:
            header.extend(
                [
                    f"action_{space}_{group}_sse",
                    f"action_{space}_{group}_count",
                    f"action_{space}_{group}_mse",
                ]
            )
    header.extend(
        [
            "contact_presence_tp",
            "contact_presence_fp",
            "contact_presence_fn",
            "contact_presence_tn",
            "contact_presence_brier_sum",
            "contact_presence_brier_count",
        ]
    )
    for field in SEGMENT_CATEGORICAL_FIELDS:
        header.extend(
            [
                f"contact_{field}_correct",
                f"contact_{field}_count",
                f"contact_{field}_nll_sum",
                f"contact_{field}_nll_count",
                f"missing_{field}_eligible_count",
                f"missing_{field}_supervised_count",
            ]
        )
    for field in SEGMENT_CONTINUOUS_FIELDS:
        header.extend(
            [
                f"contact_{field}_hard_abs_error_sum",
                f"contact_{field}_soft_abs_error_sum",
                f"contact_{field}_count",
                f"missing_{field}_eligible_count",
                f"missing_{field}_supervised_count",
            ]
        )
    for field in ("left_censored", "right_censored"):
        header.extend(
            [
                f"contact_{field}_correct",
                f"contact_{field}_count",
                f"missing_{field}_eligible_count",
                f"missing_{field}_supervised_count",
            ]
        )
    header.extend(
        [
            "contact_segment_exact_correct",
            "contact_segment_exact_count",
            "contact_grammar_valid_count",
            "contact_grammar_query_count",
            "missing_presence_eligible_count",
            "missing_presence_supervised_count",
            "contact_supervised_element_count",
            "contact_total_element_count",
            "contact_positive_segment_count",
        ]
    )
    if len(header) != len(set(header)):
        raise AssertionError("per-window CSV header has duplicate columns")
    return header


def write_per_window_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False
) -> str:
    path = Path(path).expanduser().absolute()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite per-window CSV: {path}")
    header = per_window_csv_header()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale partial output exists: {partial}")
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                if set(row) != set(header):
                    raise HeldoutEvaluationError("per-window row schema mismatch")
                writer.writerow(row)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return sha256_file(path)


def read_per_window_csv(
    path: str | Path, *, expected_sha256: str
) -> list[dict[str, Any]]:
    """Read additive rows with an exact schema and strict numeric types."""

    path = Path(path).expanduser().absolute()
    if sha256_file(path) != _require_sha256(
        expected_sha256, "per-window CSV digest"
    ):
        raise HeldoutEvaluationError("per-window CSV fingerprint differs")
    header = per_window_csv_header()
    string_fields = {"window_id", "checkpoint_bundle_sha256", "split"}
    integer_fields = {
        "split_seed",
        "checkpoint_step",
        "episode_id",
        "frame_index",
        "global_index",
        "action_noise_seed",
        "contact_presence_tp",
        "contact_presence_fp",
        "contact_presence_fn",
        "contact_presence_tn",
    } | {name for name in header if name.endswith("_count")}
    rows = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != header:
                raise HeldoutEvaluationError("per-window CSV header mismatch")
            for row_number, raw in enumerate(reader, 2):
                if set(raw) != set(header) or any(value is None for value in raw.values()):
                    raise HeldoutEvaluationError(
                        f"per-window CSV row {row_number} schema mismatch"
                    )
                row: dict[str, Any] = {}
                for name in header:
                    value = raw[name]
                    if name in string_fields:
                        if not value:
                            raise HeldoutEvaluationError(
                                f"empty {name} in per-window row {row_number}"
                            )
                        row[name] = value
                    elif name in integer_fields:
                        if not re.fullmatch(r"0|[1-9][0-9]*", value):
                            raise HeldoutEvaluationError(
                                f"invalid integer {name} in per-window row {row_number}"
                            )
                        row[name] = int(value)
                    else:
                        try:
                            number = float(value)
                        except ValueError as exc:
                            raise HeldoutEvaluationError(
                                f"invalid number {name} in per-window row {row_number}"
                            ) from exc
                        if not math.isfinite(number):
                            raise HeldoutEvaluationError(
                                f"non-finite {name} in per-window row {row_number}"
                            )
                        row[name] = number
                if row["split"] not in {"val", "test"}:
                    raise HeldoutEvaluationError(
                        f"invalid split in per-window row {row_number}"
                    )
                _require_sha256(
                    row["checkpoint_bundle_sha256"],
                    f"per-window checkpoint digest row {row_number}",
                )
                rows.append(row)
    except OSError as exc:
        raise HeldoutEvaluationError(f"cannot read per-window CSV: {path}") from exc
    if not rows:
        raise HeldoutEvaluationError("per-window CSV is empty")
    return rows


def _sum(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return float(sum(float(row[name]) for row in rows))


def reduce_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict:
    if not rows:
        raise HeldoutEvaluationError("cannot reduce an empty evaluation result")
    action = {}
    for space in ("normalized", "physical"):
        action[space] = {}
        for group in ACTION_GROUPS:
            sse = _sum(rows, f"action_{space}_{group}_sse")
            count = int(_sum(rows, f"action_{space}_{group}_count"))
            if count <= 0:
                raise HeldoutEvaluationError(f"empty active action group: {group}")
            action[space][group] = {
                "mse": sse / count,
                "squared_error_sum": sse,
                "active_element_count": count,
            }
    tp = int(_sum(rows, "contact_presence_tp"))
    fp = int(_sum(rows, "contact_presence_fp"))
    fn = int(_sum(rows, "contact_presence_fn"))
    tn = int(_sum(rows, "contact_presence_tn"))
    if tp + fn <= 0 or fp + tn <= 0:
        raise HeldoutEvaluationError(
            "held-out labels require both positive and negative supervised presence"
        )
    brier_count = int(_sum(rows, "contact_presence_brier_count"))
    categorical = {}
    missingness = {}
    for field in SEGMENT_CATEGORICAL_FIELDS:
        count = int(_sum(rows, f"contact_{field}_count"))
        if count <= 0:
            raise HeldoutEvaluationError(f"held-out labels are empty for contact field {field}")
        categorical[field] = {
            "hard_decision_accuracy": _sum(rows, f"contact_{field}_correct") / count,
            "hard_decision_count": count,
            "soft_probability_nll": _sum(rows, f"contact_{field}_nll_sum") / count,
        }
        eligible = int(_sum(rows, f"missing_{field}_eligible_count"))
        supervised = int(_sum(rows, f"missing_{field}_supervised_count"))
        missingness[field] = {
            "eligible_positive_query_count": eligible,
            "supervised_query_count": supervised,
            "coverage": _ratio(supervised, eligible),
        }
    timing = {}
    for field in SEGMENT_CONTINUOUS_FIELDS:
        count = int(_sum(rows, f"contact_{field}_count"))
        if count <= 0:
            raise HeldoutEvaluationError(f"held-out labels are empty for contact field {field}")
        timing[field] = {
            "hard_b16_mae": _sum(rows, f"contact_{field}_hard_abs_error_sum") / count,
            "soft_probability_mae": _sum(rows, f"contact_{field}_soft_abs_error_sum") / count,
            "supervised_count": count,
        }
        eligible = int(_sum(rows, f"missing_{field}_eligible_count"))
        supervised = int(_sum(rows, f"missing_{field}_supervised_count"))
        missingness[field] = {
            "eligible_positive_query_count": eligible,
            "supervised_query_count": supervised,
            "coverage": _ratio(supervised, eligible),
        }
    censor = {}
    for field in ("left_censored", "right_censored"):
        count = int(_sum(rows, f"contact_{field}_count"))
        if count <= 0:
            raise HeldoutEvaluationError(f"held-out labels are empty for contact field {field}")
        censor[field] = {
            "hard_decision_accuracy": _sum(rows, f"contact_{field}_correct") / count,
            "hard_decision_count": count,
        }
        eligible = int(_sum(rows, f"missing_{field}_eligible_count"))
        supervised = int(_sum(rows, f"missing_{field}_supervised_count"))
        missingness[field] = {
            "eligible_positive_query_count": eligible,
            "supervised_query_count": supervised,
            "coverage": _ratio(supervised, eligible),
        }
    exact_count = int(_sum(rows, "contact_segment_exact_count"))
    if exact_count <= 0:
        raise HeldoutEvaluationError("held-out labels contain no fully supervised positive segment")
    grammar_count = int(_sum(rows, "contact_grammar_query_count"))
    presence_eligible = int(_sum(rows, "missing_presence_eligible_count"))
    presence_supervised = int(_sum(rows, "missing_presence_supervised_count"))
    missingness["presence"] = {
        "eligible_query_count": presence_eligible,
        "supervised_query_count": presence_supervised,
        "coverage": _ratio(presence_supervised, presence_eligible),
    }
    supervised_elements = int(_sum(rows, "contact_supervised_element_count"))
    total_elements = int(_sum(rows, "contact_total_element_count"))
    missingness["all_elements"] = {
        "supervised_element_count": supervised_elements,
        "total_element_count": total_elements,
        "coverage": _ratio(supervised_elements, total_elements),
    }
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 0.0 if recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "window_count": len(rows),
        "action": action,
        "contact": {
            "prediction_contract": {
                "soft": (
                    "raw planner logits activated as Bernoulli/softmax probabilities; "
                    "reported only with Brier/NLL/soft-MAE, never as accuracy"
                ),
                "hard": (
                    "model action-visible hard_st B16 finite grammar output; "
                    "all decision accuracy/F1/exact metrics use this tensor"
                ),
                "presence_threshold": 0.5,
                "time_bins": TIME_BINS,
                "hard_time_grid_atol_after_low_precision_transport": (
                    HARD_TIME_GRID_ATOL
                ),
            },
            "presence": {
                "hard_decision_precision": precision,
                "hard_decision_recall": recall,
                "hard_decision_f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "soft_probability_brier": _sum(rows, "contact_presence_brier_sum") / brier_count,
                "supervised_count": brier_count,
            },
            "categorical": categorical,
            "timing": timing,
            "censor": censor,
            "segment_exact": {
                "hard_b16_fully_supervised_accuracy": (
                    _sum(rows, "contact_segment_exact_correct") / exact_count
                ),
                "fully_supervised_positive_count": exact_count,
                "tau_tolerance": 0.5 / float(TIME_BINS - 1) + 1e-7,
            },
            "grammar_validity": {
                "hard_action_visible_valid_fraction": (
                    _sum(rows, "contact_grammar_valid_count") / grammar_count
                ),
                "query_count": grammar_count,
            },
            "missingness_coverage": missingness,
        },
    }


def load_evaluation_result_bundle(path: str | Path) -> dict[str, Any]:
    """Replay a single-seed report and its additive evidence without a model."""

    result_path = Path(path).expanduser().absolute()
    result, result_sha256 = _strict_json(result_path, "held-out evaluation result")
    result = _require_exact_keys(
        result,
        {
            "result_version",
            "status",
            "scientific_claim_automatic",
            "source_receipt",
            "checkpoint_receipt",
            "code_receipt",
            "environment_receipt",
            "evaluation_contract",
            "results_by_seed_step_split",
            "per_window_csv",
        },
        "held-out evaluation result",
    )
    if (
        result["result_version"] != RESULT_VERSION
        or result["status"] != "passed"
        or result["scientific_claim_automatic"] is not False
    ):
        raise HeldoutEvaluationError("held-out result status/version contract differs")
    source = _require_exact_keys(
        result["source_receipt"],
        {
            "production_preflight_receipt",
            "split_manifest",
            "fixed_window_manifest",
            "dataset_tree",
        },
        "held-out source receipt",
    )
    preflight_entry = _require_exact_keys(
        source["production_preflight_receipt"],
        {"path_record_only", "sha256"},
        "result preflight receipt",
    )
    preflight, preflight_sha256 = load_and_replay_preflight_receipt(
        preflight_entry["path_record_only"]
    )
    if preflight_sha256 != _require_sha256(
        preflight_entry["sha256"], "result preflight receipt digest"
    ):
        raise HeldoutEvaluationError("result preflight receipt fingerprint differs")
    split_entry = _require_exact_keys(
        source["split_manifest"], {"path_record_only", "sha256"},
        "result split receipt",
    )
    if (
        split_entry["sha256"] != preflight["split_manifest_sha256"]
        or Path(split_entry["path_record_only"]).expanduser().absolute()
        != Path(preflight["split_manifest_path"]).expanduser().absolute()
    ):
        raise HeldoutEvaluationError("result split receipt differs from preflight")
    _, assignments = load_completed_group_split(
        split_entry["path_record_only"],
        expected_sha256=split_entry["sha256"],
        expected_episode_lengths=preflight["episode_lengths"],
    )
    window_entry = _require_exact_keys(
        source["fixed_window_manifest"], {"path_record_only", "sha256"},
        "result window receipt",
    )
    windows, window_manifest_sha256 = load_fixed_window_manifest(
        window_entry["path_record_only"],
        preflight_receipt=preflight,
        preflight_receipt_sha256=preflight_sha256,
        split_assignments=assignments,
    )
    if window_manifest_sha256 != _require_sha256(
        window_entry["sha256"], "result fixed-window manifest digest"
    ):
        raise HeldoutEvaluationError("result fixed-window manifest digest differs")
    dataset_tree = _require_exact_keys(
        source["dataset_tree"],
        {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
        "result dataset-tree receipt",
    )
    if (
        dataset_tree["path_record_only"] != preflight["data_root"]
        or not isinstance(dataset_tree["file_count"], int)
        or isinstance(dataset_tree["file_count"], bool)
        or dataset_tree["file_count"] <= 0
        or not isinstance(dataset_tree["total_size_bytes"], int)
        or isinstance(dataset_tree["total_size_bytes"], bool)
        or dataset_tree["total_size_bytes"] <= 0
    ):
        raise HeldoutEvaluationError("invalid result dataset-tree receipt")
    _require_sha256(dataset_tree["tree_sha256"], "result dataset-tree digest")

    checkpoint_receipt = _require_exact_keys(
        result["checkpoint_receipt"], {"manifest", "verified_artifact_trees"},
        "result checkpoint receipt",
    )
    checkpoint_entry = _require_exact_keys(
        checkpoint_receipt["manifest"], {"path_record_only", "sha256"},
        "result checkpoint-manifest receipt",
    )
    checkpoint_manifest, checkpoint_manifest_sha256 = _strict_json(
        checkpoint_entry["path_record_only"], "result checkpoint manifest"
    )
    if checkpoint_manifest_sha256 != _require_sha256(
        checkpoint_entry["sha256"], "result checkpoint-manifest digest"
    ):
        raise HeldoutEvaluationError("result checkpoint-manifest digest differs")
    if not isinstance(checkpoint_receipt["verified_artifact_trees"], list):
        raise HeldoutEvaluationError("result checkpoint artifact receipts must be a list")
    checkpoint_manifest = _require_exact_keys(
        checkpoint_manifest,
        {
            "manifest_version",
            "dataset_id",
            "production_preflight_receipt_sha256",
            "split_manifest_sha256",
            "heldout_window_manifest_sha256",
            "base_model",
            "checkpoints",
        },
        "result checkpoint manifest",
    )
    if (
        checkpoint_manifest["manifest_version"] != CHECKPOINT_MANIFEST_VERSION
        or checkpoint_manifest["dataset_id"] != preflight["dataset_id"]
        or checkpoint_manifest["production_preflight_receipt_sha256"]
        != preflight_sha256
        or checkpoint_manifest["split_manifest_sha256"]
        != preflight["split_manifest_sha256"]
        or checkpoint_manifest["heldout_window_manifest_sha256"]
        != window_manifest_sha256
    ):
        raise HeldoutEvaluationError("result checkpoint manifest source binding differs")
    checkpoint_rows = checkpoint_manifest["checkpoints"]
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != len(
        CHECKPOINT_STEPS
    ):
        raise HeldoutEvaluationError("result checkpoint manifest matrix differs")
    observed_checkpoint_keys = set()
    for index, entry in enumerate(checkpoint_rows):
        entry = _require_exact_keys(
            entry,
            {
                "split_seed",
                "step",
                "checkpoint_path",
                "checkpoint_bundle_sha256",
                "current_training_data_receipt_path",
                "current_training_data_receipt_sha256",
            },
            f"result checkpoint entry {index}",
        )
        if entry["split_seed"] != preflight["split_seed"]:
            raise HeldoutEvaluationError("result checkpoint seed differs")
        if (
            not isinstance(entry["checkpoint_path"], str)
            or not Path(entry["checkpoint_path"]).is_absolute()
            or not isinstance(entry["current_training_data_receipt_path"], str)
            or Path(entry["current_training_data_receipt_path"])
            != Path(entry["checkpoint_path"]).parent
            / "current_training_data_receipt.json"
        ):
            raise HeldoutEvaluationError(
                "result checkpoint binding path is not absolute/adjacent"
            )
        observed_checkpoint_keys.add((entry["split_seed"], entry["step"]))
        _require_sha256(
            entry["checkpoint_bundle_sha256"], "result checkpoint bundle digest"
        )
        _require_sha256(
            entry["current_training_data_receipt_sha256"],
            "result checkpoint data-binding digest",
        )
    if observed_checkpoint_keys != {
        (preflight["split_seed"], step) for step in CHECKPOINT_STEPS
    }:
        raise HeldoutEvaluationError("result checkpoint step matrix differs")
    result_base_model = _require_exact_keys(
        checkpoint_manifest["base_model"], {"path", "bundle_sha256"},
        "result base-model receipt",
    )
    if (
        not isinstance(result_base_model["path"], str)
        or not Path(result_base_model["path"]).is_absolute()
    ):
        raise HeldoutEvaluationError("result base-model path must be absolute")
    _require_sha256(
        result_base_model["bundle_sha256"],
        "result base-model bundle digest",
    )
    for index, tree in enumerate(checkpoint_receipt["verified_artifact_trees"]):
        tree = _require_exact_keys(
            tree,
            {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
            f"result checkpoint artifact tree {index}",
        )
        _require_sha256(tree["tree_sha256"], "result checkpoint artifact digest")
    recorded_trees = {
        (tree["path_record_only"], tree["tree_sha256"])
        for tree in checkpoint_receipt["verified_artifact_trees"]
    }
    expected_trees = {
        (
            str(Path(checkpoint_manifest["base_model"]["path"]).absolute()),
            checkpoint_manifest["base_model"]["bundle_sha256"],
        )
    } | {
        (
            str(Path(entry["checkpoint_path"]).absolute()),
            entry["checkpoint_bundle_sha256"],
        )
        for entry in checkpoint_rows
    }
    if recorded_trees != expected_trees or len(recorded_trees) != len(
        checkpoint_receipt["verified_artifact_trees"]
    ):
        raise HeldoutEvaluationError(
            "result checkpoint artifact receipts do not match its manifest"
        )
    episode_sets = _episode_sets(assignments[preflight["split_seed"]])
    for entry in checkpoint_rows:
        binding, binding_sha256 = _strict_json(
            entry["current_training_data_receipt_path"],
            "result checkpoint training-data binding",
        )
        if binding_sha256 != entry["current_training_data_receipt_sha256"]:
            raise HeldoutEvaluationError(
                "result checkpoint training-data binding fingerprint differs"
            )
        binding = _require_exact_keys(
            binding,
            {
                "receipt_version",
                "checkpoint_step",
                "current_training_data_receipt",
                "current_training_data_receipt_sha256",
            },
            "result checkpoint training-data binding",
        )
        current = binding["current_training_data_receipt"]
        current_sha256 = hashlib.sha256(_canonical_json_bytes(current)).hexdigest()
        if (
            binding["receipt_version"]
            != "contact-asym25-checkpoint-data-binding/v1"
            or binding["checkpoint_step"] != entry["step"]
            or binding["current_training_data_receipt_sha256"] != current_sha256
        ):
            raise HeldoutEvaluationError(
                "result checkpoint training-data binding content differs"
            )
        current = _require_exact_keys(
            current,
            {
                "receipt_version",
                "base_training_data_receipt",
                "base_training_data_receipt_sha256",
                "production_preflight_receipt_sha256",
                "heldout_window_manifest_sha256",
                "training_gate_sha256",
                "split_seed",
                "train_episode_ids",
                "validation_episode_ids",
                "heldout_episode_ids",
                "training_code_receipt",
                "training_code_receipt_sha256",
                "dataset_source_tree_receipt",
            },
            "result current training-data receipt",
        )
        current_source_tree = _require_exact_keys(
            current["dataset_source_tree_receipt"],
            {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
            "result current dataset-source receipt",
        )
        if (
            current["receipt_version"] != CURRENT_TRAINING_DATA_RECEIPT_VERSION
            or current["base_training_data_receipt"]
            != preflight["training_data_receipt"]
            or current["base_training_data_receipt_sha256"]
            != preflight["training_data_receipt_sha256"]
            or current["production_preflight_receipt_sha256"] != preflight_sha256
            or current["heldout_window_manifest_sha256"]
            != window_manifest_sha256
            or current["training_gate_sha256"] != preflight["training_gate_sha256"]
            or current["split_seed"] != preflight["split_seed"]
            or current["train_episode_ids"] != episode_sets["train"]
            or current["validation_episode_ids"] != episode_sets["val"]
            or current["heldout_episode_ids"] != episode_sets["test"]
            or current_source_tree != dataset_tree
        ):
            raise HeldoutEvaluationError(
                "result checkpoint receipt does not match source/split/windows"
            )
        code_receipt = current["training_code_receipt"]
        if (
            hashlib.sha256(_canonical_json_bytes(code_receipt)).hexdigest()
            != current["training_code_receipt_sha256"]
        ):
            raise HeldoutEvaluationError(
                "result checkpoint training code receipt digest differs"
            )
        _validate_historical_training_code_receipt(code_receipt)
        transformer_config, _ = _strict_json(
            Path(entry["checkpoint_path"]) / "config.json",
            "result checkpoint transformer config",
        )
        if (
            transformer_config.get("current_training_data_receipt") != current
            or transformer_config.get("current_training_data_receipt_sha256")
            != current_sha256
            or transformer_config.get("production_checkpoint_steps")
            != list(CHECKPOINT_STEPS)
        ):
            raise HeldoutEvaluationError(
                "result checkpoint transformer metadata differs from its receipt"
            )
    _require_exact_keys(
        result["code_receipt"],
        {"git_commit", "evaluator_version", "verified_head_files"},
        "result code receipt",
    )
    environment = _require_exact_keys(
        result["environment_receipt"],
        {
            "python",
            "platform",
            "packages",
            "cuda_record_only",
            "evaluation_device",
            "model_dtype",
            "inference_steps",
            "deterministic_algorithms",
            "augmentation_enabled",
            "eval_crop_mode",
            "batch_size",
        },
        "result environment receipt",
    )
    if (
        result["code_receipt"]["evaluator_version"] != EVALUATOR_VERSION
        or not re.fullmatch(r"[0-9a-f]{40}", str(result["code_receipt"]["git_commit"]))
        or not isinstance(result["code_receipt"]["verified_head_files"], list)
        or not result["code_receipt"]["verified_head_files"]
        or environment["model_dtype"] not in {"float32", "float16", "bfloat16"}
        or isinstance(environment["inference_steps"], bool)
        or not isinstance(environment["inference_steps"], int)
        or environment["inference_steps"] <= 0
        or environment["deterministic_algorithms"] is not True
        or environment["augmentation_enabled"] is not False
        or environment["eval_crop_mode"] != "center"
        or environment["batch_size"] != 1
    ):
        raise HeldoutEvaluationError("result code/environment receipt is invalid")

    contract = _require_exact_keys(
        result["evaluation_contract"],
        {
            "augmentation_enabled",
            "eval_crop_mode",
            "windows_per_val_seed",
            "windows_per_test_seed",
            "checkpoint_steps",
            "split_seed",
            "action_target_access",
            "oracle_contact_condition",
        },
        "result evaluation contract",
    )
    seed = preflight["split_seed"]
    if contract != {
        "augmentation_enabled": False,
        "eval_crop_mode": "center",
        "windows_per_val_seed": WINDOWS_PER_SPLIT_PER_SEED,
        "windows_per_test_seed": WINDOWS_PER_SPLIT_PER_SEED,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "split_seed": seed,
        "action_target_access": "metrics_only_after_prediction",
        "oracle_contact_condition": False,
    }:
        raise HeldoutEvaluationError("result evaluation contract differs")
    csv_entry = _require_exact_keys(
        result["per_window_csv"], {"path_record_only", "sha256", "row_count"},
        "result per-window receipt",
    )
    rows = read_per_window_csv(
        csv_entry["path_record_only"], expected_sha256=csv_entry["sha256"]
    )
    if csv_entry["row_count"] != len(CHECKPOINT_STEPS) * 512 or len(rows) != csv_entry["row_count"]:
        raise HeldoutEvaluationError("result per-window row count differs")
    manifest_windows = {
        (
            window["window_id"], window["split"], window["episode_id"],
            window["frame_index"], window["global_index"], window["action_noise_seed"],
        )
        for window in windows["windows"]
    }
    manifest_entries = checkpoint_entries_by_key(checkpoint_manifest)
    for step in CHECKPOINT_STEPS:
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
            or identities != manifest_windows
            or any(row["split_seed"] != seed for row in selected)
            or any(
                row["checkpoint_bundle_sha256"]
                != manifest_entries[(seed, step)]["checkpoint_bundle_sha256"]
                for row in selected
            )
        ):
            raise HeldoutEvaluationError(
                "result per-window identities/checkpoint binding differ"
            )
    recomputed = []
    for step in CHECKPOINT_STEPS:
        for heldout_split in ("val", "test"):
            selected = [
                row for row in rows
                if row["checkpoint_step"] == step and row["split"] == heldout_split
            ]
            if len(selected) != WINDOWS_PER_SPLIT_PER_SEED:
                raise HeldoutEvaluationError("result step/split window count differs")
            recomputed.append(
                {
                    "split_seed": seed,
                    "checkpoint_step": step,
                    "heldout_split": heldout_split,
                    "metrics": reduce_metric_rows(selected),
                }
            )
    if result["results_by_seed_step_split"] != recomputed:
        raise HeldoutEvaluationError(
            "reported metrics differ from the hashed additive per-window evidence"
        )
    return {
        "path": result_path,
        "result": result,
        "result_sha256": result_sha256,
        "preflight_receipt": preflight,
        "preflight_receipt_sha256": preflight_sha256,
        "window_manifest_sha256": window_manifest_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "per_window_csv_sha256": csv_entry["sha256"],
        "rows": rows,
    }


def build_code_receipt(repository_root: str | Path) -> dict:
    logical_paths = (
        "scripts/evaluate_contact_asym25_heldout.py",
        "world_action_model/contact_asym25_heldout_eval.py",
        "world_action_model/contact_asym25_eval_runtime.py",
        "world_action_model/contact_asym25_preflight.py",
        "world_action_model/production_split_contract.py",
        "world_action_model/contact_transition_schema.py",
        "world_action_model/asymmetric_action_schema.py",
        "world_action_model/datasets/wam_lerobot_dataset.py",
        "world_action_model/transforms/wa_transforms_lerobot_pretrain.py",
        "world_action_model/models/transformer_wa_casual_mot.py",
    )
    commit, rows = verified_head_code_manifest(repository_root, logical_paths)
    return {
        "git_commit": commit,
        "evaluator_version": EVALUATOR_VERSION,
        "verified_head_files": rows,
    }


def build_training_code_receipt(repository_root: str | Path) -> dict:
    commit, rows = verified_head_code_manifest(
        repository_root, TRAINING_CODE_LOGICAL_PATHS
    )
    return {"git_commit": commit, "verified_head_files": rows}


def build_environment_receipt(*, device: str, dtype: str, inference_steps: int) -> dict:
    packages = {}
    for name in ("numpy", "torch", "diffusers", "pandas"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = None
    cuda = None
    try:
        import torch

        cuda = {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception:
        cuda = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_record_only": cuda,
        "evaluation_device": device,
        "model_dtype": dtype,
        "inference_steps": inference_steps,
        "deterministic_algorithms": True,
        "augmentation_enabled": False,
        "eval_crop_mode": "center",
        "batch_size": 1,
    }


def checkpoint_entries_by_key(manifest: Mapping[str, Any]) -> dict[tuple[int, int], dict]:
    return {
        (int(entry["split_seed"]), int(entry["step"])): dict(entry)
        for entry in manifest["checkpoints"]
    }


def build_current_training_data_receipt(
    *,
    preflight_receipt: Mapping[str, Any],
    preflight_receipt_sha256: str,
    heldout_window_manifest_sha256: str,
    training_code_receipt: Mapping[str, Any],
    dataset_source_tree_receipt: Mapping[str, Any],
) -> tuple[dict, str]:
    """Bind the pre-sampling split receipt to windows, gate, and HEAD code."""

    base_receipt = preflight_receipt.get("training_data_receipt")
    base_digest = preflight_receipt.get("training_data_receipt_sha256")
    if not isinstance(base_receipt, Mapping):
        raise HeldoutEvaluationError("preflight is missing its training-data receipt")
    actual_base_digest = hashlib.sha256(_canonical_json_bytes(base_receipt)).hexdigest()
    if actual_base_digest != _require_sha256(base_digest, "training-data receipt digest"):
        raise HeldoutEvaluationError("preflight training-data receipt digest differs")
    code_receipt = dict(training_code_receipt)
    code_digest = hashlib.sha256(_canonical_json_bytes(code_receipt)).hexdigest()
    source_tree = _require_exact_keys(
        dict(dataset_source_tree_receipt),
        {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
        "dataset source-tree receipt",
    )
    _require_sha256(source_tree["tree_sha256"], "dataset source-tree digest")
    if (
        not isinstance(source_tree["path_record_only"], str)
        or not Path(source_tree["path_record_only"]).is_absolute()
        or isinstance(source_tree["file_count"], bool)
        or not isinstance(source_tree["file_count"], int)
        or source_tree["file_count"] <= 0
        or isinstance(source_tree["total_size_bytes"], bool)
        or not isinstance(source_tree["total_size_bytes"], int)
        or source_tree["total_size_bytes"] <= 0
    ):
        raise HeldoutEvaluationError("invalid dataset source-tree receipt")
    receipt = {
        "receipt_version": CURRENT_TRAINING_DATA_RECEIPT_VERSION,
        "base_training_data_receipt": dict(base_receipt),
        "base_training_data_receipt_sha256": actual_base_digest,
        "production_preflight_receipt_sha256": _require_sha256(
            preflight_receipt_sha256, "preflight receipt digest"
        ),
        "heldout_window_manifest_sha256": _require_sha256(
            heldout_window_manifest_sha256, "heldout window manifest digest"
        ),
        "training_gate_sha256": preflight_receipt["training_gate_sha256"],
        "split_seed": preflight_receipt["split_seed"],
        "train_episode_ids": list(preflight_receipt["train_episode_ids"]),
        "validation_episode_ids": list(preflight_receipt["validation_episode_ids"]),
        "heldout_episode_ids": list(preflight_receipt["heldout_episode_ids"]),
        "training_code_receipt": code_receipt,
        "training_code_receipt_sha256": code_digest,
        "dataset_source_tree_receipt": source_tree,
    }
    return receipt, hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()

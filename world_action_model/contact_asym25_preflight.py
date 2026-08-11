"""Fail-closed integration gate for asymmetric25 contact finetuning.

This module validates artifacts; it does not infer scientific readiness.  A
training-gate file records the independently reviewed decisions that cannot be
derived from tensors alone, while every artifact fingerprint is recomputed.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

import numpy as np

from scripts.contact_labels.contact_label_io_v1_1 import (
    COVERAGE_FIELDS,
    INVENTORY_FIELDS,
    SEGMENT_FIELDS,
    encode_episode,
    load_coverages,
    load_inventory,
    load_segments,
    read_csv_exact,
)
from scripts.contact_labels.check_formal_contact_dual_annotation_v1_1 import (
    Segment as FormalAgreementSegment,
    agreement_report as recompute_formal_agreement,
)
from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    MODEL_ACTION_DIM,
    validate_asymmetric_norm_payload,
)
from world_action_model.asymmetric_hardware_manifest import (
    calibration_capture_sha256,
    hardware_contract_sha256,
    load_hardware_manifest,
    validate_converted_dataset_info,
    validate_manifest_capture_binding,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.contact_asym25_train_norm_contract import (
    PRODUCTION_NUMERIC_CONTRACT,
    PRODUCTION_TRAIN_RAW_STATS_VERSION,
    ProductionTrainNormError,
    build_expected_train_norm_source_receipt,
    build_norm_source_tree_receipt,
    build_production_split_receipt,
    canonical_json_sha256,
    validate_production_train_norm_metadata,
    validate_production_train_norm_payload_statistics,
)
from world_action_model.matched_control_hashes import (
    COLLECTOR_VERSION,
    HASH_SCHEMA,
    IMPLEMENTED_RUNNER_VARIANTS,
    REQUIRED_HASHES,
    canonical_sha256,
)
from world_action_model.verified_io import (
    read_verified_bytes,
    strict_json_loads,
    verified_head_code_manifest,
)


TRAINING_GATE_VERSION = "contact-asym25-training-gate/v2"
PREFLIGHT_RECEIPT_VERSION = "contact-asym25-production-preflight-receipt/v1"
TRAINING_DATA_RECEIPT_VERSION = "contact-asym25-training-data-receipt/v1"
REVIEWED_BUILDER_ID = "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py"
REQUIRED_SIDECAR_KEYS = (
    "contact_target",
    "contact_loss_mask",
    "contact_confidence",
    "contact_condition_mask",
)
REQUIRED_GATE_FLAGS = (
    "human_adjudicated_sidecar_passed",
    "dual_annotation_agreement_passed",
    "group_metadata_complete",
    "strategy_claim_allowed",
    "asymmetric25_real_data_passed",
    "hardware_contract_verified",
    "matched_control_hash_collector_passed",
    "allow_server_training",
    "allow_5b_training",
)
REQUIRED_ENV_PATHS = (
    "GWP_SONIC_DATA_PATH",
    "GWP_NORM_STATS_PATH",
    "GWP_ASYMMETRIC_HARDWARE_MANIFEST",
    "GWP_ASYMMETRIC_CALIBRATION_CAPTURE",
    "GWP_CONTACT_SIDECAR_DIR",
    "GWP_CONTACT_COVERAGE_CSV",
    "GWP_CONTACT_SEGMENTS_CSV",
    "GWP_CONTACT_INVENTORY_CSV",
    "GWP_CONTACT_BUILD_CONFIG",
    "GWP_CONTACT_SPLIT_MANIFEST",
    "GWP_CONTACT_ASYM25_TRAINING_GATE",
    "GWP_SPLIT_SEED",
    "GWP_CONTACT_TRAINING_RUN_ID",
    "GWP_CONTACT_TRAINING_RUN_NONCE",
    "GWP_MATCHED_CONTROL_VARIANT_ID",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}$")
_UTC_DEADLINE_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
GATE_ACCEPTANCE_VERSION = "contact-asym25-gate-reviewer-acceptance/v1"


class ContactAsym25PreflightError(ValueError):
    """Raised when any production integration requirement is missing."""


def sha256_file(path: str | Path) -> str:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_exact_mapping(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContactAsym25PreflightError(f"{name} schema mismatch")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContactAsym25PreflightError(f"{name} must be lowercase SHA-256")
    return value


def _require_run_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not _RUN_ID_RE.fullmatch(value):
        raise ContactAsym25PreflightError(
            f"{name} must be an explicit 16-128 character audit identity"
        )
    return value


def _validate_gate_expiry(value: object) -> str:
    if not isinstance(value, str) or not _UTC_DEADLINE_RE.fullmatch(value):
        raise ContactAsym25PreflightError(
            "training gate evidence_valid_until_utc must be canonical UTC"
        )
    try:
        deadline = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContactAsym25PreflightError("invalid training gate expiry") from exc
    if deadline.tzinfo is None or deadline.utcoffset() != timezone.utc.utcoffset(None):
        raise ContactAsym25PreflightError("training gate expiry is not UTC")
    if deadline <= datetime.now(timezone.utc):
        raise ContactAsym25PreflightError("contact-asym25 training gate has expired")
    return value


def _load_json(path: str | Path, name: str) -> dict:
    payload, _ = _load_json_with_digest(path, name)
    return payload


def _load_json_with_digest(path: str | Path, name: str) -> tuple[dict, str]:
    resolved = Path(path).expanduser().absolute()
    try:
        payload_bytes, digest, _ = read_verified_bytes(resolved)
        payload = strict_json_loads(payload_bytes, name=name)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContactAsym25PreflightError(f"invalid strict JSON for {name}") from exc
    if not isinstance(payload, dict):
        raise ContactAsym25PreflightError(f"{name} must contain a JSON object")
    return payload, digest


def _validate_reviewed_meta(meta: Mapping) -> None:
    expected = {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "target_dim": SEGMENT_TARGET_DIM,
        "num_queries": 12,
        "overflow_policy": "hard_error_no_drop",
        "dropped_segments": 0,
        "horizon_frames": 48,
        "builder_identity": REVIEWED_BUILDER_ID,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ContactAsym25PreflightError(
                f"contact sidecar meta {key}={meta.get(key)!r}; expected {value!r}"
            )
    if meta.get("frame_clock") != "dataset_native_frame_index":
        raise ContactAsym25PreflightError(
            "robot contact sidecars must use dataset_native_frame_index"
        )
    target_fps = meta.get("target_fps")
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) or target_fps <= 0:
        raise ContactAsym25PreflightError("contact sidecar target_fps must be positive")
    contract = meta.get("annotation_contract")
    required_contract = {
        "annotation_role": "adjudicated",
        "review_status": "reviewed",
        "label_origin": "human_adjudicated",
        "automatic_or_proposal_ground_truth": False,
    }
    if contract != required_contract:
        raise ContactAsym25PreflightError(
            "contact sidecar lacks the reviewed/adjudicated human annotation contract"
        )
    if int(meta.get("reviewed_coverage_rows", 0)) <= 0:
        raise ContactAsym25PreflightError("contact sidecar has no reviewed coverage rows")
    if int(meta.get("unique_physical_segments", 0)) <= 0:
        raise ContactAsym25PreflightError("contact sidecar has no independent physical segments")
    expected_slices = {
        key: [value.start, value.stop] for key, value in SEGMENT_FIELD_SLICES.items()
    }
    if meta.get("field_slices") != expected_slices:
        raise ContactAsym25PreflightError("contact sidecar field_slices differ from v1.1")


def validate_contact_sidecar_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    frame_count: int | None = None,
) -> None:
    if set(arrays) != set(REQUIRED_SIDECAR_KEYS):
        raise ContactAsym25PreflightError(
            f"contact sidecar keys must be exactly {REQUIRED_SIDECAR_KEYS}"
        )
    lengths = set()
    expected_tail = (12, SEGMENT_TARGET_DIM)
    for key in REQUIRED_SIDECAR_KEYS:
        value = np.asarray(arrays[key])
        if value.ndim != 3 or value.shape[1:] != expected_tail:
            raise ContactAsym25PreflightError(
                f"{key} must have shape [T,12,{SEGMENT_TARGET_DIM}], got {value.shape}"
            )
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ContactAsym25PreflightError(f"{key} must be finite float32")
        lengths.add(int(value.shape[0]))
    if len(lengths) != 1 or (frame_count is not None and lengths != {int(frame_count)}):
        raise ContactAsym25PreflightError(
            f"contact sidecar frame counts {sorted(lengths)} do not match {frame_count}"
        )
    target = arrays["contact_target"]
    mask = arrays["contact_loss_mask"]
    confidence = arrays["contact_confidence"]
    condition = arrays["contact_condition_mask"]
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ContactAsym25PreflightError("contact_loss_mask must be binary")
    if not np.array_equal(mask, condition):
        raise ContactAsym25PreflightError("contact condition and supervision masks differ")
    if np.any((target < 0.0) | (target > 1.0)):
        raise ContactAsym25PreflightError("contact targets must lie in [0,1]")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ContactAsym25PreflightError("contact confidence must lie in [0,1]")
    if np.any(target[mask == 0] != 0.0) or np.any(confidence[mask == 0] != 0.0):
        raise ContactAsym25PreflightError("unknown contact fields must be canonical zero")
    if np.any(confidence[mask == 1] <= 0.0):
        raise ContactAsym25PreflightError("supervised contact fields need positive confidence")
    presence = SEGMENT_FIELD_SLICES["presence"]
    presence_target = target[..., presence].squeeze(-1)
    presence_mask = mask[..., presence].squeeze(-1)
    if not np.all((presence_target == 0.0) | (presence_target == 1.0)):
        raise ContactAsym25PreflightError("contact presence target must be binary")
    nonpresence_mask = mask.copy()
    nonpresence_mask[..., presence] = 0.0
    if np.any((presence_target == 0.0) & np.any(nonpresence_mask != 0.0, axis=-1)):
        raise ContactAsym25PreflightError("absent queries cannot supervise attributes")
    for field in ("side", "site", "object_role", "contact_mode"):
        field_slice = SEGMENT_FIELD_SLICES[field]
        field_mask = mask[..., field_slice]
        if not np.all(field_mask == field_mask[..., :1]):
            raise ContactAsym25PreflightError(f"{field} mask must cover the full group")
        active = field_mask[..., 0] == 1.0
        if np.any(np.abs(target[..., field_slice].sum(axis=-1)[active] - 1.0) > 1e-6):
            raise ContactAsym25PreflightError(f"supervised {field} must be one-hot")
    start = SEGMENT_FIELD_SLICES["start_tau"]
    end = SEGMENT_FIELD_SLICES["end_tau"]
    both = (mask[..., start].squeeze(-1) == 1.0) & (
        mask[..., end].squeeze(-1) == 1.0
    )
    if np.any(target[..., start].squeeze(-1)[both] > target[..., end].squeeze(-1)[both]):
        raise ContactAsym25PreflightError("contact start_tau exceeds end_tau")
    left = SEGMENT_FIELD_SLICES["left_censored"]
    right = SEGMENT_FIELD_SLICES["right_censored"]
    if not np.array_equal(mask[..., start], mask[..., left]):
        raise ContactAsym25PreflightError("start_tau and left_censored masks differ")
    if not np.array_equal(mask[..., end], mask[..., right]):
        raise ContactAsym25PreflightError("end_tau and right_censored masks differ")
    left_active = (mask[..., left].squeeze(-1) == 1.0) & (
        target[..., left].squeeze(-1) == 1.0
    )
    right_active = (mask[..., right].squeeze(-1) == 1.0) & (
        target[..., right].squeeze(-1) == 1.0
    )
    if np.any(target[..., start].squeeze(-1)[left_active] != 0.0):
        raise ContactAsym25PreflightError("left_censored requires start_tau=0")
    if np.any(target[..., end].squeeze(-1)[right_active] != 1.0):
        raise ContactAsym25PreflightError("right_censored requires end_tau=1")


def _validate_group_metadata(inventory_path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv_exact(inventory_path, INVENTORY_FIELDS)
    required = (
        "subject_id",
        "session_id",
        "operator_id",
        "source_object_instance_id",
        "target_object_instance_id",
        "liquid_id",
        "collection_batch_id",
    )
    for row_number, row in enumerate(rows, 2):
        missing = [key for key in required if not row[key]]
        if missing or row["metadata_status"] != "complete":
            raise ContactAsym25PreflightError(
                f"inventory row {row_number} has incomplete group metadata: {missing}"
            )
    return {int(row["episode_id"]): row for row in rows}


def _validate_split_manifest(
    path: Path,
    inventory: Mapping[int, object],
    inventory_rows: Mapping[int, Mapping[str, str]],
    split_policy: Mapping,
) -> dict[int, dict[str, list[int]]]:
    expected_header = [
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
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ContactAsym25PreflightError("split manifest header mismatch")
        rows = list(reader)
    group_fields_by_seed = split_policy.get("group_fields_by_seed")
    if not isinstance(group_fields_by_seed, Mapping):
        raise ContactAsym25PreflightError(
            "build config split_policy requires group_fields_by_seed"
        )
    seed_columns = ("seed_6666_split", "seed_7777_split", "seed_8888_split")
    allowed_group_fields = {
        "subject_id",
        "session_id",
        "operator_id",
        "source_object_instance_id",
        "target_object_instance_id",
        "liquid_id",
        "collection_batch_id",
    }
    for seed in seed_columns:
        fields = group_fields_by_seed.get(seed)
        if (
            not isinstance(fields, list)
            or not fields
            or any(field not in allowed_group_fields for field in fields)
        ):
            raise ContactAsym25PreflightError(f"invalid group fields for {seed}")
    seen = set()
    split_rows = {}
    for row_number, row in enumerate(rows, 2):
        episode = int(row["episode_id"])
        if episode in seen or episode not in inventory:
            raise ContactAsym25PreflightError(
                f"split row {row_number} has duplicate/unknown episode {episode}"
            )
        seen.add(episode)
        split_rows[episode] = row
        if int(row["length_frames"]) != inventory[episode].length_frames:
            raise ContactAsym25PreflightError("split/inventory episode length mismatch")
        if row["split_unit"] != "group":
            raise ContactAsym25PreflightError("episode-only diagnostic split is not production")
        if row["split_purpose"] != "production_strategy_evaluation":
            raise ContactAsym25PreflightError("split purpose is not production strategy evaluation")
        if row["ood_eligible"] != "true" or row["strategy_claim_allowed"] != "true":
            raise ContactAsym25PreflightError("split manifest is not OOD/strategy eligible")
        if row["metadata_status"] != "complete":
            raise ContactAsym25PreflightError("split manifest group metadata is incomplete")
        for seed in seed_columns:
            if row[seed] not in {"train", "val", "test"}:
                raise ContactAsym25PreflightError(f"invalid {seed} value")
    if seen != set(inventory):
        raise ContactAsym25PreflightError("split manifest does not cover every inventory episode")
    for seed in seed_columns:
        fields = group_fields_by_seed[seed]
        assignment_by_group = {}
        for episode, row in split_rows.items():
            group = tuple(inventory_rows[episode][field] for field in fields)
            split = row[seed]
            previous = assignment_by_group.setdefault(group, split)
            if previous != split:
                raise ContactAsym25PreflightError(
                    f"{seed} leaks group {fields}={group} across {previous}/{split}"
                )
        if {row[seed] for row in split_rows.values()} != {"train", "val", "test"}:
            raise ContactAsym25PreflightError(
                f"{seed} must contain non-empty train, val, and test splits"
            )
    return {
        int(seed.removeprefix("seed_").removesuffix("_split")): {
            split: sorted(
                episode
                for episode, row in split_rows.items()
                if row[seed] == split
            )
            for split in ("train", "val", "test")
        }
        for seed in seed_columns
    }


def validate_reviewed_sidecar_bundle(
    *,
    sidecar_dir: str | Path,
    coverage_csv: str | Path,
    segments_csv: str | Path,
    inventory_csv: str | Path,
    build_config_path: str | Path,
    split_manifest: str | Path,
) -> dict:
    sidecar_dir = Path(sidecar_dir).resolve()
    coverage_csv = Path(coverage_csv).resolve()
    segments_csv = Path(segments_csv).resolve()
    inventory_csv = Path(inventory_csv).resolve()
    build_config_path = Path(build_config_path).resolve()
    split_manifest = Path(split_manifest).resolve()
    for path, name in (
        (sidecar_dir, "contact sidecar directory"),
        (coverage_csv, "reviewed coverage CSV"),
        (segments_csv, "reviewed segments CSV"),
        (inventory_csv, "inventory CSV"),
        (build_config_path, "contact build config"),
        (split_manifest, "split manifest"),
    ):
        if not path.exists():
            raise ContactAsym25PreflightError(f"missing {name}: {path}")
    meta_path = sidecar_dir / "meta.json"
    meta = _load_json(meta_path, "contact sidecar meta")
    _validate_reviewed_meta(meta)
    config = _load_json(build_config_path, "contact build config")
    if not isinstance(config.get("dataset_id"), str) or not config["dataset_id"].strip():
        raise ContactAsym25PreflightError("contact build config requires a stable dataset_id")
    if config.get("sidecar_schema_version") != SEGMENT_SCHEMA_VERSION:
        raise ContactAsym25PreflightError("build config is not contact_segment_v1_1")
    if config.get("accepted_annotation_role") != "adjudicated":
        raise ContactAsym25PreflightError("build config does not require adjudication")
    if int(config.get("horizon_frames", -1)) != 48:
        raise ContactAsym25PreflightError("contact build horizon must match action horizon 48")
    if int(config.get("num_queries", -1)) != 12:
        raise ContactAsym25PreflightError("contact build num_queries must be 12")
    if config.get("frame_clock") != "dataset_native_frame_index":
        raise ContactAsym25PreflightError(
            "robot contact build clock must be dataset_native_frame_index"
        )
    target_fps = config.get("target_fps")
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) or target_fps <= 0:
        raise ContactAsym25PreflightError("contact build target_fps must be positive")
    if (
        meta.get("dataset_id") != config.get("dataset_id")
        or meta.get("frame_clock") != config.get("frame_clock")
        or float(meta.get("target_fps", -1.0)) != float(config.get("target_fps", -2.0))
        or int(meta.get("horizon_frames", -1)) != int(config.get("horizon_frames", -2))
        or int(meta.get("num_queries", -1)) != int(config.get("num_queries", -2))
    ):
        raise ContactAsym25PreflightError("sidecar meta and build config timing/schema differ")
    split_policy = config.get("split_policy")
    if not isinstance(split_policy, Mapping) or not split_policy.get(
        "strategy_claim_allowed", False
    ):
        raise ContactAsym25PreflightError("build config strategy_claim_allowed is false")
    inventory = load_inventory(inventory_csv)
    inventory_rows = _validate_group_metadata(inventory_csv)
    split_episode_ids = _validate_split_manifest(
        split_manifest, inventory, inventory_rows, split_policy
    )
    coverages = load_coverages(coverage_csv, inventory, config)
    segments = load_segments(segments_csv, inventory, coverages, config)
    if len(segments) != int(meta["unique_physical_segments"]):
        raise ContactAsym25PreflightError("independent segment count differs from sidecar meta")
    input_paths = {
        "coverage_csv": coverage_csv,
        "segments_csv": segments_csv,
        "inventory_csv": inventory_csv,
        "config": build_config_path,
        "split_manifest": split_manifest,
    }
    input_hashes = meta.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise ContactAsym25PreflightError("sidecar meta is missing input_sha256")
    for key, path in input_paths.items():
        if input_hashes.get(key) != sha256_file(path):
            raise ContactAsym25PreflightError(f"sidecar source fingerprint differs: {key}")

    coverage_by_episode = {episode: [] for episode in inventory}
    segment_by_episode = {episode: [] for episode in inventory}
    for coverage in coverages:
        coverage_by_episode[coverage.episode_id].append(coverage)
    for segment in segments:
        segment_by_episode[segment.episode_id].append(segment)
    expected_files = {f"episode_{episode:06d}.npz" for episode in inventory}
    actual_files = {path.name for path in sidecar_dir.glob("episode_*.npz")}
    if actual_files != expected_files:
        raise ContactAsym25PreflightError("sidecar episode file set differs from inventory")
    episode_hashes = {
        name: sha256_file(sidecar_dir / name) for name in sorted(expected_files)
    }
    if meta.get("episode_npz_sha256") != episode_hashes:
        raise ContactAsym25PreflightError("sidecar episode NPZ fingerprints differ from meta")
    expected_provenance = []
    for episode, item in sorted(inventory.items()):
        expected_arrays, provenance = encode_episode(
            item.length_frames,
            coverage_by_episode[episode],
            segment_by_episode[episode],
            horizon_frames=int(config["horizon_frames"]),
            num_queries=12,
            negative_confidence=float(config["negative_confidence"]),
        )
        sidecar_path = sidecar_dir / f"episode_{episode:06d}.npz"
        with np.load(sidecar_path, allow_pickle=False) as archive:
            actual_arrays = {key: archive[key] for key in archive.files}
        validate_contact_sidecar_arrays(actual_arrays, frame_count=item.length_frames)
        for key in REQUIRED_SIDECAR_KEYS:
            if not np.array_equal(actual_arrays[key], expected_arrays[key]):
                raise ContactAsym25PreflightError(
                    f"{sidecar_path.name}:{key} differs from reviewed source re-encoding"
                )
        expected_provenance.extend(
            {"episode_id": str(episode), **{k: str(v) for k, v in row.items()}}
            for row in provenance
        )
    provenance_path = sidecar_dir / "window_segment_provenance.csv"
    with provenance_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "episode_id",
            "frame_index",
            "query_index",
            "segment_id",
            "onset_event_id",
            "release_event_id",
        ]:
            raise ContactAsym25PreflightError("sidecar provenance header mismatch")
        actual_provenance = list(reader)
    if actual_provenance != expected_provenance:
        raise ContactAsym25PreflightError("sidecar provenance differs from source re-encoding")
    provenance_digest = sha256_file(provenance_path)
    if meta.get("window_segment_provenance_sha256") != provenance_digest:
        raise ContactAsym25PreflightError("sidecar provenance fingerprint differs from meta")
    bundle_digest = hashlib.sha256(
        json.dumps(
            {
                "episode_npz_sha256": episode_hashes,
                "window_segment_provenance_sha256": provenance_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if meta.get("sidecar_bundle_sha256") != bundle_digest:
        raise ContactAsym25PreflightError("sidecar bundle fingerprint differs from meta")
    return {
        "meta_sha256": sha256_file(meta_path),
        "split_manifest_sha256": sha256_file(split_manifest),
        "sidecar_bundle_sha256": bundle_digest,
        "episode_npz_sha256": episode_hashes,
        "independent_segment_count": len(segments),
        "episode_count": len(inventory),
        "episode_lengths": {
            int(episode): int(item.length_frames) for episode, item in inventory.items()
        },
        "split_episode_ids": split_episode_ids,
        "dataset_id": config["dataset_id"],
        "meta": meta,
    }


def _formal_segment_rows(rows: list[dict[str, str]]) -> list[FormalAgreementSegment]:
    result = []
    for row in rows:
        try:
            result.append(
                FormalAgreementSegment(
                    episode=int(row["episode_id"]),
                    coverage_id=row["coverage_id"],
                    segment_id=row["segment_id"],
                    start=int(row["start_frame"]),
                    end=int(row["end_frame"]),
                    start_boundary=row["start_boundary"],
                    end_boundary=row["end_boundary"],
                    side=row["side"],
                    site=row["site"],
                    object_role=row["object_role"],
                    contact_mode=row["contact_mode"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContactAsym25PreflightError(
                "formal agreement input contains an invalid segment row"
            ) from exc
    return result


def _recompute_bound_annotation_agreement(
    source_rows: Mapping[str, list[dict[str, str]]],
) -> tuple[dict[str, float], dict]:
    """Re-run the formal A/B reducer from the six gate-bound CSV files."""

    coverage_a = list(source_rows["coverage_annotator_a"])
    coverage_b = list(source_rows["coverage_annotator_b"])
    segments_a_rows = list(source_rows["segments_annotator_a"])
    segments_b_rows = list(source_rows["segments_annotator_b"])
    adjudicated_coverage = list(source_rows["coverage_adjudicated"])
    adjudicated_segments = list(source_rows["segments_adjudicated"])
    by_episode_a = {int(row["episode_id"]): row for row in coverage_a}
    by_episode_b = {int(row["episode_id"]): row for row in coverage_b}
    if (
        len(by_episode_a) != len(coverage_a)
        or len(by_episode_b) != len(coverage_b)
        or set(by_episode_a) != set(by_episode_b)
        or not by_episode_a
    ):
        raise ContactAsym25PreflightError(
            "formal agreement requires one A and one B coverage row per episode"
        )
    tasks = []
    for episode in sorted(by_episode_a):
        row_a = by_episode_a[episode]
        row_b = by_episode_b[episode]
        if (
            row_a["start_frame"] != row_b["start_frame"]
            or row_a["end_frame"] != row_b["end_frame"]
            or int(row_a["start_frame"]) != 0
            or int(row_a["end_frame"]) <= 0
        ):
            raise ContactAsym25PreflightError(
                "formal A/B coverage windows differ"
            )
        tasks.append(
            {
                "episode_id": str(episode),
                "coverage_start_frame": row_a["start_frame"],
                "coverage_end_frame": row_a["end_frame"],
            }
        )
    packets = [
        {"coverages": coverage_a, "segments": _formal_segment_rows(segments_a_rows)},
        {"coverages": coverage_b, "segments": _formal_segment_rows(segments_b_rows)},
    ]
    try:
        formal = recompute_formal_agreement(tasks, packets)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContactAsym25PreflightError(
            "formal A/B agreement reducer rejected the bound CSVs"
        ) from exc
    if (
        formal.get("report_schema_version")
        != "contact_dual_agreement_report_v1_1"
        or formal.get("all_gates_pass") is not True
    ):
        raise ContactAsym25PreflightError(
            "recomputed formal A/B agreement gates did not pass"
        )
    categorical = formal.get("categorical_agreement")
    if not isinstance(categorical, Mapping) or set(categorical) != {
        "side",
        "site",
        "object_role",
        "contact_mode",
    }:
        raise ContactAsym25PreflightError(
            "formal categorical agreement schema differs"
        )
    categorical_values = [categorical[name].get("kappa") for name in sorted(categorical)]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in categorical_values
    ):
        raise ContactAsym25PreflightError(
            "formal categorical agreement lacks finite kappa values"
        )
    metrics = {
        "coverage_kappa": formal["coverage_weighted_kappa_linear"],
        "event_f1": formal["event_f1"],
        "median_temporal_iou": formal["matched_median_tiou"],
        "boundary_mae_frames": formal["observed_boundary_median_mae_frames"],
        "categorical_kappa": min(float(value) for value in categorical_values),
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ContactAsym25PreflightError(
            "formal agreement reducer produced non-finite metrics"
        )

    parent_coverages: dict[int, set[str]] = {}
    parent_segments: dict[str, tuple[str, int]] = {}
    annotators = set()
    for role, coverages, segments in (
        ("a", coverage_a, segments_a_rows),
        ("b", coverage_b, segments_b_rows),
    ):
        annotators.update(row["annotator_id"] for row in coverages + segments)
        for row in coverages:
            parent_coverages.setdefault(int(row["episode_id"]), set()).add(
                row["coverage_id"]
            )
        for row in segments:
            parent_segments[row["segment_id"]] = (role, int(row["episode_id"]))
    adjudicators = {
        row["annotator_id"] for row in adjudicated_coverage + adjudicated_segments
    }
    if len(adjudicators) != 1 or annotators & adjudicators:
        raise ContactAsym25PreflightError(
            "adjudicator identity must be one third party distinct from A/B"
        )
    for row in adjudicated_coverage:
        episode = int(row["episode_id"])
        parents = {value for value in row["parent_coverage_ids"].split(";") if value}
        if parents != parent_coverages.get(episode):
            raise ContactAsym25PreflightError(
                "adjudicated coverage does not bind exact same-episode A/B parents"
            )
    for row in adjudicated_segments:
        episode = int(row["episode_id"])
        parents = {value for value in row["parent_segment_ids"].split(";") if value}
        if (
            not parents
            or not parents <= set(parent_segments)
            or any(parent_segments[parent][1] != episode for parent in parents)
        ):
            raise ContactAsym25PreflightError(
                "adjudicated segment contains invalid A/B parents"
            )
        roles = {parent_segments[parent][0] for parent in parents}
        if roles != {"a", "b"} and "unilateral_parent_reason=" not in row["notes"]:
            raise ContactAsym25PreflightError(
                "unilateral adjudicated segment lacks a recorded reason"
            )
    return metrics, formal


def _validate_matched_control_expectations(gate: Mapping) -> dict:
    raw = _require_exact_mapping(
        gate.get("matched_control_expectations"),
        {
            "expected_git_commit",
            "expected_run_id_by_variant",
            "expected_run_nonce_by_variant",
            "expected_report_sha256",
            "expected_trace_files",
            "expected_trace_manifest_sha256",
            "expected_global_hashes_sha256",
            "expected_hash_schema_sha256",
            "expected_control_config_sha256",
            "expected_variant_config_sha256",
            "expected_shared_model_config_sha256",
            "expected_world_size",
            "expected_rank_set",
            "expected_collector_version",
            "expected_collector_source_sha256",
            "expected_run_contract_sha256",
            "expected_external_acceptance_sha256_by_variant",
        },
        "contact matched-control expectations",
    )
    commit = raw["expected_git_commit"]
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ContactAsym25PreflightError("invalid expected matched-control Git commit")
    variants = set(IMPLEMENTED_RUNNER_VARIANTS)
    run_ids = raw["expected_run_id_by_variant"]
    run_nonces = raw["expected_run_nonce_by_variant"]
    variant_hashes = raw["expected_variant_config_sha256"]
    acceptances = raw["expected_external_acceptance_sha256_by_variant"]
    for name, mapping in (
        ("run IDs", run_ids),
        ("run nonces", run_nonces),
        ("variant config hashes", variant_hashes),
        ("external acceptances", acceptances),
    ):
        if not isinstance(mapping, dict) or set(mapping) != variants:
            raise ContactAsym25PreflightError(
                f"matched-control expected {name} variant set differs"
            )
    if any(not _RUN_ID_RE.fullmatch(str(value)) for value in run_ids.values()):
        raise ContactAsym25PreflightError("invalid matched-control run ID mapping")
    if any(not _RUN_ID_RE.fullmatch(str(value)) for value in run_nonces.values()):
        raise ContactAsym25PreflightError("invalid matched-control run nonce mapping")
    reviewed_run_identities = set(run_ids.values()) | set(run_nonces.values())
    if (
        len(set(run_ids.values())) != len(variants)
        or len(set(run_nonces.values())) != len(variants)
        or len(reviewed_run_identities) != 2 * len(variants)
    ):
        raise ContactAsym25PreflightError(
            "matched variants require globally distinct run IDs and run nonces"
        )
    for name, value in (
        ("report", raw["expected_report_sha256"]),
        ("trace manifest", raw["expected_trace_manifest_sha256"]),
        ("global hashes", raw["expected_global_hashes_sha256"]),
        ("hash schema", raw["expected_hash_schema_sha256"]),
        ("control config", raw["expected_control_config_sha256"]),
        ("shared model config", raw["expected_shared_model_config_sha256"]),
        ("collector source", raw["expected_collector_source_sha256"]),
        ("run contract", raw["expected_run_contract_sha256"]),
    ):
        _require_sha256(value, f"expected matched-control {name}")
    for value in list(variant_hashes.values()) + list(acceptances.values()):
        _require_sha256(value, "expected matched-control variant binding")
    trace_files = raw["expected_trace_files"]
    if not isinstance(trace_files, dict) or set(trace_files) != variants:
        raise ContactAsym25PreflightError("expected matched-control trace variants differ")
    canonical_trace_files = {}
    seen_paths = set()
    for variant in IMPLEMENTED_RUNNER_VARIANTS:
        entries = trace_files[variant]
        if not isinstance(entries, list) or not entries:
            raise ContactAsym25PreflightError("expected matched-control traces are empty")
        canonical_trace_files[variant] = []
        for entry in entries:
            entry = _require_exact_mapping(entry, {"path", "sha256"}, "trace entry")
            if (
                not isinstance(entry["path"], str)
                or not entry["path"]
                or entry["path"] in seen_paths
            ):
                raise ContactAsym25PreflightError("invalid/duplicate expected trace path")
            seen_paths.add(entry["path"])
            _require_sha256(entry["sha256"], "expected trace digest")
            canonical_trace_files[variant].append(dict(entry))
    if canonical_sha256(canonical_trace_files) != raw["expected_trace_manifest_sha256"]:
        raise ContactAsym25PreflightError("expected trace manifest digest differs")
    if canonical_sha256(HASH_SCHEMA) != raw["expected_hash_schema_sha256"]:
        raise ContactAsym25PreflightError("expected hash schema differs from installed code")
    world_size = raw["expected_world_size"]
    rank_set = raw["expected_rank_set"]
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or not isinstance(rank_set, list)
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in rank_set)
        or rank_set != list(range(world_size))
    ):
        raise ContactAsym25PreflightError("invalid expected matched-control topology")
    if raw["expected_collector_version"] != COLLECTOR_VERSION:
        raise ContactAsym25PreflightError("unexpected matched-control collector version")
    collector_path = Path(__file__).with_name("matched_control_hashes.py")
    if sha256_file(collector_path) != raw["expected_collector_source_sha256"]:
        raise ContactAsym25PreflightError("expected collector source differs from runtime")
    try:
        current_commit, code_rows = verified_head_code_manifest(
            Path(__file__).resolve().parents[1],
            ("world_action_model/matched_control_hashes.py",),
            expected_commit=commit,
        )
    except ValueError as exc:
        raise ContactAsym25PreflightError(
            "matched-control source is not the reviewer-pinned current HEAD blob"
        ) from exc
    if (
        current_commit != commit
        or len(code_rows) != 1
        or code_rows[0]["runtime_sha256"]
        != raw["expected_collector_source_sha256"]
    ):
        raise ContactAsym25PreflightError(
            "matched-control current HEAD code receipt differs"
        )
    return {
        **dict(raw),
        "expected_trace_files": canonical_trace_files,
        "expected_variant_config_sha256": dict(variant_hashes),
        "expected_run_id_by_variant": dict(run_ids),
        "expected_run_nonce_by_variant": dict(run_nonces),
        "expected_external_acceptance_sha256_by_variant": dict(acceptances),
        "expected_rank_set": list(rank_set),
    }


def _reviewed_gate_contract_sha256(gate: Mapping) -> str:
    contract = dict(gate)
    reports = dict(contract["evidence_reports"])
    reports.pop("gate_reviewer_acceptance", None)
    contract["evidence_reports"] = reports
    return _canonical_json_sha256(contract)


def _validate_training_gate(
    gate: Mapping,
    expected: Mapping[str, object],
    *,
    gate_path: Path,
    dataset_id: str,
    split_seed: int,
    training_run_id: str | None,
    training_run_nonce: str | None,
    matched_control_variant_id: str | None,
) -> tuple[dict[str, str], dict, dict, dict, str | None]:
    _require_exact_mapping(
        gate,
        {
            "gate_version",
            "evidence_valid_until_utc",
            *REQUIRED_GATE_FLAGS,
            "independent_segment_count",
            "artifact_fingerprints",
            "evidence_reports",
            "matched_control_expectations",
            "reviewed_by",
            "reviewed_at_utc",
            "note",
        },
        "contact-asym25 training gate",
    )
    if gate.get("gate_version") != TRAINING_GATE_VERSION:
        raise ContactAsym25PreflightError("training gate version mismatch")
    expiry = _validate_gate_expiry(gate.get("evidence_valid_until_utc"))
    failed = [name for name in REQUIRED_GATE_FLAGS if gate.get(name) is not True]
    if failed:
        raise ContactAsym25PreflightError(
            f"training gate flags are not explicitly true: {failed}"
        )
    count = gate.get("independent_segment_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ContactAsym25PreflightError("training gate independent_segment_count must be positive")
    if count != expected["independent_segment_count"]:
        raise ContactAsym25PreflightError("training gate independent segment count differs")
    fingerprints = gate.get("artifact_fingerprints")
    expected_fingerprint_keys = set(expected) - {"independent_segment_count"}
    if not isinstance(fingerprints, dict) or set(fingerprints) != expected_fingerprint_keys:
        raise ContactAsym25PreflightError("training gate artifact_fingerprints schema differs")
    for key, value in expected.items():
        if key == "independent_segment_count":
            continue
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ContactAsym25PreflightError(f"internal invalid expected fingerprint {key}")
        if fingerprints.get(key) != value:
            raise ContactAsym25PreflightError(f"training gate fingerprint differs: {key}")
    reviewer = gate.get("reviewed_by")
    if not isinstance(reviewer, str) or not _RUN_ID_RE.fullmatch(reviewer):
        raise ContactAsym25PreflightError("training gate reviewed_by is required")
    if (
        not isinstance(gate.get("reviewed_at_utc"), str)
        or not _UTC_DEADLINE_RE.fullmatch(gate["reviewed_at_utc"])
    ):
        raise ContactAsym25PreflightError("training gate reviewed_at_utc must be UTC")
    matched_expected = _validate_matched_control_expectations(gate)
    selected_variant = None
    runtime_identity_values = (
        training_run_id,
        training_run_nonce,
        matched_control_variant_id,
    )
    if any(value is not None for value in runtime_identity_values):
        if any(value is None for value in runtime_identity_values):
            raise ContactAsym25PreflightError(
                "runtime matched-control variant, run ID, and nonce are one complete set"
            )
        runtime_run_id = _require_run_identity(
            training_run_id, "runtime training_run_id"
        )
        runtime_run_nonce = _require_run_identity(
            training_run_nonce, "runtime training_run_nonce"
        )
        matching_variants = [
            variant
            for variant in IMPLEMENTED_RUNNER_VARIANTS
            if matched_expected["expected_run_id_by_variant"][variant]
            == runtime_run_id
            and matched_expected["expected_run_nonce_by_variant"][variant]
            == runtime_run_nonce
        ]
        if len(matching_variants) != 1:
            raise ContactAsym25PreflightError(
                "training run identity does not select exactly one matched transport variant"
            )
        selected_variant = matching_variants[0]
        if matched_control_variant_id != selected_variant:
            raise ContactAsym25PreflightError(
                "runtime matched-control variant differs from the reviewer gate"
            )
    entries = gate.get("evidence_reports")
    if not isinstance(entries, dict) or set(entries) != {
        "annotation_agreement",
        "hardware_verification",
        "matched_control_hash_collector",
        "gate_reviewer_acceptance",
    }:
        raise ContactAsym25PreflightError("training gate evidence_reports schema differs")
    reports = {}
    report_paths = {}
    digests = {}
    for name in (
        "annotation_agreement",
        "hardware_verification",
        "matched_control_hash_collector",
        "gate_reviewer_acceptance",
    ):
        entry = entries.get(name)
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ContactAsym25PreflightError(f"missing evidence report entry: {name}")
        report_path_value = entry.get("path")
        declared_digest = entry.get("sha256")
        if not isinstance(report_path_value, str) or not report_path_value.strip():
            raise ContactAsym25PreflightError(f"evidence report path is required: {name}")
        report_path = Path(report_path_value)
        if not report_path.is_absolute():
            report_path = gate_path.parent / report_path
        report_path = report_path.absolute()
        report_payload, actual_digest = _load_json_with_digest(
            report_path, f"{name} evidence report"
        )
        if declared_digest != actual_digest:
            raise ContactAsym25PreflightError(f"evidence report fingerprint differs: {name}")
        reports[name] = report_payload
        report_paths[name] = report_path
        digests[name] = actual_digest

    annotation = reports["annotation_agreement"]
    metrics = annotation.get("metrics")
    if (
        not isinstance(annotation, dict)
        or set(annotation)
        != {
            "report_version",
            "status",
            "independent_annotation",
            "adjudication_completed",
            "dual_annotation_agreement_passed",
            "independent_segment_count",
            "metrics",
            "source_files",
        }
        or annotation.get("report_version") != "contact-annotation-agreement/v1"
        or annotation.get("status") != "passed"
        or annotation.get("independent_annotation") is not True
        or annotation.get("adjudication_completed") is not True
        or annotation.get("dual_annotation_agreement_passed") is not True
        or annotation.get("independent_segment_count")
        != expected["independent_segment_count"]
        or not isinstance(metrics, dict)
        or set(metrics)
        != {
            "coverage_kappa",
            "event_f1",
            "median_temporal_iou",
            "boundary_mae_frames",
            "categorical_kappa",
        }
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (metrics or {}).values()
        )
        or isinstance(annotation.get("independent_segment_count"), bool)
        or not isinstance(annotation.get("independent_segment_count"), int)
    ):
        raise ContactAsym25PreflightError("annotation agreement evidence did not pass")
    annotation_sources = annotation.get("source_files")
    required_annotation_sources = {
        "coverage_annotator_a",
        "segments_annotator_a",
        "coverage_annotator_b",
        "segments_annotator_b",
        "coverage_adjudicated",
        "segments_adjudicated",
    }
    if not isinstance(annotation_sources, Mapping) or set(annotation_sources) != required_annotation_sources:
        raise ContactAsym25PreflightError("annotation report source_files are incomplete")
    annotation_source_digests = {}
    annotation_source_rows = {}
    for name, entry in annotation_sources.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
        ):
            raise ContactAsym25PreflightError(f"invalid annotation source entry: {name}")
        source_path = Path(entry["path"])
        if not source_path.is_absolute():
            source_path = report_paths["annotation_agreement"].parent / source_path
        source_path = source_path.absolute()
        try:
            source_payload, actual, _ = read_verified_bytes(source_path)
            source_fields = COVERAGE_FIELDS if name.startswith("coverage_") else SEGMENT_FIELDS
            source_rows = read_csv_exact(
                source_path, source_fields, payload=source_payload
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise ContactAsym25PreflightError(
                f"cannot read verified annotation source: {name}"
            ) from exc
        if entry.get("sha256") != actual:
            raise ContactAsym25PreflightError(f"annotation source fingerprint differs: {name}")
        annotation_source_digests[name] = actual
        annotation_source_rows[name] = source_rows
    if (
        annotation_source_digests["coverage_adjudicated"]
        != expected["coverage_csv_sha256"]
        or annotation_source_digests["segments_adjudicated"]
        != expected["segments_csv_sha256"]
    ):
        raise ContactAsym25PreflightError("annotation report is bound to different adjudicated CSVs")
    if (
        annotation_source_digests["coverage_annotator_a"]
        == annotation_source_digests["coverage_annotator_b"]
        or annotation_source_digests["segments_annotator_a"]
        == annotation_source_digests["segments_annotator_b"]
        or annotation_source_digests["coverage_adjudicated"]
        in {
            annotation_source_digests["coverage_annotator_a"],
            annotation_source_digests["coverage_annotator_b"],
        }
        or annotation_source_digests["segments_adjudicated"]
        in {
            annotation_source_digests["segments_annotator_a"],
            annotation_source_digests["segments_annotator_b"],
        }
    ):
        raise ContactAsym25PreflightError("A/B/adjudicated annotation files are not independent")

    annotator_sets = {}
    for suffix, expected_role in (
        ("annotator_a", "independent_a"),
        ("annotator_b", "independent_b"),
    ):
        rows = annotation_source_rows[f"coverage_{suffix}"] + annotation_source_rows[
            f"segments_{suffix}"
        ]
        if not rows:
            raise ContactAsym25PreflightError(f"{suffix} annotation evidence is empty")
        if any(
            row["annotation_role"] != expected_role
            or row["label_origin"] != "human_independent"
            or row["review_status"] != "submitted"
            or row["schema_version"] != "contact_segment_annotation_v1_1"
            or row["dataset_id"] != dataset_id
            or not row["annotator_id"]
            for row in rows
        ):
            raise ContactAsym25PreflightError(f"{suffix} annotation contract differs")
        coverage_rows = annotation_source_rows[f"coverage_{suffix}"]
        if any(row["frame_clock"] != "dataset_native_frame_index" for row in coverage_rows):
            raise ContactAsym25PreflightError(f"{suffix} frame clock differs")
        annotator_sets[suffix] = {row["annotator_id"] for row in rows}
    if annotator_sets["annotator_a"] & annotator_sets["annotator_b"]:
        raise ContactAsym25PreflightError("A/B annotator identities overlap")
    recomputed_metrics, _ = _recompute_bound_annotation_agreement(
        annotation_source_rows
    )
    if metrics != recomputed_metrics:
        raise ContactAsym25PreflightError(
            "stored annotation metrics differ from the formal A/B replay"
        )
    if (
        recomputed_metrics["coverage_kappa"] < 0.70
        or recomputed_metrics["event_f1"] < 0.80
        or recomputed_metrics["median_temporal_iou"] < 0.70
        or recomputed_metrics["boundary_mae_frames"] > 2.0
        or recomputed_metrics["categorical_kappa"] < 0.60
    ):
        raise ContactAsym25PreflightError("recomputed annotation agreement did not pass")
    hardware = reports["hardware_verification"]
    if (
        hardware.get("report_version") != "asymmetric25-hardware-verification/v1"
        or hardware.get("status") != "passed"
        or hardware.get("real_hardware_capture") is not True
        or hardware.get("hardware_contract_sha256")
        != expected["hardware_contract_sha256"]
        or hardware.get("calibration_capture_sha256")
        != expected["calibration_capture_sha256"]
        or hardware.get("dataset_info_sha256") != expected["dataset_info_sha256"]
        or hardware.get("episode_metadata_sha256")
        != expected["episode_metadata_sha256"]
        or hardware.get("dataset_id") != dataset_id
    ):
        raise ContactAsym25PreflightError("hardware verification evidence did not pass")
    collector = reports["matched_control_hash_collector"]
    collector_code = collector.get("collector_code")
    collector_code_valid = (
        isinstance(collector_code, dict)
        and set(collector_code) == {"path", "version", "sha256"}
        and isinstance(collector_code.get("path"), str)
        and bool(collector_code["path"])
    )
    if collector_code_valid:
        collector_path = Path(collector_code["path"])
        if not collector_path.is_absolute():
            collector_path = (
                report_paths["matched_control_hash_collector"].parent / collector_path
            )
        try:
            _, collector_runtime_digest, _ = read_verified_bytes(
                collector_path.absolute()
            )
        except (OSError, ValueError):
            collector_code_valid = False
        else:
            collector_code_valid = (
                collector_code.get("version") == COLLECTOR_VERSION
                and collector_code.get("sha256")
                == matched_expected["expected_collector_source_sha256"]
                and collector_code.get("sha256") == collector_runtime_digest
            )
    if digests["matched_control_hash_collector"] != matched_expected[
        "expected_report_sha256"
    ]:
        raise ContactAsym25PreflightError(
            "matched-control report bytes differ from the reviewer expectation"
        )
    if collector.get("trace_files") != matched_expected["expected_trace_files"]:
        raise ContactAsym25PreflightError(
            "matched-control trace manifest differs from reviewer expectations"
        )
    for entries_for_variant in matched_expected["expected_trace_files"].values():
        for trace_entry in entries_for_variant:
            trace_path = Path(trace_entry["path"])
            if not trace_path.is_absolute():
                trace_path = report_paths["matched_control_hash_collector"].parent / trace_path
            try:
                _, trace_digest, _ = read_verified_bytes(trace_path.absolute())
            except (OSError, ValueError) as exc:
                raise ContactAsym25PreflightError(
                    "cannot read verified matched-control trace"
                ) from exc
            if trace_digest != trace_entry["sha256"]:
                raise ContactAsym25PreflightError(
                    "matched-control trace bytes differ from reviewer expectations"
                )
    run_contract = collector.get("run_contract")
    run_contract = _require_exact_mapping(
        run_contract,
        {
            "initial_weights_bundle_sha256",
            "base_model_bundle_sha256",
            "training_code_receipt_sha256",
            "dataset_source_tree_sha256",
            "runtime_environment_receipt_sha256",
            "optimizer_config_sha256",
            "scheduler_config_sha256",
            "training_config_receipt",
            "training_config_sha256",
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
        },
        "matched-control stable run contract",
    )
    for field in (
        "initial_weights_bundle_sha256",
        "base_model_bundle_sha256",
        "training_code_receipt_sha256",
        "dataset_source_tree_sha256",
        "runtime_environment_receipt_sha256",
        "optimizer_config_sha256",
        "scheduler_config_sha256",
        "training_config_sha256",
        "stable_training_config_sha256",
        "control_config_sha256",
        "preflight_receipt_sha256",
        "train_window_receipt_sha256",
        "action_schema_contract_sha256",
        "dataset_info_sha256",
        "hardware_contract_sha256",
        "norm_stats_sha256",
    ):
        _require_sha256(run_contract[field], f"matched-control run contract {field}")
    training_config_receipt = run_contract["training_config_receipt"]
    if not isinstance(training_config_receipt, dict):
        raise ContactAsym25PreflightError(
            "matched-control training config receipt must be a JSON object"
        )
    try:
        training_config_sha256 = _canonical_json_sha256(training_config_receipt)
    except (TypeError, ValueError) as exc:
        raise ContactAsym25PreflightError(
            "matched-control training config receipt is not strict JSON-safe"
        ) from exc
    stable_training_config = dict(training_config_receipt)
    stable_training_config["seed"] = "__CONTACT_ASYM25_SPLIT_SEED__"
    if (
        isinstance(training_config_receipt.get("seed"), bool)
        or training_config_receipt.get("seed") != split_seed
        or training_config_sha256 != run_contract["training_config_sha256"]
        or _canonical_json_sha256(stable_training_config)
        != run_contract["stable_training_config_sha256"]
    ):
        raise ContactAsym25PreflightError(
            "matched-control training config receipt/seed normalization differs"
        )
    if (
        run_contract["precision"] != "bfloat16"
        or run_contract["action_schema_id"] != ASYMMETRIC_SCHEMA_ID
        or run_contract["contact_schema_version"] != SEGMENT_SCHEMA_VERSION
        or run_contract["contact_query_count"] != 12
        or run_contract["contact_target_dim"] != SEGMENT_TARGET_DIM
        or run_contract["contact_time_bins"] != 16
        or run_contract["dataset_id"] != dataset_id
        or run_contract["dataset_info_sha256"] != expected["dataset_info_sha256"]
        or run_contract["hardware_contract_sha256"]
        != expected["hardware_contract_sha256"]
        or run_contract["norm_stats_sha256"] != expected["norm_stats_sha256"]
        or run_contract["control_config_sha256"]
        != matched_expected["expected_control_config_sha256"]
        or _canonical_json_sha256(run_contract)
        != matched_expected["expected_run_contract_sha256"]
    ):
        raise ContactAsym25PreflightError("matched-control stable run contract differs")
    global_hashes = collector.get("global_hashes")
    valid_global_hashes = isinstance(global_hashes, dict) and bool(global_hashes)
    if valid_global_hashes:
        for logical_name, variant_rows in global_hashes.items():
            if (
                not isinstance(logical_name, str)
                or not isinstance(variant_rows, dict)
                or set(variant_rows) != set(IMPLEMENTED_RUNNER_VARIANTS)
            ):
                valid_global_hashes = False
                break
            for hashes in variant_rows.values():
                if (
                    not isinstance(hashes, dict)
                    or set(hashes) != set(REQUIRED_HASHES)
                    or any(
                        not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
                        for value in hashes.values()
                    )
                ):
                    valid_global_hashes = False
                    break
    matched_batch_count = collector.get("matched_batch_count")
    for field in (
        "training_data_receipt_sha256",
        "heldout_window_manifest_sha256",
    ):
        _require_sha256(
            collector.get(field), f"matched-control collector {field}"
        )
    if (
        collector.get("report_version") != COLLECTOR_VERSION
        or collector.get("status") != "passed"
        or collector.get("collector_implemented") is not True
        or set(collector.get("captured_hashes", [])) != set(REQUIRED_HASHES)
        or collector.get("dry_run_passed") is not True
        or collector.get("all_hashes_equal") is not True
        or collector.get("all_hashes_equal_across_variants") is not True
        or collector.get("seeds") != [split_seed]
        or collector.get("run_id") is not None
        or collector.get("run_nonce") is not None
        or collector.get("run_id_by_variant")
        != matched_expected["expected_run_id_by_variant"]
        or collector.get("run_nonce_by_variant")
        != matched_expected["expected_run_nonce_by_variant"]
        or collector.get("dataset_id") != dataset_id
        or collector.get("dataset_ids") != [dataset_id]
        or collector.get("variants") != sorted(IMPLEMENTED_RUNNER_VARIANTS)
        or collector.get("git_commit") != matched_expected["expected_git_commit"]
        or collector.get("control_config_sha256")
        != matched_expected["expected_control_config_sha256"]
        or collector.get("variant_config_sha256")
        != matched_expected["expected_variant_config_sha256"]
        or collector.get("shared_model_config_sha256")
        != matched_expected["expected_shared_model_config_sha256"]
        or collector.get("transport_override_by_variant")
        != {
            "soft_semantic": {
                "variant_id": "soft_semantic",
                "contact_semantic_transport_mode": "soft",
            },
            "hard_st": {
                "variant_id": "hard_st",
                "contact_semantic_transport_mode": "hard_st",
            },
            "hard_detached": {
                "variant_id": "hard_detached",
                "contact_semantic_transport_mode": "hard_detached",
            },
        }
        or collector.get("world_size") != matched_expected["expected_world_size"]
        or collector.get("rank_set") != matched_expected["expected_rank_set"]
        or collector.get("topology")
        != {
            "world_size": matched_expected["expected_world_size"],
            "rank_set": matched_expected["expected_rank_set"],
        }
        or collector.get("trace_manifest_sha256")
        != matched_expected["expected_trace_manifest_sha256"]
        or collector.get("global_hashes_sha256")
        != matched_expected["expected_global_hashes_sha256"]
        or collector.get("hash_schema_sha256")
        != matched_expected["expected_hash_schema_sha256"]
        or collector.get("hash_schema") != HASH_SCHEMA
        or collector.get("run_contract_sha256")
        != matched_expected["expected_run_contract_sha256"]
        or collector.get("external_acceptance_sha256_by_variant")
        != matched_expected["expected_external_acceptance_sha256_by_variant"]
        or collector.get("production_split_seed") != split_seed
        or collector.get("production_preflight_receipt_sha256")
        != run_contract["preflight_receipt_sha256"]
        or not valid_global_hashes
        or canonical_sha256(global_hashes)
        != matched_expected["expected_global_hashes_sha256"]
        or isinstance(matched_batch_count, bool)
        or not isinstance(matched_batch_count, int)
        or matched_batch_count <= 0
        or matched_batch_count != len(global_hashes or {})
        or not collector_code_valid
    ):
        raise ContactAsym25PreflightError("matched-control hash collector evidence did not pass")
    acceptance = _require_exact_mapping(
        reports["gate_reviewer_acceptance"],
        {
            "acceptance_version",
            "status",
            "accepted",
            "reviewer_identity",
            "acceptance_run_id",
            "acceptance_run_nonce",
            "evidence_valid_until_utc",
            "reviewed_gate_contract_sha256",
            "matched_control_expectations_sha256",
            "matched_control_report_sha256",
            "matched_control_run_contract_sha256",
            "reviewed_run_contract",
            "dataset_id",
            "dataset_info_sha256",
            "hardware_contract_sha256",
            "norm_stats_sha256",
            "action_schema_id",
            "contact_schema_version",
            "accepted_precision",
        },
        "contact-asym25 gate reviewer acceptance",
    )
    for field in ("acceptance_run_id", "acceptance_run_nonce"):
        _require_run_identity(acceptance[field], f"gate acceptance {field}")
    reviewed_training_identities = {
        *matched_expected["expected_run_id_by_variant"].values(),
        *matched_expected["expected_run_nonce_by_variant"].values(),
    }
    if (
        acceptance["acceptance_run_id"] == acceptance["acceptance_run_nonce"]
        or
        acceptance["acceptance_run_id"] in reviewed_training_identities
        or acceptance["acceptance_run_nonce"] in reviewed_training_identities
    ):
        raise ContactAsym25PreflightError(
            "gate acceptance identity must be independent from all reviewed runs"
        )
    if (
        acceptance["acceptance_version"] != GATE_ACCEPTANCE_VERSION
        or acceptance["status"] != "accepted"
        or acceptance["accepted"] is not True
        or acceptance["reviewer_identity"] != reviewer
        or acceptance["evidence_valid_until_utc"] != expiry
        or acceptance["reviewed_gate_contract_sha256"]
        != _reviewed_gate_contract_sha256(gate)
        or acceptance["matched_control_expectations_sha256"]
        != canonical_sha256(matched_expected)
        or acceptance["matched_control_report_sha256"]
        != digests["matched_control_hash_collector"]
        or acceptance["matched_control_run_contract_sha256"]
        != matched_expected["expected_run_contract_sha256"]
        or acceptance["reviewed_run_contract"] != run_contract
        or _canonical_json_sha256(acceptance["reviewed_run_contract"])
        != acceptance["matched_control_run_contract_sha256"]
        or acceptance["dataset_id"] != dataset_id
        or acceptance["dataset_info_sha256"] != expected["dataset_info_sha256"]
        or acceptance["hardware_contract_sha256"]
        != expected["hardware_contract_sha256"]
        or acceptance["norm_stats_sha256"] != expected["norm_stats_sha256"]
        or acceptance["action_schema_id"] != ASYMMETRIC_SCHEMA_ID
        or acceptance["contact_schema_version"] != SEGMENT_SCHEMA_VERSION
        or acceptance["accepted_precision"] != "bfloat16"
    ):
        raise ContactAsym25PreflightError("external gate reviewer acceptance differs")
    return digests, matched_expected, acceptance, run_contract, selected_variant


def _dataset_episode_lengths(data_root: Path) -> tuple[dict[int, int], str]:
    episodes_path = data_root / "meta" / "episodes.jsonl"
    rows = []
    metadata_files = []
    metadata_digests = []
    if episodes_path.is_file():
        metadata_files = [episodes_path]
        payload, digest, _ = read_verified_bytes(episodes_path)
        metadata_digests = [digest]
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = strict_json_loads(
                    line, name=f"dataset episodes.jsonl:{line_number}"
                )
            except ValueError as exc:
                raise ContactAsym25PreflightError(
                    f"invalid strict dataset episode metadata row {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ContactAsym25PreflightError(
                    f"dataset episode metadata row {line_number} must be an object"
                )
            rows.append(row)
    else:
        episodes_dir = data_root / "meta" / "episodes"
        parquet_paths = sorted(episodes_dir.rglob("*.parquet"))
        if parquet_paths:
            import pyarrow as pa
            import pyarrow.parquet as pq

            metadata_files = parquet_paths
            for path in parquet_paths:
                payload, digest, _ = read_verified_bytes(path)
                metadata_digests.append(digest)
                try:
                    table = pq.read_table(
                        pa.BufferReader(payload),
                        columns=["episode_index", "length"],
                    )
                except Exception as exc:
                    raise ContactAsym25PreflightError(
                        f"invalid verified episode metadata parquet: {path}"
                    ) from exc
                rows.extend(
                    {"episode_index": episode, "length": length}
                    for episode, length in zip(
                        table["episode_index"].to_pylist(),
                        table["length"].to_pylist(),
                        strict=True,
                    )
                )
    if not rows:
        raise ContactAsym25PreflightError(
            f"asymmetric25 dataset requires authoritative episode metadata under {data_root / 'meta'}"
        )
    result = {}
    for row_number, row in enumerate(rows, 1):
        episode = row.get("episode_index")
        length = row.get("length")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, (int, np.integer))
            or isinstance(length, bool)
            or not isinstance(length, (int, np.integer))
        ):
            raise ContactAsym25PreflightError(
                f"invalid dataset episode metadata row {row_number}"
            )
        episode = int(episode)
        length = int(length)
        if episode in result or episode < 0 or length <= 0:
            raise ContactAsym25PreflightError(
                f"duplicate/invalid dataset episode metadata row {row_number}"
            )
        result[episode] = length
    if not result:
        raise ContactAsym25PreflightError("asymmetric25 dataset episode metadata is empty")
    digest_payload = [
        {
            "path": str(path.relative_to(data_root)),
            "sha256": digest,
        }
        for path, digest in zip(metadata_files, metadata_digests, strict=True)
    ]
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result, digest


def _production_norm_vector(value: object, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (ASYMMETRIC_ACTION_DIM,) or not np.isfinite(vector).all():
        raise ContactAsym25PreflightError(
            f"verified production norm source {name} must be finite robot25"
        )
    padded = np.zeros(MODEL_ACTION_DIM, dtype=np.float64)
    padded[:ASYMMETRIC_ACTION_DIM] = vector
    return padded


def _production_feature_statistics(values: np.ndarray) -> tuple[dict, dict]:
    if (
        values.ndim != 2
        or values.shape[0] <= 0
        or values.shape[1] != MODEL_ACTION_DIM
        or not np.isfinite(values).all()
    ):
        raise ContactAsym25PreflightError("verified production norm values differ")
    q01, q99 = np.quantile(
        values,
        [0.01, 0.99],
        axis=0,
        method="linear",
    )

    def finite_list(array: np.ndarray) -> list[float]:
        result = [float(item) for item in np.asarray(array, dtype=np.float64)]
        if not all(np.isfinite(item) for item in result):
            raise ContactAsym25PreflightError("production norm statistic is non-finite")
        return result

    value_sum = np.sum(values, axis=0, dtype=np.float64)
    mean = value_sum / float(values.shape[0])
    centered_sum_squares = np.sum(
        np.square(values - mean, dtype=np.float64),
        axis=0,
        dtype=np.float64,
    )
    std = np.sqrt(centered_sum_squares / float(values.shape[0]))
    norm = {
        "mean": finite_list(mean),
        "std": finite_list(std),
        "min": finite_list(np.min(values, axis=0)),
        "max": finite_list(np.max(values, axis=0)),
        "q01": finite_list(q01),
        "q99": finite_list(q99),
    }
    raw = {
        "count": int(values.shape[0]),
        "dimension": MODEL_ACTION_DIM,
        "sum": finite_list(value_sum),
        "sum_squares": finite_list(
            np.sum(np.square(values, dtype=np.float64), axis=0, dtype=np.float64)
        ),
        "centered_sum_squares": finite_list(centered_sum_squares),
        "min": norm["min"],
        "max": norm["max"],
    }
    return norm, raw


def _recompute_production_train_norm_from_verified_raw(
    *,
    data_root: Path,
    episode_lengths: Mapping[int, int],
    train_norm_source_receipt: Mapping,
    train_norm_source_receipt_sha256: str,
) -> tuple[dict, dict, str]:
    """Independently recompute exact train-only H48 stats from verified bytes."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ContactAsym25PreflightError(
            "production norm verification requires pyarrow"
        ) from exc

    train_ids = list(train_norm_source_receipt["train_episode_ids"])
    train_set = set(train_ids)
    source_tree = train_norm_source_receipt["source_dataset_tree_receipt"]
    data_rows = [
        dict(row)
        for row in source_tree["files"]
        if row["logical_path"].startswith("data/")
        and row["logical_path"].endswith(".parquet")
    ]
    if not data_rows:
        raise ContactAsym25PreflightError("production norm source has no data parquet")
    source_data_files_sha256 = canonical_json_sha256(data_rows)
    offsets = {}
    cursor = 0
    for episode, length in sorted(episode_lengths.items()):
        offsets[int(episode)] = cursor
        cursor += int(length)
    seen_frames = {int(episode): set() for episode in episode_lengths}
    selected: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {
        episode: [] for episode in train_ids
    }
    columns = (
        "index",
        "episode_index",
        "frame_index",
        "observation.state",
        "action",
    )
    for row in data_rows:
        logical_path = row["logical_path"]
        payload, digest, identity = read_verified_bytes(
            data_root / logical_path,
            expected_sha256=row["sha256"],
        )
        if digest != row["sha256"] or identity["size_bytes"] != row["size_bytes"]:
            raise ContactAsym25PreflightError(
                f"verified norm source receipt differs: {logical_path}"
            )
        try:
            table = pq.read_table(pa.BufferReader(payload), columns=list(columns))
        except Exception as exc:
            raise ContactAsym25PreflightError(
                f"cannot read verified norm parquet bytes: {logical_path}"
            ) from exc
        if tuple(table.column_names) != columns:
            raise ContactAsym25PreflightError(
                f"verified norm parquet columns differ: {logical_path}"
            )
        values = table.to_pydict()
        for row_index in range(table.num_rows):
            episode = values["episode_index"][row_index]
            frame = values["frame_index"][row_index]
            global_index = values["index"][row_index]
            if (
                isinstance(episode, bool)
                or not isinstance(episode, int)
                or episode not in episode_lengths
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < episode_lengths[episode]
                or isinstance(global_index, bool)
                or not isinstance(global_index, int)
                or global_index != offsets[episode] + frame
                or frame in seen_frames[episode]
            ):
                raise ContactAsym25PreflightError(
                    "verified norm parquet episode/frame/index coverage differs"
                )
            seen_frames[episode].add(frame)
            if episode not in train_set:
                continue
            state = _production_norm_vector(
                values["observation.state"][row_index],
                name="observation.state",
            )
            action = _production_norm_vector(
                values["action"][row_index],
                name="action",
            )
            selected[episode].append((frame, state, action))
    for episode, length in episode_lengths.items():
        if seen_frames[episode] != set(range(length)):
            raise ContactAsym25PreflightError(
                f"verified norm parquet coverage differs for episode {episode}"
            )

    selected_raw = hashlib.sha256()
    state_chunks = []
    action_chunks = []
    episode_rows = []
    horizon_offsets = np.arange(48, dtype=np.int64)
    for episode in train_ids:
        rows = sorted(selected[episode], key=lambda item: item[0])
        length = episode_lengths[episode]
        if [row[0] for row in rows] != list(range(length)):
            raise ContactAsym25PreflightError(
                f"verified train rows differ for episode {episode}"
            )
        states = np.stack([row[1] for row in rows], axis=0)
        actions = np.stack([row[2] for row in rows], axis=0)
        for frame, state, action in rows:
            selected_raw.update(struct.pack("<qq", episode, frame))
            selected_raw.update(np.asarray(state, dtype="<f8").tobytes(order="C"))
            selected_raw.update(np.asarray(action, dtype="<f8").tobytes(order="C"))
        query = np.minimum(
            np.arange(length, dtype=np.int64)[:, None] + horizon_offsets[None, :],
            length - 1,
        )
        state_chunks.append(states)
        action_chunks.append(actions[query].reshape(length * 48, MODEL_ACTION_DIM))
        episode_rows.append({"episode_id": episode, "row_count": length})
    state_values = np.concatenate(state_chunks, axis=0)
    action_values = np.concatenate(action_chunks, axis=0)
    state_norm, state_raw = _production_feature_statistics(state_values)
    action_norm, action_raw = _production_feature_statistics(action_values)
    norm_stats = {
        "observation.state": state_norm,
        "action": action_norm,
    }
    raw_statistics = {
        "receipt_version": PRODUCTION_TRAIN_RAW_STATS_VERSION,
        "numeric_contract": PRODUCTION_NUMERIC_CONTRACT,
        "train_norm_source_receipt_sha256": train_norm_source_receipt_sha256,
        "split_seed": train_norm_source_receipt["split_seed"],
        "train_episode_ids": train_ids,
        "train_episode_ids_sha256": train_norm_source_receipt[
            "train_episode_ids_sha256"
        ],
        "source_dataset_tree_sha256": source_tree["tree_sha256"],
        "source_data_files_sha256": source_data_files_sha256,
        "selected_train_raw_sha256": selected_raw.hexdigest(),
        "selected_episode_rows": episode_rows,
        "features": {
            "observation.state": state_raw,
            "action": action_raw,
        },
    }
    if (
        state_raw["count"] != train_norm_source_receipt["num_state_samples"]
        or action_raw["count"] != train_norm_source_receipt["num_action_samples"]
    ):
        raise ContactAsym25PreflightError(
            "verified train raw counts differ from split episode lengths"
        )
    return raw_statistics, norm_stats, canonical_json_sha256(raw_statistics)


def validate_contact_asym25_training_preflight(
    *,
    data_root: str | Path,
    norm_stats_path: str | Path,
    hardware_manifest_path: str | Path,
    calibration_capture_path: str | Path,
    sidecar_dir: str | Path,
    coverage_csv: str | Path,
    segments_csv: str | Path,
    inventory_csv: str | Path,
    build_config_path: str | Path,
    split_manifest: str | Path,
    training_gate_path: str | Path,
    split_seed: int,
    training_run_id: str | None = None,
    training_run_nonce: str | None = None,
    matched_control_variant_id: str | None = None,
) -> dict:
    data_root = Path(data_root).resolve()
    norm_stats_path = Path(norm_stats_path).resolve()
    hardware_manifest_path = Path(hardware_manifest_path).resolve()
    calibration_capture_path = Path(calibration_capture_path).resolve()
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or split_seed not in {
        6666,
        7777,
        8888,
    }:
        raise ContactAsym25PreflightError(
            "production split_seed must be one of 6666, 7777, 8888"
        )
    info_path = data_root / "meta" / "info.json"
    info = _load_json(info_path, "asymmetric25 dataset info")
    manifest = load_hardware_manifest(hardware_manifest_path, mode="training")
    capture = _load_json(calibration_capture_path, "calibration capture")
    validate_manifest_capture_binding(manifest, capture)
    contract_digest = hardware_contract_sha256(manifest)
    capture_digest = calibration_capture_sha256(capture)
    embedded_manifest = info.get("asymmetric_hardware_manifest")
    if not isinstance(embedded_manifest, Mapping):
        raise ContactAsym25PreflightError("dataset lacks embedded asymmetric hardware manifest")
    validate_manifest_capture_binding(embedded_manifest, capture)
    validate_converted_dataset_info(info, embedded_manifest)
    if info.get("action_schema_id") != ASYMMETRIC_SCHEMA_ID:
        raise ContactAsym25PreflightError("dataset is not the canonical asymmetric25 schema")
    if hardware_contract_sha256(embedded_manifest) != contract_digest:
        raise ContactAsym25PreflightError("dataset and requested hardware contracts differ")
    if info.get("calibration_capture_sha256") != capture_digest:
        raise ContactAsym25PreflightError("dataset calibration capture fingerprint differs")
    dataset_episode_lengths, episode_metadata_digest = _dataset_episode_lengths(data_root)
    dataset_info_digest = sha256_file(info_path)
    total_frames = sum(dataset_episode_lengths.values())
    if int(info.get("total_episodes", -1)) != len(dataset_episode_lengths) or int(
        info.get("total_frames", -1)
    ) != total_frames:
        raise ContactAsym25PreflightError("dataset info totals differ from episode metadata")

    sidecar = validate_reviewed_sidecar_bundle(
        sidecar_dir=sidecar_dir,
        coverage_csv=coverage_csv,
        segments_csv=segments_csv,
        inventory_csv=inventory_csv,
        build_config_path=build_config_path,
        split_manifest=split_manifest,
    )
    if dataset_episode_lengths != sidecar["episode_lengths"]:
        raise ContactAsym25PreflightError(
            "contact inventory/sidecars do not match asymmetric25 dataset episodes and frame counts"
        )
    dataset_id = info.get("dataset_id")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id.strip()
        or dataset_id != sidecar["dataset_id"]
    ):
        raise ContactAsym25PreflightError(
            "contact sidecars and asymmetric25 dataset have different dataset_id values"
        )
    dataset_fps = float(info.get("fps", -1.0))
    sidecar_fps = float(sidecar["meta"].get("target_fps", -2.0))
    manifest_fps = float(manifest["timing"]["nominal_rate_hz"])
    if dataset_fps != sidecar_fps or dataset_fps != manifest_fps:
        raise ContactAsym25PreflightError(
            "dataset, contact sidecar, and hardware manifest rates differ"
        )
    if (
        info.get("frame_clock") != "dataset_native_frame_index"
        or sidecar["meta"].get("frame_clock") != info.get("frame_clock")
    ):
        raise ContactAsym25PreflightError("dataset/contact frame-clock semantics differ")
    expected_timestamp_semantics = {
        "feature": manifest["timing"]["timestamp_feature"],
        "origin": manifest["timing"]["timestamp_origin"],
        "state": manifest["timing"]["state_sample_semantics"],
        "action": manifest["timing"]["action_sample_semantics"],
    }
    if info.get("timestamp_semantics") != expected_timestamp_semantics:
        raise ContactAsym25PreflightError(
            "dataset timestamp semantics differ from the hardware manifest"
        )
    expected = {
        "independent_segment_count": sidecar["independent_segment_count"],
        "hardware_contract_sha256": contract_digest,
        "calibration_capture_sha256": capture_digest,
        "dataset_info_sha256": dataset_info_digest,
        "norm_stats_sha256": sha256_file(norm_stats_path),
        "sidecar_meta_sha256": sidecar["meta_sha256"],
        "sidecar_bundle_sha256": sidecar["sidecar_bundle_sha256"],
        "split_manifest_sha256": sidecar["split_manifest_sha256"],
        "episode_metadata_sha256": episode_metadata_digest,
        "coverage_csv_sha256": sha256_file(coverage_csv),
        "segments_csv_sha256": sha256_file(segments_csv),
        "inventory_csv_sha256": sha256_file(inventory_csv),
        "build_config_sha256": sha256_file(build_config_path),
    }
    gate_path = Path(training_gate_path).expanduser().absolute()
    gate, gate_digest = _load_json_with_digest(
        gate_path, "contact-asym25 training gate"
    )
    (
        evidence_digests,
        matched_control_expectations,
        gate_acceptance,
        matched_control_run_contract,
        selected_transport_variant,
    ) = _validate_training_gate(
        gate,
        expected,
        gate_path=gate_path,
        dataset_id=dataset_id,
        split_seed=split_seed,
        training_run_id=training_run_id,
        training_run_nonce=training_run_nonce,
        matched_control_variant_id=matched_control_variant_id,
    )
    selected_split = sidecar["split_episode_ids"].get(split_seed)
    if (
        not isinstance(selected_split, Mapping)
        or set(selected_split) != {"train", "val", "test"}
        or any(not selected_split[name] for name in ("train", "val", "test"))
    ):
        raise ContactAsym25PreflightError(
            f"production split seed {split_seed} is missing a non-empty train/val/test set"
        )
    if (
        set(selected_split["train"]) & set(selected_split["val"])
        or set(selected_split["train"]) & set(selected_split["test"])
        or set(selected_split["val"]) & set(selected_split["test"])
        or set(selected_split["train"])
        | set(selected_split["val"])
        | set(selected_split["test"])
        != set(dataset_episode_lengths)
    ):
        raise ContactAsym25PreflightError(
            "selected production split is overlapping or incomplete"
        )
    try:
        production_split_receipt, production_split_receipt_sha256 = (
            build_production_split_receipt(
                dataset_id=dataset_id,
                action_schema_id=ASYMMETRIC_SCHEMA_ID,
                split_seed=split_seed,
                train_episode_ids=selected_split["train"],
                validation_episode_ids=selected_split["val"],
                heldout_episode_ids=selected_split["test"],
                dataset_info_sha256=dataset_info_digest,
                episode_metadata_sha256=episode_metadata_digest,
                split_manifest_sha256=sidecar["split_manifest_sha256"],
            )
        )
        norm_source_tree_receipt = build_norm_source_tree_receipt(data_root)
        source_files = norm_source_tree_receipt["files"]
        info_receipts = [
            row for row in source_files if row["logical_path"] == "meta/info.json"
        ]
        episode_receipts = [
            row
            for row in source_files
            if row["logical_path"] == "meta/episodes.jsonl"
            or (
                row["logical_path"].startswith("meta/episodes/")
                and row["logical_path"].endswith(".parquet")
            )
        ]
        if (
            len(info_receipts) != 1
            or info_receipts[0]["sha256"] != dataset_info_digest
            or canonical_json_sha256(
                [
                    {"path": row["logical_path"], "sha256": row["sha256"]}
                    for row in episode_receipts
                ]
            )
            != episode_metadata_digest
        ):
            raise ContactAsym25PreflightError(
                "dataset metadata changed between verified reads"
            )
        train_norm_source_receipt, train_norm_source_receipt_sha256 = (
            build_expected_train_norm_source_receipt(
                production_split_receipt=production_split_receipt,
                production_split_receipt_sha256=production_split_receipt_sha256,
                episode_lengths=dataset_episode_lengths,
                source_dataset_tree_receipt=norm_source_tree_receipt,
            )
        )
        train_raw_statistics, expected_norm_stats, train_raw_statistics_sha256 = (
            _recompute_production_train_norm_from_verified_raw(
                data_root=data_root,
                episode_lengths=dataset_episode_lengths,
                train_norm_source_receipt=train_norm_source_receipt,
                train_norm_source_receipt_sha256=train_norm_source_receipt_sha256,
            )
        )
        norm_bytes, norm_digest, _ = read_verified_bytes(
            norm_stats_path,
            expected_sha256=expected["norm_stats_sha256"],
        )
        norm_payload = strict_json_loads(norm_bytes, name="normalization stats")
        if norm_digest != expected["norm_stats_sha256"] or not isinstance(
            norm_payload, dict
        ):
            raise ContactAsym25PreflightError(
                "verified normalization stats payload differs"
            )
        validate_asymmetric_norm_payload(
            norm_payload,
            model_dim=MODEL_ACTION_DIM,
            expected_hardware_contract_sha256=contract_digest,
            expected_calibration_capture_sha256=capture_digest,
        )
        validate_production_train_norm_metadata(
            norm_payload.get("metadata"),
            expected_source_receipt=train_norm_source_receipt,
            expected_source_receipt_sha256=train_norm_source_receipt_sha256,
            expected_raw_statistics=train_raw_statistics,
            expected_raw_statistics_sha256=train_raw_statistics_sha256,
            data_root=data_root,
        )
        validate_production_train_norm_payload_statistics(
            norm_payload,
            expected_norm_stats=expected_norm_stats,
        )
    except (ProductionTrainNormError, ValueError) as exc:
        raise ContactAsym25PreflightError(
            "normalization stats are not bound to the selected train-only split"
        ) from exc
    training_data_receipt = {
        "receipt_version": TRAINING_DATA_RECEIPT_VERSION,
        "dataset_id": dataset_id,
        "action_schema_id": ASYMMETRIC_SCHEMA_ID,
        "split_seed": split_seed,
        "train_episode_ids": list(selected_split["train"]),
        "validation_episode_ids": list(selected_split["val"]),
        "heldout_episode_ids": list(selected_split["test"]),
        "dataset_info_sha256": dataset_info_digest,
        "episode_metadata_sha256": episode_metadata_digest,
        "norm_stats_sha256": expected["norm_stats_sha256"],
        "sidecar_bundle_sha256": sidecar["sidecar_bundle_sha256"],
        "split_manifest_sha256": sidecar["split_manifest_sha256"],
        "training_gate_sha256": gate_digest,
        "preflight_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    training_data_receipt_sha256 = hashlib.sha256(
        json.dumps(
            training_data_receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    artifact_paths = {
        "data_root": str(data_root),
        "norm_stats": str(norm_stats_path),
        "hardware_manifest": str(hardware_manifest_path),
        "calibration_capture": str(calibration_capture_path),
        "sidecar_dir": str(Path(sidecar_dir).resolve()),
        "coverage_csv": str(Path(coverage_csv).resolve()),
        "segments_csv": str(Path(segments_csv).resolve()),
        "inventory_csv": str(Path(inventory_csv).resolve()),
        "build_config": str(Path(build_config_path).resolve()),
        "split_manifest": str(Path(split_manifest).resolve()),
        "training_gate": str(gate_path),
    }
    return {
        "receipt_version": PREFLIGHT_RECEIPT_VERSION,
        "status": "artifact_consistency_passed_with_explicit_external_gates",
        "scientific_gate_claimed": False,
        "action_schema_id": ASYMMETRIC_SCHEMA_ID,
        "contact_schema_version": SEGMENT_SCHEMA_VERSION,
        "contact_target_dim": SEGMENT_TARGET_DIM,
        "num_contact_tokens": 12,
        "time_bins": 16,
        "evidence_valid_until_utc": gate["evidence_valid_until_utc"],
        "split_seed": split_seed,
        "train_episode_ids": list(selected_split["train"]),
        "validation_episode_ids": list(selected_split["val"]),
        "heldout_episode_ids": list(selected_split["test"]),
        "training_data_receipt": training_data_receipt,
        "training_data_receipt_sha256": training_data_receipt_sha256,
        "production_split_receipt": production_split_receipt,
        "production_split_receipt_sha256": production_split_receipt_sha256,
        "train_norm_source_receipt": train_norm_source_receipt,
        "train_norm_source_receipt_sha256": train_norm_source_receipt_sha256,
        "train_raw_statistics_sha256": train_raw_statistics_sha256,
        "norm_source_tree_receipt": norm_source_tree_receipt,
        "dataset_id": dataset_id,
        "dataset_fps": dataset_fps,
        "horizon_frames": 48,
        "episode_lengths": {
            str(episode): int(length)
            for episode, length in sorted(dataset_episode_lengths.items())
        },
        "data_root": str(data_root),
        "norm_stats_path": str(norm_stats_path),
        "sidecar_dir": str(Path(sidecar_dir).resolve()),
        "split_manifest_path": str(Path(split_manifest).resolve()),
        "training_gate_path": str(gate_path),
        "training_gate_sha256": gate_digest,
        "preflight_code_sha256": sha256_file(Path(__file__).resolve()),
        "train_norm_contract_code_sha256": sha256_file(
            Path(__file__).with_name("contact_asym25_train_norm_contract.py")
        ),
        "artifact_paths": artifact_paths,
        "episode_npz_sha256": sidecar["episode_npz_sha256"],
        "evidence_report_sha256": evidence_digests,
        "matched_control_expectations": matched_control_expectations,
        "matched_control_run_contract": matched_control_run_contract,
        "matched_control_run_contract_sha256": matched_control_expectations[
            "expected_run_contract_sha256"
        ],
        "gate_reviewer_acceptance": gate_acceptance,
        "gate_reviewer_acceptance_sha256": evidence_digests[
            "gate_reviewer_acceptance"
        ],
        **expected,
    }


def preflight_from_environment() -> dict:
    missing = [name for name in REQUIRED_ENV_PATHS if not os.environ.get(name, "").strip()]
    if missing:
        raise ContactAsym25PreflightError(
            f"missing required contact-asym25 preflight environment inputs: {missing}"
        )
    split_seed_text = os.environ["GWP_SPLIT_SEED"]
    if split_seed_text not in {"6666", "7777", "8888"}:
        raise ContactAsym25PreflightError(
            "GWP_SPLIT_SEED must be one of 6666, 7777, 8888"
        )
    split_seed = int(split_seed_text)
    return validate_contact_asym25_training_preflight(
        data_root=os.environ["GWP_SONIC_DATA_PATH"],
        norm_stats_path=os.environ["GWP_NORM_STATS_PATH"],
        hardware_manifest_path=os.environ["GWP_ASYMMETRIC_HARDWARE_MANIFEST"],
        calibration_capture_path=os.environ["GWP_ASYMMETRIC_CALIBRATION_CAPTURE"],
        sidecar_dir=os.environ["GWP_CONTACT_SIDECAR_DIR"],
        coverage_csv=os.environ["GWP_CONTACT_COVERAGE_CSV"],
        segments_csv=os.environ["GWP_CONTACT_SEGMENTS_CSV"],
        inventory_csv=os.environ["GWP_CONTACT_INVENTORY_CSV"],
        build_config_path=os.environ["GWP_CONTACT_BUILD_CONFIG"],
        split_manifest=os.environ["GWP_CONTACT_SPLIT_MANIFEST"],
        training_gate_path=os.environ["GWP_CONTACT_ASYM25_TRAINING_GATE"],
        split_seed=split_seed,
        training_run_id=os.environ["GWP_CONTACT_TRAINING_RUN_ID"],
        training_run_nonce=os.environ["GWP_CONTACT_TRAINING_RUN_NONCE"],
        matched_control_variant_id=os.environ["GWP_MATCHED_CONTROL_VARIANT_ID"],
    )

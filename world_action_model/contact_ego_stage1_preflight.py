"""Fail-closed production gate for discrete-contact Ego Stage1 pretraining."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess

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
from world_action_model.contact_transition_schema import (
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.ego_action_schema import (
    CANONICAL_OPEN_FRACTION_SEMANTICS,
    EGO_CANONICAL_ACTION_SCHEMA_ID,
    EGO_HAND_STATUS_SEMANTICS,
    EGO_WAM_ACTION_DIM,
)
from world_action_model.ego_dataset_fingerprint import (
    EGO_STAGE1_FPS,
    EGO_STAGE1_FRAME_CLOCK,
    derive_ego_dataset_identity,
    derive_ego_t5_source_receipt,
)
from world_action_model.ego_stage1_gate_bootstrap import (
    EGO_STAGE1_GATE_VERSION,
    EGO_STAGE1_REQUIRED_ENV_PATHS,
    EGO_STAGE1_REQUIRED_GATE_FLAGS,
    EgoStage1BootstrapError,
    validate_ego_stage1_gate_freshness_and_training_nonce,
)
import world_action_model.matched_control_hashes as matched_control_hashes
from world_action_model.matched_control_hashes import (
    COLLECTOR_VERSION,
    HASH_SCHEMA,
    IMPLEMENTED_RUNNER_VARIANTS,
    canonical_sha256,
)
from world_action_model.verified_io import (
    hash_verified_regular_file,
    read_verified_bytes,
    strict_json_loads,
)


REQUIRED_GATE_FLAGS = EGO_STAGE1_REQUIRED_GATE_FLAGS
REQUIRED_ENV_PATHS = EGO_STAGE1_REQUIRED_ENV_PATHS
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REVIEWED_BUILDER_ID = "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py"
REQUIRED_SIDECAR_KEYS = (
    "contact_target", "contact_loss_mask", "contact_confidence", "contact_condition_mask"
)


class ContactEgoStage1PreflightError(ValueError):
    pass


def _is_finite_json_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    )


def sha256_file(path: str | Path) -> str:
    digest, _ = hash_verified_regular_file(path)
    return digest


def validate_contact_sidecar_arrays(arrays: Mapping[str, np.ndarray], *, frame_count: int | None = None) -> None:
    if set(arrays) != set(REQUIRED_SIDECAR_KEYS):
        raise ContactEgoStage1PreflightError("contact sidecar arrays differ from v1.1")
    lengths = set()
    for key in REQUIRED_SIDECAR_KEYS:
        value = np.asarray(arrays[key])
        if value.ndim != 3 or value.shape[1:] != (12, SEGMENT_TARGET_DIM) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ContactEgoStage1PreflightError(f"invalid reviewed sidecar array: {key}")
        lengths.add(value.shape[0])
    if len(lengths) != 1 or (frame_count is not None and lengths != {frame_count}):
        raise ContactEgoStage1PreflightError("sidecar frame count differs")
    target, mask = arrays["contact_target"], arrays["contact_loss_mask"]
    confidence, condition = arrays["contact_confidence"], arrays["contact_condition_mask"]
    if not np.all((mask == 0) | (mask == 1)) or not np.array_equal(mask, condition):
        raise ContactEgoStage1PreflightError("sidecar field masks are invalid")
    if np.any((target < 0) | (target > 1)) or np.any((confidence < 0) | (confidence > 1)):
        raise ContactEgoStage1PreflightError("sidecar values are outside [0,1]")
    if np.any(target[mask == 0] != 0) or np.any(confidence[mask == 0] != 0) or np.any(confidence[mask == 1] <= 0):
        raise ContactEgoStage1PreflightError("unknown/supervised field encoding differs")
    presence = SEGMENT_FIELD_SLICES["presence"]
    ptarget, pmask = target[..., presence].squeeze(-1), mask[..., presence].squeeze(-1)
    if not np.all((ptarget == 0) | (ptarget == 1)):
        raise ContactEgoStage1PreflightError("presence is not binary")
    other = mask.copy(); other[..., presence] = 0
    if np.any((ptarget == 0) & np.any(other != 0, axis=-1)):
        raise ContactEgoStage1PreflightError("absent query supervises attributes")
    for field in ("side", "site", "object_role", "contact_mode"):
        sl = SEGMENT_FIELD_SLICES[field]; fm = mask[..., sl]
        if not np.all(fm == fm[..., :1]):
            raise ContactEgoStage1PreflightError(f"partial categorical mask: {field}")
        active = fm[..., 0] == 1
        if np.any(np.abs(target[..., sl].sum(-1)[active] - 1) > 1e-6):
            raise ContactEgoStage1PreflightError(f"non-one-hot categorical field: {field}")
    start, end = SEGMENT_FIELD_SLICES["start_tau"], SEGMENT_FIELD_SLICES["end_tau"]
    left, right = SEGMENT_FIELD_SLICES["left_censored"], SEGMENT_FIELD_SLICES["right_censored"]
    if not np.array_equal(mask[..., start], mask[..., left]) or not np.array_equal(mask[..., end], mask[..., right]):
        raise ContactEgoStage1PreflightError("tau/censor masks differ")
    both = (mask[..., start].squeeze(-1) == 1) & (mask[..., end].squeeze(-1) == 1)
    if np.any(target[..., start].squeeze(-1)[both] > target[..., end].squeeze(-1)[both]):
        raise ContactEgoStage1PreflightError("start_tau exceeds end_tau")
    la = (mask[..., left].squeeze(-1) == 1) & (target[..., left].squeeze(-1) == 1)
    ra = (mask[..., right].squeeze(-1) == 1) & (target[..., right].squeeze(-1) == 1)
    if np.any(target[..., start].squeeze(-1)[la] != 0) or np.any(target[..., end].squeeze(-1)[ra] != 1):
        raise ContactEgoStage1PreflightError("censored tau is not canonical")


def _load_json(path: str | Path, name: str) -> dict:
    try:
        payload_bytes, _, _ = read_verified_bytes(path)
        payload = strict_json_loads(payload_bytes, name=name)
    except (OSError, ValueError) as exc:
        raise ContactEgoStage1PreflightError(f"invalid or unsafe {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ContactEgoStage1PreflightError(f"{name} must be a JSON object")
    return payload


def _load_json_with_digest(path: str | Path, name: str) -> tuple[dict, str]:
    try:
        payload_bytes, digest, _ = read_verified_bytes(path)
        payload = strict_json_loads(payload_bytes, name=name)
    except (OSError, ValueError) as exc:
        raise ContactEgoStage1PreflightError(f"invalid or unsafe {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ContactEgoStage1PreflightError(f"{name} must be a JSON object")
    return payload, digest


def _read_verified_csv(
    path: str | Path, fields: tuple[str, ...] | list[str], name: str
) -> tuple[list[dict[str, str]], str]:
    path = Path(path)
    try:
        payload, digest, _ = read_verified_bytes(path)
        rows = read_csv_exact(path, fields, payload=payload)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContactEgoStage1PreflightError(f"invalid or unsafe {name}: {path}") from exc
    return rows, digest


def _validate_reviewed_meta(meta: Mapping) -> None:
    expected = {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "target_dim": SEGMENT_TARGET_DIM,
        "num_queries": 12,
        "overflow_policy": "hard_error_no_drop",
        "dropped_segments": 0,
        "horizon_frames": 48,
        "builder_identity": REVIEWED_BUILDER_ID,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "target_fps": EGO_STAGE1_FPS,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ContactEgoStage1PreflightError(
                f"contact sidecar meta {key}={meta.get(key)!r}; expected {value!r}"
            )
    if meta.get("annotation_contract") != {
        "annotation_role": "adjudicated",
        "review_status": "reviewed",
        "label_origin": "human_adjudicated",
        "automatic_or_proposal_ground_truth": False,
    }:
        raise ContactEgoStage1PreflightError("sidecar is not human-adjudicated reviewed GT")
    reviewed_rows = meta.get("reviewed_coverage_rows")
    if isinstance(reviewed_rows, bool) or not isinstance(reviewed_rows, int) or reviewed_rows <= 0:
        raise ContactEgoStage1PreflightError("sidecar has no reviewed coverage")
    segment_count = meta.get("unique_physical_segments")
    if isinstance(segment_count, bool) or not isinstance(segment_count, int) or segment_count <= 0:
        raise ContactEgoStage1PreflightError("sidecar has no physical segments")
    expected_slices = {key: [value.start, value.stop] for key, value in SEGMENT_FIELD_SLICES.items()}
    if meta.get("field_slices") != expected_slices:
        raise ContactEgoStage1PreflightError("sidecar field_slices differ from v1.1")


def _validate_group_metadata(
    rows: list[dict[str, str]],
) -> dict[int, dict[str, str]]:
    required = (
        "subject_id",
        "session_id",
        "operator_id",
        "source_object_instance_id",
        "target_object_instance_id",
        "liquid_id",
        "collection_batch_id",
    )
    result = {}
    for row_number, row in enumerate(rows, 2):
        missing = [key for key in required if not row[key]]
        if missing or row["metadata_status"] != "complete":
            raise ContactEgoStage1PreflightError(
                f"inventory row {row_number} has incomplete group metadata: {missing}"
            )
        result[int(row["episode_id"])] = row
    return result


def _validate_split_manifest(
    rows: list[dict[str, str]],
    inventory: Mapping[int, object],
    inventory_rows: Mapping[int, Mapping[str, str]],
    split_policy: Mapping,
) -> None:
    expected_header = [
        "episode_id", "length_frames", "seed_6666_split", "seed_7777_split",
        "seed_8888_split", "split_unit", "split_purpose", "ood_eligible",
        "strategy_claim_allowed", "metadata_status",
    ]
    if any(set(row) != set(expected_header) for row in rows):
        raise ContactEgoStage1PreflightError("split manifest header mismatch")
    fields_by_seed = split_policy.get("group_fields_by_seed")
    allowed = {
        "subject_id", "session_id", "operator_id", "source_object_instance_id",
        "target_object_instance_id", "liquid_id", "collection_batch_id",
    }
    seeds = ("seed_6666_split", "seed_7777_split", "seed_8888_split")
    if not isinstance(fields_by_seed, Mapping):
        raise ContactEgoStage1PreflightError("split policy lacks group_fields_by_seed")
    for seed in seeds:
        fields = fields_by_seed.get(seed)
        if not isinstance(fields, list) or not fields or any(field not in allowed for field in fields):
            raise ContactEgoStage1PreflightError(f"invalid group fields for {seed}")
    split_rows = {}
    for row in rows:
        episode = int(row["episode_id"])
        if episode in split_rows or episode not in inventory:
            raise ContactEgoStage1PreflightError("split has duplicate/unknown episode")
        split_rows[episode] = row
        if int(row["length_frames"]) != inventory[episode].length_frames:
            raise ContactEgoStage1PreflightError("split/inventory length mismatch")
        if (
            row["split_unit"] != "group"
            or row["split_purpose"] != "production_strategy_evaluation"
            or row["ood_eligible"] != "true"
            or row["strategy_claim_allowed"] != "true"
            or row["metadata_status"] != "complete"
        ):
            raise ContactEgoStage1PreflightError("diagnostic/non-strategy split is forbidden")
        if any(row[seed] not in {"train", "val", "test"} for seed in seeds):
            raise ContactEgoStage1PreflightError("invalid split assignment")
    if set(split_rows) != set(inventory):
        raise ContactEgoStage1PreflightError("split does not cover every episode")
    for seed in seeds:
        assignment = {}
        for episode, row in split_rows.items():
            group = tuple(inventory_rows[episode][field] for field in fields_by_seed[seed])
            previous = assignment.setdefault(group, row[seed])
            if previous != row[seed]:
                raise ContactEgoStage1PreflightError(f"{seed} group leakage: {group}")
        if {row[seed] for row in split_rows.values()} != {"train", "val", "test"}:
            raise ContactEgoStage1PreflightError(f"{seed} lacks train/val/test")


def validate_ego_reviewed_sidecar_bundle(
    *, sidecar_dir: str | Path, coverage_csv: str | Path, segments_csv: str | Path,
    inventory_csv: str | Path, build_config_path: str | Path, split_manifest: str | Path,
) -> dict:
    sidecar_dir = Path(sidecar_dir).absolute()
    coverage_csv, segments_csv, inventory_csv = map(
        lambda value: Path(value).absolute(), (coverage_csv, segments_csv, inventory_csv)
    )
    build_config_path = Path(build_config_path).absolute()
    split_manifest = Path(split_manifest).absolute()
    if sidecar_dir.is_symlink() or not sidecar_dir.is_dir():
        raise ContactEgoStage1PreflightError(
            "reviewed sidecar directory must be a real directory"
        )
    for path in (sidecar_dir, coverage_csv, segments_csv, inventory_csv, build_config_path, split_manifest):
        if not path.exists():
            raise ContactEgoStage1PreflightError(f"missing sidecar artifact: {path}")
    meta_path = sidecar_dir / "meta.json"
    try:
        meta_bytes, meta_digest, _ = read_verified_bytes(meta_path)
        meta = strict_json_loads(meta_bytes, name="sidecar meta")
    except (OSError, ValueError) as exc:
        raise ContactEgoStage1PreflightError("invalid or unsafe sidecar meta") from exc
    if not isinstance(meta, dict):
        raise ContactEgoStage1PreflightError("sidecar meta must be a JSON object")
    _validate_reviewed_meta(meta)
    config, config_digest = _load_json_with_digest(
        build_config_path, "sidecar build config"
    )
    expected_config = {
        "sidecar_schema_version": SEGMENT_SCHEMA_VERSION,
        "accepted_annotation_role": "adjudicated",
        "horizon_frames": 48,
        "num_queries": 12,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "target_fps": EGO_STAGE1_FPS,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ContactEgoStage1PreflightError(f"sidecar build config differs: {key}")
    if not isinstance(config.get("dataset_id"), str) or not config["dataset_id"].strip():
        raise ContactEgoStage1PreflightError("sidecar config lacks stable dataset_id")
    for key in ("dataset_id", "frame_clock", "target_fps", "horizon_frames", "num_queries"):
        if meta.get(key) != config.get(key):
            raise ContactEgoStage1PreflightError("sidecar meta/build config differ")
    split_policy = config.get("split_policy")
    if not isinstance(split_policy, Mapping) or split_policy.get("strategy_claim_allowed") is not True:
        raise ContactEgoStage1PreflightError("sidecar build config is diagnostic/strategy=false")
    inventory_csv_rows, inventory_digest = _read_verified_csv(
        inventory_csv, INVENTORY_FIELDS, "inventory CSV"
    )
    coverage_csv_rows, coverage_digest = _read_verified_csv(
        coverage_csv, COVERAGE_FIELDS, "coverage CSV"
    )
    segments_csv_rows, segments_digest = _read_verified_csv(
        segments_csv, SEGMENT_FIELDS, "segments CSV"
    )
    split_fields = (
        "episode_id", "length_frames", "seed_6666_split", "seed_7777_split",
        "seed_8888_split", "split_unit", "split_purpose", "ood_eligible",
        "strategy_claim_allowed", "metadata_status",
    )
    split_rows, split_digest = _read_verified_csv(
        split_manifest, split_fields, "split manifest"
    )
    inventory = load_inventory(inventory_csv, rows=inventory_csv_rows)
    inventory_rows = _validate_group_metadata(inventory_csv_rows)
    _validate_split_manifest(split_rows, inventory, inventory_rows, split_policy)
    coverages = load_coverages(
        coverage_csv, inventory, config, rows=coverage_csv_rows
    )
    segments = load_segments(
        segments_csv, inventory, coverages, config, rows=segments_csv_rows
    )
    if len(segments) != int(meta["unique_physical_segments"]):
        raise ContactEgoStage1PreflightError("segment count differs from meta")
    source_digests = {
        "coverage_csv": coverage_digest,
        "segments_csv": segments_digest,
        "inventory_csv": inventory_digest,
        "config": config_digest,
        "split_manifest": split_digest,
    }
    hashes = meta.get("input_sha256")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(source_digests)
        or any(
        hashes.get(key) != digest for key, digest in source_digests.items()
        )
    ):
        raise ContactEgoStage1PreflightError("sidecar source CSV/config/split fingerprint differs")
    by_coverage = {episode: [] for episode in inventory}
    by_segment = {episode: [] for episode in inventory}
    for item in coverages:
        by_coverage[item.episode_id].append(item)
    for item in segments:
        by_segment[item.episode_id].append(item)
    expected_files = {f"episode_{episode:06d}.npz" for episode in inventory}
    if {path.name for path in sidecar_dir.glob("episode_*.npz")} != expected_files:
        raise ContactEgoStage1PreflightError("sidecar episode file set differs")
    declared_episode_hashes = meta.get("episode_npz_sha256")
    if (
        not isinstance(declared_episode_hashes, Mapping)
        or set(declared_episode_hashes) != expected_files
        or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in declared_episode_hashes.values()
        )
    ):
        raise ContactEgoStage1PreflightError("sidecar NPZ hash differs")
    episode_hashes = {}
    provenance = []
    for episode, item in sorted(inventory.items()):
        expected_arrays, rows = encode_episode(
            item.length_frames, by_coverage[episode], by_segment[episode],
            horizon_frames=48, num_queries=12,
            negative_confidence=float(config["negative_confidence"]),
        )
        name = f"episode_{episode:06d}.npz"
        try:
            payload, digest, _ = read_verified_bytes(
                sidecar_dir / name,
                expected_sha256=declared_episode_hashes[name],
            )
        except (OSError, ValueError) as exc:
            raise ContactEgoStage1PreflightError(
                f"sidecar NPZ hash differs: {name}"
            ) from exc
        episode_hashes[name] = digest
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actual = {key: archive[key] for key in archive.files}
        validate_contact_sidecar_arrays(actual, frame_count=item.length_frames)
        if any(not np.array_equal(actual[key], expected_arrays[key]) for key in REQUIRED_SIDECAR_KEYS):
            raise ContactEgoStage1PreflightError("sidecar differs from full reviewed re-encoding")
        provenance.extend({"episode_id": str(episode), **{k: str(v) for k, v in row.items()}} for row in rows)
    provenance_path = sidecar_dir / "window_segment_provenance.csv"
    provenance_fields = (
        "episode_id", "frame_index", "query_index", "segment_id",
        "onset_event_id", "release_event_id",
    )
    actual_provenance, provenance_digest = _read_verified_csv(
        provenance_path, provenance_fields, "window provenance CSV"
    )
    if actual_provenance != provenance:
        raise ContactEgoStage1PreflightError("window provenance differs from physical segment assignment")
    if meta.get("window_segment_provenance_sha256") != provenance_digest:
        raise ContactEgoStage1PreflightError("provenance hash differs")
    bundle_digest = hashlib.sha256(json.dumps({
        "episode_npz_sha256": episode_hashes,
        "window_segment_provenance_sha256": provenance_digest,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if meta.get("sidecar_bundle_sha256") != bundle_digest:
        raise ContactEgoStage1PreflightError("sidecar bundle hash differs")
    return {
        "dataset_id": config["dataset_id"], "episode_lengths": {k: int(v.length_frames) for k, v in inventory.items()},
        "episode_count": len(inventory), "independent_segment_count": len(segments),
        "episode_npz_sha256": episode_hashes, "meta_sha256": meta_digest,
        "bundle_sha256": bundle_digest, "split_sha256": split_digest, "meta": meta,
        "source_sha256": source_digests,
    }


def _validate_norm(payload: Mapping, identity: Mapping, hdf5_dir: Path) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContactEgoStage1PreflightError("norm lacks metadata")
    expected = {
        "norm_contract_version": "ego-stage1-norm/v2",
        "action_schema_id": EGO_CANONICAL_ACTION_SCHEMA_ID,
        "source_dim": EGO_WAM_ACTION_DIM,
        "model_dim": 32,
        "action_horizon": 48,
        "action_start_offset": 1,
        "dataset_id": identity["dataset_id"],
        "dataset_fingerprint_version": identity["fingerprint_version"],
        "dataset_info_identity_fields": identity["info_identity_fields"],
        "dataset_episode_identity_fields": identity["episode_identity_fields"],
        "dataset_episode_sort_key": identity["episode_sort_key"],
        "dataset_task_sort_rule": identity["task_sort_rule"],
        "dataset_canonical_json": identity["canonical_json"],
        "dataset_fingerprint_sha256": identity["dataset_fingerprint_sha256"],
        "info_identity_sha256": identity["info_identity_sha256"],
        "episode_metadata_sha256": identity["episode_metadata_sha256"],
        "dataset_portable_info_ignored_fields": identity[
            "portable_info_ignored_fields"
        ],
        "dataset_training_source_roots": identity["training_source_roots"],
        "dataset_source_manifest": identity["source_manifest"],
        "dataset_source_manifest_sha256": identity["source_manifest_sha256"],
        "dataset_source_receipt": identity["source_receipt"],
        "dataset_source_receipt_sha256": identity["source_receipt_sha256"],
        "fps": EGO_STAGE1_FPS,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "source_hand_status_semantics": EGO_HAND_STATUS_SEMANTICS,
        "target_hand_status_semantics": CANONICAL_OPEN_FRACTION_SEMANTICS,
        "hand_status_conversion": "target=1-source",
        "padding_contract": "right_zero_pad_20_to_32",
        "action_dim_mask": [True] * 20 + [False] * 12,
        "hdf5_source_role": "processed_final_hdf5_20hz",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ContactEgoStage1PreflightError(f"norm semantic binding differs: {key}")
    if metadata.get("episode_lengths") != {str(k): v for k, v in identity["episode_lengths"].items()}:
        raise ContactEgoStage1PreflightError("norm episode lengths differ")
    valid_action_samples = sum(
        sum(min(48, max(0, length - frame - 1)) for frame in range(length))
        for length in identity["episode_lengths"].values()
    )
    if (
        metadata.get("episodes") != identity["episode_count"]
        or metadata.get("state_samples") != identity["total_frames"]
        or metadata.get("action_samples") != valid_action_samples
    ):
        raise ContactEgoStage1PreflightError("norm episode/sample counts differ")
    weights = metadata.get("action_loss_weight")
    if (
        not isinstance(weights, Mapping)
        or set(weights) != {
            "dimensions_0_17",
            "grip_proxy_dimensions_18_19",
            "padding_dimensions_20_31",
        }
        or any(not _is_finite_json_number(value) for value in weights.values())
        or weights.get("dimensions_0_17") != 1.0
        or weights.get("grip_proxy_dimensions_18_19") != 0.25
        or weights.get("padding_dimensions_20_31") != 0.0
    ):
        raise ContactEgoStage1PreflightError("norm lacks action loss-weight contract")
    if metadata.get("action_time_mask") != "normalization_excludes_clamped_padding;training_masks_clamped_padding_false":
        raise ContactEgoStage1PreflightError("norm lacks action time-mask contract")
    stats = payload.get("norm_stats")
    if not isinstance(stats, Mapping) or set(stats) != {"observation.state", "action"}:
        raise ContactEgoStage1PreflightError("norm stats keys differ")
    for feature in stats.values():
        if not isinstance(feature, Mapping):
            raise ContactEgoStage1PreflightError("invalid norm feature")
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            values = feature.get(name)
            if (
                not isinstance(values, list)
                or len(values) != 32
                or any(not _is_finite_json_number(value) for value in values)
            ):
                raise ContactEgoStage1PreflightError("norm arrays must be finite 32-D")
            if any(value != 0 for value in values[20:]):
                raise ContactEgoStage1PreflightError("norm padded dimensions must be exact zero")
    expected_files = {f"episode_{episode}.hdf5" for episode in identity["episode_lengths"]}
    actual_files = {path.name for path in hdf5_dir.glob("episode_*.hdf5")}
    if actual_files != expected_files:
        raise ContactEgoStage1PreflightError("final_hdf5 episode file set differs from dataset")
    actual_hashes = {name: sha256_file(hdf5_dir / name) for name in sorted(expected_files)}
    if metadata.get("episode_hdf5_sha256") != actual_hashes:
        raise ContactEgoStage1PreflightError("norm HDF5 content fingerprints differ")
    bundle = hashlib.sha256(
        json.dumps(actual_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if metadata.get("hdf5_bundle_sha256") != bundle:
        raise ContactEgoStage1PreflightError("norm HDF5 bundle fingerprint differs")
    return bundle


def _resolve_report(gate_path: Path, entry: Mapping, name: str) -> tuple[Path, dict, str]:
    if (
        not isinstance(entry, Mapping)
        or set(entry) != {"path", "sha256"}
        or not isinstance(entry.get("path"), str)
        or not entry["path"]
        or not isinstance(entry.get("sha256"), str)
        or not _SHA256_RE.fullmatch(entry["sha256"])
    ):
        raise ContactEgoStage1PreflightError(f"missing evidence report: {name}")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = gate_path.parent / path
    path = Path(os.path.abspath(path))
    try:
        payload_bytes, digest, _ = read_verified_bytes(
            path,
            expected_sha256=entry.get("sha256"),
        )
        payload = strict_json_loads(payload_bytes, name=name)
    except (OSError, ValueError) as exc:
        raise ContactEgoStage1PreflightError(
            f"evidence report hash/JSON differs: {name}"
        ) from exc
    if not isinstance(payload, dict):
        raise ContactEgoStage1PreflightError(f"{name} must be a JSON object")
    return path, payload, digest


def _current_git_commit(start: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContactEgoStage1PreflightError(
            "cannot resolve current Git commit for Stage1 evidence binding"
        ) from exc
    commit = result.stdout.strip()
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise ContactEgoStage1PreflightError("current Git commit is not a full SHA-1")
    return commit


def _head_blob_sha256(source_path: Path) -> str:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = Path(root_result.stdout.strip())
        relative = source_path.relative_to(repo_root).as_posix()
        blob = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise ContactEgoStage1PreflightError(
            "cannot bind installed collector bytes to the current HEAD tree"
        ) from exc
    return hashlib.sha256(blob).hexdigest()


def _validate_gate_freshness_and_training_nonce(
    gate: Mapping, *, training_run_nonce: str
) -> None:
    try:
        validate_ego_stage1_gate_freshness_and_training_nonce(
            dict(gate), training_run_nonce=training_run_nonce
        )
    except EgoStage1BootstrapError as exc:
        raise ContactEgoStage1PreflightError(str(exc)) from exc


def _validate_agreement_metrics(metrics: object) -> bool:
    thresholds = {
        "coverage_kappa": ("min", 0.70, -1.0, 1.0),
        "event_f1": ("min", 0.80, 0.0, 1.0),
        "median_temporal_iou": ("min", 0.70, 0.0, 1.0),
        "boundary_mae_frames": ("max", 2.0, 0.0, math.inf),
        "categorical_kappa": ("min", 0.60, -1.0, 1.0),
    }
    if not isinstance(metrics, Mapping) or set(metrics) != set(thresholds):
        return False
    for name, (direction, threshold, lower, upper) in thresholds.items():
        value = metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            return False
        if value < lower or value > upper:
            return False
        if (direction == "min" and value < threshold) or (
            direction == "max" and value > threshold
        ):
            return False
    return True


def _validate_loaded_gate_runtime_topology(gate: Mapping) -> None:
    expectations = gate.get("matched_control_expectations")
    if not isinstance(expectations, Mapping):
        raise ContactEgoStage1PreflightError(
            "training gate lacks matched-control topology"
        )
    expected_world_size = expectations.get("expected_world_size")
    expected_rank_set = expectations.get("expected_rank_set")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size <= 0
        or not isinstance(expected_rank_set, list)
        or any(
            isinstance(rank, bool) or not isinstance(rank, int)
            for rank in expected_rank_set
        )
        or expected_rank_set != list(range(expected_world_size))
    ):
        raise ContactEgoStage1PreflightError(
            "training gate matched-control topology is invalid"
        )
    rank_text = os.environ.get("RANK", "").strip()
    world_text = os.environ.get("WORLD_SIZE", "").strip()
    if not re.fullmatch(r"[0-9]+", rank_text, flags=re.ASCII) or not re.fullmatch(
        r"[0-9]+", world_text, flags=re.ASCII
    ):
        raise ContactEgoStage1PreflightError(
            "RANK/WORLD_SIZE must be canonical ASCII-decimal integers"
        )
    rank = int(rank_text)
    world_size = int(world_text)
    if (
        world_size <= 0
        or rank < 0
        or rank >= world_size
        or world_size != expected_world_size
        or rank not in expected_rank_set
    ):
        raise ContactEgoStage1PreflightError(
            "runtime topology differs from the loaded training gate"
        )


def _validate_matched_control_expectations(gate: Mapping) -> dict:
    """Load reviewer-authored expectations without consulting the report.

    These values are deliberately part of the production gate.  Deriving any
    of them from the evidence report would let the artifact certify itself.
    """

    raw = gate.get("matched_control_expectations")
    required = {
        "expected_git_commit",
        "expected_run_id",
        "expected_run_nonce",
        "expected_report_sha256",
        "expected_trace_files",
        "expected_trace_manifest_sha256",
        "expected_global_hashes_sha256",
        "expected_hash_schema_sha256",
        "expected_control_config_sha256",
        "expected_variant_config_sha256",
        "expected_world_size",
        "expected_rank_set",
        "expected_collector_version",
        "expected_collector_source_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ContactEgoStage1PreflightError(
            "training gate matched_control_expectations are missing or have an unexpected schema"
        )
    git_commit = raw.get("expected_git_commit")
    run_id = raw.get("expected_run_id")
    run_nonce = raw.get("expected_run_nonce")
    report_sha256 = raw.get("expected_report_sha256")
    trace_files = raw.get("expected_trace_files")
    trace_manifest_sha256 = raw.get("expected_trace_manifest_sha256")
    global_hashes_sha256 = raw.get("expected_global_hashes_sha256")
    hash_schema_sha256 = raw.get("expected_hash_schema_sha256")
    control_hash = raw.get("expected_control_config_sha256")
    variant_hashes = raw.get("expected_variant_config_sha256")
    world_size = raw.get("expected_world_size")
    rank_set = raw.get("expected_rank_set")
    collector_version = raw.get("expected_collector_version")
    collector_source_sha256 = raw.get("expected_collector_source_sha256")
    if not isinstance(git_commit, str) or not _GIT_COMMIT_RE.fullmatch(git_commit):
        raise ContactEgoStage1PreflightError("invalid expected matched-control Git commit")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ContactEgoStage1PreflightError("invalid expected matched-control run ID")
    if not isinstance(run_nonce, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", run_nonce
    ):
        raise ContactEgoStage1PreflightError("invalid expected matched-control run nonce")
    for name, value in (
        ("report", report_sha256),
        ("trace manifest", trace_manifest_sha256),
        ("global hashes", global_hashes_sha256),
        ("hash schema", hash_schema_sha256),
    ):
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ContactEgoStage1PreflightError(
                f"invalid expected matched-control {name} SHA"
            )
    if not isinstance(control_hash, str) or not _SHA256_RE.fullmatch(control_hash):
        raise ContactEgoStage1PreflightError("invalid expected matched-control control config hash")
    if (
        not isinstance(variant_hashes, Mapping)
        or set(variant_hashes) != set(IMPLEMENTED_RUNNER_VARIANTS)
        or any(not isinstance(name, str) or not name for name in variant_hashes)
        or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in variant_hashes.values()
        )
    ):
        raise ContactEgoStage1PreflightError(
            "invalid expected matched-control variant config hash mapping"
        )
    if not isinstance(trace_files, Mapping) or set(trace_files) != set(
        IMPLEMENTED_RUNNER_VARIANTS
    ):
        raise ContactEgoStage1PreflightError(
            "expected matched-control trace variants differ from the implemented runner set"
        )
    canonical_trace_files = {}
    seen_trace_paths = set()
    for variant in IMPLEMENTED_RUNNER_VARIANTS:
        entries = trace_files[variant]
        if not isinstance(entries, list) or not entries:
            raise ContactEgoStage1PreflightError(
                f"expected matched-control traces are empty for {variant}"
            )
        canonical_trace_files[variant] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
                raise ContactEgoStage1PreflightError("invalid expected trace entry schema")
            path = entry["path"]
            digest = entry["sha256"]
            if (
                not isinstance(path, str)
                or not path
                or path in seen_trace_paths
                or not isinstance(digest, str)
                or not _SHA256_RE.fullmatch(digest)
            ):
                raise ContactEgoStage1PreflightError("invalid/duplicate expected trace entry")
            seen_trace_paths.add(path)
            canonical_trace_files[variant].append({"path": path, "sha256": digest})
    if canonical_sha256(canonical_trace_files) != trace_manifest_sha256:
        raise ContactEgoStage1PreflightError(
            "expected trace manifest SHA differs from expected trace entries"
        )
    if canonical_sha256(HASH_SCHEMA) != hash_schema_sha256:
        raise ContactEgoStage1PreflightError(
            "expected matched-control hash schema differs from installed code"
        )
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ContactEgoStage1PreflightError("invalid expected matched-control world size")
    if (
        not isinstance(rank_set, list)
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in rank_set)
        or rank_set != list(range(world_size))
    ):
        raise ContactEgoStage1PreflightError(
            "expected matched-control rank set must be the complete ordered topology"
        )
    if collector_version != COLLECTOR_VERSION:
        raise ContactEgoStage1PreflightError("unexpected matched-control collector version")
    if (
        not isinstance(collector_source_sha256, str)
        or not _SHA256_RE.fullmatch(collector_source_sha256)
    ):
        raise ContactEgoStage1PreflightError("invalid expected collector source hash")
    installed_source = Path(matched_control_hashes.__file__).absolute()
    try:
        installed_source_sha256, _ = hash_verified_regular_file(installed_source)
    except (OSError, ValueError) as exc:
        raise ContactEgoStage1PreflightError(
            "cannot verify installed matched-control collector source"
        ) from exc
    if installed_source_sha256 != collector_source_sha256:
        raise ContactEgoStage1PreflightError(
            "expected collector source differs from installed source bytes"
        )
    if _head_blob_sha256(installed_source) != installed_source_sha256:
        raise ContactEgoStage1PreflightError(
            "installed collector source has uncommitted bytes outside current HEAD"
        )
    if _current_git_commit(installed_source.parent) != git_commit:
        raise ContactEgoStage1PreflightError(
            "expected matched-control Git commit differs from current HEAD"
        )
    return {
        "expected_git_commit": git_commit,
        "expected_run_id": run_id,
        "expected_run_nonce": run_nonce,
        "expected_report_sha256": report_sha256,
        "expected_trace_files": canonical_trace_files,
        "expected_trace_manifest_sha256": trace_manifest_sha256,
        "expected_global_hashes_sha256": global_hashes_sha256,
        "expected_hash_schema_sha256": hash_schema_sha256,
        "expected_control_config_sha256": control_hash,
        "expected_variant_config_sha256": dict(variant_hashes),
        "expected_world_size": world_size,
        "expected_rank_set": list(rank_set),
        "expected_collector_version": collector_version,
        "expected_collector_source_sha256": collector_source_sha256,
    }


def _validate_gate(
    gate: Mapping,
    expected: Mapping,
    gate_path: Path,
    dataset_id: str,
    *,
    training_run_nonce: str,
) -> dict:
    if gate.get("gate_version") != EGO_STAGE1_GATE_VERSION:
        raise ContactEgoStage1PreflightError("training gate version differs")
    _validate_gate_freshness_and_training_nonce(
        gate,
        training_run_nonce=training_run_nonce,
    )
    failed = [name for name in REQUIRED_GATE_FLAGS if gate.get(name) is not True]
    if failed:
        raise ContactEgoStage1PreflightError(f"training gates not explicitly true: {failed}")
    gate_segment_count = gate.get("independent_segment_count")
    if (
        isinstance(gate_segment_count, bool)
        or not isinstance(gate_segment_count, int)
        or gate_segment_count != expected["independent_segment_count"]
    ):
        raise ContactEgoStage1PreflightError("gate segment count differs")
    fingerprints = gate.get("artifact_fingerprints")
    expected_fingerprint_keys = set(expected) - {"independent_segment_count"}
    if (
        not isinstance(fingerprints, Mapping)
        or set(fingerprints) != expected_fingerprint_keys
    ):
        raise ContactEgoStage1PreflightError("gate lacks artifact fingerprints")
    for key, value in expected.items():
        if key == "independent_segment_count":
            continue
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value) or fingerprints.get(key) != value:
            raise ContactEgoStage1PreflightError(f"gate fingerprint differs: {key}")
    if not isinstance(gate.get("reviewed_by"), str) or not gate["reviewed_by"]:
        raise ContactEgoStage1PreflightError("gate reviewer missing")
    if not isinstance(gate.get("reviewed_at_utc"), str) or not gate["reviewed_at_utc"].endswith("Z"):
        raise ContactEgoStage1PreflightError("gate review timestamp missing")
    matched_control_expected = _validate_matched_control_expectations(gate)
    entries = gate.get("evidence_reports")
    if not isinstance(entries, Mapping):
        raise ContactEgoStage1PreflightError("gate evidence reports missing")
    matched_entry = entries.get("matched_control_hash_collector")
    if (
        not isinstance(matched_entry, Mapping)
        or matched_entry.get("sha256")
        != matched_control_expected["expected_report_sha256"]
    ):
        raise ContactEgoStage1PreflightError(
            "matched-control report SHA is not independently bound by the gate"
        )
    reports = {name: _resolve_report(gate_path, entries.get(name), name) for name in (
        "annotation_agreement", "ego_action_grip", "matched_control_hash_collector"
    )}
    annotation_path, annotation, annotation_digest = reports["annotation_agreement"]
    metrics = annotation.get("metrics")
    if (
        annotation.get("report_version") != "contact-annotation-agreement/v1"
        or annotation.get("status") != "passed"
        or annotation.get("independent_annotation") is not True
        or annotation.get("adjudication_completed") is not True
        or annotation.get("dual_annotation_agreement_passed") is not True
        or isinstance(annotation.get("independent_segment_count"), bool)
        or not isinstance(annotation.get("independent_segment_count"), int)
        or annotation.get("independent_segment_count") != expected["independent_segment_count"]
        or not _validate_agreement_metrics(metrics)
    ):
        raise ContactEgoStage1PreflightError("A/B agreement evidence failed")
    sources = annotation.get("source_files")
    required_sources = {f"{kind}_{role}" for kind in ("coverage", "segments") for role in ("annotator_a", "annotator_b", "adjudicated")}
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise ContactEgoStage1PreflightError("annotation evidence sources incomplete")
    source_hashes, source_rows = {}, {}
    for name, entry in sources.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or not isinstance(entry.get("sha256"), str)
            or not _SHA256_RE.fullmatch(entry["sha256"])
        ):
            raise ContactEgoStage1PreflightError("invalid annotation evidence source")
        path = Path(entry["path"])
        if not path.is_absolute(): path = annotation_path.parent / path
        path = Path(os.path.abspath(path))
        fields = COVERAGE_FIELDS if name.startswith("coverage_") else SEGMENT_FIELDS
        try:
            payload, digest, _ = read_verified_bytes(
                path, expected_sha256=entry["sha256"]
            )
            rows = read_csv_exact(path, fields, payload=payload)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ContactEgoStage1PreflightError(
                "annotation source hash/content differs"
            ) from exc
        source_hashes[name] = digest
        source_rows[name] = rows
    if source_hashes["coverage_adjudicated"] != expected["coverage_csv_sha256"] or source_hashes["segments_adjudicated"] != expected["segments_csv_sha256"]:
        raise ContactEgoStage1PreflightError("agreement report binds different adjudicated CSVs")
    if len(set(source_hashes.values())) != len(source_hashes):
        raise ContactEgoStage1PreflightError("A/B/adjudicated source files are not independent")
    annotators = {}
    for suffix, role in (("annotator_a", "independent_a"), ("annotator_b", "independent_b")):
        coverage_rows = source_rows[f"coverage_{suffix}"]
        rows = coverage_rows + source_rows[f"segments_{suffix}"]
        if not rows or any(
            row["annotation_role"] != role or row["label_origin"] != "human_independent"
            or row["review_status"] != "submitted" or row["schema_version"] != "contact_segment_annotation_v1_1"
            or row["dataset_id"] != dataset_id or not row["annotator_id"] for row in rows
        ) or any(row["frame_clock"] != EGO_STAGE1_FRAME_CLOCK for row in coverage_rows):
            raise ContactEgoStage1PreflightError(f"{suffix} annotation contract differs")
        annotators[suffix] = {row["annotator_id"] for row in rows}
    if annotators["annotator_a"] & annotators["annotator_b"]:
        raise ContactEgoStage1PreflightError("A/B annotator identities overlap")
    _, grip, grip_digest = reports["ego_action_grip"]
    source_evidence = grip.get("source_evidence")
    source_evidence_valid = (
        isinstance(source_evidence, Mapping)
        and set(source_evidence) == {"path", "sha256"}
        and isinstance(source_evidence.get("path"), str)
        and bool(source_evidence.get("path"))
        and isinstance(source_evidence.get("sha256"), str)
        and bool(_SHA256_RE.fullmatch(source_evidence["sha256"]))
    )
    if source_evidence_valid:
        evidence_path = Path(source_evidence["path"])
        if not evidence_path.is_absolute():
            evidence_path = reports["ego_action_grip"][0].parent / evidence_path
        try:
            _, evidence_digest, _ = read_verified_bytes(
                Path(os.path.abspath(evidence_path)),
                expected_sha256=source_evidence["sha256"],
            )
            source_evidence_valid = evidence_digest == source_evidence["sha256"]
        except (OSError, ValueError):
            source_evidence_valid = False
    if (
        grip.get("report_version") != "ego-action-grip-evidence/v1"
        or grip.get("status") != "passed" or grip.get("source_verified") is not True
        or grip.get("transform_conversion_verified") is not True
        or grip.get("dataset_id") != dataset_id
        or grip.get("dataset_fingerprint_sha256") != expected["dataset_fingerprint_sha256"]
        or grip.get("source_receipt_sha256") != expected["source_receipt_sha256"]
        or grip.get("norm_stats_sha256") != expected["norm_stats_sha256"]
        or grip.get("hdf5_bundle_sha256") != expected["hdf5_bundle_sha256"]
        or grip.get("source_hand_status_semantics") != EGO_HAND_STATUS_SEMANTICS
        or grip.get("target_hand_status_semantics") != CANONICAL_OPEN_FRACTION_SEMANTICS
        or grip.get("source_action_dim") != 20 or grip.get("model_action_dim") != 32
        or source_evidence_valid is not True
    ):
        raise ContactEgoStage1PreflightError("Ego action/grip evidence failed")
    collector_path, collector, collector_digest = reports[
        "matched_control_hash_collector"
    ]
    if collector_digest != matched_control_expected["expected_report_sha256"]:
        raise ContactEgoStage1PreflightError(
            "matched-control report bytes differ from expectation"
        )
    required_hashes = {"ordered_sample_ids", "raw_batch", "augmentation_parameters", "visual_action_timesteps", "visual_noise", "action_noise"}
    code = collector.get("collector_code")
    valid_code = isinstance(code, Mapping) and isinstance(code.get("path"), str)
    if valid_code:
        path = Path(code["path"])
        if not path.is_absolute(): path = reports["matched_control_hash_collector"][0].parent / path
        try:
            valid_code = (
                code.get("version") == matched_control_expected["expected_collector_version"]
                and code.get("sha256")
                == matched_control_expected["expected_collector_source_sha256"]
                and code.get("sha256")
                == sha256_file(Path(os.path.abspath(path)))
            )
        except OSError:
            valid_code = False
    expected_variants = matched_control_expected["expected_variant_config_sha256"]
    expected_trace_files = matched_control_expected["expected_trace_files"]
    if collector.get("trace_files") != expected_trace_files:
        raise ContactEgoStage1PreflightError(
            "matched-control trace manifest differs from external expectations"
        )
    for entries_for_variant in expected_trace_files.values():
        for entry in entries_for_variant:
            trace_path = Path(entry["path"])
            if not trace_path.is_absolute():
                trace_path = collector_path.parent / trace_path
            trace_path = Path(os.path.abspath(trace_path))
            if sha256_file(trace_path) != entry["sha256"]:
                raise ContactEgoStage1PreflightError(
                    f"matched-control trace is missing or changed: {trace_path}"
                )
    expected_topology = {
        "world_size": matched_control_expected["expected_world_size"],
        "rank_set": matched_control_expected["expected_rank_set"],
    }
    global_hashes = collector.get("global_hashes")
    report_world_size = collector.get("world_size")
    report_rank_set = collector.get("rank_set")
    report_topology = collector.get("topology")
    valid_report_topology = (
        not isinstance(report_world_size, bool)
        and isinstance(report_world_size, int)
        and isinstance(report_rank_set, list)
        and all(
            not isinstance(rank, bool) and isinstance(rank, int)
            for rank in report_rank_set
        )
        and isinstance(report_topology, Mapping)
        and set(report_topology) == {"world_size", "rank_set"}
        and not isinstance(report_topology.get("world_size"), bool)
        and isinstance(report_topology.get("world_size"), int)
        and isinstance(report_topology.get("rank_set"), list)
        and all(
            not isinstance(rank, bool) and isinstance(rank, int)
            for rank in report_topology.get("rank_set", [])
        )
    )
    valid_global_hashes = isinstance(global_hashes, Mapping) and bool(global_hashes)
    if valid_global_hashes:
        for logical_name, variants in global_hashes.items():
            if (
                not isinstance(logical_name, str)
                or not isinstance(variants, Mapping)
                or set(variants) != set(IMPLEMENTED_RUNNER_VARIANTS)
            ):
                valid_global_hashes = False
                break
            for hashes in variants.values():
                if (
                    not isinstance(hashes, Mapping)
                    or set(hashes) != required_hashes
                    or any(
                        not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
                        for value in hashes.values()
                    )
                ):
                    valid_global_hashes = False
                    break
            if not valid_global_hashes:
                break
    if (
        collector.get("report_version")
        != matched_control_expected["expected_collector_version"]
        or collector.get("status") != "passed" or collector.get("collector_implemented") is not True
        or set(collector.get("captured_hashes", [])) != required_hashes
        or collector.get("dry_run_passed") is not True
        or collector.get("all_hashes_equal") is not True
        or collector.get("all_hashes_equal_across_variants") is not True
        or collector.get("seeds") != [6666, 7777, 8888]
        or collector.get("run_id") != matched_control_expected["expected_run_id"]
        or collector.get("run_nonce")
        != matched_control_expected["expected_run_nonce"]
        or collector.get("dataset_id") != dataset_id
        or collector.get("dataset_ids") != [dataset_id]
        or collector.get("git_commit")
        != matched_control_expected["expected_git_commit"]
        or collector.get("control_config_sha256")
        != matched_control_expected["expected_control_config_sha256"]
        or collector.get("variant_config_sha256") != expected_variants
        or collector.get("variants") != sorted(IMPLEMENTED_RUNNER_VARIANTS)
        or not valid_report_topology
        or collector.get("world_size")
        != matched_control_expected["expected_world_size"]
        or collector.get("rank_set")
        != matched_control_expected["expected_rank_set"]
        or collector.get("topology") != expected_topology
        or collector.get("trace_manifest_sha256")
        != matched_control_expected["expected_trace_manifest_sha256"]
        or collector.get("global_hashes_sha256")
        != matched_control_expected["expected_global_hashes_sha256"]
        or collector.get("hash_schema_sha256")
        != matched_control_expected["expected_hash_schema_sha256"]
        or collector.get("hash_schema") != HASH_SCHEMA
        or not valid_global_hashes
        or canonical_sha256(global_hashes)
        != matched_control_expected["expected_global_hashes_sha256"]
        or collector.get("matched_batch_count") != len(global_hashes or {})
        or not isinstance(collector.get("matched_batch_count"), int)
        or isinstance(collector.get("matched_batch_count"), bool)
        or collector.get("matched_batch_count", 0) <= 0 or not valid_code
    ):
        raise ContactEgoStage1PreflightError("matched-control collector evidence failed")
    return {
        "annotation_agreement": annotation_digest,
        "ego_action_grip": grip_digest,
        "matched_control_hash_collector": collector_digest,
        "matched_control_expectations": matched_control_expected,
    }


def validate_ego_stage1_training_preflight(
    *, data_root: str | Path, norm_stats_path: str | Path, hdf5_dir: str | Path,
    t5_embedding_dir: str | Path,
    training_run_nonce: str,
    sidecar_dir: str | Path,
    coverage_csv: str | Path, segments_csv: str | Path, inventory_csv: str | Path,
    build_config_path: str | Path, split_manifest: str | Path, training_gate_path: str | Path,
) -> dict:
    gate_path = Path(os.path.abspath(os.path.expanduser(str(training_gate_path))))
    gate = _load_json(gate_path, "Ego Stage1 training gate")
    if gate.get("gate_version") != EGO_STAGE1_GATE_VERSION:
        raise ContactEgoStage1PreflightError("training gate version differs")
    _validate_gate_freshness_and_training_nonce(
        gate, training_run_nonce=training_run_nonce
    )
    if os.environ.get("GWP_MATCHED_CONTROL_OUTPUT_DIR", "").strip():
        _validate_loaded_gate_runtime_topology(gate)

    identity = derive_ego_dataset_identity(data_root)
    t5_source = derive_ego_t5_source_receipt(
        t5_embedding_dir,
        episode_indices=list(identity["episode_lengths"]),
    )
    norm_path = Path(os.path.abspath(os.path.expanduser(str(norm_stats_path))))
    try:
        norm_bytes, norm_digest, norm_runtime_stat = read_verified_bytes(norm_path)
        norm = strict_json_loads(norm_bytes, name="Ego Stage1 norm")
    except (OSError, ValueError) as exc:
        raise ContactEgoStage1PreflightError("invalid or unsafe Ego Stage1 norm") from exc
    if not isinstance(norm, dict):
        raise ContactEgoStage1PreflightError("Ego Stage1 norm must be a JSON object")
    hdf5_bundle_sha256 = _validate_norm(norm, identity, Path(hdf5_dir).resolve())
    sidecar = validate_ego_reviewed_sidecar_bundle(
        sidecar_dir=sidecar_dir, coverage_csv=coverage_csv, segments_csv=segments_csv,
        inventory_csv=inventory_csv, build_config_path=build_config_path, split_manifest=split_manifest,
    )
    if sidecar["dataset_id"] != identity["dataset_id"]:
        raise ContactEgoStage1PreflightError("sidecar dataset_id differs from derived dataset fingerprint")
    if sidecar["episode_lengths"] != identity["episode_lengths"]:
        raise ContactEgoStage1PreflightError("sidecar episode IDs/lengths differ from Ego dataset")
    if sidecar["meta"].get("frame_clock") != identity["frame_clock"] or float(sidecar["meta"].get("target_fps", -1)) != identity["fps"]:
        raise ContactEgoStage1PreflightError("sidecar fps/frame clock differs from Ego dataset contract")
    expected = {
        "independent_segment_count": sidecar["independent_segment_count"],
        "dataset_fingerprint_sha256": identity["dataset_fingerprint_sha256"],
        "info_identity_sha256": identity["info_identity_sha256"],
        "episode_metadata_sha256": identity["episode_metadata_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "source_receipt_sha256": identity["source_receipt_sha256"],
        "t5_source_manifest_sha256": t5_source["source_manifest_sha256"],
        "t5_source_receipt_sha256": t5_source["source_receipt_sha256"],
        "norm_stats_sha256": norm_digest,
        "hdf5_bundle_sha256": hdf5_bundle_sha256,
        "sidecar_meta_sha256": sidecar["meta_sha256"],
        "sidecar_bundle_sha256": sidecar["bundle_sha256"],
        "split_manifest_sha256": sidecar["split_sha256"],
        "coverage_csv_sha256": sidecar["source_sha256"]["coverage_csv"],
        "segments_csv_sha256": sidecar["source_sha256"]["segments_csv"],
        "inventory_csv_sha256": sidecar["source_sha256"]["inventory_csv"],
        "build_config_sha256": sidecar["source_sha256"]["config"],
    }
    evidence = _validate_gate(
        gate,
        expected,
        gate_path,
        identity["dataset_id"],
        training_run_nonce=training_run_nonce,
    )
    return {
        "status": "artifact_consistency_passed_with_explicit_external_gates",
        "scientific_gate_claimed": False,
        "action_schema_id": EGO_CANONICAL_ACTION_SCHEMA_ID,
        "source_action_dim": 20, "model_action_dim": 32,
        "contact_schema_version": SEGMENT_SCHEMA_VERSION,
        "contact_target_dim": SEGMENT_TARGET_DIM, "num_contact_tokens": 12,
        "transport_mode": "hard_st", "time_bins": 16,
        "training_run_nonce": training_run_nonce,
        "evidence_valid_until_utc": gate["evidence_valid_until_utc"],
        "data_root": identity["data_root"], "norm_stats_path": str(norm_path),
        "norm_runtime_stat_record_only": norm_runtime_stat,
        "t5_embedding_dir": t5_source["source_dir_record_only"],
        "training_source_runtime_receipt_record_only": identity[
            "runtime_source_receipt_record_only"
        ],
        "t5_source_manifest": t5_source["source_manifest"],
        "t5_runtime_source_receipt_record_only": t5_source[
            "runtime_source_receipt_record_only"
        ],
        "sidecar_dir": str(Path(sidecar_dir).resolve()),
        "dataset_id": identity["dataset_id"], "episode_npz_sha256": sidecar["episode_npz_sha256"],
        "evidence_report_sha256": {
            key: value for key, value in evidence.items() if key != "matched_control_expectations"
        },
        "matched_control_expectations": evidence["matched_control_expectations"],
        **expected,
    }


def preflight_from_environment() -> dict:
    missing = [name for name in REQUIRED_ENV_PATHS if not os.environ.get(name, "").strip()]
    if missing:
        raise ContactEgoStage1PreflightError(f"missing Ego Stage1 preflight paths: {missing}")
    return validate_ego_stage1_training_preflight(
        data_root=os.environ["GWP_EGO_DATA_PATH"], norm_stats_path=os.environ["GWP_EGO_NORM_STATS_PATH"],
        hdf5_dir=os.environ["GWP_EGO_FINAL_HDF5_DIR"],
        training_run_nonce=os.environ.get("GWP_EGO_TRAINING_RUN_NONCE", ""),
        t5_embedding_dir=(
            os.environ.get("GWP_T5_EMBEDDING_DIR", "").strip()
            or str(Path(os.environ["GWP_EGO_DATA_PATH"]) / "t5_embedding")
        ),
        sidecar_dir=os.environ["GWP_CONTACT_SIDECAR_DIR"], coverage_csv=os.environ["GWP_CONTACT_COVERAGE_CSV"],
        segments_csv=os.environ["GWP_CONTACT_SEGMENTS_CSV"], inventory_csv=os.environ["GWP_CONTACT_INVENTORY_CSV"],
        build_config_path=os.environ["GWP_CONTACT_BUILD_CONFIG"], split_manifest=os.environ["GWP_CONTACT_SPLIT_MANIFEST"],
        training_gate_path=os.environ["GWP_EGO_STAGE1_TRAINING_GATE"],
    )

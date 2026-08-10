"""Fail-closed production gate for discrete-contact Ego Stage1 pretraining."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
import os
from pathlib import Path
import re

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
)


EGO_STAGE1_GATE_VERSION = "ego-discrete-contact-stage1-training-gate/v1"
REQUIRED_GATE_FLAGS = (
    "human_adjudicated_sidecar_passed",
    "dual_annotation_agreement_passed",
    "group_metadata_complete",
    "strategy_claim_allowed",
    "ego_action_grip_evidence_passed",
    "matched_control_hash_collector_passed",
    "allow_server_training",
    "allow_5b_training",
)
REQUIRED_ENV_PATHS = (
    "GWP_EGO_DATA_PATH",
    "GWP_EGO_NORM_STATS_PATH",
    "GWP_EGO_FINAL_HDF5_DIR",
    "GWP_CONTACT_SIDECAR_DIR",
    "GWP_CONTACT_COVERAGE_CSV",
    "GWP_CONTACT_SEGMENTS_CSV",
    "GWP_CONTACT_INVENTORY_CSV",
    "GWP_CONTACT_BUILD_CONFIG",
    "GWP_CONTACT_SPLIT_MANIFEST",
    "GWP_EGO_STAGE1_TRAINING_GATE",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVIEWED_BUILDER_ID = "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py"
REQUIRED_SIDECAR_KEYS = (
    "contact_target", "contact_loss_mask", "contact_confidence", "contact_condition_mask"
)


class ContactEgoStage1PreflightError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ContactEgoStage1PreflightError(f"missing {name}: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContactEgoStage1PreflightError(f"{name} must be a JSON object")
    return payload


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
    if int(meta.get("reviewed_coverage_rows", 0)) <= 0:
        raise ContactEgoStage1PreflightError("sidecar has no reviewed coverage")
    if int(meta.get("unique_physical_segments", 0)) <= 0:
        raise ContactEgoStage1PreflightError("sidecar has no physical segments")
    expected_slices = {key: [value.start, value.stop] for key, value in SEGMENT_FIELD_SLICES.items()}
    if meta.get("field_slices") != expected_slices:
        raise ContactEgoStage1PreflightError("sidecar field_slices differ from v1.1")


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
    path: Path,
    inventory: Mapping[int, object],
    inventory_rows: Mapping[int, Mapping[str, str]],
    split_policy: Mapping,
) -> None:
    expected_header = [
        "episode_id", "length_frames", "seed_6666_split", "seed_7777_split",
        "seed_8888_split", "split_unit", "split_purpose", "ood_eligible",
        "strategy_claim_allowed", "metadata_status",
    ]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ContactEgoStage1PreflightError("split manifest header mismatch")
        rows = list(reader)
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
    sidecar_dir = Path(sidecar_dir).resolve()
    coverage_csv, segments_csv, inventory_csv = map(
        lambda value: Path(value).resolve(), (coverage_csv, segments_csv, inventory_csv)
    )
    build_config_path, split_manifest = Path(build_config_path).resolve(), Path(split_manifest).resolve()
    for path in (sidecar_dir, coverage_csv, segments_csv, inventory_csv, build_config_path, split_manifest):
        if not path.exists():
            raise ContactEgoStage1PreflightError(f"missing sidecar artifact: {path}")
    meta_path = sidecar_dir / "meta.json"
    meta = _load_json(meta_path, "sidecar meta")
    _validate_reviewed_meta(meta)
    config = _load_json(build_config_path, "sidecar build config")
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
    inventory = load_inventory(inventory_csv)
    inventory_rows = _validate_group_metadata(inventory_csv)
    _validate_split_manifest(split_manifest, inventory, inventory_rows, split_policy)
    coverages = load_coverages(coverage_csv, inventory, config)
    segments = load_segments(segments_csv, inventory, coverages, config)
    if len(segments) != int(meta["unique_physical_segments"]):
        raise ContactEgoStage1PreflightError("segment count differs from meta")
    source_paths = {
        "coverage_csv": coverage_csv, "segments_csv": segments_csv,
        "inventory_csv": inventory_csv, "config": build_config_path,
        "split_manifest": split_manifest,
    }
    hashes = meta.get("input_sha256")
    if not isinstance(hashes, Mapping) or any(hashes.get(k) != sha256_file(v) for k, v in source_paths.items()):
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
    episode_hashes = {name: sha256_file(sidecar_dir / name) for name in sorted(expected_files)}
    if meta.get("episode_npz_sha256") != episode_hashes:
        raise ContactEgoStage1PreflightError("sidecar NPZ hash differs")
    provenance = []
    for episode, item in sorted(inventory.items()):
        expected_arrays, rows = encode_episode(
            item.length_frames, by_coverage[episode], by_segment[episode],
            horizon_frames=48, num_queries=12,
            negative_confidence=float(config["negative_confidence"]),
        )
        with np.load(sidecar_dir / f"episode_{episode:06d}.npz", allow_pickle=False) as archive:
            actual = {key: archive[key] for key in archive.files}
        validate_contact_sidecar_arrays(actual, frame_count=item.length_frames)
        if any(not np.array_equal(actual[key], expected_arrays[key]) for key in REQUIRED_SIDECAR_KEYS):
            raise ContactEgoStage1PreflightError("sidecar differs from full reviewed re-encoding")
        provenance.extend({"episode_id": str(episode), **{k: str(v) for k, v in row.items()}} for row in rows)
    provenance_path = sidecar_dir / "window_segment_provenance.csv"
    with provenance_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["episode_id", "frame_index", "query_index", "segment_id", "onset_event_id", "release_event_id"]:
            raise ContactEgoStage1PreflightError("provenance header mismatch")
        actual_provenance = list(reader)
    if actual_provenance != provenance:
        raise ContactEgoStage1PreflightError("window provenance differs from physical segment assignment")
    provenance_digest = sha256_file(provenance_path)
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
        "episode_npz_sha256": episode_hashes, "meta_sha256": sha256_file(meta_path),
        "bundle_sha256": bundle_digest, "split_sha256": sha256_file(split_manifest), "meta": meta,
    }


def _validate_norm(payload: Mapping, identity: Mapping, hdf5_dir: Path) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContactEgoStage1PreflightError("norm lacks metadata")
    expected = {
        "norm_contract_version": "ego-stage1-norm/v1",
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
            if not isinstance(values, list) or len(values) != 32 or not np.isfinite(values).all():
                raise ContactEgoStage1PreflightError("norm arrays must be finite 32-D")
            if any(float(value) != 0.0 for value in values[20:]):
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
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not entry["path"]:
        raise ContactEgoStage1PreflightError(f"missing evidence report: {name}")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = gate_path.parent / path
    path = path.resolve()
    digest = sha256_file(path)
    if entry.get("sha256") != digest:
        raise ContactEgoStage1PreflightError(f"evidence report hash differs: {name}")
    return path, _load_json(path, name), digest


def _validate_gate(gate: Mapping, expected: Mapping, gate_path: Path, dataset_id: str) -> dict:
    if gate.get("gate_version") != EGO_STAGE1_GATE_VERSION:
        raise ContactEgoStage1PreflightError("training gate version differs")
    failed = [name for name in REQUIRED_GATE_FLAGS if gate.get(name) is not True]
    if failed:
        raise ContactEgoStage1PreflightError(f"training gates not explicitly true: {failed}")
    if gate.get("independent_segment_count") != expected["independent_segment_count"]:
        raise ContactEgoStage1PreflightError("gate segment count differs")
    fingerprints = gate.get("artifact_fingerprints")
    if not isinstance(fingerprints, Mapping):
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
    entries = gate.get("evidence_reports")
    if not isinstance(entries, Mapping):
        raise ContactEgoStage1PreflightError("gate evidence reports missing")
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
        or annotation.get("independent_segment_count") != expected["independent_segment_count"]
        or not isinstance(metrics, Mapping)
        or float(metrics.get("coverage_kappa", -1)) < .70
        or float(metrics.get("event_f1", -1)) < .80
        or float(metrics.get("median_temporal_iou", -1)) < .70
        or float(metrics.get("boundary_mae_frames", 1e9)) > 2
        or float(metrics.get("categorical_kappa", -1)) < .60
    ):
        raise ContactEgoStage1PreflightError("A/B agreement evidence failed")
    sources = annotation.get("source_files")
    required_sources = {f"{kind}_{role}" for kind in ("coverage", "segments") for role in ("annotator_a", "annotator_b", "adjudicated")}
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise ContactEgoStage1PreflightError("annotation evidence sources incomplete")
    source_paths, source_hashes = {}, {}
    for name, entry in sources.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ContactEgoStage1PreflightError("invalid annotation evidence source")
        path = Path(entry["path"])
        if not path.is_absolute(): path = annotation_path.parent / path
        path = path.resolve(); digest = sha256_file(path)
        if entry.get("sha256") != digest: raise ContactEgoStage1PreflightError("annotation source hash differs")
        source_paths[name], source_hashes[name] = path, digest
    if source_hashes["coverage_adjudicated"] != expected["coverage_csv_sha256"] or source_hashes["segments_adjudicated"] != expected["segments_csv_sha256"]:
        raise ContactEgoStage1PreflightError("agreement report binds different adjudicated CSVs")
    if len(set(source_hashes.values())) != len(source_hashes):
        raise ContactEgoStage1PreflightError("A/B/adjudicated source files are not independent")
    annotators = {}
    for suffix, role in (("annotator_a", "independent_a"), ("annotator_b", "independent_b")):
        coverage_rows = read_csv_exact(source_paths[f"coverage_{suffix}"], COVERAGE_FIELDS)
        rows = coverage_rows + read_csv_exact(source_paths[f"segments_{suffix}"], SEGMENT_FIELDS)
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
    source_evidence_valid = isinstance(source_evidence, Mapping) and isinstance(
        source_evidence.get("path"), str
    ) and bool(source_evidence.get("path"))
    if source_evidence_valid:
        evidence_path = Path(source_evidence["path"])
        if not evidence_path.is_absolute():
            evidence_path = reports["ego_action_grip"][0].parent / evidence_path
        source_evidence_valid = source_evidence.get("sha256") == sha256_file(
            evidence_path.resolve()
        )
    if (
        grip.get("report_version") != "ego-action-grip-evidence/v1"
        or grip.get("status") != "passed" or grip.get("source_verified") is not True
        or grip.get("transform_conversion_verified") is not True
        or grip.get("dataset_id") != dataset_id
        or grip.get("dataset_fingerprint_sha256") != expected["dataset_fingerprint_sha256"]
        or grip.get("norm_stats_sha256") != expected["norm_stats_sha256"]
        or grip.get("hdf5_bundle_sha256") != expected["hdf5_bundle_sha256"]
        or grip.get("source_hand_status_semantics") != EGO_HAND_STATUS_SEMANTICS
        or grip.get("target_hand_status_semantics") != CANONICAL_OPEN_FRACTION_SEMANTICS
        or grip.get("source_action_dim") != 20 or grip.get("model_action_dim") != 32
        or source_evidence_valid is not True
    ):
        raise ContactEgoStage1PreflightError("Ego action/grip evidence failed")
    _, collector, collector_digest = reports["matched_control_hash_collector"]
    required_hashes = {"ordered_sample_ids", "raw_batch", "augmentation_parameters", "visual_action_timesteps", "visual_noise", "action_noise"}
    code = collector.get("collector_code")
    valid_code = isinstance(code, Mapping) and isinstance(code.get("path"), str)
    if valid_code:
        path = Path(code["path"])
        if not path.is_absolute(): path = reports["matched_control_hash_collector"][0].parent / path
        valid_code = bool(code.get("version")) and code.get("sha256") == sha256_file(path.resolve())
    if (
        collector.get("report_version") != "matched-control-hash-collector/v1"
        or collector.get("status") != "passed" or collector.get("collector_implemented") is not True
        or set(collector.get("captured_hashes", [])) != required_hashes
        or collector.get("dry_run_passed") is not True
        or collector.get("all_hashes_equal_across_variants") is not True
        or not isinstance(collector.get("matched_batch_count"), int)
        or isinstance(collector.get("matched_batch_count"), bool)
        or collector.get("matched_batch_count", 0) <= 0 or not valid_code
    ):
        raise ContactEgoStage1PreflightError("matched-control collector evidence failed")
    return {"annotation_agreement": annotation_digest, "ego_action_grip": grip_digest, "matched_control_hash_collector": collector_digest}


def validate_ego_stage1_training_preflight(
    *, data_root: str | Path, norm_stats_path: str | Path, hdf5_dir: str | Path,
    sidecar_dir: str | Path,
    coverage_csv: str | Path, segments_csv: str | Path, inventory_csv: str | Path,
    build_config_path: str | Path, split_manifest: str | Path, training_gate_path: str | Path,
) -> dict:
    identity = derive_ego_dataset_identity(data_root)
    norm_path = Path(norm_stats_path).resolve()
    norm = _load_json(norm_path, "Ego Stage1 norm")
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
        "norm_stats_sha256": sha256_file(norm_path),
        "hdf5_bundle_sha256": hdf5_bundle_sha256,
        "sidecar_meta_sha256": sidecar["meta_sha256"],
        "sidecar_bundle_sha256": sidecar["bundle_sha256"],
        "split_manifest_sha256": sidecar["split_sha256"],
        "coverage_csv_sha256": sha256_file(coverage_csv),
        "segments_csv_sha256": sha256_file(segments_csv),
        "inventory_csv_sha256": sha256_file(inventory_csv),
        "build_config_sha256": sha256_file(build_config_path),
    }
    gate_path = Path(training_gate_path).resolve()
    evidence = _validate_gate(_load_json(gate_path, "Ego Stage1 training gate"), expected, gate_path, identity["dataset_id"])
    return {
        "status": "artifact_consistency_passed_with_explicit_external_gates",
        "scientific_gate_claimed": False,
        "action_schema_id": EGO_CANONICAL_ACTION_SCHEMA_ID,
        "source_action_dim": 20, "model_action_dim": 32,
        "contact_schema_version": SEGMENT_SCHEMA_VERSION,
        "contact_target_dim": SEGMENT_TARGET_DIM, "num_contact_tokens": 12,
        "transport_mode": "hard_st", "time_bins": 16,
        "data_root": identity["data_root"], "norm_stats_path": str(norm_path),
        "sidecar_dir": str(Path(sidecar_dir).resolve()),
        "dataset_id": identity["dataset_id"], "episode_npz_sha256": sidecar["episode_npz_sha256"],
        "evidence_report_sha256": evidence, **expected,
    }


def preflight_from_environment() -> dict:
    missing = [name for name in REQUIRED_ENV_PATHS if not os.environ.get(name, "").strip()]
    if missing:
        raise ContactEgoStage1PreflightError(f"missing Ego Stage1 preflight paths: {missing}")
    return validate_ego_stage1_training_preflight(
        data_root=os.environ["GWP_EGO_DATA_PATH"], norm_stats_path=os.environ["GWP_EGO_NORM_STATS_PATH"],
        hdf5_dir=os.environ["GWP_EGO_FINAL_HDF5_DIR"],
        sidecar_dir=os.environ["GWP_CONTACT_SIDECAR_DIR"], coverage_csv=os.environ["GWP_CONTACT_COVERAGE_CSV"],
        segments_csv=os.environ["GWP_CONTACT_SEGMENTS_CSV"], inventory_csv=os.environ["GWP_CONTACT_INVENTORY_CSV"],
        build_config_path=os.environ["GWP_CONTACT_BUILD_CONFIG"], split_manifest=os.environ["GWP_CONTACT_SPLIT_MANIFEST"],
        training_gate_path=os.environ["GWP_EGO_STAGE1_TRAINING_GATE"],
    )

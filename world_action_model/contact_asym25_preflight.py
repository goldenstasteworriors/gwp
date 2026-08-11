"""Fail-closed integration gate for asymmetric25 contact finetuning.

This module validates artifacts; it does not infer scientific readiness.  A
training-gate file records the independently reviewed decisions that cannot be
derived from tensors alone, while every artifact fingerprint is recomputed.
"""

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
from world_action_model.asymmetric_action_schema import (
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


TRAINING_GATE_VERSION = "contact-asym25-training-gate/v1"
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
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContactAsym25PreflightError(ValueError):
    """Raised when any production integration requirement is missing."""


def sha256_file(path: str | Path) -> str:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, name: str) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ContactAsym25PreflightError(f"missing {name}: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContactAsym25PreflightError(f"{name} must contain a JSON object")
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


def _validate_training_gate(
    gate: Mapping,
    expected: Mapping[str, object],
    *,
    gate_path: Path,
    dataset_id: str,
) -> dict[str, str]:
    if gate.get("gate_version") != TRAINING_GATE_VERSION:
        raise ContactAsym25PreflightError("training gate version mismatch")
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
    if not isinstance(fingerprints, Mapping):
        raise ContactAsym25PreflightError("training gate is missing artifact_fingerprints")
    for key, value in expected.items():
        if key == "independent_segment_count":
            continue
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ContactAsym25PreflightError(f"internal invalid expected fingerprint {key}")
        if fingerprints.get(key) != value:
            raise ContactAsym25PreflightError(f"training gate fingerprint differs: {key}")
    if not isinstance(gate.get("reviewed_by"), str) or not gate["reviewed_by"].strip():
        raise ContactAsym25PreflightError("training gate reviewed_by is required")
    if not isinstance(gate.get("reviewed_at_utc"), str) or not gate["reviewed_at_utc"].endswith("Z"):
        raise ContactAsym25PreflightError("training gate reviewed_at_utc must be UTC")
    entries = gate.get("evidence_reports")
    if not isinstance(entries, Mapping):
        raise ContactAsym25PreflightError("training gate is missing evidence_reports")
    reports = {}
    report_paths = {}
    digests = {}
    for name in (
        "annotation_agreement",
        "hardware_verification",
        "matched_control_hash_collector",
    ):
        entry = entries.get(name)
        if not isinstance(entry, Mapping):
            raise ContactAsym25PreflightError(f"missing evidence report entry: {name}")
        report_path_value = entry.get("path")
        declared_digest = entry.get("sha256")
        if not isinstance(report_path_value, str) or not report_path_value.strip():
            raise ContactAsym25PreflightError(f"evidence report path is required: {name}")
        report_path = Path(report_path_value)
        if not report_path.is_absolute():
            report_path = gate_path.parent / report_path
        report_path = report_path.resolve()
        actual_digest = sha256_file(report_path)
        if declared_digest != actual_digest:
            raise ContactAsym25PreflightError(f"evidence report fingerprint differs: {name}")
        reports[name] = _load_json(report_path, f"{name} evidence report")
        report_paths[name] = report_path
        digests[name] = actual_digest

    annotation = reports["annotation_agreement"]
    metrics = annotation.get("metrics")
    if (
        annotation.get("report_version") != "contact-annotation-agreement/v1"
        or annotation.get("status") != "passed"
        or annotation.get("independent_annotation") is not True
        or annotation.get("adjudication_completed") is not True
        or annotation.get("dual_annotation_agreement_passed") is not True
        or annotation.get("independent_segment_count")
        != expected["independent_segment_count"]
        or not isinstance(metrics, Mapping)
        or float(metrics.get("coverage_kappa", -1.0)) < 0.70
        or float(metrics.get("event_f1", -1.0)) < 0.80
        or float(metrics.get("median_temporal_iou", -1.0)) < 0.70
        or float(metrics.get("boundary_mae_frames", 1e9)) > 2.0
        or float(metrics.get("categorical_kappa", -1.0)) < 0.60
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
    annotation_source_paths = {}
    for name, entry in annotation_sources.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ContactAsym25PreflightError(f"invalid annotation source entry: {name}")
        source_path = Path(entry["path"])
        if not source_path.is_absolute():
            source_path = report_paths["annotation_agreement"].parent / source_path
        actual = sha256_file(source_path.resolve())
        if entry.get("sha256") != actual:
            raise ContactAsym25PreflightError(f"annotation source fingerprint differs: {name}")
        annotation_source_digests[name] = actual
        annotation_source_paths[name] = source_path.resolve()
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
        rows = read_csv_exact(
            annotation_source_paths[f"coverage_{suffix}"], COVERAGE_FIELDS
        ) + read_csv_exact(
            annotation_source_paths[f"segments_{suffix}"], SEGMENT_FIELDS
        )
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
        coverage_rows = read_csv_exact(
            annotation_source_paths[f"coverage_{suffix}"], COVERAGE_FIELDS
        )
        if any(row["frame_clock"] != "dataset_native_frame_index" for row in coverage_rows):
            raise ContactAsym25PreflightError(f"{suffix} frame clock differs")
        annotator_sets[suffix] = {row["annotator_id"] for row in rows}
    if annotator_sets["annotator_a"] & annotator_sets["annotator_b"]:
        raise ContactAsym25PreflightError("A/B annotator identities overlap")
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
    required_hashes = {
        "ordered_sample_ids",
        "raw_batch",
        "augmentation_parameters",
        "visual_action_timesteps",
        "visual_noise",
        "action_noise",
    }
    collector_code = collector.get("collector_code")
    collector_code_valid = isinstance(collector_code, Mapping) and isinstance(
        collector_code.get("path"), str
    )
    if collector_code_valid:
        collector_path = Path(collector_code["path"])
        if not collector_path.is_absolute():
            collector_path = (
                report_paths["matched_control_hash_collector"].parent / collector_path
            )
        collector_code_valid = (
            isinstance(collector_code.get("version"), str)
            and bool(collector_code["version"].strip())
            and collector_code.get("sha256") == sha256_file(collector_path.resolve())
        )
    matched_batch_count = collector.get("matched_batch_count")
    if (
        collector.get("report_version") != "matched-control-hash-collector/v1"
        or collector.get("status") != "passed"
        or collector.get("collector_implemented") is not True
        or set(collector.get("captured_hashes", [])) != required_hashes
        or collector.get("dry_run_passed") is not True
        or collector.get("all_hashes_equal_across_variants") is not True
        or isinstance(matched_batch_count, bool)
        or not isinstance(matched_batch_count, int)
        or matched_batch_count <= 0
        or not collector_code_valid
    ):
        raise ContactAsym25PreflightError("matched-control hash collector evidence did not pass")
    return digests


def _dataset_episode_lengths(data_root: Path) -> tuple[dict[int, int], str]:
    episodes_path = data_root / "meta" / "episodes.jsonl"
    rows = []
    metadata_files = []
    if episodes_path.is_file():
        metadata_files = [episodes_path]
        rows = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        episodes_dir = data_root / "meta" / "episodes"
        parquet_paths = sorted(episodes_dir.glob("*/*.parquet")) + sorted(
            episodes_dir.glob("*.parquet")
        )
        if parquet_paths:
            import pandas as pd

            metadata_files = parquet_paths
            for path in parquet_paths:
                rows.extend(pd.read_parquet(path).to_dict(orient="records"))
    if not rows:
        raise ContactAsym25PreflightError(
            f"asymmetric25 dataset requires authoritative episode metadata under {data_root / 'meta'}"
        )
    result = {}
    for row_number, row in enumerate(rows, 1):
        try:
            episode = int(row["episode_index"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContactAsym25PreflightError(
                f"invalid dataset episode metadata row {row_number}"
            ) from exc
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
            "sha256": sha256_file(path),
        }
        for path in metadata_files
    ]
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result, digest


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

    norm_payload = _load_json(norm_stats_path, "normalization stats")
    validate_asymmetric_norm_payload(
        norm_payload,
        model_dim=MODEL_ACTION_DIM,
        expected_hardware_contract_sha256=contract_digest,
        expected_calibration_capture_sha256=capture_digest,
    )
    norm_metadata = norm_payload.get("metadata", {})
    norm_roots = norm_metadata.get("data_roots")
    if not isinstance(norm_roots, list) or {
        str(Path(value).resolve()) for value in norm_roots
    } != {str(data_root)}:
        raise ContactAsym25PreflightError("normalization stats were not computed from this data root")
    if norm_metadata.get("model_dim") != MODEL_ACTION_DIM:
        raise ContactAsym25PreflightError("normalization stats model_dim must be 32")
    if norm_metadata.get("delta_mask") != [False] * MODEL_ACTION_DIM:
        raise ContactAsym25PreflightError("asymmetric25 norm stats must use no second delta conversion")
    if norm_metadata.get("dataset_info_sha256") != dataset_info_digest:
        raise ContactAsym25PreflightError("norm stats dataset-info fingerprint differs")
    if norm_metadata.get("episode_metadata_sha256") != episode_metadata_digest:
        raise ContactAsym25PreflightError("norm stats episode-metadata fingerprint differs")
    if (
        int(norm_metadata.get("num_episodes", -1)) != len(dataset_episode_lengths)
        or int(norm_metadata.get("num_state_samples", -1)) != total_frames
        or int(norm_metadata.get("num_action_samples", -1))
        != total_frames * int(norm_metadata.get("action_horizon", -1))
        or int(norm_metadata.get("action_horizon", -1)) != 48
    ):
        raise ContactAsym25PreflightError("norm sample counts/horizon differ from dataset")

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
    gate_path = Path(training_gate_path).resolve()
    gate = _load_json(gate_path, "contact-asym25 training gate")
    evidence_digests = _validate_training_gate(
        gate,
        expected,
        gate_path=gate_path,
        dataset_id=dataset_id,
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
        "training_gate_sha256": sha256_file(gate_path),
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
        "transport_mode": "hard_st",
        "time_bins": 16,
        "split_seed": split_seed,
        "train_episode_ids": list(selected_split["train"]),
        "validation_episode_ids": list(selected_split["val"]),
        "heldout_episode_ids": list(selected_split["test"]),
        "training_data_receipt": training_data_receipt,
        "training_data_receipt_sha256": training_data_receipt_sha256,
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
        "training_gate_sha256": sha256_file(gate_path),
        "preflight_code_sha256": sha256_file(Path(__file__).resolve()),
        "artifact_paths": artifact_paths,
        "episode_npz_sha256": sidecar["episode_npz_sha256"],
        "evidence_report_sha256": evidence_digests,
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
    )

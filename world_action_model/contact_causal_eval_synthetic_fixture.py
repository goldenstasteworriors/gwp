"""CPU-only synthetic artifacts for causal evaluator admission replay.

The fixture is deliberately built from small authoritative source files and is
then accepted only by the production validators.  It is audit infrastructure;
it is never an alternative production validator and it never patches/stubs a
validator at runtime.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np


SYNTHETIC_DATASET_ID = "synthetic-causal-admission-robot25-v1"
SYNTHETIC_EPISODE_LENGTH = 304
SYNTHETIC_EPISODE_IDS = (0, 1, 2)
SYNTHETIC_EXPIRY = "2099-12-31T23:59:59Z"
SYNTHETIC_REVIEWED_AT = "2026-08-10T00:00:00Z"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _write_csv(
    path: Path, fields: tuple[str, ...] | list[str], rows: list[Mapping[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    return _sha_file(path)


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _run_checkpoint_reviewer_cli(
    *,
    repository_root: Path,
    receipt: Mapping[str, Any],
    acceptance_seed: Mapping[str, Any],
    causal: Mapping[str, Any],
    precursor: Mapping[str, Any],
    seed_artifact: Mapping[str, Any],
    annotations: Mapping[str, Any],
    runtime: Mapping[str, Any],
    traces: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Materialize one acceptance through the independent production CLI."""

    checkpoint_dir = Path(receipt["output_path_record_only"])
    internal_receipt = checkpoint_dir / "checkpoint_training_receipt.json"
    _write_json(internal_receipt, receipt)
    command = [
        __import__("sys").executable,
        "-B",
        str(repository_root / "scripts/review_contact_causal_checkpoint.py"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--robot-gate",
        str(causal["gate_path"]),
        "--gate-reviewer-acceptance",
        str(causal["gate_review_path"]),
        "--route-training-acceptance",
        str(causal["route_acceptances"][receipt["route"]["causal_route_id"]]["path"]),
        "--precursor-receipt",
        str(precursor["receipt_path"]),
        "--precursor-checkpoint-dir",
        str(precursor["checkpoint_dir"]),
        "--hardened-preflight-receipt",
        str(seed_artifact["hardened_preflight_path"]),
        "--production-preflight-receipt",
        str(causal["preflight_path"]),
        "--training-data-receipt",
        str(causal["training_data_path"]),
        "--coverage-receipt",
        str(seed_artifact["coverage"]["path"]),
        "--sidecar-dir",
        str(annotations["sidecar_dir"]),
        "--t5-embedding-dir",
        str(runtime["t5_embedding_dir"]),
        "--matched-control-report",
        str(traces["report_path"]),
    ]
    for route, path in traces["trace_path_by_route"].items():
        command.extend([f"--trace-{route.replace('_', '-')}", str(path)])
    command.extend(
        [
            "--output",
            str(output_path),
            "--reviewer-identity",
            acceptance_seed["reviewer_identity"],
            "--reviewer-attestation-nonce",
            acceptance_seed["reviewer_attestation_nonce"],
            "--acceptance-run-id",
            acceptance_seed["acceptance_run_id"],
            "--acceptance-run-nonce",
            acceptance_seed["acceptance_run_nonce"],
            "--evidence-valid-until-utc",
            acceptance_seed["evidence_valid_until_utc"],
            "--repository-root",
            str(repository_root),
        ]
    )
    subprocess.run(
        command,
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    value = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("checkpoint reviewer did not emit a JSON object")
    return value


def _build_hardware(fixture: Path, repository_root: Path) -> dict[str, Any]:
    from world_action_model.asymmetric_hardware_manifest import (
        calibration_capture_sha256,
        hardware_contract_sha256,
        validate_manifest_capture_binding,
    )

    manifest = json.loads(
        (
            repository_root
            / "configs/hardware/asymmetric25_hardware_manifest.template.json"
        ).read_text(encoding="utf-8")
    )
    manifest.pop("$schema", None)
    manifest.pop("$comment", None)
    manifest.update(
        contract_id="synthetic-causal-admission-hardware-contract-v1",
        readiness={
            "training": True,
            "deployment": False,
            "approved_by": "synthetic-hardware-reviewer",
            "approved_at_utc": SYNTHETIC_REVIEWED_AT,
        },
    )
    body_names = [f"g1_joint_{index:02d}" for index in range(29)]
    state_names = body_names + [
        "left_little_state",
        "left_ring_state",
        "left_middle_state",
        "left_index_state",
        "left_thumb_bend_state",
        "left_thumb_rotate_state",
        "right_omnipicker_aperture_state",
    ]
    command_names = body_names + [
        "left_little_command",
        "left_ring_command",
        "left_middle_command",
        "left_index_command",
        "left_thumb_bend_command",
        "left_thumb_rotate_command",
        "right_omnipicker_aperture_command",
    ]
    interface = {
        "protocol": "synthetic_verified_bus",
        "state_endpoint": "synthetic/state",
        "command_endpoint": "synthetic/command",
        "driver_revision": "synthetic-driver-r1",
        "firmware_revision": "synthetic-firmware-r1",
    }
    left = manifest["hardware"]["left_hand"]
    left.update(
        serial_number="SYNTHETIC-RH56-0001",
        state_feature="observation.raw_state",
        command_feature="raw_action",
        interface=interface,
    )
    right = manifest["hardware"]["right_gripper"]
    right.update(
        serial_number="SYNTHETIC-OMNI-0001",
        state_feature="observation.raw_state",
        command_feature="raw_action",
        interface={**interface, "state_endpoint": "synthetic/right_state"},
    )
    all_channels = list(left["channels"]) + [right["channel"]]
    for index, channel in enumerate(all_channels):
        channel["state"] = {
            "name": state_names[29 + index],
            "unit": "open_fraction",
            "closed_value": 0.0,
            "open_value": 1.0,
            "valid_min": 0.0,
            "valid_max": 1.0,
        }
        channel["command"] = {
            "name": command_names[29 + index],
            "unit": "open_fraction",
            "closed_value": 0.0,
            "open_value": 1.0,
            "valid_min": 0.0,
            "valid_max": 1.0,
            "semantics": "absolute_position_target",
            "safety_min_open_fraction": 0.0,
            "safety_max_open_fraction": 1.0,
            "max_step_open_fraction": 0.25,
        }
    manifest["wrists"].update(
        state_feature="observation.wrist_pose",
        left={
            "state_parent_frame": "torso",
            "state_child_frame": "left_tool",
            "wbc_fk_frame": "left_tool",
        },
        right={
            "state_parent_frame": "torso",
            "state_child_frame": "right_tool",
            "wbc_fk_frame": "right_tool",
        },
        translation_step_limit_m=0.1,
        rotation_step_limit_rad=0.5,
    )
    manifest["wbc"].update(
        state_feature="observation.raw_state",
        command_feature="raw_action",
        state_joint_names=body_names,
        command_joint_names=body_names,
    )
    manifest["timing"].update(
        timestamp_feature="timestamp",
        clock_domain="synthetic_monotonic_clock",
        nominal_rate_hz=30.0,
        max_period_jitter_s=0.001,
        measured_command_latency_s=0.001,
        max_command_latency_s=0.01,
        max_state_action_skew_s=0.001,
        max_camera_state_skew_s=0.001,
        latency_measurement_method="synthetic deterministic CPU fixture",
    )
    manifest["safety"].update(
        emergency_stop_endpoint="synthetic/emergency_stop",
        watchdog_timeout_s=0.1,
        max_consecutive_invalid_commands=1,
    )
    capture_base = {
        "capture_version": "asymmetric25-calibration-capture/v1",
        "verification_level": "bench_verified",
        "capture_id": "synthetic-causal-admission-calibration-v1",
        "captured_at_utc": SYNTHETIC_REVIEWED_AT,
        "operator": "synthetic-calibration-operator",
        "source_revision": "synthetic-source-r1",
        "devices": {
            "left_rh56dftp_serial": left["serial_number"],
            "right_omnipicker_serial": right["serial_number"],
        },
        "measurements": {},
        "approval": {
            "approved": True,
            "approved_by": "synthetic-hardware-reviewer",
            "approved_at_utc": SYNTHETIC_REVIEWED_AT,
        },
    }
    for name in ("hardware", "wrists", "wbc", "timing", "safety"):
        capture_base["measurements"][name] = {
            "verified": True,
            "method": "synthetic deterministic CPU fixture",
            "sample_count": 1,
            "result": deepcopy(manifest[name]),
        }
    capture_sha = calibration_capture_sha256(capture_base)
    manifest["calibration"] = {
        "verification_level": "bench_verified",
        "capture_id": capture_base["capture_id"],
        "capture_sha256": capture_sha,
        "captured_at_utc": capture_base["captured_at_utc"],
        "source_revision": capture_base["source_revision"],
        "verified_by": capture_base["approval"]["approved_by"],
    }
    validate_manifest_capture_binding(manifest, capture_base)
    manifest_path = fixture / "hardware-manifest.json"
    capture_path = fixture / "calibration-capture.json"
    _write_json(manifest_path, manifest)
    _write_json(capture_path, capture_base)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "hardware_contract_sha256": hardware_contract_sha256(manifest),
        "capture": capture_base,
        "capture_path": capture_path,
        "calibration_capture_sha256": capture_sha,
    }


def _build_dataset(fixture: Path, hardware: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from world_action_model.asymmetric_action_schema import (
        ASYMMETRIC_ACTION_NAMES,
        ASYMMETRIC_STATE_NAMES,
    )

    data_root = fixture / "dataset"
    meta_root = data_root / "meta"
    parquet_root = data_root / "data/chunk-000"
    meta_root.mkdir(parents=True, exist_ok=True)
    parquet_root.mkdir(parents=True, exist_ok=True)
    lengths = {episode: SYNTHETIC_EPISODE_LENGTH for episode in SYNTHETIC_EPISODE_IDS}
    rows: dict[str, list[Any]] = {
        "index": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "task_index": [],
        "observation.state": [],
        "action": [],
    }
    offset = 0
    for episode, length in lengths.items():
        for frame in range(length):
            base = episode * 0.1 + frame * 0.001
            rows["index"].append(offset + frame)
            rows["episode_index"].append(episode)
            rows["frame_index"].append(frame)
            rows["timestamp"].append(frame / 30.0)
            rows["task_index"].append(0)
            rows["observation.state"].append(
                [base + dimension * 0.0001 for dimension in range(25)]
            )
            rows["action"].append(
                [base * 0.5 + dimension * 0.0002 for dimension in range(25)]
            )
        offset += length
    vector_type = pa.list_(pa.float64(), 25)
    table = pa.table(
        {
            "index": pa.array(rows["index"], type=pa.int64()),
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
            "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
            "task_index": pa.array(rows["task_index"], type=pa.int64()),
            "observation.state": pa.array(
                rows["observation.state"], type=vector_type
            ),
            "action": pa.array(rows["action"], type=vector_type),
        }
    )
    parquet_path = parquet_root / "episode_000000.parquet"
    pq.write_table(table, parquet_path, compression="NONE")
    episodes_path = meta_root / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            {"episode_index": episode, "length": length, "tasks": ["synthetic"]}
            for episode, length in lengths.items()
        ],
    )
    _write_jsonl(meta_root / "tasks.jsonl", [{"task_index": 0, "task": "synthetic"}])
    info = {
        "codebase_version": "v2.1",
        "robot_type": "sonic_g1_asymmetric25",
        "dataset_id": SYNTHETIC_DATASET_ID,
        "action_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "fps": 30.0,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths.values()),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "frame_clock": "dataset_native_frame_index",
        "timestamp_semantics": {
            "feature": hardware["manifest"]["timing"]["timestamp_feature"],
            "origin": hardware["manifest"]["timing"]["timestamp_origin"],
            "state": hardware["manifest"]["timing"]["state_sample_semantics"],
            "action": hardware["manifest"]["timing"]["action_sample_semantics"],
        },
        "hardware_contract_sha256": hardware["hardware_contract_sha256"],
        "calibration_capture_sha256": hardware["calibration_capture_sha256"],
        "asymmetric_hardware_manifest": hardware["manifest"],
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [25],
                "names": list(ASYMMETRIC_STATE_NAMES),
            },
            "action": {
                "dtype": "float64",
                "shape": [25],
                "names": list(ASYMMETRIC_ACTION_NAMES),
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    info_path = meta_root / "info.json"
    _write_json(info_path, info)
    return {
        "data_root": data_root,
        "info_path": info_path,
        "episodes_path": episodes_path,
        "episode_lengths": lengths,
    }


def _annotation_row(
    *, episode: int, role: str, annotator: str, coverage_id: str, mode: str
) -> dict[str, Any]:
    submitted = role != "adjudicated"
    return {
        "schema_version": "contact_segment_annotation_v1_1",
        "dataset_id": SYNTHETIC_DATASET_ID,
        "episode_id": episode,
        "coverage_id": coverage_id,
        "annotation_role": role,
        "annotator_id": annotator,
        "review_status": "submitted" if submitted else "reviewed",
        "coverage_mode": mode,
        "start_frame": 0,
        "end_frame": SYNTHETIC_EPISODE_LENGTH,
        "frame_clock": "dataset_native_frame_index",
        "evidence_scope": "rgb_temporal",
        "label_origin": "human_independent" if submitted else "human_adjudicated",
        "review_timestamp_utc": SYNTHETIC_REVIEWED_AT,
        "parent_coverage_ids": "",
        "notes": "",
    }


def _segment_row(
    *,
    episode: int,
    role: str,
    annotator: str,
    coverage_id: str,
    segment_id: str,
    parent_segment_ids: str = "",
) -> dict[str, Any]:
    submitted = role != "adjudicated"
    sides = ("left", "right", "object_object")
    sites = ("opposition", "primary", "support")
    object_roles = ("manipulated", "tool", "receptacle")
    modes = ("stick", "slide", "roll")
    start = 80 + episode * 5
    end = 130 + episode * 5
    return {
        "schema_version": "contact_segment_annotation_v1_1",
        "dataset_id": SYNTHETIC_DATASET_ID,
        "episode_id": episode,
        "coverage_id": coverage_id,
        "segment_id": segment_id,
        "onset_event_id": f"onset-{segment_id}",
        "release_event_id": f"release-{segment_id}",
        "annotation_role": role,
        "annotator_id": annotator,
        "review_status": "submitted" if submitted else "reviewed",
        "start_frame": start,
        "end_frame": end,
        "start_boundary": "observed",
        "end_boundary": "observed",
        "side": sides[episode],
        "site": sites[episode],
        "object_role": object_roles[episode],
        "contact_mode": modes[episode],
        "confidence": 1.0,
        "evidence_scope": "rgb_temporal",
        "label_origin": "human_independent" if submitted else "human_adjudicated",
        "source_proposal_id": "",
        "parent_segment_ids": parent_segment_ids,
        "notes": "",
    }


def _build_annotations_and_sidecars(
    fixture: Path, repository_root: Path
) -> dict[str, Any]:
    from scripts.contact_labels.contact_label_io_v1_1 import (
        COVERAGE_FIELDS,
        INVENTORY_FIELDS,
        SEGMENT_FIELDS,
        encode_episode,
        load_coverages,
        load_inventory,
        load_segments,
    )
    from world_action_model.contact_transition_schema import (
        SEGMENT_FIELD_SLICES,
        SEGMENT_SCHEMA_VERSION,
        SEGMENT_TARGET_DIM,
    )

    annotation_root = fixture / "annotations"
    inventory_path = annotation_root / "episode_inventory.csv"
    inventory_rows = [
        {
            "episode_id": episode,
            "length_frames": SYNTHETIC_EPISODE_LENGTH,
            "subject_id": f"subject-{episode}",
            "session_id": f"session-{episode}",
            "operator_id": f"operator-{episode}",
            "source_object_instance_id": f"source-{episode}",
            "target_object_instance_id": f"target-{episode}",
            "liquid_id": f"liquid-{episode}",
            "collection_batch_id": f"batch-{episode}",
            "metadata_status": "complete",
        }
        for episode in SYNTHETIC_EPISODE_IDS
    ]
    _write_csv(inventory_path, INVENTORY_FIELDS, inventory_rows)
    split_fields = [
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
    assignments = {
        0: ("train", "test", "val"),
        1: ("val", "train", "test"),
        2: ("test", "val", "train"),
    }
    split_path = annotation_root / "episode_split_manifest.csv"
    _write_csv(
        split_path,
        split_fields,
        [
            {
                "episode_id": episode,
                "length_frames": SYNTHETIC_EPISODE_LENGTH,
                "seed_6666_split": assignments[episode][0],
                "seed_7777_split": assignments[episode][1],
                "seed_8888_split": assignments[episode][2],
                "split_unit": "group",
                "split_purpose": "production_strategy_evaluation",
                "ood_eligible": "true",
                "strategy_claim_allowed": "true",
                "metadata_status": "complete",
            }
            for episode in SYNTHETIC_EPISODE_IDS
        ],
    )
    config = json.loads(
        (repository_root / "configs/contact_asym25_robot_sidecar.template.json").read_text(
            encoding="utf-8"
        )
    )
    config["dataset_id"] = SYNTHETIC_DATASET_ID
    config["target_fps"] = 30.0
    config["split_policy"]["ood_eligible"] = True
    config["split_policy"]["strategy_claim_allowed"] = True
    config_path = annotation_root / "build-config.json"
    _write_json(config_path, config)

    # Every episode can appear alone as val/test under one production seed, so
    # each must carry exhaustive negative supervision as well as a physical
    # positive segment.
    modes = ("exhaustive", "exhaustive", "exhaustive")
    coverage_by_role: dict[str, list[dict[str, Any]]] = {}
    segments_by_role: dict[str, list[dict[str, Any]]] = {}
    for role, annotator, suffix in (
        ("independent_a", "annotator-independent-a", "a"),
        ("independent_b", "annotator-independent-b", "b"),
    ):
        coverage_by_role[suffix] = [
            _annotation_row(
                episode=episode,
                role=role,
                annotator=annotator,
                coverage_id=f"coverage-{suffix}-{episode}",
                mode=modes[episode],
            )
            for episode in SYNTHETIC_EPISODE_IDS
        ]
        segments_by_role[suffix] = [
            _segment_row(
                episode=episode,
                role=role,
                annotator=annotator,
                coverage_id=f"coverage-{suffix}-{episode}",
                segment_id=f"segment-{suffix}-{episode}",
            )
            for episode in SYNTHETIC_EPISODE_IDS
        ]
    adjudicated_coverages = []
    adjudicated_segments = []
    for episode in SYNTHETIC_EPISODE_IDS:
        coverage = _annotation_row(
            episode=episode,
            role="adjudicated",
            annotator="adjudicator-independent-c",
            coverage_id=f"coverage-adjudicated-{episode}",
            mode=modes[episode],
        )
        coverage["parent_coverage_ids"] = (
            f"coverage-a-{episode};coverage-b-{episode}"
        )
        adjudicated_coverages.append(coverage)
        adjudicated_segments.append(
            _segment_row(
                episode=episode,
                role="adjudicated",
                annotator="adjudicator-independent-c",
                coverage_id=coverage["coverage_id"],
                segment_id=f"segment-adjudicated-{episode}",
                parent_segment_ids=f"segment-a-{episode};segment-b-{episode}",
            )
        )
    sources: dict[str, Path] = {}
    for suffix in ("a", "b"):
        root = annotation_root / f"annotator_{suffix}"
        sources[f"coverage_annotator_{suffix}"] = root / "coverage.csv"
        sources[f"segments_annotator_{suffix}"] = root / "segments.csv"
        _write_csv(sources[f"coverage_annotator_{suffix}"], COVERAGE_FIELDS, coverage_by_role[suffix])
        _write_csv(sources[f"segments_annotator_{suffix}"], SEGMENT_FIELDS, segments_by_role[suffix])
    coverage_path = annotation_root / "coverage.csv"
    segments_path = annotation_root / "segments.csv"
    _write_csv(coverage_path, COVERAGE_FIELDS, adjudicated_coverages)
    _write_csv(segments_path, SEGMENT_FIELDS, adjudicated_segments)
    sources["coverage_adjudicated"] = coverage_path
    sources["segments_adjudicated"] = segments_path

    sidecar_dir = fixture / "sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory(inventory_path)
    coverages = load_coverages(coverage_path, inventory, config)
    segments = load_segments(segments_path, inventory, coverages, config)
    coverage_by_episode = {episode: [] for episode in inventory}
    segment_by_episode = {episode: [] for episode in inventory}
    for coverage in coverages:
        coverage_by_episode[coverage.episode_id].append(coverage)
    for segment in segments:
        segment_by_episode[segment.episode_id].append(segment)
    provenance_rows: list[dict[str, Any]] = []
    episode_hashes: dict[str, str] = {}
    for episode, item in sorted(inventory.items()):
        arrays, provenance = encode_episode(
            item.length_frames,
            coverage_by_episode[episode],
            segment_by_episode[episode],
            horizon_frames=48,
            num_queries=12,
            negative_confidence=0.1,
        )
        destination = sidecar_dir / f"episode_{episode:06d}.npz"
        with destination.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        episode_hashes[destination.name] = _sha_file(destination)
        provenance_rows.extend({"episode_id": episode, **row} for row in provenance)
    provenance_path = sidecar_dir / "window_segment_provenance.csv"
    _write_csv(
        provenance_path,
        (
            "episode_id",
            "frame_index",
            "query_index",
            "segment_id",
            "onset_event_id",
            "release_event_id",
        ),
        provenance_rows,
    )
    provenance_sha = _sha_file(provenance_path)
    bundle_sha = _sha_bytes(
        _canonical_bytes(
            {
                "episode_npz_sha256": episode_hashes,
                "window_segment_provenance_sha256": provenance_sha,
            }
        )
    )
    meta = {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "target_dim": SEGMENT_TARGET_DIM,
        "field_slices": {
            key: [value.start, value.stop]
            for key, value in SEGMENT_FIELD_SLICES.items()
        },
        "dataset_id": SYNTHETIC_DATASET_ID,
        "frame_clock": "dataset_native_frame_index",
        "target_fps": 30.0,
        "horizon_frames": 48,
        "num_queries": 12,
        "negative_confidence": 0.1,
        "overflow_policy": "hard_error_no_drop",
        "dropped_segments": 0,
        "window_assignment_sort_key": [
            "start_frame",
            "end_frame",
            "side_or_empty",
            "site_or_empty",
            "segment_id",
        ],
        "episode_lengths": {
            str(key): value.length_frames for key, value in sorted(inventory.items())
        },
        "reviewed_coverage_rows": len(coverages),
        "unique_physical_segments": len(segments),
        "window_segment_rows": len(provenance_rows),
        "input_sha256": {
            "coverage_csv": _sha_file(coverage_path),
            "segments_csv": _sha_file(segments_path),
            "inventory_csv": _sha_file(inventory_path),
            "config": _sha_file(config_path),
            "split_manifest": _sha_file(split_path),
        },
        "episode_npz_sha256": episode_hashes,
        "window_segment_provenance_sha256": provenance_sha,
        "sidecar_bundle_sha256": bundle_sha,
        "builder_identity": "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py",
        "annotation_contract": {
            "annotation_role": "adjudicated",
            "review_status": "reviewed",
            "label_origin": "human_adjudicated",
            "automatic_or_proposal_ground_truth": False,
        },
        "strategy_claim_allowed": True,
        "warning": "synthetic audit fixture; repeated window rows are not independent events",
    }
    meta_path = sidecar_dir / "meta.json"
    _write_json(meta_path, meta)
    return {
        "annotation_root": annotation_root,
        "inventory_path": inventory_path,
        "split_path": split_path,
        "config_path": config_path,
        "coverage_path": coverage_path,
        "segments_path": segments_path,
        "annotation_sources": sources,
        "sidecar_dir": sidecar_dir,
        "sidecar_meta_path": meta_path,
        "sidecar_meta_sha256": _sha_file(meta_path),
        "sidecar_bundle_sha256": bundle_sha,
        "episode_npz_sha256": episode_hashes,
    }


def _build_norm(
    fixture: Path,
    dataset: Mapping[str, Any],
    annotations: Mapping[str, Any],
    hardware: Mapping[str, Any],
    seed: int,
) -> Path:
    from world_action_model.contact_asym25_preflight import (
        _dataset_episode_lengths,
        _recompute_production_train_norm_from_verified_raw,
    )
    from world_action_model.contact_asym25_train_norm_contract import (
        build_expected_train_norm_source_receipt,
        build_norm_source_tree_receipt,
        build_production_split_receipt,
    )

    lengths, episode_meta_sha = _dataset_episode_lengths(dataset["data_root"])
    assignments = {
        6666: {"train": [0], "val": [1], "test": [2]},
        7777: {"train": [1], "val": [2], "test": [0]},
        8888: {"train": [2], "val": [0], "test": [1]},
    }[seed]
    source_tree = build_norm_source_tree_receipt(dataset["data_root"])
    split_receipt, split_sha = build_production_split_receipt(
        dataset_id=SYNTHETIC_DATASET_ID,
        action_schema_id="sonic_g1_rh56dftp6_omnipicker1_v1",
        split_seed=seed,
        train_episode_ids=assignments["train"],
        validation_episode_ids=assignments["val"],
        heldout_episode_ids=assignments["test"],
        dataset_info_sha256=_sha_file(dataset["info_path"]),
        episode_metadata_sha256=episode_meta_sha,
        split_manifest_sha256=_sha_file(annotations["split_path"]),
    )
    source_receipt, source_sha = build_expected_train_norm_source_receipt(
        production_split_receipt=split_receipt,
        production_split_receipt_sha256=split_sha,
        episode_lengths=lengths,
        source_dataset_tree_receipt=source_tree,
    )
    raw, norm, raw_sha = _recompute_production_train_norm_from_verified_raw(
        data_root=dataset["data_root"],
        episode_lengths=lengths,
        train_norm_source_receipt=source_receipt,
        train_norm_source_receipt_sha256=source_sha,
    )
    mirrored = (
        "split_seed",
        "train_episode_ids",
        "train_episode_ids_sha256",
        "production_split_receipt_sha256",
        "split_manifest_sha256",
        "source_dataset_tree_receipt",
        "action_schema_id",
        "physical_dim",
        "model_dim",
        "action_horizon",
        "delta_mask",
        "action_representation",
        "boundary",
        "padding_contract",
        "grip_validity_contract",
        "num_episodes",
        "num_state_samples",
        "num_action_samples",
        "dataset_info_sha256",
        "episode_metadata_sha256",
    )
    metadata = {
        "task_name": "contact_asym25_production_train_only",
        "data_roots": [str(Path(dataset["data_root"]).absolute())],
        "hardware_contract_sha256": hardware["hardware_contract_sha256"],
        "calibration_capture_sha256": hardware["calibration_capture_sha256"],
        "calibration_verification_level": "bench_verified",
        "quantile_sample_limit": 0,
        "production_train_norm_source_receipt": source_receipt,
        "production_train_norm_source_receipt_sha256": source_sha,
        "production_train_raw_statistics": raw,
        "production_train_raw_statistics_sha256": raw_sha,
        **{name: source_receipt[name] for name in mirrored},
    }
    path = fixture / f"seed-{seed}" / "norm-stats.json"
    _write_json(path, {"norm_stats": norm, "metadata": metadata})
    return path


def _synthetic_sha(label: str) -> str:
    return _sha_bytes(label.encode("utf-8"))


def _build_legacy_gate_and_hardened(
    fixture: Path,
    repository_root: Path,
    hardware: Mapping[str, Any],
    dataset: Mapping[str, Any],
    annotations: Mapping[str, Any],
    norm_path: Path,
    seed: int,
) -> dict[str, Any]:
    from scripts.contact_labels.contact_label_io_v1_1 import (
        COVERAGE_FIELDS,
        SEGMENT_FIELDS,
        read_csv_exact,
    )
    from world_action_model.contact_asym25_preflight import (
        REQUIRED_GATE_FLAGS,
        _dataset_episode_lengths,
        _recompute_bound_annotation_agreement,
        _reviewed_gate_contract_sha256,
        validate_contact_asym25_training_preflight,
    )
    from world_action_model.matched_control_hashes import (
        COLLECTOR_VERSION,
        HASH_SCHEMA,
        IMPLEMENTED_RUNNER_VARIANTS,
        REQUIRED_HASHES,
        canonical_sha256,
        compare_variant_traces,
        json_sha256,
    )

    seed_root = fixture / f"seed-{seed}"
    report_root = seed_root / "legacy-evidence"
    report_root.mkdir(parents=True, exist_ok=True)
    source_rows: dict[str, list[dict[str, str]]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for name, path in annotations["annotation_sources"].items():
        fields = COVERAGE_FIELDS if name.startswith("coverage_") else SEGMENT_FIELDS
        source_rows[name] = read_csv_exact(path, fields)
        source_files[name] = {"path": str(path), "sha256": _sha_file(path)}
    metrics, _ = _recompute_bound_annotation_agreement(source_rows)
    annotation_report = {
        "report_version": "contact-annotation-agreement/v1",
        "status": "passed",
        "independent_annotation": True,
        "adjudication_completed": True,
        "dual_annotation_agreement_passed": True,
        "independent_segment_count": 3,
        "metrics": metrics,
        "source_files": source_files,
    }
    annotation_report_path = report_root / "annotation-agreement.json"
    annotation_report_sha = _write_json(annotation_report_path, annotation_report)
    _, episode_meta_sha = _dataset_episode_lengths(dataset["data_root"])
    hardware_report = {
        "report_version": "asymmetric25-hardware-verification/v1",
        "status": "passed",
        "real_hardware_capture": True,
        "dataset_id": SYNTHETIC_DATASET_ID,
        "hardware_contract_sha256": hardware["hardware_contract_sha256"],
        "calibration_capture_sha256": hardware["calibration_capture_sha256"],
        "dataset_info_sha256": _sha_file(dataset["info_path"]),
        "episode_metadata_sha256": episode_meta_sha,
    }
    hardware_report_path = report_root / "hardware-verification.json"
    hardware_report_sha = _write_json(hardware_report_path, hardware_report)

    training_config = {
        "seed": seed,
        "fixture": "synthetic-causal-admission",
        "precision": "bfloat16",
    }
    stable_training_config = dict(training_config)
    stable_training_config["seed"] = "__CONTACT_ASYM25_SPLIT_SEED__"
    control_sha = _synthetic_sha(f"legacy-control-{seed}")
    run_contract = {
        "initial_weights_bundle_sha256": _synthetic_sha("legacy-initial-weights"),
        "base_model_bundle_sha256": _synthetic_sha("legacy-base-model"),
        "training_code_receipt_sha256": _synthetic_sha("legacy-training-code"),
        "dataset_source_tree_sha256": _synthetic_sha("legacy-dataset-source"),
        "runtime_environment_receipt_sha256": _synthetic_sha("legacy-runtime"),
        "optimizer_config_sha256": _synthetic_sha("legacy-optimizer"),
        "scheduler_config_sha256": _synthetic_sha("legacy-scheduler"),
        "training_config_receipt": training_config,
        "training_config_sha256": json_sha256(training_config),
        "stable_training_config_sha256": json_sha256(stable_training_config),
        "control_config_sha256": control_sha,
        "preflight_receipt_sha256": _synthetic_sha(f"legacy-preflight-{seed}"),
        "train_window_receipt_sha256": _synthetic_sha(f"legacy-windows-{seed}"),
        "precision": "bfloat16",
        "action_schema_contract_sha256": _synthetic_sha("legacy-action-schema"),
        "action_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "contact_schema_version": "contact_segment_v1_1",
        "contact_query_count": 12,
        "contact_target_dim": 22,
        "contact_time_bins": 16,
        "dataset_id": SYNTHETIC_DATASET_ID,
        "dataset_info_sha256": _sha_file(dataset["info_path"]),
        "hardware_contract_sha256": hardware["hardware_contract_sha256"],
        "norm_stats_sha256": _sha_file(norm_path),
    }
    run_contract_sha = json_sha256(run_contract)
    collector_source = repository_root / "world_action_model/matched_control_hashes.py"
    collector_source_sha = _sha_file(collector_source)
    batch_ids = [
        {
            "dataset_id": SYNTHETIC_DATASET_ID,
            "episode_index": 0,
            "frame_index": 0,
            "global_index": 0,
        }
    ]
    common_hashes = {
        name: (
            canonical_sha256(batch_ids)
            if name == "ordered_sample_ids"
            else _synthetic_sha(f"legacy-trace-{seed}-{name}")
        )
        for name in REQUIRED_HASHES
    }
    trace_paths: dict[str, Path] = {}
    for variant in IMPLEMENTED_RUNNER_VARIANTS:
        trace_path = report_root / f"legacy-{variant}.jsonl"
        trace_paths[variant] = trace_path
        record = {
            "record_version": COLLECTOR_VERSION,
            "run_id": f"legacy-training-run-{seed}-{variant}",
            "run_nonce": f"legacy-training-nonce-{seed}-{variant}",
            "variant_id": variant,
            "seed": seed,
            "global_step": 0,
            "micro_step": 0,
            "rank": 0,
            "world_size": 1,
            "variant_config_sha256": _synthetic_sha(
                f"legacy-variant-config-{seed}-{variant}"
            ),
            "control_config_sha256": control_sha,
            "shared_model_config_sha256": _synthetic_sha("legacy-shared-model"),
            "transport_override": {
                "variant_id": variant,
                "contact_semantic_transport_mode": {
                    "soft_semantic": "soft",
                    "hard_st": "hard_st",
                    "hard_detached": "hard_detached",
                }[variant],
            },
            "config_hash_scope": {
                "control": ["all_non_transport_fields"],
                "variant": ["contact_semantic_transport_mode"],
            },
            "git_commit": _git_head(repository_root),
            "collector_code_sha256": collector_source_sha,
            "training_data_receipt_sha256": _synthetic_sha(
                f"legacy-training-data-{seed}"
            ),
            "heldout_window_manifest_sha256": run_contract[
                "train_window_receipt_sha256"
            ],
            "production_split_seed": seed,
            "production_preflight_receipt_sha256": run_contract[
                "preflight_receipt_sha256"
            ],
            "run_contract": run_contract,
            "run_contract_sha256": run_contract_sha,
            "external_acceptance_sha256": _synthetic_sha(
                f"legacy-external-acceptance-{seed}-{variant}"
            ),
            "batch_ids": batch_ids,
            "diagnostic_data_paths": [str(dataset["data_root"])],
            "hashes": common_hashes,
        }
        _write_jsonl(trace_path, [record])
    collector_report_path = report_root / "matched-control-report.json"
    collector = compare_variant_traces(
        {variant: [trace_paths[variant]] for variant in IMPLEMENTED_RUNNER_VARIANTS},
        output_report=collector_report_path,
        collector_code_path=collector_source,
        required_seeds=[seed],
        allow_distinct_variant_run_identities=True,
        write_report=True,
    )
    if collector["status"] != "passed":
        raise RuntimeError(f"synthetic legacy collector failed: {collector['errors']}")
    collector_sha = _sha_file(collector_report_path)
    expected = {
        "independent_segment_count": 3,
        "hardware_contract_sha256": hardware["hardware_contract_sha256"],
        "calibration_capture_sha256": hardware["calibration_capture_sha256"],
        "dataset_info_sha256": _sha_file(dataset["info_path"]),
        "norm_stats_sha256": _sha_file(norm_path),
        "sidecar_meta_sha256": annotations["sidecar_meta_sha256"],
        "sidecar_bundle_sha256": annotations["sidecar_bundle_sha256"],
        "split_manifest_sha256": _sha_file(annotations["split_path"]),
        "episode_metadata_sha256": episode_meta_sha,
        "coverage_csv_sha256": _sha_file(annotations["coverage_path"]),
        "segments_csv_sha256": _sha_file(annotations["segments_path"]),
        "inventory_csv_sha256": _sha_file(annotations["inventory_path"]),
        "build_config_sha256": _sha_file(annotations["config_path"]),
    }
    matched_expectations = {
        "expected_git_commit": collector["git_commit"],
        "expected_run_id_by_variant": collector["run_id_by_variant"],
        "expected_run_nonce_by_variant": collector["run_nonce_by_variant"],
        "expected_report_sha256": collector_sha,
        "expected_trace_files": collector["trace_files"],
        "expected_trace_manifest_sha256": collector["trace_manifest_sha256"],
        "expected_global_hashes_sha256": collector["global_hashes_sha256"],
        "expected_hash_schema_sha256": canonical_sha256(HASH_SCHEMA),
        "expected_control_config_sha256": collector["control_config_sha256"],
        "expected_variant_config_sha256": collector["variant_config_sha256"],
        "expected_shared_model_config_sha256": collector[
            "shared_model_config_sha256"
        ],
        "expected_world_size": collector["world_size"],
        "expected_rank_set": collector["rank_set"],
        "expected_collector_version": COLLECTOR_VERSION,
        "expected_collector_source_sha256": collector_source_sha,
        "expected_run_contract_sha256": run_contract_sha,
        "expected_external_acceptance_sha256_by_variant": collector[
            "external_acceptance_sha256_by_variant"
        ],
    }
    reviewer_identity = f"legacy-gate-reviewer-{seed}"
    gate = {
        "gate_version": "contact-asym25-training-gate/v2",
        "evidence_valid_until_utc": SYNTHETIC_EXPIRY,
        **{name: True for name in REQUIRED_GATE_FLAGS},
        "independent_segment_count": 3,
        "artifact_fingerprints": {
            key: value for key, value in expected.items() if key != "independent_segment_count"
        },
        "evidence_reports": {
            "annotation_agreement": {
                "path": str(annotation_report_path),
                "sha256": annotation_report_sha,
            },
            "hardware_verification": {
                "path": str(hardware_report_path),
                "sha256": hardware_report_sha,
            },
            "matched_control_hash_collector": {
                "path": str(collector_report_path),
                "sha256": collector_sha,
            },
            "gate_reviewer_acceptance": {
                "path": str(report_root / "gate-review.json"),
                "sha256": _synthetic_sha("legacy-review-placeholder"),
            },
        },
        "matched_control_expectations": matched_expectations,
        "reviewed_by": reviewer_identity,
        "reviewed_at_utc": SYNTHETIC_REVIEWED_AT,
        "note": "synthetic CPU audit fixture",
    }
    review = {
        "acceptance_version": "contact-asym25-gate-reviewer-acceptance/v1",
        "status": "accepted",
        "accepted": True,
        "reviewer_identity": reviewer_identity,
        "acceptance_run_id": f"legacy-gate-acceptance-run-{seed}",
        "acceptance_run_nonce": f"legacy-gate-acceptance-nonce-{seed}",
        "evidence_valid_until_utc": SYNTHETIC_EXPIRY,
        "reviewed_gate_contract_sha256": _reviewed_gate_contract_sha256(gate),
        "matched_control_expectations_sha256": canonical_sha256(
            matched_expectations
        ),
        "matched_control_report_sha256": collector_sha,
        "matched_control_run_contract_sha256": run_contract_sha,
        "reviewed_run_contract": run_contract,
        "dataset_id": SYNTHETIC_DATASET_ID,
        "dataset_info_sha256": expected["dataset_info_sha256"],
        "hardware_contract_sha256": expected["hardware_contract_sha256"],
        "norm_stats_sha256": expected["norm_stats_sha256"],
        "action_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "contact_schema_version": "contact_segment_v1_1",
        "accepted_precision": "bfloat16",
    }
    review_path = report_root / "gate-review.json"
    review_sha = _write_json(review_path, review)
    gate["evidence_reports"]["gate_reviewer_acceptance"]["sha256"] = review_sha
    gate_path = seed_root / "legacy-training-gate.json"
    gate_sha = _write_json(gate_path, gate)
    hardened = validate_contact_asym25_training_preflight(
        data_root=dataset["data_root"],
        norm_stats_path=norm_path,
        hardware_manifest_path=hardware["manifest_path"],
        calibration_capture_path=hardware["capture_path"],
        sidecar_dir=annotations["sidecar_dir"],
        coverage_csv=annotations["coverage_path"],
        segments_csv=annotations["segments_path"],
        inventory_csv=annotations["inventory_path"],
        build_config_path=annotations["config_path"],
        split_manifest=annotations["split_path"],
        training_gate_path=gate_path,
        split_seed=seed,
    )
    hardened_path = seed_root / "hardened-preflight.json"
    hardened_sha = _write_json(hardened_path, hardened)
    return {
        "legacy_gate_path": gate_path,
        "legacy_gate_sha256": gate_sha,
        "legacy_gate": gate,
        "legacy_gate_review_path": review_path,
        "legacy_gate_review_sha256": review_sha,
        "legacy_matched_control_report_path": collector_report_path,
        "legacy_matched_control_report_sha256": collector_sha,
        "hardened_preflight": hardened,
        "hardened_preflight_path": hardened_path,
        "hardened_preflight_sha256": hardened_sha,
    }


def _build_precursor(
    fixture: Path,
    repository_root: Path,
    *,
    step0_parameter_sha256: str,
) -> dict[str, Any]:
    import io
    import torch

    from world_action_model.contact_causal_production import named_tensor_digest
    from world_action_model.contact_ego_causal_precursor import (
        build_live_precursor_receipt,
        freeze_precursor_action_path,
        precursor_objective_contract,
    )

    class SyntheticPrecursor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.contact_planner_tiny_weight = torch.nn.Parameter(
                torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
            )
            self.contact_condition_encoder = torch.nn.Linear(1, 1, bias=True)
            with torch.no_grad():
                self.contact_condition_encoder.weight.fill_(3.0)
                self.contact_condition_encoder.bias.fill_(4.0)

    model = SyntheticPrecursor()
    live_sha = named_tensor_digest(list(model.named_parameters()))
    if live_sha != step0_parameter_sha256:
        raise RuntimeError("synthetic precursor differs from common step-0 tensors")
    partition = freeze_precursor_action_path(model)
    frozen = named_tensor_digest(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name in set(partition["frozen_parameter_names"])
        ]
    )
    checkpoint_dir = fixture / "precursor"
    checkpoint_dir.mkdir(parents=True)
    _write_json(
        checkpoint_dir / "config.json",
        {"_class_name": "SyntheticRouteNeutralPrecursor"},
    )
    payload = io.BytesIO()
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        payload,
        _use_new_zipfile_serialization=False,
    )
    (checkpoint_dir / "diffusion_pytorch_model.bin").write_bytes(payload.getvalue())
    receipt, bundle = build_live_precursor_receipt(
        model=model,
        checkpoint_dir=checkpoint_dir,
        parameter_partition=partition,
        objective_contract=precursor_objective_contract(fixed_save_step=100),
        evidence_base={
            "gate_sha256": _synthetic_sha("synthetic-precursor-gate"),
            "root_checkpoint_bundle_sha256": _synthetic_sha(
                "synthetic-precursor-root"
            ),
            "dataset_source_tree_sha256": _synthetic_sha(
                "synthetic-precursor-dataset"
            ),
            "sidecar_bundle_sha256": _synthetic_sha(
                "synthetic-precursor-sidecar"
            ),
            "training_code_head": _git_head(repository_root),
            "training_code_source_tree_sha256": _synthetic_sha(
                "synthetic-precursor-source"
            ),
            "training_run_id": "synthetic-precursor-training-run-id",
            "training_run_nonce": "synthetic-precursor-training-run-nonce",
            "reviewer_acceptance_sha256": _synthetic_sha(
                "synthetic-precursor-review"
            ),
        },
        root_parameter_tensor_sha256=live_sha,
        frozen_root_tensor_sha256=frozen,
    )
    receipt_path = fixture / "precursor-receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_bytes_sha256": receipt_sha,
        "checkpoint_dir": checkpoint_dir,
        "bundle_sha256": bundle["bundle_sha256"],
        "parameter_tensor_sha256": live_sha,
    }


def _build_t5_and_runtime(
    fixture: Path,
    hardened: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from world_action_model.contact_causal_preflight import (
        build_robot_runtime_source_receipts_v1,
    )

    root = Path(hardened["data_root"]) / "t5_embedding"
    root.mkdir(parents=True)
    for episode in SYNTHETIC_EPISODE_IDS:
        torch.save(
            torch.full((1, 2, 4), float(episode), dtype=torch.float32),
            root / f"episode_{episode:06d}.pt",
        )
    runtime = build_robot_runtime_source_receipts_v1(
        hardened_preflight_receipt=hardened,
        t5_embedding_dir=root,
    )
    return {"t5_embedding_dir": root, **runtime}


def _build_coverage(
    seed_root: Path,
    *,
    annotations: Mapping[str, Any],
    hardened: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    from world_action_model.contact_causal_coverage import (
        recompute_dataset_supervision_coverage,
    )

    receipt = recompute_dataset_supervision_coverage(
        sidecar_dir=annotations["sidecar_dir"],
        expected_meta_sha256=annotations["sidecar_meta_sha256"],
        expected_episode_sha256=hardened["episode_npz_sha256"],
        dataset_id=SYNTHETIC_DATASET_ID,
        split_seed=seed,
        split_episode_ids={
            "train": hardened["train_episode_ids"],
            "val": hardened["validation_episode_ids"],
            "test": hardened["heldout_episode_ids"],
        },
    )
    path = seed_root / "coverage.json"
    digest = _write_json(path, receipt)
    return {"receipt": receipt, "path": path, "sha256": digest}


def _build_current_causal_traces(
    seed_root: Path,
    *,
    repository_root: Path,
    seed: int,
    source_sha256: str,
    matrix_sha256: str,
    hardened: Mapping[str, Any],
    coverage_sha256: str,
    runtime: Mapping[str, Any],
    precursor: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from world_action_model.contact_causal_matched_control import (
        build_synthetic_stable_contract_v3,
        build_trace_record,
        capture_pre_forward_rng_v3,
        compare_causal_records,
        live_optimizer_scheduler_receipts_v3,
        write_trace_jsonl_exclusive,
    )
    from world_action_model.contact_causal_production import (
        CAUSAL_ROUTES,
        build_production_route_receipt,
        sha256_json,
    )
    from world_action_model.matched_control_hashes import canonical_sha256

    target = torch.zeros(1, 1, 22)
    mask = torch.zeros_like(target)
    confidence = torch.zeros_like(target)
    action = torch.zeros(1, 2, 4)
    action_mask = {
        "dimension_mask": None,
        "time_mask": torch.ones(1, 2),
        "dimension_weight": None,
    }
    prompt = torch.zeros(1, 2, 3)
    state = torch.zeros(1, 1, 3)
    embodiment = torch.zeros(1, dtype=torch.long)
    visual_timestep = torch.tensor([10])
    action_timestep = torch.tensor([20])
    visual_noise = torch.zeros(1, 1)
    action_noise = torch.zeros_like(action)
    rng = capture_pre_forward_rng_v3()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    live = live_optimizer_scheduler_receipts_v3(
        model=model, optimizers=[optimizer], schedulers=[scheduler]
    )
    stable = build_synthetic_stable_contract_v3()
    stable["code"] = {
        "git_commit": _git_head(repository_root),
        "source_tree_sha256": source_sha256,
        "collector_source_sha256": _sha_file(
            repository_root / "world_action_model/contact_causal_matched_control.py"
        ),
    }
    stable["config"]["matrix_manifest_sha256"] = matrix_sha256
    stable["dataset"].update(
        dataset_id=hardened["dataset_id"],
        dataset_tree_sha256=hardened["norm_source_tree_receipt"]["tree_sha256"],
        dataset_info_sha256=hardened["dataset_info_sha256"],
        episode_metadata_sha256=hardened["episode_metadata_sha256"],
        split_manifest_sha256=hardened["split_manifest_sha256"],
        ordered_window_manifest_sha256=_synthetic_sha(f"current-windows-{seed}"),
    )
    stable["hardware"].update(
        hardware_contract_sha256=hardened["hardware_contract_sha256"],
        calibration_capture_sha256=hardened["calibration_capture_sha256"],
    )
    stable["normalization"]["train_only_norm_sha256"] = hardened[
        "norm_stats_sha256"
    ]
    stable["runtime"].update(
        runtime_environment_sha256=_synthetic_sha("runtime-environment-placeholder"),
        dataset_runtime_source_receipt_sha256=runtime[
            "training_source_runtime_receipt_sha256"
        ],
        t5_runtime_source_receipt_sha256=runtime["t5_runtime_source_receipt_sha256"],
    )
    stable["precursor"].update(
        precursor_receipt_bytes_sha256=precursor["receipt_bytes_sha256"],
        precursor_bundle_sha256=precursor["bundle_sha256"],
        initial_parameter_tensors_sha256=precursor["parameter_tensor_sha256"],
    )
    trace_keys = [f"step{step}/micro0/rank0" for step in (0, 100, 200)]
    stream_values = {
        "contact_target_stream_sha256": target,
        "contact_mask_stream_sha256": mask,
        "contact_confidence_stream_sha256": confidence,
        "action_target_stream_sha256": action,
        "action_mask_stream_sha256": action_mask,
        "state_input_stream_sha256": state,
        "state_mask_stream_sha256": None,
        "embodiment_id_stream_sha256": embodiment,
        "prompt_t5_stream_sha256": prompt,
    }
    stable["labels"].update(
        sidecar_meta_sha256=hardened["sidecar_meta_sha256"],
        sidecar_bundle_sha256=hardened["sidecar_bundle_sha256"],
        contact_coverage_sha256=coverage_sha256,
        **{
            name: sha256_json(
                {trace_key: canonical_sha256(value) for trace_key in trace_keys}
            )
            for name, value in stream_values.items()
        },
    )
    stable["source_receipts"].update(
        training_data_receipt_sha256=_synthetic_sha(f"training-data-{seed}"),
        preflight_receipt_sha256=_synthetic_sha(f"preflight-{seed}"),
        coverage_receipt_sha256=coverage_sha256,
        current_head_source_receipt_sha256=source_sha256,
    )
    records = {}
    paths = {}
    for route in CAUSAL_ROUTES:
        route_records = []
        for checkpoint_step in (0, 100, 200):
            route_records.append(
                build_trace_record(
                    route=route,
                    seed=seed,
                    global_step=checkpoint_step,
                    micro_step=0,
                    rank=0,
                    world_size=1,
                    run_id=f"current-training-run-{seed}-{route}",
                    run_nonce=f"current-training-nonce-{seed}-{route}",
                    stable_contract=stable,
                    route_local_contract=build_production_route_receipt(route),
                    ordered_sample_ids=[{"episode": 0, "frame": 0}],
                    raw_batch=[_synthetic_sha("current-raw-batch")],
                    augmentation_parameters=[{"crop": [0, 0, 1, 1]}],
                    visual_timestep=visual_timestep,
                    action_timestep=action_timestep,
                    visual_noise=visual_noise,
                    action_noise=action_noise,
                    contact_target=target,
                    contact_loss_mask=mask,
                    contact_confidence=confidence,
                    action_target=action,
                    action_loss_mask=action_mask,
                    state_input=state,
                    state_input_mask=None,
                    embodiment_id=embodiment,
                    prompt_t5=prompt,
                    pre_forward_rng=rng,
                    initial_parameter_tensors_sha256=precursor[
                        "parameter_tensor_sha256"
                    ],
                    optimizer_parameter_groups=live[
                        "optimizer_parameter_groups"
                    ],
                    optimizer_step0_state=live["optimizer_step0_state"],
                    scheduler_step0_state=live["scheduler_step0_state"],
                    precursor_receipt_bytes_sha256=precursor[
                        "receipt_bytes_sha256"
                    ],
                )
            )
        records[route] = route_records
        path = seed_root / f"current-{route}.jsonl"
        write_trace_jsonl_exclusive(
            path, route_records, project_root=repository_root
        )
        paths[route] = path
    report = compare_causal_records(records, required_seed=seed)
    report_path = seed_root / "current-matched-control-report.json"
    report_sha = _write_json(report_path, report)
    return {
        "records_by_route": records,
        "trace_path_by_route": paths,
        "report": report,
        "report_path": report_path,
        "report_sha256": report_sha,
        "runtime_environment_placeholder": stable["runtime"][
            "runtime_environment_sha256"
        ],
    }


def _build_causal_gate(
    seed_root: Path,
    *,
    repository_root: Path,
    seed: int,
    hardened_artifact: Mapping[str, Any],
    annotations: Mapping[str, Any],
    hardware: Mapping[str, Any],
    dataset: Mapping[str, Any],
    norm_path: Path,
    coverage: Mapping[str, Any],
    runtime: Mapping[str, Any],
    precursor: Mapping[str, Any],
    current_source_sha256: str,
    current_traces: Mapping[str, Any],
) -> dict[str, Any]:
    from world_action_model.contact_causal_gate_bootstrap import (
        ROBOT_BOOTSTRAP_CODE_PATHS,
        _gate_contract_sha,
        _route_acceptance_gate_contract_sha,
        runtime_environment_sha256_stdlib,
    )
    from world_action_model.contact_causal_preflight import (
        REQUIRED_GATE_FLAGS,
        _reviewed_utc,
        build_preflight_receipt_v2,
        build_training_data_receipt_v2,
        validate_robot_gate_v3,
    )
    from world_action_model.contact_causal_production import (
        CAUSAL_ROUTES,
        MATRIX_ID,
        MATRIX_SCHEMA_VERSION,
        PRECURSOR_ID,
        SUPERVISION_VERSION,
        build_production_route_receipt,
        sha256_json,
    )
    from world_action_model.contact_causal_matched_control import COLLECTOR_VERSION
    from world_action_model.verified_io import verified_head_code_manifest

    del _reviewed_utc
    head = _git_head(repository_root)
    _, code_rows = verified_head_code_manifest(
        repository_root, ROBOT_BOOTSTRAP_CODE_PATHS, expected_commit=head
    )
    code_blobs = {
        row["logical_path"]: row["head_blob_sha256"] for row in code_rows
    }
    runtime_environment_sha = runtime_environment_sha256_stdlib()
    matrix_sha = _sha_file(
        repository_root / "configs/contact_causal_production_matrix.v1.json"
    )
    route_bundle_sha = sha256_json(
        {route: build_production_route_receipt(route) for route in CAUSAL_ROUTES}
    )
    oracle = {
        "receipt_version": "oracle-sufficiency-contact-hard-q12-b16/v1",
        "status": "passed",
        "baseline_route": "aux_only_direct_q12",
        "oracle_route": "hard_st_semantic_reviewed_native_oracle",
        "reviewed_native_robot_labels": True,
        "independent_gate_sha256": _synthetic_sha(f"oracle-gate-{seed}"),
        "independent_reviewer_acceptance_sha256": _synthetic_sha(
            f"oracle-review-{seed}"
        ),
        "baseline_training_run_id": f"oracle-baseline-run-{seed}",
        "baseline_training_run_nonce": f"oracle-baseline-nonce-{seed}",
        "oracle_training_run_id": f"oracle-training-run-{seed}",
        "oracle_training_run_nonce": f"oracle-training-nonce-{seed}",
        "task_success_delta_percentage_points": 10.0,
        "correct_first_contact_delta_percentage_points": 15.0,
        "spill_delta_percentage_points": 0.0,
        "collision_delta_percentage_points": 0.0,
        "safety_violation_delta_percentage_points": 0.0,
        "evidence_valid_until_utc": SYNTHETIC_EXPIRY,
        "oracle_is_main_matrix_route": False,
        "predicted_routes_used_oracle": False,
    }
    oracle_path = seed_root / "oracle-sufficiency.json"
    oracle_sha = _write_json(oracle_path, oracle)
    hardened = hardened_artifact["hardened_preflight"]
    fingerprints = {
        "matrix_manifest_sha256": matrix_sha,
        "current_head_source_tree_sha256": current_source_sha256,
        "precursor_receipt_bytes_sha256": precursor["receipt_bytes_sha256"],
        "precursor_bundle_sha256": precursor["bundle_sha256"],
        "hardware_contract_sha256": hardware["hardware_contract_sha256"],
        "calibration_capture_sha256": hardware["calibration_capture_sha256"],
        "dataset_tree_sha256": hardened["norm_source_tree_receipt"]["tree_sha256"],
        "dataset_info_sha256": hardened["dataset_info_sha256"],
        "episode_metadata_sha256": hardened["episode_metadata_sha256"],
        "train_only_norm_sha256": _sha_file(norm_path),
        "sidecar_meta_sha256": annotations["sidecar_meta_sha256"],
        "sidecar_bundle_sha256": annotations["sidecar_bundle_sha256"],
        "split_manifest_sha256": _sha_file(annotations["split_path"]),
        "coverage_csv_sha256": _sha_file(annotations["coverage_path"]),
        "segments_csv_sha256": _sha_file(annotations["segments_path"]),
        "inventory_csv_sha256": _sha_file(annotations["inventory_path"]),
        "build_config_sha256": _sha_file(annotations["config_path"]),
        "dataset_contact_coverage_receipt_sha256": coverage["sha256"],
        "oracle_sufficiency_receipt_sha256": oracle_sha,
        "matched_control_report_sha256": current_traces["report_sha256"],
        "matched_control_trace_manifest_sha256": current_traces["report"][
            "trace_manifest_sha256"
        ],
        "route_contract_bundle_sha256": route_bundle_sha,
        "runtime_environment_sha256": runtime_environment_sha,
        "dataset_runtime_source_receipt_sha256": runtime[
            "training_source_runtime_receipt_sha256"
        ],
        "t5_runtime_source_receipt_sha256": runtime[
            "t5_runtime_source_receipt_sha256"
        ],
        "hardened_preflight_receipt_sha256": hardened_artifact[
            "hardened_preflight_sha256"
        ],
    }
    legacy_root = Path(hardened_artifact["legacy_gate_path"]).parent / "legacy-evidence"
    reports = {
        "annotation_agreement": {
            "path": str(legacy_root / "annotation-agreement.json"),
            "sha256": _sha_file(legacy_root / "annotation-agreement.json"),
        },
        "hardware_verification": {
            "path": str(legacy_root / "hardware-verification.json"),
            "sha256": _sha_file(legacy_root / "hardware-verification.json"),
        },
        "dataset_contact_coverage": {
            "path": str(coverage["path"]),
            "sha256": coverage["sha256"],
        },
        "oracle_sufficiency": {"path": str(oracle_path), "sha256": oracle_sha},
        "ego_precursor": {
            "path": str(precursor["receipt_path"]),
            "sha256": precursor["receipt_bytes_sha256"],
        },
        "matched_control_hash_collector": {
            "path": str(current_traces["report_path"]),
            "sha256": current_traces["report_sha256"],
        },
        "gate_reviewer_acceptance": {
            "path": str(seed_root / "gate-review.json"),
            "sha256": _synthetic_sha("causal-gate-review-placeholder"),
        },
    }
    gate = {
        "gate_version": "contact-asym25-training-gate/v3",
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "precursor_id": PRECURSOR_ID,
        "route_set": list(CAUSAL_ROUTES),
        "collector_version": COLLECTOR_VERSION,
        "supervision_version": SUPERVISION_VERSION,
        "expected_git_commit": head,
        "evidence_valid_until_utc": SYNTHETIC_EXPIRY,
        **{name: True for name in REQUIRED_GATE_FLAGS},
        "artifact_fingerprints": fingerprints,
        "evidence_reports": reports,
        "training_run_id_by_route": {
            route: f"current-training-run-{seed}-{route}" for route in CAUSAL_ROUTES
        },
        "training_run_nonce_by_route": {
            route: f"current-training-nonce-{seed}-{route}" for route in CAUSAL_ROUTES
        },
        "external_acceptance_path_by_route": {
            route: str(seed_root / f"route-acceptance-{route}.json")
            for route in CAUSAL_ROUTES
        },
        "external_acceptance_sha256_by_route": {
            route: _synthetic_sha(f"causal-route-placeholder-{route}")
            for route in CAUSAL_ROUTES
        },
        "reviewed_by": f"causal-gate-reviewer-{seed}",
        "reviewed_at_utc": SYNTHETIC_REVIEWED_AT,
        "note": "synthetic CPU audit fixture",
    }
    route_acceptances = {}
    route_contract_sha = _route_acceptance_gate_contract_sha(gate)
    evidence_sha = {
        name: entry["sha256"]
        for name, entry in reports.items()
        if name != "gate_reviewer_acceptance"
    }
    for index, route in enumerate(CAUSAL_ROUTES):
        acceptance = {
            "acceptance_version": "contact-causal-route-training-acceptance/v1",
            "status": "accepted",
            "accepted": True,
            "matrix_schema_version": MATRIX_SCHEMA_VERSION,
            "matrix_id": MATRIX_ID,
            "causal_route_id": route,
            "training_run_id": gate["training_run_id_by_route"][route],
            "training_run_nonce": gate["training_run_nonce_by_route"][route],
            "reviewer_identity": f"route-reviewer-{seed}-{index}-identity",
            "reviewer_attestation_nonce": f"route-reviewer-{seed}-{index}-nonce",
            "acceptance_run_id": f"route-acceptance-{seed}-{index}-run-id",
            "acceptance_run_nonce": f"route-acceptance-{seed}-{index}-run-nonce",
            "evidence_valid_until_utc": SYNTHETIC_EXPIRY,
            "reviewed_at_utc": SYNTHETIC_REVIEWED_AT,
            "reviewed_gate_preregistered_contract_sha256": route_contract_sha,
            "reviewed_git_commit": head,
            "reviewed_code_blob_sha256": code_blobs,
            "runtime_environment_sha256": runtime_environment_sha,
            "reviewed_artifact_fingerprints": fingerprints,
            "reviewed_evidence_report_sha256": evidence_sha,
            "oracle_used": False,
            "trainer_self_signed": False,
            "production_data_admission": True,
            "allow_5b_training": True,
            "allow_server_training": True,
            "benefit_claimed": False,
        }
        path = Path(gate["external_acceptance_path_by_route"][route])
        digest = _write_json(path, acceptance)
        gate["external_acceptance_sha256_by_route"][route] = digest
        route_acceptances[route] = {
            "value": acceptance,
            "path": path,
            "sha256": digest,
        }
    review_contract = _gate_contract_sha(
        gate, review_report_name="gate_reviewer_acceptance"
    )
    review = {
        "acceptance_version": "contact-asym25-gate-reviewer-acceptance/v2",
        "status": "accepted",
        "accepted": True,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "route_set": list(CAUSAL_ROUTES),
        "reviewer_identity": f"causal-external-reviewer-{seed}-identity",
        "reviewer_attestation_nonce": f"causal-external-reviewer-{seed}-nonce",
        "acceptance_run_id": f"causal-external-acceptance-{seed}-run-id",
        "acceptance_run_nonce": f"causal-external-acceptance-{seed}-run-nonce",
        "evidence_valid_until_utc": SYNTHETIC_EXPIRY,
        "reviewed_at_utc": SYNTHETIC_REVIEWED_AT,
        "reviewed_gate_contract_sha256": review_contract,
        "reviewed_code_blob_sha256": code_blobs,
        "current_head_source_tree_sha256": current_source_sha256,
        "hardware_contract_sha256": fingerprints["hardware_contract_sha256"],
        "dataset_tree_sha256": fingerprints["dataset_tree_sha256"],
        "train_only_norm_sha256": fingerprints["train_only_norm_sha256"],
        "sidecar_bundle_sha256": fingerprints["sidecar_bundle_sha256"],
        "precursor_receipt_bytes_sha256": precursor["receipt_bytes_sha256"],
        "matched_control_report_sha256": current_traces["report_sha256"],
        "oracle_sufficiency_receipt_sha256": oracle_sha,
        "runtime_environment_sha256": runtime_environment_sha,
        "dataset_runtime_source_receipt_sha256": fingerprints[
            "dataset_runtime_source_receipt_sha256"
        ],
        "t5_runtime_source_receipt_sha256": fingerprints[
            "t5_runtime_source_receipt_sha256"
        ],
        "hardened_preflight_receipt_sha256": fingerprints[
            "hardened_preflight_receipt_sha256"
        ],
        "oracle_is_main_matrix_route": False,
    }
    review_path = seed_root / "gate-review.json"
    review_sha = _write_json(review_path, review)
    gate["evidence_reports"]["gate_reviewer_acceptance"]["sha256"] = review_sha
    gate_path = seed_root / "robot-gate.json"
    gate_sha = _write_json(gate_path, gate)
    validate_robot_gate_v3(gate, gate_path=gate_path, load_external_review=True)
    windows_sha = _synthetic_sha(f"causal-training-windows-{seed}")
    training_data = build_training_data_receipt_v2(
        gate=gate,
        gate_sha256=gate_sha,
        seed=seed,
        dataset_id=hardened["dataset_id"],
        train_episode_ids=hardened["train_episode_ids"],
        validation_episode_ids=hardened["validation_episode_ids"],
        heldout_episode_ids=hardened["heldout_episode_ids"],
        ordered_window_manifest_sha256=windows_sha,
        coverage_receipt=coverage["receipt"],
    )
    training_path = seed_root / "training-data.json"
    training_sha = _write_json(training_path, training_data)
    preflight = build_preflight_receipt_v2(
        gate=gate,
        gate_sha256=gate_sha,
        route=CAUSAL_ROUTES[0],
        training_data_receipt=training_data,
        current_head=head,
        source_tree_sha256=current_source_sha256,
    )
    preflight_path = seed_root / "preflight.json"
    preflight_sha = _write_json(preflight_path, preflight)
    return {
        "gate": gate,
        "gate_path": gate_path,
        "gate_sha256": gate_sha,
        "gate_review": review,
        "gate_review_path": review_path,
        "gate_review_sha256": review_sha,
        "route_acceptances": route_acceptances,
        "training_data": training_data,
        "training_data_path": training_path,
        "training_data_sha256": training_sha,
        "preflight": preflight,
        "preflight_path": preflight_path,
        "preflight_sha256": preflight_sha,
        "runtime_environment_sha256": runtime_environment_sha,
    }


def build_synthetic_causal_evaluator_results(
    fixture: Mapping[str, Any],
    *,
    checkpoint_evidence: Mapping[
        tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Build nine complete results and validate every result with file replay."""

    from world_action_model.contact_asym25_heldout_eval import (
        ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
        build_fixed_window_manifest,
        load_completed_group_split,
        public_tree_receipt,
        reduce_metric_rows,
        score_heldout_window,
        snapshot_artifact_tree,
        write_per_window_csv,
    )
    from world_action_model.contact_causal_checkpoint import (
        build_checkpoint_manifest_v3,
    )
    from world_action_model.contact_causal_heldout_eval import (
        CAUSAL_EVALUATION_CONTRACT,
        PREDICTION_EVIDENCE_FORMAT,
        ROUTE_OUTPUT_EVIDENCE_FORMAT,
        _array_stream_sha256,
        _recompute_route_packet_metrics,
        validate_result_v3,
        write_route_output_evidence,
    )
    from world_action_model.contact_causal_preflight import (
        validate_preflight_receipt_v2,
        validate_training_data_receipt_v2,
    )
    from world_action_model.contact_causal_production import (
        CAUSAL_ROUTES,
        CHECKPOINT_STEPS,
        EVALUATOR_VERSION,
        MATRIX_ID,
        MATRIX_SCHEMA_VERSION,
        PREFLIGHT_RECEIPT_VERSION,
        RESULT_VERSION,
        TRAINING_DATA_RECEIPT_VERSION,
        current_head_source_tree_receipt,
        route_packet_kind,
        sha256_json,
    )
    from world_action_model.contact_transition_schema import (
        EQUAL_BIT_NUM_BITS,
        EQUAL_BIT_VALID_STATES,
        unstructured_ids_to_code,
        validated_equal_bit_mixing_matrix,
    )

    repository_root = Path(fixture["repository_root"])
    fixture_root = Path(fixture["fixture_root"])
    source = current_head_source_tree_receipt(
        repository_root, expected_commit=fixture["git_commit"]
    )
    source_sha = sha256_json(source)
    base_root = fixture_root / "base-model"
    base_root.mkdir(exist_ok=True)
    _write_json(base_root / "config.json", {"_class_name": "SyntheticBaseModel"})
    base_receipt = public_tree_receipt(snapshot_artifact_tree(base_root))
    result_paths: dict[str, Path] = {}
    for seed in (6666, 7777, 8888):
        seed_root = fixture_root / f"seed-{seed}"
        seed_artifact = fixture["seed_artifacts"][seed]
        hardened = seed_artifact["hardened_preflight"]
        hardened_path = Path(seed_artifact["hardened_preflight_path"])
        hardened_sha = seed_artifact["hardened_preflight_sha256"]
        coverage = seed_artifact["coverage"]["receipt"]
        first_receipt = checkpoint_evidence[(seed, CAUSAL_ROUTES[0], 0)][0]
        training_data = {
            "receipt_version": TRAINING_DATA_RECEIPT_VERSION,
            "matrix_schema_version": MATRIX_SCHEMA_VERSION,
            "matrix_id": MATRIX_ID,
            "dataset_id": hardened["dataset_id"],
            "hardware_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
            "split_seed": seed,
            "train_episode_ids": hardened["train_episode_ids"],
            "validation_episode_ids": hardened["validation_episode_ids"],
            "heldout_episode_ids": hardened["heldout_episode_ids"],
            "ordered_window_manifest_sha256": _synthetic_sha(f"windows-{seed}"),
            "dataset_tree_sha256": hardened["norm_source_tree_receipt"]["tree_sha256"],
            "dataset_info_sha256": hardened["dataset_info_sha256"],
            "episode_metadata_sha256": hardened["episode_metadata_sha256"],
            "hardware_contract_sha256": hardened["hardware_contract_sha256"],
            "calibration_capture_sha256": hardened["calibration_capture_sha256"],
            "train_only_norm_sha256": hardened["norm_stats_sha256"],
            "sidecar_bundle_sha256": hardened["sidecar_bundle_sha256"],
            "split_manifest_sha256": hardened["split_manifest_sha256"],
            "dataset_contact_coverage_receipt": coverage,
            "dataset_contact_coverage_receipt_sha256": sha256_json(coverage),
            "training_gate_sha256": first_receipt["evidence"]["training_gate_sha256"],
            "precursor_receipt_bytes_sha256": first_receipt["precursor"][
                "receipt_bytes_sha256"
            ],
            "precursor_bundle_sha256": first_receipt["precursor"]["bundle_sha256"],
            "human_adjudicated_robot_native_labels": True,
            "hand_active_used_as_contact_ground_truth": False,
            "grip_used_as_contact_ground_truth": False,
            "train_only_normalization": True,
        }
        validate_training_data_receipt_v2(training_data)
        causal = {
            "receipt_version": PREFLIGHT_RECEIPT_VERSION,
            "status": "artifact_consistency_passed_with_explicit_external_gates",
            "matrix_schema_version": MATRIX_SCHEMA_VERSION,
            "matrix_id": MATRIX_ID,
            "route_set": list(CAUSAL_ROUTES),
            "precursor_id": "ego-contact-planner-precursor-q12/v1",
            "git_commit": fixture["git_commit"],
            "current_head_source_tree_sha256": source_sha,
            "training_gate_sha256": training_data["training_gate_sha256"],
            "training_data_receipt": training_data,
            "training_data_receipt_sha256": sha256_json(training_data),
            "matched_control_report_sha256": first_receipt["evidence"][
                "matched_control_receipt_sha256"
            ],
            "oracle_sufficiency_receipt_sha256": _synthetic_sha(
                f"oracle-sufficiency-{seed}"
            ),
            "contact_condition_mode": "predicted",
            "oracle_used": False,
            "production_data_admission": False,
            "allow_5b_training": False,
            "allow_server_training": False,
            "benefit_claimed": False,
        }
        validate_preflight_receipt_v2(causal)
        causal_path = seed_root / "causal-preflight.json"
        causal_sha = _write_json(causal_path, causal)
        _, assignments = load_completed_group_split(
            hardened["split_manifest_path"],
            expected_sha256=hardened["split_manifest_sha256"],
            expected_episode_lengths=hardened["episode_lengths"],
        )
        windows = build_fixed_window_manifest(
            preflight_receipt=hardened,
            preflight_receipt_sha256=hardened_sha,
            split_assignments=assignments,
        )
        windows_path = seed_root / "fixed-windows.json"
        windows_sha = _write_json(windows_path, windows)
        training_data["ordered_window_manifest_sha256"] = windows_sha
        causal["training_data_receipt"] = training_data
        causal["training_data_receipt_sha256"] = sha256_json(training_data)
        causal_sha = _write_json(causal_path, causal)
        window_rows = windows["windows"]
        from world_action_model.contact_causal_eval_runtime import (
            CausalMetricTargetReader,
        )

        target_cache = seed_root / ".synthetic_target_reader_cache"
        target_cache.mkdir(mode=0o700)
        target_reader = CausalMetricTargetReader(
            preflight_receipt=hardened,
            verified_private_cache_parent=str(target_cache),
            verified_run_nonce=f"synthetic-target-reader-seed-{seed}",
        )
        try:
            target_packets = [target_reader.load(row) for row in window_rows]
            target_source_receipt = dict(target_reader.source_column_receipt)
        finally:
            target_reader.close()
            target_cache.rmdir()
        target_once = np.stack(
            [packet["contact_target"].cpu().numpy() for packet in target_packets]
        ).astype(np.float32)
        mask_once = np.stack(
            [packet["contact_loss_mask"].cpu().numpy() for packet in target_packets]
        ).astype(np.float32)
        confidence_once = np.stack(
            [packet["contact_confidence"].cpu().numpy() for packet in target_packets]
        ).astype(np.float32)
        condition_once = np.stack(
            [packet["contact_condition_mask"].cpu().numpy() for packet in target_packets]
        ).astype(np.float32)
        row_count = len(window_rows) * len(CHECKPOINT_STEPS)
        evidence_steps = np.repeat(np.asarray(CHECKPOINT_STEPS, dtype=np.int64), 512)
        evidence_ids = [row["window_id"] for _ in CHECKPOINT_STEPS for row in window_rows]
        actions = np.zeros((row_count, 48, 32), dtype=np.float32)
        action_target_once = np.stack(
            [packet["action"].cpu().numpy() for packet in target_packets]
        ).astype(np.float32)
        dim_mask_once = np.stack(
            [packet["action_dim_mask"].cpu().numpy() for packet in target_packets]
        ).astype(np.bool_)
        time_mask_once = np.stack(
            [packet["action_time_mask"].cpu().numpy() for packet in target_packets]
        ).astype(np.bool_)
        scale_once = np.stack(
            [packet["action_denorm_scale"].cpu().numpy() for packet in target_packets]
        ).astype(np.float32)
        action_targets = np.tile(action_target_once, (3, 1, 1))
        dim_masks = np.tile(dim_mask_once, (3, 1))
        time_masks = np.tile(time_mask_once, (3, 1))
        scales = np.tile(scale_once, (3, 1))
        logits = np.zeros((row_count, 12, 22), dtype=np.float32)
        hard = np.zeros_like(logits)
        targets = np.tile(target_once, (3, 1, 1))
        masks = np.tile(mask_once, (3, 1, 1))
        confidences = np.tile(confidence_once, (3, 1, 1))
        conditions = np.tile(condition_once, (3, 1, 1))
        for route in CAUSAL_ROUTES:
            route_root = seed_root / route
            route_root.mkdir(parents=True, exist_ok=True)
            receipts_and_acceptances = {}
            for step in CHECKPOINT_STEPS:
                receipt, acceptance = checkpoint_evidence[(seed, route, step)]
                receipt_path = route_root / f"receipt-step-{step}.json"
                acceptance_path = route_root / f"acceptance-step-{step}.json"
                _write_json(receipt_path, receipt)
                if (
                    not acceptance_path.is_file()
                    or acceptance_path.read_bytes() != _canonical_bytes(acceptance)
                ):
                    raise AssertionError(
                        "independent checkpoint reviewer acceptance bytes differ"
                    )
                receipts_and_acceptances[step] = (
                    receipt_path,
                    receipt,
                    acceptance_path,
                    acceptance,
                )
            checkpoint_manifest = build_checkpoint_manifest_v3(
                receipts_and_acceptances=receipts_and_acceptances
            )
            checkpoint_manifest_path = route_root / "checkpoint-manifest.json"
            checkpoint_manifest_sha = _write_json(
                checkpoint_manifest_path, checkpoint_manifest
            )
            if route == "aux_only_direct_q12":
                packets = np.empty((row_count, 0, 22), dtype=np.float32)
            elif route == "hard_st_semantic":
                packets = hard.copy()
            else:
                matrix = validated_equal_bit_mixing_matrix().astype(np.float32)
                mixed = np.einsum("nqw,bw->nqb", logits, matrix, optimize=False)
                bits = mixed >= 0.0
                weights = 1 << np.arange(EQUAL_BIT_NUM_BITS, dtype=np.uint32)
                raw_ids = (bits.astype(np.uint32) * weights).sum(-1, dtype=np.uint32)
                canonical_ids = np.where(raw_ids >= EQUAL_BIT_VALID_STATES, 0, raw_ids)
                packets = unstructured_ids_to_code(canonical_ids.astype(np.uint16))
            synthetic_environment = {
                "python": "synthetic-cpu-audit-record",
                "platform": "synthetic-cpu-audit-record",
                "packages": {
                    "numpy": np.__version__,
                    "torch": "record-only",
                    "diffusers": "record-only",
                    "pandas": "record-only",
                },
                "cuda_record_only": {
                    "torch_cuda_version": "record-only",
                    "cudnn_version": 1,
                    "device_count": 1,
                },
                "evaluation_device": "cuda:0",
                "model_dtype": "bfloat16",
                "inference_steps": 1,
                "deterministic_algorithms": True,
                "augmentation_enabled": False,
                "eval_crop_mode": "center",
                "batch_size": 1,
            }
            prediction_source_receipt = {
                "reader_role": "prediction_inputs_only",
                "arrow_schema_columns": [
                    "index", "episode_index", "frame_index", "timestamp",
                    "task_index", "observation.state",
                ],
                "removed_target_columns": ["action"],
                "action_column_read": False,
                "contact_sidecar_read": False,
            }
            prediction_metadata = {
                "format": PREDICTION_EVIDENCE_FORMAT,
                "matrix_id": MATRIX_ID,
                "seed": seed,
                "causal_route_id": route,
                "causal_preflight_sha256": causal_sha,
                "hardened_preflight_sha256": hardened_sha,
                "checkpoint_manifest_sha256": checkpoint_manifest_sha,
                "fixed_window_manifest_sha256": windows_sha,
                "fixed_window_schedule_sha256": sha256_json(
                    [
                        {
                            "window_id": row["window_id"],
                            "global_index": row["global_index"],
                            "action_noise_seed": row["action_noise_seed"],
                        }
                        for row in window_rows
                    ]
                ),
                "checkpoint_bindings": {
                    str(step): {
                        "whole_bundle_sha256": checkpoint_evidence[
                            (seed, route, step)
                        ][0]["checkpoint_artifact_hashes"]["whole_bundle_sha256"],
                        "weight_tensor_sha256": checkpoint_evidence[
                            (seed, route, step)
                        ][0]["checkpoint_artifact_hashes"]["weight_tensor_sha256"],
                        "acceptance_sha256": sha256_json(
                            checkpoint_evidence[(seed, route, step)][1]
                        ),
                    }
                    for step in CHECKPOINT_STEPS
                },
                "base_model_artifact_tree": base_receipt,
                "environment_receipt": synthetic_environment,
                "evaluator_source_receipt_sha256": source_sha,
                "prediction_source_column_receipt": prediction_source_receipt,
                "target_columns_or_sidecars_read": False,
                "row_count": row_count,
            }
            prediction_buffer = io.BytesIO()
            np.savez(
                prediction_buffer,
                metadata_json=np.frombuffer(
                    json.dumps(
                        prediction_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8"),
                    dtype=np.uint8,
                ),
                checkpoint_steps=evidence_steps,
                window_ids=np.asarray(evidence_ids, dtype=np.str_),
                action_predictions=actions,
                semantic_aux_logits=logits,
                contact_hard_semantics=hard,
                route_packets=packets,
            )
            prediction_path = route_root / "predictions.npz"
            prediction_sha = hashlib.sha256(prediction_buffer.getvalue()).hexdigest()
            prediction_path.write_bytes(prediction_buffer.getvalue())
            evidence_path = route_root / "route-output.npz"
            evidence_sha = write_route_output_evidence(
                evidence_path,
                route=route,
                checkpoint_steps=evidence_steps,
                window_ids=evidence_ids,
                action_predictions=actions,
                action_targets=action_targets,
                action_dim_masks=dim_masks,
                action_time_masks=time_masks,
                action_denorm_scales=scales,
                semantic_aux_logits=logits,
                contact_hard_semantics=hard,
                contact_targets=targets,
                contact_loss_masks=masks,
                contact_confidences=confidences,
                contact_condition_masks=conditions,
                route_packets=packets,
            )
            csv_rows = []
            for row_index, (step, window) in enumerate(
                (
                    (step, window)
                    for step in CHECKPOINT_STEPS
                    for window in window_rows
                )
            ):
                receipt = checkpoint_evidence[(seed, route, step)][0]
                csv_rows.append(
                    score_heldout_window(
                        window=window,
                        split_seed=seed,
                        checkpoint_step=step,
                        checkpoint_sha256=receipt["checkpoint_artifact_hashes"][
                            "whole_bundle_sha256"
                        ],
                        action_prediction=actions[row_index],
                        action_target=action_targets[row_index],
                        action_dim_mask=dim_masks[row_index],
                        action_time_mask=time_masks[row_index],
                        action_denorm_scale=scales[row_index],
                        action_schema_contract=ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
                        grip_active_future=None,
                        contact_logits=logits[row_index],
                        contact_hard_semantic=hard[row_index],
                        contact_target=targets[row_index],
                        contact_loss_mask=masks[row_index],
                        contact_confidence=confidences[row_index],
                        contact_condition_mask=conditions[row_index],
                    )
                )
            csv_path = route_root / "per-window.csv"
            csv_sha = write_per_window_csv(csv_path, csv_rows, overwrite=False)
            action_contract_sha = sha256_json(ASYMMETRIC25_ACTION_SCHEMA_CONTRACT)
            contact_contract_sha = sha256_json(
                {
                    "source": "shared_semantic_aux_decoder_logits",
                    "q": 12,
                    "width": 22,
                    "b": 16,
                }
            )
            outputs = []
            manifest_by_step = {
                row["step"]: row for row in checkpoint_manifest["checkpoints"]
            }
            for step in CHECKPOINT_STEPS:
                selected_mask = evidence_steps == step
                reduced = {
                    split: reduce_metric_rows(
                        [
                            row
                            for row in csv_rows
                            if row["checkpoint_step"] == step and row["split"] == split
                        ]
                    )
                    for split in ("val", "test")
                }
                packet_metrics = _recompute_route_packet_metrics(
                    route=route,
                    logits=logits[selected_mask],
                    packets=packets[selected_mask],
                    reduced_contact_by_split={
                        split: reduced[split]["contact"] for split in ("val", "test")
                    },
                )
                outputs.append(
                    {
                        "checkpoint_step": step,
                        "checkpoint_acceptance_path_record_only": manifest_by_step[step][
                            "independent_acceptance_path"
                        ],
                        "checkpoint_acceptance_sha256": manifest_by_step[step][
                            "independent_acceptance_sha256"
                        ],
                        "fixed_window_set_sha256": windows_sha,
                        "route_output": {
                            "evaluator_version": EVALUATOR_VERSION,
                            "matrix_id": MATRIX_ID,
                            "causal_route_id": route,
                            "contact_aux_prediction_source": "shared_semantic_aux_decoder_logits",
                            "contact_target_model_input": False,
                            "contact_condition_mask_model_input": False,
                            "oracle_used": False,
                            "packet_kind": route_packet_kind(route),
                            "packet_present": route != "aux_only_direct_q12",
                            "packet_sha256": (
                                None
                                if route == "aux_only_direct_q12"
                                else _array_stream_sha256(
                                    [row["window_id"] for row in window_rows],
                                    packets[selected_mask],
                                )
                            ),
                            "semantic_aux_logits_sha256": _array_stream_sha256(
                                [row["window_id"] for row in window_rows],
                                logits[selected_mask],
                            ),
                            "action_prediction_sha256": _array_stream_sha256(
                                [row["window_id"] for row in window_rows],
                                actions[selected_mask],
                            ),
                            "action_metrics_contract_sha256": action_contract_sha,
                            "packet_metrics": packet_metrics,
                        },
                        "action_metrics": {
                            split: reduced[split]["action"] for split in ("val", "test")
                        },
                        "contact_aux_metrics": {
                            split: reduced[split]["contact"] for split in ("val", "test")
                        },
                    }
                )
            result = {
                "result_version": RESULT_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
                "matrix_schema_version": MATRIX_SCHEMA_VERSION,
                "matrix_id": MATRIX_ID,
                "seed": seed,
                "causal_route_id": route,
                "causal_preflight_receipt": {
                    "path_record_only": str(causal_path.absolute()),
                    "sha256": causal_sha,
                },
                "hardened_preflight_receipt": {
                    "path_record_only": str(hardened_path.absolute()),
                    "sha256": hardened_sha,
                },
                "checkpoint_manifest": {
                    "path_record_only": str(checkpoint_manifest_path.absolute()),
                    "sha256": checkpoint_manifest_sha,
                },
                "base_model_artifact_tree": base_receipt,
                "fixed_window_manifest": {
                    "path_record_only": str(windows_path.absolute()),
                    "sha256": windows_sha,
                },
                "per_window_csv": {
                    "path_record_only": str(csv_path.absolute()),
                    "sha256": csv_sha,
                    "row_count": len(csv_rows),
                },
                "prediction_evidence": {
                    "path_record_only": str(prediction_path.absolute()),
                    "sha256": prediction_sha,
                    "row_count": row_count,
                    "format": PREDICTION_EVIDENCE_FORMAT,
                },
                "prediction_source_column_receipt": prediction_source_receipt,
                "metric_target_source_receipt": target_source_receipt,
                "prediction_worker_exited_before_target_read": True,
                "route_output_evidence": {
                    "path_record_only": str(evidence_path.absolute()),
                    "sha256": evidence_sha,
                    "row_count": row_count,
                    "format": ROUTE_OUTPUT_EVIDENCE_FORMAT,
                },
                "evaluator_source_receipt_sha256": source_sha,
                "precursor_receipt_bytes_sha256": checkpoint_manifest[
                    "precursor_receipt_bytes_sha256"
                ],
                "action_metrics_contract_sha256": action_contract_sha,
                "contact_aux_metrics_contract_sha256": contact_contract_sha,
                "evaluation_contract": CAUSAL_EVALUATION_CONTRACT,
                "environment_receipt": synthetic_environment,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "primary_checkpoint_step": 200,
                "best_checkpoint_selection_allowed": False,
                "checkpoint_outputs": outputs,
                "oracle_used": False,
                "benefit_claimed": False,
            }
            validate_result_v3(result, verify_files=True)
            result_path = route_root / "result.json"
            _write_json(result_path, result)
            result_paths[f"{seed}/{route}"] = result_path
    return {"result_path_by_seed_route": result_paths}


def _run_coverage_tamper_negatives(fixture: Mapping[str, Any]) -> list[str]:
    import shutil

    from world_action_model.contact_causal_coverage import (
        recompute_dataset_supervision_coverage,
    )
    from world_action_model.contact_causal_production import CausalProductionError

    seed = 6666
    hardened = fixture["seed_artifacts"][seed]["hardened_preflight"]
    annotations = fixture["annotations"]
    base_kwargs = {
        "sidecar_dir": annotations["sidecar_dir"],
        "expected_meta_sha256": annotations["sidecar_meta_sha256"],
        "expected_episode_sha256": hardened["episode_npz_sha256"],
        "dataset_id": SYNTHETIC_DATASET_ID,
        "split_seed": seed,
        "split_episode_ids": {
            "train": hardened["train_episode_ids"],
            "val": hardened["validation_episode_ids"],
            "test": hardened["heldout_episode_ids"],
        },
    }

    def reject(name: str, kwargs: dict[str, Any]) -> None:
        try:
            recompute_dataset_supervision_coverage(**kwargs)
        except CausalProductionError:
            return
        raise AssertionError(f"coverage replay accepted tamper: {name}")

    reject(
        "meta_sha_drift",
        {**base_kwargs, "expected_meta_sha256": _synthetic_sha("meta drift")},
    )
    episode_drift = dict(hardened["episode_npz_sha256"])
    episode_drift[sorted(episode_drift)[0]] = _synthetic_sha("episode drift")
    reject(
        "episode_sha_drift",
        {**base_kwargs, "expected_episode_sha256": episode_drift},
    )
    reject(
        "split_overlap",
        {
            **base_kwargs,
            "split_episode_ids": {
                "train": hardened["train_episode_ids"],
                "val": hardened["validation_episode_ids"],
                "test": hardened["validation_episode_ids"],
            },
        },
    )
    negative_root = Path(fixture["fixture_root"]) / "coverage-negatives"
    for name, mode in (("proxy", "proxy"), ("all-zero", "all-zero")):
        root = negative_root / name
        shutil.copytree(annotations["sidecar_dir"], root)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        hashes = {}
        for filename in sorted(meta["episode_npz_sha256"]):
            path = root / filename
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            if mode == "proxy":
                arrays["hand_active_contact_proxy"] = np.zeros(
                    (arrays["contact_target"].shape[0],), dtype=np.float32
                )
            else:
                arrays = {key: np.zeros_like(value) for key, value in arrays.items()}
            with path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            hashes[filename] = _sha_file(path)
        meta["episode_npz_sha256"] = hashes
        _write_json(root / "meta.json", meta)
        reject(
            name,
            {
                **base_kwargs,
                "sidecar_dir": root,
                "expected_meta_sha256": _sha_file(root / "meta.json"),
                "expected_episode_sha256": hashes,
            },
        )
    no_physical = negative_root / "no-physical-positive"
    shutil.copytree(annotations["sidecar_dir"], no_physical)
    provenance = no_physical / "window_segment_provenance.csv"
    provenance.write_text(
        "episode_id,frame_index,query_index,segment_id,onset_event_id,release_event_id\n",
        encoding="utf-8",
    )
    meta = json.loads((no_physical / "meta.json").read_text(encoding="utf-8"))
    meta["window_segment_provenance_sha256"] = _sha_file(provenance)
    meta["unique_physical_segments"] = 0
    _write_json(no_physical / "meta.json", meta)
    reject(
        "no_physical_positive",
        {
            **base_kwargs,
            "sidecar_dir": no_physical,
            "expected_meta_sha256": _sha_file(no_physical / "meta.json"),
        },
    )
    return [
        "meta_sha_drift",
        "episode_sha_drift",
        "split_overlap",
        "hand_active_proxy_field",
        "all_zero_supervision",
        "no_physical_positive",
    ]


def build_synthetic_causal_admission_fixture(
    workspace: str | Path,
    *,
    repository_root: str | Path,
    checkpoint_evidence: Mapping[tuple[int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build the shared authoritative dataset/sidecar/norm base.

    Later stages add the legacy and causal gates to this returned directory.
    Every path is absolute so production validators cannot accidentally resolve
    evidence relative to the process working directory.
    """

    root = Path(repository_root).expanduser().absolute()
    fixture = Path(workspace).expanduser().absolute() / "synthetic-causal-admission"
    if fixture.exists():
        raise FileExistsError(fixture)
    fixture.mkdir(parents=True)
    hardware = _build_hardware(fixture, root)
    dataset = _build_dataset(fixture, hardware)
    annotations = _build_annotations_and_sidecars(fixture, root)
    norms = {
        seed: _build_norm(fixture, dataset, annotations, hardware, seed)
        for seed in (6666, 7777, 8888)
    }
    seed_artifacts = {
        seed: _build_legacy_gate_and_hardened(
            fixture,
            root,
            hardware,
            dataset,
            annotations,
            norms[seed],
            seed,
        )
        for seed in (6666, 7777, 8888)
    }
    for seed, artifacts in seed_artifacts.items():
        artifacts["coverage"] = _build_coverage(
            fixture / f"seed-{seed}",
            annotations=annotations,
            hardened=artifacts["hardened_preflight"],
            seed=seed,
        )
    runtime = _build_t5_and_runtime(
        fixture, seed_artifacts[6666]["hardened_preflight"]
    )
    precursor = None
    causal_by_seed = None
    checkpoint_reviewer_cli_count = 0
    checkpoint_reviewer_o_excl_rejected = False
    if checkpoint_evidence is not None:
        from world_action_model.contact_causal_matched_control import (
            build_current_run_trace_prefix_receipt_v1,
        )
        from world_action_model.contact_causal_production import (
            CAUSAL_ROUTES,
            CHECKPOINT_STEPS,
            current_head_source_tree_receipt,
            sha256_json,
        )

        first = checkpoint_evidence[(6666, "aux_only_direct_q12", 0)][0]
        precursor = _build_precursor(
            fixture,
            root,
            step0_parameter_sha256=first["checkpoint_artifact_hashes"][
                "weight_tensor_sha256"
            ],
        )
        source = current_head_source_tree_receipt(
            root, expected_commit=_git_head(root)
        )
        source_sha = sha256_json(source)
        matrix_sha = _sha_file(
            root / "configs/contact_causal_production_matrix.v1.json"
        )
        causal_by_seed = {}
        for seed in (6666, 7777, 8888):
            traces = _build_current_causal_traces(
                fixture / f"seed-{seed}",
                repository_root=root,
                seed=seed,
                source_sha256=source_sha,
                matrix_sha256=matrix_sha,
                hardened=seed_artifacts[seed]["hardened_preflight"],
                coverage_sha256=seed_artifacts[seed]["coverage"]["sha256"],
                runtime=runtime,
                precursor=precursor,
            )
            causal = _build_causal_gate(
                fixture / f"seed-{seed}",
                repository_root=root,
                seed=seed,
                hardened_artifact=seed_artifacts[seed],
                annotations=annotations,
                hardware=hardware,
                dataset=dataset,
                norm_path=norms[seed],
                coverage=seed_artifacts[seed]["coverage"],
                runtime=runtime,
                precursor=precursor,
                current_source_sha256=source_sha,
                current_traces=traces,
            )
            causal["current_traces"] = traces
            causal_by_seed[seed] = causal
            for route in CAUSAL_ROUTES:
                route_acceptance = causal["route_acceptances"][route]
                run_id = traces["report"]["run_id_by_route"][route]
                run_nonce = traces["report"]["run_nonce_by_route"][route]
                for step in CHECKPOINT_STEPS:
                    receipt, acceptance = checkpoint_evidence[(seed, route, step)]
                    receipt["run_identity"].update(
                        training_run_id=run_id,
                        training_run_nonce=run_nonce,
                        external_acceptance_sha256=route_acceptance["sha256"],
                    )
                    receipt["precursor"].update(
                        receipt_bytes_sha256=precursor["receipt_bytes_sha256"],
                        bundle_sha256=precursor["bundle_sha256"],
                        parameter_tensor_sha256=precursor["parameter_tensor_sha256"],
                    )
                    receipt["stable_contract"].update(
                        git_commit=_git_head(root),
                        source_tree_sha256=source_sha,
                        matrix_manifest_sha256=matrix_sha,
                        precursor_bundle_sha256=precursor["bundle_sha256"],
                        base_weights_sha256=precursor["parameter_tensor_sha256"],
                        initial_parameter_tensor_sha256=precursor[
                            "parameter_tensor_sha256"
                        ],
                        dataset_id=SYNTHETIC_DATASET_ID,
                        dataset_tree_sha256=seed_artifacts[seed][
                            "hardened_preflight"
                        ]["norm_source_tree_receipt"]["tree_sha256"],
                        dataset_info_sha256=seed_artifacts[seed][
                            "hardened_preflight"
                        ]["dataset_info_sha256"],
                        hardware_contract_sha256=hardware[
                            "hardware_contract_sha256"
                        ],
                        calibration_capture_sha256=hardware[
                            "calibration_capture_sha256"
                        ],
                        split_manifest_sha256=_sha_file(annotations["split_path"]),
                        sidecar_bundle_sha256=annotations["sidecar_bundle_sha256"],
                        contact_label_coverage_sha256=seed_artifacts[seed][
                            "coverage"
                        ]["sha256"],
                        runtime_environment_sha256=causal[
                            "runtime_environment_sha256"
                        ],
                        dataset_runtime_source_receipt_sha256=runtime[
                            "training_source_runtime_receipt_sha256"
                        ],
                        t5_runtime_source_receipt_sha256=runtime[
                            "t5_runtime_source_receipt_sha256"
                        ],
                        train_only_norm_sha256=_sha_file(norms[seed]),
                    )
                    receipt["evidence"].update(
                        training_gate_sha256=causal["gate_sha256"],
                        production_preflight_receipt_sha256=causal[
                            "preflight_sha256"
                        ],
                        hardened_preflight_receipt_sha256=seed_artifacts[seed][
                            "hardened_preflight_sha256"
                        ],
                        training_data_receipt_sha256=causal[
                            "training_data_sha256"
                        ],
                        matched_control_receipt_sha256=traces["report_sha256"],
                        external_reviewer_acceptance_sha256=causal[
                            "gate_review_sha256"
                        ],
                        contact_coverage_receipt_sha256=seed_artifacts[seed][
                            "coverage"
                        ]["sha256"],
                        trace_manifest_sha256=traces["report"][
                            "trace_manifest_sha256"
                        ],
                        current_head_source_receipt_sha256=source_sha,
                    )
                    prefix = build_current_run_trace_prefix_receipt_v1(
                        traces["records_by_route"][route],
                        route=route,
                        seed=seed,
                        run_id=run_id,
                        run_nonce=run_nonce,
                        rank=0,
                        world_size=1,
                        through_checkpoint_step=step,
                        trace_path=traces["trace_path_by_route"][route],
                    )
                    receipt["checkpoint_file"][
                        "current_run_trace_prefix_receipt"
                    ] = prefix
                    fixture_root_for_acceptance = Path(
                        receipt["checkpoint_file"]["path_record_only"]
                    ).parent
                    acceptance.update(
                        training_receipt_sha256=sha256_json(receipt),
                        robot_gate_path_record_only=str(causal["gate_path"]),
                        robot_gate_sha256=causal["gate_sha256"],
                        gate_reviewer_acceptance_path_record_only=str(
                            causal["gate_review_path"]
                        ),
                        gate_reviewer_acceptance_sha256=causal[
                            "gate_review_sha256"
                        ],
                        route_training_acceptance_path_record_only=str(
                            route_acceptance["path"]
                        ),
                        route_training_acceptance_sha256=route_acceptance["sha256"],
                        precursor_receipt_path_record_only=str(
                            precursor["receipt_path"]
                        ),
                        precursor_receipt_bytes_sha256=precursor[
                            "receipt_bytes_sha256"
                        ],
                        precursor_checkpoint_dir_record_only=str(
                            precursor["checkpoint_dir"]
                        ),
                        precursor_bundle_sha256=precursor["bundle_sha256"],
                        hardened_preflight_path_record_only=str(
                            seed_artifacts[seed]["hardened_preflight_path"]
                        ),
                        hardened_preflight_receipt_sha256=seed_artifacts[seed][
                            "hardened_preflight_sha256"
                        ],
                        production_preflight_path_record_only=str(
                            causal["preflight_path"]
                        ),
                        production_preflight_receipt_sha256=causal[
                            "preflight_sha256"
                        ],
                        training_data_path_record_only=str(
                            causal["training_data_path"]
                        ),
                        training_data_receipt_sha256=causal[
                            "training_data_sha256"
                        ],
                        coverage_path_record_only=str(
                            seed_artifacts[seed]["coverage"]["path"]
                        ),
                        contact_coverage_receipt_sha256=seed_artifacts[seed][
                            "coverage"
                        ]["sha256"],
                        sidecar_dir_record_only=str(annotations["sidecar_dir"]),
                        t5_embedding_dir_record_only=str(
                            runtime["t5_embedding_dir"]
                        ),
                        matched_control_report_path_record_only=str(
                            traces["report_path"]
                        ),
                        matched_control_report_sha256=traces["report_sha256"],
                        matched_control_trace_manifest_sha256=traces["report"][
                            "trace_manifest_sha256"
                        ],
                        matched_control_trace_path_by_route={
                            item: str(path)
                            for item, path in traces[
                                "trace_path_by_route"
                            ].items()
                        },
                        current_run_trace_prefix_receipt_sha256=prefix[
                            "receipt_sha256"
                        ],
                        current_head_source_receipt_sha256=source_sha,
                        reviewed_current_head_sha256=_git_head(root),
                        reviewed_source_tree_sha256=source_sha,
                    )
                    reviewer_output = (
                        fixture
                        / f"seed-{seed}"
                        / route
                        / f"acceptance-step-{step}.json"
                    )
                    reviewer_output.parent.mkdir(parents=True, exist_ok=True)
                    reviewed_acceptance = _run_checkpoint_reviewer_cli(
                        repository_root=root,
                        receipt=receipt,
                        acceptance_seed=acceptance,
                        causal=causal,
                        precursor=precursor,
                        seed_artifact=seed_artifacts[seed],
                        annotations=annotations,
                        runtime=runtime,
                        traces=traces,
                        output_path=reviewer_output,
                    )
                    acceptance.clear()
                    acceptance.update(reviewed_acceptance)
                    checkpoint_reviewer_cli_count += 1
                    if not checkpoint_reviewer_o_excl_rejected:
                        try:
                            _run_checkpoint_reviewer_cli(
                                repository_root=root,
                                receipt=receipt,
                                acceptance_seed=acceptance,
                                causal=causal,
                                precursor=precursor,
                                seed_artifact=seed_artifacts[seed],
                                annotations=annotations,
                                runtime=runtime,
                                traces=traces,
                                output_path=reviewer_output,
                            )
                        except subprocess.CalledProcessError:
                            checkpoint_reviewer_o_excl_rejected = True
                        else:
                            raise AssertionError(
                                "checkpoint reviewer overwrote an acceptance"
                            )
                    del fixture_root_for_acceptance
    return {
        "fixture_version": "contact-causal-synthetic-admission-fixture/v1",
        "repository_root": root,
        "fixture_root": fixture,
        "git_commit": _git_head(root),
        "hardware": hardware,
        "dataset": dataset,
        "annotations": annotations,
        "norm_path_by_seed": norms,
        "seed_artifacts": seed_artifacts,
        "runtime": runtime,
        "precursor": precursor,
        "causal_by_seed": causal_by_seed,
        "checkpoint_reviewer_cli_count": checkpoint_reviewer_cli_count,
        "checkpoint_reviewer_o_excl_rejected": (
            checkpoint_reviewer_o_excl_rejected
        ),
    }


def run_serialized_full_evaluator_acceptance(
    *, input_path: str | Path, output_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    """Labvla subprocess entry point for the complete file-backed acceptance."""

    import sys
    import types
    from importlib.machinery import ModuleSpec

    repository_root = Path(repository_root).expanduser().absolute()
    package_init_stubs = {
        "giga_datasets": repository_root
        / "third_party/giga-datasets/giga_datasets",
        "giga_datasets.datasets": repository_root
        / "third_party/giga-datasets/giga_datasets/datasets",
        "world_action_model.datasets": repository_root
        / "world_action_model/datasets",
        "world_action_model.transforms": repository_root
        / "world_action_model/transforms",
    }
    for name, source in package_init_stubs.items():
        if name in sys.modules:
            raise RuntimeError(
                f"full evaluator imported dependency package too early: {name}"
            )
        module = types.ModuleType(name)
        module.__path__ = [str(source)]
        sys.modules[name] = module
    if "giga_train" in sys.modules:
        raise RuntimeError("full evaluator imported giga_train too early")

    class _ImportOnlyRegistry:
        @staticmethod
        def register(value):
            return value

    giga_train_stub = types.ModuleType("giga_train")
    giga_train_stub.TRANSFORMS = _ImportOnlyRegistry()
    sys.modules["giga_train"] = giga_train_stub

    def _forbidden_optional_dependency(*_args, **_kwargs):
        raise RuntimeError(
            "an import-only optional dependency was invoked by the target path"
        )

    ftfy_stub = types.ModuleType("ftfy")
    ftfy_stub.__spec__ = ModuleSpec("ftfy", loader=None)
    ftfy_stub.fix_text = _forbidden_optional_dependency
    sys.modules["ftfy"] = ftfy_stub
    decord_stub = types.ModuleType("decord")
    decord_stub.__spec__ = ModuleSpec("decord", loader=None)
    decord_stub.VideoReader = _forbidden_optional_dependency
    sys.modules["decord"] = decord_stub
    video_utils_stub = types.ModuleType("giga_datasets.video_utils")

    def _forbidden_video_utils_attribute(_name):
        return _forbidden_optional_dependency

    video_utils_stub.__getattr__ = _forbidden_video_utils_attribute
    sys.modules["giga_datasets"].video_utils = video_utils_stub

    from world_action_model.contact_causal_heldout_eval import (
        build_aggregate_v5,
        validate_aggregate_v5,
        validate_result_v3,
    )
    from world_action_model.contact_causal_production import CausalProductionError

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    evidence = {
        (int(row["seed"]), row["route"], int(row["step"])): (
            row["receipt"], row["acceptance"]
        )
        for row in payload["evidence"]
    }
    fixture = build_synthetic_causal_admission_fixture(
        payload["workspace"],
        repository_root=repository_root,
        checkpoint_evidence=evidence,
    )
    coverage_negatives = _run_coverage_tamper_negatives(fixture)
    results = build_synthetic_causal_evaluator_results(
        fixture, checkpoint_evidence=evidence
    )
    aggregate = build_aggregate_v5(
        result_path_by_seed_route=results["result_path_by_seed_route"],
        evidence=evidence,
        verify_files=True,
    )
    validate_aggregate_v5(aggregate, evidence=evidence, verify_files=True)
    tampered = deepcopy(aggregate)
    tampered["result_file_by_seed_route"][
        "6666/equal_bit_unstructured_hard_st"
    ]["route_output_evidence_sha256"] = _synthetic_sha("tampered aggregate route output")
    try:
        validate_aggregate_v5(tampered, evidence=evidence, verify_files=True)
    except CausalProductionError:
        pass
    else:
        raise AssertionError("aggregate accepted tampered raw route-output evidence")
    first_result_path = results["result_path_by_seed_route"][
        "6666/equal_bit_unstructured_hard_st"
    ]
    first_result = json.loads(first_result_path.read_text(encoding="utf-8"))
    for name, mutate in (
        (
            "csv_sha",
            lambda value: value["per_window_csv"].update(
                sha256=_synthetic_sha("tampered CSV")
            ),
        ),
        (
            "npz_sha",
            lambda value: value["route_output_evidence"].update(
                sha256=_synthetic_sha("tampered NPZ")
            ),
        ),
        (
            "packet_digest",
            lambda value: value["checkpoint_outputs"][0]["route_output"].update(
                packet_sha256=_synthetic_sha("tampered packet digest")
            ),
        ),
    ):
        changed = deepcopy(first_result)
        mutate(changed)
        try:
            validate_result_v3(changed, verify_files=True)
        except CausalProductionError:
            pass
        else:
            raise AssertionError(f"result accepted tamper: {name}")
    acceptance_tamper_evidence = deepcopy(evidence)
    receipt, acceptance = acceptance_tamper_evidence[
        (6666, "equal_bit_unstructured_hard_st", 0)
    ]
    acceptance["matched_control_report_sha256"] = _synthetic_sha(
        "tampered acceptance report"
    )
    try:
        validate_aggregate_v5(
            aggregate,
            evidence=acceptance_tamper_evidence,
            verify_files=True,
        )
    except CausalProductionError:
        pass
    else:
        raise AssertionError("aggregate accepted tampered checkpoint acceptance")
    report = {
        "acceptance_version": "contact-causal-full-evaluator-cpu-acceptance/v1",
        "status": "passed",
        "result_count": 9,
        "checkpoint_acceptance_count": 27,
        "result_path_by_seed_route": {
            key: str(path) for key, path in results["result_path_by_seed_route"].items()
        },
        "aggregate_sha256": _sha_bytes(_canonical_bytes(aggregate)),
        "verify_files": True,
        "bound_acceptance_replay_count": 27,
        "checkpoint_reviewer_cli_count": fixture[
            "checkpoint_reviewer_cli_count"
        ],
        "checkpoint_reviewer_o_excl_rejected": fixture[
            "checkpoint_reviewer_o_excl_rejected"
        ],
        "tampered_route_output_rejected": True,
        "tampered_csv_rejected": True,
        "tampered_packet_digest_rejected": True,
        "tampered_acceptance_rejected": True,
        "coverage_negative_cases": coverage_negatives,
        "stub_scope": {
            name: "package_init_only; actual target/core submodules loaded from path"
            for name in package_init_stubs
        },
        "giga_train_stub_scope": (
            "TRANSFORMS registration import only; actual transform methods executed"
        ),
        "optional_import_stub_scope": {
            "ftfy": "precomputed-T5 scorer path; invocation raises",
            "decord": "skip_video_decoding target path; invocation raises",
            "giga_datasets.video_utils": (
                "no video decode/write in target scorer; invocation raises"
            ),
        },
    }
    _write_json(Path(output_path), report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialized-full-acceptance", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--repository-root")
    args = parser.parse_args()
    if not args.serialized_full_acceptance or not all(
        (args.input, args.output, args.repository_root)
    ):
        parser.error("serialized full acceptance requires input/output/repository-root")
    run_serialized_full_evaluator_acceptance(
        input_path=args.input,
        output_path=args.output,
        repository_root=args.repository_root,
    )


if __name__ == "__main__":
    _main()

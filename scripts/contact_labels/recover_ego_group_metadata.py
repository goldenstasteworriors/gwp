#!/usr/bin/env python3
"""Audit historical Ego files without inventing production group metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.ego_group_metadata_recovery import (  # noqa: E402
    DATASET_ID,
    RECOVERY_AUDIT_VERSION,
    REQUIRED_GROUP_FIELDS,
    canonical_sha256,
    make_unfilled_supplement,
    source_evidence_sha256,
)


EPISODE_RE = re.compile(r"^episode_(\d+)\.hdf5$")
CALIBRATION_KEYS = (
    "camera_backend",
    "color_intrinsic_cx",
    "color_intrinsic_cy",
    "color_intrinsic_fx",
    "color_intrinsic_fy",
    "color_intrinsic_height",
    "color_intrinsic_width",
    "color_distortion_model",
    "color_distortion_k1",
    "color_distortion_k2",
    "color_distortion_k3",
    "color_distortion_k4",
    "color_distortion_k5",
    "color_distortion_k6",
    "color_distortion_p1",
    "color_distortion_p2",
    "depth_intrinsic_cx",
    "depth_intrinsic_cy",
    "depth_intrinsic_fx",
    "depth_intrinsic_fy",
    "depth_intrinsic_height",
    "depth_intrinsic_width",
    "depth_distortion_model",
    "depth_distortion_k1",
    "depth_distortion_k2",
    "depth_distortion_k3",
    "depth_distortion_k4",
    "depth_distortion_k5",
    "depth_distortion_k6",
    "depth_distortion_p1",
    "depth_distortion_p2",
    "depth_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--processed-hdf5-root", required=True, type=Path)
    parser.add_argument("--final-hdf5-root", required=True, type=Path)
    parser.add_argument("--lerobot-root", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--supplement-template-output", required=True, type=Path)
    parser.add_argument(
        "--diagnostic-gap-seconds",
        type=float,
        default=30.0,
        help="Gap used only to derive non-production temporal continuity blocks.",
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _source_episode_paths(root: Path) -> list[tuple[int, Path]]:
    result = []
    for path in root.iterdir():
        match = EPISODE_RE.fullmatch(path.name)
        if match:
            result.append((int(match.group(1)), path))
    result.sort()
    if not result:
        raise FileNotFoundError(f"no source episode HDF5 under {root}")
    expected = list(range(result[-1][0] + 1))
    actual = [episode for episode, _ in result]
    if actual != expected:
        raise ValueError(f"source episodes are not contiguous: {actual}")
    return result


def _iso_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9).isoformat()


def _hdf5_schema(handle: h5py.File) -> tuple[set[str], set[str], set[str]]:
    root_attrs = set(handle.attrs)
    object_attrs: set[str] = set()
    datasets: set[str] = set()

    def visit(name: str, value: Any) -> None:
        object_attrs.update(value.attrs.keys())
        if isinstance(value, h5py.Dataset):
            datasets.add(name)

    handle.visititems(visit)
    return root_attrs, object_attrs, datasets


def build_audit(
    source_root: Path,
    processed_hdf5_root: Path,
    final_hdf5_root: Path,
    lerobot_root: Path,
    *,
    diagnostic_gap_seconds: float,
) -> dict[str, Any]:
    if diagnostic_gap_seconds <= 0:
        raise ValueError("diagnostic gap must be positive")
    for label, root in (
        ("source", source_root),
        ("processed_hdf5", processed_hdf5_root),
        ("final_hdf5", final_hdf5_root),
        ("lerobot", lerobot_root),
    ):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"{label} root must be a real directory: {root}")

    info_path = lerobot_root / "meta" / "info.json"
    episodes_path = lerobot_root / "meta" / "episodes.jsonl"
    tasks_path = lerobot_root / "meta" / "tasks.jsonl"
    info = _read_json(info_path)
    episode_meta = _read_jsonl(episodes_path)
    task_rows = _read_jsonl(tasks_path)
    meta_by_episode = {int(row["episode_index"]): row for row in episode_meta}
    if len(meta_by_episode) != len(episode_meta):
        raise ValueError("LeRobot episode metadata has duplicate IDs")

    episodes = []
    calibration_signatures: set[str] = set()
    previous_end_ns: int | None = None
    continuity_block = 0
    boundary_rows = []
    dataset_names: set[str] = set()
    camera_backends: set[str] = set()
    creation_dates: set[str] = set()
    schema_unions = {
        layer: {"root_attrs": set(), "object_attrs": set(), "datasets": set()}
        for layer in ("source", "depth", "processed", "downsample", "final")
    }
    for episode, source_path in _source_episode_paths(source_root):
        depth_path = source_root / f"episode_{episode}_depth.hdf5"
        processed_path = processed_hdf5_root / f"episode_{episode}.hdf5"
        downsample_path = processed_hdf5_root / f"downsample_episode_{episode}.hdf5"
        final_path = final_hdf5_root / f"episode_{episode}.hdf5"
        if any(
            not path.is_file()
            for path in (depth_path, processed_path, downsample_path, final_path)
        ):
            raise FileNotFoundError(
                f"episode {episode} lacks depth/processed/downsample/final HDF5"
            )
        with h5py.File(source_path, "r") as source:
            attrs = {key: _json_value(value) for key, value in source.attrs.items()}
            for key, values in zip(
                ("root_attrs", "object_attrs", "datasets"), _hdf5_schema(source)
            ):
                schema_unions["source"][key].update(values)
            if int(attrs.get("episode_index", -1)) != episode:
                raise ValueError(f"{source_path}: episode_index mismatch")
            timestamps = source.get("local_timestamps_ns")
            if timestamps is None or timestamps.ndim != 1 or len(timestamps) <= 0:
                raise ValueError(f"{source_path}: invalid local_timestamps_ns")
            start_ns = int(timestamps[0])
            end_ns = int(timestamps[-1])
            if start_ns <= 0 or end_ns < start_ns:
                raise ValueError(f"{source_path}: invalid capture clock")
            source_frames = int(attrs.get("total_frames", -1))
            if source_frames != int(timestamps.shape[0]):
                raise ValueError(f"{source_path}: total_frames mismatch")
            dataset_name = str(attrs.get("dataset_name", ""))
            camera_backend = str(attrs.get("camera_backend", ""))
            creation_time = str(attrs.get("creation_time", ""))
            dataset_names.add(dataset_name)
            camera_backends.add(camera_backend)
            creation_dates.add(creation_time[:10])
        with h5py.File(depth_path, "r") as depth:
            for key, values in zip(
                ("root_attrs", "object_attrs", "datasets"), _hdf5_schema(depth)
            ):
                schema_unions["depth"][key].update(values)
            calibration = {
                key: _json_value(depth.attrs[key])
                for key in CALIBRATION_KEYS
                if key in depth.attrs
            }
            if set(calibration) != set(CALIBRATION_KEYS):
                raise ValueError(f"{depth_path}: incomplete calibration attributes")
            calibration_signature = canonical_sha256(calibration)
            calibration_signatures.add(calibration_signature)
        with h5py.File(processed_path, "r") as processed:
            for key, values in zip(
                ("root_attrs", "object_attrs", "datasets"), _hdf5_schema(processed)
            ):
                schema_unions["processed"][key].update(values)
        with h5py.File(downsample_path, "r") as downsample:
            for key, values in zip(
                ("root_attrs", "object_attrs", "datasets"), _hdf5_schema(downsample)
            ):
                schema_unions["downsample"][key].update(values)
        with h5py.File(final_path, "r") as final:
            for key, values in zip(
                ("root_attrs", "object_attrs", "datasets"), _hdf5_schema(final)
            ):
                schema_unions["final"][key].update(values)
            if "action_eef" not in final:
                raise ValueError(f"{final_path}: missing action_eef")
            processed_length = int(final["action_eef"].shape[0])
        meta = meta_by_episode.get(episode)
        if not isinstance(meta, dict) or int(meta.get("length", -1)) != processed_length:
            raise ValueError(f"episode {episode}: final HDF5/LeRobot length mismatch")
        gap_seconds = None
        if previous_end_ns is not None:
            gap_seconds = (start_ns - previous_end_ns) / 1e9
            if gap_seconds < 0:
                raise ValueError(f"episode {episode}: capture clock overlaps previous episode")
            if gap_seconds > diagnostic_gap_seconds:
                continuity_block += 1
                boundary_rows.append(
                    {
                        "previous_episode_id": episode - 1,
                        "next_episode_id": episode,
                        "recorded_gap_seconds": gap_seconds,
                    }
                )
        previous_end_ns = end_ns
        episodes.append(
            {
                "episode_id": episode,
                "source_length_frames": source_frames,
                "processed_length_frames": processed_length,
                "capture_start_local_iso": _iso_from_ns(start_ns),
                "capture_end_local_iso": _iso_from_ns(end_ns),
                "capture_start_ns": start_ns,
                "capture_end_ns": end_ns,
                "recorded_gap_from_previous_seconds": gap_seconds,
                "diagnostic_continuity_block": continuity_block,
                "diagnostic_continuity_block_status": "inferred_not_production_gt",
                "source_creation_time": creation_time,
                "camera_calibration_sha256": calibration_signature,
            }
        )

    expected_episode_ids = set(meta_by_episode)
    actual_episode_ids = {row["episode_id"] for row in episodes}
    if expected_episode_ids != actual_episode_ids:
        raise ValueError("source/LeRobot episode sets differ")
    if info.get("total_episodes") != len(episodes):
        raise ValueError("LeRobot total_episodes mismatch")
    if info.get("total_frames") != sum(row["processed_length_frames"] for row in episodes):
        raise ValueError("LeRobot total_frames mismatch")
    if float(info.get("fps", -1)) != 20.0:
        raise ValueError("expected the processed Ego dataset at 20 Hz")
    task_strings = sorted(
        {
            task
            for row in episode_meta
            for task in row.get("tasks", [])
            if isinstance(task, str)
        }
        | {
            row["task"]
            for row in task_rows
            if isinstance(row.get("task"), str)
        }
    )
    schema_names = {
        name.split("/")[-1]
        for schema in schema_unions.values()
        for values in schema.values()
        for name in values
    }
    lerobot_row_keys = {
        key for row in episode_meta + task_rows for key in row
    } | set(info)
    observed_group_fields = sorted(
        set(REQUIRED_GROUP_FIELDS) & (schema_names | lerobot_row_keys)
    )
    if observed_group_fields:
        raise ValueError(
            "group fields now exist in source metadata and must be recovered explicitly: "
            f"{observed_group_fields}"
        )

    source_evidence = {
        "path_records": {
            "source_root": str(source_root.absolute()),
            "processed_hdf5_root": str(processed_hdf5_root.absolute()),
            "final_hdf5_root": str(final_hdf5_root.absolute()),
            "lerobot_root": str(lerobot_root.absolute()),
        },
        "episode_count": len(episodes),
        "processed_frame_count": sum(row["processed_length_frames"] for row in episodes),
        "dataset_names_observed": sorted(dataset_names),
        "task_strings_observed": task_strings,
        "capture_dates_observed": sorted(creation_dates),
        "camera_backends_observed": sorted(camera_backends),
        "camera_calibration_sha256_observed": sorted(calibration_signatures),
        "hdf5_schema_unions": {
            layer: {key: sorted(values) for key, values in schema.items()}
            for layer, schema in schema_unions.items()
        },
        "required_group_fields_observed_in_artifacts": observed_group_fields,
        "lerobot_fps_observed": float(info["fps"]),
        "diagnostic_gap_threshold_seconds": diagnostic_gap_seconds,
        "diagnostic_temporal_boundaries": boundary_rows,
        "diagnostic_continuity_block_count": continuity_block + 1,
        "diagnostic_temporal_block_disclaimer": (
            "Timestamp-derived continuity blocks are inferred diagnostics only; they are not "
            "session_id, collection_batch_id, subject_id, or production OOD ground truth."
        ),
        "episodes": episodes,
    }
    provenance_matrix = [
        {
            "field": "task_family",
            "status": "observed",
            "evidence": "source HDF5 dataset_name plus LeRobot tasks.jsonl/episodes.jsonl",
            "production_group_field": False,
        },
        {
            "field": "capture_time_and_date",
            "status": "observed",
            "evidence": "source HDF5 local_timestamps_ns and creation_time",
            "production_group_field": False,
        },
        {
            "field": "camera_backend_and_calibration",
            "status": "observed",
            "evidence": "source/depth HDF5 attributes; no camera serial is stored",
            "production_group_field": False,
        },
        {
            "field": "source_object_class,target_object_class",
            "status": "observed",
            "evidence": "task text names two beakers; it does not identify either instance",
            "production_group_field": False,
        },
        *[
            {
                "field": field,
                "status": "user-required",
                "evidence": (
                    "absent from source/depth/processed HDF5 and LeRobot metadata; "
                    "requires an experiment log, capture record, or collector attestation"
                ),
                "production_group_field": True,
            }
            for field in (
                "subject_id",
                "session_id",
                "operator_id",
                "source_object_instance_id",
                "target_object_instance_id",
                "liquid_id",
                "collection_batch_id",
            )
        ],
        {
            "field": "source_geometry_id,target_geometry_id,source_material,target_material",
            "status": "user-required",
            "evidence": (
                "absent from all audited artifacts; object appearance is not instance/material GT"
            ),
            "production_group_field": False,
        },
        {
            "field": "liquid_viscosity_mpas,liquid_coloring,initial_fill_fraction",
            "status": "user-required",
            "evidence": (
                "absent from all audited artifacts; RGB color is not chemical identity or viscosity GT"
            ),
            "production_group_field": False,
        },
        {
            "field": "task_success,spill_outcome,collision_outcome",
            "status": "user-required",
            "evidence": "absent from all audited artifacts and requires reviewed outcome evidence",
            "production_group_field": False,
        },
        {
            "field": "diagnostic_continuity_block",
            "status": "recoverable",
            "evidence": (
                "derived from recorded inter-episode clock gaps using an explicit threshold"
            ),
            "production_group_field": False,
        },
    ]
    return {
        "audit_version": RECOVERY_AUDIT_VERSION,
        "dataset_id": DATASET_ID,
        "source_evidence": source_evidence,
        "source_evidence_sha256": source_evidence_sha256(source_evidence),
        "provenance_matrix": provenance_matrix,
        "recovery_conclusion": {
            "production_group_metadata_complete": False,
            "ood_split_supported": False,
            "strategy_claim_allowed": False,
            "non_ood_split_boundary_available": "diagnostic_continuity_block",
            "reason": (
                "All 66 episodes have one observed task/date/camera calibration, while every "
                "required subject/session/operator/object-instance/liquid/batch ID is absent."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    audit = build_audit(
        args.source_root,
        args.processed_hdf5_root,
        args.final_hdf5_root,
        args.lerobot_root,
        diagnostic_gap_seconds=args.diagnostic_gap_seconds,
    )
    supplement = make_unfilled_supplement(audit)
    for path, payload in (
        (args.audit_output, audit),
        (args.supplement_template_output, supplement),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"episodes={audit['source_evidence']['episode_count']} "
        f"diagnostic_blocks={audit['source_evidence']['diagnostic_continuity_block_count']} "
        "production_ood=false"
    )


if __name__ == "__main__":
    main()

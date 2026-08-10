#!/usr/bin/env python3
"""Prepare blind A/B packets for the formal contact-label v1.1 minimum set.

This script inventories immutable source media and creates frame-level evidence
indices.  It never reads contact proposals and never writes contact labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(
    "/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/"
    "7_4_pour_liquid_between_beakers"
)
DEFAULT_PROTOCOL_ROOT = REPO_ROOT / "track_a" / "contact_labels_v1_1"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROTOCOL_ROOT / "formal_dual_annotation_v1_1"
DATASET_ID = "ego_pour_liquid_between_beakers"
SCHEMA_VERSION = "contact_segment_annotation_v1_1"
PACKET_SCHEMA_VERSION = "contact_dual_assignment_packet_v1_1"
FRAME_CLOCK = "processed_action_20hz"
TARGET_FPS = 20.0
MINIMUM_EPISODES = (43, 50, 51, 58, 34, 38, 9, 54, 36, 53, 12, 0)

COVERAGE_FIELDS = (
    "schema_version", "dataset_id", "episode_id", "coverage_id",
    "annotation_role", "annotator_id", "review_status", "coverage_mode",
    "start_frame", "end_frame", "frame_clock", "evidence_scope",
    "label_origin", "review_timestamp_utc", "parent_coverage_ids", "notes",
)
SEGMENT_FIELDS = (
    "schema_version", "dataset_id", "episode_id", "coverage_id", "segment_id",
    "onset_event_id", "release_event_id", "annotation_role", "annotator_id",
    "review_status", "start_frame", "end_frame", "start_boundary",
    "end_boundary", "side", "site", "object_role", "contact_mode",
    "confidence", "evidence_scope", "label_origin", "source_proposal_id",
    "parent_segment_ids", "notes",
)
PACKET_EVIDENCE_FIELDS = (
    "task_id", "episode_id", "processed_frame", "processed_time_s",
    "video_frame", "video_time_s", "sync_error_ms", "media_id",
    "media_relpath", "video_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite-empty-templates",
        action="store_true",
        help="replace repository mother templates; never use after identities or labels are assigned",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, object]],
    *,
    newline: str = "\n",
    utf8_bom: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig" if utf8_bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_inventory(path: Path) -> dict[int, int]:
    rows = read_csv(path)
    inventory = {int(row["episode_id"]): int(row["length_frames"]) for row in rows}
    if set(inventory) != set(range(66)):
        raise ValueError("authoritative inventory must contain exactly episodes 0..65")
    return inventory


def verify_minimum_set(path: Path, inventory: dict[int, int]) -> None:
    rows = read_csv(path)
    episodes = tuple(int(row["episode_id"]) for row in rows)
    if episodes != MINIMUM_EPISODES:
        raise ValueError(f"minimum proposal drift: expected {MINIMUM_EPISODES}, got {episodes}")
    for row in rows:
        episode = int(row["episode_id"])
        if int(row["length_frames"]) != inventory[episode] or row["proposal_only"] != "true":
            raise ValueError(f"minimum proposal row for episode {episode} is inconsistent")


def video_metadata(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    result = {
        "video_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 8),
        "video_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "video_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def inventory_sources(
    source_root: Path,
    inventory: dict[int, int],
    merge_report: dict,
) -> list[dict[str, object]]:
    report_by_episode = {int(row["episode"]): row for row in merge_report["episodes"]}
    rows: list[dict[str, object]] = []
    for episode in sorted(inventory):
        processed = source_root / "vla_processed" / "final_hdf5" / f"episode_{episode}.hdf5"
        video = source_root / f"episode_{episode}.mp4"
        depth = source_root / f"episode_{episode}_depth.hdf5"
        for path in (processed, video, depth):
            if not path.is_file():
                raise FileNotFoundError(path)
        with h5py.File(processed, "r") as handle:
            processed_frames = int(handle["action_eef"].shape[0])
            camera_frames = int(handle["camera_timestamp"].shape[0])
        if processed_frames != inventory[episode] or camera_frames != processed_frames:
            raise ValueError(f"episode {episode}: processed length disagrees with inventory")
        video_meta = video_metadata(video)
        report = report_by_episode[episode]
        if int(report["frames"]) != processed_frames or int(report["video_frames"]) != video_meta["video_frames"]:
            raise ValueError(f"episode {episode}: merge report disagrees with source files")
        # Source receipts are content receipts. Same-size files are not
        # interchangeable, so every preparation run recomputes their hashes.
        processed_hash = sha256_file(processed)
        video_hash = sha256_file(video)
        depth_hash = sha256_file(depth) if episode in MINIMUM_EPISODES else ""
        rows.append({
            "episode_id": episode,
            "assignment_selected": str(episode in MINIMUM_EPISODES).lower(),
            "processed_frames": processed_frames,
            "processed_hdf5_path": str(processed),
            "processed_hdf5_bytes": processed.stat().st_size,
            "processed_hdf5_sha256": processed_hash,
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "video_sha256": video_hash,
            **video_meta,
            "depth_timestamps_path": str(depth),
            "depth_timestamps_bytes": depth.stat().st_size,
            "depth_timestamps_sha256": depth_hash,
            "depth_hash_scope": "selected_episode" if depth_hash else "not_required_for_unassigned_episode",
            "mean_sync_error_ms": round(float(report["mean_time_diff_ms"]), 6),
            "max_sync_error_ms": round(float(report["max_time_diff_ms"]), 6),
            "sync_warnings": int(report["warnings"]),
            "sync_severe_warnings": int(report["severe_warnings"]),
        })
    return rows


def nearest_video_indices(camera_ns: np.ndarray, color_us: np.ndarray) -> np.ndarray:
    color_ns = color_us.astype(np.int64) * 1000
    insertion = np.searchsorted(color_ns, camera_ns.astype(np.int64))
    insertion = np.clip(insertion, 0, len(color_ns) - 1)
    previous = np.clip(insertion - 1, 0, len(color_ns) - 1)
    use_previous = np.abs(color_ns[previous] - camera_ns) <= np.abs(color_ns[insertion] - camera_ns)
    return np.where(use_previous, previous, insertion)


def build_timeline(source_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_episode = {int(row["episode_id"]): row for row in source_rows}
    evidence: list[dict[str, object]] = []
    for task_index, episode in enumerate(MINIMUM_EPISODES):
        source = rows_by_episode[episode]
        task_id = f"contact-min12-{task_index:02d}"
        with h5py.File(str(source["processed_hdf5_path"]), "r") as handle:
            camera_ns = np.asarray(handle["camera_timestamp"], dtype=np.int64)
        with h5py.File(str(source["depth_timestamps_path"]), "r") as handle:
            color_us = np.asarray(handle["color_system_timestamps_us"], dtype=np.int64)
        video_indices = nearest_video_indices(camera_ns, color_us)
        for processed_frame, video_frame in enumerate(video_indices):
            video_frame = int(video_frame)
            sync_error_ms = abs(int(camera_ns[processed_frame]) - int(color_us[video_frame]) * 1000) / 1e6
            evidence.append({
                "task_id": task_id,
                "episode_id": episode,
                "processed_frame": processed_frame,
                "processed_time_s": f"{processed_frame / TARGET_FPS:.6f}",
                "camera_timestamp_ns": int(camera_ns[processed_frame]),
                "video_frame": video_frame,
                "video_time_s": f"{video_frame / float(source['video_fps']):.6f}",
                "video_timestamp_us": int(color_us[video_frame]),
                "sync_error_ms": f"{sync_error_ms:.6f}",
                "media_id": f"contact-media-{task_index:02d}",
                "media_relpath": f"media/{task_id}.mp4",
                "video_path": source["video_path"],
                "processed_hdf5_path": source["processed_hdf5_path"],
                "video_sha256": source["video_sha256"],
                "processed_hdf5_sha256": source["processed_hdf5_sha256"],
            })
    return evidence


def packet_tasks(inventory: dict[int, int]) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"contact-min12-{index:02d}",
            "episode_id": episode,
            "coverage_start_frame": 0,
            "coverage_end_frame": inventory[episode],
            "frame_clock": FRAME_CLOCK,
            "target_fps": f"{TARGET_FPS:.1f}",
            "context_frames_each_side": 10,
            "required_coverage_review": "full_episode",
        }
        for index, episode in enumerate(MINIMUM_EPISODES)
    ]


def create_packet(
    root: Path,
    role_letter: str,
    tasks_path: Path,
    evidence_path: Path,
    inventory: dict[int, int],
    coordinator_audit_path: Path,
    annotation_guide_path: Path,
) -> None:
    role_lower = role_letter.lower()
    role = f"independent_{role_lower}"
    packet_id = f"contact-v1.1-{role_lower}-20260810-min12"
    annotator_id = f"UNASSIGNED_ANNOTATOR_{role_letter}"
    session_id = f"UNASSIGNED_SESSION_{role_letter}"
    packet_root = root / f"annotator_{role_lower}"
    packet_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(annotation_guide_path, packet_root / "ANNOTATION_GUIDE.md")
    shutil.copyfile(tasks_path, packet_root / "tasks.csv")
    blind_evidence_rows = [
        {field: row[field] for field in PACKET_EVIDENCE_FIELDS}
        for row in read_csv(evidence_path)
    ]
    write_csv(
        packet_root / "timeline_evidence_index.csv",
        PACKET_EVIDENCE_FIELDS,
        blind_evidence_rows,
    )
    identity = {
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": packet_id,
        "annotation_role": role,
        "annotator_id": annotator_id,
        "annotation_session_id": session_id,
        "identity_status": "must_be_replaced_by_coordinator_before_distribution",
        "tasks_sha256": sha256_file(packet_root / "tasks.csv"),
        "timeline_evidence_sha256": sha256_file(packet_root / "timeline_evidence_index.csv"),
        "coordinator_audit_sha256": sha256_file(coordinator_audit_path),
        "annotation_guide_sha256": sha256_file(packet_root / "ANNOTATION_GUIDE.md"),
        "proposal_material_included": False,
        "other_annotator_material_included": False,
    }
    (packet_root / "identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    coverage_rows = []
    for index, episode in enumerate(MINIMUM_EPISODES):
        coverage_rows.append({
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "episode_id": episode,
            "coverage_id": f"{role_lower}-cov-e{episode:03d}-full",
            "annotation_role": role,
            "annotator_id": annotator_id,
            "review_status": "in_progress",
            "coverage_mode": "",
            "start_frame": 0,
            "end_frame": inventory[episode],
            "frame_clock": FRAME_CLOCK,
            "evidence_scope": "",
            "label_origin": "human_independent",
            "review_timestamp_utc": "",
            "parent_coverage_ids": "",
            "notes": f"task_id=contact-min12-{index:02d}; no label has been prefilled",
        })
    write_csv(packet_root / "coverage.csv", COVERAGE_FIELDS, coverage_rows)
    # A UTF-8 BOM distinguishes the physically independent B empty file without
    # trailing whitespace; read_csv_exact canonicalizes utf-8-sig input.
    write_csv(
        packet_root / "segments.csv",
        SEGMENT_FIELDS,
        [],
        utf8_bom=role_letter == "B",
    )
    receipt = {
        "packet_id": packet_id,
        "identity_sha256": sha256_file(packet_root / "identity.json"),
        "coverage_sha256": sha256_file(packet_root / "coverage.csv"),
        "segments_sha256": sha256_file(packet_root / "segments.csv"),
        "tasks_sha256": sha256_file(packet_root / "tasks.csv"),
        "timeline_evidence_sha256": sha256_file(packet_root / "timeline_evidence_index.csv"),
        "coordinator_audit_sha256": sha256_file(coordinator_audit_path),
        "annotation_guide_sha256": sha256_file(packet_root / "ANNOTATION_GUIDE.md"),
        "packet_file_allowlist": [
            "ANNOTATION_GUIDE.md",
            "assignment_receipt.json",
            "coverage.csv",
            "identity.json",
            "segments.csv",
            "tasks.csv",
            "timeline_evidence_index.csv",
        ],
        "optional_media_relpaths": [
            f"media/contact-min12-{index:02d}.mp4"
            for index in range(len(MINIMUM_EPISODES))
        ],
    }
    (packet_root / "assignment_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_adjudication_template(root: Path, inventory: dict[int, int]) -> None:
    destination = root / "adjudication"
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "adjudicator_id": "UNASSIGNED_ADJUDICATOR",
        "adjudication_session_id": "UNASSIGNED_ADJUDICATION_SESSION",
        "identity_status": "must_be_replaced_after_agreement_gates_pass",
        "must_be_distinct_from_annotator_a_and_b": True,
    }
    (destination / "identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for index, episode in enumerate(MINIMUM_EPISODES):
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "episode_id": episode,
            "coverage_id": f"adj-cov-e{episode:03d}-full",
            "annotation_role": "adjudicated",
            "annotator_id": "UNASSIGNED_ADJUDICATOR",
            "review_status": "in_progress",
            "coverage_mode": "",
            "start_frame": 0,
            "end_frame": inventory[episode],
            "frame_clock": FRAME_CLOCK,
            "evidence_scope": "",
            "label_origin": "human_adjudicated",
            "review_timestamp_utc": "",
            "parent_coverage_ids": f"a-cov-e{episode:03d}-full;b-cov-e{episode:03d}-full",
            "notes": f"task_id=contact-min12-{index:02d}; fill only after all agreement gates pass",
        })
    write_csv(destination / "coverage.csv", COVERAGE_FIELDS, rows)
    write_csv(destination / "segments.csv", SEGMENT_FIELDS, [])


def main() -> None:
    args = parse_args()
    protected = [
        args.output_root / f"annotator_{letter}" / name
        for letter in ("a", "b")
        for name in ("identity.json", "coverage.csv", "segments.csv")
    ]
    if any(path.exists() for path in protected) and not args.overwrite_empty_templates:
        raise FileExistsError(
            "packet annotation files already exist; refusing to overwrite identities/labels. "
            "Use a fresh --output-root, or --overwrite-empty-templates only for verified blank mother templates."
        )
    inventory_path = args.protocol_root / "episode_inventory.csv"
    minimum_path = args.protocol_root / "minimum_annotation_set.csv"
    annotation_guide_path = (
        args.protocol_root / "formal_dual_annotation_v1_1" / "LABEL_DICTIONARY.md"
    )
    merge_report_path = args.source_root / "vla_processed" / "final_hdf5" / "merge_report.json"
    inventory = read_inventory(inventory_path)
    verify_minimum_set(minimum_path, inventory)
    merge_report = json.loads(merge_report_path.read_text(encoding="utf-8"))
    if int(merge_report.get("processed_episodes", -1)) != 66 or int(merge_report.get("failed_episodes", -1)) != 0:
        raise ValueError("merge report is not the expected successful 66-episode corpus")
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = args.output_root / "coordinator" / "source_episode_manifest.csv"
    source_rows = inventory_sources(
        args.source_root,
        inventory,
        merge_report,
    )
    source_fields = tuple(source_rows[0].keys())
    write_csv(source_manifest, source_fields, source_rows)
    timeline_rows = build_timeline(source_rows)
    timeline_fields = tuple(timeline_rows[0].keys())
    coordinator_timeline = args.output_root / "coordinator" / "timeline_evidence_index.csv"
    write_csv(coordinator_timeline, timeline_fields, timeline_rows)
    tasks = packet_tasks(inventory)
    task_fields = tuple(tasks[0].keys())
    tasks_path = args.output_root / "coordinator" / "tasks.csv"
    write_csv(tasks_path, task_fields, tasks)
    audit = {
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "canonical_commit": "913811dab0a7ee74a0493f19d7bab40225546d99",
        "episodes_audited": len(source_rows),
        "processed_frames_audited": sum(int(row["processed_frames"]) for row in source_rows),
        "minimum_tasks": len(tasks),
        "minimum_processed_frames": len(timeline_rows),
        "source_manifest_sha256": sha256_file(source_manifest),
        "timeline_evidence_sha256": sha256_file(coordinator_timeline),
        "tasks_sha256": sha256_file(tasks_path),
        "inventory_sha256": sha256_file(inventory_path),
        "minimum_proposal_sha256_coordinator_only": sha256_file(minimum_path),
        "merge_report_sha256": sha256_file(merge_report_path),
        "annotation_guide_sha256": sha256_file(annotation_guide_path),
        "proposal_material_copied_to_packets": False,
        "thumbnail_assets_generated": False,
    }
    coordinator_audit = args.output_root / "coordinator" / "audit.json"
    coordinator_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    create_packet(
        args.output_root, "A", tasks_path, coordinator_timeline, inventory,
        coordinator_audit, annotation_guide_path,
    )
    create_packet(
        args.output_root, "B", tasks_path, coordinator_timeline, inventory,
        coordinator_audit, annotation_guide_path,
    )
    create_adjudication_template(args.output_root, inventory)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

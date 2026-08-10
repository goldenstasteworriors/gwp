#!/usr/bin/env python3
"""Validate assignment, independent submissions, agreement, and adjudication.

The checker is deliberately fail-closed.  It never upgrades proposals or empty
templates to labels and it cannot certify that two humans worked independently;
it only verifies the auditable packet/session/file invariants and quality gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Iterable

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_ROOT = (
    REPO_ROOT / "track_a" / "contact_labels_v1_1" / "formal_dual_annotation_v1_1"
)
PACKET_SCHEMA_VERSION = "contact_dual_assignment_packet_v1_1"
ANNOTATION_SCHEMA_VERSION = "contact_segment_annotation_v1_1"
DATASET_ID = "ego_pour_liquid_between_beakers"
FRAME_CLOCK = "processed_action_20hz"
CANONICAL_COMMIT = "913811dab0a7ee74a0493f19d7bab40225546d99"
MINIMUM_EPISODES = (43, 50, 51, 58, 34, 38, 9, 54, 36, 53, 12, 0)
UNKNOWN_PREFIXES = ("UNASSIGNED_", "REPLACE_", "PLACEHOLDER_")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")

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
TASK_FIELDS = (
    "task_id", "episode_id", "coverage_start_frame", "coverage_end_frame",
    "frame_clock", "target_fps", "context_frames_each_side",
    "required_coverage_review",
)
EVIDENCE_FIELDS = (
    "task_id", "episode_id", "processed_frame", "processed_time_s",
    "video_frame", "video_time_s", "sync_error_ms", "media_id",
    "media_relpath", "video_sha256",
)
PACKET_FILE_ALLOWLIST = {
    "ANNOTATION_GUIDE.md",
    "assignment_receipt.json",
    "coverage.csv",
    "identity.json",
    "segments.csv",
    "tasks.csv",
    "timeline_evidence_index.csv",
}
IDENTITY_FIELDS = {
    "packet_schema_version",
    "packet_id",
    "annotation_role",
    "annotator_id",
    "annotation_session_id",
    "identity_status",
    "tasks_sha256",
    "timeline_evidence_sha256",
    "coordinator_audit_sha256",
    "annotation_guide_sha256",
    "proposal_material_included",
    "other_annotator_material_included",
}
ASSIGNMENT_RECEIPT_FIELDS = {
    "packet_id",
    "identity_sha256",
    "coverage_sha256",
    "segments_sha256",
    "tasks_sha256",
    "timeline_evidence_sha256",
    "coordinator_audit_sha256",
    "annotation_guide_sha256",
    "packet_file_allowlist",
    "optional_media_relpaths",
}
ALLOWED_COVERAGE = {"unknown", "positive_only", "exhaustive"}
ALLOWED_EVIDENCE = {"rgb_direct", "rgb_temporal", "rgb_depth_temporal", "sensor_confirmed", "unknown"}
ALLOWED_START = {"observed", "left_censored", "unknown"}
ALLOWED_END = {"observed", "right_censored", "unknown"}
ALLOWED_CATEGORY = {
    "side": {"", "left", "right", "object_object"},
    "site": {"", "opposition", "primary", "support", "palm_or_jaw", "tool_tip", "other_surface"},
    "object_role": {"", "manipulated", "tool", "receptacle", "fixture"},
    "contact_mode": {"", "stick", "slide", "roll", "impact"},
}
THRESHOLDS = {
    "coverage_weighted_kappa_min": 0.70,
    "event_f1_min": 0.80,
    "matched_median_tiou_min": 0.70,
    "observed_boundary_median_mae_frames_max": 2.0,
    "categorical_kappa_min": 0.60,
}


@dataclass(frozen=True)
class Segment:
    episode: int
    coverage_id: str
    segment_id: str
    start: int
    end: int
    start_boundary: str
    end_boundary: str
    side: str
    site: str
    object_role: str
    contact_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--phase", choices=("assignment", "submission", "agreement", "adjudication"), required=True)
    parser.add_argument("--agreement-report", type=Path)
    parser.add_argument("--adjudication-root", type=Path)
    parser.add_argument("--write-report", type=Path)
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_exact(path: Path, fields: Iterable[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = list(fields)
        if reader.fieldnames != expected:
            fail(f"{path}: header mismatch; expected={expected}, actual={reader.fieldnames}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def read_identity(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != IDENTITY_FIELDS or value.get("packet_schema_version") != PACKET_SCHEMA_VERSION:
        fail(f"{path}: packet schema mismatch")
    return value


def ensure_independent_files(path_a: Path, path_b: Path, *, allow_equal_hash: bool = False) -> None:
    if path_a.resolve() == path_b.resolve():
        fail(f"A/B resolve to the same path: {path_a}")
    stat_a, stat_b = path_a.stat(), path_b.stat()
    if (stat_a.st_dev, stat_a.st_ino) == (stat_b.st_dev, stat_b.st_ino):
        fail(f"A/B are the same inode or hard link: {path_a} / {path_b}")
    if path_a.is_symlink() or path_b.is_symlink():
        fail("A/B packet files must not be symlinks")
    if not allow_equal_hash and sha256_file(path_a) == sha256_file(path_b):
        fail(f"A/B annotation files have identical SHA-256: {path_a.name}")


def validate_iso_utc(value: str, field: str) -> None:
    if not value.endswith("Z"):
        fail(f"{field} must be a UTC ISO-8601 timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} is not ISO-8601: {value!r}") from exc


def validate_sources(root: Path) -> dict[int, dict[str, str]]:
    path = root / "coordinator" / "source_episode_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 66 or {int(row["episode_id"]) for row in rows} != set(range(66)):
        fail("source manifest must contain exactly episodes 0..65")
    result = {}
    for row in rows:
        episode = int(row["episode_id"])
        for kind, size_field, hash_field in (
            ("processed_hdf5", "processed_hdf5_bytes", "processed_hdf5_sha256"),
            ("video", "video_bytes", "video_sha256"),
        ):
            source_path = Path(row[f"{kind}_path"])
            if not source_path.is_file() or source_path.stat().st_size != int(row[size_field]):
                fail(f"episode {episode}: missing or resized {kind} source")
            if not re.fullmatch(r"[0-9a-f]{64}", row[hash_field]):
                fail(f"episode {episode}: missing {kind} SHA-256")
            if sha256_file(source_path) != row[hash_field]:
                fail(f"episode {episode}: {kind} SHA-256 mismatch")
        if row["assignment_selected"] == "true":
            depth = Path(row["depth_timestamps_path"])
            if not depth.is_file() or depth.stat().st_size != int(row["depth_timestamps_bytes"]):
                fail(f"episode {episode}: missing or resized depth timestamp source")
            if not re.fullmatch(r"[0-9a-f]{64}", row["depth_timestamps_sha256"]):
                fail(f"episode {episode}: selected depth timestamp source lacks SHA-256")
            if sha256_file(depth) != row["depth_timestamps_sha256"]:
                fail(f"episode {episode}: depth timestamp source SHA-256 mismatch")
        result[episode] = row
    return result


def validate_coordinator_receipt(root: Path, sources: dict[int, dict[str, str]]) -> dict:
    audit_path = root / "coordinator" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "source_manifest_sha256": root / "coordinator" / "source_episode_manifest.csv",
        "timeline_evidence_sha256": root / "coordinator" / "timeline_evidence_index.csv",
        "tasks_sha256": root / "coordinator" / "tasks.csv",
        "inventory_sha256": root.parent / "episode_inventory.csv",
        "minimum_proposal_sha256_coordinator_only": root.parent / "minimum_annotation_set.csv",
    }
    if (
        audit.get("packet_schema_version") != PACKET_SCHEMA_VERSION
        or audit.get("canonical_commit") != CANONICAL_COMMIT
        or audit.get("episodes_audited") != 66
        or audit.get("processed_frames_audited")
        != sum(int(row["processed_frames"]) for row in sources.values())
        or audit.get("minimum_tasks") != 12
        or not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("annotation_guide_sha256", "")))
        or audit.get("proposal_material_copied_to_packets") is not False
        or audit.get("thumbnail_assets_generated") is not False
    ):
        fail("coordinator audit receipt metadata differs")
    for field, path in expected_hashes.items():
        if audit.get(field) != sha256_file(path):
            fail(f"coordinator audit receipt mismatch for {path.name}")
    first_processed = Path(sources[0]["processed_hdf5_path"])
    merge_report = first_processed.parent / "merge_report.json"
    if audit.get("merge_report_sha256") != sha256_file(merge_report):
        fail("coordinator audit receipt mismatch for merge_report.json")
    return audit


def nearest_video_indices(camera_ns: np.ndarray, color_us: np.ndarray) -> np.ndarray:
    if camera_ns.ndim != 1 or color_us.ndim != 1 or len(color_us) == 0:
        fail("source timestamp arrays must be non-empty one-dimensional arrays")
    if np.any(np.diff(camera_ns) < 0) or np.any(np.diff(color_us) < 0):
        fail("source timestamp arrays must be monotonic")
    color_ns = color_us.astype(np.int64) * 1000
    insertion = np.clip(np.searchsorted(color_ns, camera_ns.astype(np.int64)), 0, len(color_ns) - 1)
    previous = np.clip(insertion - 1, 0, len(color_ns) - 1)
    use_previous = np.abs(color_ns[previous] - camera_ns) <= np.abs(color_ns[insertion] - camera_ns)
    return np.where(use_previous, previous, insertion)


def validate_packet_allowlist(packet: Path, tasks: list[dict[str, str]]) -> None:
    expected_media = {f"media/{row['task_id']}.mp4" for row in tasks}
    for path in packet.rglob("*"):
        if path.is_symlink():
            fail(f"packet files must not be symlinks: {path}")
        relative = path.relative_to(packet).as_posix()
        if path.is_dir():
            if relative != "media":
                fail(f"unexpected packet directory: {relative}")
        elif relative not in PACKET_FILE_ALLOWLIST and relative not in expected_media:
            fail(f"unexpected packet file (possible proposal/GT leak): {relative}")
    root_files = {path.name for path in packet.iterdir() if path.is_file()}
    if root_files != PACKET_FILE_ALLOWLIST:
        fail(f"packet root file allowlist differs: {packet}")


def validate_shared_packet(root: Path, sources: dict[int, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    packet_a, packet_b = root / "annotator_a", root / "annotator_b"
    tasks_a, tasks_b = packet_a / "tasks.csv", packet_b / "tasks.csv"
    evidence_a, evidence_b = packet_a / "timeline_evidence_index.csv", packet_b / "timeline_evidence_index.csv"
    guide_a, guide_b = packet_a / "ANNOTATION_GUIDE.md", packet_b / "ANNOTATION_GUIDE.md"
    ensure_independent_files(tasks_a, tasks_b, allow_equal_hash=True)
    ensure_independent_files(evidence_a, evidence_b, allow_equal_hash=True)
    ensure_independent_files(guide_a, guide_b, allow_equal_hash=True)
    if (
        tasks_a.read_bytes() != tasks_b.read_bytes()
        or evidence_a.read_bytes() != evidence_b.read_bytes()
        or guide_a.read_bytes() != guide_b.read_bytes()
    ):
        fail("A/B must receive byte-identical guide, task coverage, and timeline evidence")
    audit = json.loads((root / "coordinator" / "audit.json").read_text(encoding="utf-8"))
    if sha256_file(guide_a) != audit.get("annotation_guide_sha256"):
        fail("packet annotation guide differs from coordinator audit")
    tasks = read_csv_exact(tasks_a, TASK_FIELDS)
    if (
        len(tasks) != 12
        or len({row["episode_id"] for row in tasks}) != 12
        or len({row["task_id"] for row in tasks}) != 12
    ):
        fail("assignment must contain 12 distinct task and episode identities")
    validate_packet_allowlist(packet_a, tasks)
    validate_packet_allowlist(packet_b, tasks)
    selected = {episode for episode, row in sources.items() if row["assignment_selected"] == "true"}
    if (
        tuple(int(row["episode_id"]) for row in tasks) != MINIMUM_EPISODES
        or set(MINIMUM_EPISODES) != selected
    ):
        fail("task episodes disagree with audited selected sources")
    for task_index, row in enumerate(tasks):
        episode = int(row["episode_id"])
        if (
            row["task_id"] != f"contact-min12-{task_index:02d}"
            or int(row["coverage_start_frame"]) != 0
            or int(row["coverage_end_frame"]) != int(sources[episode]["processed_frames"])
            or row["frame_clock"] != FRAME_CLOCK
            or float(row["target_fps"]) != 20.0
            or int(row["context_frames_each_side"]) != 10
            or row["required_coverage_review"] != "full_episode"
        ):
            fail(f"task {row['task_id']}: invalid full-episode coverage")
    evidence = read_csv_exact(evidence_a, EVIDENCE_FIELDS)
    by_task = {row["task_id"]: [] for row in tasks}
    task_by_id = {row["task_id"]: row for row in tasks}
    for row in evidence:
        if row["task_id"] not in by_task:
            fail(f"unknown evidence task {row['task_id']}")
        by_task[row["task_id"]].append(row)
        episode = int(row["episode_id"])
        if episode != int(task_by_id[row["task_id"]]["episode_id"]):
            fail(f"{row['task_id']}: evidence episode mismatch")
        task_index = tasks.index(task_by_id[row["task_id"]])
        if row["media_id"] != f"contact-media-{task_index:02d}":
            fail(f"episode {episode}: opaque media ID mismatch")
        if row["media_relpath"] != f"media/{row['task_id']}.mp4":
            fail(f"episode {episode}: isolated media relative path mismatch")
        if row["video_sha256"] != sources[episode]["video_sha256"]:
            fail(f"episode {episode}: evidence video hash mismatch")
    for task_id, rows in by_task.items():
        expected = int(task_by_id[task_id]["coverage_end_frame"])
        frames = [int(row["processed_frame"]) for row in rows]
        if frames != list(range(expected)):
            fail(f"{task_id}: evidence is not a complete ordered processed timeline")
        episode = int(task_by_id[task_id]["episode_id"])
        source = sources[episode]
        with h5py.File(source["processed_hdf5_path"], "r") as handle:
            camera_ns = np.asarray(handle["camera_timestamp"], dtype=np.int64)
        with h5py.File(source["depth_timestamps_path"], "r") as handle:
            color_us = np.asarray(handle["color_system_timestamps_us"], dtype=np.int64)
        if len(camera_ns) != expected:
            fail(f"{task_id}: source camera timestamp length differs from task")
        expected_video = nearest_video_indices(camera_ns, color_us)
        video_fps = float(source["video_fps"])
        for processed_frame, row in enumerate(rows):
            video_frame = int(expected_video[processed_frame])
            sync_error_ms = abs(
                int(camera_ns[processed_frame]) - int(color_us[video_frame]) * 1000
            ) / 1e6
            if (
                int(row["video_frame"]) != video_frame
                or not 0 <= video_frame < int(source["video_frames"])
                or abs(float(row["processed_time_s"]) - processed_frame / 20.0) > 1e-6
                or abs(float(row["video_time_s"]) - video_frame / video_fps) > 1e-6
                or abs(float(row["sync_error_ms"]) - sync_error_ms) > 1e-6
            ):
                fail(f"{task_id}: evidence timeline differs from source timestamps")
        if max(float(row["sync_error_ms"]) for row in rows) > 50.0:
            fail(f"{task_id}: severe (>50 ms) timeline sync error")
        for packet in (packet_a, packet_b):
            media_path = packet / f"media/{task_id}.mp4"
            if media_path.exists() and sha256_file(media_path) != source["video_sha256"]:
                fail(f"{packet}: isolated media content hash mismatch for {task_id}")
    return tasks, evidence


def validate_identities(root: Path, phase: str) -> tuple[dict, dict]:
    identities = tuple(read_identity(root / f"annotator_{letter}" / "identity.json") for letter in ("a", "b"))
    expected_roles = ("independent_a", "independent_b")
    for identity, expected_role in zip(identities, expected_roles):
        if identity.get("annotation_role") != expected_role:
            fail(f"identity role must be {expected_role}")
        expected_status = (
            "must_be_replaced_by_coordinator_before_distribution"
            if phase == "assignment"
            else "assigned_by_coordinator"
        )
        if identity.get("identity_status") != expected_status:
            fail(f"{expected_role}: identity_status must be {expected_status!r}")
        for field in ("packet_id", "annotator_id", "annotation_session_id"):
            value = str(identity.get(field, ""))
            if not ID_RE.fullmatch(value):
                fail(f"{expected_role}: invalid {field}")
            if phase != "assignment" and value.startswith(UNKNOWN_PREFIXES):
                fail(f"{expected_role}: coordinator must replace {field} before labeling")
        if identity.get("proposal_material_included") is not False or identity.get("other_annotator_material_included") is not False:
            fail(f"{expected_role}: packet isolation declaration is false")
        packet = root / f"annotator_{expected_role[-1]}"
        if (
            identity.get("tasks_sha256") != sha256_file(packet / "tasks.csv")
            or identity.get("timeline_evidence_sha256")
            != sha256_file(packet / "timeline_evidence_index.csv")
            or identity.get("coordinator_audit_sha256")
            != sha256_file(root / "coordinator" / "audit.json")
            or identity.get("annotation_guide_sha256")
            != sha256_file(packet / "ANNOTATION_GUIDE.md")
        ):
            fail(f"{expected_role}: immutable packet receipt differs")
    for field in ("packet_id", "annotator_id", "annotation_session_id"):
        if identities[0][field] == identities[1][field]:
            fail(f"A/B must have distinct {field}")
    return identities


def validate_coverages(
    path: Path,
    role: str,
    identity: dict,
    tasks: list[dict[str, str]],
    phase: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv_exact(path, COVERAGE_FIELDS)
    if len(rows) != len(tasks):
        fail(f"{path}: expected one full-episode coverage row per task")
    expected_episodes = [row["episode_id"] for row in tasks]
    actual_episodes = [row["episode_id"] for row in rows]
    if actual_episodes != expected_episodes or len(set(actual_episodes)) != len(tasks):
        fail(f"{path}: coverage rows must map one-to-one to the ordered task episodes")
    task_by_episode = {int(row["episode_id"]): row for row in tasks}
    by_id = {}
    for number, row in enumerate(rows, 2):
        episode = int(row["episode_id"])
        task = task_by_episode.get(episode)
        if task is None:
            fail(f"{path}:{number}: episode not assigned")
        if row["coverage_id"] in by_id or not ID_RE.fullmatch(row["coverage_id"]):
            fail(f"{path}:{number}: invalid or duplicate coverage_id")
        by_id[row["coverage_id"]] = row
        fixed = (
            row["schema_version"] == ANNOTATION_SCHEMA_VERSION
            and row["dataset_id"] == DATASET_ID
            and row["annotation_role"] == role
            and row["annotator_id"] == identity["annotator_id"]
            and row["frame_clock"] == FRAME_CLOCK
            and row["label_origin"] == "human_independent"
            and int(row["start_frame"]) == int(task["coverage_start_frame"])
            and int(row["end_frame"]) == int(task["coverage_end_frame"])
            and not row["parent_coverage_ids"]
        )
        if not fixed:
            fail(f"{path}:{number}: assignment or identity fields drifted")
        if phase == "assignment":
            expected_note = (
                f"task_id={task['task_id']}; no label has been prefilled"
            )
            if (
                row["review_status"] != "in_progress"
                or row["coverage_mode"]
                or row["evidence_scope"]
                or row["review_timestamp_utc"]
                or row["notes"] != expected_note
            ):
                fail(f"{path}:{number}: assignment template contains a prefilled label")
        else:
            if row["review_status"] != "submitted" or row["coverage_mode"] not in ALLOWED_COVERAGE:
                fail(f"{path}:{number}: coverage is not submitted with a valid mode")
            if row["evidence_scope"] not in ALLOWED_EVIDENCE:
                fail(f"{path}:{number}: invalid evidence_scope")
            validate_iso_utc(row["review_timestamp_utc"], f"{path}:{number}:review_timestamp_utc")
    return rows, by_id


def validate_segments(
    path: Path,
    role: str,
    identity: dict,
    coverage_by_id: dict[str, dict[str, str]],
    phase: str,
) -> list[Segment]:
    rows = read_csv_exact(path, SEGMENT_FIELDS)
    if phase == "assignment":
        if rows:
            fail(f"{path}: segment assignment template must be header-only")
        return []
    segment_ids: set[str] = set()
    event_ids: set[str] = set()
    result = []
    for number, row in enumerate(rows, 2):
        coverage = coverage_by_id.get(row["coverage_id"])
        if coverage is None or int(coverage["episode_id"]) != int(row["episode_id"]):
            fail(f"{path}:{number}: segment references the wrong coverage")
        if coverage["coverage_mode"] == "unknown":
            fail(f"{path}:{number}: unknown coverage cannot contain a segment")
        if not (
            row["schema_version"] == ANNOTATION_SCHEMA_VERSION
            and row["dataset_id"] == DATASET_ID
            and row["annotation_role"] == role
            and row["annotator_id"] == identity["annotator_id"]
            and row["review_status"] == "submitted"
            and row["label_origin"] == "human_independent"
            and row["evidence_scope"] in ALLOWED_EVIDENCE
            and not row["source_proposal_id"]
            and not row["parent_segment_ids"]
        ):
            fail(f"{path}:{number}: invalid independent-label provenance")
        segment_id = row["segment_id"]
        if not ID_RE.fullmatch(segment_id) or segment_id in segment_ids:
            fail(f"{path}:{number}: invalid or duplicate segment_id")
        segment_ids.add(segment_id)
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if not int(coverage["start_frame"]) <= start < end <= int(coverage["end_frame"]):
            fail(f"{path}:{number}: invalid conservative positive interval")
        if row["start_boundary"] not in ALLOWED_START or row["end_boundary"] not in ALLOWED_END:
            fail(f"{path}:{number}: invalid boundary status")
        if row["start_boundary"] == "left_censored" and start != int(coverage["start_frame"]):
            fail(f"{path}:{number}: left-censored start must equal coverage start")
        if row["end_boundary"] == "right_censored" and end != int(coverage["end_frame"]):
            fail(f"{path}:{number}: right-censored end must equal coverage end")
        onset, release = row["onset_event_id"], row["release_event_id"]
        if (row["start_boundary"] == "observed") != bool(onset):
            fail(f"{path}:{number}: onset_event_id required iff observed")
        if (row["end_boundary"] == "observed") != bool(release):
            fail(f"{path}:{number}: release_event_id required iff observed")
        for event in (onset, release):
            if event:
                if not ID_RE.fullmatch(event) or event in event_ids or event in segment_ids:
                    fail(f"{path}:{number}: invalid, duplicate, or colliding event ID")
                event_ids.add(event)
        for field, allowed in ALLOWED_CATEGORY.items():
            if row[field] not in allowed:
                fail(f"{path}:{number}: invalid {field}={row[field]!r}")
        confidence = float(row["confidence"])
        if not 0.0 < confidence <= 1.0:
            fail(f"{path}:{number}: confidence must be in (0,1]")
        result.append(Segment(
            int(row["episode_id"]), row["coverage_id"], segment_id, start, end,
            row["start_boundary"], row["end_boundary"], row["side"], row["site"],
            row["object_role"], row["contact_mode"],
        ))
    if segment_ids & event_ids:
        fail(f"{path}: segment/event ID collision")
    return result


def load_packets(root: Path, phase: str, tasks: list[dict[str, str]]) -> tuple[tuple[dict, dict], list[dict[str, object]]]:
    identities = validate_identities(root, phase)
    packet_data = []
    for index, letter in enumerate(("a", "b")):
        packet = root / f"annotator_{letter}"
        receipt = json.loads((packet / "assignment_receipt.json").read_text(encoding="utf-8"))
        expected_media = [f"media/{row['task_id']}.mp4" for row in tasks]
        if (
            set(receipt) != ASSIGNMENT_RECEIPT_FIELDS
            or receipt.get("packet_id") != identities[index]["packet_id"]
            or receipt.get("packet_file_allowlist") != sorted(PACKET_FILE_ALLOWLIST)
            or receipt.get("optional_media_relpaths") != expected_media
            or receipt.get("coordinator_audit_sha256")
            != sha256_file(root / "coordinator" / "audit.json")
            or receipt.get("annotation_guide_sha256")
            != sha256_file(packet / "ANNOTATION_GUIDE.md")
        ):
            fail(f"{packet}: assignment receipt schema or coordinator binding differs")
        for name, field in (
            ("tasks.csv", "tasks_sha256"),
            ("timeline_evidence_index.csv", "timeline_evidence_sha256"),
        ):
            if receipt.get(field) != sha256_file(packet / name):
                fail(f"{packet}: immutable assignment receipt mismatch for {name}")
        if phase == "assignment":
            for name, field in (
                ("identity.json", "identity_sha256"),
                ("coverage.csv", "coverage_sha256"),
                ("segments.csv", "segments_sha256"),
            ):
                if receipt.get(field) != sha256_file(packet / name):
                    fail(f"{packet}: assignment receipt mismatch for {name}")
        role = f"independent_{letter}"
        coverages, coverage_by_id = validate_coverages(
            packet / "coverage.csv", role, identities[index], tasks, phase
        )
        segments = validate_segments(
            packet / "segments.csv", role, identities[index], coverage_by_id, phase
        )
        segment_counts = {
            coverage_id: sum(item.coverage_id == coverage_id for item in segments)
            for coverage_id in coverage_by_id
        }
        if phase != "assignment" and any(
            row["coverage_mode"] == "positive_only"
            and segment_counts[row["coverage_id"]] == 0
            for row in coverages
        ):
            fail(f"{packet}: positive_only coverage requires at least one submitted segment")
        packet_data.append({"coverages": coverages, "segments": segments})
    ensure_independent_files(root / "annotator_a" / "identity.json", root / "annotator_b" / "identity.json")
    ensure_independent_files(root / "annotator_a" / "coverage.csv", root / "annotator_b" / "coverage.csv")
    ensure_independent_files(
        root / "annotator_a" / "segments.csv",
        root / "annotator_b" / "segments.csv",
        allow_equal_hash=False,
    )
    coverage_ids = [
        {row["coverage_id"] for row in packet["coverages"]}
        for packet in packet_data
    ]
    segment_ids = [
        {item.segment_id for item in packet["segments"]}
        for packet in packet_data
    ]
    if coverage_ids[0] & coverage_ids[1] or segment_ids[0] & segment_ids[1]:
        fail("A/B coverage and segment ID namespaces must be disjoint")
    return identities, packet_data


def weighted_kappa(values_a: list[str], values_b: list[str], ordered: list[str] | None = None) -> float | None:
    if len(values_a) != len(values_b) or not values_a:
        return None
    labels = ordered or sorted(set(values_a) | set(values_b))
    index = {label: number for number, label in enumerate(labels)}
    count = len(values_a)
    marg_a = [0] * len(labels)
    marg_b = [0] * len(labels)
    observed = 0.0
    for a, b in zip(values_a, values_b):
        marg_a[index[a]] += 1
        marg_b[index[b]] += 1
        distance = abs(index[a] - index[b]) / max(1, len(labels) - 1) if ordered else float(a != b)
        observed += distance
    expected = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            distance = abs(i - j) / max(1, len(labels) - 1) if ordered else float(i != j)
            expected += distance * marg_a[i] * marg_b[j] / count
    observed /= count
    expected /= count
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def tiou(a: Segment, b: Segment) -> float:
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return overlap / union if union else 0.0


def best_episode_matching(a_items: list[Segment], b_items: list[Segment]) -> list[tuple[int, int, float]]:
    if len(b_items) > 20:
        fail("agreement matcher supports at most 20 B segments per episode")
    edges = {}
    for i, a in enumerate(a_items):
        for j, b in enumerate(b_items):
            score = tiou(a, b)
            side_ok = not a.side or not b.side or a.side == b.side
            if score >= 0.5 and side_ok:
                edges[(i, j)] = score

    @lru_cache(maxsize=None)
    def solve(i: int, used: int) -> tuple[int, float, tuple[tuple[int, int, float], ...]]:
        if i == len(a_items):
            return 0, 0.0, ()
        best = solve(i + 1, used)
        for j in range(len(b_items)):
            if used & (1 << j) or (i, j) not in edges:
                continue
            count, score, matches = solve(i + 1, used | (1 << j))
            candidate = (count + 1, score + edges[(i, j)], ((i, j, edges[(i, j)]),) + matches)
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    return list(solve(0, 0)[2])


def agreement_report(tasks: list[dict[str, str]], packets: list[dict[str, object]]) -> dict:
    coverage_a = {int(row["episode_id"]): row for row in packets[0]["coverages"]}
    coverage_b = {int(row["episode_id"]): row for row in packets[1]["coverages"]}
    modes_a = [coverage_a[int(task["episode_id"])]["coverage_mode"] for task in tasks]
    modes_b = [coverage_b[int(task["episode_id"])]["coverage_mode"] for task in tasks]
    coverage_kappa = weighted_kappa(modes_a, modes_b, ["unknown", "positive_only", "exhaustive"])
    segments_a: list[Segment] = packets[0]["segments"]
    segments_b: list[Segment] = packets[1]["segments"]
    all_matches: list[tuple[Segment, Segment, float]] = []
    per_episode = []
    for task in tasks:
        episode = int(task["episode_id"])
        a_items = [item for item in segments_a if item.episode == episode]
        b_items = [item for item in segments_b if item.episode == episode]
        matches = best_episode_matching(a_items, b_items)
        all_matches.extend((a_items[i], b_items[j], score) for i, j, score in matches)
        per_episode.append({
            "episode_id": episode,
            "segments_a": len(a_items),
            "segments_b": len(b_items),
            "matched": len(matches),
            "event_f1": (2 * len(matches) / (len(a_items) + len(b_items))) if a_items or b_items else 0.0,
        })
    match_count = len(all_matches)
    event_f1 = 2 * match_count / (len(segments_a) + len(segments_b)) if segments_a or segments_b else 0.0
    tious = [score for _, _, score in all_matches]
    boundary_errors = []
    onset_errors = []
    release_errors = []
    for a, b, _ in all_matches:
        if a.start_boundary == b.start_boundary == "observed":
            onset_errors.append(abs(a.start - b.start))
            boundary_errors.append(abs(a.start - b.start))
        if a.end_boundary == b.end_boundary == "observed":
            release_errors.append(abs(a.end - b.end))
            boundary_errors.append(abs(a.end - b.end))
    category = {}
    for field in ("side", "site", "object_role", "contact_mode"):
        pairs = [(getattr(a, field), getattr(b, field)) for a, b, _ in all_matches if getattr(a, field) and getattr(b, field)]
        category[field] = {
            "comparable_matches": len(pairs),
            "kappa": weighted_kappa([a for a, _ in pairs], [b for _, b in pairs]),
        }
    exhaustive_episodes = [
        int(task["episode_id"])
        for task in tasks
        if coverage_a[int(task["episode_id"])]["coverage_mode"] == "exhaustive"
        and coverage_b[int(task["episode_id"])]["coverage_mode"] == "exhaustive"
    ]
    frame_a, frame_b = [], []
    for task in tasks:
        episode = int(task["episode_id"])
        if episode not in exhaustive_episodes:
            continue
        length = int(task["coverage_end_frame"])
        episode_a = [item for item in segments_a if item.episode == episode]
        episode_b = [item for item in segments_b if item.episode == episode]
        for frame in range(length):
            frame_a.append("contact" if any(item.start <= frame < item.end for item in episode_a) else "no_contact")
            frame_b.append("contact" if any(item.start <= frame < item.end for item in episode_b) else "no_contact")
    contact_kappa = weighted_kappa(frame_a, frame_b)
    positive_union = sum(a == "contact" or b == "contact" for a, b in zip(frame_a, frame_b))
    negative_union = sum(a == "no_contact" or b == "no_contact" for a, b in zip(frame_a, frame_b))
    positive_agreement = (
        sum(a == b == "contact" for a, b in zip(frame_a, frame_b)) / positive_union if positive_union else None
    )
    negative_agreement = (
        sum(a == b == "no_contact" for a, b in zip(frame_a, frame_b)) / negative_union if negative_union else None
    )
    censor_values = ("observed", "censored", "unknown")
    confusion = {boundary: {a: {b: 0 for b in censor_values} for a in censor_values} for boundary in ("start", "end")}
    for a, b, _ in all_matches:
        for boundary in ("start", "end"):
            a_value = getattr(a, f"{boundary}_boundary")
            b_value = getattr(b, f"{boundary}_boundary")
            a_value = "censored" if "censored" in a_value else a_value
            b_value = "censored" if "censored" in b_value else b_value
            confusion[boundary][a_value][b_value] += 1
    category_gate = all(
        item["comparable_matches"] >= 2
        and item["kappa"] is not None
        and item["kappa"] >= THRESHOLDS["categorical_kappa_min"]
        for item in category.values()
    )
    boundary_median = median(boundary_errors) if boundary_errors else None
    gates = {
        "coverage_weighted_kappa": coverage_kappa is not None and coverage_kappa >= THRESHOLDS["coverage_weighted_kappa_min"],
        "event_f1": event_f1 >= THRESHOLDS["event_f1_min"],
        "matched_median_tiou": bool(tious) and median(tious) >= THRESHOLDS["matched_median_tiou_min"],
        "observed_boundary_median_mae": boundary_median is not None and boundary_median <= THRESHOLDS["observed_boundary_median_mae_frames_max"],
        "categorical_kappa": category_gate,
    }
    return {
        "report_schema_version": "contact_dual_agreement_report_v1_1",
        "thresholds": THRESHOLDS,
        "coverage_weighted_kappa_linear": coverage_kappa,
        "exhaustive_intersection_episodes": exhaustive_episodes,
        "frame_contact_kappa": contact_kappa,
        "frame_positive_agreement": positive_agreement,
        "frame_negative_agreement": negative_agreement,
        "segments_a": len(segments_a),
        "segments_b": len(segments_b),
        "matched_segments": match_count,
        "event_precision": match_count / len(segments_b) if segments_b else 0.0,
        "event_recall": match_count / len(segments_a) if segments_a else 0.0,
        "event_f1": event_f1,
        "matched_median_tiou": median(tious) if tious else None,
        "observed_onset_median_mae_frames": median(onset_errors) if onset_errors else None,
        "observed_release_median_mae_frames": median(release_errors) if release_errors else None,
        "observed_boundary_median_mae_frames": boundary_median,
        "observed_boundary_median_mae_ms": boundary_median * 50.0 if boundary_median is not None else None,
        "categorical_agreement": category,
        "boundary_status_confusion": confusion,
        "per_episode": per_episode,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "failed_gate_action": "revise definitions and independently relabel three calibration episodes; do not adjudicate" if not all(gates.values()) else "eligible_for_third_party_adjudication",
    }


def validate_adjudication(root: Path, agreement: dict, identities: tuple[dict, dict], tasks: list[dict[str, str]]) -> dict:
    if not agreement.get("all_gates_pass"):
        fail("agreement gates failed; adjudication is prohibited")
    identity_path = root / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        set(identity)
        != {
            "adjudicator_id",
            "adjudication_session_id",
            "identity_status",
            "must_be_distinct_from_annotator_a_and_b",
        }
        or identity.get("identity_status") != "assigned_after_agreement_gates_pass"
        or identity.get("must_be_distinct_from_annotator_a_and_b") is not True
    ):
        fail("adjudication identity receipt is incomplete")
    for field in ("adjudicator_id", "adjudication_session_id"):
        value = str(identity.get(field, ""))
        if not ID_RE.fullmatch(value) or value.startswith(UNKNOWN_PREFIXES):
            fail(f"adjudication identity requires a real {field}")
    if identity["adjudicator_id"] in {item["annotator_id"] for item in identities}:
        fail("adjudicator must be a third person, distinct from A and B")
    if identity["adjudication_session_id"] in {item["annotation_session_id"] for item in identities}:
        fail("adjudication session must be distinct from A/B sessions")
    coverage_rows = read_csv_exact(root / "coverage.csv", COVERAGE_FIELDS)
    segment_rows = read_csv_exact(root / "segments.csv", SEGMENT_FIELDS)
    task_by_episode = {int(row["episode_id"]): row for row in tasks}
    if [int(row["episode_id"]) for row in coverage_rows] != [
        int(row["episode_id"]) for row in tasks
    ]:
        fail("adjudication requires one ordered coverage row for every task episode")
    coverage_by_id = {}
    parent_coverages: dict[str, tuple[str, int]] = {}
    expected_coverage_parents: dict[int, set[str]] = {}
    for letter in ("a", "b"):
        rows = read_csv_exact(root.parent / f"annotator_{letter}" / "coverage.csv", COVERAGE_FIELDS)
        for row in rows:
            episode = int(row["episode_id"])
            parent_coverages[row["coverage_id"]] = (letter, episode)
            expected_coverage_parents.setdefault(episode, set()).add(row["coverage_id"])
    for number, row in enumerate(coverage_rows, 2):
        episode = int(row["episode_id"])
        task = task_by_episode.get(episode)
        parents = {value for value in row["parent_coverage_ids"].split(";") if value}
        if (
            not task
            or parents != expected_coverage_parents.get(episode)
            or {parent_coverages[parent][0] for parent in parents} != {"a", "b"}
            or any(parent_coverages[parent][1] != episode for parent in parents)
        ):
            fail(
                f"adjudicated coverage row {number}: requires exact same-episode A+B parents"
            )
        if not (
            row["schema_version"] == ANNOTATION_SCHEMA_VERSION
            and row["dataset_id"] == DATASET_ID
            and row["annotation_role"] == "adjudicated"
            and row["annotator_id"] == identity["adjudicator_id"]
            and row["review_status"] == "reviewed"
            and row["coverage_mode"] in ALLOWED_COVERAGE
            and row["frame_clock"] == FRAME_CLOCK
            and row["evidence_scope"] in ALLOWED_EVIDENCE
            and row["label_origin"] == "human_adjudicated"
            and int(row["start_frame"]) == 0
            and int(row["end_frame"]) == int(task["coverage_end_frame"])
        ):
            fail(f"adjudicated coverage row {number}: invalid reviewed provenance or coverage")
        validate_iso_utc(row["review_timestamp_utc"], f"adjudicated coverage row {number}")
        if row["coverage_id"] in coverage_by_id:
            fail("duplicate adjudicated coverage_id")
        coverage_by_id[row["coverage_id"]] = row
    if len(coverage_rows) != len(tasks):
        fail("adjudication requires one full-episode coverage row per task")
    parent_segments: dict[str, tuple[str, int]] = {}
    for letter in ("a", "b"):
        rows = read_csv_exact(root.parent / f"annotator_{letter}" / "segments.csv", SEGMENT_FIELDS)
        for row in rows:
            parent_segments[row["segment_id"]] = (letter, int(row["episode_id"]))
    segment_ids, event_ids = set(), set()
    for number, row in enumerate(segment_rows, 2):
        coverage = coverage_by_id.get(row["coverage_id"])
        parents = {value for value in row["parent_segment_ids"].split(";") if value}
        if (
            coverage is None
            or int(coverage["episode_id"]) != int(row["episode_id"])
            or not parents
            or not parents <= set(parent_segments)
        ):
            fail(f"adjudicated segment row {number}: missing valid A/B parent segment IDs")
        episode = int(row["episode_id"])
        if any(parent_segments[parent][1] != episode for parent in parents):
            fail(f"adjudicated segment row {number}: parent segment belongs to another episode")
        parent_sides = {parent_segments[parent][0] for parent in parents}
        if parent_sides != {"a", "b"}:
            match = re.search(r"(?:^|;)unilateral_parent_reason=([^;]+)", row["notes"])
            if match is None or not match.group(1).strip():
                fail(
                    f"adjudicated segment row {number}: unilateral parent requires an explicit reason"
                )
        if coverage["coverage_mode"] == "unknown":
            fail(f"adjudicated segment row {number}: unknown coverage cannot contain a segment")
        if row["source_proposal_id"]:
            fail(f"adjudicated segment row {number}: proposal IDs are forbidden")
        if not (
            row["schema_version"] == ANNOTATION_SCHEMA_VERSION
            and row["dataset_id"] == DATASET_ID
            and row["annotation_role"] == "adjudicated"
            and row["annotator_id"] == identity["adjudicator_id"]
            and row["review_status"] == "reviewed"
            and row["label_origin"] == "human_adjudicated"
            and row["evidence_scope"] in ALLOWED_EVIDENCE
        ):
            fail(f"adjudicated segment row {number}: invalid human-adjudicated provenance")
        segment_id = row["segment_id"]
        if not ID_RE.fullmatch(segment_id) or segment_id in segment_ids or segment_id in parent_segments:
            fail(f"adjudicated segment row {number}: ID must be new and unique")
        segment_ids.add(segment_id)
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if not int(coverage["start_frame"]) <= start < end <= int(coverage["end_frame"]):
            fail(f"adjudicated segment row {number}: invalid interval")
        if row["start_boundary"] not in ALLOWED_START or row["end_boundary"] not in ALLOWED_END:
            fail(f"adjudicated segment row {number}: invalid boundary status")
        for boundary, event_field in (("start", "onset_event_id"), ("end", "release_event_id")):
            event = row[event_field]
            if (row[f"{boundary}_boundary"] == "observed") != bool(event):
                fail(f"adjudicated segment row {number}: {event_field} required iff observed")
            if event:
                if not ID_RE.fullmatch(event) or event in event_ids or event in segment_ids or event in parent_segments:
                    fail(f"adjudicated segment row {number}: event ID is not new/unique")
                event_ids.add(event)
        for field, allowed in ALLOWED_CATEGORY.items():
            if row[field] not in allowed:
                fail(f"adjudicated segment row {number}: invalid {field}")
        if not 0.0 < float(row["confidence"]) <= 1.0:
            fail(f"adjudicated segment row {number}: invalid confidence")
    return {
        "adjudicated_coverages": len(coverage_rows),
        "adjudicated_segments": len(segment_rows),
        "builder_eligible": True,
    }


def main() -> None:
    args = parse_args()
    sources = validate_sources(args.packet_root)
    audit = validate_coordinator_receipt(args.packet_root, sources)
    tasks, evidence = validate_shared_packet(args.packet_root, sources)
    if audit.get("minimum_tasks") != len(tasks) or audit.get(
        "minimum_processed_frames"
    ) != len(evidence):
        fail("coordinator audit task/timeline counts differ")
    identity_phase = "assignment" if args.phase == "assignment" else "submission"
    identities, packets = load_packets(args.packet_root, identity_phase, tasks)
    result: dict[str, object] = {
        "status": "valid",
        "phase": args.phase,
        "episodes_audited": len(sources),
        "tasks": len(tasks),
        "timeline_rows": len(evidence),
        "a_packet_id": identities[0]["packet_id"],
        "b_packet_id": identities[1]["packet_id"],
        "annotation_files_independent": True,
        "annotation_sha256": {
            letter: {
                name: sha256_file(args.packet_root / f"annotator_{letter}" / name)
                for name in ("identity.json", "coverage.csv", "segments.csv")
            }
            for letter in ("a", "b")
        },
    }
    if args.phase in {"agreement", "adjudication"}:
        agreement = agreement_report(tasks, packets)
        result["agreement"] = agreement
        if not agreement["all_gates_pass"]:
            result["status"] = "failed"
    else:
        agreement = None
    if args.phase == "adjudication":
        if args.adjudication_root is None:
            fail("--adjudication-root is required for adjudication phase")
        if args.agreement_report is not None:
            stored = json.loads(args.agreement_report.read_text(encoding="utf-8"))
            if "agreement" in stored:
                stored = stored["agreement"]
            if stored != agreement:
                fail("stored agreement report differs from recomputed A/B agreement")
        result["adjudication"] = validate_adjudication(
            args.adjudication_root, agreement, identities, tasks
        )
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.phase == "agreement" and result["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed preflight for the isolated contact v1.1 annotation UI.

This tool only inspects one annotator packet, a separate allowlisted media
directory, the UI file, and a caller-selected state directory.  It never opens the
coordinator tree, a source root, HDF5, proposals, or another annotator packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UI = REPO_ROOT / "tools" / "contact_annotation_ui" / "contact_annotation_ui.html"

PACKET_SCHEMA_VERSION = "contact_dual_assignment_packet_v1_1"
ANNOTATION_SCHEMA_VERSION = "contact_segment_annotation_v1_1"
SESSION_SCHEMA_VERSION = "offline_contact_annotation_ui_session_v1"
UI_BUILD_ID = "contact-annotation-ui-op34-v1"
DATASET_ID = "ego_pour_liquid_between_beakers"
FRAME_CLOCK = "processed_action_20hz"
EXPECTED_EPISODES = (43, 50, 51, 58, 34, 38, 9, 54, 36, 53, 12, 0)
UNKNOWN_PREFIXES = ("UNASSIGNED_", "REPLACE_", "PLACEHOLDER_")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

PACKET_FILES = {
    "ANNOTATION_GUIDE.md",
    "assignment_receipt.json",
    "coverage.csv",
    "identity.json",
    "segments.csv",
    "tasks.csv",
    "timeline_evidence_index.csv",
}
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

ALLOWED_COVERAGE = {"unknown", "positive_only", "exhaustive"}
ALLOWED_EVIDENCE = {"rgb_direct", "rgb_temporal", "unknown"}
ALLOWED_START = {"observed", "left_censored", "unknown"}
ALLOWED_END = {"observed", "right_censored", "unknown"}
ALLOWED_CATEGORY = {
    "side": {"", "left", "right", "object_object"},
    "site": {"", "opposition", "primary", "support", "palm_or_jaw", "tool_tip", "other_surface"},
    "object_role": {"", "manipulated", "tool", "receptacle", "fixture"},
    "contact_mode": {"", "stick", "slide", "roll", "impact"},
}


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_exact(path: Path, fields: Iterable[str] | None = None) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"{path}: expected a JSON object")
    if fields is not None and set(value) != set(fields):
        fail(f"{path}: JSON field set differs")
    return value


def read_csv_exact(path: Path, fields: Iterable[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = list(fields)
        if reader.fieldnames != expected:
            fail(f"{path}: header mismatch; expected={expected}, actual={reader.fieldnames}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def csv_roundtrip(path: Path, fields: Iterable[str]) -> None:
    expected_fields = list(fields)
    rows = read_csv_exact(path, expected_fields)
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=expected_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with StringIO(output.getvalue(), newline="") as handle:
        reader = csv.DictReader(handle)
        actual = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if reader.fieldnames != expected_fields or actual != rows:
        fail(f"{path}: CSV parse/write/parse roundtrip differs")


def validate_iso_utc(value: str, field: str) -> None:
    if not value.endswith("Z"):
        fail(f"{field}: timestamp must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field}: invalid ISO-8601 UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        fail(f"{field}: timestamp must be UTC")


def validate_id(value: object, field: str) -> str:
    text = str(value)
    if not ID_RE.fullmatch(text) or text.startswith(UNKNOWN_PREFIXES):
        fail(f"{field}: coordinator-assigned non-placeholder ID required")
    return text


def require_private(path: Path, *, directory: bool, writable: bool) -> None:
    if path.is_symlink():
        fail(f"{path}: symlinks are forbidden")
    status = path.stat()
    if directory and not stat.S_ISDIR(status.st_mode):
        fail(f"{path}: expected a directory")
    if not directory and not stat.S_ISREG(status.st_mode):
        fail(f"{path}: expected a regular file")
    mode = stat.S_IMODE(status.st_mode)
    if mode & 0o077:
        fail(f"{path}: group/world permissions forbidden, actual={oct(mode)}")
    if not mode & stat.S_IRUSR:
        fail(f"{path}: owner read permission required")
    if directory and not mode & stat.S_IXUSR:
        fail(f"{path}: owner execute permission required")
    if writable and not mode & stat.S_IWUSR:
        fail(f"{path}: owner write permission required")
    if not writable and mode & 0o222:
        fail(f"{path}: immutable/read-only path is writable, actual={oct(mode)}")
    if not directory and status.st_nlink != 1:
        fail(f"{path}: hard-linked files are forbidden")


def require_private_state_file(path: Path) -> None:
    """Validate a browser-created file below an already-private 0700 state root."""
    if path.is_symlink():
        fail(f"{path}: state symlinks are forbidden")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        fail(f"{path}: state entry must be one regular, non-hard-linked file")
    mode = stat.S_IMODE(status.st_mode)
    if not mode & stat.S_IRUSR or not mode & stat.S_IWUSR:
        fail(f"{path}: state file requires owner read/write permission")


def ensure_separate_roots(*roots: Path) -> None:
    resolved = [root.resolve(strict=True) for root in roots]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                fail("packet, media, and state roots must be pairwise disjoint")
            if (left.stat().st_dev, left.stat().st_ino) == (
                right.stat().st_dev,
                right.stat().st_ino,
            ):
                fail("packet, media, and state roots must have distinct inodes")


def validate_identity(packet_root: Path) -> dict:
    identity_fields = {
        "packet_schema_version", "packet_id", "annotation_role",
        "annotator_id", "annotation_session_id", "identity_status",
        "tasks_sha256", "timeline_evidence_sha256",
        "coordinator_audit_sha256", "annotation_guide_sha256",
        "proposal_material_included", "other_annotator_material_included",
    }
    identity = read_json_exact(packet_root / "identity.json", identity_fields)
    if identity["packet_schema_version"] != PACKET_SCHEMA_VERSION:
        fail("identity.json: packet schema differs")
    if identity["annotation_role"] not in {"independent_a", "independent_b"}:
        fail("identity.json: only independent_a or independent_b is allowed")
    if identity["identity_status"] != "assigned_by_coordinator":
        fail("identity.json: coordinator must assign identity before UI use")
    validate_id(identity["packet_id"], "packet_id")
    validate_id(identity["annotator_id"], "annotator_id")
    validate_id(identity["annotation_session_id"], "annotation_session_id")
    if identity["proposal_material_included"] is not False:
        fail("identity.json: proposal material declaration is not false")
    if identity["other_annotator_material_included"] is not False:
        fail("identity.json: other-annotator material declaration is not false")
    for field in (
        "tasks_sha256", "timeline_evidence_sha256",
        "coordinator_audit_sha256", "annotation_guide_sha256",
    ):
        if not SHA_RE.fullmatch(str(identity[field])):
            fail(f"identity.json: invalid {field}")
    return identity


def validate_tasks(packet_root: Path) -> list[dict[str, str]]:
    rows = read_csv_exact(packet_root / "tasks.csv", TASK_FIELDS)
    episodes = tuple(int(row["episode_id"]) for row in rows)
    if episodes != EXPECTED_EPISODES or len(rows) != 12:
        fail(f"tasks.csv: expected ordered episode set {EXPECTED_EPISODES}, actual={episodes}")
    for index, row in enumerate(rows):
        if (
            row["task_id"] != f"contact-min12-{index:02d}"
            or int(row["coverage_start_frame"]) != 0
            or int(row["coverage_end_frame"]) <= 0
            or row["frame_clock"] != FRAME_CLOCK
            or float(row["target_fps"]) != 20.0
            or int(row["context_frames_each_side"]) != 10
            or row["required_coverage_review"] != "full_episode"
        ):
            fail(f"tasks.csv: invalid fixed task row {row['task_id']!r}")
    return rows


def validate_timeline(
    packet_root: Path, tasks: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = read_csv_exact(packet_root / "timeline_evidence_index.csv", EVIDENCE_FIELDS)
    task_by_id = {row["task_id"]: row for row in tasks}
    grouped = {task_id: [] for task_id in task_by_id}
    media_hashes: dict[str, str] = {}
    for row in rows:
        task = task_by_id.get(row["task_id"])
        if task is None:
            fail(f"timeline: unassigned task_id={row['task_id']!r}")
        grouped[row["task_id"]].append(row)
        if row["episode_id"] != task["episode_id"]:
            fail(f"timeline: episode mismatch for {row['task_id']}")
        expected_relpath = f"media/{row['task_id']}.mp4"
        if row["media_relpath"] != expected_relpath:
            fail(f"timeline: non-allowlisted media path for {row['task_id']}")
        if not SHA_RE.fullmatch(row["video_sha256"]):
            fail(f"timeline: invalid media hash for {row['task_id']}")
        previous = media_hashes.setdefault(expected_relpath, row["video_sha256"])
        if previous != row["video_sha256"]:
            fail(f"timeline: multiple media hashes for {row['task_id']}")
    for task_id, task in task_by_id.items():
        task_rows = grouped[task_id]
        expected_length = int(task["coverage_end_frame"])
        frames = [int(row["processed_frame"]) for row in task_rows]
        if frames != list(range(expected_length)):
            fail(f"timeline: incomplete or unordered processed frames for {task_id}")
        task_index = tasks.index(task)
        for frame, row in enumerate(task_rows):
            if abs(float(row["processed_time_s"]) - frame / 20.0) > 1e-6:
                fail(f"timeline: processed 20 Hz time mismatch for {task_id}:{frame}")
            if row["media_id"] != f"contact-media-{task_index:02d}":
                fail(f"timeline: invalid opaque media ID for {task_id}")
    return rows, media_hashes


def validate_receipts(
    packet_root: Path,
    identity: dict,
    tasks: list[dict[str, str]],
    media_hashes: dict[str, str],
) -> dict:
    receipt_fields = {
        "packet_id", "identity_sha256", "coverage_sha256", "segments_sha256",
        "tasks_sha256", "timeline_evidence_sha256", "coordinator_audit_sha256",
        "annotation_guide_sha256", "packet_file_allowlist",
        "optional_media_relpaths",
    }
    receipt = read_json_exact(packet_root / "assignment_receipt.json", receipt_fields)
    expected_media = [f"media/{row['task_id']}.mp4" for row in tasks]
    if receipt["packet_id"] != identity["packet_id"]:
        fail("assignment_receipt.json: packet_id differs from identity")
    if receipt["packet_file_allowlist"] != sorted(PACKET_FILES):
        fail("assignment_receipt.json: root file allowlist differs")
    if receipt["optional_media_relpaths"] != expected_media:
        fail("assignment_receipt.json: media allowlist differs")
    immutable_hashes = {
        "tasks_sha256": packet_root / "tasks.csv",
        "timeline_evidence_sha256": packet_root / "timeline_evidence_index.csv",
        "annotation_guide_sha256": packet_root / "ANNOTATION_GUIDE.md",
    }
    for field, path in immutable_hashes.items():
        actual = sha256_file(path)
        if receipt[field] != actual or identity[field] != actual:
            fail(f"immutable packet binding differs for {path.name}")
    if receipt["coordinator_audit_sha256"] != identity["coordinator_audit_sha256"]:
        fail("coordinator audit binding differs between identity and receipt")
    if set(media_hashes) != set(expected_media):
        fail("timeline media set differs from the 12-task allowlist")
    return receipt


def validate_packet_layout(packet_root: Path) -> None:
    actual_root = {path.name for path in packet_root.iterdir()}
    if actual_root != PACKET_FILES:
        fail(
            "packet root allowlist differs; possible leaked or runtime file: "
            f"expected={sorted(PACKET_FILES)}, actual={sorted(actual_root)}"
        )
    require_private(packet_root, directory=True, writable=True)
    seen_inodes: set[tuple[int, int]] = set()
    for name in sorted(PACKET_FILES):
        path = packet_root / name
        require_private(
            path,
            directory=False,
            writable=name in {"coverage.csv", "segments.csv"},
        )
        inode = (path.stat().st_dev, path.stat().st_ino)
        if inode in seen_inodes:
            fail(f"packet files share an inode: {path}")
        seen_inodes.add(inode)


def validate_media_layout(
    media_root: Path, tasks: list[dict[str, str]]
) -> None:
    expected_media = {f"{row['task_id']}.mp4" for row in tasks}
    if {path.name for path in media_root.iterdir()} != expected_media:
        fail("media root must contain exactly the 12 allowlisted MP4 names")
    require_private(media_root, directory=True, writable=False)
    seen_inodes: set[tuple[int, int]] = set()
    for filename in sorted(expected_media):
        path = media_root / filename
        require_private(path, directory=False, writable=False)
        inode = (path.stat().st_dev, path.stat().st_ino)
        if inode in seen_inodes:
            fail(f"media files share an inode: {path}")
        seen_inodes.add(inode)


def validate_media(media_root: Path, media_hashes: dict[str, str]) -> None:
    for relpath, expected_hash in sorted(media_hashes.items()):
        path = media_root / Path(relpath).name
        if sha256_file(path) != expected_hash:
            fail(f"{relpath}: media SHA-256 differs from timeline receipt")


def validate_coverages(
    coverage_path: Path,
    identity: dict,
    tasks: list[dict[str, str]],
    phase: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv_exact(coverage_path, COVERAGE_FIELDS)
    if len(rows) != len(tasks):
        fail("coverage.csv: exactly one ordered row per task required")
    task_by_episode = {int(row["episode_id"]): row for row in tasks}
    if [int(row["episode_id"]) for row in rows] != list(EXPECTED_EPISODES):
        fail("coverage.csv: ordered episode set differs from tasks")
    by_id: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, 2):
        episode = int(row["episode_id"])
        task = task_by_episode[episode]
        if not ID_RE.fullmatch(row["coverage_id"]) or row["coverage_id"] in by_id:
            fail(f"coverage.csv:{number}: invalid or duplicate coverage_id")
        by_id[row["coverage_id"]] = row
        if not (
            row["schema_version"] == ANNOTATION_SCHEMA_VERSION
            and row["dataset_id"] == DATASET_ID
            and row["annotation_role"] == identity["annotation_role"]
            and row["annotator_id"] == identity["annotator_id"]
            and int(row["start_frame"]) == 0
            and int(row["end_frame"]) == int(task["coverage_end_frame"])
            and row["frame_clock"] == FRAME_CLOCK
            and row["label_origin"] == "human_independent"
            and not row["parent_coverage_ids"]
        ):
            fail(f"coverage.csv:{number}: fixed assignment or identity fields drifted")
        if phase == "template":
            if not (
                row["review_status"] == "in_progress"
                and row["coverage_mode"] == ""
                and row["evidence_scope"] == ""
                and row["review_timestamp_utc"] == ""
            ):
                fail(f"coverage.csv:{number}: expected an unfilled assigned template")
        else:
            if row["review_status"] != "submitted":
                fail(f"coverage.csv:{number}: review_status must be submitted")
            if row["coverage_mode"] not in ALLOWED_COVERAGE:
                fail(f"coverage.csv:{number}: invalid coverage_mode")
            if row["evidence_scope"] not in ALLOWED_EVIDENCE:
                fail(f"coverage.csv:{number}: UI supports RGB-only evidence or unknown")
            if row["notes"]:
                fail(f"coverage.csv:{number}: UI output keeps notes empty")
            validate_iso_utc(row["review_timestamp_utc"], f"coverage.csv:{number}")
    return rows, by_id


def validate_segments(
    segments_path: Path,
    identity: dict,
    coverages: list[dict[str, str]],
    coverage_by_id: dict[str, dict[str, str]],
    phase: str,
) -> list[dict[str, str]]:
    rows = read_csv_exact(segments_path, SEGMENT_FIELDS)
    if phase == "template":
        if rows:
            fail("segments.csv: assigned template must be header-only")
        return rows
    segment_ids: set[str] = set()
    event_ids: set[str] = set()
    for number, row in enumerate(rows, 2):
        coverage = coverage_by_id.get(row["coverage_id"])
        if coverage is None or row["episode_id"] != coverage["episode_id"]:
            fail(f"segments.csv:{number}: coverage reference differs")
        if coverage["coverage_mode"] == "unknown":
            fail(f"segments.csv:{number}: unknown coverage cannot contain segments")
        if not (
            row["schema_version"] == ANNOTATION_SCHEMA_VERSION
            and row["dataset_id"] == DATASET_ID
            and row["annotation_role"] == identity["annotation_role"]
            and row["annotator_id"] == identity["annotator_id"]
            and row["review_status"] == "submitted"
            and row["label_origin"] == "human_independent"
            and row["evidence_scope"] in ALLOWED_EVIDENCE
            and not row["source_proposal_id"]
            and not row["parent_segment_ids"]
        ):
            fail(f"segments.csv:{number}: independent provenance differs")
        segment_id = row["segment_id"]
        if not ID_RE.fullmatch(segment_id) or segment_id in segment_ids:
            fail(f"segments.csv:{number}: invalid or duplicate segment_id")
        segment_ids.add(segment_id)
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if not int(coverage["start_frame"]) <= start < end <= int(coverage["end_frame"]):
            fail(f"segments.csv:{number}: invalid half-open interval")
        if row["start_boundary"] not in ALLOWED_START or row["end_boundary"] not in ALLOWED_END:
            fail(f"segments.csv:{number}: invalid boundary status")
        if row["start_boundary"] == "left_censored" and start != int(coverage["start_frame"]):
            fail(f"segments.csv:{number}: left_censored must start at coverage boundary")
        if row["end_boundary"] == "right_censored" and end != int(coverage["end_frame"]):
            fail(f"segments.csv:{number}: right_censored must end at coverage boundary")
        onset, release = row["onset_event_id"], row["release_event_id"]
        if (row["start_boundary"] == "observed") != bool(onset):
            fail(f"segments.csv:{number}: onset_event_id required iff observed")
        if (row["end_boundary"] == "observed") != bool(release):
            fail(f"segments.csv:{number}: release_event_id required iff observed")
        for field, event_id in (("onset", onset), ("release", release)):
            if event_id:
                if (
                    not ID_RE.fullmatch(event_id)
                    or event_id in event_ids
                    or event_id in segment_ids
                ):
                    fail(f"segments.csv:{number}: invalid/colliding {field} event ID")
                event_ids.add(event_id)
        for field, allowed in ALLOWED_CATEGORY.items():
            if row[field] not in allowed:
                fail(f"segments.csv:{number}: invalid {field}")
        try:
            confidence = float(row["confidence"])
        except ValueError as exc:
            raise ValueError(f"segments.csv:{number}: invalid confidence") from exc
        if not 0.0 < confidence <= 1.0:
            fail(f"segments.csv:{number}: confidence must be in (0,1]")
        if row["notes"]:
            fail(f"segments.csv:{number}: UI output keeps notes empty to prevent data leakage")
    if segment_ids & event_ids:
        fail("segments.csv: segment/event ID collision")
    counts = {
        row["coverage_id"]: sum(item["coverage_id"] == row["coverage_id"] for item in rows)
        for row in coverages
    }
    for row in coverages:
        if row["coverage_mode"] == "positive_only" and counts[row["coverage_id"]] == 0:
            fail(f"coverage {row['coverage_id']}: positive_only requires a segment")
    return rows


def validate_annotation_csvs(
    coverage_path: Path,
    segments_path: Path,
    identity: dict,
    tasks: list[dict[str, str]],
    phase: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    coverages, by_id = validate_coverages(coverage_path, identity, tasks, phase)
    segments = validate_segments(segments_path, identity, coverages, by_id, phase)
    csv_roundtrip(coverage_path, COVERAGE_FIELDS)
    csv_roundtrip(segments_path, SEGMENT_FIELDS)
    return coverages, segments


def initialize_state(
    state_dir: Path,
    packet_root: Path,
    ui_file: Path,
    identity: dict,
    tasks: list[dict[str, str]],
    media_hashes: dict[str, str],
) -> Path:
    receipt_path = state_dir / "contact_annotation_ui_session.json"
    if receipt_path.exists():
        old = read_json_exact(receipt_path)
        identity_fields = (
            "packet_id", "annotation_role", "annotator_id", "annotation_session_id"
        )
        if any(old.get(field) != identity[field] for field in identity_fields):
            fail("state directory already belongs to a different packet/identity/session")
    receipt = {
        "session_schema_version": SESSION_SCHEMA_VERSION,
        "ui_build_id": UI_BUILD_ID,
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "frame_clock": FRAME_CLOCK,
        "packet_id": identity["packet_id"],
        "annotation_role": identity["annotation_role"],
        "annotator_id": identity["annotator_id"],
        "annotation_session_id": identity["annotation_session_id"],
        "tasks_sha256": sha256_file(packet_root / "tasks.csv"),
        "timeline_evidence_sha256": sha256_file(packet_root / "timeline_evidence_index.csv"),
        "annotation_guide_sha256": sha256_file(packet_root / "ANNOTATION_GUIDE.md"),
        "identity_sha256": sha256_file(packet_root / "identity.json"),
        "assignment_receipt_sha256": sha256_file(packet_root / "assignment_receipt.json"),
        "ui_sha256": sha256_file(ui_file),
        "tasks": [
            {
                "task_id": row["task_id"],
                "episode_id": int(row["episode_id"]),
                "coverage_end_frame": int(row["coverage_end_frame"]),
                "media_relpath": f"media/{row['task_id']}.mp4",
                "media_sha256": media_hashes[f"media/{row['task_id']}.mp4"],
            }
            for row in tasks
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "privacy_boundary": "three disjoint roots: exact-seven-file packet + twelve-MP4 media + identity-locked state",
    }
    payload = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    temporary = state_dir / ".contact_annotation_ui_session.json.new"
    temporary.write_text(payload, encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(receipt_path)
    os.chmod(receipt_path, 0o600)
    return receipt_path


def validate_state_session(
    state_dir: Path,
    packet_root: Path,
    ui_file: Path,
    identity: dict,
    tasks: list[dict[str, str]],
    media_hashes: dict[str, str],
) -> dict:
    path = state_dir / "contact_annotation_ui_session.json"
    if not path.is_file() or path.is_symlink():
        fail("state directory lacks a regular identity-locked session receipt")
    require_private_state_file(path)
    fields = {
        "session_schema_version", "ui_build_id", "packet_schema_version",
        "annotation_schema_version", "dataset_id", "frame_clock", "packet_id",
        "annotation_role", "annotator_id", "annotation_session_id",
        "tasks_sha256", "timeline_evidence_sha256", "annotation_guide_sha256",
        "identity_sha256", "assignment_receipt_sha256", "ui_sha256", "tasks",
        "created_utc", "privacy_boundary",
    }
    receipt = read_json_exact(path, fields)
    expected = {
        "session_schema_version": SESSION_SCHEMA_VERSION,
        "ui_build_id": UI_BUILD_ID,
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "frame_clock": FRAME_CLOCK,
        "packet_id": identity["packet_id"],
        "annotation_role": identity["annotation_role"],
        "annotator_id": identity["annotator_id"],
        "annotation_session_id": identity["annotation_session_id"],
        "tasks_sha256": sha256_file(packet_root / "tasks.csv"),
        "timeline_evidence_sha256": sha256_file(
            packet_root / "timeline_evidence_index.csv"
        ),
        "annotation_guide_sha256": sha256_file(
            packet_root / "ANNOTATION_GUIDE.md"
        ),
        "identity_sha256": sha256_file(packet_root / "identity.json"),
        "assignment_receipt_sha256": sha256_file(
            packet_root / "assignment_receipt.json"
        ),
        "ui_sha256": sha256_file(ui_file),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            fail(f"state session receipt differs for {field}")
    expected_tasks = [
        {
            "task_id": row["task_id"],
            "episode_id": int(row["episode_id"]),
            "coverage_end_frame": int(row["coverage_end_frame"]),
            "media_relpath": f"media/{row['task_id']}.mp4",
            "media_sha256": media_hashes[f"media/{row['task_id']}.mp4"],
        }
        for row in tasks
    ]
    if receipt.get("tasks") != expected_tasks:
        fail("state session receipt task/media bindings differ")
    validate_iso_utc(receipt["created_utc"], "state session receipt")
    if receipt["privacy_boundary"] != (
        "three disjoint roots: exact-seven-file packet + twelve-MP4 media + "
        "identity-locked state"
    ):
        fail("state session privacy boundary differs")
    if any(
        isinstance(value, str) and value.startswith("/")
        for value in receipt.values()
    ):
        fail("state session receipt must not contain absolute paths")
    return receipt


def validate_state_layout(state_dir: Path, *, allow_empty: bool) -> None:
    fixed = {
        "contact_annotation_ui_session.json",
        "contact_annotation_ui_autosave.json",
        "contact_annotation_ui_submission_receipt.json",
        "coverage.autosave.csv",
        "segments.autosave.csv",
        "coverage.submission.csv",
        "segments.submission.csv",
    }
    backup_re = re.compile(r"^(coverage|segments)\.precommit\.[0-9TZ-]+\.[0-9]+\.csv$")
    for path in state_dir.iterdir():
        if path.name not in fixed and not backup_re.fullmatch(path.name):
            fail(f"state directory contains an unexpected runtime file: {path.name}")
        require_private_state_file(path)
    if not allow_empty and not any(state_dir.iterdir()):
        fail("state directory is unexpectedly empty")


def validate_submission_receipt(
    state_dir: Path,
    identity: dict,
    coverage_path: Path,
    segments_path: Path,
) -> dict:
    path = state_dir / "contact_annotation_ui_submission_receipt.json"
    fields = {
        "submission_receipt_schema", "ui_build_id", "packet_id",
        "annotation_role", "annotator_id", "annotation_session_id",
        "coverage_sha256", "segments_sha256", "coverage_rows",
        "segment_rows", "submitted_utc", "checker_claim",
    }
    receipt = read_json_exact(path, fields)
    if (
        receipt["submission_receipt_schema"]
        != "offline_contact_annotation_ui_submission_v1"
        or receipt["ui_build_id"] != UI_BUILD_ID
    ):
        fail("UI submission receipt schema/build differs")
    for field in (
        "packet_id", "annotation_role", "annotator_id", "annotation_session_id"
    ):
        if receipt[field] != identity[field]:
            fail(f"UI submission receipt differs for {field}")
    if (
        receipt["coverage_sha256"] != sha256_file(coverage_path)
        or receipt["segments_sha256"] != sha256_file(segments_path)
    ):
        fail("UI submission receipt CSV hash differs")
    if (
        type(receipt["coverage_rows"]) is not int
        or type(receipt["segment_rows"]) is not int
        or receipt["checker_claim"]
        != "not evaluated here; coordinator must run the existing formal checker after isolated collection"
    ):
        fail("UI submission receipt row counts or checker claim differ")
    validate_iso_utc(receipt["submitted_utc"], "UI submission receipt")
    return receipt


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_backup(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_commit_staged(
    packet_root: Path,
    state_dir: Path,
    coverage_stage: Path,
    segments_stage: Path,
) -> dict[str, str]:
    if state_dir.stat().st_dev != packet_root.stat().st_dev:
        fail("state and packet must be on the same filesystem for atomic rename")
    for path in (coverage_stage, segments_stage):
        require_private_state_file(path)
        os.chmod(path, 0o600)
        if path.stat().st_dev != packet_root.stat().st_dev:
            fail(f"{path.name}: staged CSV is not on the packet filesystem")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    suffix = f"{stamp}.{os.getpid()}.csv"
    coverage_backup = state_dir / f"coverage.precommit.{suffix}"
    segments_backup = state_dir / f"segments.precommit.{suffix}"
    coverage_target = packet_root / "coverage.csv"
    segments_target = packet_root / "segments.csv"
    write_backup(coverage_backup, coverage_target.read_bytes())
    write_backup(segments_backup, segments_target.read_bytes())
    fsync_directory(state_dir)

    segments_replaced = False
    coverage_replaced = False
    try:
        os.replace(segments_stage, segments_target)
        segments_replaced = True
        fsync_directory(packet_root)
        os.replace(coverage_stage, coverage_target)
        coverage_replaced = True
        fsync_directory(packet_root)
    except Exception:
        if coverage_replaced and coverage_backup.exists():
            os.replace(coverage_backup, coverage_target)
        if segments_replaced and segments_backup.exists():
            os.replace(segments_backup, segments_target)
        fsync_directory(packet_root)
        raise
    return {
        "coverage_backup": str(coverage_backup),
        "segments_backup": str(segments_backup),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--ui-file", type=Path, default=DEFAULT_UI)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate an assigned, blank packet without writing")
    mode.add_argument("--initialize", action="store_true", help="validate and write an identity-locked state receipt")
    mode.add_argument("--validate-output", action="store_true", help="validate submitted UI CSV output")
    mode.add_argument(
        "--commit-staged-output",
        action="store_true",
        help="validate state-staged CSVs and atomically replace the two packet CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_root = args.packet_root.expanduser()
    media_root = args.media_root.expanduser()
    state_dir = args.state_dir.expanduser()
    ui_file = args.ui_file.expanduser()
    if not all(
        path.is_absolute()
        for path in (packet_root, media_root, state_dir, ui_file)
    ):
        fail("packet, media, state, and UI paths must be absolute")
    if not packet_root.is_dir() or not media_root.is_dir():
        fail("packet or media root does not exist")
    if not state_dir.exists():
        if not args.initialize:
            fail("state directory does not exist; --dry-run/--validate-output never creates it")
        state_dir.mkdir(mode=0o700, parents=False)
    if not state_dir.is_dir() or state_dir.is_symlink():
        fail("state directory must be a real directory")
    require_private(state_dir, directory=True, writable=True)
    ensure_separate_roots(packet_root, media_root, state_dir)
    if not ui_file.is_file() or ui_file.is_symlink():
        fail("UI file must be a regular, non-symlink file")

    identity = validate_identity(packet_root)
    tasks = validate_tasks(packet_root)
    timeline, media_hashes = validate_timeline(packet_root, tasks)
    validate_packet_layout(packet_root)
    validate_media_layout(media_root, tasks)
    validate_receipts(packet_root, identity, tasks, media_hashes)
    validate_media(media_root, media_hashes)
    csv_roundtrip(packet_root / "tasks.csv", TASK_FIELDS)
    csv_roundtrip(packet_root / "timeline_evidence_index.csv", EVIDENCE_FIELDS)
    validate_state_layout(state_dir, allow_empty=args.dry_run or args.initialize)

    state_receipt = None
    if not args.dry_run:
        if args.initialize:
            # initialize_state below may create the receipt for the first time.
            pass
        else:
            state_receipt = validate_state_session(
                state_dir,
                packet_root,
                ui_file,
                identity,
                tasks,
                media_hashes,
            )

    phase = "submission" if args.validate_output else "template"
    coverage_path = packet_root / "coverage.csv"
    segments_path = packet_root / "segments.csv"
    if args.commit_staged_output:
        coverage_stage = state_dir / "coverage.submission.csv"
        segments_stage = state_dir / "segments.submission.csv"
        for path in (coverage_stage, segments_stage):
            if not path.is_file() or path.is_symlink():
                fail(f"state directory lacks staged {path.name}")
        # The packet must still be the assigned blank template.  This prevents
        # an unreviewed UI session from silently replacing a prior submission.
        validate_annotation_csvs(
            coverage_path, segments_path, identity, tasks, "template"
        )
        coverages, segments = validate_annotation_csvs(
            coverage_stage, segments_stage, identity, tasks, "submission"
        )
        submission_receipt = validate_submission_receipt(
            state_dir, identity, coverage_stage, segments_stage
        )
        if submission_receipt["coverage_rows"] != len(coverages) or submission_receipt[
            "segment_rows"
        ] != len(segments):
            fail("UI submission receipt row counts differ")
        backups = atomic_commit_staged(
            packet_root,
            state_dir,
            coverage_stage,
            segments_stage,
        )
        validate_packet_layout(packet_root)
        coverages, segments = validate_annotation_csvs(
            coverage_path, segments_path, identity, tasks, "submission"
        )
    else:
        coverages, segments = validate_annotation_csvs(
            coverage_path, segments_path, identity, tasks, phase
        )
        backups = None

    session_receipt = None
    if args.initialize:
        session_receipt = initialize_state(
            state_dir, packet_root, ui_file, identity, tasks, media_hashes
        )
        state_receipt = validate_state_session(
            state_dir,
            packet_root,
            ui_file,
            identity,
            tasks,
            media_hashes,
        )
    result = {
        "status": "valid",
        "mode": (
            "commit_staged_output"
            if args.commit_staged_output
            else "validate_output"
            if args.validate_output
            else "initialize"
            if args.initialize
            else "dry_run"
        ),
        "packet_id": identity["packet_id"],
        "annotation_role": identity["annotation_role"],
        "annotator_id": identity["annotator_id"],
        "annotation_session_id": identity["annotation_session_id"],
        "tasks": len(tasks),
        "episodes": [int(row["episode_id"]) for row in tasks],
        "timeline_rows": len(timeline),
        "media_files": len(media_hashes),
        "coverages": len(coverages),
        "segments": len(segments),
        "csv_roundtrip": True,
        "episode_set_exact": True,
        "permissions_private": True,
        "packet_state_roots_disjoint": True,
        "media_root_external": True,
        "packet_root_exact_seven_files": True,
        "atomic_packet_replacement": bool(args.commit_staged_output),
        "precommit_backups": backups,
        "session_receipt": str(session_receipt) if session_receipt else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

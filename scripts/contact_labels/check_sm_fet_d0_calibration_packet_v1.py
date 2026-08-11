#!/usr/bin/env python3
"""Fail-closed structural checker for the SM-FET D0 calibration packet.

This checker never creates labels, proposals, GT, media, or receipts.  It only
validates the frozen fix2 packet contract with Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


PACKET_VERSION = "sm_fet_d0_calibration_packet_v1_fix2"
SCHEMA_MANIFEST_SHA256 = "8f33609a7c320052781b930543e46d174ae31f0528dd02e03e610df73e9802b7"
BLIND_EPISODES = {0, 9, 12, 34, 36, 38, 43, 50, 51, 53, 54, 58}
CALIBRATION_EPISODES = [46, 65, 1]

EXPECTED_ASSIGNMENT_ARTIFACTS = {
    "independent_a": {
        "assignment_receipt.json": "54e17f8254ca74357d016dabcf94c6e250fb3bc2eff152b920dac779aa16c8b3",
        "recipient_manifest.json": "febc2298ab7f104b9089cc1ef2b10b7a30958ef2d011fbe6141f06deba4c7bfc",
    },
    "independent_b": {
        "assignment_receipt.json": "543786f9e7be710113bf41027edf9f4631dcf500d1448eb1e91c7c61b3c4eae7",
        "recipient_manifest.json": "67a712e1b5eb6a547665bea9f5cf00aae15d00faf0614f296e7e6f32e33d5ba1",
    },
}

HEADERS = {
    "task_manifest.csv": "schema_version,task_id,dataset_id,episode_id,media_id,frame_clock,coverage_start_frame,coverage_end_frame,required_review,calibration_only,formal_blind_overlap",
    "media_allowlist.csv": "schema_version,task_id,media_id,media_filename,video_bytes,video_sha256,video_frames,video_duration_s,video_fps,video_width,video_height,video_codec,hash_status",
    "coverage.csv": "schema_version,dataset_id,episode_id,coverage_id,annotation_role,annotator_id,annotation_session_id,review_status,coverage_mode,start_frame,end_frame,frame_clock,evidence_scope,label_origin,review_timestamp_utc,parent_coverage_ids,notes",
    "segments.csv": "schema_version,dataset_id,episode_id,coverage_id,segment_id,onset_event_id,release_event_id,annotation_role,annotator_id,annotation_session_id,review_status,start_frame,end_frame,start_boundary,end_boundary,side,site,object_role,contact_mode,confidence,evidence_scope,label_origin,source_proposal_id,parent_segment_ids,notes",
    "events_v1.csv": "schema_version,dataset_id,episode_id,coverage_id,event_id,event_order,task_family,task_ontology_version,event_kind,focus_role,focus_entity_id,support_start_frame,support_end_frame,start_frame_lo,start_frame_hi,start_boundary,end_frame_lo,end_frame_hi,end_boundary,object_effect,outcome,event_kind_review_state,effect_review_state,outcome_review_state,confidence,evidence_scope,evidence_tier,active_hand,active_hand_review_state,annotation_role,annotator_id,annotation_session_id,review_status,label_origin,parent_event_ids,notes",
    "event_node_snapshots_v1.csv": "schema_version,dataset_id,episode_id,event_id,snapshot_role,frame_index,node_slot,node_role,entity_id,binding_provenance,observability,center_cell,scale_band,orientation_2d,observability_review_state,center_review_state,scale_review_state,orientation_review_state,evidence_scope,evidence_tier,annotation_role,annotator_id,annotation_session_id,review_status,label_origin,parent_snapshot_ids",
    "event_edges_v1.csv": "schema_version,dataset_id,episode_id,edge_id,event_id,edge_order,effector_side,primary_role,primary_entity_id,primary_part,primary_surface_cell,secondary_present,secondary_role,contact_segment_id,contact_mode,relation_before,relation_after,primary_part_review_state,primary_cell_review_state,contact_mode_review_state,relation_before_review_state,relation_after_review_state,confidence,evidence_scope,evidence_tier,annotation_role,annotator_id,annotation_session_id,review_status,label_origin,parent_edge_ids",
    "pose_keys_2d_v1.csv": "schema_version,dataset_id,episode_id,pose_key_id,event_id,pose_role,frame_index,hand_side,reference_entity_id,box_x0_norm,box_y0_norm,box_x1_norm,box_y1_norm,wrist_x_norm,wrist_y_norm,wrist_visibility,palm_center_x_norm,palm_center_y_norm,palm_center_visibility,thumb_tip_x_norm,thumb_tip_y_norm,thumb_tip_visibility,index_tip_x_norm,index_tip_y_norm,index_tip_visibility,middle_tip_x_norm,middle_tip_y_norm,middle_tip_visibility,ring_tip_x_norm,ring_tip_y_norm,ring_tip_visibility,little_tip_x_norm,little_tip_y_norm,little_tip_visibility,aperture_band,grasp_family,palm_axis_bin,confidence,evidence_scope,annotation_role,annotator_id,annotation_session_id,review_status,label_origin,parent_pose_key_ids",
}

ANNOTATION_FILES = {
    "coverage.csv",
    "segments.csv",
    "events_v1.csv",
    "event_node_snapshots_v1.csv",
    "event_edges_v1.csv",
    "pose_keys_2d_v1.csv",
}

RECIPIENT_FILES = {
    "ANNOTATION_HANDBOOK.md",
    "TASK_ONTOLOGY.json",
    "assignment_receipt.json",
    "coverage.csv",
    "event_edges_v1.csv",
    "event_node_snapshots_v1.csv",
    "events_v1.csv",
    "identity.json",
    "media_allowlist.csv",
    "pose_keys_2d_v1.csv",
    "recipient_exact_allowlist.json",
    "recipient_manifest.json",
    "schema_manifest.json",
    "segments.csv",
    "session.json",
    "submission_receipt.json",
    "task_manifest.csv",
}

SOURCE_TOP_LEVEL = {
    "README.md",
    "assignments",
    "coordinator",
    "frozen_inputs",
    "media_allowlist.csv",
    "schema_manifest.json",
    "task_manifest.csv",
    "templates",
}

IMMUTABLE_FILES = {
    "ANNOTATION_HANDBOOK.md",
    "TASK_ONTOLOGY.json",
    "assignment_receipt.json",
    "media_allowlist.csv",
    "recipient_exact_allowlist.json",
    "recipient_manifest.json",
    "schema_manifest.json",
    "task_manifest.csv",
}

LIFECYCLE_FILES = {"identity.json", "session.json", "submission_receipt.json"}

ASSIGNMENT_MODES = {
    **{name: 0o400 for name in IMMUTABLE_FILES},
    **{name: 0o600 for name in ANNOTATION_FILES | LIFECYCLE_FILES},
}

SUBMISSION_MODES = {name: 0o400 for name in RECIPIENT_FILES}

FORBIDDEN_PACKET_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".h5", ".hdf5", ".npz",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
}

SELECTION_LEAK_MARKERS = (
    "meniscus",
    "regrasp",
    "双手同时",
    "手—杯",
    "紫色",
    "近无色",
)


class CheckFailure(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CheckFailure(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON duplicate key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def validate_text_bytes(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    if not data or data.startswith(b"\xef\xbb\xbf") or b"\x00" in data or b"\r" in data:
        fail(f"text encoding/newline contract failed: {path}")
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        fail(f"file must end with exactly one LF: {path}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"not UTF-8: {path}: {exc}")
    return data


def validate_regular_file(path: Path, *, exact_mode: int | None = None) -> None:
    if path.is_symlink():
        fail(f"symlink forbidden: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        fail(f"missing/unreadable file {path}: {exc}")
    if not stat.S_ISREG(info.st_mode):
        fail(f"not a regular file: {path}")
    if info.st_nlink != 1:
        fail(f"hardlink forbidden: {path}")
    if exact_mode is not None and stat.S_IMODE(info.st_mode) != exact_mode:
        fail(f"mode must be {exact_mode:04o}: {path} is {stat.S_IMODE(info.st_mode):04o}")


def validate_directory(path: Path, *, exact_mode: int | None = None) -> None:
    if path.is_symlink() or not path.is_dir():
        fail(f"real directory required: {path}")
    if exact_mode is not None:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != exact_mode:
            fail(f"directory mode must be {exact_mode:04o}: {path} is {mode:04o}")


def exact_children(path: Path, expected: set[str]) -> None:
    actual = {entry.name for entry in path.iterdir()}
    if actual != expected:
        fail(f"exact allowlist mismatch at {path}: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")


def read_csv(path: Path, expected_header: str) -> list[dict[str, str]]:
    data = validate_text_bytes(path)
    first_line = data.split(b"\n", 1)[0].decode("utf-8")
    if first_line != expected_header:
        fail(f"exact CSV header mismatch: {path}")
    try:
        rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
    except csv.Error as exc:
        fail(f"invalid CSV {path}: {exc}")
    if len(set(expected_header.split(","))) != len(expected_header.split(",")):
        fail(f"checker contains duplicate expected column: {path.name}")
    return rows


def integer(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError:
        fail(f"{field} must be an integer, got {value!r}")


def validate_task_media(root: Path) -> dict[int, dict[str, Any]]:
    tasks = read_csv(root / "task_manifest.csv", HEADERS["task_manifest.csv"])
    media = read_csv(root / "media_allowlist.csv", HEADERS["media_allowlist.csv"])
    if len(tasks) != 3 or len(media) != 3:
        fail("task/media manifests must each contain exactly three rows")
    episodes = [integer(row["episode_id"], "episode_id") for row in tasks]
    if episodes != CALIBRATION_EPISODES or len(set(episodes)) != 3 or set(episodes) & BLIND_EPISODES:
        fail(f"calibration episode contract failed: {episodes}")
    media_by_task = {row["task_id"]: row for row in media}
    if len(media_by_task) != 3:
        fail("duplicate media task_id")
    result: dict[int, dict[str, Any]] = {}
    for task in tasks:
        if task["schema_version"] != PACKET_VERSION:
            fail("task schema_version mismatch")
        if task["frame_clock"] != "raw_video_frame_30fps" or task["coverage_start_frame"] != "0":
            fail("task clock/coverage start mismatch")
        if task["required_review"] != "full_episode" or task["calibration_only"] != "true" or task["formal_blind_overlap"] != "false":
            fail("task calibration flags mismatch")
        media_row = media_by_task.get(task["task_id"])
        if media_row is None or media_row["media_id"] != task["media_id"]:
            fail("task/media binding mismatch")
        if media_row["schema_version"] != PACKET_VERSION:
            fail("media schema_version mismatch")
        frames = integer(media_row["video_frames"], "video_frames")
        if integer(task["coverage_end_frame"], "coverage_end_frame") != frames:
            fail("coverage end must equal video frame count")
        if media_row["media_filename"] != f"{task['task_id']}.mp4":
            fail("neutral media filename mismatch")
        if len(media_row["video_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in media_row["video_sha256"]):
            fail("invalid media SHA-256")
        episode = integer(task["episode_id"], "episode_id")
        result[episode] = {"task": task, "media": media_row, "frames": frames}
    return result


def validate_schema_manifest(source_root: Path) -> dict[str, Any]:
    path = source_root / "schema_manifest.json"
    validate_regular_file(path)
    validate_text_bytes(path)
    if sha256_path(path) != SCHEMA_MANIFEST_SHA256:
        fail("schema manifest hash does not match frozen checker constant")
    manifest = load_json(path)
    if manifest.get("packet_schema_version") != PACKET_VERSION:
        fail("schema manifest packet version mismatch")
    schemas = manifest.get("csv_files")
    if not isinstance(schemas, dict) or set(schemas) != set(HEADERS):
        fail("schema manifest CSV set mismatch")
    for name, expected_header in HEADERS.items():
        entry = schemas[name]
        header_bytes = (expected_header + "\n").encode("utf-8")
        if entry.get("header") != expected_header or entry.get("header_sha256") != sha256_bytes(header_bytes):
            fail(f"frozen schema/header hash mismatch: {name}")
        if name in ANNOTATION_FILES:
            source_path = source_root / "templates" / name
            if entry.get("blank_file_sha256") != sha256_path(source_path):
                fail(f"frozen blank annotation hash mismatch: {name}")
        else:
            source_path = source_root / name
            if entry.get("assignment_file_sha256") != sha256_path(source_path):
                fail(f"frozen immutable CSV hash mismatch: {name}")
    return manifest


def validate_blank_annotations(root: Path, receipt: dict[str, Any] | None = None) -> None:
    expected_hashes = receipt.get("blank_annotation_sha256", {}) if receipt else {}
    for name in ANNOTATION_FILES:
        rows = read_csv(root / name, HEADERS[name])
        if rows:
            fail(f"annotation file must be blank before assignment: {root/name}")
        expected = expected_hashes.get(name)
        if receipt is not None and expected != sha256_path(root / name):
            fail(f"blank annotation hash mismatch: {name}")


def validate_recipient_manifest(root: Path, *, phase: str) -> None:
    manifest = load_json(root / "recipient_manifest.json")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != RECIPIENT_FILES - {"recipient_manifest.json"}:
        fail("recipient manifest file set mismatch")
    for name, expected in hashes.items():
        if phase == "submission" and name in ANNOTATION_FILES | LIFECYCLE_FILES:
            continue
        if sha256_path(root / name) != expected:
            fail(f"recipient manifest hash mismatch: {name}")


def parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        fail(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        fail(f"{field} is not a valid ISO-8601 timestamp")
    return parsed


def content_lock(receipt: dict[str, Any]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "content_lock_sha256"}
    wire = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(wire)


def validate_terminal_lifecycle(identity: dict[str, Any], session: dict[str, Any], submission: dict[str, Any]) -> None:
    if identity.get("identity_status") != "submitted" or session.get("session_status") != "submitted":
        fail("identity and session must both be in submitted state")
    if submission.get("submission_status") != "submitted" or submission.get("annotation_started") is not True:
        fail("submission receipt must be submitted with annotation_started=true")
    if submission.get("write_locked") is not True or submission.get("resigned_by_role") != "coordinator":
        fail("submission receipt must be coordinator-resigned and write-locked")
    opened = parse_utc(session.get("opened_at_utc"), "session.opened_at_utc")
    closed = parse_utc(session.get("closed_at_utc"), "session.closed_at_utc")
    submitted_at = parse_utc(submission.get("submitted_at_utc"), "submission.submitted_at_utc")
    if not opened < closed or submitted_at != closed:
        fail("submission lifecycle timestamps must satisfy opened < closed == submitted")


def validate_allowlist_policy(allowlist: dict[str, Any]) -> None:
    if set(allowlist.get("exact_files", [])) != RECIPIENT_FILES:
        fail("recipient exact allowlist content mismatch")
    roles = allowlist.get("file_roles")
    expected_roles = {
        "immutable_frozen_input": IMMUTABLE_FILES,
        "mutable_annotation": ANNOTATION_FILES,
        "mutable_lifecycle_metadata": LIFECYCLE_FILES,
    }
    if not isinstance(roles, dict) or set(roles) != set(expected_roles):
        fail("recipient file-role policy set mismatch")
    for role, expected in expected_roles.items():
        if set(roles.get(role, [])) != expected:
            fail(f"recipient file-role membership mismatch: {role}")
    expected_modes = {
        "assignment": {"immutable_frozen_input": "0400", "mutable_annotation": "0600", "mutable_lifecycle_metadata": "0600"},
        "submission": {"immutable_frozen_input": "0400", "mutable_annotation": "0400", "mutable_lifecycle_metadata": "0400"},
    }
    if allowlist.get("phase_modes") != expected_modes:
        fail("recipient phase mode policy mismatch")
    expected_changes = {
        "identity.json": ["identity_status"],
        "session.json": ["session_status", "opened_at_utc", "closed_at_utc"],
        "submission_receipt.json": [
            "submission_status", "annotation_started", "submitted_at_utc", "write_locked",
            "identity_sha256", "session_sha256", "annotation_sha256", "content_lock_sha256", "resigned_by_role",
        ],
    }
    if allowlist.get("allowed_lifecycle_field_changes") != expected_changes:
        fail("lifecycle field-change whitelist mismatch")


def validate_packet_root(root: Path, schema_manifest_path: Path, *, phase: str, check_acl: bool) -> dict[str, Any]:
    if phase not in {"assignment", "submission"}:
        fail(f"unknown packet validation phase: {phase}")
    validate_directory(root, exact_mode=0o700 if check_acl else None)
    exact_children(root, RECIPIENT_FILES)
    expected_modes = ASSIGNMENT_MODES if phase == "assignment" else SUBMISSION_MODES
    seen_inodes: set[tuple[int, int]] = set()
    for entry in root.iterdir():
        validate_regular_file(entry, exact_mode=expected_modes[entry.name] if check_acl else None)
        if entry.suffix.lower() in FORBIDDEN_PACKET_SUFFIXES:
            fail(f"media/HDF5/image forbidden in packet: {entry}")
        inode = (entry.stat().st_dev, entry.stat().st_ino)
        if inode in seen_inodes:
            fail(f"duplicate inode in packet: {entry}")
        seen_inodes.add(inode)
        text = validate_text_bytes(entry).decode("utf-8")
        if "/home/" in text or "file://" in text:
            fail(f"absolute source path leaked into recipient: {entry}")
        if entry.name not in {"ANNOTATION_HANDBOOK.md", "TASK_ONTOLOGY.json"} and any(marker in text for marker in SELECTION_LEAK_MARKERS):
            fail(f"episode-selection rationale leaked into recipient: {entry}")
    allowlist = load_json(root / "recipient_exact_allowlist.json")
    validate_allowlist_policy(allowlist)
    for forbidden in ("README.md", "coordinator", "selection_proposal_audit.md"):
        if forbidden not in set(allowlist.get("forbidden_recipient_paths", [])):
            fail(f"recipient denylist missing {forbidden}")
    receipt = load_json(root / "assignment_receipt.json")
    identity = load_json(root / "identity.json")
    session = load_json(root / "session.json")
    submission = load_json(root / "submission_receipt.json")
    identity_keys = {"identity_schema_version", "packet_schema_version", "packet_id", "annotator_id", "annotation_role", "identity_status"}
    session_keys = {
        "session_schema_version", "packet_schema_version", "packet_id", "annotation_session_id", "annotator_id",
        "session_status", "opened_at_utc", "closed_at_utc", "protocol_sha256", "handbook_sha256",
        "task_ontology_sha256", "task_manifest_sha256", "media_allowlist_sha256", "schema_manifest_sha256",
    }
    submission_keys = {
        "submission_receipt_schema_version", "packet_schema_version", "packet_id", "annotation_role", "annotator_id",
        "annotation_session_id", "assignment_receipt_sha256", "submission_status", "annotation_started",
        "submitted_at_utc", "write_locked", "identity_sha256", "session_sha256", "annotation_sha256",
        "content_lock_sha256", "resigned_by_role",
    }
    if set(identity) != identity_keys or set(session) != session_keys or set(submission) != submission_keys:
        fail("lifecycle metadata key set violates the field-change whitelist")
    if identity.get("identity_schema_version") != "sm_fet_d0_identity_v1_fix2" or identity.get("packet_schema_version") != PACKET_VERSION:
        fail("identity schema version mismatch")
    if session.get("session_schema_version") != "sm_fet_d0_session_v1_fix2" or session.get("packet_schema_version") != PACKET_VERSION:
        fail("session schema version mismatch")
    if submission.get("submission_receipt_schema_version") != "sm_fet_d0_submission_receipt_v1_fix2" or submission.get("packet_schema_version") != PACKET_VERSION:
        fail("submission receipt schema version mismatch")
    if receipt.get("packet_schema_version") != PACKET_VERSION:
        fail("receipt packet schema mismatch")
    role = receipt.get("annotation_role")
    expected_artifacts = EXPECTED_ASSIGNMENT_ARTIFACTS.get(role)
    if expected_artifacts is None:
        fail("unknown assignment role")
    for name, expected_hash in expected_artifacts.items():
        if sha256_path(root / name) != expected_hash:
            fail(f"coordinator-frozen assignment artifact mismatch: {name}")
    identity_tuple = (identity.get("packet_id"), identity.get("annotator_id"), identity.get("annotation_role"))
    receipt_tuple = (receipt.get("packet_id"), receipt.get("annotator_id"), receipt.get("annotation_role"))
    expected_identity_status = "assigned_by_coordinator" if phase == "assignment" else "submitted"
    if identity_tuple != receipt_tuple or identity.get("identity_status") != expected_identity_status:
        fail("identity/receipt binding mismatch")
    if session.get("packet_id") != receipt.get("packet_id") or session.get("annotator_id") != receipt.get("annotator_id"):
        fail("session/receipt binding mismatch")
    expected_session_status = "assigned_not_started" if phase == "assignment" else "submitted"
    if session.get("annotation_session_id") != receipt.get("annotation_session_id") or session.get("session_status") != expected_session_status:
        fail("session status/binding mismatch")
    if not all(receipt.get(key) is True for key in ("calibration_only", "non_gt")):
        fail("receipt must be calibration-only and non-GT")
    if receipt.get("assignment_status") != "assigned_not_started" or receipt.get("annotation_started") is not False or receipt.get("formal_blind_overlap") is not False:
        fail("assignment must be unstarted and blind-disjoint")
    if receipt.get("selection_proposal_distributed") is not False or receipt.get("source_media_copied_into_packet") is not False:
        fail("proposal/media distribution boundary failed")
    if sha256_path(schema_manifest_path) != SCHEMA_MANIFEST_SHA256 or sha256_path(root / "schema_manifest.json") != SCHEMA_MANIFEST_SHA256:
        fail("external/recipient schema manifest bytes do not match frozen checker")
    if sha256_path(schema_manifest_path) != sha256_path(root / "schema_manifest.json"):
        fail("external and recipient schema manifests differ")
    frozen = receipt.get("frozen_inputs")
    expected_frozen = {
        "annotation_handbook": ("ANNOTATION_HANDBOOK.md", 40881, "27e70a8b4ff69d4aae369dfb11e229e8daccda750befcdfc964182f4e70436c4"),
        "task_ontology": ("TASK_ONTOLOGY.json", 770, "546acd3d5a3cc3be529fc7c20b458065502a198126d62cf49d71b27ae1518f1a"),
        "schema_manifest": ("schema_manifest.json", 7525, SCHEMA_MANIFEST_SHA256),
    }
    if not isinstance(frozen, dict) or set(frozen) != set(expected_frozen):
        fail("assignment receipt frozen-input set mismatch")
    for key, (path_name, byte_count, digest) in expected_frozen.items():
        entry = frozen.get(key)
        if entry != {"recipient_path": path_name, "bytes": byte_count, "sha256": digest}:
            fail(f"assignment receipt frozen-input binding mismatch: {key}")
        path = root / path_name
        if path.stat().st_size != byte_count or sha256_path(path) != digest:
            fail(f"recipient frozen-input bytes mismatch: {path_name}")
    immutable = {
        "task_manifest.csv": "task_manifest_sha256",
        "media_allowlist.csv": "media_allowlist_sha256",
        "recipient_exact_allowlist.json": "recipient_exact_allowlist_sha256",
    }
    for name, key in immutable.items():
        if sha256_path(root / name) != receipt.get(key):
            fail(f"receipt immutable hash mismatch: {name}")
    expected_session_static = {
        "packet_id": receipt.get("packet_id"),
        "annotation_session_id": receipt.get("annotation_session_id"),
        "annotator_id": receipt.get("annotator_id"),
        "protocol_sha256": "c4f1f811f796e7c8604a0d0510229677693d6d1cc501b78f59f01cd4793e4a92",
        "handbook_sha256": "27e70a8b4ff69d4aae369dfb11e229e8daccda750befcdfc964182f4e70436c4",
        "task_ontology_sha256": "546acd3d5a3cc3be529fc7c20b458065502a198126d62cf49d71b27ae1518f1a",
        "task_manifest_sha256": sha256_path(root / "task_manifest.csv"),
        "media_allowlist_sha256": sha256_path(root / "media_allowlist.csv"),
        "schema_manifest_sha256": sha256_path(root / "schema_manifest.json"),
    }
    for key, expected in expected_session_static.items():
        if session.get(key) != expected:
            fail(f"session immutable field changed outside whitelist: {key}")
    validate_task_media(root)
    for name in HEADERS:
        read_csv(root / name, HEADERS[name])
    validate_recipient_manifest(root, phase=phase)
    baseline = receipt.get("lifecycle_baseline_sha256")
    if not isinstance(baseline, dict) or set(baseline) != {"identity_json", "session_json"}:
        fail("assignment receipt lifecycle baseline missing")
    if phase == "assignment":
        if baseline["identity_json"] != sha256_path(root / "identity.json") or baseline["session_json"] != sha256_path(root / "session.json"):
            fail("assignment lifecycle baseline hash mismatch")
        if session.get("opened_at_utc") or session.get("closed_at_utc"):
            fail("assignment session timestamps must be blank")
        if submission.get("submission_status") != "not_submitted" or submission.get("annotation_started") is not False:
            fail("assignment submission receipt must be not_submitted/unstarted")
        if submission.get("submitted_at_utc") or submission.get("write_locked") is not False:
            fail("assignment submission receipt must be open and timestamp-free")
        if any(submission.get(key) for key in ("identity_sha256", "session_sha256", "annotation_sha256", "content_lock_sha256", "resigned_by_role")):
            fail("assignment submission receipt contains premature lock material")
        validate_blank_annotations(root, receipt)
    else:
        validate_terminal_lifecycle(identity, session, submission)
        static_tuple = (submission.get("packet_id"), submission.get("annotator_id"), submission.get("annotation_role"), submission.get("annotation_session_id"))
        expected_tuple = (receipt.get("packet_id"), receipt.get("annotator_id"), receipt.get("annotation_role"), receipt.get("annotation_session_id"))
        if static_tuple != expected_tuple or submission.get("assignment_receipt_sha256") != sha256_path(root / "assignment_receipt.json"):
            fail("submission receipt immutable assignment binding mismatch")
        if submission.get("identity_sha256") != sha256_path(root / "identity.json") or submission.get("session_sha256") != sha256_path(root / "session.json"):
            fail("submission receipt lifecycle hashes mismatch")
        annotation_hashes = submission.get("annotation_sha256")
        if not isinstance(annotation_hashes, dict) or set(annotation_hashes) != ANNOTATION_FILES:
            fail("submission receipt annotation hash set mismatch")
        for name in ANNOTATION_FILES:
            if annotation_hashes[name] != sha256_path(root / name):
                fail(f"submission receipt annotation hash mismatch: {name}")
        if submission.get("content_lock_sha256") != content_lock(submission):
            fail("submission receipt content lock mismatch")
    return {"receipt": receipt, "identity": identity, "session": session, "submission": submission, "inodes": seen_inodes}


def nested_or_equal(first: Path, second: Path) -> bool:
    a = first.resolve()
    b = second.resolve()
    return a == b or a in b.parents or b in a.parents


def validate_peer(first_root: Path, first: dict[str, Any], second_root: Path, second: dict[str, Any]) -> None:
    if nested_or_equal(first_root, second_root):
        fail("A/B packet roots must be separate and non-nested")
    for field in ("packet_id", "annotator_id", "annotation_role", "annotation_session_id"):
        if first["receipt"].get(field) == second["receipt"].get(field):
            fail(f"A/B {field} must differ")
    if first["inodes"] & second["inodes"]:
        fail("A/B packet files must not share inodes/hardlinks")
    for name in ("ANNOTATION_HANDBOOK.md", "TASK_ONTOLOGY.json", "schema_manifest.json", "task_manifest.csv", "media_allowlist.csv", "recipient_exact_allowlist.json"):
        if sha256_path(first_root / name) != sha256_path(second_root / name):
            fail(f"A/B immutable neutral input mismatch: {name}")


def validate_media_root(media_root: Path, state_root: Path, packet_root: Path, peer_media_root: Path | None) -> None:
    for root in (media_root, state_root):
        validate_directory(root, exact_mode=0o700)
    if any(nested_or_equal(a, b) for a, b in ((media_root, state_root), (media_root, packet_root), (state_root, packet_root))):
        fail("packet/media/state roots must be pairwise non-nested")
    if list(state_root.iterdir()):
        fail("state root must be empty before annotation starts")
    media_rows = read_csv(packet_root / "media_allowlist.csv", HEADERS["media_allowlist.csv"])
    expected = {row["media_filename"]: row for row in media_rows}
    exact_children(media_root, set(expected))
    media_inodes: set[tuple[int, int]] = set()
    for name, row in expected.items():
        path = media_root / name
        validate_regular_file(path, exact_mode=0o400)
        if path.stat().st_size != integer(row["video_bytes"], "video_bytes") or sha256_path(path) != row["video_sha256"]:
            fail(f"media bytes/hash mismatch: {path}")
        media_inodes.add((path.stat().st_dev, path.stat().st_ino))
    if peer_media_root is not None:
        validate_directory(peer_media_root, exact_mode=0o700)
        peer_inodes = {(p.stat().st_dev, p.stat().st_ino) for p in peer_media_root.iterdir() if p.is_file()}
        if media_inodes & peer_inodes:
            fail("A/B media roots must not share hardlinks/inodes")


def row_identity(row: dict[str, str], identity: dict[str, Any], session: dict[str, Any], filename: str) -> None:
    if row.get("schema_version") != PACKET_VERSION:
        fail(f"row schema_version mismatch in {filename}")
    if row.get("annotator_id") != identity.get("annotator_id") or row.get("annotation_role") != identity.get("annotation_role"):
        fail(f"row identity mismatch in {filename}")
    if row.get("annotation_session_id") != session.get("annotation_session_id"):
        fail(f"row session mismatch in {filename}")
    if row.get("label_origin") != "human_independent_calibration":
        fail(f"label_origin must be human_independent_calibration in {filename}")
    if row.get("review_status") != "submitted":
        fail(f"review_status must be submitted in every submission row: {filename}")


def validate_reference_rows(tables: dict[str, list[dict[str, str]]], identity: dict[str, Any], session: dict[str, Any], tasks: dict[int, dict[str, Any]]) -> None:
    for filename, rows in tables.items():
        for row in rows:
            row_identity(row, identity, session, filename)
            episode = integer(row.get("episode_id", ""), f"{filename}.episode_id")
            if episode not in tasks:
                fail(f"row episode not assigned in {filename}: {episode}")
    def unique_index(filename: str, key: str) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for row in tables[filename]:
            value = row.get(key, "")
            if not value or value in result:
                fail(f"blank/duplicate {filename}.{key}: {value!r}")
            result[value] = row
        return result
    coverage = unique_index("coverage.csv", "coverage_id")
    segments = unique_index("segments.csv", "segment_id")
    events = unique_index("events_v1.csv", "event_id")
    unique_index("event_edges_v1.csv", "edge_id")
    unique_index("pose_keys_2d_v1.csv", "pose_key_id")
    coverage_by_episode: dict[int, int] = {}
    for row in tables["coverage.csv"]:
        episode = integer(row["episode_id"], "coverage episode")
        coverage_by_episode[episode] = coverage_by_episode.get(episode, 0) + 1
        start = integer(row["start_frame"], "coverage start")
        end = integer(row["end_frame"], "coverage end")
        if start != 0 or end != tasks[episode]["frames"]:
            fail("each calibration coverage row must span the full assigned episode")
        if row["frame_clock"] != "raw_video_frame_30fps" or row["coverage_mode"] != "EXHAUSTIVE":
            fail("submitted D0 coverage must use full-episode EXHAUSTIVE mode")
    if set(coverage_by_episode) != set(tasks) or any(count != 1 for count in coverage_by_episode.values()):
        fail("submission requires exactly one full-episode coverage row per assigned episode")
    def same_episode(child: dict[str, str], parent: dict[str, str], relation: str) -> None:
        if child["episode_id"] != parent["episode_id"]:
            fail(f"cross-episode reference forbidden: {relation}")
    for row in tables["events_v1.csv"]:
        parent = coverage.get(row["coverage_id"])
        if parent is None:
            fail("events_v1.csv.coverage_id is not closed")
        same_episode(row, parent, "event->coverage")
    for row in tables["segments.csv"]:
        parent = coverage.get(row["coverage_id"])
        if parent is None:
            fail("segments.csv.coverage_id is not closed")
        same_episode(row, parent, "segment->coverage")
        if row.get("source_proposal_id"):
            fail("source_proposal_id must remain empty for human calibration")
        for field in ("onset_event_id", "release_event_id"):
            if row.get(field):
                event = events.get(row[field])
                if event is None:
                    fail(f"segments.csv.{field} is not closed")
                same_episode(row, event, f"segment->{field}")
    for filename in ("event_node_snapshots_v1.csv", "event_edges_v1.csv", "pose_keys_2d_v1.csv"):
        for row in tables[filename]:
            event = events.get(row["event_id"])
            if event is None:
                fail(f"{filename}.event_id is not closed")
            same_episode(row, event, f"{filename}->event")
    for row in tables["event_edges_v1.csv"]:
        segment_id = row.get("contact_segment_id", "")
        if segment_id:
            segment = segments.get(segment_id)
            if segment is None:
                fail("event_edges_v1.csv.contact_segment_id is not closed")
            same_episode(row, segment, "edge->segment")


def phase_source(source_root: Path) -> dict[str, Any]:
    validate_directory(source_root)
    exact_children(source_root, SOURCE_TOP_LEVEL)
    exact_children(source_root / "coordinator", {"assignment_index.json", "selection_proposal_audit.md"})
    exact_children(source_root / "frozen_inputs", {"ANNOTATION_HANDBOOK.md", "TASK_ONTOLOGY.json"})
    exact_children(source_root / "templates", set(ANNOTATION_FILES) | {"identity.json", "session.json"})
    exact_children(source_root / "assignments", {"annotator_a", "annotator_b"})
    for path in source_root.rglob("*"):
        if path.is_file():
            validate_regular_file(path)
            if path.suffix.lower() in FORBIDDEN_PACKET_SUFFIXES:
                fail(f"media/HDF5/image found inside source bundle: {path}")
    validate_schema_manifest(source_root)
    frozen_source = {
        "ANNOTATION_HANDBOOK.md": (40881, "27e70a8b4ff69d4aae369dfb11e229e8daccda750befcdfc964182f4e70436c4"),
        "TASK_ONTOLOGY.json": (770, "546acd3d5a3cc3be529fc7c20b458065502a198126d62cf49d71b27ae1518f1a"),
    }
    for name, (byte_count, digest) in frozen_source.items():
        path = source_root / "frozen_inputs" / name
        if path.stat().st_size != byte_count or sha256_path(path) != digest:
            fail(f"source frozen input bytes mismatch: {name}")
    assignment_index = load_json(source_root / "coordinator" / "assignment_index.json")
    if assignment_index.get("packet_schema_version") != PACKET_VERSION or assignment_index.get("schema_manifest_sha256") != SCHEMA_MANIFEST_SHA256:
        fail("coordinator assignment index schema binding mismatch")
    indexed_recipients = assignment_index.get("recipients")
    if not isinstance(indexed_recipients, dict) or set(indexed_recipients) != {"annotator_a", "annotator_b"}:
        fail("coordinator assignment index recipient set mismatch")
    for name, values in indexed_recipients.items():
        if not isinstance(values, dict):
            fail("invalid coordinator assignment index entry")
        root = source_root / "assignments" / name
        checks = {
            "assignment_receipt_sha256": "assignment_receipt.json",
            "recipient_manifest_sha256": "recipient_manifest.json",
            "identity_sha256": "identity.json",
            "session_sha256": "session.json",
            "submission_receipt_sha256": "submission_receipt.json",
        }
        for index_key, filename in checks.items():
            if values.get(index_key) != sha256_path(root / filename):
                fail(f"coordinator assignment index hash mismatch: {name}/{filename}")
    tasks = validate_task_media(source_root)
    for name in ANNOTATION_FILES:
        rows = read_csv(source_root / "templates" / name, HEADERS[name])
        if rows:
            fail(f"source annotation template not blank: {name}")
    identity_template = load_json(source_root / "templates" / "identity.json")
    session_template = load_json(source_root / "templates" / "session.json")
    if identity_template.get("annotator_id") or identity_template.get("annotation_role") or identity_template.get("packet_id"):
        fail("source identity template must remain unassigned")
    if identity_template.get("identity_status") != "unassigned":
        fail("source identity template status mismatch")
    for key in ("packet_id", "annotation_session_id", "annotator_id", "opened_at_utc", "closed_at_utc", "protocol_sha256", "handbook_sha256", "task_ontology_sha256", "task_manifest_sha256", "media_allowlist_sha256", "schema_manifest_sha256"):
        if session_template.get(key):
            fail(f"source session template field must remain blank: {key}")
    if session_template.get("session_status") != "unassigned":
        fail("source session template status mismatch")
    a_root = source_root / "assignments" / "annotator_a"
    b_root = source_root / "assignments" / "annotator_b"
    a = validate_packet_root(a_root, source_root / "schema_manifest.json", phase="assignment", check_acl=True)
    b = validate_packet_root(b_root, source_root / "schema_manifest.json", phase="assignment", check_acl=True)
    validate_peer(a_root, a, b_root, b)
    for root in (a_root, b_root):
        if any((root / name).exists() for name in ("README.md", "coordinator", "selection_proposal_audit.md")):
            fail("coordinator proposal leaked into recipient root")
    return {"tasks": len(tasks), "recipients": 2, "annotation_templates": len(ANNOTATION_FILES)}


def phase_assignment(packet_root: Path, peer_root: Path, schema_manifest: Path) -> dict[str, Any]:
    first = validate_packet_root(packet_root, schema_manifest, phase="assignment", check_acl=True)
    peer = validate_packet_root(peer_root, schema_manifest, phase="assignment", check_acl=True)
    validate_peer(packet_root, first, peer_root, peer)
    return {"packet_id": first["receipt"]["packet_id"], "peer_packet_id": peer["receipt"]["packet_id"], "blank_files": len(ANNOTATION_FILES)}


def phase_submission(packet_root: Path, schema_manifest: Path) -> dict[str, Any]:
    packet = validate_packet_root(packet_root, schema_manifest, phase="submission", check_acl=True)
    tasks = validate_task_media(packet_root)
    tables = {name: read_csv(packet_root / name, HEADERS[name]) for name in ANNOTATION_FILES}
    validate_reference_rows(tables, packet["identity"], packet["session"], tasks)
    return {"packet_id": packet["receipt"]["packet_id"], "rows": sum(len(rows) for rows in tables.values()), "references_closed": True}


def phase_self_test() -> dict[str, Any]:
    identity = {"annotator_id": "test-a", "annotation_role": "independent_a"}
    session = {"annotation_session_id": "test-session-a"}
    def common(episode: int) -> dict[str, str]:
        return {
            "schema_version": PACKET_VERSION,
            "dataset_id": "ego_human_pour_66",
            "episode_id": str(episode),
            "annotation_role": "independent_a",
            "annotator_id": "test-a",
            "annotation_session_id": "test-session-a",
            "review_status": "submitted",
            "label_origin": "human_independent_calibration",
        }
    tasks = {46: {"frames": 324}, 65: {"frames": 384}, 1: {"frames": 599}}
    tables = {name: [] for name in ANNOTATION_FILES}
    tables["coverage.csv"] = [
        {**common(episode), "coverage_id": f"cov-{episode}", "coverage_mode": "EXHAUSTIVE", "start_frame": "0", "end_frame": str(values["frames"]), "frame_clock": "raw_video_frame_30fps"}
        for episode, values in tasks.items()
    ]
    validate_reference_rows(tables, identity, session, tasks)
    tables["events_v1.csv"] = [{**common(46), "coverage_id": "cov-46", "event_id": "evt-46-0"}]
    tables["segments.csv"] = [{**common(46), "coverage_id": "cov-46", "segment_id": "seg-46-0", "onset_event_id": "evt-46-0", "release_event_id": "", "source_proposal_id": ""}]
    tables["event_node_snapshots_v1.csv"] = [{**common(46), "event_id": "evt-46-0"}]
    tables["event_edges_v1.csv"] = [{**common(46), "edge_id": "edge-46-0", "event_id": "evt-46-0", "contact_segment_id": "seg-46-0"}]
    tables["pose_keys_2d_v1.csv"] = [{**common(46), "pose_key_id": "pose-46-0", "event_id": "evt-46-0"}]
    validate_reference_rows(tables, identity, session, tasks)
    broken = {name: [dict(row) for row in rows] for name, rows in tables.items()}
    broken["event_edges_v1.csv"][0]["contact_segment_id"] = "missing-segment"
    draft = {name: [dict(row) for row in rows] for name, rows in tables.items()}
    draft["coverage.csv"][0]["review_status"] = "draft"
    positive_only = {name: [dict(row) for row in rows] for name, rows in tables.items()}
    positive_only["coverage.csv"][0]["coverage_mode"] = "POSITIVE_ONLY"
    negatives = {
        "dangling_reference": lambda: validate_reference_rows(broken, identity, session, tasks),
        "draft_row": lambda: validate_reference_rows(draft, identity, session, tasks),
        "positive_only_coverage": lambda: validate_reference_rows(positive_only, identity, session, tasks),
        "assigned_not_started_state": lambda: validate_terminal_lifecycle(
            {"identity_status": "assigned_by_coordinator"},
            {"session_status": "assigned_not_started", "opened_at_utc": "", "closed_at_utc": ""},
            {"submission_status": "not_submitted", "annotation_started": False, "write_locked": False, "resigned_by_role": "", "submitted_at_utc": ""},
        ),
        "annotation_started_false": lambda: validate_terminal_lifecycle(
            {"identity_status": "submitted"},
            {"session_status": "submitted", "opened_at_utc": "2026-08-12T00:00:00Z", "closed_at_utc": "2026-08-12T01:00:00Z"},
            {"submission_status": "submitted", "annotation_started": False, "write_locked": True, "resigned_by_role": "coordinator", "submitted_at_utc": "2026-08-12T01:00:00Z"},
        ),
    }
    for name, attack in negatives.items():
        try:
            attack()
        except CheckFailure:
            continue
        fail(f"negative self-test unexpectedly passed: {name}")
    validate_terminal_lifecycle(
        {"identity_status": "submitted"},
        {"session_status": "submitted", "opened_at_utc": "2026-08-12T00:00:00Z", "closed_at_utc": "2026-08-12T01:00:00Z"},
        {"submission_status": "submitted", "annotation_started": True, "write_locked": True, "resigned_by_role": "coordinator", "submitted_at_utc": "2026-08-12T01:00:00Z"},
    )
    return {
        "positive_no_event_present_case": "passed",
        "positive_reference_case": "passed",
        "negative_attack_cases": sorted(negatives),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source", "assignment", "submission", "distribution", "self-test"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--peer-packet-root", type=Path)
    parser.add_argument("--schema-manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--peer-media-root", type=Path)
    parser.add_argument("--peer-state-root", type=Path)
    return parser.parse_args()


def required(value: Any, flag: str) -> Any:
    if value is None:
        fail(f"{flag} is required for this phase")
    return value


def main() -> int:
    args = parse_args()
    try:
        if args.phase == "source":
            details = phase_source(required(args.source_root, "--source-root"))
        elif args.phase == "assignment":
            details = phase_assignment(
                required(args.packet_root, "--packet-root"),
                required(args.peer_packet_root, "--peer-packet-root"),
                required(args.schema_manifest, "--schema-manifest"),
            )
        elif args.phase == "submission":
            details = phase_submission(
                required(args.packet_root, "--packet-root"),
                required(args.schema_manifest, "--schema-manifest"),
            )
        elif args.phase == "distribution":
            details = phase_assignment(
                required(args.packet_root, "--packet-root"),
                required(args.peer_packet_root, "--peer-packet-root"),
                required(args.schema_manifest, "--schema-manifest"),
            )
            validate_media_root(
                required(args.media_root, "--media-root"),
                required(args.state_root, "--state-root"),
                required(args.packet_root, "--packet-root"),
                required(args.peer_media_root, "--peer-media-root"),
            )
            validate_media_root(
                required(args.peer_media_root, "--peer-media-root"),
                required(args.peer_state_root, "--peer-state-root"),
                required(args.peer_packet_root, "--peer-packet-root"),
                required(args.media_root, "--media-root"),
            )
            distribution_roots = [
                required(args.packet_root, "--packet-root"),
                required(args.peer_packet_root, "--peer-packet-root"),
                required(args.media_root, "--media-root"),
                required(args.peer_media_root, "--peer-media-root"),
                required(args.state_root, "--state-root"),
                required(args.peer_state_root, "--peer-state-root"),
            ]
            for index, first in enumerate(distribution_roots):
                for second in distribution_roots[index + 1:]:
                    if nested_or_equal(first, second):
                        fail("all A/B packet/media/state roots must be pairwise non-nested")
            details["distribution_roots"] = "valid"
        else:
            details = phase_self_test()
    except CheckFailure as exc:
        print(json.dumps({"status": "invalid", "phase": args.phase, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"status": "valid", "phase": args.phase, **details}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Strict CSV I/O and frame-aligned encoding for contact_segment_v1_1."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_transition_schema import (  # noqa: E402
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    ContactSegment,
    encode_ordered_segments,
)


ANNOTATION_SCHEMA_VERSION = "contact_segment_annotation_v1_1"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
UNKNOWN_TOKENS = {"", "unknown", "na", "n/a", "null", "none"}

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
INVENTORY_FIELDS = (
    "episode_id", "length_frames", "subject_id", "session_id", "operator_id",
    "source_object_instance_id", "target_object_instance_id", "liquid_id",
    "collection_batch_id", "metadata_status",
)

ALLOWED_COVERAGE_MODES = {"unknown", "positive_only", "exhaustive"}
ALLOWED_START_BOUNDARIES = {"observed", "left_censored", "unknown"}
ALLOWED_END_BOUNDARIES = {"observed", "right_censored", "unknown"}
ALLOWED_SIDES = {"", "left", "right", "object_object"}
ALLOWED_SITES = {"", "opposition", "primary", "support", "palm_or_jaw", "tool_tip", "other_surface"}
ALLOWED_ROLES = {"", "manipulated", "tool", "receptacle", "fixture"}
ALLOWED_MODES = {"", "stick", "slide", "roll", "impact"}
ALLOWED_EVIDENCE = {"rgb_direct", "rgb_temporal", "rgb_depth_temporal", "sensor_confirmed", "unknown"}


@dataclass(frozen=True)
class InventoryEpisode:
    episode_id: int
    length_frames: int
    metadata_status: str


@dataclass(frozen=True)
class Coverage:
    episode_id: int
    coverage_id: str
    start_frame: int
    end_frame: int
    mode: str


@dataclass(frozen=True)
class ReviewedSegment:
    episode_id: int
    coverage_id: str
    segment_id: str
    onset_event_id: str | None
    release_event_id: str | None
    start_frame: int
    end_frame: int
    start_boundary: str
    end_boundary: str
    side: str | None
    site: str | None
    object_role: str | None
    contact_mode: str | None
    confidence: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_exact(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        expected = list(required_fields)
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: header mismatch\nexpected={expected}\nactual={reader.fieldnames}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer, got {value!r}") from exc
    return result


def _validate_id(value: str, field: str, row_number: int) -> None:
    if not ID_RE.fullmatch(value):
        raise ValueError(f"row {row_number}: invalid {field}={value!r}")


def _unknown_to_none(value: str) -> str | None:
    return None if value.lower() in UNKNOWN_TOKENS else value


def load_inventory(path: Path) -> dict[int, InventoryEpisode]:
    rows = read_csv_exact(path, INVENTORY_FIELDS)
    result: dict[int, InventoryEpisode] = {}
    for row_number, row in enumerate(rows, 2):
        episode = _parse_int(row["episode_id"], "episode_id", row_number)
        length = _parse_int(row["length_frames"], "length_frames", row_number)
        if episode < 0 or length <= 0:
            raise ValueError(f"row {row_number}: episode_id must be nonnegative and length_frames positive")
        if episode in result:
            raise ValueError(f"row {row_number}: duplicate episode_id={episode}")
        result[episode] = InventoryEpisode(episode, length, row["metadata_status"])
    if not result:
        raise ValueError(f"{path}: inventory is empty")
    return result


def load_coverages(path: Path, inventory: dict[int, InventoryEpisode], config: dict) -> list[Coverage]:
    rows = read_csv_exact(path, COVERAGE_FIELDS)
    result: list[Coverage] = []
    seen: set[str] = set()
    accepted_role = config.get("accepted_annotation_role", "adjudicated")
    for row_number, row in enumerate(rows, 2):
        if row["schema_version"] != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"row {row_number}: unsupported schema_version={row['schema_version']!r}")
        if row["dataset_id"] != config["dataset_id"]:
            raise ValueError(f"row {row_number}: dataset_id mismatch")
        if row["annotation_role"] != accepted_role or row["review_status"] != "reviewed":
            raise ValueError(f"row {row_number}: builder accepts reviewed {accepted_role!r} rows only")
        if row["label_origin"] != "human_adjudicated":
            raise ValueError(f"row {row_number}: label_origin must be human_adjudicated")
        if row["frame_clock"] != config["frame_clock"]:
            raise ValueError(f"row {row_number}: frame_clock mismatch")
        if row["coverage_mode"] not in ALLOWED_COVERAGE_MODES:
            raise ValueError(f"row {row_number}: invalid coverage_mode={row['coverage_mode']!r}")
        if row["evidence_scope"] not in ALLOWED_EVIDENCE:
            raise ValueError(f"row {row_number}: invalid evidence_scope")
        episode = _parse_int(row["episode_id"], "episode_id", row_number)
        if episode not in inventory:
            raise ValueError(f"row {row_number}: episode {episode} absent from inventory")
        start = _parse_int(row["start_frame"], "start_frame", row_number)
        end = _parse_int(row["end_frame"], "end_frame", row_number)
        if not 0 <= start < end <= inventory[episode].length_frames:
            raise ValueError(f"row {row_number}: invalid coverage [{start},{end})")
        coverage_id = row["coverage_id"]
        _validate_id(coverage_id, "coverage_id", row_number)
        if coverage_id in seen:
            raise ValueError(f"row {row_number}: duplicate coverage_id={coverage_id}")
        seen.add(coverage_id)
        result.append(Coverage(episode, coverage_id, start, end, row["coverage_mode"]))
    for episode in inventory:
        intervals = sorted((x.start_frame, x.end_frame) for x in result if x.episode_id == episode)
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"episode {episode}: overlapping coverage intervals {previous} and {current}")
    return result


def load_segments(
    path: Path,
    inventory: dict[int, InventoryEpisode],
    coverages: list[Coverage],
    config: dict,
) -> list[ReviewedSegment]:
    rows = read_csv_exact(path, SEGMENT_FIELDS)
    coverage_by_id = {coverage.coverage_id: coverage for coverage in coverages}
    result: list[ReviewedSegment] = []
    segment_ids: set[str] = set()
    event_ids: set[str] = set()
    accepted_role = config.get("accepted_annotation_role", "adjudicated")
    for row_number, row in enumerate(rows, 2):
        if row["schema_version"] != ANNOTATION_SCHEMA_VERSION or row["dataset_id"] != config["dataset_id"]:
            raise ValueError(f"row {row_number}: schema_version or dataset_id mismatch")
        if row["annotation_role"] != accepted_role or row["review_status"] != "reviewed":
            raise ValueError(f"row {row_number}: builder accepts reviewed {accepted_role!r} rows only")
        if row["label_origin"] != "human_adjudicated":
            raise ValueError(f"row {row_number}: prescreen/automatic labels cannot become GT")
        if row["evidence_scope"] not in ALLOWED_EVIDENCE:
            raise ValueError(f"row {row_number}: invalid evidence_scope")
        episode = _parse_int(row["episode_id"], "episode_id", row_number)
        if episode not in inventory:
            raise ValueError(f"row {row_number}: episode {episode} absent from inventory")
        coverage = coverage_by_id.get(row["coverage_id"])
        if coverage is None or coverage.episode_id != episode:
            raise ValueError(f"row {row_number}: coverage_id does not belong to episode")
        if coverage.mode == "unknown":
            raise ValueError(f"row {row_number}: unknown coverage cannot contain GT segments")
        segment_id = row["segment_id"]
        _validate_id(segment_id, "segment_id", row_number)
        if segment_id in segment_ids:
            raise ValueError(f"row {row_number}: duplicate segment_id={segment_id}; windows are not events")
        segment_ids.add(segment_id)
        start = _parse_int(row["start_frame"], "start_frame", row_number)
        end = _parse_int(row["end_frame"], "end_frame", row_number)
        if not coverage.start_frame <= start < end <= coverage.end_frame:
            raise ValueError(f"row {row_number}: segment [{start},{end}) outside coverage")
        start_boundary, end_boundary = row["start_boundary"], row["end_boundary"]
        if start_boundary not in ALLOWED_START_BOUNDARIES or end_boundary not in ALLOWED_END_BOUNDARIES:
            raise ValueError(f"row {row_number}: invalid boundary status")
        if start_boundary == "left_censored" and start != coverage.start_frame:
            raise ValueError(f"row {row_number}: left-censored segment must start at coverage boundary")
        if end_boundary == "right_censored" and end != coverage.end_frame:
            raise ValueError(f"row {row_number}: right-censored segment must end at coverage boundary")
        onset_id = _unknown_to_none(row["onset_event_id"])
        release_id = _unknown_to_none(row["release_event_id"])
        if (start_boundary == "observed") != (onset_id is not None):
            raise ValueError(f"row {row_number}: onset_event_id is required iff start boundary is observed")
        if (end_boundary == "observed") != (release_id is not None):
            raise ValueError(f"row {row_number}: release_event_id is required iff end boundary is observed")
        for field, event_id in (("onset_event_id", onset_id), ("release_event_id", release_id)):
            if event_id is not None:
                _validate_id(event_id, field, row_number)
                if event_id in event_ids:
                    raise ValueError(f"row {row_number}: duplicate event ID {event_id}")
                event_ids.add(event_id)
        values = {
            "side": _unknown_to_none(row["side"]),
            "site": _unknown_to_none(row["site"]),
            "object_role": _unknown_to_none(row["object_role"]),
            "contact_mode": _unknown_to_none(row["contact_mode"]),
        }
        for field, allowed in (("side", ALLOWED_SIDES), ("site", ALLOWED_SITES), ("object_role", ALLOWED_ROLES), ("contact_mode", ALLOWED_MODES)):
            if (values[field] or "") not in allowed:
                raise ValueError(f"row {row_number}: invalid {field}={values[field]!r}")
        try:
            confidence = float(row["confidence"])
        except ValueError as exc:
            raise ValueError(f"row {row_number}: invalid confidence") from exc
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"row {row_number}: confidence must be in (0,1]")
        result.append(ReviewedSegment(
            episode, coverage.coverage_id, segment_id, onset_id, release_id,
            start, end, start_boundary, end_boundary, values["side"], values["site"],
            values["object_role"], values["contact_mode"], confidence,
        ))
    if segment_ids & event_ids:
        raise ValueError(f"segment IDs and event IDs must be disjoint: {sorted(segment_ids & event_ids)}")
    return result


def interval_is_exhaustive(start: int, end: int, coverages: list[Coverage]) -> bool:
    cursor = start
    for coverage in sorted(coverages, key=lambda value: value.start_frame):
        if coverage.mode != "exhaustive" or coverage.end_frame <= cursor:
            continue
        if coverage.start_frame > cursor:
            return False
        cursor = max(cursor, coverage.end_frame)
        if cursor >= end:
            return True
    return cursor >= end


def encode_episode(
    length: int,
    coverages: list[Coverage],
    segments: list[ReviewedSegment],
    *,
    horizon_frames: int,
    num_queries: int,
    negative_confidence: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    keys = ("contact_target", "contact_loss_mask", "contact_confidence", "contact_condition_mask")
    arrays = {key: np.zeros((length, num_queries, SEGMENT_TARGET_DIM), dtype=np.float32) for key in keys}
    provenance: list[dict[str, object]] = []
    for current in range(length):
        real_end = min(length, current + horizon_frames)
        visible = [segment for segment in segments if segment.start_frame < real_end and segment.end_frame > current]
        visible.sort(key=lambda value: (
            value.start_frame,
            value.end_frame,
            value.side or "",
            value.site or "",
            value.segment_id,
        ))
        encoded_segments = []
        boundary_known: list[tuple[bool, bool]] = []
        for segment in visible:
            if segment.start_boundary == "observed":
                start_known = True
                starts_before_window = segment.start_frame < current
            else:
                # A conservative confirmed-positive start only proves that the
                # physical onset is before the window only once the window
                # begins strictly after that confirmed-positive frame.
                start_known = current > segment.start_frame
                starts_before_window = start_known
            if start_known and starts_before_window:
                local_start = 0
            else:
                local_start = max(0, segment.start_frame - current)

            if segment.end_boundary == "observed":
                end_known = True
                ends_after_window = segment.end_frame > real_end
            else:
                # A conservative confirmed-positive end proves continuation
                # beyond the window only while the window ends strictly before it.
                end_known = real_end < segment.end_frame
                ends_after_window = end_known
            if end_known and ends_after_window:
                local_end = horizon_frames
            else:
                local_end = min(horizon_frames, segment.end_frame - current)

            encoded_segments.append(ContactSegment(
                start_frame=local_start,
                end_frame=local_end,
                side=segment.side,
                site=segment.site,
                object_role=segment.object_role,
                contact_mode=segment.contact_mode,
                left_censored=starts_before_window,
                right_censored=ends_after_window,
                confidence=segment.confidence,
            ))
            boundary_known.append((start_known, end_known))
        encoded = encode_ordered_segments(
            encoded_segments,
            num_queries=num_queries,
            horizon_frames=horizon_frames,
            coverage_is_exhaustive=interval_is_exhaustive(current, real_end, coverages),
            negative_confidence=negative_confidence,
        )
        for query_index, segment in enumerate(visible):
            start_known, end_known = boundary_known[query_index]
            for is_known, tau_field, censor_field in (
                (start_known, "start_tau", "left_censored"),
                (end_known, "end_tau", "right_censored"),
            ):
                if not is_known:
                    for key in ("contact_target", "contact_loss_mask", "contact_confidence", "contact_condition_mask"):
                        encoded[key][query_index, SEGMENT_FIELD_SLICES[tau_field]] = 0.0
                        encoded[key][query_index, SEGMENT_FIELD_SLICES[censor_field]] = 0.0
            provenance.append({
                "frame_index": current,
                "query_index": query_index,
                "segment_id": segment.segment_id,
                "onset_event_id": segment.onset_event_id or "",
                "release_event_id": segment.release_event_id or "",
            })
        for key in keys:
            arrays[key][current] = encoded[key]
    return arrays, provenance


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

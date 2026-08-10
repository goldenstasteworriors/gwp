"""Read-only recovery and fail-closed validation for Ego group metadata.

The historical EgoHumanoid capture does not contain the group identifiers
needed for production OOD splits.  This module separates three things that
must not be conflated:

* values directly observed in source/processed artifacts;
* diagnostic temporal continuity blocks inferred from recorded timestamps;
* group identifiers supplied later from an authoritative record or a
  collector attestation.

Temporal gaps, filenames, file ownership, mtimes and visual appearance are
never accepted as production group truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RECOVERY_AUDIT_VERSION = "ego-group-metadata-recovery-audit/v1"
SUPPLEMENT_VERSION = "ego-group-metadata-supplement/v1"
DATASET_ID = "ego_pour_liquid_between_beakers"
REQUIRED_GROUP_FIELDS = (
    "subject_id",
    "session_id",
    "operator_id",
    "source_object_instance_id",
    "target_object_instance_id",
    "liquid_id",
    "collection_batch_id",
)
SUPPLEMENT_TOP_LEVEL_FIELDS = (
    "schema_version",
    "dataset_id",
    "source_evidence_sha256",
    "attestation",
    "episode_rows",
    "provenance_records",
)
ATTESTATION_FIELDS = ("attested_by", "attested_at_utc", "statement")
EPISODE_ROW_FIELDS = ("episode_id",) + REQUIRED_GROUP_FIELDS
PROVENANCE_RECORD_FIELDS = (
    "record_id",
    "provenance_kind",
    "evidence_ref",
    "episode_ids",
    "fields",
)
ALLOWED_PROVENANCE_KINDS = {
    "capture_record",
    "experiment_log",
    "collector_attestation",
}
ATTESTATION_STATEMENT = (
    "I attest that every non-null group value comes from the cited authoritative "
    "record or firsthand collection knowledge and is not inferred from appearance, "
    "file ownership, mtime, episode order, or temporal gaps."
)
UNKNOWN_TOKENS = {
    "",
    "unknown",
    "unk",
    "na",
    "n/a",
    "none",
    "null",
    "same",
    "default",
    "unassigned",
    "missing",
    "tbd",
    "todo",
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,127}$")
UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EgoGroupMetadataError(ValueError):
    """Raised when recovery evidence or a supplement is not admissible."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_evidence_sha256(source_evidence: Mapping[str, Any]) -> str:
    """Hash recovered evidence without binding incidental absolute mounts."""

    portable = {
        key: value for key, value in source_evidence.items() if key != "path_records"
    }
    return canonical_sha256(portable)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise EgoGroupMetadataError(f"non-finite JSON constant {value!r} in {path}")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EgoGroupMetadataError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EgoGroupMetadataError(f"cannot read strict JSON: {path}") from exc


def _require_exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if set(value) != set(expected):
        raise EgoGroupMetadataError(
            f"{label} keys differ: expected={list(expected)!r}, actual={sorted(value)!r}"
        )


def _require_non_placeholder_id(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise EgoGroupMetadataError(f"{label} must be a string")
    normalized = value.strip()
    if normalized.lower() in UNKNOWN_TOKENS or not ID_RE.fullmatch(normalized):
        raise EgoGroupMetadataError(f"{label} is empty, placeholder, or malformed")
    return normalized


def _parse_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not UTC_RE.fullmatch(value):
        raise EgoGroupMetadataError(f"{label} must be an explicit UTC timestamp with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EgoGroupMetadataError(f"{label} is not a valid timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise EgoGroupMetadataError(f"{label} is not UTC")
    return value


def load_recovery_audit(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    audit = _strict_json_load(path)
    if not isinstance(audit, dict):
        raise EgoGroupMetadataError("recovery audit must be a JSON object")
    expected = {
        "audit_version",
        "dataset_id",
        "source_evidence",
        "source_evidence_sha256",
        "provenance_matrix",
        "recovery_conclusion",
    }
    if set(audit) != expected:
        raise EgoGroupMetadataError("recovery audit top-level schema mismatch")
    if audit.get("audit_version") != RECOVERY_AUDIT_VERSION:
        raise EgoGroupMetadataError("recovery audit version mismatch")
    if audit.get("dataset_id") != DATASET_ID:
        raise EgoGroupMetadataError("recovery audit dataset_id mismatch")
    digest = audit.get("source_evidence_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise EgoGroupMetadataError("recovery audit has invalid evidence digest")
    source_evidence = audit.get("source_evidence")
    if not isinstance(source_evidence, dict):
        raise EgoGroupMetadataError("recovery audit source_evidence must be an object")
    if digest != source_evidence_sha256(source_evidence):
        raise EgoGroupMetadataError("recovery audit evidence digest mismatch")
    episodes = source_evidence.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise EgoGroupMetadataError("recovery audit has no episodes")
    seen: set[int] = set()
    for row in episodes:
        episode = row.get("episode_id") if isinstance(row, dict) else None
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise EgoGroupMetadataError("recovery audit has invalid episode_id")
        if episode in seen:
            raise EgoGroupMetadataError("recovery audit has duplicate episode_id")
        seen.add(episode)
    return audit


def make_unfilled_supplement(audit: Mapping[str, Any]) -> dict[str, Any]:
    episodes = audit["source_evidence"]["episodes"]
    return {
        "schema_version": SUPPLEMENT_VERSION,
        "dataset_id": DATASET_ID,
        "source_evidence_sha256": audit["source_evidence_sha256"],
        "attestation": {
            "attested_by": None,
            "attested_at_utc": None,
            "statement": ATTESTATION_STATEMENT,
        },
        "episode_rows": [
            {"episode_id": row["episode_id"], **{field: None for field in REQUIRED_GROUP_FIELDS}}
            for row in episodes
        ],
        "provenance_records": [],
    }


def validate_supplement(
    supplement_path: str | Path,
    audit_path: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate a metadata supplement and return an inventory-ready receipt.

    ``allow_incomplete`` exists only to lint the checked-in null template.  It
    never produces a production-complete receipt.
    """

    supplement_path = Path(supplement_path)
    audit = load_recovery_audit(audit_path)
    supplement = _strict_json_load(supplement_path)
    if not isinstance(supplement, dict):
        raise EgoGroupMetadataError("supplement must be a JSON object")
    _require_exact_keys(supplement, SUPPLEMENT_TOP_LEVEL_FIELDS, "supplement")
    if supplement["schema_version"] != SUPPLEMENT_VERSION:
        raise EgoGroupMetadataError("supplement version mismatch")
    if supplement["dataset_id"] != audit["dataset_id"]:
        raise EgoGroupMetadataError("supplement dataset_id mismatch")
    if supplement["source_evidence_sha256"] != audit["source_evidence_sha256"]:
        raise EgoGroupMetadataError("supplement is not bound to this source evidence")

    attestation = supplement["attestation"]
    if not isinstance(attestation, dict):
        raise EgoGroupMetadataError("attestation must be an object")
    _require_exact_keys(attestation, ATTESTATION_FIELDS, "attestation")
    if attestation["statement"] != ATTESTATION_STATEMENT:
        raise EgoGroupMetadataError("attestation statement was changed")

    expected_episodes = {
        row["episode_id"]: row
        for row in audit["source_evidence"]["episodes"]
    }
    episode_rows = supplement["episode_rows"]
    if not isinstance(episode_rows, list) or len(episode_rows) != len(expected_episodes):
        raise EgoGroupMetadataError("supplement must contain exactly every audited episode")
    parsed_rows: dict[int, dict[str, str | None]] = {}
    missing_cells: list[tuple[int, str]] = []
    for row_number, row in enumerate(episode_rows, 1):
        if not isinstance(row, dict):
            raise EgoGroupMetadataError(f"episode row {row_number} must be an object")
        _require_exact_keys(row, EPISODE_ROW_FIELDS, f"episode row {row_number}")
        episode = row["episode_id"]
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode not in expected_episodes
            or episode in parsed_rows
        ):
            raise EgoGroupMetadataError(f"episode row {row_number} has duplicate/unknown ID")
        parsed: dict[str, str | None] = {"episode_id": episode}  # type: ignore[dict-item]
        for field in REQUIRED_GROUP_FIELDS:
            value = row[field]
            if value is None:
                missing_cells.append((episode, field))
                parsed[field] = None
            else:
                parsed[field] = _require_non_placeholder_id(
                    value, f"episode {episode} {field}"
                )
        parsed_rows[episode] = parsed
    if set(parsed_rows) != set(expected_episodes):
        raise EgoGroupMetadataError("supplement episode set mismatch")

    records = supplement["provenance_records"]
    if not isinstance(records, list):
        raise EgoGroupMetadataError("provenance_records must be a list")
    coverage: set[tuple[int, str]] = set()
    record_ids: set[str] = set()
    for record_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise EgoGroupMetadataError(f"provenance record {record_number} must be an object")
        _require_exact_keys(record, PROVENANCE_RECORD_FIELDS, f"provenance record {record_number}")
        record_id = _require_non_placeholder_id(record["record_id"], "record_id")
        if record_id in record_ids:
            raise EgoGroupMetadataError(f"duplicate provenance record_id={record_id}")
        record_ids.add(record_id)
        if record["provenance_kind"] not in ALLOWED_PROVENANCE_KINDS:
            raise EgoGroupMetadataError(
                "provenance_kind must be capture_record, experiment_log, or collector_attestation"
            )
        evidence_ref = record["evidence_ref"]
        if not isinstance(evidence_ref, str) or len(evidence_ref.strip()) < 8:
            raise EgoGroupMetadataError(
                "provenance evidence_ref must identify an authoritative source"
            )
        episode_ids = record["episode_ids"]
        fields = record["fields"]
        if (
            not isinstance(episode_ids, list)
            or not episode_ids
            or any(
                isinstance(episode, bool)
                or not isinstance(episode, int)
                or episode not in expected_episodes
                for episode in episode_ids
            )
            or len(set(episode_ids)) != len(episode_ids)
        ):
            raise EgoGroupMetadataError("provenance record has invalid episode_ids")
        if (
            not isinstance(fields, list)
            or not fields
            or any(field not in REQUIRED_GROUP_FIELDS for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise EgoGroupMetadataError("provenance record has invalid fields")
        for episode in episode_ids:
            for field in fields:
                if parsed_rows[episode][field] is None:
                    raise EgoGroupMetadataError(
                        f"provenance record {record_id} covers null episode {episode} {field}"
                    )
                coverage.add((episode, field))

    non_null_cells = {
        (episode, field)
        for episode, row in parsed_rows.items()
        for field in REQUIRED_GROUP_FIELDS
        if row[field] is not None
    }
    uncovered = sorted(non_null_cells - coverage)
    if uncovered:
        raise EgoGroupMetadataError(
            f"non-null metadata cells lack authoritative provenance: {uncovered[:8]}"
        )

    incomplete = bool(missing_cells)
    if incomplete and not allow_incomplete:
        raise EgoGroupMetadataError(
            f"production supplement has {len(missing_cells)} missing group cells"
        )
    if not incomplete:
        _require_non_placeholder_id(attestation["attested_by"], "attested_by")
        _parse_utc(attestation["attested_at_utc"], "attested_at_utc")
    elif attestation["attested_by"] is not None or attestation["attested_at_utc"] is not None:
        raise EgoGroupMetadataError("an incomplete template must not contain a partial attestation")

    unique_values = {
        field: len({row[field] for row in parsed_rows.values() if row[field] is not None})
        for field in REQUIRED_GROUP_FIELDS
    }
    inventory_rows = []
    for episode in sorted(parsed_rows):
        evidence_row = expected_episodes[episode]
        inventory_rows.append(
            {
                "episode_id": episode,
                "length_frames": evidence_row["processed_length_frames"],
                **{field: parsed_rows[episode][field] or "" for field in REQUIRED_GROUP_FIELDS},
                "metadata_status": (
                    "complete" if not incomplete else "missing_required_group_metadata"
                ),
            }
        )
    return {
        "schema_version": SUPPLEMENT_VERSION,
        "dataset_id": DATASET_ID,
        "supplement_sha256": file_sha256(supplement_path),
        "source_evidence_sha256": audit["source_evidence_sha256"],
        "production_complete": not incomplete,
        "missing_cell_count": len(missing_cells),
        "missing_cells": [
            {"episode_id": episode, "field": field}
            for episode, field in missing_cells
        ],
        "unique_group_values": unique_values,
        "individual_axes_with_three_groups": sorted(
            field for field, count in unique_values.items() if count >= 3
        ),
        "inventory_rows": inventory_rows,
    }

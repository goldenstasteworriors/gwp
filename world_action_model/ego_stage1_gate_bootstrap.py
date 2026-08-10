"""Stdlib-only bootstrap checks for the production Ego Stage1 config.

This module intentionally does not import the dataset, trainer, NumPy, Torch,
or distributed runtime.  Its enabled path reads only the explicitly selected
training-gate JSON and launcher rank environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re

from world_action_model.verified_io import read_verified_bytes, strict_json_loads


EGO_STAGE1_GATE_VERSION = "ego-discrete-contact-stage1-training-gate/v4"
EGO_STAGE1_REQUIRED_ENV_PATHS = (
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
    "GWP_EGO_TRAINING_RUN_NONCE",
)
EGO_STAGE1_REQUIRED_GATE_FLAGS = (
    "human_adjudicated_sidecar_passed",
    "dual_annotation_agreement_passed",
    "group_metadata_complete",
    "strategy_claim_allowed",
    "ego_action_grip_evidence_passed",
    "matched_control_hash_collector_passed",
    "allow_server_training",
    "allow_5b_training",
)
_NONCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}$")
_UTC_DEADLINE_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EgoStage1BootstrapError(ValueError):
    pass


def _validate_independent_replay_gate_binding(gate: dict) -> None:
    fingerprints = gate.get("artifact_fingerprints")
    reports = gate.get("evidence_reports")
    replay_entry = (
        reports.get("t5_independent_replay")
        if isinstance(reports, dict)
        else None
    )
    replay_sha = (
        fingerprints.get("t5_independent_replay_attestation_sha256")
        if isinstance(fingerprints, dict)
        else None
    )
    if (
        not isinstance(replay_sha, str)
        or not _SHA256_RE.fullmatch(replay_sha)
        or not isinstance(replay_entry, dict)
        or set(replay_entry) != {"path", "sha256"}
        or not isinstance(replay_entry.get("path"), str)
        or not replay_entry["path"]
        or replay_entry.get("sha256") != replay_sha
    ):
        raise EgoStage1BootstrapError(
            "training gate lacks reviewer-bound independent T5 replay evidence"
        )


def _load_gate(path: str) -> dict:
    try:
        payload, _, _ = read_verified_bytes(path)
        payload = strict_json_loads(payload, name="Ego Stage1 training gate")
    except (OSError, UnicodeError, ValueError) as exc:
        raise EgoStage1BootstrapError("invalid Ego Stage1 training-gate JSON") from exc
    if not isinstance(payload, dict):
        raise EgoStage1BootstrapError("Ego Stage1 training gate must be a JSON object")
    return payload


def validate_ego_stage1_gate_freshness_and_training_nonce(
    gate: dict, *, training_run_nonce: str
) -> None:
    """Apply the shared nonce/deadline contract without heavy imports."""

    gate_nonce = gate.get("training_run_nonce")
    if not isinstance(training_run_nonce, str) or not _NONCE_RE.fullmatch(
        training_run_nonce
    ):
        raise EgoStage1BootstrapError(
            "GWP_EGO_TRAINING_RUN_NONCE must be an explicit audit nonce"
        )
    if (
        gate_nonce != training_run_nonce
        or not isinstance(gate_nonce, str)
        or not _NONCE_RE.fullmatch(gate_nonce)
    ):
        raise EgoStage1BootstrapError(
            "runtime training nonce differs from the reviewer-authored gate"
        )
    deadline = gate.get("evidence_valid_until_utc")
    if not isinstance(deadline, str) or not _UTC_DEADLINE_RE.fullmatch(deadline):
        raise EgoStage1BootstrapError("gate evidence expiry must be UTC with Z suffix")
    try:
        parsed_deadline = datetime.fromisoformat(deadline[:-1] + "+00:00")
    except ValueError as exc:
        raise EgoStage1BootstrapError("invalid gate evidence expiry") from exc
    if (
        parsed_deadline.tzinfo is None
        or parsed_deadline.utcoffset() != timezone.utc.utcoffset(None)
    ):
        raise EgoStage1BootstrapError("gate evidence expiry is not UTC")
    if parsed_deadline <= datetime.now(timezone.utc):
        raise EgoStage1BootstrapError("training gate evidence has expired")


def minimal_ego_stage1_admission_precheck_from_environment() -> bool:
    """Reject an inadmissible launch before importing NumPy/Torch/model code.

    This lightweight check can only reject.  The full production preflight
    must still re-open and validate every artifact before a config is built.
    """

    missing = [
        name
        for name in EGO_STAGE1_REQUIRED_ENV_PATHS
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise EgoStage1BootstrapError(
            f"missing Ego Stage1 preflight paths: {missing}"
        )
    gate_path = os.environ["GWP_EGO_STAGE1_TRAINING_GATE"].strip()
    gate = _load_gate(gate_path)
    if gate.get("gate_version") != EGO_STAGE1_GATE_VERSION:
        raise EgoStage1BootstrapError("training gate version differs during admission precheck")
    failed = [
        name for name in EGO_STAGE1_REQUIRED_GATE_FLAGS if gate.get(name) is not True
    ]
    if failed:
        raise EgoStage1BootstrapError(
            f"training gates not explicitly true during admission precheck: {failed}"
        )
    validate_ego_stage1_gate_freshness_and_training_nonce(
        gate,
        training_run_nonce=os.environ["GWP_EGO_TRAINING_RUN_NONCE"].strip(),
    )
    _validate_independent_replay_gate_binding(gate)
    t5_text = os.environ.get("GWP_T5_EMBEDDING_DIR", "").strip()
    if not t5_text:
        t5_text = str(Path(os.environ["GWP_EGO_DATA_PATH"]) / "t5_embedding")
    t5_root = Path(t5_text).expanduser().absolute()
    if t5_root.is_symlink() or not t5_root.is_dir():
        raise EgoStage1BootstrapError(
            f"Stage1 T5 source must be a real directory: {t5_root}"
        )
    return True


def minimal_matched_control_topology_precheck_from_environment() -> bool:
    """Reject a mismatched launcher before full preflight or config import.

    An absent collector output directory is the disabled switch and returns
    without consulting any other environment variable or opening the gate.
    """

    if not os.environ.get("GWP_MATCHED_CONTROL_OUTPUT_DIR", "").strip():
        return False
    gate_path = os.environ.get("GWP_EGO_STAGE1_TRAINING_GATE", "").strip()
    if not gate_path:
        raise EgoStage1BootstrapError(
            "GWP_EGO_STAGE1_TRAINING_GATE is required for collector topology precheck"
        )
    gate = _load_gate(gate_path)
    if gate.get("gate_version") != EGO_STAGE1_GATE_VERSION:
        raise EgoStage1BootstrapError("training gate version differs during topology precheck")
    _validate_independent_replay_gate_binding(gate)
    expectations = gate.get("matched_control_expectations")
    if not isinstance(expectations, dict):
        raise EgoStage1BootstrapError("training gate lacks matched-control expectations")
    expected_world_size = expectations.get("expected_world_size")
    expected_rank_set = expectations.get("expected_rank_set")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size <= 0
        or not isinstance(expected_rank_set, list)
        or any(
            isinstance(expected_rank, bool) or not isinstance(expected_rank, int)
            for expected_rank in expected_rank_set
        )
        or expected_rank_set != list(range(expected_world_size))
    ):
        raise EgoStage1BootstrapError("invalid external matched-control topology")
    try:
        rank_text = os.environ["RANK"].strip()
        world_size_text = os.environ["WORLD_SIZE"].strip()
    except (KeyError, AttributeError) as exc:
        raise EgoStage1BootstrapError(
            "RANK and WORLD_SIZE must be explicit base-10 integers"
        ) from exc
    if not re.fullmatch(r"[0-9]+", rank_text, flags=re.ASCII) or not re.fullmatch(
        r"[0-9]+", world_size_text, flags=re.ASCII
    ):
        raise EgoStage1BootstrapError(
            "RANK and WORLD_SIZE must be explicit base-10 integers"
        )
    rank = int(rank_text)
    world_size = int(world_size_text)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise EgoStage1BootstrapError(
            f"invalid runtime topology rank={rank}, world_size={world_size}"
        )
    if world_size != expected_world_size or rank not in expected_rank_set:
        raise EgoStage1BootstrapError(
            "actual runtime rank/world differs from the external training gate"
        )
    return True

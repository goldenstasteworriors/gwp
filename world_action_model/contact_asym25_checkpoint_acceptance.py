"""Strict checkpoint-bundle and independent acceptance contracts.

The trainer may create a bundle manifest and a training receipt, but it never
creates the reviewer acceptance consumed by held-out evaluation.  This keeps
artifact production and acceptance in separate processes/output locations.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


BUNDLE_MANIFEST_VERSION = "contact-asym25-checkpoint-bundle/v2"
TRAINING_RECEIPT_VERSION = "contact-asym25-checkpoint-training-receipt/v2"
ACCEPTANCE_VERSION = "contact-asym25-checkpoint-independent-acceptance/v1"
PRODUCTION_STEPS = (0, 100, 200)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXCLUDED_EVIDENCE_FILES = frozenset(
    {
        "checkpoint_bundle_manifest.json",
        "checkpoint_training_receipt.json",
        "checkpoint_reviewer_acceptance.json",
        "current_training_data_receipt.json",
    }
)


class CheckpointAcceptanceError(ValueError):
    """Raised when checkpoint evidence is incomplete or mutable."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CheckpointAcceptanceError(f"{name} must be lowercase SHA-256")
    return value


def _require_exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CheckpointAcceptanceError(f"{name} schema mismatch")
    return value


def strict_json_file(path: str | Path, name: str) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().absolute()
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointAcceptanceError(f"{name} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            payload = b""
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                payload += chunk
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise CheckpointAcceptanceError(f"cannot read {name}: {path}") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise CheckpointAcceptanceError(f"{name} changed while being read")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointAcceptanceError(f"invalid strict JSON in {name}") from exc
    if not isinstance(value, dict):
        raise CheckpointAcceptanceError(f"{name} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _hash_regular_file(path: Path) -> tuple[str, int]:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise CheckpointAcceptanceError(f"checkpoint entry is not regular: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise CheckpointAcceptanceError(f"checkpoint entry changed while hashing: {path}")
    return digest.hexdigest(), size


def build_bundle_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir).expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise CheckpointAcceptanceError("checkpoint path must be a real directory")
    rows: list[dict[str, Any]] = []
    for current_root, directories, files in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directories:
            candidate = current / name
            if candidate.is_symlink():
                raise CheckpointAcceptanceError(
                    f"checkpoint bundle contains symlink directory: {candidate}"
                )
        for name in files:
            relative = (current / name).relative_to(root).as_posix()
            if relative in _EXCLUDED_EVIDENCE_FILES:
                continue
            digest, size = _hash_regular_file(current / name)
            rows.append(
                {"logical_path": relative, "sha256": digest, "size_bytes": size}
            )
    rows.sort(key=lambda row: row["logical_path"])
    if not rows or not any(row["logical_path"] == "config.json" for row in rows):
        raise CheckpointAcceptanceError("checkpoint bundle must contain config.json")
    core = {
        "files": rows,
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
    }
    return {
        "manifest_version": BUNDLE_MANIFEST_VERSION,
        **core,
        "bundle_sha256": sha256_json(core),
    }


def validate_bundle_manifest(
    value: object, checkpoint_dir: str | Path
) -> dict[str, Any]:
    manifest = _require_exact(
        value,
        {
            "manifest_version",
            "files",
            "file_count",
            "total_size_bytes",
            "bundle_sha256",
        },
        "checkpoint bundle manifest",
    )
    if manifest["manifest_version"] != BUNDLE_MANIFEST_VERSION:
        raise CheckpointAcceptanceError("checkpoint bundle manifest version mismatch")
    actual = build_bundle_manifest(checkpoint_dir)
    if manifest != actual:
        raise CheckpointAcceptanceError("checkpoint bundle bytes differ from manifest")
    return manifest


def validate_training_receipt(value: object) -> dict[str, Any]:
    receipt = _require_exact(
        value,
        {
            "receipt_version",
            "checkpoint_step",
            "checkpoint_bundle_manifest_sha256",
            "checkpoint_bundle_sha256",
            "parent_checkpoint_bundle_sha256",
            "initial_weights_bundle_sha256",
            "training_run_nonce",
            "experiment_family_id",
            "variant_id",
            "split_seed",
            "training_precision",
            "contact_query_count",
            "contact_target_dim",
            "contact_time_bins",
            "action_schema_contract",
            "production_preflight_receipt_sha256",
            "heldout_window_manifest_sha256",
            "training_gate_sha256",
            "current_training_data_receipt",
            "current_training_data_receipt_sha256",
            "training_code_receipt_sha256",
            "dataset_source_tree_sha256",
            "initial_weights_tree_receipt",
        },
        "checkpoint training receipt",
    )
    if receipt["receipt_version"] != TRAINING_RECEIPT_VERSION:
        raise CheckpointAcceptanceError("checkpoint training receipt version mismatch")
    for field in (
        "checkpoint_bundle_manifest_sha256",
        "checkpoint_bundle_sha256",
        "parent_checkpoint_bundle_sha256",
        "initial_weights_bundle_sha256",
        "production_preflight_receipt_sha256",
        "heldout_window_manifest_sha256",
        "training_gate_sha256",
        "current_training_data_receipt_sha256",
        "training_code_receipt_sha256",
        "dataset_source_tree_sha256",
    ):
        _require_sha256(receipt[field], field)
    if (
        receipt["checkpoint_step"] not in PRODUCTION_STEPS
        or receipt["split_seed"] not in {6666, 7777, 8888}
        or receipt["training_precision"] != "bfloat16"
        or receipt["contact_query_count"] != 12
        or receipt["contact_target_dim"] != 22
        or receipt["contact_time_bins"] != 16
    ):
        raise CheckpointAcceptanceError("checkpoint production shape/precision differs")
    for field in ("training_run_nonce", "experiment_family_id", "variant_id"):
        if not isinstance(receipt[field], str) or len(receipt[field]) < 16:
            raise CheckpointAcceptanceError(f"invalid checkpoint {field}")
    schema = _require_exact(
        receipt["action_schema_contract"],
        {
            "schema_id",
            "model_action_dim",
            "physical_action_dim",
            "groups",
            "grip_active_mask_source",
        },
        "checkpoint action schema contract",
    )
    if not isinstance(schema["groups"], dict) or not schema["groups"]:
        raise CheckpointAcceptanceError("checkpoint action groups are empty")
    if sha256_json(receipt["current_training_data_receipt"]) != receipt[
        "current_training_data_receipt_sha256"
    ]:
        raise CheckpointAcceptanceError("embedded current training receipt differs")
    initial = _require_exact(
        receipt["initial_weights_tree_receipt"],
        {"path_record_only", "tree_sha256", "file_count", "total_size_bytes"},
        "initial weights tree receipt",
    )
    if initial["tree_sha256"] != receipt["initial_weights_bundle_sha256"]:
        raise CheckpointAcceptanceError("initial weights tree binding differs")
    return receipt


def validate_acceptance(
    value: object,
    *,
    checkpoint_dir: str | Path,
    bundle_manifest: Mapping[str, Any],
    bundle_manifest_sha256: str,
    training_receipt: Mapping[str, Any],
    training_receipt_sha256: str,
) -> dict[str, Any]:
    acceptance = _require_exact(
        value,
        {
            "acceptance_version",
            "status",
            "reviewer_identity",
            "reviewer_attestation_nonce",
            "checkpoint_path_record_only",
            "checkpoint_step",
            "split_seed",
            "training_run_nonce",
            "checkpoint_bundle_manifest_sha256",
            "checkpoint_bundle_sha256",
            "checkpoint_training_receipt_sha256",
        },
        "independent checkpoint acceptance",
    )
    if (
        acceptance["acceptance_version"] != ACCEPTANCE_VERSION
        or acceptance["status"] != "accepted"
        or acceptance["checkpoint_path_record_only"]
        != str(Path(checkpoint_dir).expanduser().absolute())
        or acceptance["checkpoint_step"] != training_receipt["checkpoint_step"]
        or acceptance["split_seed"] != training_receipt["split_seed"]
        or acceptance["training_run_nonce"] != training_receipt["training_run_nonce"]
        or acceptance["checkpoint_bundle_manifest_sha256"]
        != bundle_manifest_sha256
        or acceptance["checkpoint_bundle_sha256"]
        != bundle_manifest["bundle_sha256"]
        or acceptance["checkpoint_training_receipt_sha256"]
        != training_receipt_sha256
    ):
        raise CheckpointAcceptanceError("independent checkpoint acceptance binding differs")
    for field in ("reviewer_identity", "reviewer_attestation_nonce"):
        if not isinstance(acceptance[field], str) or len(acceptance[field]) < 16:
            raise CheckpointAcceptanceError(f"invalid independent {field}")
    return acceptance


def write_json_exclusive(path: str | Path, value: object) -> str:
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

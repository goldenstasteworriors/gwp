"""Dependency-free receipts for production asymmetric25 train-only norm stats.

The legacy norm-stat generator intentionally describes a whole dataset.  This
module defines the stricter production contract: one norm artifact belongs to
one completed group split seed and may summarize only that seed's train
episodes.  The receipt covers the exact split identity and every metadata/data
parquet byte that can contribute to the statistics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


PRODUCTION_TRAIN_NORM_RECEIPT_VERSION = (
    "contact-asym25-train-only-norm-source-receipt/v1"
)
PRODUCTION_SPLIT_RECEIPT_VERSION = "contact-asym25-production-split-receipt/v1"
NORM_SOURCE_TREE_RECEIPT_VERSION = "contact-asym25-norm-source-tree/v1"
NORM_SOURCE_TREE_SCOPE = "meta-info+episode-metadata+data-parquet/v1"
PRODUCTION_SPLIT_SEEDS = frozenset({6666, 7777, 8888})
PRODUCTION_ACTION_SCHEMA_ID = "sonic_g1_rh56dftp6_omnipicker1_v1"
PRODUCTION_MODEL_DIM = 32
PRODUCTION_ACTION_HORIZON = 48
PRODUCTION_BOUNDARY = (
    "clamp_within_episode_repeat_last_frame (matches FastLeRobotDataset)"
)
PRODUCTION_ACTION_REPRESENTATION = (
    "delta_for_true_delta_mask_dims_else_absolute_action"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProductionTrainNormError(ValueError):
    """Raised when a norm artifact is not provably train-only."""


def canonical_json_sha256(value: object) -> str:
    """Hash strict canonical JSON without accepting NaN or infinity."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionTrainNormError("receipt is not strict canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProductionTrainNormError(f"{name} must be a lowercase SHA-256")
    return value


def _episode_ids(value: object, name: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ProductionTrainNormError(f"{name} must be a non-empty episode list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ProductionTrainNormError(f"{name} contains an invalid episode ID")
        result.append(item)
    if result != sorted(set(result)):
        raise ProductionTrainNormError(f"{name} must be sorted and unique")
    return result


def build_production_split_receipt(
    *,
    dataset_id: str,
    action_schema_id: str,
    split_seed: int,
    train_episode_ids: Sequence[int],
    validation_episode_ids: Sequence[int],
    heldout_episode_ids: Sequence[int],
    dataset_info_sha256: str,
    episode_metadata_sha256: str,
    split_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Build the norm-independent canonical identity of one production split."""

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ProductionTrainNormError("production split dataset_id is empty")
    if action_schema_id != PRODUCTION_ACTION_SCHEMA_ID:
        raise ProductionTrainNormError("production split action schema differs")
    if isinstance(split_seed, bool) or split_seed not in PRODUCTION_SPLIT_SEEDS:
        raise ProductionTrainNormError("production split seed differs")
    train = _episode_ids(train_episode_ids, "train_episode_ids")
    validation = _episode_ids(validation_episode_ids, "validation_episode_ids")
    heldout = _episode_ids(heldout_episode_ids, "heldout_episode_ids")
    if set(train) & set(validation) or set(train) & set(heldout) or set(validation) & set(heldout):
        raise ProductionTrainNormError("production split episode sets overlap")
    receipt = {
        "receipt_version": PRODUCTION_SPLIT_RECEIPT_VERSION,
        "dataset_id": dataset_id,
        "action_schema_id": action_schema_id,
        "split_seed": split_seed,
        "train_episode_ids": train,
        "validation_episode_ids": validation,
        "heldout_episode_ids": heldout,
        "dataset_info_sha256": _require_sha256(
            dataset_info_sha256, "dataset info digest"
        ),
        "episode_metadata_sha256": _require_sha256(
            episode_metadata_sha256, "episode metadata digest"
        ),
        "split_manifest_sha256": _require_sha256(
            split_manifest_sha256, "split manifest digest"
        ),
    }
    return receipt, canonical_json_sha256(receipt)


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ProductionTrainNormError(f"cannot stat norm source: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProductionTrainNormError(f"norm source must be a real file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = os.lstat(path)
    except OSError as exc:
        raise ProductionTrainNormError(f"cannot hash norm source: {path}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ProductionTrainNormError(f"norm source changed while hashing: {path}")
    return digest.hexdigest(), int(before.st_size)


def build_norm_source_tree_receipt(data_root: str | Path) -> dict[str, Any]:
    """Hash exactly the metadata and parquet bytes consumed by norm generation."""

    root = Path(data_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ProductionTrainNormError(
            f"norm source dataset root must be a real directory: {root}"
        )
    info_path = root / "meta" / "info.json"
    episode_jsonl = root / "meta" / "episodes.jsonl"
    if episode_jsonl.is_file():
        episode_paths = [episode_jsonl]
    else:
        episode_paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
        episode_paths += sorted((root / "meta" / "episodes").glob("*.parquet"))
    data_paths = sorted((root / "data").glob("*/*.parquet"))
    data_paths += sorted((root / "data").glob("*.parquet"))
    paths = [info_path, *episode_paths, *data_paths]
    if not info_path.is_file() or not episode_paths or not data_paths:
        raise ProductionTrainNormError(
            "norm source tree requires meta/info.json, episode metadata, and data parquet"
        )
    logical_paths = [path.relative_to(root).as_posix() for path in paths]
    if len(logical_paths) != len(set(logical_paths)):
        raise ProductionTrainNormError("norm source tree contains duplicate files")
    rows = []
    for logical_path, path in sorted(zip(logical_paths, paths, strict=True)):
        digest, size_bytes = _hash_regular_file(path)
        rows.append(
            {
                "logical_path": logical_path,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
    return {
        "receipt_version": NORM_SOURCE_TREE_RECEIPT_VERSION,
        "scope": NORM_SOURCE_TREE_SCOPE,
        "tree_sha256": canonical_json_sha256(rows),
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
    }


def build_expected_train_norm_source_receipt(
    *,
    production_split_receipt: Mapping[str, Any],
    production_split_receipt_sha256: str,
    episode_lengths: Mapping[int, int],
    source_dataset_tree_receipt: Mapping[str, Any],
    model_dim: int = PRODUCTION_MODEL_DIM,
    action_horizon: int = PRODUCTION_ACTION_HORIZON,
) -> tuple[dict[str, Any], str]:
    """Build the exact source/count receipt required inside norm metadata."""

    split_receipt = dict(production_split_receipt)
    if canonical_json_sha256(split_receipt) != _require_sha256(
        production_split_receipt_sha256, "production split receipt digest"
    ):
        raise ProductionTrainNormError("production split receipt fingerprint differs")
    train = _episode_ids(split_receipt.get("train_episode_ids"), "train_episode_ids")
    lengths: dict[int, int] = {}
    for episode, length in episode_lengths.items():
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ProductionTrainNormError("runtime episode lengths are invalid")
        lengths[episode] = length
    missing = sorted(set(train) - set(lengths))
    if missing:
        raise ProductionTrainNormError(
            f"train norm split references missing runtime episodes: {missing}"
        )
    if model_dim != PRODUCTION_MODEL_DIM or action_horizon != PRODUCTION_ACTION_HORIZON:
        raise ProductionTrainNormError("production norm width/horizon differs")
    source_tree = dict(source_dataset_tree_receipt)
    expected_tree_keys = {
        "receipt_version",
        "scope",
        "tree_sha256",
        "file_count",
        "total_size_bytes",
    }
    if set(source_tree) != expected_tree_keys:
        raise ProductionTrainNormError("norm source-tree receipt schema differs")
    if (
        source_tree["receipt_version"] != NORM_SOURCE_TREE_RECEIPT_VERSION
        or source_tree["scope"] != NORM_SOURCE_TREE_SCOPE
    ):
        raise ProductionTrainNormError("norm source-tree receipt identity differs")
    _require_sha256(source_tree["tree_sha256"], "norm source-tree digest")
    if (
        isinstance(source_tree["file_count"], bool)
        or not isinstance(source_tree["file_count"], int)
        or source_tree["file_count"] <= 0
        or isinstance(source_tree["total_size_bytes"], bool)
        or not isinstance(source_tree["total_size_bytes"], int)
        or source_tree["total_size_bytes"] <= 0
    ):
        raise ProductionTrainNormError("norm source-tree counts are invalid")
    state_count = sum(lengths[episode] for episode in train)
    receipt = {
        "receipt_version": PRODUCTION_TRAIN_NORM_RECEIPT_VERSION,
        "dataset_id": split_receipt["dataset_id"],
        "action_schema_id": split_receipt["action_schema_id"],
        "split_seed": split_receipt["split_seed"],
        "train_episode_ids": train,
        "train_episode_ids_sha256": canonical_json_sha256(train),
        "production_split_receipt_sha256": production_split_receipt_sha256,
        "split_manifest_sha256": split_receipt["split_manifest_sha256"],
        "dataset_info_sha256": split_receipt["dataset_info_sha256"],
        "episode_metadata_sha256": split_receipt["episode_metadata_sha256"],
        "source_dataset_tree_receipt": source_tree,
        "model_dim": model_dim,
        "action_horizon": action_horizon,
        "delta_mask": [False] * model_dim,
        "action_representation": PRODUCTION_ACTION_REPRESENTATION,
        "boundary": PRODUCTION_BOUNDARY,
        "num_episodes": len(train),
        "num_state_samples": state_count,
        "num_action_samples": state_count * action_horizon,
    }
    return receipt, canonical_json_sha256(receipt)


def validate_production_train_norm_metadata(
    metadata: object,
    *,
    expected_source_receipt: Mapping[str, Any],
    expected_source_receipt_sha256: str,
    data_root: str | Path,
) -> dict[str, Any]:
    """Reject whole-dataset/another-seed norm metadata on production paths."""

    if not isinstance(metadata, Mapping):
        raise ProductionTrainNormError("production norm metadata must be an object")
    expected = dict(expected_source_receipt)
    digest = canonical_json_sha256(expected)
    if digest != _require_sha256(
        expected_source_receipt_sha256, "expected train norm receipt digest"
    ):
        raise ProductionTrainNormError("expected train norm receipt digest differs")
    actual = metadata.get("production_train_norm_source_receipt")
    actual_digest = metadata.get("production_train_norm_source_receipt_sha256")
    if actual != expected or actual_digest != digest:
        raise ProductionTrainNormError(
            "normalization stats are not the exact receipt-seed train-only artifact"
        )
    roots = metadata.get("data_roots")
    if (
        not isinstance(roots, list)
        or len(roots) != 1
        or Path(str(roots[0])).expanduser().absolute()
        != Path(data_root).expanduser().absolute()
    ):
        raise ProductionTrainNormError("production norm data root differs")
    mirrored = {
        "split_seed": expected["split_seed"],
        "train_episode_ids": expected["train_episode_ids"],
        "train_episode_ids_sha256": expected["train_episode_ids_sha256"],
        "production_split_receipt_sha256": expected[
            "production_split_receipt_sha256"
        ],
        "split_manifest_sha256": expected["split_manifest_sha256"],
        "source_dataset_tree_receipt": expected["source_dataset_tree_receipt"],
        "action_schema_id": expected["action_schema_id"],
        "model_dim": expected["model_dim"],
        "action_horizon": expected["action_horizon"],
        "delta_mask": expected["delta_mask"],
        "action_representation": expected["action_representation"],
        "boundary": expected["boundary"],
        "num_episodes": expected["num_episodes"],
        "num_state_samples": expected["num_state_samples"],
        "num_action_samples": expected["num_action_samples"],
        "dataset_info_sha256": expected["dataset_info_sha256"],
        "episode_metadata_sha256": expected["episode_metadata_sha256"],
    }
    for name, expected_value in mirrored.items():
        if metadata.get(name) != expected_value:
            raise ProductionTrainNormError(
                f"production norm metadata {name} is not train-only/bound"
            )
    return expected

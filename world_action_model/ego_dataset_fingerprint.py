"""Portable identity for processed EgoHumanoid LeRobot datasets.

The source dataset predates an explicit ``dataset_id``.  This module derives
one from stable semantic metadata without modifying the dataset or depending
on its absolute mount path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EGO_STAGE1_FPS = 20.0
EGO_STAGE1_FRAME_CLOCK = "processed_action_20hz"
EGO_DATASET_FINGERPRINT_VERSION = "ego-processed-dataset-fingerprint/v1"
EGO_INFO_IDENTITY_FIELDS = (
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "fps", "splits", "data_path", "features",
)
EGO_EPISODE_IDENTITY_FIELDS = ("episode_index", "length", "tasks")


class EgoDatasetFingerprintError(ValueError):
    pass


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise EgoDatasetFingerprintError(f"missing dataset metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EgoDatasetFingerprintError(f"metadata must be a JSON object: {path}")
    return value


def _episode_rows(meta_root: Path) -> list[dict]:
    path = meta_root / "episodes.jsonl"
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        parquet_paths = sorted((meta_root / "episodes").glob("*/*.parquet")) + sorted(
            (meta_root / "episodes").glob("*.parquet")
        )
        if not parquet_paths:
            raise EgoDatasetFingerprintError(
                f"missing authoritative JSONL/parquet episode metadata under {meta_root}"
            )
        try:
            import pandas as pd
        except ImportError as exc:
            raise EgoDatasetFingerprintError(
                "parquet episode metadata requires the project's pandas dependency"
            ) from exc
        rows = []
        for parquet_path in parquet_paths:
            rows.extend(pd.read_parquet(parquet_path).to_dict(orient="records"))
    canonical = []
    seen = set()
    for row_number, row in enumerate(rows, 1):
        try:
            episode = int(row["episode_index"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EgoDatasetFingerprintError(
                f"invalid episode metadata row {row_number}"
            ) from exc
        if episode < 0 or length <= 0 or episode in seen:
            raise EgoDatasetFingerprintError(
                f"duplicate/invalid episode metadata row {row_number}"
            )
        seen.add(episode)
        tasks = row.get("tasks", [])
        if hasattr(tasks, "tolist"):
            tasks = tasks.tolist()
        if isinstance(tasks, tuple):
            tasks = list(tasks)
        if not isinstance(tasks, list) or any(not isinstance(task, str) for task in tasks):
            raise EgoDatasetFingerprintError(f"invalid tasks in episode row {row_number}")
        canonical.append(
            {"episode_index": episode, "length": length, "tasks": sorted(tasks)}
        )
    canonical.sort(key=lambda row: row["episode_index"])
    if not canonical:
        raise EgoDatasetFingerprintError("episode metadata is empty")
    return canonical


def derive_ego_dataset_identity(data_root: str | Path) -> dict:
    """Validate Stage1 metadata and derive a path-independent dataset ID."""

    root = Path(data_root).expanduser().resolve()
    info = _load_json(root / "meta" / "info.json")
    rows = _episode_rows(root / "meta")
    required_features = {
        "observation.images.left": ("image", [480, 640, 3]),
        "action_eef": ("float32", [14]),
        "action_delta_eef": ("float32", [12]),
        "hand_status": ("float32", [2]),
        "timestamp": ("float32", [1]),
        "frame_index": ("int64", [1]),
        "episode_index": ("int64", [1]),
    }
    features = info.get("features")
    if not isinstance(features, dict):
        raise EgoDatasetFingerprintError("dataset info lacks feature metadata")
    for name, (dtype, shape) in required_features.items():
        feature = features.get(name)
        if not isinstance(feature, dict) or feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise EgoDatasetFingerprintError(
                f"processed Ego feature contract differs for {name}"
            )
    if float(info.get("fps", -1)) != EGO_STAGE1_FPS:
        raise EgoDatasetFingerprintError("Ego Stage1 dataset must be exactly 20 Hz")
    episode_lengths = {int(row["episode_index"]): int(row["length"]) for row in rows}
    if (
        int(info.get("total_episodes", -1)) != len(rows)
        or int(info.get("total_frames", -1)) != sum(episode_lengths.values())
    ):
        raise EgoDatasetFingerprintError("dataset totals differ from episode metadata")

    # Only stable, portable fields participate. Absolute paths, mtimes and
    # incidental JSON formatting are deliberately excluded.
    info_identity = {key: info.get(key) for key in EGO_INFO_IDENTITY_FIELDS}
    info_digest = _canonical_sha256(info_identity)
    episode_digest = _canonical_sha256(rows)
    payload = {
        "fingerprint_version": EGO_DATASET_FINGERPRINT_VERSION,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "fps": EGO_STAGE1_FPS,
        "info_identity_sha256": info_digest,
        "episode_metadata_sha256": episode_digest,
    }
    fingerprint = _canonical_sha256(payload)
    return {
        "dataset_id": f"ego-processed-sha256:{fingerprint}",
        "fingerprint_version": EGO_DATASET_FINGERPRINT_VERSION,
        "info_identity_fields": list(EGO_INFO_IDENTITY_FIELDS),
        "episode_identity_fields": list(EGO_EPISODE_IDENTITY_FIELDS),
        "episode_sort_key": ["episode_index"],
        "task_sort_rule": "unicode_codepoint_ascending",
        "canonical_json": "utf8_sort_keys_compact_separators_ensure_ascii_false",
        "dataset_fingerprint_sha256": fingerprint,
        "info_identity_sha256": info_digest,
        "episode_metadata_sha256": episode_digest,
        "episode_lengths": episode_lengths,
        "episode_count": len(rows),
        "total_frames": sum(episode_lengths.values()),
        "fps": EGO_STAGE1_FPS,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "data_root": str(root),
    }

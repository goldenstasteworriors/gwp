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
EGO_DATASET_FINGERPRINT_VERSION = "ego-processed-dataset-fingerprint/v2"
EGO_SOURCE_RECEIPT_VERSION = "ego-lerobot-training-source-receipt/v1"
EGO_T5_SOURCE_RECEIPT_VERSION = "ego-t5-training-source-receipt/v1"
EGO_INFO_IDENTITY_FIELDS = (
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "fps", "splits", "features",
)
EGO_EPISODE_IDENTITY_FIELDS = ("episode_index", "length", "tasks")
EGO_PORTABLE_INFO_IGNORED_FIELDS = ("data_path", "video_path")
EGO_TRAINING_SOURCE_ROOTS = ("meta", "data", "videos")


class EgoDatasetFingerprintError(ValueError):
    pass


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _training_source_manifest(
    root: Path, info: dict, episode_rows: list[dict]
) -> tuple[list[dict], dict, dict]:
    """Hash every file the Stage1 LeRobot loader can consume.

    Logical paths are independent of the mount.  ``meta/info.json`` uses
    canonical JSON with mount/path templates removed.  The v2.1 metadata path
    hashes exactly the files read by ``_LeRobotDatasetMetadata``; parquet paths
    follow ``FastLeRobotDataset.load_hf_dataset`` (which hard-codes
    ``root/data/*/*.parquet``). Image features are embedded in those parquet
    files. Video features, if declared, are resolved through ``video_path`` but
    recorded under stable per-key/per-episode logical roles. T5 conditioning is
    config-selected rather than LeRobot metadata-selected and therefore has a
    separate exact per-episode receipt below.
    """

    metadata_paths = [root / "meta" / "info.json"]
    if str(info.get("codebase_version", "")).lower() == "v2.1":
        metadata_paths.extend(
            [root / "meta" / "tasks.jsonl", root / "meta" / "episodes.jsonl"]
        )
        stats_path = root / "meta" / "stats.json"
        if stats_path.is_file():
            metadata_paths.append(stats_path)
    else:
        metadata_paths.extend(
            path for path in (root / "meta").rglob("*") if path.is_file()
        )
    data_root = root / "data"
    if data_root.is_symlink() or not data_root.is_dir():
        raise EgoDatasetFingerprintError(
            f"FastLeRobotDataset requires a real data directory: {data_root}"
        )
    data_paths = sorted(data_root.glob("*/*.parquet"))
    if not data_paths:
        raise EgoDatasetFingerprintError("processed Ego training source has no parquet data")
    features = info.get("features", {})
    embedded_image_keys = sorted(
        key for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "image"
    )
    video_keys = sorted(
        key for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    )
    video_sources = []
    if video_keys:
        template = info.get("video_path")
        if not isinstance(template, str) or not template:
            raise EgoDatasetFingerprintError("video features require info.video_path")
        chunk_size = int(info.get("chunks_size", 1000))
        if chunk_size <= 0:
            raise EgoDatasetFingerprintError("info.chunks_size must be positive")
        for row in episode_rows:
            episode = int(row["episode_index"])
            for video_key in video_keys:
                try:
                    declared = template.format(
                        episode_chunk=episode // chunk_size,
                        video_key=video_key,
                        episode_index=episode,
                    )
                except (KeyError, IndexError, ValueError) as exc:
                    raise EgoDatasetFingerprintError(
                        "cannot resolve info.video_path for loader receipt"
                    ) from exc
                actual_path = Path(declared).expanduser()
                if not actual_path.is_absolute():
                    actual_path = root / actual_path
                actual_path = actual_path.resolve()
                video_sources.append(
                    (
                        f"videos/{video_key}/episode_{episode:06d}.mp4",
                        actual_path,
                    )
                )
    paths_with_roles = []
    paths_with_roles.extend(
        (path.relative_to(root).as_posix(), "metadata", path)
        for path in sorted(set(metadata_paths))
    )
    paths_with_roles.extend(
        (path.relative_to(root).as_posix(), "parquet", path) for path in data_paths
    )
    paths_with_roles.extend((logical, "video", path) for logical, path in video_sources)
    paths_with_roles.sort(key=lambda item: item[0])
    manifest = []
    audit_counts = {role: {"files": 0, "bytes": 0} for role in (
        "metadata", "parquet", "video"
    )}
    seen_logical_paths = set()
    for relative, role, path in paths_with_roles:
        if relative in seen_logical_paths:
            raise EgoDatasetFingerprintError(
                f"duplicate logical training source path: {relative}"
            )
        seen_logical_paths.add(relative)
        if not path.is_file() or path.is_symlink():
            raise EgoDatasetFingerprintError(
                f"training source is missing or symlinked: {path}"
            )
        audit_counts[role]["files"] += 1
        audit_counts[role]["bytes"] += path.stat().st_size
        if relative == "meta/info.json":
            portable_info = {
                key: value
                for key, value in info.items()
                if key not in EGO_PORTABLE_INFO_IGNORED_FIELDS
            }
            entry = {
                "path": relative,
                "logical_role": role,
                "digest_mode": "canonical_json_without_data_or_video_path",
                "sha256": _canonical_sha256(portable_info),
            }
        else:
            entry = {
                "path": relative,
                "logical_role": role,
                "digest_mode": "raw_file_bytes",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        manifest.append(entry)
    relative_paths = {entry["path"] for entry in manifest}
    required = {"meta/info.json"}
    if (root / "meta" / "episodes.jsonl").is_file():
        required.add("meta/episodes.jsonl")
    if not required.issubset(relative_paths):
        raise EgoDatasetFingerprintError(
            f"training source manifest lacks required files: {sorted(required - relative_paths)}"
        )
    manifest_sha256 = _canonical_sha256(manifest)
    receipt = {
        "receipt_version": EGO_SOURCE_RECEIPT_VERSION,
        "manifest_sha256": manifest_sha256,
        "file_count": len(manifest),
        "files_by_logical_role": {
            role: audit_counts[role]["files"] for role in sorted(audit_counts)
        },
        "embedded_image_feature_keys": embedded_image_keys,
        "video_feature_keys": video_keys,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return manifest, receipt, audit_counts


def derive_ego_t5_source_receipt(
    t5_embedding_dir: str | Path,
    *,
    episode_indices: list[int] | tuple[int, ...],
) -> dict:
    """Hash exactly the config-selected T5 files the Stage1 wrapper reads.

    The external directory is deliberately absent from the manifest. Each
    digest is instead bound to a stable episode/logical-role key, so an
    identical copied source has the same receipt while any referenced-byte
    replacement changes it.
    """

    root = Path(t5_embedding_dir).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise EgoDatasetFingerprintError(
            f"Stage1 T5 source must be a real directory: {root}"
        )
    indices = sorted(episode_indices)
    if (
        not indices
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise EgoDatasetFingerprintError("T5 receipt requires unique non-negative episodes")
    manifest = []
    total_bytes = 0
    for episode_index in indices:
        filename = f"episode_{episode_index:06d}.pt"
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise EgoDatasetFingerprintError(
                f"missing or symlinked Stage1 T5 episode source: {path}"
            )
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        manifest.append(
            {
                "path": f"t5_embedding/episode_{episode_index:06d}.pt",
                "logical_role": "t5_episode_condition",
                "episode_index": episode_index,
                "digest_mode": "raw_file_bytes",
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
            }
        )
    manifest_sha256 = _canonical_sha256(manifest)
    receipt = {
        "receipt_version": EGO_T5_SOURCE_RECEIPT_VERSION,
        "manifest_sha256": manifest_sha256,
        "file_count": len(manifest),
        "total_bytes": total_bytes,
        "logical_role": "t5_episode_condition",
        "pattern": "episode_{episode_index:06d}.pt",
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return {
        "source_dir_record_only": str(root),
        "source_manifest": manifest,
        "source_manifest_sha256": manifest_sha256,
        "source_receipt": receipt,
        "source_receipt_sha256": receipt["receipt_sha256"],
    }


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
    source_manifest, source_receipt, source_audit_summary = _training_source_manifest(
        root, info, rows
    )

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
        "source_receipt_sha256": source_receipt["receipt_sha256"],
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
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_receipt["manifest_sha256"],
        "source_receipt": source_receipt,
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_audit_summary_record_only": source_audit_summary,
        "portable_info_ignored_fields": list(EGO_PORTABLE_INFO_IGNORED_FIELDS),
        "training_source_roots": list(EGO_TRAINING_SOURCE_ROOTS),
        "episode_lengths": episode_lengths,
        "episode_count": len(rows),
        "total_frames": sum(episode_lengths.values()),
        "fps": EGO_STAGE1_FPS,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "data_root": str(root),
    }

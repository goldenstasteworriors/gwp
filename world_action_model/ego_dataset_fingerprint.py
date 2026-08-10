"""Portable identity for processed EgoHumanoid LeRobot datasets.

The source dataset predates an explicit ``dataset_id``.  This module derives
one from stable semantic metadata without modifying the dataset or depending
on its absolute mount path.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from world_action_model.verified_io import (
    hash_verified_regular_file,
    read_verified_bytes,
    strict_json_loads,
)


EGO_STAGE1_FPS = 20.0
EGO_STAGE1_FRAME_CLOCK = "processed_action_20hz"
EGO_DATASET_FINGERPRINT_VERSION = "ego-processed-dataset-fingerprint/v2"
EGO_SOURCE_RECEIPT_VERSION = "ego-lerobot-training-source-receipt/v1"
EGO_T5_SOURCE_RECEIPT_VERSION = "ego-t5-training-source-receipt/v2"
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
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_json(path: Path) -> tuple[dict, dict[Path, tuple[str, dict[str, int]]]]:
    try:
        payload, digest, runtime_stat = read_verified_bytes(path)
        value = strict_json_loads(payload, name=str(path))
    except (OSError, ValueError) as exc:
        raise EgoDatasetFingerprintError(f"invalid dataset metadata: {path}") from exc
    if not isinstance(value, dict):
        raise EgoDatasetFingerprintError(f"metadata must be a JSON object: {path}")
    return value, {path.absolute(): (digest, runtime_stat)}


def _episode_rows(
    meta_root: Path,
) -> tuple[list[dict], dict[Path, tuple[str, dict[str, int]]]]:
    path = meta_root / "episodes.jsonl"
    verified_sources: dict[Path, tuple[str, dict[str, int]]] = {}
    if path.is_file():
        payload, digest, runtime_stat = read_verified_bytes(path)
        verified_sources[path.absolute()] = (digest, runtime_stat)
        rows = [
            strict_json_loads(line.encode("utf-8"), name=f"{path}:{line_number}")
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1)
            if line.strip()
        ]
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
            payload, digest, runtime_stat = read_verified_bytes(parquet_path)
            verified_sources[parquet_path.absolute()] = (digest, runtime_stat)
            rows.extend(pd.read_parquet(io.BytesIO(payload)).to_dict(orient="records"))
    canonical = []
    seen = set()
    for row_number, row in enumerate(rows, 1):
        episode = row.get("episode_index") if isinstance(row, dict) else None
        length = row.get("length") if isinstance(row, dict) else None
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
        ):
            raise EgoDatasetFingerprintError(f"invalid episode metadata row {row_number}")
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
    return canonical, verified_sources


def _training_source_manifest(
    root: Path,
    info: dict,
    episode_rows: list[dict],
    preverified_sources: dict[Path, tuple[str, dict[str, int]]],
) -> tuple[list[dict], dict, dict, list[dict]]:
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
        chunk_size = info.get("chunks_size", 1000)
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
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
                actual_path = actual_path.absolute()
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
    runtime_receipt = []
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
        verified = preverified_sources.get(path.absolute())
        if verified is None:
            raw_sha256, runtime_stat = hash_verified_regular_file(path)
        else:
            raw_sha256, runtime_stat = verified
        audit_counts[role]["files"] += 1
        audit_counts[role]["bytes"] += runtime_stat["size_bytes"]
        runtime_receipt.append(
            {
                "logical_path": relative,
                "logical_role": role,
                "path_record_only": str(path.absolute()),
                "sha256": raw_sha256,
                "stat": runtime_stat,
            }
        )
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
                "size_bytes": runtime_stat["size_bytes"],
                "sha256": raw_sha256,
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
    return manifest, receipt, audit_counts, runtime_receipt


def derive_ego_t5_source_receipt(
    t5_embedding_dir: str | Path,
    *,
    episode_indices: list[int] | tuple[int, ...],
    dataset_identity: dict,
    data_root: str | Path,
) -> dict:
    """Validate provenance and hash the exact Stage1 T5 source.

    The external directory is deliberately absent from the portable manifest.
    The required provenance manifest binds the runtime-derived dataset ID,
    exact per-episode prompt, model revision, checkpoint, tokenizer and encoder
    code.  Existing bare ``.pt`` files therefore cannot self-attest.
    """

    from world_action_model.ego_t5_source import (
        EGO_T5_MANIFEST_FILENAME,
        validate_ego_t5_source_manifest,
    )

    root = Path(t5_embedding_dir).expanduser().absolute()
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
    if indices != sorted(dataset_identity.get("episode_lengths", {})):
        raise EgoDatasetFingerprintError(
            "T5 episode request differs from runtime-derived dataset identity"
        )
    validation = validate_ego_t5_source_manifest(
        root,
        dataset_identity=dataset_identity,
        data_root=data_root,
        inspect_tensors=True,
    )
    provenance_size = validation["manifest_stat_record_only"]["size_bytes"]
    manifest = [
        {
            "path": f"t5_embedding/{EGO_T5_MANIFEST_FILENAME}",
            "logical_role": "t5_generation_provenance",
            "digest_mode": "raw_file_bytes",
            "size_bytes": provenance_size,
            "sha256": validation["manifest_sha256"],
        },
        *validation["portable_episode_files"],
    ]
    runtime_receipt = validation["runtime_episode_receipt_record_only"]
    total_bytes = provenance_size + sum(
        row["size_bytes"] for row in validation["portable_episode_files"]
    )
    manifest_sha256 = _canonical_sha256(manifest)
    receipt = {
        "receipt_version": EGO_T5_SOURCE_RECEIPT_VERSION,
        "manifest_sha256": manifest_sha256,
        "file_count": len(manifest),
        "episode_file_count": len(validation["portable_episode_files"]),
        "total_bytes": total_bytes,
        "logical_roles": ["t5_generation_provenance", "t5_episode_condition"],
        "pattern": "episode_{episode_index:06d}.pt",
        "provenance_manifest_sha256": validation["manifest_sha256"],
        "encoder_provenance_sha256": validation["encoder_provenance_sha256"],
        "prompt_manifest_sha256": validation["prompt_manifest_sha256"],
        "dataset_id": dataset_identity["dataset_id"],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return {
        "source_dir_record_only": str(root),
        "source_manifest": manifest,
        "source_manifest_sha256": manifest_sha256,
        "source_receipt": receipt,
        "source_receipt_sha256": receipt["receipt_sha256"],
        "runtime_source_receipt_record_only": runtime_receipt,
        "provenance_manifest": validation["manifest"],
    }


def derive_ego_dataset_identity(data_root: str | Path) -> dict:
    """Validate Stage1 metadata and derive a path-independent dataset ID."""

    root = Path(data_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise EgoDatasetFingerprintError(
            f"Stage1 LeRobot root must be a real directory: {root}"
        )
    info, verified_sources = _load_json(root / "meta" / "info.json")
    if str(info.get("codebase_version", "")).lower() != "v2.1":
        raise EgoDatasetFingerprintError(
            "production Ego Stage1 accepts only the audited LeRobot v2.1 metadata schema"
        )
    rows, episode_verified_sources = _episode_rows(root / "meta")
    verified_sources.update(episode_verified_sources)
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
    total_episodes = info.get("total_episodes")
    total_frames = info.get("total_frames")
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, int)
        or isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_episodes != len(rows)
        or total_frames != sum(episode_lengths.values())
    ):
        raise EgoDatasetFingerprintError("dataset totals differ from episode metadata")
    (
        source_manifest,
        source_receipt,
        source_audit_summary,
        runtime_source_receipt,
    ) = _training_source_manifest(root, info, rows, verified_sources)

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
        "runtime_source_receipt_record_only": runtime_source_receipt,
        "portable_info_ignored_fields": list(EGO_PORTABLE_INFO_IGNORED_FIELDS),
        "training_source_roots": list(EGO_TRAINING_SOURCE_ROOTS),
        "episode_lengths": episode_lengths,
        "episode_count": len(rows),
        "total_frames": sum(episode_lengths.values()),
        "fps": EGO_STAGE1_FPS,
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "data_root": str(root),
    }

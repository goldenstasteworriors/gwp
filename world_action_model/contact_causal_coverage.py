"""Verified dataset-level contact coverage replay for causal-main admission."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import numpy as np

from world_action_model.contact_causal_production import (
    CausalProductionError,
    SEEDS,
    sha256_json,
    validate_dataset_supervision_coverage,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.verified_io import read_verified_bytes, strict_json_loads


COVERAGE_REPLAY_VERSION = "contact-dataset-supervision-coverage-replay/v1"
GROUPED_FIELDS = tuple(
    sorted(
        SEGMENT_BINARY_FIELDS + SEGMENT_CATEGORICAL_FIELDS
        + SEGMENT_CONTINUOUS_FIELDS
    )
)
PHASES = ("onset", "hold", "release")
REQUIRED_ARRAY_KEYS = {
    "contact_target", "contact_loss_mask", "contact_confidence",
    "contact_condition_mask",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_EPISODE_RE = re.compile(r"^episode_([0-9]{6})\.npz$")


def _nonzero_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or not _SHA_RE.fullmatch(value)
        or value == "0" * 64
    ):
        raise CausalProductionError(f"{name} must be a nonzero SHA-256")
    return value


def _verified_json(path: Path, name: str):
    loader = None
    inference_loader = None
    try:
        payload, digest, _ = read_verified_bytes(path)
        value = strict_json_loads(payload, name=name)
    except (OSError, ValueError) as exc:
        raise CausalProductionError(f"cannot replay verified {name}") from exc
    if not isinstance(value, dict):
        raise CausalProductionError(f"{name} must be an object")
    return value, digest


def _grouped_counts(mask: np.ndarray, confidence: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in SEGMENT_BINARY_FIELDS + SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        result[field] = int(
            np.count_nonzero(
                (mask[..., field_slice] == 1)
                & (confidence[..., field_slice] > 0)
            )
        )
    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        result[field] = int(
            np.count_nonzero(
                np.all(mask[..., field_slice] == 1, axis=-1)
                & np.all(confidence[..., field_slice] > 0, axis=-1)
            )
        )
    if set(result) != set(GROUPED_FIELDS):
        raise CausalProductionError("grouped coverage field schema drifted")
    return {field: result[field] for field in GROUPED_FIELDS}


def _phase_counts(
    target: np.ndarray, mask: np.ndarray, confidence: np.ndarray,
) -> dict[str, int]:
    presence_slice = SEGMENT_FIELD_SLICES["presence"]
    presence = target[..., presence_slice].reshape(target.shape[:-1])
    presence_known = (
        mask[..., presence_slice].reshape(mask.shape[:-1]) == 1
    ) & (confidence[..., presence_slice].reshape(confidence.shape[:-1]) > 0)
    positive = presence_known & (presence == 1)
    start = target[..., SEGMENT_FIELD_SLICES["start_tau"]].reshape(target.shape[:-1])
    end = target[..., SEGMENT_FIELD_SLICES["end_tau"]].reshape(target.shape[:-1])
    left = target[..., SEGMENT_FIELD_SLICES["left_censored"]].reshape(target.shape[:-1])
    right = target[..., SEGMENT_FIELD_SLICES["right_censored"]].reshape(target.shape[:-1])
    def known(field: str) -> np.ndarray:
        field_slice = SEGMENT_FIELD_SLICES[field]
        return (
            mask[..., field_slice].reshape(mask.shape[:-1]) == 1
        ) & (confidence[..., field_slice].reshape(confidence.shape[:-1]) > 0)

    start_known = known("start_tau")
    end_known = known("end_tau")
    left_known = known("left_censored")
    right_known = known("right_censored")
    return {
        "onset": int(np.count_nonzero(
            positive & left_known & start_known & (left == 0) & (start >= 0)
        )),
        "hold": int(np.count_nonzero(
            positive & start_known & end_known & (end > start)
        )),
        "release": int(np.count_nonzero(
            positive & right_known & end_known & (right == 0) & (end <= 1)
        )),
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray], *, name: str) -> None:
    if set(arrays) != REQUIRED_ARRAY_KEYS:
        forbidden = sorted(
            key for key in arrays
            if "action" in key.lower() or "hand_active" in key.lower()
            or "proxy" in key.lower()
        )
        suffix = f"; forbidden proxies={forbidden}" if forbidden else ""
        raise CausalProductionError(f"{name} sidecar array schema differs{suffix}")
    shapes = {tuple(value.shape) for value in arrays.values()}
    if len(shapes) != 1:
        raise CausalProductionError(f"{name} sidecar shapes differ")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[0] <= 0 or shape[1:] != (12, SEGMENT_TARGET_DIM):
        raise CausalProductionError(f"{name} sidecar shape is not [T,12,22]")
    for key, value in arrays.items():
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise CausalProductionError(f"{name}:{key} is not finite float32")
    mask = arrays["contact_loss_mask"]
    confidence = arrays["contact_confidence"]
    if not np.logical_or(mask == 0, mask == 1).all():
        raise CausalProductionError(f"{name} loss mask is not exactly binary")
    if (confidence < 0).any() or (confidence[mask == 0] != 0).any():
        raise CausalProductionError(f"{name} confidence invents unknown supervision")


def _split_sets(
    split_episode_ids: Mapping[str, list[int]], expected_episode_ids: set[int],
) -> tuple[set[int], set[int], set[int]]:
    if not isinstance(split_episode_ids, dict) or set(split_episode_ids) != {
        "train", "val", "test"
    }:
        raise CausalProductionError("coverage split must be exact train/val/test")
    sets = []
    for name in ("train", "val", "test"):
        values = split_episode_ids[name]
        if (
            not isinstance(values, list) or not values
            or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
            or len(set(values)) != len(values)
        ):
            raise CausalProductionError(f"coverage split {name} is invalid")
        sets.append(set(values))
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise CausalProductionError("coverage split leaks episodes")
    if set.union(*sets) != expected_episode_ids:
        raise CausalProductionError("coverage split is incomplete")
    return sets[0], sets[1], sets[2]


def recompute_dataset_supervision_coverage(
    *, sidecar_dir: str | Path, expected_meta_sha256: str,
    expected_episode_sha256: Mapping[str, str], dataset_id: str,
    split_seed: int, split_episode_ids: Mapping[str, list[int]],
    label_origin: str = "human_adjudicated_robot_native",
) -> dict[str, Any]:
    """Recompute exact grouped and phase coverage from verified reviewed masks."""

    if isinstance(split_seed, bool) or split_seed not in SEEDS:
        raise CausalProductionError("coverage split seed is invalid")
    root = Path(sidecar_dir).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise CausalProductionError("coverage sidecar root must be a real directory")
    meta, meta_sha = _verified_json(root / "meta.json", "coverage sidecar meta")
    if meta_sha != _nonzero_sha(expected_meta_sha256, "coverage meta"):
        raise CausalProductionError("coverage sidecar meta bytes changed")
    if (
        meta.get("schema_version") != SEGMENT_SCHEMA_VERSION
        or meta.get("target_dim") != SEGMENT_TARGET_DIM
        or meta.get("num_queries") != 12
        or meta.get("annotation_contract")
        != {
            "annotation_role": "adjudicated", "review_status": "reviewed",
            "label_origin": "human_adjudicated",
            "automatic_or_proposal_ground_truth": False,
        }
        or meta.get("dataset_id") != dataset_id
    ):
        raise CausalProductionError("coverage source is not reviewed human contact")
    if not isinstance(expected_episode_sha256, dict) or not expected_episode_sha256:
        raise CausalProductionError("coverage episode SHA map is empty")
    episode_ids: set[int] = set()
    for filename, digest in expected_episode_sha256.items():
        match = _EPISODE_RE.fullmatch(filename) if isinstance(filename, str) else None
        if match is None:
            raise CausalProductionError("coverage episode SHA key is invalid")
        _nonzero_sha(digest, f"coverage {filename}")
        episode_ids.add(int(match.group(1)))
    if len(episode_ids) != len(expected_episode_sha256):
        raise CausalProductionError("coverage episode IDs are duplicated")
    _split_sets(split_episode_ids, episode_ids)
    if meta.get("episode_npz_sha256") != dict(expected_episode_sha256):
        raise CausalProductionError("coverage meta/expected episode SHA maps differ")
    actual_names = {path.name for path in root.glob("episode_*.npz")}
    if actual_names != set(expected_episode_sha256):
        raise CausalProductionError("coverage sidecar episode file set differs")

    totals = {field: 0 for field in GROUPED_FIELDS}
    phases = {phase: 0 for phase in PHASES}
    total_windows = supervised_windows = zero_windows = 0
    episode_receipts = {}
    for filename in sorted(expected_episode_sha256):
        try:
            payload, digest, _ = read_verified_bytes(
                root / filename, expected_sha256=expected_episode_sha256[filename]
            )
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
        except (OSError, ValueError) as exc:
            raise CausalProductionError(f"cannot replay coverage sidecar {filename}") from exc
        _validate_arrays(arrays, name=filename)
        field_counts = _grouped_counts(
            arrays["contact_loss_mask"], arrays["contact_confidence"]
        )
        phase_counts = _phase_counts(
            arrays["contact_target"], arrays["contact_loss_mask"],
            arrays["contact_confidence"],
        )
        per_window = np.zeros(arrays["contact_target"].shape[0], dtype=np.int64)
        mask = arrays["contact_loss_mask"]
        confidence = arrays["contact_confidence"]
        for field in SEGMENT_BINARY_FIELDS + SEGMENT_CONTINUOUS_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            per_window += np.count_nonzero(
                (mask[..., field_slice] == 1) & (confidence[..., field_slice] > 0),
                axis=(1, 2),
            )
        for field in SEGMENT_CATEGORICAL_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            per_window += np.count_nonzero(
                np.all(mask[..., field_slice] == 1, axis=-1)
                & np.all(confidence[..., field_slice] > 0, axis=-1), axis=1,
            )
        window_count = int(per_window.size)
        supervised = int(np.count_nonzero(per_window > 0))
        total_windows += window_count
        supervised_windows += supervised
        zero_windows += window_count - supervised
        for field in GROUPED_FIELDS:
            totals[field] += field_counts[field]
        for phase in PHASES:
            phases[phase] += phase_counts[phase]
        episode_receipts[filename] = {
            "npz_sha256": digest, "window_count": window_count,
            "supervised_window_count": supervised,
            "field_eligible_counts": field_counts, "phase_coverage": phase_counts,
        }

    provenance_path = root / "window_segment_provenance.csv"
    try:
        provenance_bytes, provenance_sha, _ = read_verified_bytes(
            provenance_path,
            expected_sha256=meta.get("window_segment_provenance_sha256"),
        )
        rows = list(csv.DictReader(io.StringIO(provenance_bytes.decode("utf-8"))))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CausalProductionError("cannot replay physical-contact provenance") from exc
    physical_events = set()
    for row in rows:
        segment_id = row.get("segment_id")
        if not segment_id:
            continue
        try:
            episode_id = int(row.get("episode_id", ""))
        except ValueError as exc:
            raise CausalProductionError("physical provenance episode ID is invalid") from exc
        if episode_id not in episode_ids:
            raise CausalProductionError("physical provenance references an unknown episode")
        physical_events.add((episode_id, segment_id))
    positive_count = len(physical_events)
    declared_positive = meta.get("unique_physical_segments")
    if (
        isinstance(declared_positive, bool) or not isinstance(declared_positive, int)
        or positive_count <= 0 or positive_count != declared_positive
    ):
        raise CausalProductionError("coverage lacks positive physical-contact events")
    if any(totals[field] <= 0 for field in GROUPED_FIELDS):
        raise CausalProductionError("coverage has an empty grouped supervision field")
    if any(phases[phase] <= 0 for phase in PHASES):
        raise CausalProductionError("coverage lacks onset/hold/release coverage")
    coverage_source = {
        "replay_version": COVERAGE_REPLAY_VERSION,
        "sidecar_meta_sha256": meta_sha,
        "provenance_sha256": provenance_sha,
        "episode_receipts": episode_receipts,
        "split_episode_ids": dict(split_episode_ids),
    }
    receipt = {
        "receipt_version": "contact-dataset-supervision-coverage/v2",
        "dataset_id": dataset_id, "split_seed": split_seed,
        "contact_schema_version": SEGMENT_SCHEMA_VERSION,
        "label_origin": label_origin, "review_status": "reviewed",
        "hand_active_used_as_contact_ground_truth": False,
        "grip_used_as_contact_ground_truth": False,
        "total_windows": total_windows, "supervised_windows": supervised_windows,
        "zero_supervision_windows": zero_windows,
        "total_grouped_eligible_items": sum(totals.values()),
        "positive_contact_event_count": positive_count,
        "field_eligible_counts": totals, "phase_coverage": phases,
        "coverage_source_sha256": sha256_json(coverage_source),
        "sidecar_bundle_sha256": _nonzero_sha(
            meta.get("sidecar_bundle_sha256"), "coverage sidecar bundle"
        ),
    }
    validate_dataset_supervision_coverage(receipt)
    return receipt


def validate_recomputed_dataset_supervision_coverage(
    receipt: Mapping[str, Any], **replay_kwargs: Any,
) -> dict[str, Any]:
    validated = validate_dataset_supervision_coverage(receipt)
    recomputed = recompute_dataset_supervision_coverage(**replay_kwargs)
    if dict(validated) != recomputed:
        raise CausalProductionError("persisted coverage differs from verified sidecar replay")
    return recomputed


def coverage_negative_acceptance_helper() -> tuple[str, ...]:
    """Formal list of fail-closed cases exercised by the shared CPU audit."""

    return (
        "empty_episode_sha_map", "all_zero_sha", "unknown_or_extra_sidecar_field",
        "action_proxy_field", "hand_active_proxy_field", "empty_grouped_field",
        "missing_onset_hold_release", "no_positive_physical_contact",
        "overlapping_split", "incomplete_split",
    )


def real_loader_synthetic_instantiation_helper(
    project_workspace: str | Path,
) -> dict[str, Any]:
    """Instantiate the production loader contract on a tiny CPU-only fixture.

    The fixture is created below the caller-supplied project workspace and is
    always removed.  It does not open videos, instantiate a model, or touch a
    training/GPU runtime.
    """

    workspace = Path(project_workspace).expanduser().absolute()
    root = workspace / ".op42_real_loader_synthetic_acceptance"
    if root.exists() or root.is_symlink():
        raise FileExistsError("synthetic loader acceptance root already exists")
    data_root = root / "dataset"
    sidecar_root = root / "sidecar"

    def write_exclusive(path: Path, payload: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("short synthetic acceptance write")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        return hashlib.sha256(payload).hexdigest()

    try:
        dataset_id = "synthetic-reviewed-loader-contract-dataset"
        info = {
            "codebase_version": "v2.1", "robot_type": "sonic_g1_asymmetric25",
            "dataset_id": dataset_id, "total_episodes": 1, "total_frames": 1,
            "total_tasks": 1, "total_videos": 0, "total_chunks": 1,
            "chunks_size": 1000, "fps": 50, "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [3], "names": None},
                "action": {"dtype": "float32", "shape": [2], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        info_payload = json.dumps(
            info, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        info_sha = write_exclusive(data_root / "meta/info.json", info_payload)
        write_exclusive(
            data_root / "meta/tasks.jsonl",
            b'{"task":"synthetic contact","task_index":0}\n',
        )
        write_exclusive(
            data_root / "meta/episodes.jsonl",
            b'{"episode_index":0,"length":1,"tasks":["synthetic contact"]}\n',
        )
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "real loader open acceptance requires existing labvla pyarrow/torch"
            ) from exc
        parquet_path = data_root / "data/chunk-000/episode_000000.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array([0.0], type=pa.float32()),
                    "frame_index": pa.array([0], type=pa.int64()),
                    "episode_index": pa.array([0], type=pa.int64()),
                    "index": pa.array([0], type=pa.int64()),
                    "task_index": pa.array([0], type=pa.int64()),
                    "observation.state": pa.array([[0.0, 0.0, 0.0]], type=pa.list_(pa.float32())),
                    "action": pa.array([[0.0, 0.0]], type=pa.list_(pa.float32())),
                }
            ),
            parquet_path,
        )
        t5_buffer = io.BytesIO()
        torch.save(
            {"t5_embedding": torch.zeros((1, 4096), dtype=torch.float32)},
            t5_buffer,
        )
        t5_path = data_root / "t5_embedding/episode_000000.pt"
        write_exclusive(t5_path, t5_buffer.getvalue())
        contact_target = np.zeros((1, 12, SEGMENT_TARGET_DIM), dtype=np.float32)
        contact_loss_mask = np.zeros_like(contact_target)
        contact_confidence = np.zeros_like(contact_target)
        contact_condition_mask = np.zeros_like(contact_target)
        contact_target[0, 0, 0] = 1.0
        contact_loss_mask[0, 0, 0] = 1.0
        contact_confidence[0, 0, 0] = 1.0
        npz_buffer = io.BytesIO()
        np.savez(
            npz_buffer,
            contact_target=contact_target,
            contact_loss_mask=contact_loss_mask,
            contact_confidence=contact_confidence,
            contact_condition_mask=contact_condition_mask,
        )
        episode_sha = write_exclusive(
            sidecar_root / "episode_000000.npz", npz_buffer.getvalue()
        )
        sidecar_meta = {
            "schema_version": SEGMENT_SCHEMA_VERSION,
            "target_dim": SEGMENT_TARGET_DIM,
            "num_queries": 12,
            "dataset_id": dataset_id,
            "episode_npz_sha256": {"episode_000000.npz": episode_sha},
            "builder_identity": (
                "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py"
            ),
            "annotation_contract": {
                "annotation_role": "adjudicated", "review_status": "reviewed",
                "label_origin": "human_adjudicated",
                "automatic_or_proposal_ground_truth": False,
            },
        }
        meta_payload = json.dumps(
            sidecar_meta, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        meta_sha = write_exclusive(sidecar_root / "meta.json", meta_payload)

        def runtime_entry(path: Path, logical_path: str, logical_role: str) -> dict[str, Any]:
            _, digest, stat_receipt = read_verified_bytes(path)
            return {
                "logical_path": logical_path, "logical_role": logical_role,
                "path_record_only": str(path.absolute()), "sha256": digest,
                "stat": stat_receipt,
            }

        source_receipt = [
            runtime_entry(data_root / "meta/info.json", "meta/info.json", "metadata"),
            runtime_entry(data_root / "meta/tasks.jsonl", "meta/tasks.jsonl", "metadata"),
            runtime_entry(data_root / "meta/episodes.jsonl", "meta/episodes.jsonl", "metadata"),
            runtime_entry(parquet_path, "data/chunk-000/episode_000000.parquet", "parquet"),
        ]
        t5_entry = runtime_entry(
            t5_path, "t5_embedding/episode_000000.pt", "t5_episode_condition"
        )
        t5_entry["episode_index"] = 0
        filter_receipt = {
            "receipt_version": "contact-asym25-training-data-receipt/v1",
            "dataset_id": dataset_id,
            "action_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
            "split_seed": 6666,
            "train_episode_ids": [0], "validation_episode_ids": [1],
            "heldout_episode_ids": [2], "dataset_info_sha256": info_sha,
            "episode_metadata_sha256": hashlib.sha256(b"episode-meta").hexdigest(),
            "norm_stats_sha256": hashlib.sha256(b"norm").hexdigest(),
            "sidecar_bundle_sha256": hashlib.sha256(b"sidecar").hexdigest(),
            "split_manifest_sha256": hashlib.sha256(b"split").hexdigest(),
            "training_gate_sha256": hashlib.sha256(b"gate").hexdigest(),
            "preflight_code_sha256": hashlib.sha256(b"preflight").hexdigest(),
        }
        private_cache_parent = root / "verified-private-cache"
        private_cache_parent.mkdir(parents=True, exist_ok=False)
        private_cache_parent.chmod(0o700)
        from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset

        loader = WAMLeRobotDataset(
            data_path=str(data_root), data_size=1, t5_load_from="dir",
            t5_cfg={"t5_embedding_dir": str(data_root / "t5_embedding")},
            episodes=[0], force_cache_sync=False, skip_video_decoding=True,
            training_source_runtime_receipt=source_receipt,
            t5_runtime_source_receipt=[t5_entry],
            verified_private_cache_parent=str(private_cache_parent),
            verified_training_run_nonce="synthetic-loader-run-nonce-v1",
            production_episode_filter_receipt=filter_receipt,
            production_episode_filter_receipt_sha256=sha256_json(filter_receipt),
            contact_sidecar_dir=str(sidecar_root), contact_sidecar_strict=True,
            contact_sidecar_require_reviewed_v1_1=True,
            contact_sidecar_expected_meta_sha256=meta_sha,
            contact_sidecar_expected_episode_sha256={
                "episode_000000.npz": episode_sha
            },
            contact_label_origin="human_adjudicated_robot_native",
            contact_human_adjudicated_only=True,
            action_contact_ground_truth=False, hand_active_contact_ground_truth=False,
            grip_contact_ground_truth=False, other_proxy_contact_ground_truth=False,
            matched_control_identity=True, matched_control_dataset_id=dataset_id,
            matched_control_dataset_info_sha256=info_sha,
        )
        loader.open()
        loaded_batch = loader[0]
        if (
            set(key for key in loaded_batch if key.startswith("contact_"))
            != {
                "contact_target", "contact_loss_mask", "contact_confidence",
                "contact_condition_mask",
            }
            or not np.array_equal(loaded_batch["contact_target"], contact_target[0])
            or any(
                name in loaded_batch
                for name in ("action_contact", "hand_active", "grip_contact")
            )
            or loaded_batch.get("matched_control_dataset_id") != dataset_id
            or int(loaded_batch["matched_control_global_index"]) != 0
            or not isinstance(loaded_batch.get("t5_embedding"), torch.Tensor)
            or tuple(loaded_batch["t5_embedding"].shape) != (1, 4096)
            or type(loader.dataset).__name__ != "FastLeRobotDataset"
            or set(loader.dataset._verified_source_entries) != {
                "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl",
                "data/chunk-000/episode_000000.parquet",
            }
        ):
            raise AssertionError(
                "real loader open/sample did not replay all verified causal inputs"
            )
        result = {
            "loader_class": type(loader).__name__,
            "runtime_loader_class": type(loader.dataset).__name__,
            "loader_opened_and_sampled": True,
            "verified_source_entry_count": len(loader.dataset._verified_source_entries),
            "verified_t5_entry_count": len(loader.t5_runtime_source_receipt),
            "matched_control_identity": loader.matched_control_identity,
            "matched_control_dataset_id": loader.matched_control_dataset_id,
            "contact_human_adjudicated_only": loader.contact_human_adjudicated_only,
            "sidecar_meta_sha256": loader.contact_sidecar_expected_meta_sha256,
            "episode_sha_map_size": len(loader.contact_sidecar_expected_episode_sha256),
            "episode_sidecar_sha256": episode_sha,
            "loaded_contact_shape": list(loaded_batch["contact_target"].shape),
            "loaded_t5_shape": list(loaded_batch["t5_embedding"].shape),
        }
        loader.close()
        loader = None
        inference_loader = WAMLeRobotDataset(
            data_path=str(data_root), data_size=1, t5_load_from="dir",
            t5_cfg={"t5_embedding_dir": str(data_root / "t5_embedding")},
            delta_frames={"observation.state": [0]}, episodes=[0],
            force_cache_sync=False, skip_video_decoding=True,
            training_source_runtime_receipt=source_receipt,
            t5_runtime_source_receipt=[t5_entry],
            verified_private_cache_parent=str(private_cache_parent),
            verified_training_run_nonce="synthetic-inference-run-nonce-v1",
            robotype="sonic_g1_asymmetric25",
        )
        inference_loader.open_target_free_inference_view()
        inference_raw = inference_loader._get_data(0)
        if (
            any(
                key == "action"
                or key.startswith("action_")
                or key.startswith("contact_")
                for key in inference_raw
            )
            or inference_loader._target_free_removed_source_columns
            != ("action",)
        ):
            raise AssertionError(
                "target-free real loader projection read an action/contact target"
            )
        result["target_free_source_projection_sampled"] = True
        result["target_free_removed_columns"] = list(
            inference_loader._target_free_removed_source_columns
        )
        inference_loader.close()
        inference_loader = None
        if result != {
            "loader_class": "WAMLeRobotDataset",
            "runtime_loader_class": "FastLeRobotDataset",
            "loader_opened_and_sampled": True,
            "verified_source_entry_count": 4,
            "verified_t5_entry_count": 1,
            "matched_control_identity": True,
            "matched_control_dataset_id": dataset_id,
            "contact_human_adjudicated_only": True,
            "sidecar_meta_sha256": meta_sha,
            "episode_sha_map_size": 1,
            "episode_sidecar_sha256": episode_sha,
            "loaded_contact_shape": [12, SEGMENT_TARGET_DIM],
            "loaded_t5_shape": [1, 4096],
            "target_free_source_projection_sampled": True,
            "target_free_removed_columns": ["action"],
        }:
            raise AssertionError("real loader lost causal admission fields")
        return result
    finally:
        if inference_loader is not None:
            inference_loader.close()
        if loader is not None:
            loader.close()
        if root.exists():
            shutil.rmtree(root)

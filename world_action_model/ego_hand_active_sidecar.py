"""Verified 20 Hz hand-tracking-validity sidecars for Ego Stage1.

``hand_status`` in the processed EgoHumanoid data is a binary square-wave
proxy.  It is only a supervised target when the corresponding upstream hand
tracker reports that hand as active.  This module keeps that validity signal
separate from both the proxy value and the physical-contact labels.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

import h5py
import numpy as np

from world_action_model.ego_dataset_fingerprint import (
    EGO_STAGE1_FPS,
    EGO_STAGE1_FRAME_CLOCK,
)
from world_action_model.verified_io import (
    hash_verified_regular_file,
    read_verified_bytes,
    strict_json_loads,
)


EGO_HAND_ACTIVE_SCHEMA_VERSION = "ego-hand-active-sidecar/v1"
EGO_HAND_ACTIVE_BUILDER = "scripts/build_ego_hand_active_sidecars.py"
EGO_HAND_ACTIVE_HORIZON = 48
EGO_HAND_ACTIVE_START_OFFSET = 1
EGO_HAND_ACTIVE_FIELD_ORDER = ("left", "right")
EGO_HAND_ACTIVE_SEMANTICS = "binary_tracker_validity_0_inactive_1_active"
EGO_HAND_ACTIVE_KEYS = (
    "ego_hand_active_current",
    "ego_hand_active_future",
    "ego_hand_active_transition_pair",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EgoHandActiveSidecarError(ValueError):
    pass


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest, _ = hash_verified_regular_file(path)
    return digest


def build_hand_active_arrays(
    current: np.ndarray,
    *,
    horizon: int = EGO_HAND_ACTIVE_HORIZON,
) -> dict[str, np.ndarray]:
    """Build current, future, and adjacent-validity masks without edge hold."""

    current = np.asarray(current)
    if (
        current.ndim != 2
        or current.shape[1] != 2
        or current.dtype != np.bool_
    ):
        raise EgoHandActiveSidecarError(
            f"current hand-active mask must be bool [T,2], got {current.dtype} {current.shape}"
        )
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise EgoHandActiveSidecarError("hand-active horizon must be positive")
    frame_count = int(current.shape[0])
    future = np.zeros((frame_count, horizon, 2), dtype=np.bool_)
    pair = np.zeros_like(future)
    for offset in range(1, horizon + 1):
        usable = frame_count - offset
        if usable <= 0:
            break
        future[:usable, offset - 1] = current[offset:]
        pair[:usable, offset - 1] = (
            current[offset - 1 : frame_count - 1] & current[offset:]
        )
    return {
        "ego_hand_active_current": current.copy(),
        "ego_hand_active_future": future,
        "ego_hand_active_transition_pair": pair,
    }


def validate_hand_active_arrays(
    arrays: dict[str, np.ndarray],
    *,
    frame_count: int,
    horizon: int = EGO_HAND_ACTIVE_HORIZON,
) -> None:
    if set(arrays) != set(EGO_HAND_ACTIVE_KEYS):
        raise EgoHandActiveSidecarError("hand-active sidecar array names differ")
    expected_shapes = {
        "ego_hand_active_current": (frame_count, 2),
        "ego_hand_active_future": (frame_count, horizon, 2),
        "ego_hand_active_transition_pair": (frame_count, horizon, 2),
    }
    for key, expected_shape in expected_shapes.items():
        value = arrays[key]
        if value.dtype != np.bool_ or value.shape != expected_shape:
            raise EgoHandActiveSidecarError(
                f"{key} must be bool {expected_shape}, got {value.dtype} {value.shape}"
            )
    expected = build_hand_active_arrays(
        arrays["ego_hand_active_current"],
        horizon=horizon,
    )
    for key in EGO_HAND_ACTIVE_KEYS[1:]:
        if not np.array_equal(arrays[key], expected[key]):
            raise EgoHandActiveSidecarError(
                f"{key} does not implement the declared future/pair clock"
            )


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Return byte-stable compressed NPZ bytes for exact bundle receipts."""

    if set(arrays) != set(EGO_HAND_ACTIVE_KEYS):
        raise EgoHandActiveSidecarError("cannot encode unexpected hand-active arrays")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for key in EGO_HAND_ACTIVE_KEYS:
            payload = io.BytesIO()
            np.lib.format.write_array(payload, arrays[key], allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def load_hand_active_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    frame_count: int,
    horizon: int = EGO_HAND_ACTIVE_HORIZON,
) -> dict[str, np.ndarray]:
    try:
        payload, _, _ = read_verified_bytes(path, expected_sha256=expected_sha256)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise EgoHandActiveSidecarError(f"invalid hand-active NPZ: {path}") from exc
    validate_hand_active_arrays(arrays, frame_count=frame_count, horizon=horizon)
    if payload != deterministic_npz_bytes(arrays):
        raise EgoHandActiveSidecarError(
            f"hand-active NPZ is not the canonical deterministic encoding: {path}"
        )
    return arrays


def _episode_files(root: Path) -> dict[int, Path]:
    if root.is_symlink() or not root.is_dir():
        raise EgoHandActiveSidecarError(f"hand-active source must be a real directory: {root}")
    result: dict[int, Path] = {}
    for path in root.glob("episode_*.hdf5"):
        match = re.fullmatch(r"episode_([0-9]+)\.hdf5", path.name)
        if match is None or path.is_symlink() or not path.is_file():
            continue
        episode = int(match.group(1))
        if episode in result:
            raise EgoHandActiveSidecarError(f"duplicate episode HDF5: {episode}")
        result[episode] = path
    return result


def read_aligned_hand_active(
    raw_path: str | Path,
    final_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Read and independently verify the raw->20 Hz validity alignment."""

    raw_path = Path(raw_path)
    final_path = Path(final_path)
    try:
        with h5py.File(raw_path, "r") as raw:
            if raw.attrs.get("hand_active_format") != "0=inactive, 1=active":
                raise EgoHandActiveSidecarError("raw hand_active semantics are missing")
            left = np.asarray(raw["left_hand_active"])
            right = np.asarray(raw["right_hand_active"])
            raw_timestamps = np.asarray(raw["local_timestamps_ns"])
            raw_interval = float(raw.attrs.get("collection_interval_s", -1.0))
        with h5py.File(final_path, "r") as final:
            final_timestamps = np.asarray(final["local_timestamps_ns"])
            hand_status = np.asarray(final["hand_status"])
            action_eef = np.asarray(final["action_eef"])
            action_delta = np.asarray(final["action_delta_eef"])
            final_interval = float(final.attrs.get("collection_interval_s", -1.0))
            downsample_rate = int(final.attrs.get("downsample_rate", -1))
            hand_description = final["hand_status"].attrs.get("description")
            hand_downsample = int(final["hand_status"].attrs.get("downsample", -1))
            hand_method = final["hand_status"].attrs.get("method")
            delta_dt = float(final["action_delta_eef"].attrs.get("dt", -1.0))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, EgoHandActiveSidecarError):
            raise
        raise EgoHandActiveSidecarError(
            f"cannot read hand-active source pair: {raw_path}, {final_path}"
        ) from exc
    raw_length = len(left)
    if (
        left.shape != (raw_length,)
        or right.shape != (raw_length,)
        or raw_timestamps.shape != (raw_length,)
        or not np.issubdtype(raw_timestamps.dtype, np.integer)
        or not np.isin(left, (0, 1)).all()
        or not np.isin(right, (0, 1)).all()
        or not np.isfinite(raw_timestamps).all()
        or raw_length <= 0
    ):
        raise EgoHandActiveSidecarError("raw hand-active arrays are malformed")
    final_length = len(final_timestamps)
    if (
        hand_status.shape != (final_length, 2)
        or action_eef.shape != (final_length, 14)
        or action_delta.shape != (final_length, 12)
        or final_timestamps.dtype != np.dtype(np.float64)
        or not np.isfinite(hand_status).all()
        or not np.isin(hand_status, (0.0, 1.0)).all()
        or not np.isfinite(final_timestamps).all()
    ):
        raise EgoHandActiveSidecarError("processed action/hand arrays are malformed")
    if (
        raw_interval != 0.01
        or final_interval != 0.05
        or delta_dt != 0.05
        or downsample_rate != 5
        or hand_downsample != 5
        or hand_method != "square_wave_approximation"
        or hand_description
        != "Binary hand status via square wave approximation: 1=closed, 0=open. Columns: [left, right]"
    ):
        raise EgoHandActiveSidecarError("raw/processed timing or hand-status semantics differ")
    downsampled_timestamps = raw_timestamps[::5]
    if (
        len(downsampled_timestamps) != final_length
        or not np.array_equal(
            downsampled_timestamps.astype(final_timestamps.dtype),
            final_timestamps,
        )
    ):
        raise EgoHandActiveSidecarError(
            "processed timestamps are not raw 100 Hz indices [::5] after the stored dtype cast"
        )
    timestamp_rounding_residual = np.abs(
        final_timestamps.astype(np.int64)
        - downsampled_timestamps.astype(np.int64)
    )
    max_timestamp_rounding_residual_ns = int(timestamp_rounding_residual.max())
    max_allowed_rounding_residual_ns = int(
        np.ceil(np.spacing(final_timestamps).max() / 2.0)
    )
    if max_timestamp_rounding_residual_ns > max_allowed_rounding_residual_ns:
        raise EgoHandActiveSidecarError(
            "processed timestamp rounding exceeds half a stored float64 ULP"
        )
    if final_length > 1 and not np.all(np.diff(final_timestamps) > 0):
        raise EgoHandActiveSidecarError("processed timestamps are not strictly increasing")
    current = np.stack((left[::5], right[::5]), axis=1).astype(np.bool_, copy=False)
    return current, hand_status.astype(np.float32, copy=False), {
        "raw_frames": raw_length,
        "raw_active_frames": [int(np.sum(left == 1)), int(np.sum(right == 1))],
        "raw_active_any": [bool(np.any(left == 1)), bool(np.any(right == 1))],
        "raw_active_indices": [
            np.flatnonzero(left == 1).tolist(),
            np.flatnonzero(right == 1).tolist(),
        ],
        "processed_frames": final_length,
        "raw_interval_seconds": raw_interval,
        "processed_interval_seconds": final_interval,
        "downsample_rate": downsample_rate,
        "processed_timestamp_storage_dtype": str(final_timestamps.dtype),
        "max_timestamp_rounding_residual_ns": max_timestamp_rounding_residual_ns,
        "max_allowed_timestamp_rounding_residual_ns": (
            max_allowed_rounding_residual_ns
        ),
    }


def _aggregate_statistics(
    per_episode: dict[int, tuple[np.ndarray, np.ndarray, dict]],
) -> dict:
    processed_frames = sum(len(active) for active, _, _ in per_episode.values())
    raw_frames = sum(int(audit["raw_frames"]) for _, _, audit in per_episode.values())
    max_timestamp_rounding_residual_ns = max(
        int(audit["max_timestamp_rounding_residual_ns"])
        for _, _, audit in per_episode.values()
    )
    max_allowed_timestamp_rounding_residual_ns = max(
        int(audit["max_allowed_timestamp_rounding_residual_ns"])
        for _, _, audit in per_episode.values()
    )
    raw_active_frames = np.zeros(2, dtype=np.int64)
    active_frames = np.zeros(2, dtype=np.int64)
    future_target_active = np.zeros(2, dtype=np.int64)
    closed_frames = np.zeros(2, dtype=np.int64)
    closed_inactive = np.zeros(2, dtype=np.int64)
    transitions = np.zeros(2, dtype=np.int64)
    transitions_active_pair = np.zeros(2, dtype=np.int64)
    zero_raw_active = [[], []]
    zero_current_active = [[], []]
    zero_future_target_supervision = [[], []]
    active_lost_by_stride = [[], []]
    both_active = 0
    neither_active = 0
    for episode, (active, hand_status, audit) in sorted(per_episode.items()):
        raw_active_frames += np.asarray(audit["raw_active_frames"], dtype=np.int64)
        active_frames += active.sum(axis=0, dtype=np.int64)
        future = build_hand_active_arrays(active)["ego_hand_active_future"]
        future_target_active += future.sum(axis=(0, 1), dtype=np.int64)
        closed = hand_status == 1.0
        closed_frames += closed.sum(axis=0, dtype=np.int64)
        closed_inactive += (closed & ~active).sum(axis=0, dtype=np.int64)
        both_active += int(np.all(active, axis=1).sum())
        neither_active += int((~np.any(active, axis=1)).sum())
        for side in range(2):
            changed = np.flatnonzero(np.diff(hand_status[:, side]) != 0.0) + 1
            transitions[side] += len(changed)
            transitions_active_pair[side] += sum(
                bool(active[index - 1, side] and active[index, side])
                for index in changed
            )
            if not bool(audit["raw_active_any"][side]):
                zero_raw_active[side].append(episode)
            if not bool(active[:, side].any()):
                zero_current_active[side].append(episode)
                raw_indices = audit["raw_active_indices"][side]
                if raw_indices:
                    active_lost_by_stride[side].append(
                        {
                            "episode_id": episode,
                            "raw_active_indices": raw_indices,
                            "raw_index_modulo_stride": [
                                int(index % audit["downsample_rate"])
                                for index in raw_indices
                            ],
                            "reason": "no_raw_active_index_is_divisible_by_stride_5",
                        }
                    )
            if not bool(future[..., side].any()):
                zero_future_target_supervision[side].append(episode)
    hands = {}
    for side, name in enumerate(EGO_HAND_ACTIVE_FIELD_ORDER):
        hands[name] = {
            "raw_100hz_active_frames": int(raw_active_frames[side]),
            "raw_100hz_active_fraction": float(raw_active_frames[side] / raw_frames),
            "active_frames": int(active_frames[side]),
            "inactive_frames": int(processed_frames - active_frames[side]),
            "active_fraction": float(active_frames[side] / processed_frames),
            "closed_proxy_frames": int(closed_frames[side]),
            "closed_proxy_while_inactive_frames": int(closed_inactive[side]),
            "closed_proxy_while_inactive_fraction_of_closed": float(
                closed_inactive[side] / max(1, closed_frames[side])
            ),
            "proxy_transition_count": int(transitions[side]),
            "proxy_transition_with_both_adjacent_frames_active_count": int(
                transitions_active_pair[side]
            ),
            "future_target_active_samples": int(future_target_active[side]),
            "zero_raw_100hz_active_episode_ids": zero_raw_active[side],
            "zero_current_20hz_active_episode_ids": zero_current_active[side],
            "zero_future_target_supervision_episode_ids": (
                zero_future_target_supervision[side]
            ),
            "raw_active_but_absent_after_exact_stride5": (
                active_lost_by_stride[side]
            ),
        }
    return {
        "episode_count": len(per_episode),
        "raw_100hz_frames": raw_frames,
        "processed_frames": processed_frames,
        "timestamp_alignment": {
            "source_index_rule": "raw_100hz_indices_[::5]",
            "stored_dtype_cast_exact": True,
            "processed_storage_dtype": "float64",
            "max_integer_rounding_residual_ns": (
                max_timestamp_rounding_residual_ns
            ),
            "max_allowed_half_ulp_rounding_residual_ns": (
                max_allowed_timestamp_rounding_residual_ns
            ),
        },
        "supervision_domain_definitions": {
            "raw_100hz_active": (
                "source left/right_hand_active before temporal downsampling"
            ),
            "current_20hz_active": (
                "raw tracker validity at the exact processed indices [::5]"
            ),
            "future_target_supervision": (
                "current_20hz_active at target frame f+offset for offsets 1..48; "
                "targets beyond the episode boundary are false"
            ),
        },
        "both_hands_active_frames": both_active,
        "both_hands_active_fraction": float(both_active / processed_frames),
        "neither_hand_active_frames": neither_active,
        "neither_hand_active_fraction": float(neither_active / processed_frames),
        "hands": hands,
    }


def load_and_validate_hand_active_bundle(
    sidecar_dir: str | Path,
    *,
    dataset_identity: dict,
    raw_hdf5_dir: str | Path,
    final_hdf5_dir: str | Path,
) -> dict:
    """Validate every source byte, NPZ byte, alignment, and aggregate statistic."""

    sidecar_root = Path(os.path.abspath(os.path.expanduser(str(sidecar_dir))))
    raw_root = Path(os.path.abspath(os.path.expanduser(str(raw_hdf5_dir))))
    final_root = Path(os.path.abspath(os.path.expanduser(str(final_hdf5_dir))))
    try:
        meta_bytes, meta_digest, meta_stat = read_verified_bytes(sidecar_root / "meta.json")
        meta = strict_json_loads(meta_bytes, name="Ego hand-active sidecar meta")
    except (OSError, ValueError) as exc:
        raise EgoHandActiveSidecarError("invalid hand-active sidecar meta") from exc
    if not isinstance(meta, dict):
        raise EgoHandActiveSidecarError("hand-active meta must be an object")
    expected_lengths = {
        str(key): int(value)
        for key, value in dataset_identity["episode_lengths"].items()
    }
    fixed = {
        "schema_version": EGO_HAND_ACTIVE_SCHEMA_VERSION,
        "builder_identity": EGO_HAND_ACTIVE_BUILDER,
        "dataset_id": dataset_identity["dataset_id"],
        "dataset_fingerprint_sha256": dataset_identity["dataset_fingerprint_sha256"],
        "source_receipt_sha256": dataset_identity["source_receipt_sha256"],
        "frame_clock": EGO_STAGE1_FRAME_CLOCK,
        "target_fps": EGO_STAGE1_FPS,
        "horizon_frames": EGO_HAND_ACTIVE_HORIZON,
        "action_start_offset": EGO_HAND_ACTIVE_START_OFFSET,
        "field_order": list(EGO_HAND_ACTIVE_FIELD_ORDER),
        "hand_active_semantics": EGO_HAND_ACTIVE_SEMANTICS,
        "episode_lengths": expected_lengths,
        "mask_contract": {
            "grip_action_value_loss": "active_at_target_frame_times_0.25",
            "future_edge_padding": "false_no_last_value_hold",
            "transition_auxiliary": "disabled_pair_mask_record_only",
            "transition_pair_mask": "active_at_target_minus_1_and_target",
            "wrist_action_dimensions_0_17": "unaffected",
            "grip_action_dimensions": [18, 19],
        },
        "physical_contact_truth": False,
    }
    for key, value in fixed.items():
        if meta.get(key) != value:
            raise EgoHandActiveSidecarError(f"hand-active meta binding differs: {key}")
    episode_hashes = meta.get("episode_npz_sha256")
    raw_hashes = meta.get("raw_hdf5_sha256")
    final_hashes = meta.get("final_hdf5_sha256")
    if not all(isinstance(value, dict) for value in (episode_hashes, raw_hashes, final_hashes)):
        raise EgoHandActiveSidecarError("hand-active source/NPZ receipts are missing")
    expected_names = {
        f"episode_{episode:06d}.npz" for episode in dataset_identity["episode_lengths"]
    }
    actual_names = {path.name for path in sidecar_root.glob("episode_*.npz")}
    if set(episode_hashes) != expected_names or actual_names != expected_names:
        raise EgoHandActiveSidecarError("hand-active NPZ episode set differs")
    raw_files = _episode_files(raw_root)
    final_files = _episode_files(final_root)
    expected_episodes = set(dataset_identity["episode_lengths"])
    if set(raw_files) != expected_episodes or set(final_files) != expected_episodes:
        raise EgoHandActiveSidecarError("hand-active raw/final HDF5 episode set differs")
    if set(raw_hashes) != {path.name for path in raw_files.values()} or set(final_hashes) != {
        path.name for path in final_files.values()
    }:
        raise EgoHandActiveSidecarError("hand-active HDF5 receipt names differ")
    per_episode: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}
    loaded_arrays: dict[int, dict[str, np.ndarray]] = {}
    for episode, frame_count in dataset_identity["episode_lengths"].items():
        raw_path = raw_files[episode]
        final_path = final_files[episode]
        if (
            raw_hashes[raw_path.name] != sha256_file(raw_path)
            or final_hashes[final_path.name] != sha256_file(final_path)
        ):
            raise EgoHandActiveSidecarError("hand-active HDF5 source bytes differ")
        current, hand_status, source_audit = read_aligned_hand_active(
            raw_path, final_path
        )
        if len(current) != frame_count:
            raise EgoHandActiveSidecarError("hand-active source length differs from dataset")
        name = f"episode_{episode:06d}.npz"
        digest = episode_hashes[name]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise EgoHandActiveSidecarError("invalid hand-active NPZ SHA")
        arrays = load_hand_active_npz(
            sidecar_root / name,
            expected_sha256=digest,
            frame_count=frame_count,
        )
        if not np.array_equal(arrays["ego_hand_active_current"], current):
            raise EgoHandActiveSidecarError("hand-active NPZ differs from raw tracker validity")
        per_episode[episode] = (current, hand_status, source_audit)
        loaded_arrays[episode] = arrays
    if canonical_sha256(episode_hashes) != meta.get("sidecar_bundle_sha256"):
        raise EgoHandActiveSidecarError("hand-active NPZ bundle SHA differs")
    if canonical_sha256(raw_hashes) != meta.get("raw_hdf5_bundle_sha256"):
        raise EgoHandActiveSidecarError("hand-active raw HDF5 bundle SHA differs")
    if canonical_sha256(final_hashes) != meta.get("final_hdf5_bundle_sha256"):
        raise EgoHandActiveSidecarError("hand-active final HDF5 bundle SHA differs")
    statistics = _aggregate_statistics(per_episode)
    if meta.get("statistics") != statistics:
        raise EgoHandActiveSidecarError("hand-active aggregate statistics differ")
    return {
        "root": str(sidecar_root),
        "meta": meta,
        "meta_sha256": meta_digest,
        "meta_runtime_stat_record_only": meta_stat,
        "episode_npz_sha256": dict(episode_hashes),
        "sidecar_bundle_sha256": meta["sidecar_bundle_sha256"],
        "raw_hdf5_bundle_sha256": meta["raw_hdf5_bundle_sha256"],
        "final_hdf5_bundle_sha256": meta["final_hdf5_bundle_sha256"],
        "statistics": statistics,
        "arrays": loaded_arrays,
    }

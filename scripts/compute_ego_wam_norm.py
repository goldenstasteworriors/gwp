#!/usr/bin/env python3
"""Compute fingerprint-bound 32-D Stage1 stats from processed Ego 20-D actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.ego_action_schema import (  # noqa: E402
    CANONICAL_OPEN_FRACTION_SEMANTICS,
    EGO_CANONICAL_ACTION_SCHEMA_ID,
    EGO_HAND_STATUS_SEMANTICS,
    EGO_WAM_ACTION_DIM,
    convert_hand_status_semantics,
    dual_eef_delta_to_action,
    dual_eef_pose_to_state,
)
from world_action_model.ego_dataset_fingerprint import (  # noqa: E402
    derive_ego_dataset_identity,
)
from world_action_model.ego_hand_active_sidecar import (  # noqa: E402
    EGO_HAND_ACTIVE_SEMANTICS,
    canonical_sha256,
    load_and_validate_hand_active_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", required=True, type=Path)
    parser.add_argument(
        "--raw-hdf5-dir",
        required=True,
        type=Path,
        help="100 Hz source containing left/right_hand_active",
    )
    parser.add_argument(
        "--hand-active-sidecar-dir",
        required=True,
        type=Path,
        help="verified 20 Hz tracker-validity sidecar bundle",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="authoritative processed LeRobot root used to derive the portable dataset ID",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument(
        "--action-start-offset",
        type=int,
        default=1,
        help="processed delta[i] maps pose i-1 to i, so WAM future chunks start at 1",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ego-grip-proxy-weight", type=float, default=0.25)
    parser.add_argument(
        "--ablation-id",
        help="required non-production identity when grip proxy weight is not 0.25",
    )
    return parser.parse_args()


def pad_feature(value: np.ndarray, model_dim: int) -> np.ndarray:
    if value.shape[-1] > model_dim:
        raise ValueError(
            f"feature dimension {value.shape[-1]} exceeds model_dim={model_dim}"
        )
    result = np.zeros(value.shape[:-1] + (model_dim,), dtype=np.float32)
    result[..., : value.shape[-1]] = value
    return result


def feature_stats(
    value: np.ndarray,
    validity: np.ndarray,
    *,
    source_dim: int,
) -> tuple[dict[str, list[float]], list[int]]:
    if value.shape != validity.shape or validity.dtype != np.bool_:
        raise ValueError("normalization value/validity arrays differ")
    names = ("mean", "std", "min", "max", "q01", "q99")
    result = {name: [] for name in names}
    counts = []
    for dimension in range(value.shape[1]):
        if dimension >= source_dim:
            selected = np.zeros(0, dtype=np.float32)
        else:
            selected = value[validity[:, dimension], dimension]
            if len(selected) == 0:
                raise ValueError(
                    f"normalization dimension {dimension} has no valid source samples"
                )
        counts.append(int(len(selected)))
        if len(selected) == 0:
            for name in names:
                result[name].append(0.0)
            continue
        result["mean"].append(float(selected.mean(dtype=np.float64)))
        result["std"].append(float(selected.std(dtype=np.float64)))
        result["min"].append(float(selected.min()))
        result["max"].append(float(selected.max()))
        result["q01"].append(float(np.quantile(selected, 0.01)))
        result["q99"].append(float(np.quantile(selected, 0.99)))
    return result, counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.expanduser().resolve()
    identity = derive_ego_dataset_identity(args.dataset_root)
    hand_active = load_and_validate_hand_active_bundle(
        args.hand_active_sidecar_dir,
        dataset_identity=identity,
        raw_hdf5_dir=args.raw_hdf5_dir,
        final_hdf5_dir=hdf5_dir,
    )
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    if args.model_dim < EGO_WAM_ACTION_DIM:
        raise ValueError(
            f"model_dim must be at least {EGO_WAM_ACTION_DIM}, got {args.model_dim}"
        )
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if args.action_start_offset < 0:
        raise ValueError("action_start_offset must be non-negative")
    if not np.isfinite(args.ego_grip_proxy_weight) or args.ego_grip_proxy_weight <= 0:
        raise ValueError("ego-grip-proxy-weight must be finite and positive")
    if args.ego_grip_proxy_weight != 0.25:
        if not args.ablation_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", args.ablation_id):
            raise ValueError("non-0.25 grip weight requires a portable --ablation-id")
        action_schema_id = f"{EGO_CANONICAL_ACTION_SCHEMA_ID}:ablation:{args.ablation_id}"
    else:
        if args.ablation_id:
            raise ValueError("--ablation-id is only valid for a non-production grip weight")
        action_schema_id = EGO_CANONICAL_ACTION_SCHEMA_ID

    state_chunks = []
    action_chunks = []
    state_validity_chunks = []
    action_validity_chunks = []
    episode_lengths = {}
    hdf5_hashes = {}
    paths = sorted(
        hdf5_dir.glob("episode_*.hdf5"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"no episode_*.hdf5 under {hdf5_dir}")
    offsets = np.arange(
        args.action_start_offset,
        args.action_start_offset + args.action_horizon,
        dtype=np.int64,
    )
    for path in paths:
        with h5py.File(path, "r") as handle:
            pose = np.asarray(handle["action_eef"], dtype=np.float32)
            delta = np.asarray(handle["action_delta_eef"], dtype=np.float32)
            source_hand_status = np.asarray(handle["hand_status"], dtype=np.float32)
        if pose.shape != (len(pose), 14) or delta.shape != (len(delta), 12) or source_hand_status.shape != (len(source_hand_status), 2):
            raise ValueError(f"invalid processed action shapes in {path}")
        hand_status = convert_hand_status_semantics(
            source_hand_status,
            source_semantics=EGO_HAND_STATUS_SEMANTICS,
            target_semantics=CANONICAL_OPEN_FRACTION_SEMANTICS,
        )
        if not (len(pose) == len(delta) == len(hand_status)):
            raise ValueError(
                f"inconsistent frame counts in {path}: "
                f"pose={len(pose)}, delta={len(delta)}, hand={len(hand_status)}"
            )
        state = dual_eef_pose_to_state(pose, hand_status)
        action = dual_eef_delta_to_action(delta, hand_status)
        active = hand_active["arrays"][int(path.stem.split("_")[-1])][
            "ego_hand_active_current"
        ]
        if active.shape != source_hand_status.shape or active.dtype != np.bool_:
            raise ValueError(f"invalid hand-active source shape in {path}")
        raw_indices = np.arange(len(action), dtype=np.int64)[:, None] + offsets[None, :]
        valid = raw_indices < len(action)
        indices = np.minimum(raw_indices, len(action) - 1)
        state_chunks.append(pad_feature(state, args.model_dim))
        action_chunks.append(pad_feature(action[indices[valid]], args.model_dim))
        state_validity = np.ones((len(state), args.model_dim), dtype=np.bool_)
        state_validity[:, 18:20] = active
        state_validity[:, EGO_WAM_ACTION_DIM:] = False
        action_validity = np.ones((int(valid.sum()), args.model_dim), dtype=np.bool_)
        action_validity[:, 18:20] = active[indices[valid]]
        action_validity[:, EGO_WAM_ACTION_DIM:] = False
        state_validity_chunks.append(state_validity)
        action_validity_chunks.append(action_validity)
        episode_lengths[str(int(path.stem.split("_")[-1]))] = len(state)
        hdf5_hashes[path.name] = sha256_file(path)

    expected_lengths = {str(key): value for key, value in identity["episode_lengths"].items()}
    if episode_lengths != expected_lengths:
        raise ValueError(
            "final_hdf5 episodes/lengths differ from authoritative LeRobot episode metadata"
        )

    states = np.concatenate(state_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    state_validity = np.concatenate(state_validity_chunks, axis=0)
    action_validity = np.concatenate(action_validity_chunks, axis=0)
    state_stats, state_valid_counts = feature_stats(
        states,
        state_validity,
        source_dim=EGO_WAM_ACTION_DIM,
    )
    action_stats, action_valid_counts = feature_stats(
        actions,
        action_validity,
        source_dim=EGO_WAM_ACTION_DIM,
    )
    hdf5_bundle_sha256 = hashlib.sha256(
        json.dumps(hdf5_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "norm_stats": {
            "observation.state": state_stats,
            "action": action_stats,
        },
        "metadata": {
            "norm_contract_version": "ego-stage1-norm/v3",
            "action_schema_id": action_schema_id,
            "source_dim": EGO_WAM_ACTION_DIM,
            "model_dim": args.model_dim,
            "action_horizon": args.action_horizon,
            "action_start_offset": args.action_start_offset,
            "episodes": len(paths),
            "state_samples": int(len(states)),
            "action_samples": int(len(actions)),
            "episode_lengths": episode_lengths,
            "dataset_id": identity["dataset_id"],
            "dataset_fingerprint_version": identity["fingerprint_version"],
            "dataset_info_identity_fields": identity["info_identity_fields"],
            "dataset_episode_identity_fields": identity["episode_identity_fields"],
            "dataset_episode_sort_key": identity["episode_sort_key"],
            "dataset_task_sort_rule": identity["task_sort_rule"],
            "dataset_canonical_json": identity["canonical_json"],
            "dataset_fingerprint_sha256": identity["dataset_fingerprint_sha256"],
            "info_identity_sha256": identity["info_identity_sha256"],
            "episode_metadata_sha256": identity["episode_metadata_sha256"],
            "dataset_portable_info_ignored_fields": identity[
                "portable_info_ignored_fields"
            ],
            "dataset_training_source_roots": identity["training_source_roots"],
            "dataset_source_manifest": identity["source_manifest"],
            "dataset_source_manifest_sha256": identity["source_manifest_sha256"],
            "dataset_source_receipt": identity["source_receipt"],
            "dataset_source_receipt_sha256": identity["source_receipt_sha256"],
            "fps": identity["fps"],
            "frame_clock": identity["frame_clock"],
            "source_hand_status_semantics": EGO_HAND_STATUS_SEMANTICS,
            "target_hand_status_semantics": CANONICAL_OPEN_FRACTION_SEMANTICS,
            "hand_status_conversion": "target=1-source",
            "hand_active_semantics": EGO_HAND_ACTIVE_SEMANTICS,
            "hand_active_meta_sha256": hand_active["meta_sha256"],
            "hand_active_sidecar_bundle_sha256": hand_active[
                "sidecar_bundle_sha256"
            ],
            "hand_active_raw_hdf5_bundle_sha256": hand_active[
                "raw_hdf5_bundle_sha256"
            ],
            "hand_active_statistics": hand_active["statistics"],
            "hand_active_statistics_sha256": canonical_sha256(
                hand_active["statistics"]
            ),
            "normalization_valid_sample_count": {
                "observation.state": state_valid_counts,
                "action": action_valid_counts,
            },
            "grip_normalization_contract": (
                "dimensions_18_19_use_only_corresponding_active_tracker_frames;"
                "inactive_proxy_values_excluded"
            ),
            "padding_contract": "right_zero_pad_20_to_32",
            "action_dim_mask": [True] * EGO_WAM_ACTION_DIM
            + [False] * (args.model_dim - EGO_WAM_ACTION_DIM),
            "action_time_mask": "normalization_excludes_clamped_padding;training_masks_clamped_padding_false",
            "action_loss_weight": {
                "dimensions_0_17": 1.0,
                "grip_proxy_dimensions_18_19": args.ego_grip_proxy_weight,
                "padding_dimensions_20_31": 0.0,
            },
            "hdf5_source_role": "processed_final_hdf5_20hz",
            "hdf5_dir_record_only_not_identity": str(hdf5_dir),
            "episode_hdf5_sha256": hdf5_hashes,
            "hdf5_bundle_sha256": hdf5_bundle_sha256,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(paths),
                "state_samples": len(states),
                "action_samples": len(actions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute 32-D WAM stats for processed EgoHumanoid dual-wrist data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.ego_action_schema import (  # noqa: E402
    EGO_WAM_ACTION_DIM,
    dual_eef_delta_to_action,
    dual_eef_pose_to_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", required=True, type=Path)
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
    return parser.parse_args()


def pad_feature(value: np.ndarray, model_dim: int) -> np.ndarray:
    if value.shape[-1] > model_dim:
        raise ValueError(
            f"feature dimension {value.shape[-1]} exceeds model_dim={model_dim}"
        )
    result = np.zeros(value.shape[:-1] + (model_dim,), dtype=np.float32)
    result[..., : value.shape[-1]] = value
    return result


def feature_stats(value: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": value.mean(axis=0, dtype=np.float64).tolist(),
        "std": value.std(axis=0, dtype=np.float64).tolist(),
        "min": value.min(axis=0).astype(np.float64).tolist(),
        "max": value.max(axis=0).astype(np.float64).tolist(),
        "q01": np.quantile(value, 0.01, axis=0).tolist(),
        "q99": np.quantile(value, 0.99, axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.expanduser().resolve()
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

    state_chunks = []
    action_chunks = []
    episode_lengths = {}
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
            hand_status = np.asarray(handle["hand_status"], dtype=np.float32)
        if not (len(pose) == len(delta) == len(hand_status)):
            raise ValueError(
                f"inconsistent frame counts in {path}: "
                f"pose={len(pose)}, delta={len(delta)}, hand={len(hand_status)}"
            )
        state = dual_eef_pose_to_state(pose, hand_status)
        action = dual_eef_delta_to_action(delta, hand_status)
        indices = np.minimum(
            np.arange(len(action), dtype=np.int64)[:, None] + offsets[None, :],
            len(action) - 1,
        )
        state_chunks.append(pad_feature(state, args.model_dim))
        action_chunks.append(pad_feature(action[indices], args.model_dim).reshape(-1, args.model_dim))
        episode_lengths[str(int(path.stem.split("_")[-1]))] = len(state)

    states = np.concatenate(state_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    payload = {
        "norm_stats": {
            "observation.state": feature_stats(states),
            "action": feature_stats(actions),
        },
        "metadata": {
            "schema": "ego_dual_eef_pose9x2_plus_grip_proxy_v1",
            "source_dim": EGO_WAM_ACTION_DIM,
            "model_dim": args.model_dim,
            "action_horizon": args.action_horizon,
            "action_start_offset": args.action_start_offset,
            "episodes": len(paths),
            "state_samples": int(len(states)),
            "action_samples": int(len(actions)),
            "episode_lengths": episode_lengths,
            "hdf5_dir": str(hdf5_dir),
            "hand_status_semantics": "binary closure proxy, not per-finger contact",
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

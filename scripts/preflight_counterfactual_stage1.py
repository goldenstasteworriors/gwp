#!/usr/bin/env python3
"""Audit stage-one counterfactual caches, split integrity, and disk capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_SCHEMA = "libero-counterfactual-split-v1"
CACHE_SCHEMA = "libero-counterfactual-robot-mask-v2"
EXPECTED_IMAGE_ORIENTATION = "lerobot_libero_visualization_flip_height_width"
EXPECTED_HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--heldout-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matched-output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--transient-gib", type=float, default=6.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_episode_file(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes = [int(token) for token in path.read_text(encoding="utf-8").split()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError(f"Episode list must be non-empty and unique: {path}")
    return episodes


def require_shape(name: str, value: np.ndarray, shape: tuple[int, ...], path: Path) -> None:
    if value.shape != shape:
        raise ValueError(f"{path}: {name} shape {value.shape}, expected {shape}")


def audit_cache(
    root: Path,
    expected_episodes: list[int],
    selection_file: Path,
    checkpoint: Path,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"Unexpected cache schema in {manifest_path}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError(f"Counterfactual cache must not contain RGB targets: {manifest_path}")
    if manifest.get("image_orientation") != EXPECTED_IMAGE_ORIENTATION:
        raise ValueError(f"Cache/training image orientation mismatch: {manifest_path}")
    if [int(value) for value in manifest.get("horizons", [])] != EXPECTED_HORIZONS:
        raise ValueError(f"Unexpected horizons in {manifest_path}")
    if Path(manifest["dataset_root"]).resolve() != Path(split_manifest["dataset_root"]).resolve():
        raise ValueError(f"Cache/split dataset root mismatch: {manifest_path}")
    if manifest.get("dataset_info_sha256") != split_manifest.get("dataset_info_sha256"):
        raise ValueError(f"Cache/split dataset hash mismatch: {manifest_path}")
    if Path(str(manifest.get("policy_id"))).resolve() != checkpoint.resolve():
        raise ValueError(f"Cache policy_id does not match checkpoint: {manifest_path}")
    if manifest.get("episode_selection_sha256") != sha256_file(selection_file):
        raise ValueError(f"Episode selection hash mismatch: {manifest_path}")

    records = list(manifest.get("episodes", []))
    got_episodes = [int(record["episode_index"]) for record in records]
    if got_episodes != expected_episodes:
        raise ValueError(f"Cache episode order/content mismatch: {manifest_path}")
    if int(manifest.get("num_episodes", -1)) != len(records):
        raise ValueError(f"Cache episode count mismatch: {manifest_path}")

    robot_pixels_by_horizon = np.zeros(5, dtype=np.int64)
    gripper_pixels_by_horizon = np.zeros(5, dtype=np.int64)
    clipped_values = 0
    action_values = 0
    for record in records:
        if int(record.get("source_frame_index", -1)) != 0:
            raise ValueError(f"Nonzero cache source frame: {record}")
        if int(record.get("action_chunk_index", -1)) != 0:
            raise ValueError(f"Nonzero action chunk index: {record}")
        prediction_path = Path(record["prediction_path"])
        cache_path = root / record["cache_file"]
        if sha256_file(prediction_path) != record["prediction_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        if sha256_file(cache_path) != record["cache_sha256"]:
            raise ValueError(f"Cache hash mismatch: {cache_path}")

        with np.load(cache_path, allow_pickle=False) as arrays:
            required = {
                "horizons",
                "raw_actions",
                "executed_actions",
                "robot_mask",
                "gripper_mask",
                "qpos",
                "eef_position",
                "eef_rotation",
                "camera_position",
                "camera_rotation",
                "camera_fovy",
            }
            missing = sorted(required - set(arrays.files))
            if missing:
                raise ValueError(f"{cache_path}: missing arrays {missing}")
            horizons = arrays["horizons"]
            raw_actions = arrays["raw_actions"]
            executed_actions = arrays["executed_actions"]
            robot_mask = arrays["robot_mask"]
            gripper_mask = arrays["gripper_mask"]
            require_shape("horizons", horizons, (5,), cache_path)
            require_shape("raw_actions", raw_actions, (48, 7), cache_path)
            require_shape("executed_actions", executed_actions, (48, 7), cache_path)
            require_shape("robot_mask", robot_mask, (5, 256, 256), cache_path)
            require_shape("gripper_mask", gripper_mask, (5, 256, 256), cache_path)
            if horizons.tolist() != EXPECTED_HORIZONS:
                raise ValueError(f"{cache_path}: wrong horizons")
            if not np.array_equal(executed_actions, np.clip(raw_actions, -1.0, 1.0)):
                raise ValueError(f"{cache_path}: executed action is not clipped raw action")
            if not np.logical_or(robot_mask == 0, robot_mask == 1).all():
                raise ValueError(f"{cache_path}: robot mask is not binary")
            if not np.logical_or(gripper_mask == 0, gripper_mask == 1).all():
                raise ValueError(f"{cache_path}: gripper mask is not binary")
            if np.logical_and(gripper_mask > 0, robot_mask == 0).any():
                raise ValueError(f"{cache_path}: gripper mask is not a subset of robot mask")
            for key in required - {"horizons", "robot_mask", "gripper_mask"}:
                if not np.isfinite(arrays[key]).all():
                    raise ValueError(f"{cache_path}: non-finite values in {key}")
            robot_pixels = robot_mask.sum(axis=(-2, -1), dtype=np.int64)
            if robot_pixels.tolist() != [int(value) for value in record["robot_pixels"]]:
                raise ValueError(f"{cache_path}: robot pixel summary mismatch")
            robot_pixels_by_horizon += robot_pixels
            gripper_pixels_by_horizon += gripper_mask.sum(axis=(-2, -1), dtype=np.int64)
            clipped_values += int(np.count_nonzero(raw_actions != executed_actions))
            action_values += int(raw_actions.size)

    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "num_episodes": len(records),
        "robot_pixels_mean": (robot_pixels_by_horizon / len(records)).tolist(),
        "gripper_pixels_mean": (gripper_pixels_by_horizon / len(records)).tolist(),
        "action_clipped_fraction": float(clipped_values / action_values),
        "metadata": {
            key: manifest.get(key)
            for key in (
                "suite",
                "policy_id",
                "camera_name",
                "image_orientation",
                "image_size",
                "num_steps_wait",
            )
        },
    }


def checkpoint_weight_bytes(checkpoint: Path) -> int:
    files = sorted(checkpoint.glob("*.bin")) + sorted(checkpoint.glob("*.safetensors"))
    total = sum(path.stat().st_size for path in files if path.is_file())
    if total <= 0:
        raise FileNotFoundError(f"No model weights in checkpoint: {checkpoint}")
    return total


def main() -> None:
    args = parse_args()
    if args.reserve_gib <= 0 or args.transient_gib <= 0:
        raise ValueError("--reserve-gib and --transient-gib must be positive")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite preflight report: {args.report}")
    if not args.project_root.is_dir() or not (args.project_root / ".git").exists():
        raise FileNotFoundError(f"Not a Git project: {args.project_root}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    for output in (args.matched_output, args.full_output):
        if output.exists():
            raise FileExistsError(f"Refusing existing stage-one output: {output}")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"Stage-one output parent does not exist: {output.parent}")
    if os.stat(args.matched_output.parent).st_dev != os.stat(args.full_output.parent).st_dev:
        raise ValueError("Matched and full outputs must use the same mounted filesystem")
    output_device = os.stat(args.matched_output.parent).st_dev
    for source in (args.checkpoint, args.train_cache, args.heldout_cache):
        if os.stat(source).st_dev != output_device:
            raise ValueError(f"Large stage-one source/output crosses filesystems: {source}")

    split_manifest_path = args.split_dir / "manifest.json"
    split_manifest = load_json(split_manifest_path)
    if split_manifest.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError(f"Unexpected split schema: {split_manifest_path}")
    train_file = args.split_dir / split_manifest["train_episodes_file"]
    heldout_file = args.split_dir / split_manifest["heldout_episodes_file"]
    train_episodes = read_episode_file(train_file)
    heldout_episodes = read_episode_file(heldout_file)
    if train_episodes != [int(value) for value in split_manifest["train_episodes"]]:
        raise ValueError("Train episode file does not match split manifest")
    if heldout_episodes != [int(value) for value in split_manifest["heldout_episodes"]]:
        raise ValueError("Held-out episode file does not match split manifest")
    overlap = sorted(set(train_episodes) & set(heldout_episodes))
    if overlap:
        raise ValueError(f"Train/held-out split overlap: {overlap}")
    selected_lengths = [
        int(split_manifest["episode_lengths"][str(episode)])
        for episode in train_episodes + heldout_episodes
    ]
    if min(selected_lengths) < 49:
        raise ValueError("Every selected episode must cover frame 0 plus a 48-step chunk")

    train_audit = audit_cache(
        args.train_cache, train_episodes, train_file, args.checkpoint, split_manifest
    )
    heldout_audit = audit_cache(
        args.heldout_cache, heldout_episodes, heldout_file, args.checkpoint, split_manifest
    )
    if train_audit["metadata"] != heldout_audit["metadata"]:
        raise ValueError("Train and held-out cache metadata differ")

    source_weight_bytes = checkpoint_weight_bytes(args.checkpoint)
    # Two BF16 checkpoints are approximately one FP32 source checkpoint.  Keep
    # an explicit reserve for logs, temporary save overhead, and filesystem health.
    reserve_bytes = int(args.reserve_gib * 1024**3)
    transient_bytes = int(args.transient_gib * 1024**3)
    required_free_bytes = source_weight_bytes + reserve_bytes + transient_bytes
    free_bytes = shutil.disk_usage(args.matched_output.parent).free
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            f"Insufficient disk: free={free_bytes / 1024**3:.2f} GiB, "
            f"required={required_free_bytes / 1024**3:.2f} GiB"
        )

    git_commit = subprocess.check_output(
        ["git", "-C", str(args.project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_status = subprocess.check_output(
        ["git", "-C", str(args.project_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if tracked_status:
        raise RuntimeError(f"Tracked project files are dirty:\n{tracked_status}")

    report = {
        "status": "passed",
        "project_root": str(args.project_root.resolve()),
        "git_commit": git_commit,
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "num_train": len(train_episodes),
        "num_heldout": len(heldout_episodes),
        "episode_overlap": 0,
        "checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_weight_gib": source_weight_bytes / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "required_free_gib": required_free_bytes / 1024**3,
        "reserve_gib": args.reserve_gib,
        "transient_gib": args.transient_gib,
        "matched_output": str(args.matched_output.resolve()),
        "full_output": str(args.full_output.resolve()),
        "train_cache": train_audit,
        "heldout_cache": heldout_audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(args.report, flush=True)


if __name__ == "__main__":
    main()

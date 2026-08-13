#!/usr/bin/env python3
"""Audit replayed demonstration masks against the logged LeRobot video/state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from giga_datasets.datasets.lerobot_dataset import (
    FastLeRobotDataset,
    _LeRobotDatasetMetadata,
)


SCHEMA_VERSION = "libero-demonstration-robot-mask-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-shift-px", type=int, default=8)
    parser.add_argument("--position-p95-mm", type=float, default=5.0)
    parser.add_argument("--rotation-p95-deg", type=float, default=2.0)
    parser.add_argument("--gripper-p95-max-abs", type=float, default=0.003)
    parser.add_argument("--shift-p95-px", type=float, default=4.0)
    parser.add_argument("--full-rgb-mae-p95", type=float, default=10.0)
    parser.add_argument("--shifted-robot-rgb-mae-p95", type=float, default=15.0)
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
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def axis_angle_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = value / angle
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def masked_rgb_mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Cannot compute masked RGB MAE for an empty mask")
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return float(difference[mask].mean())


def shifted_masked_rgb_mae(
    rendered: np.ndarray,
    real: np.ndarray,
    mask: np.ndarray,
    dx: int,
    dy: int,
) -> float:
    height, width = mask.shape
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    valid_mask = mask[source_y0:source_y1, source_x0:source_x1]
    return masked_rgb_mae(
        rendered[source_y0:source_y1, source_x0:source_x1],
        real[target_y0:target_y1, target_x0:target_x1],
        valid_mask,
    )


def best_robot_shift(
    rendered: np.ndarray,
    real: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
) -> dict[str, float | int]:
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            candidates.append(
                (shifted_masked_rgb_mae(rendered, real, mask, dx, dy), dx, dy)
            )
    mae, dx, dy = min(candidates)
    return {
        "dx": dx,
        "dy": dy,
        "magnitude_px": float(np.hypot(dx, dy)),
        "masked_rgb_mae": mae,
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Registration summary contains no finite values")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.max_shift_px < 0:
        raise ValueError("--max-shift-px must be non-negative")
    thresholds = {
        "position_error_mm.p95": args.position_p95_mm,
        "rotation_error_deg.p95": args.rotation_p95_deg,
        "gripper_max_abs.p95": args.gripper_p95_max_abs,
        "best_shift_magnitude_px.p95": args.shift_p95_px,
        "full_rgb_mae.p95": args.full_rgb_mae_p95,
        "best_shift_robot_rgb_mae.p95": args.shifted_robot_rgb_mae_p95,
    }
    if any(value <= 0 for value in thresholds.values()):
        raise ValueError("All registration thresholds must be positive")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    cache_root = args.cache.resolve()
    manifest_path = cache_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected cache schema: {manifest.get('schema_version')!r}")
    if manifest.get("rendered_rgb_role") != (
        "registration_audit_only_not_a_probe_or_video_target"
    ):
        raise ValueError("Rendered RGB is not explicitly restricted to registration audit")
    horizons = [int(value) for value in manifest.get("horizons", [])]
    if horizons != [0, 12, 24, 36, 48]:
        raise ValueError(f"Unexpected horizons: {horizons}")
    records = list(manifest.get("episodes", []))
    if len(records) != int(manifest.get("num_episodes", -1)) or not records:
        raise ValueError("Invalid episode count in demonstration-mask manifest")

    dataset_root = Path(str(manifest["dataset_root"])).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if sha256_file(info_path) != manifest.get("dataset_info_sha256"):
        raise ValueError("Dataset metadata changed after mask generation")
    repo_id = dataset_root.name
    metadata = _LeRobotDatasetMetadata(repo_id, root=dataset_root)
    episodes = [int(record["episode_index"]) for record in records]
    delta_timestamps = {
        "observation.images.image": [horizon / metadata.fps for horizon in horizons],
        "observation.state": [horizon / metadata.fps for horizon in horizons],
    }
    dataset = FastLeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        episodes=episodes,
        video_backend="pyav",
    )
    dataset._ensure_hf_dataset_loaded()
    source_to_index: dict[tuple[int, int], int] = {}
    for index, (episode, frame) in enumerate(
        zip(dataset.hf_dataset["episode_index"], dataset.hf_dataset["frame_index"])
    ):
        source_to_index[(int(episode), int(frame))] = index

    rows: list[dict[str, Any]] = []
    for record in records:
        episode = int(record["episode_index"])
        source_frame = int(record["source_frame_index"])
        source_key = (episode, source_frame)
        if source_key not in source_to_index:
            raise KeyError(f"Dataset has no source sample {source_key}")
        cache_path = cache_root / str(record["cache_file"])
        if sha256_file(cache_path) != record.get("cache_sha256"):
            raise ValueError(f"Demonstration-mask cache changed: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            if not np.array_equal(cache["horizons"], np.asarray(horizons)):
                raise ValueError(f"Horizon mismatch in {cache_path}")
            rendered_rgb = cache["rendered_rgb"].copy()
            robot_mask = cache["robot_mask"].astype(bool)
            replay_position = cache["eef_position"].copy()
            replay_rotation = cache["eef_rotation"].copy()
            replay_gripper = cache["gripper_qpos"].copy()
        sample = dataset[source_to_index[source_key]]
        real_rgb = sample["observation.images.image"].detach().cpu().numpy()
        real_rgb = np.rint(np.transpose(real_rgb, (0, 2, 3, 1)) * 255.0).astype(
            np.uint8
        )
        real_state = sample["observation.state"].detach().cpu().numpy()
        if rendered_rgb.shape != real_rgb.shape:
            raise ValueError(
                f"Rendered/real RGB shape mismatch for episode {episode}: "
                f"{rendered_rgb.shape} vs {real_rgb.shape}"
            )
        if robot_mask.shape != rendered_rgb.shape[:3]:
            raise ValueError(f"Robot-mask/RGB shape mismatch in {cache_path}")
        if real_state.shape != (len(horizons), 8):
            raise ValueError(f"Unexpected real state shape: {real_state.shape}")

        for horizon_index, horizon in enumerate(horizons):
            mask = robot_mask[horizon_index]
            background = np.logical_not(mask)
            shift = best_robot_shift(
                rendered_rgb[horizon_index],
                real_rgb[horizon_index],
                mask,
                args.max_shift_px,
            )
            row = {
                "episode_index": episode,
                "task_index": int(record["task_index"]),
                "horizon": horizon,
                "position_error_mm": float(
                    1000.0
                    * np.linalg.norm(replay_position[horizon_index] - real_state[horizon_index, :3])
                ),
                "rotation_error_deg": rotation_error_degrees(
                    replay_rotation[horizon_index],
                    axis_angle_to_matrix(real_state[horizon_index, 3:6]),
                ),
                "gripper_max_abs": float(
                    np.max(
                        np.abs(
                            replay_gripper[horizon_index]
                            - real_state[horizon_index, 6:8]
                        )
                    )
                ),
                "full_rgb_mae": float(
                    np.abs(
                        rendered_rgb[horizon_index].astype(np.float32)
                        - real_rgb[horizon_index].astype(np.float32)
                    ).mean()
                ),
                "robot_rgb_mae": masked_rgb_mae(
                    rendered_rgb[horizon_index], real_rgb[horizon_index], mask
                ),
                "background_rgb_mae": masked_rgb_mae(
                    rendered_rgb[horizon_index], real_rgb[horizon_index], background
                ),
                "best_robot_shift": shift,
                "robot_pixels": int(mask.sum()),
            }
            rows.append(row)

    metric_values = {
        "position_error_mm": [row["position_error_mm"] for row in rows],
        "rotation_error_deg": [row["rotation_error_deg"] for row in rows],
        "gripper_max_abs": [row["gripper_max_abs"] for row in rows],
        "full_rgb_mae": [row["full_rgb_mae"] for row in rows],
        "robot_rgb_mae": [row["robot_rgb_mae"] for row in rows],
        "background_rgb_mae": [row["background_rgb_mae"] for row in rows],
        "best_shift_magnitude_px": [
            row["best_robot_shift"]["magnitude_px"] for row in rows
        ],
        "best_shift_robot_rgb_mae": [
            row["best_robot_shift"]["masked_rgb_mae"] for row in rows
        ],
    }
    aggregate = {key: summarize(values) for key, values in metric_values.items()}
    checks = {
        key: float(aggregate[key.split(".")[0]][key.split(".")[1]]) <= threshold
        for key, threshold in thresholds.items()
    }
    report = {
        "schema_version": "libero-demonstration-mask-registration-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "cache": str(cache_root),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_root": str(dataset_root),
        "dataset_info_sha256": sha256_file(info_path),
        "video_backend": "pyav",
        "max_shift_px": args.max_shift_px,
        "num_episodes": len(records),
        "num_frames": len(rows),
        "thresholds": thresholds,
        "checks": checks,
        "aggregate": aggregate,
        "frames": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(args.report, flush=True)


if __name__ == "__main__":
    main()

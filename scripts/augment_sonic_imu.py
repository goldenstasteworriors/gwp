#!/usr/bin/env python3
"""Add synchronized base IMU features and a jitter score to a SONIC dataset.

The output is a lightweight LeRobot v3 overlay: parquet and metadata are newly
written, while videos and cached T5 embeddings are symlinked from the input.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


IMU_NAMES = [
    "imu_qw",
    "imu_qx",
    "imu_qy",
    "imu_qz",
    "imu_gravity_x",
    "imu_gravity_y",
    "imu_gravity_z",
    "imu_gyro_x",
    "imu_gyro_y",
    "imu_gyro_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--jitter-window",
        type=int,
        default=2,
        help="Local half-window, in 50 Hz frames, used for peak angular speed.",
    )
    return parser.parse_args()


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def continuous_unit_quaternions(values: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Encountered a zero-norm root_orientation quaternion")
    quaternions /= norms
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    quaternions[quaternions[:, 0] < 0.0] *= -1.0
    return quaternions


def angular_velocity(quaternions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    velocity = np.zeros((len(quaternions), 3), dtype=np.float64)
    if len(quaternions) < 2:
        return velocity
    previous_inverse = quaternions[:-1].copy()
    previous_inverse[:, 1:] *= -1.0
    delta = quaternion_multiply(previous_inverse, quaternions[1:])
    delta[delta[:, 0] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(delta[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[:, 0], 0.0, 1.0))
    axis = np.divide(
        delta[:, 1:],
        vector_norm[:, None],
        out=np.zeros_like(delta[:, 1:]),
        where=vector_norm[:, None] > 1e-8,
    )
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive_dt = dt[dt > 1e-6]
    fallback_dt = float(np.median(positive_dt)) if positive_dt.size else 0.02
    dt = np.where(dt > 1e-6, dt, fallback_dt)
    velocity[1:] = axis * (angle / dt)[:, None]
    velocity[0] = velocity[1]
    return velocity


def local_peak(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("--jitter-window must be non-negative")
    result = np.asarray(values, dtype=np.float64).copy()
    for offset in range(1, radius + 1):
        result[offset:] = np.maximum(result[offset:], values[:-offset])
        result[:-offset] = np.maximum(result[:-offset], values[offset:])
    return result


def data_path(info: dict, episode_index: int) -> str:
    chunk_size = max(int(info.get("chunks_size", 1000)), 1)
    return str(info["data_path"]).format(
        chunk_index=episode_index // chunk_size,
        episode_chunk=episode_index // chunk_size,
        file_index=episode_index,
        episode_index=episode_index,
    )


def episode_rows(dataset: Path) -> list[dict]:
    paths = sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))
    paths += sorted((dataset / "meta" / "episodes").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode table under {dataset / 'meta' / 'episodes'}")
    rows = []
    for path in paths:
        rows.extend(pd.read_parquet(path).to_dict(orient="records"))
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def add_imu_to_episode(processed: pd.DataFrame, raw: pd.DataFrame, radius: int):
    if len(processed) != len(raw):
        raise ValueError(f"Processed/raw length mismatch: {len(processed)} != {len(raw)}")
    if not np.array_equal(processed["frame_index"], raw["frame_index"]):
        raise ValueError("Processed/raw frame_index mismatch")

    state = np.stack(processed["observation.state"]).astype(np.float32)
    quaternions = continuous_unit_quaternions(
        np.stack(raw["observation.root_orientation"])
    )
    gravity = np.stack(raw["observation.projected_gravity"]).astype(np.float64)
    gyro = angular_velocity(quaternions, raw["timestamp"].to_numpy())
    jitter = local_peak(np.linalg.norm(gyro, axis=1), radius)
    imu = np.concatenate([quaternions, gravity, gyro], axis=1).astype(np.float32)

    result = processed.copy()
    result["observation.state"] = list(np.concatenate([state, imu], axis=1))
    result["observation.camera_jitter_score"] = list(
        jitter.astype(np.float32)[:, None]
    )
    return result, jitter


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    source_info = json.loads((source / "meta" / "info.json").read_text())
    state_feature = source_info["features"]["observation.state"]
    state_names = list(state_feature.get("names") or [])
    state_dim = int(state_feature["shape"][0])
    if state_dim + len(IMU_NAMES) > 32:
        raise ValueError(
            f"State plus IMU exceeds the model's 32-D state input: {state_dim} + {len(IMU_NAMES)}"
        )
    if len(state_names) != state_dim:
        raise ValueError("Input state metadata must provide one name per dimension")

    (output / "data" / "chunk-000").mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    for name in ("videos", "t5_embedding"):
        target = source / name
        if not target.exists():
            raise FileNotFoundError(target)
        os.symlink(target, output / name, target_is_directory=True)

    scores = []
    sources_seen = set()
    for row in episode_rows(source):
        output_episode = int(row["episode_index"])
        raw_root = Path(str(row["source_dataset"])).resolve()
        raw_episode = int(row["source_episode_index"])
        raw_info = json.loads((raw_root / "meta" / "info.json").read_text())
        processed_path = source / data_path(source_info, output_episode)
        raw_path = raw_root / data_path(raw_info, raw_episode)
        processed = pd.read_parquet(processed_path)
        raw = pd.read_parquet(
            raw_path,
            columns=[
                "timestamp",
                "frame_index",
                "observation.root_orientation",
                "observation.projected_gravity",
            ],
        )
        augmented, episode_scores = add_imu_to_episode(
            processed, raw, args.jitter_window
        )
        augmented.to_parquet(
            output / "data" / "chunk-000" / f"file-{output_episode:03d}.parquet",
            index=False,
        )
        scores.append(episode_scores)
        sources_seen.add(str(raw_root))
        if (output_episode + 1) % 10 == 0 or output_episode + 1 == source_info["total_episodes"]:
            print(f"Augmented {output_episode + 1}/{source_info['total_episodes']} episodes")

    all_scores = np.concatenate(scores)
    quantiles = {
        name: float(value)
        for name, value in zip(
            ("q50", "q75", "q90", "q95", "q99"),
            np.quantile(all_scores, (0.50, 0.75, 0.90, 0.95, 0.99)),
        )
    }
    info = source_info
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [state_dim + len(IMU_NAMES)],
        "names": state_names + IMU_NAMES,
    }
    info["features"]["observation.camera_jitter_score"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["local_peak_imu_angular_speed_rad_s"],
    }
    info["imu_augmentation"] = {
        "source_datasets": sorted(sources_seen),
        "state_layout": IMU_NAMES,
        "gyro_derivation": "backward relative root_orientation at recorded timestamps",
        "jitter_score": f"local peak gyro norm within +/-{args.jitter_window} frames",
        "jitter_score_unit": "rad/s",
        "jitter_score_quantiles": quantiles,
    }
    (output / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )
    modality_path = output / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text())
    modality.setdefault("state", {})["base_imu"] = {
        "start": state_dim,
        "end": state_dim + len(IMU_NAMES),
        "original_key": "observation.state",
    }
    modality["state"]["camera_jitter_score"] = {
        "start": 0,
        "end": 1,
        "original_key": "observation.camera_jitter_score",
    }
    modality_path.write_text(
        json.dumps(modality, indent=4) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(f"Jitter score quantiles (rad/s): {quantiles}")


if __name__ == "__main__":
    main()

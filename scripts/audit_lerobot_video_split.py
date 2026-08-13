#!/usr/bin/env python3
"""Decode every selected LeRobot video horizon and reject zero fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from giga_datasets.datasets.lerobot_dataset import FastLeRobotDataset, _LeRobotDatasetMetadata


HORIZONS = (0, 12, 24, 36, 48)
DEFAULT_VIEW_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--episodes-file", type=Path, action="append", required=True
    )
    parser.add_argument("--view-key", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_episodes(paths: list[Path]) -> tuple[list[int], list[dict[str, Any]]]:
    episodes: list[int] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        values = [int(token) for token in resolved.read_text(encoding="utf-8").split()]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Episode file must be non-empty and unique: {resolved}")
        episodes.extend(values)
        sources.append(
            {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "count": len(values),
            }
        )
    if len(episodes) != len(set(episodes)):
        raise ValueError("Episode files overlap")
    return episodes, sources


def scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def image_stats(value: Any, key: str, episode: int) -> dict[str, Any]:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 4 or int(tensor.shape[0]) != len(HORIZONS):
        raise ValueError(
            f"Episode {episode} {key} shape {tuple(tensor.shape)} does not contain "
            f"{len(HORIZONS)} horizons"
        )
    array = tensor.float().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"Episode {episode} {key} contains non-finite pixels")
    per_horizon_std = array.reshape(len(HORIZONS), -1).std(axis=1)
    per_horizon_abs_max = np.abs(array).reshape(len(HORIZONS), -1).max(axis=1)
    bad = [
        int(HORIZONS[index])
        for index in range(len(HORIZONS))
        if float(per_horizon_abs_max[index]) == 0.0
        or float(per_horizon_std[index]) <= 1e-8
    ]
    if bad:
        raise ValueError(
            f"Episode {episode} {key} has zero/constant decode fallback at horizons {bad}"
        )
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum_std": float(per_horizon_std.min()),
        "minimum_abs_max": float(per_horizon_abs_max.min()),
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    report = args.report.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite video audit: {report}")
    if not report.parent.is_dir():
        raise FileNotFoundError(report.parent)
    episodes, episode_sources = read_episodes(args.episodes_file)
    view_keys = tuple(args.view_key or DEFAULT_VIEW_KEYS)
    if not view_keys or len(view_keys) != len(set(view_keys)):
        raise ValueError("View keys must be non-empty and unique")

    metadata = _LeRobotDatasetMetadata(root.name, root=root)
    delta_timestamps = {
        key: [float(horizon) / float(metadata.fps) for horizon in HORIZONS]
        for key in view_keys
    }
    dataset = FastLeRobotDataset(
        root.name,
        root=root,
        delta_timestamps=delta_timestamps,
        episodes=sorted(episodes),
        video_backend="pyav",
    )
    dataset._ensure_hf_dataset_loaded()
    source_to_index: dict[tuple[int, int], int] = {}
    for index, (episode_index, frame_index) in enumerate(
        zip(dataset.hf_dataset["episode_index"], dataset.hf_dataset["frame_index"])
    ):
        source_to_index[(scalar_int(episode_index), scalar_int(frame_index))] = index

    records: list[dict[str, Any]] = []
    for episode in episodes:
        source = (episode, 0)
        if source not in source_to_index:
            raise ValueError(f"Selected frame-0 source is absent: {source}")
        sample = dataset[source_to_index[source]]
        if scalar_int(sample["episode_index"]) != episode:
            raise ValueError(f"Decoded wrong episode for {source}")
        if scalar_int(sample["frame_index"]) != 0:
            raise ValueError(f"Decoded wrong frame for {source}")
        views = {
            key: image_stats(sample[key], key, episode)
            for key in view_keys
        }
        records.append({"episode_index": episode, "source_frame_index": 0, "views": views})

    result = {
        "status": "passed",
        "dataset_root": str(root),
        "episode_sources": episode_sources,
        "num_episodes": len(records),
        "horizons": list(HORIZONS),
        "view_keys": list(view_keys),
        "video_backend": "pyav",
        "zero_or_constant_fallbacks": 0,
        "records": records,
    }
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()

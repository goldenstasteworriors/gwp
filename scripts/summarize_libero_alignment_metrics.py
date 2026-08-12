#!/usr/bin/env python3
"""Aggregate per-episode LIBERO action-consequence alignment metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


AGGREGATE_FIELDS = (
    "registered_mask_iou_mean",
    "raw_mask_iou_mean",
    "eef_rollout_error_px_mean",
    "eef_frame_endpoint_error_px_mean",
    "eef_rotation_error_deg_mean",
    "projection_vs_gripper_endpoint_error_px_mean",
)
HORIZON_FIELDS = (
    ("raw_mask_iou", ("raw_mask", "iou")),
    ("registered_mask_iou", ("registered_mask", "iou")),
    ("registered_mask_dice", ("registered_mask", "dice")),
    ("registered_mask_centroid_px", ("registered_mask", "centroid_px")),
    ("eef_rollout_error_px", ("eef_rollout_error_px",)),
    ("eef_frame_endpoint_error_px", ("eef_frame_endpoint_error_px",)),
    ("eef_rotation_error_deg", ("eef_rotation_error_deg",)),
    (
        "projection_vs_gripper_endpoint_error_px",
        ("gt_eef_projection_vs_gripper_video", "endpoint_error_px"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def nested_value(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def statistics(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("episode_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No episode_*/metrics.json files under {args.input_dir}")
    episodes = [json.loads(path.read_text()) for path in paths]

    aggregate = {
        field: statistics(
            [float(episode["aggregate"][field]) for episode in episodes if field in episode["aggregate"]]
        )
        for field in AGGREGATE_FIELDS
    }
    horizons: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        for row in episode["horizons"]:
            horizon = int(row["horizon"])
            for output_name, path in HORIZON_FIELDS:
                value = nested_value(row, path)
                if value is not None:
                    horizons[horizon][output_name].append(value)

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "suite": episodes[0].get("suite"),
        "num_episodes": len(episodes),
        "episode_indices": [int(episode["episode_index"]) for episode in episodes],
        "aggregate": aggregate,
        "by_horizon": {
            str(horizon): {
                field: statistics(values) for field, values in sorted(fields.items())
            }
            for horizon, fields in sorted(horizons.items())
        },
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    output = args.output or (args.input_dir / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")
    print(output)


if __name__ == "__main__":
    main()

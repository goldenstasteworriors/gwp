#!/usr/bin/env python3
"""Relabel an existing SONIC wrist-delta LeRobot dataset in camera coordinates.

The 3x3 camera-to-base rotation is required.  It must be measured or explicitly
chosen for a smoke test; the script never infers a calibration from the data.
Videos are hard-linked on the same filesystem and remain byte-for-byte equal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.camera_centric import (
    CAMERA_ACTION_SCHEMA_ID,
    validate_rotation,
    wrist_local_to_camera_action,
)
from world_action_model.asymmetric_action_schema import LEGACY_SCHEMA_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--camera-to-base-rotation", required=True, nargs=9, type=float,
        metavar="R", help="Row-major R_base_camera, rotation only",
    )
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    rotation = validate_rotation(np.asarray(args.camera_to_base_rotation).reshape(3, 3))
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("action_schema_id") != LEGACY_SCHEMA_ID:
        raise ValueError(f"expected legacy30 wrist delta schema, got {info.get('action_schema_id')!r}")
    paths = sorted((source / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {source / 'data'}")
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    for path in paths:
        frame = pd.read_parquet(path)
        state = np.stack(frame["observation.state"].to_numpy())
        action = np.stack(frame["action"].to_numpy())
        frame["action"] = list(wrist_local_to_camera_action(state, action, rotation))
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    for path in (source / "videos").rglob("*.mp4"):
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
    info["action_schema_id"] = CAMERA_ACTION_SCHEMA_ID
    info["camera_to_base_rotation"] = rotation.tolist()
    info["camera_action_reference"] = "observed camera frame; wrist relative SE(3) increment"
    info["camera_action_source_schema"] = LEGACY_SCHEMA_ID
    names = list(info["features"]["action"]["names"])
    if len(names) != 30:
        raise ValueError(f"expected 30 action names, got {len(names)}")
    for side, start in (("left", 0), ("right", 9)):
        names[start : start + 3] = [f"{side}_wrist_camera_delta_{axis}" for axis in "xyz"]
        names[start + 3 : start + 9] = [f"{side}_wrist_camera_delta_rot6d_{i}" for i in range(6)]
    info["features"]["action"]["names"] = names
    (output / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"camera-action dataset: {output}, parquet files: {len(paths)}")


if __name__ == "__main__":
    main()

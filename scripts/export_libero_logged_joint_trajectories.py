#!/usr/bin/env python3
"""Export frame-0 logged q trajectories for direct FK mask rasterization.

This is a format adapter only.  It reads the simulator-free logged action/q
dataset, exports q[0:49], and neither imports LIBERO nor executes dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


INPUT_SCHEMA = "libero-logged-action-joint-trajectory-v1"
OUTPUT_SCHEMA = "libero-logged-joint-trajectory-export-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-dataset-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    manifest_path = args.joint_dataset_split / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"Unexpected logged-q schema: {manifest_path}")
    if manifest.get("simulator_runtime_imported") is not False:
        raise ValueError("Logged-q dataset imported simulator runtime")
    if int(manifest.get("simulator_step_calls", -1)) != 0:
        raise ValueError("Logged-q dataset contains simulator steps")

    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for record in manifest.get("episodes", []):
        source_path = args.joint_dataset_split / str(record["cache_file"])
        if sha256_file(source_path) != record["cache_sha256"]:
            raise ValueError(f"Logged-q cache changed: {source_path}")
        with np.load(source_path, allow_pickle=False) as arrays:
            arm = np.asarray(arrays["arm_qpos"], dtype=np.float32)
            gripper = np.asarray(arrays["gripper_qpos"], dtype=np.float32)
        if arm.ndim != 2 or arm.shape[1] != 7 or gripper.shape != (len(arm), 2):
            raise ValueError(f"Unexpected q shape in {source_path}")
        if len(arm) < 49 or not np.isfinite(arm).all() or not np.isfinite(gripper).all():
            raise ValueError(f"Invalid 49-step logged q in {source_path}")

        episode = int(record["episode_index"])
        episode_root = args.output_dir / f"episode_{episode:06d}"
        episode_root.mkdir()
        arm_path = episode_root / "predicted_joint_qpos.npy"
        gripper_path = episode_root / "predicted_gripper_qpos.npy"
        np.save(arm_path, arm[:49])
        np.save(gripper_path, gripper[:49])
        rows.append(
            {
                "episode_index": episode,
                "task_index": int(record["task_index"]),
                "task_language": str(record["task_language"]),
                "source_frame_index": 0,
                "source_cache_file": str(source_path.resolve()),
                "source_cache_sha256": record["cache_sha256"],
                "arm_path": str(arm_path.relative_to(args.output_dir)),
                "arm_sha256": sha256_file(arm_path),
                "gripper_path": str(gripper_path.relative_to(args.output_dir)),
                "gripper_sha256": sha256_file(gripper_path),
            }
        )

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "input_schema_version": INPUT_SCHEMA,
        "joint_dataset_split": str(args.joint_dataset_split.resolve()),
        "joint_dataset_manifest_sha256": sha256_file(manifest_path),
        "q_source": "logged demonstration observations at frame 0..48",
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "num_episodes": len(rows),
        "episodes": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"num_episodes": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

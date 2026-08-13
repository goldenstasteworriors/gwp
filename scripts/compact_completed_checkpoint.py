#!/usr/bin/env python3
"""Strip resumability state only after inference and EMA weights are complete."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--required-model-dir",
        action="append",
        default=[],
        help="Inference model directory required before compaction (repeatable).",
    )
    parser.add_argument("--minimum-weight-gib", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def write_status(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="utf-8")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def weight_summary(model_dir: Path, minimum_bytes: int) -> dict[str, Any]:
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(model_dir / "config.json")
    weights = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.safetensors"))
    weights = [path for path in weights if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in weights)
    if not weights or total_bytes < minimum_bytes:
        raise RuntimeError(
            f"Incomplete inference weights in {model_dir}: {total_bytes} bytes"
        )
    return {
        "root": str(model_dir.resolve()),
        "files": [path.name for path in weights],
        "bytes": total_bytes,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    report = args.report.resolve()
    status = args.status.resolve()
    if args.training_pid <= 0:
        raise ValueError("--training-pid must be positive")
    if args.minimum_weight_gib <= 0 or args.poll_seconds <= 0:
        raise ValueError("minimum weight and polling interval must be positive")
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite compaction report: {report}")
    if not status.parent.is_dir() or not report.parent.is_dir():
        raise FileNotFoundError("Status/report parent directory must exist")
    required_names = args.required_model_dir or ["transformer", "transformer_ema"]
    if len(required_names) != len(set(required_names)):
        raise ValueError("Required model directories must be unique")
    if any(Path(name).name != name for name in required_names):
        raise ValueError("Required model directories must be direct child names")

    write_status(status, "waiting_training")
    while process_exists(args.training_pid):
        time.sleep(args.poll_seconds)

    write_status(status, "validating_inference_weights")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    minimum_bytes = int(args.minimum_weight_gib * 1024**3)
    summaries = [
        weight_summary(checkpoint / name, minimum_bytes) for name in required_names
    ]

    from world_action_model.trainer.wa_casual_trainer_mot import strip_optimizer_state

    write_status(status, "compacting")
    removed = strip_optimizer_state(checkpoint)
    if (checkpoint / "pytorch_model").exists():
        raise RuntimeError("DeepSpeed state remains after compaction")
    for name in required_names:
        if not (checkpoint / name / "config.json").is_file():
            raise RuntimeError(f"Compaction damaged inference model: {name}")
    disk = shutil.disk_usage(checkpoint)
    result = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "training_pid": args.training_pid,
        "required_inference_weights": summaries,
        "removed": removed,
        "free_bytes_after": disk.free,
    }
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_status(status, "complete")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

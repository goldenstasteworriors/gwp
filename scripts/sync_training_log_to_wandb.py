#!/usr/bin/env python3
"""Continuously mirror GigaWorld text training metrics to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import wandb


STEP_RE = re.compile(
    r"Step\[(?P<step>\d+)/(?P<max_steps>\d+)\].*?"
    r"lr: (?P<lr>[-+.\deE]+).*?"
    r"visual_loss: (?P<visual_loss>[-+.\deE]+), "
    r"action_loss: (?P<action_loss>[-+.\deE]+), "
    r"total_loss: (?P<total_loss>[-+.\deE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="gigaworld_lab")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def load_offset(path: Path) -> int:
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("offset", 0))


def save_offset(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"offset": offset}) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        config={"source_log": str(args.log)},
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    offset = load_offset(args.state)

    while True:
        try:
            with args.log.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                for line in stream:
                    match = STEP_RE.search(line)
                    if match:
                        values = match.groupdict()
                        run.log(
                            {
                                "train/step": int(values["step"]),
                                "train/max_steps": int(values["max_steps"]),
                                "train/learning_rate": float(values["lr"]),
                                "train/visual_loss": float(values["visual_loss"]),
                                "train/action_loss": float(values["action_loss"]),
                                "train/total_loss": float(values["total_loss"]),
                            }
                        )
                offset = stream.tell()
            save_offset(args.state, offset)
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            break
    run.finish()


if __name__ == "__main__":
    main()

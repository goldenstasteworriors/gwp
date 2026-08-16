#!/usr/bin/env python3
"""Run one exported SONIC history policy on an NPZ observation bundle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "third_party" / "giga-datasets",
    PROJECT_ROOT / "third_party" / "giga-models",
    PROJECT_ROOT / "third_party" / "giga-train",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

from world_action_model.inference import SonicHistoryPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observation = np.load(args.observation)
    if "images" not in observation or "states" not in observation:
        raise KeyError("Observation NPZ must contain `images` and `states`")
    progress = float(observation["episode_progress"]) if "episode_progress" in observation else 0.0
    policy = SonicHistoryPolicy(args.manifest, device=args.device)
    action = policy.predict(
        images=observation["images"],
        states=observation["states"],
        episode_progress=progress,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        action=action,
        manifest=str(args.manifest.resolve()),
        observation=str(args.observation.resolve()),
    )
    print(f"Saved action prediction {action.shape} to {args.output}")


if __name__ == "__main__":
    main()

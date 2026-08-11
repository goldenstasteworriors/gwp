#!/usr/bin/env python3
"""Run the CPU-only op42 causal-main acceptance suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_action_model.contact_causal_cpu_acceptance import run_cpu_acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root", type=Path, default=REPOSITORY_ROOT
    )
    args = parser.parse_args()
    report = run_cpu_acceptance(args.repository_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

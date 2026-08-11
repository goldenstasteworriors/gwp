#!/usr/bin/env python3
"""Run the artifact-bound contact/asymmetric25 training preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_asym25_preflight import preflight_from_environment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and optionally persist the production contact/asymmetric25 preflight receipt."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Project-local JSON output. The path must not already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.dumps(
        preflight_from_environment(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    output = args.output.expanduser().resolve(strict=False)
    if output == Path("/tmp") or Path("/tmp") in output.parents:
        raise ValueError("preflight receipts must not be written under /tmp")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(output)


if __name__ == "__main__":
    main()

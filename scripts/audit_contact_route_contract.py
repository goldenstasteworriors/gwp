#!/usr/bin/env python3
"""Write the strict CPU causal-route audit report to an explicit project path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_route_audit import run_contact_route_cpu_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CPU-only contact causal-route audit (no training/admission)."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"output parent must already exist; refusing implicit path creation: {output.parent}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit report: {output}")
    report = run_contact_route_cpu_audit()
    payload = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    output.write_text(payload, encoding="utf-8")
    if output.read_text(encoding="utf-8") != payload:
        raise RuntimeError("audit report write/read verification failed")
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare rank-local matched-control JSONL traces without running a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.matched_control_hashes import (  # noqa: E402
    ALLOWED_SEEDS,
    compare_variant_traces,
)


def _variant(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("variant must use NAME=TRACE_FILE_OR_DIRECTORY")
    return name.strip(), path.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        type=_variant,
        metavar="NAME=PATH",
        help="Repeat once per ablation variant; PATH may be a JSONL file or directory.",
    )
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--required-seed",
        action="append",
        type=int,
        choices=ALLOWED_SEEDS,
        help="Required seed; defaults to 6666, 7777, and 8888.",
    )
    parser.add_argument(
        "--distinct-variant-run-identities",
        action="store_true",
        help=(
            "Require a different run ID, run nonce, and external acceptance "
            "for every transport variant (contact-asym25 current-run mode)."
        ),
    )
    args = parser.parse_args()

    variants: dict[str, list[str]] = {}
    for name, path in args.variant:
        if name in variants:
            parser.error(f"duplicate --variant name: {name}")
        variants[name] = [path]
    report = compare_variant_traces(
        variants,
        output_report=args.output_report,
        required_seeds=args.required_seed or ALLOWED_SEEDS,
        allow_distinct_variant_run_identities=args.distinct_variant_run_identities,
    )
    print(f"{report['status']}: {Path(args.output_report).resolve()}")
    for error in report.get("errors", []):
        print(f"error: {error}", file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

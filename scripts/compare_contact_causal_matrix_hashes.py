#!/usr/bin/env python3
"""Compare exact causal-main collector-v3 route traces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_action_model.contact_causal_matched_control import (
    compare_causal_records,
    load_trace_jsonl,
)
from world_action_model.contact_causal_production import (
    CAUSAL_ROUTES,
    ensure_not_project_tmp,
    write_canonical_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    for route in CAUSAL_ROUTES:
        parser.add_argument(f"--{route.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {
        route: load_trace_jsonl(
            getattr(args, route), route=route
        )
        for route in CAUSAL_ROUTES
    }
    report = compare_causal_records(records, required_seed=args.seed)
    output = ensure_not_project_tmp(args.output, project_root=REPOSITORY_ROOT)
    digest = write_canonical_json_exclusive(output, report)
    print(f"{digest}  {output}")


if __name__ == "__main__":
    main()

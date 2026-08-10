#!/usr/bin/env python3
"""Validate authoritative Ego group metadata and optionally emit inventory CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.ego_group_metadata_recovery import (  # noqa: E402
    EgoGroupMetadataError,
    REQUIRED_GROUP_FIELDS,
    validate_supplement,
)


INVENTORY_FIELDS = (
    "episode_id",
    "length_frames",
    *REQUIRED_GROUP_FIELDS,
    "metadata_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument(
        "--allow-incomplete-template",
        action="store_true",
        help="Lint a null template; never marks it production complete.",
    )
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = validate_supplement(
        args.supplement,
        args.audit,
        allow_incomplete=args.allow_incomplete_template,
    )
    if args.inventory_output is not None:
        if not receipt["production_complete"]:
            raise EgoGroupMetadataError(
                "an incomplete template cannot emit a production episode inventory"
            )
        args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
        with args.inventory_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(receipt["inventory_rows"])
    public_receipt = {key: value for key, value in receipt.items() if key != "inventory_rows"}
    if args.receipt_output is not None:
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.write_text(
            json.dumps(public_receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    stdout_receipt = {
        key: value for key, value in public_receipt.items() if key != "missing_cells"
    }
    print(json.dumps(stdout_receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

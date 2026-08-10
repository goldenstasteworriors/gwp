#!/usr/bin/env python3
"""Validate an asymmetric-25 hardware contract and optional calibration record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.asymmetric_hardware_manifest import (
    calibration_capture_sha256,
    hardware_contract_sha256,
    load_hardware_manifest,
    validate_manifest_against_source_info,
    validate_manifest_capture_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("schema", "training", "deployment"),
        default="schema",
        help="training requires bench_verified; deployment requires robot_verified.",
    )
    parser.add_argument(
        "--source-info",
        type=Path,
        help="Optional raw LeRobot meta/info.json to bind channel names, WBC order and fps.",
    )
    parser.add_argument(
        "--calibration-capture",
        type=Path,
        help="Capture JSON; required in training/deployment mode and compared section by section.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"training", "deployment"} and args.calibration_capture is None:
        raise ValueError(
            f"--calibration-capture is required in {args.mode} mode"
        )
    manifest = load_hardware_manifest(args.manifest, mode=args.mode)
    result = {
        "valid": True,
        "mode": args.mode,
        "action_schema_id": manifest["action_schema_id"],
        "contract_id": manifest["contract_id"],
        "verification_level": manifest["calibration"]["verification_level"],
        "hardware_contract_sha256": hardware_contract_sha256(manifest),
    }
    if args.source_info is not None:
        info = json.loads(args.source_info.resolve().read_text(encoding="utf-8"))
        layout = validate_manifest_against_source_info(manifest, info)
        result["source_info"] = str(args.source_info.resolve())
        result["source_channel_indices"] = {
            "left_state": layout["left_state_indices"],
            "left_action": layout["left_action_indices"],
            "right_state": layout["right_state_index"],
            "right_action": layout["right_action_index"],
        }
    if args.calibration_capture is not None:
        capture = json.loads(
            args.calibration_capture.resolve().read_text(encoding="utf-8")
        )
        validate_manifest_capture_binding(manifest, capture)
        result["calibration_capture_sha256"] = calibration_capture_sha256(capture)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

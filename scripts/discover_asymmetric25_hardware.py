#!/usr/bin/env python3
"""Inventory host-visible interfaces for an asymmetric25 hardware bring-up.

This utility is intentionally metadata-only.  It never opens a serial, CAN,
DDS, Modbus, or network endpoint and it contains no device command path.  Its
output is a discovery aid, not a calibration capture or deployment approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs/hardware/asymmetric25_hardware_manifest.template.json"
DEFAULT_CAPTURE = REPO_ROOT / "configs/hardware/asymmetric25_calibration_capture.template.json"
ARPHRD_CAN = "280"

REQUIRED_CONFIRMATIONS = (
    {
        "area": "physical_installation",
        "items": [
            "Unitree G1 exact variant and robot serial",
            "left RH56DFTP and right OmniPicker mounting/handedness",
            "left and right tool-center frames relative to wrist_yaw links",
            "mechanical interference and cable-strain inspection",
        ],
    },
    {
        "area": "interfaces_and_revisions",
        "items": [
            (
                "RH56DFTP serial number, Modbus/CAN interface, device ID, "
                "driver revision, and firmware revision"
            ),
            (
                "OmniPicker serial number, official SDK/interface, endpoint identifier, "
                "driver revision, and firmware revision"
            ),
            (
                "G1 DDS domain, network interface, low-state/low-command endpoints, "
                "and WBC source revision"
            ),
            "emergency-stop endpoint and watchdog behavior",
        ],
    },
    {
        "area": "measured_calibration",
        "items": [
            "RH56DFTP six-channel ordered raw state and absolute-command names/units/ranges",
            "RH56DFTP per-channel open/closed values, safe limits, and maximum command step",
            (
                "OmniPicker aperture raw state/command name, unit, direction, zero, "
                "open/closed values, limits, and maximum step"
            ),
            "left/right wrist state parent/child frames and WBC FK frames on the mounted tools",
            "29-joint measured-state and absolute-command order",
        ],
    },
    {
        "area": "timing_and_safety",
        "items": [
            "hardware/synchronized clock domain and timestamp origin",
            (
                "measured rate, period jitter, command latency, state/action skew, "
                "and camera/state skew"
            ),
            (
                "translation/rotation step limits, hand/gripper limits, watchdog timeout, "
                "and invalid-command policy"
            ),
            "operator-reviewed open/close trial followed by bench and robot approval",
        ],
    },
)


def _read_text(path: Path) -> str | None:
    """Read a small sysfs metadata file without opening a device endpoint."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError, UnicodeError):
        return None


def discover_network_interfaces(sys_class_net: Path) -> list[dict[str, Any]]:
    interfaces: list[dict[str, Any]] = []
    try:
        entries = sorted(sys_class_net.iterdir(), key=lambda path: path.name)
    except (FileNotFoundError, PermissionError, OSError):
        return interfaces

    for entry in entries:
        interface_type = _read_text(entry / "type")
        interfaces.append(
            {
                "name": entry.name,
                "kind": "can" if interface_type == ARPHRD_CAN else "non_can",
                "arphrd_type": interface_type,
                "operstate": _read_text(entry / "operstate"),
                "carrier": _read_text(entry / "carrier"),
                "mtu": _read_text(entry / "mtu"),
            }
        )
    return interfaces


def discover_serial_links(serial_by_id: Path) -> list[dict[str, str | None]]:
    links: list[dict[str, str | None]] = []
    try:
        entries = sorted(serial_by_id.iterdir(), key=lambda path: path.name)
    except (FileNotFoundError, PermissionError, OSError):
        return links

    for entry in entries:
        try:
            target = str(entry.resolve(strict=False))
        except OSError:
            target = None
        links.append({"by_id_name": entry.name, "target": target})
    return links


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} {path} must contain a JSON object")
    return value


def find_placeholders(value: Any, path: str = "") -> list[str]:
    """Return paths that deliberately remain unresolved in a template."""

    missing: list[str] = []
    if value is None:
        return [path or "$"]
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$schema", "$comment"}:
                continue
            child_path = f"{path}.{key}" if path else key
            missing.extend(find_placeholders(child, child_path))
    elif isinstance(value, list):
        if not value:
            missing.append(path or "$")
        else:
            for index, child in enumerate(value):
                missing.extend(find_placeholders(child, f"{path}[{index}]"))
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--sys-class-net",
        type=Path,
        default=Path("/sys/class/net"),
        help="sysfs network-interface directory (metadata only)",
    )
    parser.add_argument(
        "--serial-by-id",
        type=Path,
        default=Path("/dev/serial/by-id"),
        help="udev stable serial-link directory (symlink metadata only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.manifest.resolve(), "manifest")
    capture = _load_json(args.capture.resolve(), "capture")
    interfaces = discover_network_interfaces(args.sys_class_net)
    serial_links = discover_serial_links(args.serial_by_id)

    report = {
        "discovery_version": "asymmetric25-host-discovery/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "metadata_only_no_device_io",
        "safety": {
            "opened_serial_endpoint": False,
            "opened_can_endpoint": False,
            "opened_network_endpoint": False,
            "initialized_dds": False,
            "sent_control_command": False,
            "is_calibration_capture": False,
            "authorizes_training_or_deployment": False,
        },
        "host": {
            "platform": platform.system(),
            "kernel_release": platform.release(),
            "network_interfaces": interfaces,
            "stable_serial_links": serial_links,
        },
        "template_audit": {
            "manifest": str(args.manifest.resolve()),
            "manifest_placeholders": find_placeholders(manifest),
            "capture": str(args.capture.resolve()),
            "capture_placeholders": find_placeholders(capture),
            "manifest_training_ready": manifest.get("readiness", {}).get("training") is True,
            "manifest_deployment_ready": manifest.get("readiness", {}).get("deployment") is True,
            "capture_approved": capture.get("approval", {}).get("approved") is True,
        },
        "discovery_interpretation": {
            "candidate_can_interfaces": [
                item["name"] for item in interfaces if item["kind"] == "can"
            ],
            "candidate_serial_links": [item["by_id_name"] for item in serial_links],
            "warning": (
                "Host-visible names are candidates only. An operator must map them to the "
                "mounted devices using official documentation and physical labels."
            ),
        },
        "required_operator_confirmations": REQUIRED_CONFIRMATIONS,
        "next_gate": (
            "Keep deployment No-Go until official drivers/protocols are identified and all "
            "manifest/capture values are measured, reviewed, hash-bound, and approved."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

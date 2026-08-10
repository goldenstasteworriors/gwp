#!/usr/bin/env python3
"""Inventory host-visible interfaces for an asymmetric25 hardware bring-up.

This utility is intentionally metadata-only.  With its validated repository
JSON inputs and allowlisted metadata roots, it never opens a serial, CAN, DDS,
Modbus, or network endpoint and contains no device command path.  Its output is
a discovery aid, not a calibration capture or deployment approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs/hardware/asymmetric25_hardware_manifest.template.json"
DEFAULT_CAPTURE = REPO_ROOT / "configs/hardware/asymmetric25_calibration_capture.template.json"
DEFAULT_MANIFEST_SCHEMA = REPO_ROOT / "schemas/asymmetric25_hardware_manifest.schema.json"
DEFAULT_CAPTURE_SCHEMA = REPO_ROOT / "schemas/asymmetric25_calibration_capture.schema.json"
DEFAULT_SYS_CLASS_NET = Path("/sys/class/net")
DEFAULT_SERIAL_BY_ID = Path("/dev/serial/by-id")
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/hardware_discovery"
FIXTURE_SYS_CLASS_NET = FIXTURE_ROOT / "sys_class_net"
FIXTURE_SERIAL_BY_ID = FIXTURE_ROOT / "serial_by_id"
MAX_JSON_BYTES = 4 * 1024 * 1024
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _open_regular_nofollow(
    path: Path,
    *,
    max_bytes: int,
    expected_identity: tuple[int, int] | None = None,
) -> str:
    """Read a bounded regular file without following its final path component."""

    if (
        not hasattr(os, "O_PATH")
        or not hasattr(os, "O_NOFOLLOW")
        or not Path("/proc/self/fd").is_dir()
    ):
        raise ValueError("safe metadata reads require Linux O_PATH/O_NOFOLLOW and /proc/self/fd")
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(path_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"refusing non-regular file {path}")
        identity = (metadata.st_dev, metadata.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(f"metadata file changed while being opened: {path}")
        if metadata.st_size > max_bytes:
            raise ValueError(f"refusing oversized metadata file {path}")
        fd = os.open(
            f"/proc/self/fd/{path_fd}",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
        )
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != identity or not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"metadata identity changed while being opened: {path}")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                block = os.read(fd, min(65536, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            payload = b"".join(chunks)
            if len(payload) > max_bytes:
                raise ValueError(f"refusing oversized metadata file {path}")
            return payload.decode("utf-8").strip()
        finally:
            os.close(fd)
    finally:
        os.close(path_fd)


def _validated_repo_json_path(path: Path, label: str) -> tuple[Path, tuple[int, int]]:
    """Accept only non-symlink regular JSON files contained in this repository."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    try:
        before = os.lstat(candidate)
    except OSError as exc:
        raise SystemExit(f"cannot inspect {label} {candidate}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"refusing non-regular or symlink {label} {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"cannot resolve {label} {candidate}: {exc}") from exc
    repo_root = REPO_ROOT.resolve(strict=True)
    if not _is_within(resolved, repo_root) or resolved.suffix.lower() != ".json":
        raise SystemExit(f"{label} must be a repository-contained .json regular file: {candidate}")
    after = os.lstat(resolved)
    if stat.S_ISLNK(after.st_mode) or not stat.S_ISREG(after.st_mode):
        raise SystemExit(f"refusing non-regular or symlink {label} {resolved}")
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise SystemExit(f"{label} changed while being validated: {candidate}")
    return resolved, (after.st_dev, after.st_ino)


def _read_text(path: Path, allowed_roots: tuple[Path, ...]) -> str | None:
    """Read a small regular metadata attribute under an allowlisted root."""

    try:
        resolved = path.resolve(strict=True)
        roots = tuple(root.resolve(strict=True) for root in allowed_roots if root.exists())
        if not any(_is_within(resolved, root) for root in roots):
            return None
        metadata = os.lstat(resolved)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return None
        return _open_regular_nofollow(
            resolved,
            max_bytes=4096,
            expected_identity=(metadata.st_dev, metadata.st_ino),
        )
    except (FileNotFoundError, PermissionError, OSError, UnicodeError, ValueError):
        return None


def discover_network_interfaces(sys_class_net: Path) -> list[dict[str, Any]]:
    interfaces: list[dict[str, Any]] = []
    try:
        entries = sorted(sys_class_net.iterdir(), key=lambda path: path.name)
    except (FileNotFoundError, PermissionError, OSError):
        return interfaces

    fixture_mode = sys_class_net == FIXTURE_SYS_CLASS_NET
    allowed_attribute_roots = (
        (FIXTURE_SYS_CLASS_NET,)
        if fixture_mode
        else (DEFAULT_SYS_CLASS_NET, Path("/sys/devices"))
    )
    for entry in entries:
        try:
            resolved_entry = entry.resolve(strict=True)
        except OSError:
            continue
        resolved_roots = tuple(
            root.resolve(strict=True) for root in allowed_attribute_roots if root.exists()
        )
        if not any(_is_within(resolved_entry, root) for root in resolved_roots):
            continue
        interface_type = _read_text(entry / "type", allowed_attribute_roots)
        interfaces.append(
            {
                "name": entry.name,
                "kind": "can" if interface_type == ARPHRD_CAN else "non_can",
                "arphrd_type": interface_type,
                "operstate": _read_text(entry / "operstate", allowed_attribute_roots),
                "carrier": _read_text(entry / "carrier", allowed_attribute_roots),
                "mtu": _read_text(entry / "mtu", allowed_attribute_roots),
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


def _validate_metadata_root(path: Path, default: Path, fixture: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if candidate.exists() or candidate.is_symlink():
        candidate_metadata = os.lstat(candidate)
        if stat.S_ISLNK(candidate_metadata.st_mode) or not stat.S_ISDIR(candidate_metadata.st_mode):
            raise SystemExit(f"refusing non-directory or symlink {label} {candidate}")
    normalized = candidate.resolve(strict=False)
    allowed = {default.resolve(strict=False), fixture.resolve(strict=False)}
    if normalized not in allowed:
        raise SystemExit(
            f"{label} must be exactly {default} or repository fixture root {fixture}; got {path}"
        )
    return normalized


def _digest_identifier(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None


def _display_path(path: Path, include_sensitive: bool) -> str:
    return str(path) if include_sensitive else path.relative_to(REPO_ROOT).as_posix()


def _redact_interfaces(
    interfaces: list[dict[str, Any]], include_sensitive: bool
) -> list[dict[str, Any]]:
    if include_sensitive:
        return interfaces
    return [
        {
            "name_sha256": _digest_identifier(item["name"]),
            "kind": item["kind"],
            "arphrd_type": item["arphrd_type"],
            "operstate": item["operstate"],
            "carrier": item["carrier"],
            "mtu": item["mtu"],
        }
        for item in interfaces
    ]


def _redact_serial_links(
    links: list[dict[str, str | None]], include_sensitive: bool
) -> list[dict[str, str | None]]:
    if include_sensitive:
        return links
    return [
        {
            "by_id_name_sha256": _digest_identifier(item["by_id_name"]),
            "target_sha256": _digest_identifier(item["target"]),
        }
        for item in links
    ]


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved, identity = _validated_repo_json_path(path, label)
    try:
        value = json.loads(
            _open_regular_nofollow(
                resolved,
                max_bytes=MAX_JSON_BYTES,
                expected_identity=identity,
            )
        )
    except (FileNotFoundError, PermissionError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} {resolved} must contain a JSON object")
    return value, resolved


def find_placeholders(value: Any, path: str = "") -> list[str]:
    """Return paths that deliberately remain unresolved in a template."""

    missing: list[str] = []
    if value is None:
        return [path or "$"]
    if isinstance(value, dict):
        if not value:
            return [path or "$"]
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
    elif isinstance(value, str) and not value.strip():
        missing.append(path or "$")
    return missing


def _resolve_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    if not reference.startswith("#/"):
        return schema
    value: Any = root
    for token in reference[2:].split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value if isinstance(value, dict) else schema


def find_schema_issues(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str = "",
) -> list[str]:
    """Return structural/type issue paths; this is not a validator replacement."""

    schema = _resolve_schema(schema, root)
    issues: list[str] = []
    for component in schema.get("allOf", []):
        issues.extend(find_schema_issues(value, component, root, path))
    expected = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected, True)
    if not type_ok:
        issues.append(path or "$")
        return issues
    if "const" in schema and value != schema["const"]:
        issues.append(path or "$")
    if "enum" in schema and value not in schema["enum"]:
        issues.append(path or "$")
    if isinstance(value, str) and len(value) < int(schema.get("minLength", 0)):
        issues.append(path or "$")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            child_path = f"{path}.{key}" if path else key
            if key not in value:
                issues.append(child_path)
        for key, child in value.items():
            if key in properties:
                child_path = f"{path}.{key}" if path else key
                issues.extend(find_schema_issues(child, properties[key], root, child_path))
    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if (minimum is not None and len(value) < int(minimum)) or (
            maximum is not None and len(value) > int(maximum)
        ):
            issues.append(path or "$")
        prefix = schema.get("prefixItems", [])
        for index, child in enumerate(value[: len(prefix)]):
            issues.extend(find_schema_issues(child, prefix[index], root, f"{path}[{index}]"))
        items = schema.get("items")
        if isinstance(items, dict):
            for index, child in enumerate(value[len(prefix) :], len(prefix)):
                issues.extend(find_schema_issues(child, items, root, f"{path}[{index}]"))
    return sorted(set(issues))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--sys-class-net",
        type=Path,
        default=DEFAULT_SYS_CLASS_NET,
        help="allowlisted sysfs directory; overrides are limited to the repository fixture root",
    )
    parser.add_argument(
        "--serial-by-id",
        type=Path,
        default=DEFAULT_SERIAL_BY_ID,
        help="allowlisted stable-link directory; overrides are limited to the repository fixture root",
    )
    parser.add_argument(
        "--include-sensitive-local-metadata",
        action="store_true",
        help=(
            "show exact host time, platform/kernel, interface/by-id names, targets, and absolute paths; "
            "keep output local and never commit it"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, manifest_path = _load_json(args.manifest, "manifest")
    capture, capture_path = _load_json(args.capture, "capture")
    manifest_schema, _ = _load_json(DEFAULT_MANIFEST_SCHEMA, "manifest schema")
    capture_schema, _ = _load_json(DEFAULT_CAPTURE_SCHEMA, "capture schema")
    sys_class_net = _validate_metadata_root(
        args.sys_class_net, DEFAULT_SYS_CLASS_NET, FIXTURE_SYS_CLASS_NET, "sys-class-net"
    )
    serial_by_id = _validate_metadata_root(
        args.serial_by_id, DEFAULT_SERIAL_BY_ID, FIXTURE_SERIAL_BY_ID, "serial-by-id"
    )
    interfaces = discover_network_interfaces(sys_class_net)
    serial_links = discover_serial_links(serial_by_id)
    include_sensitive = bool(args.include_sensitive_local_metadata)
    manifest_placeholders = find_placeholders(manifest)
    capture_placeholders = find_placeholders(capture)
    manifest_schema_issues = find_schema_issues(
        manifest, manifest_schema, manifest_schema
    )
    capture_schema_issues = find_schema_issues(capture, capture_schema, capture_schema)

    report = {
        "discovery_version": "asymmetric25-host-discovery/v2",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if include_sensitive
            else "redacted"
        ),
        "mode": "metadata_only_no_device_io",
        "sensitive_local_metadata_included": include_sensitive,
        "sensitive_output_policy": (
            "LOCAL_ONLY_DO_NOT_COMMIT" if include_sensitive else "default_redacted"
        ),
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
            "platform": platform.system() if include_sensitive else "redacted",
            "kernel_release": platform.release() if include_sensitive else "redacted",
            "network_interfaces": _redact_interfaces(interfaces, include_sensitive),
            "stable_serial_links": _redact_serial_links(serial_links, include_sensitive),
        },
        "template_audit": {
            "manifest": _display_path(manifest_path, include_sensitive),
            "manifest_placeholders": manifest_placeholders,
            "manifest_placeholder_count": len(manifest_placeholders),
            "manifest_schema_issues": manifest_schema_issues,
            "manifest_schema_issue_count": len(manifest_schema_issues),
            "manifest_schema_reference_matches": (
                (manifest_path.parent / str(manifest.get("$schema", ""))).resolve(strict=False)
                == DEFAULT_MANIFEST_SCHEMA.resolve(strict=True)
            ),
            "capture": _display_path(capture_path, include_sensitive),
            "capture_placeholders": capture_placeholders,
            "capture_placeholder_count": len(capture_placeholders),
            "capture_schema_issues": capture_schema_issues,
            "capture_schema_issue_count": len(capture_schema_issues),
            "capture_schema_reference_matches": (
                (capture_path.parent / str(capture.get("$schema", ""))).resolve(strict=False)
                == DEFAULT_CAPTURE_SCHEMA.resolve(strict=True)
            ),
            "schema_audit_scope": (
                "structural/type/required/const/enum/minLength/array-length checks only; "
                "not a replacement for validate_asymmetric_hardware_manifest.py"
            ),
            "manifest_training_ready": manifest.get("readiness", {}).get("training") is True,
            "manifest_deployment_ready": manifest.get("readiness", {}).get("deployment") is True,
            "capture_approved": capture.get("approval", {}).get("approved") is True,
        },
        "discovery_interpretation": {
            "empty_candidates_prove_hardware_absence": False,
            "candidate_can_interfaces": [
                (
                    item["name"]
                    if include_sensitive
                    else _digest_identifier(item["name"])
                )
                for item in interfaces
                if item["kind"] == "can"
            ],
            "candidate_serial_links": [
                (
                    item["by_id_name"]
                    if include_sensitive
                    else _digest_identifier(item["by_id_name"])
                )
                for item in serial_links
            ],
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

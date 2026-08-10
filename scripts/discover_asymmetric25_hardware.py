#!/usr/bin/env python3
"""Inventory host-visible interfaces for an asymmetric25 hardware bring-up.

This utility is intentionally metadata-only.  With its validated repository
JSON inputs and allowlisted metadata roots, it never opens a serial, CAN, DDS,
Modbus, or network endpoint and contains no device command path.  Its output is
a discovery aid, not a calibration capture or deployment approval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
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
DEFAULT_SYS_DEVICES = Path("/sys/devices")
DEFAULT_SERIAL_BY_ID = Path("/dev/serial/by-id")
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/hardware_discovery"
FIXTURE_SYS_CLASS_NET = FIXTURE_ROOT / "sys_class_net"
FIXTURE_SERIAL_BY_ID = FIXTURE_ROOT / "serial_by_id"
MAX_JSON_BYTES = 4 * 1024 * 1024
ARPHRD_CAN = "280"


@dataclass
class MetadataRoot:
    """A path-identity-checked directory snapshot used only through its dirfd."""

    path: Path
    fd: int | None
    identity: tuple[int, int] | None
    label: str
    fixture: bool

    def verify_current(self) -> None:
        """Reject replacement, disappearance, or symlink substitution of the root."""

        try:
            current = os.lstat(self.path)
        except FileNotFoundError:
            if self.fd is None:
                return
            raise SystemExit(f"{self.label} changed or disappeared during discovery: {self.path}")
        except OSError as exc:
            raise SystemExit(f"cannot recheck {self.label} {self.path}: {exc}") from exc
        if self.fd is None:
            raise SystemExit(f"{self.label} appeared during discovery: {self.path}")
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise SystemExit(f"{self.label} became a symlink or non-directory: {self.path}")
        if (current.st_dev, current.st_ino) != self.identity:
            raise SystemExit(f"{self.label} was replaced during discovery: {self.path}")

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


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


def _read_regular_at(directory_fd: int, name: str, max_bytes: int = 4096) -> str | None:
    """Read one final-component regular file relative to an already pinned dirfd."""

    if not name or "/" in name or name in {".", ".."}:
        return None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            return None
        if before.st_size > max_bytes:
            return None
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                return None
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                block = os.read(fd, min(4096, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            payload = b"".join(chunks)
            if len(payload) > max_bytes:
                return None
            return payload.decode("utf-8").strip()
        finally:
            os.close(fd)
    except (FileNotFoundError, PermissionError, OSError, UnicodeError):
        return None


def _open_child_directory(directory_fd: int, name: str) -> int:
    """Open one non-symlink directory component and pin its inode."""

    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"unsafe directory component {name!r}")
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"refusing non-directory or symlink component {name!r}")
    fd = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    opened = os.fstat(fd)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(fd)
        raise ValueError(f"directory component changed while opening: {name!r}")
    return fd


def _open_directory_beneath(root_fd: int, relative_path: str) -> int:
    """Walk a lexical relative directory without following any component symlink."""

    components = relative_path.split("/")
    if not relative_path or any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"unsafe relative directory path {relative_path!r}")
    current = os.dup(root_fd)
    try:
        for component in components:
            child = _open_child_directory(current, component)
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def _open_sysfs_interface_directory(
    root: MetadataRoot,
    sys_devices: MetadataRoot | None,
    name: str,
) -> int | None:
    """Pin an interface directory without resolving a live filesystem pathname."""

    assert root.fd is not None
    try:
        before = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if stat.S_ISDIR(before.st_mode):
        try:
            return _open_child_directory(root.fd, name)
        except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
            raise SystemExit(
                f"network metadata directory changed while opening: {name!r}"
            ) from exc
    if not stat.S_ISLNK(before.st_mode) or root.fixture or sys_devices is None:
        return None
    try:
        raw_target = os.readlink(name, dir_fd=root.fd)
        after = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise SystemExit(f"network metadata symlink changed during discovery: {name!r}")
        if os.path.isabs(raw_target):
            return None
        lexical_target = os.path.normpath(os.path.join(str(DEFAULT_SYS_CLASS_NET), raw_target))
        devices_prefix = f"{DEFAULT_SYS_DEVICES}{os.sep}"
        if not lexical_target.startswith(devices_prefix):
            return None
        relative_target = lexical_target[len(devices_prefix) :]
        if sys_devices.fd is None:
            return None
        interface_fd = _open_directory_beneath(sys_devices.fd, relative_target)
        final = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
        final_target = os.readlink(name, dir_fd=root.fd)
        if (
            (final.st_dev, final.st_ino) != (before.st_dev, before.st_ino)
            or final_target != raw_target
        ):
            os.close(interface_fd)
            raise SystemExit(f"network metadata symlink changed during discovery: {name!r}")
        return interface_fd
    except SystemExit:
        raise
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        raise SystemExit(f"network metadata symlink changed or became unsafe: {name!r}") from exc


def discover_network_interfaces(
    root: MetadataRoot, sys_devices: MetadataRoot | None = None
) -> list[dict[str, Any]]:
    interfaces: list[dict[str, Any]] = []
    if root.fd is None:
        root.verify_current()
        return interfaces
    try:
        entries = sorted(os.listdir(root.fd))
    except (FileNotFoundError, PermissionError, OSError):
        root.verify_current()
        return interfaces

    for name in entries:
        interface_fd = _open_sysfs_interface_directory(root, sys_devices, name)
        if interface_fd is None:
            continue
        try:
            interface_type = _read_regular_at(interface_fd, "type")
            interfaces.append(
                {
                    "name": name,
                    "kind": (
                        "can"
                        if interface_type == ARPHRD_CAN
                        else "non_can" if interface_type is not None else "unknown"
                    ),
                    "arphrd_type": interface_type,
                    "operstate": _read_regular_at(interface_fd, "operstate"),
                    "carrier": _read_regular_at(interface_fd, "carrier"),
                    "mtu": _read_regular_at(interface_fd, "mtu"),
                }
            )
        finally:
            os.close(interface_fd)
    root.verify_current()
    if sys_devices is not None:
        sys_devices.verify_current()
    return interfaces


def discover_serial_links(root: MetadataRoot) -> list[dict[str, str | None]]:
    links: list[dict[str, str | None]] = []
    if root.fd is None:
        root.verify_current()
        return links
    try:
        entries = sorted(os.listdir(root.fd))
    except (FileNotFoundError, PermissionError, OSError):
        root.verify_current()
        return links

    for name in entries:
        try:
            before = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if not stat.S_ISLNK(before.st_mode):
            links.append({"by_id_name": name, "raw_target": None, "entry_kind": "other"})
            continue
        try:
            raw_target = os.readlink(name, dir_fd=root.fd)
            after = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
            final_target = os.readlink(name, dir_fd=root.fd)
            if (
                (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or final_target != raw_target
            ):
                raise SystemExit(f"serial metadata symlink changed during discovery: {name!r}")
            links.append(
                {"by_id_name": name, "raw_target": raw_target, "entry_kind": "symlink"}
            )
        except SystemExit:
            raise
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise SystemExit(f"serial metadata symlink changed during discovery: {name!r}") from exc
    root.verify_current()
    return links


def _open_metadata_root(
    path: Path,
    default: Path,
    fixture: Path,
    label: str,
) -> MetadataRoot:
    """Allowlist and pin a metadata root using O_DIRECTORY|O_NOFOLLOW."""

    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise SystemExit("metadata discovery requires Linux directory no-follow support")
    candidate = Path(os.path.abspath(os.fspath(path)))
    normalized_default = Path(os.path.abspath(os.fspath(default)))
    normalized_fixture = Path(os.path.abspath(os.fspath(fixture)))
    if candidate not in {normalized_default, normalized_fixture}:
        raise SystemExit(
            f"{label} must be exactly {default} or repository fixture root {fixture}; got {path}"
        )
    try:
        before = os.lstat(candidate)
    except FileNotFoundError:
        return MetadataRoot(candidate, None, None, label, candidate == normalized_fixture)
    except OSError as exc:
        raise SystemExit(f"cannot inspect {label} {candidate}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise SystemExit(f"refusing non-directory or symlink {label} {candidate}")
    try:
        fd = os.open(
            candidate,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise SystemExit(f"cannot safely open {label} {candidate}: {exc}") from exc
    opened = os.fstat(fd)
    identity = (opened.st_dev, opened.st_ino)
    if not stat.S_ISDIR(opened.st_mode) or identity != (before.st_dev, before.st_ino):
        os.close(fd)
        raise SystemExit(f"{label} changed while being opened: {candidate}")
    root = MetadataRoot(candidate, fd, identity, label, candidate == normalized_fixture)
    root.verify_current()
    return root


def _display_path(path: Path, include_sensitive: bool) -> str:
    return str(path) if include_sensitive else path.relative_to(REPO_ROOT).as_posix()


def _report_interfaces(interfaces: list[dict[str, Any]], include_sensitive: bool) -> Any:
    if include_sensitive:
        return interfaces
    return {
        "total_count": len(interfaces),
        "can_count": sum(item["kind"] == "can" for item in interfaces),
        "non_can_count": sum(item["kind"] == "non_can" for item in interfaces),
        "unknown_kind_count": sum(item["kind"] == "unknown" for item in interfaces),
    }


def _report_serial_links(
    links: list[dict[str, str | None]], include_sensitive: bool
) -> Any:
    if include_sensitive:
        return links
    return {
        "total_count": len(links),
        "symlink_count": sum(item["entry_kind"] == "symlink" for item in links),
        "other_entry_count": sum(item["entry_kind"] == "other" for item in links),
    }


def _reject_nonfinite_constant(token: str) -> None:
    raise ValueError(f"non-standard/non-finite JSON constant is forbidden: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number is forbidden: {token}")
    return value


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved, identity = _validated_repo_json_path(path, label)
    try:
        value = json.loads(
            _open_regular_nofollow(
                resolved,
                max_bytes=MAX_JSON_BYTES,
                expected_identity=identity,
            ),
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
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
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    finite_numeric = numeric and (
        isinstance(value, int) or math.isfinite(value)
    )
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": finite_numeric,
    }.get(expected, True)
    if not type_ok:
        issues.append(path or "$")
        return issues
    if "const" in schema and value != schema["const"]:
        issues.append(path or "$")
    if "enum" in schema and value not in schema["enum"]:
        issues.append(path or "$")
    if numeric and not finite_numeric:
        issues.append(path or "$")
    if finite_numeric:
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(path or "$")
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(path or "$")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            issues.append(path or "$")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
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
        help=(
            "allowlisted stable-link directory; overrides are limited to the repository "
            "fixture root"
        ),
    )
    parser.add_argument(
        "--include-sensitive-local-metadata",
        action="store_true",
        help=(
            "show exact host time, platform/kernel, interface/by-id names, targets, and "
            "absolute paths; keep output local and never commit it"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, manifest_path = _load_json(args.manifest, "manifest")
    capture, capture_path = _load_json(args.capture, "capture")
    manifest_schema, _ = _load_json(DEFAULT_MANIFEST_SCHEMA, "manifest schema")
    capture_schema, _ = _load_json(DEFAULT_CAPTURE_SCHEMA, "capture schema")
    sys_class_net: MetadataRoot | None = None
    serial_by_id: MetadataRoot | None = None
    sys_devices = None
    try:
        sys_class_net = _open_metadata_root(
            args.sys_class_net, DEFAULT_SYS_CLASS_NET, FIXTURE_SYS_CLASS_NET, "sys-class-net"
        )
        serial_by_id = _open_metadata_root(
            args.serial_by_id, DEFAULT_SERIAL_BY_ID, FIXTURE_SERIAL_BY_ID, "serial-by-id"
        )
        if not sys_class_net.fixture and sys_class_net.fd is not None:
            sys_devices = _open_metadata_root(
                DEFAULT_SYS_DEVICES,
                DEFAULT_SYS_DEVICES,
                FIXTURE_SYS_CLASS_NET,
                "sys-devices",
            )
        interfaces = discover_network_interfaces(sys_class_net, sys_devices)
        serial_links = discover_serial_links(serial_by_id)
    finally:
        if sys_devices is not None:
            sys_devices.close()
        if serial_by_id is not None:
            serial_by_id.close()
        if sys_class_net is not None:
            sys_class_net.close()
    include_sensitive = bool(args.include_sensitive_local_metadata)
    manifest_placeholders = find_placeholders(manifest)
    capture_placeholders = find_placeholders(capture)
    manifest_schema_issues = find_schema_issues(
        manifest, manifest_schema, manifest_schema
    )
    capture_schema_issues = find_schema_issues(capture, capture_schema, capture_schema)

    report = {
        "discovery_version": "asymmetric25-host-discovery/v3",
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
            "network_interfaces": _report_interfaces(interfaces, include_sensitive),
            "stable_serial_links": _report_serial_links(serial_links, include_sensitive),
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
                "structural/type/required/const/enum/minLength/array-length/finite-number/"
                "numeric-bound checks only; "
                "not a replacement for validate_asymmetric_hardware_manifest.py"
            ),
            "manifest_training_ready": manifest.get("readiness", {}).get("training") is True,
            "manifest_deployment_ready": manifest.get("readiness", {}).get("deployment") is True,
            "capture_approved": capture.get("approval", {}).get("approved") is True,
        },
        "discovery_interpretation": {
            "empty_candidates_prove_hardware_absence": False,
            "candidate_can_interface_count": sum(
                item["kind"] == "can" for item in interfaces
            ),
            "candidate_serial_link_count": len(serial_links),
            **(
                {
                    "candidate_can_interfaces": [
                        item["name"] for item in interfaces if item["kind"] == "can"
                    ],
                    "candidate_serial_links": [
                        item["by_id_name"] for item in serial_links
                    ],
                }
                if include_sensitive
                else {}
            ),
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
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

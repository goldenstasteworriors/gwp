#!/usr/bin/env python3
"""Verify the exact raw LIBERO demonstrations used for cache provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "libero-raw-demo-files-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "libero_spatial_raw_demos.json",
    )
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="Check file presence and exact size without computing SHA256.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected raw-demo manifest schema: {path}")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Raw-demo manifest has no files: {path}")
    return value


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = load_manifest(manifest_path)

    missing: list[str] = []
    size_mismatches: list[dict[str, int | str]] = []
    hash_mismatches: list[dict[str, str]] = []
    verified: list[dict[str, int | str]] = []
    for record in manifest["files"]:
        name = str(record["name"])
        expected_size = int(record["size"])
        expected_hash = str(record["sha256"])
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            size_mismatches.append(
                {"name": name, "expected": expected_size, "actual": actual_size}
            )
            continue
        actual_hash = None if args.size_only else sha256_file(path)
        if actual_hash is not None and actual_hash != expected_hash:
            hash_mismatches.append(
                {"name": name, "expected": expected_hash, "actual": actual_hash}
            )
            continue
        verified.append(
            {
                "name": name,
                "size": actual_size,
                "sha256": expected_hash if actual_hash is None else actual_hash,
            }
        )

    passed = not missing and not size_mismatches and not hash_mismatches
    result = {
        "status": "passed" if passed else "incomplete",
        "mode": "size-only" if args.size_only else "size-and-sha256",
        "root": str(root),
        "manifest": str(manifest_path),
        "source": manifest.get("source"),
        "expected_files": len(manifest["files"]),
        "verified_files": len(verified),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "verified": verified,
    }
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

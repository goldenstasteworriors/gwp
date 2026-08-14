#!/usr/bin/env python3
"""Migrate a physically valid v4 mask cache to explicit v5 action arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from libero_action_encoding import (  # noqa: E402
    POLICY_ACTION_ENCODING,
    POLICY_TO_SIMULATOR_CONVERSION,
    SIMULATOR_ACTION_ENCODING,
    lerobot_action_to_libero,
)


SOURCE_SCHEMA = "libero-counterfactual-robot-mask-v4"
TARGET_SCHEMA = "libero-counterfactual-robot-mask-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(
            f"Expected {SOURCE_SCHEMA}, got {manifest.get('schema_version')!r}"
        )
    expected = {
        "policy_action_encoding": POLICY_ACTION_ENCODING,
        "simulator_action_encoding": SIMULATOR_ACTION_ENCODING,
        "policy_to_simulator_conversion": POLICY_TO_SIMULATOR_CONVERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unexpected {key} in {source / 'manifest.json'}")
    records = manifest.get("episodes")
    if not isinstance(records, list) or not records:
        raise ValueError("Source manifest contains no episodes")
    if int(manifest.get("num_episodes", -1)) != len(records):
        raise ValueError("Source manifest episode count is inconsistent")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if os.stat(source).st_dev != os.stat(output.parent).st_dev:
        raise ValueError("Source and migrated cache must stay on the same filesystem")

    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, source)
    output.mkdir()
    episodes_output = output / "episodes"
    episodes_output.mkdir()

    migrated_records: list[dict[str, Any]] = []
    try:
        for source_record in manifest["episodes"]:
            record = dict(source_record)
            relative = Path(str(record["cache_file"]))
            source_cache = (source / relative).resolve()
            try:
                source_cache.relative_to(source)
            except ValueError as error:
                raise ValueError(f"Cache file escapes source root: {source_cache}") from error
            if sha256_file(source_cache) != record.get("cache_sha256"):
                raise ValueError(f"Source cache hash mismatch: {source_cache}")
            with np.load(source_cache, allow_pickle=False) as archive:
                if "raw_actions" not in archive or "executed_actions" not in archive:
                    raise ValueError(f"v4 action arrays missing: {source_cache}")
                if (
                    "policy_actions_lerobot" in archive
                    or "simulator_actions" in archive
                ):
                    raise ValueError(f"Source already contains v5 action arrays: {source_cache}")
                policy_actions = archive["raw_actions"].copy()
                simulator_actions = archive["executed_actions"].copy()
                if policy_actions.shape != (48, 7) or simulator_actions.shape != (48, 7):
                    raise ValueError(f"Unexpected action shape: {source_cache}")
                expected_simulator = lerobot_action_to_libero(policy_actions)
                if not np.array_equal(simulator_actions, expected_simulator):
                    raise ValueError(f"Simulator action conversion mismatch: {source_cache}")
                arrays = {
                    key: archive[key].copy()
                    for key in archive.files
                    if key not in {"raw_actions", "executed_actions"}
                }
            arrays["policy_actions_lerobot"] = policy_actions
            arrays["simulator_actions"] = simulator_actions
            target_cache = output / relative
            target_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target_cache, **arrays)
            record["cache_sha256"] = sha256_file(target_cache)
            migrated_records.append(record)

        migrated = dict(manifest)
        migrated.update(
            {
                "schema_version": TARGET_SCHEMA,
                "policy_action_array": "policy_actions_lerobot",
                "simulator_action_array": "simulator_actions",
                "num_episodes": len(migrated_records),
                "episodes": migrated_records,
                "migration": {
                    "source_root": str(source),
                    "source_manifest_sha256": sha256_file(source_manifest_path),
                    "operation": (
                        "rename raw_actions to policy_actions_lerobot and "
                        "executed_actions to simulator_actions without changing values"
                    ),
                },
            }
        )
        incomplete = output / ".manifest.json.incomplete"
        incomplete.write_text(
            json.dumps(jsonable(migrated), indent=2) + "\n", encoding="utf-8"
        )
        incomplete.replace(output / "manifest.json")
    except BaseException:
        # The output did not exist before this command and contains only files
        # created by this migration, so cleanup cannot remove user-owned data.
        shutil.rmtree(output)
        raise

    print(json.dumps({"output": str(output), "num_episodes": len(migrated_records)}))


if __name__ == "__main__":
    main()

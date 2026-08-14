#!/usr/bin/env python3
"""Merge independently rendered LIBERO mask-cache shards with hash checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "libero-counterfactual-robot-mask-v4"
VARIABLE_MANIFEST_KEYS = {
    "episodes",
    "num_episodes",
    "raw_demo_files",
    "episode_selection_file",
    "episode_selection_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_episodes(path: Path) -> list[int]:
    episodes = [int(token) for token in path.read_text(encoding="utf-8").split()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError(f"Episode file must be non-empty and unique: {path}")
    return episodes


def load_manifest(shard: Path) -> dict[str, Any]:
    manifest_path = shard / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema in {manifest_path}")
    if int(manifest.get("num_episodes", -1)) != len(manifest.get("episodes", [])):
        raise ValueError(f"Invalid episode count in {manifest_path}")
    return manifest


def main() -> None:
    args = parse_args()
    shards = [path.resolve() for path in args.shards]
    if len(shards) != len(set(shards)):
        raise ValueError("Duplicate shard paths")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    requested = read_episodes(args.episodes_file)
    manifests = [load_manifest(shard) for shard in shards]
    reference = manifests[0]
    invariant_keys = set(reference) - VARIABLE_MANIFEST_KEYS
    for shard, manifest in zip(shards[1:], manifests[1:]):
        if set(manifest) - VARIABLE_MANIFEST_KEYS != invariant_keys:
            raise ValueError(f"Manifest key mismatch in shard {shard}")
        for key in invariant_keys:
            if manifest[key] != reference[key]:
                raise ValueError(f"Manifest field {key!r} differs in shard {shard}")

    episode_rows: dict[int, tuple[Path, dict[str, Any]]] = {}
    raw_demo_files: dict[int, dict[str, Any]] = {}
    for shard, manifest in zip(shards, manifests):
        for raw_demo in manifest["raw_demo_files"]:
            task_index = int(raw_demo["task_index"])
            if task_index in raw_demo_files and raw_demo_files[task_index] != raw_demo:
                raise ValueError(f"Raw-demo provenance differs for task {task_index}")
            raw_demo_files[task_index] = raw_demo
        for row in manifest["episodes"]:
            episode = int(row["episode_index"])
            if episode in episode_rows:
                raise ValueError(f"Episode {episode} appears in multiple shards")
            relative = Path(row["cache_file"])
            if (
                relative.is_absolute()
                or relative.parts[:1] != ("episodes",)
                or ".." in relative.parts
            ):
                raise ValueError(f"Unsafe cache path in shard {shard}: {relative}")
            source = shard / relative
            if not source.is_file() or sha256_file(source) != row["cache_sha256"]:
                raise ValueError(f"Cache hash mismatch for episode {episode}: {source}")
            episode_rows[episode] = (source, row)

    if set(episode_rows) != set(requested):
        missing = sorted(set(requested) - set(episode_rows))
        extra = sorted(set(episode_rows) - set(requested))
        raise ValueError(f"Shard coverage mismatch; missing={missing}, extra={extra}")

    args.output_dir.mkdir(parents=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir()
    merged_rows: list[dict[str, Any]] = []
    for episode in requested:
        source, row = episode_rows[episode]
        destination = episodes_dir / f"episode_{episode:06d}.npz"
        shutil.copy2(source, destination)
        if sha256_file(destination) != row["cache_sha256"]:
            raise RuntimeError(f"Copied cache hash mismatch for episode {episode}")
        merged_rows.append(row)

    merged = dict(reference)
    merged["raw_demo_files"] = [raw_demo_files[key] for key in sorted(raw_demo_files)]
    merged["episode_selection_file"] = str(args.episodes_file.resolve())
    merged["episode_selection_sha256"] = sha256_file(args.episodes_file)
    merged["num_episodes"] = len(merged_rows)
    merged["episodes"] = merged_rows
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "num_shards": len(shards),
                "num_episodes": len(merged_rows),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    )


if __name__ == "__main__":
    main()

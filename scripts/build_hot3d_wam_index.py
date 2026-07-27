#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys
import tarfile
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from world_action_model.datasets.hot3d_wam_dataset import ACTION_DIM, hot3d_action, hot3d_state


REQUIRED_SUFFIXES = (".image.jpg", ".lowdim.npy", ".meta.json")


def split_member_name(name: str) -> tuple[str, str] | None:
    for suffix in REQUIRED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None


def parse_key(key: str) -> tuple[str, int]:
    prefix, marker, frame_text = key.rpartition("_f")
    if not marker or not frame_text.isdigit():
        raise ValueError(f"Unexpected HOT3D sample key: {key}")
    return prefix, int(frame_text)


def init_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY,
            source_root TEXT NOT NULL,
            episode_key TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT '',
            UNIQUE(source_root, episode_key)
        );
        CREATE TABLE frames (
            episode_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            shard_path TEXT NOT NULL,
            image_offset INTEGER NOT NULL,
            image_size INTEGER NOT NULL,
            lowdim_offset INTEGER NOT NULL,
            lowdim_size INTEGER NOT NULL,
            presence INTEGER NOT NULL,
            PRIMARY KEY(episode_id, frame_index)
        );
        CREATE TABLE windows (
            id INTEGER PRIMARY KEY,
            episode_id INTEGER NOT NULL,
            start_frame INTEGER NOT NULL
        );
        CREATE INDEX frame_lookup ON frames(episode_id, frame_index);
        """
    )


def index_shard(connection: sqlite3.Connection, root: Path, shard: Path) -> int:
    records: dict[str, dict] = {}
    with tarfile.open(shard, "r:") as archive:
        for member in archive:
            parsed = split_member_name(member.name)
            if parsed is None or not member.isfile():
                continue
            key, suffix = parsed
            record = records.setdefault(key, {})
            if suffix == ".meta.json":
                payload = archive.extractfile(member).read()
                metadata = json.loads(payload)
                record["presence"] = int(metadata.get("presence", 0))
                record["language"] = str(metadata.get("language", "") or "")
            else:
                field = "image" if suffix == ".image.jpg" else "lowdim"
                record[f"{field}_offset"] = int(member.offset_data)
                record[f"{field}_size"] = int(member.size)

    inserted = 0
    for key, record in records.items():
        if not all(
            name in record
            for name in ("image_offset", "image_size", "lowdim_offset", "lowdim_size", "presence")
        ):
            continue
        episode_key, frame_index = parse_key(key)
        connection.execute(
            "INSERT OR IGNORE INTO episodes(source_root, episode_key, language) VALUES(?,?,?)",
            (str(root), episode_key, record.get("language", "")),
        )
        episode_id = connection.execute(
            "SELECT id FROM episodes WHERE source_root=? AND episode_key=?",
            (str(root), episode_key),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO frames(
                episode_id, frame_index, shard_path, image_offset, image_size,
                lowdim_offset, lowdim_size, presence
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                episode_id,
                frame_index,
                str(shard),
                record["image_offset"],
                record["image_size"],
                record["lowdim_offset"],
                record["lowdim_size"],
                record["presence"],
            ),
        )
        inserted += 1
    connection.commit()
    return inserted


def build_windows(connection: sqlite3.Connection, horizon: int) -> int:
    count = 0
    for (episode_id,) in connection.execute("SELECT id FROM episodes ORDER BY id"):
        frame_indices = [
            int(row[0])
            for row in connection.execute(
                "SELECT frame_index FROM frames WHERE episode_id=? ORDER BY frame_index",
                (episode_id,),
            )
        ]
        frame_set = set(frame_indices)
        starts = [frame for frame in frame_indices if all(frame + step in frame_set for step in range(horizon + 1))]
        connection.executemany(
            "INSERT INTO windows(episode_id, start_frame) VALUES(?,?)",
            [(episode_id, start) for start in starts],
        )
        count += len(starts)
    connection.commit()
    return count


def read_lowdim(fd_cache: dict[str, int], row: tuple) -> np.ndarray:
    shard_path, offset, size = str(row[0]), int(row[1]), int(row[2])
    fd = fd_cache.get(shard_path)
    if fd is None:
        fd = os.open(shard_path, os.O_RDONLY)
        fd_cache[shard_path] = fd
    return np.load(io.BytesIO(os.pread(fd, size, offset)), allow_pickle=False).astype(np.float32, copy=False)


def summarize(values: list[list[float]]) -> dict:
    result = {name: [] for name in ("mean", "std", "min", "max", "q01", "q99")}
    for dim_values in values:
        array = np.asarray(dim_values if dim_values else [0.0], dtype=np.float64)
        result["mean"].append(float(array.mean()))
        result["std"].append(float(max(array.std(), 1e-6)))
        result["min"].append(float(array.min()))
        result["max"].append(float(array.max()))
        result["q01"].append(float(np.quantile(array, 0.01)))
        result["q99"].append(float(np.quantile(array, 0.99)))
    return result


def compute_stats(connection: sqlite3.Connection) -> dict:
    state_values = [[] for _ in range(ACTION_DIM)]
    action_values = [[] for _ in range(ACTION_DIM)]
    fd_cache: dict[str, int] = {}
    try:
        for (episode_id,) in connection.execute("SELECT id FROM episodes ORDER BY id"):
            rows = connection.execute(
                """
                SELECT shard_path, lowdim_offset, lowdim_size, presence
                FROM frames WHERE episode_id=? ORDER BY frame_index
                """,
                (episode_id,),
            ).fetchall()
            if not rows:
                continue
            lowdims = [read_lowdim(fd_cache, row[:3]) for row in rows]
            presences = [int(row[3]) for row in rows]
            for lowdim, presence in zip(lowdims, presences):
                state, mask = hot3d_state(lowdim, presence)
                for dim in np.flatnonzero(mask):
                    state_values[int(dim)].append(float(state[dim]))
            for current, following, current_presence, following_presence in zip(
                lowdims[:-1], lowdims[1:], presences[:-1], presences[1:]
            ):
                action, mask = hot3d_action(current, following, current_presence, following_presence)
                for dim in np.flatnonzero(mask):
                    action_values[int(dim)].append(float(action[dim]))
    finally:
        for fd in fd_cache.values():
            os.close(fd)
    return {
        "model_dim": ACTION_DIM,
        "layout": {
            "base_velocity": [0, 3],
            "left_relative_wrist_pose": [3, 12],
            "right_relative_wrist_pose": [12, 21],
            "gripper": [21, 23],
            "left_fingertips": [23, 38],
            "right_fingertips": [38, 53],
        },
        "norm_stats": {
            "observation.state": summarize(state_values),
            "action": summarize(action_values),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a random-access HOT3D WAM index and 53-D normalization stats.")
    parser.add_argument("--data-root", action="append", required=True, help="HOT3D WebDataset directory; repeat for multiple roots.")
    parser.add_argument("--index", required=True, help="Output SQLite index.")
    parser.add_argument("--norm-stats", required=True, help="Output normalization JSON.")
    parser.add_argument("--action-horizon", type=int, default=48)
    args = parser.parse_args()

    roots = [Path(path).resolve() for path in args.data_root]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    index_path = Path(args.index).resolve()
    norm_path = Path(args.norm_stats).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {index_path}")

    connection = sqlite3.connect(index_path)
    try:
        init_database(connection)
        total_frames = 0
        for root in roots:
            for shard in sorted(root.glob("*.tar")):
                inserted = index_shard(connection, root, shard)
                total_frames += inserted
                print(f"[index] {shard.name}: {inserted} frames")
        total_windows = build_windows(connection, int(args.action_horizon))
        print(f"[index] total_frames={total_frames} total_windows={total_windows}")
        stats = compute_stats(connection)
        with open(norm_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        connection.execute("PRAGMA optimize")
        connection.commit()
    finally:
        connection.close()
    print(f"[done] index={index_path}")
    print(f"[done] norm_stats={norm_path}")


if __name__ == "__main__":
    main()

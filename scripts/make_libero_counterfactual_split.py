#!/usr/bin/env python3
"""Create a deterministic task-stratified split for counterfactual mask caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "libero-counterfactual-split-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=20)
    parser.add_argument("--heldout-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
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


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    frames = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["episode_index", "frame_index", "task_index"],
            )
            for path in parquet_paths
        ],
        ignore_index=True,
    )
    grouped = frames.groupby("episode_index", sort=True)
    episodes = grouped.agg(
        task_index=("task_index", "first"),
        min_frame_index=("frame_index", "min"),
        num_frames=("frame_index", "size"),
    ).reset_index()
    inconsistent = grouped["task_index"].nunique()
    inconsistent = inconsistent[inconsistent != 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Episodes contain multiple task indices: {inconsistent.index.tolist()}"
        )
    if (episodes["min_frame_index"] != 0).any():
        invalid = episodes.loc[episodes["min_frame_index"] != 0, "episode_index"].tolist()
        raise ValueError(f"Episodes do not start at frame 0: {invalid}")
    return episodes


def make_split(
    episodes: pd.DataFrame,
    *,
    train_per_task: int,
    heldout_per_task: int,
    seed: int,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    if train_per_task <= 0 or heldout_per_task <= 0:
        raise ValueError("train-per-task and heldout-per-task must both be positive")

    train: list[int] = []
    heldout: list[int] = []
    task_rows: list[dict[str, Any]] = []
    for task_index, task_episodes in episodes.groupby("task_index", sort=True):
        episode_indices = np.asarray(
            sorted(int(value) for value in task_episodes["episode_index"]),
            dtype=np.int64,
        )
        required = train_per_task + heldout_per_task
        if len(episode_indices) < required:
            raise ValueError(
                f"Task {int(task_index)} has {len(episode_indices)} episodes, need {required}"
            )
        task_rng = np.random.default_rng(np.random.SeedSequence([seed, int(task_index)]))
        shuffled = task_rng.permutation(episode_indices)
        task_train = sorted(int(value) for value in shuffled[:train_per_task])
        task_heldout = sorted(
            int(value) for value in shuffled[train_per_task : train_per_task + heldout_per_task]
        )
        train.extend(task_train)
        heldout.extend(task_heldout)
        task_rows.append(
            {
                "task_index": int(task_index),
                "available_episodes": len(episode_indices),
                "train_episodes": task_train,
                "heldout_episodes": task_heldout,
            }
        )

    train = sorted(train)
    heldout = sorted(heldout)
    overlap = sorted(set(train) & set(heldout))
    if overlap:
        raise AssertionError(f"Train/held-out overlap: {overlap}")
    return train, heldout, task_rows


def write_episode_list(path: Path, episodes: list[int]) -> None:
    path.write_text("".join(f"{episode}\n" for episode in episodes), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)

    episode_table = load_episode_table(args.dataset_root)
    train, heldout, tasks = make_split(
        episode_table,
        train_per_task=args.train_per_task,
        heldout_per_task=args.heldout_per_task,
        seed=args.seed,
    )
    selected = set(train) | set(heldout)
    selected_rows = episode_table[episode_table["episode_index"].isin(selected)]
    episode_lengths = {
        str(int(row.episode_index)): int(row.num_frames)
        for row in selected_rows.itertuples(index=False)
    }

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train_episodes.txt"
    heldout_path = args.output_dir / "heldout_episodes.txt"
    write_episode_list(train_path, train)
    write_episode_list(heldout_path, heldout)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite": args.suite,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_info_sha256": sha256_file(info_path),
        "seed": args.seed,
        "train_per_task": args.train_per_task,
        "heldout_per_task": args.heldout_per_task,
        "num_train": len(train),
        "num_heldout": len(heldout),
        "train_episodes_file": train_path.name,
        "heldout_episodes_file": heldout_path.name,
        "train_episodes": train,
        "heldout_episodes": heldout,
        "episode_lengths": episode_lengths,
        "tasks": tasks,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "suite": args.suite,
                "num_tasks": len(tasks),
                "num_train": len(train),
                "num_heldout": len(heldout),
                "overlap": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()

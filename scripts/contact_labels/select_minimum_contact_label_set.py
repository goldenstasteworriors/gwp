#!/usr/bin/env python3
"""Inventory Ego episodes, make diagnostic splits, and propose a minimum review set."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import h5py

from contact_label_io_v1_1 import INVENTORY_FIELDS, write_csv


SEEDS = (6666, 7777, 8888)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length-source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--length-key", default="action_eef")
    parser.add_argument("--minimum-count", type=int, default=12)
    parser.add_argument("--existing-hint-root", type=Path)
    return parser.parse_args()


def episode_id(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"cannot parse episode ID from {path}")
    return int(match.group(1))


def stable_rank(seed: int, episode: int) -> str:
    return hashlib.sha256(f"{seed}:{episode}".encode()).hexdigest()


def split_assignments(episodes: list[int], seed: int) -> dict[int, str]:
    ordered = sorted(episodes, key=lambda episode: stable_rank(seed, episode))
    count = len(ordered)
    train_end = round(0.70 * count)
    val_end = train_end + round(0.15 * count)
    return {
        episode: "train" if index < train_end else "val" if index < val_end else "test"
        for index, episode in enumerate(ordered)
    }


def length_strata(lengths: dict[int, int]) -> dict[int, int]:
    ordered = sorted(lengths, key=lambda episode: (lengths[episode], episode))
    return {episode: min(3, (rank * 4) // len(ordered)) for rank, episode in enumerate(ordered)}


def main() -> None:
    args = parse_args()
    if args.minimum_count < 4:
        raise ValueError("minimum-count must be at least four")
    paths = sorted(args.length_source_dir.glob("episode_*.hdf5"), key=episode_id)
    if not paths:
        raise FileNotFoundError(f"no episode HDF5 files under {args.length_source_dir}")
    lengths = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            if args.length_key not in handle:
                raise KeyError(f"{path}: missing {args.length_key}")
            lengths[episode_id(path)] = int(handle[args.length_key].shape[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = [{
        "episode_id": episode,
        "length_frames": length,
        "subject_id": "",
        "session_id": "",
        "operator_id": "",
        "source_object_instance_id": "",
        "target_object_instance_id": "",
        "liquid_id": "",
        "collection_batch_id": "",
        "metadata_status": "missing_required_group_metadata",
    } for episode, length in sorted(lengths.items())]
    write_csv(args.output_dir / "episode_inventory.csv", INVENTORY_FIELDS, inventory_rows)

    assignments = {seed: split_assignments(list(lengths), seed) for seed in SEEDS}
    split_fields = (
        "episode_id", "length_frames", "seed_6666_split", "seed_7777_split", "seed_8888_split",
        "split_unit", "split_purpose", "ood_eligible", "strategy_claim_allowed", "metadata_status",
    )
    split_rows = [{
        "episode_id": episode,
        "length_frames": lengths[episode],
        **{f"seed_{seed}_split": assignments[seed][episode] for seed in SEEDS},
        "split_unit": "episode",
        "split_purpose": "engineering_check_only",
        "ood_eligible": "false",
        "strategy_claim_allowed": "false",
        "metadata_status": "subject_session_object_missing; diagnostic_episode_split_only",
    } for episode in sorted(lengths)]
    write_csv(args.output_dir / "episode_split_manifest.csv", split_fields, split_rows)

    strata = length_strata(lengths)
    per_stratum = max(1, (args.minimum_count + 3) // 4)
    selected: list[int] = []
    for stratum in range(4):
        candidates = [episode for episode in lengths if strata[episode] == stratum]
        candidates.sort(key=lambda episode: stable_rank(SEEDS[0], episode))
        selected.extend(candidates[:per_stratum])
    selected = selected[:args.minimum_count]
    hint_episode = None
    if args.existing_hint_root is not None:
        hint_paths = sorted(args.existing_hint_root.glob("episode_*_contact_annotation.json"))
        if hint_paths:
            hint_episode = episode_id(hint_paths[0])
            if hint_episode in lengths and hint_episode not in selected:
                selected[-1] = hint_episode
    selected = sorted(set(selected), key=lambda episode: (strata[episode], stable_rank(SEEDS[0], episode)))
    proposal_fields = (
        "selection_rank", "episode_id", "length_frames", "length_stratum",
        "selection_seed", "existing_timing_hint", "proposal_only",
        "requires_independent_a", "requires_independent_b", "metadata_blocker",
    )
    proposal_rows = [{
        "selection_rank": rank,
        "episode_id": episode,
        "length_frames": lengths[episode],
        "length_stratum": strata[episode],
        "selection_seed": SEEDS[0],
        "existing_timing_hint": "present_not_ground_truth" if episode == hint_episode else "none",
        "proposal_only": "true",
        "requires_independent_a": "true",
        "requires_independent_b": "true",
        "metadata_blocker": "cannot balance subject/session/object until metadata are collected",
    } for rank, episode in enumerate(selected)]
    write_csv(args.output_dir / "minimum_annotation_set.csv", proposal_fields, proposal_rows)
    print(f"episodes={len(lengths)} minimum_set={len(selected)} seeds={SEEDS}")


if __name__ == "__main__":
    main()

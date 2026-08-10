#!/usr/bin/env python3
"""Strict static validator for contact_segment_v1_1 NPZ sidecars."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from contact_label_io_v1_1 import (
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    load_inventory,
    sha256_file,
)


REQUIRED_KEYS = (
    "contact_target", "contact_loss_mask", "contact_confidence", "contact_condition_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-dir", required=True, type=Path)
    parser.add_argument("--inventory-csv", required=True, type=Path)
    parser.add_argument("--coverage-csv", type=Path)
    parser.add_argument("--segments-csv", type=Path)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def require_binary(array: np.ndarray, name: str, path: Path) -> None:
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{path}: {name} must be binary")


def validate_archive(path: Path, length: int, num_queries: int) -> set[tuple[int, int]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(REQUIRED_KEYS):
            raise ValueError(f"{path}: keys must be exactly {REQUIRED_KEYS}, got {archive.files}")
        arrays = {key: archive[key] for key in REQUIRED_KEYS}
    expected = (length, num_queries, SEGMENT_TARGET_DIM)
    for key, array in arrays.items():
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"{path}: {key} expected float32 {expected}, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {key} contains non-finite values")
    target, mask = arrays["contact_target"], arrays["contact_loss_mask"]
    confidence, condition = arrays["contact_confidence"], arrays["contact_condition_mask"]
    require_binary(mask, "contact_loss_mask", path)
    require_binary(condition, "contact_condition_mask", path)
    if not np.array_equal(mask, condition):
        raise ValueError(f"{path}: condition mask must exactly equal field supervision mask")
    if np.any((target < 0.0) | (target > 1.0)) or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError(f"{path}: targets/confidence must lie in [0,1]")
    if np.any(target[mask == 0] != 0.0) or np.any(confidence[mask == 0] != 0.0):
        raise ValueError(f"{path}: unsupervised target/confidence entries must be zero")
    if np.any(confidence[mask == 1] <= 0.0):
        raise ValueError(f"{path}: every supervised field needs positive confidence")
    presence = SEGMENT_FIELD_SLICES["presence"]
    presence_target = target[..., presence].squeeze(-1)
    presence_mask = mask[..., presence].squeeze(-1)
    require_binary(presence_target, "presence target", path)
    for field in ("side", "site", "object_role", "contact_mode"):
        field_slice = SEGMENT_FIELD_SLICES[field]
        field_mask = mask[..., field_slice]
        if not np.all(field_mask == field_mask[..., :1]):
            raise ValueError(f"{path}: {field} mask must cover the whole one-hot group")
        active = field_mask[..., 0] == 1
        if np.any(np.abs(target[..., field_slice].sum(axis=-1)[active] - 1.0) > 1e-6):
            raise ValueError(f"{path}: supervised {field} must be one-hot")
    nonpresence_mask = mask.copy()
    nonpresence_mask[..., presence] = 0.0
    if np.any((presence_target == 0) & np.any(nonpresence_mask != 0, axis=-1)):
        raise ValueError(f"{path}: negative/unknown queries cannot supervise segment attributes")
    start = SEGMENT_FIELD_SLICES["start_tau"]
    end = SEGMENT_FIELD_SLICES["end_tau"]
    both = (mask[..., start].squeeze(-1) == 1) & (mask[..., end].squeeze(-1) == 1)
    if np.any(target[..., start].squeeze(-1)[both] > target[..., end].squeeze(-1)[both]):
        raise ValueError(f"{path}: start_tau exceeds end_tau")
    for field in ("left_censored", "right_censored"):
        field_slice = SEGMENT_FIELD_SLICES[field]
        require_binary(target[..., field_slice][mask[..., field_slice] == 1], field, path)
    left = SEGMENT_FIELD_SLICES["left_censored"]
    right = SEGMENT_FIELD_SLICES["right_censored"]
    if not np.array_equal(mask[..., start], mask[..., left]):
        raise ValueError(f"{path}: start_tau and left_censored must be jointly known or unknown")
    if not np.array_equal(mask[..., end], mask[..., right]):
        raise ValueError(f"{path}: end_tau and right_censored must be jointly known or unknown")
    left_active = (mask[..., left].squeeze(-1) == 1) & (target[..., left].squeeze(-1) == 1)
    right_active = (mask[..., right].squeeze(-1) == 1) & (target[..., right].squeeze(-1) == 1)
    if np.any(mask[..., start].squeeze(-1)[left_active] != 1) or np.any(target[..., start].squeeze(-1)[left_active] != 0):
        raise ValueError(f"{path}: left_censored=1 requires supervised start_tau=0")
    if np.any(mask[..., end].squeeze(-1)[right_active] != 1) or np.any(target[..., end].squeeze(-1)[right_active] != 1):
        raise ValueError(f"{path}: right_censored=1 requires supervised end_tau=1")
    return {tuple(map(int, index)) for index in np.argwhere(presence_target == 1)}


def main() -> None:
    args = parse_args()
    inventory = load_inventory(args.inventory_csv)
    meta_path = args.sidecar_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SEGMENT_SCHEMA_VERSION or meta.get("target_dim") != SEGMENT_TARGET_DIM:
        raise ValueError("meta.json schema/target_dim mismatch")
    num_queries = int(meta["num_queries"])
    if num_queries != 12 or meta.get("overflow_policy") != "hard_error_no_drop" or int(meta.get("dropped_segments", -1)) != 0:
        raise ValueError("production sidecars require Q=12, hard-error overflow, and zero dropped segments")
    expected_files = {f"episode_{episode:06d}.npz" for episode in inventory}
    actual_files = {path.name for path in args.sidecar_dir.glob("episode_*.npz")}
    if actual_files != expected_files:
        raise ValueError(f"episode NPZ set mismatch; missing={sorted(expected_files-actual_files)}, extra={sorted(actual_files-expected_files)}")
    positive_queries: set[tuple[int, int, int]] = set()
    for episode, item in sorted(inventory.items()):
        positives = validate_archive(args.sidecar_dir / f"episode_{episode:06d}.npz", item.length_frames, num_queries)
        positive_queries.update((episode, frame, query) for frame, query in positives)
    provenance_path = args.sidecar_dir / "window_segment_provenance.csv"
    with provenance_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = ["episode_id", "frame_index", "query_index", "segment_id", "onset_event_id", "release_event_id"]
        if reader.fieldnames != expected_header:
            raise ValueError("window_segment_provenance.csv header mismatch")
        provenance_rows = list(reader)
    provenance_queries: set[tuple[int, int, int]] = set()
    physical_ids: set[str] = set()
    for row_number, row in enumerate(provenance_rows, 2):
        key = (int(row["episode_id"]), int(row["frame_index"]), int(row["query_index"]))
        if key in provenance_queries:
            raise ValueError(f"provenance row {row_number}: duplicate window query {key}")
        if key[0] not in inventory or not 0 <= key[1] < inventory[key[0]].length_frames or not 0 <= key[2] < num_queries:
            raise ValueError(f"provenance row {row_number}: out-of-range window query {key}")
        provenance_queries.add(key)
        physical_ids.add(row["segment_id"])
    if provenance_queries != positive_queries:
        raise ValueError("positive NPZ queries and segment provenance are not one-to-one")
    if len(provenance_rows) != int(meta["window_segment_rows"]) or len(physical_ids) != int(meta["unique_physical_segments"]):
        raise ValueError("provenance counts disagree with meta.json")
    supplied = {
        "coverage_csv": args.coverage_csv,
        "segments_csv": args.segments_csv,
        "config": args.config,
        "inventory_csv": args.inventory_csv,
    }
    for key, path in supplied.items():
        if path is not None and meta["input_sha256"].get(key) != sha256_file(path):
            raise ValueError(f"{key} hash differs from build provenance")
    print(json.dumps({"status": "valid", "episodes": len(inventory), "schema": SEGMENT_SCHEMA_VERSION}, indent=2))


if __name__ == "__main__":
    main()

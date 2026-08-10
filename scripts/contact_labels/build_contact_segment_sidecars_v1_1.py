#!/usr/bin/env python3
"""Convert reviewed/adjudicated contact-segment CSVs into v1.1 NPZ sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from contact_label_io_v1_1 import (
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    encode_episode,
    load_coverages,
    load_inventory,
    load_segments,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-csv", required=True, type=Path)
    parser.add_argument("--segments-csv", required=True, type=Path)
    parser.add_argument("--inventory-csv", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("sidecar_schema_version") != SEGMENT_SCHEMA_VERSION:
        raise ValueError("config must select contact_segment_v1_1")
    if config.get("annotation_schema_version") != "contact_segment_annotation_v1_1":
        raise ValueError("config must select contact_segment_annotation_v1_1")
    if config.get("accepted_annotation_role") != "adjudicated":
        raise ValueError("only adjudicated annotations may be built")
    if config.get("overflow_policy") != "hard_error_no_drop":
        raise ValueError("overflow_policy must be hard_error_no_drop")
    horizon = int(config["horizon_frames"])
    num_queries = int(config["num_queries"])
    negative_confidence = float(config["negative_confidence"])
    if horizon <= 0 or num_queries != 12:
        raise ValueError("contact_segment_v1_1 production build requires horizon>0 and num_queries=12")
    if not 0.0 <= negative_confidence <= 1.0:
        raise ValueError("negative_confidence must lie in [0,1]")
    inventory = load_inventory(args.inventory_csv)
    coverages = load_coverages(args.coverage_csv, inventory, config)
    segments = load_segments(args.segments_csv, inventory, coverages, config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_by_episode = {episode: [] for episode in inventory}
    segment_by_episode = {episode: [] for episode in inventory}
    for coverage in coverages:
        coverage_by_episode[coverage.episode_id].append(coverage)
    for segment in segments:
        segment_by_episode[segment.episode_id].append(segment)
    provenance_rows = []
    episode_npz_sha256 = {}
    for episode, item in sorted(inventory.items()):
        arrays, provenance = encode_episode(
            item.length_frames,
            coverage_by_episode[episode],
            segment_by_episode[episode],
            horizon_frames=horizon,
            num_queries=num_queries,
            negative_confidence=negative_confidence,
        )
        destination = args.output_dir / f"episode_{episode:06d}.npz"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(destination)
        partial = destination.with_name(f".{destination.name}.partial")
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        partial.replace(destination)
        episode_npz_sha256[destination.name] = sha256_file(destination)
        for row in provenance:
            provenance_rows.append({"episode_id": episode, **row})
    provenance_path = args.output_dir / "window_segment_provenance.csv"
    write_csv(
        provenance_path,
        ("episode_id", "frame_index", "query_index", "segment_id", "onset_event_id", "release_event_id"),
        provenance_rows,
    )
    provenance_sha256 = sha256_file(provenance_path)
    sidecar_bundle_sha256 = hashlib.sha256(
        json.dumps(
            {
                "episode_npz_sha256": episode_npz_sha256,
                "window_segment_provenance_sha256": provenance_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    meta = {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "target_dim": SEGMENT_TARGET_DIM,
        "field_slices": {key: [value.start, value.stop] for key, value in SEGMENT_FIELD_SLICES.items()},
        "dataset_id": config["dataset_id"],
        "frame_clock": config["frame_clock"],
        "target_fps": config["target_fps"],
        "horizon_frames": horizon,
        "num_queries": num_queries,
        "negative_confidence": negative_confidence,
        "overflow_policy": "hard_error_no_drop",
        "dropped_segments": 0,
        "window_assignment_sort_key": [
            "start_frame", "end_frame", "side_or_empty", "site_or_empty", "segment_id"
        ],
        "episode_lengths": {str(key): value.length_frames for key, value in sorted(inventory.items())},
        "reviewed_coverage_rows": len(coverages),
        "unique_physical_segments": len(segments),
        "window_segment_rows": len(provenance_rows),
        "input_sha256": {
            "coverage_csv": sha256_file(args.coverage_csv),
            "segments_csv": sha256_file(args.segments_csv),
            "inventory_csv": sha256_file(args.inventory_csv),
            "config": sha256_file(args.config),
            "split_manifest": sha256_file(args.split_manifest),
        },
        "episode_npz_sha256": episode_npz_sha256,
        "window_segment_provenance_sha256": provenance_sha256,
        "sidecar_bundle_sha256": sidecar_bundle_sha256,
        "builder_identity": "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py",
        "annotation_contract": {
            "annotation_role": "adjudicated",
            "review_status": "reviewed",
            "label_origin": "human_adjudicated",
            "automatic_or_proposal_ground_truth": False,
        },
        "strategy_claim_allowed": bool(
            config.get("split_policy", {}).get("strategy_claim_allowed", False)
        ),
        "warning": "window_segment_rows are repeated views of unique physical segment_id values and are not independent events",
    }
    write_json(args.output_dir / "meta.json", meta)
    print(json.dumps({"episodes": len(inventory), "segments": len(segments), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

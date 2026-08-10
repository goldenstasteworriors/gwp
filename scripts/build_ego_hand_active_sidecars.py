#!/usr/bin/env python3
"""Build exact 20 Hz Ego hand-tracker-validity sidecars from raw HDF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.ego_dataset_fingerprint import derive_ego_dataset_identity  # noqa: E402
from world_action_model.ego_hand_active_sidecar import (  # noqa: E402
    EGO_HAND_ACTIVE_BUILDER,
    EGO_HAND_ACTIVE_FIELD_ORDER,
    EGO_HAND_ACTIVE_HORIZON,
    EGO_HAND_ACTIVE_SCHEMA_VERSION,
    EGO_HAND_ACTIVE_SEMANTICS,
    EGO_HAND_ACTIVE_START_OFFSET,
    _aggregate_statistics,
    build_hand_active_arrays,
    canonical_sha256,
    deterministic_npz_bytes,
    load_and_validate_hand_active_bundle,
    read_aligned_hand_active,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--raw-hdf5-dir", required=True, type=Path)
    parser.add_argument("--final-hdf5-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_files(root: Path) -> dict[int, Path]:
    result = {}
    for path in root.glob("episode_*.hdf5"):
        match = re.fullmatch(r"episode_([0-9]+)\.hdf5", path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            result[int(match.group(1))] = path
    return result


def _write_exact(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def main() -> None:
    args = parse_args()
    identity = derive_ego_dataset_identity(args.dataset_root)
    raw_root = args.raw_hdf5_dir.expanduser().resolve()
    final_root = args.final_hdf5_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    expected_episodes = set(identity["episode_lengths"])
    raw_files = _episode_files(raw_root)
    final_files = _episode_files(final_root)
    if set(raw_files) != expected_episodes or set(final_files) != expected_episodes:
        raise ValueError("raw/final HDF5 episode set differs from the processed dataset")
    if output_root.exists():
        unexpected = {
            path.name
            for path in output_root.iterdir()
            if path.name != "meta.json"
            and not re.fullmatch(r"episode_[0-9]{6}\.npz", path.name)
        }
        if unexpected:
            raise ValueError(f"refusing output directory with unexpected files: {sorted(unexpected)}")
    output_root.mkdir(parents=True, exist_ok=True)

    episode_hashes = {}
    raw_hashes = {}
    final_hashes = {}
    per_episode = {}
    for episode, frame_count in identity["episode_lengths"].items():
        raw_path = raw_files[episode]
        final_path = final_files[episode]
        current, hand_status, source_audit = read_aligned_hand_active(
            raw_path, final_path
        )
        if len(current) != frame_count:
            raise ValueError(f"episode {episode} hand-active length differs from metadata")
        arrays = build_hand_active_arrays(current)
        destination = output_root / f"episode_{episode:06d}.npz"
        _write_exact(
            destination,
            deterministic_npz_bytes(arrays),
            overwrite=args.overwrite,
        )
        episode_hashes[destination.name] = sha256_file(destination)
        raw_hashes[raw_path.name] = sha256_file(raw_path)
        final_hashes[final_path.name] = sha256_file(final_path)
        per_episode[episode] = (current, hand_status, source_audit)

    expected_outputs = {"meta.json"} | set(episode_hashes)
    existing_outputs = {path.name for path in output_root.iterdir()}
    stale = sorted(existing_outputs - expected_outputs)
    if stale:
        raise ValueError(f"refusing stale hand-active output files: {stale}")
    meta = {
        "schema_version": EGO_HAND_ACTIVE_SCHEMA_VERSION,
        "builder_identity": EGO_HAND_ACTIVE_BUILDER,
        "dataset_id": identity["dataset_id"],
        "dataset_fingerprint_sha256": identity["dataset_fingerprint_sha256"],
        "source_receipt_sha256": identity["source_receipt_sha256"],
        "frame_clock": identity["frame_clock"],
        "target_fps": identity["fps"],
        "horizon_frames": EGO_HAND_ACTIVE_HORIZON,
        "action_start_offset": EGO_HAND_ACTIVE_START_OFFSET,
        "field_order": list(EGO_HAND_ACTIVE_FIELD_ORDER),
        "hand_active_semantics": EGO_HAND_ACTIVE_SEMANTICS,
        "episode_lengths": {
            str(key): int(value) for key, value in identity["episode_lengths"].items()
        },
        "mask_contract": {
            "grip_action_value_loss": "active_at_target_frame_times_0.25",
            "future_edge_padding": "false_no_last_value_hold",
            "transition_auxiliary": "disabled_pair_mask_record_only",
            "transition_pair_mask": "active_at_target_minus_1_and_target",
            "wrist_action_dimensions_0_17": "unaffected",
            "grip_action_dimensions": [18, 19],
        },
        "physical_contact_truth": False,
        "episode_npz_sha256": episode_hashes,
        "sidecar_bundle_sha256": canonical_sha256(episode_hashes),
        "raw_hdf5_sha256": raw_hashes,
        "raw_hdf5_bundle_sha256": canonical_sha256(raw_hashes),
        "final_hdf5_sha256": final_hashes,
        "final_hdf5_bundle_sha256": canonical_sha256(final_hashes),
        "statistics": _aggregate_statistics(per_episode),
        "limitations": [
            "hand_active is tracker validity, not physical contact truth",
            "hand_status is a square-wave closure proxy, not a calibrated aperture",
            "inactive hand_status values remain record data but receive zero grip loss weight",
            "the transition-pair mask is recorded for future explicit ablations and is not consumed",
        ],
    }
    _write_exact(
        output_root / "meta.json",
        (json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        overwrite=args.overwrite,
    )
    receipt = load_and_validate_hand_active_bundle(
        output_root,
        dataset_identity=identity,
        raw_hdf5_dir=raw_root,
        final_hdf5_dir=final_root,
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "output_dir": str(output_root),
                "meta_sha256": receipt["meta_sha256"],
                "sidecar_bundle_sha256": receipt["sidecar_bundle_sha256"],
                "raw_hdf5_bundle_sha256": receipt["raw_hdf5_bundle_sha256"],
                "statistics": receipt["statistics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

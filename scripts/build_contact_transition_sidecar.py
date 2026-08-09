#!/usr/bin/env python3
"""Build frame-aligned contact-transition sidecars from interval annotations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_transition_schema import (  # noqa: E402
    ENUM_VALUES,
    FIELD_SLICES,
    SCHEMA_VERSION,
    TARGET_DIM,
    ContactEvent,
    encode_ordered_events,
)


EPISODE_RE = re.compile(r"episode_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--length-source-dir",
        type=Path,
        default=None,
        help="defaults to SOURCE_ROOT/vla_processed/final_hdf5",
    )
    parser.add_argument("--length-key", default="action_eef")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_id(path: Path) -> int:
    match = EPISODE_RE.search(path.stem)
    if match is None:
        raise ValueError(f"cannot parse episode id from {path}")
    return int(match.group(1))


def load_lengths(directory: Path, key: str) -> dict[int, int]:
    result = {}
    for path in sorted(directory.glob("episode_*.hdf5"), key=episode_id):
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"missing {key!r} in {path}")
            result[episode_id(path)] = int(handle[key].shape[0])
    if not result:
        raise FileNotFoundError(f"no episode_*.hdf5 files under {directory}")
    return result


def load_annotation_intervals(source_root: Path) -> dict[int, list[tuple[int, int]]]:
    result = {}
    pattern = "episode_*_contact_annotation.json"
    for path in sorted(source_root.glob(pattern), key=episode_id):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairs = payload.get("pairs", [])
        intervals = []
        for pair in pairs:
            onset = int(pair["contact"])
            separation = int(pair["separation"])
            if separation <= onset:
                raise ValueError(f"invalid contact interval [{onset}, {separation}) in {path}")
            intervals.append((onset, separation))
        result[episode_id(path)] = intervals
    return result


def convert_frame(frame: int, source_fps: float, target_fps: float) -> int:
    return int(round(float(frame) * target_fps / source_fps))


def build_window_events(
    intervals: list[tuple[int, int]],
    current_frame: int,
    horizon_frames: int,
    label_config: dict,
) -> list[ContactEvent]:
    end_frame = current_frame + horizon_frames
    events = []
    common = {
        "side": label_config.get("side"),
        "site": label_config.get("site"),
        "object_role": label_config.get("object_role"),
        "part_slot": label_config.get("part_slot"),
        "contact_mode": label_config.get("contact_mode"),
        "object_effect": label_config.get("object_effect"),
        "confidence": float(label_config.get("confidence", 1.0)),
    }
    for onset, separation in intervals:
        if current_frame <= onset < end_frame:
            events.append(
                ContactEvent(
                    time_offset=onset - current_frame,
                    duration=separation - onset,
                    transition="onset",
                    **common,
                )
            )
        elif onset < current_frame < separation:
            events.append(
                ContactEvent(
                    time_offset=0,
                    duration=separation - current_frame,
                    transition="hold",
                    **common,
                )
            )
        if current_frame <= separation < end_frame:
            events.append(
                ContactEvent(
                    time_offset=separation - current_frame,
                    duration=0,
                    transition="release",
                    **common,
                )
            )
    return events


def write_npz(path: Path, arrays: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(path)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    length_source_dir = (
        args.length_source_dir.expanduser().resolve()
        if args.length_source_dir is not None
        else source_root / "vla_processed" / "final_hdf5"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    annotation_fps = float(manifest["annotation_fps"])
    target_fps = float(manifest["target_fps"])
    horizon_frames = int(manifest["horizon_frames"])
    num_queries = int(manifest["num_queries"])
    defaults = dict(manifest.get("defaults", {}))
    episode_overrides = dict(manifest.get("episodes", {}))
    negative_confidence = float(defaults.pop("negative_confidence", 0.1))
    if annotation_fps <= 0 or target_fps <= 0:
        raise ValueError("annotation_fps and target_fps must be positive")

    lengths = load_lengths(length_source_dir, args.length_key)
    source_intervals = load_annotation_intervals(source_root)
    unknown_annotations = sorted(set(source_intervals) - set(lengths))
    if unknown_annotations:
        raise ValueError(f"annotations have no target episode length: {unknown_annotations}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapped_intervals = {}
    total_windows = 0
    total_positive_events = 0
    for episode, length in sorted(lengths.items()):
        arrays = {
            "contact_target": np.zeros((length, num_queries, TARGET_DIM), dtype=np.float32),
            "contact_loss_mask": np.zeros((length, num_queries, TARGET_DIM), dtype=np.float32),
            "contact_confidence": np.zeros((length, num_queries, TARGET_DIM), dtype=np.float32),
            "contact_condition_mask": np.zeros((length, num_queries, 1), dtype=np.float32),
        }
        if episode in source_intervals:
            intervals = [
                (
                    max(0, min(length, convert_frame(onset, annotation_fps, target_fps))),
                    max(0, min(length, convert_frame(separation, annotation_fps, target_fps))),
                )
                for onset, separation in source_intervals[episode]
            ]
            for onset, separation in intervals:
                if separation <= onset:
                    raise ValueError(
                        f"episode {episode} interval collapsed after FPS conversion: "
                        f"[{onset}, {separation})"
                    )
            mapped_intervals[str(episode)] = [list(interval) for interval in intervals]
            label_config = dict(defaults)
            label_config.update(episode_overrides.get(str(episode), {}))
            for frame_index in range(length):
                events = build_window_events(
                    intervals,
                    current_frame=frame_index,
                    horizon_frames=horizon_frames,
                    label_config=label_config,
                )
                encoded = encode_ordered_events(
                    events,
                    num_queries=num_queries,
                    horizon_frames=horizon_frames,
                    negative_confidence=negative_confidence,
                )
                for key, value in encoded.items():
                    arrays[key][frame_index] = value
                total_positive_events += len(events)
            total_windows += length

        destination = output_dir / f"episode_{episode:06d}.npz"
        write_npz(destination, arrays, overwrite=args.overwrite)

    field_slices = {
        name: [int(field_slice.start), int(field_slice.stop)]
        for name, field_slice in FIELD_SLICES.items()
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "target_dim": TARGET_DIM,
        "field_slices": field_slices,
        "enum_values": {name: list(values) for name, values in ENUM_VALUES.items()},
        "annotation_fps": annotation_fps,
        "target_fps": target_fps,
        "horizon_frames": horizon_frames,
        "num_queries": num_queries,
        "source_root": str(source_root),
        "length_source_dir": str(length_source_dir),
        "length_key": args.length_key,
        "episode_lengths": {str(key): value for key, value in sorted(lengths.items())},
        "annotated_episodes": sorted(source_intervals),
        "mapped_intervals": mapped_intervals,
        "total_annotated_windows": total_windows,
        "total_positive_events": total_positive_events,
        "manifest": str(manifest_path),
    }
    meta_path = output_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {meta_path}; pass --overwrite")
    partial_meta = output_dir / ".meta.json.partial"
    partial_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    partial_meta.replace(meta_path)

    summary = {
        "output_dir": str(output_dir),
        "episodes": len(lengths),
        "annotated_episodes": len(source_intervals),
        "target_dim": TARGET_DIM,
        "mapped_intervals": mapped_intervals,
        "positive_events": total_positive_events,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

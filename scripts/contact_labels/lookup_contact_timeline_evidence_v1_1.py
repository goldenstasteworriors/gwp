#!/usr/bin/env python3
"""Look up processed-frame to source-video evidence without writing images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = (
    REPO_ROOT / "track_a" / "contact_labels_v1_1" / "formal_dual_annotation_v1_1"
    / "annotator_a" / "timeline_evidence_index.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-csv", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--processed-frame", required=True, type=int)
    parser.add_argument("--context", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context < 0:
        raise ValueError("--context must be nonnegative")
    with args.evidence_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if int(row["episode_id"]) == args.episode
        ]
    if not rows:
        raise ValueError(f"episode {args.episode} is not assigned in this packet")
    by_frame = {int(row["processed_frame"]): row for row in rows}
    if args.processed_frame not in by_frame:
        raise ValueError(
            f"processed frame must be in [0,{max(by_frame)}], got {args.processed_frame}"
        )
    lo = max(0, args.processed_frame - args.context)
    hi = min(max(by_frame), args.processed_frame + args.context)
    focus = by_frame[args.processed_frame]
    result = {
        "episode_id": args.episode,
        "processed_frame": args.processed_frame,
        "processed_time_s": float(focus["processed_time_s"]),
        "video_path": focus["video_path"],
        "video_frame": int(focus["video_frame"]),
        "video_time_s": float(focus["video_time_s"]),
        "sync_error_ms": float(focus["sync_error_ms"]),
        "review_context": {
            "processed_frames_inclusive": [lo, hi],
            "video_frames_inclusive": [int(by_frame[lo]["video_frame"]), int(by_frame[hi]["video_frame"])],
            "video_times_s": [float(by_frame[lo]["video_time_s"]), float(by_frame[hi]["video_time_s"])],
        },
        "source_hashes": {
            "video_sha256": focus["video_sha256"],
            "processed_hdf5_sha256": focus["processed_hdf5_sha256"],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

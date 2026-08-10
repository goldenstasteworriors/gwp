#!/usr/bin/env python3
"""Build the Track-B 5 episode x 20 frame visibility audit.

This is a data preparation utility only.  It reads EgoHumanoid data and writes
small JPEG thumbnails, contact sheets, and CSV manifests next to this script.
It does not import or modify GigaWorld Policy model code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_SOURCE = Path(
    "/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/"
    "7_4_pour_liquid_between_beakers"
)
DEFAULT_EPISODES = (0, 29, 58, 30, 14)
FEATURES = (
    "boundary",
    "source_rim",
    "target_rim",
    "liquid_surface",
    "stream",
    "contact",
)


def robust_saliency(action_delta: np.ndarray, hand_status: np.ndarray) -> np.ndarray:
    """Motion magnitude plus a local boost around gripper-state transitions."""
    motion = np.linalg.norm(np.nan_to_num(action_delta.astype(np.float64)), axis=1)
    med = float(np.median(motion))
    mad = float(np.median(np.abs(motion - med)))
    motion_z = np.clip((motion - med) / max(1.4826 * mad, 1e-8), 0.0, 10.0)
    changed = np.zeros(len(motion), dtype=np.float64)
    if len(hand_status) > 1:
        transition = np.any(hand_status[1:] != hand_status[:-1], axis=1)
        for index in np.flatnonzero(transition) + 1:
            lo, hi = max(0, index - 2), min(len(changed), index + 3)
            changed[lo:hi] = np.maximum(changed[lo:hi], 3.0)
    return motion_z + changed


def select_phase_stratified(saliency: np.ndarray, forced: dict[int, str]) -> list[tuple[int, str]]:
    """Choose one most salient frame from each of 20 equal temporal bins."""
    selected: list[tuple[int, str]] = []
    edges = np.linspace(0, len(saliency), 21, dtype=int)
    for phase_bin, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        hi = max(hi, lo + 1)
        in_bin = [(idx, reason) for idx, reason in forced.items() if lo <= idx < hi]
        if in_bin:
            idx, reason = min(in_bin, key=lambda item: abs(item[0] - (lo + hi - 1) / 2))
        else:
            idx = lo + int(np.argmax(saliency[lo:hi]))
            reason = "phase_bin_saliency_peak"
        selected.append((idx, reason))
    if len({idx for idx, _ in selected}) != 20:
        raise RuntimeError("phase-stratified selection produced duplicate frames")
    return selected


def contact_forced_indices(source: Path, episode: int, camera_ns: np.ndarray) -> dict[int, str]:
    path = source / f"episode_{episode}_contact_annotation.json"
    if not path.exists():
        return {}
    annotation = json.loads(path.read_text(encoding="utf-8"))
    fps = float(annotation.get("fps", 30.0))
    raw_start_ns = int(camera_ns[0])
    forced: dict[int, str] = {}
    for key, reason in (
        ("contact_frame", "existing_contact_hint"),
        ("separation_frame", "existing_separation_hint"),
    ):
        if key in annotation:
            target_ns = raw_start_ns + int(float(annotation[key]) / fps * 1e9)
            forced[int(np.argmin(np.abs(camera_ns.astype(np.int64) - target_ns)))] = reason
    return forced


def nearest_video_indices(camera_ns: np.ndarray, color_us: np.ndarray) -> np.ndarray:
    color_ns = color_us.astype(np.int64) * 1000
    insertion = np.searchsorted(color_ns, camera_ns.astype(np.int64))
    insertion = np.clip(insertion, 0, len(color_ns) - 1)
    previous = np.clip(insertion - 1, 0, len(color_ns) - 1)
    use_previous = np.abs(color_ns[previous] - camera_ns) <= np.abs(color_ns[insertion] - camera_ns)
    return np.where(use_previous, previous, insertion)


def read_video_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode video frame {index}")
    return frame


def make_thumbnail(frame: np.ndarray, label: str) -> np.ndarray:
    thumb = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (320, 25), (255, 255, 255), -1)
    cv2.putText(thumb, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return thumb


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compile_prescreen(output: Path, manifest: list[dict[str, object]]) -> None:
    labels_path = output / "prescreen_labels.json"
    if not labels_path.exists():
        return
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in manifest:
        episode = str(item["episode"])
        phase_bin = int(item["phase_bin"])
        episode_labels = payload["episodes"][episode]
        row: dict[str, object] = {
            "audit_id": item["audit_id"],
            "episode": item["episode"],
            "phase_bin": phase_bin,
            "thumbnail": item["thumbnail"],
        }
        for feature in FEATURES:
            row[feature] = int(episode_labels[feature][phase_bin])
        row["confidence"] = "medium"
        row["notes"] = episode_labels["notes"] if phase_bin == 0 else ""
        rows.append(row)
    fields = ["audit_id", "episode", "phase_bin", "thumbnail", *FEATURES, "confidence", "notes"]
    write_csv(output / "prescreen_annotator_a.csv", fields, rows)
    summary: dict[str, object] = {
        "annotator": payload["annotator"],
        "frames": len(rows),
        "disclaimer": payload["disclaimer"],
        "score_definition": "0 absent/not applicable; 1 expected/present but not localizable; 2 partial/ambiguous localization; 3 clear localization",
        "per_feature": {},
    }
    for feature in FEATURES:
        counts = {str(score): sum(int(row[feature]) == score for row in rows) for score in range(4)}
        localizable = counts["2"] + counts["3"]
        expected = len(rows) - counts["0"]
        summary["per_feature"][feature] = {
            "score_counts": counts,
            "localizable_2_or_3_over_all": round(localizable / len(rows), 3),
            "localizable_2_or_3_over_expected": round(localizable / expected, 3) if expected else None,
        }
    (output / "prescreen_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build(source: Path, output: Path, episodes: tuple[int, ...]) -> None:
    if len(episodes) != 5 or len(set(episodes)) != 5:
        raise ValueError("exactly five distinct episodes are required")
    frames_dir = output / "frames"
    sheets_dir = output / "contact_sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for episode in episodes:
        processed_path = source / "vla_processed" / "final_hdf5" / f"episode_{episode}.hdf5"
        depth_path = source / f"episode_{episode}_depth.hdf5"
        video_path = source / f"episode_{episode}.mp4"
        with h5py.File(processed_path, "r") as h5:
            action_delta = np.asarray(h5["action_delta_eef"])
            hand_status = np.asarray(h5["hand_status"])
            camera_ns = np.asarray(h5["camera_timestamp"], dtype=np.int64)
        with h5py.File(depth_path, "r") as h5:
            color_us = np.asarray(h5["color_system_timestamps_us"], dtype=np.int64)
        saliency = robust_saliency(action_delta, hand_status)
        forced = contact_forced_indices(source, episode, camera_ns)
        selected = select_phase_stratified(saliency, forced)
        video_indices = nearest_video_indices(camera_ns, color_us)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        episode_thumbs: list[np.ndarray] = []
        episode_dir = frames_dir / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        for phase_bin, (action_index, reason) in enumerate(selected):
            video_index = int(video_indices[action_index])
            audit_id = f"E{episode:03d}_B{phase_bin:02d}_A{action_index:04d}_V{video_index:04d}"
            thumb = make_thumbnail(read_video_frame(capture, video_index), audit_id)
            relative_path = Path("frames") / f"episode_{episode:03d}" / f"{audit_id}.jpg"
            if not cv2.imwrite(str(output / relative_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 86]):
                raise RuntimeError(f"could not write {relative_path}")
            episode_thumbs.append(thumb)
            manifest.append(
                {
                    "audit_id": audit_id,
                    "episode": episode,
                    "phase_bin": phase_bin,
                    "processed_action_index": action_index,
                    "video_frame_index": video_index,
                    "camera_timestamp_ns": int(camera_ns[action_index]),
                    "video_timestamp_us": int(color_us[video_index]),
                    "sync_error_ms": round(abs(int(camera_ns[action_index]) - int(color_us[video_index]) * 1000) / 1e6, 4),
                    "selection_reason": reason,
                    "saliency_score": round(float(saliency[action_index]), 5),
                    "thumbnail": relative_path.as_posix(),
                    "source_video": str(video_path),
                    "source_processed_hdf5": str(processed_path),
                }
            )
        capture.release()
        sheet_rows = []
        for row_start in range(0, 20, 4):
            sheet_rows.append(np.concatenate(episode_thumbs[row_start : row_start + 4], axis=1))
        sheet = np.concatenate(sheet_rows, axis=0)
        cv2.imwrite(str(sheets_dir / f"episode_{episode:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    manifest_fields = list(manifest[0].keys())
    write_csv(output / "manifest.csv", manifest_fields, manifest)
    annotation_fields = [
        "audit_id",
        "episode",
        "phase_bin",
        "thumbnail",
        *[f"annotator_a_{feature}" for feature in FEATURES],
        "annotator_a_confidence",
        "annotator_a_notes",
        *[f"annotator_b_{feature}" for feature in FEATURES],
        "annotator_b_confidence",
        "annotator_b_notes",
        *[f"adjudicated_{feature}" for feature in FEATURES],
        "adjudication_notes",
    ]
    annotation_rows = [
        {
            "audit_id": row["audit_id"],
            "episode": row["episode"],
            "phase_bin": row["phase_bin"],
            "thumbnail": row["thumbnail"],
        }
        for row in manifest
    ]
    write_csv(output / "annotations_dual_template.csv", annotation_fields, annotation_rows)
    compile_prescreen(output, manifest)
    metadata = {
        "schema_version": "track_b_visibility_audit_v0.1",
        "source_root": str(source),
        "episodes": list(episodes),
        "frames_per_episode": 20,
        "total_frames": len(manifest),
        "selection": "one maximum-saliency frame per equal-duration phase bin; existing episode-0 contact hints override their bins",
        "saliency": "robust-normalized action_delta_eef L2 norm plus a five-frame hand_status-transition boost",
        "video_sync": "nearest color_system_timestamps_us to processed camera_timestamp (ns)",
        "max_sync_error_ms": max(float(row["sync_error_ms"]) for row in manifest),
    }
    (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def check(output: Path) -> None:
    with (output / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100, f"expected 100 rows, found {len(rows)}"
    assert len({row["audit_id"] for row in rows}) == 100
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["episode"]] = counts.get(row["episode"], 0) + 1
        assert (output / row["thumbnail"]).is_file()
        assert float(row["sync_error_ms"]) <= 20.0
    assert sorted(counts.values()) == [20] * 5
    with (output / "annotations_dual_template.csv").open(encoding="utf-8", newline="") as handle:
        annotation_rows = list(csv.DictReader(handle))
    assert len(annotation_rows) == 100
    prescreen_path = output / "prescreen_annotator_a.csv"
    if prescreen_path.exists():
        with prescreen_path.open(encoding="utf-8", newline="") as handle:
            prescreen_rows = list(csv.DictReader(handle))
        assert len(prescreen_rows) == 100
        assert all(row[feature] in {"0", "1", "2", "3"} for row in prescreen_rows for feature in FEATURES)
    print(json.dumps({"status": "ok", "frames": 100, "episodes": counts}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--episodes", type=int, nargs=5, default=DEFAULT_EPISODES)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.check_only:
        build(args.source, args.output, tuple(args.episodes))
    check(args.output)

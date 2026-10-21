"""Export right-arm/Dex1 measured absolute angles and right-wrist video to v3."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO = "observation.images.right_wrist"
JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint", "right_hand_dex1_motor",
]


def prepare(source, output, episode0=False):
    info = json.loads((source / "meta/info.json").read_text())
    indices = [info["features"]["observation.state"]["names"].index(n) for n in JOINTS]
    episodes = [json.loads(s) for s in (source / "meta/episodes.jsonl").read_text().splitlines() if s.strip()]
    tasks = [json.loads(s) for s in (source / "meta/tasks.jsonl").read_text().splitlines() if s.strip()]
    selected = [e for e in episodes if not episode0 or e["episode_index"] == 0]
    if output.exists():
        raise FileExistsError(output)
    for directory in ["data/chunk-000", f"videos/{VIDEO}/chunk-000", "meta/episodes/chunk-000"]:
        (output / directory).mkdir(parents=True)
    pd.DataFrame({"task_index": [t["task_index"] for t in tasks]}, index=pd.Index([t["task"] for t in tasks], name="task")).to_parquet(output / "meta/tasks.parquet")
    rows, receipt, total = [], [], 0
    for dest, episode in enumerate(selected):
        src = int(episode["episode_index"])
        parquet = source / info["data_path"].format(episode_chunk=src // info["chunks_size"], episode_index=src)
        raw = pd.read_parquet(parquet)
        measured = np.stack(raw["observation.state"])[:, indices].astype(np.float32)
        assert np.isfinite(measured).all()
        assert len(raw) == episode["length"]
        out = raw[["timestamp", "frame_index", "episode_index", "index", "task_index"]].copy()
        out["episode_index"] = dest
        out["index"] = np.arange(total, total + len(out), dtype=np.int64)
        out["observation.state"] = list(measured)
        # Action supervision deliberately comes from measured states, never action.wbc.
        out["action"] = list(measured.copy())
        out.to_parquet(output / f"data/chunk-000/file-{dest:03d}.parquet", index=False)
        video = source / info["video_path"].format(episode_chunk=src // info["chunks_size"], episode_index=src, video_key=VIDEO)
        shutil.copy2(video, output / f"videos/{VIDEO}/chunk-000/file-{dest:03d}.mp4")
        rows.append({
            "episode_index": dest, "source_episode_index": src, "tasks": episode["tasks"], "length": len(out),
            "data/chunk_index": 0, "data/file_index": dest,
            "dataset_from_index": total, "dataset_to_index": total + len(out),
            f"videos/{VIDEO}/chunk_index": 0, f"videos/{VIDEO}/file_index": dest,
            f"videos/{VIDEO}/from_timestamp": 0., f"videos/{VIDEO}/to_timestamp": len(out) / info["fps"],
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        })
        receipt.append({"source_episode": src, "frames": len(out), "source_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(), "right_wrist_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "angle_min": measured.min(0).tolist(), "angle_max": measured.max(0).tolist()})
        total += len(out)
    features = {VIDEO: info["features"][VIDEO]}
    for key in ["observation.state", "action"]:
        features[key] = {"dtype": "float32", "shape": [8], "names": JOINTS}
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        features[key] = info["features"][key]
    info.update(codebase_version="v3.1", robot_type="sonic_g1", total_episodes=len(rows), total_frames=total, total_videos=len(rows), total_chunks=1, chunks_size=max(1, len(rows)), splits={"train": f"0:{len(rows)}"}, features=features,
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", controlled_side="right", action_semantics="measured_absolute_joint_angle", action_schema_id="g1_right_arm7_dex1_measured_absolute_v1")
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "meta/modality.json").write_text(json.dumps({"state": {"right_joint_angles": {"start": 0, "end": 8}}, "action": {"right_joint_angles": {"start": 0, "end": 8}}, "video": {"right_wrist": {"original_key": VIDEO}}, "annotation": {"human.task_description": {"original_key": "task_index"}}}, indent=2) + "\n")
    pd.DataFrame(rows).to_parquet(output / "meta/episodes/chunk-000/file-000.parquet", index=False)
    (output / "meta/conversion_receipt.json").write_text(json.dumps({"source": str(source.resolve()), "state_source": "observation.state", "action_source": "observation.state", "units": "rad", "joints": JOINTS, "video": VIDEO, "episodes": receipt}, indent=2) + "\n")
    print(f"{output}: {len(rows)} episodes, {total} frames, measured absolute 8-D right joints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode0", action="store_true")
    args = parser.parse_args()
    prepare(args.input, args.output, args.episode0)

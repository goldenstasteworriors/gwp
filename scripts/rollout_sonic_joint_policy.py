#!/usr/bin/env python3
"""Run a reference-vision, predicted-state SONIC joint-policy rollout.

Only the first joint state comes from the reference episode.  Every subsequent
state is the first action of a newly inferred chunk.  The policy wrapper adds
the predicted left-arm deltas to that recursively predicted state, while the
six Inspire-hand channels remain absolute commands.  Reference frames are
used as visual observations at every step; reference joint states are retained
only for comparison and never fed back after initialization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "third_party" / "giga-datasets",
    PROJECT_ROOT / "third_party" / "giga-models",
    PROJECT_ROOT / "third_party" / "giga-train",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

from world_action_model.inference import SonicHistoryPolicy


EXPERIMENTS = {
    "m00_baseline": {
        "action_horizon": 48,
        "image_frame_offsets": [0, 12, 24, 36, 48],
        "state_history_offsets": [0],
        "norm_stats": "norm_stats.json",
    },
    "m12_short_horizon": {
        "action_horizon": 24,
        "image_frame_offsets": [0, 6, 12, 18, 24],
        "state_history_offsets": [-50, -25, 0],
        "norm_stats": "norm_stats_h24.json",
    },
}
DELTA_MASK = [True] * 7 + [False] * 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers",
    )
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--prompt-embedding", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths/data and write the runtime manifest without loading the model.",
    )
    return parser.parse_args()


def unique_episode_file(root: Path, episode: int, suffix: str) -> Path:
    names = (f"file-{episode:03d}.{suffix}", f"episode_{episode:06d}.{suffix}")
    matches = sorted({path.resolve() for name in names for path in root.glob(f"**/{name}")})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one episode {episode} .{suffix} under {root}, found {matches}"
        )
    return matches[0]


def load_reference_episode(dataset: Path, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path, Path]:
    parquet = unique_episode_file(dataset / "data", episode, "parquet")
    video = unique_episode_file(dataset / "videos", episode, "mp4")
    table = pd.read_parquet(parquet)
    order_key = "frame_index" if "frame_index" in table else "timestamp"
    table = table.sort_values(order_key).reset_index(drop=True)
    states = np.stack(table["observation.state"]).astype(np.float32)
    actions = np.stack(table["action"]).astype(np.float32)
    if states.shape != (len(table), 13) or actions.shape != states.shape:
        raise ValueError(f"expected finite [T,13] state/action, got {states.shape}/{actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("reference state/action contains non-finite values")
    timestamps = (
        table["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in table
        else np.arange(len(table), dtype=np.float64) / 50.0
    )
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("episode timestamps must be strictly increasing")
    fps = float(1.0 / np.median(np.diff(timestamps)))
    return states, actions, timestamps, fps, parquet, video


def load_rgb_frames(video: Path, start: int, stop: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open reference video: {video}")
    frames: list[np.ndarray] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_index in range(start, stop):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"video decode failed at frame {frame_index}: {video}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def write_runtime_manifest(args: argparse.Namespace, dataset: Path, output: Path) -> Path:
    spec = EXPERIMENTS[args.experiment]
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != "transformer" and (checkpoint / "transformer").is_dir():
        checkpoint = checkpoint / "transformer"
    norm_stats = (
        args.norm_stats.expanduser().resolve()
        if args.norm_stats
        else dataset / spec["norm_stats"]
    )
    prompt_embedding = (
        args.prompt_embedding.expanduser().resolve()
        if args.prompt_embedding
        else dataset / "t5_embedding" / "episode_000000.pt"
    )
    required = [
        checkpoint / "config.json",
        norm_stats,
        prompt_embedding,
        args.base_model.expanduser().resolve() / "vae",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inference inputs: {missing}")

    manifest = {
        "format_version": 1,
        "experiment": args.experiment,
        "transformer": str(checkpoint),
        "base_model": str(args.base_model.expanduser().resolve()),
        "norm_stats": str(norm_stats),
        "prompt_embedding": str(prompt_embedding),
        "prompt": "grab graduated cylinder and pour water into beaker",
        "evaluation_dataset": str(dataset),
        "evaluation_episodes": [args.episode],
        "image_key": "observation.images.ego_view",
        "image_frame_offsets": spec["image_frame_offsets"],
        "state_history_offsets": spec["state_history_offsets"],
        "state_history_tokens": len(spec["state_history_offsets"]),
        "visual_history_frames": 1,
        "append_state_velocity_token": False,
        "append_episode_progress_token": False,
        "action_horizon": spec["action_horizon"],
        "model_image_frames": len(spec["image_frame_offsets"]),
        "physical_action_dim": 13,
        "model_action_dim": 32,
        "delta_mask": DELTA_MASK,
        "fps": 50,
        "image_size": [320, 384],
        "num_inference_steps": args.num_inference_steps,
        "model_dtype": "bf16",
        "action_flow_shift": 5.0,
        "rollout_semantics": {
            "reference_state_used_only_at_initialization": True,
            "reference_images_used_at_every_step": True,
            "left_arm_channels_0_7": "relative_delta_integrated_from_predicted_state",
            "left_inspire_channels_7_13": "absolute_command",
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def recursive_history(
    predicted_states: list[np.ndarray], offsets: list[int], local_step: int
) -> np.ndarray:
    indices = [max(0, min(local_step, local_step + offset)) for offset in offsets]
    return np.stack([predicted_states[index] for index in indices]).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.episode < 0 or args.start_frame < 0 or args.num_steps <= 0:
        raise SystemExit("episode/start-frame must be non-negative and num-steps must be positive")
    if args.num_inference_steps <= 0:
        raise SystemExit("num-inference-steps must be positive")

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_states, reference_actions, timestamps, fps, parquet, video = load_reference_episode(
        dataset, args.episode
    )
    stop_frame = min(args.start_frame + args.num_steps, len(reference_states) - 1)
    if stop_frame <= args.start_frame:
        raise ValueError(
            f"start frame {args.start_frame} leaves no transition in {len(reference_states)} frames"
        )
    manifest_path = write_runtime_manifest(args, dataset, output)
    print(f"runtime manifest: {manifest_path}", flush=True)
    print(f"reference parquet: {parquet}", flush=True)
    print(f"reference video: {video}", flush=True)
    print(
        f"closed-loop range: [{args.start_frame}, {stop_frame}] "
        f"({stop_frame - args.start_frame} replans)",
        flush=True,
    )
    if args.validate_only:
        print("validation complete; model was not loaded")
        return

    rgb_frames = load_rgb_frames(video, args.start_frame, stop_frame)
    policy = SonicHistoryPolicy(manifest_path, device=args.device)
    spec = EXPERIMENTS[args.experiment]
    predicted_states = [reference_states[args.start_frame].copy()]
    arm_deltas: list[np.ndarray] = []
    hand_commands: list[np.ndarray] = []
    latencies: list[float] = []
    wall_start = time.time()
    total = stop_frame - args.start_frame
    for local_step, frame_index in enumerate(range(args.start_frame, stop_frame)):
        state_history = recursive_history(
            predicted_states, spec["state_history_offsets"], local_step
        )
        progress = float(frame_index / max(len(reference_states) - 1, 1))
        started = time.perf_counter()
        action_chunk = policy.predict(
            [rgb_frames[local_step]],
            state_history,
            episode_progress=progress,
            seed=args.seed + frame_index,
        )
        latencies.append(time.perf_counter() - started)
        next_state = np.asarray(action_chunk[0], dtype=np.float32)
        if next_state.shape != (13,) or not np.isfinite(next_state).all():
            raise ValueError(f"invalid predicted state at frame {frame_index}: {next_state}")
        current_state = predicted_states[-1]
        arm_deltas.append(next_state[:7] - current_state[:7])
        hand_commands.append(next_state[7:13].copy())
        predicted_states.append(next_state)
        completed = local_step + 1
        if completed == 1 or completed % 10 == 0 or completed == total:
            elapsed = time.time() - wall_start
            eta = elapsed / completed * (total - completed)
            print(
                f"[{completed}/{total}] frame={frame_index} "
                f"latency={latencies[-1]:.3f}s eta={eta:.1f}s",
                flush=True,
            )

    predicted = np.stack(predicted_states).astype(np.float32)
    arm_delta = np.stack(arm_deltas).astype(np.float32)
    hand_absolute = np.stack(hand_commands).astype(np.float32)
    frame_indices = np.arange(args.start_frame, stop_frame + 1, dtype=np.int64)
    aligned_reference = reference_states[frame_indices]
    np.savez_compressed(
        output,
        predicted_state=predicted,
        predicted_arm_delta=arm_delta,
        predicted_hand_absolute=hand_absolute,
        reference_state=aligned_reference,
        reference_action=reference_actions[frame_indices],
        frame_indices=frame_indices,
        timestamps=timestamps[frame_indices],
        latency_seconds=np.asarray(latencies, dtype=np.float64),
        initial_reference_state=reference_states[args.start_frame],
    )
    error = predicted - aligned_reference
    summary = {
        "rollout": str(output),
        "manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "experiment": args.experiment,
        "dataset": str(dataset),
        "episode": args.episode,
        "start_frame": args.start_frame,
        "end_frame_inclusive": stop_frame,
        "steps": total,
        "reference_fps": fps,
        "state_feedback": "predicted after initialization",
        "visual_input": "reference RGB at every replan",
        "left_arm": "predicted relative delta recursively integrated",
        "left_inspire_hand": "predicted absolute command",
        "mean_latency_seconds": float(np.mean(latencies)),
        "arm_rmse_vs_reference": float(np.sqrt(np.mean(error[:, :7] ** 2))),
        "hand_rmse_vs_reference": float(np.sqrt(np.mean(error[:, 7:13] ** 2))),
        "max_abs_arm_delta": float(np.max(np.abs(arm_delta))),
        "predicted_hand_range": [
            float(np.min(hand_absolute)),
            float(np.max(hand_absolute)),
        ],
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

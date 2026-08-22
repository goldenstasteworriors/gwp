#!/usr/bin/env python3
"""Run a reference-vision SONIC joint-policy rollout with chunked execution.

Each replan predicts a complete action chunk and executes a fixed prefix before
the next replan.  ``predicted`` feedback uses the reference joint state only at
initialization and recursively feeds executed action targets back as state.
``ground_truth`` feedback instead uses the reference state at every replan.
Both modes use the corresponding reference RGB frame at every replan.
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
from world_action_model.rtc import RTCConfig


EXPERIMENTS = {
    "m00_baseline": {
        "action_horizon": 48,
        "image_frame_offsets": [0, 12, 24, 36, 48],
        "state_history_offsets": [0],
        "norm_stats": "norm_stats.json",
        "delta_mask": [True] * 7 + [False] * 6,
        "prompt": "grab graduated cylinder and pour water into beaker",
        "left_arm_semantics": "relative_delta_integrated_from_predicted_state",
    },
    "m12_short_horizon": {
        "action_horizon": 24,
        "image_frame_offsets": [0, 6, 12, 18, 24],
        "state_history_offsets": [-50, -25, 0],
        "norm_stats": "norm_stats_h24.json",
        "delta_mask": [True] * 7 + [False] * 6,
        "prompt": "grab graduated cylinder and pour water into beaker",
        "left_arm_semantics": "relative_delta_integrated_from_predicted_state",
    },
    "legacy_absolute_48": {
        "action_horizon": 48,
        "image_frame_offsets": [0, 12, 24, 36, 48],
        "state_history_offsets": [0],
        "norm_stats": "norm_stats.json",
        "delta_mask": [False] * 13,
        "prompt": "shake the beaker",
        "left_arm_semantics": "absolute_joint_target",
    },
}


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
    parser.add_argument(
        "--vae-dtype",
        choices=("bf16", "float32"),
        default="bf16",
        help="VAE compute dtype; float32 is a compatibility fallback for some accelerators.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument(
        "--execution-steps",
        type=int,
        default=8,
        help="Number of 50 Hz action targets executed from each predicted chunk.",
    )
    parser.add_argument(
        "--state-feedback",
        choices=("predicted", "ground_truth"),
        default="predicted",
        help="State source at replans after initialization.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--rtc", action="store_true", help="Enable training-free RTC guidance.")
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=None,
        help="Fresh suffix retained by RTC; defaults to --execution-steps.",
    )
    parser.add_argument("--rtc-inference-delay", type=int, default=0)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc-prefix-schedule",
        choices=("exp", "linear", "ones", "hard"),
        default="exp",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Manifest metadata only; inference uses the selected prompt embedding file.",
    )
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
        "prompt": args.prompt or spec["prompt"],
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
        "delta_mask": spec["delta_mask"],
        "fps": 50,
        "image_size": [320, 384],
        "num_inference_steps": args.num_inference_steps,
        "execution_steps": args.execution_steps,
        "replan_interval_seconds": args.execution_steps / 50.0,
        "state_feedback": args.state_feedback,
        "rtc": {
            "enabled": args.rtc,
            "method": "PiGDM-guided flow matching over aligned chunk overlap",
            "execution_horizon": args.rtc_execution_horizon or args.execution_steps,
            "inference_delay": args.rtc_inference_delay,
            "max_guidance_weight": args.rtc_max_guidance_weight,
            "prefix_schedule": args.rtc_prefix_schedule,
        },
        "model_dtype": "bf16",
        "vae_dtype": args.vae_dtype,
        "action_flow_shift": 5.0,
        "rollout_semantics": {
            "reference_state_used_only_at_initialization": args.state_feedback == "predicted",
            "reference_state_used_at_every_replan": args.state_feedback == "ground_truth",
            "reference_images_used_at_every_replan": True,
            "full_action_chunk_predicted_at_every_replan": True,
            "executed_prefix_steps_per_chunk": args.execution_steps,
            "left_arm_channels_0_7": "relative_delta_integrated_to_absolute_joint_target",
            "left_arm_integration_state": args.state_feedback,
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


def reference_history(states: np.ndarray, offsets: list[int], frame_index: int) -> np.ndarray:
    indices = [max(0, min(len(states) - 1, frame_index + offset)) for offset in offsets]
    return np.stack([states[index] for index in indices]).astype(np.float32)


def _vector_stats(vectors: np.ndarray) -> dict:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, vectors.shape[-1])
    if len(vectors) == 0:
        return {"l2_mean": None, "l2_max": None, "arm_abs_mean": None, "hand_abs_mean": None}
    norms = np.linalg.norm(vectors, axis=-1)
    return {
        "l2_mean": float(np.mean(norms)),
        "l2_max": float(np.max(norms)),
        "arm_abs_mean": float(np.mean(np.abs(vectors[:, :7]))),
        "hand_abs_mean": float(np.mean(np.abs(vectors[:, 7:13]))),
    }


def compute_jitter_metrics(
    action_chunks: np.ndarray,
    replan_frame_indices: np.ndarray,
    executed_states: np.ndarray,
    reference_states: np.ndarray,
) -> dict:
    """Measure within-chunk smoothness and cross-chunk plan inconsistency."""

    chunks = np.asarray(action_chunks, dtype=np.float64)
    internal_velocity = np.diff(chunks, axis=1).reshape(-1, chunks.shape[-1])
    internal_acceleration = np.diff(chunks, n=2, axis=1).reshape(-1, chunks.shape[-1])

    overlap_differences = []
    boundary_plan_differences = []
    for index in range(1, len(chunks)):
        shift = int(replan_frame_indices[index] - replan_frame_indices[index - 1])
        if shift < 0 or shift >= chunks.shape[1]:
            continue
        overlap = min(chunks.shape[1], chunks.shape[1] - shift)
        if overlap <= 0:
            continue
        disagreement = chunks[index, :overlap] - chunks[index - 1, shift : shift + overlap]
        overlap_differences.append(disagreement)
        boundary_plan_differences.append(disagreement[0])

    overlap_array = (
        np.concatenate(overlap_differences, axis=0)
        if overlap_differences
        else np.empty((0, chunks.shape[-1]), dtype=np.float64)
    )
    boundary_array = (
        np.stack(boundary_plan_differences)
        if boundary_plan_differences
        else np.empty((0, chunks.shape[-1]), dtype=np.float64)
    )

    executed_velocity = np.diff(executed_states, axis=0)
    executed_acceleration = np.diff(executed_states, n=2, axis=0)
    reference_velocity = np.diff(reference_states, axis=0)
    reference_acceleration = np.diff(reference_states, n=2, axis=0)
    return {
        "units": "joint radians per 50-Hz step; L2 is over 13 output channels",
        "chunk_internal_velocity": _vector_stats(internal_velocity),
        "chunk_internal_acceleration": _vector_stats(internal_acceleration),
        "chunk_overlap_disagreement": _vector_stats(overlap_array),
        "chunk_boundary_plan_disagreement": _vector_stats(boundary_array),
        "executed_velocity": _vector_stats(executed_velocity),
        "executed_acceleration": _vector_stats(executed_acceleration),
        "reference_velocity": _vector_stats(reference_velocity),
        "reference_acceleration": _vector_stats(reference_acceleration),
    }


def main() -> None:
    args = parse_args()
    if args.episode < 0 or args.start_frame < 0 or args.num_steps <= 0:
        raise SystemExit("episode/start-frame must be non-negative and num-steps must be positive")
    if args.num_inference_steps <= 0 or args.execution_steps <= 0:
        raise SystemExit("num-inference-steps and execution-steps must be positive")
    if args.rtc_inference_delay < 0:
        raise SystemExit("rtc-inference-delay must be non-negative")
    spec = EXPERIMENTS[args.experiment]
    rtc_execution_horizon = args.rtc_execution_horizon or args.execution_steps
    if rtc_execution_horizon < 1 or rtc_execution_horizon > spec["action_horizon"]:
        raise SystemExit(
            f"rtc-execution-horizon must be in [1, {spec['action_horizon']}]"
        )
    if args.execution_steps > spec["action_horizon"]:
        raise SystemExit(
            f"execution-steps {args.execution_steps} exceeds action horizon "
            f"{spec['action_horizon']}"
        )

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
        f"rollout range: [{args.start_frame}, {stop_frame}] "
        f"({stop_frame - args.start_frame} control steps, "
        f"execution_steps={args.execution_steps}, state_feedback={args.state_feedback})",
        flush=True,
    )
    if args.validate_only:
        print("validation complete; model was not loaded")
        return

    rgb_frames = load_rgb_frames(video, args.start_frame, stop_frame)
    policy = SonicHistoryPolicy(manifest_path, device=args.device)
    rtc_config = None
    if args.rtc:
        rtc_config = RTCConfig(
            execution_horizon=args.rtc_execution_horizon or args.execution_steps,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_schedule=args.rtc_prefix_schedule,
        )
    predicted_states = [reference_states[args.start_frame].copy()]
    arm_deltas: list[np.ndarray] = []
    hand_commands: list[np.ndarray] = []
    latencies: list[float] = []
    action_chunks: list[np.ndarray] = []
    input_state_histories: list[np.ndarray] = []
    replan_frame_indices: list[int] = []
    executed_action_counts: list[int] = []
    executed_chunk_indices: list[int] = []
    executed_chunk_offsets: list[int] = []
    wall_start = time.time()
    total_steps = stop_frame - args.start_frame
    replan_frames = list(range(args.start_frame, stop_frame, args.execution_steps))
    total_replans = len(replan_frames)
    for replan_index, frame_index in enumerate(replan_frames):
        local_step = frame_index - args.start_frame
        if args.state_feedback == "predicted":
            state_history = recursive_history(
                predicted_states, spec["state_history_offsets"], local_step
            )
        else:
            state_history = reference_history(
                reference_states, spec["state_history_offsets"], frame_index
            )
        progress = float(frame_index / max(len(reference_states) - 1, 1))
        previous_tail = None
        if rtc_config is not None and action_chunks:
            previous_tail = action_chunks[-1][executed_action_counts[-1] :]
        started = time.perf_counter()
        action_chunk = policy.predict(
            [rgb_frames[local_step]],
            state_history,
            episode_progress=progress,
            seed=args.seed + frame_index,
            rtc_previous_chunk_tail=previous_tail,
            rtc_inference_delay=args.rtc_inference_delay,
            rtc_config=rtc_config,
        )
        latencies.append(time.perf_counter() - started)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        expected_shape = (spec["action_horizon"], 13)
        if action_chunk.shape != expected_shape or not np.isfinite(action_chunk).all():
            raise ValueError(
                f"invalid action chunk at frame {frame_index}: "
                f"{action_chunk.shape}, expected {expected_shape}"
            )
        execute_count = min(args.execution_steps, stop_frame - frame_index)
        action_chunks.append(action_chunk.copy())
        input_state_histories.append(state_history.copy())
        replan_frame_indices.append(frame_index)
        executed_action_counts.append(execute_count)
        for chunk_offset, next_state in enumerate(action_chunk[:execute_count]):
            current_state = predicted_states[-1]
            arm_deltas.append(next_state[:7] - current_state[:7])
            hand_commands.append(next_state[7:13].copy())
            predicted_states.append(next_state.copy())
            executed_chunk_indices.append(replan_index)
            executed_chunk_offsets.append(chunk_offset)

        completed_replans = replan_index + 1
        elapsed = time.time() - wall_start
        eta = elapsed / completed_replans * (total_replans - completed_replans)
        print(
            f"[{completed_replans}/{total_replans}] frame={frame_index} "
            f"executed={execute_count}/{len(action_chunk)} "
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
        action_chunks=np.stack(action_chunks).astype(np.float32),
        input_state_history=np.stack(input_state_histories).astype(np.float32),
        replan_frame_indices=np.asarray(replan_frame_indices, dtype=np.int64),
        executed_action_counts=np.asarray(executed_action_counts, dtype=np.int64),
        executed_chunk_indices=np.asarray(executed_chunk_indices, dtype=np.int64),
        executed_chunk_offsets=np.asarray(executed_chunk_offsets, dtype=np.int64),
        rtc_enabled=np.asarray(args.rtc),
        rtc_execution_horizon=np.asarray(
            args.rtc_execution_horizon or args.execution_steps, dtype=np.int64
        ),
        rtc_inference_delay=np.asarray(args.rtc_inference_delay, dtype=np.int64),
    )
    error = predicted - aligned_reference
    jitter_metrics = compute_jitter_metrics(
        np.stack(action_chunks),
        np.asarray(replan_frame_indices),
        predicted,
        aligned_reference,
    )
    summary = {
        "rollout": str(output),
        "manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "experiment": args.experiment,
        "dataset": str(dataset),
        "episode": args.episode,
        "start_frame": args.start_frame,
        "end_frame_inclusive": stop_frame,
        "steps": total_steps,
        "replans": total_replans,
        "execution_steps": args.execution_steps,
        "replan_interval_seconds": args.execution_steps / 50.0,
        "reference_fps": fps,
        "state_feedback": args.state_feedback,
        "rtc": {
            "enabled": args.rtc,
            "execution_horizon": args.rtc_execution_horizon or args.execution_steps,
            "inference_delay": args.rtc_inference_delay,
            "max_guidance_weight": args.rtc_max_guidance_weight,
            "prefix_schedule": args.rtc_prefix_schedule,
        },
        "visual_input": "reference RGB at each chunk replan",
        "left_arm": "relative delta integrated to absolute target by policy wrapper",
        "left_inspire_hand": "predicted absolute command",
        "mean_latency_seconds": float(np.mean(latencies)),
        "arm_rmse_vs_reference": float(np.sqrt(np.mean(error[:, :7] ** 2))),
        "hand_rmse_vs_reference": float(np.sqrt(np.mean(error[:, 7:13] ** 2))),
        "max_abs_arm_delta": float(np.max(np.abs(arm_delta))),
        "predicted_hand_range": [
            float(np.min(hand_absolute)),
            float(np.max(hand_absolute)),
        ],
        "jitter_metrics": jitter_metrics,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

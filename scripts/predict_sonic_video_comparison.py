#!/usr/bin/env python3
"""Render teacher-forced SONIC future-video predictions beside ground truth.

The MoT policy is trained with separate visual and action flow-matching paths,
while the deployment pipeline intentionally runs only the action expert.  This
script restores the joint denoising path for offline diagnostics.  Each chunk
is conditioned on the reference frame and state history at the chunk start;
predicted and reference frames are compared at the model's temporal offsets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


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

from world_action_model.inference.sonic_history_policy import (  # noqa: E402
    SonicHistoryPolicy,
    _center_crop_resize,
    _to_pil,
)


MODEL_FPS = 50.0
VAE_TEMPORAL_SCALE = 4
IMAGE_FRAME_OFFSETS = (0, 6, 12, 18, 24)
STATE_HISTORY_OFFSETS = (-50, -25, 0)
DEFAULT_SEGMENTS = (
    (9, 0.0, 7.2, 1.34, 1.80, 0.359),
    (11, 0.0, 4.6, 0.80, 1.74, 0.437),
    (12, 0.0, 4.1, 1.18, 1.69, 0.299),
    (13, 0.0, 4.7, 2.04, 1.69, 0.249),
    (15, 0.0, 3.6, 0.80, 1.65, 0.308),
    (28, 0.0, 3.0, 1.10, 1.66, 0.194),
    (29, 0.0, 3.2, 0.74, 1.58, 0.218),
    (66, 0.0, 4.7, 2.20, 1.71, 0.271),
    (68, 0.0, 5.2, 0.80, 1.72, 0.233),
)


@dataclass(frozen=True)
class Segment:
    episode: int
    start_seconds: float
    end_seconds: float
    peak_seconds: float
    dominant_hz: float
    angle_rms_degrees: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--prompt-embedding", type=Path)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        metavar="EP:START:END:PEAK:HZ:RMS_DEG",
        help="Repeat to override the built-in diagnostic segments.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all inputs without loading the model or writing videos.",
    )
    return parser.parse_args()


def parse_segments(values: list[str]) -> list[Segment]:
    if not values:
        return [Segment(*values) for values in DEFAULT_SEGMENTS]
    segments = []
    for raw in values:
        parts = raw.split(":")
        if len(parts) != 6:
            raise ValueError(f"invalid segment {raw!r}; expected EP:START:END:PEAK:HZ:RMS_DEG")
        segment = Segment(
            episode=int(parts[0]),
            start_seconds=float(parts[1]),
            end_seconds=float(parts[2]),
            peak_seconds=float(parts[3]),
            dominant_hz=float(parts[4]),
            angle_rms_degrees=float(parts[5]),
        )
        if segment.episode < 0 or not 0 <= segment.start_seconds < segment.end_seconds:
            raise ValueError(f"invalid episode/time range in {raw!r}")
        if not segment.start_seconds <= segment.peak_seconds <= segment.end_seconds:
            raise ValueError(f"peak time is outside the segment in {raw!r}")
        segments.append(segment)
    return segments


def unique_episode_file(root: Path, episode: int, suffix: str) -> Path:
    names = (f"file-{episode:03d}.{suffix}", f"episode_{episode:06d}.{suffix}")
    matches = sorted({path.resolve() for name in names for path in root.glob(f"**/{name}")})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one episode {episode} .{suffix} under {root}, found {matches}"
        )
    return matches[0]


def load_episode_metadata(dataset: Path, episode: int) -> tuple[np.ndarray, np.ndarray, float, Path]:
    parquet = unique_episode_file(dataset / "data", episode, "parquet")
    video = unique_episode_file(dataset / "videos", episode, "mp4")
    table = pd.read_parquet(parquet)
    order_key = "frame_index" if "frame_index" in table else "timestamp"
    table = table.sort_values(order_key).reset_index(drop=True)
    states = np.stack(table["observation.state"]).astype(np.float32)
    if states.shape != (len(table), 13) or not np.isfinite(states).all():
        raise ValueError(f"episode {episode} expected finite [T,13] states, got {states.shape}")
    timestamps = (
        table["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in table
        else np.arange(len(table), dtype=np.float64) / MODEL_FPS
    )
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"episode {episode} timestamps are not strictly increasing")
    fps = float(1.0 / np.median(np.diff(timestamps)))
    if not np.isclose(fps, MODEL_FPS, rtol=0.01, atol=0.05):
        raise ValueError(f"episode {episode} expected {MODEL_FPS} FPS, got {fps}")
    return states, timestamps, fps, video


def load_rgb_frames(video: Path, start: int, stop_inclusive: int) -> list[np.ndarray]:
    frames = []
    container = av.open(str(video))
    try:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index < start:
                continue
            if frame_index > stop_inclusive:
                break
            frames.append(frame.to_rgb().to_ndarray())
    finally:
        container.close()
    expected = stop_inclusive - start + 1
    if len(frames) != expected:
        raise RuntimeError(
            f"video decode returned {len(frames)}/{expected} frames for "
            f"[{start}, {stop_inclusive}]: {video}"
        )
    return frames


def state_history(states: np.ndarray, frame_index: int) -> np.ndarray:
    indices = [max(0, frame_index + offset) for offset in STATE_HISTORY_OFFSETS]
    return states[indices]


def _video_array(value) -> np.ndarray:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected one generated video, got {len(value)}")
        value = value[0]
    array = np.asarray(value)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError(f"expected batch size one, got generated shape {array.shape}")
        array = array[0]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected generated [T,H,W,3], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0.0, 255.0).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


@torch.inference_mode()
def predict_joint_chunk(
    policy: SonicHistoryPolicy,
    reference_image: np.ndarray,
    states: np.ndarray,
    episode_progress: float,
    seed: int,
) -> np.ndarray:
    pipe = policy.pipe
    image = _center_crop_resize(_to_pil(reference_image), policy.width, policy.height)
    image_tensor = pipe.video_processor.preprocess(
        image, height=policy.height, width=policy.width
    ).to(policy.device, dtype=torch.float32)
    state_tokens, _ = policy._state_tokens(states, episode_progress)
    state_tokens = state_tokens.unsqueeze(0).to(device=policy.device, dtype=policy.dtype)
    prompt_embeds = policy.prompt_embedding.to(device=policy.device, dtype=policy.dtype)
    generator = torch.Generator(device=policy.device).manual_seed(int(seed))

    video_condition = image_tensor.unsqueeze(2).to(dtype=pipe.vae.dtype)
    ref_latents = pipe.vae.encode(video_condition).latent_dist.mode().to(dtype=torch.float32)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(ref_latents.device, ref_latents.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(ref_latents.device, ref_latents.dtype)
    ref_latents = (ref_latents - latents_mean) * latents_std
    num_clean_latents = int(ref_latents.shape[2])
    total_latent_frames = (policy.model_image_frames - 1) // VAE_TEMPORAL_SCALE + 1
    num_future_latents = total_latent_frames - num_clean_latents
    if num_future_latents < 1:
        raise ValueError(
            f"reference uses {num_clean_latents}/{total_latent_frames} visual latents"
        )
    future_latents = torch.randn(
        (
            1,
            pipe.vae.config.z_dim,
            num_future_latents,
            ref_latents.shape[-2],
            ref_latents.shape[-1],
        ),
        generator=generator,
        device=policy.device,
        dtype=torch.float32,
    ).to(dtype=policy.dtype)
    action = torch.randn(
        (1, policy.action_horizon, policy.model_action_dim),
        generator=generator,
        device=policy.device,
        dtype=torch.float32,
    ).to(dtype=policy.dtype)
    ref_latents = ref_latents.to(dtype=policy.dtype)

    visual_scheduler = FlowMatchEulerDiscreteScheduler(shift=2.0)
    action_scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    visual_scheduler.set_timesteps(policy.num_inference_steps, device=policy.device)
    action_scheduler.set_timesteps(policy.num_inference_steps, device=policy.device)
    if len(visual_scheduler.timesteps) != len(action_scheduler.timesteps):
        raise RuntimeError("visual/action schedulers produced different step counts")

    batch_size = 1
    embodiment_id = torch.zeros((batch_size,), device=policy.device, dtype=torch.long)
    if not getattr(policy, "_joint_video_shape_logged", False):
        print(
            "joint video input latent shapes: "
            f"reference={tuple(ref_latents.shape)} future={tuple(future_latents.shape)}",
            flush=True,
        )
    for denoise_index, (visual_t, action_t) in enumerate(
        zip(visual_scheduler.timesteps, action_scheduler.timesteps)
    ):
        action_hidden, action_rope, num_state_tokens, num_action_tokens = (
            pipe.transformer.encode_action_tokens(
                state_tokens, action, embodiment_id=embodiment_id
            )
        )
        visual_hidden, visual_rope, num_clean_tokens, num_future_tokens, visual_grid = (
            pipe.transformer.encode_visual_tokens(ref_latents, future_latents)
        )
        if denoise_index == 0 and not getattr(policy, "_joint_video_shape_logged", False):
            print(
                "joint video token counts: "
                f"state={num_state_tokens} ref={num_clean_tokens} "
                f"action={num_action_tokens} future={num_future_tokens}",
                flush=True,
            )
        timestep = torch.zeros(
            (batch_size, num_state_tokens + num_clean_tokens + num_action_tokens + num_future_tokens),
            device=policy.device,
            dtype=policy.dtype,
        )
        action_start = num_state_tokens + num_clean_tokens
        timestep[:, action_start : action_start + num_action_tokens] = action_t.to(policy.dtype)
        timestep[:, action_start + num_action_tokens :] = visual_t.to(policy.dtype)
        action_ts, visual_ts = pipe.transformer._split_timestep(
            timestep,
            num_state_tokens,
            num_action_tokens,
            num_clean_tokens,
            num_future_tokens,
        )
        if action_ts.shape[1] != num_state_tokens + num_action_tokens:
            raise RuntimeError(f"unexpected action timestep shape {tuple(action_ts.shape)}")
        if visual_ts.shape[1] != num_clean_tokens + num_future_tokens:
            raise RuntimeError(f"unexpected visual timestep shape {tuple(visual_ts.shape)}")
        visual_pred, action_pred = pipe.transformer._forward_tokens(
            action_hidden_states=action_hidden,
            visual_hidden_states=visual_hidden,
            action_rotary_emb=action_rope,
            visual_rotary_emb=visual_rope,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_clean_tokens,
            return_dict=False,
            output_action=True,
            output_visual=True,
            visual_grid_info=visual_grid,
            embodiment_id=embodiment_id,
        )
        visual_pred = visual_pred[:, :, -future_latents.shape[2] :]
        future_latents = visual_scheduler.step(
            visual_pred, visual_t, future_latents, return_dict=False
        )[0]
        action = action_scheduler.step(action_pred, action_t, action, return_dict=False)[0]

    decoded_latents = torch.cat([ref_latents, future_latents], dim=2).to(pipe.vae.dtype)
    if not getattr(policy, "_joint_video_shape_logged", False):
        print(
            "joint video latent shapes: "
            f"reference={tuple(ref_latents.shape)} future={tuple(future_latents.shape)} "
            f"decode_input={tuple(decoded_latents.shape)}",
            flush=True,
        )
    latents_mean = latents_mean.to(decoded_latents.device, decoded_latents.dtype)
    latents_std = latents_std.to(decoded_latents.device, decoded_latents.dtype)
    decoded_latents = decoded_latents / latents_std + latents_mean
    decoded = pipe.vae.decode(decoded_latents, return_dict=False)[0]
    generated = pipe.video_processor.postprocess_video(decoded, output_type="np")
    generated = _video_array(generated)
    if not getattr(policy, "_joint_video_shape_logged", False):
        print(f"joint video decoded shape: {tuple(generated.shape)}", flush=True)
        policy._joint_video_shape_logged = True
    return generated


def annotate_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    time_seconds: float,
    segment: Segment,
    is_peak_sample: bool,
) -> np.ndarray:
    if ground_truth.shape != prediction.shape:
        raise ValueError(f"GT/prediction shape mismatch: {ground_truth.shape} != {prediction.shape}")
    height, width = ground_truth.shape[:2]
    header = 72
    canvas = np.zeros((height + header, width * 2, 3), dtype=np.uint8)
    canvas[header:, :width] = ground_truth
    canvas[header:, width:] = prediction
    cv2.putText(canvas, "Ground truth", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "MoT prediction (reset 0.48s)",
        (width + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    detail = (
        f"ep {segment.episode} t={time_seconds:.2f}s "
        f"{segment.dominant_hz:.2f}Hz RMS={segment.angle_rms_degrees:.3f}deg"
    )
    cv2.putText(canvas, detail, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)
    if is_peak_sample:
        cv2.rectangle(canvas, (1, 1), (width * 2 - 2, height + header - 2), (255, 40, 40), 5)
        cv2.putText(
            canvas,
            f"PEAK nearest {segment.peak_seconds:.2f}s",
            (width + 12, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 80, 80),
            2,
        )
    return canvas


def write_comparison_video(
    policy: SonicHistoryPolicy,
    dataset: Path,
    output_dir: Path,
    segment: Segment,
    seed: int,
) -> dict:
    states, timestamps, fps, video_path = load_episode_metadata(dataset, segment.episode)
    start_frame = int(round(segment.start_seconds * fps))
    requested_end = int(round(segment.end_seconds * fps))
    end_frame = min(requested_end, len(states) - 1)
    if start_frame >= end_frame:
        raise ValueError(f"episode {segment.episode} has no frames in requested segment")
    rgb_frames = load_rgb_frames(video_path, start_frame, end_frame)
    sampled_indices: list[int] = []
    gt_frames: list[np.ndarray] = []
    predicted_frames: list[np.ndarray] = []
    chunk_latencies = []

    chunk_stride = IMAGE_FRAME_OFFSETS[-1]
    for chunk_number, chunk_start in enumerate(range(start_frame, end_frame, chunk_stride)):
        valid_offsets = [offset for offset in IMAGE_FRAME_OFFSETS if chunk_start + offset <= end_frame]
        if not valid_offsets:
            continue
        reference = rgb_frames[chunk_start - start_frame]
        started = time.perf_counter()
        predicted = predict_joint_chunk(
            policy,
            reference,
            state_history(states, chunk_start),
            episode_progress=float(chunk_start / max(len(states) - 1, 1)),
            seed=seed + segment.episode * 10000 + chunk_start,
        )
        chunk_latencies.append(time.perf_counter() - started)
        if predicted.shape[0] != len(IMAGE_FRAME_OFFSETS):
            raise ValueError(
                f"model returned {predicted.shape[0]} frames; expected {len(IMAGE_FRAME_OFFSETS)}"
            )
        for offset_index, offset in enumerate(valid_offsets):
            frame_index = chunk_start + offset
            if sampled_indices and frame_index == sampled_indices[-1]:
                continue
            sampled_indices.append(frame_index)
            gt_image = _center_crop_resize(
                _to_pil(rgb_frames[frame_index - start_frame]), policy.width, policy.height
            )
            gt_frames.append(np.asarray(gt_image, dtype=np.uint8))
            predicted_frames.append(predicted[offset_index])
        print(
            f"episode={segment.episode} chunk={chunk_number + 1} "
            f"start={chunk_start / fps:.2f}s latency={chunk_latencies[-1]:.3f}s",
            flush=True,
        )

    if not sampled_indices:
        raise RuntimeError(f"episode {segment.episode} produced no comparison frames")
    peak_frame = int(round(segment.peak_seconds * fps))
    nearest_peak = min(sampled_indices, key=lambda index: abs(index - peak_frame))
    output_path = output_dir / f"episode_{segment.episode:03d}_gt_vs_pred.mp4"
    output_fps = fps / (IMAGE_FRAME_OFFSETS[1] - IMAGE_FRAME_OFFSETS[0])
    first = annotate_comparison(
        gt_frames[0], predicted_frames[0], sampled_indices[0] / fps, segment, sampled_indices[0] == nearest_peak
    )
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {output_path}")
    try:
        for frame_index, gt_image, predicted in zip(sampled_indices, gt_frames, predicted_frames):
            canvas = annotate_comparison(
                gt_image,
                predicted,
                frame_index / fps,
                segment,
                frame_index == nearest_peak,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return {
        **asdict(segment),
        "output": str(output_path),
        "source_video": str(video_path),
        "source_fps": fps,
        "comparison_fps": output_fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sampled_frames": len(sampled_indices),
        "nearest_peak_sample_seconds": nearest_peak / fps,
        "teacher_forcing_interval_seconds": chunk_stride / fps,
        "mean_chunk_latency_seconds": float(np.mean(chunk_latencies)),
    }


def validate_inputs(args: argparse.Namespace, segments: list[Segment]) -> tuple[Path, Path, Path, Path]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != "transformer" and (checkpoint / "transformer").is_dir():
        checkpoint = checkpoint / "transformer"
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    norm_stats = (
        args.norm_stats.expanduser().resolve()
        if args.norm_stats
        else dataset / "norm_stats_h24.json"
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
        PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers" / "vae",
    ]
    weight_files = sorted(checkpoint.glob("*.safetensors")) + sorted(checkpoint.glob("*.bin"))
    if not weight_files:
        required.append(checkpoint / "<transformer weights>")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inference inputs: {missing}")
    for segment in segments:
        load_episode_metadata(dataset, segment.episode)
    return checkpoint, dataset, output_dir, prompt_embedding


def main() -> None:
    args = parse_args()
    if args.num_inference_steps <= 0:
        raise SystemExit("--num-inference-steps must be positive")
    segments = parse_segments(args.segment)
    checkpoint, dataset, output_dir, prompt_embedding = validate_inputs(args, segments)
    print(f"checkpoint: {checkpoint}")
    print(f"dataset: {dataset}")
    print(f"segments: {[asdict(segment) for segment in segments]}")
    if args.validate_only:
        print("validation complete; model was not loaded and no output was written")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "transformer": str(checkpoint),
        "base_model": str((PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers").resolve()),
        "norm_stats": str(
            args.norm_stats.expanduser().resolve()
            if args.norm_stats
            else dataset / "norm_stats_h24.json"
        ),
        "prompt_embedding": str(prompt_embedding),
        "prompt": "grab graduated cylinder and pour water into beaker",
        "image_frame_offsets": list(IMAGE_FRAME_OFFSETS),
        "state_history_offsets": list(STATE_HISTORY_OFFSETS),
        "state_history_tokens": len(STATE_HISTORY_OFFSETS),
        "visual_history_frames": 1,
        "append_state_velocity_token": False,
        "append_episode_progress_token": False,
        "action_horizon": 24,
        "model_image_frames": len(IMAGE_FRAME_OFFSETS),
        "physical_action_dim": 13,
        "model_action_dim": 32,
        "delta_mask": [True] * 7 + [False] * 6,
        "fps": MODEL_FPS,
        "image_size": [320, 384],
        "num_inference_steps": args.num_inference_steps,
        "model_dtype": "bf16",
        "action_flow_shift": 5.0,
        "visual_flow_shift": 2.0,
        "vae_temporal_scale": VAE_TEMPORAL_SCALE,
        "comparison_mode": "teacher_forced_reference_reset_each_24_frames",
    }
    manifest_path = output_dir / "video_prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    policy = SonicHistoryPolicy(manifest_path, device=args.device)
    results = []
    for segment in segments:
        results.append(
            write_comparison_video(policy, dataset, output_dir, segment, args.seed)
        )
    summary = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "segments": results,
    }
    summary_path = output_dir / "video_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

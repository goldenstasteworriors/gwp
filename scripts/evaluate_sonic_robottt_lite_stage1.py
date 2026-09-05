#!/usr/bin/env python3
"""Evaluate one RoboTTT-lite stage-1 checkpoint on held-out trajectories.

This benchmark streams each held-out episode once at the policy replan rate and
measures the deterministic teacher-forced action flow-matching objective at
uniformly selected anchors.  Every checkpoint sees the same anchor frames,
image-transform seed, diffusion timestep, and action noise.  Persistent models
are also evaluated with their fast weights forcibly reset at every measured
replan, producing a paired estimate of the value of temporal memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from einops import rearrange


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

from world_action_model.datasets.wam_lerobot_dataset import (  # noqa: E402
    WAMLeRobotSequenceDataset,
)
from world_action_model.models import CasualWorldActionTransformer_MoT  # noqa: E402
from world_action_model.trainer.wa_casual_trainer_mot import (  # noqa: E402
    canonical_checkpoint_parameter_name,
    load_pretrained_weights,
)
from world_action_model.transforms.wa_transforms_lerobot_pretrain import (  # noqa: E402
    WALeRobotSequenceTransformsPretrain,
)


PHYSICAL_ACTION_DIM = 13
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 48
REPLAN_STRIDE = 8
EXECUTED_ACTION_STEPS = REPLAN_STRIDE
ACTION_FLOW_SHIFT = 5.0
VIEW_KEY = "observation.images.ego_view"
DELTA_MASK = (True,) * 7 + (False,) * 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-episode", type=int, default=24)
    parser.add_argument("--stream-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--verify-batch-equivalence", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_compact_state(path: Path) -> dict[str, torch.Tensor]:
    weight_files = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No checkpoint weights found under {path}")
    state: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            value = load_file(str(weight_file), device="cpu")
        else:
            value = torch.load(weight_file, map_location="cpu", weights_only=True)
        if isinstance(value, dict) and isinstance(value.get("model"), dict):
            value = value["model"]
        if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
            value = value["state_dict"]
        for name, tensor in value.items():
            canonical = canonical_checkpoint_parameter_name(name)
            if canonical in state:
                raise RuntimeError(f"Duplicate checkpoint tensor: {canonical}")
            state[canonical] = tensor
    return state


def load_model(
    checkpoint: Path,
    base_checkpoint: Path,
    device: str,
) -> tuple[CasualWorldActionTransformer_MoT, dict]:
    model = CasualWorldActionTransformer_MoT.from_config(checkpoint)
    model = load_pretrained_weights(
        model,
        str(base_checkpoint),
        skip_action_expert=False,
        strict_load=False,
        min_load_coverage=0.98,
    )
    compact_state = load_compact_state(checkpoint)
    model_keys = set(model.state_dict())
    unexpected = sorted(set(compact_state) - model_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected compact checkpoint keys: {unexpected[:20]}")
    missing, load_unexpected = model.load_state_dict(compact_state, strict=False)
    if load_unexpected:
        raise RuntimeError(f"Unexpected keys while overlaying checkpoint: {load_unexpected[:20]}")
    memory_keys = sorted(name for name in compact_state if name.startswith("memory_layers."))
    if not memory_keys:
        raise RuntimeError("Compact checkpoint contains no memory tensors")
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    report = {
        "base_load": model._pretrained_load_report,
        "compact_tensor_count": len(compact_state),
        "memory_tensor_count": len(memory_keys),
        "base_tensors_retained": len(missing),
    }
    return model, report


def build_dataset_and_transform(experiment: dict):
    dataset_root = Path(experiment["dataset"]).resolve()
    norm_stats = Path(experiment["norm_stats"]).resolve()
    episodes = [int(value) for value in experiment["evaluation_episodes"]]
    trained_context_length = int(experiment["context_length_replans"])
    # Evaluation is streamed one replan at a time.  This avoids repeatedly
    # decoding overlapping video windows and matches deployment-time state
    # propagation; the checkpoint itself retains its trained decay half-life.
    stream_context_length = 1
    dataset = WAMLeRobotSequenceDataset(
        data_path=str(dataset_root),
        video_backend="pyav",
        t5_load_from="dir",
        t5_cfg={
            "t5_embedding_dir": str(dataset_root / "t5_embedding"),
            "t5_embedding_pattern": "episode_{episode_index:06d}.pt",
            "t5_embedding_key": "t5_embedding",
        },
        t5_cache_size=4096,
        robotype="sonic_g1",
        episodes=episodes,
        context_length=stream_context_length,
        replan_stride=REPLAN_STRIDE,
        action_horizon=ACTION_HORIZON,
        view_keys=[VIEW_KEY],
    )
    transform = WALeRobotSequenceTransformsPretrain(
        is_train=False,
        dst_size=(320, 384),
        num_frames=ACTION_HORIZON,
        norm_path=[{"path": str(norm_stats), "data_paths": [str(dataset_root)]}],
        robotype_to_embodiment_id={"sonic_g1": 0},
        robotype_default_embodiment_id=0,
        model_action_dim=MODEL_ACTION_DIM,
        delta_mask_by_embodiment_id={"0": DELTA_MASK},
        norm_use_quantiles=True,
        norm_enable_clamp=True,
        image_cfg={"mask_generator": {"max_ref_frames": 1, "start": 1, "factor": 4}},
        image_augmentation={"enabled": False},
        num_views=1,
        view_keys=[VIEW_KEY],
        state_history_tokens=1,
        emit_action_dim_mask=True,
        emit_action_time_mask=True,
        expected_state_dim=PHYSICAL_ACTION_DIM,
        max_prompt_len=64,
        subtask_prob=0,
        context_length=stream_context_length,
        replan_stride=REPLAN_STRIDE,
    )
    return dataset, transform, episodes, trained_context_length


def select_anchors(dataset, episodes: list[int], samples_per_episode: int):
    dataset.open()
    if dataset._sequence_anchor_indices is None:
        raise RuntimeError("Sequence anchors were not initialized")
    hf_dataset = dataset.dataset.hf_dataset
    episode_column = hf_dataset["episode_index"]
    frame_column = hf_dataset["frame_index"]
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for sequence_index, row_index in enumerate(dataset._sequence_anchor_indices):
        episode = int(episode_column[row_index])
        frame = int(frame_column[row_index])
        candidates[episode].append((sequence_index, frame))

    all_anchors = []
    selected_keys = set()
    coverage = {}
    for episode in episodes:
        values = candidates[episode]
        if not values:
            raise RuntimeError(f"No sequence anchors found for episode {episode}")
        count = len(values) if samples_per_episode <= 0 else min(samples_per_episode, len(values))
        positions = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
        positions = np.unique(positions)
        chosen = [values[int(position)] for position in positions]
        all_anchors.extend((episode, frame, index) for index, frame in values)
        selected_keys.update((episode, frame) for _, frame in chosen)
        coverage[str(episode)] = {
            "available_anchors": len(values),
            "selected_anchors": len(chosen),
            "first_frame": chosen[0][1],
            "last_frame": chosen[-1][1],
        }
    return all_anchors, selected_keys, coverage


def stable_seed(base_seed: int, episode: int, frame: int, salt: int = 0) -> int:
    value = (
        int(base_seed) * 1_000_003
        + int(episode) * 97_409
        + (int(frame) + 100_000) * 65_537
        + int(salt) * 7_919
    )
    return value % (2**63 - 1)


def action_corruption(
    action: torch.Tensor,
    *,
    episode: int,
    anchor_frame: int,
    context_length: int,
    base_seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noises = []
    timesteps = []
    sigmas = []
    for step in range(context_length):
        absolute_frame = anchor_frame + (step - context_length + 1) * REPLAN_STRIDE
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed(base_seed, episode, absolute_frame)
        )
        uniform = float(torch.rand((), generator=generator).item())
        sigma = ACTION_FLOW_SHIFT * uniform / (
            1.0 + (ACTION_FLOW_SHIFT - 1.0) * uniform
        )
        timestep = int(round(sigma * 1000.0))
        sigma = timestep / 1000.0
        noises.append(torch.randn(action.shape[1:], generator=generator))
        timesteps.append(timestep)
        sigmas.append(sigma)
    noise = torch.stack(noises).to(device=device, dtype=torch.float32)
    timestep = torch.tensor(timesteps, device=device, dtype=torch.long)
    sigma = torch.tensor(sigmas, device=device, dtype=torch.float32)[:, None, None]
    action = action.to(device=device, dtype=torch.float32)
    target = noise - action
    noisy = noise * sigma + action * (1.0 - sigma)
    return noisy.to(torch.bfloat16), target, timestep, sigma


@torch.no_grad()
def encode_images(
    vae: AutoencoderKLWan,
    images: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    images = rearrange(images.to(device=vae.device, dtype=vae.dtype), "b t c h w -> b c t h w")
    latents = vae.encode(images).latent_dist.mode()
    return (latents - latent_mean) * latent_std


def masked_mean(error: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.to(dtype=error.dtype)
    return float((error * weight).sum().div(weight.sum().clamp_min(1.0)).item())


def current_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noisy_action: torch.Tensor,
    clean_action: torch.Tensor,
    sigma: torch.Tensor,
    time_mask: torch.Tensor,
    dim_mask: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    error = (prediction - target).square()
    base_mask = time_mask[:, None] & dim_mask[None, :]
    arm_mask = base_mask.clone()
    arm_mask[:, 7:] = False
    hand_mask = base_mask.clone()
    hand_mask[:, :7] = False
    first_mask = base_mask.clone()
    first_mask[1:] = False
    executed_prefix_mask = base_mask.clone()
    executed_prefix_mask[EXECUTED_ACTION_STEPS:] = False
    predicted_clean = noisy_action.float() - sigma.float() * prediction
    clean_error = (predicted_clean - clean_action.float()).square()
    return {
        "flow_mse": masked_mean(error, base_mask),
        "flow_mse_arm": masked_mean(error, arm_mask),
        "flow_mse_hand": masked_mean(error, hand_mask),
        "flow_mse_first_action": masked_mean(error, first_mask),
        "flow_mse_executed_prefix": masked_mean(error, executed_prefix_mask),
        "denoised_action_mse": masked_mean(clean_error, base_mask),
        "denoised_action_mse_executed_prefix": masked_mean(
            clean_error, executed_prefix_mask
        ),
    }


def stack_stream_steps(steps: list[dict]) -> dict:
    if not steps:
        raise ValueError("Cannot stack an empty stream segment")
    expected_keys = set(steps[0])
    if any(set(step) != expected_keys for step in steps[1:]):
        raise ValueError("Stream steps do not expose identical tensor keys")
    return {
        key: torch.cat([step[key] for step in steps], dim=0)
        for key in expected_keys
    }


@torch.no_grad()
def evaluate_stream_segment(
    model,
    vae,
    transformed: dict,
    *,
    episode: int,
    anchor_frames: list[int],
    memory_state,
    native_persistent: bool,
    measure_mask: list[bool],
    include_reset: bool,
    verify_batch_equivalence: bool,
    base_seed: int,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: str,
) -> tuple[dict, list[dict[str, dict[str, float]]]]:
    images = transformed["images"].to(device=device)
    action = transformed["action"].to(device=device)
    state = transformed["state"].to(device=device, dtype=torch.bfloat16)
    prompt = transformed["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
    embodiment = transformed["embodiment_id"].to(device=device)
    context_valid = transformed["memory_context_valid"].to(device=device).bool()
    action_time_mask = transformed["action_time_mask"].to(device=device).bool()
    action_dim_mask = transformed["action_dim_mask"].to(device=device).bool()
    if len(anchor_frames) == 1:
        # Preserve the validated deployment-equivalent singleton path exactly
        # and avoid an unnecessary concatenate-on-one operation.
        noisy_action, action_target, action_timestep, action_sigma = action_corruption(
            action,
            episode=episode,
            anchor_frame=anchor_frames[0],
            context_length=1,
            base_seed=base_seed,
            device=device,
        )
    else:
        corruptions = [
            action_corruption(
                action[index : index + 1],
                episode=episode,
                anchor_frame=frame,
                context_length=1,
                base_seed=base_seed,
                device=device,
            )
            for index, frame in enumerate(anchor_frames)
        ]
        noisy_action, action_target, action_timestep, action_sigma = tuple(
            torch.cat([values[index] for values in corruptions], dim=0)
            for index in range(4)
        )
    ref_latents = encode_images(vae, images, latent_mean, latent_std).to(torch.bfloat16)
    sequence_length = len(anchor_frames)
    if action.shape[0] != sequence_length or len(measure_mask) != sequence_length:
        raise ValueError("Stream segment metadata does not match its tensor batch")
    num_state_tokens = state.shape[1]
    num_action_tokens = action.shape[1]
    patch_t, patch_h, patch_w = model.config.patch_size
    num_ref_tokens = (
        ref_latents.shape[2] // patch_t
        * (ref_latents.shape[3] // patch_h)
        * (ref_latents.shape[4] // patch_w)
    )
    timestep = torch.zeros(
        sequence_length,
        num_state_tokens + num_ref_tokens + num_action_tokens,
        device=device,
        dtype=ref_latents.dtype,
    )
    timestep[:, num_state_tokens + num_ref_tokens :] = action_timestep[:, None]

    def forward_one(state_value, persistent: bool):
        # The existing deployment fast path does not yet expose recurrent
        # memory state.  Call the sequence-capable action-only path directly
        # while retaining eval mode and no-grad semantics.
        _, prediction, next_state = model._forward_train_action_only_no_video(
            ref_latents=ref_latents,
            timestep=timestep,
            encoder_hidden_states=prompt,
            return_dict=False,
            action=noisy_action,
            state=state,
            embodiment_id=embodiment,
            memory_sequence_length=sequence_length,
            memory_state=state_value,
            memory_update_mask=context_valid[None],
            memory_persistent=persistent,
            return_memory_state=True,
        )
        # Keep the flattened stream-step batch axis.  The caller indexes one
        # [action_horizon, action_dim] prediction per replan below.
        return prediction, next_state

    native_prediction, next_memory_state = forward_one(
        memory_state, native_persistent
    )
    if verify_batch_equivalence and memory_state is None and sequence_length > 1:
        _, individual_prediction, _ = model._forward_train_action_only_no_video(
            ref_latents=ref_latents[:1],
            timestep=timestep[:1],
            encoder_hidden_states=prompt[:1],
            return_dict=False,
            action=noisy_action[:1],
            state=state[:1],
            embodiment_id=embodiment[:1],
            memory_sequence_length=1,
            memory_state=None,
            memory_update_mask=context_valid[:1][None],
            memory_persistent=native_persistent,
            return_memory_state=True,
        )
        max_difference = float(
            (native_prediction[0].float() - individual_prediction[0].float())
            .abs()
            .max()
            .item()
        )
        print(
            f"batch_equivalence episode={episode} max_abs_diff={max_difference:.8g}",
            flush=True,
        )
        if max_difference > 1e-2:
            raise RuntimeError(
                "Batched recurrent output differs from single-step output at "
                f"the first replan: max_abs_diff={max_difference:.8g}"
            )
    reset_prediction = None
    if include_reset and native_persistent and any(measure_mask):
        reset_prediction, _ = forward_one(None, False)

    results = []
    for index, measure in enumerate(measure_mask):
        if not measure:
            results.append({})
            continue
        row = {
            "native": current_metrics(
                native_prediction[index],
                action_target[index],
                noisy_action[index],
                action[index],
                action_sigma[index],
                action_time_mask[index],
                action_dim_mask[index],
            )
        }
        if reset_prediction is not None:
            row["forced_reset"] = current_metrics(
                reset_prediction[index],
                action_target[index],
                noisy_action[index],
                action[index],
                action_sigma[index],
                action_time_mask[index],
                action_dim_mask[index],
            )
        results.append(row)
    return next_memory_state, results


def metric_summary(records: list[dict], mode: str) -> dict:
    mode_records = [row for row in records if row["mode"] == mode]
    metric_names = (
        "flow_mse",
        "flow_mse_arm",
        "flow_mse_hand",
        "flow_mse_first_action",
        "flow_mse_executed_prefix",
        "denoised_action_mse",
        "denoised_action_mse_executed_prefix",
    )
    summary = {"samples": len(mode_records)}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in mode_records], dtype=np.float64)
        summary[metric] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"{metric}_sem"] = (
            float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        )
    per_episode = {}
    for episode in sorted({row["episode"] for row in mode_records}):
        selected = [row for row in mode_records if row["episode"] == episode]
        per_episode[str(episode)] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in metric_names
        }
        per_episode[str(episode)]["samples"] = len(selected)
    summary["per_episode"] = per_episode
    return summary


def main() -> None:
    args = parse_args()
    for path in (
        args.experiment,
        args.checkpoint,
        args.base_checkpoint,
        args.base_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.samples_per_episode < 0:
        raise ValueError("samples-per-episode must be non-negative")
    if args.stream_batch_size != 1:
        raise ValueError(
            "Only stream-batch-size=1 is supported because this benchmark is "
            "defined as sequential singleton deployment inference"
        )
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    experiment = read_json(args.experiment)
    dataset, transform, episodes, trained_context_length = build_dataset_and_transform(
        experiment
    )
    all_anchors, selected_keys, coverage = select_anchors(
        dataset, episodes, args.samples_per_episode
    )
    model, load_report = load_model(args.checkpoint, args.base_checkpoint, args.device)
    vae = AutoencoderKLWan.from_pretrained(args.base_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device=args.device, dtype=torch.float32)
    vae.eval()
    latent_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(device=args.device, dtype=vae.dtype)
    latent_std = (1.0 / torch.tensor(vae.config.latents_std)).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(device=args.device, dtype=vae.dtype)

    native_persistent = bool(experiment["memory_persistent"])
    records = []
    started = time.perf_counter()
    processed = 0
    selected_total = len(selected_keys)
    last_selected_frame = {
        episode: max(frame for selected_episode, frame in selected_keys if selected_episode == episode)
        for episode in episodes
    }
    anchors_by_episode = defaultdict(list)
    for episode, frame, sequence_index in all_anchors:
        if frame <= last_selected_frame[episode]:
            anchors_by_episode[episode].append((frame, sequence_index))
    try:
        for episode in episodes:
            memory_state = None
            episode_anchors = anchors_by_episode[episode]
            if not native_persistent:
                episode_anchors = [
                    value
                    for value in episode_anchors
                    if (episode, value[0]) in selected_keys
                ]
            for batch_start in range(0, len(episode_anchors), args.stream_batch_size):
                anchor_batch = episode_anchors[
                    batch_start : batch_start + args.stream_batch_size
                ]
                frames = [frame for frame, _ in anchor_batch]
                measure_mask = [
                    (episode, frame) in selected_keys for frame in frames
                ]
                transformed_steps = []
                for frame, sequence_index in anchor_batch:
                    sample_seed = stable_seed(args.seed, episode, frame, salt=1)
                    random.seed(sample_seed)
                    np.random.seed(sample_seed % (2**32 - 1))
                    torch.manual_seed(sample_seed)
                    raw = dataset._get_data(sequence_index)
                    # LeRobot squeezes the temporal dimension for a one-frame
                    # video query, while the sequence transform requires it.
                    image = torch.as_tensor(raw[VIEW_KEY])
                    if image.ndim == 3:
                        raw[VIEW_KEY] = image.unsqueeze(0)
                    transformed_steps.append(transform(raw))
                transformed = (
                    transformed_steps[0]
                    if len(transformed_steps) == 1
                    else stack_stream_steps(transformed_steps)
                )
                memory_state, segment_results = evaluate_stream_segment(
                    model,
                    vae,
                    transformed,
                    episode=episode,
                    anchor_frames=frames,
                    memory_state=memory_state,
                    native_persistent=native_persistent,
                    measure_mask=measure_mask,
                    include_reset=not args.native_only,
                    verify_batch_equivalence=args.verify_batch_equivalence,
                    base_seed=args.seed,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    device=args.device,
                )
                for frame, mode_results in zip(frames, segment_results):
                    if not mode_results:
                        continue
                    processed += 1
                    for mode, metrics in mode_results.items():
                        records.append(
                            {
                                "episode": episode,
                                "frame": frame,
                                "mode": mode,
                                **metrics,
                            }
                        )
                    native_loss = mode_results["native"]["flow_mse"]
                    message = (
                        f"[{processed}/{selected_total}] episode={episode} "
                        f"frame={frame} native_flow_mse={native_loss:.8f}"
                    )
                    if "forced_reset" in mode_results:
                        reset_loss = mode_results["forced_reset"]["flow_mse"]
                        message += f" reset_flow_mse={reset_loss:.8f}"
                    print(message, flush=True)
    finally:
        dataset.close()

    modes = sorted({row["mode"] for row in records})
    summary = {mode: metric_summary(records, mode) for mode in modes}
    paired_memory = None
    if "forced_reset" in summary:
        native = summary["native"]["flow_mse"]
        reset = summary["forced_reset"]["flow_mse"]
        paired = []
        native_rows = {
            (row["episode"], row["frame"]): row
            for row in records
            if row["mode"] == "native"
        }
        for row in records:
            if row["mode"] != "forced_reset":
                continue
            key = (row["episode"], row["frame"])
            paired.append(row["flow_mse"] - native_rows[key]["flow_mse"])
        paired_array = np.asarray(paired, dtype=np.float64)
        paired_memory = {
            "reset_minus_native_flow_mse": float(reset - native),
            "relative_native_improvement_percent": float(
                (reset - native) / max(reset, 1e-12) * 100.0
            ),
            "paired_sample_mean": float(paired_array.mean()),
            "paired_sample_sem": float(
                paired_array.std(ddof=1) / math.sqrt(len(paired_array))
            ) if len(paired_array) > 1 else 0.0,
            "positive_means_memory_helped": True,
        }
        native_prefix = summary["native"]["flow_mse_executed_prefix"]
        reset_prefix = summary["forced_reset"]["flow_mse_executed_prefix"]
        paired_prefix = []
        for row in records:
            if row["mode"] != "forced_reset":
                continue
            key = (row["episode"], row["frame"])
            paired_prefix.append(
                row["flow_mse_executed_prefix"]
                - native_rows[key]["flow_mse_executed_prefix"]
            )
        paired_prefix_array = np.asarray(paired_prefix, dtype=np.float64)
        paired_memory.update(
            {
                "reset_minus_native_executed_prefix_flow_mse": float(
                    reset_prefix - native_prefix
                ),
                "relative_native_executed_prefix_improvement_percent": float(
                    (reset_prefix - native_prefix) / max(reset_prefix, 1e-12) * 100.0
                ),
                "paired_executed_prefix_sample_mean": float(
                    paired_prefix_array.mean()
                ),
                "paired_executed_prefix_sample_sem": float(
                    paired_prefix_array.std(ddof=1)
                    / math.sqrt(len(paired_prefix_array))
                )
                if len(paired_prefix_array) > 1
                else 0.0,
            }
        )

    result = {
        "benchmark": "sonic_robottt_lite_stage1_teacher_forced_flow_matching",
        "experiment": str(args.experiment.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "dataset": experiment["dataset"],
        "norm_stats": experiment["norm_stats"],
        "episodes": episodes,
        "trained_context_length_replans": trained_context_length,
        "context_seconds": experiment["context_seconds"],
        "evaluation_stream": "full held-out episode at stride-8 replans",
        "native_persistent": native_persistent,
        "samples_per_episode_requested": args.samples_per_episode,
        "selected_samples": selected_total,
        "anchor_coverage": coverage,
        "seed": args.seed,
        "stream_batch_size": args.stream_batch_size,
        "batch_equivalence_verified": args.verify_batch_equivalence,
        "metric_scope": {
            "primary": "normalized action flow-field MSE at the current replan",
            "arm_dimensions": [0, 7],
            "hand_dimensions": [7, 13],
            "action_horizon": ACTION_HORIZON,
            "executed_action_prefix": EXECUTED_ACTION_STEPS,
            "note": "Teacher-forced offline metric; not a closed-loop success rate.",
        },
        "load_report": load_report,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
        "paired_memory_effect": paired_memory,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": summary, "paired_memory_effect": paired_memory}, indent=2))


if __name__ == "__main__":
    main()

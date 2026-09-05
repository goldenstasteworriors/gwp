#!/usr/bin/env python3
"""Evaluate one RoboTTT-lite stage-1 checkpoint on held-out trajectories.

This benchmark measures the deterministic teacher-forced action flow-matching
objective used during stage-1 training.  Every checkpoint sees the same anchor
frames, image-transform seed, diffusion timestep, and action noise.  Persistent
models are also evaluated with their fast weights forcibly reset at every
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
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--native-only", action="store_true")
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
    context_length = int(experiment["context_length_replans"])
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
        context_length=context_length,
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
        context_length=context_length,
        replan_stride=REPLAN_STRIDE,
    )
    return dataset, transform, episodes, context_length


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

    selected = []
    coverage = {}
    for episode in episodes:
        values = candidates[episode]
        if not values:
            raise RuntimeError(f"No sequence anchors found for episode {episode}")
        count = len(values) if samples_per_episode <= 0 else min(samples_per_episode, len(values))
        positions = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
        positions = np.unique(positions)
        chosen = [values[int(position)] for position in positions]
        selected.extend((episode, frame, index) for index, frame in chosen)
        coverage[str(episode)] = {
            "available_anchors": len(values),
            "selected_anchors": len(chosen),
            "first_frame": chosen[0][1],
            "last_frame": chosen[-1][1],
        }
    return selected, coverage


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
    predicted_clean = noisy_action.float() - sigma.float() * prediction
    clean_error = (predicted_clean - clean_action.float()).square()
    return {
        "flow_mse": masked_mean(error, base_mask),
        "flow_mse_arm": masked_mean(error, arm_mask),
        "flow_mse_hand": masked_mean(error, hand_mask),
        "flow_mse_first_action": masked_mean(error, first_mask),
        "denoised_action_mse": masked_mean(clean_error, base_mask),
    }


@torch.no_grad()
def evaluate_modes(
    model,
    vae,
    transformed: dict,
    *,
    episode: int,
    anchor_frame: int,
    context_length: int,
    segment_length: int,
    native_persistent: bool,
    include_reset: bool,
    base_seed: int,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: str,
) -> dict[str, dict[str, float]]:
    images = transformed["images"].to(device=device)
    action = transformed["action"].to(device=device)
    state = transformed["state"].to(device=device, dtype=torch.bfloat16)
    prompt = transformed["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
    embodiment = transformed["embodiment_id"].to(device=device)
    context_valid = transformed["memory_context_valid"].to(device=device).bool()
    action_time_mask = transformed["action_time_mask"].to(device=device).bool()
    action_dim_mask = transformed["action_dim_mask"].to(device=device).bool()
    noisy_action, action_target, action_timestep, action_sigma = action_corruption(
        action,
        episode=episode,
        anchor_frame=anchor_frame,
        context_length=context_length,
        base_seed=base_seed,
        device=device,
    )

    encoded_segments = []
    for start in range(0, context_length, segment_length):
        stop = min(start + segment_length, context_length)
        encoded_segments.append(
            (
                start,
                stop,
                encode_images(
                    vae, images[start:stop], latent_mean, latent_std
                ).to(torch.bfloat16),
            )
        )

    modes = {"native": native_persistent}
    if include_reset and native_persistent:
        modes["forced_reset"] = False
    results = {}
    for mode_name, persistent in modes.items():
        memory_state = None
        final_prediction = None
        for start, stop, ref_latents in encoded_segments:
            count = stop - start
            num_state_tokens = state[start:stop].shape[1]
            num_action_tokens = action[start:stop].shape[1]
            patch_t, patch_h, patch_w = model.config.patch_size
            num_ref_tokens = (
                ref_latents.shape[2] // patch_t
                * (ref_latents.shape[3] // patch_h)
                * (ref_latents.shape[4] // patch_w)
            )
            timestep = torch.zeros(
                count,
                num_state_tokens + num_ref_tokens + num_action_tokens,
                device=device,
                dtype=ref_latents.dtype,
            )
            timestep[
                :, num_state_tokens + num_ref_tokens :
            ] = action_timestep[start:stop, None]
            _, prediction, memory_state = model(
                ref_latents=ref_latents,
                timestep=timestep,
                encoder_hidden_states=prompt[start:stop],
                return_dict=False,
                action=noisy_action[start:stop],
                state=state[start:stop],
                embodiment_id=embodiment[start:stop],
                action_only=True,
                no_video_tokens=True,
                memory_sequence_length=count,
                memory_state=memory_state,
                memory_update_mask=context_valid[start:stop][None],
                memory_persistent=persistent,
                return_memory_state=True,
            )
            final_prediction = prediction[-1]
        if final_prediction is None:
            raise RuntimeError("No action prediction was produced")
        results[mode_name] = current_metrics(
            final_prediction,
            action_target[-1],
            noisy_action[-1],
            action[-1],
            action_sigma[-1],
            action_time_mask[-1],
            action_dim_mask[-1],
        )
    return results


def metric_summary(records: list[dict], mode: str) -> dict:
    mode_records = [row for row in records if row["mode"] == mode]
    metric_names = (
        "flow_mse",
        "flow_mse_arm",
        "flow_mse_hand",
        "flow_mse_first_action",
        "denoised_action_mse",
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
    if args.samples_per_episode < 0 or args.segment_length < 1:
        raise ValueError("samples-per-episode must be non-negative and segment-length positive")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    experiment = read_json(args.experiment)
    dataset, transform, episodes, context_length = build_dataset_and_transform(experiment)
    anchors, coverage = select_anchors(dataset, episodes, args.samples_per_episode)
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
    try:
        for sample_number, (episode, frame, sequence_index) in enumerate(anchors, 1):
            sample_seed = stable_seed(args.seed, episode, frame, salt=1)
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32 - 1))
            torch.manual_seed(sample_seed)
            transformed = transform(dataset._get_data(sequence_index))
            mode_results = evaluate_modes(
                model,
                vae,
                transformed,
                episode=episode,
                anchor_frame=frame,
                context_length=context_length,
                segment_length=args.segment_length,
                native_persistent=native_persistent,
                include_reset=not args.native_only,
                base_seed=args.seed,
                latent_mean=latent_mean,
                latent_std=latent_std,
                device=args.device,
            )
            for mode, metrics in mode_results.items():
                records.append(
                    {"episode": episode, "frame": frame, "mode": mode, **metrics}
                )
            native_loss = mode_results["native"]["flow_mse"]
            message = (
                f"[{sample_number}/{len(anchors)}] episode={episode} frame={frame} "
                f"native_flow_mse={native_loss:.8f}"
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

    result = {
        "benchmark": "sonic_robottt_lite_stage1_teacher_forced_flow_matching",
        "experiment": str(args.experiment.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "dataset": experiment["dataset"],
        "norm_stats": experiment["norm_stats"],
        "episodes": episodes,
        "context_length_replans": context_length,
        "context_seconds": experiment["context_seconds"],
        "native_persistent": native_persistent,
        "samples_per_episode_requested": args.samples_per_episode,
        "selected_samples": len(anchors),
        "anchor_coverage": coverage,
        "seed": args.seed,
        "segment_length": args.segment_length,
        "metric_scope": {
            "primary": "normalized action flow-field MSE at the current replan",
            "arm_dimensions": [0, 7],
            "hand_dimensions": [7, 13],
            "action_horizon": ACTION_HORIZON,
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

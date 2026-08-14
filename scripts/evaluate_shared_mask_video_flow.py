#!/usr/bin/env python3
"""Evaluate held-out shared mask-video flow matching and action sensitivity."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "third_party" / "giga-models",
    PROJECT_ROOT / "third_party" / "giga-datasets",
    PROJECT_ROOT / "third_party" / "giga-train",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from diffusers.models import AutoencoderKLWan  # noqa: E402

from evaluate_counterfactual_mask_head import (  # noqa: E402
    build_dataset,
    encode_reference,
    manifest_episode_indices,
    normalize_cached_actions,
    read_manifest,
    resolve_dtype,
    sha256_file,
    taskwise_shuffle_indices,
)
from world_action_model.models import CasualWorldActionTransformer_MoT  # noqa: E402


SCHEMA_VERSION = "shared-mask-video-flow-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--view-keys",
        nargs="+",
        default=("observation.images.image", "observation.images.wrist_image"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=(250, 500, 750)
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--save-per-sample", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def require_clean_tracked_worktree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("Refusing evaluation from a tracked-dirty worktree")


def validate_training_manifest(
    path: Path, checkpoint: Path, cache: Path
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or int(value.get("exit_code", -1)) != 0:
        raise ValueError("Training manifest is not complete")
    common = value.get("common", {})
    settings = value.get("variant_settings", {})
    if common.get("alignment_mode") != "shared_mask_video_flow":
        raise ValueError("Training manifest used a different alignment mode")
    if settings.get("alignment_mode") != "shared_mask_video_flow":
        raise ValueError("Variant settings used a different alignment mode")
    if Path(str(common.get("heldout_cache"))).resolve() != cache.resolve():
        raise ValueError("Training manifest refers to a different held-out cache")
    output_dir = Path(str(value.get("output_dir"))).resolve()
    try:
        checkpoint.resolve().relative_to(output_dir)
    except ValueError as error:
        raise ValueError("Checkpoint is outside the training output") from error
    return value


def aggregate(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot aggregate empty or non-finite metrics")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_mask_video(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    initial = sample["counterfactual_initial_robot_mask"]
    future = sample["counterfactual_robot_masks"]
    if initial.ndim != 2 or future.ndim != 3 or future.shape[0] != 4:
        raise ValueError(
            f"Unexpected counterfactual mask shapes: {initial.shape}, {future.shape}"
        )
    video = torch.cat((initial[None], future), dim=0)
    video = video.clamp(0.0, 1.0)[None, :, None].repeat(1, 1, 3, 1, 1)
    return video * 2.0 - 1.0


def encode_video(
    vae: AutoencoderKLWan,
    video: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    value = video.permute(0, 2, 1, 3, 4).to(device=vae.device, dtype=vae.dtype)
    latent = vae.encode(value).latent_dist.mode()
    return (latent - latent_mean) * latent_std


def latent_valid_mask(
    pixel_valid: torch.Tensor, latent_height: int, latent_width: int
) -> torch.Tensor:
    if pixel_valid.ndim != 3:
        raise ValueError(f"Expected [T,H,W] valid mask, got {pixel_valid.shape}")
    spatial = pixel_valid.float().amin(dim=0, keepdim=True)[None]
    spatial = F.interpolate(
        spatial, size=(latent_height, latent_width), mode="nearest"
    )
    return spatial.unsqueeze(2)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> float:
    expanded = torch.broadcast_to(valid.float(), prediction.shape)
    denominator = expanded.sum().clamp_min(1.0)
    value = ((prediction.float() - target.float()).pow(2) * expanded).sum()
    return float((value / denominator).item())


@torch.inference_mode()
def predict_flow(
    transformer: CasualWorldActionTransformer_MoT,
    ref_latents: torch.Tensor,
    noisy_future: torch.Tensor,
    action: torch.Tensor,
    sample: dict[str, torch.Tensor],
    timestep_value: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = ref_latents.device
    state = sample["state"].unsqueeze(0).to(device=device, dtype=dtype)
    prompt = sample["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
    embodiment_id = sample["embodiment_id"].reshape(1).to(device=device)
    action = action.unsqueeze(0).to(device=device, dtype=dtype)
    patch_height, patch_width = transformer.config.patch_size[-2:]
    frame_tokens = (
        ref_latents.shape[-2]
        * ref_latents.shape[-1]
        // (int(patch_height) * int(patch_width))
    )
    num_visual_tokens = frame_tokens * (1 + noisy_future.shape[2])
    timestep = torch.zeros(
        1,
        state.shape[1] + action.shape[1] + num_visual_tokens,
        device=device,
        dtype=dtype,
    )
    timestep[:, state.shape[1] + action.shape[1] + frame_tokens :] = timestep_value
    # Call the exact training path while keeping every module in eval mode. The
    # public inference path intentionally defaults future tokens to RGB and does
    # not expose the mask-video modality selector.
    output = transformer._forward_train(
        ref_latents=ref_latents.to(dtype=dtype),
        noisy_latents=noisy_future.to(dtype=dtype),
        timestep=timestep,
        encoder_hidden_states=prompt,
        return_dict=True,
        action=action,
        state=state,
        embodiment_id=embodiment_id,
        output_counterfactual_mask=False,
        output_action=False,
        output_visual=True,
        future_visual_modality="mask",
    )
    return output.sample[:, :, 1:].float()


def main() -> None:
    args = parse_args()
    inputs = (
        args.checkpoint,
        args.training_manifest,
        args.vae,
        args.dataset,
        args.cache,
        args.norm_stats,
    )
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not args.timesteps or any(value <= 0 or value >= 1000 for value in args.timesteps):
        raise ValueError("Every evaluation timestep must be between 1 and 999")
    if len(args.timesteps) != len(set(args.timesteps)):
        raise ValueError("Evaluation timesteps must be unique")
    require_clean_tracked_worktree()

    training = validate_training_manifest(
        args.training_manifest, args.checkpoint, args.cache
    )
    manifest = read_manifest(args.cache)
    episodes = manifest_episode_indices(manifest)
    cached_actions = normalize_cached_actions(args.cache, manifest, args.norm_stats)
    dataset = build_dataset(args)
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    if num_samples > len(episodes):
        raise ValueError("Dataset contains more samples than the cache manifest")
    shuffle_indices = taskwise_shuffle_indices(manifest, num_samples)
    if shuffle_indices is None:
        raise ValueError("Every task needs at least two samples for action sensitivity")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = resolve_dtype(args.model_dtype)
    transformer = CasualWorldActionTransformer_MoT.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    ).to(device)
    transformer.requires_grad_(False).eval()
    if not bool(getattr(transformer.config, "use_mask_video_modality_embedding", False)):
        raise ValueError("Checkpoint does not enable the mask-video modality embedding")
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype).to(device)
    vae.requires_grad_(False).eval()
    latent_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latent_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)

    metrics: dict[str, dict[int, list[float]]] = {
        key: {value: [] for value in args.timesteps}
        for key in ("correct_action_mse", "shuffled_action_mse", "zero_flow_mse")
    }
    per_sample: list[dict[str, Any]] = []
    try:
        for index in range(num_samples):
            sample = dataset[index]
            expected_action = torch.from_numpy(cached_actions[index])
            if not torch.allclose(
                sample["counterfactual_action"], expected_action, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(f"Normalized policy action mismatch at sample {index}")
            ref_latents = encode_reference(
                vae, sample["ref_images"], latent_mean, latent_std
            ).to(dtype=dtype)
            mask_latents = encode_video(
                vae, build_mask_video(sample), latent_mean, latent_std
            )
            clean_future = mask_latents[:, :, 1:]
            valid = latent_valid_mask(
                sample["counterfactual_robot_mask_valid"],
                clean_future.shape[-2],
                clean_future.shape[-1],
            ).to(device=device)
            shuffled_action = torch.from_numpy(cached_actions[shuffle_indices[index]])
            row: dict[str, Any] = {
                "index": index,
                "episode_index": episodes[index],
                "task_index": int(manifest["episodes"][index]["task_index"]),
                "shuffle_index": int(shuffle_indices[index]),
                "timesteps": {},
            }
            for timestep_index, timestep_value in enumerate(args.timesteps):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + index * 10007 + timestep_index * 1000003
                )
                noise = torch.randn(
                    clean_future.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                sigma = float(timestep_value) / 1000.0
                noisy_future = noise * sigma + clean_future * (1.0 - sigma)
                target = noise.float() - clean_future.float()
                correct = predict_flow(
                    transformer,
                    ref_latents,
                    noisy_future,
                    sample["counterfactual_action"],
                    sample,
                    timestep_value,
                    dtype,
                )
                shuffled = predict_flow(
                    transformer,
                    ref_latents,
                    noisy_future,
                    shuffled_action,
                    sample,
                    timestep_value,
                    dtype,
                )
                values = {
                    "correct_action_mse": masked_mse(correct, target, valid),
                    "shuffled_action_mse": masked_mse(shuffled, target, valid),
                    "zero_flow_mse": masked_mse(torch.zeros_like(target), target, valid),
                }
                for key, value in values.items():
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite {key} at sample {index}")
                    metrics[key][timestep_value].append(value)
                row["timesteps"][str(timestep_value)] = values
            if args.save_per_sample:
                per_sample.append(row)
            print(
                json.dumps(
                    {
                        "sample": index + 1,
                        "num_samples": num_samples,
                        "episode_index": episodes[index],
                    }
                ),
                flush=True,
            )
    finally:
        dataset.close()

    aggregate_by_timestep = {
        key: {str(step): aggregate(values) for step, values in by_step.items()}
        for key, by_step in metrics.items()
    }
    flattened = {
        key: [value for step in args.timesteps for value in by_step[step]]
        for key, by_step in metrics.items()
    }
    aggregate_all = {key: aggregate(values) for key, values in flattened.items()}
    correct_mean = float(aggregate_all["correct_action_mse"]["mean"])
    shuffled_mean = float(aggregate_all["shuffled_action_mse"]["mean"])
    zero_mean = float(aggregate_all["zero_flow_mse"]["mean"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
        "training_manifest": str(args.training_manifest.resolve()),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "training_comparison_id": training["comparison_id"],
        "training_variant": training["variant"],
        "counterfactual_mask_weight": training["variant_settings"][
            "counterfactual_mask_weight"
        ],
        "vae": str(args.vae.resolve()),
        "dataset": str(args.dataset.resolve()),
        "cache": str(args.cache.resolve()),
        "cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
        "policy_action_array": manifest["policy_action_array"],
        "simulator_action_array": manifest["simulator_action_array"],
        "norm_stats": str(args.norm_stats.resolve()),
        "num_samples": num_samples,
        "num_views": len(args.view_keys),
        "view_keys": list(args.view_keys),
        "model_dtype": args.model_dtype,
        "timesteps": list(args.timesteps),
        "seed": args.seed,
        "shuffle_semantics": "cyclic derangement within task",
        "aggregate_by_timestep": aggregate_by_timestep,
        "aggregate_all": aggregate_all,
        "correct_action_advantage_over_shuffled_mse": shuffled_mean - correct_mean,
        "correct_action_relative_reduction_vs_zero": 1.0 - correct_mean / zero_mean,
    }
    if args.save_per_sample:
        report["per_sample"] = per_sample
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

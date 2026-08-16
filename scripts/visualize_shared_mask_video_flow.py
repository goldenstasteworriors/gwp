#!/usr/bin/env python3
"""Render standalone-FK targets and shared mask-video flow predictions.

The free-generation rows start from pure Gaussian noise and never read future
target latents.  The diagnostic row reproduces the one-step noisy-target flow
evaluation and is labelled separately so it cannot be mistaken for generation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
from diffusers.models import AutoencoderKLWan  # noqa: E402
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

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
from evaluate_shared_mask_video_flow import (  # noqa: E402
    build_mask_video,
    encode_video,
    git_commit,
    predict_flow,
    require_clean_tracked_worktree,
    validate_training_manifest,
)
from world_action_model.models import CasualWorldActionTransformer_MoT  # noqa: E402


SCHEMA_VERSION = "shared-mask-video-flow-visualization-v1"
HORIZONS = (12, 24, 36, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--view-keys",
        nargs="+",
        default=("observation.images.image", "observation.images.wrist_image"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--free-rollout-steps", type=int, default=20)
    parser.add_argument("--diagnostic-timestep", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def select_task_diverse_indices(
    manifest: dict[str, Any], num_samples: int
) -> list[int]:
    selected: list[int] = []
    seen_tasks: set[int] = set()
    for index, record in enumerate(manifest["episodes"]):
        task_index = int(record["task_index"])
        if task_index in seen_tasks:
            continue
        selected.append(index)
        seen_tasks.add(task_index)
        if len(selected) == num_samples:
            break
    if len(selected) < num_samples:
        for index in range(len(manifest["episodes"])):
            if index not in selected:
                selected.append(index)
            if len(selected) == num_samples:
                break
    if len(selected) != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples but cache has {len(selected)} available"
        )
    return selected


@torch.inference_mode()
def rollout_from_pure_noise(
    transformer: CasualWorldActionTransformer_MoT,
    ref_latents: torch.Tensor,
    initial_noise: torch.Tensor,
    action: torch.Tensor,
    sample: dict[str, torch.Tensor],
    dtype: torch.dtype,
    steps: int,
) -> torch.Tensor:
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    scheduler.set_timesteps(steps, device=ref_latents.device)
    latent = initial_noise.clone()
    for timestep in scheduler.timesteps:
        flow = predict_flow(
            transformer,
            ref_latents,
            latent,
            action,
            sample,
            float(timestep.item()),
            dtype,
        )
        latent = scheduler.step(
            flow, timestep, latent, return_dict=False
        )[0]
    return latent.float()


@torch.inference_mode()
def decode_future_masks(
    vae: AutoencoderKLWan,
    initial_mask_latent: torch.Tensor,
    future_latent: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> np.ndarray:
    normalized = torch.cat((initial_mask_latent, future_latent), dim=2)
    decoded = vae.decode(normalized / latent_std + latent_mean).sample
    if decoded.ndim != 5 or decoded.shape[1] != 3 or decoded.shape[2] != 5:
        raise ValueError(f"Unexpected decoded mask video shape: {tuple(decoded.shape)}")
    probability = (
        ((decoded[0].float().clamp(-1.0, 1.0) + 1.0) * 0.5)
        .mean(dim=0)
        .cpu()
        .numpy()
    )
    return probability[1:]


def valid_agent_crop(valid: np.ndarray) -> tuple[slice, slice]:
    support = np.asarray(valid, dtype=bool).any(axis=0)
    ys, xs = np.nonzero(support)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Standalone mask validity is empty")
    return slice(int(ys.min()), int(ys.max()) + 1), slice(
        int(xs.min()), int(xs.max()) + 1
    )


def binary_iou(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    target = target > 0.5
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def sequence_metrics(
    probability: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float]:
    return {
        "pixel_mse": float(np.mean((probability - target) ** 2)),
        "pixel_mae": float(np.mean(np.abs(probability - target))),
        "mean_iou": float(
            np.mean(
                [
                    binary_iou(probability[index], target[index], threshold)
                    for index in range(target.shape[0])
                ]
            )
        ),
    }


def rgb_context(frame: torch.Tensor, crop: tuple[slice, slice]) -> np.ndarray:
    value = frame.detach().float().cpu().numpy().transpose(1, 2, 0)
    if value.min() < -0.05:
        value = (value + 1.0) * 0.5
    value = np.clip(value, 0.0, 1.0)
    return (value[crop] * 255.0).round().astype(np.uint8)


def grayscale(value: np.ndarray) -> Image.Image:
    pixels = (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def binary_overlay(
    probability: np.ndarray, target: np.ndarray, threshold: float
) -> Image.Image:
    prediction = probability >= threshold
    truth = target > 0.5
    image = np.zeros((*truth.shape, 3), dtype=np.uint8)
    image[np.logical_and(truth, ~prediction)] = (30, 220, 80)
    image[np.logical_and(~truth, prediction)] = (240, 50, 80)
    image[np.logical_and(truth, prediction)] = (245, 245, 245)
    return Image.fromarray(image, mode="RGB")


def difference_heatmap(value: np.ndarray) -> Image.Image:
    strength = np.clip(np.abs(value) * 8.0, 0.0, 1.0)
    image = np.zeros((*strength.shape, 3), dtype=np.uint8)
    image[..., 0] = (strength * 255.0).round().astype(np.uint8)
    image[..., 1] = (strength * 150.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def make_temporal_sheet(
    target: np.ndarray,
    free_correct: np.ndarray,
    free_shuffled: np.ndarray,
    diagnostic: np.ndarray,
    threshold: float,
    title: str,
) -> Image.Image:
    rows: list[tuple[str, list[Image.Image]]] = [
        ("Standalone FK target", [grayscale(value) for value in target]),
        ("Free generation", [grayscale(value) for value in free_correct]),
        (
            "Free binary overlay",
            [
                binary_overlay(free_correct[index], target[index], threshold)
                for index in range(4)
            ],
        ),
        ("Shuffled-action free", [grayscale(value) for value in free_shuffled]),
        (
            "|correct - shuffled| x8",
            [
                difference_heatmap(free_correct[index] - free_shuffled[index])
                for index in range(4)
            ],
        ),
        ("Noisy-target diagnostic", [grayscale(value) for value in diagnostic]),
    ]
    tile_height, tile_width = target.shape[-2:]
    label_width, header_height = 230, 62
    sheet = Image.new(
        "RGB",
        (label_width + 4 * tile_width, header_height + len(rows) * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), title[:150], fill="black")
    draw.text(
        (8, 25),
        "Overlay: white=TP, green=target-only, red=prediction-only",
        fill="black",
    )
    for column, horizon in enumerate(HORIZONS):
        draw.text(
            (label_width + column * tile_width + 8, 43),
            f"horizon {horizon}",
            fill="black",
        )
    for row_index, (label, images) in enumerate(rows):
        y = header_height + row_index * tile_height
        draw.text((8, y + 8), label, fill="black")
        for column, image in enumerate(images):
            sheet.paste(image, (label_width + column * tile_width, y))
    return sheet


def make_gif_frames(
    target: np.ndarray,
    free_correct: np.ndarray,
    free_shuffled: np.ndarray,
    title: str,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_index, horizon in enumerate(HORIZONS):
        tiles = (
            grayscale(target[frame_index]),
            grayscale(free_correct[frame_index]),
            grayscale(free_shuffled[frame_index]),
        )
        height, width = target.shape[-2:]
        canvas = Image.new("RGB", (3 * width, height + 42), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"{title[:90]} | horizon {horizon}", fill="black")
        for column, label in enumerate(("FK target", "free correct", "free shuffled")):
            draw.text((column * width + 6, 23), label, fill="black")
            canvas.paste(tiles[column], (column * width, 42))
        frames.append(canvas)
    return frames


def sha256_outputs(paths: list[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths}


def main() -> None:
    args = parse_args()
    for path in (
        args.checkpoint,
        args.training_manifest,
        args.vae,
        args.dataset,
        args.cache,
        args.norm_stats,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")
    if args.num_samples <= 0 or args.free_rollout_steps <= 0:
        raise ValueError("Sample count and free rollout steps must be positive")
    if not 0 < args.diagnostic_timestep < 1000:
        raise ValueError("Diagnostic timestep must be in (0,1000)")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("Threshold must be in (0,1)")
    require_clean_tracked_worktree()

    training = validate_training_manifest(
        args.training_manifest, args.checkpoint, args.cache
    )
    manifest = read_manifest(args.cache)
    strict_false = (
        "benchmark_environment_imported",
        "benchmark_environment_constructed",
        "benchmark_state_read",
        "controller_used",
        "contact_used",
    )
    if any(manifest.get(key) is not False for key in strict_false):
        raise ValueError("Visualization target is not standalone FK-only")
    episodes = manifest_episode_indices(manifest)
    shuffle_indices = taskwise_shuffle_indices(manifest, len(episodes))
    if shuffle_indices is None:
        raise ValueError("Every task needs at least two samples for shuffled controls")
    selected = select_task_diverse_indices(manifest, args.num_samples)
    cached_actions = normalize_cached_actions(args.cache, manifest, args.norm_stats)
    dataset = build_dataset(args)

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
        raise ValueError("Checkpoint does not enable shared mask-video generation")
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype).to(device)
    vae.requires_grad_(False).eval()
    latent_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latent_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)

    args.output_dir.mkdir(parents=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir()
    output_paths: list[Path] = []
    sample_reports: list[dict[str, Any]] = []
    overview_rows: list[tuple[str, list[Image.Image]]] = []
    try:
        for order, index in enumerate(selected):
            sample = dataset[index]
            expected_action = torch.from_numpy(cached_actions[index])
            if not torch.allclose(
                sample["counterfactual_action"], expected_action, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(f"Normalized action mismatch at sample {index}")
            shuffle_index = int(shuffle_indices[index])
            shuffled_action = torch.from_numpy(cached_actions[shuffle_index])
            ref_latents = encode_reference(
                vae, sample["ref_images"], latent_mean, latent_std
            ).to(dtype=dtype)
            mask_latents = encode_video(
                vae, build_mask_video(sample), latent_mean, latent_std
            )
            clean_future = mask_latents[:, :, 1:]
            initial_mask_latent = mask_latents[:, :, :1]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + index * 10007
            )
            noise = torch.randn(
                clean_future.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )

            free_correct_latent = rollout_from_pure_noise(
                transformer,
                ref_latents,
                noise,
                sample["counterfactual_action"],
                sample,
                dtype,
                args.free_rollout_steps,
            )
            free_shuffled_latent = rollout_from_pure_noise(
                transformer,
                ref_latents,
                noise,
                shuffled_action,
                sample,
                dtype,
                args.free_rollout_steps,
            )

            sigma = args.diagnostic_timestep / 1000.0
            noisy_target = noise * sigma + clean_future * (1.0 - sigma)
            diagnostic_flow = predict_flow(
                transformer,
                ref_latents,
                noisy_target,
                sample["counterfactual_action"],
                sample,
                args.diagnostic_timestep,
                dtype,
            )
            diagnostic_latent = noisy_target.float() - sigma * diagnostic_flow

            free_correct = decode_future_masks(
                vae,
                initial_mask_latent,
                free_correct_latent,
                latent_mean,
                latent_std,
            )
            free_shuffled = decode_future_masks(
                vae,
                initial_mask_latent,
                free_shuffled_latent,
                latent_mean,
                latent_std,
            )
            diagnostic = decode_future_masks(
                vae,
                initial_mask_latent,
                diagnostic_latent,
                latent_mean,
                latent_std,
            )
            target_full = sample["counterfactual_robot_masks"].cpu().numpy()
            valid_full = sample["counterfactual_robot_mask_valid"].cpu().numpy()
            crop = valid_agent_crop(valid_full)
            target = target_full[(slice(None),) + crop]
            free_correct = free_correct[(slice(None),) + crop]
            free_shuffled = free_shuffled[(slice(None),) + crop]
            diagnostic = diagnostic[(slice(None),) + crop]
            initial_target = sample["counterfactual_initial_robot_mask"].cpu().numpy()[crop]
            context = rgb_context(sample["ref_images"][0], crop)

            record = manifest["episodes"][index]
            title = (
                f"episode {episodes[index]} task {record['task_index']}: "
                f"{record.get('task_language', '')}"
            )
            temporal_path = samples_dir / f"episode_{episodes[index]:06d}_temporal.png"
            make_temporal_sheet(
                target,
                free_correct,
                free_shuffled,
                diagnostic,
                args.threshold,
                title,
            ).save(temporal_path)
            output_paths.append(temporal_path)
            gif_path = samples_dir / f"episode_{episodes[index]:06d}_free_generation.gif"
            gif_frames = make_gif_frames(target, free_correct, free_shuffled, title)
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=700,
                loop=0,
            )
            output_paths.append(gif_path)

            metrics = {
                "free_correct": sequence_metrics(
                    free_correct, target, args.threshold
                ),
                "free_shuffled": sequence_metrics(
                    free_shuffled, target, args.threshold
                ),
                "noisy_target_diagnostic": sequence_metrics(
                    diagnostic, target, args.threshold
                ),
                "free_correct_vs_shuffled_probability_mae": float(
                    np.mean(np.abs(free_correct - free_shuffled))
                ),
            }
            sample_reports.append(
                {
                    "order": order,
                    "dataset_index": index,
                    "episode_index": episodes[index],
                    "task_index": int(record["task_index"]),
                    "task_language": record.get("task_language"),
                    "shuffle_dataset_index": shuffle_index,
                    "shuffle_episode_index": episodes[shuffle_index],
                    "maximum_supported_alpha": record.get("maximum_supported_alpha"),
                    "metrics": metrics,
                    "temporal_png": str(temporal_path.relative_to(args.output_dir)),
                    "free_generation_gif": str(gif_path.relative_to(args.output_dir)),
                }
            )
            overview_rows.append(
                (
                    f"ep {episodes[index]} / task {record['task_index']}",
                    [
                        Image.fromarray(context, mode="RGB"),
                        grayscale(initial_target),
                        grayscale(target[-1]),
                        grayscale(free_correct[-1]),
                        binary_overlay(free_correct[-1], target[-1], args.threshold),
                        grayscale(free_shuffled[-1]),
                        difference_heatmap(free_correct[-1] - free_shuffled[-1]),
                    ],
                )
            )
            print(
                json.dumps(
                    {
                        "completed": order + 1,
                        "num_samples": len(selected),
                        "episode_index": episodes[index],
                        "metrics": metrics,
                    }
                ),
                flush=True,
            )
    finally:
        dataset.close()

    tile_size, label_width, header_height = 192, 150, 54
    column_labels = (
        "RGB context",
        "initial FK",
        "h48 FK target",
        "h48 free",
        "h48 overlay",
        "h48 shuffled",
        "|difference| x8",
    )
    overview = Image.new(
        "RGB",
        (
            label_width + len(column_labels) * tile_size,
            header_height + len(overview_rows) * tile_size,
        ),
        "white",
    )
    draw = ImageDraw.Draw(overview)
    for column, label in enumerate(column_labels):
        draw.text((label_width + column * tile_size + 4, 20), label, fill="black")
    for row, (label, images) in enumerate(overview_rows):
        y = header_height + row * tile_size
        draw.text((5, y + 8), label, fill="black")
        for column, image in enumerate(images):
            overview.paste(
                image.resize((tile_size, tile_size), Image.Resampling.NEAREST),
                (label_width + column * tile_size, y),
            )
    overview_path = args.output_dir / "overview.png"
    overview.save(overview_path)
    output_paths.append(overview_path)

    manifest_path = args.output_dir / "manifest.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
        "training_manifest": str(args.training_manifest.resolve()),
        "training_comparison_id": training["comparison_id"],
        "dataset": str(args.dataset.resolve()),
        "target_cache": str(args.cache.resolve()),
        "target_cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
        "target_schema_version": manifest["schema_version"],
        "target_definition": (
            "specified arm q + gripper q + explicit camera intrinsics/extrinsics -> "
            "standalone robot-only mj_kinematics FK -> CPU mesh rasterization"
        ),
        "simulator_interaction": False,
        "target_provenance": {
            key: manifest.get(key)
            for key in (
                "benchmark_environment_imported",
                "benchmark_environment_constructed",
                "benchmark_state_read",
                "controller_used",
                "contact_used",
                "fk_api",
                "dynamics",
                "dynamics_attempted_calls",
                "legacy_rollout_masks_read",
                "legacy_rollout_qpos_read",
                "simulator_actions_read",
            )
        },
        "free_generation": {
            "initialization": "pure Gaussian noise",
            "future_target_latent_used": False,
            "flow_scheduler": "FlowMatchEulerDiscreteScheduler(shift=5.0)",
            "steps": args.free_rollout_steps,
        },
        "noisy_target_diagnostic": {
            "is_free_generation": False,
            "timestep": args.diagnostic_timestep,
            "sigma": args.diagnostic_timestep / 1000.0,
            "future_target_latent_fraction_in_input": 1.0
            - args.diagnostic_timestep / 1000.0,
        },
        "seed": args.seed,
        "threshold": args.threshold,
        "sample_selection": "first episode from each distinct task, then cache order",
        "samples": sample_reports,
        "output_sha256": sha256_outputs(output_paths),
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a counterfactual robot-mask head and action-conditioning controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT,
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

from world_action_model.alignment import (  # noqa: E402
    copy_initial_mask_baseline,
    counterfactual_mask_metrics,
    per_horizon_mean_mask_baseline,
)
from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset  # noqa: E402
from world_action_model.models import CasualWorldActionTransformer_MoT  # noqa: E402
from world_action_model.transforms.wa_transforms_lerobot_pretrain import (  # noqa: E402
    WALeRobotTransformsPretrain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mean-mask-cache", type=Path)
    parser.add_argument(
        "--view-keys",
        nargs="+",
        default=("observation.images.image", "observation.images.wrist_image"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-per-sample", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def read_manifest(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "libero-counterfactual-robot-mask-v1":
        raise ValueError(f"Unsupported cache schema: {manifest.get('schema_version')!r}")
    if [int(value) for value in manifest.get("horizons", [])] != [0, 12, 24, 36, 48]:
        raise ValueError(f"Unexpected cache horizons: {manifest.get('horizons')}")
    return manifest


def normalize_cached_actions(
    cache_root: Path,
    manifest: dict[str, Any],
    norm_stats_path: Path,
    *,
    model_action_dim: int = 16,
) -> np.ndarray:
    stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))["norm_stats"]["action"]
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    actions = []
    for record in manifest["episodes"]:
        with np.load(cache_root / record["cache_file"], allow_pickle=False) as cache:
            action = cache["executed_actions"].astype(np.float32)
        if action.ndim != 2 or action.shape[0] != 48:
            raise ValueError(f"Expected [48,D] cached action, got {action.shape}")
        effective_dim = min(action.shape[-1], low.shape[0], high.shape[0])
        normalized = np.zeros((48, model_action_dim), dtype=np.float32)
        action_range = np.maximum(high[:effective_dim] - low[:effective_dim], 1e-8)
        normalized[:, :effective_dim] = (
            2.0 * (action[:, :effective_dim] - low[:effective_dim]) / action_range - 1.0
        )
        actions.append(np.clip(normalized, -1.0, 1.0))
    return np.stack(actions)


def load_mean_mask_prior(cache_root: Path, num_views: int) -> np.ndarray:
    manifest = read_manifest(cache_root)
    future_masks = []
    for record in manifest["episodes"]:
        with np.load(cache_root / record["cache_file"], allow_pickle=False) as cache:
            future_masks.append(cache["robot_mask"][1:].copy())
    prior = per_horizon_mean_mask_baseline(np.stack(future_masks))
    if num_views > 1:
        prior = np.pad(prior, ((0, 0), (0, 0), (0, prior.shape[-1] * (num_views - 1))))
    return prior


def build_dataset(args: argparse.Namespace) -> WAMLeRobotDataset:
    horizons = [0, 12, 24, 36, 48]
    view_keys = list(args.view_keys)
    dataset = WAMLeRobotDataset(
        data_path=str(args.dataset),
        delta_info={"action": 48},
        delta_frames={key: horizons for key in view_keys},
        t5_load_from="dir",
        t5_cfg={
            "t5_embedding_dir": str(args.dataset / "t5_embedding"),
            "t5_embedding_pattern": "episode_{episode_index:06d}.pt",
            "t5_embedding_key": "t5_embedding",
        },
        t5_cache_size=256,
        robotype="franka",
        counterfactual_mask_cache=str(args.cache),
        video_backend="pyav",
    )
    transform = WALeRobotTransformsPretrain(
        is_train=False,
        dst_size=(256, 256),
        num_frames=48,
        fps=20,
        norm_path=[{"path": str(args.norm_stats), "data_paths": [str(args.dataset)]}],
        robotype_to_embodiment_id={"franka": 0},
        robotype_default_embodiment_id=0,
        model_action_dim=16,
        delta_mask_by_embodiment_id={"0": [False] * 7},
        norm_use_quantiles=True,
        norm_enable_clamp=True,
        num_views=len(view_keys),
        view_keys=view_keys,
        view_concat_mode="horizontal",
        image_cfg={"mask_generator": {"max_ref_frames": 1, "start": 1, "factor": 4}},
        max_prompt_len=64,
        subtask_prob=0,
    )
    dataset.set_transform(transform)
    return dataset


def encode_reference(
    vae: AutoencoderKLWan,
    ref_images: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    images = ref_images[:1].unsqueeze(0).permute(0, 2, 1, 3, 4)
    images = images.to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(images).latent_dist.mode()
    return (latents - latent_mean) * latent_std


@torch.inference_mode()
def predict_mask_logits(
    transformer: CasualWorldActionTransformer_MoT,
    ref_latents: torch.Tensor,
    action: torch.Tensor,
    sample: dict[str, torch.Tensor],
    future_noise: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    state = sample["state"].unsqueeze(0).to(device=ref_latents.device, dtype=dtype)
    prompt = sample["prompt_embeds"].unsqueeze(0).to(device=ref_latents.device, dtype=dtype)
    embodiment_id = sample["embodiment_id"].reshape(1).to(device=ref_latents.device)
    action = action.unsqueeze(0).to(device=ref_latents.device, dtype=dtype)

    num_state_tokens = state.shape[1]
    num_action_tokens = action.shape[1]
    patch_height, patch_width = transformer.config.patch_size[-2:]
    frame_tokens = (
        ref_latents.shape[-2]
        * ref_latents.shape[-1]
        // (int(patch_height) * int(patch_width))
    )
    num_latent_tokens = frame_tokens * (1 + future_noise.shape[2])
    timestep = torch.zeros(
        1,
        num_state_tokens + num_action_tokens + num_latent_tokens,
        device=ref_latents.device,
        dtype=dtype,
    )
    timestep[:, num_state_tokens + num_action_tokens + frame_tokens :] = 1000
    output = transformer(
        ref_latents=ref_latents.to(dtype=dtype),
        noisy_latents=future_noise,
        timestep=timestep,
        encoder_hidden_states=prompt,
        return_dict=True,
        action=action,
        state=state,
        embodiment_id=embodiment_id,
        output_counterfactual_mask=True,
        output_action=False,
        output_visual=False,
    )
    return output["counterfactual_mask_logits"].float()


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[None, None]
        return F.interpolate(mask.float(), size=size, mode="nearest")[0, 0].cpu().numpy()
    if mask.ndim == 3:
        mask = mask[None]
        return F.interpolate(mask.float(), size=size, mode="nearest")[0].cpu().numpy()
    raise ValueError(f"Expected [H,W] or [T,H,W] mask, got {tuple(mask.shape)}")


def evaluate_prediction(
    probability: np.ndarray,
    sample: dict[str, torch.Tensor],
    size: tuple[int, int],
    threshold: float,
) -> dict[str, Any]:
    target = resize_mask(sample["counterfactual_robot_masks"], size)
    initial = resize_mask(sample["counterfactual_initial_robot_mask"], size)
    valid = resize_mask(sample["counterfactual_robot_mask_valid"], size)
    gripper = resize_mask(sample["counterfactual_gripper_masks"], size)
    return counterfactual_mask_metrics(
        probability,
        target,
        initial,
        valid_masks=valid,
        target_gripper_masks=gripper,
        horizons=sample["counterfactual_horizons"].tolist(),
        threshold=threshold,
    )


def evaluate_binary_baseline(
    prediction: np.ndarray,
    sample: dict[str, torch.Tensor],
    size: tuple[int, int],
) -> dict[str, Any]:
    prediction_tensor = torch.from_numpy(prediction.astype(np.float32))
    prediction = resize_mask(prediction_tensor, size)
    return evaluate_prediction(prediction, sample, size, threshold=0.5)


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    horizons = results[0]["horizons"]
    per_horizon = []
    for horizon_index, horizon in enumerate(horizons):
        row: dict[str, Any] = {"horizon": int(horizon)}
        for region in ("full", "dynamic", "new_occupancy"):
            row[region] = {}
            for metric in ("iou", "dice", "precision", "recall"):
                row[region][metric] = finite_mean(
                    [result["per_horizon"][horizon_index][region][metric] for result in results]
                )
            row[region]["empty_union_fraction"] = float(
                np.mean(
                    [
                        result["per_horizon"][horizon_index][region]["empty_union"]
                        for result in results
                    ]
                )
            )
        row["gripper_region_recall"] = finite_mean(
            [result["per_horizon"][horizon_index]["gripper_region_recall"] for result in results]
        )
        per_horizon.append(row)

    mean = {}
    for region in ("full", "dynamic", "new_occupancy"):
        for metric in ("iou", "dice", "precision", "recall"):
            mean[f"{region}_{metric}"] = finite_mean(
                [result["mean"][f"{region}_{metric}"] for result in results]
            )
        mean[f"{region}_empty_union_fraction"] = float(
            np.mean(
                [result["mean"][f"{region}_empty_union_fraction"] for result in results]
            )
        )
    mean["gripper_region_recall"] = finite_mean(
        [result["mean"]["gripper_region_recall"] for result in results]
    )
    return {"num_samples": len(results), "per_horizon": per_horizon, "mean": mean}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.vae, args.dataset, args.cache, args.norm_stats):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = resolve_dtype(args.model_dtype)

    manifest = read_manifest(args.cache)
    cached_actions = normalize_cached_actions(args.cache, manifest, args.norm_stats)
    dataset = build_dataset(args)
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    if cached_actions.shape[0] < num_samples:
        raise ValueError("Counterfactual cache contains fewer actions than dataset samples")

    transformer = CasualWorldActionTransformer_MoT.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    ).to(device)
    transformer.eval()
    if int(transformer.config.counterfactual_mask_frames_per_latent) <= 0:
        raise ValueError("Checkpoint does not contain an enabled counterfactual mask head")
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype).to(device)
    vae.eval()
    latent_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latent_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)

    mean_prior = None
    if args.mean_mask_cache is not None:
        mean_prior = load_mean_mask_prior(args.mean_mask_cache, len(args.view_keys))

    modes: dict[str, list[dict[str, Any]]] = {
        "correct_action": [],
        "copy_initial": [],
    }
    if num_samples > 1:
        modes["shuffled_action"] = []
    if mean_prior is not None:
        modes["per_horizon_mean_mask"] = []
    per_sample = []

    try:
        for index in range(num_samples):
            sample = dataset[index]
            expected_action = torch.from_numpy(cached_actions[index])
            if not torch.allclose(
                sample["counterfactual_action"], expected_action, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(f"Normalized cached action mismatch at sample {index}")

            ref_latents = encode_reference(
                vae, sample["ref_images"], latent_mean, latent_std
            ).to(dtype=dtype)
            generator = torch.Generator(device=device).manual_seed(args.seed + index * 10007)
            future_noise = torch.randn(
                1,
                vae.config.z_dim,
                1,
                ref_latents.shape[-2],
                ref_latents.shape[-1],
                device=device,
                dtype=dtype,
                generator=generator,
            )
            logits = predict_mask_logits(
                transformer,
                ref_latents,
                sample["counterfactual_action"],
                sample,
                future_noise,
                dtype,
            )
            size = tuple(logits.shape[-2:])
            correct = evaluate_prediction(
                logits.sigmoid()[0].cpu().numpy(), sample, size, args.threshold
            )
            modes["correct_action"].append(correct)

            initial = sample["counterfactual_initial_robot_mask"].cpu().numpy()
            copy_prediction = copy_initial_mask_baseline(initial, 4)
            modes["copy_initial"].append(
                evaluate_binary_baseline(copy_prediction, sample, size)
            )
            sample_result: dict[str, Any] = {"index": index, "correct_action": correct}

            if num_samples > 1:
                # A half-dataset cyclic shift is a deterministic derangement and
                # is less likely than an adjacent sample to retain the same task.
                shuffle_index = (index + max(1, num_samples // 2)) % num_samples
                shuffled_action = torch.from_numpy(cached_actions[shuffle_index])
                shuffled_logits = predict_mask_logits(
                    transformer,
                    ref_latents,
                    shuffled_action,
                    sample,
                    future_noise,
                    dtype,
                )
                shuffled = evaluate_prediction(
                    shuffled_logits.sigmoid()[0].cpu().numpy(),
                    sample,
                    size,
                    args.threshold,
                )
                modes["shuffled_action"].append(shuffled)
                sample_result["shuffle_index"] = shuffle_index
                sample_result["shuffled_action"] = shuffled

            if mean_prior is not None:
                mean_result = evaluate_binary_baseline(mean_prior, sample, size)
                modes["per_horizon_mean_mask"].append(mean_result)
                sample_result["per_horizon_mean_mask"] = mean_result
            if args.save_per_sample:
                per_sample.append(sample_result)
    finally:
        dataset.close()

    aggregate = {mode: aggregate_metrics(results) for mode, results in modes.items()}
    sensitivity = None
    if "shuffled_action" in aggregate:
        sensitivity = {
            key: aggregate["correct_action"]["mean"][key]
            - aggregate["shuffled_action"]["mean"][key]
            for key in (
                "full_iou",
                "dynamic_iou",
                "new_occupancy_iou",
                "gripper_region_recall",
            )
        }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "vae": str(args.vae.resolve()),
        "dataset": str(args.dataset.resolve()),
        "cache": str(args.cache.resolve()),
        "mean_mask_cache": (
            str(args.mean_mask_cache.resolve()) if args.mean_mask_cache is not None else None
        ),
        "seed": args.seed,
        "threshold": args.threshold,
        "mask_resolution": list(size),
        "aggregate": aggregate,
        "correct_minus_shuffled": sensitivity,
    }
    if args.save_per_sample:
        report["per_sample"] = per_sample
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report), indent=2, allow_nan=False), flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

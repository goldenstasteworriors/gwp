#!/usr/bin/env python3
"""Evaluate a counterfactual robot-mask head and action-conditioning controls."""

from __future__ import annotations

import argparse
import hashlib
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
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402

from world_action_model.alignment import (  # noqa: E402
    LatentRobotMaskProbe,
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
    parser.add_argument(
        "--alignment-mode",
        choices=("hidden_head", "clean_latent_probe"),
        default="hidden_head",
    )
    parser.add_argument("--mask-probe", type=Path)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "libero-counterfactual-robot-mask-v5",
        "libero-policy-action-standalone-mask-v2",
        "libero-support-constrained-action-standalone-mask-v2",
    }:
        raise ValueError(f"Unsupported cache schema: {manifest.get('schema_version')!r}")
    if manifest.get("schema_version") in {
        "libero-policy-action-standalone-mask-v2",
        "libero-support-constrained-action-standalone-mask-v2",
    }:
        calls = manifest.get("dynamics_attempted_calls", {})
        if manifest.get("dynamics") != "forbidden" or set(calls) != {
            "mj_step",
            "mj_step1",
            "mj_step2",
            "mj_forward",
            "mj_forwardSkip",
        } or any(int(value) != 0 for value in calls.values()):
            raise ValueError("Kinematic counterfactual cache attempted dynamics")
        if any(
            manifest.get(key) is not False
            for key in (
                "legacy_rollout_masks_read",
                "legacy_rollout_qpos_read",
                "simulator_actions_read",
            )
        ):
            raise ValueError("Kinematic counterfactual cache read a legacy oracle field")
        if manifest.get("fk_api") != "mujoco.mj_kinematics" or any(
            manifest.get(key) is not False
            for key in (
                "benchmark_environment_imported",
                "benchmark_environment_constructed",
                "benchmark_state_read",
                "controller_used",
                "contact_used",
            )
        ):
            raise ValueError("Counterfactual cache is not standalone FK-only")
    if (
        manifest.get("image_orientation")
        != "lerobot_libero_visualization_flip_height_width"
    ):
        raise ValueError("Counterfactual masks are not aligned to the training-video orientation")
    if [int(value) for value in manifest.get("horizons", [])] != [0, 12, 24, 36, 48]:
        raise ValueError(f"Unexpected cache horizons: {manifest.get('horizons')}")
    return manifest


def manifest_episode_indices(manifest: dict[str, Any]) -> list[int]:
    episodes = [int(record["episode_index"]) for record in manifest.get("episodes", [])]
    if not episodes:
        raise ValueError("Counterfactual cache manifest contains no episode records")
    if len(episodes) != len(set(episodes)):
        raise ValueError("Counterfactual cache manifest contains duplicate episode records")
    return episodes


def validate_mean_mask_cache(
    evaluation_manifest: dict[str, Any],
    mean_mask_manifest: dict[str, Any],
) -> None:
    if (
        evaluation_manifest.get("dataset_info_sha256")
        != mean_mask_manifest.get("dataset_info_sha256")
    ):
        raise ValueError("Mean-mask and evaluation caches refer to different dataset metadata")
    if evaluation_manifest.get("suite") != mean_mask_manifest.get("suite"):
        raise ValueError("Mean-mask and evaluation caches refer to different LIBERO suites")
    if evaluation_manifest.get("policy_id") != mean_mask_manifest.get("policy_id"):
        raise ValueError("Mean-mask and evaluation caches were generated by different policies")
    for key in (
        "horizons",
        "camera_name",
        "image_orientation",
        "libero_config_sha256",
        "raw_demo_matching",
        "raw_demo_files",
        "action_match_atol",
        "source_state_atol",
        "policy_action_array",
        "simulator_action_array",
        "image_size",
        "num_steps_wait",
    ):
        if evaluation_manifest.get(key) != mean_mask_manifest.get(key):
            raise ValueError(f"Mean-mask and evaluation cache metadata differ for {key}")
    evaluation_episodes = set(manifest_episode_indices(evaluation_manifest))
    mean_mask_episodes = set(manifest_episode_indices(mean_mask_manifest))
    overlap = sorted(evaluation_episodes & mean_mask_episodes)
    if overlap:
        raise ValueError(
            "Mean-mask cache leaks evaluation episodes: "
            f"{overlap[:20]}{'...' if len(overlap) > 20 else ''}"
        )


def taskwise_shuffle_indices(
    manifest: dict[str, Any],
    num_samples: int,
) -> list[int] | None:
    """Derange actions within each task so prompt semantics stay fixed."""

    records = list(manifest.get("episodes", []))[:num_samples]
    if len(records) != num_samples:
        raise ValueError(
            f"Manifest has {len(records)} records but evaluation requested {num_samples}"
        )
    by_task: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        by_task.setdefault(int(record["task_index"]), []).append(index)
    if any(len(indices) < 2 for indices in by_task.values()):
        return None
    shuffled = [-1] * num_samples
    for indices in by_task.values():
        for source, target in zip(indices, indices[1:] + indices[:1]):
            shuffled[source] = target
    if any(index < 0 for index in shuffled):
        raise AssertionError("Taskwise shuffle left an unassigned sample")
    if any(index == shuffled[index] for index in range(num_samples)):
        raise AssertionError("Taskwise shuffle contains a fixed point")
    return shuffled


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
            action = cache["policy_actions_lerobot"].astype(np.float32)
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
    alignment_mode: str,
    mask_probe: LatentRobotMaskProbe | None,
    flow_rollout_steps: int = 1,
    teacher_forced_clean_latents: torch.Tensor | None = None,
    teacher_forced_sigma: float | None = None,
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
    common = {
        "ref_latents": ref_latents.to(dtype=dtype),
        "noisy_latents": future_noise,
        "timestep": timestep,
        "encoder_hidden_states": prompt,
        "return_dict": True,
        "action": action,
        "state": state,
        "embodiment_id": embodiment_id,
    }
    if alignment_mode == "hidden_head":
        output = transformer(
            **common,
            output_counterfactual_mask=True,
        )
        return output["counterfactual_mask_logits"].float()
    if alignment_mode != "clean_latent_probe":
        raise ValueError(f"Unsupported alignment mode: {alignment_mode!r}")
    if mask_probe is None:
        raise ValueError("clean_latent_probe evaluation requires a frozen mask probe")
    if flow_rollout_steps <= 0:
        raise ValueError("flow_rollout_steps must be positive")
    if teacher_forced_clean_latents is not None:
        if teacher_forced_sigma is None or not 0.0 < teacher_forced_sigma < 1.0:
            raise ValueError("Teacher-forced clean latents require sigma in (0,1)")
        if flow_rollout_steps != 1:
            raise ValueError("Teacher-forced diagnostic currently requires one flow step")
        if teacher_forced_clean_latents.shape != future_noise.shape:
            raise ValueError("Teacher-forced clean/future-noise shapes differ")
        predicted_clean = (
            future_noise.float() * teacher_forced_sigma
            + teacher_forced_clean_latents.float() * (1.0 - teacher_forced_sigma)
        )
        timestep.zero_()
        timestep[:, num_state_tokens + num_action_tokens + frame_tokens :] = (
            round(teacher_forced_sigma * 1000.0)
        )
        common["noisy_latents"] = predicted_clean.to(dtype=dtype)
        common["timestep"] = timestep
        output = transformer(
            **common,
            output_counterfactual_mask=False,
        )
        video_output = output["sample"] if isinstance(output, dict) else output.sample
        predicted_flow = video_output[:, :, 1:].float()
        if predicted_flow.shape != future_noise.shape:
            raise ValueError(
                "Predicted flow/future-noise shape mismatch: "
                f"{tuple(predicted_flow.shape)} vs {tuple(future_noise.shape)}"
            )
        predicted_clean = predicted_clean - teacher_forced_sigma * predicted_flow
    else:
        if teacher_forced_sigma is not None:
            raise ValueError("teacher_forced_sigma requires clean latents")
        scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
        scheduler.set_timesteps(flow_rollout_steps, device=ref_latents.device)
        predicted_clean = future_noise.float()
        for scheduler_timestep in scheduler.timesteps:
            timestep.zero_()
            timestep[:, num_state_tokens + num_action_tokens + frame_tokens :] = (
                scheduler_timestep.to(dtype=timestep.dtype)
            )
            common["noisy_latents"] = predicted_clean.to(dtype=dtype)
            common["timestep"] = timestep
            output = transformer(
                **common,
                output_counterfactual_mask=False,
            )
            video_output = output["sample"] if isinstance(output, dict) else output.sample
            predicted_flow = video_output[:, :, 1:].float()
            if predicted_flow.shape != future_noise.shape:
                raise ValueError(
                    "Predicted flow/future-noise shape mismatch: "
                    f"{tuple(predicted_flow.shape)} vs {tuple(future_noise.shape)}"
                )
            predicted_clean = scheduler.step(
                predicted_flow,
                scheduler_timestep,
                predicted_clean,
                return_dict=False,
            )[0]
    return mask_probe(predicted_clean).float()


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
    if args.alignment_mode == "clean_latent_probe":
        if args.mask_probe is None or not args.mask_probe.is_dir():
            raise FileNotFoundError(
                "--mask-probe is required for clean_latent_probe evaluation"
            )
    elif args.mask_probe is not None:
        raise ValueError("--mask-probe is only valid with clean_latent_probe")

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
    shuffle_indices = taskwise_shuffle_indices(manifest, num_samples)

    transformer = CasualWorldActionTransformer_MoT.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    ).to(device)
    transformer.eval()
    if (
        args.alignment_mode == "hidden_head"
        and int(transformer.config.counterfactual_mask_frames_per_latent) <= 0
    ):
        raise ValueError("Checkpoint does not contain an enabled counterfactual mask head")
    mask_probe = None
    if args.alignment_mode == "clean_latent_probe":
        mask_probe = LatentRobotMaskProbe.from_pretrained(args.mask_probe).to(
            device=device, dtype=torch.float32
        )
        mask_probe.requires_grad_(False).eval()
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype).to(device)
    vae.eval()
    if mask_probe is not None:
        if mask_probe.latent_channels != int(vae.config.z_dim):
            raise ValueError("Mask probe/VAE latent channel mismatch")
        if mask_probe.frames_per_latent != int(vae.config.scale_factor_temporal):
            raise ValueError("Mask probe/VAE temporal scale mismatch")
        if mask_probe.spatial_scale != int(vae.config.scale_factor_spatial):
            raise ValueError("Mask probe/VAE spatial scale mismatch")
    latent_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latent_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)

    mean_prior = None
    mean_mask_manifest = None
    if args.mean_mask_cache is not None:
        mean_mask_manifest = read_manifest(args.mean_mask_cache)
        validate_mean_mask_cache(manifest, mean_mask_manifest)
        mean_prior = load_mean_mask_prior(args.mean_mask_cache, len(args.view_keys))

    modes: dict[str, list[dict[str, Any]]] = {
        "correct_action": [],
        "copy_initial": [],
    }
    if shuffle_indices is not None:
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
                args.alignment_mode,
                mask_probe,
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

            if shuffle_indices is not None:
                shuffle_index = shuffle_indices[index]
                shuffled_action = torch.from_numpy(cached_actions[shuffle_index])
                shuffled_logits = predict_mask_logits(
                    transformer,
                    ref_latents,
                    shuffled_action,
                    sample,
                    future_noise,
                    dtype,
                    args.alignment_mode,
                    mask_probe,
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
        "alignment_mode": args.alignment_mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "vae": str(args.vae.resolve()),
        "dataset": str(args.dataset.resolve()),
        "cache": str(args.cache.resolve()),
        "mean_mask_cache": (
            str(args.mean_mask_cache.resolve()) if args.mean_mask_cache is not None else None
        ),
        "evaluation_episode_count": len(manifest_episode_indices(manifest)),
        "mean_mask_episode_count": (
            len(manifest_episode_indices(mean_mask_manifest))
            if mean_mask_manifest is not None
            else None
        ),
        "shuffle_semantics": (
            "cyclic derangement within task" if shuffle_indices is not None else None
        ),
        "seed": args.seed,
        "threshold": args.threshold,
        "mask_resolution": list(size),
        "mask_probe": (
            str(args.mask_probe.resolve()) if args.mask_probe is not None else None
        ),
        "mask_probe_config_sha256": (
            sha256_file(args.mask_probe / LatentRobotMaskProbe.CONFIG_NAME)
            if args.mask_probe is not None
            else None
        ),
        "mask_probe_weights_sha256": (
            sha256_file(args.mask_probe / LatentRobotMaskProbe.WEIGHTS_NAME)
            if args.mask_probe is not None
            else None
        ),
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

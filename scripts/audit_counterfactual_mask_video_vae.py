#!/usr/bin/env python3
"""Audit whether the frozen Wan VAE preserves counterfactual robot masks.

This is a representation gate for the proposed shared video-output alignment:
robot-mask videos would use the same VAE latent space and RGB ``proj_out`` as
the original video-flow objective.  Training that objective is not allowed
until an encode/decode round trip preserves control-relevant mask geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_action_model.alignment import binary_mask_metrics, counterfactual_mask_metrics


SCHEMA_VERSION = "counterfactual-mask-video-vae-audit-v1"
CACHE_SCHEMA = "libero-counterfactual-robot-mask-v3"
EXPECTED_HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Wan-VAE reconstruction of cached counterfactual mask videos"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-initial-iou-mean", type=float, default=0.90)
    parser.add_argument("--min-full-iou-mean", type=float, default=0.90)
    parser.add_argument("--min-full-iou-p05", type=float, default=0.80)
    parser.add_argument("--min-dynamic-iou-mean", type=float, default=0.80)
    parser.add_argument("--min-dynamic-iou-p05", type=float, default=0.60)
    parser.add_argument("--min-new-iou-mean", type=float, default=0.80)
    parser.add_argument("--min-new-iou-p05", type=float, default=0.60)
    parser.add_argument("--min-gripper-recall-mean", type=float, default=0.90)
    parser.add_argument("--min-gripper-recall-p05", type=float, default=0.75)
    parser.add_argument("--max-pixel-mae-mean", type=float, default=0.05)
    parser.add_argument("--max-pixel-mae-p95", type=float, default=0.10)
    return parser.parse_args()


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
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"Unsupported cache schema: {manifest.get('schema_version')!r}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError("Counterfactual cache unexpectedly contains RGB supervision")
    if (
        manifest.get("image_orientation")
        != "lerobot_libero_visualization_flip_height_width"
    ):
        raise ValueError("Counterfactual masks do not use the training-video orientation")
    if [int(value) for value in manifest.get("horizons", [])] != EXPECTED_HORIZONS:
        raise ValueError(f"Unexpected cache horizons: {manifest.get('horizons')}")
    records = manifest.get("episodes")
    if not isinstance(records, list) or not records:
        raise ValueError("Counterfactual cache contains no episode records")
    return manifest


def pad_views(mask: np.ndarray, num_views: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected [T,H,W] mask sequence, got {mask.shape}")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if num_views == 1:
        return mask
    return np.pad(mask, ((0, 0), (0, 0), (0, mask.shape[-1] * (num_views - 1))))


def evaluate_reconstruction(
    probabilities: np.ndarray,
    robot_masks: np.ndarray,
    gripper_masks: np.ndarray,
    *,
    num_views: int,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one decoded five-frame mask video against its binary source."""

    robot_masks = np.asarray(robot_masks)
    gripper_masks = np.asarray(gripper_masks)
    if robot_masks.ndim != 3 or robot_masks.shape[0] != len(EXPECTED_HORIZONS):
        raise ValueError(f"Expected five robot masks, got {robot_masks.shape}")
    if gripper_masks.shape != robot_masks.shape:
        raise ValueError(
            f"Gripper masks {gripper_masks.shape} do not match robot masks {robot_masks.shape}"
        )
    if not np.isin(robot_masks, (0, 1)).all():
        raise ValueError("Robot masks are not binary")
    if not np.isin(gripper_masks, (0, 1)).all():
        raise ValueError("Gripper masks are not binary")
    if np.any(np.logical_and(gripper_masks > 0, robot_masks == 0)):
        raise ValueError("Gripper mask is not a subset of the robot mask")

    target = pad_views(robot_masks, num_views).astype(np.float32)
    gripper = pad_views(gripper_masks, num_views).astype(np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != target.shape:
        raise ValueError(
            f"Decoded probability shape {probabilities.shape} does not match {target.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Decoded mask probabilities contain non-finite values")

    valid = np.zeros_like(target, dtype=bool)
    valid[..., : robot_masks.shape[-1]] = True
    predicted_binary = probabilities >= threshold
    single_view_width = robot_masks.shape[-1]
    initial = binary_mask_metrics(
        predicted_binary[0, :, :single_view_width],
        target[0, :, :single_view_width] > 0,
    )
    future = counterfactual_mask_metrics(
        probabilities[1:],
        target[1:],
        target[0],
        valid_masks=valid[1:],
        target_gripper_masks=gripper[1:],
        horizons=EXPECTED_HORIZONS[1:],
        threshold=threshold,
    )
    pixel_mae = float(np.abs(probabilities - target)[valid].mean())
    return {
        "initial": initial,
        "future": future,
        "valid_pixel_mae": pixel_mae,
    }


def finite_distribution(values: list[float], expected_count: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size != expected_count:
        raise ValueError(
            f"Expected {expected_count} finite values, got {finite.size}/{array.size}"
        )
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one reconstruction sample is required")
    metrics = {
        "initial_iou": [sample["metrics"]["initial"]["iou"] for sample in samples],
        "full_iou": [
            sample["metrics"]["future"]["mean"]["full_iou"] for sample in samples
        ],
        "dynamic_iou": [
            sample["metrics"]["future"]["mean"]["dynamic_iou"] for sample in samples
        ],
        "new_occupancy_iou": [
            sample["metrics"]["future"]["mean"]["new_occupancy_iou"]
            for sample in samples
        ],
        "gripper_region_recall": [
            sample["metrics"]["future"]["mean"]["gripper_region_recall"]
            for sample in samples
        ],
        "valid_pixel_mae": [
            sample["metrics"]["valid_pixel_mae"] for sample in samples
        ],
    }
    return {
        name: finite_distribution(values, len(samples)) for name, values in metrics.items()
    }


def evaluate_gate(
    aggregate: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    checks = {
        "initial_iou_mean": (
            aggregate["initial_iou"]["mean"]
            >= thresholds["min_initial_iou_mean"]
        ),
        "full_iou_mean": (
            aggregate["full_iou"]["mean"] >= thresholds["min_full_iou_mean"]
        ),
        "full_iou_p05": (
            aggregate["full_iou"]["p05"] >= thresholds["min_full_iou_p05"]
        ),
        "dynamic_iou_mean": (
            aggregate["dynamic_iou"]["mean"]
            >= thresholds["min_dynamic_iou_mean"]
        ),
        "dynamic_iou_p05": (
            aggregate["dynamic_iou"]["p05"]
            >= thresholds["min_dynamic_iou_p05"]
        ),
        "new_occupancy_iou_mean": (
            aggregate["new_occupancy_iou"]["mean"]
            >= thresholds["min_new_iou_mean"]
        ),
        "new_occupancy_iou_p05": (
            aggregate["new_occupancy_iou"]["p05"]
            >= thresholds["min_new_iou_p05"]
        ),
        "gripper_recall_mean": (
            aggregate["gripper_region_recall"]["mean"]
            >= thresholds["min_gripper_recall_mean"]
        ),
        "gripper_recall_p05": (
            aggregate["gripper_region_recall"]["p05"]
            >= thresholds["min_gripper_recall_p05"]
        ),
        "pixel_mae_mean": (
            aggregate["valid_pixel_mae"]["mean"]
            <= thresholds["max_pixel_mae_mean"]
        ),
        "pixel_mae_p95": (
            aggregate["valid_pixel_mae"]["p95"]
            <= thresholds["max_pixel_mae_p95"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.incomplete"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.num_views <= 0:
        raise ValueError("--num-views must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    cache_root = args.cache.expanduser().resolve()
    vae_root = args.vae.expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    if not vae_root.is_dir():
        raise FileNotFoundError(vae_root)
    vae_config = vae_root / "config.json"
    if not vae_config.is_file():
        raise FileNotFoundError(vae_config)
    manifest = read_manifest(cache_root)
    records = list(manifest["episodes"])
    if args.max_samples:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No cache records selected")

    import torch
    from diffusers.models import AutoencoderKLWan

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    vae = AutoencoderKLWan.from_pretrained(vae_root, torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval().to(device=device, dtype=dtype)
    latent_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latent_scale = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)

    samples: list[dict[str, Any]] = []
    for record in records:
        cache_path = cache_root / str(record["cache_file"])
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        expected_sha = str(record.get("cache_sha256", ""))
        actual_sha = sha256_file(cache_path)
        if not expected_sha or actual_sha != expected_sha:
            raise ValueError(f"Cache SHA256 mismatch: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            horizons = [int(value) for value in cache["horizons"].tolist()]
            robot_masks = cache["robot_mask"].copy()
            gripper_masks = cache["gripper_mask"].copy()
        if horizons != EXPECTED_HORIZONS:
            raise ValueError(f"Unexpected horizons in {cache_path}: {horizons}")

        mask_video = pad_views(robot_masks, args.num_views).astype(np.float32)
        mask_video = np.repeat(mask_video[:, None], 3, axis=1)
        video = torch.from_numpy(mask_video * 2.0 - 1.0).unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
        with torch.inference_mode():
            encoded = vae.encode(video).latent_dist.mode()
            normalized = (encoded - latent_mean) * latent_scale
            decoded = vae.decode(normalized / latent_scale + latent_mean).sample
        if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[1] != 3:
            raise ValueError(f"Unexpected decoded video shape: {tuple(decoded.shape)}")
        if decoded.shape[2:] != video.shape[2:]:
            raise ValueError(
                f"Decoded video shape {tuple(decoded.shape)} does not match input "
                f"{tuple(video.shape)}"
            )
        probabilities = (
            ((decoded[0].float().clamp(-1.0, 1.0) + 1.0) * 0.5)
            .mean(dim=0)
            .cpu()
            .numpy()
        )
        metrics = evaluate_reconstruction(
            probabilities,
            robot_masks,
            gripper_masks,
            num_views=args.num_views,
            threshold=args.threshold,
        )
        samples.append(
            {
                "episode_index": int(record["episode_index"]),
                "task_index": int(record["task_index"]),
                "cache_file": str(cache_path),
                "cache_sha256": actual_sha,
                "latent_shape": list(normalized.shape),
                "metrics": metrics,
            }
        )

    thresholds = {
        "min_initial_iou_mean": args.min_initial_iou_mean,
        "min_full_iou_mean": args.min_full_iou_mean,
        "min_full_iou_p05": args.min_full_iou_p05,
        "min_dynamic_iou_mean": args.min_dynamic_iou_mean,
        "min_dynamic_iou_p05": args.min_dynamic_iou_p05,
        "min_new_iou_mean": args.min_new_iou_mean,
        "min_new_iou_p05": args.min_new_iou_p05,
        "min_gripper_recall_mean": args.min_gripper_recall_mean,
        "min_gripper_recall_p05": args.min_gripper_recall_p05,
        "max_pixel_mae_mean": args.max_pixel_mae_mean,
        "max_pixel_mae_p95": args.max_pixel_mae_p95,
    }
    aggregate = aggregate_samples(samples)
    gate = evaluate_gate(aggregate, thresholds)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": gate["status"],
        "purpose": (
            "representation gate before training policy-action mask-video flow through "
            "the shared Wan VAE and RGB video output path"
        ),
        "counterfactual_rgb_used": False,
        "cache": str(cache_root),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "cache_schema_version": manifest["schema_version"],
        "policy_id": manifest.get("policy_id"),
        "vae": str(vae_root),
        "vae_config_sha256": sha256_file(vae_config),
        "device": str(device),
        "dtype": args.dtype,
        "num_views": args.num_views,
        "threshold": args.threshold,
        "horizons": EXPECTED_HORIZONS,
        "num_samples": len(samples),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "gate": gate,
        "samples": samples,
        "git_commit": git_commit(PROJECT_ROOT),
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()

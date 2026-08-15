#!/usr/bin/env python3
"""Measure the expert-action versus policy-action video-conditioning gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluate_counterfactual_mask_head import (  # noqa: E402
    aggregate_metrics,
    build_dataset,
    encode_reference,
    evaluate_prediction,
    manifest_episode_indices,
    normalize_cached_actions,
    predict_mask_logits,
    read_manifest,
    resolve_dtype,
    taskwise_shuffle_indices,
)
from world_action_model.alignment import (  # noqa: E402
    LatentRobotMaskProbe,
    counterfactual_mask_metrics,
)
from world_action_model.models import CasualWorldActionTransformer_MoT  # noqa: E402


DEMO_SCHEMA = "libero-logged-action-kinematic-mask-v1"
RENDERER_AUDIT_SCHEMA = "libero-kinematic-mask-pipeline-audit-v1"
GAP_KEYS = ("dynamic_iou", "new_occupancy_iou", "gripper_region_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--demonstration-cache", type=Path, required=True)
    parser.add_argument("--renderer-audit", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--mask-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--view-keys",
        nargs="+",
        default=("observation.images.image", "observation.images.wrist_image"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_demo_inputs(
    demo_root: Path,
    renderer_audit_path: Path,
    counterfactual_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = demo_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != DEMO_SCHEMA:
        raise ValueError(f"Unexpected demonstration cache schema: {manifest_path}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError("Demonstration-mask cache must not provide an RGB target")
    calls = manifest.get("dynamics_attempted_calls", {})
    if manifest.get("dynamics") != "forbidden" or set(calls) != {
        "env_step",
        "sim_step",
        "mj_step",
    } or any(
        int(value) != 0 for value in calls.values()
    ):
        raise ValueError("Logged-q demonstration masks attempted dynamics")
    if manifest.get("action_array") != "expert_actions_lerobot":
        raise ValueError("Unexpected expert action semantics")
    for key in ("dataset_root", "dataset_info_sha256", "suite", "horizons"):
        if manifest.get(key) != counterfactual_manifest.get(key):
            raise ValueError(f"Demo/counterfactual cache mismatch for {key}")
    records = list(manifest.get("episodes", []))
    counterfactual_records = list(counterfactual_manifest.get("episodes", []))
    demo_keys = [
        (int(record["episode_index"]), int(record["source_frame_index"]))
        for record in records
    ]
    counterfactual_keys = [
        (int(record["episode_index"]), int(record["source_frame_index"]))
        for record in counterfactual_records
    ]
    if demo_keys != counterfactual_keys:
        raise ValueError("Demo/counterfactual episode order differs")
    for demo_record, counterfactual_record in zip(records, counterfactual_records):
        for key in (
            "task_index",
            "libero_task_index",
            "task_language",
            "raw_demo_name",
            "raw_hdf5_path",
            "raw_hdf5_sha256",
        ):
            if demo_record.get(key) != counterfactual_record.get(key):
                raise ValueError(
                    f"Demo/counterfactual provenance differs for episode "
                    f"{demo_record['episode_index']} key {key}"
                )
        cache_path = demo_root / str(demo_record["cache_file"])
        if sha256_file(cache_path) != demo_record.get("cache_sha256"):
            raise ValueError(f"Demonstration cache changed: {cache_path}")
    renderer_audit = load_json(renderer_audit_path)
    if renderer_audit.get("schema_version") != RENDERER_AUDIT_SCHEMA:
        raise ValueError("Unexpected renderer audit schema")
    attempted = renderer_audit.get("dynamics_prohibition", {}).get(
        "attempted_calls", {}
    )
    registration = renderer_audit.get("renderer_registration", {}).get(
        "robot_mask_iou", {}
    )
    if set(attempted) != {"env_step", "sim_step", "mj_step"} or any(
        int(value) != 0 for value in attempted.values()
    ) or float(
        registration.get("mean", -1.0)
    ) < 0.85 or float(registration.get("p05", -1.0)) < 0.80:
        raise RuntimeError("Direct-q renderer audit did not pass")
    return manifest, records, renderer_audit


def normalize_demo_actions(
    demo_root: Path,
    records: list[dict[str, Any]],
    norm_stats_path: Path,
    model_action_dim: int = 16,
) -> np.ndarray:
    stats = load_json(norm_stats_path)["norm_stats"]["action"]
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    normalized_actions = []
    for record in records:
        with np.load(demo_root / str(record["cache_file"]), allow_pickle=False) as cache:
            action = cache["expert_actions_lerobot"].astype(np.float32)
        if action.ndim != 2 or action.shape[0] != 48 or action.shape[1] < 7:
            raise ValueError(f"Expected [48,>=7] demo action, got {action.shape}")
        effective_dim = min(action.shape[-1], low.shape[0], high.shape[0])
        normalized = np.zeros((48, model_action_dim), dtype=np.float32)
        action_range = np.maximum(high[:effective_dim] - low[:effective_dim], 1e-8)
        normalized[:, :effective_dim] = (
            2.0 * (action[:, :effective_dim] - low[:effective_dim]) / action_range - 1.0
        )
        normalized_actions.append(np.clip(normalized, -1.0, 1.0))
    return np.stack(normalized_actions)


def load_demo_target(
    demo_root: Path,
    record: dict[str, Any],
    num_views: int,
) -> dict[str, np.ndarray]:
    with np.load(demo_root / str(record["cache_file"]), allow_pickle=False) as cache:
        robot = cache["robot_mask"].astype(np.float32)
        gripper = cache["gripper_mask"].astype(np.float32)
        horizons = cache["horizons"].astype(np.int64)
    if robot.shape != (5, 256, 256) or gripper.shape != robot.shape:
        raise ValueError(f"Unexpected demo mask shape for episode {record['episode_index']}")
    if horizons.tolist() != [0, 12, 24, 36, 48]:
        raise ValueError(f"Unexpected demo horizons for episode {record['episode_index']}")
    valid = np.ones_like(robot, dtype=np.float32)
    if num_views > 1:
        unknown_width = robot.shape[-1] * (num_views - 1)
        pad = ((0, 0), (0, 0), (0, unknown_width))
        robot = np.pad(robot, pad)
        gripper = np.pad(gripper, pad)
        valid = np.pad(valid, pad)
    return {
        "initial": robot[0],
        "future": robot[1:],
        "gripper": gripper[1:],
        "valid": valid[1:],
        "horizons": horizons[1:],
    }


def resize_array(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(value).float().unsqueeze(0)
    return F.interpolate(tensor, size=size, mode="nearest")[0].numpy()


def evaluate_demo_prediction(
    probability: np.ndarray,
    target: dict[str, np.ndarray],
    size: tuple[int, int],
    threshold: float,
) -> dict[str, Any]:
    return counterfactual_mask_metrics(
        probability,
        resize_array(target["future"], size),
        resize_array(target["initial"][None], size)[0],
        valid_masks=resize_array(target["valid"], size),
        target_gripper_masks=resize_array(target["gripper"], size),
        horizons=target["horizons"].tolist(),
        threshold=threshold,
    )


def paired_bootstrap(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        end = min(samples, start + 1000)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def metric_vector(results: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([result["mean"][key] for result in results], dtype=np.float64)


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
    for path in (
        args.checkpoint,
        args.vae,
        args.dataset,
        args.cache,
        args.demonstration_cache,
        args.renderer_audit,
        args.norm_stats,
        args.mask_probe,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 0.0 < args.threshold < 1.0 or args.bootstrap_samples <= 0:
        raise ValueError("Invalid threshold/bootstrap count")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = resolve_dtype(args.model_dtype)

    counterfactual_manifest = read_manifest(args.cache)
    demo_manifest, demo_records, renderer_audit = validate_demo_inputs(
        args.demonstration_cache.resolve(),
        args.renderer_audit.resolve(),
        counterfactual_manifest,
    )
    predicted_actions = normalize_cached_actions(
        args.cache, counterfactual_manifest, args.norm_stats
    )
    expert_actions = normalize_demo_actions(
        args.demonstration_cache, demo_records, args.norm_stats
    )
    dataset = build_dataset(args)
    num_samples = len(dataset)
    if num_samples != len(demo_records) or num_samples != len(predicted_actions):
        raise ValueError("Dataset/demo/counterfactual sample counts differ")
    shuffle_indices = taskwise_shuffle_indices(counterfactual_manifest, num_samples)
    if shuffle_indices is None:
        raise ValueError("Action-gap diagnostic requires at least two samples per task")

    transformer = CasualWorldActionTransformer_MoT.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    ).to(device).eval()
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype).to(device).eval()
    probe = LatentRobotMaskProbe.from_pretrained(args.mask_probe).to(
        device=device, dtype=torch.float32
    ).requires_grad_(False).eval()
    if probe.latent_channels != int(vae.config.z_dim):
        raise ValueError("Probe/VAE latent channel mismatch")
    if probe.frames_per_latent != int(vae.config.scale_factor_temporal):
        raise ValueError("Probe/VAE temporal scale mismatch")
    if probe.spatial_scale != int(vae.config.scale_factor_spatial):
        raise ValueError("Probe/VAE spatial scale mismatch")
    latent_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latent_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)

    modes: dict[str, list[dict[str, Any]]] = {
        "expert_correct": [],
        "expert_shuffled": [],
        "policy_correct": [],
        "policy_shuffled": [],
    }
    per_sample = []
    try:
        for index in range(num_samples):
            sample = dataset[index]
            expected_expert = torch.from_numpy(expert_actions[index])
            expected_policy = torch.from_numpy(predicted_actions[index])
            if not torch.allclose(sample["action"], expected_expert, atol=1e-5, rtol=1e-5):
                raise ValueError(f"Expert action mismatch at sample {index}")
            if not torch.allclose(
                sample["counterfactual_action"], expected_policy, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(f"Policy action mismatch at sample {index}")
            ref_latents = encode_reference(
                vae, sample["ref_images"], latent_mean, latent_std
            ).to(dtype=dtype)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + index * 10007
            )
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
            shuffle_index = shuffle_indices[index]
            action_by_mode = {
                "expert_correct": torch.from_numpy(expert_actions[index]),
                "expert_shuffled": torch.from_numpy(expert_actions[shuffle_index]),
                "policy_correct": sample["counterfactual_action"],
                "policy_shuffled": torch.from_numpy(predicted_actions[shuffle_index]),
            }
            demo_target = load_demo_target(
                args.demonstration_cache, demo_records[index], len(args.view_keys)
            )
            sample_row: dict[str, Any] = {
                "index": index,
                "episode_index": int(demo_records[index]["episode_index"]),
                "shuffle_index": shuffle_index,
            }
            for mode, action in action_by_mode.items():
                logits = predict_mask_logits(
                    transformer,
                    ref_latents,
                    action,
                    sample,
                    future_noise,
                    dtype,
                    "clean_latent_probe",
                    probe,
                )
                probability = logits.sigmoid()[0].cpu().numpy()
                size = tuple(logits.shape[-2:])
                if mode.startswith("expert_"):
                    metrics = evaluate_demo_prediction(
                        probability, demo_target, size, args.threshold
                    )
                else:
                    metrics = evaluate_prediction(
                        probability, sample, size, args.threshold
                    )
                modes[mode].append(metrics)
                sample_row[mode] = metrics["mean"]
            per_sample.append(sample_row)
    finally:
        dataset.close()

    aggregate = {mode: aggregate_metrics(results) for mode, results in modes.items()}
    sensitivity = {"expert": {}, "policy": {}}
    gap = {}
    bootstrap = {"expert_sensitivity": {}, "policy_sensitivity": {}, "gap": {}}
    checks = {}
    for key_index, key in enumerate(GAP_KEYS):
        expert_values = metric_vector(modes["expert_correct"], key) - metric_vector(
            modes["expert_shuffled"], key
        )
        policy_values = metric_vector(modes["policy_correct"], key) - metric_vector(
            modes["policy_shuffled"], key
        )
        gap_values = expert_values - policy_values
        sensitivity["expert"][key] = float(expert_values.mean())
        sensitivity["policy"][key] = float(policy_values.mean())
        gap[key] = float(gap_values.mean())
        expert_bootstrap = paired_bootstrap(
            expert_values, args.bootstrap_samples, args.seed + key_index
        )
        policy_bootstrap = paired_bootstrap(
            policy_values, args.bootstrap_samples, args.seed + 100 + key_index
        )
        gap_bootstrap = paired_bootstrap(
            gap_values, args.bootstrap_samples, args.seed + 200 + key_index
        )
        bootstrap["expert_sensitivity"][key] = expert_bootstrap
        bootstrap["policy_sensitivity"][key] = policy_bootstrap
        bootstrap["gap"][key] = gap_bootstrap
        checks[f"expert_action_sensitive.{key}"] = (
            expert_bootstrap["ci95_low"] > 0.0
        )
        checks[f"expert_sensitivity_exceeds_policy.{key}"] = (
            gap_bootstrap["ci95_low"] > 0.0
        )
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": "clean-latent-action-conditioning-gap-v2",
        "status": status,
        "alignment_training_allowed": status == "passed",
        "gate_semantics": "paired_bootstrap_ci95_low_strictly_positive",
        "checkpoint": str(args.checkpoint.resolve()),
        "vae": str(args.vae.resolve()),
        "dataset": str(args.dataset.resolve()),
        "counterfactual_cache": str(args.cache.resolve()),
        "counterfactual_manifest_sha256": sha256_file(args.cache / "manifest.json"),
        "demonstration_cache": str(args.demonstration_cache.resolve()),
        "demonstration_manifest_sha256": sha256_file(
            args.demonstration_cache / "manifest.json"
        ),
        "renderer_audit": str(args.renderer_audit.resolve()),
        "renderer_audit_sha256": sha256_file(args.renderer_audit),
        "renderer_registration": renderer_audit["renderer_registration"],
        "mask_probe": str(args.mask_probe.resolve()),
        "mask_probe_config_sha256": sha256_file(
            args.mask_probe / LatentRobotMaskProbe.CONFIG_NAME
        ),
        "mask_probe_weights_sha256": sha256_file(
            args.mask_probe / LatentRobotMaskProbe.WEIGHTS_NAME
        ),
        "num_samples": num_samples,
        "episodes": manifest_episode_indices(counterfactual_manifest),
        "input_views": list(args.view_keys),
        "view_concat_mode": "horizontal",
        "noise_pairing": "same future noise for expert/policy and correct/shuffled",
        "shuffle_semantics": "cyclic derangement within task",
        "seed": args.seed,
        "threshold": args.threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "mask_resolution": list(size),
        "aggregate": aggregate,
        "action_sensitivity": sensitivity,
        "expert_minus_policy_sensitivity": gap,
        "bootstrap": bootstrap,
        "checks": checks,
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": status,
                    "action_sensitivity": sensitivity,
                    "expert_minus_policy_sensitivity": gap,
                    "checks": checks,
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

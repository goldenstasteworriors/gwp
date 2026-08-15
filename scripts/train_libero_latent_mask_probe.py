#!/usr/bin/env python3
"""Fit and gate a frozen clean-video-latent robot-mask probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from einops import rearrange
from giga_datasets.datasets.lerobot_dataset import (
    FastLeRobotDataset,
    _LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_action_model.alignment import (
    LatentRobotMaskProbe,
    copy_initial_mask_baseline,
    counterfactual_mask_metrics,
    masked_bce_dice_loss,
    per_horizon_mean_mask_baseline,
)


CACHE_SCHEMA = "libero-joint-kinematic-robot-mask-v1"
RENDERER_AUDIT_SCHEMA = "libero-kinematic-mask-pipeline-audit-v1"
EXPECTED_HORIZONS = [0, 12, 24, 36, 48]
PROBE_VIEW_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PROBE_VIEW_CONCAT_MODE = "horizontal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--heldout-cache", type=Path, required=True)
    parser.add_argument("--renderer-audit", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
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


def read_cache(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"Unexpected cache schema in {manifest_path}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError("Probe cache must not contain a video-generation target")
    if manifest.get("q_source") != "logged_demonstration_joint_trajectory":
        raise ValueError("Probe targets must come from logged q, not predicted q")
    if manifest.get("dynamics") != "forbidden" or any(
        int(value) != 0
        for value in manifest.get("dynamics_attempted_calls", {}).values()
    ):
        raise ValueError("Probe cache attempted simulator dynamics")
    if manifest.get("robot_visibility") != "amodal_robot_only_no_task_object_occlusion":
        raise ValueError("Unexpected robot-mask visibility semantics")
    if [int(value) for value in manifest.get("horizons", [])] != EXPECTED_HORIZONS:
        raise ValueError(f"Unexpected horizons in {manifest_path}")
    records = list(manifest.get("episodes", []))
    if not records or len(records) != int(manifest.get("num_episodes", -1)):
        raise ValueError(f"Invalid episode records in {manifest_path}")
    for record in records:
        cache_path = root / str(record["cache_file"])
        if sha256_file(cache_path) != record.get("cache_sha256"):
            raise ValueError(f"Probe mask cache changed: {cache_path}")
    return manifest, records


def validate_renderer_audit(report_path: Path) -> dict[str, Any]:
    report = load_json(report_path)
    if report.get("schema_version") != RENDERER_AUDIT_SCHEMA:
        raise ValueError(f"Unexpected renderer audit schema in {report_path}")
    attempted = report.get("dynamics_prohibition", {}).get("attempted_calls", {})
    if set(attempted) != {"env_step", "sim_step", "mj_step"} or any(
        int(value) != 0 for value in attempted.values()
    ):
        raise RuntimeError("Renderer audit attempted simulator dynamics")
    registration = report.get("renderer_registration", {}).get("robot_mask_iou", {})
    if float(registration.get("mean", -1.0)) < 0.85 or float(
        registration.get("p05", -1.0)
    ) < 0.80:
        raise RuntimeError("Direct-q renderer registration gate did not pass")
    return report


class DemoMaskDataset(Dataset):
    def __init__(self, root: Path, manifest: dict[str, Any], records: list[dict[str, Any]]):
        self.root = root
        self.records = records
        self.dataset_root = Path(str(manifest["dataset_root"])).resolve()
        repo_id = self.dataset_root.name
        metadata = _LeRobotDatasetMetadata(repo_id, root=self.dataset_root)
        self.dataset = FastLeRobotDataset(
            repo_id,
            root=self.dataset_root,
            delta_timestamps={
                key: [horizon / metadata.fps for horizon in EXPECTED_HORIZONS]
                for key in PROBE_VIEW_KEYS
            },
            episodes=[int(record["episode_index"]) for record in records],
            video_backend="pyav",
        )
        self.dataset._ensure_hf_dataset_loaded()
        source_to_index = {}
        for index, (episode, frame) in enumerate(
            zip(
                self.dataset.hf_dataset["episode_index"],
                self.dataset.hf_dataset["frame_index"],
            )
        ):
            source_to_index[(int(episode), int(frame))] = index
        self.sample_indices = [
            source_to_index[
                (int(record["episode_index"]), int(record.get("source_frame_index", 0)))
            ]
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        sample = self.dataset[self.sample_indices[index]]
        views = [sample[key].to(dtype=torch.float32) for key in PROBE_VIEW_KEYS]
        if any(view.shape != views[0].shape for view in views[1:]):
            raise ValueError(
                f"Probe view shapes differ for episode {record['episode_index']}: "
                f"{[tuple(view.shape) for view in views]}"
            )
        images = torch.cat(views, dim=-1)
        with np.load(self.root / str(record["cache_file"]), allow_pickle=False) as cache:
            masks = torch.from_numpy(cache["robot_mask"].copy()).to(dtype=torch.float32)
            gripper_masks = torch.from_numpy(cache["gripper_mask"].copy()).to(
                dtype=torch.float32
            )
        unknown_width = int(masks.shape[-1]) * (len(PROBE_VIEW_KEYS) - 1)
        valid = torch.ones_like(masks)
        masks = F.pad(masks, (0, unknown_width), value=0.0)
        gripper_masks = F.pad(gripper_masks, (0, unknown_width), value=0.0)
        valid = F.pad(valid, (0, unknown_width), value=0.0)
        return {
            "images": images,
            "initial_mask": masks[0],
            "future_masks": masks[1:],
            "future_gripper_masks": gripper_masks[1:],
            "future_mask_valid": valid[1:],
            "episode_index": torch.tensor(int(record["episode_index"])),
        }


@torch.no_grad()
def encode_video(
    vae: AutoencoderKLWan,
    images: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    images = images.to(device=vae.device, dtype=vae.dtype)
    images = images * 2.0 - 1.0
    images = rearrange(images, "b t c h w -> b c t h w")
    latents = vae.encode(images).latent_dist.mode()
    return (latents - latents_mean) * latents_std


def aggregate_metrics(per_sample: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("full_iou", "dynamic_iou", "new_occupancy_iou", "gripper_region_recall")
    return {
        key: float(np.mean([sample["mean"][key] for sample in per_sample]))
        for key in keys
    }


@torch.no_grad()
def evaluate(
    probe: LatentRobotMaskProbe,
    vae: AutoencoderKLWan,
    loader: DataLoader,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    threshold: float,
    mean_mask: np.ndarray,
) -> dict[str, Any]:
    probe.eval()
    methods: dict[str, list[dict[str, Any]]] = {
        "probe": [],
        "copy_initial": [],
        "train_mean": [],
    }
    episode_rows = []
    for batch in loader:
        latents = encode_video(vae, batch["images"], latents_mean, latents_std)
        future_latents = latents[:, :, 1:]
        logits = probe(future_latents.to(dtype=torch.float32))
        probabilities = logits.sigmoid().cpu().numpy()
        targets = batch["future_masks"].numpy()
        grippers = batch["future_gripper_masks"].numpy()
        valid = batch["future_mask_valid"].numpy()
        initial = batch["initial_mask"].numpy()
        if probabilities.shape != targets.shape:
            probabilities = F.interpolate(
                torch.from_numpy(probabilities), size=targets.shape[-2:], mode="bilinear"
            ).numpy()
        for batch_index in range(len(probabilities)):
            candidates = {
                "probe": probabilities[batch_index],
                "copy_initial": copy_initial_mask_baseline(
                    initial[batch_index], targets.shape[1]
                ),
                "train_mean": mean_mask,
            }
            row = {"episode_index": int(batch["episode_index"][batch_index])}
            for method, prediction in candidates.items():
                metrics = counterfactual_mask_metrics(
                    prediction,
                    targets[batch_index],
                    initial[batch_index],
                    valid_masks=valid[batch_index],
                    target_gripper_masks=grippers[batch_index],
                    horizons=EXPECTED_HORIZONS[1:],
                    threshold=threshold,
                )
                methods[method].append(metrics)
                row[method] = metrics["mean"]
            episode_rows.append(row)
    return {
        "aggregate": {
            method: aggregate_metrics(results) for method, results in methods.items()
        },
        "episodes": episode_rows,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.batch_size <= 0 or args.epochs <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch/epoch/worker count")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0,1)")
    train_root = args.train_cache.resolve()
    heldout_root = args.heldout_cache.resolve()
    train_manifest, train_records = read_cache(train_root)
    heldout_manifest, heldout_records = read_cache(heldout_root)
    if Path(str(train_manifest["dataset_root"])).resolve() != Path(
        str(heldout_manifest["dataset_root"])
    ).resolve():
        raise ValueError("Train/held-out probe caches use different datasets")
    train_episodes = {int(record["episode_index"]) for record in train_records}
    heldout_episodes = {int(record["episode_index"]) for record in heldout_records}
    if train_episodes & heldout_episodes:
        raise ValueError("Train/held-out probe caches overlap")
    renderer_audit = validate_renderer_audit(args.renderer_audit)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    vae = AutoencoderKLWan.from_pretrained(args.vae.resolve())
    vae.requires_grad_(False)
    vae.eval().to(device=device, dtype=torch.float32)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=device).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    probe = LatentRobotMaskProbe(
        latent_channels=int(vae.config.z_dim),
        frames_per_latent=int(vae.config.scale_factor_temporal),
        spatial_scale=int(vae.config.scale_factor_spatial),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_dataset = DemoMaskDataset(train_root, train_manifest, train_records)
    heldout_dataset = DemoMaskDataset(heldout_root, heldout_manifest, heldout_records)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    training_masks = []
    for record in train_records:
        with np.load(train_root / str(record["cache_file"]), allow_pickle=False) as cache:
            masks = cache["robot_mask"][1:].copy()
            training_masks.append(
                np.pad(
                    masks,
                    (
                        (0, 0),
                        (0, 0),
                        (0, masks.shape[-1] * (len(PROBE_VIEW_KEYS) - 1)),
                    ),
                )
            )
    mean_mask = per_horizon_mean_mask_baseline(np.stack(training_masks), args.threshold)

    history = []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        losses = []
        for batch in train_loader:
            latents = encode_video(vae, batch["images"], latents_mean, latents_std)
            logits = probe(latents[:, :, 1:].to(dtype=torch.float32))
            target = batch["future_masks"].to(device=device, dtype=torch.float32)
            valid = batch["future_mask_valid"].to(device=device, dtype=torch.float32)
            if logits.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
                valid = F.interpolate(valid, size=logits.shape[-2:], mode="nearest")
            loss = masked_bce_dice_loss(logits, target, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not np.isfinite(losses).all():
            raise RuntimeError(f"Non-finite probe loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(json.dumps(history[-1]), flush=True)

    heldout = evaluate(
        probe,
        vae,
        heldout_loader,
        latents_mean,
        latents_std,
        args.threshold,
        mean_mask,
    )
    probe_metrics = heldout["aggregate"]["probe"]
    minimums = {
        "full_iou": 0.65,
        "dynamic_iou": 0.40,
        "new_occupancy_iou": 0.40,
        "gripper_region_recall": 0.70,
    }
    checks = {}
    for key, minimum in minimums.items():
        checks[f"absolute.{key}"] = float(probe_metrics[key]) >= minimum
        checks[f"beats_copy_initial.{key}"] = float(probe_metrics[key]) > float(
            heldout["aggregate"]["copy_initial"][key]
        )
        checks[f"beats_train_mean.{key}"] = float(probe_metrics[key]) > float(
            heldout["aggregate"]["train_mean"][key]
        )
    status = "passed" if all(checks.values()) else "failed"
    args.output_dir.mkdir(parents=True)
    probe_dir = args.output_dir / "probe"
    probe.eval().requires_grad_(False)
    probe.save_pretrained(probe_dir, approved_for_alignment=status == "passed")
    report = {
        "schema_version": "libero-latent-mask-probe-training-v1",
        "status": status,
        "seed": args.seed,
        "train_cache": str(train_root),
        "heldout_cache": str(heldout_root),
        "train_manifest_sha256": sha256_file(train_root / "manifest.json"),
        "heldout_manifest_sha256": sha256_file(heldout_root / "manifest.json"),
        "renderer_audit": str(args.renderer_audit.resolve()),
        "renderer_audit_sha256": sha256_file(args.renderer_audit),
        "renderer_registration": renderer_audit["renderer_registration"],
        "probe_target_semantics": (
            "real camera video latent paired with logged q rendered by direct FK "
            "and robot-mesh rasterization; no dynamics replay"
        ),
        "vae": str(args.vae.resolve()),
        "input_views": list(PROBE_VIEW_KEYS),
        "view_concat_mode": PROBE_VIEW_CONCAT_MODE,
        "mask_supervision_view": PROBE_VIEW_KEYS[0],
        "probe": str(probe_dir),
        "probe_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "probe_hidden_channels": probe.hidden_channels,
        "probe_config_sha256": sha256_file(probe_dir / probe.CONFIG_NAME),
        "probe_weights_sha256": sha256_file(probe_dir / probe.WEIGHTS_NAME),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
        },
        "minimums": minimums,
        "checks": checks,
        "history": history,
        "heldout": heldout,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "aggregate": heldout["aggregate"], "checks": checks}, indent=2))
    print(report_path, flush=True)


if __name__ == "__main__":
    main()

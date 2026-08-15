#!/usr/bin/env python3
"""Launch one audited arm of the stage-one counterfactual-mask comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


VARIANTS = ("matched", "full")
ALIGNMENT_MODES = (
    "hidden_head",
    "clean_latent_probe",
    "shared_mask_video_flow",
)
EXPECTED_CACHE_SCHEMAS = {
    "libero-counterfactual-robot-mask-v5",
    "libero-support-constrained-action-kinematic-mask-v1",
}
EXPECTED_IMAGE_ORIENTATION = "lerobot_libero_visualization_flip_height_width"
EXPECTED_TRAIN_EPISODES = 200
EXPECTED_HELDOUT_EPISODES = 50
EXPECTED_REGISTRATION_THRESHOLDS = {
    "position_error_mm.p95": 5.0,
    "rotation_error_deg.p95": 2.0,
    "gripper_max_abs.p95": 0.003,
    "best_shift_magnitude_px.p95": 4.0,
    "full_rgb_mae.p95": 10.0,
    "best_shift_robot_rgb_mae.p95": 15.0,
}
EXPECTED_PROBE_MINIMUMS = {
    "full_iou": 0.65,
    "dynamic_iou": 0.40,
    "new_occupancy_iou": 0.40,
    "gripper_region_recall": 0.70,
}
EXPECTED_MASK_VIDEO_VAE_THRESHOLDS = {
    "min_initial_iou_mean": 0.90,
    "min_full_iou_mean": 0.90,
    "min_full_iou_p05": 0.80,
    "min_dynamic_iou_mean": 0.80,
    "min_dynamic_iou_p05": 0.60,
    "min_new_iou_mean": 0.80,
    "min_new_iou_p05": 0.60,
    "min_gripper_recall_mean": 0.90,
    "min_gripper_recall_p05": 0.75,
    "max_pixel_mae_mean": 0.05,
    "max_pixel_mae_p95": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--wan-pretrained", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument(
        "--alignment-mode", choices=ALIGNMENT_MODES, default="hidden_head"
    )
    parser.add_argument("--mask-probe-report", type=Path)
    parser.add_argument("--action-gap-report", type=Path)
    parser.add_argument("--mask-video-vae-report", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mask-weight", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--required-env-name", default="gigaworld-policy")
    parser.add_argument(
        "--execution-profile",
        choices=("four_device", "single_device_fallback"),
        default="four_device",
    )
    parser.add_argument("--distributed-failure-manifest", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the launch record without creating output or training.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def filesystem_id(path: Path) -> int:
    """Return the filesystem id across OverlayFS lower/upper inodes."""

    return int(os.statvfs(path).f_fsid)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def git_output(project_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), *args], text=True
    ).strip()


def validate_cache_manifest(
    cache_root: Path,
    audit: dict[str, Any],
    expected_count: int,
    checkpoint: Path,
) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    manifest = load_json(manifest_path)
    schema = manifest.get("schema_version")
    if schema not in EXPECTED_CACHE_SCHEMAS:
        raise ValueError(f"Unexpected cache schema: {manifest_path}")
    if audit.get("schema_version") not in {None, schema}:
        raise ValueError(f"Preflight/cache schema mismatch: {manifest_path}")
    if schema == "libero-support-constrained-action-kinematic-mask-v1":
        calls = manifest.get("dynamics_attempted_calls", {})
        if (
            manifest.get("dynamics") != "forbidden"
            or set(calls) != {"env_step", "sim_step", "mj_step"}
            or any(int(value) != 0 for value in calls.values())
        ):
            raise ValueError(f"Kinematic cache attempted dynamics: {manifest_path}")
        if any(
            manifest.get(key) is not False
            for key in (
                "legacy_rollout_masks_read",
                "legacy_rollout_qpos_read",
                "simulator_actions_read",
            )
        ):
            raise ValueError(f"Kinematic cache read an oracle field: {manifest_path}")
    if manifest.get("image_orientation") != EXPECTED_IMAGE_ORIENTATION:
        raise ValueError(f"Cache/training image orientation mismatch: {manifest_path}")
    if int(manifest.get("num_episodes", -1)) != expected_count:
        raise ValueError(
            f"{manifest_path}: expected {expected_count} episodes, "
            f"got {manifest.get('num_episodes')}"
        )
    if Path(str(manifest.get("policy_id"))).resolve() != checkpoint:
        raise ValueError(f"Cache policy/checkpoint mismatch: {manifest_path}")
    if sha256_file(manifest_path) != audit.get("manifest_sha256"):
        raise ValueError(f"Cache manifest changed after preflight: {manifest_path}")
    dataset_root = Path(str(manifest.get("dataset_root"))).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    return manifest


def latest_input_mtime_ns(paths: list[Path]) -> int:
    latest = 0
    for root in paths:
        if root.is_file():
            latest = max(latest, root.stat().st_mtime_ns)
            continue
        for dirpath, _, filenames in os.walk(root):
            directory = Path(dirpath)
            latest = max(latest, directory.stat().st_mtime_ns)
            for filename in filenames:
                latest = max(latest, (directory / filename).stat().st_mtime_ns)
    return latest


def manifest_episode_indices(manifest: dict[str, Any]) -> list[int]:
    records = list(manifest.get("episodes", []))
    if len(records) != int(manifest.get("num_episodes", -1)) or not records:
        raise ValueError("Invalid cache episode records")
    return [int(record["episode_index"]) for record in records]


def validate_clean_latent_gates(
    args: argparse.Namespace,
    checkpoint: Path,
    vae: Path,
    heldout_cache: Path,
    heldout_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if args.alignment_mode != "clean_latent_probe":
        if args.mask_probe_report is not None or args.action_gap_report is not None:
            raise ValueError(
                "Probe/action-gap reports are only valid with clean_latent_probe"
            )
        return None
    if args.mask_probe_report is None or args.action_gap_report is None:
        raise ValueError(
            "clean_latent_probe requires --mask-probe-report and --action-gap-report"
        )

    probe_report_path = args.mask_probe_report.resolve()
    gap_report_path = args.action_gap_report.resolve()
    probe_report = load_json(probe_report_path)
    gap_report = load_json(gap_report_path)
    if probe_report.get("schema_version") != "libero-latent-mask-probe-training-v1":
        raise ValueError(f"Unexpected probe report schema: {probe_report_path}")
    expected_probe_checks = {
        f"{prefix}.{metric}"
        for metric in EXPECTED_PROBE_MINIMUMS
        for prefix in ("absolute", "beats_copy_initial", "beats_train_mean")
    }
    probe_checks = probe_report.get("checks", {})
    if (
        probe_report.get("status") != "passed"
        or set(probe_checks) != expected_probe_checks
        or not all(value is True for value in probe_checks.values())
    ):
        raise RuntimeError(f"Latent-mask probe gate did not pass: {probe_report_path}")
    probe_root = Path(str(probe_report.get("probe"))).resolve()
    probe_config = probe_root / "config.json"
    probe_weights = probe_root / "model.safetensors"
    if not probe_config.is_file() or not probe_weights.is_file():
        raise FileNotFoundError(f"Incomplete latent-mask probe: {probe_root}")
    if sha256_file(probe_config) != probe_report.get("probe_config_sha256"):
        raise ValueError("Probe config changed after held-out gate")
    if sha256_file(probe_weights) != probe_report.get("probe_weights_sha256"):
        raise ValueError("Probe weights changed after held-out gate")
    probe_config_value = load_json(probe_config)
    expected_probe_config = {
        "class_name": "LatentRobotMaskProbe",
        "latent_channels": 48,
        "frames_per_latent": 4,
        "spatial_scale": 8,
        "hidden_channels": 128,
        "approved_for_alignment": True,
    }
    if probe_config_value != expected_probe_config:
        raise ValueError(
            f"Unexpected approved probe config: {probe_config_value}, "
            f"expected {expected_probe_config}"
        )
    if int(probe_report.get("probe_parameters", -1)) != 88448:
        raise ValueError("Unexpected latent-mask probe parameter count")
    if probe_report.get("input_views") != [
        "observation.images.image",
        "observation.images.wrist_image",
    ]:
        raise ValueError("Probe was not trained on the fixed two-view input")
    if probe_report.get("view_concat_mode") != "horizontal":
        raise ValueError("Probe view layout is not horizontal")
    if probe_report.get("mask_supervision_view") != "observation.images.image":
        raise ValueError("Probe mask supervision is not restricted to agent-view")
    if Path(str(probe_report.get("vae"))).resolve() != vae:
        raise ValueError("Probe gate used a different VAE")
    if int(probe_report.get("seed", -1)) != 20260813:
        raise ValueError("Probe gate used a different seed")
    expected_probe_hyperparameters = {
        "batch_size": 4,
        "epochs": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "threshold": 0.5,
    }
    if probe_report.get("hyperparameters") != expected_probe_hyperparameters:
        raise ValueError("Probe gate used a different fixed training protocol")
    if probe_report.get("minimums") != EXPECTED_PROBE_MINIMUMS:
        raise ValueError("Probe gate used different held-out minimums")
    registration_thresholds = probe_report.get("registration_thresholds", {})
    if set(registration_thresholds) != {"train", "heldout"} or any(
        registration_thresholds.get(split) != EXPECTED_REGISTRATION_THRESHOLDS
        for split in ("train", "heldout")
    ):
        raise ValueError("Probe gate used different registration thresholds")

    if gap_report.get("schema_version") != "clean-latent-action-conditioning-gap-v2":
        raise ValueError(f"Unexpected action-gap report schema: {gap_report_path}")
    if gap_report.get("gate_semantics") != (
        "paired_bootstrap_ci95_low_strictly_positive"
    ):
        raise ValueError("Action-gap report used different statistical gate semantics")
    expected_gap_checks = {
        f"{prefix}.{metric}"
        for metric in (
            "dynamic_iou",
            "new_occupancy_iou",
            "gripper_region_recall",
        )
        for prefix in (
            "expert_action_sensitive",
            "expert_sensitivity_exceeds_policy",
        )
    }
    gap_checks = gap_report.get("checks", {})
    if (
        gap_report.get("status") != "passed"
        or gap_report.get("alignment_training_allowed") is not True
        or set(gap_checks) != expected_gap_checks
        or not all(value is True for value in gap_checks.values())
    ):
        raise RuntimeError(f"Expert-policy action gap did not pass: {gap_report_path}")
    gap_bootstrap = gap_report.get("bootstrap", {})
    for section in ("expert_sensitivity", "gap"):
        section_values = gap_bootstrap.get(section, {})
        for metric in (
            "dynamic_iou",
            "new_occupancy_iou",
            "gripper_region_recall",
        ):
            interval = section_values.get(metric, {})
            lower = interval.get("ci95_low")
            if (
                not isinstance(lower, (int, float))
                or isinstance(lower, bool)
                or not float(lower) > 0.0
            ):
                raise RuntimeError(
                    f"Action-gap bootstrap lower bound did not pass: "
                    f"{section}.{metric}={lower!r}"
                )
    if Path(str(gap_report.get("checkpoint"))).resolve() != checkpoint:
        raise ValueError("Action-gap report used a different baseline checkpoint")
    if Path(str(gap_report.get("vae"))).resolve() != vae:
        raise ValueError("Action-gap report used a different VAE")
    dataset_root = Path(str(heldout_manifest.get("dataset_root"))).resolve()
    if Path(str(gap_report.get("dataset"))).resolve() != dataset_root:
        raise ValueError("Action-gap report used a different held-out dataset")
    if Path(str(gap_report.get("counterfactual_cache"))).resolve() != heldout_cache:
        raise ValueError("Action-gap report used a different held-out cache")
    if gap_report.get("counterfactual_manifest_sha256") != sha256_file(
        heldout_cache / "manifest.json"
    ):
        raise ValueError("Held-out counterfactual cache changed after action-gap gate")
    if Path(str(gap_report.get("mask_probe"))).resolve() != probe_root:
        raise ValueError("Action-gap report used a different mask probe")
    if gap_report.get("mask_probe_config_sha256") != sha256_file(probe_config):
        raise ValueError("Action-gap/probe config hash mismatch")
    if gap_report.get("mask_probe_weights_sha256") != sha256_file(probe_weights):
        raise ValueError("Action-gap/probe weights hash mismatch")
    heldout_episodes = manifest_episode_indices(heldout_manifest)
    if [int(value) for value in gap_report.get("episodes", [])] != heldout_episodes:
        raise ValueError("Action-gap report used different held-out episodes/order")
    if gap_report.get("input_views") != probe_report.get("input_views"):
        raise ValueError("Action-gap/probe input views differ")
    if gap_report.get("view_concat_mode") != probe_report.get("view_concat_mode"):
        raise ValueError("Action-gap/probe view layouts differ")
    if int(gap_report.get("num_samples", -1)) != EXPECTED_HELDOUT_EPISODES:
        raise ValueError("Action-gap report did not use all held-out episodes")
    if int(gap_report.get("seed", -1)) != 20260813:
        raise ValueError("Action-gap report used a different seed")
    if float(gap_report.get("threshold", -1.0)) != 0.5:
        raise ValueError("Action-gap report used a different mask threshold")
    if int(gap_report.get("bootstrap_samples", -1)) != 10000:
        raise ValueError("Action-gap report used a different bootstrap count")
    if gap_report.get("noise_pairing") != (
        "same future noise for expert/policy and correct/shuffled"
    ):
        raise ValueError("Action-gap report did not use paired future noise")
    if gap_report.get("shuffle_semantics") != "cyclic derangement within task":
        raise ValueError("Action-gap report used different shuffled-action controls")
    demonstration_cache = Path(str(gap_report.get("demonstration_cache"))).resolve()
    demonstration_registration = Path(
        str(gap_report.get("demonstration_registration"))
    ).resolve()
    if Path(str(probe_report.get("heldout_cache"))).resolve() != demonstration_cache:
        raise ValueError("Probe/action-gap demonstration caches differ")
    if probe_report.get("heldout_manifest_sha256") != gap_report.get(
        "demonstration_manifest_sha256"
    ):
        raise ValueError("Probe/action-gap demonstration manifests differ")
    if (
        Path(str(probe_report.get("heldout_registration"))).resolve()
        != demonstration_registration
    ):
        raise ValueError("Probe/action-gap registration reports differ")
    if probe_report.get("heldout_registration_sha256") != gap_report.get(
        "demonstration_registration_sha256"
    ):
        raise ValueError("Probe/action-gap registration hashes differ")
    return {
        "probe_report": str(probe_report_path),
        "probe_report_sha256": sha256_file(probe_report_path),
        "probe_root": str(probe_root),
        "probe_config_sha256": sha256_file(probe_config),
        "probe_weights_sha256": sha256_file(probe_weights),
        "action_gap_report": str(gap_report_path),
        "action_gap_report_sha256": sha256_file(gap_report_path),
        "demonstration_cache": str(demonstration_cache),
        "demonstration_manifest_sha256": gap_report[
            "demonstration_manifest_sha256"
        ],
        "demonstration_registration": str(demonstration_registration),
        "demonstration_registration_sha256": gap_report[
            "demonstration_registration_sha256"
        ],
    }


def validate_shared_mask_video_gate(
    args: argparse.Namespace,
    vae: Path,
    heldout_cache: Path,
    heldout_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if args.alignment_mode != "shared_mask_video_flow":
        if args.mask_video_vae_report is not None:
            raise ValueError(
                "--mask-video-vae-report is only valid with shared_mask_video_flow"
            )
        return None
    if args.mask_video_vae_report is None:
        raise ValueError(
            "shared_mask_video_flow requires --mask-video-vae-report"
        )

    report_path = args.mask_video_vae_report.resolve()
    report = load_json(report_path)
    checks = report.get("gate", {}).get("checks", {})
    expected_checks = {
        "initial_iou_mean",
        "full_iou_mean",
        "full_iou_p05",
        "dynamic_iou_mean",
        "dynamic_iou_p05",
        "new_occupancy_iou_mean",
        "new_occupancy_iou_p05",
        "gripper_recall_mean",
        "gripper_recall_p05",
        "pixel_mae_mean",
        "pixel_mae_p95",
    }
    if (
        report.get("schema_version") != "counterfactual-mask-video-vae-audit-v1"
        or report.get("status") != "passed"
        or report.get("gate", {}).get("status") != "passed"
        or set(checks) != expected_checks
        or not all(value is True for value in checks.values())
    ):
        raise RuntimeError(f"Mask-video VAE gate did not pass: {report_path}")
    manifest_path = heldout_cache / "manifest.json"
    if Path(str(report.get("cache"))).resolve() != heldout_cache:
        raise ValueError("Mask-video VAE report uses a different held-out cache")
    if report.get("cache_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Held-out cache changed after mask-video VAE gate")
    if report.get("cache_schema_version") != heldout_manifest.get("schema_version"):
        raise ValueError("Unexpected cache schema in mask-video VAE report")
    if report.get("policy_id") != heldout_manifest.get("policy_id"):
        raise ValueError("Mask-video VAE report uses a different policy cache")
    if Path(str(report.get("vae"))).resolve() != vae:
        raise ValueError("Mask-video VAE report uses a different VAE")
    vae_config = vae / "config.json"
    if report.get("vae_config_sha256") != sha256_file(vae_config):
        raise ValueError("VAE config changed after mask-video representation gate")
    if int(report.get("num_samples", -1)) != EXPECTED_HELDOUT_EPISODES:
        raise ValueError("Mask-video VAE gate must cover all held-out episodes")
    if report.get("thresholds") != EXPECTED_MASK_VIDEO_VAE_THRESHOLDS:
        raise ValueError("Mask-video VAE gate thresholds differ from the protocol")
    if (
        int(report.get("num_views", -1)) != 2
        or float(report.get("threshold", -1.0)) != 0.5
        or report.get("dtype") != "fp32"
        or [int(value) for value in report.get("horizons", [])]
        != [0, 12, 24, 36, 48]
        or report.get("counterfactual_rgb_used") is not False
    ):
        raise ValueError("Unexpected mask-video VAE evaluation protocol")
    return {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "num_samples": int(report["num_samples"]),
        "thresholds": report["thresholds"],
        "checks": checks,
    }


def build_launch_record(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    report_path = args.preflight_report.resolve()
    report = load_json(report_path)
    if report.get("status") != "passed":
        raise ValueError(f"Preflight report did not pass: {report_path}")

    project_root = Path(str(report.get("project_root"))).resolve()
    script_project_root = Path(__file__).resolve().parents[1]
    if project_root != script_project_root:
        raise ValueError(
            f"Preflight project {project_root} does not match launcher project "
            f"{script_project_root}"
        )
    if not (project_root / ".git").exists():
        raise FileNotFoundError(f"Not a Git project: {project_root}")
    current_commit = git_output(project_root, "rev-parse", "HEAD")
    if current_commit != report.get("git_commit"):
        raise RuntimeError(
            "Git commit changed after preflight; regenerate the preflight report"
        )
    tracked_status = git_output(
        project_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_status:
        raise RuntimeError(f"Tracked project files are dirty:\n{tracked_status}")

    if int(report.get("num_train", -1)) != EXPECTED_TRAIN_EPISODES:
        raise ValueError("Stage one requires exactly 200 training episodes")
    if int(report.get("num_heldout", -1)) != EXPECTED_HELDOUT_EPISODES:
        raise ValueError("Stage one requires exactly 50 held-out episodes")
    if int(report.get("episode_overlap", -1)) != 0:
        raise ValueError("Stage-one train/held-out episode overlap must be zero")

    checkpoint = Path(str(report.get("checkpoint"))).resolve()
    train_cache = Path(str(report.get("train_cache", {}).get("root"))).resolve()
    heldout_cache = Path(str(report.get("heldout_cache", {}).get("root"))).resolve()
    split_manifest = Path(str(report.get("split_manifest"))).resolve()
    output_key = "matched_output" if args.variant == "matched" else "full_output"
    counterpart_key = "full_output" if args.variant == "matched" else "matched_output"
    output_dir = Path(str(report.get(output_key))).resolve()
    counterpart_dir = Path(str(report.get(counterpart_key))).resolve()
    wan_pretrained = args.wan_pretrained.resolve()
    vae = wan_pretrained / "vae"
    norm_stats = args.norm_stats.resolve()

    for directory in (checkpoint, train_cache, heldout_cache, wan_pretrained):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)
    if not split_manifest.is_file():
        raise FileNotFoundError(split_manifest)
    if sha256_file(split_manifest) != report.get("split_manifest_sha256"):
        raise ValueError("Split manifest changed after preflight")
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing stage-one output: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(output_dir.parent)

    train_manifest = validate_cache_manifest(
        train_cache,
        report["train_cache"],
        EXPECTED_TRAIN_EPISODES,
        checkpoint,
    )
    heldout_manifest = validate_cache_manifest(
        heldout_cache,
        report["heldout_cache"],
        EXPECTED_HELDOUT_EPISODES,
        checkpoint,
    )
    dataset_root = Path(str(train_manifest["dataset_root"])).resolve()
    if Path(str(heldout_manifest["dataset_root"])).resolve() != dataset_root:
        raise ValueError("Train/held-out caches use different datasets")
    clean_latent_gates = validate_clean_latent_gates(
        args,
        checkpoint,
        vae,
        heldout_cache,
        heldout_manifest,
    )
    shared_mask_video_gate = validate_shared_mask_video_gate(
        args,
        vae,
        heldout_cache,
        heldout_manifest,
    )

    t5_embedding_dir = dataset_root / "t5_embedding"
    if not t5_embedding_dir.is_dir():
        raise FileNotFoundError(t5_embedding_dir)

    output_filesystem = filesystem_id(output_dir.parent)
    stage_sources = [
        checkpoint,
        train_cache,
        heldout_cache,
        dataset_root,
        wan_pretrained,
        norm_stats,
    ]
    if clean_latent_gates is not None:
        stage_sources.extend(
            [
                Path(clean_latent_gates["probe_report"]),
                Path(clean_latent_gates["probe_root"]),
                Path(clean_latent_gates["action_gap_report"]),
                Path(clean_latent_gates["demonstration_cache"]),
                Path(clean_latent_gates["demonstration_registration"]),
            ]
        )
    if shared_mask_video_gate is not None:
        stage_sources.append(Path(shared_mask_video_gate["report"]))
    for source in stage_sources:
        if filesystem_id(source) != output_filesystem:
            raise ValueError(f"Large stage-one source/output crosses filesystems: {source}")

    latest_mtime_paths = [checkpoint, train_cache, heldout_cache, split_manifest]
    if clean_latent_gates is not None:
        latest_mtime_paths.extend(
            [
                Path(clean_latent_gates["probe_report"]),
                Path(clean_latent_gates["probe_root"]),
                Path(clean_latent_gates["action_gap_report"]),
                Path(clean_latent_gates["demonstration_cache"]),
                Path(clean_latent_gates["demonstration_registration"]),
            ]
        )
    if shared_mask_video_gate is not None:
        latest_mtime_paths.append(Path(shared_mask_video_gate["report"]))
    latest_mtime = latest_input_mtime_ns(latest_mtime_paths)
    if report_path.stat().st_mtime_ns < latest_mtime:
        raise RuntimeError(
            "A checkpoint/cache/split input is newer than the preflight report; "
            "regenerate the report"
        )
    counterpart_manifest = counterpart_dir / "stage1_launch_manifest.json"
    if counterpart_dir.exists() and not counterpart_manifest.is_file():
        raise FileExistsError(
            f"Counterpart output exists without a launch manifest: {counterpart_dir}"
        )
    if counterpart_manifest.is_file():
        required_free_gib = (
            float(report.get("source_checkpoint_weight_gib", 0.0)) / 2.0
            + float(report.get("reserve_gib", 0.0))
            + float(report.get("transient_gib", 0.0))
        )
    else:
        required_free_gib = float(report.get("required_free_gib", 0.0))
    current_free_gib = shutil.disk_usage(output_dir.parent).free / 1024**3
    if current_free_gib < required_free_gib:
        raise RuntimeError(
            f"Insufficient disk at launch: free={current_free_gib:.2f} GiB, "
            f"required={required_free_gib:.2f} GiB"
        )

    if not args.gpu_ids or len(args.gpu_ids) != len(set(args.gpu_ids)):
        raise ValueError("--gpu-ids must be non-empty and unique")
    if any(gpu_id < 0 for gpu_id in args.gpu_ids):
        raise ValueError("--gpu-ids must be non-negative")
    for name in (
        "steps",
        "batch_size_per_gpu",
        "gradient_accumulation_steps",
        "num_workers",
        "log_interval",
    ):
        require_positive(f"--{name.replace('_', '-')}", int(getattr(args, name)))
    if args.mask_weight <= 0.0:
        raise ValueError("--mask-weight must be positive")
    fixed_protocol = {
        "gpu_ids": [0, 1, 2, 3] if args.execution_profile == "four_device" else [0],
        "steps": 200,
        "batch_size_per_gpu": 1,
        "gradient_accumulation_steps": (
            4 if args.execution_profile == "four_device" else 16
        ),
        "num_workers": 4,
        "seed": 6666,
        "mask_weight": 0.1,
    }
    actual_protocol = {
        "gpu_ids": args.gpu_ids,
        "steps": args.steps,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "mask_weight": args.mask_weight,
    }
    if actual_protocol != fixed_protocol:
        raise ValueError(
            "Stage-one protocol is fixed; received "
            f"{actual_protocol}, expected {fixed_protocol}"
        )
    distributed_failure = None
    if args.execution_profile == "single_device_fallback":
        if args.distributed_failure_manifest is None:
            raise ValueError(
                "single_device_fallback requires --distributed-failure-manifest"
            )
        failure_path = args.distributed_failure_manifest.resolve()
        distributed_failure = load_json(failure_path)
        if (
            distributed_failure.get("status") != "failed"
            or int(distributed_failure.get("exit_code", -1)) != 1
            or distributed_failure.get("variant") != "matched"
            or distributed_failure.get("common", {}).get("alignment_mode")
            != args.alignment_mode
            or Path(
                str(distributed_failure.get("common", {}).get("train_cache"))
            ).resolve()
            != train_cache
        ):
            raise ValueError("Distributed failure manifest does not match this stage")
    elif args.distributed_failure_manifest is not None:
        raise ValueError(
            "--distributed-failure-manifest is only valid for single_device_fallback"
        )

    effective_global_batch = (
        args.batch_size_per_gpu
        * len(args.gpu_ids)
        * args.gradient_accumulation_steps
    )
    common = {
        "protocol": {
            "hidden_head": "counterfactual-mask-stage1-v1",
            "clean_latent_probe": "counterfactual-clean-latent-stage1-v1",
            "shared_mask_video_flow": (
                "counterfactual-shared-mask-video-flow-stage1-v1"
            ),
        }[args.alignment_mode],
        "alignment_mode": args.alignment_mode,
        "project_root": str(project_root),
        "preflight_report": str(report_path),
        "preflight_report_sha256": sha256_file(report_path),
        "git_commit": current_commit,
        "checkpoint": str(checkpoint),
        "train_cache": str(train_cache),
        "heldout_cache": str(heldout_cache),
        "dataset_root": str(dataset_root),
        "wan_pretrained": str(wan_pretrained),
        "norm_stats": str(norm_stats),
        "gpu_ids": args.gpu_ids,
        "steps": args.steps,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch": effective_global_batch,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "mask_weight_when_enabled": args.mask_weight,
        "log_interval": args.log_interval,
        "python": sys.executable,
        "required_env_name": args.required_env_name,
        "execution_profile": args.execution_profile,
    }
    if distributed_failure is not None:
        common["distributed_failure"] = {
            "manifest": str(args.distributed_failure_manifest.resolve()),
            "manifest_sha256": sha256_file(args.distributed_failure_manifest.resolve()),
            "comparison_id": distributed_failure.get("comparison_id"),
            "exit_code": distributed_failure.get("exit_code"),
        }
    if clean_latent_gates is not None:
        common["clean_latent_gates"] = clean_latent_gates
    if shared_mask_video_gate is not None:
        common["shared_mask_video_gate"] = shared_mask_video_gate
    comparison_id = sha256_json(common)

    env_overrides = {
        "GWP_LIBERO_DATA_PATH": str(dataset_root),
        "GWP_NORM_STATS_PATH": str(norm_stats),
        "GWP_WAN_PRETRAINED": str(wan_pretrained),
        "GWP05_TRANSFORMER_PRETRAINED": str(checkpoint),
        "GWP05_PROJECT_DIR": str(output_dir),
        "GWP_T5_EMBEDDING_DIR": str(t5_embedding_dir),
        "GWP_LIBERO_DATA_PATHS": "",
        "GWP_VIEW_KEYS": "observation.images.image,observation.images.wrist_image",
        "GWP_VIEW_CONCAT_MODE": "horizontal",
        "GWP_GPU_IDS": ",".join(str(gpu_id) for gpu_id in args.gpu_ids),
        "GWP_BATCH_SIZE_PER_GPU": str(args.batch_size_per_gpu),
        "GWP_NUM_WORKERS": str(args.num_workers),
        "GWP_MAX_STEPS": str(args.steps),
        "GWP_GRAD_ACCUM_STEPS": str(args.gradient_accumulation_steps),
        "GWP_LOG_INTERVAL": str(args.log_interval),
        "GWP_WITH_EMA": "0",
        "GWP_RESUME": "0",
        "GWP_DISABLE_CHECKPOINT": "0",
        "GWP_CHECKPOINT_INTERVAL": str(args.steps),
        "GWP_CHECKPOINT_TOTAL_LIMIT": "1",
        "GWP_CHECKPOINT_SAVE_OPTIMIZER": "1",
        "GWP_CHECKPOINT_MODEL_DTYPE": "bf16",
        "GWP_STRIP_OPTIMIZER_AFTER_CHECKPOINT": "1",
        "GWP_CONSEQUENCE_MODE": "off",
        "GWP_COUNTERFACTUAL_MASK_CACHE_PATHS": str(train_cache),
        "GWP_COUNTERFACTUAL_MASK_INTERVAL": "1",
        "GWP_COUNTERFACTUAL_MASK_START_STEP": "0",
        "GWP_COUNTERFACTUAL_MASK_TRAINABLE_SCOPE": "all",
        "GWP_COUNTERFACTUAL_MASK_ALIGNMENT_MODE": args.alignment_mode,
        "GWP_COUNTERFACTUAL_MASK_PROBE_PRETRAINED": (
            clean_latent_gates["probe_root"]
            if clean_latent_gates is not None
            else ""
        ),
        "GWP_COUNTERFACTUAL_MASK_BCE_WEIGHT": "1.0",
        "GWP_COUNTERFACTUAL_MASK_DICE_WEIGHT": "1.0",
        "GWP_SEED": str(args.seed),
        "GWP_VISUAL_LOSS_WEIGHT": "1.0",
        # Preserve the official 30k baseline objective; the mask term is the
        # only intended scientific difference between matched and full.
        "GWP_ACTION_LOSS_WEIGHT": "5.0",
        "GWP_COUNTERFACTUAL_MASK_WEIGHT": (
            "0.0" if args.variant == "matched" else str(args.mask_weight)
        ),
        "GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL": (
            "1" if args.variant == "matched" else "0"
        ),
    }
    command = [
        sys.executable,
        str(project_root / "scripts" / "train.py"),
        "--config",
        "giga_world_policy_0_5_libero_finetune",
    ]
    record = {
        "status": "validated" if args.dry_run else "launching",
        "variant": args.variant,
        "comparison_id": comparison_id,
        "common": common,
        "variant_settings": {
            "alignment_mode": args.alignment_mode,
            "mask_probe": (
                clean_latent_gates["probe_root"]
                if clean_latent_gates is not None
                else None
            ),
            "mask_video_vae_report": (
                shared_mask_video_gate["report"]
                if shared_mask_video_gate is not None
                else None
            ),
            "counterfactual_mask_weight": float(
                env_overrides["GWP_COUNTERFACTUAL_MASK_WEIGHT"]
            ),
            "counterfactual_mask_compute_control": (
                env_overrides["GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL"] == "1"
            ),
        },
        "output_dir": str(output_dir),
        "counterpart_output_dir": str(counterpart_dir),
        "python": sys.executable,
        "command": command,
        "environment": env_overrides,
        "free_gib_at_validation": current_free_gib,
        "required_free_gib_at_validation": required_free_gib,
    }

    if counterpart_manifest.is_file():
        counterpart = load_json(counterpart_manifest)
        if counterpart.get("variant") == args.variant:
            raise ValueError("Counterpart manifest has the same variant")
        if counterpart.get("comparison_id") != comparison_id:
            raise ValueError(
                "Counterpart launch used different common settings; refusing an "
                "unmatched comparison"
            )
        if counterpart.get("status") != "complete" or int(
            counterpart.get("exit_code", -1)
        ) != 0:
            raise RuntimeError(
                "Counterpart launch has not completed successfully; refusing to "
                "start the second arm"
            )
    return record, env_overrides


def main() -> None:
    args = parse_args()
    if not args.dry_run and args.required_env_name:
        active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        prefix_name = Path(sys.prefix).name
        if args.required_env_name not in {active_env, prefix_name}:
            raise RuntimeError(
                f"Launch requires conda env {args.required_env_name!r}; "
                f"active={active_env!r}, prefix={prefix_name!r}"
            )

    record, env_overrides = build_launch_record(args)
    print(json.dumps(record, indent=2), flush=True)
    if args.dry_run:
        return

    output_dir = Path(record["output_dir"])
    output_dir.mkdir(parents=False, exist_ok=False)
    manifest_path = output_dir / "stage1_launch_manifest.json"
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    for inherited_name in (
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(inherited_name, None)
    environment.update(env_overrides)
    exit_code = 1
    try:
        completed = subprocess.run(
            record["command"],
            cwd=Path(record["common"]["project_root"]),
            env=environment,
            check=False,
        )
        exit_code = int(completed.returncode)
    finally:
        record["status"] = "complete" if exit_code == 0 else "failed"
        record["exit_code"] = exit_code
        manifest_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "stage1_exit_code.txt").write_text(
            f"{exit_code}\n", encoding="utf-8"
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

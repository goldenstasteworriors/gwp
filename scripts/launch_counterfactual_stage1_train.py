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
EXPECTED_CACHE_SCHEMA = "libero-counterfactual-robot-mask-v3"
EXPECTED_IMAGE_ORIENTATION = "lerobot_libero_visualization_flip_height_width"
EXPECTED_TRAIN_EPISODES = 200
EXPECTED_HELDOUT_EPISODES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--wan-pretrained", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mask-weight", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--required-env-name", default="gigaworld-policy")
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
    if manifest.get("schema_version") != EXPECTED_CACHE_SCHEMA:
        raise ValueError(f"Unexpected cache schema: {manifest_path}")
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
    validate_cache_manifest(
        heldout_cache,
        report["heldout_cache"],
        EXPECTED_HELDOUT_EPISODES,
        checkpoint,
    )
    dataset_root = Path(str(train_manifest["dataset_root"])).resolve()

    t5_embedding_dir = dataset_root / "t5_embedding"
    if not t5_embedding_dir.is_dir():
        raise FileNotFoundError(t5_embedding_dir)

    output_device = os.stat(output_dir.parent).st_dev
    for source in (
        checkpoint,
        train_cache,
        heldout_cache,
        dataset_root,
        wan_pretrained,
        norm_stats,
    ):
        if os.stat(source).st_dev != output_device:
            raise ValueError(f"Large stage-one source/output crosses filesystems: {source}")

    latest_mtime = latest_input_mtime_ns(
        [checkpoint, train_cache, heldout_cache, split_manifest]
    )
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
        "gpu_ids": [2, 3, 4, 5],
        "steps": 200,
        "batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
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

    effective_global_batch = (
        args.batch_size_per_gpu
        * len(args.gpu_ids)
        * args.gradient_accumulation_steps
    )
    common = {
        "protocol": "counterfactual-mask-stage1-v1",
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
    }
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

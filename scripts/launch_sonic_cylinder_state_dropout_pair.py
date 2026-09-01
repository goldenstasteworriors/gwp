#!/usr/bin/env python3
"""Launch matched 0.5 state-condition-dropout runs for the fixed-thumb baselines."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_EPISODES = "0-69,73-115"
EVALUATION_EPISODES = [70, 71, 72, 116, 117, 118]
PROMPT = "grab graduated cylinder and pour water into beaker"


@dataclass(frozen=True)
class RunSpec:
    name: str
    description: str
    gpu_id: int
    delta_mask: tuple[bool, ...]
    norm_stats_name: str


RUNS = (
    RunSpec(
        name="m01_state_dropout05_all_absolute_gpu1",
        description=(
            "Fixed-thumb all-absolute baseline with sample-level proprioceptive "
            "condition dropout."
        ),
        gpu_id=1,
        delta_mask=(False,) * 13,
        norm_stats_name="norm_stats_all_absolute_h48_train_0-69_73-115.json",
    ),
    RunSpec(
        name="m01_state_dropout05_armdelta_handabs_gpu3",
        description=(
            "Fixed-thumb arm-delta/hand-absolute baseline with sample-level "
            "proprioceptive condition dropout."
        ),
        gpu_id=3,
        delta_mask=(True,) * 7 + (False,) * 6,
        norm_stats_name="norm_stats_armdelta_handabs_h48_train_0-69_73-115.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dropout-prob", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--checkpoint-total-limit", type=int, default=1)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=50.0,
        help="Refuse to launch both full-model runs below this free-space guard.",
    )
    parser.add_argument("--startup-wait-seconds", type=float, default=5.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the dated run plan without launching training.",
    )
    return parser.parse_args()


def bool_spec(values: tuple[bool, ...]) -> str:
    return ",".join("true" if value else "false" for value in values)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dated_name(spec: RunSpec, run_date: str) -> str:
    return f"{run_date}_{spec.name}"


def run_environment(
    args: argparse.Namespace,
    spec: RunSpec,
    output_dir: Path,
    run_name: str,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "WANDB_NAME": run_name,
            "GWP_SONIC_DATA_PATH": str(args.dataset),
            "GWP_NORM_STATS_PATH": str(args.dataset / spec.norm_stats_name),
            "GWP05_TRANSFORMER_PRETRAINED": str(
                PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5"
            ),
            "GWP_WAN_PRETRAINED": str(
                PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers"
            ),
            "GWP05_PROJECT_DIR": str(output_dir),
            "GWP_GPU_IDS": str(spec.gpu_id),
            "GWP_PHYSICAL_ACTION_DIM": "13",
            "GWP_ACTION_DELTA_MASK": bool_spec(spec.delta_mask),
            "GWP_STATE_CONDITION_DROPOUT_PROB": str(args.state_dropout_prob),
            "GWP_IMAGE_AUGMENTATION": "1",
            "GWP_BATCH_SIZE_PER_GPU": "1",
            "GWP_NUM_WORKERS": "8",
            "GWP_TRAIN_EPISODES": TRAIN_EPISODES,
            "GWP_MAX_STEPS": str(args.max_steps),
            "GWP_GRAD_ACCUM_STEPS": "4",
            "GWP_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
            "GWP_CHECKPOINT_TOTAL_LIMIT": str(args.checkpoint_total_limit),
            "GWP_CHECKPOINT_WEIGHTS_ONLY": "1",
            "GWP_CHECKPOINT_EXPORT_DTYPE": "bf16",
            "GWP_LOG_INTERVAL": "1",
            "GWP_WITH_EMA": "0",
            "GWP_ACTIVATION_CHECKPOINTING": "1",
            "GWP_DEEPSPEED_CONFIG": "accelerate_configs/zero3.json",
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
        }
    )
    return env


def validate(args: argparse.Namespace, run_date: str) -> None:
    if not 0.0 <= args.state_dropout_prob <= 1.0:
        raise ValueError("--state-dropout-prob must be in [0, 1]")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    for spec in RUNS:
        norm_path = args.dataset / spec.norm_stats_name
        if not norm_path.is_file():
            raise FileNotFoundError(norm_path)
    for required in (
        PROJECT_ROOT / ".venv-ppu" / "bin" / "python",
        PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5",
        PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    planned_dirs = [args.output_root / dated_name(spec, run_date) for spec in RUNS]
    for output_dir in planned_dirs:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    free_gib = shutil.disk_usage(args.output_root.parent).free / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            "Insufficient free space for two checkpoint streams: "
            f"free={free_gib:.2f} GiB required={args.min_free_gib:.2f} GiB"
        )


def plan(args: argparse.Namespace, run_date: str) -> list[dict[str, object]]:
    return [
        {
            "name": dated_name(spec, run_date),
            "gpu_id": spec.gpu_id,
            "output_dir": str(args.output_root / dated_name(spec, run_date)),
            "state_condition_dropout_prob": args.state_dropout_prob,
            "action_delta_mask": list(spec.delta_mask),
        }
        for spec in RUNS
    ]


def start_run(
    args: argparse.Namespace,
    spec: RunSpec,
    run_date: str,
) -> dict[str, object]:
    run_name = dated_name(spec, run_date)
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = run_environment(args, spec, output_dir, run_name)
    experiment = {
        "name": run_name,
        "description": spec.description,
        "dataset": str(args.dataset),
        "norm_stats": str(args.dataset / spec.norm_stats_name),
        "prompt": PROMPT,
        "gpu_ids": str(spec.gpu_id),
        "action_delta_mask": list(spec.delta_mask),
        "state_condition_dropout_prob": args.state_dropout_prob,
        "training_episode_spec": TRAIN_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "max_steps": args.max_steps,
        "batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_total_limit": args.checkpoint_total_limit,
        "checkpoint_export_dtype": "bf16",
        "transformer_pretrained": str(PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output_dir / "experiment.json", experiment)
    env_record = {
        key: environment[key]
        for key in sorted(environment)
        if key.startswith("GWP_") or key.startswith("GWP05_") or key == "WANDB_NAME"
    }
    write_json(output_dir / "launch_environment.json", env_record)

    log_handle = (output_dir / "launcher.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(PROJECT_ROOT / ".venv-ppu" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            "giga_world_policy_0_5_sonic_g1_finetune",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (output_dir / "launcher.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "name": run_name,
        "gpu_id": spec.gpu_id,
        "pid": process.pid,
        "process": process,
    }


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    run_date = time.strftime("%Y%m%d")
    validate(args, run_date)
    if args.dry_run:
        print(json.dumps(plan(args, run_date), indent=2, ensure_ascii=False))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = []
    try:
        for spec in RUNS:
            started.append(start_run(args, spec, run_date))
        time.sleep(args.startup_wait_seconds)
        failures = []
        for item in started:
            return_code = item["process"].poll()
            if return_code is not None:
                failures.append((item["name"], item["pid"], return_code))
        if failures:
            raise RuntimeError(f"Training exited during startup: {failures}")
    except BaseException:
        for item in started:
            process = item["process"]
            if process.poll() is None:
                os.killpg(process.pid, 15)
        raise

    print(
        json.dumps(
            [
                {"name": item["name"], "gpu_id": item["gpu_id"], "pid": item["pid"]}
                for item in started
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

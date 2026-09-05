#!/usr/bin/env python3
"""Launch the four-way SONIC RoboTTT-lite memory-activation diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from launch_sonic_robottt_lite_stage1 import (
    ACTION_DELTA_MASK,
    CODE_ROOT,
    EVALUATION_EPISODES,
    NORM_STATS_NAME,
    RUNTIME_ROOT,
    TRAIN_EPISODES,
    configure_ppu_environment,
    validate_dataset,
    write_json,
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    gpu_id: int
    persistent: bool
    memory_only: bool
    current_image_dropout_prob: float
    description: str


RUNS = (
    RunSpec(
        "m10_nomemory_joint_control_c32",
        0,
        False,
        False,
        0.0,
        "Action head plus reset-every-replan memory as a compute-matched control.",
    ),
    RunSpec(
        "m11_memory_only_c32_gate001_tbptt16",
        1,
        True,
        True,
        0.0,
        "Persistent C32 memory-only tuning with a stronger gate and TBPTT horizon.",
    ),
    RunSpec(
        "m12_joint_c32_gate001_tbptt16",
        2,
        True,
        False,
        0.0,
        "Persistent C32 joint action-head and memory tuning.",
    ),
    RunSpec(
        "m13_joint_c32_gate001_tbptt16_imgdrop015",
        3,
        True,
        False,
        0.15,
        "Joint C32 tuning with supervised-tail current-image dropout.",
    ),
)

CONTEXT_LENGTH = 32
TBPTT_STEPS = 16
MEMORY_GATE_INIT = 0.01
MEMORY_OUTER_LR = 6e-5
ACTION_HEAD_LR = 5e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint-total-limit", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    parser.add_argument("--startup-wait-seconds", type=float, default=60.0)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_name(spec: RunSpec, suffix: str) -> str:
    training_date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    value = f"{training_date}_{spec.name}_stageA_gpu{spec.gpu_id}"
    return f"{value}_{suffix}" if suffix else value


def validate(args: argparse.Namespace) -> None:
    if args.max_steps < 1 or args.checkpoint_interval < 1:
        raise ValueError("Training and checkpoint intervals must be positive")
    if args.checkpoint_total_limit < 1:
        raise ValueError("checkpoint-total-limit must be positive")
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_suffix):
        raise ValueError("run suffix contains unsupported characters")
    required = (
        args.dataset,
        args.checkpoint,
        args.dataset / NORM_STATS_NAME,
        RUNTIME_ROOT / ".venv-ppu" / "bin" / "python",
        RUNTIME_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    validate_dataset(args.dataset)
    for spec in RUNS:
        output_dir = args.output_root / run_name(spec, args.run_suffix)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    free_gib = shutil.disk_usage(args.output_root.parent).free / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"Insufficient free space: {free_gib:.2f} GiB < {args.min_free_gib:.2f} GiB"
        )


def experiment_record(args: argparse.Namespace, spec: RunSpec) -> dict[str, object]:
    name = run_name(spec, args.run_suffix)
    return {
        "name": name,
        "stage": "stageA_memory_activation_diagnostic",
        "description": spec.description,
        "gpu_id": spec.gpu_id,
        "dataset": str(args.dataset),
        "norm_stats": str(args.dataset / NORM_STATS_NAME),
        "transformer_pretrained": str(args.checkpoint),
        "output_dir": str(args.output_root / name),
        "context_length_replans": CONTEXT_LENGTH,
        "context_seconds": CONTEXT_LENGTH * 8 / 50.0,
        "replan_stride_raw_frames": 8,
        "memory_persistent": spec.persistent,
        "train_memory_only": spec.memory_only,
        "train_action_head_and_memory": not spec.memory_only,
        "memory_layers": [7, 15, 23, 29],
        "memory_gate_init": MEMORY_GATE_INIT,
        "memory_inner_lr": 0.01,
        "memory_outer_lr": MEMORY_OUTER_LR,
        "action_head_lr": None if spec.memory_only else ACTION_HEAD_LR,
        "memory_context_half_life": CONTEXT_LENGTH,
        "tbptt_supervised_tail_steps": TBPTT_STEPS,
        "current_image_dropout_prob": spec.current_image_dropout_prob,
        "training_episode_spec": TRAIN_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "action_semantics": {
            "arm_joints_0_to_6": "relative_joint_delta",
            "hand_joints_7_to_12": "absolute_joint_position",
            "thumb_bend_input_and_target": 1.0,
            "thumb_rotate_input_and_target": 0.0,
        },
        "max_steps": args.max_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_total_limit": args.checkpoint_total_limit,
        "checkpoint_export": "bf16_action_head_plus_memory",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
        ).strip(),
    }


def build_environment(
    args: argparse.Namespace,
    spec: RunSpec,
    output_dir: Path,
    name: str,
) -> dict[str, str]:
    env = dict(os.environ)
    configure_ppu_environment(env)
    mask = ",".join("true" if value else "false" for value in ACTION_DELTA_MASK)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "WANDB_NAME": name,
            "WANDB_PROJECT": "sonic-cylinder-robottt-lite-diagnostic",
            "WANDB_DIR": str(output_dir / "wandb"),
            "GWP_LOG_WITH": "wandb",
            "GWP_SONIC_DATA_PATH": str(args.dataset),
            "GWP_NORM_STATS_PATH": str(args.dataset / NORM_STATS_NAME),
            "GWP_T5_LOAD_FROM": "dir",
            "GWP_T5_EMBEDDING_DIR": str(args.dataset / "t5_embedding"),
            "GWP05_TRANSFORMER_PRETRAINED": str(args.checkpoint),
            "GWP_WAN_PRETRAINED": str(
                RUNTIME_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers"
            ),
            "GWP05_PROJECT_DIR": str(output_dir),
            "GWP_GPU_IDS": str(spec.gpu_id),
            "GWP_PHYSICAL_ACTION_DIM": "13",
            "GWP_EXPECTED_STATE_DIM": "13",
            "GWP_ACTION_DELTA_MASK": mask,
            "GWP_TRAIN_EPISODES": TRAIN_EPISODES,
            "GWP_ACTION_HORIZON": "48",
            "GWP_MEMORY_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
            "GWP_MEMORY_CONTEXT_HALF_LIFE": str(CONTEXT_LENGTH),
            "GWP_MEMORY_REPLAN_STRIDE": "8",
            "GWP_MEMORY_LAYER_INDICES": "7,15,23,29",
            "GWP_MEMORY_DIM": "256",
            "GWP_MEMORY_HIDDEN_DIM": "256",
            "GWP_MEMORY_INNER_LR": "0.01",
            "GWP_MEMORY_GATE_INIT": str(MEMORY_GATE_INIT),
            "GWP_MEMORY_TBPTT_STEPS": str(TBPTT_STEPS),
            "GWP_MEMORY_PERSISTENT": "1" if spec.persistent else "0",
            "GWP_TRAIN_MEMORY_ONLY": "1" if spec.memory_only else "0",
            "GWP_TRAIN_ACTION_HEAD_ONLY": "0" if spec.memory_only else "1",
            "GWP_LEARNING_RATE": str(MEMORY_OUTER_LR),
            "GWP_MEMORY_LEARNING_RATE": str(MEMORY_OUTER_LR),
            "GWP_ACTION_HEAD_LEARNING_RATE": (
                "" if spec.memory_only else str(ACTION_HEAD_LR)
            ),
            "GWP_MEMORY_CURRENT_IMAGE_DROPOUT_PROB": str(
                spec.current_image_dropout_prob
            ),
            "GWP_VISUAL_LOSS_WEIGHT": "0",
            "GWP_IMAGE_AUGMENTATION": "1",
            "GWP_MAX_REF_FRAMES": "1",
            "GWP_MIN_REF_LATENTS": "1",
            "GWP_BATCH_SIZE_PER_GPU": "1",
            "GWP_GRAD_ACCUM_STEPS": "1",
            "GWP_NUM_WORKERS": "4",
            "GWP_MAX_STEPS": str(args.max_steps),
            "GWP_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
            "GWP_CHECKPOINT_TOTAL_LIMIT": str(args.checkpoint_total_limit),
            "GWP_CHECKPOINT_WEIGHTS_ONLY": "1",
            "GWP_CHECKPOINT_EXPORT_MODE": "action_head_only",
            "GWP_CHECKPOINT_EXPORT_DTYPE": "bf16",
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_WITH_EMA": "0",
            "GWP_MODEL_GRADIENT_CHECKPOINTING": "0",
            "GWP_ACTIVATION_CHECKPOINTING": "0",
            "GWP_DEEPSPEED_CONFIG": "accelerate_configs/zero2.json",
            "GWP_LOG_INTERVAL": "1",
            "HF_DATASETS_CACHE": str(RUNTIME_ROOT / ".cache" / "huggingface"),
        }
    )
    return env


def start_run(args: argparse.Namespace, spec: RunSpec) -> dict[str, object]:
    record = experiment_record(args, spec)
    output_dir = Path(record["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    record["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(output_dir / "experiment.json", record)
    env = build_environment(args, spec, output_dir, str(record["name"]))
    write_json(
        output_dir / "launch_environment.json",
        {
            key: env[key]
            for key in sorted(env)
            if key.startswith(("GWP_", "GWP05_", "WANDB_"))
        },
    )
    log_handle = (output_dir / "launcher.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(RUNTIME_ROOT / ".venv-ppu" / "bin" / "python"),
            str(CODE_ROOT / "scripts" / "train.py"),
            "--config",
            "giga_world_policy_0_5_sonic_g1_finetune",
        ],
        cwd=CODE_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (output_dir / "launcher.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return {**record, "pid": process.pid, "process": process}


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate(args)
    plan = [experiment_record(args, spec) for spec in RUNS]
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = []
    try:
        for spec in RUNS:
            started.append(start_run(args, spec))
        time.sleep(args.startup_wait_seconds)
        failures = [
            (item["name"], item["pid"], item["process"].poll())
            for item in started
            if item["process"].poll() not in (None, 0)
        ]
        if failures:
            raise RuntimeError(f"Training exited during startup: {failures}")
    except BaseException:
        for item in started:
            process = item["process"]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise
    print(
        json.dumps(
            [
                {
                    "name": item["name"],
                    "gpu_id": item["gpu_id"],
                    "pid": item["pid"],
                    "output_dir": item["output_dir"],
                }
                for item in started
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

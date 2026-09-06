#!/usr/bin/env python3
"""Launch one sequential SONIC RoboTTT curriculum stage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
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


STAGES = {
    "stage2_c32": {
        "context_length": 32,
        "max_steps": 2000,
        "description": "All-segment TBPTT warm transition at the validated C32 context.",
    },
    "stage2_c64": {
        "context_length": 64,
        "max_steps": 3000,
        "description": "All-segment TBPTT curriculum at C64 context.",
    },
    "stage2_full": {
        "context_length": 168,
        "max_steps": 3000,
        "description": "Episode-scale all-segment TBPTT at the dataset maximum of 168 replans.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overlay-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-total-limit", type=int, default=2)
    parser.add_argument("--tbptt-steps", type=int, default=16)
    parser.add_argument("--memory-learning-rate", type=float, default=3e-5)
    parser.add_argument("--action-learning-rate", type=float, default=2e-6)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    parser.add_argument("--startup-wait-seconds", type=float, default=60.0)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parsed_gpu_ids(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("gpu ids must be a non-empty unique comma-separated list")
    return values


def run_name(args: argparse.Namespace) -> str:
    training_date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    context_length = STAGES[args.stage]["context_length"]
    value = f"{training_date}_{args.stage}_c{context_length}_allseg_tbptt16_actionio"
    return f"{value}_{args.run_suffix}" if args.run_suffix else value


def validate(args: argparse.Namespace) -> None:
    parsed_gpu_ids(args.gpu_ids)
    max_steps = args.max_steps or int(STAGES[args.stage]["max_steps"])
    if max_steps < 1 or args.tbptt_steps < 1:
        raise ValueError("max steps and TBPTT steps must be positive")
    if args.checkpoint_interval < 1 or args.checkpoint_total_limit < 1:
        raise ValueError("checkpoint settings must be positive")
    if args.memory_learning_rate <= 0 or args.action_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_suffix):
        raise ValueError("run suffix contains unsupported characters")
    required = (
        args.dataset,
        args.checkpoint,
        args.overlay_checkpoint,
        args.dataset / NORM_STATS_NAME,
        RUNTIME_ROOT / ".venv-ppu" / "bin" / "python",
        RUNTIME_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    validate_dataset(args.dataset)
    output_dir = args.output_root / run_name(args)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    free_gib = shutil.disk_usage(args.output_root.parent).free / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"Insufficient free space: {free_gib:.2f} GiB < {args.min_free_gib:.2f} GiB"
        )


def experiment_record(args: argparse.Namespace) -> dict[str, object]:
    context_length = int(STAGES[args.stage]["context_length"])
    max_steps = args.max_steps or int(STAGES[args.stage]["max_steps"])
    return {
        "name": run_name(args),
        "stage": args.stage,
        "description": STAGES[args.stage]["description"],
        "dataset": str(args.dataset),
        "norm_stats": str(args.dataset / NORM_STATS_NAME),
        "transformer_pretrained": str(args.checkpoint),
        "transformer_overlay_pretrained": str(args.overlay_checkpoint),
        "output_dir": str(args.output_root / run_name(args)),
        "gpu_ids": parsed_gpu_ids(args.gpu_ids),
        "context_length_replans": context_length,
        "context_seconds": context_length * 8 / 50.0,
        "replan_stride_raw_frames": 8,
        "memory_persistent": True,
        "memory_history_residual_only": True,
        "memory_context_half_life": context_length,
        "tbptt_segment_steps": args.tbptt_steps,
        "supervise_all_tbptt_segments": True,
        "sequence_action_forcing": "independent flow timestep per action chunk",
        "trainable_profile": "memory_action_io",
        "memory_learning_rate": args.memory_learning_rate,
        "action_learning_rate": args.action_learning_rate,
        "training_episode_spec": TRAIN_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "action_semantics": {
            "arm_joints_0_to_6": "relative_joint_delta",
            "hand_joints_7_to_12": "absolute_joint_position",
            "thumb_bend_input_and_target": 1.0,
            "thumb_rotate_input_and_target": 0.0,
        },
        "max_steps": max_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_total_limit": args.checkpoint_total_limit,
        "checkpoint_export": "bf16_action_head_plus_memory",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
        ).strip(),
    }


def build_environment(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    record = experiment_record(args)
    env = dict(os.environ)
    configure_ppu_environment(env)
    mask = ",".join("true" if value else "false" for value in ACTION_DELTA_MASK)
    runtime_dir = args.output_root.parent / ".robottt_runtime" / args.stage
    context_length = int(record["context_length_replans"])
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "WANDB_NAME": str(record["name"]),
            "WANDB_PROJECT": "sonic-cylinder-robottt-curriculum",
            "WANDB_DIR": str(output_dir / "wandb"),
            "GWP_LOG_WITH": "wandb",
            "GWP_SONIC_DATA_PATH": str(args.dataset),
            "GWP_NORM_STATS_PATH": str(args.dataset / NORM_STATS_NAME),
            "GWP_T5_LOAD_FROM": "dir",
            "GWP_T5_EMBEDDING_DIR": str(args.dataset / "t5_embedding"),
            "GWP05_TRANSFORMER_PRETRAINED": str(args.checkpoint),
            "GWP05_TRANSFORMER_OVERLAY_PRETRAINED": str(args.overlay_checkpoint),
            "GWP_WAN_PRETRAINED": str(
                RUNTIME_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers"
            ),
            "GWP05_PROJECT_DIR": str(output_dir),
            "GWP_GPU_IDS": args.gpu_ids,
            "GWP_PHYSICAL_ACTION_DIM": "13",
            "GWP_EXPECTED_STATE_DIM": "13",
            "GWP_ACTION_DELTA_MASK": mask,
            "GWP_TRAIN_EPISODES": TRAIN_EPISODES,
            "GWP_ACTION_HORIZON": "48",
            "GWP_MEMORY_CONTEXT_LENGTH": str(context_length),
            "GWP_MEMORY_CONTEXT_HALF_LIFE": str(context_length),
            "GWP_MEMORY_REPLAN_STRIDE": "8",
            "GWP_MEMORY_LAYER_INDICES": "7,15,23,29",
            "GWP_MEMORY_DIM": "256",
            "GWP_MEMORY_HIDDEN_DIM": "256",
            "GWP_MEMORY_INNER_LR": "0.05",
            "GWP_MEMORY_GATE_INIT": "0.05",
            "GWP_MEMORY_TBPTT_STEPS": str(args.tbptt_steps),
            "GWP_MEMORY_SUPERVISE_ALL_SEGMENTS": "1",
            "GWP_MEMORY_PERSISTENT": "1",
            "GWP_MEMORY_HISTORY_RESIDUAL_ONLY": "1",
            "GWP_TRAIN_MEMORY_ONLY": "0",
            "GWP_TRAIN_ACTION_HEAD_ONLY": "1",
            "GWP_TRAINABLE_PROFILE": "memory_action_io",
            "GWP_LEARNING_RATE": str(args.memory_learning_rate),
            "GWP_MEMORY_LEARNING_RATE": str(args.memory_learning_rate),
            "GWP_ACTION_HEAD_LEARNING_RATE": str(args.action_learning_rate),
            "GWP_MEMORY_CURRENT_IMAGE_DROPOUT_PROB": "0",
            "GWP_STATE_HISTORY_DROPOUT_PROB": "0",
            "GWP_VISUAL_LOSS_WEIGHT": "0",
            "GWP_IMAGE_AUGMENTATION": "1",
            "GWP_MAX_REF_FRAMES": "1",
            "GWP_MIN_REF_LATENTS": "1",
            "GWP_BATCH_SIZE_PER_GPU": "1",
            "GWP_GRAD_ACCUM_STEPS": "1",
            "GWP_NUM_WORKERS": "2" if context_length > 64 else "4",
            "GWP_MAX_STEPS": str(record["max_steps"]),
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
            "TMPDIR": str(runtime_dir),
        }
    )
    return env


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.overlay_checkpoint = args.overlay_checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    validate(args)
    record = experiment_record(args)
    output_dir = Path(record["output_dir"])
    if args.dry_run:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    record["started_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    write_json(output_dir / "experiment.json", record)
    env = build_environment(args, output_dir)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "launch_environment.json",
        {
            key: env[key]
            for key in sorted(env)
            if key.startswith(("GWP_", "GWP05_", "WANDB_")) or key == "TMPDIR"
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
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_json(
        output_dir / "launcher_pid.json",
        {"pid": process.pid, "started_at": record["started_at"]},
    )
    time.sleep(args.startup_wait_seconds)
    return_code = process.poll()
    if return_code not in (None, 0):
        raise RuntimeError(
            f"Training exited during startup with return code {return_code}; "
            f"see {output_dir / 'launcher.log'}"
        )
    print(json.dumps({"pid": process.pid, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

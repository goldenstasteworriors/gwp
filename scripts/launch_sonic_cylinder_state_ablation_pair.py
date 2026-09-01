#!/usr/bin/env python3
"""Launch matched SONIC state-dropout and compact-history experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_EPISODES = "0-69,73-115"
EVALUATION_EPISODES = [70, 71, 72, 116, 117, 118]
PROMPT = "grab graduated cylinder and pour water into beaker"
ACTION_DELTA_MASK = (True,) * 7 + (False,) * 6
NORM_STATS_NAME = "norm_stats_armdelta_handabs_h48_train_0-69_73-115.json"
COMPACT_STATE_HISTORY_OFFSETS = tuple(range(-7, 1))
EXPECTED_FIXED_JOINTS = {
    "left_hand_thumb_bend": 1.0,
    "left_hand_thumb_rotate": 0.0,
}
PPU_RUNTIME_ENV_KEYS = {
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_TOOLKIT_ROOT",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PPU_SDK",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    description: str
    state_condition_dropout_prob: float
    state_history_offsets: tuple[int, ...]


RUNS = (
    RunSpec(
        name="m02_state_dropout05_armdelta_handabs",
        description=(
            "Fixed-thumb arm-delta/hand-absolute baseline with sample-level "
            "proprioceptive condition dropout 0.5."
        ),
        state_condition_dropout_prob=0.5,
        state_history_offsets=(0,),
    ),
    RunSpec(
        name="m03_state_history8_compact_armdelta_handabs",
        description=(
            "Fixed-thumb arm-delta/hand-absolute baseline with eight consecutive "
            "state frames spanning the most recent 140 ms at 50 Hz."
        ),
        state_condition_dropout_prob=0.0,
        state_history_offsets=COMPACT_STATE_HISTORY_OFFSETS,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gpu-ids",
        default="1,3",
        help="Two distinct GPU ids in dropout,history order.",
    )
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--checkpoint-total-limit", type=int, default=1)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=50.0,
        help="Refuse to launch both full-model runs below this free-space guard.",
    )
    parser.add_argument("--startup-wait-seconds", type=float, default=20.0)
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional safe suffix for a non-overwriting retry, for example retry1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print the dated run plan without training.",
    )
    return parser.parse_args()


def bool_spec(values: tuple[bool, ...]) -> str:
    return ",".join("true" if value else "false" for value in values)


def int_spec(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def parse_gpu_ids(raw: str) -> tuple[int, int]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != len(RUNS):
        raise ValueError(f"--gpu-ids must contain exactly {len(RUNS)} ids")
    if len(set(values)) != len(values) or min(values) < 0:
        raise ValueError("--gpu-ids must contain distinct non-negative ids")
    return values


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dated_name(spec: RunSpec, run_date: str, run_suffix: str, gpu_id: int) -> str:
    base = f"{run_date}_{spec.name}_gpu{gpu_id}"
    return f"{base}_{run_suffix}" if run_suffix else base


def prepend_env_path(env: dict[str, str], key: str, *paths: Path) -> None:
    values = [str(path) for path in paths]
    current = env.get(key, "")
    if current:
        values.append(current)
    env[key] = os.pathsep.join(values)


def configure_ppu_sdk_environment(env: dict[str, str]) -> None:
    sdk_root = Path(env.get("PPU_SDK", "/usr/local/PPU_SDK")).resolve()
    cuda_root = sdk_root / "CUDA_SDK"
    env.update(
        {
            "PPU_SDK": str(sdk_root),
            "CUDA_PATH": str(cuda_root),
            "CUDA_HOME": str(cuda_root),
            "CUDA_TOOLKIT_ROOT": str(cuda_root),
        }
    )
    prepend_env_path(
        env,
        "PATH",
        cuda_root / "bin",
        sdk_root / "asight" / "bin",
        sdk_root / "bin",
        sdk_root / "ppu-smi" / "bin",
    )
    prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        Path("/opt/accl-p"),
        cuda_root / "lib64",
        sdk_root / "lib",
        sdk_root / "sailSHMEM" / "lib",
    )
    prepend_env_path(
        env,
        "LIBRARY_PATH",
        cuda_root / "lib64",
        sdk_root / "lib",
    )


def validate_derived_dataset(dataset: Path) -> None:
    report_path = dataset / "fixed_joint_conversion_report.json"
    info_path = dataset / "meta" / "info.json"
    if not report_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(
            "Dataset must be produced by create_sonic_fixed_joint_dataset.py "
            f"and contain {report_path.name} plus meta/info.json"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixed_joints = report.get("fixed_joint_values")
    if fixed_joints != EXPECTED_FIXED_JOINTS:
        raise ValueError(
            f"Unexpected fixed joints: {fixed_joints}; expected {EXPECTED_FIXED_JOINTS}"
        )
    for key in (
        "max_source_alignment_error",
        "max_output_alignment_error",
        "max_fixed_joint_error",
    ):
        if float(report.get(key, float("inf"))) > 1e-6:
            raise ValueError(f"Dataset validation failed: {key}={report.get(key)}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    derived = info.get("derived_dataset", {})
    if derived.get("supervision_source") != "corrected observation.state[t+1]":
        raise ValueError("Dataset action supervision is not corrected observation.state[t+1]")
    if derived.get("action_wbc_used") is not False:
        raise ValueError("Dataset does not explicitly forbid action.wbc supervision")


def run_environment(
    args: argparse.Namespace,
    spec: RunSpec,
    gpu_id: int,
    output_dir: Path,
    run_name: str,
) -> dict[str, str]:
    env = dict(os.environ)
    configure_ppu_sdk_environment(env)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "WANDB_NAME": run_name,
            "GWP_SONIC_DATA_PATH": str(args.dataset),
            "GWP_NORM_STATS_PATH": str(args.dataset / NORM_STATS_NAME),
            "GWP05_TRANSFORMER_PRETRAINED": str(
                PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5"
            ),
            "GWP_WAN_PRETRAINED": str(
                PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers"
            ),
            "GWP05_PROJECT_DIR": str(output_dir),
            "GWP_GPU_IDS": str(gpu_id),
            "GWP_PHYSICAL_ACTION_DIM": "13",
            "GWP_ACTION_DELTA_MASK": bool_spec(ACTION_DELTA_MASK),
            "GWP_TRAINING_TIME_RTC_DELAY_STEPS": "0",
            "GWP_STATE_CONDITION_DROPOUT_PROB": str(
                spec.state_condition_dropout_prob
            ),
            "GWP_STATE_HISTORY_OFFSETS": int_spec(spec.state_history_offsets),
            "GWP_STATE_HISTORY_DROPOUT_PROB": "0",
            "GWP_APPEND_STATE_VELOCITY_TOKEN": "0",
            "GWP_APPEND_EPISODE_PROGRESS_TOKEN": "0",
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
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint_interval <= 0 or args.checkpoint_total_limit <= 0:
        raise ValueError("Checkpoint interval and retention limit must be positive")
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_suffix):
        raise ValueError("--run-suffix may contain only letters, numbers, '_' and '-'")
    validate_derived_dataset(args.dataset)
    norm_path = args.dataset / NORM_STATS_NAME
    if not norm_path.is_file():
        raise FileNotFoundError(norm_path)
    for required in (
        PROJECT_ROOT / ".venv-ppu" / "bin" / "python",
        PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5",
        PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers",
        Path("/usr/local/PPU_SDK/lib/libhggcrt1.so"),
        Path("/usr/local/PPU_SDK/CUDA_SDK/lib64"),
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    if COMPACT_STATE_HISTORY_OFFSETS != tuple(range(-7, 1)):
        raise ValueError("Compact history must contain eight consecutive frames ending at zero")
    for output_dir in (
        args.output_root / dated_name(spec, run_date, args.run_suffix, gpu_id)
        for spec, gpu_id in zip(RUNS, args.gpu_ids, strict=True)
    ):
        if output_dir.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")

    free_gib = shutil.disk_usage(args.output_root.parent).free / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            "Insufficient free space for two checkpoint streams: "
            f"free={free_gib:.2f} GiB required={args.min_free_gib:.2f} GiB"
        )


def run_record(
    args: argparse.Namespace,
    spec: RunSpec,
    gpu_id: int,
    run_date: str,
) -> dict[str, object]:
    run_name = dated_name(spec, run_date, args.run_suffix, gpu_id)
    return {
        "name": run_name,
        "description": spec.description,
        "dataset": str(args.dataset),
        "norm_stats": str(args.dataset / NORM_STATS_NAME),
        "prompt": PROMPT,
        "gpu_id": gpu_id,
        "output_dir": str(args.output_root / run_name),
        "action_delta_mask": list(ACTION_DELTA_MASK),
        "action_semantics": {
            "arm_joints_0_to_6": "relative_joint_delta",
            "hand_joints_7_to_12": "absolute_joint_position",
        },
        "state_condition_dropout_prob": spec.state_condition_dropout_prob,
        "state_history_offsets": list(spec.state_history_offsets),
        "state_history_tokens": len(spec.state_history_offsets),
        "state_history_span_frames": -min(spec.state_history_offsets),
        "state_history_span_seconds_at_50hz": -min(spec.state_history_offsets) / 50.0,
        "training_time_rtc_delay_steps": 0,
        "training_episode_spec": TRAIN_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "max_steps": args.max_steps,
        "batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_total_limit": args.checkpoint_total_limit,
        "checkpoint_export_dtype": "bf16",
        "transformer_pretrained": str(
            PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5"
        ),
    }


def start_run(
    args: argparse.Namespace,
    spec: RunSpec,
    gpu_id: int,
    run_date: str,
) -> dict[str, object]:
    record = run_record(args, spec, gpu_id, run_date)
    output_dir = Path(str(record["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    record["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(output_dir / "experiment.json", record)

    environment = run_environment(args, spec, gpu_id, output_dir, str(record["name"]))
    env_record = {
        key: environment[key]
        for key in sorted(environment)
        if key.startswith("GWP_")
        or key.startswith("GWP05_")
        or key == "WANDB_NAME"
        or key in PPU_RUNTIME_ENV_KEYS
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
        "name": record["name"],
        "gpu_id": gpu_id,
        "output_dir": str(output_dir),
        "pid": process.pid,
        "process": process,
    }


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    run_date = time.strftime("%Y%m%d")
    validate(args, run_date)
    if args.dry_run:
        print(
            json.dumps(
                [
                    run_record(args, spec, gpu_id, run_date)
                    for spec, gpu_id in zip(RUNS, args.gpu_ids, strict=True)
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = []
    try:
        for spec, gpu_id in zip(RUNS, args.gpu_ids, strict=True):
            started.append(start_run(args, spec, gpu_id, run_date))
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
                {
                    "name": item["name"],
                    "gpu_id": item["gpu_id"],
                    "output_dir": item["output_dir"],
                    "pid": item["pid"],
                }
                for item in started
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

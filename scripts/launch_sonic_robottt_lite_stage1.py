#!/usr/bin/env python3
"""Launch the four-way SONIC RoboTTT-lite stage-1 context comparison."""

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


CODE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path("/workspace/WM/giga-world-policy")
TRAIN_EPISODES = "0-69,73-115"
EVALUATION_EPISODES = [70, 71, 72, 116, 117, 118]
ACTION_DELTA_MASK = (True,) * 7 + (False,) * 6
NORM_STATS_NAME = "norm_stats_armdelta_handabs_h48_train_0-69_73-115.json"
EXPECTED_FIXED_JOINTS = {
    "left_hand_thumb_bend": 1.0,
    "left_hand_thumb_rotate": 0.0,
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    gpu_id: int
    context_length: int
    persistent: bool
    description: str


RUNS = (
    RunSpec(
        "m00_nomemory_matched_compute_c64",
        0,
        64,
        False,
        "TTT-KVB compute-matched control; fast weights reset at every replan.",
    ),
    RunSpec(
        "m01_robottt_lite_c16",
        1,
        16,
        True,
        "Persistent TTT-KVB memory with 16 replans (2.56 s at stride 8/50 Hz).",
    ),
    RunSpec(
        "m02_robottt_lite_c32",
        2,
        32,
        True,
        "Persistent TTT-KVB memory with 32 replans (5.12 s at stride 8/50 Hz).",
    ),
    RunSpec(
        "m03_robottt_lite_c64",
        3,
        64,
        True,
        "Persistent TTT-KVB memory with 64 replans (10.24 s at stride 8/50 Hz).",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-total-limit", type=int, default=2)
    parser.add_argument("--tbptt-steps", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=120.0)
    parser.add_argument("--startup-wait-seconds", type=float, default=45.0)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def configure_ppu_environment(env: dict[str, str]) -> None:
    sdk_root = Path("/usr/local/PPU_SDK")
    cuda_root = sdk_root / "CUDA_SDK"
    env.update(
        {
            "PPU_SDK": str(sdk_root),
            "CUDA_PATH": str(cuda_root),
            "CUDA_HOME": str(cuda_root),
            "CUDA_TOOLKIT_ROOT": str(cuda_root),
        }
    )
    for key, paths in {
        "PATH": [cuda_root / "bin", sdk_root / "bin", sdk_root / "ppu-smi" / "bin"],
        "LD_LIBRARY_PATH": [
            Path("/opt/accl-p"),
            cuda_root / "lib64",
            sdk_root / "lib",
            sdk_root / "sailSHMEM" / "lib",
        ],
        "LIBRARY_PATH": [cuda_root / "lib64", sdk_root / "lib"],
        "PYTHONPATH": [
            CODE_ROOT,
            CODE_ROOT / "third_party" / "giga-train",
            CODE_ROOT / "third_party" / "giga-models",
            CODE_ROOT / "third_party" / "giga-datasets",
        ],
    }.items():
        existing = env.get(key)
        values = [str(path) for path in paths]
        if existing:
            values.append(existing)
        env[key] = os.pathsep.join(values)


def validate_dataset(dataset: Path) -> None:
    report_path = dataset / "fixed_joint_conversion_report.json"
    info_path = dataset / "meta" / "info.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("fixed_joint_values") != EXPECTED_FIXED_JOINTS:
        raise ValueError("Dataset fixed-joint contract does not match the baseline")
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
        raise ValueError("Action supervision is not observation.state[t+1]")
    if derived.get("action_wbc_used") is not False:
        raise ValueError("Dataset does not explicitly forbid action.wbc supervision")


def run_name(spec: RunSpec, suffix: str) -> str:
    name = f"20260905_{spec.name}_stage1_gpu{spec.gpu_id}"
    return f"{name}_{suffix}" if suffix else name


def validate(args: argparse.Namespace) -> None:
    if args.max_steps < 1 or args.tbptt_steps < 1:
        raise ValueError("max steps and TBPTT steps must be positive")
    if args.checkpoint_interval < 1 or args.checkpoint_total_limit < 1:
        raise ValueError("checkpoint settings must be positive")
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_suffix):
        raise ValueError("run suffix contains unsupported characters")
    for required in (
        args.dataset,
        args.checkpoint,
        args.dataset / NORM_STATS_NAME,
        RUNTIME_ROOT / ".venv-ppu" / "bin" / "python",
        RUNTIME_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
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
        "stage": "stage1_memory_only",
        "description": spec.description,
        "gpu_id": spec.gpu_id,
        "dataset": str(args.dataset),
        "norm_stats": str(args.dataset / NORM_STATS_NAME),
        "transformer_pretrained": str(args.checkpoint),
        "output_dir": str(args.output_root / name),
        "context_length_replans": spec.context_length,
        "context_seconds": spec.context_length * 8 / 50.0,
        "replan_stride_raw_frames": 8,
        "memory_persistent": spec.persistent,
        "memory_layers": [7, 15, 23, 29],
        "memory_fast_model": "two_layer_mlp_256x256_ttt_kvb",
        "memory_gate_init": 0.001,
        "memory_context_half_life": spec.context_length,
        "tbptt_supervised_tail_steps": args.tbptt_steps,
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
            "WANDB_PROJECT": "sonic-cylinder-robottt-lite",
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
            "GWP_MEMORY_CONTEXT_LENGTH": str(spec.context_length),
            "GWP_MEMORY_CONTEXT_HALF_LIFE": str(spec.context_length),
            "GWP_MEMORY_REPLAN_STRIDE": "8",
            "GWP_MEMORY_LAYER_INDICES": "7,15,23,29",
            "GWP_MEMORY_DIM": "256",
            "GWP_MEMORY_HIDDEN_DIM": "256",
            "GWP_MEMORY_INNER_LR": "0.01",
            "GWP_MEMORY_GATE_INIT": "0.001",
            "GWP_MEMORY_TBPTT_STEPS": str(args.tbptt_steps),
            "GWP_MEMORY_PERSISTENT": "1" if spec.persistent else "0",
            "GWP_TRAIN_MEMORY_ONLY": "1",
            "GWP_TRAIN_ACTION_HEAD_ONLY": "0",
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
            if key.startswith("GWP_")
            or key.startswith("GWP05_")
            or key.startswith("WANDB_")
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
                os.killpg(process.pid, 15)
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

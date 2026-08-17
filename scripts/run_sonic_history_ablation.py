#!/usr/bin/env python3
"""Run the controlled SONIC late-stage/history ablation queue on configurable GPU groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PROMPT = "Pick up the pipette and keep holding it firmly."
STEP_RE = re.compile(r"Step\[(?P<step>\d+)/\d+\].*?total_loss: (?P<loss>[-+.\deE]+)")
CHECKPOINT_RE = re.compile(r"checkpoint_.*_step_(\d+)$")


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    env: dict[str, str] = field(default_factory=dict)


EXPERIMENTS = [
    Experiment("m00_baseline", "Correct prompt, original state/image context and unmasked action loss."),
    Experiment("m01_dim_mask", "Mask the 19 padded model action dimensions.", {"GWP_MASK_PADDED_ACTION_DIMS": "1"}),
    Experiment(
        "m02_dim_time_mask",
        "Mask padded action dimensions and episode-tail padded timesteps.",
        {"GWP_MASK_PADDED_ACTION_DIMS": "1", "GWP_MASK_PADDED_ACTION_STEPS": "1"},
    ),
    Experiment(
        "m03_state_history_1s",
        "Three state tokens spanning one second.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
        },
    ),
    Experiment(
        "m04_state_history_2s",
        "Five state tokens spanning two seconds.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-100,-75,-50,-25,0",
        },
    ),
    Experiment(
        "m05_state_history_velocity",
        "One-second state history plus an explicit normalized velocity token.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_APPEND_STATE_VELOCITY_TOKEN": "1",
        },
    ),
    Experiment(
        "m06_state_history_phase",
        "One-second state history plus normalized episode phase.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_APPEND_EPISODE_PROGRESS_TOKEN": "1",
        },
    ),
    Experiment(
        "m07_visual_history",
        "Five visual frames ending at the current observation (two clean VAE latents).",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_IMAGE_FRAME_OFFSETS": "-24,-18,-12,-6,0,12,24,36,48",
            "GWP_MAX_REF_FRAMES": "5",
            "GWP_MIN_REF_LATENTS": "2",
        },
    ),
    Experiment(
        "m08_state_visual_history",
        "One-second state history combined with visual history.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_IMAGE_FRAME_OFFSETS": "-24,-18,-12,-6,0,12,24,36,48",
            "GWP_MAX_REF_FRAMES": "5",
            "GWP_MIN_REF_LATENTS": "2",
        },
    ),
    Experiment(
        "m09_history_dropout",
        "Combined history with 30 percent state-history dropout.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_STATE_HISTORY_DROPOUT_PROB": "0.3",
            "GWP_IMAGE_FRAME_OFFSETS": "-24,-18,-12,-6,0,12,24,36,48",
            "GWP_MAX_REF_FRAMES": "5",
            "GWP_MIN_REF_LATENTS": "2",
        },
    ),
    Experiment(
        "m10_tail_weighted",
        "One-second state history with linearly increasing horizon loss weight.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_ACTION_TEMPORAL_WEIGHT_MODE": "linear",
        },
    ),
    Experiment(
        "m11_smooth_terminal",
        "One-second state history with velocity consistency and terminal losses.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_ACTION_SMOOTHNESS_WEIGHT": "0.1",
            "GWP_ACTION_TERMINAL_WEIGHT": "0.2",
        },
    ),
    Experiment(
        "m12_short_horizon",
        "One-second state history with a 24-step action horizon.",
        {
            "GWP_MASK_PADDED_ACTION_DIMS": "1",
            "GWP_MASK_PADDED_ACTION_STEPS": "1",
            "GWP_STATE_HISTORY_OFFSETS": "-50,-25,0",
            "GWP_ACTION_HORIZON": "24",
            "GWP_IMAGE_FRAME_OFFSETS": "0,6,12,18,24",
        },
    ),
]


@dataclass
class RunningJob:
    experiment: Experiment
    gpu_ids: str
    output_dir: Path
    train_process: subprocess.Popen | AttachedProcess
    train_log: object
    wandb_process: subprocess.Popen | AttachedProcess
    wandb_log: object
    convergence_reason: str | None = None
    warm_start_checkpoint: str | None = None
    warm_start_step: int | None = None


@dataclass
class AttachedProcess:
    """A minimal ``Popen``-like handle for a process adopted after scheduler restart."""

    pid: int
    checker: Callable[[int], bool]

    def poll(self) -> int | None:
        return None if self.checker(self.pid) else 0

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(f"attached pid {self.pid}", timeout)
            time.sleep(0.2)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sonic_pipette_pick_hold_history_ablation_20260817",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/workspace/WM/datasets/8_14_shake_beaker_1_success_left13_all_pipette_hold_v3"),
    )
    parser.add_argument("--prompt", default=TASK_PROMPT)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--min-convergence-step", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--benchmark-stride", type=int, default=100)
    parser.add_argument("--benchmark-max-per-episode", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--gpu-groups",
        default="0,1;2,3",
        help="Semicolon-separated GPU groups. One group runs experiments sequentially.",
    )
    parser.add_argument(
        "--adopt-running",
        action="store_true",
        help="Adopt matching live jobs recorded in scheduler_state.json after a scheduler restart.",
    )
    return parser.parse_args()


def parse_gpu_groups(raw: str) -> tuple[str, ...]:
    groups = []
    assigned = set()
    for raw_group in raw.split(";"):
        ids = [item.strip() for item in raw_group.split(",") if item.strip()]
        if not ids or any(not item.isdigit() for item in ids):
            raise ValueError(f"Invalid GPU group {raw_group!r} in {raw!r}")
        duplicate = assigned.intersection(ids)
        if duplicate:
            raise ValueError(f"GPU ids assigned to multiple groups: {sorted(duplicate)}")
        assigned.update(ids)
        groups.append(",".join(ids))
    if not groups:
        raise ValueError("At least one GPU group is required")
    return tuple(groups)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}, "started_at": time.time()}


def _pid_matches_training_output(pid: int, output_dir: Path) -> bool:
    """Avoid treating a recycled PID or an unrelated training process as ours."""
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        environ = (proc / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    expected = f"GWP05_PROJECT_DIR={output_dir}".encode()
    return "scripts/train.py" in cmdline and expected in environ


def active_saved_jobs(state: dict, output_root: Path) -> list[dict]:
    active = []
    for record in state.get("running", []):
        name = str(record.get("name", ""))
        pid = int(record.get("pid", -1))
        if name and pid > 0 and _pid_matches_training_output(pid, output_root / name):
            active.append({**record, "pid": pid, "name": name})
    return active


def _pid_matches_wandb_sync(pid: int, output_dir: Path) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return (
        "sync_training_log_to_wandb.py" in cmdline
        and str(output_dir / "launcher.log") in cmdline
    )


def experiment_by_name(name: str) -> Experiment:
    for experiment in EXPERIMENTS:
        if experiment.name == name:
            return experiment
    raise KeyError(f"Unknown experiment recorded in scheduler state: {name}")


def attach_saved_job(args: argparse.Namespace, record: dict) -> RunningJob:
    experiment = experiment_by_name(str(record["name"]))
    output_dir = args.output_root / experiment.name
    train_pid = int(record["pid"])
    if not _pid_matches_training_output(train_pid, output_dir):
        raise RuntimeError(f"Cannot adopt stale training pid {train_pid} for {experiment.name}")
    wandb_pid_path = output_dir / "wandb_sync.pid"
    if not wandb_pid_path.is_file():
        raise RuntimeError(f"Cannot adopt {experiment.name}: missing {wandb_pid_path}")
    wandb_pid = int(wandb_pid_path.read_text(encoding="utf-8").strip())
    if not _pid_matches_wandb_sync(wandb_pid, output_dir):
        raise RuntimeError(f"Cannot adopt {experiment.name}: W&B sync pid {wandb_pid} is not active")
    metadata = {}
    experiment_path = output_dir / "experiment.json"
    if experiment_path.is_file():
        metadata = json.loads(experiment_path.read_text(encoding="utf-8"))
    return RunningJob(
        experiment=experiment,
        gpu_ids=str(record["gpu_ids"]),
        output_dir=output_dir,
        train_process=AttachedProcess(
            train_pid,
            lambda pid, path=output_dir: _pid_matches_training_output(pid, path),
        ),
        train_log=(output_dir / "launcher.log").open("a", encoding="utf-8"),
        wandb_process=AttachedProcess(
            wandb_pid,
            lambda pid, path=output_dir: _pid_matches_wandb_sync(pid, path),
        ),
        wandb_log=(output_dir / "wandb_sync.log").open("a", encoding="utf-8"),
        warm_start_checkpoint=metadata.get("warm_start_checkpoint"),
        warm_start_step=metadata.get("warm_start_step"),
    )


def validate_prompt_assets(dataset: Path, prompt: str) -> None:
    """Fail closed if metadata, task lookup, and cached embeddings disagree."""
    if not prompt.strip():
        raise ValueError("The task prompt must not be empty")

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_prompt = info.get("task_prompt")
    if dataset_prompt != prompt:
        raise ValueError(f"Dataset task_prompt mismatch: {dataset_prompt!r} != {prompt!r}")

    import pyarrow.parquet as pq

    task_table = pq.read_table(dataset / "meta" / "tasks.parquet")
    task_column = next(
        (name for name in ("task", "name", "__index_level_0__") if name in task_table.column_names),
        None,
    )
    if task_column is None:
        raise KeyError(f"No task text column in {dataset / 'meta' / 'tasks.parquet'}")
    task_texts = {str(value) for value in task_table[task_column].to_pylist()}
    if task_texts != {prompt}:
        raise ValueError(f"Dataset task table mismatch: {sorted(task_texts)!r} != {[prompt]!r}")

    provenance_path = dataset / "t5_embedding" / "prompt_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("prompt") != prompt:
        raise ValueError(
            f"T5 prompt provenance mismatch: {provenance.get('prompt')!r} != {prompt!r}"
        )
    episode_indices = provenance.get("episode_indices")
    if not isinstance(episode_indices, list) or not episode_indices:
        raise ValueError(f"No episode_indices recorded in {provenance_path}")
    missing = [
        index
        for index in episode_indices
        if not (dataset / "t5_embedding" / f"episode_{int(index):06d}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing prompt embeddings for episodes: {missing}")


def read_losses(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    matches = STEP_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace"))
    return [(int(match.group("step")), float(match.group("loss"))) for match in matches]


def curve_has_converged(losses: list[tuple[int, float]], minimum_step: int) -> bool:
    if not losses or losses[-1][0] < minimum_step or len(losses) < 200:
        return False
    previous = [loss for _, loss in losses[-200:-100]]
    recent = [loss for _, loss in losses[-100:]]
    previous_mean = sum(previous) / len(previous)
    recent_mean = sum(recent) / len(recent)
    relative_gain = (previous_mean - recent_mean) / max(abs(previous_mean), 1e-8)
    recent_variance = sum((loss - recent_mean) ** 2 for loss in recent) / len(recent)
    coefficient_of_variation = math.sqrt(recent_variance) / max(abs(recent_mean), 1e-8)
    first_half = sum(recent[:50]) / 50
    second_half = sum(recent[50:]) / 50
    relative_drift = abs(second_half - first_half) / max(abs(recent_mean), 1e-8)
    return relative_gain < 0.04 and coefficient_of_variation < 0.35 and relative_drift < 0.08


def latest_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    candidates = []
    for path in (output_dir / "models").glob("checkpoint_*_step_*"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=None, key=lambda item: item[0])


def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    """Return true only after the trainer has finished all checkpoint writes.

    The checkpoint directory is created before DeepSpeed optimizer shards and
    inference weights are written.  Treating directory creation as completion
    can terminate a converged job in the middle of its save and destroy the
    only retained inference checkpoint.
    """
    required_files = (
        checkpoint_dir / "transformer" / "config.json",
        checkpoint_dir / "transformer" / "diffusion_pytorch_model.bin",
        checkpoint_dir / "scheduler.bin",
        checkpoint_dir / "custom_checkpoint_0.pkl",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_files):
        return False
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in checkpoint_dir.glob("random_states_*.pkl")
    )


def latest_complete_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    candidates = []
    for path in (output_dir / "models").glob("checkpoint_*_step_*"):
        match = CHECKPOINT_RE.match(path.name)
        if match and checkpoint_is_complete(path):
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=None, key=lambda item: item[0])


def next_attempt_number(output_root: Path, experiment_name: str) -> int:
    archive_root = output_root / "_failed_attempts"
    archived = list(archive_root.glob(f"{experiment_name}_attempt_*"))
    next_attempt = len(archived) + 1
    experiment_path = output_root / experiment_name / "experiment.json"
    if experiment_path.is_file():
        try:
            previous = json.loads(experiment_path.read_text(encoding="utf-8"))
            next_attempt = max(next_attempt, int(previous.get("attempt", 0)) + 1)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return next_attempt


def archive_incomplete_attempt(output_root: Path, experiment_name: str) -> Path | None:
    """Move an interrupted partial-checkpoint attempt aside before a clean retry."""
    output_dir = output_root / experiment_name
    if not output_dir.is_dir():
        return None
    if latest_checkpoint(output_dir) is None or latest_complete_checkpoint(output_dir) is not None:
        return None
    attempt = next_attempt_number(output_root, experiment_name)
    archive_root = output_root / "_failed_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    target = archive_root / (
        f"{experiment_name}_attempt_{attempt:02d}_incomplete_checkpoint_{timestamp}"
    )
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(output_dir), str(target))
    return target


def stop_process_group(process: subprocess.Popen, timeout: float = 20.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def job_environment(
    args: argparse.Namespace,
    experiment: Experiment,
    gpu_ids: str,
    output_dir: Path,
    transformer_pretrained: Path | None = None,
) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "GWP_SONIC_DATA_PATH": str(args.dataset),
            "GWP_NORM_STATS_PATH": str(args.dataset / "norm_stats.json"),
            "GWP05_TRANSFORMER_PRETRAINED": str(
                transformer_pretrained or PROJECT_ROOT / "weights" / "Giga-World-Policy-0.5"
            ),
            "GWP_WAN_PRETRAINED": str(PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers"),
            "GWP05_PROJECT_DIR": str(output_dir),
            "GWP_GPU_IDS": gpu_ids,
            "GWP_PHYSICAL_ACTION_DIM": "13",
            "GWP_IMAGE_AUGMENTATION": "1",
            "GWP_BATCH_SIZE_PER_GPU": "1",
            "GWP_NUM_WORKERS": "8",
            "GWP_TRAIN_EPISODES": "0-19",
            "GWP_MAX_STEPS": str(args.max_steps),
            "GWP_GRAD_ACCUM_STEPS": "4",
            "GWP_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
            "GWP_CHECKPOINT_TOTAL_LIMIT": "1",
            "GWP_CHECKPOINT_WEIGHTS_ONLY": "1",
            "GWP_LOG_INTERVAL": "1",
            "GWP_WITH_EMA": "0",
            "GWP_ACTIVATION_CHECKPOINTING": "1",
            "GWP_DEEPSPEED_CONFIG": "accelerate_configs/zero3.json",
        }
    )
    env.update(experiment.env)
    return env


def start_job(args: argparse.Namespace, experiment: Experiment, gpu_ids: str) -> RunningJob:
    output_dir = args.output_root / experiment.name
    archived_attempt = archive_incomplete_attempt(args.output_root, experiment.name)
    attempt = next_attempt_number(args.output_root, experiment.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    warm_start = latest_complete_checkpoint(output_dir)
    warm_start_step = warm_start[0] if warm_start is not None else None
    warm_start_transformer = warm_start[1] / "transformer" if warm_start is not None else None
    experiment_path = output_dir / "experiment.json"
    write_json(
        experiment_path,
        {
            "name": experiment.name,
            "description": experiment.description,
            "env": experiment.env,
            "dataset": str(args.dataset),
            "prompt": args.prompt,
            "training_episodes": list(range(20)),
            "evaluation_episodes": [20, 21, 22, 23],
            "attempt": attempt,
            "archived_previous_attempt": str(archived_attempt) if archived_attempt else None,
            "gpu_ids": gpu_ids,
            "warm_start_checkpoint": str(warm_start[1]) if warm_start is not None else None,
            "warm_start_step": warm_start_step,
        },
    )
    train_log = (output_dir / "launcher.log").open("a", encoding="utf-8")
    train_process = subprocess.Popen(
        [
            str(PROJECT_ROOT / ".venv-ppu" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            "giga_world_policy_0_5_sonic_g1_finetune",
        ],
        cwd=PROJECT_ROOT,
        env=job_environment(
            args,
            experiment,
            gpu_ids,
            output_dir,
            transformer_pretrained=warm_start_transformer,
        ),
        stdin=subprocess.DEVNULL,
        stdout=train_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (output_dir / "launcher.pid").write_text(f"{train_process.pid}\n", encoding="utf-8")

    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)
    wandb_log = (output_dir / "wandb_sync.log").open("a", encoding="utf-8")
    wandb_env = dict(os.environ)
    wandb_env["WANDB_DIR"] = str(wandb_dir)
    wandb_suffix = "" if attempt == 1 else f"-attempt{attempt}"
    wandb_run_id = (
        f"sonic-pipette-pick-hold-history-{experiment.name}-20260817{wandb_suffix}"
    )
    wandb_process = subprocess.Popen(
        [
            str(PROJECT_ROOT / ".venv-ppu" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "sync_training_log_to_wandb.py"),
            "--log",
            str(output_dir / "launcher.log"),
            "--state",
            str(output_dir / "wandb_sync_state.json"),
            "--entity",
            "ttyyc-hust",
            "--project",
            "gigaworld_lab",
            "--run-id",
            wandb_run_id,
            "--run-name",
            wandb_run_id,
            "--config-json",
            str(experiment_path),
        ],
        cwd=PROJECT_ROOT,
        env=wandb_env,
        stdin=subprocess.DEVNULL,
        stdout=wandb_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (output_dir / "wandb_sync.pid").write_text(f"{wandb_process.pid}\n", encoding="utf-8")
    return RunningJob(
        experiment=experiment,
        gpu_ids=gpu_ids,
        output_dir=output_dir,
        train_process=train_process,
        train_log=train_log,
        wandb_process=wandb_process,
        wandb_log=wandb_log,
        warm_start_checkpoint=str(warm_start[1]) if warm_start is not None else None,
        warm_start_step=warm_start_step,
    )


def finish_job_logging(job: RunningJob) -> None:
    time.sleep(5)
    stop_process_group(job.wandb_process, timeout=10)
    job.train_log.close()
    job.wandb_log.close()


def export_inference_model(args: argparse.Namespace, job: RunningJob) -> Path:
    latest = latest_complete_checkpoint(job.output_dir)
    if latest is None:
        raise RuntimeError(f"No complete checkpoint found for {job.experiment.name}")
    checkpoint_step, checkpoint_dir = latest
    transformer_source = checkpoint_dir / "transformer"
    if not transformer_source.is_dir():
        raise FileNotFoundError(transformer_source)

    export_dir = job.output_dir / "inference_model"
    export_dir.mkdir(exist_ok=True)
    transformer_target = export_dir / "transformer"
    if transformer_target.exists():
        shutil.rmtree(transformer_target)
    shutil.move(str(transformer_source), str(transformer_target))
    shutil.copy2(args.dataset / "norm_stats.json", export_dir / "norm_stats.json")
    shutil.copy2(
        args.dataset / "t5_embedding" / "episode_000000.pt",
        export_dir / "prompt_embedding.pt",
    )

    warm_start_transformer = (
        Path(job.warm_start_checkpoint) / "transformer"
        if job.warm_start_checkpoint is not None
        else None
    )
    env = job_environment(
        args,
        job.experiment,
        job.gpu_ids,
        job.output_dir,
        transformer_pretrained=warm_start_transformer,
    )
    image_offsets = [int(value) for value in env.get("GWP_IMAGE_FRAME_OFFSETS", "0,12,24,36,48").split(",")]
    state_offsets = [int(value) for value in env.get("GWP_STATE_HISTORY_OFFSETS", "0").split(",")]
    visual_history_frames = int(env.get("GWP_MAX_REF_FRAMES", "1"))
    manifest = {
        "format_version": 1,
        "experiment": job.experiment.name,
        "description": job.experiment.description,
        "transformer": "transformer",
        "base_model": str(PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B-Diffusers"),
        "norm_stats": "norm_stats.json",
        "prompt_embedding": "prompt_embedding.pt",
        "prompt": args.prompt,
        "evaluation_dataset": str(args.dataset),
        "evaluation_episodes": [20, 21, 22, 23],
        "training_episodes": list(range(20)),
        "image_key": "observation.images.ego_view",
        "image_frame_offsets": image_offsets,
        "state_history_offsets": state_offsets,
        "state_history_tokens": len(state_offsets),
        "visual_history_frames": visual_history_frames,
        "append_state_velocity_token": env.get("GWP_APPEND_STATE_VELOCITY_TOKEN", "0") == "1",
        "append_episode_progress_token": env.get("GWP_APPEND_EPISODE_PROGRESS_TOKEN", "0") == "1",
        "action_horizon": int(env.get("GWP_ACTION_HORIZON", "48")),
        "model_image_frames": len(image_offsets),
        "physical_action_dim": 13,
        "model_action_dim": 32,
        "delta_mask": [False] * 13,
        "fps": 50,
        "image_size": [320, 384],
        "num_inference_steps": 10,
        "model_dtype": "bf16",
        "action_flow_shift": 5.0,
        "training_checkpoint_step": checkpoint_step,
        "warm_start_checkpoint": job.warm_start_checkpoint,
        "warm_start_step": job.warm_start_step,
        "convergence_reason": job.convergence_reason or "max_steps",
        "training_env": {key: value for key, value in env.items() if key.startswith("GWP_")},
    }
    manifest_path = export_dir / "manifest.json"
    write_json(manifest_path, manifest)

    # Checkpoints contain large optimizer shards that are not needed for direct
    # inference.  This scheduler only removes checkpoint trees it created.
    shutil.rmtree(checkpoint_dir)
    return manifest_path


def run_benchmark(args: argparse.Namespace, manifest: Path, device: str) -> Path:
    output = manifest.parent / "benchmark.json"
    command = [
        str(PROJECT_ROOT / ".venv-ppu" / "bin" / "python"),
        str(PROJECT_ROOT / "scripts" / "evaluate_sonic_history.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--device",
        device,
        "--sample-stride",
        str(args.benchmark_stride),
        "--max-samples-per-episode",
        str(args.benchmark_max_per_episode),
    ]
    with (manifest.parent / "benchmark.log").open("a", encoding="utf-8") as stream:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return output


def recover_exported_models(args: argparse.Namespace, state: dict) -> None:
    """Resume benchmark/finalization without retraining an already exported model."""
    for experiment in EXPERIMENTS:
        export_dir = args.output_root / experiment.name / "inference_model"
        manifest = export_dir / "manifest.json"
        benchmark = export_dir / "benchmark.json"
        if not manifest.is_file():
            continue
        if not benchmark.is_file():
            benchmark = run_benchmark(args, manifest, "cuda:0")
        state["completed"][experiment.name] = {
            "manifest": str(manifest),
            "benchmark": str(benchmark),
            "completed_at": time.time(),
            "recovered_from_export": True,
        }
        state["failed"].pop(experiment.name, None)


def recover_complete_checkpoints(args: argparse.Namespace, state: dict) -> None:
    """Export and benchmark a converged checkpoint left by an interrupted scheduler."""
    for experiment in EXPERIMENTS:
        if experiment.name in state["completed"]:
            continue
        output_dir = args.output_root / experiment.name
        checkpoint = latest_complete_checkpoint(output_dir)
        if checkpoint is None:
            continue
        checkpoint_step, _ = checkpoint
        losses = read_losses(output_dir / "launcher.log")
        converged = checkpoint_step >= args.min_convergence_step and curve_has_converged(
            losses, args.min_convergence_step
        )
        reached_limit = checkpoint_step >= args.max_steps
        if not (converged or reached_limit):
            continue
        reason = (
            f"rolling_curve_converged_at_step_{losses[-1][0]}"
            if converged
            else "max_steps"
        )
        recovered_job = RunningJob(
            experiment=experiment,
            gpu_ids="0,1",
            output_dir=output_dir,
            train_process=None,
            train_log=None,
            wandb_process=None,
            wandb_log=None,
            convergence_reason=reason,
        )
        manifest = export_inference_model(args, recovered_job)
        benchmark = run_benchmark(args, manifest, "cuda:0")
        state["completed"][experiment.name] = {
            "manifest": str(manifest),
            "benchmark": str(benchmark),
            "completed_at": time.time(),
            "recovered_from_checkpoint": True,
        }
        state["failed"].pop(experiment.name, None)


def refresh_results(output_root: Path) -> None:
    rows = []
    for experiment in EXPERIMENTS:
        path = output_root / experiment.name / "inference_model" / "benchmark.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        post = result["post5"]
        rows.append(
            {
                "experiment": experiment.name,
                "post5_first_step_mse": post.get("first_step_mse"),
                "post5_horizon_mse": post.get("horizon_mse"),
                "post5_jitter_mse": result.get("post5_jitter_mse"),
                "post5_stall_rate": post.get("stall"),
                "post5_horizon_stall_rate": post.get("horizon_stall"),
                "post5_velocity_mse": post.get("velocity_mse"),
                "post5_acceleration_mse": post.get("acceleration_mse"),
                "post5_excess_jitter_energy": post.get("excess_jitter_energy"),
                "post5_hold_horizon_mse": post.get("hold_horizon_mse"),
                "post5_hold_stall_rate": post.get("hold_horizon_stall"),
                "post5_horizon_mse_vs_hold_ratio": post.get("horizon_mse_vs_hold_ratio"),
                "repeat_history_degradation": post.get("repeat_history_degradation"),
                "reverse_history_degradation": post.get("reverse_history_degradation"),
            }
        )
    if not rows:
        return
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "results.json", rows)


def monitor_export_and_benchmark_jobs(
    args: argparse.Namespace,
    state: dict,
    state_path: Path,
    running: list[RunningJob],
) -> None:
    write_json(
        state_path,
        {
            **state,
            "running": [
                {"name": job.experiment.name, "pid": job.train_process.pid, "gpu_ids": job.gpu_ids}
                for job in running
            ],
        },
    )

    while any(job.train_process.poll() is None for job in running):
        for job in running:
            if job.train_process.poll() is not None:
                continue
            losses = read_losses(job.output_dir / "launcher.log")
            checkpoint = latest_complete_checkpoint(job.output_dir)
            if (
                checkpoint is not None
                and checkpoint[0] >= args.min_convergence_step
                and curve_has_converged(losses, args.min_convergence_step)
            ):
                job.convergence_reason = f"rolling_curve_converged_at_step_{losses[-1][0]}"
                stop_process_group(job.train_process)
        time.sleep(args.poll_seconds)

    manifests = []
    for job in running:
        return_code = job.train_process.poll()
        finish_job_logging(job)
        if (
            return_code not in (0, -signal.SIGTERM)
            and latest_complete_checkpoint(job.output_dir) is None
        ):
            state["failed"][job.experiment.name] = {
                "return_code": return_code,
                "time": time.time(),
            }
            write_json(state_path, {**state, "running": []})
            raise RuntimeError(
                f"Training failed before a checkpoint for {job.experiment.name}: return_code={return_code}"
            )
        manifest = export_inference_model(args, job)
        manifests.append((job, manifest))

    for job, manifest in manifests:
        benchmark = run_benchmark(args, manifest, "cuda:0")
        state["completed"][job.experiment.name] = {
            "manifest": str(manifest),
            "benchmark": str(benchmark),
            "completed_at": time.time(),
        }
        state["failed"].pop(job.experiment.name, None)
        write_json(state_path, {**state, "running": []})
        refresh_results(args.output_root)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    gpu_groups = parse_gpu_groups(args.gpu_groups)
    validate_prompt_assets(args.dataset, args.prompt)
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be present in the scheduler environment")
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "scheduler_state.json"
    state = load_state(state_path)
    active = active_saved_jobs(state, args.output_root)
    if active and not args.adopt_running:
        details = ", ".join(f"{item['name']} (pid={item['pid']})" for item in active)
        raise RuntimeError(
            "Refusing to launch duplicate training while saved jobs are still active: " + details
        )
    recover_exported_models(args, state)
    recover_complete_checkpoints(args, state)
    refresh_results(args.output_root)
    write_json(state_path, state)
    write_json(
        args.output_root / "experiment_registry.json",
        [
            {"name": item.name, "description": item.description, "env": item.env}
            for item in EXPERIMENTS
        ],
    )

    if active:
        adopted = [attach_saved_job(args, record) for record in active]
        monitor_export_and_benchmark_jobs(args, state, state_path, adopted)

    pending = [item for item in EXPERIMENTS if item.name not in state["completed"]]
    for group_start in range(0, len(pending), len(gpu_groups)):
        experiments = pending[group_start : group_start + len(gpu_groups)]
        running = [
            start_job(args, experiment, gpu_groups[index])
            for index, experiment in enumerate(experiments)
        ]
        monitor_export_and_benchmark_jobs(args, state, state_path, running)

    write_json(state_path, {**state, "running": [], "finished_at": time.time()})
    print(f"Completed {len(EXPERIMENTS)} experiments under {args.output_root}")


if __name__ == "__main__":
    main()

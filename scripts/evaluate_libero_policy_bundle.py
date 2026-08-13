#!/usr/bin/env python3
"""Run one audited open-loop and closed-loop LIBERO policy evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--open-loop-python", type=Path, required=True)
    parser.add_argument("--closed-loop-python", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rollouts-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--model-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--server-timeout", type=int, default=600)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def checkpoint_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    weights = sorted(path.glob("*.bin")) + sorted(path.glob("*.safetensors"))
    return bool(weights) and all(file.is_file() and file.stat().st_size > 0 for file in weights)


def wait_for_server(log_path: Path, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    marker = "Serving LIBERO GWP policy"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Inference server exited with code {process.returncode}: {log_path}"
            )
        if log_path.exists() and marker in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(2)
    raise TimeoutError(f"Inference server did not become ready: {log_path}")


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def require_clean_tracked_worktree(project_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Refusing evaluation from a tracked-dirty worktree")


def checkpoint_summary(path: Path) -> dict[str, Any]:
    weights = sorted(path.glob("*.bin")) + sorted(path.glob("*.safetensors"))
    return {
        "root": str(path.resolve()),
        "config_sha256": sha256_file(path / "config.json"),
        "weight_files": [
            {"name": file.name, "bytes": file.stat().st_size} for file in weights
        ],
        "weight_bytes": sum(file.stat().st_size for file in weights),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    inputs = (
        args.checkpoint,
        args.vae,
        args.norm_stats,
        args.dataset_root,
        args.episodes_file,
        args.policy_python,
        args.open_loop_python,
        args.closed_loop_python,
    )
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    if not checkpoint_ready(args.checkpoint):
        raise RuntimeError(f"Checkpoint is incomplete: {args.checkpoint}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    if args.gpu_id < 0 or args.port <= 0:
        raise ValueError("gpu-id must be non-negative and port must be positive")
    if args.rollouts_per_task <= 0 or args.max_steps <= 0 or args.replan_steps <= 0:
        raise ValueError("rollout count and step settings must be positive")
    require_clean_tracked_worktree(project_root)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    open_output = output_dir / "open_loop_heldout"
    closed_output = output_dir / "closed_loop"
    server_log = output_dir / "policy_server.log"
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "libero-policy-evaluation-bundle-v1",
        "status": "running",
        "git_commit": git_commit(project_root),
        "checkpoint": checkpoint_summary(args.checkpoint),
        "vae": str(args.vae.resolve()),
        "vae_config_sha256": sha256_file(args.vae / "config.json"),
        "norm_stats": str(args.norm_stats.resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_info_sha256": sha256_file(
            args.dataset_root / "meta" / "info.json"
        ),
        "suite": args.suite,
        "episodes_file": str(args.episodes_file.resolve()),
        "episodes_file_sha256": sha256_file(args.episodes_file),
        "gpu_id": args.gpu_id,
        "port": args.port,
        "rollouts_per_task": args.rollouts_per_task,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "num_inference_steps": args.num_inference_steps,
        "model_dtype": args.model_dtype,
        "seed": args.seed,
    }
    write_json(manifest_path, manifest)

    server_command = [
        str(args.policy_python),
        str(project_root / "scripts" / "inference_libero_gwp05.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--vae",
        str(args.vae),
        "--norm-stats",
        str(args.norm_stats),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        f"cuda:{args.gpu_id}",
        "--model-dtype",
        args.model_dtype,
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--num-views",
        "2",
        "--seed",
        str(args.seed),
    ]
    server_file = server_log.open("w", encoding="utf-8")
    server: subprocess.Popen | None = None
    try:
        server = subprocess.Popen(
            server_command,
            cwd=project_root,
            stdout=server_file,
            stderr=subprocess.STDOUT,
        )
        wait_for_server(server_log, server, args.server_timeout)
        common_policy_metadata = [
            "--policy-checkpoint",
            str(args.checkpoint),
            "--policy-num-inference-steps",
            str(args.num_inference_steps),
            "--policy-model-dtype",
            args.model_dtype,
            "--policy-seed",
            str(args.seed),
        ]
        open_command = [
            str(args.open_loop_python),
            str(project_root / "scripts" / "evaluate_libero_open_loop.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--suite",
            args.suite,
            "--episodes-file",
            str(args.episodes_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--num-views",
            "2",
            *common_policy_metadata,
            "--output-dir",
            str(open_output),
        ]
        run_logged(
            open_command,
            cwd=project_root,
            log_path=output_dir / "open_loop.log",
        )

        closed_command = [
            str(args.closed_loop_python),
            str(project_root / "scripts" / "evaluate_libero_closed_loop.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--suite",
            args.suite,
            "--rollouts-per-task",
            str(args.rollouts_per_task),
            "--max-steps",
            str(args.max_steps),
            "--replan-steps",
            str(args.replan_steps),
            "--num-views",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--seed",
            str(args.seed),
            *common_policy_metadata,
            "--output-dir",
            str(closed_output),
        ]
        closed_env = os.environ.copy()
        closed_env["MUJOCO_GL"] = "egl"
        closed_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        run_logged(
            closed_command,
            cwd=project_root,
            log_path=output_dir / "closed_loop.log",
            env=closed_env,
        )

        open_summary = open_output / "summary.json"
        closed_summary = closed_output / "summary.json"
        if not open_summary.is_file() or not closed_summary.is_file():
            raise FileNotFoundError("Evaluation bundle did not produce both summaries")
        manifest.update(
            {
                "status": "complete",
                "open_loop_summary": str(open_summary.resolve()),
                "open_loop_summary_sha256": sha256_file(open_summary),
                "closed_loop_summary": str(closed_summary.resolve()),
                "closed_loop_summary_sha256": sha256_file(closed_summary),
            }
        )
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        write_json(manifest_path, manifest)
        raise
    finally:
        if server is not None:
            terminate(server)
        server_file.close()

    print(json.dumps(manifest, indent=2))
    print(manifest_path)


if __name__ == "__main__":
    main()

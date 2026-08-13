#!/usr/bin/env python3
"""Evaluate a two-view GWP checkpoint on the four standard LIBERO suites."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


SUITES = (
    ("libero_spatial", 280),
    ("libero_object", 280),
    ("libero_goal", 300),
    ("libero_10", 520),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=4, default=(2, 3, 4, 5))
    parser.add_argument("--base-port", type=int, default=11520)
    parser.add_argument("--rollouts-per-task", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--wait-for-checkpoint", type=int, default=0, metavar="SECONDS")
    parser.add_argument("--training-pid", type=int, default=None)
    return parser.parse_args()


def wait_for_server(log_path: Path, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    marker = "Serving LIBERO GWP policy"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Inference server exited with code {process.returncode}: {log_path}")
        if log_path.exists() and marker in log_path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(2)
    raise TimeoutError(f"Inference server did not become ready within {timeout}s: {log_path}")


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_checkpoint(path: Path, timeout: int, training_pid: int | None) -> None:
    if path.is_dir():
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if training_pid is not None and not process_alive(training_pid):
            raise RuntimeError(f"Training process {training_pid} exited before checkpoint appeared: {path}")
        time.sleep(60)
        if path.is_dir():
            return
    raise TimeoutError(f"Checkpoint did not appear within {timeout}s: {path}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    wait_for_checkpoint(args.checkpoint, args.wait_for_checkpoint, args.training_pid)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    servers: list[subprocess.Popen] = []
    server_logs = []
    clients: list[tuple[str, subprocess.Popen, object]] = []
    try:
        for index, ((suite, _), gpu_id) in enumerate(zip(SUITES, args.gpu_ids)):
            port = args.base_port + index
            log_path = args.output_dir / f"{suite}_server.log"
            log_file = log_path.open("w", encoding="utf-8")
            server_logs.append(log_file)
            command = [
                str(args.policy_python),
                str(project_root / "scripts" / "inference_libero_gwp05.py"),
                "--checkpoint", str(args.checkpoint),
                "--vae", str(args.vae),
                "--norm-stats", str(args.norm_stats),
                "--port", str(port),
                "--device", f"cuda:{gpu_id}",
                "--num-views", "2",
                "--num-inference-steps", str(args.num_inference_steps),
                "--seed", str(args.seed),
            ]
            servers.append(subprocess.Popen(command, cwd=project_root, stdout=log_file, stderr=subprocess.STDOUT))

        for (suite, _), process, log_file in zip(SUITES, servers, server_logs):
            wait_for_server(Path(log_file.name), process, args.server_timeout)

        for index, ((suite, max_steps), gpu_id) in enumerate(zip(SUITES, args.gpu_ids)):
            port = args.base_port + index
            suite_output = args.output_dir / suite
            log_file = (args.output_dir / f"{suite}_eval.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["MUJOCO_GL"] = "egl"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            command = [
                str(args.eval_python),
                str(project_root / "scripts" / "evaluate_libero_closed_loop.py"),
                "--dataset-root", str(args.dataset_root / suite),
                "--suite", suite,
                "--rollouts-per-task", str(args.rollouts_per_task),
                "--max-steps", str(max_steps),
                "--replan-steps", str(args.replan_steps),
                "--num-views", "2",
                "--port", str(port),
                "--seed", str(args.seed),
                "--output-dir", str(suite_output),
            ]
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            clients.append((suite, process, log_file))

        failures = []
        for suite, process, log_file in clients:
            returncode = process.wait()
            log_file.close()
            if returncode != 0:
                failures.append((suite, returncode))
        if failures:
            raise RuntimeError(f"LIBERO evaluation failures: {failures}")

        suites = {}
        total_success = 0.0
        total_rollouts = 0
        for suite, _ in SUITES:
            summary_path = args.output_dir / suite / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            num_rollouts = int(summary["num_rollouts"])
            success_rate = float(summary["success_rate"])
            suites[suite] = {
                "num_rollouts": num_rollouts,
                "success_rate": success_rate,
                "success_rate_by_task": summary["success_rate_by_task"],
            }
            total_success += success_rate * num_rollouts
            total_rollouts += num_rollouts
        aggregate = {
            "checkpoint": str(args.checkpoint),
            "num_rollouts": total_rollouts,
            "success_rate": total_success / total_rollouts,
            "suites": suites,
        }
        path = args.output_dir / "summary.json"
        path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(aggregate, indent=2), flush=True)
        print(path, flush=True)
    finally:
        for _, process, log_file in clients:
            if not log_file.closed:
                log_file.close()
            terminate(process)
        for process in servers:
            terminate(process)
        for log_file in server_logs:
            if not log_file.closed:
                log_file.close()


if __name__ == "__main__":
    main()

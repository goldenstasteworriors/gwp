#!/usr/bin/env python3
"""Run closed-loop LIBERO rollouts against a GWP socket server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIGA_MODELS = PROJECT_ROOT / "third_party" / "giga-models"
if str(GIGA_MODELS) not in sys.path:
    sys.path.insert(0, str(GIGA_MODELS))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

from giga_models.sockets import RobotInferenceClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--rollouts-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-views", type=int, choices=(1, 2), default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11421)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-num-inference-steps", type=int, required=True)
    parser.add_argument(
        "--policy-model-dtype", choices=("bf16", "fp16", "fp32"), required=True
    )
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def task_prompt_files(
    dataset_root: Path, suite: Any
) -> tuple[dict[int, Path], dict[int, dict[str, Any]]]:
    """Match dataset prompts to suite tasks by language, never by integer order."""

    paths = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    frames = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "task_index"]) for path in paths],
        ignore_index=True,
    )
    metadata_paths = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"No episode metadata under {dataset_root}")
    metadata = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "tasks"]) for path in metadata_paths],
        ignore_index=True,
    )
    episode_languages: dict[int, str] = {}
    for _, row in metadata.iterrows():
        episode = int(row["episode_index"])
        tasks = list(row["tasks"])
        if len(tasks) != 1 or not str(tasks[0]).strip():
            raise ValueError(f"Episode {episode} must have one language task: {tasks}")
        episode_languages[episode] = str(tasks[0]).strip()

    dataset_by_language: dict[str, tuple[int, int, Path]] = {}
    for dataset_task_id, group in frames.groupby("task_index", sort=True):
        episodes = sorted({int(value) for value in group["episode_index"]})
        languages = {episode_languages[episode] for episode in episodes}
        if len(languages) != 1:
            raise ValueError(
                f"Dataset task {dataset_task_id} has multiple languages: {sorted(languages)}"
            )
        language = next(iter(languages))
        if language in dataset_by_language:
            raise ValueError(f"Duplicate dataset task language: {language!r}")
        episode = episodes[0]
        prompt = dataset_root / "t5_embedding" / f"episode_{episode:06d}.pt"
        if not prompt.is_file():
            raise FileNotFoundError(prompt)
        dataset_by_language[language] = (int(dataset_task_id), episode, prompt)

    prompts: dict[int, Path] = {}
    mapping: dict[int, dict[str, Any]] = {}
    for suite_task_id in range(suite.n_tasks):
        language = str(suite.get_task(suite_task_id).language).strip()
        if language not in dataset_by_language:
            raise ValueError(f"Suite task language is absent from dataset: {language!r}")
        dataset_task_id, episode, prompt = dataset_by_language[language]
        prompts[suite_task_id] = prompt
        mapping[suite_task_id] = {
            "dataset_task_id": dataset_task_id,
            "prompt_episode_index": episode,
            "task_language": language,
            "prompt_path": str(prompt.resolve()),
        }
    return prompts, mapping


def load_prompt(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("t5_embedding", next(iter(value.values())))
    value = torch.as_tensor(value).float()
    return value[:64]


def policy_state(observation: dict[str, Any]) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    robot = observation["robot_state"]
    position = np.asarray(robot["eef"]["pos"], dtype=np.float32)
    axis_angle = np.asarray(quat2axisangle(robot["eef"]["quat"]), dtype=np.float32)
    gripper = np.asarray(robot["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([position, axis_angle, gripper], axis=0)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def configure_libero_path(dataset_root: Path) -> Path:
    """Resolve a non-interactive LIBERO config before importing LIBERO."""

    configured = os.environ.get("LIBERO_CONFIG_PATH")
    if configured:
        config_path = Path(configured).expanduser().resolve() / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"LIBERO_CONFIG_PATH does not contain config.yaml: {config_path}"
            )
        return config_path
    resolved = dataset_root.resolve()
    for parent in (resolved, *resolved.parents):
        config_path = parent / "libero_config" / "config.yaml"
        if config_path.is_file():
            os.environ["LIBERO_CONFIG_PATH"] = str(config_path.parent)
            return config_path
    raise FileNotFoundError(
        f"Could not discover libero_config/config.yaml from {dataset_root}"
    )


def load_libero_runtime() -> tuple[Any, Any]:
    """Load LeRobot's wrapper or an equivalent official-LIBERO fallback.

    The simulator's supported Python 3.8 environment cannot install current
    LeRobot.  Keep evaluation semantics fixed by mirroring the subset of
    ``lerobot.envs.libero.LiberoEnv`` used by this script: benchmark initial
    states, ten settling no-ops, dual raw cameras, relative control, and the
    same robot-state fields.
    """

    try:
        from lerobot.envs.libero import LiberoEnv, _get_suite

        return LiberoEnv, _get_suite
    except ModuleNotFoundError as error:
        if error.name != "lerobot":
            raise

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    def get_suite(name: str) -> Any:
        suites = benchmark.get_benchmark_dict()
        if name not in suites:
            raise ValueError(
                f"Unknown LIBERO suite {name!r}; available: {sorted(suites)}"
            )
        suite = suites[name]()
        if not getattr(suite, "tasks", None):
            raise ValueError(f"Suite {name!r} has no tasks")
        return suite

    class OfficialLiberoEnv:
        """Minimal single-environment equivalent of LeRobot's wrapper."""

        def __init__(
            self,
            task_suite: Any,
            task_id: int,
            task_suite_name: str = "",
            episode_length: int = 1000,
            camera_name: str = "agentview_image",
            obs_type: str = "pixels_agent_pos",
            init_states: bool = True,
            episode_index: int = 0,
            n_envs: int = 1,
            num_steps_wait: int = 10,
        ) -> None:
            del task_suite_name
            if obs_type != "pixels_agent_pos":
                raise ValueError(f"Unsupported fallback observation type: {obs_type}")
            if n_envs != 1:
                raise ValueError("Closed-loop fallback supports exactly one environment")
            camera_names = [
                value.strip() for value in camera_name.split(",") if value.strip()
            ]
            if not camera_names or len(camera_names) > 2:
                raise ValueError(f"Expected one or two camera names, got {camera_names}")

            task = task_suite.get_task(task_id)
            self.task_description = task.language
            self._camera_names = camera_names
            self._episode_length = int(episode_length)
            self._num_steps_wait = int(num_steps_wait)
            self._steps = 0
            self._initial_state = None
            if init_states:
                initial_states = task_suite.get_task_init_states(task_id)
                self._initial_state = np.asarray(
                    initial_states[episode_index % len(initial_states)], dtype=np.float64
                )
            bddl_path = (
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            )
            self._env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=256,
                camera_widths=256,
                control_freq=20,
            )
            self._env.reset()

        def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
            images = {"image": raw_obs[self._camera_names[0]]}
            if len(self._camera_names) == 2:
                images["image2"] = raw_obs[self._camera_names[1]]
            return {
                "pixels": images,
                "robot_state": {
                    "eef": {
                        "pos": raw_obs["robot0_eef_pos"],
                        "quat": raw_obs["robot0_eef_quat"],
                        "mat": self._env.robots[0].controller.ee_ori_mat,
                    },
                    "gripper": {
                        "qpos": raw_obs["robot0_gripper_qpos"],
                        "qvel": raw_obs["robot0_gripper_qvel"],
                    },
                    "joints": {
                        "pos": raw_obs["robot0_joint_pos"],
                        "vel": raw_obs["robot0_joint_vel"],
                    },
                },
            }

        def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, bool]]:
            self._env.seed(seed)
            raw_obs = self._env.reset()
            if self._initial_state is not None:
                raw_obs = self._env.set_init_state(self._initial_state)
            for _ in range(self._num_steps_wait):
                raw_obs, _, _, _ = self._env.step([0, 0, 0, 0, 0, 0, -1])
            for robot in self._env.robots:
                robot.controller.use_delta = True
            self._steps = 0
            return self._format_raw_obs(raw_obs), {"is_success": False}

        def step(
            self, action: np.ndarray
        ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            if action.ndim != 1:
                raise ValueError(f"Expected a 1-D action, got {action.shape}")
            raw_obs, reward, done, info = self._env.step(action)
            self._steps += 1
            success = bool(self._env.check_success())
            info.update({"done": done, "is_success": success})
            terminated = bool(done or success)
            truncated = self._steps >= self._episode_length and not terminated
            return self._format_raw_obs(raw_obs), reward, terminated, truncated, info

        def close(self) -> None:
            self._env.close()

    return OfficialLiberoEnv, get_suite


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = RobotInferenceClient(host=args.host, port=args.port)

    configure_libero_path(args.dataset_root)
    LiberoEnv, _get_suite = load_libero_runtime()

    suite = _get_suite(args.suite)
    prompts, prompt_mapping = task_prompt_files(args.dataset_root, suite)
    rows: list[dict[str, Any]] = []
    prompt_sent: set[int] = set()
    for task_id in args.task_ids:
        if task_id < 0 or task_id >= suite.n_tasks:
            raise ValueError(f"Invalid task id {task_id}; suite has {suite.n_tasks} tasks")
        prompt = load_prompt(prompts[task_id])
        for rollout in range(args.rollouts_per_task):
            camera_names = (
                "agentview_image,robot0_eye_in_hand_image"
                if args.num_views == 2
                else "agentview_image"
            )
            env = LiberoEnv(
                suite,
                task_id,
                args.suite,
                episode_length=args.max_steps,
                camera_name=camera_names,
                obs_type="pixels_agent_pos",
                init_states=True,
                episode_index=rollout,
                n_envs=1,
                num_steps_wait=10,
            )
            observation, _ = env.reset(seed=args.seed + rollout)
            executed: list[np.ndarray] = []
            success = False
            step = 0
            chunk = 0
            while step < args.max_steps and not success:
                request = {
                    # Official LeRobot LIBERO recordings use the visualization
                    # convention, which flips the raw robosuite observation on
                    # both image axes. Match the training-video orientation.
                    "image": observation["pixels"]["image"][::-1, ::-1].copy(),
                    "state": policy_state(observation),
                    "prompt_key": f"{args.suite}:{task_id}",
                    "task_id": task_id,
                    "episode_index": rollout,
                    "chunk_index": chunk,
                }
                if args.num_views == 2:
                    request["wrist_image"] = observation["pixels"]["image2"][::-1, ::-1].copy()
                if task_id not in prompt_sent:
                    request["prompt_embedding"] = prompt
                    prompt_sent.add(task_id)
                action_chunk = np.asarray(client.inference(request), dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                    raise ValueError(f"Policy returned malformed actions: {action_chunk.shape}")
                for action in action_chunk[: args.replan_steps, :7]:
                    observation, reward, terminated, truncated, _ = env.step(
                        np.clip(action, -1.0, 1.0)
                    )
                    executed.append(action.copy())
                    step += 1
                    success = success or bool(reward > 0.5)
                    if success or terminated or truncated or step >= args.max_steps:
                        break
                chunk += 1

            rollout_dir = args.output_dir / f"task_{task_id:02d}" / f"rollout_{rollout:02d}"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            np.save(rollout_dir / "executed_actions.npy", np.asarray(executed, dtype=np.float32))
            row = {
                "suite": args.suite,
                "task_id": task_id,
                "task_description": env.task_description,
                "rollout": rollout,
                "success": success,
                "steps": step,
                "chunks": chunk,
            }
            (rollout_dir / "result.json").write_text(json.dumps(to_jsonable(row), indent=2) + "\n")
            rows.append(row)
            print(json.dumps(to_jsonable(row)), flush=True)
            env.close()

    by_task = {
        str(task_id): float(np.mean([row["success"] for row in rows if row["task_id"] == task_id]))
        for task_id in args.task_ids
    }
    summary = {
        "schema_version": "libero-closed-loop-v3",
        "suite": args.suite,
        "num_rollouts": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "success_rate_by_task": by_task,
        "protocol": {
            "dataset_root": str(args.dataset_root.resolve()),
            "task_ids": [int(task_id) for task_id in args.task_ids],
            "rollouts_per_task": int(args.rollouts_per_task),
            "max_steps": int(args.max_steps),
            "replan_steps": int(args.replan_steps),
            "num_views": int(args.num_views),
            "seed": int(args.seed),
            "policy_checkpoint": str(args.policy_checkpoint.resolve()),
            "policy_num_inference_steps": args.policy_num_inference_steps,
            "policy_model_dtype": args.policy_model_dtype,
            "policy_seed": args.policy_seed,
            "task_prompt_mapping": {
                str(key): value for key, value in prompt_mapping.items()
            },
        },
        "rollouts": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n")
    print(json.dumps(to_jsonable({key: value for key, value in summary.items() if key != "rollouts"}), indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()

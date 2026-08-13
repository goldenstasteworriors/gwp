#!/usr/bin/env python3
"""Replay matched expert actions and render masks for real-video probe fitting.

This cache is separate from the policy-action counterfactual cache. Its masks
are candidate labels for the logged demonstration frames and must pass a later
RGB/state registration audit before they train a frozen latent probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from generate_libero_counterfactual_mask_cache import (
    IMAGE_ORIENTATION,
    capture_state,
    configure_libero_path,
    filter_raw_demo_actions,
    flatten_observation_state,
    jsonable,
    robot_geom_ids,
    sha256_array,
    sha256_file,
    stack_captures,
)


SCHEMA_VERSION = "libero-demonstration-robot-mask-v1"
COUNTERFACTUAL_SCHEMA = "libero-counterfactual-robot-mask-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--source-state-atol", type=float, default=1e-5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.source_state_atol <= 0:
        raise ValueError("--source-state-atol must be positive")

    source_root = args.counterfactual_cache.resolve()
    source_manifest_path = source_root / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("schema_version") != COUNTERFACTUAL_SCHEMA:
        raise ValueError(
            f"Expected {COUNTERFACTUAL_SCHEMA}, got "
            f"{source_manifest.get('schema_version')!r}"
        )
    if source_manifest.get("image_orientation") != IMAGE_ORIENTATION:
        raise ValueError("Counterfactual cache has an incompatible image orientation")
    if source_manifest.get("rgb_supervision") is not None:
        raise ValueError("Counterfactual selection cache unexpectedly contains RGB targets")
    horizons = [int(value) for value in source_manifest.get("horizons", [])]
    if horizons != [0, 12, 24, 36, 48]:
        raise ValueError(f"Unexpected horizons: {horizons}")
    records = list(source_manifest.get("episodes", []))
    if not records:
        raise ValueError("Counterfactual cache has no episodes")

    dataset_root = Path(str(source_manifest["dataset_root"])).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if sha256_file(info_path) != source_manifest.get("dataset_info_sha256"):
        raise ValueError("Dataset metadata changed after counterfactual cache generation")
    libero_config_path = configure_libero_path(dataset_root)

    from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

    suite_name = str(source_manifest["suite"])
    suite = _get_suite(suite_name)
    args.output_dir.mkdir(parents=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir()

    output_records: list[dict[str, Any]] = []
    for source_record in records:
        episode = int(source_record["episode_index"])
        source_cache_path = source_root / str(source_record["cache_file"])
        if sha256_file(source_cache_path) != source_record.get("cache_sha256"):
            raise ValueError(f"Counterfactual episode cache changed: {source_cache_path}")
        with np.load(source_cache_path, allow_pickle=False) as source_cache:
            raw_initial_state = source_cache["raw_initial_state"].copy()
            source_observation_state = source_cache["source_observation_state"].copy()

        raw_path = Path(str(source_record["raw_hdf5_path"])).resolve()
        if sha256_file(raw_path) != source_record.get("raw_hdf5_sha256"):
            raise ValueError(f"Raw demonstration HDF5 changed: {raw_path}")
        demo_name = str(source_record["raw_demo_name"])
        with h5py.File(raw_path, "r") as raw_file:
            demo_path = f"data/{demo_name}"
            if demo_path not in raw_file:
                raise KeyError(f"Missing {demo_path} in {raw_path}")
            raw_demo = raw_file[demo_path]
            expert_actions = filter_raw_demo_actions(raw_demo["actions"][()])
            stored_initial_state = np.asarray(raw_demo["states"][0], dtype=np.float64)
        if sha256_array(stored_initial_state) != source_record.get(
            "raw_initial_state_sha256"
        ):
            raise ValueError(f"Raw initial-state provenance mismatch for episode {episode}")
        if not np.array_equal(stored_initial_state, raw_initial_state):
            raise ValueError(f"Raw initial state changed for episode {episode}")
        if len(expert_actions) < max(horizons):
            raise ValueError(
                f"Episode {episode} has {len(expert_actions)} expert actions, "
                f"need {max(horizons)}"
            )

        env = LiberoEnv(
            task_suite=suite,
            task_id=int(source_record["libero_task_index"]),
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image",
            init_states=False,
            n_envs=1,
            num_steps_wait=0,
        )
        try:
            env.reset(seed=args.seed)
            raw_obs = env._env.set_init_state(raw_initial_state)
            for _ in range(int(source_manifest["num_steps_wait"])):
                raw_obs, _, _, _ = env._env.step(get_libero_dummy_action())
            obs = env._format_raw_obs(raw_obs)
            rendered_initial_state = flatten_observation_state(obs)
            source_state_max_abs = float(
                np.max(np.abs(rendered_initial_state - source_observation_state))
            )
            if source_state_max_abs > args.source_state_atol:
                raise ValueError(
                    f"Episode {episode} frame-0 state mismatch: {source_state_max_abs} "
                    f"> {args.source_state_atol}"
                )
            robot_ids, gripper_ids = robot_geom_ids(env)

            def capture(current_obs: dict[str, Any]) -> dict[str, np.ndarray]:
                return capture_state(
                    env,
                    current_obs,
                    robot_ids,
                    gripper_ids,
                    str(source_manifest["camera_name"]),
                    int(source_manifest["image_size"][1]),
                    int(source_manifest["image_size"][0]),
                )

            captures = {0: capture(obs)}
            for step in range(1, max(horizons) + 1):
                obs, *_ = env.step(expert_actions[step - 1])
                if step in horizons:
                    captures[step] = capture(obs)
        finally:
            env.close()

        arrays = stack_captures([captures[horizon] for horizon in horizons])
        output_path = episodes_dir / f"episode_{episode:06d}.npz"
        np.savez_compressed(
            output_path,
            horizons=np.asarray(horizons, dtype=np.int32),
            expert_actions=expert_actions[: max(horizons)],
            raw_initial_state=raw_initial_state,
            source_observation_state=source_observation_state,
            rendered_initial_observation_state=rendered_initial_state,
            **arrays,
        )
        output_record = {
            key: source_record[key]
            for key in (
                "episode_index",
                "source_frame_index",
                "task_index",
                "libero_task_index",
                "task_language",
                "raw_demo_name",
                "raw_hdf5_path",
                "raw_hdf5_sha256",
                "raw_initial_state_sha256",
            )
        }
        output_record.update(
            {
                "source_counterfactual_cache_file": str(source_cache_path),
                "source_counterfactual_cache_sha256": sha256_file(source_cache_path),
                "source_state_max_abs": source_state_max_abs,
                "cache_file": str(output_path.relative_to(args.output_dir)),
                "cache_sha256": sha256_file(output_path),
                "robot_pixels": [int(mask.sum()) for mask in arrays["robot_mask"]],
            }
        )
        output_records.append(output_record)
        print(json.dumps(jsonable(output_record)), flush=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_semantics": (
            "robot embodiment on the matched logged demonstration trajectory; "
            "candidate labels require a separate real-RGB registration audit before "
            "probe training"
        ),
        "rgb_supervision": None,
        "suite": suite_name,
        "dataset_root": str(dataset_root),
        "dataset_info_sha256": sha256_file(info_path),
        "source_counterfactual_cache": str(source_root),
        "source_counterfactual_manifest_sha256": sha256_file(source_manifest_path),
        "libero_config_path": str(libero_config_path),
        "libero_config_sha256": sha256_file(libero_config_path),
        "source_frame_index": 0,
        "horizons": horizons,
        "camera_name": source_manifest["camera_name"],
        "image_orientation": IMAGE_ORIENTATION,
        "image_size": source_manifest["image_size"],
        "num_steps_wait": source_manifest["num_steps_wait"],
        "seed": args.seed,
        "source_state_atol": args.source_state_atol,
        "num_episodes": len(output_records),
        "episodes": output_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()

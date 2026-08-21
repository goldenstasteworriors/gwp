#!/usr/bin/env python3
"""Bypass the model and audit SONIC training/deployment normalization end to end."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "third_party" / "giga-datasets",
    PROJECT_ROOT / "third_party" / "giga-models",
    PROJECT_ROOT / "third_party" / "giga-train",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

import world_action_model.datasets  # noqa: F401,E402 - register dataset classes
import world_action_model.transforms  # noqa: F401,E402 - register transforms
from giga_datasets import load_dataset  # noqa: E402
from giga_train import build_transform, load_config  # noqa: E402
from world_action_model.inference.sonic_history_policy import (  # noqa: E402
    decode_sonic_action,
    encode_sonic_state_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def scalar_int(value) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def find_sample_index(dataset, episode: int, frame: int) -> int:
    dataset.open()
    backend = getattr(dataset, "dataset", None)
    table = getattr(backend, "hf_dataset", None)
    if table is None:
        raise TypeError(
            "The audit expects one WAMLeRobotDataset entry with an hf_dataset backend"
        )
    episodes = table["episode_index"]
    frames = table["frame_index"]
    for index, (episode_value, frame_value) in enumerate(zip(episodes, frames)):
        if scalar_int(episode_value) == episode and scalar_int(frame_value) == frame:
            return index
    raise KeyError(
        f"episode/frame {episode}/{frame} is not present in the training split"
    )


def to_numpy(value, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def metric_block(error: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_abs": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def main() -> None:
    args = parse_args()
    if args.episode < 0 or args.frame < 0:
        raise SystemExit("episode and frame must be non-negative")
    training_config_path = args.training_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(str(training_config_path))
    train_config = config["dataloaders"]["train"]
    data_configs = train_config["data_or_config"]
    if not isinstance(data_configs, list) or len(data_configs) != 1:
        raise ValueError(
            "Normalization audit currently requires exactly one training dataset entry"
        )
    dataset_config = data_configs[0]
    transform_config = train_config["transform"]
    dataset = load_dataset(dataset_config)
    transform = build_transform(transform_config)
    sample_index = find_sample_index(dataset, args.episode, args.frame)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw = dataset[sample_index]
    processed = transform(raw)

    physical_dim = len(transform_config["delta_mask_by_embodiment_id"]["0"])
    model_dim = int(transform_config["model_action_dim"])
    action_horizon = int(transform_config["num_frames"])
    delta_mask = np.asarray(
        transform_config["delta_mask_by_embodiment_id"]["0"], dtype=bool
    )
    if physical_dim != 13 or model_dim != 32:
        raise ValueError(
            f"Expected SONIC 13-D physical/32-D model actions, got {physical_dim}/{model_dim}"
        )

    raw_state = to_numpy(raw[transform_config.get("state_key", "observation.state")])
    raw_action = to_numpy(raw[transform_config.get("action_key", "action")])
    if raw_state.ndim == 1:
        raw_state = raw_state[None]
    if raw_action.ndim == 1:
        raw_action = raw_action[None]
    raw_state = raw_state[:, :physical_dim]
    raw_action = raw_action[:action_horizon, :physical_dim]
    current_state = raw_state[-1]
    training_action = to_numpy(processed["action"])
    training_state = to_numpy(processed["state"])

    norm_specs = transform_config["norm_path"]
    if isinstance(norm_specs, (str, os.PathLike)):
        norm_path = Path(norm_specs)
    else:
        first_norm_spec = norm_specs[0]
        norm_path = Path(
            first_norm_spec["path"]
            if isinstance(first_norm_spec, dict)
            else first_norm_spec
        )
    norm_path = norm_path.expanduser().resolve()
    stats = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]
    state_low = torch.tensor(stats["observation.state"]["q01"], dtype=torch.float32)
    state_high = torch.tensor(stats["observation.state"]["q99"], dtype=torch.float32)
    action_low = torch.tensor(stats["action"]["q01"], dtype=torch.float32)
    action_high = torch.tensor(stats["action"]["q99"], dtype=torch.float32)

    deployment_state, deployment_current = encode_sonic_state_tokens(
        raw_state,
        state_low=state_low,
        state_high=state_high,
        physical_action_dim=physical_dim,
        model_action_dim=model_dim,
        required_state_history=int(transform_config.get("state_history_tokens", 1)),
        append_state_velocity_token=bool(
            transform_config.get("append_state_velocity_token", False)
        ),
        append_episode_progress_token=bool(
            transform_config.get("append_episode_progress_token", False)
        ),
        episode_progress=float(to_numpy(raw.get("episode_progress", 0.0))),
    )
    decoded_action = decode_sonic_action(
        training_action,
        action_low=action_low,
        action_high=action_high,
        delta_mask=delta_mask,
        current_state=deployment_current,
        physical_action_dim=physical_dim,
    ).cpu().numpy()

    represented_action = raw_action.copy()
    represented_action[:, delta_mask] -= current_state[delta_mask]
    low = action_low[:physical_dim].numpy()
    high = action_high[:physical_dim].numpy()
    unclamped_normalized = (represented_action - low) / np.maximum(high - low, 1e-8) * 2.0 - 1.0
    expected_training_action = np.clip(unclamped_normalized, -1.0, 1.0)
    clipped_entries = np.abs(unclamped_normalized) > 1.0
    roundtrip_error = decoded_action - raw_action
    normalized_action_error = training_action[:, :physical_dim] - expected_training_action
    state_token_error = training_state - deployment_state.cpu().numpy()

    frame_indices = np.arange(args.frame, args.frame + len(raw_action), dtype=np.int64)
    timestamps = frame_indices.astype(np.float64) / 50.0
    np.savez_compressed(
        output,
        predicted_state=decoded_action.astype(np.float32),
        reference_state=raw_action.astype(np.float32),
        reference_action=raw_action.astype(np.float32),
        frame_indices=frame_indices,
        timestamps=timestamps,
        raw_current_state=current_state.astype(np.float32),
        represented_action=represented_action.astype(np.float32),
        unclamped_normalized_action=unclamped_normalized.astype(np.float32),
        training_normalized_action=training_action.astype(np.float32),
        expected_training_normalized_action=expected_training_action.astype(np.float32),
        training_normalized_state=training_state.astype(np.float32),
        deployment_normalized_state=deployment_state.cpu().numpy().astype(np.float32),
        normalization_clipped_entries=clipped_entries,
        roundtrip_error=roundtrip_error.astype(np.float32),
    )

    per_dimension_max = np.max(np.abs(roundtrip_error), axis=0)
    summary = {
        "audit": "training_transform_to_deployment_decoder_model_bypass",
        "training_config": str(training_config_path),
        "dataset": str(Path(dataset_config["data_path"]).resolve()),
        "norm_stats": str(norm_path),
        "episode": args.episode,
        "frame": args.frame,
        "sample_index_in_training_split": sample_index,
        "action_horizon": action_horizon,
        "delta_mask": delta_mask.tolist(),
        "training_normalized_action_vs_formula": metric_block(normalized_action_error),
        "training_state_vs_deployment_state_tokens": metric_block(state_token_error),
        "decoded_vs_raw_action_all": metric_block(roundtrip_error),
        "decoded_vs_raw_action_arm": metric_block(roundtrip_error[:, :7]),
        "decoded_vs_raw_action_hand": metric_block(roundtrip_error[:, 7:13]),
        "normalization_clipped_entry_count": int(np.count_nonzero(clipped_entries)),
        "normalization_total_physical_entries": int(clipped_entries.size),
        "normalization_clipped_fraction": float(np.mean(clipped_entries)),
        "per_dimension_max_abs_roundtrip_error": per_dimension_max.tolist(),
        "roundtrip_is_exact_without_clipping": bool(
            np.max(np.abs(roundtrip_error[~clipped_entries])) < 1e-5
            if np.any(~clipped_entries)
            else True
        ),
        "rollout": str(output),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format_version": 1,
        "audit": summary["audit"],
        "experiment": "m00_training_normalization_roundtrip",
        "transformer": None,
        "norm_stats": str(norm_path),
        "evaluation_dataset": summary["dataset"],
        "evaluation_episodes": [args.episode],
        "action_horizon": action_horizon,
        "physical_action_dim": physical_dim,
        "model_action_dim": model_dim,
        "delta_mask": delta_mask.tolist(),
        "fps": 50,
        "rollout_semantics": {
            "model_bypassed": True,
            "left_arm_channels_0_7": "training_delta_target_decoded_by_deployment_then_integrated",
            "left_inspire_channels_7_13": "training_absolute_target_decoded_by_deployment",
            "predicted_state": "deployment-decoded training target",
            "reference_state": "raw dataset action target",
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

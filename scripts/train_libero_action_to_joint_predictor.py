#!/usr/bin/env python3
"""Train a simulator-free action-chunk -> joint-trajectory predictor.

The model consumes the current arm/gripper q and a 48-step LeRobot action
chunk.  It predicts the 48 realized joint configurations without importing or
querying LIBERO / MuJoCo.  Held-out episode predictions are exported using the
direct-q mask renderer's input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_SCHEMA = "libero-logged-action-joint-trajectory-v1"
REPORT_SCHEMA = "libero-action-to-joint-predictor-v1"
HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-episodes-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def set_seed(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class ActionToJointGRU(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.initial_hidden = nn.Sequential(
            nn.Linear(9, hidden_dim * num_layers),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            input_size=7 + 9,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, q0_normalized: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = actions.shape
        initial = self.initial_hidden(q0_normalized).reshape(
            batch, self.num_layers, self.hidden_dim
        )
        initial = initial.permute(1, 0, 2).contiguous()
        context = q0_normalized[:, None, :].expand(batch, steps, 9)
        hidden, _ = self.gru(torch.cat([actions, context], dim=-1), initial)
        return self.output(hidden)


def load_episode_arrays(
    split_root: Path, record: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = split_root / str(record["cache_file"])
    if sha256_file(cache_path) != record["cache_sha256"]:
        raise ValueError(f"Joint supervision cache changed: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as arrays:
        actions = np.asarray(arrays["policy_actions_lerobot"], dtype=np.float32)
        arm = np.asarray(arrays["arm_qpos"], dtype=np.float32)
        gripper = np.asarray(arrays["gripper_qpos"], dtype=np.float32)
    q = np.concatenate([arm, gripper], axis=-1)
    if actions.shape != (len(q), 7) or q.shape[1] != 9:
        raise ValueError(f"Unexpected action/q shape: {cache_path}")
    return actions, q


def partition_train_records(
    records: list[dict[str, Any]], validation_per_task: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(int(record["task_index"]), []).append(record)
    fit: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for task, task_records in sorted(by_task.items()):
        task_records = sorted(task_records, key=lambda value: int(value["episode_index"]))
        if len(task_records) <= validation_per_task:
            raise ValueError(f"Task {task} has too few train episodes")
        validation.extend(task_records[-validation_per_task:])
        fit.extend(task_records[:-validation_per_task])
    return fit, validation


def build_windows(
    split_root: Path,
    records: list[dict[str, Any]],
    horizon: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    q0_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    for record in records:
        actions, q = load_episode_arrays(split_root, record)
        for start in range(len(actions) - horizon):
            q0_values.append(q[start])
            action_values.append(actions[start : start + horizon])
            target_values.append(q[start + 1 : start + horizon + 1])
            keys.append((int(record["episode_index"]), start))
    return (
        np.stack(q0_values).astype(np.float32),
        np.stack(action_values).astype(np.float32),
        np.stack(target_values).astype(np.float32),
        keys,
    )


def compute_q_statistics(
    split_root: Path, records: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    for record in records:
        _, q = load_episode_arrays(split_root, record)
        values.append(q)
    combined = np.concatenate(values, axis=0).astype(np.float64)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    if np.any(std < 1e-6):
        raise ValueError(f"Degenerate q standard deviation: {std}")
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_q(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean) / std).astype(np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    q0: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    q_mean: np.ndarray,
    q_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    weighted_loss = 0.0
    count = 0
    for start in range(0, len(q0), batch_size):
        stop = min(len(q0), start + batch_size)
        q0_tensor = torch.from_numpy(normalize_q(q0[start:stop], q_mean, q_std)).to(device)
        action_tensor = torch.from_numpy(actions[start:stop]).to(device)
        target_normalized = normalize_q(targets[start:stop], q_mean, q_std)
        output = model(q0_tensor, action_tensor)
        output_np = output.cpu().numpy()
        loss = float(np.mean((output_np - target_normalized) ** 2))
        weighted_loss += loss * (stop - start)
        count += stop - start
        predictions.append(output_np * q_std + q_mean)
    return weighted_loss / count, np.concatenate(predictions, axis=0)


def metric_summary(
    predictions: np.ndarray, targets: np.ndarray, keys: list[tuple[int, int]]
) -> dict[str, Any]:
    error = predictions - targets
    abs_error = np.abs(error)
    first_window = np.asarray([start == 0 for _, start in keys], dtype=bool)

    def group(mask: np.ndarray) -> dict[str, Any]:
        selected = abs_error[mask]
        result: dict[str, Any] = {
            "num_windows": int(mask.sum()),
            "arm_q_mae_rad_all_steps": float(selected[..., :7].mean()),
            "gripper_q_mae_all_steps": float(selected[..., 7:].mean()),
            "by_horizon": {},
        }
        for horizon in HORIZONS[1:]:
            index = horizon - 1
            result["by_horizon"][str(horizon)] = {
                "arm_q_mae_rad": float(selected[:, index, :7].mean()),
                "arm_q_rmse_rad": float(
                    np.sqrt(np.mean(error[mask, index, :7] ** 2))
                ),
                "gripper_q_mae": float(selected[:, index, 7:].mean()),
            }
        return result

    return {
        "all_windows": group(np.ones(len(keys), dtype=bool)),
        "episode_frame0_windows": group(first_window),
    }


def export_frame0_predictions(
    output_root: Path,
    heldout_root: Path,
    records: list[dict[str, Any]],
    model: nn.Module,
    q_mean: np.ndarray,
    q_std: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    predictions_root = output_root / "predictions"
    predictions_root.mkdir()
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for record in records:
            actions, q = load_episode_arrays(heldout_root, record)
            if len(actions) <= 48:
                raise ValueError(f"Held-out episode too short: {record}")
            q0_tensor = torch.from_numpy(
                normalize_q(q[None, 0], q_mean, q_std)
            ).to(device)
            action_tensor = torch.from_numpy(actions[None, :48]).to(device)
            predicted = model(q0_tensor, action_tensor).cpu().numpy()[0]
            predicted = predicted * q_std + q_mean
            dense = np.concatenate([q[0:1], predicted], axis=0).astype(np.float32)
            episode = int(record["episode_index"])
            episode_root = predictions_root / f"episode_{episode:06d}"
            episode_root.mkdir()
            arm_path = episode_root / "predicted_joint_qpos.npy"
            gripper_path = episode_root / "predicted_gripper_qpos.npy"
            np.save(arm_path, dense[:, :7])
            np.save(gripper_path, dense[:, 7:])
            rows.append(
                {
                    "episode_index": episode,
                    "task_index": int(record["task_index"]),
                    "arm_path": str(arm_path.relative_to(output_root)),
                    "arm_sha256": sha256_file(arm_path),
                    "gripper_path": str(gripper_path.relative_to(output_root)),
                    "gripper_sha256": sha256_file(gripper_path),
                    "layout": "dense_49_rows_q0_plus_48_predicted_post_action_states",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.device != "cpu":
        raise ValueError("This gate is intentionally CPU-only; use --device cpu")
    if min(
        args.hidden_dim,
        args.num_layers,
        args.epochs,
        args.batch_size,
        args.validation_episodes_per_task,
        args.num_threads,
    ) <= 0:
        raise ValueError("Positive model/training arguments required")
    set_seed(args.seed, args.num_threads)
    root_manifest_path = args.dataset_root / "manifest.json"
    root_manifest = load_json(root_manifest_path)
    if root_manifest.get("schema_version") != DATA_SCHEMA:
        raise ValueError(f"Unexpected joint dataset schema: {root_manifest_path}")
    if root_manifest.get("simulator_runtime_imported") is not False:
        raise ValueError("Joint dataset imported simulator runtime")
    if int(root_manifest.get("simulator_step_calls", -1)) != 0:
        raise ValueError("Joint dataset contains simulator steps")
    train_root = args.dataset_root / "train"
    heldout_root = args.dataset_root / "heldout"
    train_manifest = load_json(train_root / "manifest.json")
    heldout_manifest = load_json(heldout_root / "manifest.json")
    train_records = list(train_manifest["episodes"])
    heldout_records = list(heldout_manifest["episodes"])
    fit_records, validation_records = partition_train_records(
        train_records, args.validation_episodes_per_task
    )

    q_mean, q_std = compute_q_statistics(train_root, fit_records)
    fit_q0, fit_actions, fit_targets, _ = build_windows(train_root, fit_records)
    val_q0, val_actions, val_targets, val_keys = build_windows(
        train_root, validation_records
    )
    held_q0, held_actions, held_targets, held_keys = build_windows(
        heldout_root, heldout_records
    )
    fit_dataset = TensorDataset(
        torch.from_numpy(normalize_q(fit_q0, q_mean, q_std)),
        torch.from_numpy(fit_actions),
        torch.from_numpy(normalize_q(fit_targets, q_mean, q_std)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        fit_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    device = torch.device("cpu")
    model = ActionToJointGRU(args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        examples = 0
        for q0_batch, action_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(q0_batch.to(device), action_batch.to(device))
            loss = torch.mean((output - target_batch.to(device)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.item()) * len(q0_batch)
            examples += len(q0_batch)
        train_loss = running / examples
        validation_loss, _ = evaluate(
            model,
            val_q0,
            val_actions,
            val_targets,
            q_mean,
            q_std,
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_loss, validation_predictions = evaluate(
        model,
        val_q0,
        val_actions,
        val_targets,
        q_mean,
        q_std,
        args.batch_size,
        device,
    )
    heldout_loss, heldout_predictions = evaluate(
        model,
        held_q0,
        held_actions,
        held_targets,
        q_mean,
        q_std,
        args.batch_size,
        device,
    )
    copy_predictions = np.repeat(held_q0[:, None, :], 48, axis=1)
    checkpoint_path = args.output_dir / "action_to_joint_gru.pt"
    torch.save(
        {
            "schema_version": REPORT_SCHEMA,
            "model_state_dict": best_state,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "q_mean": q_mean,
            "q_std": q_std,
            "horizon": 48,
            "action_encoding": root_manifest.get("action_encoding"),
        },
        checkpoint_path,
    )
    prediction_rows = export_frame0_predictions(
        args.output_dir,
        heldout_root,
        heldout_records,
        model,
        q_mean,
        q_std,
        device,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "trained_and_heldout_q_evaluated",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_manifest_sha256": sha256_file(root_manifest_path),
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
        "device": "cpu",
        "seed": args.seed,
        "configuration": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "validation_episodes_per_task": args.validation_episodes_per_task,
        },
        "data": {
            "fit_episodes": len(fit_records),
            "validation_episodes": len(validation_records),
            "heldout_episodes": len(heldout_records),
            "fit_windows": len(fit_q0),
            "validation_windows": len(val_q0),
            "heldout_windows": len(held_q0),
            "episode_overlap": root_manifest.get("episode_overlap"),
        },
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_validation,
        "final_validation_normalized_mse": validation_loss,
        "heldout_normalized_mse": heldout_loss,
        "validation_metrics": metric_summary(
            validation_predictions, val_targets, val_keys
        ),
        "heldout_metrics": metric_summary(
            heldout_predictions, held_targets, held_keys
        ),
        "heldout_copy_q0_baseline": metric_summary(
            copy_predictions, held_targets, held_keys
        ),
        "q_mean": q_mean.tolist(),
        "q_std": q_std.tolist(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_layout": (
            "dense 49 rows: observed q0 followed by 48 predicted post-action q states"
        ),
        "predictions": prediction_rows,
        "history": history,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["heldout_metrics"], sort_keys=True))
    print(json.dumps(report["heldout_copy_q0_baseline"], sort_keys=True))


if __name__ == "__main__":
    main()


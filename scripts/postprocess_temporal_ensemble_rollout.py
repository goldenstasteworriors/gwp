#!/usr/bin/env python3
"""Build offline temporal-ensemble rollouts from saved full action chunks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METHODS = (
    "lerobot_30old_70new",
    "linear_ramp",
    "cosine_ramp",
    "act_exp",
    "cogact_adaptive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--act-k", type=float, default=0.01)
    parser.add_argument("--cogact-alpha", type=float, default=0.1)
    parser.add_argument("--cogact-history", type=int, default=8)
    parser.add_argument("--ramp-new-start", type=float, default=0.2)
    parser.add_argument("--ramp-new-end", type=float, default=0.9)
    return parser.parse_args()


def load_record(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as record:
        arrays = {key: np.asarray(record[key]) for key in record.files}
    required = {
        "action_chunks",
        "replan_frame_indices",
        "predicted_state",
        "reference_state",
        "frame_indices",
        "initial_reference_state",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"rollout is missing arrays required for temporal ensemble: {sorted(missing)}")
    chunks = arrays["action_chunks"]
    replans = arrays["replan_frame_indices"]
    predicted = arrays["predicted_state"]
    if chunks.ndim != 3 or chunks.shape[-1] != 13:
        raise ValueError(f"expected action_chunks [N,H,13], got {chunks.shape}")
    if replans.shape != (len(chunks),):
        raise ValueError(f"replan/chunk mismatch: {replans.shape}/{chunks.shape}")
    if predicted.ndim != 2 or predicted.shape[-1] != 13:
        raise ValueError(f"expected predicted_state [T,13], got {predicted.shape}")
    if not all(np.isfinite(arrays[key]).all() for key in required):
        raise ValueError("rollout contains non-finite required arrays")
    return arrays


def valid_predictions(
    chunks: np.ndarray,
    replans: np.ndarray,
    action_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = action_frame - replans
    valid = np.flatnonzero((offsets >= 0) & (offsets < chunks.shape[1]))
    if len(valid) == 0:
        raise ValueError(f"no chunk covers action frame {action_frame}")
    predictions = np.stack([chunks[index, offsets[index]] for index in valid])
    return predictions.astype(np.float64), offsets[valid]


def ramp_new_weight(
    offset: int,
    execution_steps: int,
    start: float,
    end: float,
    cosine: bool,
) -> float:
    if execution_steps <= 1:
        phase = 1.0
    else:
        phase = min(max(offset / float(execution_steps - 1), 0.0), 1.0)
    if cosine:
        phase = 0.5 - 0.5 * math.cos(math.pi * phase)
    return start + (end - start) * phase


def combine_predictions(
    predictions: np.ndarray,
    offsets: np.ndarray,
    method: str,
    execution_steps: int,
    args: argparse.Namespace,
) -> np.ndarray:
    newest = predictions[-1]
    if len(predictions) == 1:
        return newest
    if method == "lerobot_30old_70new":
        return 0.3 * predictions[-2] + 0.7 * newest
    if method in {"linear_ramp", "cosine_ramp"}:
        new_weight = ramp_new_weight(
            int(offsets[-1]),
            execution_steps,
            args.ramp_new_start,
            args.ramp_new_end,
            cosine=method == "cosine_ramp",
        )
        return (1.0 - new_weight) * predictions[-2] + new_weight * newest
    if method == "act_exp":
        weights = np.exp(-args.act_k * np.arange(len(predictions), dtype=np.float64))
        weights /= weights.sum()
        return np.sum(weights[:, None] * predictions, axis=0)
    if method == "cogact_adaptive":
        predictions = predictions[-args.cogact_history :]
        reference = predictions[-1]
        norms = np.linalg.norm(predictions, axis=1) * np.linalg.norm(reference)
        similarities = np.sum(predictions * reference[None], axis=1) / (norms + 1e-7)
        weights = np.exp(args.cogact_alpha * similarities)
        weights /= weights.sum()
        return np.sum(weights[:, None] * predictions, axis=0)
    raise ValueError(method)


def execution_steps_from_replans(replans: np.ndarray, fallback: int) -> int:
    differences = np.diff(replans)
    if len(differences) == 0:
        return fallback
    positive = differences[differences > 0]
    if len(positive) == 0:
        return fallback
    return int(np.median(positive))


def ensemble_trajectory(
    arrays: dict[str, np.ndarray],
    method: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = arrays["action_chunks"].astype(np.float64)
    replans = arrays["replan_frame_indices"].astype(np.int64)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    execution_steps = execution_steps_from_replans(replans, chunks.shape[1])
    actions = []
    for action_frame in frame_indices[:-1]:
        predictions, offsets = valid_predictions(chunks, replans, int(action_frame))
        actions.append(
            combine_predictions(predictions, offsets, method, execution_steps, args)
        )
    action_array = np.asarray(actions, dtype=np.float32)
    initial_state = np.asarray(arrays["initial_reference_state"], dtype=np.float32)
    predicted = np.concatenate([initial_state[None], action_array], axis=0)
    return predicted, action_array


def smoothness_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    frame_indices: np.ndarray,
    replans: np.ndarray,
) -> dict[str, float | int]:
    step_delta = np.diff(predicted, axis=0)
    step_l2 = np.linalg.norm(step_delta, axis=1)
    boundary_action_indices = replans[1:] - frame_indices[0]
    boundary_action_indices = boundary_action_indices[
        (boundary_action_indices > 0) & (boundary_action_indices < len(step_l2))
    ].astype(np.int64)
    boundary_mask = np.zeros(len(step_l2), dtype=bool)
    boundary_mask[boundary_action_indices] = True
    interior_mask = ~boundary_mask
    error = predicted - reference

    def selected_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if np.any(mask) else 0.0

    return {
        "steps": int(len(predicted) - 1),
        "boundaries": int(boundary_mask.sum()),
        "step_l2_mean": float(step_l2.mean()),
        "step_l2_max": float(step_l2.max()),
        "interior_step_l2_mean": selected_mean(step_l2, interior_mask),
        "boundary_step_l2_mean": selected_mean(step_l2, boundary_mask),
        "boundary_step_l2_max": float(step_l2[boundary_mask].max()) if np.any(boundary_mask) else 0.0,
        "reference_rmse": float(np.sqrt(np.mean(error * error))),
        "arm_step_abs_mean": float(np.abs(step_delta[:, :7]).mean()),
        "hand_step_abs_mean": float(np.abs(step_delta[:, 7:]).mean()),
    }


def write_variant(
    source: Path,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    method: str,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, float | int]]:
    predicted, actions = ensemble_trajectory(arrays, method, args)
    output = output_dir / f"{source.stem}__te_{method}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    updated = dict(arrays)
    updated["predicted_state"] = predicted
    updated["predicted_action"] = actions
    updated["predicted_arm_delta"] = np.diff(predicted[:, :7], axis=0).astype(np.float32)
    updated["predicted_hand_absolute"] = actions[:, 7:13].astype(np.float32)
    updated["temporal_ensemble_method"] = np.asarray(method)
    np.savez_compressed(output, **updated)

    source_manifest = source.with_suffix(".manifest.json")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.is_file() else {}
    manifest["postprocess"] = {
        "type": "offline_temporal_ensemble",
        "method": method,
        "source_rollout": str(source),
        "counterfactual_only": True,
        "note": (
            "Saved chunks were inferred from the original rollout states. For predicted-state "
            "feedback this isolates action smoothing but is not a re-inferred closed-loop rollout."
        ),
        "act_k": args.act_k,
        "cogact_alpha": args.cogact_alpha,
        "cogact_history": args.cogact_history,
        "ramp_new_start": args.ramp_new_start,
        "ramp_new_end": args.ramp_new_end,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = smoothness_metrics(
        predicted,
        np.asarray(arrays["reference_state"], dtype=np.float32),
        np.asarray(arrays["frame_indices"], dtype=np.int64),
        np.asarray(arrays["replan_frame_indices"], dtype=np.int64),
    )
    output.with_suffix(".json").write_text(
        json.dumps({"rollout": str(output), "method": method, "metrics": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, metrics


def main() -> None:
    args = parse_args()
    if args.cogact_history <= 0:
        raise SystemExit("--cogact-history must be positive")
    if not 0.0 <= args.ramp_new_start <= args.ramp_new_end <= 1.0:
        raise SystemExit("ramp weights must satisfy 0 <= start <= end <= 1")
    source = args.rollout.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    arrays = load_record(source)
    baseline_metrics = smoothness_metrics(
        np.asarray(arrays["predicted_state"], dtype=np.float32),
        np.asarray(arrays["reference_state"], dtype=np.float32),
        np.asarray(arrays["frame_indices"], dtype=np.int64),
        np.asarray(arrays["replan_frame_indices"], dtype=np.int64),
    )
    summaries = {"baseline_direct_switch": {"output": str(source), "metrics": baseline_metrics}}
    print("baseline_direct_switch:")
    print(json.dumps(baseline_metrics, indent=2))
    for method in args.methods:
        output, metrics = write_variant(source, output_dir, arrays, method, args)
        summaries[method] = {"output": str(output), "metrics": metrics}
        print(f"{method}: {output}")
        print(json.dumps(metrics, indent=2))
    summary_path = output_dir / f"{source.stem}__temporal_ensemble_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

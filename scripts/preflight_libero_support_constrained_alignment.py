#!/usr/bin/env python3
"""Preflight simulator-free support-constrained action/mask sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_SCHEMA = "libero-counterfactual-split-v1"
EXPERT_SCHEMA = "libero-logged-action-kinematic-mask-v1"
TARGET_SCHEMA = "libero-support-constrained-action-kinematic-mask-v1"
SUPPORT_SCHEMA = "libero-action-to-joint-input-support-audit-v1"
TARGET_AUDIT_SCHEMA = "libero-kinematic-alignment-target-audit-v1"
HORIZONS = [0, 12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--expert-train-cache", type=Path, required=True)
    parser.add_argument("--expert-heldout-cache", type=Path, required=True)
    parser.add_argument("--target-train-cache", type=Path, required=True)
    parser.add_argument("--target-heldout-cache", type=Path, required=True)
    parser.add_argument("--input-support-report", type=Path, required=True)
    parser.add_argument("--target-train-audit", type=Path, required=True)
    parser.add_argument("--target-heldout-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_episode_file(path: Path) -> list[int]:
    values = [int(token) for token in path.read_text(encoding="utf-8").split()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid episode file: {path}")
    return values


def validate_cache(
    root: Path,
    schema: str,
    expected_episodes: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != schema:
        raise ValueError(f"Unexpected cache schema: {manifest_path}")
    if manifest.get("rgb_supervision") is not None:
        raise ValueError(f"Cache contains RGB supervision: {manifest_path}")
    if manifest.get("horizons") != HORIZONS:
        raise ValueError(f"Unexpected horizons: {manifest_path}")
    calls = manifest.get("dynamics_attempted_calls", {})
    if manifest.get("dynamics") != "forbidden" or set(calls) != {
        "env_step",
        "sim_step",
        "mj_step",
    } or any(int(value) != 0 for value in calls.values()):
        raise ValueError(f"Dynamics attempt in {manifest_path}: {calls}")
    records = list(manifest.get("episodes", []))
    episodes = [int(record["episode_index"]) for record in records]
    if episodes != expected_episodes or len(records) != int(
        manifest.get("num_episodes", -1)
    ):
        raise ValueError(f"Episode order/count mismatch: {manifest_path}")
    alpha_counts: dict[str, int] = {}
    for record in records:
        if int(record.get("source_frame_index", -1)) != 0:
            raise ValueError(f"Nonzero source frame: {record}")
        cache_path = root / str(record["cache_file"])
        if sha256_file(cache_path) != record["cache_sha256"]:
            raise ValueError(f"Cache changed: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as arrays:
            required = {
                "horizons",
                "robot_mask",
                "gripper_mask",
                "predicted_joint_qpos",
                "predicted_gripper_qpos",
            }
            if schema == TARGET_SCHEMA:
                required.add("policy_actions_lerobot")
            else:
                required.add("expert_actions_lerobot")
            if not required.issubset(arrays.files):
                raise ValueError(f"Missing arrays in {cache_path}")
            if arrays["horizons"].tolist() != HORIZONS:
                raise ValueError(f"Wrong per-file horizons: {cache_path}")
            if arrays["robot_mask"].shape != (5, 256, 256):
                raise ValueError(f"Wrong robot mask shape: {cache_path}")
            if arrays["gripper_mask"].shape != (5, 256, 256):
                raise ValueError(f"Wrong gripper mask shape: {cache_path}")
            if np.logical_and(
                arrays["gripper_mask"] > 0, arrays["robot_mask"] == 0
            ).any():
                raise ValueError(f"Gripper mask not subset: {cache_path}")
        if schema == TARGET_SCHEMA:
            alpha = float(record["maximum_supported_alpha"])
            alpha_counts[f"{alpha:.2f}"] = alpha_counts.get(f"{alpha:.2f}", 0) + 1
    if schema == TARGET_SCHEMA and alpha_counts != manifest.get("alpha_counts"):
        raise ValueError(f"Alpha counts changed: {manifest_path}")
    return manifest, {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "num_episodes": len(records),
        "alpha_counts": alpha_counts if schema == TARGET_SCHEMA else None,
    }


def validate_target_audit(
    path: Path,
    expert_root: Path,
    target_root: Path,
) -> dict[str, Any]:
    report = load_json(path)
    if report.get("schema_version") != TARGET_AUDIT_SCHEMA:
        raise ValueError(f"Unexpected target audit schema: {path}")
    if report.get("status") != "passed" or not all(report.get("checks", {}).values()):
        raise ValueError(f"Target audit failed: {path}")
    if Path(str(report.get("expert_cache"))).resolve() != expert_root.resolve():
        raise ValueError(f"Target audit/expert cache mismatch: {path}")
    if Path(str(report.get("policy_cache"))).resolve() != target_root.resolve():
        raise ValueError(f"Target audit/target cache mismatch: {path}")
    if report.get("expert_manifest_sha256") != sha256_file(
        expert_root / "manifest.json"
    ) or report.get("policy_manifest_sha256") != sha256_file(
        target_root / "manifest.json"
    ):
        raise ValueError(f"Target audit input changed: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "checks": report["checks"],
    }


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    if not (args.project_root / ".git").exists():
        raise FileNotFoundError(args.project_root)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    split_path = args.split_dir / "manifest.json"
    split = load_json(split_path)
    if split.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError("Unexpected split schema")
    train_file = args.split_dir / str(split["train_episodes_file"])
    heldout_file = args.split_dir / str(split["heldout_episodes_file"])
    train_episodes = read_episode_file(train_file)
    heldout_episodes = read_episode_file(heldout_file)
    if train_episodes != [int(value) for value in split["train_episodes"]]:
        raise ValueError("Train split file/manifest mismatch")
    if heldout_episodes != [int(value) for value in split["heldout_episodes"]]:
        raise ValueError("Held-out split file/manifest mismatch")
    overlap = sorted(set(train_episodes) & set(heldout_episodes))
    if overlap:
        raise ValueError(f"Train/held-out overlap: {overlap}")

    expert_train, expert_train_summary = validate_cache(
        args.expert_train_cache, EXPERT_SCHEMA, train_episodes
    )
    expert_heldout, expert_heldout_summary = validate_cache(
        args.expert_heldout_cache, EXPERT_SCHEMA, heldout_episodes
    )
    target_train, target_train_summary = validate_cache(
        args.target_train_cache, TARGET_SCHEMA, train_episodes
    )
    target_heldout, target_heldout_summary = validate_cache(
        args.target_heldout_cache, TARGET_SCHEMA, heldout_episodes
    )
    manifests = (expert_train, expert_heldout, target_train, target_heldout)
    for key in ("dataset_root", "dataset_info_sha256", "suite", "horizons"):
        if len({json.dumps(manifest.get(key), sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"Cache metadata differs for {key}")
    if Path(str(target_train["policy_id"])).resolve() != args.checkpoint.resolve() or Path(
        str(target_heldout["policy_id"])
    ).resolve() != args.checkpoint.resolve():
        raise ValueError("Target cache baseline checkpoint mismatch")

    support = load_json(args.input_support_report)
    if support.get("schema_version") != SUPPORT_SCHEMA:
        raise ValueError("Unexpected input-support report")
    if support.get("status") != "failed" or support.get("decision") != (
        "filter_or_retrain_action_to_q_before_alignment"
    ):
        raise ValueError("Expected full-policy support gate to fail before filtering")
    if target_train.get("input_support_report_sha256") != sha256_file(
        args.input_support_report
    ) or target_heldout.get("input_support_report_sha256") != sha256_file(
        args.input_support_report
    ):
        raise ValueError("Target caches use another input-support report")
    selected_alphas = [
        float(record["maximum_supported_alpha"])
        for split_name in ("train", "heldout")
        for record in support["maximum_supported_alpha"][split_name]
    ]
    if min(selected_alphas) < 0.5:
        raise ValueError("A selected curriculum alpha is below the predeclared useful range")

    train_audit = validate_target_audit(
        args.target_train_audit, args.expert_train_cache, args.target_train_cache
    )
    heldout_audit = validate_target_audit(
        args.target_heldout_audit,
        args.expert_heldout_cache,
        args.target_heldout_cache,
    )
    git_commit = subprocess.check_output(
        ["git", "-C", str(args.project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.project_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if tracked:
        raise RuntimeError(f"Tracked project files are dirty:\n{tracked}")
    report = {
        "schema_version": "libero-support-constrained-alignment-preflight-v1",
        "status": "passed",
        "decision": "ready_for_clean_latent_probe_and_action_gap_gates",
        "project_root": str(args.project_root.resolve()),
        "git_commit": git_commit,
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": sha256_file(split_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": target_train["dataset_root"],
        "num_train": len(train_episodes),
        "num_heldout": len(heldout_episodes),
        "episode_overlap": 0,
        "expert_train_cache": expert_train_summary,
        "expert_heldout_cache": expert_heldout_summary,
        "target_train_cache": target_train_summary,
        "target_heldout_cache": target_heldout_summary,
        "input_support_report": str(args.input_support_report.resolve()),
        "input_support_report_sha256": sha256_file(args.input_support_report),
        "full_policy_accepted_fraction": {
            "train": support["policy_train"]["accepted_fraction"],
            "heldout": support["policy_heldout"]["accepted_fraction"],
        },
        "curriculum_alpha_counts": {
            "train": target_train["alpha_counts"],
            "heldout": target_heldout["alpha_counts"],
        },
        "target_train_audit": train_audit,
        "target_heldout_audit": heldout_audit,
        "simulator_runtime_imported": False,
        "simulator_step_calls": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()

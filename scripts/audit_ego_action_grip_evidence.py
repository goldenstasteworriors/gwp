#!/usr/bin/env python3
"""Create reviewed, artifact-bound Ego action/grip semantics evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_ego_stage1_preflight import _validate_norm  # noqa: E402
from world_action_model.ego_action_schema import (  # noqa: E402
    CANONICAL_OPEN_FRACTION_SEMANTICS,
    EGO_HAND_STATUS_SEMANTICS,
    convert_hand_status_semantics,
)
from world_action_model.ego_dataset_fingerprint import derive_ego_dataset_identity  # noqa: E402
from world_action_model.ego_hand_active_sidecar import (  # noqa: E402
    EGO_HAND_ACTIVE_FIELD_ORDER,
    EGO_HAND_ACTIVE_SEMANTICS,
    load_and_validate_hand_active_bundle,
    sha256_file,
)
from world_action_model.verified_io import (  # noqa: E402
    read_verified_bytes,
    strict_json_loads,
    verified_head_code_manifest,
)


REPORT_VERSION = "ego-action-grip-evidence/v2"
SOURCE_RECEIPT_VERSION = "ego-action-grip-source-receipt/v1"
REVIEW_STATUS = "semantics_verified_missingness_masked"
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)

UPSTREAM_SOURCES = {
    "data_alignment/human_data_process/add_hand_status.py": (
        "hand_status = np.stack([left_status, right_status], axis=1)",
        "1=closed, 0=open. Columns: [left, right]",
        "binary signal (1 = closed, 0 = open)",
    ),
    "data_alignment/human_data_process/process_human_eef_pipeline.py": (
        "delta_eef[i, :6] = compute_delta_eef_single(left_eef[i-1], left_eef[i])",
        "delta_eef[i, 6:] = compute_delta_eef_single(right_eef[i-1], right_eef[i])",
        "delta_T = np.linalg.inv(T_prev) @ T_curr",
        "action_eef = np.hstack([left_eef_ds, right_eef_ds])",
    ),
    "data_alignment/convert_to_lerobot.py": (
        '"hand_status": np.array(hand_status_data[i], dtype=np.float32)',
        '"action_delta_eef": np.array(action_delta_eef[i], dtype=np.float32)',
        "fps=fps",
    ),
    "scripts/deploy.py": (
        "left_q = self.left_open_q * (1 - hand_status[0]) + self.left_close_q * hand_status[0]",
        "right_q = self.right_open_q * (1 - hand_status[1]) + self.right_close_q * hand_status[1]",
        "hand_status: 0.0 = open, 1.0 = close",
    ),
}

WAM_SOURCES = (
    "world_action_model/ego_action_schema.py",
    "world_action_model/ego_hand_active_sidecar.py",
    "world_action_model/datasets/wam_lerobot_dataset.py",
    "world_action_model/transforms/wa_transforms_lerobot_pretrain.py",
    "world_action_model/trainer/wa_casual_trainer_mot.py",
    "world_action_model/contact_ego_stage1_preflight.py",
    "world_action_model/ego_stage1_gate_bootstrap.py",
    "scripts/build_ego_hand_active_sidecars.py",
    "scripts/compute_ego_wam_norm.py",
    "scripts/audit_ego_action_grip_evidence.py",
    "configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--raw-hdf5-dir", required=True, type=Path)
    parser.add_argument("--final-hdf5-dir", required=True, type=Path)
    parser.add_argument("--hand-active-sidecar-dir", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--source-receipt-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect Git source repository: {root}") from exc


def _source_manifest(root: Path, sources: dict[str, tuple[str, ...]]) -> tuple[str, list[dict]]:
    commit = _git_bytes(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    rows = []
    for logical_path, required_fragments in sources.items():
        head_payload = _git_bytes(root, "show", f"{commit}:{logical_path}")
        runtime_payload = (root / logical_path).read_bytes()
        for label, payload in (("HEAD", head_payload), ("worktree", runtime_payload)):
            text = payload.decode("utf-8")
            missing = [fragment for fragment in required_fragments if fragment not in text]
            if missing:
                raise ValueError(
                    f"{logical_path} {label} lacks required semantic evidence: {missing}"
                )
        rows.append(
            {
                "logical_path": logical_path,
                "git_blob_sha1": _git_bytes(
                    root, "rev-parse", "--verify", f"{commit}:{logical_path}"
                ).decode("ascii").strip(),
                "head_blob_sha256": hashlib.sha256(head_payload).hexdigest(),
                "audited_worktree_sha256": hashlib.sha256(runtime_payload).hexdigest(),
                "worktree_matches_head": runtime_payload == head_payload,
                "required_semantic_fragments_sha256": hashlib.sha256(
                    json.dumps(required_fragments, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    return commit, rows


def _write_json(path: Path, value: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _delta_and_clock_audit(final_root: Path, raw_root: Path, episodes: list[int]) -> dict:
    max_translation_residual = 0.0
    max_rotation_residual = 0.0
    max_first_delta = 0.0
    max_quaternion_norm_error = 0.0
    max_timestamp_downsample_residual_ns = 0.0
    min_step_seconds = float("inf")
    max_step_seconds = 0.0
    step_sum = 0.0
    step_count = 0
    for episode in episodes:
        with h5py.File(final_root / f"episode_{episode}.hdf5", "r") as final:
            pose = np.asarray(final["action_eef"], dtype=np.float64)
            delta = np.asarray(final["action_delta_eef"], dtype=np.float64)
            timestamps = np.asarray(final["local_timestamps_ns"], dtype=np.int64)
        with h5py.File(raw_root / f"episode_{episode}.hdf5", "r") as raw:
            raw_timestamps = np.asarray(raw["local_timestamps_ns"], dtype=np.int64)
        max_timestamp_downsample_residual_ns = max(
            max_timestamp_downsample_residual_ns,
            float(np.max(np.abs(timestamps - raw_timestamps[::5][: len(timestamps)]))),
        )
        max_first_delta = max(max_first_delta, float(np.max(np.abs(delta[0]))))
        for pose_offset, delta_offset in ((0, 0), (7, 6)):
            quaternion = pose[:, pose_offset + 3 : pose_offset + 7]
            max_quaternion_norm_error = max(
                max_quaternion_norm_error,
                float(np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0))),
            )
            rotation = Rotation.from_quat(quaternion)
            relative = rotation[:-1].inv() * rotation[1:]
            translation = rotation[:-1].inv().apply(
                pose[1:, pose_offset : pose_offset + 3]
                - pose[:-1, pose_offset : pose_offset + 3]
            )
            actual = delta[1:, delta_offset : delta_offset + 6]
            max_translation_residual = max(
                max_translation_residual,
                float(np.max(np.abs(translation - actual[:, :3]))),
            )
            wrapped = (
                relative.as_euler("xyz") - actual[:, 3:] + np.pi
            ) % (2.0 * np.pi) - np.pi
            max_rotation_residual = max(
                max_rotation_residual,
                float(np.max(np.abs(wrapped))),
            )
        steps = np.diff(timestamps) / 1e9
        min_step_seconds = min(min_step_seconds, float(steps.min()))
        max_step_seconds = max(max_step_seconds, float(steps.max()))
        step_sum += float(steps.sum())
        step_count += len(steps)
    passed = (
        max_translation_residual <= 1e-12
        and max_rotation_residual <= 1e-12
        and max_first_delta == 0.0
        and max_quaternion_norm_error <= 1e-12
        and max_timestamp_downsample_residual_ns == 0.0
        and min_step_seconds > 0.0
    )
    return {
        "passed": passed,
        "delta_definition": "delta[t]=pose[t-1]^-1_times_pose[t];delta[0]=identity",
        "wrist_field_order": "left_xyz_rpy_then_right_xyz_rpy",
        "max_translation_residual": max_translation_residual,
        "max_wrapped_xyz_euler_residual": max_rotation_residual,
        "max_first_frame_delta_abs": max_first_delta,
        "max_quaternion_norm_error": max_quaternion_norm_error,
        "max_raw_100hz_to_processed_20hz_timestamp_residual_ns": (
            max_timestamp_downsample_residual_ns
        ),
        "processed_timestamp_step_seconds": {
            "min": min_step_seconds,
            "max": max_step_seconds,
            "mean": step_sum / step_count,
        },
    }


def main() -> None:
    args = parse_args()
    if not args.reviewed_by.strip():
        raise ValueError("reviewed-by must be non-empty")
    if not _UTC_RE.fullmatch(args.reviewed_at_utc):
        raise ValueError("reviewed-at-utc must be canonical whole-second UTC with Z")
    datetime.strptime(args.reviewed_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    identity = derive_ego_dataset_identity(args.dataset_root)
    hand_active = load_and_validate_hand_active_bundle(
        args.hand_active_sidecar_dir,
        dataset_identity=identity,
        raw_hdf5_dir=args.raw_hdf5_dir,
        final_hdf5_dir=args.final_hdf5_dir,
    )
    norm_bytes, norm_digest, _ = read_verified_bytes(args.norm_stats)
    norm = strict_json_loads(norm_bytes, name="Ego Stage1 norm")
    if not isinstance(norm, dict):
        raise ValueError("Ego Stage1 norm must be an object")
    hdf5_bundle = _validate_norm(
        norm,
        identity,
        args.final_hdf5_dir.resolve(),
        hand_active,
    )
    polarity_probe = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    converted = convert_hand_status_semantics(
        polarity_probe,
        source_semantics=EGO_HAND_STATUS_SEMANTICS,
        target_semantics=CANONICAL_OPEN_FRACTION_SEMANTICS,
    )
    if not np.array_equal(converted, 1.0 - polarity_probe):
        raise ValueError("installed WAM hand-status conversion failed the polarity probe")

    upstream_root = args.upstream_repo.expanduser().resolve()
    upstream_commit, upstream_files = _source_manifest(upstream_root, UPSTREAM_SOURCES)
    wam_commit, wam_files = verified_head_code_manifest(REPO_ROOT, WAM_SOURCES)
    source_receipt = {
        "receipt_version": SOURCE_RECEIPT_VERSION,
        "upstream": {
            "repository_path_record_only": str(upstream_root),
            "git_commit": upstream_commit,
            "files": upstream_files,
            "historical_dataset_generator_commit_recorded_in_hdf5": False,
        },
        "wam": {
            "repository_path_record_only": str(REPO_ROOT),
            "git_commit": wam_commit,
            "files": wam_files,
        },
        "verified_claims": {
            "source_hand_status_semantics": EGO_HAND_STATUS_SEMANTICS,
            "source_hand_field_order": list(EGO_HAND_ACTIVE_FIELD_ORDER),
            "source_wrist_field_order": "left_then_right",
            "target_hand_status_semantics": CANONICAL_OPEN_FRACTION_SEMANTICS,
            "conversion": "target=1-source",
            "grip_action_dimensions": [18, 19],
            "grip_proxy_loss_weight_when_active": 0.25,
        },
        "limitations": [
            "the HDF5 files do not record the exact historical generator Git commit",
            "source-code evidence establishes the checked-out semantic contract, not physical contact",
        ],
    }
    _write_json(args.source_receipt_output, source_receipt, overwrite=args.overwrite)
    source_receipt_digest = sha256_file(args.source_receipt_output)
    clock_audit = _delta_and_clock_audit(
        args.final_hdf5_dir.expanduser().resolve(),
        args.raw_hdf5_dir.expanduser().resolve(),
        list(identity["episode_lengths"]),
    )
    if clock_audit["passed"] is not True:
        raise ValueError("Ego wrist delta/time alignment audit failed")
    report = {
        "report_version": REPORT_VERSION,
        "status": REVIEW_STATUS,
        "source_verified": True,
        "transform_conversion_verified": True,
        "missingness_masked": True,
        "physical_contact_truth": False,
        "dataset_id": identity["dataset_id"],
        "dataset_fingerprint_sha256": identity["dataset_fingerprint_sha256"],
        "source_receipt_sha256": identity["source_receipt_sha256"],
        "norm_stats_sha256": norm_digest,
        "hdf5_bundle_sha256": hdf5_bundle,
        "hand_active_meta_sha256": hand_active["meta_sha256"],
        "hand_active_sidecar_bundle_sha256": hand_active["sidecar_bundle_sha256"],
        "hand_active_raw_hdf5_bundle_sha256": hand_active[
            "raw_hdf5_bundle_sha256"
        ],
        "source_action_dim": 20,
        "model_action_dim": 32,
        "source_hand_status_semantics": EGO_HAND_STATUS_SEMANTICS,
        "target_hand_status_semantics": CANONICAL_OPEN_FRACTION_SEMANTICS,
        "hand_status_conversion": "target=1-source",
        "hand_field_order": list(EGO_HAND_ACTIVE_FIELD_ORDER),
        "grip_action_dimensions": [18, 19],
        "grip_proxy_loss_weight_when_active": 0.25,
        "inactive_grip_loss_weight": 0.0,
        "wrist_action_dimensions_unaffected": list(range(18)),
        "hand_active_semantics": EGO_HAND_ACTIVE_SEMANTICS,
        "transition_auxiliary": {
            "enabled": False,
            "pair_mask_recorded": True,
            "validity": "active_at_target_minus_1_and_target",
        },
        "dataset_scope": {
            "episode_indices": list(identity["episode_lengths"]),
            "episode_count": identity["episode_count"],
            "total_frames": identity["total_frames"],
            "fps": identity["fps"],
            "frame_clock": identity["frame_clock"],
        },
        "hand_active_statistics": hand_active["statistics"],
        "delta_and_clock_audit": clock_audit,
        "source_evidence": {
            "path": os.path.relpath(
                args.source_receipt_output.resolve(),
                start=args.report_output.resolve().parent,
            ),
            "sha256": source_receipt_digest,
        },
        "reviewed_by": args.reviewed_by,
        "reviewed_at_utc": args.reviewed_at_utc,
        "limitations": [
            "hand_status is a binary square-wave closure proxy, not physical contact truth",
            "hand_active is tracker validity, not evidence that an object was touched",
            "inactive proxy values are retained as record data but excluded from grip normalization and loss",
            "zero-supervision hand/episode pairs cannot support a bimanual truth claim",
            "the transition-pair mask is record-only because Stage1 has no transition auxiliary loss",
            "the exact historical HDF5 generator commit was not recorded",
        ],
    }
    _write_json(args.report_output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": REVIEW_STATUS,
                "report": str(args.report_output.resolve()),
                "report_sha256": sha256_file(args.report_output),
                "source_receipt_sha256": source_receipt_digest,
                "hand_active_statistics": hand_active["statistics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

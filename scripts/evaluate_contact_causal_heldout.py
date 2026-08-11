#!/usr/bin/env python3
"""Run or offline-replay the causal-main held-out evaluator v2."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _reject_project_bytecode_cache(root: Path) -> None:
    """Prevent tracked-source imports from being shadowed by repository pyc files."""

    sys.dont_write_bytecode = True
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        relative = Path(current).relative_to(root)
        directory_names[:] = [
            name
            for name in directory_names
            if not (relative == Path(".") and name in {".git", "outputs"})
        ]
        if "__pycache__" in directory_names or any(name.endswith(".pyc") for name in file_names):
            raise RuntimeError("causal evaluator forbids repository bytecode caches")


_reject_project_bytecode_cache(REPOSITORY_ROOT)


def _stdlib_bootstrap_evaluator_head(root: Path) -> str:
    """Verify the HEAD-declared evaluator tree before NumPy/project imports."""

    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    manifest_payload = subprocess.run(
        [
            "git", "-C", str(root), "cat-file", "blob",
            f"{head}:configs/contact_causal_production_matrix.v1.json",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout
    manifest = json.loads(manifest_payload.decode("utf-8"))
    logical_paths = manifest.get("source_logical_paths")
    if (
        not isinstance(logical_paths, list)
        or "scripts/evaluate_contact_causal_heldout.py" not in logical_paths
        or len(logical_paths) != len(set(logical_paths))
    ):
        raise RuntimeError("causal evaluator HEAD source manifest differs")
    for logical_path in logical_paths:
        if (
            not isinstance(logical_path, str)
            or Path(logical_path).is_absolute()
            or ".." in Path(logical_path).parts
        ):
            raise RuntimeError("causal evaluator HEAD source path is unsafe")
        expected = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", f"{head}:{logical_path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        path = root / logical_path
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeError("causal evaluator runtime source is unsafe")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or b"".join(chunks) != expected
        ):
            raise RuntimeError(
                f"causal evaluator source differs from HEAD: {logical_path}"
            )
    if subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip() != head:
        raise RuntimeError("causal evaluator HEAD changed during bootstrap")
    return head


_EVALUATOR_BOOTSTRAP_HEAD = _stdlib_bootstrap_evaluator_head(REPOSITORY_ROOT)

import numpy as np


if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_action_model.contact_asym25_heldout_eval import (  # noqa: E402
    ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
    HeldoutEvaluationError,
    build_environment_receipt,
    build_fixed_window_manifest,
    load_and_replay_preflight_receipt,
    load_completed_group_split,
    load_fixed_window_manifest,
    public_tree_receipt,
    reduce_metric_rows,
    score_heldout_window,
    snapshot_artifact_tree,
    verify_artifact_tree_unchanged,
    write_per_window_csv,
    write_strict_json,
)
from world_action_model.contact_causal_checkpoint import (  # noqa: E402
    validate_checkpoint_manifest_v3,
)
from world_action_model.contact_causal_heldout_eval import (  # noqa: E402
    CAUSAL_EVALUATION_CONTRACT,
    build_aggregate_v5,
    ROUTE_OUTPUT_EVIDENCE_FORMAT,
    _recompute_route_packet_metrics,
    validate_result_v3,
    write_route_output_evidence,
)
from world_action_model.contact_causal_preflight import (  # noqa: E402
    validate_preflight_receipt_v2,
)
from world_action_model.contact_causal_production import (  # noqa: E402
    CAUSAL_ROUTES,
    CHECKPOINT_STEPS,
    EVALUATOR_VERSION,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    RESULT_VERSION,
    CausalProductionError,
    ensure_not_project_tmp,
    sha256_json,
    strict_json_file,
)


def _source_contract(causal_path: Path, hardened_path: Path):
    causal, causal_sha, _ = strict_json_file(causal_path, name="causal preflight v2")
    validate_preflight_receipt_v2(causal)
    if causal["git_commit"] != _EVALUATOR_BOOTSTRAP_HEAD:
        raise CausalProductionError(
            "causal evaluator preflight differs from stdlib bootstrap HEAD"
        )
    try:
        hardened, hardened_sha = load_and_replay_preflight_receipt(hardened_path)
        _, assignments = load_completed_group_split(
            hardened["split_manifest_path"],
            expected_sha256=hardened["split_manifest_sha256"],
            expected_episode_lengths=hardened["episode_lengths"],
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("hardened v2 held-out source replay failed") from exc
    training = causal["training_data_receipt"]
    mirrors = {
        "dataset_id": "dataset_id",
        "split_seed": "split_seed",
        "dataset_info_sha256": "dataset_info_sha256",
        "hardware_contract_sha256": "hardware_contract_sha256",
        "calibration_capture_sha256": "calibration_capture_sha256",
        "train_only_norm_sha256": "norm_stats_sha256",
        "sidecar_bundle_sha256": "sidecar_bundle_sha256",
        "split_manifest_sha256": "split_manifest_sha256",
    }
    if any(training[new] != hardened.get(old) for new, old in mirrors.items()):
        raise CausalProductionError("causal preflight differs from mature replay")
    return causal, causal_sha, hardened, hardened_sha, assignments


def _digest_array(digest: Any, window_id: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    for payload in (
        window_id.encode("utf-8"), str(array.dtype).encode("ascii"),
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


PREDICTION_EVIDENCE_FORMAT = "contact-causal-prediction-evidence-npz/v1"
_PREDICTION_KEYS = {
    "metadata_json",
    "checkpoint_steps",
    "window_ids",
    "action_predictions",
    "semantic_aux_logits",
    "contact_hard_semantics",
    "route_packets",
}


def _write_exclusive_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write for causal prediction evidence")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(payload).hexdigest()


def _write_prediction_evidence(
    path: Path,
    *,
    metadata: dict,
    checkpoint_steps: list[int],
    window_ids: list[str],
    action_predictions: np.ndarray,
    semantic_aux_logits: np.ndarray,
    contact_hard_semantics: np.ndarray,
    route_packets: np.ndarray,
) -> str:
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    arrays = {
        "metadata_json": np.frombuffer(metadata_bytes, dtype=np.uint8),
        "checkpoint_steps": np.asarray(checkpoint_steps, dtype=np.int64),
        "window_ids": np.asarray(window_ids, dtype=np.str_),
        "action_predictions": np.asarray(action_predictions, dtype=np.float32),
        "semantic_aux_logits": np.asarray(semantic_aux_logits, dtype=np.float32),
        "contact_hard_semantics": np.asarray(contact_hard_semantics, dtype=np.float32),
        "route_packets": np.asarray(route_packets, dtype=np.float32),
    }
    row_count = len(checkpoint_steps)
    if (
        row_count != 3 * 512
        or len(window_ids) != row_count
        or any(array.shape[0] != row_count for key, array in arrays.items()
               if key not in {"metadata_json", "checkpoint_steps", "window_ids"})
        or any(
            not np.isfinite(array).all()
            for key, array in arrays.items()
            if key not in {"metadata_json", "checkpoint_steps", "window_ids"}
        )
    ):
        raise CausalProductionError("prediction evidence row/finite contract differs")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return _write_exclusive_bytes(path, buffer.getvalue())


def _load_prediction_evidence(path: Path) -> tuple[dict, dict[str, np.ndarray], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != _PREDICTION_KEYS:
            raise CausalProductionError("prediction evidence NPZ schema differs")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    try:
        metadata = json.loads(arrays.pop("metadata_json").tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CausalProductionError("prediction evidence metadata is invalid") from exc
    if not isinstance(metadata, dict):
        raise CausalProductionError("prediction evidence metadata must be an object")
    return metadata, arrays, digest


def prepare_windows(args: argparse.Namespace) -> None:
    output = ensure_not_project_tmp(args.output, project_root=REPOSITORY_ROOT)
    try:
        hardened, hardened_sha = load_and_replay_preflight_receipt(
            args.hardened_preflight_receipt
        )
        _, assignments = load_completed_group_split(
            hardened["split_manifest_path"],
            expected_sha256=hardened["split_manifest_sha256"],
            expected_episode_lengths=hardened["episode_lengths"],
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("cannot prepare mature fixed windows") from exc
    manifest = build_fixed_window_manifest(
        preflight_receipt=hardened,
        preflight_receipt_sha256=hardened_sha,
        split_assignments=assignments,
    )
    digest = write_strict_json(output, manifest)
    print(json.dumps({"output": str(output), "sha256": digest}, sort_keys=True))


def predict_phase(args: argparse.Namespace) -> None:
    """Run checkpoints in a target-free worker and persist predictions O_EXCL."""

    output = ensure_not_project_tmp(args.output, project_root=REPOSITORY_ROOT)
    causal, causal_sha, hardened, hardened_sha, assignments = _source_contract(
        args.causal_preflight_receipt, args.hardened_preflight_receipt
    )
    try:
        windows, windows_sha = load_fixed_window_manifest(
            args.window_manifest,
            preflight_receipt=hardened,
            preflight_receipt_sha256=hardened_sha,
            split_assignments=assignments,
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("prediction fixed windows replay failed") from exc
    manifest, manifest_sha, _ = strict_json_file(
        args.checkpoint_manifest, name="causal prediction checkpoint manifest"
    )
    manifest, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
    seed = causal["training_data_receipt"]["split_seed"]
    route = manifest["causal_route_id"]
    selected_windows = [window for window in windows["windows"] if window["seed"] == seed]
    if len(selected_windows) != 512:
        raise CausalProductionError("prediction worker requires 512 fixed windows")
    base_snapshot = snapshot_artifact_tree(args.base_model)
    test_only_tiny_runtime = args.test_only_tiny_runtime
    if test_only_tiny_runtime and (
        causal["production_data_admission"] is not False
        or causal["allow_5b_training"] is not False
        or causal["allow_server_training"] is not False
        or args.device != "cpu"
    ):
        raise CausalProductionError(
            "test-only tiny replay requires a non-production CPU preflight"
        )
    environment = build_environment_receipt(
        device=args.device,
        dtype=("float32" if test_only_tiny_runtime else "bfloat16"),
        inference_steps=args.inference_steps,
    )
    environment["test_only_tiny_runtime"] = test_only_tiny_runtime
    cache_parent = output.parent / ".prediction_private_hf_cache"
    cache_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_parent.chmod(0o700)
    evaluator_nonce = "causal-predict-" + hashlib.sha256(
        (manifest["training_run_nonce"] + ":" + route + ":" + str(seed)).encode()
    ).hexdigest()
    from world_action_model.contact_causal_eval_runtime import (
        CausalProductionHeldoutRuntime,
        CausalTestOnlyTinyReplayRuntime,
    )

    runtime_class = (
        CausalTestOnlyTinyReplayRuntime
        if test_only_tiny_runtime
        else CausalProductionHeldoutRuntime
    )
    runtime = runtime_class(
        preflight_receipt=hardened,
        base_model_path=args.base_model,
        device=args.device,
        dtype=("float32" if test_only_tiny_runtime else "bfloat16"),
        inference_steps=args.inference_steps,
        verified_private_cache_parent=str(cache_parent),
        verified_run_nonce=evaluator_nonce,
        base_model_source_snapshot=base_snapshot,
    )
    steps: list[int] = []
    window_ids: list[str] = []
    actions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    hard_semantics: list[np.ndarray] = []
    packets: list[np.ndarray] = []
    try:
        for step in CHECKPOINT_STEPS:
            receipt, _ = accepted[step]
            runtime.load_checkpoint(
                str(Path(receipt["checkpoint_file"]["path_record_only"]).parent),
                checkpoint_receipt=receipt,
            )
            if runtime.causal_route != route:
                raise CausalProductionError("prediction checkpoint route differs")
            for window in selected_windows:
                packet = runtime.load_inference_window(window)
                prediction = runtime.predict(
                    packet, action_noise_seed=window["action_noise_seed"]
                )
                steps.append(step)
                window_ids.append(window["window_id"])
                actions.append(prediction["action_prediction"])
                logits.append(prediction["contact_logits"])
                hard_semantics.append(prediction["contact_hard_semantic"])
                if route != "aux_only_direct_q12":
                    route_packet = prediction["route_packet"]
                    if route_packet is None:
                        raise CausalProductionError("prediction worker omitted route packet")
                    packets.append(route_packet)
            runtime.unload_checkpoint()
        prediction_source_receipt = dict(runtime.prediction_source_column_receipt)
        prediction_source_receipt["test_only_import_harness"] = (
            dict(runtime.test_only_import_harness)
            if test_only_tiny_runtime
            else None
        )
    finally:
        runtime.close()
    cache_parent.rmdir()
    verify_artifact_tree_unchanged(base_snapshot)
    schedule = [
        {
            "window_id": window["window_id"],
            "global_index": window["global_index"],
            "action_noise_seed": window["action_noise_seed"],
        }
        for window in selected_windows
    ]
    checkpoint_bindings = {
        str(step): {
            "whole_bundle_sha256": accepted[step][0]["checkpoint_artifact_hashes"][
                "whole_bundle_sha256"
            ],
            "weight_tensor_sha256": accepted[step][0]["checkpoint_artifact_hashes"][
                "weight_tensor_sha256"
            ],
            "acceptance_sha256": sha256_json(accepted[step][1]),
        }
        for step in CHECKPOINT_STEPS
    }
    metadata = {
        "format": PREDICTION_EVIDENCE_FORMAT,
        "matrix_id": MATRIX_ID,
        "seed": seed,
        "causal_route_id": route,
        "causal_preflight_sha256": causal_sha,
        "hardened_preflight_sha256": hardened_sha,
        "checkpoint_manifest_sha256": manifest_sha,
        "fixed_window_manifest_sha256": windows_sha,
        "fixed_window_schedule_sha256": sha256_json(schedule),
        "checkpoint_bindings": checkpoint_bindings,
        "base_model_artifact_tree": public_tree_receipt(base_snapshot),
        "environment_receipt": environment,
        "evaluator_source_receipt_sha256": causal[
            "current_head_source_tree_sha256"
        ],
        "prediction_source_column_receipt": prediction_source_receipt,
        "target_columns_or_sidecars_read": False,
        "row_count": len(steps),
    }
    packet_array = (
        np.empty((len(steps), 0, 22), dtype=np.float32)
        if route == "aux_only_direct_q12"
        else np.stack(packets).astype(np.float32, copy=False)
    )
    digest = _write_prediction_evidence(
        output,
        metadata=metadata,
        checkpoint_steps=steps,
        window_ids=window_ids,
        action_predictions=np.stack(actions),
        semantic_aux_logits=np.stack(logits),
        contact_hard_semantics=np.stack(hard_semantics),
        route_packets=packet_array,
    )
    print(json.dumps({"prediction": str(output), "sha256": digest}, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    output_dir = ensure_not_project_tmp(args.output_dir, project_root=REPOSITORY_ROOT)
    if args.overwrite:
        raise CausalProductionError("production causal evaluation forbids overwrite")
    output_dir.mkdir(parents=True, exist_ok=False)
    causal, causal_sha, hardened, hardened_sha, assignments = _source_contract(
        args.causal_preflight_receipt, args.hardened_preflight_receipt
    )
    try:
        windows, windows_sha = load_fixed_window_manifest(
            args.window_manifest,
            preflight_receipt=hardened,
            preflight_receipt_sha256=hardened_sha,
            split_assignments=assignments,
        )
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("fixed held-out windows replay failed") from exc
    manifest, manifest_sha, _ = strict_json_file(
        args.checkpoint_manifest, name="causal checkpoint manifest v3"
    )
    manifest, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
    seed = causal["training_data_receipt"]["split_seed"]
    route = manifest["causal_route_id"]
    if manifest["seed"] != seed or manifest["causal_route_id"] != route:
        raise CausalProductionError("checkpoint manifest differs from causal preflight")
    try:
        base_model_snapshot = snapshot_artifact_tree(args.base_model)
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("base-model artifact tree replay failed") from exc
    environment_receipt = build_environment_receipt(
        device=args.device,
        dtype="bfloat16",
        inference_steps=args.inference_steps,
    )
    environment_receipt["test_only_tiny_runtime"] = False
    selected_windows = [window for window in windows["windows"] if window["seed"] == seed]
    if len(selected_windows) != 512:
        raise CausalProductionError("causal evaluator needs 256 val + 256 test windows")

    prediction_path = output_dir / "contact_causal_predictions.npz"
    pycache_prefix = output_dir / ".prediction_worker_pycache"
    pycache_prefix.mkdir(mode=0o700, exist_ok=False)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).absolute()),
        "_predict-phase",
        "--causal-preflight-receipt", str(args.causal_preflight_receipt.absolute()),
        "--hardened-preflight-receipt", str(args.hardened_preflight_receipt.absolute()),
        "--window-manifest", str(args.window_manifest.absolute()),
        "--checkpoint-manifest", str(args.checkpoint_manifest.absolute()),
        "--base-model", str(args.base_model.absolute()),
        "--output", str(prediction_path.absolute()),
        "--device", args.device,
        "--inference-steps", str(args.inference_steps),
    ]
    child_environment = dict(os.environ)
    child_environment["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
    try:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=child_environment,
            check=True,
        )
    finally:
        pycache_prefix.rmdir()
    prediction_metadata, prediction_arrays, prediction_sha = (
        _load_prediction_evidence(prediction_path)
    )
    schedule = [
        {
            "window_id": window["window_id"],
            "global_index": window["global_index"],
            "action_noise_seed": window["action_noise_seed"],
        }
        for window in selected_windows
    ]
    expected_bindings = {
        str(step): {
            "whole_bundle_sha256": accepted[step][0]["checkpoint_artifact_hashes"][
                "whole_bundle_sha256"
            ],
            "weight_tensor_sha256": accepted[step][0]["checkpoint_artifact_hashes"][
                "weight_tensor_sha256"
            ],
            "acceptance_sha256": sha256_json(accepted[step][1]),
        }
        for step in CHECKPOINT_STEPS
    }
    expected_prediction_metadata = {
        "format": PREDICTION_EVIDENCE_FORMAT,
        "matrix_id": MATRIX_ID,
        "seed": seed,
        "causal_route_id": route,
        "causal_preflight_sha256": causal_sha,
        "hardened_preflight_sha256": hardened_sha,
        "checkpoint_manifest_sha256": manifest_sha,
        "fixed_window_manifest_sha256": windows_sha,
        "fixed_window_schedule_sha256": sha256_json(schedule),
        "checkpoint_bindings": expected_bindings,
        "base_model_artifact_tree": public_tree_receipt(base_model_snapshot),
        "environment_receipt": environment_receipt,
        "evaluator_source_receipt_sha256": causal[
            "current_head_source_tree_sha256"
        ],
        "prediction_source_column_receipt": prediction_metadata.get(
            "prediction_source_column_receipt"
        ),
        "target_columns_or_sidecars_read": False,
        "row_count": 3 * 512,
    }
    if prediction_metadata != expected_prediction_metadata:
        raise CausalProductionError("prediction worker evidence binding differs")
    expected_steps = np.repeat(np.asarray(CHECKPOINT_STEPS, dtype=np.int64), 512)
    expected_window_ids = np.asarray(
        [window["window_id"] for _ in CHECKPOINT_STEPS for window in selected_windows],
        dtype=np.str_,
    )
    if (
        not np.array_equal(prediction_arrays["checkpoint_steps"], expected_steps)
        or not np.array_equal(prediction_arrays["window_ids"], expected_window_ids)
        or any(
            array.shape[0] != 3 * 512
            for array in prediction_arrays.values()
        )
    ):
        raise CausalProductionError("prediction worker row order differs")

    cache_parent = output_dir / ".metric_target_private_hf_cache"
    cache_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_stat = cache_parent.lstat()
    if (
        cache_parent.is_symlink()
        or not cache_parent.is_dir()
        or cache_stat.st_uid != os.geteuid()
        or stat.S_IMODE(cache_stat.st_mode) != 0o700
    ):
        raise CausalProductionError(
            "causal evaluator private HF cache parent must be owned mode 0700"
        )
    scorer_nonce = "causal-score-" + hashlib.sha256(
        (
            manifest["training_run_nonce"] + ":" + route + ":" + str(seed)
        ).encode("utf-8")
    ).hexdigest()

    from world_action_model.contact_causal_eval_runtime import (
        CausalMetricTargetReader,
    )

    target_reader = CausalMetricTargetReader(
        preflight_receipt=hardened,
        verified_private_cache_parent=str(cache_parent),
        verified_run_nonce=scorer_nonce,
    )
    rows = []
    step_digests = {}
    evidence_steps = []
    evidence_window_ids = []
    evidence_actions = []
    evidence_action_targets = []
    evidence_action_dim_masks = []
    evidence_action_time_masks = []
    evidence_action_denorm_scales = []
    evidence_logits = []
    evidence_hard_semantics = []
    evidence_contact_targets = []
    evidence_contact_loss_masks = []
    evidence_contact_confidences = []
    evidence_contact_condition_masks = []
    evidence_packets = prediction_arrays["route_packets"]
    try:
        targets_by_window = {
            window["window_id"]: target_reader.load(window)
            for window in selected_windows
        }
        target_source_receipt = dict(target_reader.source_column_receipt)
        for row_index, (step, window_id) in enumerate(
            zip(expected_steps.tolist(), expected_window_ids.tolist(), strict=True)
        ):
            receipt, _ = accepted[step]
            window = selected_windows[row_index % 512]
            targets = targets_by_window[window_id]
            prediction = {
                "action_prediction": prediction_arrays["action_predictions"][row_index],
                "contact_logits": prediction_arrays["semantic_aux_logits"][row_index],
                "contact_hard_semantic": prediction_arrays[
                    "contact_hard_semantics"
                ][row_index],
            }
            if row_index % 512 == 0:
                digests = {
                    "action": hashlib.sha256(),
                    "logits": hashlib.sha256(),
                    "packet": hashlib.sha256(),
                }
            _digest_array(digests["action"], window_id, prediction["action_prediction"])
            _digest_array(digests["logits"], window_id, prediction["contact_logits"])
            if route != "aux_only_direct_q12":
                _digest_array(
                    digests["packet"], window_id, evidence_packets[row_index]
                )
            evidence_steps.append(step)
            evidence_window_ids.append(window_id)
            evidence_actions.append(prediction["action_prediction"])
            evidence_action_targets.append(targets["action"].cpu().numpy())
            evidence_action_dim_masks.append(
                targets["action_dim_mask"].cpu().numpy()
            )
            evidence_action_time_masks.append(
                targets["action_time_mask"].cpu().numpy()
            )
            evidence_action_denorm_scales.append(
                targets["action_denorm_scale"].cpu().numpy()
            )
            evidence_logits.append(prediction["contact_logits"])
            evidence_hard_semantics.append(prediction["contact_hard_semantic"])
            evidence_contact_targets.append(
                targets["contact_target"].cpu().numpy()
            )
            evidence_contact_loss_masks.append(
                targets["contact_loss_mask"].cpu().numpy()
            )
            evidence_contact_confidences.append(
                targets["contact_confidence"].cpu().numpy()
            )
            evidence_contact_condition_masks.append(
                targets["contact_condition_mask"].cpu().numpy()
            )
            rows.append(
                score_heldout_window(
                    window=window,
                    split_seed=seed,
                    checkpoint_step=step,
                    checkpoint_sha256=receipt["checkpoint_artifact_hashes"]["whole_bundle_sha256"],
                    action_prediction=prediction["action_prediction"],
                    action_target=targets["action"].cpu().numpy(),
                    action_dim_mask=targets["action_dim_mask"].cpu().numpy(),
                    action_time_mask=targets["action_time_mask"].cpu().numpy(),
                    action_denorm_scale=targets["action_denorm_scale"].cpu().numpy(),
                    action_schema_contract=ASYMMETRIC25_ACTION_SCHEMA_CONTRACT,
                    grip_active_future=None,
                    contact_logits=prediction["contact_logits"],
                    contact_hard_semantic=prediction["contact_hard_semantic"],
                    contact_target=targets["contact_target"].cpu().numpy(),
                    contact_loss_mask=targets["contact_loss_mask"].cpu().numpy(),
                    contact_confidence=targets["contact_confidence"].cpu().numpy(),
                    contact_condition_mask=targets["contact_condition_mask"].cpu().numpy(),
                )
            )
            if row_index % 512 == 511:
                step_digests[step] = {
                    name: digest.hexdigest() for name, digest in digests.items()
                }
    finally:
        target_reader.close()
        cache_parent.rmdir()
    try:
        verify_artifact_tree_unchanged(base_model_snapshot)
    except HeldoutEvaluationError as exc:
        raise CausalProductionError("base-model artifact changed during evaluation") from exc
    if len(rows) != 1536:
        raise CausalProductionError("causal evaluator emitted an incomplete row matrix")
    csv_path = output_dir / "contact_causal_heldout_per_window.csv"
    csv_sha = write_per_window_csv(csv_path, rows, overwrite=args.overwrite)
    route_evidence_path = output_dir / "contact_causal_route_outputs.npz"
    action_evidence = np.stack(evidence_actions).astype(np.float32, copy=False)
    logits_evidence = np.stack(evidence_logits).astype(np.float32, copy=False)
    route_packet_evidence = (
        np.empty((len(evidence_steps), 0, 22), dtype=np.float32)
        if route == "aux_only_direct_q12"
        else np.stack(evidence_packets).astype(np.float32, copy=False)
    )
    route_evidence_sha = write_route_output_evidence(
        route_evidence_path,
        route=route,
        checkpoint_steps=evidence_steps,
        window_ids=evidence_window_ids,
        action_predictions=action_evidence,
        action_targets=np.stack(evidence_action_targets),
        action_dim_masks=np.stack(evidence_action_dim_masks),
        action_time_masks=np.stack(evidence_action_time_masks),
        action_denorm_scales=np.stack(evidence_action_denorm_scales),
        semantic_aux_logits=logits_evidence,
        contact_hard_semantics=np.stack(evidence_hard_semantics),
        contact_targets=np.stack(evidence_contact_targets),
        contact_loss_masks=np.stack(evidence_contact_loss_masks),
        contact_confidences=np.stack(evidence_contact_confidences),
        contact_condition_masks=np.stack(evidence_contact_condition_masks),
        route_packets=route_packet_evidence,
        overwrite=args.overwrite,
    )
    action_contract_sha = sha256_json(ASYMMETRIC25_ACTION_SCHEMA_CONTRACT)
    contact_contract_sha = sha256_json(
        {"source": "shared_semantic_aux_decoder_logits", "q": 12, "width": 22, "b": 16}
    )
    checkpoint_outputs = []
    manifest_entries = {row["step"]: row for row in manifest["checkpoints"]}
    for step in CHECKPOINT_STEPS:
        reduced = {
            split: reduce_metric_rows(
                [row for row in rows if row["checkpoint_step"] == step and row["split"] == split]
            )
            for split in ("val", "test")
        }
        if route == "aux_only_direct_q12":
            packet_sha = None
            packet_metrics = {
                "kind": "none_direct_reference",
                "semantic_field_metrics_allowed": False,
            }
        else:
            packet_sha = step_digests[step]["packet"]
            step_mask = np.asarray(evidence_steps, dtype=np.int64) == step
            packet_metrics = _recompute_route_packet_metrics(
                route=route,
                logits=logits_evidence[step_mask],
                packets=route_packet_evidence[step_mask],
                reduced_contact_by_split={
                    split: reduced[split]["contact"] for split in ("val", "test")
                },
            )
        route_output = {
            "evaluator_version": EVALUATOR_VERSION,
            "matrix_id": MATRIX_ID,
            "causal_route_id": route,
            "contact_aux_prediction_source": "shared_semantic_aux_decoder_logits",
            "contact_target_model_input": False,
            "contact_condition_mask_model_input": False,
            "oracle_used": False,
            "packet_kind": {
                "aux_only_direct_q12": "none_direct_reference",
                "equal_bit_unstructured_hard_st": "unstructured_code",
                "hard_st_semantic": "semantic_grammar",
            }[route],
            "packet_present": route != "aux_only_direct_q12",
            "packet_sha256": packet_sha,
            "semantic_aux_logits_sha256": step_digests[step]["logits"],
            "action_prediction_sha256": step_digests[step]["action"],
            "action_metrics_contract_sha256": action_contract_sha,
            "packet_metrics": packet_metrics,
        }
        checkpoint_outputs.append(
            {
                "checkpoint_step": step,
                "checkpoint_acceptance_path_record_only": manifest_entries[step]["independent_acceptance_path"],
                "checkpoint_acceptance_sha256": manifest_entries[step]["independent_acceptance_sha256"],
                "fixed_window_set_sha256": windows_sha,
                "route_output": route_output,
                "action_metrics": {split: reduced[split]["action"] for split in ("val", "test")},
                "contact_aux_metrics": {split: reduced[split]["contact"] for split in ("val", "test")},
            }
        )
    result = {
        "result_version": RESULT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "seed": seed,
        "causal_route_id": route,
        "causal_preflight_receipt": {"path_record_only": str(args.causal_preflight_receipt.absolute()), "sha256": causal_sha},
        "hardened_preflight_receipt": {"path_record_only": str(args.hardened_preflight_receipt.absolute()), "sha256": hardened_sha},
        "checkpoint_manifest": {"path_record_only": str(args.checkpoint_manifest.absolute()), "sha256": manifest_sha},
        "base_model_artifact_tree": public_tree_receipt(base_model_snapshot),
        "fixed_window_manifest": {"path_record_only": str(args.window_manifest.absolute()), "sha256": windows_sha},
        "per_window_csv": {"path_record_only": str(csv_path.absolute()), "sha256": csv_sha, "row_count": len(rows)},
        "prediction_evidence": {
            "path_record_only": str(prediction_path.absolute()),
            "sha256": prediction_sha,
            "row_count": len(evidence_steps),
            "format": PREDICTION_EVIDENCE_FORMAT,
        },
        "prediction_source_column_receipt": prediction_metadata[
            "prediction_source_column_receipt"
        ],
        "metric_target_source_receipt": target_source_receipt,
        "prediction_worker_exited_before_target_read": True,
        "route_output_evidence": {
            "path_record_only": str(route_evidence_path.absolute()),
            "sha256": route_evidence_sha,
            "row_count": len(evidence_steps),
            "format": ROUTE_OUTPUT_EVIDENCE_FORMAT,
        },
        "evaluator_source_receipt_sha256": causal["current_head_source_tree_sha256"],
        "precursor_receipt_bytes_sha256": manifest["precursor_receipt_bytes_sha256"],
        "action_metrics_contract_sha256": action_contract_sha,
        "contact_aux_metrics_contract_sha256": contact_contract_sha,
        "evaluation_contract": CAUSAL_EVALUATION_CONTRACT,
        "environment_receipt": environment_receipt,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "primary_checkpoint_step": 200,
        "best_checkpoint_selection_allowed": False,
        "checkpoint_outputs": checkpoint_outputs,
        "oracle_used": False,
        "benefit_claimed": False,
    }
    validate_result_v3(result, verify_files=True, replay_predictions=True)
    result_path = output_dir / "contact_causal_heldout_result.json"
    result_sha = write_strict_json(result_path, result, overwrite=args.overwrite)
    print(json.dumps({"result": str(result_path), "sha256": result_sha}, sort_keys=True))


def aggregate(args: argparse.Namespace) -> None:
    if len(args.result) != 9:
        raise CausalProductionError("causal aggregate requires exactly nine results")
    paths = {}
    evidence = {}
    for path in args.result:
        result, _, _ = strict_json_file(path, name="causal held-out result")
        validate_result_v3(result, verify_files=True, replay_predictions=True)
        key = f"{result['seed']}/{result['causal_route_id']}"
        if key in paths:
            raise CausalProductionError("duplicate causal aggregate seed/route")
        paths[key] = path
        manifest, _, _ = strict_json_file(
            result["checkpoint_manifest"]["path_record_only"],
            name="causal aggregate checkpoint manifest",
        )
        _, accepted = validate_checkpoint_manifest_v3(manifest, verify_files=True)
        for step, pair in accepted.items():
            evidence[(result["seed"], result["causal_route_id"], step)] = pair
    ordered_paths = {
        f"{seed}/{route}": paths[f"{seed}/{route}"]
        for seed in (6666, 7777, 8888)
        for route in CAUSAL_ROUTES
    }
    output = build_aggregate_v5(
        result_path_by_seed_route=ordered_paths,
        evidence=evidence,
        verify_files=True,
        replay_predictions=True,
    )
    target = ensure_not_project_tmp(args.output, project_root=REPOSITORY_ROOT)
    digest = write_strict_json(target, output, overwrite=args.overwrite)
    print(json.dumps({"aggregate": str(target), "sha256": digest}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Causal-main held-out evaluator v2")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-windows")
    prepare.add_argument("--hardened-preflight-receipt", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--causal-preflight-receipt", type=Path, required=True)
    run.add_argument("--hardened-preflight-receipt", type=Path, required=True)
    run.add_argument("--window-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-manifest", type=Path, required=True)
    run.add_argument("--base-model", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--inference-steps", type=int, default=10)
    run.add_argument("--overwrite", action="store_true")
    predict = commands.add_parser("_predict-phase", help=argparse.SUPPRESS)
    predict.add_argument("--causal-preflight-receipt", type=Path, required=True)
    predict.add_argument("--hardened-preflight-receipt", type=Path, required=True)
    predict.add_argument("--window-manifest", type=Path, required=True)
    predict.add_argument("--checkpoint-manifest", type=Path, required=True)
    predict.add_argument("--base-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", required=True)
    predict.add_argument("--inference-steps", type=int, required=True)
    predict.add_argument(
        "--test-only-tiny-runtime",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    merge = commands.add_parser("aggregate")
    merge.add_argument("--result", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    {
        "prepare-windows": prepare_windows,
        "evaluate": evaluate,
        "_predict-phase": predict_phase,
        "aggregate": aggregate,
    }[
        args.command
    ](args)


if __name__ == "__main__":
    main()

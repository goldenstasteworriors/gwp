#!/usr/bin/env python3
"""Generate one exact robot25 train-only norm behind a reviewed split gate.

This is deliberately separate from ``compute_wam_task_norm.py``.  The legacy
tool pools a whole dataset; this production tool validates a completed group
split, selects one seed's train episodes, reads content-verified parquet bytes,
and emits exact (not sampled) statistics for only those episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_action_model.asymmetric_action_schema import (  # noqa: E402
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    MODEL_ACTION_DIM,
    validate_asymmetric_norm_payload,
)
from world_action_model.asymmetric_hardware_manifest import (  # noqa: E402
    hardware_contract_sha256,
    validate_converted_dataset_info,
)
from world_action_model.contact_asym25_train_norm_contract import (  # noqa: E402
    PRODUCTION_NUMERIC_CONTRACT,
    PRODUCTION_TRAIN_RAW_STATS_VERSION,
    build_expected_train_norm_source_receipt,
    build_norm_source_tree_receipt,
    build_production_split_receipt,
    canonical_json_sha256,
    validate_production_train_norm_metadata,
    validate_production_train_norm_payload_statistics,
)
from world_action_model.verified_io import (  # noqa: E402
    hash_verified_regular_file,
    read_verified_bytes,
    strict_json_loads,
)


GENERATION_GATE_VERSION = "contact-asym25-train-norm-generation-gate/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--inventory-csv", required=True)
    parser.add_argument("--build-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-seed", required=True, type=int, choices=(6666, 7777, 8888))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare-generation-request",
        help="Write a false authorization request; never computes or writes norm values.",
    )
    mode.add_argument("--output", help="Write a new immutable production norm JSON.")
    parser.add_argument(
        "--generation-gate",
        help="Reviewer-authored true gate required with --output.",
    )
    return parser.parse_args()


def _new_json_file(path_value: str, value: object) -> None:
    path = Path(path_value).expanduser().absolute()
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or path.exists() or path.is_symlink():
        raise ValueError(f"output must be a new file under a real existing directory: {path}")
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_verified_json(path: Path, expected_sha256: str, name: str) -> dict:
    payload, digest, _ = read_verified_bytes(path, expected_sha256=expected_sha256)
    value = strict_json_loads(payload, name=name)
    if digest != expected_sha256 or not isinstance(value, dict):
        raise ValueError(f"verified {name} differs")
    return value


def _read_verified_json_unpinned(path_value: str, name: str) -> tuple[dict, str]:
    payload, digest, _ = read_verified_bytes(Path(path_value).expanduser().absolute())
    value = strict_json_loads(payload, name=name)
    if not isinstance(value, dict):
        raise ValueError(f"verified {name} must be an object")
    return value, digest


def _read_csv_bytes(payload: bytes, expected_header: list[str], name: str) -> list[dict]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"verified {name} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != expected_header:
        raise ValueError(f"verified {name} header differs")
    rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"verified {name} rows differ")
    return rows


def _verified_episode_lengths(data_root: Path, source_tree: dict) -> tuple[dict[int, int], str]:
    episode_receipts = [
        row
        for row in source_tree["files"]
        if row["logical_path"] == "meta/episodes.jsonl"
        or (
            row["logical_path"].startswith("meta/episodes/")
            and row["logical_path"].endswith(".parquet")
        )
    ]
    if not episode_receipts:
        raise ValueError("source tree lacks authoritative episode metadata")
    rows = []
    if [row["logical_path"] for row in episode_receipts] == ["meta/episodes.jsonl"]:
        receipt = episode_receipts[0]
        payload, _, _ = read_verified_bytes(
            data_root / receipt["logical_path"], expected_sha256=receipt["sha256"]
        )
        for line_number, line in enumerate(payload.splitlines(), 1):
            if line.strip():
                value = strict_json_loads(
                    line, name=f"verified episodes.jsonl:{line_number}"
                )
                if not isinstance(value, dict):
                    raise ValueError("episode metadata row must be an object")
                rows.append(value)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for receipt in episode_receipts:
            payload, _, _ = read_verified_bytes(
                data_root / receipt["logical_path"],
                expected_sha256=receipt["sha256"],
            )
            table = pq.read_table(
                pa.BufferReader(payload), columns=["episode_index", "length"]
            )
            rows.extend(
                {"episode_index": episode, "length": length}
                for episode, length in zip(
                    table["episode_index"].to_pylist(),
                    table["length"].to_pylist(),
                    strict=True,
                )
            )
    lengths = {}
    for row in rows:
        episode = row.get("episode_index")
        length = row.get("length")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode < 0
            or episode in lengths
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ValueError("authoritative episode metadata differs")
        lengths[episode] = length
    digest_rows = [
        {"path": row["logical_path"], "sha256": row["sha256"]}
        for row in episode_receipts
    ]
    return lengths, canonical_json_sha256(digest_rows)


def _validated_completed_group_split(
    *,
    inventory_path: str,
    build_config_path: str,
    split_manifest_path: str,
    episode_lengths: dict[int, int],
) -> tuple[str, dict[int, dict[str, list[int]]], dict[str, str]]:
    inventory_header = [
        "episode_id",
        "length_frames",
        "subject_id",
        "session_id",
        "operator_id",
        "source_object_instance_id",
        "target_object_instance_id",
        "liquid_id",
        "collection_batch_id",
        "metadata_status",
    ]
    inventory_bytes, inventory_sha256, _ = read_verified_bytes(
        Path(inventory_path).expanduser().absolute()
    )
    inventory_rows = _read_csv_bytes(
        inventory_bytes, inventory_header, "group inventory"
    )
    group_fields = inventory_header[2:9]
    inventory = {}
    for row in inventory_rows:
        try:
            episode = int(row["episode_id"])
            length = int(row["length_frames"])
        except ValueError as exc:
            raise ValueError("group inventory episode/length differs") from exc
        if (
            episode in inventory
            or episode not in episode_lengths
            or length != episode_lengths[episode]
            or row["metadata_status"] != "complete"
            or any(not row[field] for field in group_fields)
        ):
            raise ValueError("group inventory is incomplete or differs from dataset")
        inventory[episode] = row
    if set(inventory) != set(episode_lengths):
        raise ValueError("group inventory does not cover every dataset episode")
    config, build_config_sha256 = _read_verified_json_unpinned(
        build_config_path, "contact build config"
    )
    dataset_id = config.get("dataset_id")
    policy = config.get("split_policy")
    group_fields_by_seed = policy.get("group_fields_by_seed") if isinstance(policy, dict) else None
    seed_columns = {
        6666: "seed_6666_split",
        7777: "seed_7777_split",
        8888: "seed_8888_split",
    }
    if (
        not isinstance(dataset_id, str)
        or not dataset_id.strip()
        or not isinstance(policy, dict)
        or policy.get("strategy_claim_allowed") is not True
        or not isinstance(group_fields_by_seed, dict)
    ):
        raise ValueError("contact build config does not authorize a production split")
    for column in seed_columns.values():
        fields = group_fields_by_seed.get(column)
        if not isinstance(fields, list) or not fields or any(
            field not in group_fields for field in fields
        ):
            raise ValueError(f"production group fields differ for {column}")
    split_header = [
        "episode_id",
        "length_frames",
        "seed_6666_split",
        "seed_7777_split",
        "seed_8888_split",
        "split_unit",
        "split_purpose",
        "ood_eligible",
        "strategy_claim_allowed",
        "metadata_status",
    ]
    split_bytes, split_sha256, _ = read_verified_bytes(
        Path(split_manifest_path).expanduser().absolute()
    )
    split_rows = _read_csv_bytes(split_bytes, split_header, "group split manifest")
    by_episode = {}
    for row in split_rows:
        try:
            episode = int(row["episode_id"])
            length = int(row["length_frames"])
        except ValueError as exc:
            raise ValueError("group split episode/length differs") from exc
        if (
            episode in by_episode
            or episode not in inventory
            or length != episode_lengths[episode]
            or row["split_unit"] != "group"
            or row["split_purpose"] != "production_strategy_evaluation"
            or row["ood_eligible"] != "true"
            or row["strategy_claim_allowed"] != "true"
            or row["metadata_status"] != "complete"
            or any(row[column] not in {"train", "val", "test"} for column in seed_columns.values())
        ):
            raise ValueError("group split is incomplete or non-production")
        by_episode[episode] = row
    if set(by_episode) != set(inventory):
        raise ValueError("group split does not cover every inventory episode")
    assignments = {}
    for seed, column in seed_columns.items():
        fields = group_fields_by_seed[column]
        assignment_by_group = {}
        for episode, row in by_episode.items():
            group = tuple(inventory[episode][field] for field in fields)
            previous = assignment_by_group.setdefault(group, row[column])
            if previous != row[column]:
                raise ValueError(f"group leakage detected for seed {seed}")
        if {row[column] for row in by_episode.values()} != {"train", "val", "test"}:
            raise ValueError(f"seed {seed} has an empty train/val/test partition")
        assignments[seed] = {
            name: sorted(
                episode
                for episode, row in by_episode.items()
                if row[column] == name
            )
            for name in ("train", "val", "test")
        }
    return dataset_id, assignments, {
        "inventory_csv_sha256": inventory_sha256,
        "build_config_sha256": build_config_sha256,
        "split_manifest_sha256": split_sha256,
    }


def _robot25(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != (ASYMMETRIC_ACTION_DIM,) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite robot25")
    padded = np.zeros(MODEL_ACTION_DIM, dtype=np.float64)
    padded[:ASYMMETRIC_ACTION_DIM] = raw
    return padded


def _feature_statistics(values: np.ndarray) -> tuple[dict, dict]:
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != MODEL_ACTION_DIM:
        raise ValueError("production statistic array shape differs")
    q = np.quantile(values, (0.01, 0.99), axis=0, method="linear")

    def values_list(array: np.ndarray) -> list[float]:
        result = [float(item) for item in np.asarray(array, dtype=np.float64)]
        if not all(np.isfinite(item) for item in result):
            raise ValueError("production statistic is non-finite")
        return result

    value_sum = values.sum(axis=0, dtype=np.float64)
    mean = value_sum / float(values.shape[0])
    centered_sum_squares = np.square(
        values - mean, dtype=np.float64
    ).sum(axis=0, dtype=np.float64)
    std = np.sqrt(centered_sum_squares / float(values.shape[0]))
    norm = {
        "mean": values_list(mean),
        "std": values_list(std),
        "min": values_list(values.min(axis=0)),
        "max": values_list(values.max(axis=0)),
        "q01": values_list(q[0]),
        "q99": values_list(q[1]),
    }
    sufficient = {
        "count": int(values.shape[0]),
        "dimension": MODEL_ACTION_DIM,
        "sum": values_list(value_sum),
        "sum_squares": values_list(
            np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        ),
        "centered_sum_squares": values_list(centered_sum_squares),
        "min": norm["min"],
        "max": norm["max"],
    }
    return norm, sufficient


def compute_verified_train_statistics(
    *,
    data_root: Path,
    episode_lengths: dict[int, int],
    source_receipt: dict,
    source_receipt_sha256: str,
) -> tuple[dict, dict, str]:
    """Generator-side implementation; preflight has an independent recomputation."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    train_ids = list(source_receipt["train_episode_ids"])
    train_set = set(train_ids)
    source_tree = source_receipt["source_dataset_tree_receipt"]
    parquet_receipts = [
        dict(row)
        for row in source_tree["files"]
        if row["logical_path"].startswith("data/")
        and row["logical_path"].endswith(".parquet")
    ]
    if not parquet_receipts:
        raise ValueError("source receipt contains no data parquet")
    source_data_files_sha256 = canonical_json_sha256(parquet_receipts)
    global_offsets = {}
    next_index = 0
    for episode_id, episode_length in sorted(episode_lengths.items()):
        global_offsets[episode_id] = next_index
        next_index += episode_length
    coverage = {episode_id: set() for episode_id in episode_lengths}
    selected = {episode_id: {} for episode_id in train_ids}
    required_columns = [
        "index",
        "episode_index",
        "frame_index",
        "observation.state",
        "action",
    ]
    for receipt in parquet_receipts:
        logical_path = receipt["logical_path"]
        payload, digest, identity = read_verified_bytes(
            data_root / logical_path,
            expected_sha256=receipt["sha256"],
        )
        if digest != receipt["sha256"] or identity["size_bytes"] != receipt["size_bytes"]:
            raise ValueError(f"verified parquet receipt differs: {logical_path}")
        table = pq.read_table(pa.BufferReader(payload), columns=required_columns)
        if table.column_names != required_columns:
            raise ValueError(f"verified parquet columns differ: {logical_path}")
        batch = table.to_pydict()
        for position in range(table.num_rows):
            episode_id = batch["episode_index"][position]
            frame_index = batch["frame_index"][position]
            sample_index = batch["index"][position]
            if (
                isinstance(episode_id, bool)
                or not isinstance(episode_id, int)
                or episode_id not in episode_lengths
                or isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < episode_lengths[episode_id]
                or isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index != global_offsets[episode_id] + frame_index
                or frame_index in coverage[episode_id]
            ):
                raise ValueError("parquet episode/frame/index coverage differs")
            coverage[episode_id].add(frame_index)
            if episode_id in train_set:
                selected[episode_id][frame_index] = (
                    _robot25(batch["observation.state"][position], "observation.state"),
                    _robot25(batch["action"][position], "action"),
                )
    for episode_id, episode_length in episode_lengths.items():
        if coverage[episode_id] != set(range(episode_length)):
            raise ValueError(f"parquet coverage differs for episode {episode_id}")

    state_batches = []
    action_batches = []
    selected_raw_digest = hashlib.sha256()
    selected_episode_rows = []
    future = np.arange(48, dtype=np.int64)
    for episode_id in train_ids:
        episode_length = episode_lengths[episode_id]
        rows = selected[episode_id]
        if sorted(rows) != list(range(episode_length)):
            raise ValueError(f"selected train rows differ for episode {episode_id}")
        states = np.stack([rows[frame][0] for frame in range(episode_length)])
        actions = np.stack([rows[frame][1] for frame in range(episode_length)])
        for frame_index in range(episode_length):
            selected_raw_digest.update(struct.pack("<qq", episode_id, frame_index))
            selected_raw_digest.update(
                np.asarray(states[frame_index], dtype="<f8").tobytes(order="C")
            )
            selected_raw_digest.update(
                np.asarray(actions[frame_index], dtype="<f8").tobytes(order="C")
            )
        indices = np.minimum(
            np.arange(episode_length, dtype=np.int64)[:, None] + future[None, :],
            episode_length - 1,
        )
        state_batches.append(states)
        action_batches.append(
            actions[indices].reshape(episode_length * 48, MODEL_ACTION_DIM)
        )
        selected_episode_rows.append(
            {"episode_id": episode_id, "row_count": episode_length}
        )
    state_norm, state_sufficient = _feature_statistics(
        np.concatenate(state_batches, axis=0)
    )
    action_norm, action_sufficient = _feature_statistics(
        np.concatenate(action_batches, axis=0)
    )
    norm_stats = {"observation.state": state_norm, "action": action_norm}
    raw_statistics = {
        "receipt_version": PRODUCTION_TRAIN_RAW_STATS_VERSION,
        "numeric_contract": PRODUCTION_NUMERIC_CONTRACT,
        "train_norm_source_receipt_sha256": source_receipt_sha256,
        "split_seed": source_receipt["split_seed"],
        "train_episode_ids": train_ids,
        "train_episode_ids_sha256": source_receipt["train_episode_ids_sha256"],
        "source_dataset_tree_sha256": source_tree["tree_sha256"],
        "source_data_files_sha256": source_data_files_sha256,
        "selected_train_raw_sha256": selected_raw_digest.hexdigest(),
        "selected_episode_rows": selected_episode_rows,
        "features": {
            "observation.state": state_sufficient,
            "action": action_sufficient,
        },
    }
    if (
        state_sufficient["count"] != source_receipt["num_state_samples"]
        or action_sufficient["count"] != source_receipt["num_action_samples"]
    ):
        raise ValueError("raw train-only counts differ from split receipt")
    return raw_statistics, norm_stats, canonical_json_sha256(raw_statistics)


def _build_generation_request(
    *,
    split_receipt: dict,
    split_receipt_sha256: str,
    group_split_input_sha256: dict[str, str],
    source_tree_receipt: dict,
) -> dict:
    generator_sha256, _ = hash_verified_regular_file(Path(__file__))
    contract_sha256, _ = hash_verified_regular_file(
        REPOSITORY_ROOT / "world_action_model/contact_asym25_train_norm_contract.py"
    )
    return {
        "gate_version": GENERATION_GATE_VERSION,
        "allow_norm_generation": False,
        "group_metadata_complete": True,
        "split_seed": split_receipt["split_seed"],
        "production_split_receipt": split_receipt,
        "production_split_receipt_sha256": split_receipt_sha256,
        "group_split_input_sha256": group_split_input_sha256,
        "source_dataset_tree_receipt": source_tree_receipt,
        "source_dataset_tree_receipt_sha256": canonical_json_sha256(
            source_tree_receipt
        ),
        "generator_code_sha256": generator_sha256,
        "norm_contract_code_sha256": contract_sha256,
        "reviewed_by": None,
        "reviewed_at_utc": None,
    }


def _validate_generation_gate(path_value: str | None, expected_request: dict) -> None:
    if not path_value:
        raise ValueError("--generation-gate is required with --output")
    path = Path(path_value).expanduser().absolute()
    payload, _, _ = read_verified_bytes(path)
    gate = strict_json_loads(payload, name="norm generation gate")
    if not isinstance(gate, dict) or set(gate) != set(expected_request):
        raise ValueError("norm generation gate schema differs")
    fixed = dict(expected_request)
    fixed["allow_norm_generation"] = True
    if gate.get("group_metadata_complete") is not True:
        raise ValueError("norm generation gate group metadata is not complete")
    if gate.get("allow_norm_generation") is not True:
        raise ValueError("norm generation gate is false")
    if not isinstance(gate.get("reviewed_by"), str) or not gate["reviewed_by"].strip():
        raise ValueError("norm generation gate reviewer is missing")
    reviewed_at = gate.get("reviewed_at_utc")
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ValueError("norm generation gate UTC review time is missing")
    fixed["reviewed_by"] = gate["reviewed_by"]
    fixed["reviewed_at_utc"] = reviewed_at
    if gate != fixed:
        raise ValueError("norm generation gate fingerprints differ")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().absolute()
    if data_root.is_symlink() or not data_root.is_dir():
        raise ValueError("production dataset root must be a real directory")
    source_tree_receipt = build_norm_source_tree_receipt(data_root)
    episode_lengths, episode_metadata_sha256 = _verified_episode_lengths(
        data_root, source_tree_receipt
    )
    dataset_id, assignments, group_split_input_sha256 = (
        _validated_completed_group_split(
            inventory_path=args.inventory_csv,
            build_config_path=args.build_config,
            split_manifest_path=args.split_manifest,
            episode_lengths=episode_lengths,
        )
    )
    split_manifest_sha256 = group_split_input_sha256["split_manifest_sha256"]
    selected = assignments[args.split_seed]
    info_row = next(
        row
        for row in source_tree_receipt["files"]
        if row["logical_path"] == "meta/info.json"
    )
    info = _read_verified_json(
        data_root / "meta/info.json", info_row["sha256"], "dataset info"
    )
    if (
        info.get("action_schema_id") != ASYMMETRIC_SCHEMA_ID
        or info.get("dataset_id") != dataset_id
        or int(info.get("total_episodes", -1)) != len(episode_lengths)
        or int(info.get("total_frames", -1)) != sum(episode_lengths.values())
    ):
        raise ValueError("dataset info differs from the robot25 completed group split")
    embedded_hardware = info.get("asymmetric_hardware_manifest")
    if not isinstance(embedded_hardware, dict):
        raise ValueError("dataset info lacks the asymmetric hardware manifest")
    validate_converted_dataset_info(info, embedded_hardware)
    split_receipt, split_receipt_sha256 = build_production_split_receipt(
        dataset_id=dataset_id,
        action_schema_id=ASYMMETRIC_SCHEMA_ID,
        split_seed=args.split_seed,
        train_episode_ids=selected["train"],
        validation_episode_ids=selected["val"],
        heldout_episode_ids=selected["test"],
        dataset_info_sha256=info_row["sha256"],
        episode_metadata_sha256=episode_metadata_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    source_receipt, source_receipt_sha256 = build_expected_train_norm_source_receipt(
        production_split_receipt=split_receipt,
        production_split_receipt_sha256=split_receipt_sha256,
        episode_lengths=episode_lengths,
        source_dataset_tree_receipt=source_tree_receipt,
    )
    request = _build_generation_request(
        split_receipt=split_receipt,
        split_receipt_sha256=split_receipt_sha256,
        group_split_input_sha256=group_split_input_sha256,
        source_tree_receipt=source_tree_receipt,
    )
    if args.prepare_generation_request:
        _new_json_file(args.prepare_generation_request, request)
        print(f"Wrote non-authorizing generation request: {args.prepare_generation_request}")
        return
    _validate_generation_gate(args.generation_gate, request)
    raw_statistics, norm_stats, raw_statistics_sha256 = (
        compute_verified_train_statistics(
            data_root=data_root,
            episode_lengths=episode_lengths,
            source_receipt=source_receipt,
            source_receipt_sha256=source_receipt_sha256,
        )
    )
    mirrored_names = (
        "split_seed",
        "train_episode_ids",
        "train_episode_ids_sha256",
        "production_split_receipt_sha256",
        "split_manifest_sha256",
        "source_dataset_tree_receipt",
        "action_schema_id",
        "physical_dim",
        "model_dim",
        "action_horizon",
        "delta_mask",
        "action_representation",
        "boundary",
        "padding_contract",
        "grip_validity_contract",
        "num_episodes",
        "num_state_samples",
        "num_action_samples",
        "dataset_info_sha256",
        "episode_metadata_sha256",
    )
    metadata = {
        "task_name": "contact_asym25_production_train_only",
        "data_roots": [str(data_root)],
        "hardware_contract_sha256": hardware_contract_sha256(embedded_hardware),
        "calibration_capture_sha256": info["calibration_capture_sha256"],
        "calibration_verification_level": embedded_hardware["calibration"][
            "verification_level"
        ],
        "quantile_sample_limit": 0,
        "production_train_norm_source_receipt": source_receipt,
        "production_train_norm_source_receipt_sha256": source_receipt_sha256,
        "production_train_raw_statistics": raw_statistics,
        "production_train_raw_statistics_sha256": raw_statistics_sha256,
        **{name: source_receipt[name] for name in mirrored_names},
    }
    result = {"norm_stats": norm_stats, "metadata": metadata}
    validate_asymmetric_norm_payload(
        result,
        model_dim=MODEL_ACTION_DIM,
        expected_hardware_contract_sha256=metadata["hardware_contract_sha256"],
        expected_calibration_capture_sha256=metadata[
            "calibration_capture_sha256"
        ],
    )
    validate_production_train_norm_metadata(
        metadata,
        expected_source_receipt=source_receipt,
        expected_source_receipt_sha256=source_receipt_sha256,
        expected_raw_statistics=raw_statistics,
        expected_raw_statistics_sha256=raw_statistics_sha256,
        data_root=data_root,
    )
    validate_production_train_norm_payload_statistics(
        result,
        expected_norm_stats=norm_stats,
    )
    _new_json_file(args.output, result)
    print(
        f"Wrote seed-{args.split_seed} train-only production norm: {args.output}; "
        f"episodes={len(selected['train'])} states={metadata['num_state_samples']} "
        f"actions={metadata['num_action_samples']}"
    )


if __name__ == "__main__":
    main()

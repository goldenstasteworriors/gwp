#!/usr/bin/env python3
"""Verify paired simulation rollouts and evaluate preregistered oracle gates.

The command-line path is intentionally execution-only: a ``spec_only_no_go``
manifest cannot be scored.  This prevents synthetic test records or a scenario
design from being reported as physical simulator evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from validate_sim_oracle_manifest import (
    DEFAULT_READINESS,
    DEFAULT_SPEC,
    ManifestError,
    load_json_value,
    load_object,
    sha256_file,
    validate_manifest,
)


RECORD_VERSION = "wam-sim-oracle-rollout-record/v1"
REQUIRED_EVENTS = ("first_contact", "pour_onset", "pour_offset", "release")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIELD_METRICS = (
    "boundary_F1_at_2px",
    "source_rim_chamfer_px",
    "target_rim_chamfer_px",
    "liquid_surface_F1_at_2px",
    "stream_AUPRC",
    "contact_AUPRC",
    "temporal_jitter_px",
)


class ResultError(ValueError):
    pass


_ARTIFACT_HASH_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_sha256(path: Path) -> str:
    """Hash once per immutable path/size/mtime tuple and detect in-hash changes."""
    resolved = path.expanduser().resolve()
    before = resolved.stat()
    key = (
        str(resolved),
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cached = _ARTIFACT_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(resolved)
    after = resolved.stat()
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"artifact changed while hashing: {resolved}",
    )
    _ARTIFACT_HASH_CACHE[key] = digest
    return digest


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path} must be a number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{path} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{path} must be >= {minimum}")
    return result


def _probability(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    _require(0.0 <= result <= 1.0, f"{path} must lie in [0, 1]")
    return result


def _scan_prohibited(value: Any, prohibited: list[str], path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _scan_prohibited(key, prohibited, f"{path}.<key>")
            _scan_prohibited(child, prohibited, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_prohibited(child, prohibited, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in prohibited:
            _require(term.lower() not in lowered, f"{path} contains prohibited ground-truth term {term!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = load_json_value(line)
            except (json.JSONDecodeError, ManifestError) as exc:
                raise ResultError(f"{path}:{line_number}: {exc}") from exc
            _require(isinstance(value, dict), f"{path}:{line_number} must be a JSON object")
            records.append(value)
    _require(records, f"{path} contains no records")
    return records


def _check_native_receipt(
    receipt: Any,
    spec: dict[str, Any],
    record_path: str,
    *,
    verify_artifacts: bool,
) -> None:
    _require(isinstance(receipt, dict), f"{record_path}.native_gt_receipt must be an object")
    required = spec["native_ground_truth_contract"]["required_channels"]
    _require(set(receipt) == set(required), f"{record_path}.native_gt_receipt channel set mismatch")
    allowed = set(spec["native_ground_truth_contract"]["allowed_source_methods"])
    tolerance = float(spec["timebase"]["camera_to_state_tolerance_s"])
    timeline_hashes: set[str] = set()
    state_namespaces: set[str] = set()
    for channel, expected_method in required.items():
        item = receipt[channel]
        path = f"{record_path}.native_gt_receipt.{channel}"
        _require(isinstance(item, dict), f"{path} must be an object")
        _require(
            set(item)
            == {
                "source_method",
                "artifact_path",
                "artifact_sha256",
                "native_timeline_sha256",
                "state_id_namespace",
                "max_camera_state_skew_s",
            },
            f"{path} keys mismatch",
        )
        method = item.get("source_method")
        _require(method in allowed, f"{path}.source_method is not allowed")
        _require(method == expected_method, f"{path}.source_method={method!r}; expected {expected_method!r}")
        artifact_hash = item.get("artifact_sha256")
        timeline_hash = item.get("native_timeline_sha256")
        namespace = item.get("state_id_namespace")
        _require(isinstance(artifact_hash, str) and SHA256_RE.fullmatch(artifact_hash) is not None, f"{path}.artifact_sha256 invalid")
        _require(isinstance(timeline_hash, str) and SHA256_RE.fullmatch(timeline_hash) is not None, f"{path}.native_timeline_sha256 invalid")
        _require(isinstance(namespace, str) and namespace.strip(), f"{path}.state_id_namespace missing")
        _require(isinstance(item.get("artifact_path"), str) and item["artifact_path"].strip(), f"{path}.artifact_path missing")
        timeline_hashes.add(timeline_hash)
        state_namespaces.add(namespace)
        skew = _finite_number(item.get("max_camera_state_skew_s"), f"{path}.max_camera_state_skew_s", minimum=0.0)
        _require(skew <= tolerance, f"{path} camera-state skew exceeds {tolerance}")
        if verify_artifacts:
            artifact_path = Path(item.get("artifact_path", "")).expanduser().resolve()
            _require(artifact_path.is_file(), f"{path}.artifact_path does not exist")
            _require(_artifact_sha256(artifact_path) == artifact_hash, f"{path} artifact sha256 mismatch")
    _require(len(timeline_hashes) == 1, f"{record_path} GT channels do not share one native timeline")
    _require(len(state_namespaces) == 1, f"{record_path} GT channels do not share one state_id namespace")


def _check_variant_runtime_receipt(
    receipt: Any,
    variant_spec: dict[str, Any],
    record_path: str,
    *,
    verify_artifacts: bool,
) -> None:
    path = f"{record_path}.variant_runtime_receipt"
    _require(isinstance(receipt, dict), f"{path} must be an object")
    _require(
        set(receipt)
        == {
            "action_visible_inputs",
            "contact_supervision",
            "route",
            "variant_config_path",
            "variant_config_sha256",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            "code_revision",
        },
        f"{path} keys mismatch",
    )
    for key in ("action_visible_inputs", "contact_supervision", "route"):
        _require(receipt[key] == variant_spec[key], f"{path}.{key} does not match the preregistered variant")
    for kind in ("variant_config", "checkpoint_manifest"):
        artifact_path = receipt.get(f"{kind}_path")
        artifact_hash = receipt.get(f"{kind}_sha256")
        _require(isinstance(artifact_path, str) and artifact_path.strip(), f"{path}.{kind}_path missing")
        _require(isinstance(artifact_hash, str) and SHA256_RE.fullmatch(artifact_hash) is not None, f"{path}.{kind}_sha256 invalid")
        if verify_artifacts:
            resolved = Path(artifact_path).expanduser().resolve()
            _require(resolved.is_file(), f"{path}.{kind}_path does not exist")
            _require(_artifact_sha256(resolved) == artifact_hash, f"{path}.{kind}_sha256 mismatch")
    code_revision = receipt.get("code_revision")
    _require(isinstance(code_revision, str) and re.fullmatch(r"^[0-9a-f]{40}$", code_revision) is not None, f"{path}.code_revision must be a full git commit")
    if verify_artifacts:
        _check_checkpoint_bundle_manifest(
            Path(receipt["checkpoint_manifest_path"]).expanduser().resolve(),
            code_revision,
            path,
        )


def _check_checkpoint_bundle_manifest(
    manifest_path: Path,
    expected_code_revision: str,
    parent_path: str,
) -> None:
    path = f"{parent_path}.checkpoint_manifest"
    try:
        manifest = load_object(manifest_path)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        raise ResultError(f"{path} is invalid: {exc}") from exc
    _require(
        set(manifest)
        == {
            "schema_version",
            "checkpoint_id",
            "code_revision",
            "created_at_utc",
            "shards",
        },
        f"{path} keys mismatch",
    )
    _require(manifest["schema_version"] == "wam-checkpoint-bundle/v1", f"{path}.schema_version invalid")
    _require(isinstance(manifest["checkpoint_id"], str) and manifest["checkpoint_id"].strip(), f"{path}.checkpoint_id missing")
    _require(manifest["code_revision"] == expected_code_revision, f"{path}.code_revision mismatch")
    _require(isinstance(manifest["created_at_utc"], str) and manifest["created_at_utc"].endswith("Z"), f"{path}.created_at_utc invalid")
    shards = manifest["shards"]
    _require(isinstance(shards, list) and shards, f"{path}.shards must be non-empty")
    seen_paths: set[Path] = set()
    for index, item in enumerate(shards):
        item_path = f"{path}.shards[{index}]"
        _require(isinstance(item, dict) and set(item) == {"path", "sha256", "size_bytes"}, f"{item_path} keys mismatch")
        shard_path = Path(item.get("path", "")).expanduser().resolve()
        shard_hash = item.get("sha256")
        shard_size = item.get("size_bytes")
        _require(shard_path not in seen_paths, f"{item_path}.path is duplicated")
        seen_paths.add(shard_path)
        _require(shard_path.is_file(), f"{item_path}.path does not exist")
        _require(isinstance(shard_hash, str) and SHA256_RE.fullmatch(shard_hash) is not None, f"{item_path}.sha256 invalid")
        _require(isinstance(shard_size, int) and shard_size >= 0, f"{item_path}.size_bytes invalid")
        _require(shard_path.stat().st_size == shard_size, f"{item_path}.size_bytes mismatch")
        _require(_artifact_sha256(shard_path) == shard_hash, f"{item_path}.sha256 mismatch")


def _check_outcomes(outcomes: Any, record_path: str) -> dict[str, Any]:
    _require(isinstance(outcomes, dict), f"{record_path}.outcomes must be an object")
    required = {
        "initial_source_ml",
        "final_source_ml",
        "final_target_ml",
        "outside_ml",
        "overflow_crossed_ml",
        "peak_forbidden_impulse_ns",
        "max_penetration_m",
        "stable_duration_s",
        "transfer_fraction",
        "spill_fraction",
        "overflow_fraction",
        "mass_residual_fraction",
        "spill_positive",
        "collision_positive",
        "success",
        "correct_first_contact",
    }
    _require(set(outcomes) == required, f"{record_path}.outcomes keys mismatch")
    nonnegative = {
        key: _finite_number(outcomes[key], f"{record_path}.outcomes.{key}", minimum=0.0)
        for key in required
        if key not in {"spill_positive", "collision_positive", "success", "correct_first_contact"}
    }
    initial = nonnegative["initial_source_ml"]
    _require(initial > 0.0, f"{record_path}.outcomes.initial_source_ml must be positive")
    derived_transfer = nonnegative["final_target_ml"] / initial
    derived_spill = nonnegative["outside_ml"] / initial
    derived_overflow = nonnegative["overflow_crossed_ml"] / initial
    accounted = nonnegative["final_source_ml"] + nonnegative["final_target_ml"] + nonnegative["outside_ml"]
    derived_mass = abs(initial - accounted) / initial
    derived_collision = nonnegative["peak_forbidden_impulse_ns"] > 0.5 or nonnegative["max_penetration_m"] > 0.002
    derived_spill_positive = derived_spill > 0.02 or derived_overflow > 0.01
    derived_success = (
        derived_transfer >= 0.8
        and derived_spill <= 0.02
        and not derived_collision
        and nonnegative["stable_duration_s"] >= 1.0
    )
    for key, derived in (
        ("transfer_fraction", derived_transfer),
        ("spill_fraction", derived_spill),
        ("overflow_fraction", derived_overflow),
        ("mass_residual_fraction", derived_mass),
    ):
        _require(abs(nonnegative[key] - derived) <= 1e-9, f"{record_path}.outcomes.{key} is not derived exactly")
    for key in ("spill_positive", "collision_positive", "success", "correct_first_contact"):
        _require(isinstance(outcomes[key], bool), f"{record_path}.outcomes.{key} must be boolean")
    _require(outcomes["spill_positive"] == derived_spill_positive, f"{record_path}.outcomes.spill_positive mismatch")
    _require(outcomes["collision_positive"] == derived_collision, f"{record_path}.outcomes.collision_positive mismatch")
    _require(outcomes["success"] == derived_success, f"{record_path}.outcomes.success mismatch")
    return {**nonnegative, **{key: outcomes[key] for key in ("spill_positive", "collision_positive", "success", "correct_first_contact")}}


def _event_time(item: Any, path: str) -> float:
    _require(isinstance(item, dict), f"{path} must be an object")
    return _finite_number(item.get("time_s"), f"{path}.time_s", minimum=0.0)


def _check_events(events: Any, outcomes: dict[str, Any], record_path: str) -> dict[str, float]:
    _require(isinstance(events, dict) and set(events) == {"native", "predicted"}, f"{record_path}.events keys mismatch")
    native = events["native"]
    predicted = events["predicted"]
    _require(isinstance(native, dict) and set(native) == set(REQUIRED_EVENTS), f"{record_path}.events.native keys mismatch")
    _require(isinstance(predicted, dict) and set(predicted) == set(REQUIRED_EVENTS), f"{record_path}.events.predicted keys mismatch")
    native_times = {name: _event_time(native[name], f"{record_path}.events.native.{name}") for name in REQUIRED_EVENTS}
    predicted_times = {name: _event_time(predicted[name], f"{record_path}.events.predicted.{name}") for name in REQUIRED_EVENTS}
    _require(list(native_times.values()) == sorted(native_times.values()), f"{record_path}.native events are not ordered")
    for source_name, source in (("native", native), ("predicted", predicted)):
        first = source["first_contact"]
        _require(
            set(first) == {"time_s", "side", "edge", "site"},
            f"{record_path}.events.{source_name}.first_contact keys mismatch",
        )
        for key in ("side", "edge", "site"):
            _require(isinstance(first.get(key), str) and first[key], f"{record_path}.events.{source_name}.first_contact.{key} missing")
        for name in REQUIRED_EVENTS[1:]:
            _require(
                set(source[name]) == {"time_s"},
                f"{record_path}.events.{source_name}.{name} keys mismatch",
            )
    native_first = native["first_contact"]
    predicted_first = predicted["first_contact"]
    correct = (
        all(predicted_first[key] == native_first[key] for key in ("side", "edge", "site"))
        and abs(predicted_times["first_contact"] - native_times["first_contact"]) <= 0.10
    )
    _require(outcomes["correct_first_contact"] == correct, f"{record_path}.outcomes.correct_first_contact mismatch")
    return {
        f"{name}_abs_error_s": abs(predicted_times[name] - native_times[name])
        for name in REQUIRED_EVENTS
    }


def _check_field_metrics(
    metrics: Any,
    receipt: Any,
    record_path: str,
    *,
    verify_artifacts: bool,
) -> dict[str, float | None]:
    path = f"{record_path}.field_metrics"
    _require(isinstance(metrics, dict) and set(metrics) == set(FIELD_METRICS), f"{path} keys mismatch")
    normalized: dict[str, float | None] = {}
    probability_metrics = {
        "boundary_F1_at_2px",
        "liquid_surface_F1_at_2px",
        "stream_AUPRC",
        "contact_AUPRC",
    }
    for name in FIELD_METRICS:
        value = metrics[name]
        if value is None:
            normalized[name] = None
            continue
        number = _finite_number(value, f"{path}.{name}", minimum=0.0)
        if name in probability_metrics:
            _require(number <= 1.0, f"{path}.{name} must lie in [0, 1]")
        normalized[name] = number

    if all(value is None for value in normalized.values()):
        _require(receipt is None, f"{record_path}.field_metric_receipt must be null when no field metric exists")
        return normalized

    receipt_path = f"{record_path}.field_metric_receipt"
    _require(isinstance(receipt, dict), f"{receipt_path} must be an object")
    _require(
        set(receipt)
        == {
            "prediction_artifact_path",
            "prediction_artifact_sha256",
            "evaluator_path",
            "evaluator_sha256",
            "evaluator_revision",
        },
        f"{receipt_path} keys mismatch",
    )
    for kind in ("prediction_artifact", "evaluator"):
        artifact_path = receipt.get(f"{kind}_path")
        artifact_hash = receipt.get(f"{kind}_sha256")
        _require(isinstance(artifact_path, str) and artifact_path.strip(), f"{receipt_path}.{kind}_path missing")
        _require(isinstance(artifact_hash, str) and SHA256_RE.fullmatch(artifact_hash) is not None, f"{receipt_path}.{kind}_sha256 invalid")
        if verify_artifacts:
            resolved = Path(artifact_path).expanduser().resolve()
            _require(resolved.is_file(), f"{receipt_path}.{kind}_path does not exist")
            _require(_artifact_sha256(resolved) == artifact_hash, f"{receipt_path}.{kind}_sha256 mismatch")
    revision = receipt.get("evaluator_revision")
    _require(isinstance(revision, str) and re.fullmatch(r"^[0-9a-f]{40}$", revision) is not None, f"{receipt_path}.evaluator_revision must be a full git commit")
    return normalized


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    grouped: dict[float, list[bool]] = defaultdict(list)
    for score, label in zip(scores, labels):
        grouped[score].append(label)
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        group_positive = sum(group)
        true_positive += group_positive
        false_positive += len(group) - group_positive
        precision = true_positive / (true_positive + false_positive)
        recall_increment = group_positive / positives
        average_precision += precision * recall_increment
    return average_precision


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else math.nan


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    spill_labels = [item["outcomes"]["spill_positive"] for item in items]
    collision_labels = [item["outcomes"]["collision_positive"] for item in items]
    spill_scores = [item["risk_scores"]["spill"] for item in items]
    collision_scores = [item["risk_scores"]["collision"] for item in items]
    aggregate = {
        "n": len(items),
        "success_rate": _mean(float(item["outcomes"]["success"]) for item in items),
        "correct_first_contact_rate": _mean(float(item["outcomes"]["correct_first_contact"]) for item in items),
        "transfer_fraction": _mean(item["outcomes"]["transfer_fraction"] for item in items),
        "spill_fraction": _mean(item["outcomes"]["spill_fraction"] for item in items),
        "overflow_fraction": _mean(item["outcomes"]["overflow_fraction"] for item in items),
        "collision_rate": _mean(float(item["outcomes"]["collision_positive"]) for item in items),
        "mass_conservation_error": _mean(item["outcomes"]["mass_residual_fraction"] for item in items),
        "spill_AUPRC": _average_precision(spill_labels, spill_scores),
        "spill_AUROC": _roc_auc(spill_labels, spill_scores),
        "collision_AUPRC": _average_precision(collision_labels, collision_scores),
        "Brier_score": _mean(
            (spill_score - float(spill_label)) ** 2
            for spill_score, spill_label in zip(spill_scores, spill_labels)
        ),
        **{
            f"{name}_MAE_ms": 1000.0 * _mean(item["event_errors"][f"{name}_abs_error_s"] for item in items)
            for name in REQUIRED_EVENTS
        },
        "event_F1_at_100ms": _mean(
            float(item["event_errors"][f"{name}_abs_error_s"] <= 0.10)
            for item in items
            for name in REQUIRED_EVENTS
        ),
    }
    for name in FIELD_METRICS:
        values = [item["field_metrics"][name] for item in items if item["field_metrics"][name] is not None]
        aggregate[name] = _mean(values) if values else None
        aggregate[f"{name}_n"] = len(values)
    return aggregate


def verify_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    spec: dict[str, Any],
    execution_manifest_sha256: str,
    *,
    require_complete: bool,
    verify_artifacts: bool,
) -> dict[str, Any]:
    scenarios = {item["scenario_id"]: item for item in manifest["scenarios"]}
    variant_specs = {item["id"]: item for item in spec["benchmark_variants"]}
    prohibited = list(spec["native_ground_truth_contract"]["prohibited_source_terms"])
    normalized: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    scenario_control_hashes: dict[str, set[str]] = defaultdict(set)
    scenario_gt_hashes: dict[str, set[str]] = defaultdict(set)
    run_ids_by_variant: dict[str, set[str]] = defaultdict(set)

    for index, record in enumerate(records):
        path = f"records[{index}]"
        required_record_keys = {
            "schema_version",
            "run_id",
            "variant_id",
            "variant_contract_sha256",
            "variant_runtime_receipt",
            "scenario_id",
            "execution_manifest_sha256",
            "control_fingerprint_sha256",
            "native_gt_receipt",
            "outcomes",
            "events",
            "risk_scores",
            "field_metrics",
            "field_metric_receipt",
        }
        _require(
            set(record) == required_record_keys,
            f"{path} keys mismatch",
        )
        _require(record.get("schema_version") == RECORD_VERSION, f"{path}.schema_version invalid")
        _scan_prohibited(record, prohibited, path)
        variant_id = record.get("variant_id")
        scenario_id = record.get("scenario_id")
        _require(variant_id in variant_specs, f"{path}.variant_id invalid")
        _require(scenario_id in scenarios, f"{path}.scenario_id invalid")
        pair = (variant_id, scenario_id)
        _require(pair not in seen_pairs, f"duplicate variant/scenario record {pair}")
        seen_pairs.add(pair)
        expected_variant_hash = _canonical_sha256(variant_specs[variant_id])
        _require(record.get("variant_contract_sha256") == expected_variant_hash, f"{path}.variant_contract_sha256 mismatch")
        _check_variant_runtime_receipt(
            record.get("variant_runtime_receipt"),
            variant_specs[variant_id],
            path,
            verify_artifacts=verify_artifacts,
        )
        _require(record.get("execution_manifest_sha256") == execution_manifest_sha256, f"{path}.execution_manifest_sha256 mismatch")
        control_hash = record.get("control_fingerprint_sha256")
        _require(isinstance(control_hash, str) and SHA256_RE.fullmatch(control_hash) is not None, f"{path}.control_fingerprint_sha256 invalid")
        scenario_control_hashes[scenario_id].add(control_hash)
        run_id = record.get("run_id")
        _require(isinstance(run_id, str) and run_id.strip(), f"{path}.run_id missing")
        run_ids_by_variant[variant_id].add(run_id)
        _check_native_receipt(record.get("native_gt_receipt"), spec, path, verify_artifacts=verify_artifacts)
        scenario_gt_hashes[scenario_id].add(_canonical_sha256(record["native_gt_receipt"]))
        outcomes = _check_outcomes(record.get("outcomes"), path)
        event_errors = _check_events(record.get("events"), outcomes, path)
        risk_scores = record.get("risk_scores")
        _require(isinstance(risk_scores, dict) and set(risk_scores) == {"spill", "collision"}, f"{path}.risk_scores keys mismatch")
        risk_scores = {name: _probability(value, f"{path}.risk_scores.{name}") for name, value in risk_scores.items()}
        field_metrics = _check_field_metrics(
            record.get("field_metrics"),
            record.get("field_metric_receipt"),
            path,
            verify_artifacts=verify_artifacts,
        )
        normalized.append(
            {
                "variant_id": variant_id,
                "scenario_id": scenario_id,
                "scenario": scenarios[scenario_id],
                "outcomes": outcomes,
                "event_errors": event_errors,
                "risk_scores": risk_scores,
                "field_metrics": field_metrics,
            }
        )

    for scenario_id, hashes in scenario_control_hashes.items():
        _require(len(hashes) == 1, f"matched-control fingerprint differs across variants for {scenario_id}")
    for scenario_id, hashes in scenario_gt_hashes.items():
        _require(len(hashes) == 1, f"native GT receipt differs across variants for {scenario_id}")
    for variant_id, run_ids in run_ids_by_variant.items():
        _require(len(run_ids) == 1, f"variant {variant_id} mixes multiple run_id values")

    if require_complete:
        expected_pairs = {
            (variant_id, scenario_id)
            for variant_id in variant_specs
            for scenario_id in scenarios
        }
        missing = expected_pairs - seen_pairs
        extra = seen_pairs - expected_pairs
        _require(not missing and not extra, f"incomplete paired matrix: missing={len(missing)}, extra={len(extra)}")

    by_variant_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_variant_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in normalized:
        variant = item["variant_id"]
        scenario = item["scenario"]
        by_variant_split[(variant, scenario["split"])].append(item)
        by_variant_condition[(variant, scenario["condition_id"])].append(item)

    metrics = {
        f"{variant}/{split}": _aggregate(items)
        for (variant, split), items in sorted(by_variant_split.items())
    }
    mass_fail_fraction = _mean(
        float(item["outcomes"]["mass_residual_fraction"] > 0.01)
        for item in normalized
    )
    repeatability_violations: list[dict[str, Any]] = []
    for (variant, condition), items in sorted(by_variant_condition.items()):
        if len(items) < 2:
            continue
        transfers = [item["outcomes"]["transfer_fraction"] for item in items]
        spread = max(transfers) - min(transfers)
        if spread > 0.05 + 1e-12:
            repeatability_violations.append(
                {"variant_id": variant, "condition_id": condition, "spread": spread}
            )

    def metric(variant: str, name: str) -> float | None:
        value = metrics.get(f"{variant}/ood_test", {}).get(name)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    direct_success = metric("direct_action", "success_rate")
    oracle_success = metric("oracle_contact", "success_rate")
    predicted_success = metric("predicted_contact", "success_rate")
    direct_contact = metric("direct_action", "correct_first_contact_rate")
    oracle_contact = metric("oracle_contact", "correct_first_contact_rate")
    predicted_contact = metric("predicted_contact", "correct_first_contact_rate")

    oracle_success_gain = None if direct_success is None or oracle_success is None else oracle_success - direct_success
    oracle_contact_gain = None if direct_contact is None or oracle_contact is None else oracle_contact - direct_contact
    predicted_success_recovery = None
    predicted_contact_recovery = None
    if oracle_success_gain is not None and oracle_success_gain > 0 and predicted_success is not None:
        predicted_success_recovery = (predicted_success - direct_success) / oracle_success_gain
    if oracle_contact_gain is not None and oracle_contact_gain > 0 and predicted_contact is not None:
        predicted_contact_recovery = (predicted_contact - direct_contact) / oracle_contact_gain

    rgb_spill_auprc = metric("rgb_only", "spill_AUPRC")
    oracle_fields_spill_auprc = metric("oracle_all_fields", "spill_AUPRC")
    rgb_collision = metric("rgb_only", "collision_rate")
    oracle_fields_collision = metric("oracle_all_fields", "collision_rate")
    spill_gain = None if rgb_spill_auprc is None or oracle_fields_spill_auprc is None else oracle_fields_spill_auprc - rgb_spill_auprc
    collision_reduction = None
    if rgb_collision is not None and rgb_collision > 0 and oracle_fields_collision is not None:
        collision_reduction = (rgb_collision - oracle_fields_collision) / rgb_collision

    gates = {
        "mass_conservation": {
            "pass": mass_fail_fraction <= 0.01,
            "fraction_above_1pct": mass_fail_fraction,
            "threshold_max": 0.01,
        },
        "repeatability": {
            "pass": not repeatability_violations,
            "violation_count": len(repeatability_violations),
            "threshold_max_transfer_spread": 0.05,
            "violations": repeatability_violations,
        },
        "oracle_sufficiency": {
            "pass": oracle_success_gain is not None and oracle_contact_gain is not None and oracle_success_gain >= 0.10 and oracle_contact_gain >= 0.15,
            "ood_success_gain": oracle_success_gain,
            "ood_correct_first_contact_gain": oracle_contact_gain,
            "thresholds": {"success": 0.10, "correct_first_contact": 0.15},
        },
        "predicted_recovery": {
            "pass": predicted_success_recovery is not None and predicted_contact_recovery is not None and predicted_success_recovery >= 0.70 and predicted_contact_recovery >= 0.70,
            "success_recovery": predicted_success_recovery,
            "correct_first_contact_recovery": predicted_contact_recovery,
            "threshold_min": 0.70,
        },
        "field_value": {
            "pass": (spill_gain is not None and spill_gain >= 0.05) or (collision_reduction is not None and collision_reduction >= 0.20),
            "ood_spill_AUPRC_gain": spill_gain,
            "ood_collision_relative_reduction": collision_reduction,
            "thresholds": {"spill_AUPRC_gain": 0.05, "collision_relative_reduction": 0.20},
        },
    }
    all_pass = require_complete and all(item["pass"] for item in gates.values())
    return {
        "schema_version": "wam-sim-oracle-verification-report/v1",
        "benchmark_id": spec["benchmark_id"],
        "execution_manifest_sha256": execution_manifest_sha256,
        "record_count": len(normalized),
        "paired_matrix_required": require_complete,
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all_pass,
        "claim_scope": "controlled simulator oracle ceiling only",
        "real_transparent_reconstruction_claim_allowed": False,
        "real_depth_validity_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic only; overall_pass remains false for an incomplete matrix.",
    )
    args = parser.parse_args()
    try:
        validate_manifest(args.manifest, args.spec, args.readiness, "execution")
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_object(manifest_path)
        spec = load_object(args.spec.expanduser().resolve())
        records = _read_jsonl(args.records.expanduser().resolve())
        report = verify_records(
            records,
            manifest,
            spec,
            sha256_file(manifest_path),
            require_complete=not args.allow_incomplete,
            verify_artifacts=True,
        )
    except (OSError, json.JSONDecodeError, ManifestError, ResultError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

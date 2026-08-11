"""Matched-control collector v3 for the causal-main route matrix.

This is a separate registry and trace schema from ``matched_control_hashes``
v2.  The legacy collector remains exact to soft/hard-ST/detached transports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import torch

from world_action_model.contact_causal_production import (
    CAUSAL_ROUTES,
    COLLECTOR_VERSION,
    CausalProductionError,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    ROUTE_LOCAL_JSON_POINTERS,
    SEEDS,
    canonicalize_exact_pointers,
    ensure_exact_causal_route,
    ensure_not_project_tmp,
    sha256_json,
)
from world_action_model.contact_supervision_contract import (
    grouped_contact_supervision_weights,
)
from world_action_model.matched_control_hashes import canonical_sha256
from world_action_model.matched_control_hashes import (
    SAMPLE_AUGMENTATION_KEY,
    SAMPLE_IDENTITY_KEY,
    SAMPLE_RAW_HASH_KEY,
)
from world_action_model.verified_io import read_verified_bytes, strict_json_loads


TRACE_VERSION = "contact-causal-matched-trace/v3"
COLLECTOR_CONFIG_KEYS = {
    "collector_version",
    "enabled",
    "matrix_schema_version",
    "matrix_id",
    "causal_route_id",
    "seed",
    "run_id",
    "run_nonce",
    "output_dir",
    "stable_contract",
    "route_local_contract",
    "initial_parameter_tensors_sha256",
    "optimizer_parameter_groups",
    "optimizer_step0_state",
    "scheduler_step0_state",
    "precursor_receipt_bytes_sha256",
    "current_head_source_tree_sha256",
    "dataset_contact_coverage_receipt_sha256",
    "all_hashes_equal_across_routes",
    "production_data_admission",
    "allow_5b_training",
    "allow_server_training",
    "benefit_claimed",
}
REQUIRED_HASHES = (
    "ordered_sample_ids",
    "raw_batch",
    "augmentation_parameters",
    "visual_action_timesteps",
    "visual_noise",
    "action_noise",
    "contact_target",
    "contact_loss_mask",
    "contact_confidence",
    "contact_supervision_counts",
    "action_target",
    "action_loss_mask",
    "prompt_t5",
    "pre_forward_rng",
    "initial_parameter_tensors",
    "optimizer_parameter_groups",
    "optimizer_step0_state",
    "scheduler_step0_state",
    "precursor_receipt_bytes",
)


class CausalMatchedControlCollector:
    """Rank-local v3 collector with the trainer's existing record interface."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rank: int,
        world_size: int,
        trainer_seed: int,
        project_dir: str | Path,
    ) -> None:
        config = validate_collector_config_v3(config)
        self.route = ensure_exact_causal_route(config["causal_route_id"])
        self.seed = config["seed"]
        if self.seed != trainer_seed or self.seed not in SEEDS:
            raise CausalProductionError("causal collector seed differs from trainer")
        self.rank = rank
        self.world_size = world_size
        if (
            isinstance(rank, bool)
            or isinstance(world_size, bool)
            or not isinstance(rank, int)
            or not isinstance(world_size, int)
            or world_size <= 0
            or not 0 <= rank < world_size
        ):
            raise CausalProductionError("causal collector runtime topology is invalid")
        self.run_id = config.get("run_id")
        self.run_nonce = config.get("run_nonce")
        self.stable_contract = config.get("stable_contract")
        self.route_local_contract = config.get("route_local_contract")
        self.initial_parameter_tensors_sha256 = config.get(
            "initial_parameter_tensors_sha256"
        )
        self.optimizer_parameter_groups = config.get("optimizer_parameter_groups")
        self.optimizer_step0_state = config.get("optimizer_step0_state")
        self.scheduler_step0_state = config.get("scheduler_step0_state")
        self.precursor_receipt_bytes_sha256 = config.get(
            "precursor_receipt_bytes_sha256"
        )
        output_dir = ensure_not_project_tmp(
            config.get("output_dir", ""), project_root=project_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        safe = f"{self.run_id}__{self.run_nonce}__{self.route}__seed{self.seed}__rank{rank:05d}.jsonl"
        self.output_path = output_dir / safe
        if self.output_path.exists():
            raise FileExistsError("refusing to mix causal matched-control traces")
        self.output_path.touch(exist_ok=False)
        self._micro_steps: dict[int, int] = {}

    @staticmethod
    def _decode_json_list(value: object, *, name: str) -> list[Any]:
        if not isinstance(value, (list, tuple)) or not value:
            raise CausalProductionError(f"causal collector batch lacks {name}")
        decoded = []
        for item in value:
            if not isinstance(item, str):
                raise CausalProductionError(f"causal collector {name} must contain JSON strings")
            decoded.append(strict_json_loads(item.encode("utf-8"), name=name))
        return decoded

    def record(
        self,
        *,
        global_step: int,
        batch_dict: Mapping[str, Any],
        visual_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        visual_noise: torch.Tensor,
        action_noise: torch.Tensor,
    ) -> dict[str, Any]:
        micro_step = self._micro_steps.get(global_step, 0)
        self._micro_steps[global_step] = micro_step + 1
        identities = self._decode_json_list(
            batch_dict.get(SAMPLE_IDENTITY_KEY), name="ordered sample identities"
        )
        augmentation = self._decode_json_list(
            batch_dict.get(SAMPLE_AUGMENTATION_KEY), name="augmentation parameters"
        )
        raw_hashes = batch_dict.get(SAMPLE_RAW_HASH_KEY)
        if not isinstance(raw_hashes, (list, tuple)) or not raw_hashes:
            raise CausalProductionError("causal collector batch lacks raw input hashes")
        required_batch = (
            "contact_target",
            "contact_loss_mask",
            "contact_confidence",
            "action",
            "prompt_embeds",
        )
        missing = [name for name in required_batch if name not in batch_dict]
        if missing:
            raise CausalProductionError(f"causal collector batch fields missing: {missing}")
        action_mask = batch_dict.get("action_loss_mask")
        if action_mask is None:
            action_mask = torch.ones(
                batch_dict["action"].shape[:-1],
                device=batch_dict["action"].device,
            )
        record = build_trace_record(
            route=self.route,
            seed=self.seed,
            global_step=global_step,
            micro_step=micro_step,
            rank=self.rank,
            world_size=self.world_size,
            run_id=self.run_id,
            run_nonce=self.run_nonce,
            stable_contract=self.stable_contract,
            route_local_contract=self.route_local_contract,
            ordered_sample_ids=identities,
            raw_batch=list(raw_hashes),
            augmentation_parameters=augmentation,
            visual_timestep=visual_timestep,
            action_timestep=action_timestep,
            visual_noise=visual_noise,
            action_noise=action_noise,
            contact_target=batch_dict["contact_target"],
            contact_loss_mask=batch_dict["contact_loss_mask"],
            contact_confidence=batch_dict["contact_confidence"],
            action_target=batch_dict["action"],
            action_loss_mask=action_mask,
            prompt_t5=batch_dict["prompt_embeds"],
            pre_forward_rng={"cpu_rng_state": torch.get_rng_state()},
            initial_parameter_tensors_sha256=(
                self.initial_parameter_tensors_sha256
            ),
            optimizer_parameter_groups=self.optimizer_parameter_groups,
            optimizer_step0_state=self.optimizer_step0_state,
            scheduler_step0_state=self.scheduler_step0_state,
            precursor_receipt_bytes_sha256=(
                self.precursor_receipt_bytes_sha256
            ),
        )
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def validate_collector_config_v3(value: object) -> dict[str, Any]:
    config = _exact(value, COLLECTOR_CONFIG_KEYS, "causal collector v3 config")
    route = ensure_exact_causal_route(config["causal_route_id"])
    if (
        config["collector_version"] != COLLECTOR_VERSION
        or config["enabled"] is not True
        or config["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or config["matrix_id"] != MATRIX_ID
        or config["seed"] not in SEEDS
        or isinstance(config["seed"], bool)
        or config["all_hashes_equal_across_routes"] is not False
        or config["production_data_admission"] is not False
        or config["allow_5b_training"] is not False
        or config["allow_server_training"] is not False
        or config["benefit_claimed"] is not False
    ):
        raise CausalProductionError("causal collector v3 identity/admission differs")
    import re

    identities = []
    for name in ("run_id", "run_nonce"):
        item = config[name]
        if not isinstance(item, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", item
        ):
            raise CausalProductionError(f"causal collector {name} is invalid")
        identities.append(item)
    if len(set(identities)) != 2:
        raise CausalProductionError("causal collector run identity/nonce overlap")
    if not isinstance(config["output_dir"], str) or not config["output_dir"]:
        raise CausalProductionError("causal collector output directory is missing")
    if not isinstance(config["stable_contract"], dict) or not config["stable_contract"]:
        raise CausalProductionError("causal collector stable contract is missing")
    if not isinstance(config["route_local_contract"], dict) or not config["route_local_contract"]:
        raise CausalProductionError("causal collector route-local contract is missing")
    if config["route_local_contract"].get("causal_route_id") != route:
        raise CausalProductionError("causal collector route-local contract differs")
    for name in (
        "initial_parameter_tensors_sha256",
        "precursor_receipt_bytes_sha256",
        "current_head_source_tree_sha256",
        "dataset_contact_coverage_receipt_sha256",
    ):
        _sha(config[name], f"causal collector {name}")
    for name in (
        "optimizer_parameter_groups",
        "optimizer_step0_state",
        "scheduler_step0_state",
    ):
        if config[name] is None:
            raise CausalProductionError(f"causal collector {name} is missing")
    return config


def contact_supervision_count_receipt(mask: torch.Tensor, confidence: torch.Tensor) -> dict[str, Any]:
    weights = grouped_contact_supervision_weights(mask, confidence)
    field_counts = {
        field: int((weight > 0).sum().item()) for field, weight in sorted(weights.items())
    }
    denominators = {
        field: float(weight.sum().item()) for field, weight in sorted(weights.items())
    }
    total = sum(field_counts.values())
    return {
        "field_eligible_counts": field_counts,
        "field_weighted_denominators": denominators,
        "total_eligible_count": total,
        "zero_supervision": total == 0,
    }


def build_trace_record(
    *,
    route: str,
    seed: int,
    global_step: int,
    micro_step: int,
    rank: int,
    world_size: int,
    run_id: str,
    run_nonce: str,
    stable_contract: Mapping[str, Any],
    route_local_contract: Mapping[str, Any],
    ordered_sample_ids: object,
    raw_batch: object,
    augmentation_parameters: object,
    visual_timestep: torch.Tensor,
    action_timestep: torch.Tensor,
    visual_noise: torch.Tensor,
    action_noise: torch.Tensor,
    contact_target: torch.Tensor,
    contact_loss_mask: torch.Tensor,
    contact_confidence: torch.Tensor,
    action_target: torch.Tensor,
    action_loss_mask: torch.Tensor,
    prompt_t5: torch.Tensor,
    pre_forward_rng: object,
    initial_parameter_tensors_sha256: str,
    optimizer_parameter_groups: object,
    optimizer_step0_state: object,
    scheduler_step0_state: object,
    precursor_receipt_bytes_sha256: str,
) -> dict[str, Any]:
    route = ensure_exact_causal_route(route)
    if seed not in SEEDS:
        raise CausalProductionError("causal collector seed must be 6666/7777/8888")
    for name, value in (
        ("global_step", global_step),
        ("micro_step", micro_step),
        ("rank", rank),
        ("world_size", world_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CausalProductionError(f"collector {name} must be a native integer")
    if global_step < 0 or micro_step < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise CausalProductionError("collector step/rank bounds differ")
    import re

    for name, value in (("run_id", run_id), ("run_nonce", run_nonce)):
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", value
        ):
            raise CausalProductionError(f"collector {name} is invalid")
    if run_id == run_nonce:
        raise CausalProductionError("collector run ID/nonce must be distinct")
    for value, name in (
        (initial_parameter_tensors_sha256, "initial parameter tensor"),
        (precursor_receipt_bytes_sha256, "precursor receipt bytes"),
    ):
        _sha(value, name)
    counts = contact_supervision_count_receipt(contact_loss_mask, contact_confidence)
    hashes = {
        "ordered_sample_ids": canonical_sha256(ordered_sample_ids),
        "raw_batch": canonical_sha256(raw_batch),
        "augmentation_parameters": canonical_sha256(augmentation_parameters),
        "visual_action_timesteps": canonical_sha256(
            {"visual": visual_timestep, "action": action_timestep}
        ),
        "visual_noise": canonical_sha256(visual_noise),
        "action_noise": canonical_sha256(action_noise),
        "contact_target": canonical_sha256(contact_target),
        "contact_loss_mask": canonical_sha256(contact_loss_mask),
        "contact_confidence": canonical_sha256(contact_confidence),
        "contact_supervision_counts": canonical_sha256(counts),
        "action_target": canonical_sha256(action_target),
        "action_loss_mask": canonical_sha256(action_loss_mask),
        "prompt_t5": canonical_sha256(prompt_t5),
        "pre_forward_rng": canonical_sha256(pre_forward_rng),
        "initial_parameter_tensors": initial_parameter_tensors_sha256,
        "optimizer_parameter_groups": canonical_sha256(optimizer_parameter_groups),
        "optimizer_step0_state": canonical_sha256(optimizer_step0_state),
        "scheduler_step0_state": canonical_sha256(scheduler_step0_state),
        "precursor_receipt_bytes": precursor_receipt_bytes_sha256,
    }
    return {
        "record_version": COLLECTOR_VERSION,
        "trace_version": TRACE_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "causal_route_id": route,
        "seed": seed,
        "global_step": global_step,
        "micro_step": micro_step,
        "rank": rank,
        "world_size": world_size,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "stable_contract": dict(stable_contract),
        "stable_contract_sha256": sha256_json(stable_contract),
        "route_local_contract": dict(route_local_contract),
        "route_local_contract_sha256": sha256_json(route_local_contract),
        "contact_supervision_counts": counts,
        "hashes": hashes,
    }


def validate_trace_record(value: object) -> dict[str, Any]:
    record = _exact(
        value,
        {
            "record_version",
            "trace_version",
            "matrix_schema_version",
            "matrix_id",
            "causal_route_id",
            "seed",
            "global_step",
            "micro_step",
            "rank",
            "world_size",
            "run_id",
            "run_nonce",
            "stable_contract",
            "stable_contract_sha256",
            "route_local_contract",
            "route_local_contract_sha256",
            "contact_supervision_counts",
            "hashes",
        },
        "causal matched trace",
    )
    if (
        record["record_version"] != COLLECTOR_VERSION
        or record["trace_version"] != TRACE_VERSION
        or record["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or record["matrix_id"] != MATRIX_ID
    ):
        raise CausalProductionError("causal matched trace version/matrix differs")
    ensure_exact_causal_route(record["causal_route_id"])
    if record["seed"] not in SEEDS:
        raise CausalProductionError("causal matched trace seed differs")
    for name in ("seed", "global_step", "micro_step", "rank", "world_size"):
        if isinstance(record[name], bool) or not isinstance(record[name], int):
            raise CausalProductionError(f"causal matched trace {name} is invalid")
    if record["world_size"] <= 0 or not 0 <= record["rank"] < record["world_size"]:
        raise CausalProductionError("causal matched trace topology differs")
    if sha256_json(record["stable_contract"]) != record["stable_contract_sha256"]:
        raise CausalProductionError("causal matched stable contract digest differs")
    if sha256_json(record["route_local_contract"]) != record["route_local_contract_sha256"]:
        raise CausalProductionError("causal matched route-local digest differs")
    hashes = record["hashes"]
    if not isinstance(hashes, dict) or list(hashes) != list(REQUIRED_HASHES):
        raise CausalProductionError("causal matched hash schema/order differs")
    for name, digest in hashes.items():
        _sha(digest, f"causal matched hash {name}")
    counts = _exact(
        record["contact_supervision_counts"],
        {
            "field_eligible_counts",
            "field_weighted_denominators",
            "total_eligible_count",
            "zero_supervision",
        },
        "trace supervision counts",
    )
    if (
        isinstance(counts["total_eligible_count"], bool)
        or not isinstance(counts["total_eligible_count"], int)
        or counts["total_eligible_count"] < 0
        or counts["zero_supervision"] is not (counts["total_eligible_count"] == 0)
    ):
        raise CausalProductionError("trace supervision zero/count semantics differ")
    return record


def compare_causal_records(
    records_by_route: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_seed: int,
) -> dict[str, Any]:
    if list(records_by_route) != list(CAUSAL_ROUTES):
        raise CausalProductionError("collector v3 requires the exact ordered causal route set")
    indexed: dict[str, dict[tuple[int, int, int], Mapping[str, Any]]] = {}
    global_identities: set[str] = set()
    for route in CAUSAL_ROUTES:
        records = records_by_route[route]
        if not records:
            raise CausalProductionError(f"causal collector trace is empty: {route}")
        current: dict[tuple[int, int, int], Mapping[str, Any]] = {}
        route_ids: set[tuple[str, str]] = set()
        for record in records:
            validate_trace_record(record)
            if record["causal_route_id"] != route or record["seed"] != required_seed:
                raise CausalProductionError("collector trace route/seed differs")
            key = (record["global_step"], record["micro_step"], record["rank"])
            if key in current:
                raise CausalProductionError("collector trace key is duplicated")
            current[key] = record
            route_ids.add((record["run_id"], record["run_nonce"]))
        if len(route_ids) != 1:
            raise CausalProductionError("collector route contains multiple run identities")
        run_id, run_nonce = next(iter(route_ids))
        if run_id in global_identities or run_nonce in global_identities:
            raise CausalProductionError("collector run identity is globally reused")
        global_identities.update({run_id, run_nonce})
        indexed[route] = current
    baseline = indexed[CAUSAL_ROUTES[0]]
    if any(set(records) != set(baseline) for records in indexed.values()):
        raise CausalProductionError("collector route step/micro/rank sets differ")
    for key, reference in baseline.items():
        for route in CAUSAL_ROUTES[1:]:
            record = indexed[route][key]
            if record["stable_contract"] != reference["stable_contract"]:
                raise CausalProductionError("collector stable contract differs across routes")
            if record["hashes"] != reference["hashes"]:
                raise CausalProductionError("collector observed batch/RNG/label/control hashes differ")
            if record["contact_supervision_counts"] != reference["contact_supervision_counts"]:
                raise CausalProductionError("collector supervision counts differ across routes")
    topology = {
        record["world_size"] for records in indexed.values() for record in records.values()
    }
    if len(topology) != 1:
        raise CausalProductionError("collector topology differs across traces")
    world_size = next(iter(topology))
    ranks = sorted({key[2] for key in baseline})
    if ranks != list(range(world_size)):
        raise CausalProductionError("collector rank set is incomplete")
    global_hashes = {
        f"step{key[0]}/micro{key[1]}/rank{key[2]}": baseline[key]["hashes"]
        for key in sorted(baseline)
    }
    return {
        "report_version": COLLECTOR_VERSION,
        "status": "passed",
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "routes": list(CAUSAL_ROUTES),
        "seed": required_seed,
        "matched_record_count": len(baseline),
        "world_size": world_size,
        "rank_set": ranks,
        "required_hashes": list(REQUIRED_HASHES),
        "all_hashes_equal_across_routes": True,
        "global_hashes_sha256": sha256_json(global_hashes),
        "run_id_by_route": {
            route: next(iter({record["run_id"] for record in indexed[route].values()}))
            for route in CAUSAL_ROUTES
        },
        "run_nonce_by_route": {
            route: next(iter({record["run_nonce"] for record in indexed[route].values()}))
            for route in CAUSAL_ROUTES
        },
    }


def load_trace_jsonl(path: str | Path, *, route: str) -> list[dict[str, Any]]:
    route = ensure_exact_causal_route(route)
    try:
        payload, _, _ = read_verified_bytes(path)
        text = payload.decode("utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        raise CausalProductionError("cannot read causal matched trace") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        try:
            value = strict_json_loads(
                line.encode("utf-8"), name=f"causal matched trace line {line_number}"
            )
        except ValueError as exc:
            raise CausalProductionError("invalid strict causal matched JSONL") from exc
        validate_trace_record(value)
        if value["causal_route_id"] != route:
            raise CausalProductionError("causal trace file route differs")
        records.append(value)
    if not records:
        raise CausalProductionError("causal matched trace is empty")
    return records


def write_trace_jsonl_exclusive(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    project_root: str | Path,
) -> None:
    output = ensure_not_project_tmp(path, project_root=project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output, flags, 0o600)
    try:
        for record in records:
            validate_trace_record(record)
            payload = json.dumps(
                record, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8") + b"\n"
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("short write for causal matched trace")
                offset += written
        os.fsync(fd)
    finally:
        os.close(fd)

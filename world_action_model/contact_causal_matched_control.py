"""Matched-control collector v3 for the causal-main route matrix.

This is a separate registry and trace schema from ``matched_control_hashes``
v2.  The legacy collector remains exact to soft/hard-ST/detached transports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
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
    build_production_route_receipt,
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
STABLE_CONTRACT_VERSION = "contact-causal-matched-stable-contract/v3"
STABLE_CONTRACT_KEYS = {
    "stable_contract_version", "code", "config", "dataset", "hardware",
    "normalization", "runtime", "precursor", "labels", "topology", "source_receipts",
}
STABLE_RECEIPT_KEYS = {
    "code": {"git_commit", "source_tree_sha256", "collector_source_sha256"},
    "config": {
        "matrix_manifest_sha256", "control_config_sha256",
        "transform_config_sha256", "optimizer_config_sha256",
        "scheduler_config_sha256",
    },
    "dataset": {
        "dataset_id", "dataset_tree_sha256", "dataset_info_sha256",
        "episode_metadata_sha256", "split_manifest_sha256",
        "ordered_window_manifest_sha256",
    },
    "hardware": {
        "hardware_schema_id", "hardware_contract_sha256",
        "calibration_capture_sha256",
    },
    "normalization": {"train_only_norm_sha256"},
    "runtime": {
        "runtime_environment_sha256",
        "dataset_runtime_source_receipt_sha256",
        "t5_runtime_source_receipt_sha256",
    },
    "precursor": {
        "precursor_receipt_bytes_sha256", "precursor_bundle_sha256",
        "initial_parameter_tensors_sha256",
    },
    "labels": {
        "sidecar_meta_sha256", "sidecar_bundle_sha256",
        "contact_coverage_sha256", "contact_target_stream_sha256",
        "contact_mask_stream_sha256", "contact_confidence_stream_sha256",
        "action_target_stream_sha256", "action_mask_stream_sha256",
        "state_input_stream_sha256", "state_mask_stream_sha256",
        "embodiment_id_stream_sha256", "prompt_t5_stream_sha256",
    },
    "topology": {"world_size", "rank_set"},
    "source_receipts": {
        "training_data_receipt_sha256", "preflight_receipt_sha256",
        "coverage_receipt_sha256", "current_head_source_receipt_sha256",
    },
}
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
    "write_report",
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
    "code_receipt",
    "config_receipt",
    "dataset_receipt",
    "hardware_receipt",
    "normalization_receipt",
    "runtime_receipt",
    "precursor_receipt",
    "labels_receipt",
    "topology_receipt",
    "source_receipts",
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
    "state_input",
    "state_input_mask",
    "embodiment_id",
    "prompt_t5",
    "pre_forward_rng",
    "initial_parameter_tensors",
    "optimizer_parameter_groups",
    "optimizer_step0_state",
    "scheduler_step0_state",
    "precursor_receipt_bytes",
)


def _live_state_receipt(value: Any) -> Any:
    """Convert a live training-state object to strict, typed JSON."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes_sha256": canonical_sha256(tensor),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "bytes_sha256": canonical_sha256(array),
        }
    if isinstance(value, np.generic):
        return _live_state_receipt(np.asarray(value))
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise CausalProductionError("live optimizer/scheduler state is non-finite")
        return value
    if isinstance(value, Mapping):
        rows = []
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise CausalProductionError(
                    "live optimizer/scheduler mapping key must be string/integer"
                )
            rows.append(
                {
                    "key_type": "integer" if isinstance(key, int) else "string",
                    "key": str(key),
                    "value": _live_state_receipt(item),
                }
            )
        rows.sort(key=lambda row: (row["key_type"], row["key"]))
        return {"kind": "mapping", "entries": rows}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return {
            "kind": "sequence",
            "items": [_live_state_receipt(item) for item in value],
        }
    raise CausalProductionError(
        f"unsupported live optimizer/scheduler state: {type(value).__name__}"
    )


def live_optimizer_scheduler_receipts_v3(
    *, model: Any, optimizers: Sequence[Any], schedulers: Sequence[Any]
) -> dict[str, Any]:
    """Read exact named membership/order and state from actual training objects."""

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise CausalProductionError("causal collector live model has no parameters")
    by_identity = {id(parameter): name for name, parameter in named_parameters}
    groups = []
    covered_names = []
    optimizer_states = []
    for optimizer_index, optimizer in enumerate(optimizers):
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        if not hasattr(raw_optimizer, "param_groups") or not hasattr(
            raw_optimizer, "state"
        ):
            raise CausalProductionError("causal collector optimizer object is invalid")
        for group_index, group in enumerate(raw_optimizer.param_groups):
            names = []
            for parameter in group.get("params", []):
                if id(parameter) not in by_identity:
                    raise CausalProductionError(
                        "causal collector optimizer references a parameter outside the live model"
                    )
                names.append(by_identity[id(parameter)])
            covered_names.extend(names)
            groups.append(
                {
                    "optimizer_index": optimizer_index,
                    "optimizer_type": (
                        f"{raw_optimizer.__class__.__module__}."
                        f"{raw_optimizer.__class__.__qualname__}"
                    ),
                    "group_index": group_index,
                    "parameter_names_in_order": names,
                    "hyperparameters": {
                        key: _live_state_receipt(item)
                        for key, item in group.items()
                        if key != "params"
                    },
                }
            )
        state_rows = []
        for parameter, state in raw_optimizer.state.items():
            if id(parameter) not in by_identity:
                raise CausalProductionError(
                    "causal collector optimizer state references an unknown parameter"
                )
            state_rows.append(
                {
                    "parameter_name": by_identity[id(parameter)],
                    "state": _live_state_receipt(state),
                }
            )
        state_rows.sort(key=lambda row: row["parameter_name"])
        optimizer_states.append(
            {
                "optimizer_index": optimizer_index,
                "optimizer_type": (
                    f"{raw_optimizer.__class__.__module__}."
                    f"{raw_optimizer.__class__.__qualname__}"
                ),
                "parameter_state": state_rows,
            }
        )
    expected_names = [name for name, _ in named_parameters]
    if len(covered_names) != len(set(covered_names)) or set(covered_names) != set(
        expected_names
    ):
        raise CausalProductionError(
            "causal collector optimizer groups must cover every live parameter exactly once"
        )
    scheduler_states = []
    for scheduler_index, scheduler in enumerate(schedulers):
        if not hasattr(scheduler, "state_dict"):
            raise CausalProductionError("causal collector scheduler object is invalid")
        scheduler_states.append(
            {
                "scheduler_index": scheduler_index,
                "scheduler_type": (
                    f"{scheduler.__class__.__module__}."
                    f"{scheduler.__class__.__qualname__}"
                ),
                "state": _live_state_receipt(scheduler.state_dict()),
            }
        )
    if not groups or not optimizer_states or not scheduler_states:
        raise CausalProductionError(
            "causal collector requires live optimizer groups/state and scheduler state"
        )
    return {
        "optimizer_parameter_groups": groups,
        "optimizer_step0_state": optimizer_states,
        "scheduler_step0_state": scheduler_states,
    }


def capture_pre_forward_rng_v3() -> dict[str, Any]:
    """Capture all process-global RNG states immediately before model forward."""

    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = list(torch.cuda.get_rng_state_all())
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_by_device": cuda_states,
    }


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
        stable_topology = config["stable_contract"]["topology"]
        if (
            stable_topology["world_size"] != world_size
            or stable_topology["rank_set"] != list(range(world_size))
        ):
            raise CausalProductionError("collector runtime differs from stable topology receipt")
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
        self._live_optimizer_parameter_groups = None
        self._live_optimizer_step0_state = None
        self._live_scheduler_step0_state = None
        self.precursor_receipt_bytes_sha256 = config.get(
            "precursor_receipt_bytes_sha256"
        )
        self.write_report = config["write_report"]
        self.output_path = None
        self._output_fd = None
        if self.write_report:
            output_dir = ensure_not_project_tmp(
                config.get("output_dir", ""), project_root=project_dir
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            safe = f"{self.run_id}__{self.run_nonce}__{self.route}__seed{self.seed}__rank{rank:05d}.jsonl"
            self.output_path = output_dir / safe
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            self._output_fd = os.open(self.output_path, flags, 0o600)
        self._micro_steps: dict[int, int] = {}
        self._record_sha256_by_key: dict[str, str] = {}
        self._records: list[dict[str, Any]] = []

    def bind_live_training_state(
        self, *, model: Any, optimizers: Sequence[Any], schedulers: Sequence[Any]
    ) -> dict[str, Any]:
        """Bind this run to the actual step-0 model/optimizer/scheduler objects."""

        live = live_optimizer_scheduler_receipts_v3(
            model=model, optimizers=optimizers, schedulers=schedulers
        )
        expected = {
            "optimizer_parameter_groups": self.optimizer_parameter_groups,
            "optimizer_step0_state": self.optimizer_step0_state,
            "scheduler_step0_state": self.scheduler_step0_state,
        }
        if live != expected:
            raise CausalProductionError(
                "causal collector configured optimizer/scheduler receipts differ from live step-0 objects"
            )
        if self._live_optimizer_parameter_groups is not None and live != {
            "optimizer_parameter_groups": self._live_optimizer_parameter_groups,
            "optimizer_step0_state": self._live_optimizer_step0_state,
            "scheduler_step0_state": self._live_scheduler_step0_state,
        }:
            raise CausalProductionError("causal collector live step-0 binding changed")
        self._live_optimizer_parameter_groups = live["optimizer_parameter_groups"]
        self._live_optimizer_step0_state = live["optimizer_step0_state"]
        self._live_scheduler_step0_state = live["scheduler_step0_state"]
        return live

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
        action_target: torch.Tensor,
        action_loss_mask: object,
        state_input: torch.Tensor,
        state_input_mask: object,
        embodiment_id: torch.Tensor,
        prompt_t5: torch.Tensor,
        pre_forward_rng: object,
    ) -> dict[str, Any]:
        if self._live_optimizer_parameter_groups is None:
            raise CausalProductionError(
                "causal collector must bind actual step-0 optimizer/scheduler objects before recording"
            )
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
        if (
            len(raw_hashes) != len(identities)
            or len(augmentation) != len(identities)
            or any(_sha(item, "causal raw batch") != item for item in raw_hashes)
        ):
            raise CausalProductionError("causal collector batch trace lengths/hashes differ")
        required_batch = (
            "contact_target",
            "contact_loss_mask",
            "contact_confidence",
        )
        missing = [name for name in required_batch if name not in batch_dict]
        if missing:
            raise CausalProductionError(f"causal collector batch fields missing: {missing}")
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
            action_target=action_target,
            action_loss_mask=action_loss_mask,
            state_input=state_input,
            state_input_mask=state_input_mask,
            embodiment_id=embodiment_id,
            prompt_t5=prompt_t5,
            pre_forward_rng=pre_forward_rng,
            initial_parameter_tensors_sha256=(
                self.initial_parameter_tensors_sha256
            ),
            optimizer_parameter_groups=self._live_optimizer_parameter_groups,
            optimizer_step0_state=self._live_optimizer_step0_state,
            scheduler_step0_state=self._live_scheduler_step0_state,
            precursor_receipt_bytes_sha256=(
                self.precursor_receipt_bytes_sha256
            ),
        )
        if self.write_report:
            payload = json.dumps(
                record, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8") + b"\n"
            offset = 0
            while offset < len(payload):
                written = os.write(self._output_fd, payload[offset:])
                if written <= 0:
                    raise OSError("short write for causal collector trace")
                offset += written
            os.fsync(self._output_fd)
        key = f"step{global_step}/micro{micro_step}/rank{self.rank}"
        if key in self._record_sha256_by_key:
            raise CausalProductionError("causal collector trace key was reused")
        self._record_sha256_by_key[key] = record["record_sha256"]
        self._records.append(record)
        return record

    def current_run_trace_prefix_receipt(self, *, through_step: int) -> dict[str, Any]:
        if isinstance(through_step, bool) or not isinstance(through_step, int) or through_step < 0:
            raise CausalProductionError("causal trace prefix step is invalid")
        if self.output_path is None:
            raise CausalProductionError(
                "production trace prefix requires write_report=true/O_EXCL trace"
            )
        return build_current_run_trace_prefix_receipt_v1(
            self._records,
            route=self.route, seed=self.seed, run_id=self.run_id,
            run_nonce=self.run_nonce, rank=self.rank, world_size=self.world_size,
            through_checkpoint_step=through_step, trace_path=self.output_path,
        )

    def close(self) -> None:
        if self._output_fd is not None:
            os.close(self._output_fd)
            self._output_fd = None


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def validate_stable_contract_v3(value: object) -> dict[str, Any]:
    contract = _exact(value, STABLE_CONTRACT_KEYS, "causal collector stable contract")
    if contract["stable_contract_version"] != STABLE_CONTRACT_VERSION:
        raise CausalProductionError("causal collector stable contract version differs")
    for section, keys in STABLE_RECEIPT_KEYS.items():
        payload = _exact(contract[section], keys, f"stable contract {section}")
        if section == "dataset":
            if not isinstance(payload["dataset_id"], str) or not payload["dataset_id"]:
                raise CausalProductionError("stable dataset identity is missing")
        elif section == "hardware":
            if (
                payload["hardware_schema_id"]
                != "sonic_g1_rh56dftp6_omnipicker1_v1"
            ):
                raise CausalProductionError("stable target-hardware identity differs")
        elif section == "topology":
            world = payload["world_size"]
            ranks = payload["rank_set"]
            if (
                isinstance(world, bool) or not isinstance(world, int) or world <= 0
                or not isinstance(ranks, list) or ranks != list(range(world))
            ):
                raise CausalProductionError("stable collector topology differs")
            continue
        elif section == "code":
            import re

            if not isinstance(payload["git_commit"], str) or not re.fullmatch(
                r"[0-9a-f]{40}", payload["git_commit"]
            ):
                raise CausalProductionError("stable code receipt lacks current HEAD")
        for name, item in payload.items():
            if name.endswith("_sha256"):
                _sha(item, f"stable {section} {name}")
    return contract


def stable_receipt_hashes_v3(value: object) -> dict[str, str]:
    contract = validate_stable_contract_v3(value)
    return {
        ("source_receipts" if section == "source_receipts" else f"{section}_receipt"):
        sha256_json(contract[section])
        for section in STABLE_RECEIPT_KEYS
    }


def runtime_control_config_receipts_v3(
    training_config: Mapping[str, Any], *, route: str
) -> dict[str, str]:
    """Hash the live route-neutral control, transform, optimizer and scheduler.

    Only the versioned causal-route leaves and evidence objects that are
    validated independently are removed.  Any future training-control field is
    therefore stable by default and changes the control digest.
    """

    route = ensure_exact_causal_route(route)
    if not isinstance(training_config, Mapping):
        raise CausalProductionError("collector runtime training config is missing")
    value = deepcopy(dict(training_config))
    value.pop("project_dir", None)
    value.pop("contact_causal_main_integration", None)
    models = value.get("models")
    if not isinstance(models, dict):
        raise CausalProductionError("collector runtime model config is missing")
    for name in (
        "causal_training_data_receipt", "causal_preflight_receipt",
        "causal_checkpoint_receipt_base", "matched_control_collector",
    ):
        models.pop(name, None)
    for name in (
        "contact_causal_route", "production_causal_route_id",
    ):
        if name not in models:
            raise CausalProductionError(f"collector control lacks route leaf: {name}")
        models[name] = "__CAUSAL_ROUTE_LOCAL__"
    transformer = models.get("transformer")
    if not isinstance(transformer, dict):
        raise CausalProductionError("collector runtime transformer config is missing")
    route_leaves = {
        "contact_causal_route": route,
        "contact_bottleneck": route != "aux_only_direct_q12",
        "contact_semantic_bottleneck": route != "aux_only_direct_q12",
    }
    for name, expected in route_leaves.items():
        if transformer.get(name) != expected:
            raise CausalProductionError(f"collector transformer route leaf differs: {name}")
        transformer[name] = "__CAUSAL_ROUTE_LOCAL__"
    required = ("dataloaders", "train", "optimizers", "schedulers", "models")
    if any(name not in value for name in required):
        raise CausalProductionError("collector runtime control config is incomplete")
    control = {name: value[name] for name in sorted(value)}
    train_loader = value["dataloaders"].get("train")
    if not isinstance(train_loader, dict) or not isinstance(
        train_loader.get("transform"), dict
    ):
        raise CausalProductionError("collector runtime transform config is missing")
    return {
        "control_config_sha256": sha256_json(control),
        "transform_config_sha256": sha256_json(train_loader["transform"]),
        "optimizer_config_sha256": sha256_json(value["optimizers"]),
        "scheduler_config_sha256": sha256_json(value["schedulers"]),
    }


def validate_collector_runtime_bindings_v3(
    config: Mapping[str, Any], *, training_config: Mapping[str, Any],
    matrix_manifest_sha256: str, training_data_receipt: Mapping[str, Any],
    training_data_receipt_sha256: str, preflight_receipt_sha256: str,
    coverage_receipt_sha256: str, precursor_binding: Mapping[str, Any],
    sidecar_meta_sha256: str, source_tree_receipt: Mapping[str, Any],
    runtime_environment_sha256: str,
    dataset_runtime_source_receipt_sha256: str,
    t5_runtime_source_receipt_sha256: str,
) -> dict[str, Any]:
    """Recompute every stable source section from the admitted live inputs."""

    config = validate_collector_config_v3(config)
    stable = config["stable_contract"]
    route = config["causal_route_id"]
    for name, value in (
        ("matrix manifest", matrix_manifest_sha256),
        ("training-data receipt", training_data_receipt_sha256),
        ("preflight receipt", preflight_receipt_sha256),
        ("coverage receipt", coverage_receipt_sha256),
        ("sidecar meta", sidecar_meta_sha256),
        ("runtime environment", runtime_environment_sha256),
        ("dataset runtime source receipt", dataset_runtime_source_receipt_sha256),
        ("T5 runtime source receipt", t5_runtime_source_receipt_sha256),
    ):
        _sha(value, name)
    source_sha = sha256_json(source_tree_receipt)
    source_files = source_tree_receipt.get("source_files")
    if not isinstance(source_files, list):
        raise CausalProductionError("collector current-HEAD source receipt is incomplete")
    collector_rows = [
        row for row in source_files
        if isinstance(row, dict)
        and row.get("logical_path")
        == "world_action_model/contact_causal_matched_control.py"
    ]
    if len(collector_rows) != 1:
        raise CausalProductionError("collector source is absent from current-HEAD receipt")
    collector_source_sha = collector_rows[0].get("runtime_sha256")
    _sha(collector_source_sha, "runtime collector source")
    expected_sections = {
        "code": {
            "git_commit": source_tree_receipt.get("git_commit"),
            "source_tree_sha256": source_sha,
            "collector_source_sha256": collector_source_sha,
        },
        "config": {
            "matrix_manifest_sha256": matrix_manifest_sha256,
            **runtime_control_config_receipts_v3(training_config, route=route),
        },
        "dataset": {
            "dataset_id": training_data_receipt.get("dataset_id"),
            "dataset_tree_sha256": training_data_receipt.get("dataset_tree_sha256"),
            "dataset_info_sha256": training_data_receipt.get("dataset_info_sha256"),
            "episode_metadata_sha256": training_data_receipt.get(
                "episode_metadata_sha256"
            ),
            "split_manifest_sha256": training_data_receipt.get(
                "split_manifest_sha256"
            ),
            "ordered_window_manifest_sha256": training_data_receipt.get(
                "ordered_window_manifest_sha256"
            ),
        },
        "hardware": {
            "hardware_schema_id": training_data_receipt.get("hardware_schema_id"),
            "hardware_contract_sha256": training_data_receipt.get(
                "hardware_contract_sha256"
            ),
            "calibration_capture_sha256": training_data_receipt.get(
                "calibration_capture_sha256"
            ),
        },
        "normalization": {
            "train_only_norm_sha256": training_data_receipt.get(
                "train_only_norm_sha256"
            )
        },
        "runtime": {
            "runtime_environment_sha256": runtime_environment_sha256,
            "dataset_runtime_source_receipt_sha256": (
                dataset_runtime_source_receipt_sha256
            ),
            "t5_runtime_source_receipt_sha256": t5_runtime_source_receipt_sha256,
        },
        "precursor": {
            "precursor_receipt_bytes_sha256": precursor_binding.get(
                "receipt_bytes_sha256"
            ),
            "precursor_bundle_sha256": precursor_binding.get("bundle_sha256"),
            "initial_parameter_tensors_sha256": precursor_binding.get(
                "parameter_tensor_sha256"
            ),
        },
        "source_receipts": {
            "training_data_receipt_sha256": training_data_receipt_sha256,
            "preflight_receipt_sha256": preflight_receipt_sha256,
            "coverage_receipt_sha256": coverage_receipt_sha256,
            "current_head_source_receipt_sha256": source_sha,
        },
    }
    for section, expected in expected_sections.items():
        if stable[section] != expected:
            raise CausalProductionError(
                f"collector live {section} receipt differs from stable contract"
            )
    label_expected = {
        "sidecar_meta_sha256": sidecar_meta_sha256,
        "sidecar_bundle_sha256": training_data_receipt.get(
            "sidecar_bundle_sha256"
        ),
        "contact_coverage_sha256": coverage_receipt_sha256,
    }
    if any(stable["labels"].get(name) != value for name, value in label_expected.items()):
        raise CausalProductionError("collector live label receipts differ")
    return config


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
    if not isinstance(config["write_report"], bool):
        raise CausalProductionError("causal collector write_report must be boolean")
    if config["write_report"]:
        if not isinstance(config["output_dir"], str) or not config["output_dir"]:
            raise CausalProductionError("causal collector output directory is missing")
    elif config["output_dir"] is not None:
        raise CausalProductionError("write_report=false requires null output_dir for zero writes")
    validate_stable_contract_v3(config["stable_contract"])
    if not isinstance(config["route_local_contract"], dict) or not config["route_local_contract"]:
        raise CausalProductionError("causal collector route-local contract is missing")
    if config["route_local_contract"].get("causal_route_id") != route:
        raise CausalProductionError("causal collector route-local contract differs")
    if config["route_local_contract"] != build_production_route_receipt(route):
        raise CausalProductionError("causal collector route receipt is not exact")
    for name in (
        "initial_parameter_tensors_sha256",
        "precursor_receipt_bytes_sha256",
        "current_head_source_tree_sha256",
        "dataset_contact_coverage_receipt_sha256",
    ):
        _sha(config[name], f"causal collector {name}")
    stable = config["stable_contract"]
    mirrors = {
        "initial_parameter_tensors_sha256": stable["precursor"][
            "initial_parameter_tensors_sha256"
        ],
        "precursor_receipt_bytes_sha256": stable["precursor"][
            "precursor_receipt_bytes_sha256"
        ],
        "current_head_source_tree_sha256": stable["code"]["source_tree_sha256"],
        "dataset_contact_coverage_receipt_sha256": stable["source_receipts"][
            "coverage_receipt_sha256"
        ],
    }
    if any(config[name] != expected for name, expected in mirrors.items()):
        raise CausalProductionError("causal collector config/stable receipt mirrors differ")
    for name in (
        "optimizer_parameter_groups",
        "optimizer_step0_state",
        "scheduler_step0_state",
    ):
        if config[name] is None:
            raise CausalProductionError(f"causal collector {name} is missing")
    topology = config["stable_contract"]["topology"]
    if topology["world_size"] <= 0:
        raise CausalProductionError("causal collector stable topology is invalid")
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
    action_loss_mask: object,
    state_input: torch.Tensor,
    state_input_mask: object,
    embodiment_id: torch.Tensor,
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
    stable_contract = validate_stable_contract_v3(stable_contract)
    receipt_hashes = stable_receipt_hashes_v3(stable_contract)
    hashes = {
        **receipt_hashes,
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
        "state_input": canonical_sha256(state_input),
        "state_input_mask": canonical_sha256(state_input_mask),
        "embodiment_id": canonical_sha256(embodiment_id),
        "prompt_t5": canonical_sha256(prompt_t5),
        "pre_forward_rng": canonical_sha256(pre_forward_rng),
        "initial_parameter_tensors": initial_parameter_tensors_sha256,
        "optimizer_parameter_groups": canonical_sha256(optimizer_parameter_groups),
        "optimizer_step0_state": canonical_sha256(optimizer_step0_state),
        "scheduler_step0_state": canonical_sha256(scheduler_step0_state),
        "precursor_receipt_bytes": precursor_receipt_bytes_sha256,
    }
    schema_descriptor = {
        "record_keys": [
            "record_version", "trace_version", "matrix_schema_version", "matrix_id",
            "causal_route_id", "seed", "global_step", "micro_step", "rank",
            "world_size", "run_id", "run_nonce", "stable_contract",
            "stable_contract_sha256", "stable_receipt_hashes", "route_local_contract",
            "route_local_contract_sha256", "contact_supervision_counts", "hashes",
            "trace_schema_sha256", "record_sha256",
        ],
        "required_hashes": list(REQUIRED_HASHES),
        "stable_contract_keys": sorted(STABLE_CONTRACT_KEYS),
    }
    record = {
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
        "stable_receipt_hashes": receipt_hashes,
        "route_local_contract": dict(route_local_contract),
        "route_local_contract_sha256": sha256_json(route_local_contract),
        "contact_supervision_counts": counts,
        "hashes": hashes,
        "trace_schema_sha256": sha256_json(schema_descriptor),
    }
    record["record_sha256"] = sha256_json(record)
    return record


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
            "stable_receipt_hashes",
            "route_local_contract",
            "route_local_contract_sha256",
            "contact_supervision_counts",
            "hashes",
            "trace_schema_sha256",
            "record_sha256",
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
    if record["route_local_contract"] != build_production_route_receipt(
        record["causal_route_id"]
    ):
        raise CausalProductionError("causal trace route-local receipt is not exact")
    if record["seed"] not in SEEDS:
        raise CausalProductionError("causal matched trace seed differs")
    for name in ("seed", "global_step", "micro_step", "rank", "world_size"):
        if isinstance(record[name], bool) or not isinstance(record[name], int):
            raise CausalProductionError(f"causal matched trace {name} is invalid")
    if record["world_size"] <= 0 or not 0 <= record["rank"] < record["world_size"]:
        raise CausalProductionError("causal matched trace topology differs")
    stable = validate_stable_contract_v3(record["stable_contract"])
    if sha256_json(stable) != record["stable_contract_sha256"]:
        raise CausalProductionError("causal matched stable contract digest differs")
    receipt_hashes = stable_receipt_hashes_v3(stable)
    if record["stable_receipt_hashes"] != receipt_hashes:
        raise CausalProductionError("causal matched stable receipt hashes differ")
    if sha256_json(record["route_local_contract"]) != record["route_local_contract_sha256"]:
        raise CausalProductionError("causal matched route-local digest differs")
    hashes = record["hashes"]
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_HASHES):
        raise CausalProductionError("causal matched hash schema/order differs")
    for name, digest in hashes.items():
        _sha(digest, f"causal matched hash {name}")
    if any(hashes[name] != digest for name, digest in receipt_hashes.items()):
        raise CausalProductionError("causal matched receipt hash was not recomputed")
    if (
        hashes["initial_parameter_tensors"]
        != stable["precursor"]["initial_parameter_tensors_sha256"]
        or hashes["precursor_receipt_bytes"]
        != stable["precursor"]["precursor_receipt_bytes_sha256"]
    ):
        raise CausalProductionError("causal trace precursor/initial tensor mirrors differ")
    schema_descriptor = {
        "record_keys": [
            "record_version", "trace_version", "matrix_schema_version", "matrix_id",
            "causal_route_id", "seed", "global_step", "micro_step", "rank",
            "world_size", "run_id", "run_nonce", "stable_contract",
            "stable_contract_sha256", "stable_receipt_hashes", "route_local_contract",
            "route_local_contract_sha256", "contact_supervision_counts", "hashes",
            "trace_schema_sha256", "record_sha256",
        ],
        "required_hashes": list(REQUIRED_HASHES),
        "stable_contract_keys": sorted(STABLE_CONTRACT_KEYS),
    }
    if record["trace_schema_sha256"] != sha256_json(schema_descriptor):
        raise CausalProductionError("causal matched trace schema hash differs")
    unsigned = dict(record)
    claimed_record_sha = unsigned.pop("record_sha256")
    _sha(claimed_record_sha, "causal matched record")
    if claimed_record_sha != sha256_json(unsigned):
        raise CausalProductionError("causal matched record hash differs")
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
    stream_bindings = {
        "contact_target_stream_sha256": "contact_target",
        "contact_mask_stream_sha256": "contact_loss_mask",
        "contact_confidence_stream_sha256": "contact_confidence",
        "action_target_stream_sha256": "action_target",
        "action_mask_stream_sha256": "action_loss_mask",
        "state_input_stream_sha256": "state_input",
        "state_mask_stream_sha256": "state_input_mask",
        "embodiment_id_stream_sha256": "embodiment_id",
        "prompt_t5_stream_sha256": "prompt_t5",
    }
    stable_labels = next(iter(baseline.values()))["stable_contract"]["labels"]
    for receipt_name, hash_name in stream_bindings.items():
        replayed = sha256_json(
            {
                f"step{key[0]}/micro{key[1]}/rank{key[2]}": baseline[key][
                    "hashes"
                ][hash_name]
                for key in sorted(baseline)
            }
        )
        if stable_labels[receipt_name] != replayed:
            raise CausalProductionError(
                f"collector stable label stream was not recomputed: {receipt_name}"
            )
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
    trace_manifest = {
        route: {
            f"step{key[0]}/micro{key[1]}/rank{key[2]}":
            indexed[route][key]["record_sha256"]
            for key in sorted(indexed[route])
        }
        for route in CAUSAL_ROUTES
    }
    schema_hashes = {
        record["trace_schema_sha256"]
        for records in indexed.values() for record in records.values()
    }
    if len(schema_hashes) != 1:
        raise CausalProductionError("collector trace schema hashes differ")
    report = {
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
        "trace_manifest_sha256": sha256_json(trace_manifest),
        "schema_sha256": next(iter(schema_hashes)),
        "run_id_by_route": {
            route: next(iter({record["run_id"] for record in indexed[route].values()}))
            for route in CAUSAL_ROUTES
        },
        "run_nonce_by_route": {
            route: next(iter({record["run_nonce"] for record in indexed[route].values()}))
            for route in CAUSAL_ROUTES
        },
    }
    report["report_sha256"] = sha256_json(report)
    return validate_collector_report_v3(report)


def validate_collector_report_v3(value: object) -> dict[str, Any]:
    report = _exact(
        value,
        {
            "report_version", "status", "matrix_schema_version", "matrix_id",
            "routes", "seed", "matched_record_count", "world_size", "rank_set",
            "required_hashes", "all_hashes_equal_across_routes",
            "global_hashes_sha256", "trace_manifest_sha256", "schema_sha256",
            "run_id_by_route", "run_nonce_by_route", "report_sha256",
        },
        "causal collector v3 report",
    )
    if (
        report["report_version"] != COLLECTOR_VERSION
        or report["status"] != "passed"
        or report["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or report["matrix_id"] != MATRIX_ID
        or report["routes"] != list(CAUSAL_ROUTES)
        or report["required_hashes"] != list(REQUIRED_HASHES)
        or report["all_hashes_equal_across_routes"] is not True
        or isinstance(report["seed"], bool) or report["seed"] not in SEEDS
        or isinstance(report["matched_record_count"], bool)
        or not isinstance(report["matched_record_count"], int)
        or report["matched_record_count"] <= 0
        or isinstance(report["world_size"], bool)
        or not isinstance(report["world_size"], int) or report["world_size"] <= 0
        or report["rank_set"] != list(range(report["world_size"]))
    ):
        raise CausalProductionError("causal collector v3 report identity differs")
    for name in (
        "global_hashes_sha256", "trace_manifest_sha256", "schema_sha256",
        "report_sha256",
    ):
        _sha(report[name], f"causal collector report {name}")
    for name in ("run_id_by_route", "run_nonce_by_route"):
        mapping = report[name]
        if not isinstance(mapping, dict) or list(mapping) != list(CAUSAL_ROUTES):
            raise CausalProductionError(f"causal collector report {name} differs")
    identities = list(report["run_id_by_route"].values()) + list(
        report["run_nonce_by_route"].values()
    )
    import re

    if (
        len(set(identities)) != len(identities)
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", item)
            for item in identities
        )
    ):
        raise CausalProductionError("causal collector report identities overlap")
    unsigned = dict(report)
    claimed = unsigned.pop("report_sha256")
    if claimed != sha256_json(unsigned):
        raise CausalProductionError("causal collector report hash differs")
    return report


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


def build_current_run_trace_prefix_receipt_v1(
    records: Sequence[Mapping[str, Any]],
    *,
    route: str,
    seed: int,
    run_id: str,
    run_nonce: str,
    rank: int,
    world_size: int,
    through_checkpoint_step: int,
    trace_path: str | Path,
) -> dict[str, Any]:
    route = ensure_exact_causal_route(route)
    trace_path = Path(trace_path).expanduser().absolute()
    selected: dict[str, str] = {}
    for record in records:
        validate_trace_record(record)
        if (
            record["causal_route_id"] != route
            or record["seed"] != seed
            or record["run_id"] != run_id
            or record["run_nonce"] != run_nonce
            or record["rank"] != rank
            or record["world_size"] != world_size
        ):
            raise CausalProductionError("current-run trace prefix identity differs")
        if record["global_step"] > through_checkpoint_step:
            continue
        key = (
            f"step{record['global_step']}/micro{record['micro_step']}"
            f"/rank{record['rank']}"
        )
        if key in selected:
            raise CausalProductionError("current-run trace prefix key is duplicated")
        selected[key] = record["record_sha256"]
    unsigned = {
        "receipt_version": "contact-causal-current-run-trace-prefix/v1",
        "causal_route_id": route,
        "seed": seed,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "rank": rank,
        "world_size": world_size,
        "through_checkpoint_step": through_checkpoint_step,
        "record_count": len(selected),
        "record_sha256_by_key": dict(sorted(selected.items())),
        "trace_path_record_only": str(trace_path),
    }
    return {**unsigned, "receipt_sha256": sha256_json(unsigned)}


def validate_current_run_trace_prefix_receipt_v1(value: object) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "receipt_version", "causal_route_id", "seed", "run_id", "run_nonce",
            "rank", "world_size", "through_checkpoint_step", "record_count",
            "record_sha256_by_key", "trace_path_record_only", "receipt_sha256",
        },
        "current-run trace prefix receipt",
    )
    route = ensure_exact_causal_route(receipt["causal_route_id"])
    if receipt["receipt_version"] != "contact-causal-current-run-trace-prefix/v1":
        raise CausalProductionError("current-run trace prefix version differs")
    for name in ("seed", "rank", "world_size", "through_checkpoint_step", "record_count"):
        if isinstance(receipt[name], bool) or not isinstance(receipt[name], int):
            raise CausalProductionError(f"current-run trace prefix {name} differs")
    if (
        receipt["seed"] not in SEEDS
        or receipt["world_size"] <= 0
        or not 0 <= receipt["rank"] < receipt["world_size"]
        or receipt["through_checkpoint_step"] < 0
        or receipt["record_count"] < 0
    ):
        raise CausalProductionError("current-run trace prefix bounds differ")
    for name in ("run_id", "run_nonce"):
        if not isinstance(receipt[name], str) or len(receipt[name]) < 16:
            raise CausalProductionError("current-run trace prefix identity differs")
    path = Path(receipt["trace_path_record_only"])
    if not path.is_absolute():
        raise CausalProductionError("current-run trace prefix path must be absolute")
    rows = receipt["record_sha256_by_key"]
    if not isinstance(rows, dict) or list(rows) != sorted(rows):
        raise CausalProductionError("current-run trace prefix manifest is not sorted")
    if len(rows) != receipt["record_count"]:
        raise CausalProductionError("current-run trace prefix count differs")
    for key, digest in rows.items():
        if not isinstance(key, str) or not key:
            raise CausalProductionError("current-run trace prefix key is invalid")
        _sha(digest, "current-run trace record")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    _sha(claimed, "current-run trace prefix receipt")
    if claimed != sha256_json(unsigned):
        raise CausalProductionError("current-run trace prefix receipt digest differs")
    return receipt


def replay_current_run_trace_prefix_v1(value: object) -> dict[str, Any]:
    receipt = validate_current_run_trace_prefix_receipt_v1(value)
    records = load_trace_jsonl(
        receipt["trace_path_record_only"], route=receipt["causal_route_id"]
    )
    rebuilt = build_current_run_trace_prefix_receipt_v1(
        records,
        route=receipt["causal_route_id"], seed=receipt["seed"],
        run_id=receipt["run_id"], run_nonce=receipt["run_nonce"],
        rank=receipt["rank"], world_size=receipt["world_size"],
        through_checkpoint_step=receipt["through_checkpoint_step"],
        trace_path=receipt["trace_path_record_only"],
    )
    if rebuilt != receipt:
        raise CausalProductionError("current-run trace prefix replay differs")
    return receipt


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


def build_synthetic_stable_contract_v3(*, world_size: int = 1) -> dict[str, Any]:
    """Build typed non-production inputs for formal CPU-only collector tests."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise CausalProductionError("synthetic collector world size is invalid")

    def digest(label: str) -> str:
        import hashlib

        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    contract = {
        "stable_contract_version": STABLE_CONTRACT_VERSION,
        "code": {
            "git_commit": "1" * 40,
            "source_tree_sha256": digest("source-tree"),
            "collector_source_sha256": digest("collector-source"),
        },
        "config": {
            "matrix_manifest_sha256": digest("matrix"),
            "control_config_sha256": digest("control"),
            "transform_config_sha256": digest("transform"),
            "optimizer_config_sha256": digest("optimizer"),
            "scheduler_config_sha256": digest("scheduler"),
        },
        "dataset": {
            "dataset_id": "synthetic-reviewed-target-hardware-dataset",
            "dataset_tree_sha256": digest("dataset-tree"),
            "dataset_info_sha256": digest("dataset-info"),
            "episode_metadata_sha256": digest("episodes"),
            "split_manifest_sha256": digest("split"),
            "ordered_window_manifest_sha256": digest("windows"),
        },
        "hardware": {
            "hardware_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
            "hardware_contract_sha256": digest("hardware"),
            "calibration_capture_sha256": digest("capture"),
        },
        "normalization": {"train_only_norm_sha256": digest("norm")},
        "runtime": {
            "runtime_environment_sha256": digest("runtime-environment"),
            "dataset_runtime_source_receipt_sha256": digest("dataset-runtime-source"),
            "t5_runtime_source_receipt_sha256": digest("t5-runtime-source"),
        },
        "precursor": {
            "precursor_receipt_bytes_sha256": digest("precursor-receipt"),
            "precursor_bundle_sha256": digest("precursor-bundle"),
            "initial_parameter_tensors_sha256": digest("initial-parameters"),
        },
        "labels": {
            "sidecar_meta_sha256": digest("sidecar-meta"),
            "sidecar_bundle_sha256": digest("sidecar-bundle"),
            "contact_coverage_sha256": digest("coverage"),
            "contact_target_stream_sha256": digest("contact-target"),
            "contact_mask_stream_sha256": digest("contact-mask"),
            "contact_confidence_stream_sha256": digest("contact-confidence"),
            "action_target_stream_sha256": digest("action-target"),
            "action_mask_stream_sha256": digest("action-mask"),
            "state_input_stream_sha256": digest("state-input"),
            "state_mask_stream_sha256": digest("state-mask"),
            "embodiment_id_stream_sha256": digest("embodiment-id"),
            "prompt_t5_stream_sha256": digest("prompt-t5"),
        },
        "topology": {"world_size": world_size, "rank_set": list(range(world_size))},
        "source_receipts": {
            "training_data_receipt_sha256": digest("training-data"),
            "preflight_receipt_sha256": digest("preflight"),
            "coverage_receipt_sha256": digest("coverage-receipt"),
            "current_head_source_receipt_sha256": digest("head-source"),
        },
    }
    return validate_stable_contract_v3(contract)


def collector_cpu_recording_acceptance_helper(project_dir: str | Path) -> dict[str, Any]:
    """Instantiate the real collector and record a CPU batch with zero writes."""

    import hashlib

    stable = build_synthetic_stable_contract_v3()
    route = CAUSAL_ROUTES[0]
    model = torch.nn.Module()
    model.register_parameter(
        "shared_weight", torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    live = live_optimizer_scheduler_receipts_v3(
        model=model, optimizers=[optimizer], schedulers=[scheduler]
    )
    config = {
        "collector_version": COLLECTOR_VERSION, "enabled": True,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION, "matrix_id": MATRIX_ID,
        "causal_route_id": route, "seed": SEEDS[0],
        "run_id": "synthetic-collector-run-id-v3",
        "run_nonce": "synthetic-collector-run-nonce-v3",
        "output_dir": None, "write_report": False,
        "stable_contract": stable,
        "route_local_contract": build_production_route_receipt(route),
        "initial_parameter_tensors_sha256": stable["precursor"][
            "initial_parameter_tensors_sha256"
        ],
        **live,
        "precursor_receipt_bytes_sha256": stable["precursor"][
            "precursor_receipt_bytes_sha256"
        ],
        "current_head_source_tree_sha256": stable["code"]["source_tree_sha256"],
        "dataset_contact_coverage_receipt_sha256": stable["source_receipts"][
            "coverage_receipt_sha256"
        ],
        "all_hashes_equal_across_routes": False,
        "production_data_admission": False, "allow_5b_training": False,
        "allow_server_training": False, "benefit_claimed": False,
    }
    collector = CausalMatchedControlCollector(
        config, rank=0, world_size=1, trainer_seed=SEEDS[0], project_dir=project_dir
    )
    collector.bind_live_training_state(
        model=model, optimizers=[optimizer], schedulers=[scheduler]
    )
    raw = hashlib.sha256(b"synthetic-raw-batch").hexdigest()
    batch = {
        SAMPLE_IDENTITY_KEY: [json.dumps({"episode": 1, "frame": 2})],
        SAMPLE_AUGMENTATION_KEY: [json.dumps({"crop": [0, 0, 1, 1]})],
        SAMPLE_RAW_HASH_KEY: [raw],
        "contact_target": torch.zeros(1, 1, 22),
        "contact_loss_mask": torch.zeros(1, 1, 22),
        "contact_confidence": torch.zeros(1, 1, 22),
        "action": torch.zeros(1, 2, 4),
        "action_loss_mask": torch.ones(1, 2),
        "prompt_embeds": torch.zeros(1, 2, 3),
    }
    record = collector.record(
        global_step=0, batch_dict=batch,
        visual_timestep=torch.tensor([10]), action_timestep=torch.tensor([20]),
        visual_noise=torch.zeros(1, 1), action_noise=torch.zeros(1, 2, 4),
        action_target=torch.ones(1, 2, 4),
        action_loss_mask={
            "dimension_mask": None,
            "time_mask": torch.ones(1, 2),
            "dimension_weight": None,
        },
        state_input=torch.zeros(1, 1, 3), state_input_mask=None,
        embodiment_id=torch.zeros(1, dtype=torch.long),
        prompt_t5=batch["prompt_embeds"],
        pre_forward_rng=capture_pre_forward_rng_v3(),
    )
    collector.close()
    validate_trace_record(record)
    if collector.output_path is not None:
        raise AssertionError("write_report=false performed an output allocation")
    return record

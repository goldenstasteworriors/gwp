"""Deterministic, opt-in matched-control traces for WAM ablations.

The training-time collector is deliberately rank-local.  It never invokes a
distributed collective and it never samples randomness.  Global hashes are
formed only by :func:`compare_variant_traces` after the runs have finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_action_model.verified_io import read_verified_bytes, strict_json_loads


COLLECTOR_VERSION = "matched-control-hash-collector/v2"
TRACE_VERSION = "wam-matched-control-sample-trace/v1"
ALLOWED_SEEDS = (6666, 7777, 8888)
IMPLEMENTED_RUNNER_VARIANTS = (
    "soft_semantic",
    "hard_st",
    "hard_detached",
)
REQUIRED_HASHES = (
    "ordered_sample_ids",
    "raw_batch",
    "augmentation_parameters",
    "visual_action_timesteps",
    "visual_noise",
    "action_noise",
)
HASH_SCHEMA = {
    "ordered_sample_ids": (
        "ordered [dataset_id, episode_index, frame_index, global_index] identities"
    ),
    "raw_batch": "ordered authoritative pre-augmentation per-sample input hashes",
    "augmentation_parameters": "ordered explicit sampled transform parameter traces",
    "visual_action_timesteps": (
        "typed pair containing the independent visual_timestep and action_timestep tensors"
    ),
    "visual_noise": "visual flow-noise tensor",
    "action_noise": "masked action flow-noise tensor actually used by the objective",
}
SAMPLE_IDENTITY_KEY = "matched_control_sample_identity"
SAMPLE_DATA_PATH_KEY = "matched_control_diagnostic_data_path"
SAMPLE_RAW_HASH_KEY = "matched_control_raw_input_sha256"
SAMPLE_AUGMENTATION_KEY = "matched_control_augmentation_trace"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class MatchedControlError(ValueError):
    """Raised when a trace is incomplete or controls are not matched."""


def normalize_path(path: str | os.PathLike[str]) -> str:
    """Return a normalized absolute diagnostic path.

    Absolute paths are never part of the cross-machine sample identity; they
    are retained only to diagnose an unexpected dataset mount.
    """

    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def _update_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.view(torch.uint8).numpy().tobytes(order="C")


def _update_canonical(digest: Any, value: Any) -> None:
    """Stream a typed canonical representation into ``digest``.

    Tensor/ndarray hashes include dtype, shape, and contiguous raw bytes.
    Scalars are type-tagged; Python floats use their IEEE-754 binary64 bytes.
    """

    if value is None:
        digest.update(b"n")
    elif isinstance(value, bool):
        digest.update(b"b1" if value else b"b0")
    elif isinstance(value, int) and not isinstance(value, bool):
        digest.update(b"i")
        _update_length_prefixed(digest, str(value).encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"f")
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        digest.update(b"s")
        _update_length_prefixed(digest, value.encode("utf-8"))
    elif isinstance(value, bytes):
        digest.update(b"y")
        _update_length_prefixed(digest, value)
    elif isinstance(value, torch.Tensor):
        digest.update(b"t")
        _update_canonical(digest, str(value.dtype))
        _update_canonical(digest, tuple(int(dim) for dim in value.shape))
        _update_length_prefixed(digest, _tensor_bytes(value))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"a")
        _update_canonical(digest, array.dtype.str)
        _update_canonical(digest, tuple(int(dim) for dim in array.shape))
        _update_length_prefixed(digest, array.tobytes(order="C"))
    elif isinstance(value, np.generic):
        _update_canonical(digest, np.asarray(value))
    elif isinstance(value, Mapping):
        digest.update(b"m")
        keys = sorted(value)
        _update_canonical(digest, len(keys))
        for key in keys:
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            _update_canonical(digest, key)
            _update_canonical(digest, value[key])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        digest.update(b"q")
        _update_canonical(digest, len(value))
        for item in value:
            _update_canonical(digest, item)
    else:
        raise TypeError(f"unsupported canonical hash type: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_canonical(digest, value)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"__float64_be__": struct.pack(">d", value).hex()}
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return {
            "__numpy_scalar__": {
                "dtype": array.dtype.str,
                "raw_bytes": array.tobytes().hex(),
            }
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {key: _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def json_sha256(value: Any) -> str:
    """SHA-256 of strict canonical JSON for cross-receipt contracts."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_exclusive_durable(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while persisting matched-control report")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_sample_identity(
    *, dataset_id: Any, episode_index: Any, frame_index: Any, global_index: Any
) -> dict[str, Any]:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise MatchedControlError("matched-control identity requires a non-empty dataset_id")
    dataset_id = dataset_id.strip()

    def scalar_int(name: str, value: Any) -> int:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            raise MatchedControlError(f"{name} cannot be boolean")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise MatchedControlError(f"invalid {name}: {value!r}") from exc

    return {
        "dataset_id": dataset_id,
        "episode_index": scalar_int("episode_index", episode_index),
        "frame_index": scalar_int("frame_index", frame_index),
        "global_index": scalar_int("global_index", global_index),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover_git_commit(start: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else "unavailable"


def _safe_name(value: str) -> str:
    value = _SAFE_NAME_RE.sub("_", value.strip())
    if not value:
        raise MatchedControlError("run and variant names must not be empty")
    return value


def _ensure_project_output(output_dir: Path, project_dir: Path) -> None:
    if output_dir == Path("/tmp") or Path("/tmp") in output_dir.parents:
        raise MatchedControlError("matched-control traces must not be written under /tmp")
    try:
        output_dir.relative_to(project_dir)
    except ValueError as exc:
        raise MatchedControlError(
            f"collector output_dir must be inside project_dir: {project_dir}"
        ) from exc


class MatchedControlCollector:
    """Write one canonical rank-local JSONL row per observed micro-batch."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rank: int,
        world_size: int,
        trainer_seed: int,
        project_dir: str | os.PathLike[str],
    ) -> None:
        if config.get("enabled") is not True:
            raise MatchedControlError("do not construct a disabled collector")
        configured_seed = config.get("seed", trainer_seed)
        if (
            isinstance(configured_seed, bool)
            or not isinstance(configured_seed, int)
            or isinstance(trainer_seed, bool)
            or not isinstance(trainer_seed, int)
        ):
            raise MatchedControlError("collector and trainer seeds must be native integers")
        self.seed = configured_seed
        if self.seed != trainer_seed:
            raise MatchedControlError(
                f"collector seed {self.seed} differs from trainer seed {trainer_seed}"
            )
        if self.seed not in ALLOWED_SEEDS:
            raise MatchedControlError(
                f"collector seed must be one of {ALLOWED_SEEDS}, got {self.seed}"
            )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
        ):
            raise MatchedControlError("rank and world_size must be native integers")
        self.rank = rank
        self.world_size = world_size
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise MatchedControlError(
                f"invalid distributed identity rank={self.rank}, world_size={self.world_size}"
            )
        self.run_id = str(config.get("run_id", "")).strip()
        self.run_nonce = str(config.get("run_nonce", "")).strip()
        self.variant_id = str(config.get("variant_id", "")).strip()
        safe_run = _safe_name(self.run_id)
        safe_nonce = _safe_name(self.run_nonce)
        safe_variant = _safe_name(self.variant_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", self.run_nonce):
            raise MatchedControlError(
                "run_nonce must be an explicit 16-128 character audit nonce"
            )
        if self.variant_id not in IMPLEMENTED_RUNNER_VARIANTS:
            raise MatchedControlError(
                f"variant_id is not an implemented runner: {self.variant_id!r}"
            )
        self.variant_config_sha256 = str(
            config.get("variant_config_sha256", "")
        ).strip()
        self.control_config_sha256 = str(
            config.get("control_config_sha256", "")
        ).strip()
        if not _HEX64_RE.fullmatch(self.variant_config_sha256):
            raise MatchedControlError(
                "collector requires an explicit 64-hex variant_config_sha256"
            )
        if not _HEX64_RE.fullmatch(self.control_config_sha256):
            raise MatchedControlError(
                "collector requires an explicit 64-hex control_config_sha256"
            )
        self.shared_model_config_sha256 = config.get(
            "shared_model_config_sha256"
        )
        self.transport_override = config.get("transport_override")
        expected_transport = {
            "soft_semantic": "soft",
            "hard_st": "hard_st",
            "hard_detached": "hard_detached",
        }[self.variant_id]
        if (
            not isinstance(self.shared_model_config_sha256, str)
            or not _HEX64_RE.fullmatch(self.shared_model_config_sha256)
            or not isinstance(self.transport_override, Mapping)
            or dict(self.transport_override)
            != {
                "variant_id": self.variant_id,
                "contact_semantic_transport_mode": expected_transport,
            }
        ):
            raise MatchedControlError(
                "collector transport override/shared model contract differs"
            )
        self.transport_override = dict(self.transport_override)
        self.training_data_receipt_sha256 = config.get(
            "training_data_receipt_sha256"
        )
        self.production_split_seed = config.get("production_split_seed")
        self.heldout_window_manifest_sha256 = config.get(
            "heldout_window_manifest_sha256"
        )
        if self.training_data_receipt_sha256 is not None:
            if (
                not isinstance(self.training_data_receipt_sha256, str)
                or not _HEX64_RE.fullmatch(self.training_data_receipt_sha256)
                or self.production_split_seed != self.seed
                or not isinstance(self.heldout_window_manifest_sha256, str)
                or not _HEX64_RE.fullmatch(self.heldout_window_manifest_sha256)
            ):
                raise MatchedControlError(
                    "production matched-control receipt/split seed differs"
                )
        elif (
            self.production_split_seed is not None
            or self.heldout_window_manifest_sha256 is not None
        ):
            raise MatchedControlError(
                "production_split_seed requires training_data_receipt_sha256"
            )
        self.production_preflight_receipt_sha256 = config.get(
            "production_preflight_receipt_sha256"
        )
        self.run_contract = config.get("run_contract")
        self.run_contract_sha256 = config.get("run_contract_sha256")
        self.external_acceptance_sha256 = config.get(
            "external_acceptance_sha256"
        )
        production_contract_values = (
            self.production_preflight_receipt_sha256,
            self.run_contract,
            self.run_contract_sha256,
            self.external_acceptance_sha256,
        )
        if any(value is not None for value in production_contract_values):
            if any(value is None for value in production_contract_values):
                raise MatchedControlError(
                    "current-run contract fields must be supplied as one complete set"
                )
            if (
                not isinstance(self.production_preflight_receipt_sha256, str)
                or not _HEX64_RE.fullmatch(self.production_preflight_receipt_sha256)
                or not isinstance(self.run_contract, Mapping)
                or not self.run_contract
                or not isinstance(self.run_contract_sha256, str)
                or not _HEX64_RE.fullmatch(self.run_contract_sha256)
                or json_sha256(self.run_contract) != self.run_contract_sha256
                or not isinstance(self.external_acceptance_sha256, str)
                or not _HEX64_RE.fullmatch(self.external_acceptance_sha256)
            ):
                raise MatchedControlError("invalid current-run matched-control contract")
            self.run_contract = dict(self.run_contract)
        elif self.training_data_receipt_sha256 is not None:
            # Legacy production receipts remain readable for Ego v6, but the
            # contact-asym25 v2 gate never accepts their report.
            self.run_contract = None
        self.config_hash_scope = config.get("config_hash_scope")
        if not isinstance(self.config_hash_scope, Mapping) or set(
            self.config_hash_scope
        ) != {"control", "variant"}:
            raise MatchedControlError(
                "config_hash_scope must define exactly 'control' and 'variant'"
            )
        self.config_hash_scope = dict(self.config_hash_scope)

        expected_version = config.get("expected_collector_version")
        if expected_version is not None and expected_version != COLLECTOR_VERSION:
            raise MatchedControlError(
                "collector version differs from the production expectation"
            )
        expected_world_size = config.get("expected_world_size")
        if expected_world_size is not None and (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
        ):
            raise MatchedControlError("expected_world_size must be a native integer")
        if expected_world_size is not None and self.world_size != expected_world_size:
            raise MatchedControlError(
                "runtime world_size differs from the production expectation"
            )
        configured_runtime_world = config.get("runtime_world_size")
        configured_runtime_rank = config.get("runtime_rank")
        for name, value in (
            ("runtime_world_size", configured_runtime_world),
            ("runtime_rank", configured_runtime_rank),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise MatchedControlError(f"{name} must be a native integer")
        if (
            configured_runtime_world is not None
            and self.world_size != configured_runtime_world
        ) or (
            configured_runtime_rank is not None and self.rank != configured_runtime_rank
        ):
            raise MatchedControlError(
                "trainer rank/world differs from the config-import runtime topology"
            )
        expected_rank_set = config.get("expected_rank_set")
        if expected_rank_set is not None and (
            not isinstance(expected_rank_set, list)
            or any(
                isinstance(expected_rank, bool) or not isinstance(expected_rank, int)
                for expected_rank in expected_rank_set
            )
            or expected_rank_set != list(range(self.world_size))
            or self.rank not in expected_rank_set
        ):
            raise MatchedControlError(
                "runtime rank topology differs from the production expectation"
            )
        self.git_commit = _discover_git_commit(Path(__file__).resolve().parent)
        self.collector_code_sha256 = _sha256_file(Path(__file__).resolve())
        expected_git_commit = config.get("expected_git_commit")
        if expected_git_commit is not None and self.git_commit != expected_git_commit:
            raise MatchedControlError(
                "runtime Git commit differs from the production expectation"
            )
        expected_source_sha256 = config.get("expected_collector_source_sha256")
        if (
            expected_source_sha256 is not None
            and self.collector_code_sha256 != expected_source_sha256
        ):
            raise MatchedControlError(
                "runtime collector source SHA differs from the production expectation"
            )
        expected_run_id = config.get("expected_run_id")
        expected_run_nonce = config.get("expected_run_nonce")
        if expected_run_id is not None and self.run_id != expected_run_id:
            raise MatchedControlError("runtime run_id differs from the production expectation")
        if expected_run_nonce is not None and self.run_nonce != expected_run_nonce:
            raise MatchedControlError(
                "runtime run_nonce differs from the production expectation"
            )

        project_path = Path(project_dir).expanduser().resolve()
        output_dir_value = config.get("output_dir")
        if not output_dir_value:
            raise MatchedControlError("collector output_dir must be explicitly configured")
        output_dir = Path(str(output_dir_value)).expanduser().resolve()
        _ensure_project_output(output_dir, project_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{safe_run}__{safe_nonce}__{safe_variant}__seed{self.seed}"
            f"__rank{self.rank:05d}.jsonl"
        )
        self.output_path = output_dir / filename
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to mix matched-control runs in existing file: {self.output_path}"
            )
        self.output_path.touch(exist_ok=False)
        self._micro_steps: dict[int, int] = {}

    @staticmethod
    def _require_string_batch(batch_dict: Mapping[str, Any], key: str) -> list[str]:
        value = batch_dict.get(key)
        if not isinstance(value, (list, tuple)) or not value:
            raise MatchedControlError(f"batch is missing non-empty trace field {key!r}")
        if any(not isinstance(item, str) or not item for item in value):
            raise MatchedControlError(f"trace field {key!r} must be a list of strings")
        return list(value)

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
        identity_json = self._require_string_batch(batch_dict, SAMPLE_IDENTITY_KEY)
        raw_hashes = self._require_string_batch(batch_dict, SAMPLE_RAW_HASH_KEY)
        augmentation_json = self._require_string_batch(batch_dict, SAMPLE_AUGMENTATION_KEY)
        diagnostic_paths = self._require_string_batch(batch_dict, SAMPLE_DATA_PATH_KEY)
        batch_size = len(identity_json)
        if any(len(values) != batch_size for values in (raw_hashes, augmentation_json, diagnostic_paths)):
            raise MatchedControlError("matched-control trace fields have different batch lengths")
        if any(not _HEX64_RE.fullmatch(value) for value in raw_hashes):
            raise MatchedControlError("raw input trace contains a non-SHA256 value")
        try:
            batch_ids = [
                strict_json_loads(value.encode("utf-8"), name="sample identity")
                for value in identity_json
            ]
            augmentation = [
                strict_json_loads(value.encode("utf-8"), name="augmentation trace")
                for value in augmentation_json
            ]
        except ValueError as exc:
            raise MatchedControlError("invalid canonical JSON in sample trace") from exc
        for item in augmentation:
            if item.get("trace_version") != TRACE_VERSION:
                raise MatchedControlError("unexpected augmentation trace version")

        global_step = int(global_step)
        micro_step = self._micro_steps.get(global_step, 0)
        self._micro_steps[global_step] = micro_step + 1
        hashes = {
            "ordered_sample_ids": canonical_sha256(batch_ids),
            "raw_batch": canonical_sha256(raw_hashes),
            "augmentation_parameters": canonical_sha256(augmentation),
            "visual_action_timesteps": canonical_sha256(
                {"action": action_timestep, "visual": visual_timestep}
            ),
            "visual_noise": canonical_sha256(visual_noise),
            "action_noise": canonical_sha256(action_noise),
        }
        record = {
            "record_version": COLLECTOR_VERSION,
            "run_id": self.run_id,
            "run_nonce": self.run_nonce,
            "variant_id": self.variant_id,
            "seed": self.seed,
            "global_step": global_step,
            "micro_step": micro_step,
            "rank": self.rank,
            "world_size": self.world_size,
            "variant_config_sha256": self.variant_config_sha256,
            "control_config_sha256": self.control_config_sha256,
            "shared_model_config_sha256": self.shared_model_config_sha256,
            "transport_override": self.transport_override,
            "config_hash_scope": self.config_hash_scope,
            "git_commit": self.git_commit,
            "collector_code_sha256": self.collector_code_sha256,
            "training_data_receipt_sha256": self.training_data_receipt_sha256,
            "heldout_window_manifest_sha256": self.heldout_window_manifest_sha256,
            "production_split_seed": self.production_split_seed,
            "production_preflight_receipt_sha256": (
                self.production_preflight_receipt_sha256
            ),
            "run_contract": self.run_contract,
            "run_contract_sha256": self.run_contract_sha256,
            "external_acceptance_sha256": self.external_acceptance_sha256,
            "batch_ids": batch_ids,
            "diagnostic_data_paths": diagnostic_paths,
            "hashes": hashes,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def _trace_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    raise MatchedControlError(f"trace path does not exist: {path}")


def _read_variant_records(
    paths: Sequence[Path], expected_variant: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    file_digests: list[dict[str, str]] = []
    for path in paths:
        payload, trace_digest, _ = read_verified_bytes(path)
        file_digests.append(
            {"path": str(path), "sha256": trace_digest}
        )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MatchedControlError(f"trace is not UTF-8: {path}") from exc
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = strict_json_loads(
                    line.encode("utf-8"),
                    name=f"matched-control trace {path}:{line_number}",
                )
            except ValueError as exc:
                raise MatchedControlError(f"invalid JSONL {path}:{line_number}") from exc
            if record.get("record_version") != COLLECTOR_VERSION:
                raise MatchedControlError(f"unexpected record version in {path}:{line_number}")
            if record.get("variant_id") != expected_variant:
                raise MatchedControlError(
                    f"variant mismatch in {path}:{line_number}: {record.get('variant_id')!r}"
                )
            hashes = record.get("hashes")
            if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_HASHES):
                raise MatchedControlError(f"missing required hashes in {path}:{line_number}")
            if any(not isinstance(value, str) or not _HEX64_RE.fullmatch(value) for value in hashes.values()):
                raise MatchedControlError(f"invalid SHA256 hash in {path}:{line_number}")
            batch_ids = record.get("batch_ids")
            if not isinstance(batch_ids, list) or not batch_ids:
                raise MatchedControlError(f"missing ordered batch_ids in {path}:{line_number}")
            for identity in batch_ids:
                if not isinstance(identity, dict) or set(identity) != {
                    "dataset_id",
                    "episode_index",
                    "frame_index",
                    "global_index",
                }:
                    raise MatchedControlError(
                        f"invalid sample identity schema in {path}:{line_number}"
                    )
                if not isinstance(identity["dataset_id"], str) or not identity[
                    "dataset_id"
                ].strip():
                    raise MatchedControlError(
                        f"empty sample dataset_id in {path}:{line_number}"
                    )
                if any(
                    isinstance(identity[name], bool)
                    or not isinstance(identity[name], int)
                    for name in ("episode_index", "frame_index", "global_index")
                ):
                    raise MatchedControlError(
                        f"non-integer sample identity in {path}:{line_number}"
                    )
            if hashes["ordered_sample_ids"] != canonical_sha256(batch_ids):
                raise MatchedControlError(
                    f"ordered sample identity hash is inconsistent in {path}:{line_number}"
                )
            diagnostic_paths = record.get("diagnostic_data_paths")
            if (
                not isinstance(diagnostic_paths, list)
                or len(diagnostic_paths) != len(batch_ids)
                or any(not isinstance(value, str) for value in diagnostic_paths)
            ):
                raise MatchedControlError(
                    f"invalid diagnostic_data_paths in {path}:{line_number}"
                )
            for name in (
                "variant_config_sha256",
                "control_config_sha256",
                "collector_code_sha256",
                "shared_model_config_sha256",
            ):
                if not _HEX64_RE.fullmatch(str(record.get(name, ""))):
                    raise MatchedControlError(f"missing {name} in {path}:{line_number}")
            expected_transport = {
                "soft_semantic": "soft",
                "hard_st": "hard_st",
                "hard_detached": "hard_detached",
            }[expected_variant]
            if record.get("transport_override") != {
                "variant_id": expected_variant,
                "contact_semantic_transport_mode": expected_transport,
            }:
                raise MatchedControlError(
                    f"transport override differs in {path}:{line_number}"
                )
            if not isinstance(record.get("config_hash_scope"), dict) or set(
                record["config_hash_scope"]
            ) != {"control", "variant"}:
                raise MatchedControlError(f"missing config_hash_scope in {path}:{line_number}")
            if not isinstance(record.get("run_id"), str) or not record["run_id"].strip():
                raise MatchedControlError(f"missing run_id in {path}:{line_number}")
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}",
                str(record.get("run_nonce", "")),
            ):
                raise MatchedControlError(f"missing/invalid run_nonce in {path}:{line_number}")
            if not isinstance(record.get("git_commit"), str) or not record[
                "git_commit"
            ].strip():
                raise MatchedControlError(f"missing git_commit in {path}:{line_number}")
            training_receipt = record.get("training_data_receipt_sha256")
            window_receipt = record.get("heldout_window_manifest_sha256")
            production_seed = record.get("production_split_seed")
            if training_receipt is not None:
                if (
                    not isinstance(training_receipt, str)
                    or not _HEX64_RE.fullmatch(training_receipt)
                    or not isinstance(window_receipt, str)
                    or not _HEX64_RE.fullmatch(window_receipt)
                    or isinstance(production_seed, bool)
                    or production_seed not in ALLOWED_SEEDS
                    or production_seed != record.get("seed")
                ):
                    raise MatchedControlError(
                        f"invalid production receipt/window/seed in {path}:{line_number}"
                    )
            elif window_receipt is not None or production_seed is not None:
                raise MatchedControlError(
                    f"incomplete production receipt binding in {path}:{line_number}"
                )
            preflight_receipt = record.get("production_preflight_receipt_sha256")
            run_contract = record.get("run_contract")
            run_contract_sha256 = record.get("run_contract_sha256")
            external_acceptance = record.get("external_acceptance_sha256")
            current_run_values = (
                preflight_receipt,
                run_contract,
                run_contract_sha256,
                external_acceptance,
            )
            if any(value is not None for value in current_run_values):
                if (
                    any(value is None for value in current_run_values)
                    or not isinstance(preflight_receipt, str)
                    or not _HEX64_RE.fullmatch(preflight_receipt)
                    or not isinstance(run_contract, dict)
                    or not run_contract
                    or not isinstance(run_contract_sha256, str)
                    or not _HEX64_RE.fullmatch(run_contract_sha256)
                    or json_sha256(run_contract) != run_contract_sha256
                    or not isinstance(external_acceptance, str)
                    or not _HEX64_RE.fullmatch(external_acceptance)
                ):
                    raise MatchedControlError(
                        f"invalid current-run contract in {path}:{line_number}"
                    )
            for name in ("seed", "global_step", "micro_step", "rank", "world_size"):
                value = record.get(name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise MatchedControlError(f"invalid {name} in {path}:{line_number}")
            if record["seed"] not in ALLOWED_SEEDS:
                raise MatchedControlError(f"unsupported seed in {path}:{line_number}")
            if (
                record["global_step"] < 0
                or record["micro_step"] < 0
                or record["world_size"] <= 0
                or not 0 <= record["rank"] < record["world_size"]
            ):
                raise MatchedControlError(
                    f"invalid step/rank bounds in {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise MatchedControlError(f"no matched-control records for variant {expected_variant!r}")
    return records, file_digests


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["seed"]),
        int(record["global_step"]),
        int(record["micro_step"]),
        int(record["rank"]),
    )


def compare_variant_traces(
    variants: Mapping[str, Sequence[str | os.PathLike[str]]],
    *,
    output_report: str | os.PathLike[str],
    collector_code_path: str | os.PathLike[str] = __file__,
    required_seeds: Sequence[int] = ALLOWED_SEEDS,
    allow_distinct_variant_run_identities: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    """Compare rank-local traces and optionally write a gate-consumable report."""

    report_path = Path(output_report).expanduser().resolve()
    if report_path == Path("/tmp") or Path("/tmp") in report_path.parents:
        raise MatchedControlError("comparison reports must not be written under /tmp")
    collector_path = Path(collector_code_path).expanduser().resolve()
    collector_code = {
        "path": os.path.relpath(collector_path, report_path.parent),
        "version": COLLECTOR_VERSION,
        "sha256": _sha256_file(collector_path),
    }
    report: dict[str, Any] = {
        "report_version": COLLECTOR_VERSION,
        "status": "failed",
        "collector_implemented": True,
        "collector_code": collector_code,
        "dry_run_passed": False,
        "matched_batch_count": 0,
        "all_hashes_equal": False,
        "all_hashes_equal_across_variants": False,
        "captured_hashes": list(REQUIRED_HASHES),
        "hash_schema": HASH_SCHEMA,
        "seeds": [],
        "run_id": None,
        "run_nonce": None,
        "run_id_by_variant": {},
        "run_nonce_by_variant": {},
        "dataset_id": None,
        "dataset_ids": [],
        "variants": sorted(variants),
        "global_hashes": {},
        "trace_files": {},
        "variant_config_sha256": {},
        "shared_model_config_sha256": None,
        "transport_override_by_variant": {},
        "control_config_sha256": None,
        "config_hash_scope": None,
        "git_commit": None,
        "training_data_receipt_sha256": None,
        "heldout_window_manifest_sha256": None,
        "production_split_seed": None,
        "production_preflight_receipt_sha256": None,
        "run_contract": None,
        "run_contract_sha256": None,
        "external_acceptance_sha256_by_variant": {},
        "world_size": None,
        "rank_set": None,
        "topology": None,
        "trace_manifest_sha256": None,
        "global_hashes_sha256": None,
        "hash_schema_sha256": canonical_sha256(HASH_SCHEMA),
        "errors": [],
    }
    try:
        if set(variants) != set(IMPLEMENTED_RUNNER_VARIANTS):
            raise MatchedControlError(
                "comparison variants must equal the implemented runner set: "
                f"{list(IMPLEMENTED_RUNNER_VARIANTS)}"
            )
        indexed: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]] = {}
        trace_files: dict[str, list[dict[str, str]]] = {}
        observed_pairs: set[tuple[str, str, str]] = set()
        for variant, raw_paths in sorted(variants.items()):
            paths: list[Path] = []
            for raw_path in raw_paths:
                paths.extend(_trace_paths(Path(raw_path).expanduser().resolve()))
            deduplicated_paths = sorted(set(paths))
            if len(deduplicated_paths) != len(paths):
                raise MatchedControlError(f"duplicate trace file supplied for {variant}")
            paths = deduplicated_paths
            records, file_digests = _read_variant_records(paths, variant)
            trace_files[variant] = [
                {
                    "path": os.path.relpath(entry["path"], report_path.parent),
                    "sha256": entry["sha256"],
                }
                for entry in file_digests
            ]
            run_variant_pairs = {
                (
                    str(record.get("run_id", "")),
                    str(record.get("run_nonce", "")),
                    str(record.get("variant_id", "")),
                )
                for record in records
            }
            if len(run_variant_pairs) != 1:
                raise MatchedControlError(f"multiple run/variant identities for {variant}")
            pair = next(iter(run_variant_pairs))
            if pair in observed_pairs:
                raise MatchedControlError(f"duplicate run/variant identity: {pair}")
            observed_pairs.add(pair)
            by_key: dict[tuple[int, int, int, int], dict[str, Any]] = {}
            for record in records:
                key = _record_key(record)
                if key in by_key:
                    raise MatchedControlError(f"duplicate record key {key} for {variant}")
                by_key[key] = record
            indexed[variant] = by_key

        run_id_by_variant = {
            variant: run_id for run_id, _, variant in observed_pairs
        }
        run_nonce_by_variant = {
            variant: run_nonce for _, run_nonce, variant in observed_pairs
        }
        run_ids = set(run_id_by_variant.values())
        run_nonces = set(run_nonce_by_variant.values())
        if allow_distinct_variant_run_identities:
            if (
                len(run_ids) != len(indexed)
                or len(run_nonces) != len(indexed)
                or any(
                    not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}", value
                    )
                    for value in run_ids | run_nonces
                )
            ):
                raise MatchedControlError(
                    "current-run matched variants require distinct 16-128 character "
                    "run IDs and nonces"
                )
        elif len(run_ids) != 1 or len(run_nonces) != 1:
            raise MatchedControlError("run identity differs across variants")
        baseline_name = sorted(indexed)[0]
        baseline = indexed[baseline_name]
        baseline_keys = set(baseline)
        for variant, records in indexed.items():
            if set(records) != baseline_keys:
                raise MatchedControlError(f"step/seed/rank set differs for {variant}")
        all_records = [record for records in indexed.values() for record in records.values()]
        dataset_ids = {
            identity["dataset_id"]
            for record in all_records
            for identity in record["batch_ids"]
        }
        if len(dataset_ids) != 1:
            raise MatchedControlError(
                f"matched-control report requires one dataset_id, got {sorted(dataset_ids)}"
            )
        control_hashes = {record["control_config_sha256"] for record in all_records}
        shared_model_hashes = {
            record["shared_model_config_sha256"] for record in all_records
        }
        hash_scopes = {
            json.dumps(record["config_hash_scope"], sort_keys=True, separators=(",", ":"))
            for record in all_records
        }
        git_commits = {record.get("git_commit") for record in all_records}
        code_hashes = {record["collector_code_sha256"] for record in all_records}
        versions = {record["record_version"] for record in all_records}
        training_receipt_hashes = {
            record.get("training_data_receipt_sha256") for record in all_records
        }
        production_split_seeds = {
            record.get("production_split_seed") for record in all_records
        }
        heldout_window_hashes = {
            record.get("heldout_window_manifest_sha256") for record in all_records
        }
        preflight_receipt_hashes = {
            record.get("production_preflight_receipt_sha256") for record in all_records
        }
        run_contracts = {
            canonical_json_dumps(record.get("run_contract")) for record in all_records
        }
        run_contract_hashes = {
            record.get("run_contract_sha256") for record in all_records
        }
        if len(control_hashes) != 1:
            raise MatchedControlError("control_config_sha256 differs across variants")
        if len(shared_model_hashes) != 1:
            raise MatchedControlError(
                "matched variants differ outside the reviewed transport override"
            )
        if len(hash_scopes) != 1:
            raise MatchedControlError("config_hash_scope differs across variants")
        if len(git_commits) != 1:
            raise MatchedControlError("git_commit differs across variants")
        if len(code_hashes) != 1 or next(iter(code_hashes)) != collector_code["sha256"]:
            raise MatchedControlError("collector code SHA differs across traces or current code")
        if versions != {COLLECTOR_VERSION}:
            raise MatchedControlError("collector version differs across variants")
        if (
            len(training_receipt_hashes) != 1
            or len(production_split_seeds) != 1
            or len(heldout_window_hashes) != 1
        ):
            raise MatchedControlError(
                "training-data receipt, split seed, or held-out windows differ across variants"
            )
        if (
            len(preflight_receipt_hashes) != 1
            or len(run_contracts) != 1
            or len(run_contract_hashes) != 1
        ):
            raise MatchedControlError(
                "preflight or stable current-run contract differs across variants"
            )
        matched_training_receipt_sha256 = next(iter(training_receipt_hashes))
        matched_production_split_seed = next(iter(production_split_seeds))
        matched_heldout_window_sha256 = next(iter(heldout_window_hashes))
        matched_preflight_receipt_sha256 = next(iter(preflight_receipt_hashes))
        matched_run_contract = strict_json_loads(
            next(iter(run_contracts)).encode("utf-8"), name="matched run contract"
        )
        matched_run_contract_sha256 = next(iter(run_contract_hashes))
        if (matched_training_receipt_sha256 is None) != (
            matched_production_split_seed is None
        ) or (matched_training_receipt_sha256 is None) != (
            matched_heldout_window_sha256 is None
        ):
            raise MatchedControlError(
                "incomplete production matched-control training-data binding"
            )
        current_run_fields = (
            matched_preflight_receipt_sha256,
            matched_run_contract,
            matched_run_contract_sha256,
        )
        if any(value is not None for value in current_run_fields) and (
            any(value is None for value in current_run_fields)
            or not isinstance(matched_run_contract, dict)
            or json_sha256(matched_run_contract) != matched_run_contract_sha256
        ):
            raise MatchedControlError("incomplete current-run contract binding")
        if allow_distinct_variant_run_identities:
            training_config_receipt = matched_run_contract.get(
                "training_config_receipt"
            )
            stable_training_config = (
                dict(training_config_receipt)
                if isinstance(training_config_receipt, dict)
                else None
            )
            if stable_training_config is not None:
                stable_training_config["seed"] = (
                    "__CONTACT_ASYM25_SPLIT_SEED__"
                )
            if (
                matched_run_contract.get("preflight_receipt_sha256")
                != matched_preflight_receipt_sha256
                or matched_run_contract.get("control_config_sha256")
                != next(iter(control_hashes))
                or not isinstance(training_config_receipt, dict)
                or isinstance(training_config_receipt.get("seed"), bool)
                or training_config_receipt.get("seed")
                != matched_production_split_seed
                or json_sha256(training_config_receipt)
                != matched_run_contract.get("training_config_sha256")
                or json_sha256(stable_training_config)
                != matched_run_contract.get("stable_training_config_sha256")
            ):
                raise MatchedControlError(
                    "current-run preflight/control/training-config contract differs"
                )
        world_sizes = {int(record["world_size"]) for record in all_records}
        if len(world_sizes) != 1:
            raise MatchedControlError("world_size changes within the matched-control report")
        report_world_size = next(iter(world_sizes))
        report_rank_set = sorted({int(record["rank"]) for record in all_records})
        if report_rank_set != list(range(report_world_size)):
            raise MatchedControlError(
                "report rank set is incomplete for its world_size"
            )
        variant_hashes: dict[str, str] = {}
        transport_overrides: dict[str, dict[str, str]] = {}
        external_acceptance_hashes: dict[str, str | None] = {}
        for variant, records in indexed.items():
            values = {record["variant_config_sha256"] for record in records.values()}
            if len(values) != 1:
                raise MatchedControlError(f"variant config hash changes within {variant}")
            variant_hashes[variant] = next(iter(values))
            override_values = {
                canonical_json_dumps(record["transport_override"])
                for record in records.values()
            }
            if len(override_values) != 1:
                raise MatchedControlError(
                    f"transport override changes within {variant}"
                )
            transport_overrides[variant] = strict_json_loads(
                next(iter(override_values)).encode("utf-8"),
                name=f"transport override {variant}",
            )
            acceptance_values = {
                record.get("external_acceptance_sha256")
                for record in records.values()
            }
            if len(acceptance_values) != 1:
                raise MatchedControlError(
                    f"external acceptance changes within {variant}"
                )
            external_acceptance_hashes[variant] = next(iter(acceptance_values))
        if allow_distinct_variant_run_identities and (
            any(
                not isinstance(value, str) or not _HEX64_RE.fullmatch(value)
                for value in external_acceptance_hashes.values()
            )
            or len(set(external_acceptance_hashes.values())) != len(indexed)
        ):
            raise MatchedControlError(
                "current-run matched variants require distinct external acceptances"
            )
        observed_seeds = sorted({key[0] for key in baseline_keys})
        if observed_seeds != sorted(int(seed) for seed in required_seeds):
            raise MatchedControlError(
                f"seed set {observed_seeds} differs from required {sorted(required_seeds)}"
            )
        if any(seed not in ALLOWED_SEEDS for seed in observed_seeds):
            raise MatchedControlError(f"unsupported seed in trace: {observed_seeds}")

        logical_keys = sorted({key[:3] for key in baseline_keys})
        global_hashes: dict[str, dict[str, dict[str, str]]] = {}
        for logical_key in logical_keys:
            seed, global_step, micro_step = logical_key
            logical_name = f"seed{seed}/step{global_step}/micro{micro_step}"
            global_hashes[logical_name] = {}
            baseline_world = None
            for variant, records in sorted(indexed.items()):
                rank_records = [record for key, record in records.items() if key[:3] == logical_key]
                worlds = {int(record["world_size"]) for record in rank_records}
                if len(worlds) != 1:
                    raise MatchedControlError(f"inconsistent world_size for {variant} {logical_name}")
                world_size = worlds.pop()
                ranks = sorted(int(record["rank"]) for record in rank_records)
                if ranks != list(range(world_size)):
                    raise MatchedControlError(
                        f"rank set {ranks} is incomplete for world_size={world_size}"
                    )
                if baseline_world is None:
                    baseline_world = world_size
                elif world_size != baseline_world:
                    raise MatchedControlError(f"world_size differs across variants at {logical_name}")
                rank_records.sort(key=lambda record: int(record["rank"]))
                global_hashes[logical_name][variant] = {
                    field: canonical_sha256(
                        [
                            {"rank": int(record["rank"]), "sha256": record["hashes"][field]}
                            for record in rank_records
                        ]
                    )
                    for field in REQUIRED_HASHES
                }

            base_records = baseline
            for key in sorted(key for key in baseline_keys if key[:3] == logical_key):
                base_record = base_records[key]
                for variant, records in indexed.items():
                    record = records[key]
                    if record.get("batch_ids") != base_record.get("batch_ids"):
                        raise MatchedControlError(f"batch identity differs for {variant} at {key}")
                    for field in REQUIRED_HASHES:
                        if record["hashes"][field] != base_record["hashes"][field]:
                            raise MatchedControlError(f"{field} differs for {variant} at {key}")
            reference_globals = global_hashes[logical_name][baseline_name]
            if any(values != reference_globals for values in global_hashes[logical_name].values()):
                raise MatchedControlError(f"global rank combination differs at {logical_name}")

        report.update(
            status="passed",
            dry_run_passed=True,
            matched_batch_count=len(logical_keys),
            all_hashes_equal=True,
            all_hashes_equal_across_variants=True,
            seeds=observed_seeds,
            run_id=(None if allow_distinct_variant_run_identities else next(iter(run_ids))),
            run_nonce=(
                None if allow_distinct_variant_run_identities else next(iter(run_nonces))
            ),
            run_id_by_variant=run_id_by_variant,
            run_nonce_by_variant=run_nonce_by_variant,
            dataset_id=next(iter(dataset_ids)),
            dataset_ids=sorted(dataset_ids),
            global_hashes=global_hashes,
            trace_files=trace_files,
            variant_config_sha256=variant_hashes,
            shared_model_config_sha256=next(iter(shared_model_hashes)),
            transport_override_by_variant=transport_overrides,
            control_config_sha256=next(iter(control_hashes)),
            config_hash_scope=strict_json_loads(
                next(iter(hash_scopes)).encode("utf-8"),
                name="matched-control config hash scope",
            ),
            git_commit=next(iter(git_commits)),
            training_data_receipt_sha256=matched_training_receipt_sha256,
            heldout_window_manifest_sha256=matched_heldout_window_sha256,
            production_split_seed=matched_production_split_seed,
            production_preflight_receipt_sha256=(
                matched_preflight_receipt_sha256
            ),
            run_contract=matched_run_contract,
            run_contract_sha256=matched_run_contract_sha256,
            external_acceptance_sha256_by_variant=external_acceptance_hashes,
            world_size=report_world_size,
            rank_set=report_rank_set,
            topology={
                "world_size": report_world_size,
                "rank_set": report_rank_set,
            },
            trace_manifest_sha256=canonical_sha256(trace_files),
            global_hashes_sha256=canonical_sha256(global_hashes),
            hash_schema_sha256=canonical_sha256(HASH_SCHEMA),
            errors=[],
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report["errors"] = [str(exc)]

    if write_report:
        _write_json_exclusive_durable(report_path, report)
    return report

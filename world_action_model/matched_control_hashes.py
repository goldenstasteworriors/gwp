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


COLLECTOR_VERSION = "matched-control-hash-collector/v1"
TRACE_VERSION = "wam-matched-control-sample-trace/v1"
ALLOWED_SEEDS = (6666, 7777, 8888)
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
        if not bool(config.get("enabled", False)):
            raise MatchedControlError("do not construct a disabled collector")
        self.seed = int(config.get("seed", trainer_seed))
        if self.seed != int(trainer_seed):
            raise MatchedControlError(
                f"collector seed {self.seed} differs from trainer seed {trainer_seed}"
            )
        if self.seed not in ALLOWED_SEEDS:
            raise MatchedControlError(
                f"collector seed must be one of {ALLOWED_SEEDS}, got {self.seed}"
            )
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise MatchedControlError(
                f"invalid distributed identity rank={self.rank}, world_size={self.world_size}"
            )
        self.run_id = str(config.get("run_id", "")).strip()
        self.variant_id = str(config.get("variant_id", "")).strip()
        safe_run = _safe_name(self.run_id)
        safe_variant = _safe_name(self.variant_id)
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
        self.config_hash_scope = config.get("config_hash_scope")
        if not isinstance(self.config_hash_scope, Mapping) or set(
            self.config_hash_scope
        ) != {"control", "variant"}:
            raise MatchedControlError(
                "config_hash_scope must define exactly 'control' and 'variant'"
            )
        self.config_hash_scope = dict(self.config_hash_scope)

        project_path = Path(project_dir).expanduser().resolve()
        output_dir_value = config.get("output_dir")
        if not output_dir_value:
            raise MatchedControlError("collector output_dir must be explicitly configured")
        output_dir = Path(str(output_dir_value)).expanduser().resolve()
        _ensure_project_output(output_dir, project_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{safe_run}__{safe_variant}__seed{self.seed}__rank{self.rank:05d}.jsonl"
        )
        self.output_path = output_dir / filename
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to mix matched-control runs in existing file: {self.output_path}"
            )
        self.output_path.touch(exist_ok=False)
        self.git_commit = str(config.get("git_commit") or _discover_git_commit(Path(__file__).resolve().parent))
        self.collector_code_sha256 = _sha256_file(Path(__file__).resolve())
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
            batch_ids = [json.loads(value) for value in identity_json]
            augmentation = [json.loads(value) for value in augmentation_json]
        except json.JSONDecodeError as exc:
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
            "variant_id": self.variant_id,
            "seed": self.seed,
            "global_step": global_step,
            "micro_step": micro_step,
            "rank": self.rank,
            "world_size": self.world_size,
            "variant_config_sha256": self.variant_config_sha256,
            "control_config_sha256": self.control_config_sha256,
            "config_hash_scope": self.config_hash_scope,
            "git_commit": self.git_commit,
            "collector_code_sha256": self.collector_code_sha256,
            "batch_ids": batch_ids,
            "diagnostic_data_paths": diagnostic_paths,
            "hashes": hashes,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
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
        payload = path.read_bytes()
        file_digests.append(
            {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
        )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MatchedControlError(f"trace is not UTF-8: {path}") from exc
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
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
            ):
                if not _HEX64_RE.fullmatch(str(record.get(name, ""))):
                    raise MatchedControlError(f"missing {name} in {path}:{line_number}")
            if not isinstance(record.get("config_hash_scope"), dict) or set(
                record["config_hash_scope"]
            ) != {"control", "variant"}:
                raise MatchedControlError(f"missing config_hash_scope in {path}:{line_number}")
            if not isinstance(record.get("run_id"), str) or not record["run_id"].strip():
                raise MatchedControlError(f"missing run_id in {path}:{line_number}")
            if not isinstance(record.get("git_commit"), str) or not record[
                "git_commit"
            ].strip():
                raise MatchedControlError(f"missing git_commit in {path}:{line_number}")
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
) -> dict[str, Any]:
    """Compare rank-local traces and always write a gate-consumable report."""

    report_path = Path(output_report).expanduser().resolve()
    if report_path == Path("/tmp") or Path("/tmp") in report_path.parents:
        raise MatchedControlError("comparison reports must not be written under /tmp")
    report_path.parent.mkdir(parents=True, exist_ok=True)
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
        "dataset_id": None,
        "dataset_ids": [],
        "variants": sorted(variants),
        "global_hashes": {},
        "trace_files": {},
        "variant_config_sha256": {},
        "control_config_sha256": None,
        "config_hash_scope": None,
        "git_commit": None,
        "errors": [],
    }
    try:
        if len(variants) < 2:
            raise MatchedControlError("comparison requires at least two variants")
        indexed: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]] = {}
        trace_files: dict[str, list[dict[str, str]]] = {}
        observed_pairs: set[tuple[str, str]] = set()
        for variant, raw_paths in sorted(variants.items()):
            paths: list[Path] = []
            for raw_path in raw_paths:
                paths.extend(_trace_paths(Path(raw_path).expanduser().resolve()))
            deduplicated_paths = sorted(set(paths))
            if len(deduplicated_paths) != len(paths):
                raise MatchedControlError(f"duplicate trace file supplied for {variant}")
            paths = deduplicated_paths
            records, trace_files[variant] = _read_variant_records(paths, variant)
            run_variant_pairs = {
                (str(record.get("run_id", "")), str(record.get("variant_id", "")))
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

        run_ids = {run_id for run_id, _ in observed_pairs}
        if len(run_ids) != 1:
            raise MatchedControlError("run_id differs across variants")
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
        hash_scopes = {
            json.dumps(record["config_hash_scope"], sort_keys=True, separators=(",", ":"))
            for record in all_records
        }
        git_commits = {record.get("git_commit") for record in all_records}
        code_hashes = {record["collector_code_sha256"] for record in all_records}
        versions = {record["record_version"] for record in all_records}
        if len(control_hashes) != 1:
            raise MatchedControlError("control_config_sha256 differs across variants")
        if len(hash_scopes) != 1:
            raise MatchedControlError("config_hash_scope differs across variants")
        if len(git_commits) != 1:
            raise MatchedControlError("git_commit differs across variants")
        if len(code_hashes) != 1 or next(iter(code_hashes)) != collector_code["sha256"]:
            raise MatchedControlError("collector code SHA differs across traces or current code")
        if versions != {COLLECTOR_VERSION}:
            raise MatchedControlError("collector version differs across variants")
        variant_hashes: dict[str, str] = {}
        for variant, records in indexed.items():
            values = {record["variant_config_sha256"] for record in records.values()}
            if len(values) != 1:
                raise MatchedControlError(f"variant config hash changes within {variant}")
            variant_hashes[variant] = next(iter(values))
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
            run_id=next(iter(run_ids)),
            dataset_id=next(iter(dataset_ids)),
            dataset_ids=sorted(dataset_ids),
            global_hashes=global_hashes,
            trace_files=trace_files,
            variant_config_sha256=variant_hashes,
            control_config_sha256=next(iter(control_hashes)),
            config_hash_scope=json.loads(next(iter(hash_scopes))),
            git_commit=next(iter(git_commits)),
            errors=[],
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report["errors"] = [str(exc)]

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

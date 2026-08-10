"""Strict provenance and validation for Ego Stage1 UMT5 sources.

The ordinary WAM loader accepts episode-level ``.pt`` tensors, but those bytes
alone cannot establish which text, tokenizer, model revision, or encoder code
created them.  This module defines the fail-closed production manifest used by
Ego Stage1.  It deliberately does not contain an "attest old files" shortcut:
the manifest must be emitted by the pinned-model generator in
``scripts/prepare_ego_t5_source.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from world_action_model.verified_io import (
    hash_verified_regular_file,
    read_verified_bytes,
    strict_json_loads,
)


EGO_T5_MANIFEST_FILENAME = "t5_source_manifest.json"
EGO_T5_MANIFEST_VERSION = "ego-t5-source-manifest/v1"
EGO_T5_GENERATOR_IDENTITY = "scripts/prepare_ego_t5_source.py"
EGO_T5_PROMPT_SELECTION = "first_non_quality_task/v1"
EGO_T5_PROMPT_CLEANING = "wan_huggingface_tokenizer/whitespace/v1"
EGO_T5_ENCODER_BACKEND = "wan.modules.t5.T5EncoderModel"
EGO_T5_ENCODER_ARCHITECTURE = "umt5-xxl-encoder"
EGO_T5_OUTPUT_DTYPE = "float32"
EGO_T5_OUTPUT_WIDTH = 4096
EGO_T5_LOADER_MAX_PROMPT_LEN = 64
EGO_T5_TEXT_LEN = 512
EGO_T5_MODEL_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
EGO_T5_CHECKPOINT_LOGICAL_PATH = "models_t5_umt5-xxl-enc-bf16.pth"
EGO_T5_TOKENIZER_FILES = (
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
EGO_T5_ENCODER_CODE_FILES = (
    "wan/modules/t5.py",
    "wan/modules/tokenizers.py",
)
QUALITY_TASK_LABELS = frozenset({"qualified", "unqualified", "success", "failure"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_EPISODE_FILE_RE = re.compile(r"^episode_([0-9]{6})\.pt$")


class EgoT5SourceError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _absolute_no_resolve(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _require_exact_keys(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EgoT5SourceError(f"invalid {name} schema: {actual}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EgoT5SourceError(f"invalid {name} SHA-256")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EgoT5SourceError(f"invalid {name}")
    return value


def load_episode_prompts(
    data_root: str | os.PathLike[str],
    *,
    expected_episode_lengths: Mapping[int, int],
) -> list[dict]:
    """Read the exact v2.1 prompt source used by the audited Ego dataset."""

    root = _absolute_no_resolve(data_root)
    path = root / "meta" / "episodes.jsonl"
    try:
        payload, _, _ = read_verified_bytes(path)
    except (OSError, ValueError) as exc:
        raise EgoT5SourceError(
            "production Ego T5 generation requires meta/episodes.jsonl"
        ) from exc
    rows = []
    seen = set()
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = strict_json_loads(line, name=f"{path}:{line_number}")
        except ValueError as exc:
            raise EgoT5SourceError("invalid Ego episode prompt metadata") from exc
        if not isinstance(row, dict):
            raise EgoT5SourceError("Ego episode prompt row must be an object")
        episode_index = row.get("episode_index")
        length = row.get("length")
        tasks = row.get("tasks")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or episode_index in seen
            or isinstance(length, bool)
            or not isinstance(length, int)
            or expected_episode_lengths.get(episode_index) != length
            or not isinstance(tasks, list)
            or not all(isinstance(task, str) for task in tasks)
        ):
            raise EgoT5SourceError("episode prompt row differs from dataset identity")
        candidates = [
            task.strip()
            for task in tasks
            if task.strip() and task.strip().lower() not in QUALITY_TASK_LABELS
        ]
        if not candidates:
            raise EgoT5SourceError(f"episode {episode_index} has no usable task prompt")
        prompt = candidates[0]
        rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256(prompt),
            }
        )
        seen.add(episode_index)
    rows.sort(key=lambda item: item["episode_index"])
    if [row["episode_index"] for row in rows] != sorted(expected_episode_lengths):
        raise EgoT5SourceError("episode prompt inventory differs from dataset identity")
    return rows


def tensor_semantic_receipt(payload: bytes, *, name: str) -> dict:
    """Safely load one tensor and hash its logical dtype/shape/value bytes."""

    try:
        import torch

        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise EgoT5SourceError(f"cannot safely load T5 tensor: {name}") from exc
    if not isinstance(value, torch.Tensor):
        raise EgoT5SourceError(f"T5 artifact must contain a tensor directly: {name}")
    value = value.detach().cpu().contiguous()
    if (
        value.ndim != 2
        or not 1 <= int(value.shape[0]) <= EGO_T5_LOADER_MAX_PROMPT_LEN
        or int(value.shape[1]) != EGO_T5_OUTPUT_WIDTH
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise EgoT5SourceError(
            f"T5 tensor must be finite float32 [1..64,4096], got "
            f"{value.dtype} {tuple(value.shape)}: {name}"
        )
    shape = [int(dim) for dim in value.shape]
    raw = value.numpy().tobytes(order="C")
    header = json.dumps(
        {"dtype": EGO_T5_OUTPUT_DTYPE, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "dtype": EGO_T5_OUTPUT_DTYPE,
        "shape": shape,
        "semantic_sha256": hashlib.sha256(header + b"\0" + raw).hexdigest(),
    }


def _validate_source_file_manifest(
    rows: object,
    *,
    expected_paths: tuple[str, ...],
    name: str,
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != len(expected_paths):
        raise EgoT5SourceError(f"invalid {name} file manifest")
    out = []
    for row, expected_path in zip(rows, expected_paths):
        row = _require_exact_keys(
            row,
            {"logical_path", "sha256", "size_bytes"},
            f"{name} file",
        )
        if row["logical_path"] != expected_path:
            raise EgoT5SourceError(f"unexpected {name} logical path")
        _require_sha256(row["sha256"], f"{name} file")
        _require_positive_int(row["size_bytes"], f"{name} size")
        out.append(dict(row))
    return out


def validate_ego_t5_source_manifest(
    t5_embedding_dir: str | os.PathLike[str],
    *,
    dataset_identity: Mapping[str, Any],
    data_root: str | os.PathLike[str],
    inspect_tensors: bool = True,
) -> dict:
    """Validate provenance plus every config-selected Stage1 tensor."""

    root = _absolute_no_resolve(t5_embedding_dir)
    if root.is_symlink() or not root.is_dir():
        raise EgoT5SourceError(f"T5 source must be a real directory: {root}")
    expected_entries = {EGO_T5_MANIFEST_FILENAME} | {
        f"episode_{int(index):06d}.pt"
        for index in dataset_identity.get("episode_lengths", {})
    }
    actual_entries = {entry.name for entry in root.iterdir()}
    if actual_entries != expected_entries:
        raise EgoT5SourceError(
            "T5 source directory must contain only the provenance manifest and "
            "the exact runtime-derived episode files"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in root.iterdir()):
        raise EgoT5SourceError("T5 source entries must be non-symlink regular files")
    manifest_path = root / EGO_T5_MANIFEST_FILENAME
    try:
        manifest_bytes, manifest_sha256, manifest_stat = read_verified_bytes(manifest_path)
        manifest = strict_json_loads(manifest_bytes, name=str(manifest_path))
    except (OSError, ValueError) as exc:
        raise EgoT5SourceError(
            f"missing/invalid pinned-model T5 provenance manifest: {manifest_path}"
        ) from exc
    manifest = _require_exact_keys(
        manifest,
        {
            "manifest_version",
            "status",
            "dataset",
            "encoder",
            "generator",
            "episodes",
            "episode_count",
            "unique_prompt_count",
            "prompt_manifest_sha256",
        },
        "T5 source manifest",
    )
    if (
        manifest["manifest_version"] != EGO_T5_MANIFEST_VERSION
        or manifest["status"] != "generated_from_pinned_model"
    ):
        raise EgoT5SourceError("T5 source is not pinned-model generated")

    dataset = _require_exact_keys(
        manifest["dataset"],
        {
            "dataset_id",
            "fingerprint_version",
            "episode_metadata_sha256",
            "prompt_source_logical_path",
            "prompt_selection",
            "prompt_cleaning",
        },
        "T5 dataset provenance",
    )
    if (
        dataset["dataset_id"] != dataset_identity.get("dataset_id")
        or dataset["fingerprint_version"] != dataset_identity.get("fingerprint_version")
        or dataset["episode_metadata_sha256"]
        != dataset_identity.get("episode_metadata_sha256")
        or dataset["prompt_source_logical_path"] != "meta/episodes.jsonl"
        or dataset["prompt_selection"] != EGO_T5_PROMPT_SELECTION
        or dataset["prompt_cleaning"] != EGO_T5_PROMPT_CLEANING
    ):
        raise EgoT5SourceError("T5 manifest dataset/prompt provenance differs")

    encoder = _require_exact_keys(
        manifest["encoder"],
        {
            "backend",
            "architecture",
            "model_repo_id",
            "model_revision",
            "tokenizer_repo_id",
            "tokenizer_revision",
            "tokenizer_subdir",
            "checkpoint",
            "tokenizer_manifest",
            "tokenizer_manifest_sha256",
            "encoder_code_manifest",
            "encoder_code_manifest_sha256",
            "text_len",
            "loader_max_prompt_len",
            "add_special_tokens",
            "clean",
            "compute_dtype",
            "output_dtype",
            "output_width",
        },
        "T5 encoder provenance",
    )
    if (
        encoder["backend"] != EGO_T5_ENCODER_BACKEND
        or encoder["architecture"] != EGO_T5_ENCODER_ARCHITECTURE
        or encoder["model_repo_id"] != EGO_T5_MODEL_REPO_ID
        or not isinstance(encoder["model_revision"], str)
        or not _GIT_COMMIT_RE.fullmatch(encoder["model_revision"])
        or encoder["tokenizer_repo_id"] != EGO_T5_MODEL_REPO_ID
        or encoder["tokenizer_revision"] != encoder["model_revision"]
        or encoder["tokenizer_subdir"] != "google/umt5-xxl"
        or encoder["text_len"] != EGO_T5_TEXT_LEN
        or encoder["loader_max_prompt_len"] != EGO_T5_LOADER_MAX_PROMPT_LEN
        or encoder["add_special_tokens"] is not True
        or encoder["clean"] != "whitespace"
        or encoder["compute_dtype"] not in {"float32", "bfloat16"}
        or encoder["output_dtype"] != EGO_T5_OUTPUT_DTYPE
        or encoder["output_width"] != EGO_T5_OUTPUT_WIDTH
    ):
        raise EgoT5SourceError("unsupported or incomplete T5 encoder provenance")
    checkpoint = _require_exact_keys(
        encoder["checkpoint"],
        {"logical_path", "sha256", "size_bytes", "hf_etag"},
        "T5 checkpoint provenance",
    )
    if checkpoint["logical_path"] != EGO_T5_CHECKPOINT_LOGICAL_PATH:
        raise EgoT5SourceError("unexpected T5 checkpoint logical path")
    _require_sha256(checkpoint["sha256"], "T5 checkpoint")
    _require_positive_int(checkpoint["size_bytes"], "T5 checkpoint size")
    _require_sha256(checkpoint["hf_etag"], "T5 checkpoint HF etag")
    tokenizer_manifest = _validate_source_file_manifest(
        encoder["tokenizer_manifest"],
        expected_paths=EGO_T5_TOKENIZER_FILES,
        name="tokenizer",
    )
    code_manifest = _validate_source_file_manifest(
        encoder["encoder_code_manifest"],
        expected_paths=EGO_T5_ENCODER_CODE_FILES,
        name="encoder code",
    )
    if (
        encoder["tokenizer_manifest_sha256"] != canonical_sha256(tokenizer_manifest)
        or encoder["encoder_code_manifest_sha256"] != canonical_sha256(code_manifest)
    ):
        raise EgoT5SourceError("T5 tokenizer/encoder code manifest SHA differs")

    generator = _require_exact_keys(
        manifest["generator"],
        {
            "identity",
            "git_commit",
            "source_sha256",
            "device_type",
            "runtime_versions",
            "one_encode_per_unique_prompt",
            "episode_file_pattern",
        },
        "T5 generator provenance",
    )
    runtime_versions = generator["runtime_versions"]
    if (
        generator["identity"] != EGO_T5_GENERATOR_IDENTITY
        or not isinstance(generator["git_commit"], str)
        or not _GIT_COMMIT_RE.fullmatch(generator["git_commit"])
        or not isinstance(generator["source_sha256"], str)
        or not _SHA256_RE.fullmatch(generator["source_sha256"])
        or generator["device_type"] not in {"cpu", "cuda"}
        or not isinstance(runtime_versions, dict)
        or set(runtime_versions)
        != {"python", "torch", "transformers", "ftfy", "regex"}
        or any(
            not isinstance(value, str) or not value
            for value in runtime_versions.values()
        )
        or generator["one_encode_per_unique_prompt"] is not True
        or generator["episode_file_pattern"] != "episode_{episode_index:06d}.pt"
    ):
        raise EgoT5SourceError("invalid T5 generator provenance")
    if (
        (generator["device_type"] == "cpu" and encoder["compute_dtype"] != "float32")
        or (generator["device_type"] == "cuda" and encoder["compute_dtype"] != "bfloat16")
    ):
        raise EgoT5SourceError("T5 device/compute dtype provenance is inconsistent")

    expected_prompts = load_episode_prompts(
        data_root,
        expected_episode_lengths=dataset_identity["episode_lengths"],
    )
    expected_by_episode = {row["episode_index"]: row for row in expected_prompts}
    episodes = manifest["episodes"]
    if not isinstance(episodes, list) or len(episodes) != len(expected_prompts):
        raise EgoT5SourceError("T5 manifest episode count differs")
    runtime_receipt = []
    portable_files = []
    prompt_to_tensor = {}
    tensor_to_prompt = {}
    normalized_rows = []
    for row in episodes:
        row = _require_exact_keys(
            row,
            {
                "episode_index",
                "prompt",
                "prompt_sha256",
                "file",
                "tensor",
            },
            "T5 episode",
        )
        episode_index = row["episode_index"]
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index not in expected_by_episode
        ):
            raise EgoT5SourceError("invalid T5 episode index")
        expected_prompt = expected_by_episode.pop(episode_index)
        if (
            row["prompt"] != expected_prompt["prompt"]
            or row["prompt_sha256"] != expected_prompt["prompt_sha256"]
        ):
            raise EgoT5SourceError("T5 episode prompt differs from dataset metadata")
        file_row = _require_exact_keys(
            row["file"],
            {"logical_path", "sha256", "size_bytes"},
            "T5 episode file",
        )
        filename = f"episode_{episode_index:06d}.pt"
        if file_row["logical_path"] != filename or not _EPISODE_FILE_RE.fullmatch(
            filename
        ):
            raise EgoT5SourceError("T5 episode file path differs")
        _require_sha256(file_row["sha256"], "T5 episode file")
        _require_positive_int(file_row["size_bytes"], "T5 episode file size")
        path = root / filename
        try:
            actual_sha256, runtime_stat = hash_verified_regular_file(
                path, expected_sha256=file_row["sha256"]
            )
        except (OSError, ValueError) as exc:
            raise EgoT5SourceError(f"invalid T5 episode file: {path}") from exc
        if runtime_stat["size_bytes"] != file_row["size_bytes"]:
            raise EgoT5SourceError("T5 episode file size differs")
        tensor_row = _require_exact_keys(
            row["tensor"],
            {"dtype", "shape", "semantic_sha256"},
            "T5 tensor",
        )
        _require_sha256(tensor_row["semantic_sha256"], "T5 tensor semantic")
        if (
            tensor_row["dtype"] != EGO_T5_OUTPUT_DTYPE
            or not isinstance(tensor_row["shape"], list)
            or len(tensor_row["shape"]) != 2
            or isinstance(tensor_row["shape"][0], bool)
            or not isinstance(tensor_row["shape"][0], int)
            or not 1 <= tensor_row["shape"][0] <= EGO_T5_LOADER_MAX_PROMPT_LEN
            or tensor_row["shape"][1] != EGO_T5_OUTPUT_WIDTH
        ):
            raise EgoT5SourceError("invalid T5 tensor contract in manifest")
        if inspect_tensors:
            payload, _, _ = read_verified_bytes(path, expected_sha256=actual_sha256)
            actual_tensor = tensor_semantic_receipt(payload, name=str(path))
            if actual_tensor != tensor_row:
                raise EgoT5SourceError("T5 tensor semantic receipt differs")
        prompt = row["prompt"]
        semantic_sha = tensor_row["semantic_sha256"]
        previous_tensor = prompt_to_tensor.setdefault(prompt, semantic_sha)
        previous_prompt = tensor_to_prompt.setdefault(semantic_sha, prompt)
        if previous_tensor != semantic_sha:
            raise EgoT5SourceError("identical prompts produced different T5 tensors")
        if previous_prompt != prompt:
            raise EgoT5SourceError("different prompts share one T5 tensor")
        normalized_rows.append(dict(row))
        portable_files.append(
            {
                "path": f"t5_embedding/{filename}",
                "logical_role": "t5_episode_condition",
                "episode_index": episode_index,
                "digest_mode": "raw_file_bytes",
                "size_bytes": file_row["size_bytes"],
                "sha256": actual_sha256,
            }
        )
        runtime_receipt.append(
            {
                "logical_path": f"t5_embedding/{filename}",
                "logical_role": "t5_episode_condition",
                "episode_index": episode_index,
                "path_record_only": str(path),
                "sha256": actual_sha256,
                "stat": runtime_stat,
            }
        )
    if expected_by_episode:
        raise EgoT5SourceError("T5 manifest is missing episodes")
    normalized_rows.sort(key=lambda item: item["episode_index"])
    prompt_manifest = [
        {
            "episode_index": row["episode_index"],
            "prompt": row["prompt"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in normalized_rows
    ]
    if (
        manifest["episodes"] != normalized_rows
        or isinstance(manifest["episode_count"], bool)
        or not isinstance(manifest["episode_count"], int)
        or manifest["episode_count"] != len(normalized_rows)
        or isinstance(manifest["unique_prompt_count"], bool)
        or not isinstance(manifest["unique_prompt_count"], int)
        or manifest["unique_prompt_count"] != len(prompt_to_tensor)
        or manifest["prompt_manifest_sha256"] != canonical_sha256(prompt_manifest)
    ):
        raise EgoT5SourceError("T5 episode/prompt manifest summary differs")

    encoder_provenance_sha256 = canonical_sha256(encoder)
    return {
        "manifest": manifest,
        "manifest_path_record_only": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_stat_record_only": manifest_stat,
        "encoder_provenance_sha256": encoder_provenance_sha256,
        "prompt_manifest_sha256": manifest["prompt_manifest_sha256"],
        "portable_episode_files": portable_files,
        "runtime_episode_receipt_record_only": runtime_receipt,
        "episode_count": len(normalized_rows),
        "unique_prompt_count": len(prompt_to_tensor),
    }

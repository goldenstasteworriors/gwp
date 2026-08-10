"""Strict independent-replay evidence for production Ego Stage1 T5 sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from world_action_model.ego_t5_source import (
    EGO_T5_GENERATION_CODE_FILES,
    canonical_sha256,
)
from world_action_model.verified_io import hash_verified_regular_file


EGO_T5_REPLAY_ATTESTATION_VERSION = (
    "ego-t5-independent-replay-attestation/v1"
)
EGO_T5_REPLAY_VERIFIER_IDENTITY = "scripts/verify_ego_t5_replay.py"
EGO_T5_REPLAY_COMPARISON_MODE = (
    "exact-float32-contiguous-tensor-semantic-sha256/v1"
)
EGO_T5_REPLAY_CODE_FILES = (
    *EGO_T5_GENERATION_CODE_FILES,
    "world_action_model/ego_t5_replay_attestation.py",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class EgoT5ReplayAttestationError(ValueError):
    pass


def _exact(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EgoT5ReplayAttestationError(f"invalid {name} schema: {actual}")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EgoT5ReplayAttestationError(f"invalid {name} SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EgoT5ReplayAttestationError(f"invalid {name}")
    return value


def current_git_commit() -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EgoT5ReplayAttestationError(
            "independent verifier requires a Git checkout"
        ) from exc
    if not _GIT_SHA1_RE.fullmatch(value):
        raise EgoT5ReplayAttestationError("invalid verifier Git commit")
    return value


def current_replay_code_manifest() -> list[dict]:
    rows = []
    for logical_path in EGO_T5_REPLAY_CODE_FILES:
        path = PROJECT_ROOT / logical_path
        digest, runtime_stat = hash_verified_regular_file(path)
        try:
            blob = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT),
                    "rev-parse",
                    f"HEAD:{logical_path}",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EgoT5ReplayAttestationError(
                f"verifier code is absent from current HEAD: {logical_path}"
            ) from exc
        if not _GIT_SHA1_RE.fullmatch(blob):
            raise EgoT5ReplayAttestationError(
                f"invalid verifier code blob identity: {logical_path}"
            )
        rows.append(
            {
                "logical_path": logical_path,
                "git_blob_sha1": blob,
                "sha256": digest,
                "size_bytes": runtime_stat["size_bytes"],
            }
        )
    return rows


def _validate_code_manifest(rows: object) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != len(EGO_T5_REPLAY_CODE_FILES):
        raise EgoT5ReplayAttestationError("invalid replay verifier code manifest")
    normalized = []
    for row, expected_path in zip(rows, EGO_T5_REPLAY_CODE_FILES):
        row = _exact(
            row,
            {"logical_path", "git_blob_sha1", "sha256", "size_bytes"},
            "replay verifier code file",
        )
        if (
            row["logical_path"] != expected_path
            or not isinstance(row["git_blob_sha1"], str)
            or not _GIT_SHA1_RE.fullmatch(row["git_blob_sha1"])
        ):
            raise EgoT5ReplayAttestationError(
                "unexpected replay verifier code identity"
            )
        _sha256(row["sha256"], "replay verifier code")
        _positive_int(row["size_bytes"], "replay verifier code size")
        normalized.append(dict(row))
    return normalized


def expected_replay_results(generation_manifest: Mapping[str, Any]) -> list[dict]:
    grouped: dict[str, dict] = {}
    order = []
    for episode in generation_manifest["episodes"]:
        prompt = episode["prompt"]
        if prompt not in grouped:
            grouped[prompt] = {
                "prompt": prompt,
                "prompt_sha256": episode["prompt_sha256"],
                "episode_indices": [],
                "source_tensor": dict(episode["tensor"]),
            }
            order.append(prompt)
        row = grouped[prompt]
        if (
            row["prompt_sha256"] != episode["prompt_sha256"]
            or row["source_tensor"] != episode["tensor"]
        ):
            raise EgoT5ReplayAttestationError(
                "generation manifest prompt/tensor grouping is inconsistent"
            )
        row["episode_indices"].append(episode["episode_index"])
    return [
        {
            **grouped[prompt],
            "replay_tensor": dict(grouped[prompt]["source_tensor"]),
            "exact_match": True,
        }
        for prompt in order
    ]


def validate_ego_t5_replay_attestation(
    attestation: object,
    *,
    dataset_identity: Mapping[str, Any],
    t5_source: Mapping[str, Any],
    verify_current_code: bool = True,
) -> dict:
    """Validate one already strictly-parsed attestation against recomputed source."""

    attestation = _exact(
        attestation,
        {
            "attestation_version",
            "status",
            "independent_replay",
            "dataset_binding",
            "source_binding",
            "assets",
            "verifier",
            "replay_contract",
            "results",
            "results_manifest_sha256",
        },
        "independent T5 replay attestation",
    )
    if (
        attestation["attestation_version"] != EGO_T5_REPLAY_ATTESTATION_VERSION
        or attestation["status"] != "passed"
        or attestation["independent_replay"] is not True
    ):
        raise EgoT5ReplayAttestationError(
            "independent T5 replay attestation did not pass"
        )

    dataset = _exact(
        attestation["dataset_binding"],
        {"dataset_id", "episode_metadata_sha256"},
        "replay dataset binding",
    )
    if (
        dataset["dataset_id"] != dataset_identity["dataset_id"]
        or dataset["episode_metadata_sha256"]
        != dataset_identity["episode_metadata_sha256"]
    ):
        raise EgoT5ReplayAttestationError("replay dataset binding differs")

    receipt = t5_source["source_receipt"]
    generation_manifest = t5_source["provenance_manifest"]
    source = _exact(
        attestation["source_binding"],
        {
            "artifact_manifest_sha256",
            "t5_source_manifest_sha256",
            "t5_source_receipt_sha256",
            "encoder_provenance_sha256",
            "prompt_manifest_sha256",
            "generation_code_manifest_sha256",
        },
        "replay source binding",
    )
    expected_source = {
        "artifact_manifest_sha256": receipt["provenance_manifest_sha256"],
        "t5_source_manifest_sha256": t5_source["source_manifest_sha256"],
        "t5_source_receipt_sha256": t5_source["source_receipt_sha256"],
        "encoder_provenance_sha256": receipt["encoder_provenance_sha256"],
        "prompt_manifest_sha256": receipt["prompt_manifest_sha256"],
        "generation_code_manifest_sha256": generation_manifest["generator"][
            "code_manifest_sha256"
        ],
    }
    if source != expected_source:
        raise EgoT5ReplayAttestationError("replay source binding differs")

    encoder = generation_manifest["encoder"]
    assets = _exact(
        attestation["assets"],
        {
            "model_revision",
            "checkpoint_sha256",
            "checkpoint_hf_etag",
            "tokenizer_manifest_sha256",
            "encoder_code_manifest_sha256",
        },
        "replay asset binding",
    )
    expected_assets = {
        "model_revision": encoder["model_revision"],
        "checkpoint_sha256": encoder["checkpoint"]["sha256"],
        "checkpoint_hf_etag": encoder["checkpoint"]["hf_etag"],
        "tokenizer_manifest_sha256": encoder["tokenizer_manifest_sha256"],
        "encoder_code_manifest_sha256": encoder["encoder_code_manifest_sha256"],
    }
    if assets != expected_assets:
        raise EgoT5ReplayAttestationError("replay asset binding differs")

    verifier = _exact(
        attestation["verifier"],
        {
            "identity",
            "git_commit",
            "code_manifest",
            "code_manifest_sha256",
            "runtime_versions",
            "device_type",
            "process_isolation",
            "threat_model_boundary",
        },
        "replay verifier",
    )
    code_manifest = _validate_code_manifest(verifier["code_manifest"])
    if (
        verifier["identity"] != EGO_T5_REPLAY_VERIFIER_IDENTITY
        or not isinstance(verifier["git_commit"], str)
        or not _GIT_SHA1_RE.fullmatch(verifier["git_commit"])
        or verifier["code_manifest_sha256"] != canonical_sha256(code_manifest)
        or verifier["device_type"] not in {"cpu", "cuda"}
        or verifier["process_isolation"]
        != "separate_cli_process_no_gate_or_source_write_access"
        or verifier["threat_model_boundary"]
        != (
            "ordinary pathname races fail closed; same-UID process-memory or "
            "forged-metadata attacks are outside this verifier boundary"
        )
        or not isinstance(verifier["runtime_versions"], dict)
        or set(verifier["runtime_versions"])
        != {
            "python",
            "torch",
            "transformers",
            "tokenizers",
            "ftfy",
            "regex",
            "torch_load_weights_only",
        }
        or any(
            not isinstance(value, str) or not value
            for value in verifier["runtime_versions"].values()
        )
        or verifier["runtime_versions"]["torch_load_weights_only"]
        != "required_and_supported"
    ):
        raise EgoT5ReplayAttestationError("invalid replay verifier identity")
    if verify_current_code:
        current_manifest = current_replay_code_manifest()
        if (
            verifier["git_commit"] != current_git_commit()
            or code_manifest != current_manifest
        ):
            raise EgoT5ReplayAttestationError(
                "current verifier code differs from replay attestation"
            )

    expected_results = expected_replay_results(generation_manifest)
    contract = _exact(
        attestation["replay_contract"],
        {
            "comparison_mode",
            "all_unique_prompts",
            "episode_count",
            "unique_prompt_count",
            "replay_count",
            "distinct_output_directory",
        },
        "replay contract",
    )
    if (
        contract["comparison_mode"] != EGO_T5_REPLAY_COMPARISON_MODE
        or contract["all_unique_prompts"] is not True
        or contract["distinct_output_directory"] is not True
        or contract["episode_count"] != generation_manifest["episode_count"]
        or contract["unique_prompt_count"]
        != generation_manifest["unique_prompt_count"]
        or contract["replay_count"] != len(expected_results)
        or any(
            isinstance(contract[key], bool) or not isinstance(contract[key], int)
            for key in ("episode_count", "unique_prompt_count", "replay_count")
        )
    ):
        raise EgoT5ReplayAttestationError("replay contract differs")
    results = attestation["results"]
    if not isinstance(results, list) or results != expected_results:
        raise EgoT5ReplayAttestationError(
            "replay results are incomplete, reordered, or do not exactly match"
        )
    _sha256(attestation["results_manifest_sha256"], "replay results manifest")
    if attestation["results_manifest_sha256"] != canonical_sha256(results):
        raise EgoT5ReplayAttestationError("replay results manifest SHA differs")
    return dict(attestation)


def canonical_attestation_bytes(attestation: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            attestation,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

"""Checkpoint manifest/acceptance v3 helpers for causal-main Stage2 runs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

from world_action_model.contact_causal_production import (
    ACCEPTANCE_VERSION,
    CHECKPOINT_MANIFEST_VERSION,
    CHECKPOINT_STEPS,
    CausalProductionError,
    MATRIX_ID,
    MATRIX_SCHEMA_VERSION,
    PRECURSOR_ID,
    TRAINING_RECEIPT_VERSION,
    ensure_exact_causal_route,
    sha256_json,
    strict_json_file,
    validate_acceptance_v3,
    validate_training_receipt_v5,
    verify_checkpoint_file,
)
from world_action_model.verified_io import (
    hash_verified_regular_file,
    strict_json_loads,
)


SAFE_TENSOR_MANIFEST_VERSION = "contact-causal-safe-tensor-manifest/v1"
SAFE_WEIGHT_FILES = {
    "diffusion_pytorch_model.bin": "pytorch_weights_only",
    "diffusion_pytorch_model.safetensors": "safetensors",
}


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CausalProductionError(f"{name} schema mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CausalProductionError(f"{name} must be lowercase SHA-256")
    return value


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise CausalProductionError("checkpoint weights must be dense torch tensors")
    value = tensor.detach().to(device="cpu").contiguous()
    try:
        return value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CausalProductionError("cannot obtain canonical checkpoint tensor bytes") from exc


def load_safe_weight_mapping(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load only a safetensors file or a weights-only tensor mapping."""

    import torch

    weight_path = Path(path).expanduser().absolute()
    try:
        serialization = SAFE_WEIGHT_FILES[weight_path.name]
    except KeyError as exc:
        raise CausalProductionError("checkpoint weight filename/format is not approved") from exc
    try:
        if serialization == "safetensors":
            from safetensors.torch import load_file

            raw = load_file(str(weight_path), device="cpu")
        else:
            raw = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as exc:
        raise CausalProductionError("checkpoint weight file is not safely loadable") from exc
    if (
        not isinstance(raw, dict)
        or not raw
        or any(not isinstance(name, str) or not name for name in raw)
        or any(not isinstance(tensor, torch.Tensor) for tensor in raw.values())
    ):
        raise CausalProductionError(
            "checkpoint must contain one non-empty top-level string-to-tensor mapping"
        )
    return dict(raw), serialization


def build_safe_tensor_manifest(path: str | Path) -> dict[str, Any]:
    mapping, serialization = load_safe_weight_mapping(path)
    rows = []
    for name in sorted(mapping):
        tensor = mapping[name].detach().to(device="cpu").contiguous()
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "bytes_sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
            }
        )
    core = {
        "serialization": serialization,
        "tensor_count": len(rows),
        "total_numel": sum(row["numel"] for row in rows),
        "tensors": rows,
    }
    return {
        "manifest_version": SAFE_TENSOR_MANIFEST_VERSION,
        **core,
        "tensor_mapping_sha256": sha256_json(core),
    }


def validate_safe_tensor_manifest(value: object) -> dict[str, Any]:
    manifest = _exact(
        value,
        {
            "manifest_version",
            "serialization",
            "tensor_count",
            "total_numel",
            "tensors",
            "tensor_mapping_sha256",
        },
        "safe tensor manifest",
    )
    if (
        manifest["manifest_version"] != SAFE_TENSOR_MANIFEST_VERSION
        or manifest["serialization"] not in set(SAFE_WEIGHT_FILES.values())
        or isinstance(manifest["tensor_count"], bool)
        or not isinstance(manifest["tensor_count"], int)
        or manifest["tensor_count"] <= 0
        or isinstance(manifest["total_numel"], bool)
        or not isinstance(manifest["total_numel"], int)
        or manifest["total_numel"] <= 0
    ):
        raise CausalProductionError("safe tensor manifest identity/count differs")
    rows = manifest["tensors"]
    if not isinstance(rows, list) or len(rows) != manifest["tensor_count"]:
        raise CausalProductionError("safe tensor manifest rows differ")
    names = []
    total = 0
    for row in rows:
        row = _exact(
            row,
            {"name", "shape", "dtype", "numel", "bytes_sha256"},
            "safe tensor row",
        )
        if (
            not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(row["dtype"], str)
            or not row["dtype"].startswith("torch.")
            or not isinstance(row["shape"], list)
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in row["shape"])
            or isinstance(row["numel"], bool)
            or not isinstance(row["numel"], int)
            or row["numel"] < 0
        ):
            raise CausalProductionError("safe tensor row metadata differs")
        expected_numel = 1
        for dim in row["shape"]:
            expected_numel *= dim
        if expected_numel != row["numel"]:
            raise CausalProductionError("safe tensor shape/numel differs")
        _sha(row["bytes_sha256"], "safe tensor bytes")
        names.append(row["name"])
        total += row["numel"]
    if names != sorted(set(names)) or total != manifest["total_numel"]:
        raise CausalProductionError("safe tensor names/total differ")
    core = {
        "serialization": manifest["serialization"],
        "tensor_count": manifest["tensor_count"],
        "total_numel": manifest["total_numel"],
        "tensors": rows,
    }
    if sha256_json(core) != manifest["tensor_mapping_sha256"]:
        raise CausalProductionError("safe tensor mapping digest differs")
    return manifest


def inspect_causal_checkpoint_bundle(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Rebuild the exact config + unique safe-weight bundle from disk."""

    from world_action_model.contact_asym25_checkpoint_acceptance import (
        build_bundle_manifest,
    )
    from world_action_model.verified_io import read_verified_bytes

    root = Path(checkpoint_dir).expanduser().absolute()
    try:
        bundle = build_bundle_manifest(root)
    except (OSError, ValueError) as exc:
        raise CausalProductionError("causal checkpoint bundle layout differs") from exc
    weight_name = bundle["weight_file"]
    if weight_name not in SAFE_WEIGHT_FILES:
        raise CausalProductionError("causal checkpoint uses an unapproved weight file")
    try:
        config_payload, config_sha256, config_stat = read_verified_bytes(
            root / "config.json"
        )
        config = strict_json_loads(config_payload, name="causal checkpoint config")
        weight_sha256, weight_stat = hash_verified_regular_file(root / weight_name)
    except (OSError, ValueError) as exc:
        raise CausalProductionError("causal checkpoint config/weight bytes are invalid") from exc
    if not isinstance(config, dict) or not config or config_stat["size_bytes"] <= 0:
        raise CausalProductionError("causal checkpoint config must be a non-empty JSON object")
    tensor_manifest = build_safe_tensor_manifest(root / weight_name)
    validate_safe_tensor_manifest(tensor_manifest)
    return {
        "bundle_manifest": bundle,
        "config": config,
        "config_sha256": config_sha256,
        "weight_file": weight_name,
        "weight_file_sha256": weight_sha256,
        "weight_file_size_bytes": weight_stat["size_bytes"],
        "tensor_manifest": tensor_manifest,
        "tensor_manifest_sha256": sha256_json(tensor_manifest),
    }


def validate_weight_mapping_against_topology(
    path: str | Path, topology: Mapping[str, Any]
) -> dict[str, str]:
    """Bind safely loaded state_dict tensors to the live model partition."""

    from world_action_model.contact_causal_production import named_tensor_digest

    mapping, _ = load_safe_weight_mapping(path)
    parameter_rows = topology.get("persistent_parameter_schema")
    buffer_rows = topology.get("persistent_buffer_schema")
    if not isinstance(parameter_rows, list) or not isinstance(buffer_rows, list):
        raise CausalProductionError("checkpoint topology schemas are absent")
    expected_rows = {}
    parameter_names = []
    buffer_names = []
    for rows, names, label in (
        (parameter_rows, parameter_names, "parameter"),
        (buffer_rows, buffer_names, "buffer"),
    ):
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"name", "shape", "dtype"}:
                raise CausalProductionError(f"checkpoint {label} topology row differs")
            expected_rows[row["name"]] = (row["shape"], row["dtype"])
            names.append(row["name"])
    if set(mapping) != set(expected_rows) or len(expected_rows) != len(mapping):
        raise CausalProductionError("checkpoint state_dict names differ from live topology")
    for name, tensor in mapping.items():
        shape, dtype = expected_rows[name]
        if list(tensor.shape) != shape or str(tensor.dtype) != dtype:
            raise CausalProductionError(
                f"checkpoint tensor metadata differs from live topology: {name}"
            )
    result = {
        "parameter_tensor_sha256": named_tensor_digest(
            [(name, mapping[name]) for name in parameter_names]
        ),
        "buffer_tensor_sha256": named_tensor_digest(
            [(name, mapping[name]) for name in buffer_names]
        ),
    }
    if (
        result["parameter_tensor_sha256"] != topology.get("parameter_tensor_sha256")
        or result["buffer_tensor_sha256"] != topology.get("buffer_tensor_sha256")
    ):
        raise CausalProductionError("checkpoint tensor bytes differ from live topology")
    return result


def validate_checkpoint_manifest_v3(
    value: object, *, verify_files: bool = True
) -> tuple[dict[str, Any], dict[int, tuple[dict[str, Any], dict[str, Any]]]]:
    manifest = _exact(
        value,
        {
            "manifest_version",
            "matrix_schema_version",
            "matrix_id",
            "precursor_id",
            "precursor_receipt_bytes_sha256",
            "precursor_bundle_sha256",
            "seed",
            "causal_route_id",
            "training_run_id",
            "training_run_nonce",
            "primary_checkpoint_step",
            "best_checkpoint_selection_allowed",
            "checkpoints",
        },
        "causal checkpoint manifest",
    )
    route = ensure_exact_causal_route(manifest["causal_route_id"])
    if (
        manifest["manifest_version"] != CHECKPOINT_MANIFEST_VERSION
        or manifest["matrix_schema_version"] != MATRIX_SCHEMA_VERSION
        or manifest["matrix_id"] != MATRIX_ID
        or manifest["precursor_id"] != PRECURSOR_ID
        or manifest["primary_checkpoint_step"] != 200
        or manifest["best_checkpoint_selection_allowed"] is not False
    ):
        raise CausalProductionError("causal checkpoint manifest identity differs")
    _sha(manifest["precursor_receipt_bytes_sha256"], "manifest precursor receipt")
    _sha(manifest["precursor_bundle_sha256"], "manifest precursor bundle")
    entries = manifest["checkpoints"]
    if not isinstance(entries, list) or len(entries) != 3:
        raise CausalProductionError("causal checkpoint manifest needs step 0/100/200")
    accepted: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw_entry in entries:
        entry = _exact(
            raw_entry,
            {
                "step",
                "training_receipt_path",
                "training_receipt_sha256",
                "independent_acceptance_path",
                "independent_acceptance_sha256",
                "checkpoint_file_path",
                "checkpoint_file_sha256",
            },
            "causal checkpoint manifest entry",
        )
        step = entry["step"]
        if isinstance(step, bool) or step not in CHECKPOINT_STEPS or step in accepted:
            raise CausalProductionError("causal checkpoint step is invalid/duplicate")
        for name in (
            "training_receipt_path",
            "independent_acceptance_path",
            "checkpoint_file_path",
        ):
            if not isinstance(entry[name], str) or not Path(entry[name]).is_absolute():
                raise CausalProductionError("causal checkpoint evidence paths must be absolute")
        receipt, receipt_digest, _ = strict_json_file(
            entry["training_receipt_path"], name=f"causal training receipt step {step}"
        )
        validate_training_receipt_v5(receipt)
        acceptance, acceptance_digest, _ = strict_json_file(
            entry["independent_acceptance_path"],
            name=f"causal independent acceptance step {step}",
        )
        validate_acceptance_v3(acceptance, receipt=receipt)
        if (
            receipt_digest != entry["training_receipt_sha256"]
            or acceptance_digest != entry["independent_acceptance_sha256"]
            or receipt["checkpoint_file"]["path_record_only"]
            != entry["checkpoint_file_path"]
            or receipt["checkpoint_file"]["sha256"]
            != entry["checkpoint_file_sha256"]
            or receipt["checkpoint_step"] != step
            or receipt["seed"] != manifest["seed"]
            or receipt["route"]["causal_route_id"] != route
            or receipt["run_identity"]["training_run_id"]
            != manifest["training_run_id"]
            or receipt["run_identity"]["training_run_nonce"]
            != manifest["training_run_nonce"]
            or receipt["precursor"]["receipt_bytes_sha256"]
            != manifest["precursor_receipt_bytes_sha256"]
            or receipt["precursor"]["bundle_sha256"]
            != manifest["precursor_bundle_sha256"]
        ):
            raise CausalProductionError("causal checkpoint manifest entry binding differs")
        if verify_files:
            verify_checkpoint_file(receipt)
        accepted[step] = (receipt, acceptance)
    if set(accepted) != set(CHECKPOINT_STEPS):
        raise CausalProductionError("causal checkpoint step set differs")
    rows = [accepted[step][0] for step in CHECKPOINT_STEPS]
    if (
        rows[0]["parent_checkpoint_weight_sha256"]
        != rows[0]["precursor"]["parameter_tensor_sha256"]
        or rows[1]["parent_checkpoint_weight_sha256"]
        != rows[0]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]
        or rows[2]["parent_checkpoint_weight_sha256"]
        != rows[1]["checkpoint_artifact_hashes"]["weight_tensor_sha256"]
    ):
        raise CausalProductionError("causal checkpoint parent chain differs")
    return manifest, accepted


def build_checkpoint_manifest_v3(
    *,
    receipts_and_acceptances: Mapping[
        int, tuple[str | Path, Mapping[str, Any], str | Path, Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    if set(receipts_and_acceptances) != set(CHECKPOINT_STEPS):
        raise CausalProductionError("checkpoint manifest builder needs step 0/100/200")
    first = receipts_and_acceptances[0][1]
    validate_training_receipt_v5(first)
    entries = []
    for step in CHECKPOINT_STEPS:
        receipt_path, receipt, acceptance_path, acceptance = receipts_and_acceptances[step]
        validate_training_receipt_v5(receipt)
        validate_acceptance_v3(acceptance, receipt=receipt)
        entries.append(
            {
                "step": step,
                "training_receipt_path": str(Path(receipt_path).absolute()),
                "training_receipt_sha256": sha256_json(receipt),
                "independent_acceptance_path": str(Path(acceptance_path).absolute()),
                "independent_acceptance_sha256": sha256_json(acceptance),
                "checkpoint_file_path": receipt["checkpoint_file"]["path_record_only"],
                "checkpoint_file_sha256": receipt["checkpoint_file"]["sha256"],
            }
        )
    return {
        "manifest_version": CHECKPOINT_MANIFEST_VERSION,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "precursor_id": PRECURSOR_ID,
        "precursor_receipt_bytes_sha256": first["precursor"]["receipt_bytes_sha256"],
        "precursor_bundle_sha256": first["precursor"]["bundle_sha256"],
        "seed": first["seed"],
        "causal_route_id": first["route"]["causal_route_id"],
        "training_run_id": first["run_identity"]["training_run_id"],
        "training_run_nonce": first["run_identity"]["training_run_nonce"],
        "primary_checkpoint_step": 200,
        "best_checkpoint_selection_allowed": False,
        "checkpoints": entries,
    }

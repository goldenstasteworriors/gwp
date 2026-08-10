#!/usr/bin/env python3
"""Plan, generate, or validate a provenance-bound Ego Stage1 T5 source.

Generation always starts from a nonexistent output directory and a pinned WAN
UMT5 checkpoint.  Existing unproven ``.pt`` files are never retroactively
attested.  Staging stays next to the requested output, never in ``/tmp``.
"""

from __future__ import annotations

import argparse
import ctypes
from importlib import import_module, metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_secure_cli_imports() -> Path:
    """Isolate project imports from timestamp-based bytecode caches."""

    if not sys.dont_write_bytecode:
        raise RuntimeError("secure Ego T5 CLI execution requires python -B")

    root_real = os.path.realpath(PROJECT_ROOT)
    scripts_real = os.path.realpath(PROJECT_ROOT / "scripts")
    retained_paths = []
    for entry in sys.path:
        candidate = os.path.abspath(entry or os.getcwd())
        candidate_real = os.path.realpath(candidate)
        if candidate_real in {root_real, scripts_real}:
            continue
        shadow_markers = (
            Path(candidate) / "world_action_model" / "__init__.py",
            Path(candidate) / "scripts" / "compute_t5_embedding.py",
            Path(candidate) / "third_party" / "wan" / "__init__.py",
        )
        if any(marker.exists() or marker.is_symlink() for marker in shadow_markers):
            raise RuntimeError(
                f"another sys.path entry can resolve project modules: {candidate}"
            )
        retained_paths.append(entry)
    sys.path[:] = [str(PROJECT_ROOT), *retained_paths]

    import importlib.util

    package_spec = importlib.util.find_spec("world_action_model")
    expected_package = PROJECT_ROOT / "world_action_model" / "__init__.py"
    if (
        package_spec is None
        or not isinstance(package_spec.origin, str)
        or not os.path.samefile(package_spec.origin, expected_package)
    ):
        raise RuntimeError("world_action_model import origin is not this checkout")

    def walk_error(exc: OSError) -> None:
        raise RuntimeError("cannot audit project bytecode caches") from exc

    for code_root in (
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "world_action_model",
        PROJECT_ROOT / "third_party" / "wan",
    ):
        for _, directory_names, file_names in os.walk(
            code_root,
            followlinks=False,
            onerror=walk_error,
        ):
            if "__pycache__" in directory_names or any(
                name.endswith(".pyc") for name in file_names
            ):
                raise RuntimeError(
                    f"project code root contains forbidden bytecode cache: {code_root}"
                )
    if any(PROJECT_ROOT.glob(".gwp-secure-pycache-*")):
        raise RuntimeError("stale secure pycache prefix exists in project root")

    prefix = PROJECT_ROOT / (
        f".gwp-secure-pycache-{os.getpid()}-{os.urandom(16).hex()}"
    )
    os.mkdir(prefix, mode=0o700)
    if prefix.is_symlink() or not prefix.is_dir() or prefix.stat().st_mode & 0o077:
        raise RuntimeError("secure pycache prefix is not a private real directory")
    if any(prefix.iterdir()):
        raise RuntimeError("secure pycache prefix was not created empty")
    sys.pycache_prefix = str(prefix)

    def cleanup() -> None:
        if prefix.exists() and prefix.is_dir() and not prefix.is_symlink():
            shutil.rmtree(prefix)

    import atexit

    atexit.register(cleanup)
    return prefix


_SECURE_CLI_PYCACHE = None
if __name__ == "__main__":
    try:
        _SECURE_CLI_PYCACHE = _bootstrap_secure_cli_imports()
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import world_action_model as _wam_package
import world_action_model.ego_dataset_fingerprint as _fingerprint_module
import world_action_model.ego_t5_source as _t5_source_module
import world_action_model.verified_io as _verified_io_module
from world_action_model.ego_dataset_fingerprint import derive_ego_dataset_identity
from world_action_model.ego_t5_source import (
    EGO_T5_CHECKPOINT_LOGICAL_PATH,
    EGO_T5_ENCODER_ARCHITECTURE,
    EGO_T5_ENCODER_BACKEND,
    EGO_T5_ENCODER_CODE_FILES,
    EGO_T5_GENERATOR_IDENTITY,
    EGO_T5_GENERATION_CODE_FILES,
    EGO_T5_LOADER_MAX_PROMPT_LEN,
    EGO_T5_MANIFEST_FILENAME,
    EGO_T5_MANIFEST_VERSION,
    EGO_T5_MODEL_REPO_ID,
    EGO_T5_OUTPUT_DTYPE,
    EGO_T5_OUTPUT_WIDTH,
    EGO_T5_PROMPT_CLEANING,
    EGO_T5_PROMPT_SELECTION,
    EGO_T5_TEXT_LEN,
    EGO_T5_TOKENIZER_FILES,
    EgoT5SourceError,
    canonical_sha256,
    load_episode_prompts,
    tensor_semantic_receipt,
    validate_ego_t5_source_manifest,
)
from world_action_model.verified_io import (
    assert_python_module_origin,
    hash_verified_regular_file,
    load_verified_head_python_module,
    read_verified_bytes,
    verified_head_code_manifest,
)


GENERATION_CODE_FILES = EGO_T5_GENERATION_CODE_FILES
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _require_secure_cli_runtime() -> None:
    prefix = _SECURE_CLI_PYCACHE
    if (
        prefix is None
        or sys.pycache_prefix != str(prefix)
        or prefix.is_symlink()
        or not prefix.is_dir()
        or any(prefix.iterdir())
    ):
        raise EgoT5SourceError(
            "generation is allowed only through the python -B secure CLI launcher"
        )


def _json_dump(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise EgoT5SourceError("secure staging requires POSIX directory/no-follow flags")
    fd = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive_file(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        raise EgoT5SourceError("secure staging requires POSIX O_NOFOLLOW")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while creating verified artifact")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise EgoT5SourceError("atomic no-replace publication requires renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _current_code_manifest(*, expected_git_commit: str | None = None) -> list[dict]:
    try:
        _, rows = verified_head_code_manifest(
            PROJECT_ROOT,
            GENERATION_CODE_FILES,
            expected_commit=expected_git_commit,
        )
    except ValueError as exc:
        raise EgoT5SourceError("generator/WAN code is not identical to HEAD bytes") from exc
    return rows


def _assert_code_manifest(expected: list[dict], *, expected_git_commit: str) -> None:
    if _current_code_manifest(expected_git_commit=expected_git_commit) != expected:
        raise EgoT5SourceError("generator/WAN code changed during T5 encoding")


def _assert_generation_module_origins() -> None:
    try:
        for module, logical_path in (
            (_wam_package, "world_action_model/__init__.py"),
            (
                _fingerprint_module,
                "world_action_model/ego_dataset_fingerprint.py",
            ),
            (_t5_source_module, "world_action_model/ego_t5_source.py"),
            (_verified_io_module, "world_action_model/verified_io.py"),
        ):
            assert_python_module_origin(module, PROJECT_ROOT / logical_path)
    except ValueError as exc:
        raise EgoT5SourceError("loaded project module origin differs") from exc


def _load_verified_compute_module(generator_receipt: dict):
    logical_path = "scripts/compute_t5_embedding.py"
    matches = [
        row
        for row in generator_receipt["code_manifest"]
        if row["logical_path"] == logical_path
    ]
    if len(matches) != 1:
        raise EgoT5SourceError("generation code manifest lacks unique compute child")
    try:
        return load_verified_head_python_module(
            PROJECT_ROOT,
            logical_path,
            f"_gwp_verified_compute_{os.getpid()}_{os.urandom(8).hex()}",
            expected_commit=generator_receipt["git_commit"],
            expected_row=matches[0],
        )
    except ValueError as exc:
        raise EgoT5SourceError(
            "compute child source/origin differs from current HEAD"
        ) from exc


def _require_current_wan_code_path(wan_code_path: Path) -> None:
    expected = PROJECT_ROOT / "third_party"
    try:
        same_directory = os.path.samefile(wan_code_path, expected)
    except OSError as exc:
        raise EgoT5SourceError("cannot bind WAN code to the current checkout") from exc
    if not same_directory:
        raise EgoT5SourceError(
            "WAN code path must be this checkout's current-HEAD third_party directory"
        )


def _create_tokenizer_snapshot(
    *,
    model_root: Path,
    snapshot_dir: Path,
    tokenizer_manifest: list[dict],
    runtime_receipts: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    os.mkdir(snapshot_dir, mode=0o700)
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise EgoT5SourceError("tokenizer snapshot must be a private real directory")
    if snapshot_dir.stat().st_mode & 0o077:
        raise EgoT5SourceError("tokenizer snapshot permissions are not private")
    expected_by_path = {row["logical_path"]: row for row in tokenizer_manifest}
    if set(expected_by_path) != set(EGO_T5_TOKENIZER_FILES):
        raise EgoT5SourceError("tokenizer provenance file inventory differs")
    snapshot_receipts = {}
    for logical_path in EGO_T5_TOKENIZER_FILES:
        row = expected_by_path[logical_path]
        payload, digest, _ = read_verified_bytes(
            model_root / logical_path,
            expected_sha256=row["sha256"],
            expected_stat=runtime_receipts[logical_path],
        )
        if digest != row["sha256"] or len(payload) != row["size_bytes"]:
            raise EgoT5SourceError("tokenizer source differs from encoder provenance")
        destination = snapshot_dir / Path(logical_path).name
        _write_exclusive_file(destination, payload)
        copied_digest, copied_stat = hash_verified_regular_file(
            destination, expected_sha256=row["sha256"]
        )
        if copied_digest != row["sha256"] or copied_stat["size_bytes"] != row["size_bytes"]:
            raise EgoT5SourceError("private tokenizer snapshot differs after copy")
        snapshot_receipts[destination.name] = copied_stat
    expected_names = {Path(path).name for path in EGO_T5_TOKENIZER_FILES}
    if {entry.name for entry in snapshot_dir.iterdir()} != expected_names:
        raise EgoT5SourceError("private tokenizer snapshot contains unexpected files")
    _fsync_directory(snapshot_dir)
    return snapshot_receipts


def _verify_tokenizer_snapshot(
    snapshot_dir: Path,
    tokenizer_manifest: list[dict],
    snapshot_receipts: dict[str, dict[str, int]],
) -> None:
    expected_names = {Path(path).name for path in EGO_T5_TOKENIZER_FILES}
    if (
        snapshot_dir.is_symlink()
        or not snapshot_dir.is_dir()
        or {entry.name for entry in snapshot_dir.iterdir()} != expected_names
    ):
        raise EgoT5SourceError("private tokenizer snapshot inventory changed")
    expected_by_name = {
        Path(row["logical_path"]).name: row for row in tokenizer_manifest
    }
    for name in sorted(expected_names):
        row = expected_by_name[name]
        digest, runtime_stat = hash_verified_regular_file(
            snapshot_dir / name,
            expected_sha256=row["sha256"],
            expected_stat=snapshot_receipts[name],
        )
        if digest != row["sha256"] or runtime_stat["size_bytes"] != row["size_bytes"]:
            raise EgoT5SourceError("private tokenizer snapshot changed during load")


def _source_file_row(root: Path, logical_path: str) -> tuple[dict, dict[str, int]]:
    path = root / logical_path
    digest, stat = hash_verified_regular_file(path)
    return (
        {
            "logical_path": logical_path,
            "sha256": digest,
            "size_bytes": stat["size_bytes"],
        },
        stat,
    )


def build_encoder_provenance(
    *,
    model_root: Path,
    wan_code_path: Path,
    model_revision: str,
    checkpoint_hf_etag: str,
    device_type: str,
    expected_checkpoint_sha256: str | None = None,
    include_runtime_receipts: bool = False,
) -> dict | tuple[dict, dict[str, dict[str, int]]]:
    if model_root.is_symlink() or not model_root.is_dir():
        raise EgoT5SourceError(f"model root must be a real directory: {model_root}")
    if wan_code_path.is_symlink() or not wan_code_path.is_dir():
        raise EgoT5SourceError(
            f"WAN code root must be a real directory: {wan_code_path}"
        )
    if len(model_revision) != 40 or any(
        char not in "0123456789abcdef" for char in model_revision
    ):
        raise EgoT5SourceError("model revision must be a full lowercase 40-hex commit")
    if (
        not isinstance(checkpoint_hf_etag, str)
        or len(checkpoint_hf_etag) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_hf_etag)
    ):
        raise EgoT5SourceError(
            "checkpoint HF etag must be an external reviewer-supplied SHA-256"
        )
    checkpoint, checkpoint_stat = _source_file_row(
        model_root, EGO_T5_CHECKPOINT_LOGICAL_PATH
    )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint["sha256"] != expected_checkpoint_sha256
    ):
        raise EgoT5SourceError("checkpoint differs from external expected SHA-256")
    checkpoint["hf_etag"] = checkpoint_hf_etag
    tokenizer_rows = [
        _source_file_row(model_root, path) for path in EGO_T5_TOKENIZER_FILES
    ]
    code_rows = [
        _source_file_row(wan_code_path, path) for path in EGO_T5_ENCODER_CODE_FILES
    ]
    tokenizer_manifest = [row for row, _ in tokenizer_rows]
    code_manifest = [row for row, _ in code_rows]
    compute_dtype = "float32" if device_type == "cpu" else "bfloat16"
    provenance = {
        "backend": EGO_T5_ENCODER_BACKEND,
        "architecture": EGO_T5_ENCODER_ARCHITECTURE,
        "model_repo_id": EGO_T5_MODEL_REPO_ID,
        "model_revision": model_revision,
        "tokenizer_repo_id": EGO_T5_MODEL_REPO_ID,
        "tokenizer_revision": model_revision,
        "tokenizer_subdir": "google/umt5-xxl",
        "checkpoint": checkpoint,
        "tokenizer_manifest": tokenizer_manifest,
        "tokenizer_manifest_sha256": canonical_sha256(tokenizer_manifest),
        "encoder_code_manifest": code_manifest,
        "encoder_code_manifest_sha256": canonical_sha256(code_manifest),
        "text_len": EGO_T5_TEXT_LEN,
        "loader_max_prompt_len": EGO_T5_LOADER_MAX_PROMPT_LEN,
        "add_special_tokens": True,
        "clean": "whitespace",
        "compute_dtype": compute_dtype,
        "output_dtype": EGO_T5_OUTPUT_DTYPE,
        "output_width": EGO_T5_OUTPUT_WIDTH,
    }
    if not include_runtime_receipts:
        return provenance
    runtime_receipts = {
        EGO_T5_CHECKPOINT_LOGICAL_PATH: checkpoint_stat,
        **{
            row["logical_path"]: runtime_stat
            for row, runtime_stat in tokenizer_rows + code_rows
        },
    }
    return provenance, runtime_receipts


def _runtime_precheck(device_type: str) -> dict:
    try:
        import_module("ftfy")
        import_module("regex")
        torch = import_module("torch")
        transformers = import_module("transformers")
        tokenizers = import_module("tokenizers")
        ftfy_version = metadata.version("ftfy")
        regex_version = metadata.version("regex")
    except ImportError as exc:
        raise EgoT5SourceError(
            "generation environment requires torch, transformers, ftfy, and regex"
        ) from exc
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise EgoT5SourceError(
            "generation requires a torch.load version with explicit weights_only support"
        )
    if device_type == "cuda" and not torch.cuda.is_available():
        raise EgoT5SourceError(
            "requested CUDA generation but the project runtime reports no device"
        )
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "tokenizers": str(tokenizers.__version__),
        "ftfy": ftfy_version,
        "regex": regex_version,
        "torch_load_weights_only": "required_and_supported",
    }


def _git_generation_receipt(device_type: str, runtime_versions: dict) -> dict:
    try:
        commit, code_manifest = verified_head_code_manifest(
            PROJECT_ROOT,
            GENERATION_CODE_FILES,
        )
    except ValueError as exc:
        raise EgoT5SourceError(
            "generator code must exactly match current HEAD blob bytes"
        ) from exc
    return {
        "identity": EGO_T5_GENERATOR_IDENTITY,
        "git_commit": commit,
        "code_manifest": code_manifest,
        "code_manifest_sha256": canonical_sha256(code_manifest),
        "device_type": device_type,
        "runtime_versions": runtime_versions,
        "one_encode_per_unique_prompt": True,
        "episode_file_pattern": "episode_{episode_index:06d}.pt",
        "claim_scope": "generator_authored_generation_claim_not_independent_proof",
        "threat_model_boundary": (
            "ordinary pathname races fail closed; same-UID process-memory or "
            "forged-metadata attacks are outside this verifier boundary"
        ),
    }


def _save_tensor(path: Path, tensor) -> tuple[dict, dict]:
    import torch

    tensor = tensor.detach().cpu().to(dtype=torch.float32).contiguous().clone()
    if (
        tensor.ndim != 2
        or not 1 <= int(tensor.shape[0]) <= EGO_T5_LOADER_MAX_PROMPT_LEN
        or int(tensor.shape[1]) != EGO_T5_OUTPUT_WIDTH
        or not bool(torch.isfinite(tensor).all())
    ):
        raise EgoT5SourceError(
            f"encoder output is not finite float32 [1..64,4096]: {tuple(tensor.shape)}"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as output_file:
            fd = None
            torch.save(tensor, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
    finally:
        if fd is not None:
            os.close(fd)
    payload, file_sha256, stat = read_verified_bytes(path)
    tensor_receipt = tensor_semantic_receipt(payload, name=str(path))
    return (
        {
            "logical_path": path.name,
            "sha256": file_sha256,
            "size_bytes": stat["size_bytes"],
        },
        tensor_receipt,
    )


def generate(args: argparse.Namespace) -> dict:
    _require_secure_cli_runtime()
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))
    output_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    model_root = Path(os.path.abspath(os.path.expanduser(args.model_root)))
    wan_code_path = Path(os.path.abspath(os.path.expanduser(args.wan_code_path)))
    _require_current_wan_code_path(wan_code_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise EgoT5SourceError(
            "strict generation requires a nonexistent output directory; "
            "old files cannot be attested"
        )
    if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
        raise EgoT5SourceError("output parent must be an existing real directory")
    if args.device == "cpu":
        device_type = "cpu"
    elif args.device == "cuda" or (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        device_type = "cuda"
    else:
        raise EgoT5SourceError("device must be exactly cpu, cuda, or cuda:<index>")

    runtime_versions = _runtime_precheck(device_type)
    generator_receipt = _git_generation_receipt(device_type, runtime_versions)
    _assert_generation_module_origins()
    identity = derive_ego_dataset_identity(data_root)
    prompts = load_episode_prompts(
        data_root, expected_episode_lengths=identity["episode_lengths"]
    )
    encoder_provenance, asset_runtime_receipts = build_encoder_provenance(
        model_root=model_root,
        wan_code_path=wan_code_path,
        model_revision=args.model_revision,
        checkpoint_hf_etag=args.checkpoint_hf_etag,
        device_type=device_type,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        include_runtime_receipts=True,
    )
    staging = output_dir.parent / (
        f".{output_dir.name}.staging-{os.getpid()}-{os.urandom(16).hex()}"
    )
    os.mkdir(staging, mode=0o700)
    tokenizer_snapshot = staging / ".tokenizer-snapshot"
    try:
        _assert_code_manifest(
            generator_receipt["code_manifest"],
            expected_git_commit=generator_receipt["git_commit"],
        )
        compute_module = _load_verified_compute_module(generator_receipt)
        _encode_t5 = compute_module._encode_t5
        _init_wan_t5_encoder = compute_module._init_wan_t5_encoder
        _assert_code_manifest(
            generator_receipt["code_manifest"],
            expected_git_commit=generator_receipt["git_commit"],
        )
        snapshot_receipts = _create_tokenizer_snapshot(
            model_root=model_root,
            snapshot_dir=tokenizer_snapshot,
            tokenizer_manifest=encoder_provenance["tokenizer_manifest"],
            runtime_receipts=asset_runtime_receipts,
        )
        encoder = _init_wan_t5_encoder(
            wan_path=str(model_root),
            device=args.device,
            text_len=EGO_T5_TEXT_LEN,
            wan_code_path=str(wan_code_path),
            checkpoint_sha256=encoder_provenance["checkpoint"]["sha256"],
            checkpoint_stat=asset_runtime_receipts[
                EGO_T5_CHECKPOINT_LOGICAL_PATH
            ],
            tokenizer_path_override=str(tokenizer_snapshot),
            tokenizer_local_files_only=True,
            code_git_commit=generator_receipt["git_commit"],
            code_manifest=generator_receipt["code_manifest"],
        )
        _assert_code_manifest(
            generator_receipt["code_manifest"],
            expected_git_commit=generator_receipt["git_commit"],
        )
        _verify_tokenizer_snapshot(
            tokenizer_snapshot,
            encoder_provenance["tokenizer_manifest"],
            snapshot_receipts,
        )
        shutil.rmtree(tokenizer_snapshot)
        _fsync_directory(staging)
        encoded = {}
        episode_rows = []
        prompt_to_semantic = {}
        semantic_to_prompt = {}
        for row in prompts:
            prompt = row["prompt"]
            if prompt not in encoded:
                _assert_code_manifest(
                    generator_receipt["code_manifest"],
                    expected_git_commit=generator_receipt["git_commit"],
                )
                encoded[prompt] = _encode_t5(encoder, prompt, device=args.device)
                _assert_code_manifest(
                    generator_receipt["code_manifest"],
                    expected_git_commit=generator_receipt["git_commit"],
                )
            path = staging / f"episode_{row['episode_index']:06d}.pt"
            file_receipt, tensor_receipt = _save_tensor(path, encoded[prompt])
            semantic = tensor_receipt["semantic_sha256"]
            if prompt in prompt_to_semantic and prompt_to_semantic[prompt] != semantic:
                raise EgoT5SourceError("identical prompts produced different tensors")
            if semantic in semantic_to_prompt and semantic_to_prompt[semantic] != prompt:
                raise EgoT5SourceError("different prompts produced one identical tensor")
            prompt_to_semantic[prompt] = semantic
            semantic_to_prompt[semantic] = prompt
            episode_rows.append(
                {
                    "episode_index": row["episode_index"],
                    "prompt": prompt,
                    "prompt_sha256": row["prompt_sha256"],
                    "file": file_receipt,
                    "tensor": tensor_receipt,
                }
            )
        _assert_code_manifest(
            generator_receipt["code_manifest"],
            expected_git_commit=generator_receipt["git_commit"],
        )
        prompt_manifest = [
            {
                "episode_index": row["episode_index"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
            }
            for row in episode_rows
        ]
        manifest = {
            "manifest_version": EGO_T5_MANIFEST_VERSION,
            "status": "generation_claim_from_pinned_assets",
            "claim_scope": "generator_authored_not_independent_replay_proof",
            "dataset": {
                "dataset_id": identity["dataset_id"],
                "fingerprint_version": identity["fingerprint_version"],
                "episode_metadata_sha256": identity["episode_metadata_sha256"],
                "prompt_source_logical_path": "meta/episodes.jsonl",
                "prompt_selection": EGO_T5_PROMPT_SELECTION,
                "prompt_cleaning": EGO_T5_PROMPT_CLEANING,
            },
            "encoder": encoder_provenance,
            "generator": generator_receipt,
            "episodes": episode_rows,
            "episode_count": len(episode_rows),
            "unique_prompt_count": len(prompt_to_semantic),
            "prompt_manifest_sha256": canonical_sha256(prompt_manifest),
        }
        manifest_path = staging / EGO_T5_MANIFEST_FILENAME
        _write_exclusive_file(
            manifest_path,
            (_json_dump(manifest) + "\n").encode("utf-8"),
        )
        _fsync_directory(staging)
        validation = validate_ego_t5_source_manifest(
            staging,
            dataset_identity=identity,
            data_root=data_root,
            inspect_tensors=True,
        )
        _rename_noreplace(staging, output_dir)
        _fsync_directory(output_dir.parent)
        return {
            "status": "generation_claim_validated_not_independently_replayed",
            "dataset_id": identity["dataset_id"],
            "output_dir_record_only": str(output_dir),
            "episode_count": validation["episode_count"],
            "unique_prompt_count": validation["unique_prompt_count"],
            "manifest_sha256": validation["manifest_sha256"],
            "encoder_provenance_sha256": validation["encoder_provenance_sha256"],
        }
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def plan(args: argparse.Namespace) -> dict:
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))
    identity = derive_ego_dataset_identity(data_root)
    prompts = load_episode_prompts(
        data_root, expected_episode_lengths=identity["episode_lengths"]
    )
    prompt_counts = {}
    for row in prompts:
        prompt_counts[row["prompt"]] = prompt_counts.get(row["prompt"], 0) + 1
    return {
        "status": "generation_plan_only",
        "dataset_id": identity["dataset_id"],
        "episode_metadata_sha256": identity["episode_metadata_sha256"],
        "episode_count": len(prompts),
        "unique_prompt_count": len(prompt_counts),
        "prompt_counts": prompt_counts,
        "model_repo_id_required": EGO_T5_MODEL_REPO_ID,
        "text_len": EGO_T5_TEXT_LEN,
        "loader_max_prompt_len": EGO_T5_LOADER_MAX_PROMPT_LEN,
        "output_contract": "finite float32 [1..64,4096] direct tensor",
        "existing_files_are_not_attested": True,
    }


def runtime_check(args: argparse.Namespace) -> dict:
    _require_secure_cli_runtime()
    if args.device == "cpu":
        device_type = "cpu"
    elif args.device == "cuda" or (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        device_type = "cuda"
    else:
        raise EgoT5SourceError("device must be exactly cpu, cuda, or cuda:<index>")
    versions = _runtime_precheck(device_type)
    receipt = _git_generation_receipt(device_type, versions)
    _assert_generation_module_origins()
    compute_module = _load_verified_compute_module(receipt)
    encoder_class = compute_module._load_wan_t5_encoder_class(
        args.wan_code_path,
        code_git_commit=receipt["git_commit"],
        code_manifest=receipt["code_manifest"],
    )
    return {
        "status": "runtime_import_ready",
        "device_type": device_type,
        "runtime_versions": versions,
        "encoder_class": encoder_class.__name__,
        "model_loaded": False,
        "accelerator_forward_executed": False,
    }


def validate(args: argparse.Namespace) -> dict:
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))
    identity = derive_ego_dataset_identity(data_root)
    validation = validate_ego_t5_source_manifest(
        args.t5_embedding_dir,
        dataset_identity=identity,
        data_root=data_root,
        inspect_tensors=True,
    )
    if any((args.model_root, args.wan_code_path, args.model_revision)):
        if not all((args.model_root, args.wan_code_path, args.model_revision)):
            raise EgoT5SourceError(
                "asset verification requires model-root, wan-code-path, and "
                "model-revision together"
            )
        device_type = validation["manifest"]["generator"]["device_type"]
        actual_encoder = build_encoder_provenance(
            model_root=Path(os.path.abspath(os.path.expanduser(args.model_root))),
            wan_code_path=Path(os.path.abspath(os.path.expanduser(args.wan_code_path))),
            model_revision=args.model_revision,
            checkpoint_hf_etag=validation["manifest"]["encoder"]["checkpoint"][
                "hf_etag"
            ],
            device_type=device_type,
            expected_checkpoint_sha256=validation["manifest"]["encoder"][
                "checkpoint"
            ]["sha256"],
        )
        if actual_encoder != validation["manifest"]["encoder"]:
            raise EgoT5SourceError("current model/tokenizer/code assets differ from manifest")
    return {
        "status": "validated",
        "dataset_id": identity["dataset_id"],
        "episode_count": validation["episode_count"],
        "unique_prompt_count": validation["unique_prompt_count"],
        "manifest_sha256": validation["manifest_sha256"],
        "encoder_provenance_sha256": validation["encoder_provenance_sha256"],
        "prompt_manifest_sha256": validation["prompt_manifest_sha256"],
        "model_assets_rehashed": bool(args.model_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser(
        "plan", help="audit prompts without loading a model"
    )
    plan_parser.add_argument("--data-root", required=True)

    runtime_parser = subparsers.add_parser(
        "runtime-check", help="check encoder imports without loading model weights"
    )
    runtime_parser.add_argument("--wan-code-path", required=True)
    runtime_parser.add_argument("--device", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="encode from a pinned model into a new directory"
    )
    generate_parser.add_argument("--data-root", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--model-root", required=True)
    generate_parser.add_argument("--wan-code-path", required=True)
    generate_parser.add_argument("--model-revision", required=True)
    generate_parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="external reviewer-supplied SHA-256 for the checkpoint bytes",
    )
    generate_parser.add_argument(
        "--checkpoint-hf-etag",
        required=True,
        help="external reviewer-supplied 64-hex Hugging Face etag",
    )
    generate_parser.add_argument("--device", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate manifest, prompts, and all tensor bytes"
    )
    validate_parser.add_argument("--data-root", required=True)
    validate_parser.add_argument("--t5-embedding-dir", required=True)
    validate_parser.add_argument("--model-root")
    validate_parser.add_argument("--wan-code-path")
    validate_parser.add_argument("--model-revision")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "plan":
            result = plan(args)
        elif args.command == "runtime-check":
            result = runtime_check(args)
        elif args.command == "generate":
            result = generate(args)
        else:
            result = validate(args)
    except (OSError, ValueError, EgoT5SourceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(_json_dump(result))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independently replay every unique Ego Stage1 T5 prompt.

This command must run as a separate process after generation.  It never reads
or writes a training gate and cannot retrofit evidence into the T5 source.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_secure_cli_imports() -> Path:
    """Isolate project imports from timestamp-based bytecode caches."""

    if not sys.dont_write_bytecode:
        raise RuntimeError("secure Ego T5 replay requires python -B")
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
import world_action_model.ego_t5_replay_attestation as _attestation_module
import world_action_model.ego_t5_source as _t5_source_module
import world_action_model.verified_io as _verified_io_module
from world_action_model.ego_dataset_fingerprint import (
    derive_ego_dataset_identity,
    derive_ego_t5_source_receipt,
)
from world_action_model.ego_t5_replay_attestation import (
    EGO_T5_REPLAY_COMPARISON_MODE,
    EGO_T5_REPLAY_ATTESTATION_VERSION,
    EGO_T5_REPLAY_VERIFIER_IDENTITY,
    canonical_attestation_bytes,
    current_git_commit,
    current_replay_code_manifest,
    expected_replay_results,
    sha256_bytes,
    validate_ego_t5_replay_attestation,
)
from world_action_model.ego_t5_source import (
    EGO_T5_CHECKPOINT_LOGICAL_PATH,
    EGO_T5_TEXT_LEN,
    EgoT5SourceError,
    canonical_sha256,
    tensor_semantic_bytes,
)
from world_action_model.verified_io import (
    assert_python_module_origin,
    load_verified_head_python_module,
    read_verified_bytes,
)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _real_parent(path: Path, name: str) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise EgoT5SourceError(f"{name} parent must be an existing real directory")


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
            "replay is allowed only through the python -B secure CLI launcher"
        )


def _assert_replay_module_origins() -> None:
    try:
        for module, logical_path in (
            (_wam_package, "world_action_model/__init__.py"),
            (
                _fingerprint_module,
                "world_action_model/ego_dataset_fingerprint.py",
            ),
            (
                _attestation_module,
                "world_action_model/ego_t5_replay_attestation.py",
            ),
            (_t5_source_module, "world_action_model/ego_t5_source.py"),
            (_verified_io_module, "world_action_model/verified_io.py"),
        ):
            assert_python_module_origin(module, PROJECT_ROOT / logical_path)
    except ValueError as exc:
        raise EgoT5SourceError("loaded replay module origin differs") from exc


def _load_verified_prepare_module(*, git_commit: str, code_manifest: list[dict]):
    logical_path = "scripts/prepare_ego_t5_source.py"
    matches = [
        row for row in code_manifest if row["logical_path"] == logical_path
    ]
    if len(matches) != 1:
        raise EgoT5SourceError("replay code manifest lacks unique prepare helper")
    try:
        return load_verified_head_python_module(
            PROJECT_ROOT,
            logical_path,
            f"_gwp_verified_prepare_{os.getpid()}_{os.urandom(8).hex()}",
            expected_commit=git_commit,
            expected_row=matches[0],
        )
    except ValueError as exc:
        raise EgoT5SourceError(
            "prepare helper source/origin differs from current HEAD"
        ) from exc


def replay(args: argparse.Namespace) -> dict:
    _require_secure_cli_runtime()
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))
    source_dir = Path(os.path.abspath(os.path.expanduser(args.t5_embedding_dir)))
    replay_output = Path(os.path.abspath(os.path.expanduser(args.replay_output_dir)))
    attestation_output = Path(
        os.path.abspath(os.path.expanduser(args.attestation_output))
    )
    model_root = Path(os.path.abspath(os.path.expanduser(args.model_root)))
    wan_code_path = Path(os.path.abspath(os.path.expanduser(args.wan_code_path)))
    if replay_output.exists() or replay_output.is_symlink():
        raise EgoT5SourceError("independent replay output must not already exist")
    if attestation_output.exists() or attestation_output.is_symlink():
        raise EgoT5SourceError("independent replay attestation must not already exist")
    _real_parent(replay_output, "replay output")
    _real_parent(attestation_output, "attestation output")
    if (
        replay_output == source_dir
        or source_dir in replay_output.parents
        or attestation_output == source_dir
        or source_dir in attestation_output.parents
    ):
        raise EgoT5SourceError("independent replay outputs must be outside the T5 source")
    if args.device == "cpu":
        device_type = "cpu"
    elif args.device == "cuda" or (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        device_type = "cuda"
    else:
        raise EgoT5SourceError("device must be exactly cpu, cuda, or cuda:<index>")

    verifier_commit = current_git_commit()
    verifier_code = current_replay_code_manifest(
        expected_git_commit=verifier_commit
    )
    _assert_replay_module_origins()
    prepare_module = _load_verified_prepare_module(
        git_commit=verifier_commit,
        code_manifest=verifier_code,
    )
    if current_replay_code_manifest(
        expected_git_commit=verifier_commit
    ) != verifier_code:
        raise EgoT5SourceError("replay verifier code changed during prepare import")
    prepare_module._require_current_wan_code_path(wan_code_path)
    runtime_versions = prepare_module._runtime_precheck(device_type)
    identity = derive_ego_dataset_identity(data_root)
    t5_source = derive_ego_t5_source_receipt(
        source_dir,
        episode_indices=list(identity["episode_lengths"]),
        dataset_identity=identity,
        data_root=data_root,
    )
    generation_manifest = t5_source["provenance_manifest"]
    actual_encoder, runtime_receipts = prepare_module.build_encoder_provenance(
        model_root=model_root,
        wan_code_path=wan_code_path,
        model_revision=args.model_revision,
        checkpoint_hf_etag=args.checkpoint_hf_etag,
        device_type=device_type,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        include_runtime_receipts=True,
    )
    if actual_encoder != generation_manifest["encoder"]:
        raise EgoT5SourceError(
            "independent replay assets differ from the generation claim"
        )

    staging = replay_output.parent / (
        f".{replay_output.name}.staging-{os.getpid()}-{os.urandom(16).hex()}"
    )
    os.mkdir(staging, mode=0o700)
    tokenizer_snapshot = staging / ".tokenizer-snapshot"
    try:
        before_code = current_replay_code_manifest(
            expected_git_commit=verifier_commit
        )
        if before_code != verifier_code:
            raise EgoT5SourceError("replay verifier code changed before encoder load")
        compute_module = prepare_module._load_verified_compute_module(
            {
                "git_commit": verifier_commit,
                "code_manifest": verifier_code,
            }
        )
        _encode_t5 = compute_module._encode_t5
        _init_wan_t5_encoder = compute_module._init_wan_t5_encoder
        if current_replay_code_manifest(
            expected_git_commit=verifier_commit
        ) != verifier_code:
            raise EgoT5SourceError("replay verifier code changed during compute import")
        snapshot_receipts = prepare_module._create_tokenizer_snapshot(
            model_root=model_root,
            snapshot_dir=tokenizer_snapshot,
            tokenizer_manifest=actual_encoder["tokenizer_manifest"],
            runtime_receipts=runtime_receipts,
        )
        encoder = _init_wan_t5_encoder(
            wan_path=str(model_root),
            device=args.device,
            text_len=EGO_T5_TEXT_LEN,
            wan_code_path=str(wan_code_path),
            checkpoint_sha256=actual_encoder["checkpoint"]["sha256"],
            checkpoint_stat=runtime_receipts[EGO_T5_CHECKPOINT_LOGICAL_PATH],
            tokenizer_path_override=str(tokenizer_snapshot),
            tokenizer_local_files_only=True,
            code_git_commit=verifier_commit,
            code_manifest=verifier_code,
        )
        if current_replay_code_manifest(
            expected_git_commit=verifier_commit
        ) != verifier_code:
            raise EgoT5SourceError("replay verifier code changed during encoder load")
        prepare_module._verify_tokenizer_snapshot(
            tokenizer_snapshot,
            actual_encoder["tokenizer_manifest"],
            snapshot_receipts,
        )
        shutil.rmtree(tokenizer_snapshot)
        prepare_module._fsync_directory(staging)

        expected = expected_replay_results(generation_manifest)
        episode_by_index = {
            row["episode_index"]: row for row in generation_manifest["episodes"]
        }
        results = []
        for source_result in expected:
            if current_replay_code_manifest(
                expected_git_commit=verifier_commit
            ) != verifier_code:
                raise EgoT5SourceError("replay verifier code changed before encoding")
            replay_tensor = _encode_t5(
                encoder,
                source_result["prompt"],
                device=args.device,
            )
            if current_replay_code_manifest(
                expected_git_commit=verifier_commit
            ) != verifier_code:
                raise EgoT5SourceError("replay verifier code changed during encoding")
            replay_path = staging / (
                f"prompt_{source_result['prompt_sha256']}.pt"
            )
            replay_file, replay_receipt = prepare_module._save_tensor(
                replay_path, replay_tensor
            )
            source_episode = episode_by_index[source_result["episode_indices"][0]]
            source_path = source_dir / source_episode["file"]["logical_path"]
            source_payload, _, _ = read_verified_bytes(
                source_path,
                expected_sha256=source_episode["file"]["sha256"],
            )
            source_receipt, source_semantic_bytes = tensor_semantic_bytes(
                source_payload,
                name=str(source_path),
            )
            replay_payload, _, _ = read_verified_bytes(
                replay_path,
                expected_sha256=replay_file["sha256"],
            )
            saved_replay_receipt, replay_semantic_bytes = tensor_semantic_bytes(
                replay_payload,
                name=str(replay_path),
            )
            exact = (
                source_receipt == source_result["source_tensor"]
                and replay_receipt == saved_replay_receipt
                and source_semantic_bytes == replay_semantic_bytes
            )
            result = {
                "prompt": source_result["prompt"],
                "prompt_sha256": source_result["prompt_sha256"],
                "episode_indices": source_result["episode_indices"],
                "source_tensor": source_result["source_tensor"],
                "replay_tensor": replay_receipt,
                "exact_match": exact,
            }
            if not exact:
                raise EgoT5SourceError(
                    "independent replay tensor differs from generation source"
                )
            results.append(result)
        if len(results) != generation_manifest["unique_prompt_count"]:
            raise EgoT5SourceError("independent replay did not cover every unique prompt")
        if current_replay_code_manifest(
            expected_git_commit=verifier_commit
        ) != verifier_code:
            raise EgoT5SourceError("replay verifier code changed during encoding")

        encoder_manifest = generation_manifest["encoder"]
        receipt = t5_source["source_receipt"]
        attestation = {
            "attestation_version": EGO_T5_REPLAY_ATTESTATION_VERSION,
            "status": "passed",
            "independent_replay": True,
            "dataset_binding": {
                "dataset_id": identity["dataset_id"],
                "episode_metadata_sha256": identity["episode_metadata_sha256"],
            },
            "source_binding": {
                "artifact_manifest_sha256": receipt[
                    "provenance_manifest_sha256"
                ],
                "t5_source_manifest_sha256": t5_source[
                    "source_manifest_sha256"
                ],
                "t5_source_receipt_sha256": t5_source[
                    "source_receipt_sha256"
                ],
                "encoder_provenance_sha256": receipt[
                    "encoder_provenance_sha256"
                ],
                "prompt_manifest_sha256": receipt["prompt_manifest_sha256"],
                "generation_code_manifest_sha256": generation_manifest[
                    "generator"
                ]["code_manifest_sha256"],
            },
            "assets": {
                "model_revision": encoder_manifest["model_revision"],
                "checkpoint_sha256": encoder_manifest["checkpoint"]["sha256"],
                "checkpoint_hf_etag": encoder_manifest["checkpoint"]["hf_etag"],
                "tokenizer_manifest_sha256": encoder_manifest[
                    "tokenizer_manifest_sha256"
                ],
                "encoder_code_manifest_sha256": encoder_manifest[
                    "encoder_code_manifest_sha256"
                ],
            },
            "verifier": {
                "identity": EGO_T5_REPLAY_VERIFIER_IDENTITY,
                "git_commit": verifier_commit,
                "code_manifest": verifier_code,
                "code_manifest_sha256": canonical_sha256(verifier_code),
                "runtime_versions": runtime_versions,
                "device_type": device_type,
                "process_isolation": (
                    "separate_cli_process_no_gate_or_source_write_access"
                ),
                "threat_model_boundary": (
                    "ordinary pathname races fail closed; same-UID process-memory or "
                    "forged-metadata attacks are outside this verifier boundary"
                ),
            },
            "replay_contract": {
                "comparison_mode": EGO_T5_REPLAY_COMPARISON_MODE,
                "all_unique_prompts": True,
                "episode_count": generation_manifest["episode_count"],
                "unique_prompt_count": generation_manifest[
                    "unique_prompt_count"
                ],
                "replay_count": len(results),
                "distinct_output_directory": True,
            },
            "results": results,
            "results_manifest_sha256": canonical_sha256(results),
        }
        validate_ego_t5_replay_attestation(
            attestation,
            dataset_identity=identity,
            t5_source=t5_source,
            verify_current_code=True,
        )
        prepare_module._fsync_directory(staging)
        prepare_module._rename_noreplace(staging, replay_output)
        prepare_module._fsync_directory(replay_output.parent)

        payload = canonical_attestation_bytes(attestation)
        attestation_staging = attestation_output.parent / (
            f".{attestation_output.name}.staging-{os.getpid()}-"
            f"{os.urandom(16).hex()}"
        )
        try:
            prepare_module._write_exclusive_file(attestation_staging, payload)
            prepare_module._rename_noreplace(attestation_staging, attestation_output)
            prepare_module._fsync_directory(attestation_output.parent)
        finally:
            if attestation_staging.exists() and attestation_staging.is_file():
                attestation_staging.unlink()
        return {
            "status": "independent_replay_attestation_published",
            "dataset_id": identity["dataset_id"],
            "replay_output_dir_record_only": str(replay_output),
            "attestation_path_record_only": str(attestation_output),
            "attestation_sha256": sha256_bytes(payload),
            "unique_prompt_count": len(results),
            "all_unique_prompts": True,
            "gate_updated": False,
        }
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--t5-embedding-dir", required=True)
    parser.add_argument("--replay-output-dir", required=True)
    parser.add_argument("--attestation-output", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--wan-code-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-hf-etag", required=True)
    parser.add_argument("--device", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = replay(args)
    except (OSError, ValueError, EgoT5SourceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(_json(result))


if __name__ == "__main__":
    main()

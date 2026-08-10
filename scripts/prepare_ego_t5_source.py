#!/usr/bin/env python3
"""Plan, generate, or validate a provenance-bound Ego Stage1 T5 source.

Generation always starts from a nonexistent output directory and a pinned WAN
UMT5 checkpoint.  Existing unproven ``.pt`` files are never retroactively
attested.  Staging stays next to the requested output, never in ``/tmp``.
"""

from __future__ import annotations

import argparse
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_action_model.ego_dataset_fingerprint import derive_ego_dataset_identity
from world_action_model.ego_t5_source import (
    EGO_T5_CHECKPOINT_LOGICAL_PATH,
    EGO_T5_ENCODER_ARCHITECTURE,
    EGO_T5_ENCODER_BACKEND,
    EGO_T5_ENCODER_CODE_FILES,
    EGO_T5_GENERATOR_IDENTITY,
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
from world_action_model.verified_io import hash_verified_regular_file, read_verified_bytes


def _json_dump(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _source_file_row(root: Path, logical_path: str) -> dict:
    path = root / logical_path
    digest, stat = hash_verified_regular_file(path)
    return {
        "logical_path": logical_path,
        "sha256": digest,
        "size_bytes": stat["size_bytes"],
    }


def _checkpoint_hf_metadata(model_root: Path, model_revision: str) -> str:
    metadata_path = (
        model_root
        / ".cache"
        / "huggingface"
        / "download"
        / f"{EGO_T5_CHECKPOINT_LOGICAL_PATH}.metadata"
    )
    try:
        payload, _, _ = read_verified_bytes(metadata_path)
        fields = payload.decode("utf-8").split()
    except (OSError, UnicodeError, ValueError) as exc:
        raise EgoT5SourceError(
            "checkpoint requires Hugging Face local-dir revision metadata"
        ) from exc
    if len(fields) < 2 or fields[0] != model_revision:
        raise EgoT5SourceError("checkpoint HF metadata revision differs")
    etag = fields[1]
    if len(etag) != 64 or any(char not in "0123456789abcdef" for char in etag):
        raise EgoT5SourceError("checkpoint HF metadata etag is not SHA-256 shaped")
    return etag


def build_encoder_provenance(
    *,
    model_root: Path,
    wan_code_path: Path,
    model_revision: str,
    device_type: str,
) -> dict:
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
    checkpoint = _source_file_row(model_root, EGO_T5_CHECKPOINT_LOGICAL_PATH)
    checkpoint["hf_etag"] = _checkpoint_hf_metadata(model_root, model_revision)
    tokenizer_manifest = [
        _source_file_row(model_root, path) for path in EGO_T5_TOKENIZER_FILES
    ]
    code_manifest = [
        _source_file_row(wan_code_path, path) for path in EGO_T5_ENCODER_CODE_FILES
    ]
    compute_dtype = "float32" if device_type == "cpu" else "bfloat16"
    return {
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


def _runtime_precheck(device_type: str) -> dict:
    try:
        import_module("ftfy")
        import_module("regex")
        torch = import_module("torch")
        transformers = import_module("transformers")
        ftfy_version = metadata.version("ftfy")
        regex_version = metadata.version("regex")
    except ImportError as exc:
        raise EgoT5SourceError(
            "generation environment requires torch, transformers, ftfy, and regex"
        ) from exc
    if device_type == "cuda" and not torch.cuda.is_available():
        raise EgoT5SourceError(
            "requested CUDA generation but the project runtime reports no device"
        )
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "ftfy": ftfy_version,
        "regex": regex_version,
    }


def _git_generation_receipt(device_type: str, runtime_versions: dict) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EgoT5SourceError("generator must run from a Git checkout") from exc
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise EgoT5SourceError("generator Git commit is not full length")
    generator_path = PROJECT_ROOT / EGO_T5_GENERATOR_IDENTITY
    source_sha256, _ = hash_verified_regular_file(generator_path)
    relevant = [
        EGO_T5_GENERATOR_IDENTITY,
        "scripts/compute_t5_embedding.py",
        "world_action_model/ego_t5_source.py",
        *[f"third_party/{path}" for path in EGO_T5_ENCODER_CODE_FILES],
    ]
    tracked = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", *relevant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        raise EgoT5SourceError("generator/encoder implementation is not fully tracked")
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "diff", "--quiet", "HEAD", "--", *relevant],
        check=False,
    )
    if result.returncode != 0:
        raise EgoT5SourceError("generator/encoder implementation has uncommitted changes")
    return {
        "identity": EGO_T5_GENERATOR_IDENTITY,
        "git_commit": commit,
        "source_sha256": source_sha256,
        "device_type": device_type,
        "runtime_versions": runtime_versions,
        "one_encode_per_unique_prompt": True,
        "episode_file_pattern": "episode_{episode_index:06d}.pt",
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
    torch.save(tensor, path)
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
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))
    output_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    model_root = Path(os.path.abspath(os.path.expanduser(args.model_root)))
    wan_code_path = Path(os.path.abspath(os.path.expanduser(args.wan_code_path)))
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
    identity = derive_ego_dataset_identity(data_root)
    prompts = load_episode_prompts(
        data_root, expected_episode_lengths=identity["episode_lengths"]
    )
    encoder_provenance = build_encoder_provenance(
        model_root=model_root,
        wan_code_path=wan_code_path,
        model_revision=args.model_revision,
        device_type=device_type,
    )
    staging = output_dir.parent / (
        f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    try:
        from scripts.compute_t5_embedding import _encode_t5, _init_wan_t5_encoder

        encoder = _init_wan_t5_encoder(
            wan_path=str(model_root),
            device=args.device,
            text_len=EGO_T5_TEXT_LEN,
            wan_code_path=str(wan_code_path),
        )
        encoded = {}
        episode_rows = []
        prompt_to_semantic = {}
        semantic_to_prompt = {}
        for row in prompts:
            prompt = row["prompt"]
            if prompt not in encoded:
                encoded[prompt] = _encode_t5(encoder, prompt, device=args.device)
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
            "status": "generated_from_pinned_model",
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
        manifest_path.write_text(_json_dump(manifest) + "\n", encoding="utf-8")
        validation = validate_ego_t5_source_manifest(
            staging,
            dataset_identity=identity,
            data_root=data_root,
            inspect_tensors=True,
        )
        os.replace(staging, output_dir)
        return {
            "status": "generated_and_validated",
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
    if args.device == "cpu":
        device_type = "cpu"
    elif args.device == "cuda" or (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        device_type = "cuda"
    else:
        raise EgoT5SourceError("device must be exactly cpu, cuda, or cuda:<index>")
    versions = _runtime_precheck(device_type)
    from scripts.compute_t5_embedding import _load_wan_t5_encoder_class

    encoder_class = _load_wan_t5_encoder_class(args.wan_code_path)
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
            device_type=device_type,
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

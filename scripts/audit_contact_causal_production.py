#!/usr/bin/env python3
"""Run the CPU-only op42 causal-main acceptance suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root", type=Path, default=REPOSITORY_ROOT
    )
    parser.add_argument("--loader-runtime-only", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.expanduser().absolute()
    if args.loader_runtime_only:
        stubs = {
            "giga_datasets": repository_root / "third_party/giga-datasets/giga_datasets",
            "giga_datasets.datasets": repository_root
            / "third_party/giga-datasets/giga_datasets/datasets",
            "world_action_model.datasets": repository_root
            / "world_action_model/datasets",
        }
        for name, source in stubs.items():
            module = types.ModuleType(name)
            module.__path__ = [str(source)]
            sys.modules[name] = module
        from world_action_model.contact_causal_coverage import (
            real_loader_synthetic_instantiation_helper,
        )
        from world_action_model.contact_causal_matched_control import (
            collector_cpu_recording_acceptance_helper,
        )

        loader = real_loader_synthetic_instantiation_helper(repository_root)
        collector = collector_cpu_recording_acceptance_helper(repository_root)
        target_module = sys.modules.get(
            "world_action_model.datasets.wam_lerobot_dataset"
        )
        target_path = Path(getattr(target_module, "__file__", "")).absolute()
        expected_path = repository_root / "world_action_model/datasets/wam_lerobot_dataset.py"
        if target_path != expected_path:
            raise RuntimeError("loader runtime harness did not import the real target module")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "runtime_environment": "labvla",
                    "stub_scope": {
                        name: "package_init_only; actual target/core submodules loaded from path"
                        for name in stubs
                    },
                    "actual_loader_module": str(target_path),
                    "loader_constructor_and_sidecar": loader,
                    "collector_record_sha256": collector["record_sha256"],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    from world_action_model.contact_causal_cpu_acceptance import run_cpu_acceptance

    report = run_cpu_acceptance(args.repository_root)
    loader_process = subprocess.run(
        [
            "conda", "run", "-n", "labvla", "python", "-B",
            str(Path(__file__).absolute()), "--repository-root",
            str(repository_root), "--loader-runtime-only",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    loader_runtime = json.loads(loader_process.stdout)
    if loader_runtime.get("status") != "passed":
        raise RuntimeError("real loader/collector CPU runtime acceptance did not pass")
    report["real_loader_runtime_acceptance"] = loader_runtime
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

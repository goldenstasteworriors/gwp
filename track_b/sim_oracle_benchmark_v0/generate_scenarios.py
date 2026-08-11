#!/usr/bin/env python3
"""Generate a deterministic, engine-independent pouring benchmark manifest.

This script does not import or launch a simulator.  Its output remains
``spec_only_no_go`` until an independently reviewed engine adapter supplies an
execution binding that passes ``validate_sim_oracle_manifest.py --mode execution``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "benchmark_spec.json"
MANIFEST_VERSION = "wam-sim-oracle-scenario-manifest/v1"
GT_CONTRACT_ID = "native-state-no-edit/v1"
_STRIDES = (1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _pick(values: list[Any], condition_index: int, factor_index: int, offset: int) -> Any:
    stride = _STRIDES[factor_index % len(_STRIDES)]
    return values[(condition_index * stride + offset + factor_index) % len(values)]


def _base_factors(
    in_domain: dict[str, list[Any]], condition_index: int, split_offset: int
) -> dict[str, Any]:
    return {
        key: _pick(values, condition_index, factor_index, split_offset)
        for factor_index, (key, values) in enumerate(in_domain.items())
    }


def _apply_ood_override(
    factors: dict[str, Any],
    ood_overrides: dict[str, dict[str, list[Any]]],
    condition_index: int,
) -> str:
    schedule = [
        (axis, field, value)
        for axis, fields in ood_overrides.items()
        for field, values in fields.items()
        for value in values
    ]
    axis, field, value = schedule[condition_index % len(schedule)]
    factors[field] = value
    return axis


def _embodiment(role_assignment: str) -> dict[str, Any]:
    if role_assignment == "left_stabilize_right_pour":
        left_role, right_role = "target_stabilizer", "source_transport_and_tilt"
    elif role_assignment == "left_pour_right_stabilize":
        left_role, right_role = "source_transport_and_tilt", "target_stabilizer"
    else:
        raise ValueError(f"unsupported role_assignment {role_assignment!r}")
    return {
        "action_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
        "left": {
            "device": "Inspire Robots RH56DFTP",
            "active_dof": 6,
            "role": left_role,
        },
        "right": {
            "device": "AGIBOT OmniPicker",
            "active_dof": 1,
            "jaw_count": 2,
            "role": right_role,
        },
    }


def build_manifest(spec_path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = _load_json(spec_path)
    if spec.get("schema_version") != "wam-sim-oracle-benchmark-spec/v1":
        raise ValueError("unsupported benchmark spec version")
    if spec.get("status") != "spec_only_no_go":
        raise ValueError("generator only accepts a fail-closed spec_only_no_go source spec")

    factor_space = spec["factor_space"]
    in_domain = factor_space["in_domain"]
    ood_overrides = factor_space["ood_overrides"]
    design = spec["design"]
    generator_seed = int(design["generator_seed"])
    scenarios: list[dict[str, Any]] = []
    split_offsets = {"sanity": 0, "id_test": 101, "ood_test": 211}

    for split, split_design in design["splits"].items():
        conditions = int(split_design["conditions"])
        replicates = int(split_design["replicates_per_condition"])
        slug = split.replace("_", "-")
        for condition_index in range(conditions):
            factors = _base_factors(
                in_domain, condition_index, split_offsets[split]
            )
            ood_axis = None
            if split == "ood_test":
                ood_axis = _apply_ood_override(
                    factors, ood_overrides, condition_index
                )
            condition_id = f"simb-{slug}-c{condition_index:03d}"
            for replicate_index in range(replicates):
                seed_base = (
                    generator_seed
                    + split_offsets[split] * 100_000
                    + condition_index * 100
                    + replicate_index
                )
                scenarios.append(
                    {
                        "scenario_id": f"{condition_id}-r{replicate_index:02d}",
                        "condition_id": condition_id,
                        "split": split,
                        "replicate_index": replicate_index,
                        "seeds": {
                            "physics": seed_base + 1,
                            "render": seed_base + 2,
                            "policy": seed_base + 3,
                        },
                        "factors": factors.copy(),
                        "ood_axis": ood_axis,
                        "embodiment": _embodiment(factors["role_assignment"]),
                        "ground_truth_contract_id": GT_CONTRACT_ID,
                    }
                )

    variants = [variant["id"] for variant in spec["benchmark_variants"]]
    return {
        "schema_version": MANIFEST_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_spec_sha256": _sha256_file(spec_path),
        "generation": {
            "generator": Path(__file__).name,
            "generator_sha256": _sha256_file(Path(__file__).resolve()),
            "generator_seed": generator_seed,
            "method": design["method"],
        },
        "status": "spec_only_no_go",
        "variant_ids": variants,
        "execution_binding": None,
        "scenarios": scenarios,
    }


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    condition_counts: dict[str, set[str]] = {}
    for scenario in manifest["scenarios"]:
        split = scenario["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        condition_counts.setdefault(split, set()).add(scenario["condition_id"])
    return {
        "benchmark_id": manifest["benchmark_id"],
        "status": manifest["status"],
        "records_by_split": split_counts,
        "conditions_by_split": {
            key: len(value) for key, value in condition_counts.items()
        },
        "variant_count": len(manifest["variant_ids"]),
        "physical_rollout_slots_if_complete": len(manifest["scenarios"])
        * len(manifest["variant_ids"]),
        "contains_simulated_results": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Build in memory and print design counts without writing a manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing explicit --output path.",
    )
    args = parser.parse_args()
    if (args.output is None) == (not args.summary_only):
        parser.error("choose exactly one of --output or --summary-only")

    manifest = build_manifest(args.spec)
    if args.summary_only:
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0

    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**_summary(manifest), "output": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

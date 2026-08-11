#!/usr/bin/env python3
"""Fail-closed validator for simulation scenario and execution manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "benchmark_spec.json"
DEFAULT_READINESS = HERE / "local_readiness_20260811.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _json_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _flatten_ood_values(spec: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for axis, fields in spec["factor_space"]["ood_overrides"].items():
        for field, values in fields.items():
            result[field].update(_json_signature(value) for value in values)
    return result


def _validate_scenario_design(
    manifest: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    require(
        manifest.get("schema_version") == "wam-sim-oracle-scenario-manifest/v1",
        "unsupported scenario manifest version",
    )
    require(manifest.get("benchmark_id") == spec.get("benchmark_id"), "benchmark_id mismatch")
    require(
        manifest.get("benchmark_spec_sha256") == sha256_file(_SPEC_PATH),
        "benchmark_spec_sha256 mismatch",
    )
    expected_variants = [item["id"] for item in spec["benchmark_variants"]]
    require(manifest.get("variant_ids") == expected_variants, "variant_ids must exactly match spec order")
    require(manifest.get("status") in {"spec_only_no_go", "execution_ready"}, "invalid status")

    generation = manifest.get("generation")
    require(isinstance(generation, dict), "generation must be an object")
    require(
        generation.get("generator_seed") == spec["design"]["generator_seed"],
        "generator_seed mismatch",
    )
    generator_hash = generation.get("generator_sha256")
    require(isinstance(generator_hash, str) and SHA256_RE.fullmatch(generator_hash) is not None, "invalid generator_sha256")
    generator_name = generation.get("generator")
    require(
        isinstance(generator_name, str)
        and generator_name
        and Path(generator_name).name == generator_name,
        "generation.generator must be a local filename",
    )
    generator_path = _MANIFEST_PATH.parent / generator_name
    if not generator_path.is_file():
        generator_path = _SPEC_PATH.parent / generator_name
    require(generator_path.is_file(), f"generator file is missing: {generator_path}")
    require(
        sha256_file(generator_path) == generator_hash,
        "generator_sha256 does not match the generator beside the manifest",
    )

    scenarios = manifest.get("scenarios")
    require(isinstance(scenarios, list) and scenarios, "scenarios must be a non-empty array")
    ids: set[str] = set()
    seed_tuples: set[tuple[int, int, int]] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    in_domain = spec["factor_space"]["in_domain"]
    in_keys = set(in_domain)
    all_ood_values = _flatten_ood_values(spec)
    ood_overrides = spec["factor_space"]["ood_overrides"]

    for index, scenario in enumerate(scenarios):
        path = f"scenarios[{index}]"
        require(isinstance(scenario, dict), f"{path} must be an object")
        scenario_id = scenario.get("scenario_id")
        require(isinstance(scenario_id, str) and scenario_id not in ids, f"{path}.scenario_id invalid or duplicate")
        ids.add(scenario_id)
        split = scenario.get("split")
        require(split in spec["design"]["splits"], f"{path}.split invalid")
        counts[split] += 1
        condition_id = scenario.get("condition_id")
        require(isinstance(condition_id, str) and scenario_id.startswith(condition_id + "-r"), f"{path}.condition_id mismatch")
        replicate = scenario.get("replicate_index")
        require(isinstance(replicate, int) and replicate >= 0, f"{path}.replicate_index invalid")
        groups[(split, condition_id)].append(scenario)

        seeds = scenario.get("seeds")
        require(isinstance(seeds, dict) and set(seeds) == {"physics", "render", "policy"}, f"{path}.seeds invalid")
        seed_tuple = tuple(seeds[name] for name in ("physics", "render", "policy"))
        require(all(isinstance(value, int) and value >= 0 for value in seed_tuple), f"{path}.seeds must be non-negative integers")
        require(seed_tuple not in seed_tuples, f"{path}.seeds duplicate another scenario")
        seed_tuples.add(seed_tuple)

        factors = scenario.get("factors")
        require(isinstance(factors, dict) and set(factors) == in_keys, f"{path}.factors keys mismatch")
        ood_axis = scenario.get("ood_axis")
        if split != "ood_test":
            require(ood_axis is None, f"{path}.ood_axis must be null outside ood_test")
            for field, value in factors.items():
                require(value in in_domain[field], f"{path}.factors.{field} is not in-domain")
                require(_json_signature(value) not in all_ood_values.get(field, set()), f"{path}.factors.{field} leaks an OOD value")
        else:
            require(ood_axis in ood_overrides, f"{path}.ood_axis invalid")
            primary_fields = ood_overrides[ood_axis]
            has_primary_ood = any(
                factors[field] in values for field, values in primary_fields.items()
            )
            require(has_primary_ood, f"{path} has no value from its primary OOD axis")
            for field, value in factors.items():
                allowed = list(in_domain[field])
                allowed.extend(
                    candidate
                    for fields in ood_overrides.values()
                    for candidate in fields.get(field, [])
                )
                require(value in allowed, f"{path}.factors.{field} is outside declared factor space")

        require(scenario.get("ground_truth_contract_id") == "native-state-no-edit/v1", f"{path}.ground_truth_contract_id invalid")
        embodiment = scenario.get("embodiment")
        require(isinstance(embodiment, dict), f"{path}.embodiment missing")
        require(embodiment.get("action_schema_id") == spec["embodiment_contract"]["action_schema_id"], f"{path}.embodiment action schema mismatch")
        require(embodiment.get("left", {}).get("device") == "Inspire Robots RH56DFTP", f"{path}.left device mismatch")
        require(embodiment.get("right", {}).get("device") == "AGIBOT OmniPicker", f"{path}.right device mismatch")

    for split, split_design in spec["design"]["splits"].items():
        conditions = int(split_design["conditions"])
        replicates = int(split_design["replicates_per_condition"])
        require(counts[split] == conditions * replicates, f"{split} record count mismatch")
        split_groups = [items for (group_split, _), items in groups.items() if group_split == split]
        require(len(split_groups) == conditions, f"{split} condition count mismatch")
        for items in split_groups:
            require(len(items) == replicates, f"{split} replicate count mismatch")
            require(sorted(item["replicate_index"] for item in items) == list(range(replicates)), f"{split} replicate indices must be contiguous")
            factor_signatures = {_json_signature(item["factors"]) for item in items}
            require(len(factor_signatures) == 1, f"{split} factors changed across replicates")
            require(len({item["ood_axis"] for item in items}) == 1, f"{split} OOD axis changed across replicates")

    id_conditions = {
        scenario["condition_id"]: scenario["factors"]
        for scenario in scenarios
        if scenario["split"] == "id_test"
    }
    for field, values in in_domain.items():
        observed = {_json_signature(factors[field]) for factors in id_conditions.values()}
        expected = {_json_signature(value) for value in values}
        require(expected <= observed, f"id_test does not cover every declared value for {field}")

    ood_conditions = {
        scenario["condition_id"]: scenario
        for scenario in scenarios
        if scenario["split"] == "ood_test"
    }
    for axis, fields in ood_overrides.items():
        axis_records = [
            scenario for scenario in ood_conditions.values() if scenario["ood_axis"] == axis
        ]
        require(axis_records, f"ood_test does not cover axis {axis}")
        for field, values in fields.items():
            observed = {
                _json_signature(scenario["factors"][field]) for scenario in axis_records
            }
            expected = {_json_signature(value) for value in values}
            require(
                expected <= observed,
                f"ood_test axis {axis} does not cover every declared value for {field}",
            )

    return {
        "scenario_count": len(scenarios),
        "condition_count": len(groups),
        "records_by_split": dict(sorted(counts.items())),
        "variant_count": len(expected_variants),
    }


def _validate_execution_binding(
    manifest: dict[str, Any], readiness: dict[str, Any]
) -> dict[str, Any]:
    require(manifest.get("status") == "execution_ready", "execution mode requires status=execution_ready")
    require(readiness.get("overall_status") == "go", "local readiness is not go")
    binding = manifest.get("execution_binding")
    require(isinstance(binding, dict), "execution_binding must be an object")
    required_capabilities = {
        "native_liquid_particles_or_cells",
        "viscosity_control",
        "source_rim_flux",
        "per_vessel_contained_volume",
        "outside_volume",
        "mass_residual",
        "native_contact_points_normals_impulses",
        "native_rgb",
        "native_depth",
        "native_instance_segmentation",
        "same_state_multimodal_capture",
    }
    capabilities = binding.get("capabilities")
    require(isinstance(capabilities, dict), "execution_binding.capabilities missing")
    require(required_capabilities <= set(capabilities), "execution_binding.capabilities incomplete")
    for name in required_capabilities:
        require(capabilities[name] is True, f"execution capability {name} is not proven true")

    engine = binding.get("engine")
    require(isinstance(engine, dict), "execution_binding.engine missing")
    for key in ("id", "version", "adapter_revision", "adapter_path", "adapter_sha256"):
        require(isinstance(engine.get(key), str) and engine[key], f"execution_binding.engine.{key} missing")
    require(SHA256_RE.fullmatch(engine["adapter_sha256"]) is not None, "invalid adapter_sha256")
    adapter_path = Path(engine["adapter_path"]).expanduser().resolve()
    require(adapter_path.is_file(), f"engine adapter does not exist: {adapter_path}")
    require(sha256_file(adapter_path) == engine["adapter_sha256"], "engine adapter sha256 mismatch")

    assets = binding.get("assets")
    require(isinstance(assets, list), "execution_binding.assets must be an array")
    required_roles = {
        "g1_body",
        "left_rh56dftp",
        "right_omnipicker",
        "transparent_source_vessel",
        "opaque_source_vessel",
        "transparent_target_vessel",
        "opaque_target_vessel",
    }
    by_role = {item.get("role"): item for item in assets if isinstance(item, dict)}
    require(required_roles <= set(by_role), "execution_binding.assets missing required roles")
    for role in required_roles:
        item = by_role[role]
        path = Path(item.get("path", "")).expanduser().resolve()
        digest = item.get("sha256")
        require(path.is_file(), f"asset {role} does not exist: {path}")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"asset {role} sha256 invalid")
        require(sha256_file(path) == digest, f"asset {role} sha256 mismatch")
        receipt_path = Path(item.get("calibration_receipt_path", "")).expanduser().resolve()
        receipt_hash = item.get("calibration_receipt_sha256")
        require(receipt_path.is_file(), f"asset {role} calibration receipt is missing")
        require(isinstance(receipt_hash, str) and SHA256_RE.fullmatch(receipt_hash) is not None, f"asset {role} calibration receipt sha256 invalid")
        require(sha256_file(receipt_path) == receipt_hash, f"asset {role} calibration receipt sha256 mismatch")

    clocks = binding.get("clocks")
    require(isinstance(clocks, dict), "execution_binding.clocks missing")
    skew = clocks.get("measured_max_camera_state_skew_s")
    require(isinstance(skew, (int, float)) and math.isfinite(skew) and 0 <= skew <= 0.005, "camera-state skew exceeds 0.005 s")
    matched = binding.get("matched_controls")
    require(isinstance(matched, dict), "execution_binding.matched_controls missing")
    require(
        matched.get("fingerprint_scope")
        == "scenario_factors_plus_physics_render_policy_seeds_and_policy_stochastic_schedule_excluding_variant_inputs",
        "matched-control fingerprint scope mismatch",
    )
    require(matched.get("all_paired_controls_passed") is True, "matched controls did not pass")
    report_path = Path(matched.get("report_path", "")).expanduser().resolve()
    report_hash = matched.get("report_sha256")
    require(report_path.is_file(), "matched-control report is missing")
    require(isinstance(report_hash, str) and SHA256_RE.fullmatch(report_hash) is not None, "matched-control report sha256 invalid")
    require(sha256_file(report_path) == report_hash, "matched-control report sha256 mismatch")
    approval = binding.get("approval")
    require(isinstance(approval, dict), "execution_binding.approval missing")
    require(isinstance(approval.get("reviewed_by"), str) and approval["reviewed_by"].strip(), "execution review identity missing")
    require(approval.get("engine_physics_verified") is True, "engine physics verification is false")
    require(approval.get("transparent_render_verified") is True, "transparent render verification is false")
    require(approval.get("mass_conservation_verified") is True, "mass conservation verification is false")

    receipt = readiness.get("execution_receipt")
    require(isinstance(receipt, dict), "go readiness requires execution_receipt")
    require(
        receipt.get("benchmark_spec_sha256") == manifest.get("benchmark_spec_sha256"),
        "readiness execution_receipt is bound to a different benchmark spec",
    )
    require(receipt.get("engine_id") == engine["id"], "readiness engine_id mismatch")
    require(receipt.get("engine_version") == engine["version"], "readiness engine_version mismatch")
    reviewers = receipt.get("reviewers")
    require(
        isinstance(reviewers, list)
        and len(reviewers) >= 2
        and len(set(reviewers)) == len(reviewers)
        and all(isinstance(item, str) and item.strip() for item in reviewers),
        "readiness requires at least two distinct reviewers",
    )
    evidence_path = Path(receipt.get("evidence_bundle_path", "")).expanduser().resolve()
    evidence_hash = receipt.get("evidence_bundle_sha256")
    require(evidence_path.is_file(), "readiness evidence bundle is missing")
    require(isinstance(evidence_hash, str) and SHA256_RE.fullmatch(evidence_hash) is not None, "readiness evidence bundle sha256 invalid")
    require(sha256_file(evidence_path) == evidence_hash, "readiness evidence bundle sha256 mismatch")
    capability_tests = receipt.get("capability_tests")
    require(isinstance(capability_tests, dict), "readiness capability_tests missing")
    require(
        required_capabilities <= set(capability_tests)
        and all(capability_tests[name] is True for name in required_capabilities),
        "readiness capability tests are incomplete or false",
    )
    require(receipt.get("approved_for_benchmark") is True, "readiness is not approved_for_benchmark")
    require(isinstance(receipt.get("approved_at_utc"), str) and receipt["approved_at_utc"].endswith("Z"), "readiness approved_at_utc missing")
    return {"execution_binding": "validated", "engine": engine["id"]}


_SPEC_PATH = DEFAULT_SPEC
_MANIFEST_PATH = HERE / "scenario_manifest_spec_only.json"


def validate_manifest(
    manifest_path: Path,
    spec_path: Path,
    readiness_path: Path,
    mode: str,
) -> dict[str, Any]:
    global _SPEC_PATH, _MANIFEST_PATH
    _SPEC_PATH = spec_path.resolve()
    _MANIFEST_PATH = manifest_path.resolve()
    spec = load_object(_SPEC_PATH)
    manifest = load_object(_MANIFEST_PATH)
    report = _validate_scenario_design(manifest, spec)
    if mode == "spec":
        require(manifest.get("status") == "spec_only_no_go", "spec mode requires status=spec_only_no_go")
        require(manifest.get("execution_binding") is None, "spec-only manifest must not contain execution_binding")
        report["mode"] = "spec"
        report["execution_allowed"] = False
    else:
        report.update(_validate_execution_binding(manifest, load_object(readiness_path.resolve())))
        report["mode"] = "execution"
        report["execution_allowed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--mode", choices=("spec", "execution"), default="spec")
    args = parser.parse_args()
    try:
        report = validate_manifest(args.manifest, args.spec, args.readiness, args.mode)
    except (OSError, json.JSONDecodeError, ManifestError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({"valid": True, **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

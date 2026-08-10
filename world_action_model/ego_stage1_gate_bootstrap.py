"""Stdlib-only bootstrap checks for the production Ego Stage1 config.

This module intentionally does not import the dataset, trainer, NumPy, Torch,
or distributed runtime.  Its enabled path reads only the explicitly selected
training-gate JSON and launcher rank environment.
"""

from __future__ import annotations

import os
import re

from world_action_model.verified_io import read_verified_bytes, strict_json_loads


EGO_STAGE1_GATE_VERSION = "ego-discrete-contact-stage1-training-gate/v3"


class EgoStage1BootstrapError(ValueError):
    pass


def _load_gate(path: str) -> dict:
    try:
        payload, _, _ = read_verified_bytes(path)
        payload = strict_json_loads(payload, name="Ego Stage1 training gate")
    except (OSError, UnicodeError, ValueError) as exc:
        raise EgoStage1BootstrapError("invalid Ego Stage1 training-gate JSON") from exc
    if not isinstance(payload, dict):
        raise EgoStage1BootstrapError("Ego Stage1 training gate must be a JSON object")
    return payload


def minimal_matched_control_topology_precheck_from_environment() -> bool:
    """Reject a mismatched launcher before full preflight or config import.

    An absent collector output directory is the disabled switch and returns
    without consulting any other environment variable or opening the gate.
    """

    if not os.environ.get("GWP_MATCHED_CONTROL_OUTPUT_DIR", "").strip():
        return False
    gate_path = os.environ.get("GWP_EGO_STAGE1_TRAINING_GATE", "").strip()
    if not gate_path:
        raise EgoStage1BootstrapError(
            "GWP_EGO_STAGE1_TRAINING_GATE is required for collector topology precheck"
        )
    gate = _load_gate(gate_path)
    if gate.get("gate_version") != EGO_STAGE1_GATE_VERSION:
        raise EgoStage1BootstrapError("training gate version differs during topology precheck")
    expectations = gate.get("matched_control_expectations")
    if not isinstance(expectations, dict):
        raise EgoStage1BootstrapError("training gate lacks matched-control expectations")
    expected_world_size = expectations.get("expected_world_size")
    expected_rank_set = expectations.get("expected_rank_set")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size <= 0
        or not isinstance(expected_rank_set, list)
        or any(
            isinstance(expected_rank, bool) or not isinstance(expected_rank, int)
            for expected_rank in expected_rank_set
        )
        or expected_rank_set != list(range(expected_world_size))
    ):
        raise EgoStage1BootstrapError("invalid external matched-control topology")
    try:
        rank_text = os.environ["RANK"].strip()
        world_size_text = os.environ["WORLD_SIZE"].strip()
    except (KeyError, AttributeError) as exc:
        raise EgoStage1BootstrapError(
            "RANK and WORLD_SIZE must be explicit base-10 integers"
        ) from exc
    if not re.fullmatch(r"[0-9]+", rank_text, flags=re.ASCII) or not re.fullmatch(
        r"[0-9]+", world_size_text, flags=re.ASCII
    ):
        raise EgoStage1BootstrapError(
            "RANK and WORLD_SIZE must be explicit base-10 integers"
        )
    rank = int(rank_text)
    world_size = int(world_size_text)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise EgoStage1BootstrapError(
            f"invalid runtime topology rank={rank}, world_size={world_size}"
        )
    if world_size != expected_world_size or rank not in expected_rank_set:
        raise EgoStage1BootstrapError(
            "actual runtime rank/world differs from the external training gate"
        )
    return True

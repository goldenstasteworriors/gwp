"""Dependency-free production split receipt validation.

The dataset adapter imports this module before it opens a LeRobot dataset so
production episode filtering can fail closed before any sample/window access.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from collections.abc import Mapping, Sequence


PRODUCTION_TRAINING_DATA_RECEIPT_VERSION = (
    "contact-asym25-training-data-receipt/v1"
)
PRODUCTION_ACTION_SCHEMA_ID = "sonic_g1_rh56dftp6_omnipicker1_v1"
PRODUCTION_SPLIT_SEEDS = frozenset({6666, 7777, 8888})
PRODUCTION_EPISODE_KEYS = (
    "train_episode_ids",
    "validation_episode_ids",
    "heldout_episode_ids",
)
PRODUCTION_RECEIPT_HASH_KEYS = (
    "dataset_info_sha256",
    "episode_metadata_sha256",
    "norm_stats_sha256",
    "sidecar_bundle_sha256",
    "split_manifest_sha256",
    "training_gate_sha256",
    "preflight_code_sha256",
)
PRODUCTION_TRAINING_DATA_RECEIPT_KEYS = frozenset(
    {
        "receipt_version",
        "dataset_id",
        "action_schema_id",
        "split_seed",
        *PRODUCTION_EPISODE_KEYS,
        *PRODUCTION_RECEIPT_HASH_KEYS,
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Mapping) -> str:
    """Return the strict canonical JSON hash used by production receipts."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("production training-data receipt is not strict JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_production_episode_filter_receipt(
    receipt: object,
    receipt_sha256: object,
    configured_episode_ids: object,
) -> tuple[dict, str, frozenset[int]]:
    """Validate and bind an exact train-only filter before dataset opening."""

    if not isinstance(receipt, dict) or set(receipt) != set(
        PRODUCTION_TRAINING_DATA_RECEIPT_KEYS
    ):
        raise ValueError("production training-data receipt schema mismatch")
    digest = canonical_json_sha256(receipt)
    if (
        receipt.get("receipt_version")
        != PRODUCTION_TRAINING_DATA_RECEIPT_VERSION
        or not isinstance(receipt_sha256, str)
        or not _SHA256_RE.fullmatch(receipt_sha256)
        or digest != receipt_sha256
    ):
        raise ValueError("production training-data receipt fingerprint differs")
    if (
        not isinstance(receipt.get("dataset_id"), str)
        or not receipt["dataset_id"].strip()
        or receipt.get("action_schema_id") != PRODUCTION_ACTION_SCHEMA_ID
        or isinstance(receipt.get("split_seed"), bool)
        or receipt.get("split_seed") not in PRODUCTION_SPLIT_SEEDS
        or any(
            not isinstance(receipt.get(key), str)
            or not _SHA256_RE.fullmatch(receipt[key])
            for key in PRODUCTION_RECEIPT_HASH_KEYS
        )
    ):
        raise ValueError("production training-data receipt identity/hash fields invalid")

    episode_sets: list[set[int]] = []
    for key in PRODUCTION_EPISODE_KEYS:
        values = receipt.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in values
            )
            or values != sorted(set(values))
        ):
            raise ValueError(f"invalid production receipt {key}")
        episode_sets.append(set(values))
    if any(
        episode_sets[left] & episode_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("production train/val/test episode sets overlap")
    if (
        not isinstance(configured_episode_ids, Sequence)
        or isinstance(configured_episode_ids, (str, bytes))
        or list(configured_episode_ids) != receipt["train_episode_ids"]
    ):
        raise ValueError(
            "production dataset episodes must exactly equal receipt train_episode_ids"
        )
    return dict(receipt), digest, frozenset(receipt["train_episode_ids"])


def require_train_episode(
    episode_index: object, allowed_episode_ids: frozenset[int]
) -> int:
    """Reject any sample emitted outside the receipt-bound train split."""

    if isinstance(episode_index, bool) or type(episode_index).__name__ == "bool_":
        raise ValueError("production dataset emitted an invalid episode_index")
    try:
        value = operator.index(episode_index)
    except TypeError as exc:
        raise ValueError("production dataset emitted an invalid episode_index") from exc
    if value not in allowed_episode_ids:
        raise ValueError(
            "production dataset yielded a val/test episode outside its train filter"
        )
    return value

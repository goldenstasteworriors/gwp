import io
import json
import os
import pickle
import re

import numpy as np
import torch
from giga_datasets import utils
from giga_datasets.datasets.base_dataset import BaseDataset
from giga_datasets.datasets.dataset import ConcatDataset, load_config, register_dataset
from giga_datasets.datasets.lerobot_dataset import FastLeRobotDataset, _LeRobotDatasetMetadata
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.ego_hand_active_sidecar import (
    EGO_HAND_ACTIVE_BUILDER,
    EGO_HAND_ACTIVE_FIELD_ORDER,
    EGO_HAND_ACTIVE_HORIZON,
    EGO_HAND_ACTIVE_SCHEMA_VERSION,
    EGO_HAND_ACTIVE_SEMANTICS,
    load_hand_active_npz,
)
from world_action_model.production_split_contract import (
    require_train_episode,
    validate_production_episode_filter_receipt,
)
from world_action_model.verified_io import read_verified_bytes, strict_json_loads

_SUBTASK_MARKER_RE = re.compile(r"\bsubtask\s*:\s*", flags=re.IGNORECASE)
_T5_LOAD_FROM_CHOICES = ("dir", "pkl", "path")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_NONCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}$")
_VERIFIED_STAT_KEYS = {
    "device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns"
}


def _validate_runtime_receipt_entry(
    entry: object, *, expected_keys: set[str], name: str
) -> dict:
    if not isinstance(entry, dict) or set(entry) != expected_keys:
        raise ValueError(f"invalid {name} receipt schema")
    runtime_stat = entry.get("stat")
    if (
        not isinstance(entry.get("path_record_only"), str)
        or not os.path.isabs(entry["path_record_only"])
        or not isinstance(entry.get("sha256"), str)
        or not _SHA256_RE.fullmatch(entry["sha256"])
        or not isinstance(runtime_stat, dict)
        or set(runtime_stat) != _VERIFIED_STAT_KEYS
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in runtime_stat.values()
        )
    ):
        raise ValueError(f"invalid {name} content/stat receipt")
    return entry


@register_dataset
class WAMLeRobotDataset(BaseDataset):
    """LeRobot dataset adapter for WAM-only sidecar fields."""

    def __init__(
        self,
        data_path: str,
        data_size: int | None = None,
        delta_info: dict[str, int] | None = None,
        delta_frames: dict[str, list[int]] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        t5_load_from: str | None = None,
        t5_cfg: dict | None = None,
        t5_cache_size: int = 256,
        contact_sidecar_dir: str | None = None,
        contact_sidecar_pattern: str = "episode_{episode_index:06d}.npz",
        contact_sidecar_cache_size: int = 8,
        contact_sidecar_strict: bool = False,
        contact_sidecar_require_reviewed_v1_1: bool = False,
        contact_sidecar_expected_meta_sha256: str | None = None,
        contact_sidecar_expected_episode_sha256: dict[str, str] | None = None,
        contact_label_origin: str | None = None,
        contact_human_adjudicated_only: bool = False,
        action_contact_ground_truth: bool = False,
        hand_active_contact_ground_truth: bool = False,
        grip_contact_ground_truth: bool = False,
        other_proxy_contact_ground_truth: bool = False,
        ego_hand_active_sidecar_dir: str | None = None,
        ego_hand_active_sidecar_pattern: str = "episode_{episode_index:06d}.npz",
        ego_hand_active_sidecar_cache_size: int = 8,
        ego_hand_active_sidecar_strict: bool = False,
        ego_hand_active_expected_meta_sha256: str | None = None,
        ego_hand_active_expected_episode_sha256: dict[str, str] | None = None,
        training_source_runtime_receipt: list[dict] | None = None,
        t5_runtime_source_receipt: list[dict] | None = None,
        verified_private_cache_parent: str | None = None,
        verified_training_run_nonce: str | None = None,
        matched_control_identity: bool = False,
        matched_control_dataset_id: str | None = None,
        matched_control_dataset_info_sha256: str | None = None,
        production_episode_filter_receipt: dict | None = None,
        production_episode_filter_receipt_sha256: str | None = None,
        meta_name: str | None = None,
        robotype: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(data_path=data_path)
        self.data_size = data_size
        self.delta_info = delta_info
        self.delta_frames = delta_frames
        self.delta_timestamps = delta_timestamps
        self.robotype = robotype
        self.t5_load_from = str(t5_load_from)
        if self.t5_load_from not in _T5_LOAD_FROM_CHOICES:
            raise ValueError(f"invalid t5_load_from: {self.t5_load_from!r}, must be one of {_T5_LOAD_FROM_CHOICES}")
        if self.t5_load_from == "dir":
            self.t5_embedding_dir = str(t5_cfg.get("t5_embedding_dir", None))
            self.t5_embedding_pattern = str(t5_cfg.get("t5_embedding_pattern", "episode_{episode_index:06d}.pt"))
        elif self.t5_load_from == "pkl":
            self.prompt_t5_lookup_path = str(t5_cfg.get("prompt_t5_lookup_path", None))
            self._prompt_t5_lookup = None
        elif self.t5_load_from == "path":
            self.t5_embedding_path = str(t5_cfg.get("t5_embedding_path", None))
        self.t5_embedding_key = str(t5_cfg.get("t5_embedding_key", "t5_embedding"))
        self.t5_cache_size = int(t5_cache_size) if t5_cache_size is not None else 0
        self._t5_cache = {}
        self._t5_cache_order = []
        self.training_source_runtime_receipt = list(
            training_source_runtime_receipt or []
        )
        self.t5_runtime_source_receipt = list(t5_runtime_source_receipt or [])
        self.verified_private_cache_parent = (
            None
            if verified_private_cache_parent is None
            else os.path.abspath(verified_private_cache_parent)
        )
        self.verified_training_run_nonce = verified_training_run_nonce
        if self.training_source_runtime_receipt and (
            self.verified_private_cache_parent is None
            or not isinstance(self.verified_training_run_nonce, str)
            or not _TRAINING_NONCE_RE.fullmatch(
                self.verified_training_run_nonce
            )
        ):
            raise ValueError(
                "verified LeRobot source requires private-cache parent and run nonce"
            )
        for entry in self.training_source_runtime_receipt:
            _validate_runtime_receipt_entry(
                entry,
                expected_keys={
                    "logical_path", "logical_role", "path_record_only", "sha256", "stat"
                },
                name="LeRobot source",
            )
        seen_t5_paths = set()
        for entry in self.t5_runtime_source_receipt:
            entry = _validate_runtime_receipt_entry(
                entry,
                expected_keys={
                    "logical_path", "logical_role", "episode_index",
                    "path_record_only", "sha256", "stat",
                },
                name="T5 source",
            )
            episode_index = entry.get("episode_index")
            filename = f"episode_{episode_index:06d}.pt" if (
                not isinstance(episode_index, bool) and isinstance(episode_index, int)
                and episode_index >= 0
            ) else None
            expected_path = (
                None if filename is None or self.t5_load_from != "dir"
                else os.path.abspath(os.path.join(self.t5_embedding_dir, filename))
            )
            if (
                filename is None
                or entry.get("logical_role") != "t5_episode_condition"
                or entry.get("logical_path") != f"t5_embedding/{filename}"
                or expected_path != os.path.abspath(entry["path_record_only"])
                or expected_path in seen_t5_paths
            ):
                raise ValueError("invalid/duplicate T5 runtime source receipt")
            seen_t5_paths.add(expected_path)
        self._t5_receipt_by_path = {
            os.path.abspath(str(entry.get("path_record_only"))): entry
            for entry in self.t5_runtime_source_receipt
        }
        self.contact_sidecar_dir = (
            None if contact_sidecar_dir is None else str(contact_sidecar_dir)
        )
        self.contact_sidecar_pattern = str(contact_sidecar_pattern)
        self.contact_sidecar_cache_size = max(0, int(contact_sidecar_cache_size))
        self.contact_sidecar_strict = bool(contact_sidecar_strict)
        self.contact_sidecar_require_reviewed_v1_1 = bool(
            contact_sidecar_require_reviewed_v1_1
        )
        self.contact_sidecar_expected_meta_sha256 = (
            None
            if contact_sidecar_expected_meta_sha256 is None
            else str(contact_sidecar_expected_meta_sha256)
        )
        self.contact_sidecar_expected_episode_sha256 = dict(
            contact_sidecar_expected_episode_sha256 or {}
        )
        self.contact_label_origin = contact_label_origin
        self.contact_human_adjudicated_only = bool(contact_human_adjudicated_only)
        proxy_flags = {
            "action": action_contact_ground_truth,
            "hand_active": hand_active_contact_ground_truth,
            "grip": grip_contact_ground_truth,
            "other": other_proxy_contact_ground_truth,
        }
        if self.contact_human_adjudicated_only and (
            self.contact_label_origin
            not in {"human_adjudicated", "human_adjudicated_robot_native"}
            or any(value is not False for value in proxy_flags.values())
        ):
            raise ValueError(
                "human-adjudicated contact forbids action/hand_active/grip/proxy ground truth"
            )
        self.ego_hand_active_sidecar_dir = (
            None
            if ego_hand_active_sidecar_dir is None
            else str(ego_hand_active_sidecar_dir)
        )
        self.ego_hand_active_sidecar_pattern = str(ego_hand_active_sidecar_pattern)
        self.ego_hand_active_sidecar_cache_size = max(
            0, int(ego_hand_active_sidecar_cache_size)
        )
        self.ego_hand_active_sidecar_strict = bool(ego_hand_active_sidecar_strict)
        self.ego_hand_active_expected_meta_sha256 = (
            None
            if ego_hand_active_expected_meta_sha256 is None
            else str(ego_hand_active_expected_meta_sha256)
        )
        self.ego_hand_active_expected_episode_sha256 = dict(
            ego_hand_active_expected_episode_sha256 or {}
        )
        self.matched_control_identity = bool(matched_control_identity)
        self.matched_control_dataset_id = (
            None
            if matched_control_dataset_id is None
            else str(matched_control_dataset_id).strip()
        )
        if not self.matched_control_dataset_id:
            self.matched_control_dataset_id = None
        self.matched_control_dataset_info_sha256 = (
            None if matched_control_dataset_info_sha256 is None
            else str(matched_control_dataset_info_sha256)
        )
        if self.matched_control_identity:
            info_path = os.path.join(self.data_path, "meta", "info.json")
            info_receipt = next(
                (
                    entry
                    for entry in self.training_source_runtime_receipt
                    if entry.get("logical_path") == "meta/info.json"
                ),
                None,
            )
            try:
                info_bytes, info_sha256, _ = read_verified_bytes(
                    info_path,
                    expected_sha256=(info_receipt or {}).get("sha256")
                    or self.matched_control_dataset_info_sha256,
                    expected_stat=(info_receipt or {}).get("stat"),
                )
                info_payload = strict_json_loads(info_bytes, name="LeRobot info")
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "matched-control identity cannot verify dataset metadata"
                ) from exc
            metadata_value = info_payload.get("dataset_id")
            if (
                not isinstance(self.matched_control_dataset_info_sha256, str)
                or not _SHA256_RE.fullmatch(self.matched_control_dataset_info_sha256)
                or self.matched_control_dataset_info_sha256 == "0" * 64
                or info_sha256 != self.matched_control_dataset_info_sha256
            ):
                raise ValueError(
                    "matched-control identity requires preflight-bound dataset info SHA"
                )
            metadata_dataset_id = (
                metadata_value.strip() if isinstance(metadata_value, str) else ""
            )
            if not metadata_dataset_id and not self.matched_control_dataset_id:
                raise ValueError(
                    "matched-control identity requires either meta/info.json dataset_id "
                    "or an explicit preflight-bound matched_control_dataset_id"
                )
            if (
                metadata_dataset_id
                and self.matched_control_dataset_id is not None
                and self.matched_control_dataset_id != metadata_dataset_id
            ):
                raise ValueError(
                    "configured matched_control_dataset_id differs from meta/info.json"
                )
            if metadata_dataset_id:
                self.matched_control_dataset_id = metadata_dataset_id
        self.production_episode_filter_receipt = None
        self.production_episode_filter_receipt_sha256 = None
        self.production_allowed_episode_ids = None
        if (production_episode_filter_receipt is None) != (
            production_episode_filter_receipt_sha256 is None
        ):
            raise ValueError(
                "production episode filtering requires both receipt and receipt SHA"
            )
        if production_episode_filter_receipt is not None:
            receipt, digest, allowed_ids = (
                validate_production_episode_filter_receipt(
                    production_episode_filter_receipt,
                    production_episode_filter_receipt_sha256,
                    kwargs.get("episodes"),
                )
            )
            self.production_episode_filter_receipt = receipt
            self.production_episode_filter_receipt_sha256 = digest
            self.production_allowed_episode_ids = allowed_ids
        if self.contact_sidecar_require_reviewed_v1_1:
            if not self.contact_sidecar_strict or self.contact_sidecar_dir is None:
                raise ValueError(
                    "reviewed contact_segment_v1_1 requires a strict sidecar directory"
                )
            if (
                not isinstance(self.contact_sidecar_expected_meta_sha256, str)
                or not _SHA256_RE.fullmatch(self.contact_sidecar_expected_meta_sha256)
                or self.contact_sidecar_expected_meta_sha256 == "0" * 64
                or not self.contact_sidecar_expected_episode_sha256
                or any(
                    not isinstance(value, str)
                    or not _SHA256_RE.fullmatch(value)
                    or value == "0" * 64
                    for value in self.contact_sidecar_expected_episode_sha256.values()
                )
            ):
                raise ValueError("reviewed sidecar requires exact preflight SHA receipts")
            meta_path = os.path.join(self.contact_sidecar_dir, "meta.json")
            try:
                meta_bytes, _, _ = read_verified_bytes(
                    meta_path,
                    expected_sha256=self.contact_sidecar_expected_meta_sha256,
                )
                meta = strict_json_loads(meta_bytes, name="reviewed sidecar meta")
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "contact sidecar meta fingerprint differs from preflight"
                ) from exc
            if (
                meta.get("schema_version") != SEGMENT_SCHEMA_VERSION
                or meta.get("target_dim") != SEGMENT_TARGET_DIM
                or meta.get("num_queries") != 12
                or meta.get("builder_identity")
                != "scripts/contact_labels/build_contact_segment_sidecars_v1_1.py"
                or meta.get("annotation_contract")
                != {
                    "annotation_role": "adjudicated",
                    "review_status": "reviewed",
                    "label_origin": "human_adjudicated",
                    "automatic_or_proposal_ground_truth": False,
                }
            ):
                raise ValueError("contact sidecar meta is not reviewed contact_segment_v1_1")
            if self.contact_human_adjudicated_only and (
                meta["annotation_contract"]["label_origin"] != "human_adjudicated"
                or meta["annotation_contract"]["automatic_or_proposal_ground_truth"]
                is not False
            ):
                raise ValueError("reviewed contact sidecar contains proxy-derived labels")
        if self.ego_hand_active_sidecar_dir is not None:
            if (
                not self.ego_hand_active_sidecar_strict
                or not isinstance(self.ego_hand_active_expected_meta_sha256, str)
                or not _SHA256_RE.fullmatch(self.ego_hand_active_expected_meta_sha256)
                or not self.ego_hand_active_expected_episode_sha256
                or any(
                    not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
                    for value in self.ego_hand_active_expected_episode_sha256.values()
                )
            ):
                raise ValueError(
                    "Ego hand-active sidecars require strict preflight SHA receipts"
                )
            try:
                meta_bytes, _, _ = read_verified_bytes(
                    os.path.join(self.ego_hand_active_sidecar_dir, "meta.json"),
                    expected_sha256=self.ego_hand_active_expected_meta_sha256,
                )
                hand_active_meta = strict_json_loads(
                    meta_bytes,
                    name="Ego hand-active sidecar meta",
                )
            except (OSError, ValueError) as exc:
                raise ValueError("invalid Ego hand-active sidecar meta") from exc
            if (
                not isinstance(hand_active_meta, dict)
                or hand_active_meta.get("schema_version")
                != EGO_HAND_ACTIVE_SCHEMA_VERSION
                or hand_active_meta.get("builder_identity") != EGO_HAND_ACTIVE_BUILDER
                or hand_active_meta.get("horizon_frames") != EGO_HAND_ACTIVE_HORIZON
                or hand_active_meta.get("action_start_offset") != 1
                or hand_active_meta.get("field_order")
                != list(EGO_HAND_ACTIVE_FIELD_ORDER)
                or hand_active_meta.get("hand_active_semantics")
                != EGO_HAND_ACTIVE_SEMANTICS
                or hand_active_meta.get("physical_contact_truth") is not False
                or hand_active_meta.get("episode_npz_sha256")
                != self.ego_hand_active_expected_episode_sha256
            ):
                raise ValueError("Ego hand-active sidecar semantic contract differs")
            self.ego_hand_active_meta = hand_active_meta
        else:
            if (
                self.ego_hand_active_sidecar_strict
                or self.ego_hand_active_expected_meta_sha256 is not None
                or self.ego_hand_active_expected_episode_sha256
            ):
                raise ValueError("Ego hand-active receipts require a sidecar directory")
            self.ego_hand_active_meta = None
        self._contact_sidecar_cache = {}
        self._contact_sidecar_cache_order = []
        self._ego_hand_active_cache = {}
        self._ego_hand_active_cache_order = []
        self.meta_name = meta_name
        self.kwargs = kwargs
        self.dataset = None

    @classmethod
    def load(cls, data_or_config: str | dict) -> BaseDataset:
        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        data_path = config.get("data_path", None)
        if isinstance(data_path, (list, tuple)):
            sub_configs = []
            for path in data_path:
                sub_config = dict(config)
                sub_config["data_path"] = path
                sub_config["_class_name"] = cls.__name__
                sub_configs.append(sub_config)
            return ConcatDataset.load(sub_configs)
        return cls(**config)

    def _init_t5_source(self) -> None:
        if self.t5_load_from is None:
            return
        if self.t5_load_from == "path":
            if not self.t5_embedding_path:
                raise KeyError("t5_load_from='path' requires t5_cfg['t5_embedding_path']")
            self._fixed_t5_embedding = self._load_t5_from_path(self.t5_embedding_path)
            if self._fixed_t5_embedding is None:
                raise FileNotFoundError(f"missing t5 embedding file: {self.t5_embedding_path}")
            return
        if self.t5_load_from == "pkl":
            if not self.prompt_t5_lookup_path:
                raise KeyError("t5_load_from='pkl' requires t5_cfg['prompt_t5_lookup_path']")
            with open(self.prompt_t5_lookup_path, "rb") as f:
                self._prompt_t5_lookup = pickle.load(f)
            return
        if self.t5_embedding_dir is None:
            t5_dir = os.path.join(self.data_path, "t5_embedding")
            if os.path.isdir(t5_dir):
                self.t5_embedding_dir = t5_dir
        if self.t5_embedding_dir is None:
            raise FileNotFoundError(
                f"t5_load_from='dir' requires t5_cfg['t5_embedding_dir'] "
                f"or {self.data_path}/t5_embedding"
            )

    def open(self) -> None:
        if self.dataset is None:
            repo_id = os.path.basename(self.data_path)
            if self.training_source_runtime_receipt:
                info_receipt = next(
                    entry
                    for entry in self.training_source_runtime_receipt
                    if entry.get("logical_path") == "meta/info.json"
                )
                info_bytes, _, _ = read_verified_bytes(
                    info_receipt["path_record_only"],
                    expected_sha256=info_receipt["sha256"],
                    expected_stat=info_receipt["stat"],
                )
                info_payload = strict_json_loads(info_bytes, name="LeRobot info")
                dataset_fps = float(info_payload["fps"])
            else:
                dataset_meta = _LeRobotDatasetMetadata(repo_id, root=self.data_path)
                dataset_fps = dataset_meta.fps
            delta_timestamps = {}
            if self.delta_info is not None:
                for delta_name, delta_size in self.delta_info.items():
                    delta_timestamps[delta_name] = [i / dataset_fps for i in range(int(delta_size))]
            if self.delta_frames is not None:
                for delta_name, frames in dict(self.delta_frames).items():
                    delta_timestamps[delta_name] = [float(i) / dataset_fps for i in frames]
            if self.delta_timestamps is not None:
                for delta_name, stamps in dict(self.delta_timestamps).items():
                    delta_timestamps[delta_name] = [float(t) for t in stamps]
            if not delta_timestamps:
                delta_timestamps = {"action": [0.0]}

            fast_kwargs = dict(self.kwargs)
            if "video_backend" not in fast_kwargs or fast_kwargs.get("video_backend", None) is None:
                fast_kwargs["video_backend"] = "pyav"
            self.dataset = FastLeRobotDataset(
                repo_id,
                root=self.data_path,
                delta_timestamps=delta_timestamps,
                verified_source_receipt=self.training_source_runtime_receipt or None,
                verified_private_cache_parent=self.verified_private_cache_parent,
                verified_training_run_nonce=self.verified_training_run_nonce,
                **fast_kwargs,
            )
            if self.production_allowed_episode_ids is not None:
                actual_episodes = getattr(self.dataset, "episodes", None)
                if (
                    actual_episodes is None
                    or list(actual_episodes)
                    != self.production_episode_filter_receipt["train_episode_ids"]
                ):
                    raise ValueError(
                        "FastLeRobotDataset did not retain the exact production train filter"
                    )
                info = getattr(self.dataset.meta, "info", None)
                if (
                    not isinstance(info, dict)
                    or info.get("dataset_id")
                    != self.production_episode_filter_receipt["dataset_id"]
                ):
                    raise ValueError(
                        "production episode filter receipt dataset_id differs from runtime data"
                    )
            if self.data_size is not None:
                assert self.data_size == len(self.dataset)
            else:
                self.data_size = len(self.dataset)

            if self.robotype is None:
                self.robotype = self._load_robotype()
                if self.robotype is None:
                    raise KeyError(f"missing robotype in meta/info.* under {self.data_path}")
            self._init_t5_source()

    @staticmethod
    def _split_main_subtask(task_text) -> tuple[str | None, str | None]:
        text = str(task_text or "").strip()
        if not text:
            return None, None
        match = _SUBTASK_MARKER_RE.search(text)
        if match is None:
            return text.lower(), None
        main_task = text[: match.start()].strip().lower()
        subtask = text[match.end() :].strip().lower()
        return main_task or None, subtask or None

    def _extract_t5_embedding(self, obj) -> torch.Tensor:
        if isinstance(obj, dict):
            if self.t5_embedding_key in obj:
                return obj[self.t5_embedding_key]
            if "prompt_embeds" in obj:
                return obj["prompt_embeds"]
            condition_dict = obj.get("condition_dict")
            if isinstance(condition_dict, dict) and "prompt_embeds" in condition_dict:
                return condition_dict["prompt_embeds"]
            return next(iter(obj.values()))
        return obj

    def _load_t5_from_path(self, path: str):
        if path in self._t5_cache:
            return self._t5_cache[path]
        receipt = self._t5_receipt_by_path.get(os.path.abspath(path))
        if self.t5_runtime_source_receipt and receipt is None:
            raise ValueError(f"missing preflight T5 receipt for {path}")
        try:
            payload, _, _ = read_verified_bytes(
                path,
                expected_sha256=(receipt or {}).get("sha256"),
                expected_stat=(receipt or {}).get("stat"),
            )
        except (OSError, ValueError):
            if self.t5_runtime_source_receipt:
                raise
            return None
        emb = self._extract_t5_embedding(
            torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        )
        if self.t5_runtime_source_receipt and (
            not isinstance(emb, torch.Tensor)
            or emb.ndim != 2
            or not 1 <= int(emb.shape[0]) <= 64
            or int(emb.shape[1]) != 4096
            or emb.dtype != torch.float32
            or not bool(torch.isfinite(emb).all())
        ):
            raise ValueError(
                f"verified Stage1 T5 source must be a finite float32 [1..64,4096] tensor: {path}"
            )
        if self.t5_cache_size > 0:
            self._t5_cache[path] = emb
            self._t5_cache_order.append(path)
            if len(self._t5_cache_order) > self.t5_cache_size:
                old = self._t5_cache_order.pop(0)
                self._t5_cache.pop(old, None)
        return emb

    def _load_t5_from_lookup(self, lookup: dict, prompt_text: str | None):
        if not prompt_text:
            return None
        entry = lookup.get(prompt_text)
        if entry is None:
            return None
        if isinstance(entry.get("t5_embedding"), torch.Tensor):
            return entry["t5_embedding"]
        return self._load_t5_from_path(str(entry.get("t5_embedding_path", "") or ""))

    def _load_robotype(self):
        if self.training_source_runtime_receipt and self.dataset is not None:
            info = self.dataset.meta.info
            if not isinstance(info, dict):
                raise ValueError("verified LeRobot metadata is not an object")
        else:
            for name in ('info.json', 'info.yaml', 'info.yml'):
                path = os.path.join(self.data_path, 'meta', name)
                if os.path.isfile(path):
                    info = utils.load_file(path)
                    break
            else:
                raise FileNotFoundError(f"missing meta/info.(json|yaml|yml) under {self.data_path}")
        for key in ('robotype', 'robot_type', 'robot', 'robot_name', 'robot_model'):
            if key in info:
                robotype = info[key]
                return robotype.strip() if isinstance(robotype, str) else robotype
        return None

    def _get_metadata_data_size(self):
        if self.training_source_runtime_receipt:
            receipt_by_logical = {
                entry["logical_path"]: entry
                for entry in self.training_source_runtime_receipt
            }
            info_receipt = receipt_by_logical.get("meta/info.json")
            if info_receipt is None:
                raise ValueError("verified LeRobot receipt lacks meta/info.json")
            info_bytes, _, _ = read_verified_bytes(
                info_receipt["path_record_only"],
                expected_sha256=info_receipt["sha256"],
                expected_stat=info_receipt["stat"],
            )
            info = strict_json_loads(info_bytes, name="LeRobot info")
            episodes = self.kwargs.get("episodes")
            if episodes is not None:
                episodes_receipt = receipt_by_logical.get("meta/episodes.jsonl")
                if episodes_receipt is None:
                    raise ValueError(
                        "verified LeRobot receipt lacks meta/episodes.jsonl"
                    )
                episode_bytes, _, _ = read_verified_bytes(
                    episodes_receipt["path_record_only"],
                    expected_sha256=episodes_receipt["sha256"],
                    expected_stat=episodes_receipt["stat"],
                )
                lengths = {}
                for line_number, line in enumerate(episode_bytes.splitlines(), 1):
                    if not line.strip():
                        continue
                    row = strict_json_loads(
                        line,
                        name=f"LeRobot episodes line {line_number}",
                    )
                    episode_index = row.get("episode_index") if isinstance(row, dict) else None
                    length = row.get("length") if isinstance(row, dict) else None
                    if (
                        isinstance(episode_index, bool)
                        or not isinstance(episode_index, int)
                        or episode_index < 0
                        or isinstance(length, bool)
                        or not isinstance(length, int)
                        or length <= 0
                        or episode_index in lengths
                    ):
                        raise ValueError("invalid verified LeRobot episode metadata")
                    lengths[episode_index] = length
                requested = []
                for value in episodes:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError("invalid verified LeRobot episode selection")
                    requested.append(value)
                missing = sorted(set(requested) - set(lengths))
                if missing:
                    raise IndexError(f"episodes missing from verified metadata: {missing}")
                return sum(lengths[index] for index in requested)
            total_frames = info.get("total_frames") if isinstance(info, dict) else None
            if (
                isinstance(total_frames, bool)
                or not isinstance(total_frames, int)
                or total_frames <= 0
            ):
                raise ValueError("invalid verified LeRobot total_frames")
            return total_frames

        for name in ('info.json', 'info.yaml', 'info.yml'):
            path = os.path.join(self.data_path, 'meta', name)
            if os.path.isfile(path):
                info = utils.load_file(path)
                break
        else:
            return None

        episodes = self.kwargs.get("episodes")
        episodes_path = os.path.join(self.data_path, "meta", "episodes.jsonl")
        if episodes is not None and os.path.isfile(episodes_path):
            lengths = {}
            with open(episodes_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    lengths[int(row["episode_index"])] = int(row["length"])
            missing = sorted(set(int(index) for index in episodes) - set(lengths))
            if missing:
                raise IndexError(f"episodes missing from {episodes_path}: {missing}")
            return sum(lengths[int(index)] for index in episodes)

        try:
            return int(info.get('total_frames'))
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        if self.dataset is not None:
            close_verified = getattr(self.dataset, "close_verified_sources", None)
            if close_verified is not None:
                close_verified()
            self.dataset = None
        self._prompt_t5_lookup = None
        self._fixed_t5_embedding = None
        self._t5_cache.clear()
        self._t5_cache_order.clear()
        self._contact_sidecar_cache.clear()
        self._contact_sidecar_cache_order.clear()
        self._ego_hand_active_cache.clear()
        self._ego_hand_active_cache_order.clear()
        super().close()

    def __len__(self) -> int:
        if self.data_size is None:
            self.data_size = self._get_metadata_data_size()
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index: int) -> dict:
        data_dict = self.dataset[index]
        if self.production_allowed_episode_ids is not None:
            require_train_episode(
                data_dict.get("episode_index"), self.production_allowed_episode_ids
            )
        data_dict['robotype'] = self.robotype
        data_dict['data_path'] = self.data_path
        if self.matched_control_identity:
            data_dict['matched_control_dataset_id'] = self.matched_control_dataset_id
            data_dict['matched_control_global_index'] = self._scalar_int(
                data_dict.get("index", index), "authoritative global index"
            )
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            data_dict[self.meta_name] = self.dataset.meta
        self._load_t5_embedding(data_dict)
        self._load_contact_sidecar(data_dict)
        self._load_ego_hand_active_sidecar(data_dict)
        return data_dict

    @staticmethod
    def _scalar_int(value, name: str) -> int:
        if hasattr(value, "item"):
            value = value.item()
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name}={value!r} for contact sidecar lookup") from exc

    def _read_contact_sidecar(self, episode_index: int):
        if episode_index in self._contact_sidecar_cache:
            return self._contact_sidecar_cache[episode_index]
        path = os.path.join(
            self.contact_sidecar_dir,
            self.contact_sidecar_pattern.format(episode_index=episode_index),
        )
        if not os.path.isfile(path):
            if self.contact_sidecar_strict:
                raise FileNotFoundError(f"missing contact sidecar: {path}")
            return None
        if self.contact_sidecar_require_reviewed_v1_1:
            expected_digest = self.contact_sidecar_expected_episode_sha256.get(
                os.path.basename(path)
            )
            if expected_digest is None:
                raise ValueError(f"missing preflight NPZ fingerprint for {path}")
        else:
            expected_digest = None
        try:
            payload, _, _ = read_verified_bytes(
                path,
                expected_sha256=expected_digest,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"contact sidecar NPZ changed after preflight: {path}") from exc
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            sidecar = {key: archive[key] for key in archive.files}
        required = {
            "contact_target",
            "contact_loss_mask",
            "contact_confidence",
            "contact_condition_mask",
        }
        missing = sorted(required - set(sidecar))
        if missing:
            raise KeyError(f"contact sidecar {path} is missing arrays: {missing}")
        lengths = {int(sidecar[key].shape[0]) for key in required}
        if len(lengths) != 1:
            raise ValueError(f"inconsistent frame counts in contact sidecar {path}: {lengths}")
        if self.contact_sidecar_require_reviewed_v1_1:
            expected_tail = (12, SEGMENT_TARGET_DIM)
            for key in required:
                value = sidecar[key]
                if (
                    value.ndim != 3
                    or value.shape[1:] != expected_tail
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"reviewed sidecar {path}:{key} must be finite float32 "
                        f"[T,12,{SEGMENT_TARGET_DIM}], got {value.dtype} {value.shape}"
                    )
        if self.contact_sidecar_cache_size > 0:
            self._contact_sidecar_cache[episode_index] = sidecar
            self._contact_sidecar_cache_order.append(episode_index)
            if len(self._contact_sidecar_cache_order) > self.contact_sidecar_cache_size:
                old = self._contact_sidecar_cache_order.pop(0)
                self._contact_sidecar_cache.pop(old, None)
        return sidecar

    def _load_contact_sidecar(self, data_dict: dict) -> None:
        if self.contact_sidecar_dir is None:
            return
        episode_index = self._scalar_int(data_dict.get("episode_index"), "episode_index")
        frame_index = self._scalar_int(data_dict.get("frame_index"), "frame_index")
        sidecar = self._read_contact_sidecar(episode_index)
        if sidecar is None:
            return
        frame_count = int(sidecar["contact_target"].shape[0])
        if not 0 <= frame_index < frame_count:
            raise IndexError(
                f"frame_index={frame_index} outside contact sidecar episode "
                f"{episode_index} length={frame_count}"
            )
        for key, array in sidecar.items():
            if key.startswith("contact_"):
                data_dict[key] = array[frame_index]

    def _read_ego_hand_active_sidecar(self, episode_index: int):
        if episode_index in self._ego_hand_active_cache:
            return self._ego_hand_active_cache[episode_index]
        path = os.path.join(
            self.ego_hand_active_sidecar_dir,
            self.ego_hand_active_sidecar_pattern.format(episode_index=episode_index),
        )
        name = os.path.basename(path)
        expected_digest = self.ego_hand_active_expected_episode_sha256.get(name)
        if expected_digest is None:
            raise ValueError(f"missing preflight Ego hand-active NPZ fingerprint: {name}")
        lengths = self.ego_hand_active_meta.get("episode_lengths", {})
        frame_count = lengths.get(str(episode_index))
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise ValueError(f"missing Ego hand-active episode length: {episode_index}")
        try:
            arrays = load_hand_active_npz(
                path,
                expected_sha256=expected_digest,
                frame_count=frame_count,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid Ego hand-active sidecar: {path}") from exc
        if self.ego_hand_active_sidecar_cache_size > 0:
            self._ego_hand_active_cache[episode_index] = arrays
            self._ego_hand_active_cache_order.append(episode_index)
            if (
                len(self._ego_hand_active_cache_order)
                > self.ego_hand_active_sidecar_cache_size
            ):
                old = self._ego_hand_active_cache_order.pop(0)
                self._ego_hand_active_cache.pop(old, None)
        return arrays

    def _load_ego_hand_active_sidecar(self, data_dict: dict) -> None:
        if self.ego_hand_active_sidecar_dir is None:
            return
        episode_index = self._scalar_int(data_dict.get("episode_index"), "episode_index")
        frame_index = self._scalar_int(data_dict.get("frame_index"), "frame_index")
        arrays = self._read_ego_hand_active_sidecar(episode_index)
        frame_count = int(arrays["ego_hand_active_current"].shape[0])
        if not 0 <= frame_index < frame_count:
            raise IndexError(
                f"frame_index={frame_index} outside Ego hand-active sidecar "
                f"episode {episode_index} length={frame_count}"
            )
        for key, value in arrays.items():
            data_dict[key] = value[frame_index]

    def _load_t5_embedding(self, data_dict: dict) -> None:
        if self.t5_load_from is None:
            return
        if self.t5_load_from == "path":
            data_dict[self.t5_embedding_key] = self._fixed_t5_embedding
            return
        if self.t5_load_from == "pkl":
            main_task, subtask = self._split_main_subtask(data_dict.get("task"))
            main_lookup = self._prompt_t5_lookup.get("main_task", {})
            sub_lookup = self._prompt_t5_lookup.get("subtask", {})
            main_emb = self._load_t5_from_lookup(main_lookup, main_task)
            sub_emb = self._load_t5_from_lookup(sub_lookup, subtask)
            data_dict[self.t5_embedding_key] = main_emb
            if sub_emb is not None:
                data_dict['subtask_t5_embedding'] = sub_emb
            return
        if self.t5_load_from == "dir":
            episode_index = data_dict.get("episode_index", None)
            if hasattr(episode_index, "item"):
                try:
                    episode_index = episode_index.item()
                except Exception:
                    return
            try:
                episode_index = int(episode_index)
            except Exception:
                return
            path = os.path.join(
                self.t5_embedding_dir,
                self.t5_embedding_pattern.format(episode_index=episode_index),
            )
            emb = self._load_t5_from_path(path)
            if emb is not None:
                data_dict[self.t5_embedding_key] = emb

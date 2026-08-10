import json
import hashlib
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

_SUBTASK_MARKER_RE = re.compile(r"\bsubtask\s*:\s*", flags=re.IGNORECASE)
_T5_LOAD_FROM_CHOICES = ("dir", "pkl", "path")


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
        matched_control_identity: bool = False,
        matched_control_dataset_id: str | None = None,
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
        self.matched_control_identity = bool(matched_control_identity)
        self.matched_control_dataset_id = (
            None
            if matched_control_dataset_id is None
            else str(matched_control_dataset_id).strip()
        )
        if not self.matched_control_dataset_id:
            self.matched_control_dataset_id = None
        if self.matched_control_identity:
            info_path = os.path.join(self.data_path, "meta", "info.json")
            if not os.path.isfile(info_path):
                raise FileNotFoundError(
                    f"matched-control identity requires dataset metadata: {info_path}"
                )
            with open(info_path, "r", encoding="utf-8") as handle:
                metadata_value = json.load(handle).get("dataset_id")
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
        if self.contact_sidecar_require_reviewed_v1_1:
            if not self.contact_sidecar_strict or self.contact_sidecar_dir is None:
                raise ValueError(
                    "reviewed contact_segment_v1_1 requires a strict sidecar directory"
                )
            meta_path = os.path.join(self.contact_sidecar_dir, "meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"missing reviewed contact sidecar meta: {meta_path}")
            with open(meta_path, "rb") as handle:
                meta_bytes = handle.read()
            meta_digest = hashlib.sha256(meta_bytes).hexdigest()
            if meta_digest != self.contact_sidecar_expected_meta_sha256:
                raise ValueError("contact sidecar meta fingerprint differs from preflight")
            meta = json.loads(meta_bytes.decode("utf-8"))
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
        self._contact_sidecar_cache = {}
        self._contact_sidecar_cache_order = []
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
            dataset_meta = _LeRobotDatasetMetadata(repo_id, root=self.data_path)
            delta_timestamps = {}
            if self.delta_info is not None:
                for delta_name, delta_size in self.delta_info.items():
                    delta_timestamps[delta_name] = [i / dataset_meta.fps for i in range(int(delta_size))]
            if self.delta_frames is not None:
                for delta_name, frames in dict(self.delta_frames).items():
                    delta_timestamps[delta_name] = [float(i) / dataset_meta.fps for i in frames]
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
                **fast_kwargs,
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
        if not os.path.isfile(path):
            return None
        emb = self._extract_t5_embedding(torch.load(path, map_location="cpu"))
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
            self.dataset = None
        self._prompt_t5_lookup = None
        self._fixed_t5_embedding = None
        self._t5_cache.clear()
        self._t5_cache_order.clear()
        self._contact_sidecar_cache.clear()
        self._contact_sidecar_cache_order.clear()
        super().close()

    def __len__(self) -> int:
        if self.data_size is None:
            self.data_size = self._get_metadata_data_size()
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index: int) -> dict:
        data_dict = self.dataset[index]
        data_dict['robotype'] = self.robotype
        data_dict['data_path'] = self.data_path
        if self.matched_control_identity:
            data_dict['matched_control_dataset_id'] = self.matched_control_dataset_id
            data_dict['matched_control_global_index'] = int(index)
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            data_dict[self.meta_name] = self.dataset.meta
        self._load_t5_embedding(data_dict)
        self._load_contact_sidecar(data_dict)
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
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_digest:
                raise ValueError(f"contact sidecar NPZ changed after preflight: {path}")
        with np.load(path, allow_pickle=False) as archive:
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

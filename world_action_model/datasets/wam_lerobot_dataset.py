import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from giga_datasets import utils
from giga_datasets.datasets.base_dataset import BaseDataset
from giga_datasets.datasets.dataset import ConcatDataset, load_config, register_dataset
from giga_datasets.datasets.lerobot_dataset import FastLeRobotDataset, _LeRobotDatasetMetadata

_SUBTASK_MARKER_RE = re.compile(r"\bsubtask\s*:\s*", flags=re.IGNORECASE)
_T5_LOAD_FROM_CHOICES = ("dir", "pkl", "path")
_COUNTERFACTUAL_MASK_SCHEMA = "libero-counterfactual-robot-mask-v1"


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        meta_name: str | None = None,
        robotype: str | None = None,
        counterfactual_mask_cache: str | None = None,
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
        self.meta_name = meta_name
        self.counterfactual_mask_cache = counterfactual_mask_cache
        self._counterfactual_cache_root = None
        self._counterfactual_records = None
        self._counterfactual_sample_indices = None
        if counterfactual_mask_cache is not None:
            self._load_counterfactual_manifest(counterfactual_mask_cache)
        self.kwargs = kwargs
        self.dataset = None

    def _load_counterfactual_manifest(self, cache_root: str) -> None:
        root = Path(cache_root).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != _COUNTERFACTUAL_MASK_SCHEMA:
            raise ValueError(
                f"Unsupported counterfactual mask schema: {manifest.get('schema_version')!r}"
            )
        if manifest.get("rgb_supervision") is not None:
            raise ValueError("Counterfactual mask cache must not contain RGB supervision")
        if int(manifest.get("action_chunk_index", -1)) != 0:
            raise ValueError("Only action_chunk_index=0 counterfactual caches are supported")
        manifest_horizons = [int(value) for value in manifest.get("horizons", [])]
        if self.delta_frames is not None:
            for key, frames in self.delta_frames.items():
                if [int(value) for value in frames] != manifest_horizons:
                    raise ValueError(
                        f"Counterfactual cache horizons do not match delta_frames[{key!r}]"
                    )
        records = list(manifest.get("episodes", []))
        if not records:
            raise ValueError(f"No episode records in {manifest_path}")
        seen = set()
        for record in records:
            key = (int(record["episode_index"]), int(record["source_frame_index"]))
            if key in seen:
                raise ValueError(f"Duplicate counterfactual cache source: {key}")
            seen.add(key)
            cache_path = root / record["cache_file"]
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            if _sha256_file(cache_path) != record["cache_sha256"]:
                raise ValueError(f"Counterfactual cache hash mismatch: {cache_path}")
        info_path = Path(self.data_path) / "meta" / "info.json"
        if _sha256_file(info_path) != manifest["dataset_info_sha256"]:
            raise ValueError(
                f"Counterfactual cache dataset metadata does not match {self.data_path}"
            )
        self._counterfactual_cache_root = root
        self._counterfactual_records = records
        self.data_size = len(records)

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
            if self._counterfactual_records is not None:
                cache_episodes = sorted(
                    {int(record["episode_index"]) for record in self._counterfactual_records}
                )
                configured_episodes = fast_kwargs.get("episodes")
                if configured_episodes is not None and sorted(configured_episodes) != cache_episodes:
                    raise ValueError(
                        "Configured episodes do not match counterfactual cache episodes"
                    )
                fast_kwargs["episodes"] = cache_episodes
            if "video_backend" not in fast_kwargs or fast_kwargs.get("video_backend", None) is None:
                fast_kwargs["video_backend"] = "pyav"
            self.dataset = FastLeRobotDataset(
                repo_id,
                root=self.data_path,
                delta_timestamps=delta_timestamps,
                **fast_kwargs,
            )
            if self._counterfactual_records is not None:
                self._build_counterfactual_sample_indices()
            elif self.data_size is not None:
                assert self.data_size == len(self.dataset)
            else:
                self.data_size = len(self.dataset)

            if self.robotype is None:
                self.robotype = self._load_robotype()
                if self.robotype is None:
                    raise KeyError(f"missing robotype in meta/info.* under {self.data_path}")
            self._init_t5_source()

    def _build_counterfactual_sample_indices(self) -> None:
        self.dataset._ensure_hf_dataset_loaded()
        episode_indices = self.dataset.hf_dataset["episode_index"]
        frame_indices = self.dataset.hf_dataset["frame_index"]
        source_to_index = {}
        for index, (episode_index, frame_index) in enumerate(zip(episode_indices, frame_indices)):
            if hasattr(episode_index, "item"):
                episode_index = episode_index.item()
            if hasattr(frame_index, "item"):
                frame_index = frame_index.item()
            source_to_index[(int(episode_index), int(frame_index))] = index
        sample_indices = []
        for record in self._counterfactual_records:
            key = (int(record["episode_index"]), int(record["source_frame_index"]))
            if key not in source_to_index:
                raise ValueError(f"Counterfactual cache source is absent from dataset: {key}")
            sample_indices.append(source_to_index[key])
        self._counterfactual_sample_indices = sample_indices

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

        try:
            return int(info.get('total_frames'))
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset = None
        self._counterfactual_sample_indices = None
        self._prompt_t5_lookup = None
        self._fixed_t5_embedding = None
        self._t5_cache.clear()
        self._t5_cache_order.clear()
        super().close()

    def __len__(self) -> int:
        if self.data_size is None:
            self.data_size = self._get_metadata_data_size()
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index: int) -> dict:
        counterfactual_record = None
        if self._counterfactual_records is not None:
            if self._counterfactual_sample_indices is None:
                raise RuntimeError("Counterfactual sample index is not initialized")
            counterfactual_record = self._counterfactual_records[index]
            index = self._counterfactual_sample_indices[index]
        data_dict = self.dataset[index]
        data_dict['robotype'] = self.robotype
        data_dict['data_path'] = self.data_path
        if counterfactual_record is not None:
            episode_index = int(data_dict["episode_index"].item())
            frame_index = int(data_dict["frame_index"].item())
            expected_source = (
                int(counterfactual_record["episode_index"]),
                int(counterfactual_record["source_frame_index"]),
            )
            if (episode_index, frame_index) != expected_source:
                raise ValueError(
                    f"Counterfactual sample mismatch: got {(episode_index, frame_index)}, "
                    f"expected {expected_source}"
                )
            cache_path = self._counterfactual_cache_root / counterfactual_record["cache_file"]
            with np.load(cache_path, allow_pickle=False) as cache:
                data_dict["counterfactual_action"] = torch.from_numpy(
                    cache["executed_actions"].copy()
                )
                data_dict["counterfactual_robot_masks"] = torch.from_numpy(
                    cache["robot_mask"].copy()
                )
                data_dict["counterfactual_gripper_masks"] = torch.from_numpy(
                    cache["gripper_mask"].copy()
                )
                data_dict["counterfactual_horizons"] = torch.from_numpy(
                    cache["horizons"].copy()
                )
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            data_dict[self.meta_name] = self.dataset.meta
        self._load_t5_embedding(data_dict)
        return data_dict

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

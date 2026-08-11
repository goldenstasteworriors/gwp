import os
import random

import numpy as np
import torch
import torch.nn.functional as torch_F
from giga_train import TRANSFORMS
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from world_action_model.transforms.wa_transforms import WATransforms
from world_action_model.ego_action_schema import (
    CANONICAL_OPEN_FRACTION_SEMANTICS,
    convert_hand_status_semantics,
    dual_eef_delta_to_action,
    dual_eef_pose_to_state,
    ego_grip_loss_weight_multiplier,
)
from world_action_model.ego_hand_active_sidecar import build_future_action_time_mask
from world_action_model.asymmetric_action_schema import ASYMMETRIC_SCHEMA_ID
from world_action_model.matched_control_hashes import (
    SAMPLE_AUGMENTATION_KEY,
    SAMPLE_DATA_PATH_KEY,
    SAMPLE_IDENTITY_KEY,
    SAMPLE_RAW_HASH_KEY,
    TRACE_VERSION,
    build_sample_identity,
    canonical_json_dumps,
    canonical_sha256,
    normalize_path,
)
from world_action_model.verified_io import read_verified_bytes, strict_json_loads


@TRANSFORMS.register
class WALeRobotTransformsPretrain(WATransforms):
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        robotype_to_embodiment_id=None,
        robotype_default_embodiment_id=0,
        model_action_dim=None,
        delta_mask_by_embodiment_id=None,
        norm_use_quantiles=False,
        norm_enable_clamp=False,
        image_cfg=None,
        image_augmentation=None,
        num_views=1,
        view_keys=None,
        view_concat_mode="t",
        state_key="observation.state",
        action_key="action",
        ego_dual_eef_cfg=None,
        ego_grip_semantics=None,
        ego_hand_active_future_key=None,
        ego_hand_active_transition_pair_key=None,
        require_exact_effective_dim=False,
        expected_action_schema_id=None,
        expected_hardware_contract_sha256=None,
        expected_calibration_capture_sha256=None,
        ego_grip_proxy_weight=0.25,
        task_key="task",
        emit_action_denorm_scale=False,
        max_prompt_len=64,
        subtask_prob=0,
        matched_control_capture=False,
        eval_crop_mode="legacy_random",
    ):
        if norm_path is None:
            raise ValueError("norm_path list is none")

        if isinstance(norm_path, (str, os.PathLike)):
            norm_specs = [{
                "path": str(norm_path),
                "data_paths": None,
                "sha256": None,
                "runtime_stat_record_only": None,
            }]
        else:
            norm_specs = []
            for item in norm_path:
                if isinstance(item, (str, os.PathLike)):
                    norm_specs.append({
                        "path": str(item),
                        "data_paths": None,
                        "sha256": None,
                        "runtime_stat_record_only": None,
                    })
                elif isinstance(item, dict):
                    path = item.get("path")
                    if path is None:
                        raise ValueError(f"norm_path entry missing path: {item}")
                    data_paths = item.get("data_paths", None)
                    if data_paths is not None:
                        data_paths = {os.path.normpath(str(p)) for p in data_paths}
                    norm_specs.append({
                        "path": str(path),
                        "data_paths": data_paths,
                        "sha256": item.get("sha256"),
                        "runtime_stat_record_only": item.get(
                            "runtime_stat_record_only"
                        ),
                    })
                else:
                    raise TypeError(f"Unsupported norm_path entry type: {type(item).__name__}")
            if len(norm_specs) == 0:
                raise ValueError("norm_path list is empty")
        norm_paths = [spec["path"] for spec in norm_specs]
        loaded_stats_dicts = []
        for spec in norm_specs:
            payload, _, _ = read_verified_bytes(
                spec["path"],
                expected_sha256=spec["sha256"],
                expected_stat=spec["runtime_stat_record_only"],
            )
            stats_dict = strict_json_loads(payload, name=f"norm stats {spec['path']}")
            if not isinstance(stats_dict, dict):
                raise ValueError(f"normalization stats must be an object: {spec['path']}")
            loaded_stats_dicts.append(stats_dict)

        super().__init__(
            is_train=is_train,
            dst_size=dst_size,
            num_frames=num_frames,
            fps=fps,
            norm_path=norm_paths[0],
            norm_payload=loaded_stats_dicts[0],
            image_cfg=image_cfg,
            num_views=num_views,
        )

        self.robotype_default_embodiment_id = int(robotype_default_embodiment_id)
        if robotype_to_embodiment_id is None:
            robotype_to_embodiment_id = {"aloha": 0, "agilex": 0, "agibot": 1, "ur5": 3, "arx5": 4}
        self.robotype_to_embodiment_id = {str(k): int(v) for k, v in dict(robotype_to_embodiment_id).items()}
        self.model_action_dim = None if model_action_dim is None else int(model_action_dim)
        if delta_mask_by_embodiment_id is None:
            raise ValueError("delta_mask_by_embodiment_id must be provided")
        self.delta_mask_by_embodiment_id = {
            int(k): np.asarray(v, dtype=bool) for k, v in dict(delta_mask_by_embodiment_id).items()
        }
        if len(self.delta_mask_by_embodiment_id) == 0:
            raise ValueError("delta_mask_by_embodiment_id must not be empty")

        self.norm_use_quantiles = bool(norm_use_quantiles)
        self.norm_enable_clamp = bool(norm_enable_clamp)

        self.norm_paths = norm_paths
        self.norm_specs = norm_specs
        self.stats_dicts = loaded_stats_dicts
        for json_path in self.norm_paths:
            print("Loading stats dict from:", json_path)

        if view_keys is None:
            view_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        self.view_keys = list(view_keys)
        self.view_concat_mode = str(view_concat_mode)
        self.state_key = state_key
        self.action_key = action_key
        self.ego_dual_eef_cfg = (
            None if ego_dual_eef_cfg is None else dict(ego_dual_eef_cfg)
        )
        self.ego_grip_semantics = (
            None if ego_grip_semantics is None else str(ego_grip_semantics)
        )
        self.ego_hand_active_future_key = (
            None
            if ego_hand_active_future_key is None
            else str(ego_hand_active_future_key)
        )
        self.ego_hand_active_transition_pair_key = (
            None
            if ego_hand_active_transition_pair_key is None
            else str(ego_hand_active_transition_pair_key)
        )
        self.require_exact_effective_dim = bool(require_exact_effective_dim)
        self.expected_action_schema_id = (
            None if expected_action_schema_id is None else str(expected_action_schema_id)
        )
        self.expected_hardware_contract_sha256 = (
            None
            if expected_hardware_contract_sha256 is None
            else str(expected_hardware_contract_sha256)
        )
        self.expected_calibration_capture_sha256 = (
            None
            if expected_calibration_capture_sha256 is None
            else str(expected_calibration_capture_sha256)
        )
        if self.ego_grip_semantics is not None and self.ego_dual_eef_cfg is None:
            raise ValueError("ego_grip_semantics requires ego_dual_eef_cfg")
        if (self.ego_hand_active_future_key is None) != (
            self.ego_hand_active_transition_pair_key is None
        ):
            raise ValueError(
                "Ego hand-active future and transition-pair keys must be configured together"
            )
        if self.ego_hand_active_future_key is not None and self.ego_dual_eef_cfg is None:
            raise ValueError("Ego hand-active masks require ego_dual_eef_cfg")
        if (
            self.ego_dual_eef_cfg is not None
            and self.expected_action_schema_id == ASYMMETRIC_SCHEMA_ID
            and self.ego_grip_semantics is None
        ):
            raise ValueError(
                "cross-embodiment asymmetric25 input requires explicit ego_grip_semantics"
            )
        self.ego_grip_proxy_weight = float(ego_grip_proxy_weight)
        if not 0.0 <= self.ego_grip_proxy_weight <= 1.0:
            raise ValueError("ego_grip_proxy_weight must be in [0, 1]")
        if self.expected_action_schema_id is not None:
            for json_path, stats_dict in zip(self.norm_paths, self.stats_dicts):
                metadata = stats_dict.get("metadata")
                schema_id = metadata.get("action_schema_id") if isinstance(metadata, dict) else None
                if schema_id != self.expected_action_schema_id:
                    raise ValueError(
                        f"normalization stats {json_path} action_schema_id={schema_id!r}; "
                        f"expected {self.expected_action_schema_id!r}"
                    )
                if self.expected_hardware_contract_sha256 is not None:
                    actual = metadata.get("hardware_contract_sha256")
                    if actual != self.expected_hardware_contract_sha256:
                        raise ValueError(
                            f"normalization stats {json_path} hardware contract digest differs"
                        )
                if self.expected_calibration_capture_sha256 is not None:
                    actual = metadata.get("calibration_capture_sha256")
                    if actual != self.expected_calibration_capture_sha256:
                        raise ValueError(
                            f"normalization stats {json_path} calibration capture digest differs"
                        )
        self.task_key = task_key
        self.emit_action_denorm_scale = bool(emit_action_denorm_scale)
        self.max_prompt_len = int(max_prompt_len)
        self.subtask_prob = float(subtask_prob)
        self.image_augmentation = dict(image_augmentation or {})
        self.matched_control_capture = bool(matched_control_capture)
        self.eval_crop_mode = str(eval_crop_mode)
        if self.eval_crop_mode not in {"legacy_random", "center"}:
            raise ValueError(
                "eval_crop_mode must be 'legacy_random' or 'center', got "
                f"{self.eval_crop_mode!r}"
            )
        self._warned_unknown_robotype = False

    def _parse_robotype(self, robotype):
        if robotype is None:
            return None
        if isinstance(robotype, bytes):
            robotype = robotype.decode("utf-8", errors="ignore")
        if hasattr(robotype, "item"):
            try:
                robotype = robotype.item()
            except Exception:
                pass
        if isinstance(robotype, str):
            robotype = robotype.strip()
        return robotype

    def _get_embodiment_id(self, data_dict) -> int:
        robotype = self._parse_robotype(data_dict.get("robotype", None))
        if robotype in self.robotype_to_embodiment_id:
            return int(self.robotype_to_embodiment_id[robotype])
        if isinstance(robotype, str):
            robotype_l = robotype.lower()
            if "agibot" in robotype_l and "agibot" in self.robotype_to_embodiment_id:
                return int(self.robotype_to_embodiment_id["agibot"])
            if "aloha" in robotype_l and "aloha" in self.robotype_to_embodiment_id:
                return int(self.robotype_to_embodiment_id["aloha"])
            if "agilex" in robotype_l and "agilex" in self.robotype_to_embodiment_id:
                return int(self.robotype_to_embodiment_id["agilex"])
        if not self._warned_unknown_robotype:
            print(f"Unknown robotype={robotype!r}, fallback to {self.robotype_default_embodiment_id}")
            self._warned_unknown_robotype = True
        return self.robotype_default_embodiment_id

    def _get_stats_dict(self, embodiment_id: int, data_dict=None):
        if data_dict is not None:
            data_path = data_dict.get("data_path")
            if data_path is not None:
                data_path = os.path.normpath(str(data_path))
                for spec, stats_dict in zip(self.norm_specs, self.stats_dicts):
                    data_paths = spec.get("data_paths")
                    if data_paths is not None and data_path in data_paths:
                        return stats_dict
        if not self.stats_dicts:
            return self.stats_dict
        if 0 <= embodiment_id < len(self.stats_dicts):
            return self.stats_dicts[embodiment_id]
        if not self._warned_unknown_robotype:
            print(f"embodiment_id={embodiment_id} out of range for norm_paths (len={len(self.stats_dicts)}), fallback to 0")
            self._warned_unknown_robotype = True
        return self.stats_dicts[0]

    def _get_delta_mask(self, embodiment_id: int):
        return self.delta_mask_by_embodiment_id.get(int(embodiment_id))

    def _to_nchw_uint8(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[0] in (1, 3):
                x = x[None, ...]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)[None, ...]
            else:
                x = x[None, ...]
        if x.dim() != 4:
            raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype != torch.uint8:
            x_f = x.to(dtype=torch.float32)
            x_max = float(x_f.max().item()) if x_f.numel() > 0 else 0.0
            if x_max <= 1.0:
                x_f = x_f * 255.0
            x = x_f.clamp(0.0, 255.0).to(dtype=torch.uint8)
        return x

    def _process_images(
        self,
        input_images: torch.Tensor,
        dst_width: int,
        dst_height: int,
        *,
        trace: dict | None = None,
        view_key: str | None = None,
    ) -> torch.Tensor:
        view_trace = None
        if trace is not None:
            view_trace = {
                "view_key": str(view_key),
                "input_dtype": str(input_images.dtype),
                "input_shape": [int(dim) for dim in input_images.shape],
                "augmentation_enabled": bool(
                    self.is_train and self.image_augmentation.get("enabled", False)
                ),
                "sampled_augmentation": None,
                "resize_crop": None,
            }
            trace["views"].append(view_trace)
        input_images = input_images.to(dtype=torch.float32) / 255.0
        if self.is_train and self.image_augmentation.get("enabled", False):
            input_images = self._augment_images(
                input_images,
                trace=None if view_trace is None else view_trace,
            )
        height = int(input_images.shape[2])
        width = int(input_images.shape[3])
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        if not self.is_train and self.eval_crop_mode == "center":
            x1 = (new_width - dst_width) // 2
            y1 = (new_height - dst_height) // 2
        else:
            x1 = random.randint(0, new_width - dst_width)
            y1 = random.randint(0, new_height - dst_height)
        if view_trace is not None:
            view_trace["resize_crop"] = {
                "new_height": new_height,
                "new_width": new_width,
                "top": y1,
                "left": x1,
                "height": int(dst_height),
                "width": int(dst_width),
            }
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        input_images = self.normalize(input_images)
        return input_images

    def _augment_images(
        self, input_images: torch.Tensor, *, trace: dict | None = None
    ) -> torch.Tensor:
        """Apply one shared augmentation sample to every temporal frame."""
        cfg = self.image_augmentation
        height, width = input_images.shape[-2:]
        sampled = {
            "crop": None,
            "rotation_degrees": None,
            "jitter_application_order": [],
        }

        crop_scale = float(cfg.get("crop_scale", 1.0))
        if not 0.0 < crop_scale <= 1.0:
            raise ValueError(f"image_augmentation.crop_scale must be in (0, 1], got {crop_scale}")
        if crop_scale < 1.0:
            crop_height = max(1, int(round(height * crop_scale)))
            crop_width = max(1, int(round(width * crop_scale)))
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            sampled["crop"] = {
                "top": top,
                "left": left,
                "height": crop_height,
                "width": crop_width,
            }
            input_images = F.resized_crop(
                input_images,
                top,
                left,
                crop_height,
                crop_width,
                (height, width),
                InterpolationMode.BILINEAR,
                antialias=True,
            )

        rotation_degrees = float(cfg.get("rotation_degrees", 0.0))
        if rotation_degrees > 0.0:
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            sampled["rotation_degrees"] = angle
            input_images = F.rotate(
                input_images,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

        jitter = []
        for name, fn in (
            ("brightness", F.adjust_brightness),
            ("contrast", F.adjust_contrast),
            ("saturation", F.adjust_saturation),
        ):
            strength = float(cfg.get(name, 0.0))
            if strength < 0.0:
                raise ValueError(f"image_augmentation.{name} must be non-negative, got {strength}")
            if strength > 0.0:
                factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
                jitter.append((name, fn, factor))
        random.shuffle(jitter)
        for name, fn, factor in jitter:
            sampled["jitter_application_order"].append(
                {"name": name, "factor": factor}
            )
            input_images = fn(input_images, factor)
        if trace is not None:
            trace["sampled_augmentation"] = sampled
        return input_images.clamp_(0.0, 1.0)

    def _to_prompt_tensor(self, prompt_embed) -> torch.Tensor:
        if isinstance(prompt_embed, np.ndarray):
            prompt_embed = torch.from_numpy(prompt_embed)
        return prompt_embed.to(dtype=torch.float32)

    def _build_prompt_embeds(
        self, data_dict, *, trace: dict | None = None
    ) -> torch.Tensor:
        prompt_embed = data_dict.get("t5_embedding", None)
        assert prompt_embed is not None, "t5_embedding must be provided"
        prompt_embed = self._to_prompt_tensor(prompt_embed)
        subtask_prompt_embed = data_dict.get('subtask_t5_embedding', None)
        subtask_draw = None
        subtask_selected = False
        if self.is_train and subtask_prompt_embed is not None:
            subtask_draw = random.random()
            subtask_selected = subtask_draw < self.subtask_prob
        if subtask_selected:
            subtask_prompt_embed = self._to_prompt_tensor(subtask_prompt_embed)
            prompt_embed = torch.cat([prompt_embed, subtask_prompt_embed], dim=0)
        if trace is not None:
            trace["subtask_prompt"] = {
                "eligible": bool(self.is_train and subtask_prompt_embed is not None),
                "draw": subtask_draw,
                "probability": self.subtask_prob,
                "selected": subtask_selected,
            }
        prompt_embed = prompt_embed[:self.max_prompt_len]

        return torch_F.pad(prompt_embed, (0, 0, 0, self.max_prompt_len - prompt_embed.shape[0]), value=0)

    def __call__(self, data_dict):
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        dst_width, dst_height = self.dst_size

        matched_trace = None
        if self.matched_control_capture:
            sample_identity = build_sample_identity(
                dataset_id=data_dict.get("matched_control_dataset_id"),
                episode_index=data_dict.get("episode_index"),
                frame_index=data_dict.get("frame_index"),
                global_index=data_dict.get("matched_control_global_index"),
            )
            raw_inputs = {
                "pixels": [
                    {"key": key, "value": data_dict[key]}
                    for key in self.view_keys[: self.num_views]
                ],
                "state_action": (
                    {
                        "mode": "ego_dual_eef_sources",
                        "pose": data_dict[str(self.ego_dual_eef_cfg.get("pose_key", "action_eef"))],
                        "delta": data_dict[str(self.ego_dual_eef_cfg.get("delta_key", "action_delta_eef"))],
                        "hand_status": data_dict[str(self.ego_dual_eef_cfg.get("hand_status_key", "hand_status"))],
                        **(
                            {
                                "hand_active_future": data_dict[
                                    self.ego_hand_active_future_key
                                ],
                                "hand_active_transition_pair": data_dict[
                                    self.ego_hand_active_transition_pair_key
                                ],
                                "delta_is_pad": data_dict[
                                    f"{self.ego_dual_eef_cfg.get('delta_key', 'action_delta_eef')}_is_pad"
                                ],
                                "hand_status_is_pad": data_dict[
                                    f"{self.ego_dual_eef_cfg.get('hand_status_key', 'hand_status')}_is_pad"
                                ],
                            }
                            if self.ego_hand_active_future_key is not None
                            else {}
                        ),
                    }
                    if self.ego_dual_eef_cfg is not None
                    else {
                        "mode": "direct_state_action",
                        "state": data_dict[self.state_key],
                        "action": data_dict[self.action_key],
                    }
                ),
                "conditioning": {
                    key: data_dict[key]
                    for key in (
                        "t5_embedding",
                        "subtask_t5_embedding",
                        "contact_target",
                        "contact_loss_mask",
                        "contact_confidence",
                        "contact_condition",
                        "contact_condition_mask",
                        "action_time_mask",
                        "action_loss_mask",
                        "action_loss_weight",
                        f"{self.action_key}_is_pad",
                        "action_is_pad",
                    )
                    if key in data_dict and data_dict[key] is not None
                },
            }
            matched_trace = {
                "trace_version": TRACE_VERSION,
                "views": [],
                "reference_mask": None,
                "subtask_prompt": None,
            }
            matched_identity_json = canonical_json_dumps(sample_identity)
            matched_raw_sha256 = canonical_sha256(raw_inputs)
            matched_diagnostic_path = normalize_path(data_dict.get("data_path", ""))

        if "robotype" not in data_dict:
            raise KeyError("Missing robotype key")
        embodiment_id = self._get_embodiment_id(data_dict)
        stats_dict = self._get_stats_dict(embodiment_id, data_dict=data_dict)

        if self.num_views == 3:
            if self.view_concat_mode == "t":
                cam_high = self._to_nchw_uint8(torch.as_tensor(data_dict[self.view_keys[0]]))
                cam_left = self._to_nchw_uint8(torch.as_tensor(data_dict[self.view_keys[1]]))
                cam_right = self._to_nchw_uint8(torch.as_tensor(data_dict[self.view_keys[2]]))

                top_h = int(dst_height) // 2
                bottom_h = int(dst_height) - top_h
                left_w = int(dst_width) // 2
                right_w = int(dst_width) - left_w

                cam_high = self._process_images(
                    cam_high, dst_width=int(dst_width), dst_height=top_h,
                    trace=matched_trace, view_key=self.view_keys[0]
                )
                cam_left = self._process_images(
                    cam_left, dst_width=left_w, dst_height=bottom_h,
                    trace=matched_trace, view_key=self.view_keys[1]
                )
                cam_right = self._process_images(
                    cam_right, dst_width=right_w, dst_height=bottom_h,
                    trace=matched_trace, view_key=self.view_keys[2]
                )

                cam_bottom = torch.cat([cam_left, cam_right], dim=-1)
                input_images = torch.cat([cam_high, cam_bottom], dim=-2)
            elif self.view_concat_mode == "horizontal":
                views = []
                for k in self.view_keys[:3]:
                    v = data_dict[k]
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)
                    if not isinstance(v, torch.Tensor):
                        raise TypeError(f"Unsupported image type for {k}: {type(v)}")
                    v = self._to_nchw_uint8(v)
                    v = self._process_images(
                        v, dst_width=dst_width, dst_height=dst_height,
                        trace=matched_trace, view_key=k,
                    )
                    views.append(v)
                input_images = torch.cat(views, dim=-1)
            else:
                raise ValueError(f"Unsupported view_concat_mode={self.view_concat_mode!r}")
        else:
            views = []
            for k in self.view_keys[: self.num_views]:
                v = data_dict[k]
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                v = self._to_nchw_uint8(torch.as_tensor(v))
                v = self._process_images(
                    v, dst_width=dst_width, dst_height=dst_height,
                    trace=matched_trace, view_key=k,
                )
                views.append(v)

            if len(views) == 1:
                input_images = views[0]
            else:
                input_images = torch.cat(views, dim=-1)

        data_dict["input_images"] = input_images

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_images"].shape[0])
            if matched_trace is not None:
                matched_trace["reference_mask"] = {
                    "num_ref_frames": int(ref_masks.sum().item()),
                    "num_ref_latents": int(ref_latent_masks.sum().item()),
                }
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = data_dict["input_images"].clone() * ref_masks
            data_dict["input_ref_images"] = ref_images
            data_dict["input_ref_masks"] = ref_latent_masks

        action_is_precomputed_delta = False
        ego_grip_active_mask = None
        ego_grip_transition_pair_mask = None
        ego_source_time_mask = None
        if self.ego_dual_eef_cfg is not None:
            cfg = self.ego_dual_eef_cfg
            pose_key = str(cfg.get("pose_key", "action_eef"))
            delta_key = str(cfg.get("delta_key", "action_delta_eef"))
            hand_status_key = str(cfg.get("hand_status_key", "hand_status"))
            pose = np.asarray(data_dict[pose_key], dtype=np.float32)
            delta = np.asarray(data_dict[delta_key], dtype=np.float32)
            hand_status = np.asarray(data_dict[hand_status_key], dtype=np.float32)
            if self.ego_grip_semantics is not None:
                hand_status = convert_hand_status_semantics(
                    hand_status,
                    source_semantics=self.ego_grip_semantics,
                    target_semantics=CANONICAL_OPEN_FRACTION_SEMANTICS,
                )
            chunk_start_offset = int(cfg.get("chunk_start_offset", 0))
            if chunk_start_offset < 0:
                raise ValueError("ego_dual_eef_cfg.chunk_start_offset must be non-negative")
            state_hand_status = hand_status[0] if hand_status.ndim == 2 else hand_status
            if self.ego_hand_active_future_key is not None:
                delta_pad_key = f"{delta_key}_is_pad"
                hand_pad_key = f"{hand_status_key}_is_pad"
                if delta_pad_key not in data_dict or hand_pad_key not in data_dict:
                    raise KeyError(
                        "production Ego action chunks require both source pad masks: "
                        f"{delta_pad_key}, {hand_pad_key}"
                    )
                ego_source_time_mask = torch.from_numpy(
                    build_future_action_time_mask(
                        data_dict[delta_pad_key],
                        data_dict[hand_pad_key],
                        source_frames=int(delta.shape[0]),
                        chunk_start_offset=chunk_start_offset,
                        expected_future_frames=int(self.num_frames),
                    )
                )
            if chunk_start_offset:
                if delta.ndim != 2 or hand_status.ndim != 2:
                    raise ValueError(
                        "non-zero ego chunk_start_offset requires chunked delta and hand status"
                    )
                delta = delta[chunk_start_offset:]
                hand_status = hand_status[chunk_start_offset:]
            state = torch.from_numpy(dual_eef_pose_to_state(pose, state_hand_status))
            action = torch.from_numpy(dual_eef_delta_to_action(delta, hand_status))
            action_is_precomputed_delta = True
            if self.ego_hand_active_future_key is not None:
                if ego_source_time_mask.numel() != int(action.shape[0]):
                    raise ValueError(
                        "Ego f+1 pad mask does not align to the future action chunk"
                    )
                if self.ego_hand_active_future_key not in data_dict:
                    raise KeyError(
                        f"missing required Ego hand-active mask: {self.ego_hand_active_future_key}"
                    )
                if self.ego_hand_active_transition_pair_key not in data_dict:
                    raise KeyError(
                        "missing required Ego hand-active transition-pair mask: "
                        f"{self.ego_hand_active_transition_pair_key}"
                    )
                ego_grip_active_mask = torch.as_tensor(
                    data_dict[self.ego_hand_active_future_key],
                    dtype=torch.bool,
                )
                ego_grip_transition_pair_mask = torch.as_tensor(
                    data_dict[self.ego_hand_active_transition_pair_key],
                    dtype=torch.bool,
                )
                expected_grip_shape = (int(action.shape[0]), 2)
                if (
                    tuple(ego_grip_active_mask.shape) != expected_grip_shape
                    or tuple(ego_grip_transition_pair_mask.shape)
                    != expected_grip_shape
                ):
                    raise ValueError(
                        "Ego hand-active masks must align to the future action chunk: "
                        f"expected {expected_grip_shape}, got "
                        f"{tuple(ego_grip_active_mask.shape)} and "
                        f"{tuple(ego_grip_transition_pair_mask.shape)}"
                    )
        else:
            action = data_dict[self.action_key]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            state = data_dict[self.state_key]
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)

        action = action.to(dtype=torch.float32)
        state = state.to(dtype=torch.float32)

        if action.dim() == 1:
            action = action[None, :]
        if state.dim() == 1:
            state = state[None, :]
        if state.dim() == 2 and state.shape[0] > 1:
            state = state[:1]

        source_time_mask = data_dict.get("action_time_mask", data_dict.get("action_loss_mask"))
        if ego_source_time_mask is not None:
            if source_time_mask is not None:
                explicit_time_mask = torch.as_tensor(
                    source_time_mask,
                    dtype=torch.bool,
                ).flatten()
                if not torch.equal(explicit_time_mask.cpu(), ego_source_time_mask.cpu()):
                    raise ValueError(
                        "explicit Ego action_time_mask differs from source pad masks"
                    )
            source_time_mask = ego_source_time_mask
        elif source_time_mask is None:
            is_pad = data_dict.get(f"{self.action_key}_is_pad", data_dict.get("action_is_pad"))
            if is_pad is not None:
                source_time_mask = ~torch.as_tensor(is_pad, dtype=torch.bool)
        if source_time_mask is None:
            action_time_mask = torch.ones(action.shape[0], dtype=torch.bool, device=action.device)
        else:
            action_time_mask = torch.as_tensor(
                source_time_mask,
                dtype=torch.bool,
                device=action.device,
            ).flatten()
            if action_time_mask.numel() != action.shape[0]:
                raise ValueError(
                    f"action_time_mask length {action_time_mask.numel()} does not match "
                    f"action chunk length {action.shape[0]}"
                )

        source_action_weight = data_dict.get("action_loss_weight")
        if source_action_weight is None:
            action_loss_weight = torch.ones_like(action)
        else:
            action_loss_weight = torch.as_tensor(
                source_action_weight,
                dtype=torch.float32,
                device=action.device,
            )
            if action_loss_weight.ndim == 1:
                action_loss_weight = (
                    action_loss_weight.unsqueeze(0).expand(action.shape[0], -1).clone()
                )
            if action_loss_weight.shape != action.shape:
                raise ValueError(
                    f"action_loss_weight shape {tuple(action_loss_weight.shape)} does not match "
                    f"action shape {tuple(action.shape)}"
                )
        if self.ego_dual_eef_cfg is not None and action_loss_weight.shape[-1] >= 20:
            if ego_grip_active_mask is not None:
                multiplier = ego_grip_loss_weight_multiplier(
                    ego_grip_active_mask.cpu().numpy(),
                    action_dim=int(action_loss_weight.shape[-1]),
                    active_weight=self.ego_grip_proxy_weight,
                )
                action_loss_weight *= torch.as_tensor(
                    multiplier,
                    dtype=action_loss_weight.dtype,
                    device=action_loss_weight.device,
                )
            else:
                action_loss_weight[:, 18:20] *= self.ego_grip_proxy_weight

        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
                action_time_mask = action_time_mask[:t]
                action_loss_weight = action_loss_weight[:t]
                if ego_grip_active_mask is not None:
                    ego_grip_active_mask = ego_grip_active_mask[:t]
                    ego_grip_transition_pair_mask = ego_grip_transition_pair_mask[:t]
            else:
                pad = torch.zeros((t - cur_t, action.shape[1]), dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=0)
                action_time_mask = torch.cat(
                    [
                        action_time_mask,
                        torch.zeros(t - cur_t, dtype=torch.bool, device=action.device),
                    ],
                    dim=0,
                )
                action_loss_weight = torch.cat(
                    [
                        action_loss_weight,
                        torch.zeros(
                            (t - cur_t, action_loss_weight.shape[1]),
                            dtype=action_loss_weight.dtype,
                            device=action.device,
                        ),
                    ],
                    dim=0,
                )
                if ego_grip_active_mask is not None:
                    ego_grip_active_mask = torch.cat(
                        (
                            ego_grip_active_mask,
                            torch.zeros((t - cur_t, 2), dtype=torch.bool),
                        ),
                        dim=0,
                    )
                    ego_grip_transition_pair_mask = torch.cat(
                        (
                            ego_grip_transition_pair_mask,
                            torch.zeros((t - cur_t, 2), dtype=torch.bool),
                        ),
                        dim=0,
                    )

        base = self._get_delta_mask(embodiment_id)
        assert base is not None, f"embodiment_id {embodiment_id} not found in delta_mask_by_embodiment_id"
        effective_dim = int(base.shape[-1])
        if self.require_exact_effective_dim:
            if state.shape[-1] != effective_dim or action.shape[-1] != effective_dim:
                raise ValueError(
                    f"embodiment {embodiment_id} requires exact physical dim {effective_dim}; "
                    f"got state={state.shape[-1]} action={action.shape[-1]}. "
                    "Legacy SONIC 30-D data cannot be truncated into the asymmetric 25-D schema."
                )
        state = state[:, :effective_dim]
        action = action[:, :effective_dim]
        action_loss_weight = action_loss_weight[:, :effective_dim]

        assert self.model_action_dim is not None, "model_action_dim must be provided"
        d = int(self.model_action_dim)
        if state.shape[-1] > d:
            state = state[..., :d]
        if state.shape[-1] < d:
            state = torch_F.pad(state, (0, d - int(state.shape[-1])), value=0.0)
        if action.shape[-1] > d:
            action = action[..., :d]
        if action.shape[-1] < d:
            action = torch_F.pad(action, (0, d - int(action.shape[-1])), value=0.0)
        if action_loss_weight.shape[-1] < d:
            action_loss_weight = torch_F.pad(
                action_loss_weight,
                (0, d - int(action_loss_weight.shape[-1])),
                value=0.0,
            )
        elif action_loss_weight.shape[-1] > d:
            action_loss_weight = action_loss_weight[..., :d]

        if d > len(base):
            base = np.pad(base, (0, d - len(base)), constant_values=False)
        else:
            base = base[:d]

        mask_t = torch.as_tensor(base, dtype=torch.bool, device=action.device)
        idx = torch.nonzero(mask_t, as_tuple=False).flatten()
        delta = action.clone()
        if idx.numel() > 0 and not action_is_precomputed_delta:
            delta[:, idx] = action[:, idx] - state[:, idx]

        def _to_1d(x, device):
            return torch.as_tensor(x, dtype=torch.float32, device=device).flatten()

        def _normalize_feature(x, stats, eff_dim: int, device):
            out = torch.zeros_like(x)
            eps = 1e-8
            if self.norm_use_quantiles:
                low = _to_1d(stats["q01"], device)
                high = _to_1d(stats["q99"], device)
                if eff_dim > 0:
                    rng = (high[:eff_dim] - low[:eff_dim]).clamp_min(eps)
                    out[..., :eff_dim] = (x[..., :eff_dim] - low[:eff_dim]) / rng * 2.0 - 1.0
            else:
                mean = _to_1d(stats["mean"], device)
                std = _to_1d(stats["std"], device)
                if eff_dim > 0:
                    out[..., :eff_dim] = (x[..., :eff_dim] - mean[:eff_dim]) / std[:eff_dim].clamp_min(eps)
            if self.norm_enable_clamp:
                out = out.clamp(-1.0, 1.0)
            return out

        norm_stats = stats_dict["norm_stats"]
        norm_state = _normalize_feature(
            state,
            norm_stats["observation.state"],
            eff_dim=effective_dim,
            device=state.device,
        )
        norm_delta = _normalize_feature(
            delta,
            norm_stats["action"],
            eff_dim=effective_dim,
            device=action.device,
        )

        prompt_embeds = self._build_prompt_embeds(data_dict, trace=matched_trace)

        out = {}
        out["fps"] = torch.tensor(self.fps, dtype=torch.float32)
        out["images"] = data_dict["input_images"]
        out["ref_images"] = data_dict.get("input_ref_images", None)
        out["ref_masks"] = data_dict.get("input_ref_masks", None)
        out["prompt_embeds"] = prompt_embeds
        out["action"] = norm_delta
        out["state"] = norm_state
        out["embodiment_id"] = torch.tensor(int(embodiment_id), dtype=torch.long)
        valid_dim_mask = torch.zeros(d, dtype=torch.bool, device=action.device)
        valid_dim_mask[:effective_dim] = True
        out["action_dim_mask"] = valid_dim_mask
        out["action_time_mask"] = action_time_mask
        # Keep the existing trainer-facing name while exposing the explicit
        # contract requested by new datasets.
        out["action_loss_mask"] = action_time_mask
        out["action_loss_weight"] = (
            action_loss_weight
            * action_time_mask[:, None].to(dtype=action_loss_weight.dtype)
            * valid_dim_mask[None, :].to(dtype=action_loss_weight.dtype)
        )
        if ego_grip_transition_pair_mask is not None:
            # Record-only.  Stage1 currently has no transition auxiliary loss;
            # any future explicit ablation must use both-adjacent-frame validity.
            out["ego_grip_transition_loss_mask"] = (
                ego_grip_transition_pair_mask.to(device=action.device)
                & action_time_mask[:, None]
            )
        if matched_trace is not None:
            out[SAMPLE_IDENTITY_KEY] = matched_identity_json
            out[SAMPLE_DATA_PATH_KEY] = matched_diagnostic_path
            out[SAMPLE_RAW_HASH_KEY] = matched_raw_sha256
            out[SAMPLE_AUGMENTATION_KEY] = canonical_json_dumps(matched_trace)
        dim = out["action"].shape[-1]
        if self.emit_action_denorm_scale:
            scale = torch.ones(dim, dtype=torch.float32, device=action.device)
            action_stats = norm_stats["action"]
            eps = 1e-8
            if self.norm_use_quantiles:
                low = _to_1d(action_stats["q01"], action.device)
                high = _to_1d(action_stats["q99"], action.device)
                
                scale[:effective_dim] = (high[:effective_dim] - low[:effective_dim]).clamp_min(eps) / 2.0
            else:
                std = _to_1d(action_stats["std"], action.device)
                scale[:effective_dim] = std[:effective_dim].clamp_min(eps)
            out["action_denorm_scale"] = scale

        # Optional contact-transition sidecars are already aligned to the
        # action chunk by the dataset.  Keep them geometry-free and pass them
        # through without action normalization; confidence/masks are consumed
        # directly by CasualWATrainerMoT.
        for key in (
            "contact_target",
            "contact_loss_mask",
            "contact_confidence",
            "contact_condition",
            "contact_condition_mask",
        ):
            if key in data_dict and data_dict[key] is not None:
                out[key] = torch.as_tensor(data_dict[key], dtype=torch.float32)

        keys = list(out.keys())
        for k in keys:
            if out[k] is None:
                out.pop(k)

        return out

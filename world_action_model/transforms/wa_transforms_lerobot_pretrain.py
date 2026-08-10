import json
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
)
from world_action_model.asymmetric_action_schema import ASYMMETRIC_SCHEMA_ID


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
        require_exact_effective_dim=False,
        expected_action_schema_id=None,
        expected_hardware_contract_sha256=None,
        expected_calibration_capture_sha256=None,
        ego_grip_proxy_weight=0.25,
        task_key="task",
        emit_action_denorm_scale=False,
        max_prompt_len=64,
        subtask_prob=0,
    ):
        if norm_path is None:
            raise ValueError("norm_path list is none")

        if isinstance(norm_path, (str, os.PathLike)):
            norm_specs = [{"path": str(norm_path), "data_paths": None}]
        else:
            norm_specs = []
            for item in norm_path:
                if isinstance(item, (str, os.PathLike)):
                    norm_specs.append({"path": str(item), "data_paths": None})
                elif isinstance(item, dict):
                    path = item.get("path")
                    if path is None:
                        raise ValueError(f"norm_path entry missing path: {item}")
                    data_paths = item.get("data_paths", None)
                    if data_paths is not None:
                        data_paths = {os.path.normpath(str(p)) for p in data_paths}
                    norm_specs.append({"path": str(path), "data_paths": data_paths})
                else:
                    raise TypeError(f"Unsupported norm_path entry type: {type(item).__name__}")
            if len(norm_specs) == 0:
                raise ValueError("norm_path list is empty")
        norm_paths = [spec["path"] for spec in norm_specs]

        super().__init__(
            is_train=is_train,
            dst_size=dst_size,
            num_frames=num_frames,
            fps=fps,
            norm_path=norm_paths[0],
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
        self.stats_dicts = []
        for json_path in self.norm_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                self.stats_dicts.append(json.load(f))
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

    def _process_images(self, input_images: torch.Tensor, dst_width: int, dst_height: int) -> torch.Tensor:
        input_images = input_images.to(dtype=torch.float32) / 255.0
        if self.is_train and self.image_augmentation.get("enabled", False):
            input_images = self._augment_images(input_images)
        height = int(input_images.shape[2])
        width = int(input_images.shape[3])
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        input_images = self.normalize(input_images)
        return input_images

    def _augment_images(self, input_images: torch.Tensor) -> torch.Tensor:
        """Apply one shared augmentation sample to every temporal frame."""
        cfg = self.image_augmentation
        height, width = input_images.shape[-2:]

        crop_scale = float(cfg.get("crop_scale", 1.0))
        if not 0.0 < crop_scale <= 1.0:
            raise ValueError(f"image_augmentation.crop_scale must be in (0, 1], got {crop_scale}")
        if crop_scale < 1.0:
            crop_height = max(1, int(round(height * crop_scale)))
            crop_width = max(1, int(round(width * crop_scale)))
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
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
                jitter.append((fn, factor))
        random.shuffle(jitter)
        for fn, factor in jitter:
            input_images = fn(input_images, factor)
        return input_images.clamp_(0.0, 1.0)

    def _to_prompt_tensor(self, prompt_embed) -> torch.Tensor:
        if isinstance(prompt_embed, np.ndarray):
            prompt_embed = torch.from_numpy(prompt_embed)
        return prompt_embed.to(dtype=torch.float32)

    def _build_prompt_embeds(self, data_dict) -> torch.Tensor:
        prompt_embed = data_dict.get("t5_embedding", None)
        assert prompt_embed is not None, "t5_embedding must be provided"
        prompt_embed = self._to_prompt_tensor(prompt_embed)
        subtask_prompt_embed = data_dict.get('subtask_t5_embedding', None)
        if (
            self.is_train
            and subtask_prompt_embed is not None
            and random.random() < self.subtask_prob
        ):
            subtask_prompt_embed = self._to_prompt_tensor(subtask_prompt_embed)
            prompt_embed = torch.cat([prompt_embed, subtask_prompt_embed], dim=0)
        prompt_embed = prompt_embed[:self.max_prompt_len]

        return torch_F.pad(prompt_embed, (0, 0, 0, self.max_prompt_len - prompt_embed.shape[0]), value=0)

    def __call__(self, data_dict):
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        dst_width, dst_height = self.dst_size

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

                cam_high = self._process_images(cam_high, dst_width=int(dst_width), dst_height=top_h)
                cam_left = self._process_images(cam_left, dst_width=left_w, dst_height=bottom_h)
                cam_right = self._process_images(cam_right, dst_width=right_w, dst_height=bottom_h)

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
                    v = self._process_images(v, dst_width=dst_width, dst_height=dst_height)
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
                v = self._process_images(v, dst_width=dst_width, dst_height=dst_height)
                views.append(v)

            if len(views) == 1:
                input_images = views[0]
            else:
                input_images = torch.cat(views, dim=-1)

        data_dict["input_images"] = input_images

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_images"].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = data_dict["input_images"].clone() * ref_masks
            data_dict["input_ref_images"] = ref_images
            data_dict["input_ref_masks"] = ref_latent_masks

        action_is_precomputed_delta = False
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
        if source_time_mask is None:
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
            action_loss_weight[:, 18:20] *= self.ego_grip_proxy_weight

        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
                action_time_mask = action_time_mask[:t]
                action_loss_weight = action_loss_weight[:t]
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

        prompt_embeds = self._build_prompt_embeds(data_dict)

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

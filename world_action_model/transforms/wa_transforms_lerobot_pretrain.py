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
        task_key="task",
        emit_action_denorm_scale=False,
        max_prompt_len=64,
        subtask_prob=0,
        state_history_tokens=1,
        state_history_dropout_prob=0.0,
        state_condition_dropout_prob=0.0,
        append_episode_progress_token=False,
        append_state_velocity_token=False,
        emit_action_dim_mask=False,
        emit_action_time_mask=False,
        expected_state_dim=None,
        jitter_score_key=None,
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
        self.task_key = task_key
        self.emit_action_denorm_scale = bool(emit_action_denorm_scale)
        self.max_prompt_len = int(max_prompt_len)
        self.subtask_prob = float(subtask_prob)
        self.state_history_tokens = int(state_history_tokens)
        if self.state_history_tokens < 1:
            raise ValueError("state_history_tokens must be at least 1")
        self.state_history_dropout_prob = float(state_history_dropout_prob)
        if not 0.0 <= self.state_history_dropout_prob <= 1.0:
            raise ValueError("state_history_dropout_prob must be in [0, 1]")
        self.state_condition_dropout_prob = float(state_condition_dropout_prob)
        if not 0.0 <= self.state_condition_dropout_prob <= 1.0:
            raise ValueError("state_condition_dropout_prob must be in [0, 1]")
        self.append_episode_progress_token = bool(append_episode_progress_token)
        self.append_state_velocity_token = bool(append_state_velocity_token)
        self.emit_action_dim_mask = bool(emit_action_dim_mask)
        self.emit_action_time_mask = bool(emit_action_time_mask)
        self.expected_state_dim = (
            None if expected_state_dim is None else int(expected_state_dim)
        )
        self.jitter_score_key = (
            None if jitter_score_key is None else str(jitter_score_key)
        )
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
            raise ValueError(
                "image_augmentation.crop_scale must be in (0, 1], "
                f"got {crop_scale}"
            )
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
            input_images = F.rotate(
                input_images,
                random.uniform(-rotation_degrees, rotation_degrees),
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
                raise ValueError(
                    f"image_augmentation.{name} must be non-negative, "
                    f"got {strength}"
                )
            if strength > 0.0:
                factor = random.uniform(
                    max(0.0, 1.0 - strength),
                    1.0 + strength,
                )
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
        if state.dim() != 2:
            raise ValueError(f"Expected state with shape [history, dim], got {tuple(state.shape)}")
        if self.expected_state_dim is not None and state.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Expected {self.expected_state_dim}-D state, got {state.shape[-1]}. "
                "The IMU ablation requires an IMU-augmented dataset."
            )
        if state.shape[0] > self.state_history_tokens:
            state = state[-self.state_history_tokens :]
        elif state.shape[0] < self.state_history_tokens:
            pad = state[:1].expand(self.state_history_tokens - state.shape[0], -1)
            state = torch.cat([pad, state], dim=0)

        action_valid = torch.ones(action.shape[0], dtype=torch.bool, device=action.device)
        if self.emit_action_time_mask and "action_is_pad" in data_dict:
            action_is_pad = torch.as_tensor(data_dict["action_is_pad"], device=action.device).flatten().bool()
            action_valid = ~action_is_pad[: action.shape[0]]

        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
                action_valid = action_valid[:t]
            else:
                pad = torch.zeros((t - cur_t, action.shape[1]), dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=0)
                action_valid = torch.cat(
                    [action_valid, torch.zeros(t - cur_t, dtype=torch.bool, device=action.device)]
                )
                
        base = self._get_delta_mask(embodiment_id)
        assert base is not None, f"embodiment_id {embodiment_id} not found in delta_mask_by_embodiment_id"
        assert self.model_action_dim is not None, "model_action_dim must be provided"
        d = int(self.model_action_dim)
        effective_action_dim = min(int(base.shape[-1]), int(action.shape[-1]), d)
        effective_state_dim = min(int(state.shape[-1]), d)
        state = state[:, :effective_state_dim]
        action = action[:, :effective_action_dim]
        if state.shape[-1] > d:
            state = state[..., :d]
        if state.shape[-1] < d:
            state = torch_F.pad(state, (0, d - int(state.shape[-1])), value=0.0)
        if action.shape[-1] > d:
            action = action[..., :d]
        if action.shape[-1] < d:
            action = torch_F.pad(action, (0, d - int(action.shape[-1])), value=0.0)

        if d > len(base):
            base = np.pad(base, (0, d - len(base)), constant_values=False)
        else:
            base = base[:d]

        mask_t = torch.as_tensor(base, dtype=torch.bool, device=action.device)
        idx = torch.nonzero(mask_t, as_tuple=False).flatten()
        delta = action.clone()
        if idx.numel() > 0:
            delta[:, idx] = action[:, idx] - state[-1:, idx]

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
            eff_dim=effective_state_dim,
            device=state.device,
        )
        norm_delta = _normalize_feature(
            delta,
            norm_stats["action"],
            eff_dim=effective_action_dim,
            device=action.device,
        )

        if self.is_train and self.state_history_dropout_prob > 0.0 and norm_state.shape[0] > 1:
            drop = torch.rand(norm_state.shape[0] - 1, device=norm_state.device) < self.state_history_dropout_prob
            norm_state[:-1] = torch.where(drop[:, None], norm_state[-1:], norm_state[:-1])

        # Drop only the model's proprioceptive condition.  The action target
        # (including delta conversion above) has already been computed from the
        # real current state, so this cannot corrupt supervision.  Keeping the
        # token count fixed also preserves attention layouts and inference
        # checkpoint compatibility while forcing a subset of samples through
        # the visual path.
        if (
            self.is_train
            and self.state_condition_dropout_prob > 0.0
            and torch.rand((), device=norm_state.device)
            < self.state_condition_dropout_prob
        ):
            norm_state.zero_()

        state_tokens = [norm_state]
        if self.append_state_velocity_token:
            state_tokens.append(norm_state[-1:] - norm_state[:1])
        if self.append_episode_progress_token:
            progress = torch.as_tensor(
                data_dict.get("episode_progress", 0.0),
                dtype=norm_state.dtype,
                device=norm_state.device,
            ).clamp(0.0, 1.0)
            phase = torch.zeros((1, d), dtype=norm_state.dtype, device=norm_state.device)
            phase[0, 0] = progress * 2.0 - 1.0
            if d > 1:
                phase[0, 1] = torch.sin(progress * (2.0 * torch.pi))
            if d > 2:
                phase[0, 2] = torch.cos(progress * (2.0 * torch.pi))
            state_tokens.append(phase)
        norm_state = torch.cat(state_tokens, dim=0)

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
        if self.emit_action_dim_mask:
            action_dim_mask = torch.zeros(d, dtype=torch.bool, device=action.device)
            action_dim_mask[:effective_action_dim] = True
            out["action_dim_mask"] = action_dim_mask
        if self.emit_action_time_mask:
            out["action_time_mask"] = action_valid
        if self.jitter_score_key is not None:
            if self.jitter_score_key not in data_dict:
                raise KeyError(
                    f"Missing configured jitter score key {self.jitter_score_key!r}"
                )
            out["visual_jitter_score"] = torch.as_tensor(
                data_dict[self.jitter_score_key], dtype=torch.float32
            ).reshape(-1)
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

        keys = list(out.keys())
        for k in keys:
            if out[k] is None:
                out.pop(k)

        return out

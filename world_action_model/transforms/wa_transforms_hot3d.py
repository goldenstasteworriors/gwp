from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as torch_F
from giga_train import TRANSFORMS
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from world_action_model.transforms.wa_transforms import WATransforms


@TRANSFORMS.register
class WAHot3DTransforms(WATransforms):
    """Image processing and masked 53-D normalization for HOT3D windows."""

    def __init__(
        self,
        norm_path: str,
        dst_size=(320, 384),
        fps: int = 30,
        image_cfg=None,
        max_prompt_len: int = 64,
        embodiment_id: int = 0,
    ) -> None:
        super().__init__(
            is_train=True,
            dst_size=dst_size,
            num_frames=48,
            fps=fps,
            norm_path=norm_path,
            image_cfg=image_cfg,
            num_views=1,
        )
        with open(norm_path, "r", encoding="utf-8") as handle:
            self.stats = json.load(handle)["norm_stats"]
        self.max_prompt_len = int(max_prompt_len)
        self.embodiment_id = int(embodiment_id)

    def _process_images(self, images: torch.Tensor) -> torch.Tensor:
        dst_width, dst_height = (int(x) for x in self.dst_size)
        images = images.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        height, width = int(images.shape[-2]), int(images.shape[-1])
        scale = max(float(dst_height) / height, float(dst_width) / width)
        new_height, new_width = int(round(height * scale)), int(round(width * scale))
        images = F.resize(images, (new_height, new_width), InterpolationMode.BILINEAR)
        top = random.randint(0, new_height - dst_height)
        left = random.randint(0, new_width - dst_width)
        images = F.crop(images, top, left, dst_height, dst_width)
        return self.normalize(images)

    @staticmethod
    def _normalize(values: torch.Tensor, stats: dict, mask: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=values.device)
        std = torch.as_tensor(stats["std"], dtype=torch.float32, device=values.device).clamp_min(1e-6)
        normalized = (values - mean) / std
        return normalized * mask.to(dtype=normalized.dtype)

    def __call__(self, data_dict: dict) -> dict:
        images = self._process_images(torch.as_tensor(data_dict["observation.images.ego_view"]))
        ref_masks, ref_latent_masks = self.mask_generator.get_mask(images.shape[0])
        state = torch.as_tensor(data_dict["observation.state"], dtype=torch.float32)
        action = torch.as_tensor(data_dict["action"], dtype=torch.float32)
        state_dim_mask = torch.as_tensor(data_dict["state_dim_mask"], dtype=torch.bool)
        action_dim_mask = torch.as_tensor(data_dict["action_dim_mask"], dtype=torch.bool)
        prompt = torch.as_tensor(data_dict["t5_embedding"], dtype=torch.float32)[: self.max_prompt_len]
        prompt = torch_F.pad(prompt, (0, 0, 0, self.max_prompt_len - prompt.shape[0]))
        out = {
            "fps": torch.tensor(self.fps, dtype=torch.float32),
            "images": images,
            "ref_images": images.clone() * ref_masks[:, None, None, None],
            "ref_masks": ref_latent_masks[None, :, None, None],
            "prompt_embeds": prompt,
            "state": self._normalize(state, self.stats["observation.state"], state_dim_mask),
            "action": self._normalize(action, self.stats["action"], action_dim_mask),
            "state_dim_mask": state_dim_mask,
            "action_dim_mask": action_dim_mask,
            "action_loss_mask": torch.as_tensor(data_dict["action_loss_mask"], dtype=torch.bool),
            "embodiment_id": torch.tensor(self.embodiment_id, dtype=torch.long),
        }
        for key in (
            "contact_target",
            "contact_loss_mask",
            "contact_confidence",
            "contact_condition",
            "contact_condition_mask",
        ):
            if key in data_dict and data_dict[key] is not None:
                out[key] = torch.as_tensor(data_dict[key], dtype=torch.float32)
        return out

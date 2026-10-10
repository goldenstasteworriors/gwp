"""View-consistent camera rotation augmentation for camera-action WAM data."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as torch_functional
from giga_train import TRANSFORMS
from scipy.spatial.transform import Rotation

from world_action_model.camera_centric import matrix_to_rot6d, rot6d_to_matrix
from world_action_model.transforms.wa_transforms_lerobot_pretrain import (
    WALeRobotTransformsPretrain,
)


def rotate_camera_action(action: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    """Express camera-frame 30-D relative actions in a rotated virtual camera."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 30:
        raise ValueError(f"expected [T,30] camera action, got {action.shape}")
    result = action.copy()
    for start in (0, 9):
        result[:, start : start + 3] = (old_to_new @ action[:, start : start + 3].T).T
        matrices = rot6d_to_matrix(action[:, start + 3 : start + 9])
        result[:, start + 3 : start + 9] = matrix_to_rot6d(
            old_to_new @ matrices @ old_to_new.T
        )
    return result


def rotate_camera_video(
    images: torch.Tensor,
    old_to_new: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    reference_width: int,
    reference_height: int,
) -> torch.Tensor:
    """Warp every frame with the same pure-rotation homography K R^-1 K^-1."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected NCHW RGB video, got {tuple(images.shape)}")
    frames, _, height, width = images.shape
    sx, sy = width / reference_width, height / reference_height
    fx, fy = fx * sx, fy * sy
    cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=images.device),
        torch.arange(width, dtype=torch.float32, device=images.device),
        indexing="ij",
    )
    rays = torch.stack(((xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)), dim=-1)
    new_to_old = torch.as_tensor(old_to_new.T.copy(), dtype=torch.float32, device=images.device)
    old_rays = rays @ new_to_old.T
    z = old_rays[..., 2].clamp_min(1e-6)
    old_x = fx * old_rays[..., 0] / z + cx
    old_y = fy * old_rays[..., 1] / z + cy
    grid = torch.stack((2 * old_x / (width - 1) - 1, 2 * old_y / (height - 1) - 1), dim=-1)
    warped = torch_functional.grid_sample(
        images.float(), grid[None].expand(frames, -1, -1, -1),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return warped.round().clamp(0, 255).to(dtype=images.dtype)


@TRANSFORMS.register
class WACamVLATransforms(WALeRobotTransformsPretrain):
    """Apply a sampled camera pose to both video and camera-frame action."""

    def __init__(self, camera_view_augmentation: dict, **kwargs):
        super().__init__(**kwargs)
        if self.num_views != 1:
            raise ValueError("WACamVLATransforms currently supports one head camera")
        self.camera_view_augmentation = dict(camera_view_augmentation)
        required = {"fx", "fy", "cx", "cy", "width", "height", "max_rotation_degrees"}
        if not required.issubset(self.camera_view_augmentation):
            raise ValueError(f"camera_view_augmentation missing {sorted(required - set(self.camera_view_augmentation))}")
        if float(self.camera_view_augmentation["max_rotation_degrees"]) < 0:
            raise ValueError("max_rotation_degrees must be non-negative")

    def __call__(self, data_dict):
        if not self.is_train:
            return super().__call__(data_dict)
        cfg = self.camera_view_augmentation
        limit = float(cfg["max_rotation_degrees"])
        angles_deg = np.asarray([random.uniform(-limit, limit) for _ in range(3)])
        old_to_new = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
        key = self.view_keys[0]
        images = self._to_nchw_uint8(torch.as_tensor(data_dict[key]))
        result = dict(data_dict)
        result[key] = rotate_camera_video(
            images, old_to_new,
            fx=float(cfg["fx"]), fy=float(cfg["fy"]),
            cx=float(cfg["cx"]), cy=float(cfg["cy"]),
            reference_width=int(cfg["width"]), reference_height=int(cfg["height"]),
        )
        result[self.action_key] = rotate_camera_action(
            np.asarray(data_dict[self.action_key]), old_to_new
        )
        transformed = super().__call__(result)
        transformed["camera_view_delta"] = torch.as_tensor(
            np.deg2rad(angles_deg), dtype=torch.float32
        )
        return transformed

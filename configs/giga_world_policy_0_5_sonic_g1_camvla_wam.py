"""CamVLA-inspired SONIC G1 camera-action WAM finetuning experiment.

Use only with a 30-D dataset relabeled by convert_sonic_camera_actions.py.
The camera-to-base rotation in that dataset is an explicit calibration input.
"""

import copy
import os

from configs.giga_world_policy_0_5_sonic_g1_finetune import config as base_config


config = copy.deepcopy(base_config)
config["dataloaders"]["train"]["transform"]["type"] = "WACamVLATransforms"
config["dataloaders"]["train"]["transform"]["camera_view_augmentation"] = dict(
    # Measured D435i color intrinsics from SONIC's FK camera tuner.  Replace
    # these values if the video profile/camera differs.
    fx=float(os.environ.get("GWP_CAM_FX", "607.061")),
    fy=float(os.environ.get("GWP_CAM_FY", "607.014")),
    cx=float(os.environ.get("GWP_CAM_CX", "331.952")),
    cy=float(os.environ.get("GWP_CAM_CY", "246.989")),
    width=int(os.environ.get("GWP_CAM_WIDTH", "640")),
    height=int(os.environ.get("GWP_CAM_HEIGHT", "480")),
    max_rotation_degrees=float(os.environ.get("GWP_CAM_MAX_ROTATION_DEG", "5")),
)
# Cropping and in-plane rotation without a matching action relabel would
# introduce extra geometry noise.  Keep appearance jitter only here.
config["dataloaders"]["train"]["transform"]["image_augmentation"] = dict(
    enabled=True,
    crop_scale=1.0,
    rotation_degrees=0.0,
    brightness=0.15,
    contrast=0.15,
    saturation=0.15,
)
config["models"]["camera_pose_loss_weight"] = float(
    os.environ.get("GWP_CAM_POSE_LOSS_WEIGHT", "0.05")
)

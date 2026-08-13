"""Frozen latent-to-mask probe used to supervise the actual video output path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn


class LatentRobotMaskProbe(nn.Module):
    """Decode standardized clean video latents into per-frame robot-mask logits."""

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "model.safetensors"

    def __init__(
        self,
        latent_channels: int,
        frames_per_latent: int = 4,
        spatial_scale: int = 8,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.frames_per_latent = int(frames_per_latent)
        self.spatial_scale = int(spatial_scale)
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if self.frames_per_latent <= 0:
            raise ValueError("frames_per_latent must be positive")
        if self.spatial_scale <= 0:
            raise ValueError("spatial_scale must be positive")
        self.projection = nn.Conv3d(
            self.latent_channels,
            self.frames_per_latent * self.spatial_scale**2,
            kernel_size=1,
        )

    def forward(self, clean_latents: torch.Tensor) -> torch.Tensor:
        if clean_latents.ndim != 5:
            raise ValueError(
                "Expected clean latents with shape [B,C,T,H,W], got "
                f"{tuple(clean_latents.shape)}"
            )
        if clean_latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got "
                f"{clean_latents.shape[1]}"
            )
        patches = self.projection(clean_latents)
        batch, _, latent_frames, height, width = patches.shape
        patches = patches.reshape(
            batch,
            self.frames_per_latent,
            self.spatial_scale,
            self.spatial_scale,
            latent_frames,
            height,
            width,
        )
        # Time-major physical frames followed by the two pixel-shuffle axes.
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3)
        return patches.reshape(
            batch,
            latent_frames * self.frames_per_latent,
            height * self.spatial_scale,
            width * self.spatial_scale,
        )

    def save_pretrained(
        self,
        path: str | Path,
        *,
        approved_for_alignment: bool,
    ) -> None:
        from safetensors.torch import save_file

        root = Path(path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        config = {
            "class_name": type(self).__name__,
            "latent_channels": self.latent_channels,
            "frames_per_latent": self.frames_per_latent,
            "spatial_scale": self.spatial_scale,
            "approved_for_alignment": bool(approved_for_alignment),
        }
        config_path = root / self.CONFIG_NAME
        temporary_config = root / f".{self.CONFIG_NAME}.incomplete"
        weights_path = root / self.WEIGHTS_NAME
        temporary_weights = root / f".{self.WEIGHTS_NAME}.incomplete"
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()},
            str(temporary_weights),
        )
        os.replace(temporary_weights, weights_path)
        temporary_config.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_config, config_path)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LatentRobotMaskProbe":
        from safetensors.torch import load_file

        root = Path(path).expanduser().resolve()
        config_path = root / cls.CONFIG_NAME
        weights_path = root / cls.WEIGHTS_NAME
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Expected a JSON object in {config_path}")
        expected_class = config.pop("class_name", cls.__name__)
        if expected_class != cls.__name__:
            raise ValueError(
                f"Probe class {expected_class!r} in {config_path} is not {cls.__name__!r}"
            )
        if config.pop("approved_for_alignment", False) is not True:
            raise RuntimeError(
                f"Latent-mask probe did not pass its held-out alignment gate: {root}"
            )
        probe = cls(**config)
        missing, unexpected = probe.load_state_dict(load_file(str(weights_path)), strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Invalid latent-mask probe weights: missing={missing}, unexpected={unexpected}"
            )
        return probe

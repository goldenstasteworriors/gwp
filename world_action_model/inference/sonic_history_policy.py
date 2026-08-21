from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as torch_F
from accelerate import init_empty_weights
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.inference_openloop import WAPipeline, load_fixed_prompt_embedding, resolve_torch_dtype
from world_action_model.models import CasualWorldActionTransformer_MoT


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected one image with three dimensions, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _center_crop_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    src_width, src_height = image.size
    if float(height) / src_height < float(width) / src_width:
        new_height = int(round(float(width) / src_width * src_height))
        new_width = width
    else:
        new_height = height
        new_width = int(round(float(height) / src_height * src_width))
    image = vision_F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return vision_F.crop(image, top, left, height, width)


def encode_sonic_state_tokens(
    states,
    *,
    state_low: torch.Tensor,
    state_high: torch.Tensor,
    physical_action_dim: int,
    model_action_dim: int,
    required_state_history: int,
    append_state_velocity_token: bool = False,
    append_episode_progress_token: bool = False,
    episode_progress: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the deployment state normalization without loading a policy model."""
    states = torch.as_tensor(states, dtype=torch.float32, device=state_low.device)
    if states.ndim == 1:
        states = states[None]
    if states.ndim != 2:
        raise ValueError(f"Expected state history [T,D], got {tuple(states.shape)}")
    if physical_action_dim <= 0 or model_action_dim < physical_action_dim:
        raise ValueError(
            f"Invalid physical/model action dimensions: {physical_action_dim}/{model_action_dim}"
        )
    if required_state_history < 1:
        raise ValueError("required_state_history must be at least one")
    if state_low.numel() < physical_action_dim or state_high.numel() < physical_action_dim:
        raise ValueError("State normalization statistics are shorter than physical_action_dim")

    states = states[:, :physical_action_dim]
    if states.shape[0] > required_state_history:
        states = states[-required_state_history:]
    elif states.shape[0] < required_state_history:
        states = torch.cat(
            [states[:1].expand(required_state_history - states.shape[0], -1), states],
            dim=0,
        )
    current_state = states[-1].clone()
    low = state_low.flatten()[:physical_action_dim]
    high = state_high.flatten()[:physical_action_dim]
    state_range = (high - low).clamp_min(1e-8)
    norm_state = ((states - low) / state_range * 2.0 - 1.0).clamp(-1.0, 1.0)
    norm_state = torch_F.pad(
        norm_state,
        (0, model_action_dim - physical_action_dim),
        value=0.0,
    )
    tokens = [norm_state]
    if append_state_velocity_token:
        tokens.append(norm_state[-1:] - norm_state[:1])
    if append_episode_progress_token:
        progress = float(np.clip(episode_progress, 0.0, 1.0))
        phase = torch.zeros(
            (1, model_action_dim),
            dtype=norm_state.dtype,
            device=norm_state.device,
        )
        phase[0, 0] = progress * 2.0 - 1.0
        if model_action_dim > 1:
            phase[0, 1] = math.sin(progress * 2.0 * math.pi)
        if model_action_dim > 2:
            phase[0, 2] = math.cos(progress * 2.0 * math.pi)
        tokens.append(phase)
    return torch.cat(tokens, dim=0), current_state


def decode_sonic_action(
    normalized_action,
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    delta_mask,
    current_state,
    physical_action_dim: int,
) -> torch.Tensor:
    """Apply the exact deployment action denormalization/integration path."""
    predicted = torch.as_tensor(
        normalized_action,
        dtype=torch.float32,
        device=action_low.device,
    )
    if predicted.ndim == 1:
        predicted = predicted[None]
    if predicted.ndim != 2 or predicted.shape[1] < physical_action_dim:
        raise ValueError(
            f"Expected normalized action [T,D>=physical_dim], got {tuple(predicted.shape)}"
        )
    if action_low.numel() < physical_action_dim or action_high.numel() < physical_action_dim:
        raise ValueError("Action normalization statistics are shorter than physical_action_dim")
    predicted = predicted[:, :physical_action_dim]
    low = action_low.flatten()[:physical_action_dim]
    high = action_high.flatten()[:physical_action_dim]
    predicted = (predicted + 1.0) * 0.5 * (high - low) + low

    delta_mask = torch.as_tensor(
        delta_mask,
        dtype=torch.bool,
        device=predicted.device,
    ).flatten()[:physical_action_dim]
    if delta_mask.numel() != physical_action_dim:
        raise ValueError("delta_mask length must equal physical_action_dim")
    if delta_mask.any():
        current_state = torch.as_tensor(
            current_state,
            dtype=torch.float32,
            device=predicted.device,
        ).flatten()[:physical_action_dim]
        if current_state.numel() != physical_action_dim:
            raise ValueError("current_state length must equal physical_action_dim")
        predicted[:, delta_mask] += current_state[delta_mask]
    return predicted


def _load_transformer_checkpoint(path: Path, dtype: torch.dtype):
    from safetensors.torch import load_file

    state_dict = {}
    weight_files = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No transformer weights found under {path}")
    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            checkpoint = load_file(str(weight_file), device="cpu")
        else:
            checkpoint = torch.load(weight_file, map_location="cpu", weights_only=True)
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        state_dict.update(checkpoint)

    # Activation checkpoint wrappers are transparent during training, but
    # DeepSpeed includes their private module segment in saved parameter keys.
    state_dict = {
        key.replace("._checkpoint_wrapped_module.", "."): value.to(dtype=dtype)
        for key, value in state_dict.items()
    }
    with init_empty_weights():
        transformer = CasualWorldActionTransformer_MoT.from_config(path)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {path}: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    return transformer.eval()


class SonicHistoryPolicy:
    """Inference-only wrapper for one exported SONIC history experiment."""

    def __init__(self, manifest_path: str | Path, device: str = "cuda:0") -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_dir = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.device = str(device)
        self.dtype = resolve_torch_dtype(self.manifest.get("model_dtype", "bf16"))

        checkpoint = _resolve_path(self.manifest["transformer"], self.manifest_dir)
        base_model = _resolve_path(self.manifest["base_model"], self.manifest_dir)
        norm_stats = _resolve_path(self.manifest["norm_stats"], self.manifest_dir)
        prompt_embedding = _resolve_path(self.manifest["prompt_embedding"], self.manifest_dir)

        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        vae = AutoencoderKLWan.from_pretrained(
            base_model,
            subfolder="vae",
            torch_dtype=self.dtype,
        )
        transformer = _load_transformer_checkpoint(checkpoint, self.dtype)
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=float(self.manifest.get("action_flow_shift", 5.0))
        )
        self.pipe = WAPipeline(
            tokenizer=None,
            text_encoder=None,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            expand_timesteps=True,
        )
        self.pipe.to(device=self.device, dtype=self.dtype)

        stats = json.loads(norm_stats.read_text(encoding="utf-8"))["norm_stats"]
        self.physical_action_dim = int(self.manifest["physical_action_dim"])
        self.model_action_dim = int(self.manifest.get("model_action_dim", 32))
        self.action_horizon = int(self.manifest["action_horizon"])
        self.model_image_frames = int(self.manifest["model_image_frames"])
        self.required_state_history = int(self.manifest.get("state_history_tokens", 1))
        self.required_visual_history = int(self.manifest.get("visual_history_frames", 1))
        self.append_state_velocity_token = bool(
            self.manifest.get("append_state_velocity_token", False)
        )
        self.append_episode_progress_token = bool(
            self.manifest.get("append_episode_progress_token", False)
        )
        self.width, self.height = [int(value) for value in self.manifest.get("image_size", [320, 384])]
        self.num_inference_steps = int(self.manifest.get("num_inference_steps", 10))
        self.delta_mask = np.asarray(
            self.manifest.get("delta_mask", [False] * self.physical_action_dim),
            dtype=bool,
        )[: self.physical_action_dim]

        self.state_low = torch.tensor(
            stats["observation.state"]["q01"][: self.physical_action_dim],
            dtype=torch.float32,
            device=self.device,
        )
        self.state_high = torch.tensor(
            stats["observation.state"]["q99"][: self.physical_action_dim],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_low = torch.tensor(
            stats["action"]["q01"][: self.physical_action_dim],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_high = torch.tensor(
            stats["action"]["q99"][: self.physical_action_dim],
            dtype=torch.float32,
            device=self.device,
        )
        self.prompt_embedding = load_fixed_prompt_embedding(str(prompt_embedding), self.device)

    def _state_tokens(self, states, episode_progress: float) -> tuple[torch.Tensor, torch.Tensor]:
        return encode_sonic_state_tokens(
            states,
            state_low=self.state_low,
            state_high=self.state_high,
            physical_action_dim=self.physical_action_dim,
            model_action_dim=self.model_action_dim,
            required_state_history=self.required_state_history,
            append_state_velocity_token=self.append_state_velocity_token,
            append_episode_progress_token=self.append_episode_progress_token,
            episode_progress=episode_progress,
        )

    @torch.inference_mode()
    def predict(
        self,
        images: Sequence,
        states,
        episode_progress: float = 0.0,
        seed: int = 0,
    ) -> np.ndarray:
        if len(images) < self.required_visual_history:
            raise ValueError(
                f"Expected {self.required_visual_history} visual history frames, got {len(images)}"
            )
        images = images[-self.required_visual_history :]
        images = [
            _center_crop_resize(_to_pil(image), self.width, self.height)
            for image in images
        ]
        state_tokens, current_state = self._state_tokens(states, episode_progress)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        predicted = self.pipe(
            image=images,
            action_chunk=self.action_horizon,
            state=state_tokens,
            prompt_embeds=self.prompt_embedding,
            height=self.height,
            width=self.width,
            num_frames=self.model_image_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            return_dict=False,
            action_dim=self.model_action_dim,
            generator=generator,
        )[0, :, : self.physical_action_dim].float()
        predicted = decode_sonic_action(
            predicted,
            action_low=self.action_low,
            action_high=self.action_high,
            delta_mask=self.delta_mask,
            current_state=current_state,
            physical_action_dim=self.physical_action_dim,
        )
        return predicted.cpu().numpy()

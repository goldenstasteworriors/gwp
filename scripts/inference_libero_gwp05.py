#!/usr/bin/env python3
"""Serve a GWP-0.5 LIBERO action policy over the existing robot socket API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in [
    PROJECT_ROOT,
    PROJECT_ROOT / "third_party" / "giga-models",
    PROJECT_ROOT / "third_party" / "giga-datasets",
    PROJECT_ROOT / "third_party" / "giga-train",
    PROJECT_ROOT / "scripts",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")

from giga_models.sockets import RobotInferenceServer  # noqa: E402
from inference_openloop_gwp0 import WAPipeline, resolve_torch_dtype  # noqa: E402
from world_action_model.models import CasualWorldActionTransformer_MoT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11421)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--model-dtype", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def as_prompt_embedding(value: object, device: str, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, dict):
        value = value.get("t5_embedding", next(iter(value.values())))
    value = torch.as_tensor(value).float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"Expected prompt embedding [B,L,D], got {tuple(value.shape)}")
    value = value[:, :64]
    if value.shape[1] < 64:
        value = F.pad(value, (0, 0, 0, 64 - value.shape[1]))
    return value.to(device=device, dtype=dtype)


def as_pil_image(value: object) -> PIL.Image.Image:
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {value.shape}")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = np.moveaxis(value, 0, -1)
    if np.issubdtype(value.dtype, np.floating) and value.max(initial=0.0) <= 1.0:
        value = value * 255.0
    return PIL.Image.fromarray(np.clip(value, 0, 255).astype(np.uint8), mode="RGB")


def build_policy(args: argparse.Namespace):
    device = str(args.device)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    dtype = resolve_torch_dtype(args.model_dtype)
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype)
    transformer = CasualWorldActionTransformer_MoT.from_pretrained(args.checkpoint, torch_dtype=dtype)
    transformer.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe = WAPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        image_processor=None,
        image_encoder=None,
        transformer=transformer,
        transformer_2=None,
        expand_timesteps=True,
    )
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    stats = json.loads(args.norm_stats.read_text())["norm_stats"]
    state_min = torch.tensor(stats["observation.state"]["q01"][:8], device=device)
    state_max = torch.tensor(stats["observation.state"]["q99"][:8], device=device)
    action_min = torch.tensor(stats["action"]["q01"][:7], device=device)
    action_max = torch.tensor(stats["action"]["q99"][:7], device=device)
    state_range = (state_max - state_min).clamp_min(1e-8)
    action_range = (action_max - action_min).clamp_min(1e-8)
    prompt_cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def inference(self, data):
        image = as_pil_image(data["image"])
        state = torch.as_tensor(data["state"], dtype=torch.float32, device=device).reshape(-1)[:8]
        normalized_state = 2.0 * (state - state_min) / state_range - 1.0
        normalized_state = F.pad(normalized_state, (0, 16 - normalized_state.shape[-1])).unsqueeze(0)

        prompt_key = str(data.get("prompt_key", data.get("task_id", "default")))
        prompt_value = data.get("prompt_embedding")
        if prompt_value is not None:
            prompt_cache[prompt_key] = as_prompt_embedding(prompt_value, device, dtype)
        if prompt_key not in prompt_cache:
            raise ValueError(f"No prompt embedding cached for {prompt_key!r}")

        episode_index = int(data.get("episode_index", 0))
        chunk_index = int(data.get("chunk_index", 0))
        generator = torch.Generator(device=device).manual_seed(
            args.seed + episode_index * 10007 + chunk_index
        )
        normalized_action = pipe(
            image=image,
            action_chunk=48,
            state=normalized_state,
            prompt_embeds=prompt_cache[prompt_key],
            height=256,
            width=256,
            num_frames=5,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
            return_dict=False,
            action_dim=16,
        )[0, :, :7].float()
        action = ((normalized_action + 1.0) * 0.5) * action_range + action_min
        return action.cpu().numpy().astype(np.float32)

    pipe.inference = types.MethodType(inference, pipe)
    return pipe


def main() -> None:
    args = parse_args()
    policy = build_policy(args)
    print(f"Serving LIBERO GWP policy on {args.host}:{args.port}", flush=True)
    RobotInferenceServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()

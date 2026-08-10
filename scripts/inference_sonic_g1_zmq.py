#!/usr/bin/env python3
"""Real-time GigaWorld-Policy-0.5 ZMQ server for the SONIC G1.

The wire protocol is compatible with DiT4DiT's ``PolicyClient``: requests and
responses use msgpack, the endpoint is ``get_action``, and the returned array is
stored under ``data.unnormalized_actions``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
import zmq
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_openloop import WAPipeline, load_fixed_prompt_embedding
from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    LEGACY_SCHEMA_ID,
    LEGACY_SONIC_ACTION_DIM,
    validate_asymmetric_norm_payload,
)
from world_action_model.models import CasualWorldActionTransformer_MoT


MODEL_ACTION_DIM = 32
IMAGE_SIZE = (320, 384)


def _pack_default(obj: Any) -> dict:
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(type(obj).__name__)


def _unpack_object(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _load_stats(
    path: Path,
    device: torch.device,
    *,
    physical_dim: int,
    action_schema_id: str,
) -> tuple[torch.Tensor, ...]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if action_schema_id == ASYMMETRIC_SCHEMA_ID:
        validate_asymmetric_norm_payload(payload, model_dim=MODEL_ACTION_DIM)
    stats = payload["norm_stats"]
    state_low = torch.tensor(stats["observation.state"]["q01"][:physical_dim], device=device)
    state_high = torch.tensor(stats["observation.state"]["q99"][:physical_dim], device=device)
    action_low = torch.tensor(stats["action"]["q01"][:physical_dim], device=device)
    action_high = torch.tensor(stats["action"]["q99"][:physical_dim], device=device)
    for name, value in (
        ("state q01", state_low),
        ("state q99", state_high),
        ("action q01", action_low),
        ("action q99", action_high),
    ):
        if value.numel() != physical_dim:
            raise ValueError(f"normalization {name} has {value.numel()} values; expected {physical_dim}")
    return state_low, state_high, action_low, action_high


def _load_pipeline(
    checkpoint: Path,
    base_model: Path,
    device: str,
) -> WAPipeline:
    dtype = torch.bfloat16
    transformer = CasualWorldActionTransformer_MoT.from_config(
        checkpoint / "config.json"
    )
    weight_path = checkpoint / "diffusion_pytorch_model.bin"
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        key.replace("._checkpoint_wrapped_module", ""): value
        for key, value in state_dict.items()
    }
    incompatible = transformer.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not strictly match the SONIC transformer: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    transformer.to(dtype=dtype).eval()

    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipeline = WAPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        expand_timesteps=True,
    )
    pipeline.to(device=device, dtype=dtype)
    return pipeline


def _as_pil(image: Any) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"image must have shape [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return ImageOps.fit(Image.fromarray(array, mode="RGB"), IMAGE_SIZE, method=Image.Resampling.BILINEAR)


class SonicG1Policy:
    def __init__(
        self,
        pipeline,
        norm_stats: Path,
        device: str,
        prompt_embedding: torch.Tensor | None,
        inference_steps: int,
        action_schema: str = "legacy30",
    ) -> None:
        self.pipeline = pipeline
        self.device = torch.device(device)
        self.prompt_embedding = prompt_embedding
        self.inference_steps = inference_steps
        self.action_schema = str(action_schema)
        if self.action_schema == "legacy30":
            self.physical_dim = LEGACY_SONIC_ACTION_DIM
            self.action_schema_id = LEGACY_SCHEMA_ID
        elif self.action_schema == "asymmetric25":
            self.physical_dim = ASYMMETRIC_ACTION_DIM
            self.action_schema_id = ASYMMETRIC_SCHEMA_ID
        else:
            raise ValueError(f"unsupported action_schema={self.action_schema!r}")
        self.state_low, self.state_high, self.action_low, self.action_high = _load_stats(
            norm_stats,
            self.device,
            physical_dim=self.physical_dim,
            action_schema_id=self.action_schema_id,
        )

    @torch.inference_mode()
    def predict_action(self, **request: Any) -> dict[str, np.ndarray | float]:
        examples = request.get("examples")
        if not isinstance(examples, list) or len(examples) != 1:
            raise ValueError("request must contain exactly one item in examples")
        example = examples[0]

        state = torch.as_tensor(example["state"], dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if tuple(state.shape) != (1, self.physical_dim):
            raise ValueError(
                f"SONIC GigaWorld {self.action_schema} state must have shape "
                f"[1,{self.physical_dim}], got {tuple(state.shape)}"
            )
        state_range = (self.state_high - self.state_low).clamp_min(1e-8)
        normalized_state = ((state - self.state_low) / state_range * 2.0 - 1.0).clamp(-1.0, 1.0)
        normalized_state = F.pad(normalized_state, (0, MODEL_ACTION_DIM - self.physical_dim))

        images = example.get("image")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError("example.image must contain exactly the ego-view image")
        prompt = example.get("lang")
        prompt_embedding = self.prompt_embedding
        if prompt_embedding is None:
            raise ValueError("server requires --fixed-t5-path for this component-only Wan base model")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        normalized_action = self.pipeline(
            height=IMAGE_SIZE[1],
            width=IMAGE_SIZE[0],
            action_chunk=48,
            state=normalized_state,
            num_frames=5,
            guidance_scale=0.0,
            num_inference_steps=self.inference_steps,
            image=_as_pil(images[0]),
            return_dict=False,
            prompt_embeds=prompt_embedding,
            prompt=None,
            action_dim=MODEL_ACTION_DIM,
            generator=None,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started

        normalized_action = normalized_action[..., : self.physical_dim]
        action_range = (self.action_high - self.action_low).clamp_min(1e-8)
        action = ((normalized_action + 1.0) * 0.5 * action_range + self.action_low)
        action = action[0].float().cpu().numpy()
        return {"unnormalized_actions": action, "inference_latency_s": latency}


class MsgpackPolicyServer:
    def __init__(self, policy: SonicG1Policy, host: str, port: int) -> None:
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def run(self) -> None:
        logging.info("SONIC G1 server listening on %s", self.socket.getsockopt_string(zmq.LAST_ENDPOINT))
        while self.running:
            try:
                request = msgpack.unpackb(self.socket.recv(), object_hook=_unpack_object)
                endpoint = request.get("endpoint", "get_action")
                if endpoint == "ping":
                    response = {"ok": True, "status": "ok", "message": "Server is running"}
                elif endpoint == "get_metadata":
                    response = {
                        "ok": True,
                        "status": "ok",
                        "data": {
                            "policy": "giga-world-policy-0.5-sonic-g1",
                            "action_schema": self.policy.action_schema,
                            "action_schema_id": self.policy.action_schema_id,
                            "state_dim": self.policy.physical_dim,
                            "action_dim": self.policy.physical_dim,
                            "action_horizon": 48,
                        },
                    }
                elif endpoint == "kill":
                    self.running = False
                    response = {"ok": True, "status": "ok", "message": "Server shutting down"}
                elif endpoint in {"get_action", "predict_action"}:
                    output = self.policy.predict_action(**request.get("data", {}))
                    response = {"ok": True, "status": "ok", "type": "inference_result", "data": output}
                else:
                    raise ValueError(f"unknown endpoint: {endpoint}")
            except Exception as exc:
                logging.exception("Inference request failed")
                response = {
                    "ok": False,
                    "status": "error",
                    "error": {"message": str(exc), "traceback": traceback.format_exc()},
                }
            self.socket.send(msgpack.packb(response, default=_pack_default))
        self.socket.close()
        self.context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--fixed-t5-path", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument(
        "--action-schema",
        choices=("legacy30", "asymmetric25"),
        default="legacy30",
        help="Legacy 30-D remains the default; asymmetric25 requires schema-labelled norm stats.",
    )
    parser.add_argument("--no-compile-transformer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = _load_pipeline(args.checkpoint, args.base_model, args.device)
    if not args.no_compile_transformer:
        from inference_openloop import compile_policy_action_blocks

        compiled = compile_policy_action_blocks(pipeline)
        logging.info("Compiled modules: %s", compiled)
    fixed_t5 = load_fixed_prompt_embedding(str(args.fixed_t5_path), args.device)
    policy = SonicG1Policy(
        pipeline,
        args.norm_stats,
        args.device,
        fixed_t5,
        args.inference_steps,
        args.action_schema,
    )
    MsgpackPolicyServer(policy, args.host, args.port).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

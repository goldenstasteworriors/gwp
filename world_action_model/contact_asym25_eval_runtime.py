"""Heavy, deterministic runtime for the asymmetric25 held-out evaluator."""

from __future__ import annotations

from collections.abc import Mapping
import random
from pathlib import Path
from typing import Any

import numpy as np

from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
    MODEL_ACTION_DIM,
)
from world_action_model.contact_asym25_heldout_eval import (
    HORIZON_FRAMES,
    IMAGE_FRAME_OFFSETS,
    NUM_CONTACT_QUERIES,
    TIME_BINS,
    HeldoutEvaluationError,
)
from world_action_model.contact_transition_schema import (
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)


class ProductionHeldoutRuntime:
    """Load source windows and sample one checkpoint without target leakage."""

    def __init__(
        self,
        *,
        preflight_receipt: Mapping[str, Any],
        base_model_path: str | Path,
        device: str,
        dtype: str,
        inference_steps: int,
    ) -> None:
        import torch
        from diffusers.models import AutoencoderKLWan
        from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset
        from world_action_model.transforms.wa_transforms_lerobot_pretrain import (
            WALeRobotTransformsPretrain,
        )

        if dtype not in {"float32", "float16", "bfloat16"}:
            raise HeldoutEvaluationError(
                "runtime dtype must be float32, float16, or bfloat16"
            )
        if not isinstance(inference_steps, int) or isinstance(inference_steps, bool) or inference_steps <= 0:
            raise HeldoutEvaluationError("inference_steps must be a positive integer")
        if device.startswith("cpu") and dtype != "float32":
            raise HeldoutEvaluationError("CPU production evaluation requires float32")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype_name = dtype
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.inference_steps = inference_steps
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.preflight = dict(preflight_receipt)
        data_root = Path(self.preflight["data_root"])
        self.dataset = WAMLeRobotDataset(
            data_path=str(data_root),
            data_size=None,
            delta_info={"action": HORIZON_FRAMES},
            delta_frames={
                "observation.images.ego_view": list(IMAGE_FRAME_OFFSETS)
            },
            force_cache_sync=True,
            video_backend="pyav",
            t5_load_from="dir",
            t5_cfg={
                "t5_embedding_dir": str(data_root / "t5_embedding"),
                "t5_embedding_pattern": "episode_{episode_index:06d}.pt",
                "t5_embedding_key": "t5_embedding",
            },
            t5_cache_size=128,
            contact_sidecar_dir=self.preflight["sidecar_dir"],
            contact_sidecar_pattern="episode_{episode_index:06d}.npz",
            contact_sidecar_cache_size=8,
            contact_sidecar_strict=True,
            contact_sidecar_require_reviewed_v1_1=True,
            contact_sidecar_expected_meta_sha256=self.preflight[
                "sidecar_meta_sha256"
            ],
            contact_sidecar_expected_episode_sha256=self.preflight[
                "episode_npz_sha256"
            ],
            robotype="sonic_g1_asymmetric25",
        )
        norm_path = Path(self.preflight["norm_stats_path"])
        self.transform = WALeRobotTransformsPretrain(
            is_train=False,
            eval_crop_mode="center",
            dst_size=(320, 384),
            num_frames=HORIZON_FRAMES,
            fps=float(self.preflight["dataset_fps"]),
            norm_path=[
                {
                    "path": str(norm_path),
                    "data_paths": [str(data_root)],
                    "sha256": self.preflight["norm_stats_sha256"],
                }
            ],
            robotype_to_embodiment_id={"sonic_g1_asymmetric25": 0},
            robotype_default_embodiment_id=0,
            model_action_dim=MODEL_ACTION_DIM,
            delta_mask_by_embodiment_id={"0": [False] * ASYMMETRIC_ACTION_DIM},
            require_exact_effective_dim=True,
            expected_action_schema_id=ASYMMETRIC_SCHEMA_ID,
            expected_hardware_contract_sha256=self.preflight[
                "hardware_contract_sha256"
            ],
            expected_calibration_capture_sha256=self.preflight[
                "calibration_capture_sha256"
            ],
            norm_use_quantiles=True,
            norm_enable_clamp=True,
            num_views=1,
            view_keys=["observation.images.ego_view"],
            image_cfg={"mask_generator": {"max_ref_frames": 1, "start": 1, "factor": 4}},
            image_augmentation={"enabled": False},
            max_prompt_len=64,
            subtask_prob=0,
            matched_control_capture=False,
            emit_action_denorm_scale=True,
        )
        self.dataset.open()

        self.base_model_path = str(Path(base_model_path).expanduser().absolute())
        self.vae = AutoencoderKLWan.from_pretrained(
            self.base_model_path,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.vae.requires_grad_(False)
        self.vae.eval().to(self.device, dtype=self.dtype)
        self.latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.device,
            dtype=self.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        self.latents_std = (
            1.0
            / torch.tensor(
                self.vae.config.latents_std,
                device=self.device,
                dtype=self.dtype,
            ).view(1, self.vae.config.z_dim, 1, 1, 1)
        )
        self.transformer = None
        self.scheduler = None

    @staticmethod
    def _scalar_int(value: object, name: str) -> int:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            raise HeldoutEvaluationError(f"{name} cannot be boolean")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise HeldoutEvaluationError(f"invalid {name}") from exc

    def load_window(self, window: Mapping[str, Any]) -> dict[str, Any]:
        random.seed(0)
        np.random.seed(0)
        self.torch.manual_seed(0)
        raw = self.dataset._get_data(int(window["global_index"]))
        episode = self._scalar_int(raw.get("episode_index"), "episode_index")
        frame = self._scalar_int(raw.get("frame_index"), "frame_index")
        authoritative_index = self._scalar_int(raw.get("index"), "global index")
        if (
            episode != int(window["episode_id"])
            or frame != int(window["frame_index"])
            or authoritative_index != int(window["global_index"])
        ):
            raise HeldoutEvaluationError(
                "fixed window identity differs from the runtime LeRobot sample"
            )
        transformed = self.transform(raw)
        required = {
            "ref_images",
            "prompt_embeds",
            "action",
            "state",
            "embodiment_id",
            "action_dim_mask",
            "action_time_mask",
            "action_denorm_scale",
            "contact_target",
            "contact_loss_mask",
            "contact_confidence",
            "contact_condition_mask",
        }
        missing = sorted(required - set(transformed))
        if missing:
            raise HeldoutEvaluationError(
                f"held-out transformed window lacks required fields: {missing}"
            )
        if "contact_condition" in transformed:
            raise HeldoutEvaluationError(
                "held-out evaluation forbids oracle contact_condition input"
            )
        if not bool(transformed["action_time_mask"].all()):
            raise HeldoutEvaluationError("fixed held-out window unexpectedly contains padding")
        return transformed

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from world_action_model.models import CasualWorldActionTransformer_MoT

        self.unload_checkpoint()
        checkpoint_path = str(Path(checkpoint_path).expanduser().absolute())
        model = CasualWorldActionTransformer_MoT.from_pretrained(
            checkpoint_path,
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        expected = {
            "in_action_channels": MODEL_ACTION_DIM,
            "out_action_channels": MODEL_ACTION_DIM,
            "num_contact_tokens": NUM_CONTACT_QUERIES,
            "contact_target_dim": SEGMENT_TARGET_DIM,
            "contact_condition_dim": SEGMENT_TARGET_DIM,
            "contact_bottleneck": True,
            "contact_schema_version": SEGMENT_SCHEMA_VERSION,
            "contact_semantic_bottleneck": True,
            "contact_semantic_transport_mode": "hard_st",
            "contact_time_bins": TIME_BINS,
        }
        for name, value in expected.items():
            if getattr(model.config, name, None) != value:
                raise HeldoutEvaluationError(
                    f"checkpoint model config {name} differs from production"
                )
        model.requires_grad_(False)
        model.eval().to(self.device, dtype=self.dtype)
        model._enable_action_only_prefix_cache = True
        self.transformer = model
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)

    def unload_checkpoint(self) -> None:
        if self.transformer is not None:
            if hasattr(self.transformer, "reset_action_only_prefix_cache"):
                self.transformer.reset_action_only_prefix_cache()
            self.transformer.to("cpu")
            del self.transformer
            self.transformer = None
        self.scheduler = None
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _encode_reference(self, ref_images: Any) -> Any:
        torch = self.torch
        reference = ref_images[:1].unsqueeze(0).permute(0, 2, 1, 3, 4)
        reference = reference.to(self.device, dtype=self.vae.dtype)
        with torch.inference_mode():
            latent = self.vae.encode(reference).latent_dist.mode()
        return (latent - self.latents_mean) * self.latents_std

    def predict(self, transformed: Mapping[str, Any], *, action_noise_seed: int) -> dict[str, np.ndarray]:
        torch = self.torch
        if self.transformer is None or self.scheduler is None:
            raise HeldoutEvaluationError("no checkpoint is loaded")
        ref_latents = self._encode_reference(transformed["ref_images"])
        state = transformed["state"].unsqueeze(0).to(self.device, dtype=self.dtype)
        prompt = transformed["prompt_embeds"].unsqueeze(0).to(
            self.device, dtype=self.dtype
        )
        embodiment_id = transformed["embodiment_id"].reshape(1).to(self.device)
        dim_mask = transformed["action_dim_mask"].to(self.device, dtype=self.dtype)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(action_noise_seed))
        action = torch.randn(
            (1, HORIZON_FRAMES, MODEL_ACTION_DIM),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        action = action * dim_mask[None, None, :]
        patch = tuple(int(value) for value in self.transformer.config.patch_size)
        if any(
            int(ref_latents.shape[axis + 2]) % patch[axis] != 0
            for axis in range(3)
        ):
            raise HeldoutEvaluationError("reference latent shape is not patch divisible")
        num_ref_tokens = int(
            np.prod(
                [
                    int(ref_latents.shape[axis + 2]) // patch[axis]
                    for axis in range(3)
                ]
            )
        )
        num_state_tokens = int(state.shape[1])
        timestep_template = torch.zeros(
            (1, num_state_tokens + num_ref_tokens + HORIZON_FRAMES),
            device=self.device,
            dtype=self.dtype,
        )
        empty_noisy_latents = ref_latents[:, :, :0]
        self.scheduler.set_timesteps(self.inference_steps, device=self.device)
        first_logits = None
        first_hard = None
        self.transformer.reset_action_only_prefix_cache()
        try:
            with torch.inference_mode():
                for timestep_value in self.scheduler.timesteps:
                    timestep = timestep_template.clone()
                    timestep[:, num_state_tokens + num_ref_tokens :] = timestep_value.to(
                        timestep.dtype
                    )
                    output = self.transformer(
                        ref_latents=ref_latents,
                        noisy_latents=empty_noisy_latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt,
                        return_dict=True,
                        action=action,
                        state=state,
                        embodiment_id=embodiment_id,
                        action_only=True,
                    )
                    required_outputs = {
                        "action_pred",
                        "contact_pred",
                        "contact_semantic",
                    }
                    if not isinstance(output, dict) or not required_outputs.issubset(
                        output
                    ):
                        raise HeldoutEvaluationError(
                            "action-only checkpoint did not expose contact outputs"
                        )
                    logits = output["contact_pred"].detach()
                    hard = output["contact_semantic"].detach()
                    if first_logits is None:
                        first_logits = logits.clone()
                        first_hard = hard.clone()
                    elif not torch.equal(first_logits, logits) or not torch.equal(
                        first_hard, hard
                    ):
                        raise HeldoutEvaluationError(
                            "cached contact plan changed across action denoising steps"
                        )
                    action = self.scheduler.step(
                        output["action_pred"],
                        timestep_value,
                        action,
                        return_dict=False,
                    )[0]
                    action = action * dim_mask[None, None, :]
        finally:
            self.transformer.reset_action_only_prefix_cache()
        return {
            "action_prediction": action[0].float().cpu().numpy(),
            "contact_logits": first_logits[0].float().cpu().numpy(),
            "contact_hard_semantic": first_hard[0].float().cpu().numpy(),
        }

    def close(self) -> None:
        self.unload_checkpoint()
        if self.dataset is not None:
            self.dataset.close()
        if getattr(self, "vae", None) is not None:
            self.vae.to("cpu")
            del self.vae
            self.vae = None

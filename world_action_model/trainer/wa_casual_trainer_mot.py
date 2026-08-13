import functools
import os

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from giga_models import utils as gm_utils
from world_action_model.alignment import masked_bce_dice_loss, policy_only_surrogate
from world_action_model.models import CasualWorldActionTransformer_MoT
from giga_train import ModuleDict, Trainer

class CasualWATrainerMoT(Trainer):
    """WAM pretraining trainer for the Mixture-of-Transformers policy."""

    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        self.visual_flow_shift = float(model_config.visual_flow_shift)
        self.action_flow_shift = float(model_config.action_flow_shift)
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.num_embodiments = int(model_config.get("num_embodiments", 1))
        self.visual_loss_weight = float(model_config.get("visual_loss_weight", 1.0))
        self.action_loss_weight = float(model_config.get("action_loss_weight", 1.0))
        self.consequence_loss_weight = float(model_config.get("consequence_loss_weight", 0.0))
        self.consequence_gradient_mode = str(model_config.get("consequence_gradient_mode", "policy_only"))
        self.consequence_interval = int(model_config.get("consequence_interval", 1))
        self.consequence_start_step = int(model_config.get("consequence_start_step", 0))
        self.consequence_visual_sigma = float(model_config.get("consequence_visual_sigma", 1.0))
        self.counterfactual_mask_loss_weight = float(
            model_config.get("counterfactual_mask_loss_weight", 0.0)
        )
        self.counterfactual_mask_bce_weight = float(
            model_config.get("counterfactual_mask_bce_weight", 1.0)
        )
        self.counterfactual_mask_dice_weight = float(
            model_config.get("counterfactual_mask_dice_weight", 1.0)
        )
        self.counterfactual_mask_interval = int(
            model_config.get("counterfactual_mask_interval", 1)
        )
        self.counterfactual_mask_start_step = int(
            model_config.get("counterfactual_mask_start_step", 0)
        )
        valid_gradient_modes = {"policy_only", "joint", "video_only"}
        if self.consequence_gradient_mode not in valid_gradient_modes:
            raise ValueError(
                f"consequence_gradient_mode={self.consequence_gradient_mode!r} must be one of "
                f"{sorted(valid_gradient_modes)}"
            )
        if self.consequence_interval <= 0:
            raise ValueError("consequence_interval must be positive")
        if not 0.0 < self.consequence_visual_sigma <= 1.0:
            raise ValueError("consequence_visual_sigma must be in (0, 1]")
        if self.counterfactual_mask_loss_weight < 0.0:
            raise ValueError("counterfactual_mask_loss_weight must be non-negative")
        if self.counterfactual_mask_interval <= 0:
            raise ValueError("counterfactual_mask_interval must be positive")
        if self.counterfactual_mask_loss_weight > 0.0 and self.consequence_loss_weight > 0.0:
            raise ValueError(
                "Counterfactual mask supervision cannot be combined with the legacy demo-RGB "
                "consequence loss"
            )

        vae_pretrained = model_config.get("vae_pretrained", os.path.join(pretrained, "vae"))
        vae_dtype = model_config.get("vae_dtype", torch.float32)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        transformer_pretrained = model_config.get("transformer_pretrained", os.path.join(pretrained, "transformer"))
        transformer_cfg = model_config.get("transformer")
        if (
            self.counterfactual_mask_loss_weight > 0.0
            and int(transformer_cfg.get("counterfactual_mask_frames_per_latent", 0)) <= 0
        ):
            raise ValueError(
                "counterfactual_mask_frames_per_latent must be enabled when mask loss is active"
            )
        transformer = CasualWorldActionTransformer_MoT(**transformer_cfg)
        transformer = load_pretrained_weights(
            transformer,
            transformer_pretrained,
            skip_action_expert=model_config.get("skip_action_expert", False),
            strict_load=model_config.get("strict_load", False),
        )
        transformer.to(self.device)
        if model_config.get("enable_gradient_checkpointing", True):
            transformer.enable_gradient_checkpointing()

        model = dict(transformer=transformer)
        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()
        return model

    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        bs = images.shape[0]
        prompt_embeds = batch_dict["prompt_embeds"]
        action = batch_dict["action"]
        state = batch_dict["state"]
        embodiment_id = batch_dict["embodiment_id"]

        visual_timestep, visual_sigma = self.get_timestep_and_sigma(bs, images.ndim, self.visual_flow_shift)
        action_timestep, action_sigma = self.get_timestep_and_sigma(bs, action.ndim, self.action_flow_shift)

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * visual_sigma + visual_latents * (1 - visual_sigma)

        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)

        prompt_embeds = prompt_embeds.to(self.dtype)
        if "ref_images" not in batch_dict:
            raise ValueError("CasualWATrainerMoT requires ref_images in batch_dict")

        if not self.expand_timesteps:
            ref_images = batch_dict["ref_images"]
            ref_latents = self.forward_vae(ref_images)
            num_frames = images.shape[1]
            batch_size = ref_latents.shape[0]
            latent_height = ref_latents.shape[-2]
            latent_width = ref_latents.shape[-1]
            mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2).to(ref_latents.device)
            condition = torch.concat([mask_lat_size, ref_latents], dim=1)
            insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
        else:
            num_latent_frames = visual_latents.shape[2]
            latent_height = visual_latents.shape[-2]
            latent_width = visual_latents.shape[-1]
            ref_images = batch_dict["ref_images"][:, :1]
            ref_latents = self.forward_vae(ref_images)
            first_frame_mask = torch.ones(
                bs,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=visual_latents.dtype,
                device=visual_latents.device,
            )
            first_frame_mask[:, :, 0] = 0
            insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens
        timestep = torch.zeros(
            bs,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        timestep[:, num_state_tokens + num_clean_latent_tokens : num_state_tokens + num_clean_latent_tokens + num_action_tokens] = (
            action_timestep[:, None]
        )
        timestep[:, num_state_tokens + num_action_tokens + num_clean_latent_tokens :] = visual_timestep[:, None]

        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            embodiment_id=embodiment_id,
        )

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        action_loss = (action_pred.float() - action_target.float()).pow(2).mean()
        losses = {
            "visual_loss": self.visual_loss_weight * visual_loss,
            "action_loss": self.action_loss_weight * action_loss,
        }
        if self._should_compute_counterfactual_mask_loss():
            required_keys = (
                "counterfactual_action",
                "counterfactual_robot_masks",
                "counterfactual_robot_mask_valid",
            )
            missing_keys = [key for key in required_keys if key not in batch_dict]
            if missing_keys:
                raise KeyError(f"Counterfactual mask batch is missing keys: {missing_keys}")
            counterfactual_action = batch_dict["counterfactual_action"].to(
                device=state.device,
                dtype=self.dtype,
            )
            if self.action_repeats > 1:
                counterfactual_action = counterfactual_action.repeat(
                    1, self.action_repeats, 1
                )
            if counterfactual_action.shape[1] != num_action_tokens:
                raise ValueError(
                    f"Counterfactual action tokens {counterfactual_action.shape[1]} do not match "
                    f"training action tokens {num_action_tokens}"
                )

            counterfactual_future_noise = torch.randn_like(visual_latents[:, :, 1:]).to(
                self.dtype
            )
            counterfactual_timestep = torch.zeros_like(timestep)
            counterfactual_timestep[
                :, num_state_tokens + num_action_tokens + num_clean_latent_tokens :
            ] = 1000
            counterfactual_output = transformer(
                ref_latents=ref_latents,
                noisy_latents=counterfactual_future_noise,
                timestep=counterfactual_timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=True,
                action=counterfactual_action.detach(),
                state=state,
                embodiment_id=embodiment_id,
                output_counterfactual_mask=True,
                output_action=False,
                output_visual=False,
            )
            mask_logits = counterfactual_output["counterfactual_mask_logits"].float()
            mask_target = batch_dict["counterfactual_robot_masks"].to(
                device=mask_logits.device,
                dtype=torch.float32,
            )
            mask_valid = batch_dict["counterfactual_robot_mask_valid"].to(
                device=mask_logits.device,
                dtype=torch.float32,
            )
            mask_target = F.interpolate(mask_target, size=mask_logits.shape[-2:], mode="nearest")
            mask_valid = F.interpolate(mask_valid, size=mask_logits.shape[-2:], mode="nearest")
            raw_mask_loss = masked_bce_dice_loss(
                mask_logits,
                mask_target,
                mask_valid,
                bce_weight=self.counterfactual_mask_bce_weight,
                dice_weight=self.counterfactual_mask_dice_weight,
            )
            losses["counterfactual_mask_loss"] = (
                self.counterfactual_mask_loss_weight * raw_mask_loss
            )
        if self._should_compute_consequence_loss():
            predicted_clean_action = noisy_action.float() - action_sigma.float() * action_pred.float()
            consequence_action = predicted_clean_action
            if self.consequence_gradient_mode == "video_only":
                consequence_action = consequence_action.detach()

            consequence_sigma = self.consequence_visual_sigma
            consequence_future_latents = (
                visual_noise[:, :, 1:] * consequence_sigma
                + visual_latents[:, :, 1:] * (1.0 - consequence_sigma)
            ).to(self.dtype)
            consequence_timestep = torch.zeros_like(timestep)
            consequence_visual_timestep = int(round(1000.0 * consequence_sigma))
            consequence_timestep[:, num_state_tokens + num_action_tokens + num_clean_latent_tokens :] = (
                consequence_visual_timestep
            )

            consequence_visual_pred, _ = transformer(
                ref_latents=ref_latents,
                noisy_latents=consequence_future_latents,
                timestep=consequence_timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                action=consequence_action.to(self.dtype),
                state=state,
                embodiment_id=embodiment_id,
            )
            raw_consequence_loss = (
                (consequence_visual_pred.float() - visual_target.float()) * first_frame_mask
            ).pow(2).mean()

            if self.consequence_gradient_mode == "policy_only":
                consequence_loss = policy_only_surrogate(raw_consequence_loss, predicted_clean_action)
            else:
                consequence_loss = raw_consequence_loss
            losses["consequence_loss"] = self.consequence_loss_weight * consequence_loss
        return losses

    def _should_compute_consequence_loss(self) -> bool:
        if self.consequence_loss_weight <= 0.0:
            return False
        step = int(getattr(self, "_cur_step", 0))
        if step < self.consequence_start_step:
            return False
        return (step - self.consequence_start_step) % self.consequence_interval == 0

    def _should_compute_counterfactual_mask_loss(self) -> bool:
        if self.counterfactual_mask_loss_weight <= 0.0:
            return False
        step = int(getattr(self, "_cur_step", 0))
        if step < self.counterfactual_mask_start_step:
            return False
        return (
            step - self.counterfactual_mask_start_step
        ) % self.counterfactual_mask_interval == 0

    def save_checkpoint_step(self) -> None:
        if bool(self.kwargs.get("disable_checkpoint", False)):
            return
        super().save_checkpoint_step()

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b t c h w -> b c t h w")
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim, flow_shift):
        sigma = torch.rand(batch_size).to(self.device)
        sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

def load_pretrained_weights(model, pretrained_path, skip_action_expert=False, strict_load=False):
    from safetensors.torch import load_file

    pretrained_path = gm_utils.get_model_path(pretrained_path)
    if os.path.isdir(pretrained_path):
        weight_files = sorted(
            os.path.join(pretrained_path, f)
            for f in os.listdir(pretrained_path)
            if f.endswith((".safetensors", ".bin"))
        )
    else:
        weight_files = [pretrained_path]

    state_dicts = {}
    for weight_file in weight_files:
        print(f"Loading weights from {weight_file}")
        if weight_file.endswith(".bin"):
            checkpoint = torch.load(weight_file, map_location="cpu")
        else:
            checkpoint = load_file(weight_file)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dicts.update(checkpoint)

    if not strict_load and hasattr(model, "load_from_wan_pretrained_state_dict"):
        loaded, skipped, unexpected, missing = model.load_from_wan_pretrained_state_dict(
            state_dicts,
            skip_action_expert=skip_action_expert,
        )
        if unexpected:
            print(f"[WARNING] Unexpected keys in Wan checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
        if skipped:
            print(f"[WARNING] Skipped keys during MoT remap: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        if missing:
            print(f"[WARNING] Missing keys after MoT remap: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print(f"Loaded {len(loaded)} tensors from Wan2.2 into MoT.")
        return model

    missing_keys, unexpected_keys = model.load_state_dict(state_dicts, strict=False)
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys in state_dict: {unexpected_keys}")
    if missing_keys:
        print(f"[WARNING] Missing keys in state_dict: {missing_keys}")
    return model

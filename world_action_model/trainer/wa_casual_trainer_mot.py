import functools
import json
import os

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from giga_models import utils as gm_utils
from world_action_model.models import CasualWorldActionTransformer_MoT
from giga_train import ModuleDict, Trainer


_ACTION_HEAD_PARAMETER_PREFIXES = (
    "action_condition_embedder.",
    "action_norm_out.",
    "state_encoder.",
    "action_encoder.",
    "action_decoder.",
)


def is_action_head_parameter(name: str) -> bool:
    """Return whether a MoT parameter belongs to the action-specific branch."""
    prefixes_to_strip = ("module.", "_orig_mod.", "transformer.")
    stripped = name
    changed = True
    while changed:
        changed = False
        for prefix in prefixes_to_strip:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
                break
    return (
        stripped == "action_scale_shift_table"
        or stripped.startswith(_ACTION_HEAD_PARAMETER_PREFIXES)
        or (stripped.startswith("blocks.") and ".action_expert." in stripped)
    )


def canonical_checkpoint_parameter_name(name: str) -> str:
    """Remove activation-checkpoint wrapper components from exported keys."""
    return name.replace("._checkpoint_wrapped_module", "")


def split_action_head_state_dict(state_dict):
    """Split a state dict into canonical action-head and non-action tensors."""
    action_head_state = {}
    non_action_state = {}
    for name, tensor in state_dict.items():
        canonical_name = canonical_checkpoint_parameter_name(name)
        destination = (
            action_head_state
            if is_action_head_parameter(canonical_name)
            else non_action_state
        )
        if canonical_name in destination:
            raise RuntimeError(f"Duplicate canonical checkpoint key {canonical_name!r}")
        destination[canonical_name] = tensor
    return action_head_state, non_action_state


class CasualWATrainerMoT(Trainer):
    """WAM pretraining trainer for the Mixture-of-Transformers policy."""

    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        self.visual_flow_shift = float(model_config.visual_flow_shift)
        self.action_flow_shift = float(model_config.action_flow_shift)
        self.train_action_head_only = bool(model_config.get("train_action_head_only", False))
        checkpoint_export_mode = str(
            model_config.get("checkpoint_export_mode", "auto")
        )
        if checkpoint_export_mode == "auto":
            checkpoint_export_mode = (
                "action_head_only" if self.train_action_head_only else "full"
            )
        valid_export_modes = {
            "full",
            "action_head_only",
            "action_head_plus_latest_non_action",
        }
        if checkpoint_export_mode not in valid_export_modes:
            raise ValueError(
                f"checkpoint_export_mode must be one of {sorted(valid_export_modes)}, "
                f"got {checkpoint_export_mode!r}"
            )
        if self.train_action_head_only and checkpoint_export_mode != "action_head_only":
            raise ValueError(
                "train_action_head_only requires checkpoint_export_mode=action_head_only"
            )
        self.checkpoint_export_mode = checkpoint_export_mode
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.num_embodiments = int(model_config.get("num_embodiments", 1))
        self.visual_loss_weight = float(model_config.get("visual_loss_weight", 1.0))
        self.action_loss_weight = float(model_config.get("action_loss_weight", 1.0))
        self.action_temporal_weight_mode = str(
            model_config.get("action_temporal_weight_mode", "uniform")
        )
        if self.action_temporal_weight_mode not in {"uniform", "linear", "quadratic"}:
            raise ValueError(
                "action_temporal_weight_mode must be uniform, linear, or quadratic; "
                f"got {self.action_temporal_weight_mode!r}"
            )
        self.action_smoothness_weight = float(model_config.get("action_smoothness_weight", 0.0))
        self.action_terminal_weight = float(model_config.get("action_terminal_weight", 0.0))

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
        transformer = CasualWorldActionTransformer_MoT(**transformer_cfg)
        transformer = load_pretrained_weights(
            transformer,
            transformer_pretrained,
            skip_action_expert=model_config.get("skip_action_expert", False),
            strict_load=model_config.get("strict_load", False),
        )
        transformer.to(self.device)
        if self.train_action_head_only:
            transformer.requires_grad_(False)
            trainable_names = []
            for name, parameter in transformer.named_parameters():
                if is_action_head_parameter(name):
                    parameter.requires_grad_(True)
                    trainable_names.append(name)
            if not trainable_names:
                raise RuntimeError("No action-head parameters matched the MoT model")
            self.visual_loss_weight = 0.0
            trainable_count = sum(
                parameter.numel()
                for parameter in transformer.parameters()
                if parameter.requires_grad
            )
            total_count = sum(parameter.numel() for parameter in transformer.parameters())
            print(
                "Action-head-only training enabled: "
                f"{len(trainable_names)} tensors, {trainable_count:,}/{total_count:,} parameters trainable"
            )
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
            ref_images = batch_dict["ref_images"]
            all_ref_latents = self.forward_vae(ref_images)
            ref_masks = batch_dict.get("ref_masks")
            if ref_masks is None:
                num_clean_latents = 1
            else:
                ref_counts = ref_masks.reshape(bs, ref_masks.shape[2], -1)[:, :, 0].sum(dim=1).long()
                if not torch.equal(ref_counts, ref_counts[:1].expand_as(ref_counts)):
                    raise ValueError(
                        "All samples in a batch must use the same number of visual history latents"
                    )
                num_clean_latents = int(ref_counts[0].item())
            if not 1 <= num_clean_latents < num_latent_frames:
                raise ValueError(
                    f"Expected 1 <= clean visual latents < {num_latent_frames}, got {num_clean_latents}"
                )
            first_frame_mask = torch.ones(
                bs,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=visual_latents.dtype,
                device=visual_latents.device,
            )
            first_frame_mask[:, :, :num_clean_latents] = 0
            insert_noisy_latents = (
                (1 - first_frame_mask) * all_ref_latents
                + first_frame_mask * noisy_latents
            )

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)
        if not self.expand_timesteps:
            num_clean_latents = 1
        ref_latents = insert_noisy_latents[:, :, :num_clean_latents]
        noisy_latents = insert_noisy_latents[:, :, num_clean_latents:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens * num_clean_latents
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
            action_only=self.train_action_head_only,
        )

        if visual_pred is None:
            visual_loss = action_pred.new_zeros(())
        else:
            visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        action_error = (action_pred.float() - action_target.float()).pow(2)
        action_dim_mask = batch_dict.get("action_dim_mask")
        if action_dim_mask is None:
            action_dim_mask = torch.ones_like(action_error[:, 0], dtype=torch.bool)
        else:
            action_dim_mask = action_dim_mask.to(action_error.device).bool()
        action_time_mask = batch_dict.get("action_time_mask")
        if action_time_mask is None:
            action_time_mask = torch.ones_like(action_error[:, :, 0], dtype=torch.bool)
        else:
            action_time_mask = action_time_mask.to(action_error.device).bool()

        action_steps = action_error.shape[1]
        progress = torch.linspace(0.0, 1.0, action_steps, device=action_error.device)
        if self.action_temporal_weight_mode == "linear":
            temporal_weight = 0.5 + progress
        elif self.action_temporal_weight_mode == "quadratic":
            temporal_weight = 0.25 + progress.square()
        else:
            temporal_weight = torch.ones_like(progress)
        temporal_weight = temporal_weight / temporal_weight.mean()
        loss_weight = (
            action_time_mask[:, :, None]
            * action_dim_mask[:, None, :]
            * temporal_weight[None, :, None]
        ).to(action_error.dtype)
        action_loss = (action_error * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)

        pred_clean_action = noisy_action.float() - action_sigma.float() * action_pred.float()
        if self.action_smoothness_weight > 0.0 and action_steps > 1:
            pred_velocity = pred_clean_action[:, 1:] - pred_clean_action[:, :-1]
            target_velocity = action.float()[:, 1:] - action.float()[:, :-1]
            velocity_error = (pred_velocity - target_velocity).pow(2)
            velocity_mask = (
                action_time_mask[:, 1:]
                & action_time_mask[:, :-1]
            )[:, :, None] & action_dim_mask[:, None, :]
            velocity_mask = velocity_mask.to(velocity_error.dtype)
            smoothness_loss = (velocity_error * velocity_mask).sum() / velocity_mask.sum().clamp_min(1.0)
            action_loss = action_loss + self.action_smoothness_weight * smoothness_loss

        if self.action_terminal_weight > 0.0:
            last_valid = action_time_mask.long().sum(dim=1).sub(1).clamp_min(0)
            batch_indices = torch.arange(bs, device=action_error.device)
            terminal_error = (
                pred_clean_action[batch_indices, last_valid]
                - action.float()[batch_indices, last_valid]
            ).pow(2)
            terminal_mask = action_dim_mask.to(terminal_error.dtype)
            terminal_loss = (terminal_error * terminal_mask).sum() / terminal_mask.sum().clamp_min(1.0)
            action_loss = action_loss + self.action_terminal_weight * terminal_loss
        return {
            "visual_loss": visual_loss * self.visual_loss_weight,
            "action_loss": action_loss * self.action_loss_weight,
        }

    def save_model_hook(self, models, weights, output_dir):
        """Export full weights or compact action/non-action artifacts."""
        compact_modes = {
            "action_head_only",
            "action_head_plus_latest_non_action",
        }
        if self.checkpoint_export_mode in compact_modes and self.is_main_process:
            if len(weights) == 0:
                weights.append(self.accelerator.get_state_dict(models[0]))
            if len(weights) != 1:
                raise RuntimeError(f"Expected one model state dict, got {len(weights)}")
            action_head_state, non_action_state = split_action_head_state_dict(
                weights[0]
            )
            if not action_head_state:
                raise RuntimeError("Action-head checkpoint filter matched no tensors")
            if self.checkpoint_export_mode == "action_head_plus_latest_non_action":
                if not non_action_state:
                    raise RuntimeError("Non-action checkpoint filter matched no tensors")
                self._save_latest_non_action_checkpoint(
                    models[0], non_action_state, output_dir
                )
            weights[0] = action_head_state
            self.logger.info(
                "Save action-head-only checkpoint with %d tensors",
                len(action_head_state),
            )
        return super().save_model_hook(models, weights, output_dir)

    def _save_latest_non_action_checkpoint(self, model, state_dict, output_dir):
        """Atomically replace the one retained non-action checkpoint."""
        latest_root = os.path.join(self.model_dir, "latest_non_action")
        transformer_dir = os.path.join(latest_root, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(
            model, keep_torch_compile=False
        )
        transformer = (
            unwrapped["transformer"]
            if isinstance(unwrapped, ModuleDict)
            else unwrapped
        )
        if hasattr(transformer, "save_config"):
            transformer.save_config(transformer_dir)

        transformer_state = {}
        for name, tensor in state_dict.items():
            if name.startswith("transformer."):
                name = name[len("transformer.") :]
            transformer_state[name] = tensor

        weights_path = os.path.join(
            transformer_dir, "diffusion_pytorch_model.bin"
        )
        partial_weights_path = weights_path + ".partial"
        if os.path.exists(partial_weights_path):
            os.remove(partial_weights_path)
        torch.save(transformer_state, partial_weights_path)
        os.replace(partial_weights_path, weights_path)

        metadata = {
            "source_checkpoint": os.path.basename(output_dir),
            "step": self.cur_step,
            "tensor_count": len(transformer_state),
        }
        metadata_path = os.path.join(latest_root, "metadata.json")
        partial_metadata_path = metadata_path + ".partial"
        with open(partial_metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
        os.replace(partial_metadata_path, metadata_path)
        self.logger.info(
            "Save latest non-action checkpoint with %d tensors to %s",
            len(transformer_state),
            weights_path,
        )

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

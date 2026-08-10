import functools
import os

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from giga_models import utils as gm_utils
from world_action_model.contact_transition_schema import (
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)
from world_action_model.matched_control_hashes import MatchedControlCollector
from world_action_model.models import CasualWorldActionTransformer_MoT
from world_action_model.trainer.wa_casual_trainer_pretrain import (
    _as_dim_mask,
    _as_time_mask,
    masked_mse,
)
from giga_train import ModuleDict, Trainer


def masked_contact_loss(
    prediction,
    target,
    mask=None,
    confidence=None,
    loss_type="bce",
):
    """Confidence-weighted loss for ordered contact-event query targets."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"contact target shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if loss_type == "bce":
        element_loss = F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )
    elif loss_type == "smooth_l1":
        element_loss = F.smooth_l1_loss(
            prediction.float(), target.float(), reduction="none"
        )
    else:
        raise ValueError(
            f"unsupported contact_loss_type={loss_type!r}; expected 'bce' or 'smooth_l1'"
        )

    weight = torch.ones_like(element_loss)
    for name, value in (("contact_loss_mask", mask), ("contact_confidence", confidence)):
        if value is None:
            continue
        value = value.to(device=prediction.device, dtype=element_loss.dtype)
        while value.ndim < element_loss.ndim:
            value = value.unsqueeze(-1)
        try:
            value = torch.broadcast_to(value, element_loss.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} cannot broadcast to "
                f"contact prediction {tuple(element_loss.shape)}"
            ) from exc
        weight = weight * value

    denominator = weight.sum().clamp_min(1.0)
    return (element_loss * weight).sum() / denominator


def apply_contact_query_assignment(
    prediction,
    target,
    mask,
    confidence,
    matcher=None,
):
    """Return query-aligned targets, leaving a clean Hungarian hook.

    v1.1 sidecars currently use deterministic segment ordering.  A future
    Hungarian matcher can be supplied as a callable with this same signature
    and must return ``(target, mask, confidence)`` aligned to prediction query
    order.  Keeping assignment outside the grouped loss avoids baking an
    ordering assumption into the loss implementation.
    """

    if matcher is None:
        return target, mask, confidence
    aligned = matcher(prediction, target, mask, confidence)
    if not isinstance(aligned, tuple) or len(aligned) != 3:
        raise TypeError("contact matcher must return (target, mask, confidence)")
    return aligned


def grouped_contact_segment_loss(
    prediction,
    target,
    mask=None,
    confidence=None,
    matcher=None,
):
    """Field-balanced v1.1 BCE/CE/Smooth-L1 loss on raw planner logits."""

    if prediction.shape != target.shape or prediction.shape[-1] != SEGMENT_TARGET_DIM:
        raise ValueError(
            "v1.1 contact shape mismatch: "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}, "
            f"expected last dim {SEGMENT_TARGET_DIM}"
        )
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if mask is None:
        mask = torch.ones_like(target)
    else:
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)
        mask = torch.broadcast_to(mask, target.shape)
    if confidence is None:
        confidence = torch.ones_like(target)
    else:
        confidence = confidence.to(device=prediction.device, dtype=prediction.dtype)
        confidence = torch.broadcast_to(confidence, target.shape)

    target, mask, confidence = apply_contact_query_assignment(
        prediction, target, mask, confidence, matcher
    )
    components = []
    active_components = []

    def weighted_mean(values, weights):
        denominator = weights.sum()
        value = (values * weights).sum() / denominator.clamp_min(1.0)
        return value, (denominator > 0).to(dtype=value.dtype)

    for field in SEGMENT_BINARY_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        values = F.binary_cross_entropy_with_logits(
            prediction[..., field_slice].float(),
            target[..., field_slice].float(),
            reduction="none",
        )
        weights = (mask[..., field_slice] * confidence[..., field_slice]).float()
        component, active = weighted_mean(values, weights)
        components.append(component)
        active_components.append(active)

    for field in SEGMENT_CATEGORICAL_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        field_target = target[..., field_slice].float()
        # Categorical supervision is valid only when the complete one-hot
        # group is known.  Partial category masks are treated as unknown.
        field_weight = (
            mask[..., field_slice].amin(dim=-1)
            * confidence[..., field_slice].amin(dim=-1)
        ).float()
        values = -(field_target * F.log_softmax(
            prediction[..., field_slice].float(), dim=-1
        )).sum(dim=-1)
        component, active = weighted_mean(values, field_weight)
        components.append(component)
        active_components.append(active)

    for field in SEGMENT_CONTINUOUS_FIELDS:
        field_slice = SEGMENT_FIELD_SLICES[field]
        values = F.smooth_l1_loss(
            prediction[..., field_slice].float().sigmoid(),
            target[..., field_slice].float(),
            reduction="none",
        )
        weights = (mask[..., field_slice] * confidence[..., field_slice]).float()
        component, active = weighted_mean(values, weights)
        components.append(component)
        active_components.append(active)

    values = torch.stack(components)
    active = torch.stack(active_components)
    return (values * active).sum() / active.sum().clamp_min(1.0)


def _as_action_loss_weight(
    weight: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize per-dimension action weights to a broadcastable [B,T,D]."""

    weight = torch.as_tensor(weight, device=device)
    if weight.ndim == 1:
        weight = weight[None, None, :]
    elif weight.ndim == 2:
        if tuple(weight.shape) == (seq_len, dim):
            weight = weight[None, :, :]
        elif tuple(weight.shape) == (batch_size, dim):
            weight = weight[:, None, :]
        else:
            raise ValueError(
                "rank-2 action_loss_weight must be [T,D] or [B,D], got "
                f"{tuple(weight.shape)}"
            )
    elif weight.ndim != 3:
        raise ValueError(
            f"action_loss_weight must have 1/2/3 dims, got {tuple(weight.shape)}"
        )
    expected = (batch_size, seq_len, dim)
    try:
        return torch.broadcast_to(weight, expected)
    except RuntimeError as exc:
        raise ValueError(
            f"action_loss_weight shape {tuple(weight.shape)} cannot broadcast to {expected}"
        ) from exc


def masked_weighted_action_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim_mask: torch.Tensor | None,
    time_mask: torch.Tensor | None,
    action_loss_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve ``masked_mse`` exactly unless explicit dimension weights exist."""

    if action_loss_weight is None:
        return masked_mse(pred, target, dim_mask=dim_mask, time_mask=time_mask)

    sq = (pred - target).pow(2)
    weights = _as_action_loss_weight(
        action_loss_weight,
        batch_size=pred.shape[0],
        seq_len=pred.shape[1],
        dim=pred.shape[2],
        device=pred.device,
    ).to(dtype=sq.dtype)
    if dim_mask is not None:
        weights = weights * dim_mask.to(dtype=sq.dtype, device=sq.device)

    # Normalize dimensions independently per token, matching masked_mse when
    # all explicit weights are one while allowing the ego grip channels to be
    # downweighted to 0.25.
    per_token = (sq * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    if time_mask is None:
        valid_token = (weights.sum(dim=-1) > 0).to(dtype=per_token.dtype)
        return (per_token * valid_token).sum() / valid_token.sum().clamp_min(1.0)
    time_mask = time_mask.to(dtype=per_token.dtype, device=per_token.device)
    valid_token = (weights.sum(dim=-1) > 0).to(dtype=per_token.dtype)
    time_weight = time_mask * valid_token
    return (per_token * time_weight).sum() / time_weight.sum().clamp_min(1.0)


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
        self.contact_loss_weight = float(model_config.get("contact_loss_weight", 0.0))
        self.contact_loss_type = str(model_config.get("contact_loss_type", "bce"))
        self.contact_schema_version = str(
            model_config.get("contact_schema_version", "contact_transition_v1")
        )
        self.contact_condition_mode = str(
            model_config.get("contact_condition_mode", "predicted")
        )
        self.require_contact_targets = bool(
            model_config.get("require_contact_targets", False)
        )
        self.matched_control_collector = None
        matched_control_config = model_config.get("matched_control_collector")
        if matched_control_config is not None and bool(
            matched_control_config.get("enabled", False)
        ):
            self.matched_control_collector = MatchedControlCollector(
                matched_control_config,
                rank=self.process_index,
                world_size=self.num_processes,
                trainer_seed=self.seed,
                project_dir=self.project_dir,
            )
        if self.contact_condition_mode not in {"predicted", "oracle"}:
            raise ValueError(
                "contact_condition_mode must be 'predicted' or 'oracle', got "
                f"{self.contact_condition_mode!r}"
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

        dim_mask = None
        if "action_dim_mask" in batch_dict:
            dim_mask = _as_dim_mask(
                batch_dict["action_dim_mask"],
                batch_size=bs,
                seq_len=action.shape[1],
                dim=action.shape[2],
                device=action.device,
            )

        # Missing embodiment dimensions must not inject random flow noise into
        # the shared action stream.  Mask both endpoints before interpolation,
        # then use the same mask for the objective below.
        flow_mask = 1.0 if dim_mask is None else dim_mask.to(dtype=action.dtype, device=action.device)
        action_for_flow = action * flow_mask
        action_noise = torch.randn_like(action) * flow_mask
        action_target = (action_noise - action_for_flow) * flow_mask
        noisy_action = action_noise * action_sigma + action_for_flow * (1 - action_sigma)
        if self.matched_control_collector is not None:
            self.matched_control_collector.record(
                global_step=self.cur_step,
                batch_dict=batch_dict,
                visual_timestep=visual_timestep,
                action_timestep=action_timestep,
                visual_noise=visual_noise,
                action_noise=action_noise,
            )

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

        contact_target = batch_dict.get("contact_target")
        contact_condition = batch_dict.get("contact_condition")
        contact_condition_mask = batch_dict.get("contact_condition_mask")
        if self.contact_condition_mode == "oracle":
            if contact_target is None:
                raise ValueError(
                    "oracle contact conditioning requires contact_target in the batch"
                )
            contact_condition = contact_target
            if contact_condition_mask is None:
                contact_condition_mask = batch_dict.get("contact_loss_mask")
                if (
                    self.contact_schema_version != SEGMENT_SCHEMA_VERSION
                    and contact_condition_mask is not None
                    and contact_condition_mask.ndim == 3
                ):
                    contact_condition_mask = contact_condition_mask.amax(dim=-1)

        model_output = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )
        if len(model_output) == 3:
            visual_pred, action_pred, contact_pred = model_output
        else:
            visual_pred, action_pred = model_output
            contact_pred = None

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        time_mask = None
        if "action_loss_mask" in batch_dict:
            raw_time_mask = batch_dict["action_loss_mask"]
            if self.action_repeats > 1:
                raw_time_mask = raw_time_mask.repeat(1, self.action_repeats)
            time_mask = _as_time_mask(
                raw_time_mask,
                batch_size=bs,
                seq_len=action.shape[1],
                device=action_pred.device,
            )
        action_loss_weight = batch_dict.get("action_loss_weight")
        if action_loss_weight is not None and self.action_repeats > 1:
            action_loss_weight = action_loss_weight.repeat(
                1, self.action_repeats, 1
            )
        action_loss = masked_weighted_action_mse(
            action_pred.float(),
            action_target.float(),
            dim_mask=dim_mask,
            time_mask=time_mask,
            action_loss_weight=action_loss_weight,
        )
        losses = {
            "visual_loss": visual_loss,
            "action_loss": action_loss,
        }
        if contact_pred is not None and contact_target is not None:
            if self.contact_schema_version == SEGMENT_SCHEMA_VERSION:
                contact_loss = grouped_contact_segment_loss(
                    contact_pred,
                    contact_target,
                    mask=batch_dict.get("contact_loss_mask"),
                    confidence=batch_dict.get("contact_confidence"),
                )
            else:
                contact_loss = masked_contact_loss(
                    contact_pred,
                    contact_target,
                    mask=batch_dict.get("contact_loss_mask"),
                    confidence=batch_dict.get("contact_confidence"),
                    loss_type=self.contact_loss_type,
                )
            losses["contact_loss"] = self.contact_loss_weight * contact_loss
        elif self.require_contact_targets:
            missing = "contact_target" if contact_target is None else "contact prediction head"
            raise ValueError(f"required {missing} is missing")
        return losses

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

    is_native_mot = any(".visual_expert." in key or ".action_expert." in key for key in state_dicts)
    if not strict_load and is_native_mot and hasattr(model, "load_from_mot_pretrained_state_dict"):
        loaded, skipped, unexpected = model.load_from_mot_pretrained_state_dict(
            state_dicts,
            skip_action_expert=skip_action_expert,
        )
        if unexpected:
            print(f"[WARNING] Unexpected keys in MoT checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
        if skipped:
            print(f"[WARNING] Skipped keys during MoT load: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        print(f"Loaded {len(loaded)} tensors from native MoT checkpoint.")
        return model

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

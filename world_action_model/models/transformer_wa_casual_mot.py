# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Mixture-of-Transformers (MoT) for world-action policy.
# Action stream (state + action) and visual stream (ref + future) use separate
# transformer parameters with global self-attention across both streams.

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from world_action_model.contact_transition_schema import (
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
)

from .transformer_wa_casual import (
    WanRotaryPosEmbed,
    WanRotaryPosEmbed1D,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
    _get_qkv_projections,
    WanAttnProcessor,
    WanAttention,
    FeedForward,
)
from .action_projectors import EmbodimentSpecificActionDecoder, EmbodimentSpecificActionEncoder

logger = logging.get_logger(__name__)

def _copy_param_sliced(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy the overlapping leading slice while preserving newly added dims."""
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    if dst.ndim != src.ndim:
        raise ValueError(f"Cannot slice-load: src shape {tuple(src.shape)} -> dst shape {tuple(dst.shape)}")
    overlap = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[overlap].copy_(src[overlap])

def _load_module_state_dict_sliced(module: nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str = "", skip=False) -> Tuple[list, list]:
    """Load *state_dict* into *module*, slicing tensors when shapes differ."""
    loaded, skipped = [], []
    module_state = module.state_dict()
    for key, dst in module_state.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if full_key not in state_dict or skip:
            skipped.append(full_key)
            continue
        src = state_dict[full_key]
        try:
            _copy_param_sliced(dst, src)
            loaded.append(full_key)
        except ValueError as exc:
            logger.warning("Skip loading %s: %s", full_key, exc)
            skipped.append(full_key)
    return loaded, skipped

def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)

def _chunk_temb(
    temb: torch.Tensor, scale_shift_table: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        return (
            shift_msa.squeeze(2),
            scale_msa.squeeze(2),
            gate_msa.squeeze(2),
            c_shift_msa.squeeze(2),
            c_scale_msa.squeeze(2),
            c_gate_msa.squeeze(2),
        )
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (scale_shift_table + temb.float()).chunk(
        6, dim=1
    )
    return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

def build_mot_attention_mask(
    num_state_tokens: int,
    num_action_tokens: int,
    num_ref_tokens: int,
    num_future_tokens: int,
    batch_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    num_contact_tokens: int = 0,
    action_attends_ref: bool = True,
    state_attends_ref: bool = True,
    contact_attends_ref: bool = True,
) -> torch.Tensor:
    """
    Causal mask for concatenated sequence [action_stream | visual_stream].

    action_stream layout: [state, contact, action]
    visual_stream layout: [ref, future]

    Contact queries are a current-observation latent: they may read state,
    reference-image and other contact tokens, but never noisy target actions or
    future-video tokens.  ``action_attends_ref=False`` forces the action stream
    through the contact bottleneck instead of retaining a direct visual path.

    With ``num_contact_tokens=0`` this is identical to the original
    [state, ref, action, future] mask.
    """
    la = num_state_tokens + num_contact_tokens + num_action_tokens
    lv = num_ref_tokens + num_future_tokens
    l_total = la + lv

    mask = torch.full((l_total, l_total), float("-inf"), device=device)
    ref_idx = torch.arange(la, la + num_ref_tokens, device=device)
    future_idx = torch.arange(la + num_ref_tokens, l_total, device=device)
    state_idx = torch.arange(0, num_state_tokens, device=device)
    contact_idx = torch.arange(
        num_state_tokens,
        num_state_tokens + num_contact_tokens,
        device=device,
    )
    action_only_idx = torch.arange(num_state_tokens + num_contact_tokens, la, device=device)

    def allow_rows(cols: torch.Tensor, rows: torch.Tensor) -> None:
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        mask[row_grid, col_grid] = 0.0

    state_cols = [state_idx]
    if state_attends_ref:
        state_cols.append(ref_idx)
    allow_rows(torch.cat(state_cols), state_idx)
    state_ref_cols = torch.cat([state_idx, ref_idx])
    allow_rows(state_ref_cols, ref_idx)

    if num_contact_tokens:
        contact_cols = [state_idx, contact_idx]
        if contact_attends_ref:
            contact_cols.insert(1, ref_idx)
        allow_rows(torch.cat(contact_cols), contact_idx)

    action_cols = [state_idx, contact_idx, action_only_idx]
    if action_attends_ref:
        action_cols.insert(1, ref_idx)
    allow_rows(torch.cat(action_cols), action_only_idx)

    allow_rows(torch.arange(0, l_total, device=device), future_idx)

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, l_total, l_total).to(dtype=dtype)

class ActionExpertBlock(nn.Module):
    def __init__(self, action_expert_dim: int, wan_dim: int, action_ffn_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = FP32LayerNorm(action_expert_dim, eps, elementwise_affine=False)
        self.norm3 = FP32LayerNorm(action_expert_dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=action_expert_dim,
            heads=num_heads,
            dim_head=wan_dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )
        self.attn2 = WanAttention(
            dim=action_expert_dim,
            heads=num_heads,
            dim_head=wan_dim // num_heads,
            eps=eps,
            added_kv_proj_dim=None,
            cross_attention_dim_head=wan_dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(action_expert_dim, eps, elementwise_affine=True)
        self.ffn = FeedForward(action_expert_dim, inner_dim=action_ffn_dim, activation_fn="gelu-approximate")
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, action_expert_dim) / action_expert_dim**0.5)

    # def load_from_dense_state_dict(self, dense_sd: Dict[str, torch.Tensor]) -> Tuple[list, list]:
    #     """Load Wan2.2 block weights; slice feature dims where action expert is narrower."""
    #     return _load_module_state_dict_sliced(self, dense_sd)

    def load_from_dense_block(self, dense_block: "WanTransformerBlock") -> Tuple[list, list]:
        return self.load_from_dense_state_dict(dense_block.state_dict())

class WanMoTTransformerBlock(nn.Module):
    """MoT block with separate action/visual experts and global self-attention."""

    def __init__(
        self,
        action_expert_dim: int,
        wan_dim: int,
        action_ffn_dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.action_expert = ActionExpertBlock(
            action_expert_dim, wan_dim, action_ffn_dim, num_heads, eps
        )
        self.visual_expert = WanTransformerBlock(
            wan_dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
        )

    def _expert_qkv(
        self,
        hidden_states: torch.Tensor,
        expert: WanTransformerBlock,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_x = (expert.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        query, key, value = _get_qkv_projections(expert.attn1, norm_x, None)
        query = expert.attn1.norm_q(query)
        key = expert.attn1.norm_k(key)

        heads = expert.attn1.heads
        query = query.unflatten(2, (heads, -1))
        key = key.unflatten(2, (heads, -1))
        value = value.unflatten(2, (heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb[0], rotary_emb[1])
            key = _apply_rotary_emb(key, rotary_emb[0], rotary_emb[1])

        return query, key, value

    def _expert_out_proj(
        self,
        attn_output: torch.Tensor,
        expert: WanTransformerBlock,
    ) -> torch.Tensor:
        out = attn_output.flatten(2, 3)
        out = expert.attn1.to_out[0](out)
        out = expert.attn1.to_out[1](out)
        return out

    def _build_cross_attn_kv_cache(
        self,
        expert: WanTransformerBlock,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if encoder_hidden_states is None:
            return None

        attn = expert.attn2
        if attn.fused_projections:
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        key = attn.norm_k(key)

        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        return {"key": key.detach(), "value": value.detach()}

    def _expert_cross_attn_output(
        self,
        hidden_states: torch.Tensor,
        expert: WanTransformerBlock,
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attn = expert.attn2
        query = attn.to_q(hidden_states)
        query = attn.norm_q(query)
        query = query.unflatten(2, (attn.heads, -1))

        if encoder_kv_cache is None:
            kv_cache = self._build_cross_attn_kv_cache(expert, encoder_hidden_states)
            if kv_cache is None:
                return torch.zeros_like(hidden_states)
        else:
            kv_cache = encoder_kv_cache

        attn_output = dispatch_attention_fn(
            query,
            kv_cache["key"],
            kv_cache["value"],
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.type_as(query)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)
        return attn_output

    def _expert_cross_attn_ffn(
        self,
        hidden_states: torch.Tensor,
        expert: WanTransformerBlock,
        encoder_hidden_states: Optional[torch.Tensor],
        c_shift_msa: torch.Tensor,
        c_scale_msa: torch.Tensor,
        c_gate_msa: torch.Tensor,
        encoder_kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or encoder_kv_cache is not None:
            norm_hidden_states = expert.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self._expert_cross_attn_output(
                norm_hidden_states, expert, encoder_hidden_states, encoder_kv_cache
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = (expert.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = expert.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        encoder_visual_hidden_states: torch.Tensor,
        encoder_action_hidden_states: torch.Tensor,
        action_temb: torch.Tensor,
        visual_temb: torch.Tensor,
        action_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        visual_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_shift, a_scale, a_gate, a_c_shift, a_c_scale, a_c_gate = _chunk_temb(
            action_temb, self.action_expert.scale_shift_table
        )
        v_shift, v_scale, v_gate, v_c_shift, v_c_scale, v_c_gate = _chunk_temb(
            visual_temb, self.visual_expert.scale_shift_table
        )

        q_action, k_action, v_action = self._expert_qkv(
            action_hidden_states, self.action_expert, a_shift, a_scale, action_rotary_emb
        )
        q_visual, k_visual, v_visual = self._expert_qkv(
            visual_hidden_states, self.visual_expert, v_shift, v_scale, visual_rotary_emb
        )

        query = torch.cat([q_action, q_visual], dim=1)
        key = torch.cat([k_action, k_visual], dim=1)
        value = torch.cat([v_action, v_visual], dim=1)

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=self_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        la = action_hidden_states.shape[1]
        attn_action = attn_output[:, :la]
        attn_visual = attn_output[:, la:]

        out_action = self._expert_out_proj(attn_action, self.action_expert)
        out_visual = self._expert_out_proj(attn_visual, self.visual_expert)

        action_hidden_states = (
            action_hidden_states.float() + out_action.float() * a_gate
        ).type_as(action_hidden_states)
        visual_hidden_states = (
            visual_hidden_states.float() + out_visual.float() * v_gate
        ).type_as(visual_hidden_states)

        action_hidden_states = self._expert_cross_attn_ffn(
            action_hidden_states,
            self.action_expert,
            encoder_action_hidden_states,
            a_c_shift,
            a_c_scale,
            a_c_gate,
        )
        visual_hidden_states = self._expert_cross_attn_ffn(
            visual_hidden_states,
            self.visual_expert,
            encoder_visual_hidden_states,
            v_c_shift,
            v_c_scale,
            v_c_gate,
        )

        return action_hidden_states, visual_hidden_states

    def forward_prefix_cache(
        self,
        action_prefix_hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        encoder_visual_hidden_states: torch.Tensor,
        encoder_action_hidden_states: torch.Tensor,
        action_prefix_temb: torch.Tensor,
        visual_temb: torch.Tensor,
        action_prefix_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        visual_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        a_shift, a_scale, a_gate, a_c_shift, a_c_scale, a_c_gate = _chunk_temb(
            action_prefix_temb, self.action_expert.scale_shift_table
        )
        v_shift, v_scale, v_gate, v_c_shift, v_c_scale, v_c_gate = _chunk_temb(
            visual_temb, self.visual_expert.scale_shift_table
        )

        q_action, k_action, v_action = self._expert_qkv(
            action_prefix_hidden_states, self.action_expert, a_shift, a_scale, action_prefix_rotary_emb
        )
        q_visual, k_visual, v_visual = self._expert_qkv(
            visual_hidden_states, self.visual_expert, v_shift, v_scale, visual_rotary_emb
        )

        query = torch.cat([q_action, q_visual], dim=1)
        key = torch.cat([k_action, k_visual], dim=1)
        value = torch.cat([v_action, v_visual], dim=1)

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=self_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        la = action_prefix_hidden_states.shape[1]
        attn_action = attn_output[:, :la]
        attn_visual = attn_output[:, la:]

        out_action = self._expert_out_proj(attn_action, self.action_expert)
        out_visual = self._expert_out_proj(attn_visual, self.visual_expert)

        action_prefix_hidden_states = (
            action_prefix_hidden_states.float() + out_action.float() * a_gate
        ).type_as(action_prefix_hidden_states)
        visual_hidden_states = (
            visual_hidden_states.float() + out_visual.float() * v_gate
        ).type_as(visual_hidden_states)

        action_encoder_kv_cache = self._build_cross_attn_kv_cache(
            self.action_expert, encoder_action_hidden_states
        )
        action_prefix_hidden_states = self._expert_cross_attn_ffn(
            action_prefix_hidden_states,
            self.action_expert,
            encoder_action_hidden_states,
            a_c_shift,
            a_c_scale,
            a_c_gate,
            action_encoder_kv_cache,
        )
        visual_hidden_states = self._expert_cross_attn_ffn(
            visual_hidden_states,
            self.visual_expert,
            encoder_visual_hidden_states,
            v_c_shift,
            v_c_scale,
            v_c_gate,
        )

        block_cache = {"key": key, "value": value}
        if action_encoder_kv_cache is not None:
            block_cache["action_encoder_key"] = action_encoder_kv_cache["key"]
            block_cache["action_encoder_value"] = action_encoder_kv_cache["value"]
        return action_prefix_hidden_states, visual_hidden_states, block_cache

    def forward_action_with_prefix_cache(
        self,
        action_hidden_states: torch.Tensor,
        encoder_action_hidden_states: torch.Tensor,
        action_temb: torch.Tensor,
        action_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        prefix_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        a_shift, a_scale, a_gate, a_c_shift, a_c_scale, a_c_gate = _chunk_temb(
            action_temb, self.action_expert.scale_shift_table
        )

        q_action, k_action, v_action = self._expert_qkv(
            action_hidden_states, self.action_expert, a_shift, a_scale, action_rotary_emb
        )
        key = torch.cat([prefix_cache["key"], k_action], dim=1)
        value = torch.cat([prefix_cache["value"], v_action], dim=1)

        attn_output = dispatch_attention_fn(
            q_action,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        out_action = self._expert_out_proj(attn_output, self.action_expert)
        action_hidden_states = (
            action_hidden_states.float() + out_action.float() * a_gate
        ).type_as(action_hidden_states)

        action_encoder_kv_cache = None
        if "action_encoder_key" in prefix_cache and "action_encoder_value" in prefix_cache:
            action_encoder_kv_cache = {
                "key": prefix_cache["action_encoder_key"],
                "value": prefix_cache["action_encoder_value"],
            }
        action_hidden_states = self._expert_cross_attn_ffn(
            action_hidden_states,
            self.action_expert,
            encoder_action_hidden_states,
            a_c_shift,
            a_c_scale,
            a_c_gate,
            action_encoder_kv_cache,
        )
        return action_hidden_states

    def load_from_dense_state_dict(self, dense_sd: Dict[str, torch.Tensor], skip_action_expert: bool = False) -> Tuple[list, list]:
        visual_loaded, visual_skipped = _load_module_state_dict_sliced(self.visual_expert, dense_sd)
        if skip_action_expert:
            print("--------------------------------")
            print('WARNING: Action expert weights are skipped')
            print("--------------------------------")
        action_loaded, action_skipped = _load_module_state_dict_sliced(self.action_expert, dense_sd, skip=skip_action_expert)
        action_loaded = [f"action_expert.{k}" for k in action_loaded]
        action_skipped = [f"action_expert.{k}" for k in action_skipped]
        visual_loaded = [f"visual_expert.{k}" for k in visual_loaded]
        visual_skipped = [f"visual_expert.{k}" for k in visual_skipped]
        return visual_loaded + action_loaded, visual_skipped + action_skipped

    def load_from_dense_block(self, dense_block: WanTransformerBlock) -> Tuple[list, list]:
        return self.load_from_dense_state_dict(dense_block.state_dict())

class CasualWorldActionTransformer_MoT(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    Mixture-of-Transformers world-action model.

    ``forward`` accepts the same arguments as :class:`CasualWorldActionTransformer`
    (``ref_latents``, ``noisy_latents``, ``timestep``, ``encoder_hidden_states``,
    ``state``, ``action``, etc.) so existing training / inference code can swap models
    without changes. Token-level execution lives in :meth:`_forward_tokens`.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "action_condition_embedder", "norm"]
    _no_split_modules = ["WanMoTTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "action_scale_shift_table", "norm1", "norm2", "norm3"]
    _repeated_blocks = ["WanMoTTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        action_expert_dim: int = 1024,
        action_ffn_dim: int = 4096,
        in_action_channels: int = 32,
        out_action_channels: int = 32,
        num_embodiments: int = 4,
        num_contact_tokens: int = 0,
        contact_target_dim: int = 0,
        contact_condition_dim: Optional[int] = None,
        contact_bottleneck: bool = False,
        contact_schema_version: str = "contact_transition_v1",
        contact_semantic_bottleneck: bool = False,
        contact_planner_num_heads: int = 8,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        out_action_channels = int(out_action_channels if out_action_channels is not None else in_action_channels)

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.action_condition_embedder = WanTimeTextImageEmbedding(
            dim=action_expert_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=action_expert_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=None,
            pos_embed_seq_len=None,
        )

        self.blocks = nn.ModuleList(
            [
                WanMoTTransformerBlock(
                    action_expert_dim=action_expert_dim,
                    wan_dim=inner_dim,
                    action_ffn_dim=action_ffn_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        self.action_rope = WanRotaryPosEmbed1D(attention_head_dim, rope_max_seq_len)
        self.state_encoder = EmbodimentSpecificActionEncoder(
            action_dim=in_action_channels,
            inner_dim=action_expert_dim,
            num_embodiments=num_embodiments,
        )
        self.action_encoder = EmbodimentSpecificActionEncoder(
            action_dim=in_action_channels,
            inner_dim=action_expert_dim,
            num_embodiments=num_embodiments,
        )
        self.action_norm_out = FP32LayerNorm(action_expert_dim, eps, elementwise_affine=False)
        self.action_scale_shift_table = nn.Parameter(torch.randn(1, 2, action_expert_dim) / action_expert_dim**0.5)
        self.action_decoder = EmbodimentSpecificActionDecoder(
            inner_dim=action_expert_dim,
            action_dim=out_action_channels,
            num_embodiments=num_embodiments,
        )

        self.num_contact_tokens = int(num_contact_tokens)
        self.contact_target_dim = int(contact_target_dim)
        self.contact_condition_dim = int(
            contact_condition_dim
            if contact_condition_dim is not None
            else contact_target_dim
        )
        self.contact_bottleneck = bool(contact_bottleneck)
        self.contact_schema_version = str(contact_schema_version)
        self.contact_semantic_bottleneck = bool(contact_semantic_bottleneck)
        if self.num_contact_tokens < 0:
            raise ValueError("num_contact_tokens must be non-negative")
        if self.contact_target_dim < 0 or self.contact_condition_dim < 0:
            raise ValueError("contact target/condition dimensions must be non-negative")
        if self.contact_bottleneck and not self.num_contact_tokens:
            raise ValueError("contact_bottleneck=True requires num_contact_tokens > 0")
        if self.contact_semantic_bottleneck:
            if not self.contact_bottleneck:
                raise ValueError(
                    "contact_semantic_bottleneck=True requires contact_bottleneck=True"
                )
            if self.contact_schema_version != SEGMENT_SCHEMA_VERSION:
                raise ValueError(
                    "semantic bottleneck requires contact_schema_version="
                    f"{SEGMENT_SCHEMA_VERSION!r}"
                )
            if self.contact_target_dim != SEGMENT_TARGET_DIM:
                raise ValueError(
                    "semantic bottleneck requires contact_target_dim="
                    f"{SEGMENT_TARGET_DIM}, got {self.contact_target_dim}"
                )
            if self.contact_condition_dim != SEGMENT_TARGET_DIM:
                raise ValueError(
                    "semantic bottleneck requires contact_condition_dim="
                    f"{SEGMENT_TARGET_DIM}, got {self.contact_condition_dim}"
                )
            if action_expert_dim % int(contact_planner_num_heads) != 0:
                raise ValueError(
                    "action_expert_dim must be divisible by contact_planner_num_heads"
                )

        if self.num_contact_tokens:
            self.contact_query_tokens = nn.Parameter(
                torch.randn(1, self.num_contact_tokens, action_expert_dim)
                / action_expert_dim**0.5
            )
            self.contact_condition_encoder = (
                nn.Linear(self.contact_condition_dim, action_expert_dim)
                if self.contact_condition_dim
                else None
            )
            self.contact_decoder = (
                nn.Linear(action_expert_dim, self.contact_target_dim)
                if self.contact_target_dim
                else None
            )
            if self.contact_semantic_bottleneck:
                self.contact_planner_ref_proj = nn.Linear(inner_dim, action_expert_dim)
                self.contact_planner_text_proj = nn.Linear(text_dim, action_expert_dim)
                self.contact_planner_attention = nn.MultiheadAttention(
                    action_expert_dim,
                    int(contact_planner_num_heads),
                    batch_first=True,
                )
                self.contact_planner_norm1 = nn.LayerNorm(action_expert_dim, eps=eps)
                self.contact_planner_ffn = nn.Sequential(
                    nn.Linear(action_expert_dim, action_expert_dim * 2),
                    nn.GELU(),
                    nn.Linear(action_expert_dim * 2, action_expert_dim),
                )
                self.contact_planner_norm2 = nn.LayerNorm(action_expert_dim, eps=eps)
            else:
                self.contact_planner_ref_proj = None
                self.contact_planner_text_proj = None
                self.contact_planner_attention = None
                self.contact_planner_norm1 = None
                self.contact_planner_ffn = None
                self.contact_planner_norm2 = None
        else:
            self.register_parameter("contact_query_tokens", None)
            self.contact_condition_encoder = None
            self.contact_decoder = None
            self.contact_planner_ref_proj = None
            self.contact_planner_text_proj = None
            self.contact_planner_attention = None
            self.contact_planner_norm1 = None
            self.contact_planner_ffn = None
            self.contact_planner_norm2 = None

    def _encode_state_tokens(self, state: torch.Tensor, embodiment_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if getattr(self.state_encoder, "uses_embodiment_id", False):
            return self.state_encoder(state, embodiment_id=embodiment_id)
        return self.state_encoder(state)

    def _encode_action_tokens_only(self, action: torch.Tensor, embodiment_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if getattr(self.action_encoder, "uses_embodiment_id", False):
            return self.action_encoder(action, embodiment_id=embodiment_id)
        return self.action_encoder(action)

    def _decode_action_tokens(self, action_states: torch.Tensor, embodiment_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if getattr(self.action_decoder, "uses_embodiment_id", False):
            return self.action_decoder(action_states, embodiment_id=embodiment_id)
        return self.action_decoder(action_states)

    def load_from_wan_pretrained_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        verbose: bool = True,
        skip_action_expert: bool = False,
    ) -> Tuple[list, list, list, list]:
        """
        Load weights from :class:`CasualWorldActionTransformer` (Wan2.2).

        - Shared modules (``rope``, ``patch_embedding``, ``condition_embedder``, ``norm_out``,
          ``proj_out``, ``scale_shift_table``) are loaded directly when keys match.
        - ``condition_embedder`` is also sliced into ``action_condition_embedder``.
        - ``norm_out`` / ``scale_shift_table`` are also sliced into ``action_norm_out`` /
          ``action_scale_shift_table``.
        - ``blocks.{i}.*`` is mapped to ``blocks.{i}.visual_expert.*`` (full) and
          ``blocks.{i}.action_expert.*`` (1D linear interpolation + alpha scaling when
          ``action_expert_dim`` / ``action_ffn_dim`` differ from Wan hidden / FFN sizes).
        - ``action_condition_embedder`` / ``action_scale_shift_table`` use the same
          interpolation when feature dims differ.
        - ``action_encoder`` / ``action_decoder`` are loaded with interpolation when present.
        """
        wan_dim = self.config.num_attention_heads * self.config.attention_head_dim
        loaded_keys: list = []
        skipped_keys: list = []
        unexpected_keys: list = []

        # Non-block weights shared with the dense Wan model.
        shared_prefixes = (
            "rope.",
            "patch_embedding.",
            "condition_embedder.",
            "norm_out.",
            "proj_out.",
            "scale_shift_table",
        )
        own_state = self.state_dict()
        for key, tensor in state_dict.items():
            if key.startswith("blocks."):
                continue
            if key.startswith("state_encoder.") or key.startswith("action_encoder.") or key.startswith("action_decoder."):
                if skip_action_expert:
                    skipped_keys.append(key)
                continue
            if not any(key == p.rstrip(".") or key.startswith(p) for p in shared_prefixes):
                unexpected_keys.append(key)
                continue
            if key not in own_state:
                unexpected_keys.append(key)
                continue
            try:
                _copy_param_sliced(own_state[key], tensor)
                loaded_keys.append(key)
            except ValueError as exc:
                if verbose:
                    logger.warning("Skip shared weight %s: %s", key, exc)
                skipped_keys.append(key)

        # Per-layer MoT blocks from dense Wan blocks.
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            block_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if not block_sd:
                if verbose:
                    logger.warning("No Wan weights found for %s", prefix)
                continue
            block_loaded, block_skipped = block.load_from_dense_state_dict(block_sd, skip_action_expert=skip_action_expert)
            loaded_keys.extend(f"{prefix}{k}" for k in block_loaded)
            skipped_keys.extend(f"{prefix}{k}" for k in block_skipped)
        if skip_action_expert:
            print("--------------------------------")
            print('WARNING: Action I/O weights are skipped')
            print("--------------------------------")
        else:
            # Action/state I/O (optional in WAM policy checkpoints).  Older
            # joint WAM checkpoints use action_encoder for both state and action;
            # use that as the default state encoder initialization when no
            # state_encoder-specific weights are present.
            for module_name, source_prefixes in (
                ("state_encoder", ("state_encoder", "action_encoder")),
                ("action_encoder", ("action_encoder",)),
                ("action_decoder", ("action_decoder",)),
            ):
                if not hasattr(self, module_name):
                    continue
                module = getattr(self, module_name)
                module_sd = {}
                for source_prefix in source_prefixes:
                    source_sd = {
                        f"{module_name}.{k[len(source_prefix) + 1:]}": v
                        for k, v in state_dict.items()
                        if k.startswith(f"{source_prefix}.")
                    }
                    if source_sd:
                        module_sd = source_sd
                        break
                if not module_sd:
                    continue
                mod_loaded, mod_skipped = _load_module_state_dict_sliced(
                    module, module_sd, prefix=module_name
                )
                loaded_keys.extend(mod_loaded)
                skipped_keys.extend(mod_skipped)

            # Action output head from Wan visual output head (interpolate when action_expert_dim != wan_dim).
            if "scale_shift_table" in state_dict and "action_scale_shift_table" in own_state:
                try:
                    _copy_param_sliced(own_state["action_scale_shift_table"], state_dict["scale_shift_table"])
                    loaded_keys.append("action_scale_shift_table")
                except ValueError as exc:
                    if verbose:
                        logger.warning("Skip action_scale_shift_table from scale_shift_table: %s", exc)
                    skipped_keys.append("action_scale_shift_table")

            cond_sd = {k[len("condition_embedder.") :]: v for k, v in state_dict.items() if k.startswith("condition_embedder.")}
            if cond_sd:
                cond_loaded, cond_skipped = _load_module_state_dict_sliced(
                    self.action_condition_embedder, cond_sd
                )
                loaded_keys.extend(
                    f"action_condition_embedder.{k}" for k in cond_loaded
                )
                skipped_keys.extend(f"action_condition_embedder.{k}" for k in cond_skipped)

        # Commit in-place updates back to parameters.
        self.load_state_dict(own_state, strict=True)

        if verbose:
            logger.info(
                "MoT Wan preload: loaded=%d skipped=%d unexpected=%d (wan_dim=%d action_expert_dim=%d)",
                len(loaded_keys),
                len(skipped_keys),
                len(unexpected_keys),
                wan_dim,
                self.config.action_expert_dim,
            )
        return loaded_keys, skipped_keys, unexpected_keys, [
            k for k in own_state if k not in loaded_keys and not k.startswith("blocks.")
        ]

    def load_from_mot_pretrained_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        skip_action_expert: bool = False,
    ) -> Tuple[list, list, list]:
        """Load a native MoT checkpoint and retain initialization in expanded dimensions."""
        own_state = self.state_dict()
        loaded, skipped, unexpected = [], [], []
        action_prefixes = (
            "state_encoder.",
            "action_encoder.",
            "action_decoder.",
            "action_condition_embedder.",
            "action_norm_out.",
            "action_scale_shift_table",
        )
        for key, src in state_dict.items():
            if key not in own_state:
                unexpected.append(key)
                continue
            if skip_action_expert and (
                key.startswith(action_prefixes) or ".action_expert." in key
            ):
                skipped.append(key)
                continue
            try:
                _copy_param_sliced(own_state[key], src)
                loaded.append(key)
            except ValueError as exc:
                logger.warning("Skip native MoT weight %s: %s", key, exc)
                skipped.append(key)
        self.load_state_dict(own_state, strict=True)
        return loaded, skipped, unexpected

    @staticmethod
    def activate_contact_segment_logits(logits: torch.Tensor) -> torch.Tensor:
        """Map raw v1.1 planner logits to the auditable 22-D semantics."""

        if logits.shape[-1] != SEGMENT_TARGET_DIM:
            raise ValueError(
                f"expected {SEGMENT_TARGET_DIM}-D contact logits, got {logits.shape[-1]}"
            )
        semantic = torch.empty_like(logits)
        for field in SEGMENT_BINARY_FIELDS + SEGMENT_CONTINUOUS_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            semantic[..., field_slice] = logits[..., field_slice].sigmoid()
        for field in SEGMENT_CATEGORICAL_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            semantic[..., field_slice] = logits[..., field_slice].softmax(dim=-1)
        return semantic

    def plan_contact_semantic(
        self,
        state: torch.Tensor,
        ref_visual_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plan v1.1 segments and optionally merge field-masked oracle values.

        The returned action-visible tensor is exactly 22-D.  Raw planner
        hidden states never leave this method and are never appended to the
        action stream.
        """

        if not self.contact_semantic_bottleneck:
            raise RuntimeError("contact semantic planner is not enabled")
        if ref_visual_hidden_states is None or ref_visual_hidden_states.shape[1] == 0:
            raise ValueError("semantic contact planner requires reference visual tokens")
        state_states = self._encode_state_tokens(state, embodiment_id=embodiment_id)
        batch_size = state_states.shape[0]
        planner_query = self.contact_query_tokens.to(
            device=state_states.device, dtype=state_states.dtype
        ).expand(batch_size, -1, -1)
        ref_context = self.contact_planner_ref_proj(
            ref_visual_hidden_states.to(dtype=state_states.dtype)
        )
        text_context = self.contact_planner_text_proj(
            encoder_hidden_states.to(dtype=state_states.dtype)
        )
        planner_context = torch.cat([state_states, ref_context, text_context], dim=1)
        attended, _ = self.contact_planner_attention(
            planner_query, planner_context, planner_context, need_weights=False
        )
        planner_hidden = self.contact_planner_norm1(planner_query + attended)
        planner_hidden = self.contact_planner_norm2(
            planner_hidden + self.contact_planner_ffn(planner_hidden)
        )
        contact_logits = self.contact_decoder(planner_hidden)
        predicted_semantic = self.activate_contact_segment_logits(contact_logits)

        if contact_condition is None:
            return contact_logits, predicted_semantic
        expected = (batch_size, self.num_contact_tokens, SEGMENT_TARGET_DIM)
        if tuple(contact_condition.shape) != expected:
            raise ValueError(
                f"contact_condition shape mismatch: got {tuple(contact_condition.shape)}, "
                f"expected {expected}"
            )
        if contact_condition_mask is None:
            raise ValueError("masked oracle merge requires contact_condition_mask")
        mask = contact_condition_mask.to(
            device=predicted_semantic.device, dtype=predicted_semantic.dtype
        )
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.shape not in {expected, expected[:2] + (1,)}:
            raise ValueError(
                "contact_condition_mask must have shape [B,Q], [B,Q,1], "
                f"or [B,Q,{SEGMENT_TARGET_DIM}], got {tuple(mask.shape)}"
            )
        mask = torch.broadcast_to(mask, expected).clamp(0.0, 1.0)
        oracle = contact_condition.to(
            device=predicted_semantic.device, dtype=predicted_semantic.dtype
        )
        semantic = mask * oracle + (1.0 - mask) * predicted_semantic.detach()
        return contact_logits, semantic

    def encode_action_tokens(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
        contact_semantic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int, int, int]:
        """Encode [state, contact queries, action] and build action RoPE.

        ``contact_condition`` is optional privileged/oracle conditioning for
        stage-gate experiments.  The default predicted-contact path supplies no
        condition, so queries must infer the interaction plan from current
        state, reference image and language through the causal mask.
        """
        state_states = self._encode_state_tokens(state, embodiment_id=embodiment_id)
        action_states = self._encode_action_tokens_only(action, embodiment_id=embodiment_id)
        contact_states = state_states.new_zeros(state_states.shape[0], 0, state_states.shape[-1])
        if self.num_contact_tokens and self.contact_semantic_bottleneck:
            if contact_semantic is None:
                raise ValueError(
                    "semantic bottleneck requires planned contact_semantic"
                )
            expected = (
                state_states.shape[0],
                self.num_contact_tokens,
                SEGMENT_TARGET_DIM,
            )
            if tuple(contact_semantic.shape) != expected:
                raise ValueError(
                    f"contact_semantic shape mismatch: got {tuple(contact_semantic.shape)}, "
                    f"expected {expected}"
                )
            # This 22 -> action width projection is the only planner-to-action
            # route.  In particular, planner_hidden is not concatenated here.
            contact_states = self.contact_condition_encoder(
                contact_semantic.to(
                    device=state_states.device, dtype=state_states.dtype
                )
            )
        elif self.num_contact_tokens:
            contact_states = self.contact_query_tokens.to(
                device=state_states.device,
                dtype=state_states.dtype,
            ).expand(state_states.shape[0], -1, -1)
            if contact_condition is not None:
                if self.contact_condition_encoder is None:
                    raise ValueError(
                        "contact_condition was provided but contact_condition_dim is zero"
                    )
                expected = (
                    state_states.shape[0],
                    self.num_contact_tokens,
                    self.contact_condition_dim,
                )
                if tuple(contact_condition.shape) != expected:
                    raise ValueError(
                        f"contact_condition shape mismatch: got {tuple(contact_condition.shape)}, "
                        f"expected {expected}"
                    )
                encoded_condition = self.contact_condition_encoder(
                    contact_condition.to(device=state_states.device, dtype=state_states.dtype)
                )
                if contact_condition_mask is not None:
                    condition_mask = contact_condition_mask.to(
                        device=state_states.device,
                        dtype=state_states.dtype,
                    )
                    if condition_mask.ndim == 2:
                        condition_mask = condition_mask.unsqueeze(-1)
                    if condition_mask.shape not in {
                        encoded_condition.shape,
                        encoded_condition.shape[:2] + (1,),
                    }:
                        raise ValueError(
                            "contact_condition_mask must have shape [B,Q], [B,Q,1], "
                            "or match the encoded contact condition"
                        )
                    encoded_condition = encoded_condition * condition_mask
                contact_states = contact_states + encoded_condition

        action_hidden_states = torch.cat([state_states, contact_states, action_states], dim=1)
        action_rotary_emb = self.action_rope(action_hidden_states)
        return (
            action_hidden_states,
            action_rotary_emb,
            state_states.shape[1],
            contact_states.shape[1],
            action_states.shape[1],
        )

    def encode_visual_tokens(
        self,
        ref_latents: torch.Tensor,
        future_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int, int, Optional[Dict[str, int]]]:
        """Encode ref (+ optional future) latents into visual stream tokens and RoPE."""
        if future_latents is not None:
            visual_input = torch.cat([ref_latents, future_latents], dim=2)
        else:
            visual_input = ref_latents

        visual_rotary_emb = self.rope(visual_input)
        visual_hidden_states = self.patch_embedding(visual_input)
        visual_hidden_states = visual_hidden_states.flatten(2).transpose(1, 2)

        p_t, p_h, p_w = self.config.patch_size
        _, _, num_frames, height, width = visual_input.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_ref_tokens = post_patch_width * post_patch_height
        num_future_tokens = visual_hidden_states.shape[1] - num_ref_tokens
        grid_info = None
        if future_latents is not None:
            grid_info = {
                "post_patch_num_frames": post_patch_num_frames,
                "post_patch_height": post_patch_height,
                "post_patch_width": post_patch_width,
                "p_t": p_t,
                "p_h": p_h,
                "p_w": p_w,
            }
        return visual_hidden_states, visual_rotary_emb, num_ref_tokens, num_future_tokens, grid_info

    def encode_joint_streams(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        ref_latents: torch.Tensor,
        future_latents: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        int,
        int,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        int,
        Optional[Dict[str, int]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Encode visual/action streams and run the opt-in semantic planner."""

        visual_hidden_states, visual_rotary_emb, num_ref_tokens, _, grid_info = (
            self.encode_visual_tokens(ref_latents, future_latents)
        )
        contact_logits = None
        contact_semantic = None
        if self.contact_semantic_bottleneck:
            contact_logits, contact_semantic = self.plan_contact_semantic(
                state,
                visual_hidden_states[:, :num_ref_tokens],
                encoder_hidden_states,
                embodiment_id=embodiment_id,
                contact_condition=contact_condition,
                contact_condition_mask=contact_condition_mask,
            )
        action_hidden_states, action_rotary_emb, num_state_tokens, num_contact_tokens, _ = (
            self.encode_action_tokens(
                state,
                action,
                embodiment_id=embodiment_id,
                contact_condition=contact_condition,
                contact_condition_mask=contact_condition_mask,
                contact_semantic=contact_semantic,
            )
        )
        return (
            action_hidden_states,
            action_rotary_emb,
            num_state_tokens,
            num_contact_tokens,
            visual_hidden_states,
            visual_rotary_emb,
            num_ref_tokens,
            grid_info,
            contact_logits,
            contact_semantic,
        )

    def _split_timestep(
        self,
        timestep: torch.Tensor,
        num_state_tokens: int,
        num_contact_tokens: int,
        num_action_only_tokens: int,
        num_ref_tokens: int,
        num_future_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split timesteps and insert clean contact-query timesteps if needed.

        Existing callers provide [state, ref, action, future].  For debugging,
        [state, ref, contact, action, future] is also accepted.  Contact queries
        always use timestep zero because they represent a clean current-plan
        prefix rather than a flow-matched target.
        """
        if timestep.ndim != 2:
            raise ValueError(
                f"per-token timestep must be rank 2, got shape {tuple(timestep.shape)}"
            )

        base_len = (
            num_state_tokens
            + num_ref_tokens
            + num_action_only_tokens
            + num_future_tokens
        )
        extended_len = base_len + num_contact_tokens
        if timestep.shape[1] not in {base_len, extended_len}:
            raise ValueError(
                f"timestep token count mismatch: got {timestep.shape[1]}, "
                f"expected {base_len} (without contact) or {extended_len} (with contact)"
            )

        state_end = num_state_tokens
        ref_end = state_end + num_ref_tokens
        if timestep.shape[1] == extended_len and num_contact_tokens:
            contact_end = ref_end + num_contact_tokens
            contact_timestep = timestep[:, ref_end:contact_end] * 0
            action_end = contact_end + num_action_only_tokens
            action_timestep = timestep[:, contact_end:action_end]
        else:
            contact_timestep = timestep.new_zeros(
                timestep.shape[0], num_contact_tokens
            )
            action_end = ref_end + num_action_only_tokens
            action_timestep = timestep[:, ref_end:action_end]

        state_timestep = timestep[:, :state_end]
        ref_timestep = timestep[:, state_end:ref_end]
        future_timestep = timestep[:, action_end:action_end + num_future_tokens]
        return (
            torch.cat([state_timestep, contact_timestep, action_timestep], dim=1),
            torch.cat([ref_timestep, future_timestep], dim=1),
        )

    def _process_blocks(
        self,
        action_hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        encoder_visual_hidden_states: torch.Tensor,
        encoder_action_hidden_states: torch.Tensor,
        action_temb: torch.Tensor,
        visual_temb: torch.Tensor,
        action_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        visual_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        self_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                action_hidden_states, visual_hidden_states = self._gradient_checkpointing_func(
                    block,
                    action_hidden_states,
                    visual_hidden_states,
                    encoder_visual_hidden_states,
                    encoder_action_hidden_states,
                    action_temb,
                    visual_temb,
                    action_rotary_emb,
                    visual_rotary_emb,
                    self_attention_mask,
                )
            else:
                action_hidden_states, visual_hidden_states = block(
                    action_hidden_states,
                    visual_hidden_states,
                    encoder_visual_hidden_states,
                    encoder_action_hidden_states,
                    action_temb,
                    visual_temb,
                    action_rotary_emb,
                    visual_rotary_emb,
                    self_attention_mask,
                )
        return action_hidden_states, visual_hidden_states

    def _empty_visual_stream(
        self,
        batch_size: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        head_dim = self.config.attention_head_dim
        visual_hidden_states = torch.zeros(batch_size, 0, dim, device=device, dtype=dtype)
        visual_rotary_emb = (
            torch.zeros(batch_size, 0, 1, head_dim, device=device, dtype=dtype),
            torch.zeros(batch_size, 0, 1, head_dim, device=device, dtype=dtype),
        )
        return visual_hidden_states, visual_rotary_emb

    def forward(self, *args, **kwargs):
        action_only = kwargs.pop("action_only", False)
        no_video_tokens = kwargs.pop("no_video_tokens", False)
        if self.training:
            if action_only:
                if no_video_tokens:
                    return self._forward_train_action_only_no_video(*args, **kwargs)
                return self._forward_train_action_only(*args, **kwargs)
            return self._forward_train(*args, **kwargs)
        if action_only:
            return self._forward_inference_action_only(*args, **kwargs)
        return self._forward_inference(*args, **kwargs)

    def _forward_tokens(
        self,
        action_hidden_states: torch.Tensor,
        visual_hidden_states: Optional[torch.Tensor],
        action_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        visual_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        num_state_tokens: int,
        num_contact_tokens: int,
        num_ref_tokens: int,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_action: bool = True,
        output_visual: bool = True,
        visual_grid_info: Optional[Dict[str, int]] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_pred_override: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size = action_hidden_states.shape[0]
        la = action_hidden_states.shape[1]
        if visual_hidden_states is None:
            visual_hidden_states, visual_rotary_emb = self._empty_visual_stream(
                batch_size,
                action_hidden_states.shape[2],
                action_hidden_states.device,
                action_hidden_states.dtype,
            )
        lv = visual_hidden_states.shape[1]
        num_action_only_tokens = la - num_state_tokens - num_contact_tokens
        if num_action_only_tokens < 0:
            raise ValueError(
                "action stream is shorter than its state/contact prefix: "
                f"la={la}, state={num_state_tokens}, contact={num_contact_tokens}"
            )
        num_future_tokens = lv - num_ref_tokens
        action_timestep, visual_timestep = self._split_timestep(
            timestep,
            num_state_tokens,
            num_contact_tokens,
            num_action_only_tokens,
            num_ref_tokens,
            num_future_tokens,
        )

        if action_timestep.ndim == 2:
            ts_action_seq_len = action_timestep.shape[1]
            action_timestep = action_timestep.flatten()
        else:
            ts_action_seq_len = None
        if visual_timestep.ndim == 2:
            ts_visual_seq_len = visual_timestep.shape[1]
            visual_timestep = visual_timestep.flatten()
        else:
            ts_visual_seq_len = None

        action_temb, action_timestep_proj, encoder_action_hidden_states, _ = self.action_condition_embedder(
            action_timestep, encoder_hidden_states, None, timestep_seq_len=ts_action_seq_len
        )
        visual_temb, visual_timestep_proj, encoder_visual_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                visual_timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_visual_seq_len
            )
        )
        if ts_action_seq_len is not None:
            action_timestep_proj = action_timestep_proj.unflatten(2, (6, -1))
            visual_timestep_proj = visual_timestep_proj.unflatten(2, (6, -1))
        else:
            action_timestep_proj = action_timestep_proj.unflatten(1, (6, -1))
            visual_timestep_proj = visual_timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        self_attention_mask = build_mot_attention_mask(
            num_state_tokens,
            num_action_only_tokens,
            num_ref_tokens,
            num_future_tokens,
            batch_size,
            self.config.num_attention_heads,
            action_hidden_states.device,
            action_hidden_states.dtype,
            num_contact_tokens=num_contact_tokens,
            action_attends_ref=not self.contact_bottleneck,
            state_attends_ref=not self.contact_semantic_bottleneck,
            contact_attends_ref=not self.contact_semantic_bottleneck,
        )

        action_hidden_states, visual_hidden_states = self._process_blocks(
            action_hidden_states,
            visual_hidden_states,
            encoder_visual_hidden_states,
            encoder_action_hidden_states,
            action_timestep_proj,
            visual_timestep_proj,
            action_rotary_emb,
            visual_rotary_emb,
            self_attention_mask,
        )

        action_pred = None
        contact_pred = contact_pred_override
        video_output = None

        if output_action:
            if action_temb.ndim == 3:
                action_shift, action_scale = (
                    self.action_scale_shift_table.unsqueeze(0).to(action_temb.device) + action_temb.unsqueeze(2)
                ).chunk(2, dim=2)
                action_shift = action_shift.squeeze(2)
                action_scale = action_scale.squeeze(2)
            else:
                action_shift, action_scale = (
                    self.action_scale_shift_table.to(action_temb.device) + action_temb.unsqueeze(1)
                ).chunk(2, dim=1)
            action_shift = action_shift.to(action_hidden_states.device)
            action_scale = action_scale.to(action_hidden_states.device)
            # action_hidden_states = (action_hidden_states.float() + action_scale * action_shift).type_as(action_hidden_states)
            action_hidden_states = (self.action_norm_out(action_hidden_states.float()) * (1 + action_scale) + action_shift).type_as(action_hidden_states)
            contact_end = num_state_tokens + num_contact_tokens
            if (
                contact_pred is None
                and num_contact_tokens
                and self.contact_decoder is not None
            ):
                contact_slice = action_hidden_states[:, num_state_tokens:contact_end]
                contact_pred = self.contact_decoder(contact_slice)
            action_slice = action_hidden_states[:, contact_end:]
            action_pred = self._decode_action_tokens(action_slice, embodiment_id=embodiment_id)

        if output_visual:
            if visual_grid_info is None:
                raise ValueError("visual_grid_info is required when output_visual=True")
            if visual_temb.ndim == 3:
                shift, scale = (self.scale_shift_table.unsqueeze(0).to(visual_temb.device) + visual_temb.unsqueeze(2)).chunk(2, dim=2)
                shift = shift.squeeze(2)
                scale = scale.squeeze(2)
            else:
                shift, scale = (self.scale_shift_table.to(visual_temb.device) + visual_temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(visual_hidden_states.device)
            scale = scale.to(visual_hidden_states.device)
            visual_for_out = (self.norm_out(visual_hidden_states.float()) * (1 + scale) + shift).type_as(
                visual_hidden_states
            )
            visual_for_out = self.proj_out(visual_for_out)

            g = visual_grid_info
            video_output = visual_for_out.reshape(
                batch_size,
                g["post_patch_num_frames"],
                g["post_patch_height"],
                g["post_patch_width"],
                g["p_t"],
                g["p_h"],
                g["p_w"],
                -1,
            )
            video_output = video_output.permute(0, 7, 1, 4, 2, 5, 3, 6)
            video_output = video_output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if contact_pred is not None:
                return video_output, action_pred, contact_pred
            return video_output, action_pred

        if output_action and output_visual:
            output = {"sample": video_output, "action_pred": action_pred}
            if contact_pred is not None:
                output["contact_pred"] = contact_pred
            return output
        if output_visual:
            return Transformer2DModelOutput(sample=video_output)
        output = {"action_pred": action_pred}
        if contact_pred is not None:
            output["contact_pred"] = contact_pred
        return output
    
    def _forward_train(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        (
            action_hidden_states,
            action_rotary_emb,
            num_state_tokens,
            num_contact_tokens,
            visual_hidden_states,
            visual_rotary_emb,
            num_ref_tokens,
            grid_info,
            contact_logits,
            _,
        ) = self.encode_joint_streams(
            state,
            action,
            ref_latents,
            noisy_latents,
            encoder_hidden_states,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )
        return self._forward_tokens(
            action_hidden_states=action_hidden_states,
            visual_hidden_states=visual_hidden_states,
            action_rotary_emb=action_rotary_emb,
            visual_rotary_emb=visual_rotary_emb,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            num_state_tokens=num_state_tokens,
            num_contact_tokens=num_contact_tokens,
            num_ref_tokens=num_ref_tokens,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            output_action=True,
            output_visual=True,
            visual_grid_info=grid_info,
            embodiment_id=embodiment_id,
            contact_pred_override=contact_logits,
        )

    def _forward_train_action_only(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        (
            action_hidden_states,
            action_rotary_emb,
            num_state_tokens,
            num_contact_tokens,
            visual_hidden_states,
            visual_rotary_emb,
            num_ref_tokens,
            _,
            contact_logits,
            _,
        ) = self.encode_joint_streams(
            state,
            action,
            ref_latents,
            noisy_latents,
            encoder_hidden_states,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )
        return self._forward_tokens(
            action_hidden_states=action_hidden_states,
            visual_hidden_states=visual_hidden_states,
            action_rotary_emb=action_rotary_emb,
            visual_rotary_emb=visual_rotary_emb,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            num_state_tokens=num_state_tokens,
            num_contact_tokens=num_contact_tokens,
            num_ref_tokens=num_ref_tokens,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            output_action=True,
            output_visual=False,
            embodiment_id=embodiment_id,
            contact_pred_override=contact_logits,
        )

    def _forward_train_action_only_no_video(
        self,
        ref_latents: torch.Tensor = None,
        noisy_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        if ref_latents is None:
            if self.contact_semantic_bottleneck:
                raise ValueError(
                    "semantic contact planner requires ref_latents in action-only training"
                )
            action_hidden_states, action_rotary_emb, num_state_tokens, num_contact_tokens, _ = self.encode_action_tokens(
                state,
                action,
                embodiment_id=embodiment_id,
                contact_condition=contact_condition,
                contact_condition_mask=contact_condition_mask,
            )
            visual_hidden_states, visual_rotary_emb = self._empty_visual_stream(
                action_hidden_states.shape[0],
                action_hidden_states.shape[2],
                action_hidden_states.device,
                action_hidden_states.dtype,
            )
            num_ref_tokens = 0
            contact_logits = None
        else:
            (
                action_hidden_states,
                action_rotary_emb,
                num_state_tokens,
                num_contact_tokens,
                visual_hidden_states,
                visual_rotary_emb,
                num_ref_tokens,
                _,
                contact_logits,
                _,
            ) = self.encode_joint_streams(
                state,
                action,
                ref_latents,
                None,
                encoder_hidden_states,
                embodiment_id=embodiment_id,
                contact_condition=contact_condition,
                contact_condition_mask=contact_condition_mask,
            )
        return self._forward_tokens(
            action_hidden_states=action_hidden_states,
            visual_hidden_states=visual_hidden_states,
            action_rotary_emb=action_rotary_emb,
            visual_rotary_emb=visual_rotary_emb,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            num_state_tokens=num_state_tokens,
            num_contact_tokens=num_contact_tokens,
            num_ref_tokens=num_ref_tokens,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            output_action=True,
            output_visual=False,
            embodiment_id=embodiment_id,
            contact_pred_override=contact_logits,
        )

    def _forward_inference(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        (
            action_hidden_states,
            action_rotary_emb,
            num_state_tokens,
            num_contact_tokens,
            visual_hidden_states,
            visual_rotary_emb,
            num_ref_tokens,
            grid_info,
            contact_logits,
            _,
        ) = self.encode_joint_streams(
            state,
            action,
            ref_latents,
            noisy_latents,
            encoder_hidden_states,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )
        return self._forward_tokens(
            action_hidden_states=action_hidden_states,
            visual_hidden_states=visual_hidden_states,
            action_rotary_emb=action_rotary_emb,
            visual_rotary_emb=visual_rotary_emb,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            num_state_tokens=num_state_tokens,
            num_contact_tokens=num_contact_tokens,
            num_ref_tokens=num_ref_tokens,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            output_action=True,
            output_visual=True,
            visual_grid_info=grid_info,
            embodiment_id=embodiment_id,
            contact_pred_override=contact_logits,
        )

    def reset_action_only_prefix_cache(self) -> None:
        self._action_only_prefix_cache = None

    def _condition_time_only(
        self,
        embedder: WanTimeTextImageEmbedding,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep = embedder.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(embedder.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = embedder.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = embedder.time_proj(embedder.act_fn(temb))
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        return temb, timestep_proj

    def _get_or_build_action_only_prefix_cache(
        self,
        action_prefix_hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        action_prefix_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        visual_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        action_prefix_timestep: torch.Tensor,
        visual_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor],
        num_state_tokens: int,
        num_contact_tokens: int,
        contact_semantic: Optional[torch.Tensor] = None,
        contact_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        cache = getattr(self, "_action_only_prefix_cache", None)
        if cache is not None:
            return cache

        if action_prefix_timestep.ndim == 2:
            ts_action_prefix_seq_len = action_prefix_timestep.shape[1]
            action_prefix_timestep = action_prefix_timestep.flatten()
        else:
            ts_action_prefix_seq_len = None
        if visual_timestep.ndim == 2:
            ts_visual_seq_len = visual_timestep.shape[1]
            visual_timestep = visual_timestep.flatten()
        else:
            ts_visual_seq_len = None

        action_temb, action_timestep_proj, encoder_action_hidden_states, _ = self.action_condition_embedder(
            action_prefix_timestep,
            encoder_hidden_states,
            None,
            timestep_seq_len=ts_action_prefix_seq_len,
        )
        visual_temb, visual_timestep_proj, encoder_visual_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                visual_timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_visual_seq_len,
            )
        )
        if ts_action_prefix_seq_len is not None:
            action_timestep_proj = action_timestep_proj.unflatten(2, (6, -1))
        else:
            action_timestep_proj = action_timestep_proj.unflatten(1, (6, -1))
        if ts_visual_seq_len is not None:
            visual_timestep_proj = visual_timestep_proj.unflatten(2, (6, -1))
        else:
            visual_timestep_proj = visual_timestep_proj.unflatten(1, (6, -1))

        prefix_attention_mask = build_mot_attention_mask(
            num_state_tokens=num_state_tokens,
            num_action_tokens=0,
            num_ref_tokens=visual_hidden_states.shape[1],
            num_future_tokens=0,
            batch_size=action_prefix_hidden_states.shape[0],
            num_heads=self.config.num_attention_heads,
            device=action_prefix_hidden_states.device,
            dtype=action_prefix_hidden_states.dtype,
            num_contact_tokens=num_contact_tokens,
            action_attends_ref=not self.contact_bottleneck,
            state_attends_ref=not self.contact_semantic_bottleneck,
            contact_attends_ref=not self.contact_semantic_bottleneck,
        )

        block_caches = []
        for block in self.blocks:
            action_prefix_hidden_states, visual_hidden_states, block_cache = block.forward_prefix_cache(
                action_prefix_hidden_states,
                visual_hidden_states,
                encoder_visual_hidden_states,
                encoder_action_hidden_states,
                action_timestep_proj,
                visual_timestep_proj,
                action_prefix_rotary_emb,
                visual_rotary_emb,
                prefix_attention_mask,
            )
            cached_key = block_cache["key"]
            cached_value = block_cache["value"]
            if self.contact_bottleneck:
                # Cached action queries must not regain a direct reference-token
                # path. Contact keys already absorb the reference observation
                # through the preceding prefix layers.
                prefix_len = action_prefix_hidden_states.shape[1]
                cached_key = cached_key[:, :prefix_len]
                cached_value = cached_value[:, :prefix_len]
            cached_block = {"key": cached_key.detach(), "value": cached_value.detach()}
            if "action_encoder_key" in block_cache and "action_encoder_value" in block_cache:
                cached_block["action_encoder_key"] = block_cache["action_encoder_key"].detach()
                cached_block["action_encoder_value"] = block_cache["action_encoder_value"].detach()
            block_caches.append(cached_block)

        cache = {
            "blocks": block_caches,
            "encoder_action_hidden_states": encoder_action_hidden_states.detach(),
            "num_ref_tokens": int(visual_hidden_states.shape[1]),
        }
        if contact_semantic is not None:
            cache["contact_semantic"] = contact_semantic.detach()
        if contact_logits is not None:
            cache["contact_logits"] = contact_logits.detach()
        self._action_only_prefix_cache = cache
        return cache

    def forward_action_stack_with_prefix_cache(
        self,
        action_hidden_states: torch.Tensor,
        encoder_action_hidden_states: torch.Tensor,
        action_timestep_proj: torch.Tensor,
        action_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        prefix_cache_blocks: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        for block, block_cache in zip(self.blocks, prefix_cache_blocks):
            action_hidden_states = block.forward_action_with_prefix_cache(
                action_hidden_states,
                encoder_action_hidden_states,
                action_timestep_proj,
                action_rotary_emb,
                block_cache,
            )
        return action_hidden_states

    def _forward_inference_action_only_cached(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        try:
            cache = getattr(self, "_action_only_prefix_cache", None)
            contact_logits = None
            contact_semantic = None
            if cache is None:
                visual_hidden_states, visual_rotary_emb, num_ref_tokens, _, _ = (
                    self.encode_visual_tokens(ref_latents)
                )
                if self.contact_semantic_bottleneck:
                    contact_logits, contact_semantic = self.plan_contact_semantic(
                        state,
                        visual_hidden_states[:, :num_ref_tokens],
                        encoder_hidden_states,
                        embodiment_id=embodiment_id,
                        contact_condition=contact_condition,
                        contact_condition_mask=contact_condition_mask,
                    )
            else:
                visual_hidden_states = None
                visual_rotary_emb = None
                num_ref_tokens = int(cache["num_ref_tokens"])
                if self.contact_semantic_bottleneck:
                    if "contact_semantic" not in cache:
                        raise RuntimeError(
                            "semantic action-only cache is missing contact_semantic"
                        )
                    contact_semantic = cache["contact_semantic"]

            (
                action_hidden_states,
                action_rotary_emb,
                num_state_tokens,
                num_contact_tokens,
                num_action_tokens,
            ) = self.encode_action_tokens(
                state,
                action,
                embodiment_id=embodiment_id,
                contact_condition=contact_condition,
                contact_condition_mask=contact_condition_mask,
                contact_semantic=contact_semantic,
            )
            interleaved_len = num_state_tokens + num_ref_tokens + num_action_tokens
            if timestep.ndim == 2:
                timestep = timestep[:, :interleaved_len]

            action_timestep, visual_timestep = self._split_timestep(
                timestep,
                num_state_tokens,
                num_contact_tokens,
                num_action_tokens,
                num_ref_tokens,
                0,
            )
            num_prefix_tokens = num_state_tokens + num_contact_tokens
            action_prefix_hidden_states = action_hidden_states[:, :num_prefix_tokens]
            action_hidden_states = action_hidden_states[:, num_prefix_tokens:]
            action_prefix_rotary_emb = (
                action_rotary_emb[0][:, :num_prefix_tokens],
                action_rotary_emb[1][:, :num_prefix_tokens],
            )
            action_rotary_emb = (
                action_rotary_emb[0][:, num_prefix_tokens:],
                action_rotary_emb[1][:, num_prefix_tokens:],
            )
            action_prefix_timestep = action_timestep[:, :num_prefix_tokens]
            action_timestep = action_timestep[:, num_prefix_tokens:]

            if cache is None:
                cache = self._get_or_build_action_only_prefix_cache(
                    action_prefix_hidden_states,
                    visual_hidden_states,
                    action_prefix_rotary_emb,
                    visual_rotary_emb,
                    action_prefix_timestep,
                    visual_timestep,
                    encoder_hidden_states,
                    encoder_hidden_states_image,
                    num_state_tokens,
                    num_contact_tokens,
                    contact_semantic=contact_semantic,
                    contact_logits=contact_logits,
                )

            if action_timestep.ndim == 2:
                ts_action_seq_len = action_timestep.shape[1]
                action_timestep = action_timestep.flatten()
            else:
                ts_action_seq_len = None
            action_temb, action_timestep_proj = self._condition_time_only(
                self.action_condition_embedder,
                action_timestep,
                encoder_hidden_states,
                ts_action_seq_len,
            )
            encoder_action_hidden_states = cache["encoder_action_hidden_states"]

            compiled_action_stack = getattr(self, "_compiled_forward_action_stack_with_prefix_cache", None)
            if compiled_action_stack is not None:
                action_hidden_states = compiled_action_stack(
                    action_hidden_states,
                    encoder_action_hidden_states,
                    action_timestep_proj,
                    action_rotary_emb,
                    cache["blocks"],
                )
            else:
                action_hidden_states = self.forward_action_stack_with_prefix_cache(
                    action_hidden_states,
                    encoder_action_hidden_states,
                    action_timestep_proj,
                    action_rotary_emb,
                    cache["blocks"],
                )

            if action_temb.ndim == 3:
                action_shift, action_scale = (
                    self.action_scale_shift_table.unsqueeze(0).to(action_temb.device) + action_temb.unsqueeze(2)
                ).chunk(2, dim=2)
                action_shift = action_shift.squeeze(2)
                action_scale = action_scale.squeeze(2)
            else:
                action_shift, action_scale = (
                    self.action_scale_shift_table.to(action_temb.device) + action_temb.unsqueeze(1)
                ).chunk(2, dim=1)
            action_shift = action_shift.to(action_hidden_states.device)
            action_scale = action_scale.to(action_hidden_states.device)
            action_hidden_states = (
                self.action_norm_out(action_hidden_states.float()) * (1 + action_scale) + action_shift
            ).type_as(action_hidden_states)
            return self._decode_action_tokens(action_hidden_states, embodiment_id=embodiment_id)
        finally:
            if USE_PEFT_BACKEND:
                unscale_lora_layers(self, lora_scale)

    def _forward_inference_action_only(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "GWP0.5 action-only inference requires torch.no_grad()/inference_mode() "
                "so the prefix KV cache can be used."
            )
        if ref_latents is None:
            raise ValueError("GWP0.5 action-only inference requires ref_latents to build the prefix KV cache.")
        if not getattr(self, "_enable_action_only_prefix_cache", False):
            raise RuntimeError("GWP0.5 action-only inference requires _enable_action_only_prefix_cache=True.")

        return self._forward_inference_action_only_cached(
            noisy_latents=noisy_latents,
            ref_latents=ref_latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            state=state,
            action=action,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )

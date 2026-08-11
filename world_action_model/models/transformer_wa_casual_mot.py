# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Mixture-of-Transformers (MoT) for world-action policy.
# Action stream (state + action) and visual stream (ref + future) use separate
# transformer parameters with global self-attention across both streams.

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from world_action_model.contact_transition_schema import (
    CONTACT_SEMANTIC_TRANSPORT_MODES,
    EQUAL_BIT_MIXING_MATRIX,
    EQUAL_BIT_MIXING_MATRIX_SHA256,
    EQUAL_BIT_NUM_BITS,
    EQUAL_BIT_VALID_STATES,
    SEGMENT_BINARY_FIELDS,
    SEGMENT_CATEGORICAL_FIELDS,
    SEGMENT_CONTINUOUS_FIELDS,
    SEGMENT_FIELD_SLICES,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TARGET_DIM,
    contact_transport_capacity_metadata,
    validated_equal_bit_mixing_matrix,
)
from world_action_model.contact_route_contract import (
    ContactCausalRoute,
    contact_packet_receipt,
    contact_route_metadata,
    resolve_contact_causal_route,
    route_action_attends_reference,
    route_action_uses_packet,
    route_rejects_oracle,
    route_uses_standalone_planner,
    validate_contact_route_model_config,
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
        contact_semantic_transport_mode: str = "soft",
        contact_time_bins: int = 16,
        contact_causal_route: Optional[str] = None,
        contact_planner_precursor: bool = False,
        production_matrix_id: Optional[str] = None,
        current_training_data_receipt: Optional[Dict[str, Any]] = None,
        current_training_data_receipt_sha256: Optional[str] = None,
        production_checkpoint_steps: Optional[Tuple[int, ...]] = None,
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
        self.contact_semantic_transport_mode = str(contact_semantic_transport_mode)
        self.contact_time_bins = int(contact_time_bins)
        self.contact_planner_precursor = bool(contact_planner_precursor)
        self.production_matrix_id = str(production_matrix_id or "")
        if self.production_matrix_id and self.production_matrix_id != (
            "contact-asym25-causal-main-q12-b16-stage2/v1"
        ):
            raise ValueError("unsupported contact causal production matrix ID")
        self.production_causal_matrix = bool(self.production_matrix_id)
        if not self.contact_planner_precursor and not self.production_matrix_id:
            # Keep historical Ego v6/op38 saved configs free of op42-only
            # default keys. New precursor/matrix runs retain both explicit
            # fields in their checkpoint config.
            legacy_config = dict(self._internal_dict)
            legacy_config.pop("contact_planner_precursor", None)
            legacy_config.pop("production_matrix_id", None)
            defaults = legacy_config.get("_use_default_values")
            if isinstance(defaults, list):
                legacy_config["_use_default_values"] = [
                    name
                    for name in defaults
                    if name not in {"contact_planner_precursor", "production_matrix_id"}
                ]
            self._internal_dict = FrozenDict(legacy_config)
        self.contact_causal_route = resolve_contact_causal_route(
            contact_causal_route,
            contact_semantic_bottleneck=self.contact_semantic_bottleneck,
            contact_semantic_transport_mode=self.contact_semantic_transport_mode,
        )
        self.contact_causal_route_explicit = contact_causal_route is not None and str(
            contact_causal_route
        ).strip() != ""
        if self.num_contact_tokens < 0:
            raise ValueError("num_contact_tokens must be non-negative")
        if self.contact_target_dim < 0 or self.contact_condition_dim < 0:
            raise ValueError("contact target/condition dimensions must be non-negative")
        if self.contact_bottleneck and not self.num_contact_tokens:
            raise ValueError("contact_bottleneck=True requires num_contact_tokens > 0")
        if self.contact_semantic_transport_mode not in CONTACT_SEMANTIC_TRANSPORT_MODES:
            raise ValueError(
                "unsupported contact_semantic_transport_mode="
                f"{self.contact_semantic_transport_mode!r}; expected one of "
                f"{CONTACT_SEMANTIC_TRANSPORT_MODES}"
            )
        if self.contact_time_bins < 2:
            raise ValueError(
                f"contact_time_bins must be at least 2, got {self.contact_time_bins}"
            )
        if (
            self.contact_semantic_transport_mode != "soft"
            and not self.contact_semantic_bottleneck
            and self.contact_causal_route is not ContactCausalRoute.AUX_ONLY_DIRECT_Q12
        ):
            raise ValueError(
                "hard contact semantic transport requires "
                "contact_semantic_bottleneck=True"
            )
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

        validate_contact_route_model_config(
            self.contact_causal_route if self.contact_causal_route_explicit else None,
            num_contact_tokens=self.num_contact_tokens,
            contact_target_dim=self.contact_target_dim,
            contact_condition_dim=self.contact_condition_dim,
            contact_bottleneck=self.contact_bottleneck,
            contact_schema_version=self.contact_schema_version,
            contact_semantic_bottleneck=self.contact_semantic_bottleneck,
            contact_time_bins=self.contact_time_bins,
        )
        self.contact_planner_enabled = self.contact_semantic_bottleneck or (
            self.contact_causal_route_explicit
            and route_uses_standalone_planner(self.contact_causal_route)
        ) or self.contact_planner_precursor
        self.contact_action_uses_packet = self.contact_semantic_bottleneck or (
            self.contact_causal_route_explicit
            and route_action_uses_packet(self.contact_causal_route)
        )
        self.contact_action_attends_reference = (
            route_action_attends_reference(self.contact_causal_route)
            if self.contact_causal_route_explicit or self.contact_semantic_bottleneck
            else not self.contact_bottleneck
        )
        self.contact_route_metadata = (
            contact_route_metadata(
                self.contact_causal_route,
                num_queries=self.num_contact_tokens,
                time_bins=self.contact_time_bins,
            )
            if self.contact_causal_route is not None
            else None
        )

        self.contact_transport_metadata = contact_transport_capacity_metadata(
            mode=self.contact_semantic_transport_mode,
            time_bins=self.contact_time_bins,
            num_queries=self.num_contact_tokens,
        )
        if self.contact_semantic_bottleneck:
            logger.info(
                "Contact semantic transport capacity: %s",
                self.contact_transport_metadata,
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
            if self.contact_planner_enabled:
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

        if self.contact_causal_route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
            self.register_buffer(
                "equal_bit_mixing_matrix",
                torch.tensor(EQUAL_BIT_MIXING_MATRIX, dtype=torch.float32),
                persistent=False,
            )

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

    def transport_contact_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        """Apply the configured action-visible semantic channel.

        ``soft`` returns the exact input tensor for v1.1 compatibility.
        Hard modes expose only grammar-valid finite states: binary fields are
        Bernoulli decisions, categorical groups are one-hot argmax decisions,
        and timing fields use uniformly spaced levels including 0 and 1.
        Every non-presence field is canonicalized to zero for absent queries.
        """

        if not torch.isfinite(semantic).all():
            raise ValueError("contact semantic transport requires finite values")
        if self.contact_semantic_transport_mode == "soft":
            return semantic
        if semantic.shape[-1] != SEGMENT_TARGET_DIM:
            raise ValueError(
                f"expected {SEGMENT_TARGET_DIM}-D contact semantic, "
                f"got {semantic.shape[-1]}"
            )

        hard = torch.zeros_like(semantic)
        for field in SEGMENT_BINARY_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            hard[..., field_slice] = (
                semantic[..., field_slice] >= 0.5
            ).to(dtype=semantic.dtype)
        for field in SEGMENT_CATEGORICAL_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            index = semantic[..., field_slice].argmax(dim=-1)
            hard[..., field_slice] = F.one_hot(
                index,
                num_classes=field_slice.stop - field_slice.start,
            ).to(dtype=semantic.dtype)
        time_scale = float(self.contact_time_bins - 1)
        for field in SEGMENT_CONTINUOUS_FIELDS:
            field_slice = SEGMENT_FIELD_SLICES[field]
            value = semantic[..., field_slice].clamp(0.0, 1.0)
            hard[..., field_slice] = torch.round(value * time_scale) / time_scale

        start_slice = SEGMENT_FIELD_SLICES["start_tau"]
        end_slice = SEGMENT_FIELD_SLICES["end_tau"]
        left_slice = SEGMENT_FIELD_SLICES["left_censored"]
        right_slice = SEGMENT_FIELD_SLICES["right_censored"]
        start = hard[..., start_slice]
        end = hard[..., end_slice]
        left_censored = hard[..., left_slice]
        right_censored = hard[..., right_slice]

        # Remove invalid-combination side channels.  Only uncensored segments
        # need sorting; a censored boundary has a unique canonical value.
        uncensored = (1.0 - left_censored) * (1.0 - right_censored)
        ordered_start = torch.minimum(start, end)
        ordered_end = torch.maximum(start, end)
        start = uncensored * ordered_start + (1.0 - uncensored) * start
        end = uncensored * ordered_end + (1.0 - uncensored) * end
        start = torch.where(left_censored.bool(), torch.zeros_like(start), start)
        end = torch.where(right_censored.bool(), torch.ones_like(end), end)
        hard[..., start_slice] = start
        hard[..., end_slice] = end

        presence = hard[..., SEGMENT_FIELD_SLICES["presence"]]
        for field in (
            SEGMENT_CATEGORICAL_FIELDS
            + SEGMENT_CONTINUOUS_FIELDS
            + ("left_censored", "right_censored")
        ):
            field_slice = SEGMENT_FIELD_SLICES[field]
            hard[..., field_slice] = hard[..., field_slice] * presence

        # IEEE-754 distinguishes +0 and -0 at the byte level.  Signed zero is
        # not a semantic grammar state, so canonicalize every numeric zero in
        # the complete transported tensor to the positive zero from
        # ``zeros_like`` before either hard return path.
        hard = torch.where(hard == 0, torch.zeros_like(hard), hard)

        if self.contact_semantic_transport_mode == "hard_st":
            # Subtract the identical semantic tensor before adding the hard
            # value.  This keeps the forward pass bit-exactly on the finite
            # grammar alphabet while retaining an identity surrogate gradient.
            # ``semantic + (hard - semantic).detach()`` can leave float32
            # cancellation residues that depend on the pre-quantized value.
            return hard.detach() + (semantic - semantic.detach())
        return hard.detach()

    def transport_unstructured_contact_logits(
        self, contact_logits: torch.Tensor
    ) -> torch.Tensor:
        """Map 22 logits to the exact 48,673-state equal-bit alphabet.

        Sixteen fixed Rademacher projections produce little-endian hard bits.
        Raw ids 48,673..65,535 all canonicalize to id zero.  The last six
        action-visible dimensions are literal positive zero.  Forward values
        are exact hard states; backward uses a finite sigmoid ST surrogate.
        """

        if self.contact_causal_route is not ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
            raise RuntimeError("equal-bit transport is not enabled")
        if contact_logits.shape[-1] != SEGMENT_TARGET_DIM:
            raise ValueError(
                f"expected {SEGMENT_TARGET_DIM}-D contact logits, got {contact_logits.shape[-1]}"
            )
        if not torch.isfinite(contact_logits).all():
            raise ValueError("equal-bit transport requires finite planner logits")
        matrix = self._validated_equal_bit_mixing_matrix(
            device=contact_logits.device, dtype=contact_logits.dtype
        )
        mixed_logits = F.linear(contact_logits, matrix) / math.sqrt(
            float(SEGMENT_TARGET_DIM)
        )
        soft_bits = mixed_logits.sigmoid()
        raw_bits = (soft_bits >= 0.5).to(dtype=contact_logits.dtype)
        integer_weights = (2 ** torch.arange(
            EQUAL_BIT_NUM_BITS, device=contact_logits.device, dtype=torch.int64
        )).view(*([1] * (raw_bits.ndim - 1)), EQUAL_BIT_NUM_BITS)
        raw_id = (
            raw_bits.to(dtype=torch.int64) * integer_weights
        ).sum(dim=-1)
        canonical_id = torch.where(
            raw_id < EQUAL_BIT_VALID_STATES,
            raw_id,
            torch.zeros_like(raw_id),
        )
        shifts = torch.arange(
            EQUAL_BIT_NUM_BITS, device=contact_logits.device, dtype=torch.int64
        ).view(*([1] * canonical_id.ndim), EQUAL_BIT_NUM_BITS)
        canonical_bits = (
            (canonical_id.unsqueeze(-1) >> shifts) & 1
        ).to(dtype=contact_logits.dtype)
        hard = torch.zeros(
            *contact_logits.shape[:-1],
            SEGMENT_TARGET_DIM,
            device=contact_logits.device,
            dtype=contact_logits.dtype,
        )
        hard[..., :EQUAL_BIT_NUM_BITS] = canonical_bits
        hard = torch.where(hard == 0, torch.zeros_like(hard), hard)
        soft_packet = torch.cat(
            [
                soft_bits,
                torch.zeros(
                    *soft_bits.shape[:-1],
                    SEGMENT_TARGET_DIM - EQUAL_BIT_NUM_BITS,
                    device=soft_bits.device,
                    dtype=soft_bits.dtype,
                ),
            ],
            dim=-1,
        )
        return hard.detach() + (soft_packet - soft_packet.detach())

    def _validated_equal_bit_mixing_matrix(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Validate the nonpersistent runtime copy, then build from source bytes."""

        self._validated_contact_route_metadata()
        source = validated_equal_bit_mixing_matrix()
        runtime = getattr(self, "equal_bit_mixing_matrix", None)
        if runtime is None or tuple(runtime.shape) != tuple(source.shape):
            raise RuntimeError("equal-bit runtime matrix shape differs from its receipt")
        if not torch.is_floating_point(runtime) or not torch.isfinite(runtime).all():
            raise RuntimeError("equal-bit runtime matrix dtype/values are invalid")
        runtime_int8 = runtime.detach().to(device="cpu", dtype=torch.int8).contiguous()
        source_int8 = torch.from_numpy(source)
        if not torch.equal(runtime_int8, source_int8) or not torch.equal(
            runtime.detach().to(device="cpu", dtype=torch.float32),
            source_int8.to(dtype=torch.float32),
        ):
            raise RuntimeError("equal-bit runtime matrix differs from source constants")
        runtime_sha = hashlib.sha256(runtime_int8.numpy().tobytes(order="C")).hexdigest()
        if runtime_sha != EQUAL_BIT_MIXING_MATRIX_SHA256:
            raise RuntimeError("equal-bit runtime matrix digest differs from its receipt")
        return torch.tensor(source, device=device, dtype=dtype)

    def _plan_contact_logits(
        self,
        state: torch.Tensor,
        ref_visual_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw auxiliary logits; planner hidden never leaves this method."""

        if not self.contact_planner_enabled:
            raise RuntimeError("contact planner is not enabled")
        if ref_visual_hidden_states is None or ref_visual_hidden_states.shape[1] == 0:
            raise ValueError("contact planner requires reference visual tokens")
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
        return self.contact_decoder(planner_hidden)

    def _transport_planned_semantic(
        self,
        contact_logits: torch.Tensor,
        contact_condition: Optional[torch.Tensor],
        contact_condition_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Activate and optionally oracle-merge semantic planner logits."""

        batch_size = contact_logits.shape[0]
        predicted_semantic = self.activate_contact_segment_logits(contact_logits)

        if contact_condition is None:
            if contact_condition_mask is not None:
                raise ValueError("contact_condition_mask requires contact_condition")
            return self.transport_contact_semantic(predicted_semantic)
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
        if not torch.isfinite(mask).all():
            raise ValueError("contact_condition_mask requires finite values")
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
        if not torch.isfinite(oracle).all():
            raise ValueError("contact_condition requires finite values")
        predicted_for_merge = predicted_semantic
        if self.contact_semantic_transport_mode == "soft":
            # Preserve the f3ae3bf oracle teacher-forcing behavior bitwise.
            predicted_for_merge = predicted_for_merge.detach()
        semantic = mask * oracle + (1.0 - mask) * predicted_for_merge
        return self.transport_contact_semantic(semantic)

    def plan_contact_semantic(
        self,
        state: torch.Tensor,
        ref_visual_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility semantic planner; hard-ST numerics remain unchanged."""

        if not self.contact_semantic_bottleneck:
            raise RuntimeError("contact semantic planner is not enabled")
        contact_logits = self._plan_contact_logits(
            state,
            ref_visual_hidden_states,
            encoder_hidden_states,
            embodiment_id=embodiment_id,
        )
        return contact_logits, self._transport_planned_semantic(
            contact_logits, contact_condition, contact_condition_mask
        )

    def plan_contact_route(
        self,
        state: torch.Tensor,
        ref_visual_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return auxiliary logits and the sole optional action-visible packet."""

        if (self.production_causal_matrix or route_rejects_oracle(self.contact_causal_route)) and (
            contact_condition is not None or contact_condition_mask is not None
        ):
            raise ValueError(
                f"route {self.contact_causal_route.value!r} rejects oracle/condition inputs"
            )
        if self.contact_causal_route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12:
            logits = self._plan_contact_logits(
                state,
                ref_visual_hidden_states,
                encoder_hidden_states,
                embodiment_id=embodiment_id,
            )
            # Match the hard-semantic planner/decoder/transport forward, then
            # discard the complete output before action token construction.
            self.transport_contact_semantic(
                self.activate_contact_segment_logits(logits)
            )
            return logits, None
        if self.contact_causal_route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
            logits = self._plan_contact_logits(
                state,
                ref_visual_hidden_states,
                encoder_hidden_states,
                embodiment_id=embodiment_id,
            )
            return logits, self.transport_unstructured_contact_logits(logits)
        return self.plan_contact_semantic(
            state,
            ref_visual_hidden_states,
            encoder_hidden_states,
            embodiment_id=embodiment_id,
            contact_condition=contact_condition,
            contact_condition_mask=contact_condition_mask,
        )

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
        if self.num_contact_tokens and self.contact_action_uses_packet:
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
        elif self.num_contact_tokens and not self.contact_planner_enabled:
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
        if self.contact_planner_enabled:
            contact_logits, contact_semantic = self.plan_contact_route(
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
        contact_precursor = kwargs.pop("contact_precursor", False)
        if self.training:
            if contact_precursor:
                if not self.contact_planner_precursor:
                    raise RuntimeError("contact precursor forward requires explicit precursor config")
                return self._forward_train_contact_precursor(*args, **kwargs)
            if action_only:
                if no_video_tokens:
                    return self._forward_train_action_only_no_video(*args, **kwargs)
                return self._forward_train_action_only(*args, **kwargs)
            return self._forward_train(*args, **kwargs)
        if action_only:
            return self._forward_inference_action_only(*args, **kwargs)
        return self._forward_inference(*args, **kwargs)

    def _forward_train_contact_precursor(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_condition_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], Transformer2DModelOutput]:
        """Train visual WAM + contact planner without constructing action tokens.

        Contact targets and masks remain trainer-only loss inputs.  This method
        rejects every model-side oracle input and never calls the action
        encoder, action decoder, or contact-condition encoder.
        """

        if contact_condition is not None or contact_condition_mask is not None:
            raise ValueError("route-neutral precursor rejects oracle/model-side contact inputs")
        if encoder_hidden_states_image is not None:
            raise ValueError("route-neutral precursor rejects an extra image condition path")
        visual_hidden_states, visual_rotary_emb, num_ref_tokens, _, grid_info = (
            self.encode_visual_tokens(ref_latents, noisy_latents)
        )
        contact_logits = self._plan_contact_logits(
            state,
            visual_hidden_states[:, :num_ref_tokens],
            encoder_hidden_states,
            embodiment_id=embodiment_id,
        )
        state_states = self._encode_state_tokens(state, embodiment_id=embodiment_id)
        action_rotary_emb = self.action_rope(state_states)
        return self._forward_tokens(
            action_hidden_states=state_states,
            visual_hidden_states=visual_hidden_states,
            action_rotary_emb=action_rotary_emb,
            visual_rotary_emb=visual_rotary_emb,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            num_state_tokens=state_states.shape[1],
            num_contact_tokens=0,
            num_ref_tokens=num_ref_tokens,
            encoder_hidden_states_image=None,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            output_action=False,
            output_visual=True,
            visual_grid_info=grid_info,
            embodiment_id=embodiment_id,
            contact_pred_override=contact_logits,
        )

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
        if self.contact_action_uses_packet and encoder_hidden_states_image is not None:
            raise ValueError(
                "strict contact route rejects nonempty encoder_hidden_states_image"
            )
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
            action_attends_ref=self.contact_action_attends_reference,
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
            if self.contact_planner_enabled:
                raise ValueError(
                    "contact planner route requires ref_latents in action-only training"
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
        self._action_only_prefix_cache_packet_receipt_sha256 = None

    def _validated_contact_route_metadata(self) -> Mapping[str, Any]:
        if self.contact_causal_route is None or self.contact_route_metadata is None:
            raise RuntimeError("contact route metadata is unavailable")
        expected = contact_route_metadata(
            self.contact_causal_route,
            num_queries=self.num_contact_tokens,
            time_bins=self.contact_time_bins,
        )
        if self.contact_route_metadata != expected:
            raise RuntimeError("runtime contact route metadata differs from source contract")
        return expected

    @staticmethod
    def _canonical_packet_bytes(packet: torch.Tensor) -> bytes:
        return (
            packet.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

    def _validate_hard_semantic_packet(self, packet: torch.Tensor) -> None:
        """Validate the exact B16 grammar without treating packets as logits."""

        flat = packet.reshape(-1, SEGMENT_TARGET_DIM)
        presence = flat[:, SEGMENT_FIELD_SLICES["presence"]].reshape(-1)
        if not torch.logical_or(presence == 0, presence == 1).all():
            raise RuntimeError("hard semantic cache presence must be binary")
        absent = presence == 0
        if bool(absent.any()) and not bool((flat[absent, 1:] == 0).all()):
            raise RuntimeError("absent hard semantic cache packet is not canonical")
        present = flat[presence == 1]
        for field in SEGMENT_CATEGORICAL_FIELDS:
            values = present[:, SEGMENT_FIELD_SLICES[field]]
            if not torch.logical_or(values == 0, values == 1).all() or not bool(
                (values.sum(dim=-1) == 1).all()
            ):
                raise RuntimeError(
                    f"hard semantic cache field {field!r} is not exactly one-hot"
                )
        for field in ("left_censored", "right_censored"):
            values = present[:, SEGMENT_FIELD_SLICES[field]]
            if not torch.logical_or(values == 0, values == 1).all():
                raise RuntimeError(
                    f"hard semantic cache field {field!r} is not binary"
                )
        if present.numel():
            lattice = torch.arange(
                self.contact_time_bins, device=packet.device, dtype=packet.dtype
            ) / float(self.contact_time_bins - 1)
            start = present[:, SEGMENT_FIELD_SLICES["start_tau"]].reshape(-1)
            end = present[:, SEGMENT_FIELD_SLICES["end_tau"]].reshape(-1)
            if not bool((start[:, None] == lattice[None, :]).any(dim=-1).all()) or not bool(
                (end[:, None] == lattice[None, :]).any(dim=-1).all()
            ):
                raise RuntimeError("hard semantic cache times are outside the B16 lattice")
            left = present[:, SEGMENT_FIELD_SLICES["left_censored"]].reshape(-1).bool()
            right = present[:, SEGMENT_FIELD_SLICES["right_censored"]].reshape(-1).bool()
            if bool((left & (start != 0)).any()):
                raise RuntimeError("left-censored cache packet has nonzero start_tau")
            if bool((right & (end != 1)).any()):
                raise RuntimeError("right-censored cache packet has end_tau != 1")
            uncensored = ~(left | right)
            if bool((uncensored & (start > end)).any()):
                raise RuntimeError("uncensored cache packet has start_tau > end_tau")

    def _validate_cached_contact_packet(
        self, packet: torch.Tensor, cache: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        encoder = cache.get("encoder_action_hidden_states")
        if not isinstance(packet, torch.Tensor) or not isinstance(encoder, torch.Tensor):
            raise RuntimeError("contact cache packet/encoder must be tensors")
        expected_shape = (encoder.shape[0], self.num_contact_tokens, SEGMENT_TARGET_DIM)
        if tuple(packet.shape) != expected_shape:
            raise RuntimeError(
                f"contact cache packet shape {tuple(packet.shape)} != {expected_shape}"
            )
        if not torch.is_floating_point(packet) or packet.dtype != encoder.dtype:
            raise RuntimeError("contact cache packet dtype differs from its encoder state")
        if packet.device != encoder.device:
            raise RuntimeError("contact cache packet device differs from its encoder state")
        if not torch.isfinite(packet).all():
            raise RuntimeError("contact cache packet requires finite values")
        zeros = packet == 0
        if bool(zeros.any()) and bool(torch.signbit(packet[zeros]).any()):
            raise RuntimeError("contact cache packet contains signed negative zero")

        if self.contact_causal_route is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST:
            bits = packet[..., :EQUAL_BIT_NUM_BITS]
            if not torch.logical_or(bits == 0, bits == 1).all():
                raise RuntimeError("equal-bit cache prefix is not binary")
            if not bool((packet[..., EQUAL_BIT_NUM_BITS:] == 0).all()):
                raise RuntimeError("equal-bit cache suffix is not canonical +0")
            weights = (2 ** torch.arange(
                EQUAL_BIT_NUM_BITS, device=packet.device, dtype=torch.int64
            )).view(*([1] * (bits.ndim - 1)), EQUAL_BIT_NUM_BITS)
            raw_id = (bits.to(dtype=torch.int64) * weights).sum(dim=-1)
            if bool((raw_id >= EQUAL_BIT_VALID_STATES).any()):
                raise RuntimeError("equal-bit cache contains an invalid raw id")
        else:
            route_metadata = self._validated_contact_route_metadata()
            finite_semantic_packet = bool(
                route_metadata.get("action_visible_packet_finite_semantic_grammar", False)
            )
            if finite_semantic_packet and (
                route_metadata.get("action_visible_packet")
                != "canonical_contact_semantic_22d"
                or self.contact_semantic_transport_mode not in {"hard_st", "hard_detached"}
                or self.contact_schema_version != SEGMENT_SCHEMA_VERSION
                or self.contact_target_dim != SEGMENT_TARGET_DIM
                or self.contact_condition_dim != SEGMENT_TARGET_DIM
            ):
                raise RuntimeError("finite semantic route receipt contradicts model schema")
            if not finite_semantic_packet:
                return contact_packet_receipt(
                    route_receipt=route_metadata,
                    packet_bytes=self._canonical_packet_bytes(packet),
                    packet_dtype=str(packet.dtype),
                    packet_shape=tuple(packet.shape),
                )
            if self.contact_time_bins != 16:
                raise RuntimeError("finite hard semantic cache requires the B16 grammar")
            self._validate_hard_semantic_packet(packet)

        return contact_packet_receipt(
            route_receipt=self._validated_contact_route_metadata(),
            packet_bytes=self._canonical_packet_bytes(packet),
            packet_dtype=str(packet.dtype),
            packet_shape=tuple(packet.shape),
        )

    def _validate_contact_action_only_cache(self, cache: Mapping[str, Any]) -> None:
        """Fail closed if route-specific cache contents are missing or excessive."""

        if self.contact_route_metadata is not None and cache.get(
            "contact_route_receipt_sha256"
        ) != self._validated_contact_route_metadata()["receipt_sha256"]:
            raise RuntimeError("action-only cache route receipt differs from the model")
        if self.contact_causal_route is ContactCausalRoute.AUX_ONLY_DIRECT_Q12:
            forbidden = {
                "contact_semantic",
                "contact_logits",
                "contact_code",
                "contact_packet_receipt",
            }
            if forbidden.intersection(cache):
                raise RuntimeError("direct route cache contains forbidden planner values")
            return
        if self.contact_action_uses_packet:
            equal_bit = (
                self.contact_causal_route
                is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
            )
            packet_key = "contact_code" if equal_bit else "contact_semantic"
            invalid_layout = packet_key not in cache
            if equal_bit:
                invalid_layout = invalid_layout or bool(
                    {"contact_semantic", "contact_logits"}.intersection(cache)
                )
            if invalid_layout:
                raise RuntimeError(
                    f"contact cache is missing or contradicts canonical {packet_key}"
                )
            recomputed = self._validate_cached_contact_packet(cache[packet_key], cache)
            if cache.get("contact_packet_receipt") != recomputed:
                raise RuntimeError("contact cache packet receipt differs from canonical bytes")
            expected_sha = getattr(
                self, "_action_only_prefix_cache_packet_receipt_sha256", None
            )
            if expected_sha is None or recomputed["receipt_sha256"] != expected_sha:
                raise RuntimeError("contact cache packet differs from its model-held receipt")

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
            self._validate_contact_action_only_cache(cache)
            return cache
        if self.contact_action_uses_packet and encoder_hidden_states_image is not None:
            raise ValueError(
                "strict contact cache rejects nonempty encoder_hidden_states_image"
            )

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
            action_attends_ref=self.contact_action_attends_reference,
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
            if not self.contact_action_attends_reference:
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
        if self.contact_route_metadata is not None:
            cache["contact_route_receipt_sha256"] = self._validated_contact_route_metadata()[
                "receipt_sha256"
            ]
        if (
            self.contact_causal_route
            is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
            and contact_semantic is not None
        ):
            packet = contact_semantic.detach()
            cache["contact_code"] = packet
        elif contact_semantic is not None:
            cache["contact_semantic"] = contact_semantic.detach()
        if (
            contact_logits is not None
            and self.contact_causal_route
            is not ContactCausalRoute.AUX_ONLY_DIRECT_Q12
            and self.contact_causal_route
            is not ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
        ):
            cache["contact_logits"] = contact_logits.detach()
        if self.contact_action_uses_packet:
            packet_key = (
                "contact_code"
                if self.contact_causal_route
                is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
                else "contact_semantic"
            )
            receipt = self._validate_cached_contact_packet(cache[packet_key], cache)
            cache["contact_packet_receipt"] = receipt
            self._action_only_prefix_cache_packet_receipt_sha256 = receipt[
                "receipt_sha256"
            ]
        self._action_only_prefix_cache = cache
        self._validate_contact_action_only_cache(cache)
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
        if self.contact_action_uses_packet and encoder_hidden_states_image is not None:
            raise ValueError(
                "strict contact route rejects nonempty encoder_hidden_states_image"
            )
        if (self.production_causal_matrix or route_rejects_oracle(self.contact_causal_route)) and (
            contact_condition is not None or contact_condition_mask is not None
        ):
            raise ValueError(
                f"route {self.contact_causal_route.value!r} rejects oracle/condition inputs"
            )
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
            if cache is not None:
                self._validate_contact_action_only_cache(cache)
            contact_logits = None
            contact_semantic = None
            if cache is None:
                visual_hidden_states, visual_rotary_emb, num_ref_tokens, _, _ = (
                    self.encode_visual_tokens(ref_latents)
                )
                if self.contact_planner_enabled:
                    contact_logits, contact_semantic = self.plan_contact_route(
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
                if self.contact_action_uses_packet:
                    packet_key = (
                        "contact_code"
                        if self.contact_causal_route
                        is ContactCausalRoute.EQUAL_BIT_UNSTRUCTURED_HARD_ST
                        else "contact_semantic"
                    )
                    if packet_key not in cache:
                        raise RuntimeError(
                            f"strict action-only cache is missing {packet_key}"
                        )
                    contact_semantic = cache[packet_key]

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
            action_pred = self._decode_action_tokens(
                action_hidden_states, embodiment_id=embodiment_id
            )
            if not return_dict:
                return action_pred
            output = {"action_pred": action_pred}
            if contact_logits is not None:
                output["contact_pred"] = contact_logits
            elif "contact_logits" in cache:
                output["contact_pred"] = cache["contact_logits"]
            if "contact_semantic" in cache:
                output["contact_semantic"] = cache["contact_semantic"]
            if "contact_code" in cache:
                output["contact_code"] = cache["contact_code"]
            return output
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

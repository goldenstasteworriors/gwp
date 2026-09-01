"""Real-Time Chunking utilities for action-flow policies.

``RTCConfig`` implements the original training-free, PiGDM-style inference
guidance from "Real-Time Execution of Action Chunking Flow Policies"
(arXiv:2506.07339).  ``training_time_rtc_condition`` implements the clean
prefix conditioning used by training-time RTC.  Both operate on normalized
model actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import torch


PrefixSchedule = Literal["exp", "linear", "ones", "hard"]


@dataclass(frozen=True)
class RTCConfig:
    """Inference-only RTC parameters."""

    execution_horizon: int = 10
    max_guidance_weight: float = 5.0
    prefix_schedule: PrefixSchedule = "exp"

    def __post_init__(self) -> None:
        if self.execution_horizon < 1:
            raise ValueError("RTC execution_horizon must be positive")
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError("RTC max_guidance_weight must be finite and positive")
        if self.prefix_schedule not in ("exp", "linear", "ones", "hard"):
            raise ValueError(f"unsupported RTC prefix_schedule={self.prefix_schedule!r}")


@dataclass(frozen=True)
class TrainingTimeRTCConfig:
    """Fixed simulated execution delay for training-time RTC.

    ``simulated_delay_steps`` is the number of initial action targets that are
    already committed while a new chunk is inferred.  Zero leaves the baseline
    flow-matching objective unchanged.
    """

    simulated_delay_steps: int = 0

    def __post_init__(self) -> None:
        if self.simulated_delay_steps < 0:
            raise ValueError("training-time RTC simulated_delay_steps must be non-negative")


def training_time_rtc_condition(
    action: torch.Tensor,
    noisy_action: torch.Tensor,
    action_sigma: torch.Tensor,
    action_timestep: torch.Tensor,
    *,
    config: TrainingTimeRTCConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Condition a flow-training action chunk on a clean committed prefix.

    The first ``simulated_delay_steps`` targets remain clean and receive
    timestep/sigma zero.  The returned prefix mask marks positions which must
    be excluded from every action prediction loss.
    """

    if action.ndim != 3:
        raise ValueError(f"action must be [B, H, A], got {tuple(action.shape)}")
    if noisy_action.shape != action.shape:
        raise ValueError("noisy_action must have the same shape as action")
    if action_timestep.ndim != 1 or action_timestep.shape[0] != action.shape[0]:
        raise ValueError("action_timestep must be [B]")
    if action_sigma.ndim != 3 or action_sigma.shape[0] != action.shape[0]:
        raise ValueError("action_sigma must be [B, 1, 1]")

    batch_size, horizon, _ = action.shape
    delay = config.simulated_delay_steps
    if delay >= horizon:
        raise ValueError(
            "training-time RTC simulated_delay_steps must be smaller than "
            f"the action horizon ({delay} >= {horizon})"
        )

    prefix_mask = torch.zeros(
        batch_size, horizon, dtype=torch.bool, device=action.device
    )
    if delay == 0:
        action_timesteps = action_timestep[:, None].expand(-1, horizon)
        action_sigmas = action_sigma.expand(-1, horizon, -1)
        return noisy_action, action_sigmas, action_timesteps, prefix_mask

    prefix_mask[:, :delay] = True
    noisy_action = torch.where(prefix_mask[:, :, None], action, noisy_action)
    action_sigmas = action_sigma.expand(-1, horizon, -1).clone()
    action_sigmas[prefix_mask] = 0
    action_timesteps = action_timestep[:, None].expand(-1, horizon).clone()
    action_timesteps[prefix_mask] = 0
    return noisy_action, action_sigmas, action_timesteps, prefix_mask


def prefix_attention_weights(
    *,
    horizon: int,
    prefix_length: int,
    inference_delay: int,
    execution_horizon: int,
    schedule: PrefixSchedule = "exp",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the frozen-prefix plus soft-overlap RTC mask."""

    if horizon < 1 or prefix_length < 0:
        raise ValueError("horizon must be positive and prefix_length non-negative")
    if inference_delay < 0 or execution_horizon < 1:
        raise ValueError("inference_delay must be non-negative and execution_horizon positive")

    effective_execution_horizon = max(execution_horizon, inference_delay)
    overlap_end = min(prefix_length, max(horizon - effective_execution_horizon, 0))
    frozen_end = min(int(inference_delay), overlap_end)
    weights = torch.zeros(horizon, device=device, dtype=dtype)
    if frozen_end:
        weights[:frozen_end] = 1
    soft_count = overlap_end - frozen_end
    if soft_count <= 0 or schedule == "hard":
        return weights
    if schedule == "ones":
        weights[frozen_end:overlap_end] = 1
        return weights

    soft = torch.linspace(1, 0, soft_count + 2, device=device, dtype=dtype)[1:-1]
    if schedule == "exp":
        soft = soft * torch.expm1(soft) / math.expm1(1.0)
    elif schedule != "linear":
        raise ValueError(f"unsupported RTC prefix schedule {schedule!r}")
    weights[frozen_end:overlap_end] = soft
    return weights


def guided_flow_velocity(
    sample: torch.Tensor,
    base_velocity_fn: Callable[[torch.Tensor], torch.Tensor],
    previous_chunk_tail: torch.Tensor | None,
    *,
    noise_level: torch.Tensor | float,
    inference_delay: int,
    config: RTCConfig,
) -> torch.Tensor:
    """Apply one RTC PiGDM correction to a flow-matching velocity field."""

    if previous_chunk_tail is None or previous_chunk_tail.shape[-2] == 0:
        return base_velocity_fn(sample)
    if sample.ndim != 3:
        raise ValueError("RTC sample must have shape [B, H, A]")

    prefix = torch.as_tensor(previous_chunk_tail, device=sample.device, dtype=sample.dtype)
    if prefix.ndim == 2:
        prefix = prefix.unsqueeze(0)
    if prefix.ndim != 3 or prefix.shape[0] not in (1, sample.shape[0]):
        raise ValueError("RTC prefix must have shape [H, A] or [B, H, A]")
    if prefix.shape[-1] > sample.shape[-1]:
        raise ValueError("RTC prefix action dimension exceeds the model action dimension")
    if prefix.shape[0] == 1 and sample.shape[0] > 1:
        prefix = prefix.expand(sample.shape[0], -1, -1)

    prefix_length = min(prefix.shape[-2], sample.shape[-2])
    target = torch.zeros_like(sample)
    target[:, :prefix_length, : prefix.shape[-1]] = prefix[:, :prefix_length]
    weights = prefix_attention_weights(
        horizon=sample.shape[-2],
        prefix_length=prefix_length,
        inference_delay=inference_delay,
        execution_horizon=config.execution_horizon,
        schedule=config.prefix_schedule,
        device=sample.device,
        dtype=sample.dtype,
    ).view(1, -1, 1)
    if not torch.count_nonzero(weights):
        return base_velocity_fn(sample)

    with torch.enable_grad():
        differentiable_sample = sample.detach().requires_grad_(True)
        velocity = base_velocity_fn(differentiable_sample)
        sigma = torch.as_tensor(noise_level, device=sample.device, dtype=sample.dtype)
        clean_estimate = differentiable_sample - sigma * velocity
        error = (target - clean_estimate) * weights
        correction = torch.autograd.grad(
            clean_estimate,
            differentiable_sample,
            grad_outputs=error.detach(),
            retain_graph=False,
            create_graph=False,
        )[0]

    sigma_f32 = sigma.float().clamp(0, 1)
    one_minus = 1 - sigma_f32
    denominator = one_minus.square().clamp_min(torch.finfo(torch.float32).eps)
    inv_r2 = (denominator + sigma_f32.square()) / denominator
    raw_weight = (one_minus / sigma_f32.clamp_min(torch.finfo(torch.float32).eps)) * inv_r2
    guidance_weight = torch.nan_to_num(
        raw_weight,
        nan=0.0,
        posinf=config.max_guidance_weight,
        neginf=0.0,
    ).clamp(0, config.max_guidance_weight).to(sample.dtype)
    return (velocity - guidance_weight * correction).detach()

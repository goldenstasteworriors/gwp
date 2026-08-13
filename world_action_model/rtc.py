"""Training-free Real-Time Chunking guidance for action flow inference.

The implementation follows the PiGDM-style flow correction from
"Real-Time Execution of Action Chunking Flow Policies" (arXiv:2506.07339).
It operates on normalized model actions; callers are responsible for aligning
the unexecuted tail of the previous chunk with the new chunk's first timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import torch


PrefixSchedule = Literal["exp", "linear", "ones", "hard"]


@dataclass(frozen=True)
class RTCConfig:
    """Inference-only RTC parameters.

    ``execution_horizon`` is the number of fresh actions reserved at the end
    of a generated chunk. Consequently, at most ``H - execution_horizon``
    actions are guided by the aligned tail of the previous chunk.
    """

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


def align_previous_chunk(
    previous_chunk: torch.Tensor,
    executed_steps: int,
) -> torch.Tensor:
    """Return the previous chunk tail aligned to the new observation time."""

    if previous_chunk.ndim not in (2, 3):
        raise ValueError("previous_chunk must have shape [H, A] or [B, H, A]")
    if executed_steps < 0:
        raise ValueError("executed_steps cannot be negative")
    time_axis = previous_chunk.ndim - 2
    horizon = previous_chunk.shape[time_axis]
    start = min(int(executed_steps), horizon)
    return previous_chunk.narrow(time_axis, start, horizon - start)


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
    """Build the paper's frozen-prefix plus soft-overlap RTC mask."""

    if horizon < 1 or prefix_length < 0:
        raise ValueError("horizon must be positive and prefix_length non-negative")
    if inference_delay < 0 or execution_horizon < 1:
        raise ValueError("inference_delay must be non-negative and execution_horizon positive")

    overlap_end = min(prefix_length, max(horizon - execution_horizon, 0))
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

    # Exclude both endpoints: the frozen region ends at one and the fresh
    # region starts at zero. This avoids a second hard edge in the mask.
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
    """Apply one RTC PiGDM correction to a flow-matching velocity field.

    ``noise_level`` follows Diffusers' FlowMatch scheduler convention: one at
    the noisy endpoint and zero at the clean endpoint.
    """

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

    # Paper Eq. (2)-(4), expressed in the scheduler's decreasing-noise sigma.
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

"""Gradient-routing utilities for action-conditioned consequence losses."""

from __future__ import annotations

import torch


def policy_only_surrogate(
    consequence_loss: torch.Tensor,
    predicted_clean_action: torch.Tensor,
) -> torch.Tensor:
    """Keep the consequence value while routing its gradient only through actions.

    The returned scalar has the same numerical value as ``consequence_loss``.
    Its gradient is the vector-Jacobian product of the consequence model with
    respect to ``predicted_clean_action``, followed by the policy graph that
    produced those actions. Parameters used only by the consequence forward do
    not receive gradients from this loss.
    """

    if consequence_loss.ndim != 0:
        consequence_loss = consequence_loss.mean()
    if not predicted_clean_action.requires_grad:
        raise ValueError("predicted_clean_action must require gradients")

    action_gradient = torch.autograd.grad(
        consequence_loss,
        predicted_clean_action,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0].detach()
    surrogate = (predicted_clean_action * action_gradient).sum()
    return consequence_loss.detach() + surrogate - surrogate.detach()

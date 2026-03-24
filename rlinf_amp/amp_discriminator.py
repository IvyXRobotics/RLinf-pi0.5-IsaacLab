"""
AMPDiscriminator
----------------
Standalone AMP (Adversarial Motion Priors) discriminator MLP.

Architecture:
    Input: [s_t || s_{t+1}]  →  Linear → ReLU → ... → Linear(1)

Reward formula (no gradient):
    amp_reward = amp_reward_coef * clamp(1 - (1/4)*(d - 1)^2, min=0)

When task_reward_lerp > 0, the final reward is blended:
    reward = (1 - task_reward_lerp) * amp_reward + task_reward_lerp * task_reward

This is a self-contained copy that does NOT depend on beyondAMP being installed.
It mirrors ``beyondAMP.modules.amp_discriminator.AMPDiscriminator`` exactly so
the two can be swapped freely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class AMPDiscriminator(nn.Module):
    """Style-reward discriminator for AMP.

    Parameters
    ----------
    input_dim : int
        Dimension of a single AMP observation (NOT the concatenated pair).
        The network receives ``2 * input_dim`` features (s_t concat s_{t+1}).
    amp_reward_coef : float
        Scalar that scales the discriminator reward signal.
    hidden_layer_sizes : list[int]
        Hidden layer widths, e.g. ``[1024, 512, 256]``.
    device : str
        Torch device.
    task_reward_lerp : float
        Mix ratio α ∈ [0, 1].  0 = pure AMP, 1 = pure task reward.
        Matches beyondAMP default of 0.9 (mostly task-driven).
    """

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: list[int],
        device: str,
        task_reward_lerp: float = 0.0,
    ):
        super().__init__()

        self.device = device
        self.input_dim = input_dim  # per-obs dim; network input = 2 * input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        layers = []
        in_dim = input_dim * 2  # concatenated pair
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers).to(device)
        self.amp_linear = nn.Linear(in_dim, 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 2*input_dim)  →  (B, 1) logit"""
        return self.amp_linear(self.trunk(x))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """WGAN-GP style gradient penalty on expert data.

        Penalises large discriminator gradients w.r.t. expert inputs,
        which stabilises training.

        Parameters
        ----------
        expert_state, expert_next_state : (B, input_dim)
        lambda_ : gradient penalty coefficient
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Push gradient norm toward 0 (not 1 like standard WGAN-GP)
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute AMP style reward.

        Parameters
        ----------
        state, next_state : (B, input_dim)
        task_reward       : (B,)
        normalizer        : optional AMPNormalizer

        Returns
        -------
        reward     : (B,)  — mixed reward (or pure AMP if lerp == 0)
        logit      : (B,)  — raw discriminator output
        amp_reward : (B,)  — discriminator reward before mixing
        """
        was_training = self.training
        self.eval()

        if normalizer is not None:
            state      = normalizer.normalize(state)
            next_state = normalizer.normalize(next_state)

        d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
        amp_reward = self.amp_reward_coef * torch.clamp(
            1.0 - 0.25 * (d - 1.0).pow(2), min=0.0
        )

        if self.task_reward_lerp > 0:
            reward = (
                (1.0 - self.task_reward_lerp) * amp_reward
                + self.task_reward_lerp * task_reward.unsqueeze(-1)
            )
        else:
            reward = amp_reward

        if was_training:
            self.train()

        return reward.squeeze(-1), d.squeeze(-1), amp_reward.squeeze(-1)

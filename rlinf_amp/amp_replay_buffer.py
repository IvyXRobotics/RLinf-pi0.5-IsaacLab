"""
AMPReplayBuffer
---------------
Fixed-capacity ring buffer that stores policy (s_t, s_{t+1}) AMP transitions
collected during rollout.  The discriminator is trained against samples from
this buffer (policy side) and the HDF5MotionDataset (expert side).

This is a direct port of beyondAMP's ``ReplayBuffer`` with no external deps.
"""

from __future__ import annotations

import numpy as np
import torch


class AMPReplayBuffer:
    """Fixed-capacity ring buffer for AMP transitions.

    Parameters
    ----------
    obs_dim : int
        Dimension of a single AMP observation (e.g. 25 for stack-cube default).
    buffer_size : int
        Maximum number of transitions to keep (beyondAMP default: 100 000).
    device : str
        Torch device.
    """

    def __init__(self, obs_dim: int, buffer_size: int, device: str):
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size
        self.device = device

        self.states      = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)

        self._ptr        = 0   # write pointer
        self.num_samples = 0   # how many valid entries exist

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> None:
        """Add a batch of transitions to the buffer.

        Handles wrap-around when the batch crosses the buffer boundary.

        Parameters
        ----------
        states, next_states : (B, obs_dim)
        """
        B = states.shape[0]
        end = self._ptr + B

        if end > self.buffer_size:
            # Split: fill to end, wrap remainder
            tail = self.buffer_size - self._ptr
            self.states     [self._ptr:]  = states     [:tail]
            self.next_states[self._ptr:]  = next_states[:tail]
            self.states     [:end - self.buffer_size] = states     [tail:]
            self.next_states[:end - self.buffer_size] = next_states[tail:]
        else:
            self.states     [self._ptr:end] = states
            self.next_states[self._ptr:end] = next_states

        self.num_samples = min(self.buffer_size, max(end, self.num_samples))
        self._ptr = end % self.buffer_size

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns
        -------
        (states, next_states) : (batch_size, obs_dim)
        """
        idx = np.random.choice(self.num_samples, size=batch_size, replace=True)
        return (
            self.states     [idx].to(self.device),
            self.next_states[idx].to(self.device),
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ):
        """Yield ``(states, next_states)`` for ``num_mini_batch`` iterations."""
        for _ in range(num_mini_batch):
            yield self.sample(mini_batch_size)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AMPReplayBuffer("
            f"capacity={self.buffer_size}, "
            f"filled={self.num_samples}, "
            f"obs_dim={self.obs_dim})"
        )

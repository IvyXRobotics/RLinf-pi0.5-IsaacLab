"""
AMPNormalizer
-------------
Running mean / std normalizer for AMP observations.

The discriminator sees normalised observations during both training and
inference, which keeps gradients stable across features with different
magnitudes (e.g. joint angles vs EEF positions).

State is maintained on CPU as numpy arrays (matches beyondAMP convention)
and converted to torch for GPU-side normalisation.
"""

from __future__ import annotations

import numpy as np
import torch


class AMPNormalizer:
    """Online running-mean / running-std normalizer.

    Parameters
    ----------
    obs_dim : int
        Dimension of a single AMP observation vector.
    epsilon : float
        Small constant to prevent division by zero.
    clip_val : float
        Clip normalised values to [-clip_val, clip_val].
    """

    def __init__(
        self,
        obs_dim: int,
        epsilon: float = 1e-8,
        clip_val: float = 5.0,
    ):
        self.obs_dim  = obs_dim
        self.epsilon  = epsilon
        self.clip_val = clip_val

        self.mean  = np.zeros(obs_dim, dtype=np.float64)
        self.var   = np.ones(obs_dim,  dtype=np.float64)
        self.count = 0

    # ------------------------------------------------------------------
    # Update (called with CPU numpy batches during training)
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Welford online update.

        Parameters
        ----------
        x : (B, obs_dim)  numpy float array
        """
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta       = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total_count
        m_a      = self.var   * self.count
        m_b      = batch_var  * batch_count
        m2       = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var  = m2 / total_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = total_count

    # ------------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a torch tensor using the running statistics.

        Parameters
        ----------
        x : (B, obs_dim)  on any device

        Returns
        -------
        (B, obs_dim) normalised tensor on the same device as x
        """
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(
            np.sqrt(self.var + self.epsilon), dtype=x.dtype, device=x.device
        )
        return torch.clamp((x - mean) / std, -self.clip_val, self.clip_val)

    # ------------------------------------------------------------------
    # Serialisation (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mean":  self.mean.copy(),
            "var":   self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean  = state["mean"].copy()
        self.var   = state["var"].copy()
        self.count = int(state["count"])

"""
PKLMotionDataset
----------------
Loads the pi0_gim_tshirt_dagger rollout dataset from .pkl files and provides
the same (s_t, s_{t+1}) transition sampling API as HDF5MotionDataset.

Dataset layout expected:
    <root>/good/rollout_*.pkl      <- successful rollouts (expert, label +1)
    <root>/failure/rollout_*.pkl   <- failed rollouts     (policy, label -1)

Each .pkl file is a list of timestep dicts with keys:
    qpos:           {'left_arm': [7-dim], 'right_arm': [7-dim]}
    gripper_target: {'left_arm': float,   'right_arm': float}

AMP observation (16-dim):
    [left_qpos(7) | right_qpos(7) | left_gripper(1) | right_gripper(1)]
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch


OBS_DIM = 16   # 7 + 7 + 1 + 1


def _extract_obs(step: dict) -> np.ndarray:
    """Extract 16-dim AMP observation from one timestep dict."""
    left_q  = np.array(step["qpos"]["left_arm"],  dtype=np.float32)   # (7,)
    right_q = np.array(step["qpos"]["right_arm"], dtype=np.float32)   # (7,)
    left_g  = np.array([step["gripper_target"]["left_arm"]],  dtype=np.float32)  # (1,)
    right_g = np.array([step["gripper_target"]["right_arm"]], dtype=np.float32)  # (1,)
    return np.concatenate([left_q, right_q, left_g, right_g])          # (16,)


class PKLMotionDataset:
    """Rollout dataset backed by .pkl files in good/ and failure/ folders.

    Provides the same interface as HDF5MotionDataset:

        ``sample_batch(batch_size)``    -> ``(t_idx, tp1_idx)``
        ``build_transition(t, tp1)``    -> ``(obs_t, obs_tp1)``
        ``sample_expert(batch_size)``   -> ``(obs_t, obs_tp1)`` from good/ only
        ``sample_policy(batch_size)``   -> ``(obs_t, obs_tp1)`` from failure/ only
        ``observation_dim``             -> 16

    Parameters
    ----------
    root_dir : str
        Path to the folder containing good/ and failure/ subdirectories.
    device : str
        Torch device for returned tensors.
    rollout_limit : int or None
        If set, load only the first N rollouts from each split (for debugging).
    """

    observation_dim: int = OBS_DIM

    def __init__(
        self,
        root_dir: str,
        device: str = "cpu",
        rollout_limit: int | None = None,
    ):
        self.root_dir = root_dir
        self.device   = device

        expert_obs, expert_lens = self._load_split("good",    rollout_limit)
        policy_obs, policy_lens = self._load_split("failure", rollout_limit)

        # Contiguous tensors  (total_T, 16)
        self._expert_obs: torch.Tensor = torch.tensor(expert_obs, dtype=torch.float32).to(device)
        self._policy_obs: torch.Tensor = torch.tensor(policy_obs, dtype=torch.float32).to(device)

        # Valid transition indices (no episode boundary crossing)
        self._expert_t, self._expert_tp1 = self._build_indices(expert_lens, device)
        self._policy_t, self._policy_tp1 = self._build_indices(policy_lens, device)

        # Combined pool (all transitions, used by sample_batch / build_transition)
        combined_obs = np.concatenate([expert_obs, policy_obs], axis=0)
        self._all_obs: torch.Tensor = torch.tensor(combined_obs, dtype=torch.float32).to(device)

        # Offset policy indices into the combined array
        expert_total = len(expert_obs)
        all_t   = torch.cat([self._expert_t,   self._policy_t   + expert_total])
        all_tp1 = torch.cat([self._expert_tp1, self._policy_tp1 + expert_total])
        self._all_t,   self._all_tp1   = all_t, all_tp1

        self.total_dataset_size = len(expert_obs) + len(policy_obs)
        self._num_expert = len(expert_obs)
        self._num_policy = len(policy_obs)

        print(
            f"[PKLMotionDataset] root={root_dir} | "
            f"expert_steps={self._num_expert} ({len(expert_lens)} rollouts) | "
            f"policy_steps={self._num_policy} ({len(policy_lens)} rollouts) | "
            f"obs_dim={OBS_DIM}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_split(
        self, split: str, rollout_limit: int | None
    ) -> Tuple[np.ndarray, List[int]]:
        """Load all pkl files from <root>/<split>/ and return (obs_array, lengths)."""
        folder = os.path.join(self.root_dir, split)
        paths  = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        if not paths:
            raise FileNotFoundError(f"No .pkl files found in {folder}")
        if rollout_limit is not None:
            paths = paths[:rollout_limit]

        all_obs: List[np.ndarray] = []
        lengths: List[int]        = []

        for path in paths:
            with open(path, "rb") as f:
                steps = pickle.load(f)
            if len(steps) < 2:
                continue
            traj = np.stack([_extract_obs(s) for s in steps], axis=0)  # (T, 16)
            all_obs.append(traj)
            lengths.append(len(traj))

        return np.concatenate(all_obs, axis=0), lengths  # (total_T, 16)

    @staticmethod
    def _build_indices(
        lengths: List[int], device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_t, idx_tp1 = [], []
        offset = 0
        for L in lengths:
            t = torch.arange(offset, offset + L - 1, dtype=torch.long)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += L
        return (
            torch.cat(idx_t).to(device),
            torch.cat(idx_tp1).to(device),
        )

    # ------------------------------------------------------------------
    # Sampling API (mirrors HDF5MotionDataset)
    # ------------------------------------------------------------------

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random transition indices from the combined pool."""
        idx = torch.randint(0, len(self._all_t), (batch_size,), device=self.device)
        return self._all_t[idx], self._all_tp1[idx]

    def build_transition(
        self, t: torch.Tensor, tp1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (obs_t, obs_tp1) for given indices into the combined pool."""
        return self._all_obs[t], self._all_obs[tp1]

    def sample_expert(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample transitions from the good/ (expert) split only."""
        idx = torch.randint(0, len(self._expert_t), (batch_size,), device=self.device)
        t   = self._expert_t[idx]
        tp1 = self._expert_tp1[idx]
        return self._expert_obs[t], self._expert_obs[tp1]

    def sample_policy(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample transitions from the failure/ (policy) split only."""
        idx = torch.randint(0, len(self._policy_t), (batch_size,), device=self.device)
        t   = self._policy_t[idx]
        tp1 = self._policy_tp1[idx]
        return self._policy_obs[t], self._policy_obs[tp1]

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ):
        """Yield (expert_obs_t, expert_obs_tp1) tuples — mirrors HDF5MotionDataset."""
        for _ in range(num_mini_batch):
            yield self.sample_expert(mini_batch_size)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PKLMotionDataset("
            f"expert_steps={self._num_expert}, "
            f"policy_steps={self._num_policy}, "
            f"obs_dim={OBS_DIM})"
        )

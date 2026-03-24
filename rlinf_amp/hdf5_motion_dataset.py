"""
HDF5MotionDataset
-----------------
Loads the rendered.hdf5 stack-cube demonstration dataset directly and provides
the same (s_t, s_{t+1}) transition sampling API that the AMP discriminator needs
— without converting to .npz.

Option B (default): AMP observation = eef_pos only (3-dim).
    eef_pos (3) — end-effector XYZ, already present in the RLinf states vector
    No env-side changes required.

Option A (upgrade): set amp_obs_terms=["joint_pos","joint_vel","eef_pos","eef_quat"]
    Richer 25-dim style signal, but requires adding amp_obs to _wrap_obs.

Only obs-level data is used; camera images and actions are ignored.

Dataset layout expected (mirrors rendered.hdf5):
    /data/demo_N/obs/eef_pos          (T, 3)   float32  ← used by default
    /data/demo_N/obs/joint_pos        (T, 9)   float32  ← Option A
    /data/demo_N/obs/joint_vel        (T, 9)   float32  ← Option A
    /data/demo_N/obs/eef_quat         (T, 4)   float32  ← Option A
"""

from __future__ import annotations

from typing import List, Tuple

import h5py
import numpy as np
import torch


# Supported AMP observation terms and their slice widths
_TERM_DIMS = {
    "joint_pos": 9,
    "joint_vel": 9,
    "eef_pos":   3,
    "eef_quat":  4,
}

# Option B default: eef_pos only (3-dim) — no env patch required.
# Upgrade to ["joint_pos", "joint_vel", "eef_pos", "eef_quat"] for Option A.
DEFAULT_AMP_OBS_TERMS: List[str] = ["eef_pos"]


class HDF5MotionDataset:
    """Demonstration dataset backed by a single HDF5 file.

    Provides the same interface as beyondAMP's ``MotionDataset`` so it can be
    dropped in wherever a motion dataset is expected:

        ``sample_batch(batch_size)``  → ``(t_idx, tp1_idx)``
        ``build_transition(t, tp1)``  → ``(obs_t, obs_tp1)``  [concatenated AMP obs]
        ``feed_forward_generator(n_batches, batch_size)``
        ``observation_dim``           → int (per-timestep AMP obs size)

    Parameters
    ----------
    hdf5_path : str
        Path to ``rendered.hdf5``.
    amp_obs_terms : list[str], optional
        Ordered list of field names to include in the AMP observation.
        Defaults to ``["joint_pos", "joint_vel", "eef_pos", "eef_quat"]``.
    device : str
        Torch device to place all tensors on.
    demo_limit : int or None
        If set, only load the first N demos (useful for debugging).
    """

    def __init__(
        self,
        hdf5_path: str,
        amp_obs_terms: List[str] = None,
        device: str = "cpu",
        demo_limit: int | None = None,
    ):
        self.hdf5_path = hdf5_path
        self.amp_obs_terms = amp_obs_terms or DEFAULT_AMP_OBS_TERMS
        self.device = device

        for term in self.amp_obs_terms:
            if term not in _TERM_DIMS:
                raise ValueError(
                    f"Unknown AMP obs term '{term}'. "
                    f"Supported: {list(_TERM_DIMS.keys())}"
                )

        self._load(demo_limit)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, demo_limit: int | None):
        """Read all demos from HDF5 and build contiguous tensors."""
        buffers: dict[str, list[torch.Tensor]] = {t: [] for t in self.amp_obs_terms}
        traj_lengths: list[int] = []

        with h5py.File(self.hdf5_path, "r") as f:
            demos = sorted(f["data"].keys())
            if demo_limit is not None:
                demos = demos[:demo_limit]

            for demo in demos:
                T = f[f"data/{demo}/obs/joint_pos"].shape[0]
                if T < 2:
                    # Need at least one transition
                    continue
                traj_lengths.append(T)
                for term in self.amp_obs_terms:
                    arr = f[f"data/{demo}/obs/{term}"][:]  # (T, dim)
                    buffers[term].append(torch.tensor(arr, dtype=torch.float32))

        # Concatenate into single large tensors  (total_T, dim)
        self._data: dict[str, torch.Tensor] = {}
        for term in self.amp_obs_terms:
            self._data[term] = torch.cat(buffers[term], dim=0).to(self.device)

        self.total_dataset_size = sum(traj_lengths)
        self._traj_lengths = traj_lengths

        # Per-term dims
        self._term_dims = {t: _TERM_DIMS[t] for t in self.amp_obs_terms}
        self.observation_dim = sum(self._term_dims[t] for t in self.amp_obs_terms)

        # Valid (t, t+1) transition indices — no crossing of episode boundaries
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths)

    def _build_transition_indices(
        self, traj_lengths: list[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_t, idx_tp1 = [], []
        offset = 0
        for L in traj_lengths:
            t = torch.arange(offset, offset + L - 1, dtype=torch.long)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += L
        return (
            torch.cat(idx_t).to(self.device),
            torch.cat(idx_tp1).to(self.device),
        )

    # ------------------------------------------------------------------
    # Term access (mirrors beyondAMP property names)
    # ------------------------------------------------------------------

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._data["joint_pos"]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._data["joint_vel"]

    @property
    def eef_pos(self) -> torch.Tensor:
        return self._data["eef_pos"]

    @property
    def eef_quat(self) -> torch.Tensor:
        return self._data["eef_quat"]

    # ------------------------------------------------------------------
    # Sampling API
    # ------------------------------------------------------------------

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random valid transition indices.

        Returns
        -------
        t, tp1 : LongTensors of shape (batch_size,)
        """
        idx = torch.randint(
            0, len(self.index_t), (batch_size,), device=self.device
        )
        return self.index_t[idx], self.index_tp1[idx]

    def build_transition(
        self, t: torch.Tensor, tp1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather and concatenate AMP observations at indices t and t+1.

        Returns
        -------
        obs_t, obs_tp1 : Tensors of shape (batch_size, observation_dim)
        """
        parts_t, parts_tp1 = [], []
        for term in self.amp_obs_terms:
            data = self._data[term]
            parts_t.append(data[t])
            parts_tp1.append(data[tp1])
        return torch.cat(parts_t, dim=-1), torch.cat(parts_tp1, dim=-1)

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ):
        """Yield ``(obs_t, obs_tp1)`` tuples for discriminator training.

        Mirrors beyondAMP's ``MotionDataset.feed_forward_generator``.
        """
        for _ in range(num_mini_batch):
            t, tp1 = self.sample_batch(mini_batch_size)
            yield self.build_transition(t, tp1)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HDF5MotionDataset("
            f"demos={len(self._traj_lengths)}, "
            f"total_steps={self.total_dataset_size}, "
            f"transitions={len(self.index_t)}, "
            f"obs_dim={self.observation_dim}, "
            f"terms={self.amp_obs_terms})"
        )

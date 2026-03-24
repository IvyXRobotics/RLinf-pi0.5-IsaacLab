"""
AMPActorMixin
-------------
A mixin / companion that adds AMP (Adversarial Motion Priors) discriminator
training to RLinf's ``EmbodiedFSDPActor``.

Usage
-----
Subclass EmbodiedFSDPActor and mix in AMPActorMixin:

    class AMPEmbodiedFSDPActor(AMPActorMixin, EmbodiedFSDPActor):
        pass

The mixin overrides two methods:

    compute_advantages_and_returns()
        1.  Extracts AMP obs from ``curr_obs["states"][..., 0:3]`` (eef_pos) and
            ``next_obs["states"][..., 0:3]`` — no env patch required (Option B).
        2.  Stores policy transitions in the AMP replay buffer.
        3.  Runs the discriminator to compute style rewards.
        4.  Mixes AMP reward into task reward:
                total_reward = (1 - alpha) * amp_reward + alpha * task_reward
        5.  Replaces ``rollout_batch["rewards"]`` with the mixed reward.
        6.  Calls ``super().compute_advantages_and_returns()`` so GAE/returns
            are computed on the mixed rewards.

    run_training()
        1.  Calls ``super().run_training()`` — full PPO update unchanged.
        2.  Then runs ``_train_discriminator()`` — discriminator update using
            the current replay buffer and the HDF5 motion dataset.

Configuration (omegaconf / yaml)
---------------------------------
Add the following section to the training yaml:

    amp:
      enabled: true
      hdf5_path: "/path/to/rendered.hdf5"
      obs_terms: [eef_pos]            # Option B: 3-dim, no env changes needed
      reward_coef: 0.5               # amp_reward_coef
      task_reward_lerp: 0.9          # alpha (0=pure AMP, 1=pure task)
      discr_hidden_dims: [256, 256]
      discr_lr: 1e-4
      discr_weight_decay_trunk: 1e-4
      discr_weight_decay_head: 1e-2
      grad_pen_lambda: 10.0
      replay_buffer_size: 100000
      mini_batch_size: 512           # per discriminator update
      num_discr_updates: 4           # discriminator updates per PPO step
      demo_limit: null               # null = load all demos
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.optim as optim

from rlinf_amp.amp_discriminator   import AMPDiscriminator
from rlinf_amp.amp_normalizer       import AMPNormalizer
from rlinf_amp.amp_replay_buffer    import AMPReplayBuffer
from rlinf_amp.hdf5_motion_dataset  import HDF5MotionDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


class AMPActorMixin:
    """Mixin that adds AMP discriminator training to EmbodiedFSDPActor.

    Must appear before EmbodiedFSDPActor in the MRO:

        class AMPEmbodiedFSDPActor(AMPActorMixin, EmbodiedFSDPActor):
            pass
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_amp(self) -> None:
        """Create AMP components from ``self.cfg.amp``.

        Call this at the end of ``init_worker()`` (after the base class
        finishes setting up the FSDP model & optimizer).
        """
        amp_cfg = self.cfg.amp
        device  = self.device  # from EmbodiedFSDPActor

        # --- Motion dataset (expert side) ----------------------------
        self.amp_motion_dataset = HDF5MotionDataset(
            hdf5_path    = amp_cfg.hdf5_path,
            amp_obs_terms= list(amp_cfg.obs_terms),
            device       = device,
            demo_limit   = amp_cfg.get("demo_limit", None),
        )
        obs_dim = self.amp_motion_dataset.observation_dim
        self.log_info(f"[AMP] Motion dataset: {self.amp_motion_dataset}")

        # --- Discriminator -------------------------------------------
        self.amp_discriminator = AMPDiscriminator(
            input_dim          = obs_dim,
            amp_reward_coef    = float(amp_cfg.reward_coef),
            hidden_layer_sizes = list(amp_cfg.discr_hidden_dims),
            device             = device,
            task_reward_lerp   = float(amp_cfg.task_reward_lerp),
        )

        # Separate optimizer for the discriminator (doesn't touch FSDP)
        self.amp_discr_optimizer = optim.Adam(
            [
                {"params": self.amp_discriminator.trunk.parameters(),
                 "weight_decay": float(amp_cfg.get("discr_weight_decay_trunk", 1e-4))},
                {"params": self.amp_discriminator.amp_linear.parameters(),
                 "weight_decay": float(amp_cfg.get("discr_weight_decay_head", 1e-2))},
            ],
            lr=float(amp_cfg.get("discr_lr", 1e-4)),
        )

        # --- Replay buffer (policy side) -----------------------------
        self.amp_replay_buffer = AMPReplayBuffer(
            obs_dim     = obs_dim,
            buffer_size = int(amp_cfg.get("replay_buffer_size", 100_000)),
            device      = device,
        )

        # --- Normalizer ----------------------------------------------
        self.amp_normalizer = AMPNormalizer(obs_dim)

        # --- Config shortcuts ----------------------------------------
        self._amp_obs_terms       = list(amp_cfg.obs_terms)
        self._amp_grad_pen_lambda = float(amp_cfg.get("grad_pen_lambda", 10.0))
        self._amp_mini_batch_size = int(amp_cfg.get("mini_batch_size", 512))
        self._amp_num_discr_upd   = int(amp_cfg.get("num_discr_updates", 4))

        # --- Reward saving -------------------------------------------
        self._amp_save_rewards  = bool(amp_cfg.get("save_rewards", False))
        self._amp_reward_dir    = amp_cfg.get("reward_save_path", None)
        self._amp_step_count    = 0
        if self._amp_save_rewards and self._amp_reward_dir:
            os.makedirs(self._amp_reward_dir, exist_ok=True)
            self.log_info(f"[AMP] Saving rewards to: {self._amp_reward_dir}")

        self.log_info(
            f"[AMP] Discriminator input_dim={obs_dim}×2, "
            f"hidden={list(amp_cfg.discr_hidden_dims)}, "
            f"replay_capacity={self.amp_replay_buffer.buffer_size}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_amp_obs(
        self, obs_dict: dict
    ) -> torch.Tensor | None:
        """Extract AMP obs (eef_pos) from ``states[..., 0:3]``.

        Option B: no separate amp_obs key needed.  eef_pos is the first 3
        elements of the states vector produced by IsaaclabStackCubeEnv._wrap_obs:
            states = [eef_pos(3), eef_axis_angle(3), gripper_pos(2)]

        Shape from batch: [T, B, state_dim] → returns (T*B, 3).
        Returns None if states is absent.
        """
        states = obs_dict.get("states", None)
        if states is None:
            return None
        # Flatten leading dims to (N, state_dim), then slice eef_pos
        flat = states.reshape(-1, states.shape[-1])
        return flat[:, :3]  # eef_pos

    # ------------------------------------------------------------------
    # Override: compute_advantages_and_returns
    # ------------------------------------------------------------------

    def compute_advantages_and_returns(self) -> dict:
        """Inject AMP reward into task reward, then run base GAE.

        The base method reads ``self.rollout_batch["rewards"]``.  We replace
        those rewards with the mixed reward before calling super().
        """
        # Guard: skip if AMP not initialised (e.g. during eval-only runs)
        if not hasattr(self, "amp_discriminator"):
            return super().compute_advantages_and_returns()

        curr_obs = self.rollout_batch.get("curr_obs", {})
        next_obs = self.rollout_batch.get("next_obs", {})

        amp_obs_t   = self._extract_amp_obs(curr_obs)
        amp_obs_tp1 = self._extract_amp_obs(next_obs)

        if amp_obs_t is None or amp_obs_tp1 is None:
            self.log_warning(
                "[AMP] 'states' key missing from rollout obs — skipping AMP reward."
            )
            return super().compute_advantages_and_returns()

        amp_obs_t   = amp_obs_t.to(self.device)
        amp_obs_tp1 = amp_obs_tp1.to(self.device)

        # --- Store in replay buffer -----------------------------------
        # Split into per-env transitions for the ring buffer
        self.amp_replay_buffer.insert(
            amp_obs_t.detach().cpu(),
            amp_obs_tp1.detach().cpu(),
        )
        # Update normaliser with current batch
        self.amp_normalizer.update(amp_obs_t.detach().cpu().numpy())

        # --- Compute AMP reward ---------------------------------------
        task_reward = self.rollout_batch["rewards"]  # [T, B, ...]
        flat_task   = task_reward.reshape(-1).to(self.device)

        mixed_reward, _, amp_reward_only = self.amp_discriminator.predict_amp_reward(
            state       = amp_obs_t,
            next_state  = amp_obs_tp1,
            task_reward = flat_task,
            normalizer  = self.amp_normalizer,
        )

        # Reshape back to original reward shape
        self.rollout_batch["rewards"] = mixed_reward.reshape(task_reward.shape).cpu()

        # Stash for metrics logging
        self._amp_last_amp_reward_mean  = amp_reward_only.mean().item()
        self._amp_last_task_reward_mean = flat_task.mean().item()

        # --- Save per-step reward arrays to disk ----------------------
        if self._amp_save_rewards and self._amp_reward_dir:
            np.savez_compressed(
                os.path.join(self._amp_reward_dir, f"step_{self._amp_step_count:06d}.npz"),
                amp_reward   = amp_reward_only.detach().cpu().numpy(),   # (N,)
                task_reward  = flat_task.detach().cpu().numpy(),          # (N,)
                mixed_reward = mixed_reward.detach().cpu().numpy(),       # (N,)
            )
        self._amp_step_count += 1

        # --- Normal GAE on mixed rewards ------------------------------
        rollout_metrics = super().compute_advantages_and_returns()

        # Append AMP-specific metrics
        rollout_metrics["amp_reward"] = self._amp_last_amp_reward_mean
        rollout_metrics["task_reward_raw"] = self._amp_last_task_reward_mean
        return rollout_metrics

    # ------------------------------------------------------------------
    # Override: run_training
    # ------------------------------------------------------------------

    def run_training(self) -> dict:
        """Run PPO training, then train the discriminator."""
        training_metrics = super().run_training()

        if not hasattr(self, "amp_discriminator"):
            return training_metrics

        # Only train discriminator once the replay buffer has enough data
        if self.amp_replay_buffer.num_samples < self._amp_mini_batch_size:
            return training_metrics

        discr_metrics = self._train_discriminator()
        training_metrics.update(discr_metrics)
        return training_metrics

    def _train_discriminator(self) -> dict:
        """Run ``num_discr_updates`` discriminator gradient steps.

        Returns a dict of scalar metrics for logging.
        """
        self.amp_discriminator.train()

        total_amp_loss    = 0.0
        total_grad_pen    = 0.0
        total_expert_pred = 0.0
        total_policy_pred = 0.0

        for _ in range(self._amp_num_discr_upd):
            # --- Sample policy transitions from replay buffer ----------
            policy_s, policy_sn = self.amp_replay_buffer.sample(
                self._amp_mini_batch_size
            )
            policy_s  = policy_s .to(self.device)
            policy_sn = policy_sn.to(self.device)

            # --- Sample expert transitions from HDF5 dataset -----------
            t, tp1 = self.amp_motion_dataset.sample_batch(self._amp_mini_batch_size)
            expert_s, expert_sn = self.amp_motion_dataset.build_transition(t, tp1)
            expert_s  = expert_s .to(self.device)
            expert_sn = expert_sn.to(self.device)

            # --- Normalise -------------------------------------------
            policy_s  = self.amp_normalizer.normalize(policy_s)
            policy_sn = self.amp_normalizer.normalize(policy_sn)
            expert_s  = self.amp_normalizer.normalize(expert_s)
            expert_sn = self.amp_normalizer.normalize(expert_sn)

            # --- Discriminator forward --------------------------------
            policy_d = self.amp_discriminator(
                torch.cat([policy_s, policy_sn], dim=-1)
            )
            expert_d = self.amp_discriminator(
                torch.cat([expert_s, expert_sn], dim=-1)
            )

            # AMP loss: expert → +1, policy → -1
            expert_loss = torch.nn.functional.mse_loss(
                expert_d, torch.ones_like(expert_d)
            )
            policy_loss = torch.nn.functional.mse_loss(
                policy_d, -torch.ones_like(policy_d)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # Gradient penalty on expert inputs
            grad_pen = self.amp_discriminator.compute_grad_pen(
                expert_s, expert_sn, lambda_=self._amp_grad_pen_lambda
            )

            loss = amp_loss + grad_pen

            # --- Backward -------------------------------------------
            self.amp_discr_optimizer.zero_grad()
            loss.backward()
            self.amp_discr_optimizer.step()

            # --- Normaliser update (CPU) ------------------------------
            self.amp_normalizer.update(
                policy_s.detach().cpu().numpy()
            )
            self.amp_normalizer.update(
                expert_s.detach().cpu().numpy()
            )

            total_amp_loss    += amp_loss.item()
            total_grad_pen    += grad_pen.item()
            total_expert_pred += expert_d.mean().item()
            total_policy_pred += policy_d.mean().item()

        n = self._amp_num_discr_upd
        return {
            "amp_discr_loss":    total_amp_loss    / n,
            "amp_grad_pen":      total_grad_pen    / n,
            "amp_expert_pred":   total_expert_pred / n,
            "amp_policy_pred":   total_policy_pred / n,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_amp_state(self, save_dir: str) -> None:
        """Save discriminator weights and normaliser state."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "discriminator": self.amp_discriminator.state_dict(),
                "discr_optimizer": self.amp_discr_optimizer.state_dict(),
                "normalizer": self.amp_normalizer.state_dict(),
            },
            os.path.join(save_dir, "amp_state.pt"),
        )

    def load_amp_state(self, save_dir: str) -> None:
        """Restore discriminator weights and normaliser state."""
        path = os.path.join(save_dir, "amp_state.pt")
        if not os.path.isfile(path):
            self.log_warning(f"[AMP] No checkpoint found at {path}, starting fresh.")
            return
        state = torch.load(path, map_location=self.device)
        self.amp_discriminator.load_state_dict(state["discriminator"])
        self.amp_discr_optimizer.load_state_dict(state["discr_optimizer"])
        self.amp_normalizer.load_state_dict(state["normalizer"])
        self.log_info(f"[AMP] Loaded checkpoint from {path}")

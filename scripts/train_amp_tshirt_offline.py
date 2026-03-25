"""
Offline AMP discriminator training + per-frame reward scoring for the tshirt task.

Usage
-----
python scripts/train_amp_tshirt_offline.py \
    --data_dir /home/ivy/Documents/Gatech/Research/Lidar/HLM/RL_VLA_Project/Dataset/pi05_gim_tshirt_rollout_data/pi0_gim_tshirt_dagger \
    --output_dir ./amp_tshirt_output \
    --num_epochs 200 \
    --mini_batch_size 256 \
    --num_discr_updates 4

After training, reward curves are saved to:
    <output_dir>/rewards/good/rollout_*.npz
    <output_dir>/rewards/failure/rollout_*.npz

Each .npz contains:
    amp_reward   (T-1,)   per-transition AMP reward (0 to reward_coef)
"""

import argparse
import glob
import os
import pickle

import numpy as np
import torch
import torch.optim as optim

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlinf_amp.amp_discriminator import AMPDiscriminator
from rlinf_amp.amp_normalizer    import AMPNormalizer
from rlinf_amp.pkl_motion_dataset import PKLMotionDataset, OBS_DIM, _extract_obs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",         default="/home/ivy/Documents/Gatech/Research/Lidar/HLM/RL_VLA_Project/Dataset/pi05_gim_tshirt_rollout_data/pi0_gim_tshirt_dagger")
    p.add_argument("--output_dir",       default="./amp_tshirt_output")
    p.add_argument("--num_epochs",       type=int,   default=200)
    p.add_argument("--mini_batch_size",  type=int,   default=256)
    p.add_argument("--num_discr_updates",type=int,   default=4)
    p.add_argument("--hidden_dims",      nargs="+",  type=int, default=[256, 256])
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--grad_pen_lambda",  type=float, default=10.0)
    p.add_argument("--reward_coef",      type=float, default=0.5)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rollout_limit",    type=int,   default=None)
    return p.parse_args()


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")

    # --- Dataset ---
    dataset = PKLMotionDataset(
        root_dir      = args.data_dir,
        device        = args.device,
        rollout_limit = args.rollout_limit,
    )

    # --- Discriminator ---
    discriminator = AMPDiscriminator(
        input_dim          = OBS_DIM,
        amp_reward_coef    = args.reward_coef,
        hidden_layer_sizes = args.hidden_dims,
        device             = args.device,
    )

    optimizer = optim.Adam(
        [
            {"params": discriminator.trunk.parameters(),      "weight_decay": 1e-4},
            {"params": discriminator.amp_linear.parameters(), "weight_decay": 1e-2},
        ],
        lr=args.lr,
    )

    normalizer = AMPNormalizer(OBS_DIM)

    # Warm up normalizer with all expert obs
    expert_np = dataset._expert_obs.cpu().numpy()
    normalizer.update(expert_np)

    # --- Training loop ---
    print(f"\nTraining discriminator for {args.num_epochs} epochs...")
    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0.0
        epoch_expert_pred = 0.0
        epoch_policy_pred = 0.0

        for _ in range(args.num_discr_updates):
            # Sample expert and policy transitions
            expert_s, expert_sn = dataset.sample_expert(args.mini_batch_size)
            policy_s, policy_sn = dataset.sample_policy(args.mini_batch_size)

            # Normalise
            expert_s  = normalizer.normalize(expert_s)
            expert_sn = normalizer.normalize(expert_sn)
            policy_s  = normalizer.normalize(policy_s)
            policy_sn = normalizer.normalize(policy_sn)

            # Forward
            expert_d = discriminator(torch.cat([expert_s, expert_sn], dim=-1))
            policy_d = discriminator(torch.cat([policy_s, policy_sn], dim=-1))

            # MSE loss: expert -> +1, policy -> -1
            expert_loss = torch.nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss    = 0.5 * (expert_loss + policy_loss)

            grad_pen = discriminator.compute_grad_pen(
                expert_s, expert_sn, lambda_=args.grad_pen_lambda
            )

            loss = amp_loss + grad_pen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            normalizer.update(policy_s.detach().cpu().numpy())

            epoch_loss        += amp_loss.item()
            epoch_expert_pred += expert_d.mean().item()
            epoch_policy_pred += policy_d.mean().item()

        n = args.num_discr_updates
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.num_epochs} | "
                f"loss={epoch_loss/n:.4f} | "
                f"expert_pred={epoch_expert_pred/n:.3f} | "
                f"policy_pred={epoch_policy_pred/n:.3f}"
            )

    # --- Save checkpoint ---
    ckpt_path = os.path.join(args.output_dir, "amp_state_tshirt.pt")
    torch.save(
        {
            "discriminator": discriminator.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "normalizer":    normalizer.state_dict(),
            "obs_dim":       OBS_DIM,
            "hidden_dims":   args.hidden_dims,
            "reward_coef":   args.reward_coef,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved to: {ckpt_path}")

    # --- Score all rollouts ---
    score_all(args, discriminator, normalizer)


def score_all(args, discriminator, normalizer):
    """Run discriminator on every frame of every rollout and save reward curves."""
    discriminator.eval()

    for split in ["good", "failure"]:
        folder   = os.path.join(args.data_dir, split)
        out_dir  = os.path.join(args.output_dir, "rewards", split)
        os.makedirs(out_dir, exist_ok=True)

        paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        print(f"\nScoring {len(paths)} rollouts from {split}/...")

        for path in paths:
            with open(path, "rb") as f:
                steps = pickle.load(f)
            if len(steps) < 2:
                continue

            # Build obs array (T, 16)
            obs = np.stack([_extract_obs(s) for s in steps], axis=0)
            obs_t   = torch.tensor(obs[:-1], dtype=torch.float32).to(args.device)
            obs_tp1 = torch.tensor(obs[1:],  dtype=torch.float32).to(args.device)

            obs_t   = normalizer.normalize(obs_t)
            obs_tp1 = normalizer.normalize(obs_tp1)

            with torch.no_grad():
                d = discriminator(torch.cat([obs_t, obs_tp1], dim=-1))  # (T-1, 1)
                # AMP reward formula: reward_coef * clamp(1 - 0.25*(d-1)^2, min=0)
                amp_reward = discriminator.amp_reward_coef * torch.clamp(
                    1.0 - 0.25 * (d - 1.0) ** 2, min=0.0
                )
                amp_reward = amp_reward.squeeze(-1).cpu().numpy()  # (T-1,)

            stem    = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{stem}.npz")
            np.savez_compressed(out_path, amp_reward=amp_reward)

        print(f"  Saved reward curves to {out_dir}/")

    print("\nDone. Load a reward curve with:")
    print("  data = np.load('<output_dir>/rewards/good/rollout_*.npz')")
    print("  reward_curve = data['amp_reward']   # shape (T-1,)")


if __name__ == "__main__":
    args = parse_args()
    train(args)

"""
Visualize AMP reward curves for good vs failure rollouts.

Usage
-----
python scripts/visualize_amp_rewards.py --output_dir ./amp_tshirt_output
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./amp_tshirt_output")
    p.add_argument("--n_samples",  type=int, default=10, help="rollouts to overlay per plot")
    return p.parse_args()


def load_rewards(folder, n=None):
    paths = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if n:
        paths = paths[:n]
    return [np.load(p)["amp_reward"] for p in paths], [os.path.basename(p) for p in paths]


def main():
    args   = parse_args()
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    good_dir    = os.path.join(args.output_dir, "rewards", "good")
    failure_dir = os.path.join(args.output_dir, "rewards", "failure")

    good_rewards,    good_names    = load_rewards(good_dir)
    failure_rewards, failure_names = load_rewards(failure_dir)

    # ── 1. Overlay: individual reward curves (good vs failure) ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for r in good_rewards[:args.n_samples]:
        t = np.linspace(0, 1, len(r))
        axes[0].plot(t, r, alpha=0.4, linewidth=0.8, color="steelblue")
    axes[0].set_title(f"Good rollouts (n={len(good_rewards)})")
    axes[0].set_xlabel("Normalized timestep")
    axes[0].set_ylabel("AMP reward")

    for r in failure_rewards[:args.n_samples]:
        t = np.linspace(0, 1, len(r))
        axes[1].plot(t, r, alpha=0.4, linewidth=0.8, color="tomato")
    axes[1].set_title(f"Failure rollouts (n={len(failure_rewards)})")
    axes[1].set_xlabel("Normalized timestep")

    fig.suptitle("AMP reward curves — individual rollouts")
    plt.tight_layout()
    out = os.path.join(vis_dir, "reward_curves_overlay.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    # ── 2. Mean ± std band (good vs failure) ───────────────────────────────
    def interp_to_fixed(rewards, n_points=500):
        out = []
        for r in rewards:
            t_orig  = np.linspace(0, 1, len(r))
            t_fixed = np.linspace(0, 1, n_points)
            out.append(np.interp(t_fixed, t_orig, r))
        return np.array(out)

    N = 500
    good_mat    = interp_to_fixed(good_rewards,    N)
    failure_mat = interp_to_fixed(failure_rewards, N)
    t = np.linspace(0, 1, N)

    fig, ax = plt.subplots(figsize=(10, 4))
    for mat, color, label in [
        (good_mat,    "steelblue", "Good"),
        (failure_mat, "tomato",    "Failure"),
    ]:
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ax.plot(t, mean, color=color, linewidth=2, label=label)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Normalized timestep")
    ax.set_ylabel("AMP reward")
    ax.set_title("Mean ± std AMP reward — good vs failure")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(vis_dir, "reward_mean_std.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    # ── 3. Histogram: reward distribution ──────────────────────────────────
    good_all    = np.concatenate(good_rewards)
    failure_all = np.concatenate(failure_rewards)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(good_all,    bins=80, alpha=0.6, color="steelblue", label="Good",    density=True)
    ax.hist(failure_all, bins=80, alpha=0.6, color="tomato",    label="Failure", density=True)
    ax.set_xlabel("AMP reward")
    ax.set_ylabel("Density")
    ax.set_title("AMP reward distribution — good vs failure")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(vis_dir, "reward_histogram.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    # ── 4. Heatmap: reward over time for all rollouts ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, title, cmap in [
        (axes[0], good_mat,    "Good rollouts",    "Blues"),
        (axes[1], failure_mat, "Failure rollouts", "Reds"),
    ]:
        im = ax.imshow(
            mat, aspect="auto", cmap=cmap,
            extent=[0, 1, 0, mat.shape[0]],
            origin="lower", vmin=0, vmax=0.5,
        )
        ax.set_xlabel("Normalized timestep")
        ax.set_ylabel("Rollout index")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="AMP reward")

    fig.suptitle("AMP reward heatmap (each row = one rollout)")
    plt.tight_layout()
    out = os.path.join(vis_dir, "reward_heatmap.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    # ── Summary stats ───────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Good    — mean={good_all.mean():.4f}  std={good_all.std():.4f}  "
          f"max={good_all.max():.4f}  min={good_all.min():.4f}")
    print(f"  Failure — mean={failure_all.mean():.4f}  std={failure_all.std():.4f}  "
          f"max={failure_all.max():.4f}  min={failure_all.min():.4f}")
    print(f"\nAll plots saved to: {vis_dir}/")


if __name__ == "__main__":
    main()

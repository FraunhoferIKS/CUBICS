"""Run the CUBICS canonical simulation (Rain × Wind × Time scenario).

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

try:
    from cubics import default_config, load_config, make_default_scenario, run_simulation
    from cubics.assurance.simulation import SimulationResult
except ImportError:
    sys.exit(
        "cubics package not found. Run: pip install -e . from the repo root."
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ── Plotting ──────────────────────────────────────────────────────────────────

MILESTONE_COLORS = ["#cccccc", "#66c2a5", "#fc8d62", "#8da0cb"]


def plot_situational_progression(
    result: SimulationResult,
    failure_rates: list[float],
    output_path: Path,
) -> None:
    """Save a 2×4 grid of Beta PDF evolution per situation (Figure 1)."""
    x = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for j, lbl in enumerate(result.situations):
        ax = axes[j]
        for m_idx, m in enumerate(result.milestones):
            r = result.history_R[lbl][m_idx]
            s = result.history_S[lbl][m_idx]
            y = beta_dist.pdf(x, 1 + r, 1 + s)
            ax.plot(
                x, y,
                color=MILESTONE_COLORS[m_idx],
                label=f"Iter {m}" if j == 0 else "",
                linewidth=2,
            )
            if m == result.milestones[-1]:
                ax.fill_between(x, 0, y, color=MILESTONE_COLORS[m_idx], alpha=0.3)

        ax.axvline(1.0 - failure_rates[j], color="red", linestyle="--", alpha=0.6)
        parts = lbl.split("_._")
        title = (
            f"Rain: {parts[0]}\nWind: {parts[1]}\nToD: {parts[2]}"
            if len(parts) == 3 else lbl
        )
        ax.set_title(f"$s_{j}$ | {title}", fontsize=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 35)

    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def plot_marginal_vs_agnostic(
    result: SimulationResult,
    situation_probs: list[float],
    failure_rates: list[float],
    output_path: Path,
) -> None:
    """Save the two-panel SL marginal vs agnostic plot (Figure 2)."""
    x = np.linspace(0, 1, 300)
    total_gt_success = sum(p * (1 - f) for p, f in zip(situation_probs, failure_rates))
    num_runs = result.milestones[-1]

    fig, (ax_m, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    for m_idx, m in enumerate(result.milestones):
        op = result.marginal_history[m_idx]
        u_safe = max(op.u, 1e-6)
        alpha_m = 1 + (2.0 * op.b[0]) / u_safe
        beta_m = 1 + (2.0 * op.b[1]) / u_safe
        y_m = beta_dist.pdf(x, alpha_m, beta_m)
        ax_m.plot(x, y_m, color=MILESTONE_COLORS[m_idx], linewidth=3, label=f"Iter {m}")
        if m == num_runs:
            ax_m.fill_between(x, 0, y_m, color=MILESTONE_COLORS[m_idx], alpha=0.3)

    ax_m.set_title("Subjective Logic Marginal Guarantee (Deduction)", fontweight="bold")
    ax_m.set_facecolor("#f4faff")
    ax_m.axvline(total_gt_success, color="red", linestyle="-", linewidth=2, label="System GT")

    for m_idx, m in enumerate(result.milestones):
        R_global = sum(result.history_R[lbl][m_idx] for lbl in result.situations)
        S_global = sum(result.history_S[lbl][m_idx] for lbl in result.situations)
        y_a = beta_dist.pdf(x, 1 + R_global, 1 + S_global)
        ax_a.plot(x, y_a, color=MILESTONE_COLORS[m_idx], linewidth=3)
        if m == num_runs:
            ax_a.fill_between(x, 0, y_a, color=MILESTONE_COLORS[m_idx], alpha=0.3)

    ax_a.set_title("Agnostic Guarantee (Pooled Evidence)", fontweight="bold")
    ax_a.set_xlabel("Probability of Success ($p$)", fontsize=15)
    ax_a.set_facecolor("#fff4f4")
    ax_a.axvline(total_gt_success, color="red", linestyle="-", linewidth=2)

    for ax in [ax_m, ax_a]:
        ax.set_xlim(0, 1)
        ax.set_ylabel(r"Density $f(p| \alpha, \beta)$")

    fig.suptitle("Overall System Reliability", y=1.02, fontsize=15)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    p.add_argument("--num-runs", type=int, default=None, help="Number of simulation iterations.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory for output figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else default_config()
    if args.num_runs is not None:
        cfg.simulation_num_runs = args.num_runs
    if args.seed is not None:
        cfg.simulation_seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    context, conditional, probs, fail_rates = make_default_scenario()

    result = run_simulation(
        context,
        conditional,
        probs,
        fail_rates,
        num_runs=cfg.simulation_num_runs,
        seed=cfg.simulation_seed,
        prior_weight=cfg.sl_prior_weight,
        context_prior_weight=cfg.sl_context_prior_weight,
    )

    plot_situational_progression(
        result, fail_rates, cfg.output_dir / "situational_progression.png"
    )
    plot_marginal_vs_agnostic(
        result, probs, fail_rates, cfg.output_dir / "marginal_vs_agnostic.png"
    )

    print(f"\nOutputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()

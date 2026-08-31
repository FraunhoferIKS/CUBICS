"""RQ2: Marginal vs Agnostic Guarantee Analysis.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import beta as beta_dist

from cubics.sl import BinOp, MultiOp

try:
    from cubics import default_config, load_config
    from cubics.data import load_metrics_csv
except ImportError:
    sys.exit("cubics package not found. Run: pip install -e . from the repo root.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Classes included in --paper-results Table 2 multi-class comparison.
# "bike" matches the class label used in BDD100K / test_metrics_complete.csv.
PAPER_CLASSES = ["person", "bike", "car", "truck"]


# ── Local SL plotting helpers ─────────────────────────────────────────────────

def _op_to_beta(op: BinOp):
    """Convert a BinOp opinion to a scipy Beta distribution."""
    u_safe = max(op.u, 1e-9)
    alpha = ((2.0 * op.b) / u_safe) + (2.0 * op.a)
    beta_val = ((2.0 * op.d) / u_safe) + (2.0 * (1.0 - op.a))
    return beta_dist(alpha, beta_val)


def _plot_opinion(
    op: BinOp,
    ax,
    label: str,
    color: str,
    linestyle: str = "-",
    uncertainty_bars: bool = False,
) -> None:
    """Plot the Beta PDF of a BinOp opinion on *ax*."""
    x = np.linspace(0, 1, 1000)
    rv = _op_to_beta(op)
    y = rv.pdf(x)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2, label=label)
    ax.fill_between(x, 0, y, color=color, alpha=0.1)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_scenario_data(
    csv_path: Path,
    target_class: str = "person",
) -> tuple[list[tuple[str, int, int]], tuple[str, int, int]]:
    """Load per-scenario TP/FN from the evaluation CSV.

    Args:
        csv_path: Path to ``test_metrics_complete.csv``.
        target_class: YOLO class name to filter on.

    Returns:
        A tuple of ``(scenarios, baseline)`` where each scenario is
        ``(label, TP, FN)`` and *baseline* is the overall test baseline row.

    Raises:
        AssertionError: If no situational scenarios or no baseline are found.
    """
    df = load_metrics_csv(csv_path)
    df_class = df[df["Class"] == target_class]

    baseline = None
    scenarios = []

    for _, row in df_class.iterrows():
        scenario = row["Scenario"]
        tp = int(row["True_Positives"])
        fn = int(row["False_Negatives"])

        if "Overall" in scenario or "Baseline" in scenario:
            baseline = (scenario, tp, fn)
        elif "+" in scenario:
            scenarios.append((scenario, tp, fn))

    assert len(scenarios) > 0, f"No situational scenarios found for class '{target_class}'"
    assert baseline is not None, f"No overall baseline found for class '{target_class}'"

    return scenarios, baseline


# ── SL Construction ───────────────────────────────────────────────────────────

def build_marginal_vs_agnostic(
    scenarios: list[tuple[str, int, int]],
    baseline: tuple[str, int, int],
    prior_weight: float = 2.0,
    verbose: bool = True,
) -> tuple[MultiOp, BinOp, list[str], npt.NDArray, npt.NDArray, MultiOp, float]:
    """Build SL opinions and compute marginal and agnostic guarantees.

    Args:
        scenarios: List of ``(label, TP, FN)`` for each situational scenario.
        baseline: ``(label, TP, FN)`` for the overall test baseline row.
        prior_weight: SL non-informative prior weight W (default 2.0).
        verbose: Print per-scenario table and intermediate opinions (default True).

    Returns:
        Tuple of ``(omega_marginal, omega_agnostic, labels, TPs, FNs, omega_S,
        gt_success)``.
    """
    W = prior_weight
    n = len(scenarios)
    labels = [s[0] for s in scenarios]
    TPs = np.array([s[1] for s in scenarios], dtype=float)
    FNs = np.array([s[2] for s in scenarios], dtype=float)
    totals = TPs + FNs

    if verbose:
        print(f"\n{'Scenario':<25} | {'TP':>6} | {'FN':>6} | {'Total':>7} | {'Success Rate':>12}")
        print("-" * 68)
        for lbl, tp, fn in scenarios:
            print(f"{lbl:<25} | {tp:>6.0f} | {fn:>6.0f} | {tp+fn:>7.0f} | {tp/(tp+fn):>12.4f}")

    # Context opinion ω_S from total evidence per scenario
    total_ctx = totals.sum()
    b_ctx = totals / (W + total_ctx)
    u_ctx = W / (W + total_ctx)
    a_ctx = np.ones(n) / n
    omega_S = MultiOp(b=b_ctx, u=u_ctx, a=a_ctx, domain="Situation", states=labels)
    situation_freqs = totals / total_ctx

    if verbose:
        print(f"\nContext opinion ω_S: u={omega_S.u:.6f}")
        for lbl, p in zip(labels, omega_S.prob()):
            print(f"  P({lbl}) = {p:.4f}")

    # Conditional opinions P(Guarantee | s_i)
    b_Z_given_S = np.zeros((n, 2))
    u_Z_given_S = np.zeros(n)
    for i in range(n):
        r, s = TPs[i], FNs[i]
        b_Z_given_S[i, 0] = r / (W + r + s)
        b_Z_given_S[i, 1] = s / (W + r + s)
        u_Z_given_S[i] = W / (W + r + s)

    # Marginal guarantee via SL multinomial deduction
    omega_marginal = omega_S.deduce(
        b_Y_given_x=b_Z_given_S,
        u_Y_given_x=u_Z_given_S,
        states_Y=["Guarantee", "No Guarantee"],
        name_Y="Guarantee",
    )

    # Agnostic guarantee from the pre-computed overall baseline row
    R_total, S_total = float(baseline[1]), float(baseline[2])
    omega_agnostic = BinOp.from_rsa(R_total, S_total)

    # Ground-truth weighted success rate
    success_rates = TPs / totals
    gt_success = float(np.sum(situation_freqs * success_rates))

    if verbose:
        print(f"\n> SL Marginal Guarantee: {omega_marginal}")
        print(f"> Agnostic Guarantee:   {omega_agnostic}")
        print(f"> Weighted GT success:  {gt_success:.4f}")

    return omega_marginal, omega_agnostic, labels, TPs, FNs, omega_S, gt_success


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_marginal_vs_agnostic(
    omega_marginal: MultiOp,
    omega_agnostic: BinOp,
    gt_success: float,
    output_path: Path,
) -> None:
    """Save a two-panel plot: SL marginal (top) vs agnostic (bottom)."""
    x = np.linspace(0, 1, 1000)
    fig, (ax_m, ax_a) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    c_m, c_a = "#2166ac", "#b2182b"

    u_safe = max(omega_marginal.u, 1e-9)
    alpha_m = 1 + (2.0 * omega_marginal.b[0]) / u_safe
    beta_m = 1 + (2.0 * omega_marginal.b[1]) / u_safe
    y_m = beta_dist.pdf(x, alpha_m, beta_m)
    ax_m.plot(x, y_m, color=c_m, linewidth=2.5, label="SL Marginal")
    ax_m.fill_between(x, 0, y_m, color=c_m, alpha=0.15)
    E_m = omega_marginal.prob()[0]
    ax_m.axvline(E_m, color=c_m, linestyle=":", linewidth=1.5, alpha=0.7, label=f"E[marginal] = {E_m:.4f}")
    ax_m.axvline(gt_success, color="red", linewidth=2, alpha=0.8, label=f"Weighted GT = {gt_success:.4f}")
    ax_m.set_facecolor("#f4faff")
    ax_m.legend(loc="upper left", fontsize=15)

    rv_a = _op_to_beta(omega_agnostic)
    y_a = rv_a.pdf(x)
    ax_a.plot(x, y_a, color=c_a, linewidth=2.5, label="Agnostic")
    ax_a.fill_between(x, 0, y_a, color=c_a, alpha=0.15)
    E_a = omega_agnostic.prob()
    ax_a.axvline(E_a, color=c_a, linestyle=":", linewidth=1.5, alpha=0.7, label=f"E[agnostic] = {E_a:.4f}")
    ax_a.axvline(gt_success, color="red", linewidth=2, alpha=0.8, label=f"Weighted GT = {gt_success:.4f}")
    ax_a.set_facecolor("#fff4f4")
    ax_a.set_xlabel("Probability of Success ($p$)", fontsize=15)
    ax_a.legend(loc="upper left", fontsize=15)

    for ax in [ax_m, ax_a]:
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density", fontsize=15)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def plot_situational_guarantees(
    labels: list[str],
    TPs: npt.NDArray,
    FNs: npt.NDArray,
    output_path: Path,
) -> None:
    """Save a grid of per-scenario Beta PDFs."""
    n = len(labels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()
    x = np.linspace(0, 1, 500)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for j in range(n):
        ax = axes[j]
        r, s = TPs[j], FNs[j]
        y = beta_dist.pdf(x, 1 + r, 1 + s)
        ax.plot(x, y, color=colors[j], linewidth=2)
        ax.fill_between(x, 0, y, color=colors[j], alpha=0.2)
        gt = r / (r + s)
        ax.axvline(gt, color="red", linestyle="--", alpha=0.6, label=f"GT={gt:.3f}")
        ax.set_title(f"$s_{{{j}}}$ | {labels[j]}", fontsize=12)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=12)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Situational Guarantee Distributions (Beta PDFs)", y=1.02, fontsize=15)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def plot_scenario_overlays(
    scenarios: list[tuple[str, int, int]],
    baseline: tuple[str, int, int],
    output_path: Path,
) -> None:
    """Save best/mid/worst scenario overlays against the global baseline."""
    _, bl_tp, bl_fn = baseline
    op_baseline = BinOp.from_rsa(bl_tp, bl_fn)

    scenario_ops = {lbl: BinOp.from_rsa(tp, fn) for lbl, tp, fn in scenarios}
    success_rates = {lbl: tp / (tp + fn) for lbl, tp, fn in scenarios}
    sorted_s = sorted(success_rates, key=success_rates.get, reverse=True)
    best, worst, mid = sorted_s[0], sorted_s[-1], sorted_s[len(sorted_s) // 2]

    overlay = [(best, "green", "--"), (mid, "orange", "-."), (worst, "red", ":")]
    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_opinion(op_baseline, ax=ax, label="Global Baseline", color="black", linestyle="-")
    for lbl, color, ls in overlay:
        _plot_opinion(scenario_ops[lbl], ax=ax, label=lbl, color=color, linestyle=ls)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Probability of Success ($p$)", fontsize=15)
    ax.set_ylabel(r"Density $f(p|\alpha,\beta)$", fontsize=15)
    ax.set_title("RQ2: Global Baseline vs Selected Scenarios", fontweight="bold", fontsize=15)
    ax.legend(fontsize=15)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def analyze_rq2_benefit(
    csv_path: Path,
    target_class: str = "person",
    output_path: Path | None = None,
) -> None:
    """Print per-scenario SL opinions (Table 1) and save a combined analysis figure.

    Only the person class is used for Table 1 in --paper-results mode; call this
    function directly with a different *target_class* for exploratory use.
    """
    df = load_metrics_csv(csv_path)
    df_class = df[df["Class"] == target_class]

    highlight = [
        "Overall Test Baseline",
        "Clear + Daytime",
        "Rainy + Night",
        "Foggy + Dawn/dusk",
    ]

    opinions: dict[str, BinOp] = {}
    print(f"\n--- RQ2 Analysis for Class: {target_class.upper()} ---")
    print(
        f"{'Scenario':<25} | {'TP':>6} | {'FN':>6} | {'Belief':>7} | "
        f"{'Disbelief':>9} | {'Uncertainty':>11} | {'E[x]':>6}"
    )
    print("-" * 88)

    for scenario in highlight:
        row = df_class[df_class["Scenario"] == scenario]
        if row.empty:
            continue
        r = int(row["True_Positives"].values[0])
        s = int(row["False_Negatives"].values[0])
        op = BinOp.from_rsa(r, s)
        opinions[scenario] = op
        print(
            f"{scenario:<25} | {r:>6} | {s:>6} | {op.b:>7.4f} | "
            f"{op.d:>9.4f} | {op.u:>11.4f} | {op.prob():>6.4f}"
        )

    if "Overall Test Baseline" not in opinions or "Clear + Daytime" not in opinions:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_opinion(opinions["Overall Test Baseline"], ax=ax, label="Global Baseline",
                  color="black", linestyle="-")
    if "Clear + Daytime" in opinions:
        _plot_opinion(opinions["Clear + Daytime"], ax=ax, label="Scenario: Clear Day",
                      color="green", linestyle="--")
    if "Foggy + Dawn/dusk" in opinions:
        _plot_opinion(opinions["Foggy + Dawn/dusk"], ax=ax, label="Scenario: Foggy Dawn",
                      color="red", linestyle=":")

    ax.set_ylim(0, 120)
    ax.set_xlim(0, 1)
    ax.legend()

    out = output_path or Path("outputs") / "rq2_combined_analysis.svg"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


def print_summary(omega_marginal: MultiOp, omega_agnostic: BinOp, gt_success: float) -> None:
    E_m = omega_marginal.prob()[0]
    E_a = omega_agnostic.prob()
    u_m = omega_marginal.u
    u_a = omega_agnostic.u

    print("\n" + "=" * 60)
    print("SUMMARY: Marginal vs Agnostic")
    print("=" * 60)
    print(f"  SL Marginal E[Guarantee]:    {E_m:.4f}")
    print(f"  Agnostic E[Guarantee]:       {E_a:.4f}")
    print(f"  Weighted GT Success:         {gt_success:.4f}")
    print(f"  Difference (Marg - Agn):     {E_m - E_a:+.4f}")
    print(f"\n  SL Marginal Uncertainty:     {u_m:.6f}")
    print(f"  Agnostic Uncertainty:        {u_a:.6f}")
    print(f"  Uncertainty ratio (M/A):     {u_m / u_a:.1f}x")
    print(f"  → Marginal has {'MORE' if u_m > u_a else 'LESS'} uncertainty than Agnostic")
    print("=" * 60)


# ── Paper-results mode ────────────────────────────────────────────────────────

def run_paper_results(cfg) -> None:
    """Reproduce both paper tables in a single deterministic pass.

    Table 1 — per-situation SL opinions: person class only.
    Table 2 — marginal vs global pooled guarantee: one row per class in
    PAPER_CLASSES, showing E[Safe], u, and the uncertainty ratio.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1 + all figures: person class only ---
    print("\n" + "=" * 70)
    print("TABLE 1 — Per-situation SL opinions (person class)")
    print("=" * 70)

    scenarios_p, baseline_p = load_scenario_data(cfg.metrics_csv, "person")
    omega_m_p, omega_a_p, labels_p, TPs_p, FNs_p, _, gt_p = build_marginal_vs_agnostic(
        scenarios_p, baseline_p, prior_weight=cfg.sl_prior_weight
    )

    plot_marginal_vs_agnostic(
        omega_m_p, omega_a_p, gt_p,
        cfg.output_dir / "rq2_marginal_vs_agnostic.svg",
    )
    plot_situational_guarantees(
        labels_p, TPs_p, FNs_p,
        cfg.output_dir / "rq2_situational_guarantees.png",
    )
    plot_scenario_overlays(
        scenarios_p, baseline_p,
        cfg.output_dir / "rq2_scenario_overlays.png",
    )
    analyze_rq2_benefit(
        cfg.metrics_csv,
        target_class="person",
        output_path=cfg.output_dir / "rq2_combined_analysis.svg",
    )

    # --- Table 2: marginal vs pooled, multiple classes ---
    print("\n" + "=" * 70)
    print("TABLE 2 — Marginal vs Global Pooled Guarantee (multiple classes)")
    print("=" * 70)
    header = (
        f"{'Class':<12} | {'E[Marginal]':>11} | {'u[Marginal]':>11} | "
        f"{'E[Pooled]':>9} | {'u[Pooled]':>9} | {'u Ratio':>8}"
    )
    print(header)
    print("-" * len(header))

    for cls in PAPER_CLASSES:
        try:
            scenarios, baseline = load_scenario_data(cfg.metrics_csv, cls)
        except AssertionError as exc:
            print(f"{cls:<12} | {'(insufficient data)':>60}")
            continue
        omega_m, omega_a, _, _, _, _, _ = build_marginal_vs_agnostic(
            scenarios, baseline, prior_weight=cfg.sl_prior_weight, verbose=False
        )
        E_m = omega_m.prob()[0]
        u_m = omega_m.u
        E_a = omega_a.prob()
        u_a = omega_a.u
        ratio = u_m / u_a if u_a > 0 else float("inf")
        print(
            f"{cls:<12} | {E_m:>11.4f} | {u_m:>11.6f} | "
            f"{E_a:>9.4f} | {u_a:>9.6f} | {ratio:>7.1f}x"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    p.add_argument("--csv", type=Path, default=None, help="Override metrics CSV path.")
    p.add_argument("--class-name", type=str, default=None, dest="target_class",
                   help="Object class to analyse (default: person).")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory for output figures.")
    p.add_argument(
        "--paper-results",
        action="store_true",
        help=(
            "Reproduce paper tables exactly: Table 1 for the person class, "
            "Table 2 for all PAPER_CLASSES. Ignores --class-name."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else default_config()
    if args.csv is not None:
        cfg.metrics_csv = args.csv
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    if args.paper_results:
        run_paper_results(cfg)
        return

    # Single-class exploratory path (unchanged behaviour)
    if args.target_class is not None:
        cfg.target_class = args.target_class

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios, baseline = load_scenario_data(cfg.metrics_csv, cfg.target_class)
    print(f"Loaded {len(scenarios)} scenarios + baseline for class '{cfg.target_class}'")

    omega_marginal, omega_agnostic, labels, TPs, FNs, omega_S, gt_success = \
        build_marginal_vs_agnostic(scenarios, baseline, prior_weight=cfg.sl_prior_weight)

    plot_marginal_vs_agnostic(
        omega_marginal, omega_agnostic, gt_success,
        cfg.output_dir / "rq2_marginal_vs_agnostic.png",
    )
    plot_situational_guarantees(
        labels, TPs, FNs,
        cfg.output_dir / "rq2_situational_guarantees.png",
    )
    plot_scenario_overlays(
        scenarios, baseline,
        cfg.output_dir / "rq2_scenario_overlays.png",
    )
    analyze_rq2_benefit(
        cfg.metrics_csv, cfg.target_class,
        output_path=cfg.output_dir / "rq2_combined_analysis.svg",
    )

    print_summary(omega_marginal, omega_agnostic, gt_success)


if __name__ == "__main__":
    main()

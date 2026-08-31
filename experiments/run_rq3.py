"""RQ3: Robustness and Sensitivity Analysis.

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

from subjective_logic.core import BinOp

try:
    from cubics import default_config, load_config
    from cubics.data import load_metrics_csv
except ImportError:
    sys.exit("cubics package not found. Run: pip install -e . from the repo root.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ── Experiment A ──────────────────────────────────────────────────────────────

def experiment_a_data_scarcity(df, prior_weight: float = 2.0) -> None:
    """A: Show graceful degradation with scarce evidence (truck at foggy night)."""
    print("\n--- A. Data Scarcity (Graceful Degradation) ---")
    row = df[(df["Scenario"] == "Foggy + Night") & (df["Class"] == "truck")]
    if row.empty:
        print("  Skipped: 'Foggy + Night' truck row not found.")
        return

    r = int(row["True_Positives"].values[0])
    s = int(row["False_Negatives"].values[0])
    op = BinOp.from_rsa(r, s, a=0.5)

    print(f"  Scenario: Foggy Night Truck")
    print(f"  Evidence: TP={r}, FN={s} (N={r+s})")
    print(f"  YOLO Recall: {row['Recall'].values[0]:.4f}")
    print(f"  CUBICS E[x]: {op.prob():.4f}  u={op.u:.4f}")
    print("  Finding: High uncertainty pulls E[x] toward the 0.5 prior instead"
          " of blindly reporting 0% or 100% recall from tiny N.")


# ── Experiment B ──────────────────────────────────────────────────────────────

def experiment_b_prior_sensitivity(
    df,
    output_path: Path,
    prior_weight: float = 2.0,
) -> None:
    """B: Sensitivity to base-rate prior for data-rich vs data-scarce scenarios."""
    print("\n--- B. Prior Sensitivity (Data-Rich vs Data-Scarce) ---")
    person_df = df[df["Class"] == "person"]

    row_rich = person_df[person_df["Scenario"] == "Clear + Daytime"]
    row_poor = person_df[person_df["Scenario"] == "Foggy + Dawn/dusk"]
    if row_rich.empty or row_poor.empty:
        print("  Skipped: required scenarios not found.")
        return

    r_rich = int(row_rich["True_Positives"].values[0])
    s_rich = int(row_rich["False_Negatives"].values[0])
    r_poor = int(row_poor["True_Positives"].values[0])
    s_poor = int(row_poor["False_Negatives"].values[0])

    priors = [0.1, 0.5, 0.9]
    rich_E, poor_E = [], []

    print(f"{'Base Rate (a)':<15} | {'Clear Day E[x]':<25} | {'Foggy Dawn E[x]':<25}")
    print("-" * 70)
    for a in priors:
        op_r = BinOp.from_rsa(r_rich, s_rich, a=a)
        op_p = BinOp.from_rsa(r_poor, s_poor, a=a)
        rich_E.append(op_r.prob())
        poor_E.append(op_p.prob())
        print(f"a = {a:<11} | E = {op_r.prob():.4f} (u={op_r.u:.4f})     | E = {op_p.prob():.4f} (u={op_p.u:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(priors, rich_E, marker="o", color="green", label="Clear Day (Data-Rich)")
    plt.plot(priors, poor_E, marker="s", color="red", label="Foggy Dawn (Data-Scarce)")
    plt.title("RQ3: Sensitivity to Prior Selection ($a$)")
    plt.xlabel("Selected Base Rate Prior ($a$)")
    plt.ylabel("Subjective Logic Expected Probability ($E[x]$)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


# ── Experiment C ──────────────────────────────────────────────────────────────

def experiment_c_label_noise(df, prior_weight: float = 2.0) -> None:
    """C: Show mathematical robustness to 5% label noise injection."""
    print("\n--- C. Robustness to Label Noise (Misclassification) ---")
    person_df = df[df["Class"] == "person"]

    row_target = person_df[person_df["Scenario"] == "Snowy + Night"]
    row_rich = person_df[person_df["Scenario"] == "Clear + Daytime"]
    if row_target.empty or row_rich.empty:
        print("  Skipped: required scenarios not found.")
        return

    r_t = int(row_target["True_Positives"].values[0])
    s_t = int(row_target["False_Negatives"].values[0])
    r_rich = int(row_rich["True_Positives"].values[0])
    s_rich = int(row_rich["False_Negatives"].values[0])

    op_clean = BinOp.from_rsa(r_t, s_t)
    noise_ratio = 0.05
    r_noise = int(r_rich * noise_ratio)
    s_noise = int(s_rich * noise_ratio)
    op_noisy = BinOp.from_rsa(r_t + r_noise, s_t + s_noise)

    delta_E = op_noisy.prob() - op_clean.prob()
    print(f"  Original Snowy Night (N={r_t+s_t}):            E = {op_clean.prob():.4f}")
    print(f"  Noisy Snowy Night (+5% Clear Day injected): E = {op_noisy.prob():.4f}")
    print(f"  Delta E: {delta_E:+.4f}")
    print("  Finding: Additive SL evidence aggregation limits the impact of a "
          "5% systematic metadata error to a negligible shift.")


# ── Experiment D ──────────────────────────────────────────────────────────────

def experiment_d_context_misclassification(
    df,
    output_path: Path,
    prior_weight: float = 2.0,
) -> None:
    """D: Robustness curve across 0–50% context misclassification noise."""
    print("\n--- D. Context Misclassification Sensitivity Curve ---")
    person_df = df[df["Class"] == "person"]

    row_clear = person_df[person_df["Scenario"] == "Clear + Daytime"]
    row_snowy = person_df[person_df["Scenario"] == "Snowy + Night"]
    row_foggy = person_df[person_df["Scenario"] == "Foggy + Dawn/dusk"]

    if row_clear.empty or row_snowy.empty or row_foggy.empty:
        print("  Skipped: required scenarios not found.")
        return

    r_clear = int(row_clear["True_Positives"].iloc[0])
    s_clear = int(row_clear["False_Negatives"].iloc[0])
    r_snow = int(row_snowy["True_Positives"].iloc[0])
    s_snow = int(row_snowy["False_Negatives"].iloc[0])
    r_fog = int(row_foggy["True_Positives"].iloc[0])
    s_fog = int(row_foggy["False_Negatives"].iloc[0])

    noise_levels = np.linspace(0, 0.5, 20)
    snowy_E, foggy_E = [], []

    for noise in noise_levels:
        r_poison = int(r_clear * noise)
        s_poison = int(s_clear * noise)
        snowy_E.append(BinOp.from_rsa(r_snow + r_poison, s_snow + s_poison).prob())
        foggy_E.append(BinOp.from_rsa(r_fog + r_poison, s_fog + s_poison).prob())

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels * 100, snowy_E, label="Snowy Night (Data-Rich)", color="blue", linewidth=2)
    plt.plot(noise_levels * 100, foggy_E, label="Foggy Dawn (Data-Scarce)", color="red",
             linestyle="--", linewidth=2)
    plt.axhline(
        y=BinOp.from_rsa(r_clear, s_clear).prob(), color="green", alpha=0.3,
        label="Clear Day Performance (Noise Source)"
    )
    plt.xlabel("Context Classifier Error Rate (% Noise Injected)", fontsize=15)
    plt.ylabel("Expected Probability $E[x]$", fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")

    print(f"\n{'Noise %':<10} | {'Snowy E[x]':<15} | {'Foggy E[x]':<15}")
    for i in [0, 5, 10, 19]:
        print(f"{noise_levels[i]*100:>8.1f}% | {snowy_E[i]:>13.4f} | {foggy_E[i]:>13.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    p.add_argument("--csv", type=Path, default=None, help="Override metrics CSV path.")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory for output figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else default_config()
    if args.csv is not None:
        cfg.metrics_csv = args.csv
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" RQ3: SENSITIVITY AND ROBUSTNESS ANALYSIS")
    print("=" * 60)

    df = load_metrics_csv(cfg.metrics_csv)

    experiment_a_data_scarcity(df, prior_weight=cfg.sl_prior_weight)
    experiment_b_prior_sensitivity(
        df,
        output_path=cfg.output_dir / "rq3_prior_sensitivity.svg",
        prior_weight=cfg.sl_prior_weight,
    )
    experiment_c_label_noise(df, prior_weight=cfg.sl_prior_weight)
    experiment_d_context_misclassification(
        df,
        output_path=cfg.output_dir / "rq3_context_sensitivity_curve.svg",
        prior_weight=cfg.sl_prior_weight,
    )


if __name__ == "__main__":
    main()

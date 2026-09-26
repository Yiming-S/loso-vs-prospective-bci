#!/usr/bin/env python3
"""Run the theory Monte-Carlo grid and save table + figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.theory import (run_grid, simulate, summarize, analytic_gap_per_t,  # noqa: E402
                        simulate_tslr, tslr_decomposition)
from src.config import RESULTS_DIR, FIG_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-tslr", action="store_true",
                        help="reuse an existing theory_tslr.csv when only refreshing figures")
    args = parser.parse_args()
    grid = run_grid(seed=0)
    grid.to_csv(RESULTS_DIR / "theory_grid.csv", index=False)
    print(grid.round(4).to_string(index=False))

    # per-t curve for the canonical random-walk regime
    df = simulate(T=8, tau=0.3, sigma=0.5, mu=0.0, seed=1)
    _, per_t = summarize(df)
    per_t.to_csv(RESULTS_DIR / "theory_per_t.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    # (a) ratio across regimes
    g = grid.copy()
    regime_labels = [name.replace("_", " ") for name in g["name"]]
    ax[0].bar(regime_labels, g["ratio"], color="#4477AA")
    ax[0].axhline(1.0, color="k", lw=1, ls="--")
    ax[0].set_ylabel(r"$E[\mathrm{MSE}_{forward}] / E[\mathrm{MSE}_{LOSO}]$")
    ax[0].set_title("(a) Selected toy-model regimes\n(ratio > 1 means forward is worse in that setting)")
    ax[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(g["ratio"]):
        ax[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    # (b) per-t
    ax[1].plot(per_t.index, per_t["mse_loso"], "o-", label="LOSO", color="#4477AA")
    ax[1].plot(per_t.index, per_t["mse_forward"], "s-", label="forward (expanding)", color="#EE6677")
    ax[1].set_xlabel("target session t"); ax[1].set_ylabel("E[MSE]")
    ax[1].set_title("(b) MSE by target session\n(random walk, T=8)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_theory.png", dpi=150)
    print("saved", FIG_DIR / "fig5_theory.png")

    # (regime diagnostic) analytic per-t optimism gap as drift grows, T=15:
    # the shape can flip from monotone-decreasing to inverted-U as tau increases.
    # This is an illustration inside the mean-estimator model, not an estimator
    # of the empirical data's latent drift regime.
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    colors = {0.05: "#004488", 0.15: "#4477AA", 0.30: "#EE6677", 0.60: "#AA3377"}
    for tau in (0.05, 0.15, 0.30, 0.60):
        c = analytic_gap_per_t(T=15, tau=tau, sigma=0.5, mu=0.0)
        ax2.plot(c["t"], c["gap"], "o-", ms=4, color=colors[tau],
                 label=fr"$\tau={tau:.2f}$")
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlabel("target session t")
    ax2.set_ylabel(r"$E[\mathrm{MSE}_{forward}] - E[\mathrm{MSE}_{LOSO}]$")
    ax2.set_title("Toy mean-estimator sensitivity to assumed drift\n"
                  "curve shape is parameter-dependent and not an empirical drift estimate")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig9_theory_regime.png", dpi=150)
    print("saved", FIG_DIR / "fig9_theory_regime.png")

    # (bridge to the real classifier + metric) fit actual logistic regression on
    # drifting tangent-space features; show the size/look-ahead AUC decomposition.
    taus = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]
    tslr_path = RESULTS_DIR / "theory_tslr.csv"
    if args.reuse_tslr and tslr_path.exists():
        tdf = pd.read_csv(tslr_path)
    else:
        recs = [{"tau": tau, **tslr_decomposition(
            simulate_tslr(T=12, tau=tau, reps=60, seed=0))} for tau in taus]
        tdf = pd.DataFrame(recs)
        tdf.to_csv(tslr_path, index=False)
    print(tdf.round(4).to_string(index=False))
    fig3, ax3 = plt.subplots(figsize=(7, 4.2))
    ax3.plot(tdf["tau"], tdf["total"], "o-", color="#17222b", label="total gap")
    ax3.plot(tdf["tau"], tdf["size"], "o-", color="#0c6e77", label="size (extra data)")
    ax3.plot(tdf["tau"], tdf["look_ahead"], "o-", color="#a9741c",
             label="sampled composition (same size)")
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xlabel("session drift  tau")
    ax3.set_ylabel("AUC difference (synthetic logistic regression)")
    ax3.set_title("One synthetic logistic-regression illustration\n"
                  "AUC decomposition depends on the chosen generative parameters")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "fig10_tslr_bridge.png", dpi=150)
    print("saved", FIG_DIR / "fig10_tslr_bridge.png")


if __name__ == "__main__":
    main()

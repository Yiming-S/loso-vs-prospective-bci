#!/usr/bin/env python3
"""Analyses and manuscript assets for the estimand-focused major revision.

This script operates on the audited result-level CSV files.  It makes the
conditioning event explicit, reports participant- and target-weighted results,
adds dataset-level meta-analysis and practical-impact summaries, places every
contrast on an auditable support set, and renders the revised vector figures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PAPER = ROOT / "paper"
# Physical widths match the manuscript's final two-column and one-column sizes.
WIDTH_DOUBLE = 177.53 / 25.4
WIDTH_SINGLE = 85.29 / 25.4
FIGSIZE = (WIDTH_DOUBLE, 3.5)
INK = "#171717"
MUTED = "#595959"
MID_GRAY = "#999999"
SUPPORT = "#EEEEEE"
WHITE = "#FFFFFF"
CLASSICAL = ("csp_lda", "ts_lr")
MODEL_LABEL = {"csp_lda": "CSP+LDA", "ts_lr": "TS+LR"}
DATASETS = (
    "BNCI2014_004", "Zhou2016", "Zhou2020", "Kumar2024", "Ma2020",
    "Stieger2021",
)
COLORS = {"csp_lda": INK, "ts_lr": INK}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": MID_GRAY,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "hatch.linewidth": 0.5,
    "axes.prop_cycle": plt.cycler(color=[INK, MUTED, MID_GRAY]),
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save_figure(fig, stem, bottom=0.0, layout=True):
    """Preserve final-size typography and each figure's content-driven height."""
    if layout:
        fig.tight_layout(rect=(0, bottom, 1, 1), pad=0.7)
    for suffix, kwargs in (
        ("svg", {"format": "svg"}),
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 180}),
    ):
        fig.savefig(FIGURES / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def figure_protocol_logic():
    """Chronological donor matrix and the two explicitly defined averages."""
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.01, 0.98, "(a) Donor availability at the same target",
            ha="left", va="top", fontsize=10, weight="bold")
    positions = (0.39, 0.51, 0.63, 0.75, 0.90)
    ax.add_patch(FancyArrowPatch((0.35, 0.86), (0.95, 0.86),
                                arrowstyle="->", mutation_scale=9,
                                linewidth=0.7, color=MUTED))
    ax.text(0.65, 0.895, "Recorded session order", ha="center",
            va="center", fontsize=8, color=MUTED)
    ax.text(0.51, 0.805, "Prior donors", ha="center", fontsize=8)
    ax.text(0.75, 0.805, "Test target", ha="center", fontsize=8)
    ax.text(0.90, 0.805, "Later session", ha="center", fontsize=8)

    def protocol_row(ypos, name, interpretation, include_future):
        ax.text(0.01, ypos + 0.018, name, ha="left", va="center",
                fontsize=9, weight="bold")
        ax.text(0.01, ypos - 0.038, interpretation, ha="left", va="center",
                fontsize=8, color=MUTED)
        for index, xpos in enumerate(positions):
            role = "prior" if index < 3 else "target" if index == 3 else "later"
            face = "#D9D9D9" if role == "prior" else WHITE
            linewidth = 1.8 if role == "target" else 0.7
            linestyle = "--" if role == "later" and not include_future else "-"
            hatch = "////" if role == "later" and include_future else None
            ax.add_patch(Rectangle((xpos - 0.04, ypos - 0.047), 0.08, 0.094,
                                   facecolor=face, edgecolor=INK if role == "target" else MUTED,
                                   linewidth=linewidth, linestyle=linestyle, hatch=hatch))
            ax.text(xpos, ypos, f"S{index}", ha="center", va="center", fontsize=9,
                    weight="bold" if role == "target" else "normal",
                    bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.7}
                    if hatch else None)
        ax.text(0.90, ypos - 0.085, "included donor" if include_future else "unavailable at $t$",
                ha="center", va="center", fontsize=8, color=MUTED)

    protocol_row(0.70, "LOSO", "Completed trajectory", True)
    protocol_row(0.515, "Expanding window", "History available at $t$", False)
    ax.text(0.50, 0.36, "Paired comparison: same test trials and decoder",
            ha="center", va="center", fontsize=9)
    ax.text(0.50, 0.302, r"$\Delta=\mathrm{AUC}_{LOSO}-\mathrm{AUC}_{EXP}$",
            ha="center", va="center", fontsize=9)
    ax.plot([0.01, 0.99], [0.255, 0.255], color=MID_GRAY, lw=0.6)
    ax.text(0.01, 0.22, "(b) Analysis populations", ha="left", va="top",
            fontsize=10, weight="bold")
    ax.text(0.01, 0.13, "Later donor available (primary)", fontsize=9, weight="bold")
    ax.text(0.01, 0.058, r"$t=1,\ldots,T-2$; at least one later donor", fontsize=8)
    ax.text(0.54, 0.13, "All origins (sensitivity)", fontsize=9, weight="bold")
    ax.text(0.54, 0.058, r"$t=1,\ldots,T-1$; final origin has $\Delta=0$", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.99)
    save_figure(fig, "fig0_protocol_logic", layout=False)


def t_summary(values):
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(x)
    mean = float(np.mean(x)) if n else math.nan
    median = float(np.median(x)) if n else math.nan
    if n > 1:
        se = float(stats.sem(x))
        half = float(stats.t.ppf(0.975, n - 1) * se)
        t_value, p_value = stats.ttest_1samp(x, 0.0)
        wilcoxon = stats.wilcoxon(x, alternative="two-sided") if np.any(x != 0) else None
    else:
        se = half = t_value = p_value = math.nan
        wilcoxon = None
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "q1": float(np.quantile(x, 0.25)) if n else math.nan,
        "q3": float(np.quantile(x, 0.75)) if n else math.nan,
        "sd": float(np.std(x, ddof=1)) if n > 1 else math.nan,
        "se": se,
        "ci": [mean - half, mean + half] if n > 1 else [math.nan, math.nan],
        "t": float(t_value),
        "p": float(p_value),
        "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon is not None else math.nan,
        "prob_gt_0": float(np.mean(x > 0)) if n else math.nan,
        "prob_gt_001": float(np.mean(x > 0.01)) if n else math.nan,
        "prob_gt_002": float(np.mean(x > 0.02)) if n else math.nan,
    }


def target_weighted_summary(target_rows, seed=20260720, n_boot=5000):
    """Point estimate over participant-target means; participant-cluster bootstrap CI."""
    d = target_rows.groupby(
        ["subject_uid", "target_session"], as_index=False
    )["delta"].mean()
    subject = d.groupby("subject_uid")["delta"].agg(["sum", "count"])
    point = float(subject["sum"].sum() / subject["count"].sum())
    rng = np.random.default_rng(seed)
    sums = subject["sum"].to_numpy()
    counts = subject["count"].to_numpy()
    n = len(subject)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        index = rng.integers(0, n, n)
        boot[b] = sums[index].sum() / counts[index].sum()
    return {
        "n_subjects": int(n),
        "n_targets": int(len(d)),
        "mean": point,
        "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "ci_method": "participant-cluster percentile bootstrap, 5000 resamples",
    }


def random_effects_meta(participant, value="delta"):
    rows = []
    for dataset, frame in participant.groupby("dataset"):
        x = frame[value].dropna().to_numpy()
        rows.append({
            "dataset": dataset,
            "n": int(len(x)),
            "estimate": float(np.mean(x)),
            "se": float(stats.sem(x)),
        })
    d = pd.DataFrame(rows)
    y = d["estimate"].to_numpy()
    v = np.square(d["se"].to_numpy())

    def restricted_objective(tau2):
        w = 1.0 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (
            np.sum(np.log(v + tau2)) + np.log(np.sum(w)) +
            np.sum(w * np.square(y - mu))
        )

    upper = max(0.01, float(np.var(y, ddof=1) * 10))
    result = optimize.minimize_scalar(
        restricted_objective, bounds=(0.0, upper), method="bounded",
        options={"xatol": 1e-12},
    )
    tau2 = float(max(0.0, result.x))
    w = 1.0 / (v + tau2)
    mean = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    k = len(d)
    crit = float(stats.t.ppf(0.975, k - 1))
    pred_crit = float(stats.t.ppf(0.975, max(1, k - 2)))
    fixed_w = 1.0 / v
    fixed_mean = np.sum(fixed_w * y) / np.sum(fixed_w)
    Q = float(np.sum(fixed_w * np.square(y - fixed_mean)))
    i2 = float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0
    hk_scale = float(np.sum(w * np.square(y - mean)) / (k - 1))
    hk_se = float(np.sqrt(hk_scale / np.sum(w)))
    meta = {
        "k": int(k),
        "estimate": mean,
        "ci": [mean - crit * se, mean + crit * se],
        "prediction_interval": [
            mean - pred_crit * math.sqrt(tau2 + se * se),
            mean + pred_crit * math.sqrt(tau2 + se * se),
        ],
        "tau2": tau2,
        "i2": i2,
        "Q": Q,
        "method": "REML variance component; t interval across six datasets",
        "ci_formula": "mu +/- t_(k-1, .975) / sqrt(sum(w)); w=1/(v+tau2)",
        "prediction_interval_formula": "mu +/- t_(k-2, .975)*sqrt(tau2+1/sum(w))",
        "hksj_sensitivity": {
            "scale": hk_scale,
            "se": hk_se,
            "ci": [mean - crit * hk_se, mean + crit * hk_se],
            "formula": "mu +/- t_(k-1, .975)*sqrt(q/sum(w)); q=sum(w*(y-mu)^2)/(k-1)",
        },
    }
    return d, meta


def participant_frame(rows):
    return rows.groupby(
        ["dataset", "subject", "subject_uid"], as_index=False
    )["delta"].mean()


def leave_one_dataset_out(rows):
    """Point-only exclusions with the primary equal-participant weighting.

    Exclusion changes the empirical participant population; it does not create
    a new sampling interval or estimate uncertainty about unseen datasets.
    Only the two fixed classical decoders enter the participant means.
    """
    frame = rows[rows["model"].isin(CLASSICAL)].copy()
    if frame.empty or frame["delta"].isna().any():
        raise ValueError("Dataset exclusions require nonmissing classical results")
    participant = participant_frame(frame)
    if participant["subject_uid"].duplicated().any():
        raise ValueError("Each participant identifier must map to one dataset")
    if participant["dataset"].nunique() < 2:
        raise ValueError("Dataset exclusions require at least two datasets")
    records = []
    for dataset in sorted(participant["dataset"].unique()):
        included = participant[participant["dataset"] != dataset]
        excluded = participant[participant["dataset"] == dataset]
        records.append({
            "excluded_dataset": dataset,
            "excluded_participants": int(len(excluded)),
            "remaining_datasets": int(included["dataset"].nunique()),
            "remaining_participants": int(len(included)),
            "mean": float(included["delta"].mean()),
        })
    return records


def estimate_bundle(rows):
    participant = participant_frame(rows)
    participant_summary = t_summary(participant["delta"])
    target_summary = target_weighted_summary(rows)
    dataset_values = participant.groupby("dataset")["delta"].mean()
    dataset_summary = t_summary(dataset_values)
    meta_rows, meta = random_effects_meta(participant)
    return participant, {
        "participant_weighted": participant_summary,
        "target_weighted": target_summary,
        "dataset_weighted": dataset_summary,
        "random_effects": meta,
    }, meta_rows


def support_record(name, d):
    targets = d[["dataset", "subject", "target_session"]].drop_duplicates()
    by_dataset = targets.groupby("dataset").size().to_dict()
    return {
        "contrast": name,
        "participants": int(d["subject_uid"].nunique()),
        "participant_targets": int(len(targets)),
        "fits": int(len(d)),
        "datasets": "; ".join(f"{k}:{v}" for k, v in sorted(by_dataset.items())),
        "target_range": f"{int(d['target_session'].min())}--{int(d['target_session'].max())}",
        "mean_history_size": float(d["target_session"].mean()),
    }


def contrast_summary(d, left, right):
    x = d.dropna(subset=[left, right]).copy()
    x["contrast_value"] = x[left] - x[right]
    participant = x.groupby("subject_uid")["contrast_value"].mean()
    result = t_summary(participant)
    result.update({"left": left, "right": right, "n_fits": int(len(x))})
    return result


def monte_carlo_summary(long_df, supports):
    """Marginal plug-in SEs and a covariance-agnostic triangle bound.

    Stored draw SDs use ddof=0. Convert their squared values to sample variances
    before estimating each draw-mean variance. Shared donor draws induce unknown
    covariances, so no independent-row aggregate MCSE is reported. The triangle
    inequality bounds aggregate SE by sum(abs(weight)*marginal SE); inserting
    five-draw marginal estimates gives a plug-in bound, not a confidence bound.
    """
    matched = long_df[
        (long_df["model"].isin(CLASSICAL)) &
        (long_df["protocol"] == "loso_matched") &
        (long_df["target_session"] >= 1)
    ].copy()
    keys = ["dataset", "subject", "model", "target_session"]
    matched = matched[keys + ["auc_sd_draws", "n_draws"]]
    records = {}
    for name, support in supports.items():
        frame = support[keys + ["subject_uid"]].merge(
            matched, on=keys, how="left", validate="one_to_one"
        )
        if frame[["auc_sd_draws", "n_draws"]].isna().any().any():
            raise ValueError(f"Missing draw variability on {name} support")
        if (frame["n_draws"] <= 1).any():
            raise ValueError("At least two draws are needed for marginal variance")
        # sample_SD / sqrt(D) = stored_population_SD / sqrt(D-1)
        frame["marginal_se"] = frame["auc_sd_draws"] / np.sqrt(frame["n_draws"] - 1)
        n_participants = frame["subject_uid"].nunique()
        row_counts = frame.groupby("subject_uid")["subject_uid"].transform("size")
        frame["weight"] = 1.0 / (n_participants * row_counts)
        if not np.isclose(frame["weight"].sum(), 1.0):
            raise AssertionError("Monte Carlo aggregation weights do not sum to one")
        records[name] = {
            "participants": int(n_participants),
            "participant_targets": int(len(frame[keys[:-2] + ["target_session"]].drop_duplicates())),
            "model_target_rows": int(len(frame)),
            "draws": sorted(frame["n_draws"].astype(int).unique().tolist()),
            "median_marginal_se": float(frame["marginal_se"].median()),
            "q95_marginal_se": float(frame["marginal_se"].quantile(0.95)),
            "max_marginal_se": float(frame["marginal_se"].max()),
            "triangle_se_bound_plugin": float((frame["weight"] * frame["marginal_se"]).sum()),
        }
    return {
        "by_support": records,
        "marginal_se_formula": "stored_ddof0_SD/sqrt(D-1)",
        "aggregation_weights": "Equal participants, then equal targets and decoders within participant",
        "triangle_bound_formula": "SE(sum(a_j X_j)) <= sum(abs(a_j)*SE(X_j))",
        "note": "The triangle expression allows arbitrary covariance, including shared donor draws across decoders and datasets. Marginal SEs estimated from five draws are plugged in; this is not an exact MCSE, an upper confidence limit, or evidence of reference invariance.",
    }


def matched_trial_summary(long_df, supports):
    """Describe trial-count mismatch despite equal donor-session counts."""
    keys = ["dataset", "subject", "model", "target_session"]
    source = long_df[long_df["model"].isin(CLASSICAL) &
                     long_df["protocol"].isin(["loso_matched", "forward_expanding"])]
    values = source.pivot(index=keys, columns="protocol", values="n_train_trials").reset_index()
    records = []
    for name, support in supports.items():
        frame = support[keys].merge(values, on=keys, how="left", validate="one_to_one")
        if frame[["loso_matched", "forward_expanding"]].isna().any().any():
            raise ValueError(f"Missing training-trial counts on {name} support")
        frame["trial_count_difference"] = frame["loso_matched"] - frame["forward_expanding"]
        for dataset, group in frame.groupby("dataset"):
            delta = group["trial_count_difference"]
            records.append({
                "support": name, "dataset": dataset, "model_target_rows": len(group),
                "unequal_trial_count_rows": int((~np.isclose(delta, 0.0, atol=1e-9)).sum()),
                "min_trial_count_difference": float(delta.min()),
                "max_trial_count_difference": float(delta.max()),
            })
    return records


def decoder_margin_frame(frame):
    means = frame.groupby(["subject_uid", "model"])[["loso", "forward_expanding"]].mean().unstack("model")
    margins = pd.DataFrame({
        "loso_margin": means["loso"]["ts_lr"] - means["loso"]["csp_lda"],
        "expanding_margin": means["forward_expanding"]["ts_lr"] - means["forward_expanding"]["csp_lda"],
    })
    margins["flip"] = margins["loso_margin"] * margins["expanding_margin"] < 0
    return margins


def operational_impact(frame):
    margins = decoder_margin_frame(frame)
    absolute = frame.groupby("subject_uid")[["loso", "forward_expanding"]].mean()
    sensitivity = []
    for threshold in (0.0, 0.005, 0.01, 0.02):
        selected = margins["flip"] & (margins["loso_margin"].abs() > threshold) & (margins["expanding_margin"].abs() > threshold)
        sensitivity.append({"bilateral_margin_threshold": threshold,
                            "n": int(selected.sum()), "percent": float(100 * selected.mean())})
    result = {
        "model_selection_changes": int(margins["flip"].sum()),
        "model_selection_change_percent": float(100 * margins["flip"].mean()),
        "csp_to_ts": int(((margins["loso_margin"] < 0) & (margins["expanding_margin"] > 0)).sum()),
        "ts_to_csp": int(((margins["loso_margin"] > 0) & (margins["expanding_margin"] < 0)).sum()),
        "ties_loso": int((margins["loso_margin"] == 0).sum()),
        "ties_expanding": int((margins["expanding_margin"] == 0).sum()),
        "bilateral_margin_sensitivity": sensitivity,
        "direction": "LOSO observed higher-AUC decoder to expanding-window observed higher-AUC decoder",
        "interpretation": "Observed participant mean AUC ordering; not prospective model selection or participant-specific uncertainty",
        "by_decoder": {model: t_summary(group.groupby("subject_uid")["delta"].mean())
                       for model, group in frame.groupby("model")},
    }
    for suffix, threshold in (("070", 0.70), ("075", 0.75)):
        result[f"threshold_{suffix}_loso_only"] = int(((absolute["loso"] >= threshold) & (absolute["forward_expanding"] < threshold)).sum())
        result[f"threshold_{suffix}_forward_only"] = int(((absolute["loso"] < threshold) & (absolute["forward_expanding"] >= threshold)).sum())
    return result


def figure_decoder_margins(all_rows, conditional_rows):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_DOUBLE, 4.1))
    frames = [decoder_margin_frame(rows) for rows in (conditional_rows, all_rows)]
    # Match the previous full-range panels: union of every observed point and
    # the near-zero window, with Matplotlib's default five-percent margins.
    limits = []
    for column in ("loso_margin", "expanding_margin"):
        values = np.concatenate([frame[column].to_numpy() for frame in frames] +
                                [np.array([-0.04, 0.04])])
        lo, hi = values.min(), values.max()
        pad = 0.05 * (hi - lo)
        limits.append((lo - pad, hi + pad))
    for col, (name, margins) in enumerate(zip(("Conditional (primary)", "All origins"), frames)):
        flip = margins["flip"]
        for row in (0, 1):
            ax = axes[row, col]
            # Opaque identical fills form a union; the crossing is not darker.
            ax.axhspan(-0.01, 0.01, facecolor=SUPPORT, edgecolor="none", zorder=0)
            ax.axvspan(-0.01, 0.01, facecolor=SUPPORT, edgecolor="none", zorder=0)
            ax.axhline(0, color=MUTED, lw=0.65, zorder=1)
            ax.axvline(0, color=MUTED, lw=0.65, zorder=1)
            for mask, color, marker, label in (
                (~flip, MID_GRAY, "o", "Same observed ordering"),
                (flip, INK, "D", "Opposite observed ordering"),
            ):
                ax.scatter(margins.loc[mask, "loso_margin"],
                           margins.loc[mask, "expanding_margin"],
                           s=18 if marker == "o" else 24, color=color, marker=marker,
                           edgecolors=WHITE, linewidths=0.3, label=label, zorder=3)
            panel = chr(ord("a") + 2 * row + col)
            ax.set_title(f"({panel}) {name}: {'full range' if row == 0 else 'near zero'}",
                         loc="left", pad=6)
            ax.tick_params(length=3, pad=3)
            if row == 0:
                ax.set(xlim=limits[0], ylim=limits[1])
                ax.add_patch(Rectangle((-0.04, -0.04), 0.08, 0.08,
                                       fill=False, edgecolor=MUTED, lw=0.7, ls="--"))
                ax.text(0.97, 0.04, f"Opposite: {int(flip.sum())}/{len(margins)}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 2})
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            else:
                ax.set(xlim=(-0.04, 0.04), ylim=(-0.04, 0.04),
                       xticks=(-0.04, -0.02, 0, 0.02, 0.04),
                       yticks=(-0.04, -0.02, 0, 0.02, 0.04))
            if col == 1:
                ax.tick_params(labelleft=False)
    fig.supxlabel("LOSO decoder difference (TS+LR − CSP+LDA)", y=0.115, fontsize=9)
    fig.supylabel("Expanding-window decoder difference\n(TS+LR − CSP+LDA)",
                  x=0.012, y=0.575, fontsize=9, linespacing=1.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.05),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.8)
    fig.text(0.52, 0.024, "Gray region: either absolute decoder difference ≤ 0.01 AUC.",
             ha="center", va="bottom", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.125, right=0.98, bottom=0.21, top=0.94,
                        hspace=0.40, wspace=0.17)
    save_figure(fig, "fig6_decoder_margins", layout=False)


def drift_influence(inner):
    import statsmodels.formula.api as smf

    d = inner.dropna(subset=["delta", "drift_z"]).copy()
    formula = "delta ~ drift_z + target_session + C(model) + C(dataset)"
    ols = smf.ols(formula, d).fit()
    cooks = ols.get_influence().cooks_distance[0]
    cutoff = 4.0 / len(d)
    keep = cooks <= cutoff
    trimmed = smf.ols(formula, d.loc[keep]).fit(
        cov_type="cluster", cov_kwds={"groups": d.loc[keep, "subject_uid"]}
    )
    clustered = ols.get_robustcov_results(
        cov_type="cluster", groups=d["subject_uid"]
    )
    names = list(ols.params.index)
    index = names.index("drift_z")
    return {
        "formula": formula,
        "n": int(len(d)),
        "n_subjects": int(d["subject_uid"].nunique()),
        "residual_covariance": "independent working residuals; inference clustered by participant",
        "drift_beta_clustered": float(clustered.params[index]),
        "drift_ci_clustered": [float(v) for v in clustered.conf_int()[index]],
        "drift_p_clustered": float(clustered.pvalues[index]),
        "cooks_cutoff": cutoff,
        "n_above_cooks_cutoff": int(np.sum(~keep)),
        "trimmed_drift_beta": float(trimmed.params["drift_z"]),
        "trimmed_drift_ci": [float(v) for v in trimmed.conf_int().loc["drift_z"]],
        "trimmed_drift_p": float(trimmed.pvalues["drift_z"]),
    }


def ma_day_summary(block_rows):
    path = RESULTS / "sensitivity" / "ma2020_day_level.csv"
    if not path.exists():
        return {}, pd.DataFrame()
    day = pd.read_csv(path)
    piv = day.pivot_table(
        index=["subject", "model", "target_day"], columns="protocol",
        values=["auc", "balanced_accuracy"],
    ).reset_index()
    rows = []
    for metric in ("auc", "balanced_accuracy"):
        value = piv[(metric, "loso")] - piv[(metric, "forward_expanding")]
        work = pd.DataFrame({
            "subject": piv[("subject", "")],
            "model": piv[("model", "")],
            "target": piv[("target_day", "")],
            "delta": value,
        })
        for label, frame in (
            ("conditional", work[work["target"] < 2]),
            ("all_origin", work),
        ):
            participant = frame.groupby("subject")["delta"].mean()
            rec = t_summary(participant)
            rec.update({"metric": metric, "estimand": label, "time_unit": "day"})
            rows.append(rec)

    block = block_rows[block_rows["dataset"] == "Ma2020"]
    for label, frame in (
        ("conditional", block[~block["is_structural_zero"]]),
        ("all_origin", block),
    ):
        participant = frame.groupby("subject_uid")["delta"].mean()
        rec = t_summary(participant)
        rec.update({"metric": "auc", "estimand": label, "time_unit": "block"})
        rows.append(rec)
    out = pd.DataFrame(rows)
    return {f"{r['time_unit']}_{r['metric']}_{r['estimand']}": r for r in rows}, out


def figure_forest(inner):
    participant = inner.groupby(
        ["dataset", "model", "subject_uid"], as_index=False
    )["delta"].mean()
    rows = []
    for (dataset, model), frame in participant.groupby(["dataset", "model"]):
        rec = t_summary(frame["delta"])
        rec.update({"dataset": dataset, "model": model})
        rows.append(rec)
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 2.64))
    ybase = np.arange(len(DATASETS))[::-1]
    offsets = {"csp_lda": 0.13, "ts_lr": -0.13}
    for model in CLASSICAL:
        sub = d[d["model"] == model].set_index("dataset").reindex(DATASETS)
        y = ybase + offsets[model]
        ax.errorbar(
            sub["mean"], y,
            xerr=[sub["mean"] - sub["ci"].map(lambda x: x[0]),
                  sub["ci"].map(lambda x: x[1]) - sub["mean"]],
            fmt="o" if model == "csp_lda" else "s", capsize=2.5,
            color=COLORS[model], markerfacecolor=INK if model == "csp_lda" else WHITE,
            markeredgecolor=INK, markeredgewidth=0.9, elinewidth=1.0,
            markersize=5, label=MODEL_LABEL[model],
        )
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(ybase)
    ax.set_yticklabels(DATASETS)
    ax.set_xlabel("Conditional mean AUC difference (LOSO − expanding window)")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.6)
    save_figure(fig, "fig1_gap_forest")
    return d


def figure_estimands(bundle, cond_participant, all_participant, common_contrasts=None):
    """(a) session-count-matched comparisons; (b) participant distributions.

    The four aggregate means of the two summaries are tabulated in
    Table 2, so the figure gives the matched-reference partition instead."""
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 2.95),
                           gridspec_kw={"width_ratios": [1.0, 1.05]})
    small = 8
    cc = common_contrasts
    rows = [
        ("LOSO \u2212 matched", cc["primary_conditional"]["quantity_uniform"], True),
        ("Matched \u2212 expanding", cc["primary_conditional"]["composition_uniform"], True),
        ("LOSO \u2212 matched", cc["strict_future"]["quantity_uniform"], False),
        ("Matched \u2212 expanding", cc["strict_future"]["composition_uniform"], False),
        ("Future \u2212 expanding", cc["strict_future"]["strict_future"], False),
    ]
    yc = np.array([4.6, 3.6, 2.0, 1.0, 0.0])
    for (label, rec, filled), ypos in zip(rows, yc):
        ax[0].errorbar(rec["mean"], ypos,
                       xerr=[[rec["mean"] - rec["ci"][0]], [rec["ci"][1] - rec["mean"]]],
                       fmt="o", capsize=2.5, color=INK, markersize=5,
                       markerfacecolor=INK if filled else WHITE,
                       markeredgewidth=0.9, elinewidth=1.0)
    ax[0].axvline(0, color=MUTED, lw=0.7)
    ax[0].set_yticks(yc)
    ax[0].set_yticklabels([r[0] for r in rows], fontsize=small)
    for ypos, text in ((5.35, "Later donor available (1,001 targets)"),
                       (2.75, "Strict-future support (560 targets)")):
        ax[0].text(0.02, ypos, text, fontsize=small, color=MUTED, ha="left",
                   va="center", transform=ax[0].get_yaxis_transform())
    ax[0].set_ylim(-0.6, 5.75)
    ax[0].set_xlim(-0.015, 0.062)
    ax[0].set_xticks([0.0, 0.02, 0.04, 0.06])
    ax[0].set_xlabel("Mean AUC difference")
    ax[0].tick_params(axis="y", length=0)
    ax[0].set_title("(a) Session-count-matched comparisons", loc="left", pad=8)
    ax[0].grid(axis="x")
    ax[0].set_axisbelow(True)

    bins = np.linspace(
        min(cond_participant["delta"].min(), all_participant["delta"].min()),
        max(cond_participant["delta"].max(), all_participant["delta"].max()), 22,
    )
    ax[1].hist(cond_participant["delta"], bins=bins, histtype="step", lw=1.15,
               color=INK, label="Conditional (primary)", linestyle="-")
    ax[1].hist(all_participant["delta"], bins=bins, histtype="step", lw=1.15,
               color=MUTED, label="All origins", linestyle="--")
    peak_count = max(np.histogram(frame["delta"], bins=bins)[0].max()
                     for frame in (all_participant, cond_participant))
    # Keep direct reference labels in empty headroom, not on the observations.
    ax[1].set_ylim(0, 1.38 * peak_count)
    for threshold, style, position in ((0.0, "-", 0.93), (0.01, ":", 0.84),
                                        (0.02, ":", 0.75)):
        ax[1].axvline(threshold, color=MID_GRAY, lw=0.7, ls=style, zorder=0)
        ax[1].annotate(f"{threshold:g}", (threshold, position),
                       xycoords=("data", "axes fraction"), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=8,
                       color=MUTED, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.5})
    ax[1].set_xlabel("Participant-level AUC difference")
    ax[1].set_ylabel("Participants")
    ax[1].set_xticks([-0.05, 0.0, 0.05, 0.10])
    ax[1].set_xticklabels(["\u22120.05", "0", "0.05", "0.10"])
    ax[1].set_title("(b) Participant distribution", loc="left", pad=8)
    # Legend sits in the empty right tail, clear of the reference-line labels.
    ax[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.70),
                 handlelength=2.2)
    save_figure(fig, "fig2_estimands_distribution")

def figure_history(inner):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_DOUBLE, 3.48), sharey=True)
    for index, (ax, dataset) in enumerate(zip(axes.flat, DATASETS)):
        d = inner[inner["dataset"] == dataset]
        rows = []
        for surplus, frame in d.groupby("sessions_after"):
            participant = frame.groupby("subject_uid")["delta"].mean()
            rec = t_summary(participant)
            rows.append((float(surplus), rec))
        rows.sort(key=lambda x: x[0])
        x = np.array([r[0] for r in rows])
        mean = np.array([r[1]["mean"] for r in rows])
        low = np.array([r[1]["ci"][0] for r in rows])
        high = np.array([r[1]["ci"][1] for r in rows])
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", capsize=2,
                    color=INK, ecolor=MUTED, elinewidth=0.9, ms=3.8)
        ax.axhline(0, color=MUTED, lw=0.65)
        ax.set_title(f"({chr(ord('a') + index)}) {dataset}", fontsize=10, loc="left", pad=6)
        xmin = min(1.0, float(x.min())) - 0.5
        xmax = float(x.max()) + 0.5
        ax.set_xlim(xmin, xmax)
        tick_index = (np.arange(len(x)) if len(x) <= 6 else
                      np.unique(np.round(np.linspace(0, len(x) - 1, 5)).astype(int)))
        ax.set_xticks(x[tick_index])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.supxlabel("Number of later sessions", y=0.025, fontsize=9)
    fig.supylabel("Mean AUC difference (LOSO − expanding window)", x=0.018, fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.94,
                        hspace=0.42, wspace=0.20)
    save_figure(fig, "fig3_history_by_dataset", layout=False)


def figure_drift(inner):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=False, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        d = inner[(inner["dataset"] == dataset)].dropna(subset=["drift_z", "delta"])
        if len(d) >= 100:
            ax.hexbin(d["drift_z"], d["delta"], gridsize=18, mincnt=1,
                      cmap="Greys", linewidths=0)
        else:
            ax.scatter(d["drift_z"], d["delta"], s=10, alpha=0.6, color=MUTED)
        if d["drift_z"].nunique() > 1:
            slope, intercept, _, _ = stats.theilslopes(d["delta"], d["drift_z"])
            x = np.linspace(d["drift_z"].min(), d["drift_z"].max(), 30)
            ax.plot(x, intercept + slope * x, color=INK, lw=1.2)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("adjacent covariance distance (z)")
    axes[0, 0].set_ylabel("LOSO - expanding window")
    axes[1, 0].set_ylabel("LOSO - expanding window")
    save_figure(fig, "fig4_drift_facets")


def figure_ma_sensitivity(ma_frame):
    if ma_frame.empty:
        return
    d = ma_frame[ma_frame["metric"] == "auc"].copy()
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.33))
    for j, unit in enumerate(("block", "day")):
        for offset, estimand, label in ((-0.11, "all_origin", "All origins"),
                                        (0.11, "conditional", "Conditional")):
            row = d[(d["time_unit"] == unit) & (d["estimand"] == estimand)].iloc[0]
            ax.errorbar(j + offset, row["mean"],
                        yerr=[[row["mean"] - row["ci"][0]],
                              [row["ci"][1] - row["mean"]]],
                        fmt="o", capsize=2.5, color=INK, markersize=5,
                        markerfacecolor=INK if estimand == "all_origin" else WHITE,
                        markeredgewidth=0.9, elinewidth=1.0,
                        label=label if j == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xticks((0, 1), ("15 blocks", "3 days"))
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Mean AUC difference")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.53, 0.005), frameon=False, columnspacing=1.1)
    save_figure(fig, "fig5_ma_time_unit", bottom=0.15)


def fmt(value, digits=3, signed=False):
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def text_minus(value):
    """Typeset a leading minus in text-mode macros as a true minus sign."""
    value = str(value)
    return "$-$" + value[1:] if value.startswith("-") else value


def pvalue(value):
    """Return a math-mode relation for p: '= 0.95', '= 0.007', or '< 0.001'."""
    if not np.isfinite(value):
        return "= \\mathrm{NA}"
    if value < 0.001:
        return "< 0.001"
    if value < 0.01:
        return f"= {value:.3f}"
    return f"= {value:.2f}"

def estimands_table_tex(summary):
    """Balance the two estimands without an ambiguous mixed-unit count column."""
    cond = summary["estimands"]["conditional"]
    all_est = summary["estimands"]["all_origin"]
    participants = all_est["participant_weighted"]["n"]
    datasets = all_est["random_effects"]["k"]
    rows = []
    for label, key in (
        ("Participant-weighted", "participant_weighted"),
        ("Target-weighted", "target_weighted"),
        ("Dataset-weighted", "dataset_weighted"),
        ("Random-effects", "random_effects"),
    ):
        record = all_est[key]
        conditional_record = cond[key]
        population = (f"{participants} participants" if key in
                      ("participant_weighted", "target_weighted")
                      else f"{datasets} datasets")
        rows.append(
            f"{label} ({population}) & {conditional_record['mean'] if 'mean' in conditional_record else conditional_record['estimate']:+.3f} & "
            f"$[{conditional_record['ci'][0]:+.3f}, {conditional_record['ci'][1]:+.3f}]$ & "
            f"{record['mean'] if 'mean' in record else record['estimate']:+.3f} & "
            f"$[{record['ci'][0]:+.3f}, {record['ci'][1]:+.3f}]$ \\\\"
        )
    return "\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table*}[!t]", r"\centering", r"\settableformat",
        r"\caption{LOSO minus expanding-window AUC under the primary conditional estimand and the all-origin sensitivity summary. Conditional targets have a later donor; all origins also include the final structural zero. Target-weighted CIs use a participant-cluster bootstrap.}",
        r"\label{tab:estimands}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l S[table-format=+1.3,retain-explicit-plus=true] c S[table-format=+1.3,retain-explicit-plus=true] c@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Conditional (primary)}} & \multicolumn{2}{c}{\textbf{All origins}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"\textbf{Aggregation} & {\textbf{Mean $\boldsymbol{\Delta}$}} & \textbf{95\% CI} & {\textbf{Mean $\boldsymbol{\Delta}$}} & \textbf{95\% CI} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}",
        r"\end{table*}", "",
    ])


def write_tables(summary, cond_participant, all_participant, meta_rows,
                 support, common_contrasts, forest, ma_frame, wide):
    cond = summary["estimands"]["conditional"]
    all_est = summary["estimands"]["all_origin"]
    (PAPER / "table_estimands.tex").write_text(estimands_table_tex(summary))

    all_practical = t_summary(all_participant["delta"])
    cond_practical = t_summary(cond_participant["delta"])
    practical_rows = [
        f"Median & ${cond_practical['median']:+.3f}$ & ${all_practical['median']:+.3f}$ \\\\",
        f"IQR & $[{cond_practical['q1']:+.3f}, {cond_practical['q3']:+.3f}]$ & "
        f"$[{all_practical['q1']:+.3f}, {all_practical['q3']:+.3f}]$ \\\\",
        r"\addlinespace[0.4em]",
        r"\multicolumn{3}{@{}l}{\textit{Participants above each threshold, $n$ (\%)}} \\",
    ]
    for threshold in (0.0, 0.01, 0.02):
        a = int((all_participant["delta"] > threshold).sum())
        c = int((cond_participant["delta"] > threshold).sum())
        practical_rows.append(
            f"$\\Delta>{threshold:g}$ & {c} ({100*c/len(cond_participant):.1f}\\%) & "
            f"{a} ({100*a/len(all_participant):.1f}\\%) \\\\"
        )
    (PAPER / "table_practical_impact.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[!t]", r"\centering", r"\settableformat",
        r"\caption{Observed participant-level magnitude and counts, $n$ (\%), of the protocol difference.}",
        r"\label{tab:practical}",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}",
        r"\toprule", r"\textbf{Metric} & \textbf{Conditional (primary)} & \textbf{All origins} \\", r"\midrule",
        *practical_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ]))

    all_decision = summary["practical"]["by_estimand"]["all_origin"]
    cond_decision = summary["practical"]["by_estimand"]["conditional"]
    decision_rows = [
        "Opposite ordering & "
        r"\multicolumn{1}{r}{"
        f"{cond_decision['model_selection_changes']} "
        f"({cond_decision['model_selection_change_percent']:.1f}\\%)" + r"} & \multicolumn{1}{r}{" +
        f"{all_decision['model_selection_changes']} "
        f"({all_decision['model_selection_change_percent']:.1f}\\%)" + r"} \\",
        f"CSP+LDA $\\to$ TS+LR & {cond_decision['csp_to_ts']} & {all_decision['csp_to_ts']} \\\\ ",
        f"TS+LR $\\to$ CSP+LDA & {cond_decision['ts_to_csp']} & {all_decision['ts_to_csp']} \\\\ ",
        r"\addlinespace[0.4em]",
        r"\multicolumn{3}{@{}l}{\textit{Opposite ordering with both margins above}} \\",
    ]
    for a, c in zip(all_decision["bilateral_margin_sensitivity"][1:], cond_decision["bilateral_margin_sensitivity"][1:]):
        decision_rows.append(f"${a['bilateral_margin_threshold']:g}$ AUC & {c['n']} & {a['n']} \\\\")
    for suffix, threshold in (("070", "0.70"), ("075", "0.75")):
        decision_rows.extend([
            r"\addlinespace[0.4em]",
            rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{AUC $\geq {threshold}$ under one protocol only}}}} \\",
            f"LOSO only & {cond_decision[f'threshold_{suffix}_loso_only']} & "
            f"{all_decision[f'threshold_{suffix}_loso_only']} \\\\",
            f"Expanding window only & {cond_decision[f'threshold_{suffix}_forward_only']} & "
            f"{all_decision[f'threshold_{suffix}_forward_only']} \\\\",
        ])
    (PAPER / "table_decision_impact.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[!t]", r"\centering", r"\settableformat",
        r"\caption{Observed participant-specific decoder ordering and threshold crossings ($n=138$). Arrows go from LOSO to expanding-window ordering. Bilateral margins require both absolute TS+LR-minus-CSP+LDA mean AUC differences to exceed the threshold. Threshold crossings average both decoders and targets within participant.}",
        r"\label{tab:decisions}",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}X S[table-format=3.0] S[table-format=3.0]@{}}",
        r"\toprule", r"\textbf{Outcome} & {\textbf{Conditional (primary)}} & {\textbf{All origins}} \\", r"\midrule",
        *decision_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ]))

    support_rows = []
    conditional_support = wide[~wide["is_structural_zero"]]
    strict_support = conditional_support.dropna(subset=["future_matched", "loso_matched"])
    target_keys = ["dataset", "subject", "target_session"]
    for dataset in DATASETS:
        groups = [frame[frame["dataset"] == dataset] for frame in
                  (wide, conditional_support, strict_support)]
        counts = [len(group[target_keys].drop_duplicates()) for group in groups]
        label = dataset.replace("_", r"\_")
        support_rows.append(
            f"{label} & {groups[0]['subject_uid'].nunique()} & "
            f"{counts[0]} & {counts[1]} & {counts[2]} \\\\"
        )
    totals = [len(frame[target_keys].drop_duplicates()) for frame in
              (wide, conditional_support, strict_support)]
    means = [frame["target_session"].mean() for frame in
             (wide, conditional_support, strict_support)]
    support_rows.extend([
        r"\midrule",
        f"Total & {wide['subject_uid'].nunique()} & {totals[0]} & {totals[1]} & {totals[2]} \\\\",
        r"\shortstack[l]{Mean donor-\\session count} & {---} & "
        f"\\multicolumn{{1}}{{r}}{{{means[0]:.2f}}} & "
        f"\\multicolumn{{1}}{{r}}{{{means[1]:.2f}}} & "
        f"\\multicolumn{{1}}{{r}}{{{means[2]:.2f}}} \\\\",
    ])
    (PAPER / "table_support.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[t]", r"\centering", r"\settableformat",
        r"\caption{Observed support by dataset: $n$ is the number of participants; All origins, Cond.\ (conditional), and Strict future count participant--targets shared by the two decoders. The session-count-matched reference uses conditional support; the future-only arm requires strict-future support, $t\leq(T-1)/2$. Mean donor-session count is the target-weighted expanding-window count $t$.}",
        r"\label{tab:support}",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}X S[table-format=3.0] S[table-format=4.0] S[table-format=4.0] S[table-format=3.0]@{}}",
        r"\toprule",
        r"\textbf{Dataset} & {$\boldsymbol{n}$} & \multicolumn{3}{c}{\textbf{Shared participant--targets}} \\",
        r"\cmidrule(l){3-5}",
        r" & & {\shortstack{\textbf{All}\\\textbf{origins}}} & {\textbf{Cond.}} & {\shortstack{\textbf{Strict}\\\textbf{future}}} \\",
        r"\midrule", *support_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ]))

    contrast_rows = []
    labels = {
        "total": "LOSO $-$ expanding window",
        "quantity_uniform": "LOSO $-$ session-count-matched reference",
        "composition_uniform": "Session-count-matched reference $-$ expanding window",
        "strict_future": "Strict future $-$ expanding window",
    }
    for support_key, support_label in (("primary_conditional", "Later donor available: 1,001 participant--targets"),
                                       ("strict_future", "Strict-future common support: 560 participant--targets")):
        if contrast_rows:
            contrast_rows.append(r"\addlinespace[0.5em]")
        contrast_rows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{support_label}}}}} \\")
        contrast_rows.append(r"\addlinespace[0.2em]")
        for key, record in common_contrasts[support_key].items():
            contrast_rows.append(
                f"{labels[key]} & {record['n']} & {record['mean']:+.3f} & "
                f"$[{record['ci'][0]:+.3f}, {record['ci'][1]:+.3f}]$ \\\\"
            )
    (PAPER / "table_common_support.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table*}[!t]", r"\centering", r"\settableformat",
        r"\caption{Partition of the LOSO minus expanding-window difference under the session-count-matched reference (Equation~\eqref{eq:partition}), on the full conditional support and on the strict-future support required by the future-only arm. The two components sum to the total within each support, up to rounding. All intervals are participant-level $t$ intervals; those involving a sampled arm condition on the five realized donor draws (Appendix~\ref{app:uncertainty}).}",
        r"\label{tab:common-support}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X S[table-format=3.0] S[table-format=+1.3,retain-explicit-plus=true] r@{}}",
        r"\toprule",
        r"\textbf{Contrast} & {\textbf{Participants}} & {\textbf{Mean $\boldsymbol{\Delta}$}} & \textbf{95\% CI} \\", r"\midrule",
        *contrast_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}", "",
    ]))

    time_rows = [
        r"BNCI2014\_004 & Session & Five sessions; screening then feedback & 120--160 & Exact intervals unavailable \\",
        r"Zhou2016 & Session & Three sessions & 90--119 & Several days to several months \\",
        r"Zhou2020 & Session & Six or seven sessions; about 2-day spacing over 2 weeks & 100--180 & Participant 20 has six sessions \\",
        r"Kumar2024 & Session & Six separate days & 60--80 & Calibration followed by feedback \\",
        r"Ma2020 & Block/day & Fifteen blocks; five per day on three days & 40/block & Both units analyzed \\",
        r"Stieger2021 & Day & Seven or 11 recorded days & 21--190 & Left/right subset retained \\",
    ]
    (PAPER / "table_time_structure.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[H]", r"\centering", r"\scriptsize",
        r"\caption{Time units represented by the released dataset labels. Session-index lag is interpreted within dataset and is not a common elapsed-time scale.}",
        r"\label{tab:time}", r"\begin{tabularx}{\linewidth}{llXrX}", r"\toprule",
        r"Dataset & Unit & Recording schedule & Trials/unit & Interval information \\", r"\midrule",
        *time_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ]))

    sensitivity_rows = [
        r"\multicolumn{3}{@{}l}{\textit{Alternative policies (conditional targets): LOSO minus}} \\",
    ]
    for label, column in (("Immediately previous session", "delta_last"),
                          ("First session", "delta_first")):
        p = wide[~wide["is_structural_zero"]].groupby("subject_uid")[column].mean()
        s = t_summary(p)
        sensitivity_rows.append(
            f"{label} & {s['n']} & ${s['mean']:+.3f}\\;"
            f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]$ \\\\"
        )
    if not ma_frame.empty:
        for unit, unit_label in (("block", "blocks"), ("day", "days")):
            sensitivity_rows.extend([
                r"\addlinespace[0.4em]",
                rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{Ma2020: {unit_label}}}}} \\",
            ])
            for estimand, label in (("all_origin", "All origins"),
                                   ("conditional", "Conditional")):
                row = ma_frame[(ma_frame["time_unit"] == unit) &
                               (ma_frame["metric"] == "auc") &
                               (ma_frame["estimand"] == estimand)].iloc[0]
                sensitivity_rows.append(
                    f"{label} & {int(row['n'])} & ${row['mean']:+.3f}\\;"
                    f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]$ \\\\"
                )
    (PAPER / "table_sensitivity.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[!t]", r"\centering", r"\settableformat",
        r"\caption{Alternative training policies and Ma2020 time-unit sensitivity.}",
        r"\label{tab:sensitivity}", r"\par\smallskip",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}X S[table-format=3.0] r@{}}",
        r"\toprule", r"\textbf{Analysis} & {$\boldsymbol{n}$} & \shortstack[r]{\textbf{Mean $\boldsymbol{\Delta}$}\\\textbf{95\% CI}} \\", r"\midrule",
        *sensitivity_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ]))

    macros = {
        "CondGapMean": fmt(cond["participant_weighted"]["mean"], signed=True),
        "CondGapLo": fmt(cond["participant_weighted"]["ci"][0], signed=True),
        "CondGapHi": fmt(cond["participant_weighted"]["ci"][1], signed=True),
        "AllGapMean": fmt(all_est["participant_weighted"]["mean"], signed=True),
        "AllGapLo": fmt(all_est["participant_weighted"]["ci"][0], signed=True),
        "AllGapHi": fmt(all_est["participant_weighted"]["ci"][1], signed=True),
        "CondTargetMean": fmt(cond["target_weighted"]["mean"], signed=True),
        "AllTargetMean": fmt(all_est["target_weighted"]["mean"], signed=True),
        "DatasetMean": fmt(cond["dataset_weighted"]["mean"], signed=True),
        "MetaMean": fmt(cond["random_effects"]["estimate"], signed=True),
        "MetaLo": fmt(cond["random_effects"]["ci"][0], signed=True),
        "MetaHi": fmt(cond["random_effects"]["ci"][1], signed=True),
        "MetaPredLo": fmt(cond["random_effects"]["prediction_interval"][0], signed=True),
        "MetaPredHi": fmt(cond["random_effects"]["prediction_interval"][1], signed=True),
        "MetaI": f"{100*cond['random_effects']['i2']:.1f}",
        "AllDatasetMean": fmt(all_est["dataset_weighted"]["mean"], signed=True),
        "AllMetaMean": fmt(all_est["random_effects"]["estimate"], signed=True),
        "AllMetaLo": fmt(all_est["random_effects"]["ci"][0], signed=True),
        "AllMetaHi": fmt(all_est["random_effects"]["ci"][1], signed=True),
        "AllMetaPredLo": fmt(all_est["random_effects"]["prediction_interval"][0], signed=True),
        "AllMetaPredHi": fmt(all_est["random_effects"]["prediction_interval"][1], signed=True),
        "AllMetaI": f"{100*all_est['random_effects']['i2']:.1f}",
        "CondMedian": fmt(cond["participant_weighted"]["median"], signed=True),
        "CondIQRLo": fmt(cond["participant_weighted"]["q1"], signed=True),
        "CondIQRHi": fmt(cond["participant_weighted"]["q3"], signed=True),
        "AllMedian": fmt(all_est["participant_weighted"]["median"], signed=True),
        "AllIQRLo": fmt(all_est["participant_weighted"]["q1"], signed=True),
        "AllIQRHi": fmt(all_est["participant_weighted"]["q3"], signed=True),
        "CondPositive": f"{100*cond['participant_weighted']['prob_gt_0']:.1f}",
        "CondGtOne": f"{100*cond['participant_weighted']['prob_gt_001']:.1f}",
        "CondGtTwo": f"{100*cond['participant_weighted']['prob_gt_002']:.1f}",
        "AllPositive": f"{100*all_est['participant_weighted']['prob_gt_0']:.1f}",
        "AllGtOne": f"{100*all_est['participant_weighted']['prob_gt_001']:.1f}",
        "AllGtTwo": f"{100*all_est['participant_weighted']['prob_gt_002']:.1f}",
        "CSPGapMean": fmt(summary["practical"]["by_decoder"]["csp_lda"]["mean"], signed=True),
        "CSPGapLo": fmt(summary["practical"]["by_decoder"]["csp_lda"]["ci"][0], signed=True),
        "CSPGapHi": fmt(summary["practical"]["by_decoder"]["csp_lda"]["ci"][1], signed=True),
        "TSGapMean": fmt(summary["practical"]["by_decoder"]["ts_lr"]["mean"], signed=True),
        "TSGapLo": fmt(summary["practical"]["by_decoder"]["ts_lr"]["ci"][0], signed=True),
        "TSGapHi": fmt(summary["practical"]["by_decoder"]["ts_lr"]["ci"][1], signed=True),
        "AllCSPGapMean": fmt(summary["practical"]["by_estimand"]["all_origin"]["by_decoder"]["csp_lda"]["mean"], signed=True),
        "AllCSPGapLo": fmt(summary["practical"]["by_estimand"]["all_origin"]["by_decoder"]["csp_lda"]["ci"][0], signed=True),
        "AllCSPGapHi": fmt(summary["practical"]["by_estimand"]["all_origin"]["by_decoder"]["csp_lda"]["ci"][1], signed=True),
        "AllTSGapMean": fmt(summary["practical"]["by_estimand"]["all_origin"]["by_decoder"]["ts_lr"]["mean"], signed=True),
        "AllTSGapLo": fmt(summary["practical"]["by_estimand"]["all_origin"]["by_decoder"]["ts_lr"]["ci"][0], signed=True),
        "AllTSGapHi": fmt(summary["practical"]["by_estimand"]["all_origin"]["by_decoder"]["ts_lr"]["ci"][1], signed=True),
        "RankChangeN": str(summary["practical"]["model_selection_changes"]),
        "CondRankChangeN": str(summary["practical"]["model_selection_changes"]),
        "RankChangePct": f"{summary['practical']['model_selection_change_percent']:.1f}",
        "ThresholdSeventyN": str(summary["practical"]["threshold_070_loso_only"]),
        "ThresholdSeventyFiveN": str(summary["practical"]["threshold_075_loso_only"]),
        "AllRankChangeN": str(summary["practical"]["by_estimand"]["all_origin"]["model_selection_changes"]),
        "AllRankChangePct": f"{summary['practical']['by_estimand']['all_origin']['model_selection_change_percent']:.1f}",
        "AllThresholdSeventyN": str(summary["practical"]["by_estimand"]["all_origin"]["threshold_070_loso_only"]),
        "AllThresholdSeventyFiveN": str(summary["practical"]["by_estimand"]["all_origin"]["threshold_075_loso_only"]),
        "DriftRobustBeta": fmt(summary["drift_influence"]["drift_beta_clustered"], 4, True),
        "DriftRobustLo": fmt(summary["drift_influence"]["drift_ci_clustered"][0], 4, True),
        "DriftRobustHi": fmt(summary["drift_influence"]["drift_ci_clustered"][1], 4, True),
        "DriftRobustP": pvalue(summary["drift_influence"]["drift_p_clustered"]),
        "DriftInfluentialN": str(summary["drift_influence"]["n_above_cooks_cutoff"]),
        "CondMatchedTriangleSE": fmt(summary["matched_monte_carlo"]["by_support"]["conditional"]["triangle_se_bound_plugin"], 4),
        "StrictMatchedTriangleSE": fmt(summary["matched_monte_carlo"]["by_support"]["strict_future"]["triangle_se_bound_plugin"], 4),
        "CondMatchedRowQSE": fmt(summary["matched_monte_carlo"]["by_support"]["conditional"]["q95_marginal_se"], 4),
        "StrictMatchedRowQSE": fmt(summary["matched_monte_carlo"]["by_support"]["strict_future"]["q95_marginal_se"], 4),
    }
    for support_name, prefix in (("primary_conditional", "Uniform"), ("strict_future", "Strict")):
        for field, label in (("total", "Total"), ("quantity_uniform", "Count"), ("composition_uniform", "Composition")):
            record = common_contrasts[support_name][field]
            for stat, value in (("Mean", record["mean"]), ("Lo", record["ci"][0]), ("Hi", record["ci"][1])):
                macros[prefix + label + stat] = fmt(value, signed=True)
    for key, prefix in (("all_origin", "All"), ("conditional", "Cond")):
        impact = summary["practical"]["by_estimand"][key]
        macros[prefix + "FlipMarginOneN"] = str(impact["bilateral_margin_sensitivity"][2]["n"])
        macros[prefix + "FlipCSPtoTSN"] = str(impact["csp_to_ts"])
        macros[prefix + "FlipTStoCSPN"] = str(impact["ts_to_csp"])
        hk = summary["estimands"][key]["random_effects"]["hksj_sensitivity"]["ci"]
        macros[prefix + "MetaHKLo"] = fmt(hk[0], signed=True)
        macros[prefix + "MetaHKHi"] = fmt(hk[1], signed=True)
        exclusions = summary["leave_one_dataset_out"]["by_estimand"][key]
        macros["LoDO" + prefix + "Min"] = fmt(min(row["mean"] for row in exclusions), 3, True)
        macros["LoDO" + prefix + "Max"] = fmt(max(row["mean"] for row in exclusions), 3, True)
        without_stieger = next(row for row in exclusions if row["excluded_dataset"] == "Stieger2021")
        macros["LoDOWithoutStieger" + prefix] = fmt(without_stieger["mean"], 3, True)
    future_record = common_contrasts["strict_future"]["strict_future"]
    for stat, value in (("Mean", future_record["mean"]), ("Lo", future_record["ci"][0]), ("Hi", future_record["ci"][1])):
        macros["StrictFuture" + stat] = fmt(value, signed=True)
    if not ma_frame.empty:
        for unit, prefix in (("block", "MaBlock"), ("day", "MaDay")):
            for estimand, suffix in (("conditional", "Cond"), ("all_origin", "All")):
                row = ma_frame[(ma_frame["time_unit"] == unit) &
                               (ma_frame["metric"] == "auc") &
                               (ma_frame["estimand"] == estimand)].iloc[0]
                macros[prefix + suffix] = fmt(row["mean"], signed=True)
        ba = ma_frame[(ma_frame["time_unit"] == "day") &
                      (ma_frame["metric"] == "balanced_accuracy") &
                      (ma_frame["estimand"] == "conditional")].iloc[0]
        macros["MaDayBA"] = fmt(ba["mean"], signed=True)
        macros["MaDayBALo"] = fmt(ba["ci"][0], signed=True)
        macros["MaDayBAHi"] = fmt(ba["ci"][1], signed=True)

    # Unsigned variants for sentences whose direction is stated in words
    # ("scored 0.025 AUC above ..."); negative values keep their minus sign.
    for key in ("CondGapMean", "UniformCountMean", "UniformCountLo", "UniformCountHi",
                "UniformCompositionMean", "UniformCompositionLo", "UniformCompositionHi",
                "StrictFutureMean", "StrictFutureLo", "StrictFutureHi"):
        value = macros[key]
        macros[key + "Unsigned"] = value[1:] if value.startswith("+") else value

    (PAPER / "revision_results.tex").write_text(
        "% Generated by scripts/revision_analysis.py; do not edit.\n" +
        "\n".join(rf"\newcommand{{\{key}}}{{{text_minus(value)}}}" for key, value in macros.items()) +
        "\n"
    )


def main():
    wide = pd.read_csv(RESULTS / "gap_wide.csv")
    wide = wide[wide["model"].isin(CLASSICAL)].copy()
    long_df = pd.read_csv(RESULTS / "auc_long.csv")
    conditional_rows = wide[~wide["is_structural_zero"]].copy()
    all_rows = wide.copy()
    exclusions = {
        "all_origin": leave_one_dataset_out(all_rows),
        "conditional": leave_one_dataset_out(conditional_rows),
    }
    pd.DataFrame([
        {"estimand": estimand, **record}
        for estimand, records in exclusions.items() for record in records
    ]).to_csv(RESULTS / "leave_one_dataset_out.csv", index=False)

    # The final-origin contrast must be exactly zero, not merely close to zero.
    zeros = all_rows[all_rows["is_structural_zero"]]["delta"]
    if not np.allclose(zeros, 0.0, atol=1e-12):
        raise RuntimeError("final-origin structural zeros failed numerical verification")

    cond_participant, cond_bundle, cond_meta = estimate_bundle(conditional_rows)
    all_participant, all_bundle, all_meta = estimate_bundle(all_rows)
    cond_meta["estimand"] = "conditional"
    all_meta["estimand"] = "all_origin"
    meta_rows = pd.concat([cond_meta, all_meta], ignore_index=True)

    impacts = []
    for label, frame in (("conditional", cond_participant), ("all_origin", all_participant)):
        x = frame.copy()
        x["estimand"] = label
        impacts.append(x)
    impacts = pd.concat(impacts, ignore_index=True)
    impacts.to_csv(RESULTS / "participant_impacts.csv", index=False)
    meta_rows.to_csv(RESULTS / "dataset_meta.csv", index=False)

    support_rows = []
    support_rows.append(support_record("Conditional LOSO - expanding", conditional_rows))
    support_rows.append(support_record("All origins LOSO - expanding", all_rows))
    uniform = conditional_rows.dropna(subset=["loso_matched"])
    support_rows.append(support_record("Uniform donor-count-matched", uniform))
    strict = conditional_rows.dropna(subset=["future_matched", "loso_matched"])
    support_rows.append(support_record("Strict future common support", strict))
    support = pd.DataFrame(support_rows)
    support.to_csv(RESULTS / "contrast_support.csv", index=False)

    inclusion = wide.groupby(["dataset", "subject"], as_index=False).agg(
        T=("T", "first"),
        n_models=("model", "nunique"),
        all_targets=("target_session", "nunique"),
        conditional_targets=("is_structural_zero", lambda x: int((~x).sum() / len(CLASSICAL))),
    )
    inclusion["expected_models"] = len(CLASSICAL)
    inclusion["complete"] = (
        (inclusion["n_models"] == len(CLASSICAL)) &
        (inclusion["all_targets"] == inclusion["T"] - 1) &
        (inclusion["conditional_targets"] == inclusion["T"] - 2)
    )
    inclusion.to_csv(RESULTS / "inclusion_manifest.csv", index=False)

    common_contrasts = {}
    for name, frame in (("primary_conditional", uniform), ("strict_future", strict)):
        common_contrasts[name] = {
            "total": contrast_summary(frame, "loso", "forward_expanding"),
            "quantity_uniform": contrast_summary(frame, "loso", "loso_matched"),
            "composition_uniform": contrast_summary(frame, "loso_matched", "forward_expanding"),
        }
        components = common_contrasts[name]
        if not np.isclose(components["total"]["mean"],
                          components["quantity_uniform"]["mean"] + components["composition_uniform"]["mean"]):
            raise AssertionError(f"Decomposition is not additive on {name}")
    common_contrasts["strict_future"]["strict_future"] = contrast_summary(strict, "future_matched", "forward_expanding")

    conditional_impact = operational_impact(conditional_rows)
    all_origin_impact = operational_impact(all_rows)
    # Preserve the previous conditional keys while exposing both estimands explicitly.
    practical = dict(conditional_impact)
    practical.update({
        "by_estimand": {
            "conditional": conditional_impact,
            "all_origin": all_origin_impact,
        },
        "negative_by_dataset_conditional": {
            dataset: float(np.mean(frame["delta"] < 0))
            for dataset, frame in cond_participant.groupby("dataset")
        },
        "negative_by_dataset_all_origin": {
            dataset: float(np.mean(frame["delta"] < 0))
            for dataset, frame in all_participant.groupby("dataset")
        },
    })

    ma_dict, ma_frame = ma_day_summary(wide)
    figure_protocol_logic()
    forest = figure_forest(conditional_rows)
    figure_estimands(
        {"conditional": cond_bundle, "all_origin": all_bundle},
        cond_participant, all_participant,
        common_contrasts=common_contrasts,
    )
    figure_history(conditional_rows)
    figure_drift(conditional_rows)
    figure_ma_sensitivity(ma_frame)
    figure_decoder_margins(all_rows, conditional_rows)
    margins_output = []
    dataset_ordering_output = []
    for estimand, frame in (("all_origin", all_rows), ("conditional", conditional_rows)):
        margins = decoder_margin_frame(frame).reset_index()
        margins["estimand"] = estimand
        margins_output.append(margins)
        dataset_means = (
            frame.groupby(["dataset", "subject_uid", "model"])[["loso", "forward_expanding"]].mean()
            .groupby(["dataset", "model"]).mean().unstack("model")
        )
        dataset_ordering = pd.DataFrame({
            "loso_margin_ts_minus_csp": dataset_means["loso"]["ts_lr"] - dataset_means["loso"]["csp_lda"],
            "expanding_margin_ts_minus_csp": dataset_means["forward_expanding"]["ts_lr"] - dataset_means["forward_expanding"]["csp_lda"],
        }).reset_index()
        dataset_ordering["estimand"] = estimand
        dataset_ordering_output.append(dataset_ordering)
    pd.concat(margins_output, ignore_index=True).to_csv(RESULTS / "decoder_margins.csv", index=False)
    pd.concat(dataset_ordering_output, ignore_index=True).to_csv(RESULTS / "dataset_decoder_margins.csv", index=False)

    mc_supports = {"conditional": uniform, "strict_future": strict}
    trial_discrepancies = matched_trial_summary(long_df, mc_supports)
    pd.DataFrame(trial_discrepancies).to_csv(RESULTS / "matched_trial_counts.csv", index=False)

    summary = {
        "estimands": {"conditional": cond_bundle, "all_origin": all_bundle},
        "practical": practical,
        "common_support_contrasts": common_contrasts,
        "matched_monte_carlo": monte_carlo_summary(long_df, mc_supports),
        "matched_trial_count_discrepancies": trial_discrepancies,
        "drift_influence": drift_influence(conditional_rows),
        "ma2020_time_unit": ma_dict,
        "structural_zero_rows": int(len(zeros)),
        "leave_one_dataset_out": {
            "weighting": "Equal remaining participants; equal available target-decoder rows within each participant",
            "inference": "Point-only dataset-exclusion sensitivity; no confidence interval or new-dataset prediction claim",
            "by_estimand": exclusions,
        },
    }
    (RESULTS / "revision_summary.json").write_text(json.dumps(summary, indent=2))
    forest.to_csv(RESULTS / "gap_forest.csv", index=False)
    write_tables(summary, cond_participant, all_participant, meta_rows,
                 support, common_contrasts, forest, ma_frame, wide)
    print(json.dumps({
        "conditional": cond_bundle["participant_weighted"],
        "all_origin": all_bundle["participant_weighted"],
        "random_effects": cond_bundle["random_effects"],
        "practical": practical,
        "ma2020": ma_dict,
    }, indent=2))


if __name__ == "__main__":
    main()

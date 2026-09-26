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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PAPER = ROOT / "paper"
FIGSIZE = (9.0, 4.8)
CLASSICAL = ("csp_lda", "ts_lr")
MODEL_LABEL = {"csp_lda": "CSP+LDA", "ts_lr": "TS+LR"}
DATASETS = (
    "BNCI2014_004", "Zhou2016", "Zhou2020", "Kumar2024", "Ma2020",
    "Stieger2021",
)
COLORS = {"csp_lda": "#4477AA", "ts_lr": "#EE7733"}

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save_figure(fig, stem, bottom=0.0):
    fig.set_size_inches(*FIGSIZE, forward=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    for suffix, kwargs in (
        ("svg", {"format": "svg"}),
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 180}),
    ):
        fig.savefig(FIGURES / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def figure_protocol_logic():
    """Show the single information-set contrast that organizes the manuscript."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    colors = {
        "past": "#3E6D9C", "target": "#B23A7A", "future": "#E8792B",
        "inactive_fill": "#F4F5F6", "inactive_edge": "#C6CBD1",
        "ink": "#202428", "muted": "#60676E", "rule": "#D8DCE0",
        "primary_fill": "#EEF4FA", "support_fill": "#FDF3EB",
    }

    ax.text(0.015, 0.955, "(a) Donor availability at the same target",
            ha="left", va="top", fontsize=13, weight="bold", color=colors["ink"])
    ax.text(0.48, 0.845, "training donors", ha="center", va="center",
            fontsize=10.5, color=colors["muted"])
    ax.text(0.88, 0.845, "test target", ha="center", va="center",
            fontsize=10.5, color=colors["muted"])
    ax.text(0.37, 0.805, "prior sessions", ha="center", va="center",
            fontsize=9.8, color=colors["muted"])
    ax.text(0.65, 0.805, "later donor", ha="center", va="center",
            fontsize=9.8, color=colors["future"])

    chip_width, chip_height = 0.082, 0.105
    past_x = (0.27, 0.37, 0.47)
    future_x, target_x = 0.65, 0.88

    def chip(xpos, ypos, label, role):
        if role == "past":
            face, edge, text_color, linewidth = colors["past"], colors["past"], "white", 1.4
        elif role == "future":
            face, edge, text_color, linewidth = colors["future"], colors["future"], "white", 1.4
        elif role == "target":
            face, edge, text_color, linewidth = "white", colors["target"], colors["ink"], 2.0
        else:
            face, edge, text_color, linewidth = (
                colors["inactive_fill"], colors["inactive_edge"], colors["muted"], 1.2
            )
        box = FancyBboxPatch(
            (xpos - chip_width / 2, ypos - chip_height / 2), chip_width, chip_height,
            boxstyle="round,pad=0.005,rounding_size=0.013",
            linewidth=linewidth, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)
        if role == "inactive":
            box.set_linestyle("--")
        ax.text(xpos, ypos, label, ha="center", va="center", fontsize=11.5,
                color=text_color, weight="bold" if role == "target" else "normal")

    def protocol_row(ypos, name, interpretation, include_future):
        ax.text(0.015, ypos + 0.018, name, ha="left", va="center",
                fontsize=12.2 if include_future else 11.0,
                weight="bold", color=colors["ink"])
        ax.text(0.015, ypos - 0.038, interpretation, ha="left", va="center",
                fontsize=9.8, color=colors["muted"])
        for index, xpos in enumerate(past_x):
            chip(xpos, ypos, f"S{index}", "past")
        chip(future_x, ypos, "S4", "future" if include_future else "inactive")
        if not include_future:
            ax.text(future_x, ypos - 0.090, "unavailable at $t$", ha="center",
                    va="center", fontsize=8.7, color=colors["muted"])
            ax.plot([0.585, 0.585], [ypos - 0.07, ypos + 0.07],
                    color=colors["inactive_edge"], lw=1.1, ls="--")
        ax.add_patch(FancyArrowPatch(
            (0.72, ypos), (0.81, ypos), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.25, color=colors["muted"],
        ))
        chip(target_x, ypos, "S3", "target")
        ax.text(target_x, ypos - 0.083, r"$t=3$", ha="center", va="center",
                fontsize=9.5, color=colors["target"])

    protocol_row(0.675, "LOSO", "completed trajectory", True)
    protocol_row(0.48, "Expanding window", "history available at $t$", False)

    ax.text(0.525, 0.345,
            r"paired comparison: same test trials and decoder; $\Delta=\mathrm{AUC}_{LOSO}-\mathrm{AUC}_{EXP}$",
            ha="center", va="center", fontsize=10.5, color=colors["ink"])
    ax.plot([0.015, 0.985], [0.305, 0.305], color=colors["rule"], lw=1.1)

    ax.text(0.015, 0.275, "(b) Analysis populations", ha="left", va="top",
            fontsize=13, weight="bold", color=colors["ink"])

    def estimand_card(x0, width, fill, accent, eyebrow, title, formula, note):
        card = FancyBboxPatch(
            (x0, 0.035), width, 0.185,
            boxstyle="round,pad=0.009,rounding_size=0.014",
            linewidth=0.8, edgecolor=colors["rule"], facecolor=fill,
        )
        ax.add_patch(card)
        ax.plot([x0 + 0.012, x0 + 0.012], [0.057, 0.198], color=accent,
                lw=4.0, solid_capstyle="round")
        ax.text(x0 + 0.035, 0.185, eyebrow, ha="left", va="center",
                fontsize=9.6, weight="bold", color=colors["muted"])
        ax.text(x0 + 0.035, 0.142, title, ha="left", va="center",
                fontsize=12.2, weight="bold", color=colors["ink"])
        ax.text(x0 + 0.035, 0.094, formula, ha="left", va="center",
                fontsize=11.2, color=colors["ink"])
        ax.text(x0 + width - 0.025, 0.094, note, ha="right", va="center",
                fontsize=9.8, color=colors["muted"])

    estimand_card(
        0.02, 0.465, colors["primary_fill"], colors["past"], "PRIMARY",
        "All prospective origins", r"$t=1,\ldots,T-1$", r"final origin: $\Delta=0$",
    )
    estimand_card(
        0.515, 0.465, colors["support_fill"], colors["future"], "SUPPORTING",
        "Later donor available", r"$t=1,\ldots,T-2$", "conditional summary",
    )
    save_figure(fig, "fig0_protocol_logic")


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
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for ax, (name, rows) in zip(axes, (("(a) All origins", all_rows), ("(b) Later donor available", conditional_rows))):
        margins = decoder_margin_frame(rows)
        flip = margins["flip"]
        ax.axhspan(-0.01, 0.01, color="#E5E7EB", alpha=0.55, zorder=0)
        ax.axvspan(-0.01, 0.01, color="#E5E7EB", alpha=0.55, zorder=0)
        ax.axhline(0, color="#60676E", lw=0.7)
        ax.axvline(0, color="#60676E", lw=0.7)
        for mask, color, label in ((~flip, "#4477AA", "Same observed ordering"), (flip, "#CC6677", "Opposite observed ordering")):
            ax.scatter(margins.loc[mask, "loso_margin"], margins.loc[mask, "expanding_margin"],
                       s=23, alpha=0.75, c=color, edgecolors="white", linewidths=0.35, label=label)
        ax.set_title(name, loc="left")
        ax.set_xlabel("LOSO decoder difference (TS+LR − CSP+LDA)")
        ax.text(0.035, 0.965, f"Opposite ordering: {int(flip.sum())}/138",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Expanding-window decoder difference")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.text(0.51, 0.015, "Shaded bands: absolute decoder difference ≤ 0.01 AUC; positive differences favor TS+LR.",
             ha="center", va="bottom", fontsize=9)
    save_figure(fig, "fig6_decoder_margins", bottom=0.075)


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
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ybase = np.arange(len(DATASETS))[::-1]
    offsets = {"csp_lda": 0.13, "ts_lr": -0.13}
    for model in CLASSICAL:
        sub = d[d["model"] == model].set_index("dataset").reindex(DATASETS)
        y = ybase + offsets[model]
        ax.errorbar(
            sub["mean"], y,
            xerr=[sub["mean"] - sub["ci"].map(lambda x: x[0]),
                  sub["ci"].map(lambda x: x[1]) - sub["mean"]],
            fmt="o", capsize=3, color=COLORS[model], label=MODEL_LABEL[model],
        )
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(ybase)
    ax.set_yticklabels(DATASETS)
    ax.set_xlabel("all-origin mean AUC difference (LOSO - expanding window)")
    ax.set_title("Primary protocol difference by dataset and decoder (95% CI)")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "fig1_gap_forest")
    return d


def figure_estimands(bundle, cond_participant, all_participant):
    labels = [
        "all origins\nparticipant-weighted",
        "all origins\ntarget-weighted",
        "conditional\nparticipant-weighted",
        "conditional\ntarget-weighted",
    ]
    records = [
        bundle["all_origin"]["participant_weighted"],
        bundle["all_origin"]["target_weighted"],
        bundle["conditional"]["participant_weighted"],
        bundle["conditional"]["target_weighted"],
    ]
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
    y = np.arange(4)[::-1]
    means = np.array([r["mean"] for r in records])
    lows = np.array([r["ci"][0] for r in records])
    highs = np.array([r["ci"][1] for r in records])
    ax[0].errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", capsize=3,
                   color="#AA3377")
    ax[0].axvline(0, color="black", lw=0.9)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("mean AUC difference")
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax[0].set_title("(a) Estimand and weighting")

    bins = np.linspace(
        min(cond_participant["delta"].min(), all_participant["delta"].min()),
        max(cond_participant["delta"].max(), all_participant["delta"].max()), 22,
    )
    ax[1].hist(all_participant["delta"], bins=bins, histtype="step", lw=2,
               color="#4477AA", label="all origins (primary)")
    ax[1].hist(cond_participant["delta"], bins=bins, histtype="step", lw=2,
               color="#EE7733", label="conditional")
    ax[1].axvline(0, color="black", lw=0.9)
    ax[1].axvline(0.01, color="#777777", lw=0.8, ls="--")
    ax[1].axvline(0.02, color="#777777", lw=0.8, ls=":")
    ax[1].set_xlabel("participant-level mean AUC difference")
    ax[1].set_ylabel("participants")
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax[1].set_title("(b) Participant distribution")
    ax[1].legend(frameon=False)
    save_figure(fig, "fig2_estimands_distribution")


def figure_history(inner):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
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
                    color="#AA3377", ms=4)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("future sessions")
        xmin = min(1.0, float(x.min())) - 0.5
        xmax = float(x.max()) + 0.5
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(math.ceil(xmin), math.floor(xmax) + 1))
    axes[0, 0].set_ylabel("mean gap")
    axes[1, 0].set_ylabel("mean gap")
    save_figure(fig, "fig3_history_by_dataset")


def figure_drift(inner):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=False, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        d = inner[(inner["dataset"] == dataset)].dropna(subset=["drift_z", "delta"])
        if len(d) >= 100:
            ax.hexbin(d["drift_z"], d["delta"], gridsize=18, mincnt=1,
                      cmap="Blues", linewidths=0)
        else:
            ax.scatter(d["drift_z"], d["delta"], s=10, alpha=0.45, color="#4477AA")
        if d["drift_z"].nunique() > 1:
            slope, intercept, _, _ = stats.theilslopes(d["delta"], d["drift_z"])
            x = np.linspace(d["drift_z"].min(), d["drift_z"].max(), 30)
            ax.plot(x, intercept + slope * x, color="#CC3311", lw=1.2)
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
    order = [
        ("block", "conditional"), ("block", "all_origin"),
        ("day", "conditional"), ("day", "all_origin"),
    ]
    labels = ["15 blocks\nconditional", "15 blocks\nall origins",
              "3 days\nconditional", "3 days\nall origins"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(order))
    for j, key in enumerate(order):
        row = d[(d["time_unit"] == key[0]) & (d["estimand"] == key[1])].iloc[0]
        ax.errorbar(j, row["mean"],
                    yerr=[[row["mean"] - row["ci"][0]], [row["ci"][1] - row["mean"]]],
                    fmt="o", capsize=4, color="#AA3377")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean AUC difference")
    ax.set_title("Ma2020 time-unit sensitivity (participant-level 95% CI)")
    save_figure(fig, "fig5_ma_time_unit")


def fmt(value, digits=3, signed=False):
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def pvalue(value):
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        exponent = math.floor(math.log10(value))
        mantissa = value / (10 ** exponent)
        return rf"\ensuremath{{{mantissa:.1f}\times10^{{{exponent}}}}}"
    return f"{value:.4f}"


def write_tables(summary, cond_participant, all_participant, meta_rows,
                 support, common_contrasts, forest, ma_frame, wide):
    cond = summary["estimands"]["conditional"]
    all_est = summary["estimands"]["all_origin"]
    rows = []
    for label, record in (
        ("All origins, participant-weighted", all_est["participant_weighted"]),
        ("All origins, target-weighted", all_est["target_weighted"]),
        ("All origins, dataset-weighted", all_est["dataset_weighted"]),
        ("All origins, random-effects", all_est["random_effects"]),
        ("Conditional, participant-weighted", cond["participant_weighted"]),
        ("Conditional, target-weighted", cond["target_weighted"]),
        ("Conditional, dataset-weighted", cond["dataset_weighted"]),
        ("Conditional, random-effects", cond["random_effects"]),
    ):
        n = record.get("n", record.get("n_subjects", record.get("k", 6)))
        rows.append(
            f"{label} & {n} & {record['mean'] if 'mean' in record else record['estimate']:+.3f} "
            f"[{record['ci'][0]:+.3f}, {record['ci'][1]:+.3f}] \\\\"
        )
    (PAPER / "table_estimands.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table*}[!t]", r"\centering", r"\settableformat",
        r"\caption{LOSO minus expanding-window AUC under the primary all-origin estimand and the conditional summary. The all-origin estimand includes the final structural zero. Conditional estimates require at least one later session. Target-weighted intervals use a participant-cluster bootstrap.}",
        r"\label{tab:estimands}",
        r"\begin{tabularx}{0.80\textwidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}",
        r"\toprule",
        r"Estimand and aggregation & Units & Mean difference [95\% CI] \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}", "",
    ]))

    all_practical = t_summary(all_participant["delta"])
    cond_practical = t_summary(cond_participant["delta"])
    practical_rows = [
        f"Median [IQR] & {all_practical['median']:+.3f} "
        f"[{all_practical['q1']:+.3f}, {all_practical['q3']:+.3f}] & "
        f"{cond_practical['median']:+.3f} "
        f"[{cond_practical['q1']:+.3f}, {cond_practical['q3']:+.3f}] \\\\ ",
    ]
    for threshold in (0.0, 0.01, 0.02):
        a = int((all_participant["delta"] > threshold).sum())
        c = int((cond_participant["delta"] > threshold).sum())
        practical_rows.append(
            f"Participants with $\\Delta>{threshold:g}$ & {a} ({100*a/len(all_participant):.1f}\\%) & "
            f"{c} ({100*c/len(cond_participant):.1f}\\%) \\\\"
        )
    (PAPER / "table_practical_impact.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[!t]", r"\centering", r"\settableformat",
        r"\caption{Observed participant-level magnitude and counts, $n$ (\%), of the protocol difference.}",
        r"\label{tab:practical}",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}",
        r"\toprule", r"Metric & All origins & Conditional \\", r"\midrule",
        *practical_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ]))

    all_decision = summary["practical"]["by_estimand"]["all_origin"]
    cond_decision = summary["practical"]["by_estimand"]["conditional"]
    decision_rows = [
        "Opposite decoder ordering & "
        f"{all_decision['model_selection_changes']} "
        f"({all_decision['model_selection_change_percent']:.1f}\\%) & "
        f"{cond_decision['model_selection_changes']} "
        f"({cond_decision['model_selection_change_percent']:.1f}\\%) \\\\ ",
        f"CSP+LDA $\\to$ TS+LR & {all_decision['csp_to_ts']} & {cond_decision['csp_to_ts']} \\\\ ",
        f"TS+LR $\\to$ CSP+LDA & {all_decision['ts_to_csp']} & {cond_decision['ts_to_csp']} \\\\ ",
        f"LOSO-only AUC $\\geq0.70$ & {all_decision['threshold_070_loso_only']} & "
        f"{cond_decision['threshold_070_loso_only']} \\\\ ",
        f"Expanding-only AUC $\\geq0.70$ & {all_decision['threshold_070_forward_only']} & "
        f"{cond_decision['threshold_070_forward_only']} \\\\ ",
        f"LOSO-only AUC $\\geq0.75$ & {all_decision['threshold_075_loso_only']} & "
        f"{cond_decision['threshold_075_loso_only']} \\\\ ",
        f"Expanding-only AUC $\\geq0.75$ & {all_decision['threshold_075_forward_only']} & "
        f"{cond_decision['threshold_075_forward_only']} \\\\ ",
    ]
    margin_rows = []
    for a, c in zip(all_decision["bilateral_margin_sensitivity"][1:], cond_decision["bilateral_margin_sensitivity"][1:]):
        margin_rows.append(f"Flip, both margins $>{a['bilateral_margin_threshold']:g}$ & {a['n']} & {c['n']} \\\\")
    decision_rows[3:3] = margin_rows
    (PAPER / "table_decision_impact.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[!t]", r"\centering", r"\settableformat",
        r"\caption{Observed participant-specific decoder ordering and illustrative threshold crossings ($n=138$). Arrows go from LOSO to expanding-window ordering. Bilateral margins require both absolute TS+LR-minus-CSP+LDA mean AUC differences to exceed the threshold. Threshold crossings average both decoders and targets within participant. These are descriptive point-estimate comparisons.}",
        r"\label{tab:decisions}",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}",
        r"\toprule", r"Outcome & All origins & Conditional \\", r"\midrule",
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
        f"Mean donor-session count & --- & {means[0]:.2f} & {means[1]:.2f} & {means[2]:.2f} \\\\",
    ])
    (PAPER / "table_support.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table*}[!t]", r"\centering", r"\settableformat",
        r"\caption{Observed participant--target support by dataset. The two decoders share these targets. The uniform donor-count-matched reference uses the conditional support; the future-only reference requires strict-future support, $t\leq(T-1)/2$. Mean donor-session count is the target-weighted expanding-window count $t$.}",
        r"\label{tab:support}",
        r"\begin{tabularx}{0.90\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrr@{}}",
        r"\toprule",
        r"Dataset & Participants & All origins & Conditional & Strict future \\",
        r"\midrule", *support_rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}", "",
    ]))

    contrast_rows = []
    labels = {
        "total": "LOSO $-$ expanding window",
        "quantity_uniform": "LOSO $-$ donor-count-matched reference",
        "composition_uniform": "Donor-count-matched reference $-$ expanding window",
        "strict_future": "Strict future $-$ expanding window",
    }
    for support_key, support_label in (("primary_conditional", "Later donor available: 1,001 participant--targets"),
                                       ("strict_future", "Strict-future common support: 560 participant--targets")):
        contrast_rows.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{{support_label}}}}} \\")
        for key, record in common_contrasts[support_key].items():
            contrast_rows.append(
                f"{labels[key]} & {record['n']} & {record['mean']:+.4f} "
                f"[{record['ci'][0]:+.4f}, {record['ci'][1]:+.4f}] \\\\"
            )
    (PAPER / "table_common_support.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table*}[!t]", r"\centering", r"\settableformat",
        r"\caption{Reference-dependent descriptive decomposition. Uniform donor-session-count matching is analyzed on the full conditional support and, separately, on the strict-future support required by the future-only reference. The two uniform-reference components sum to the total contrast within each support. Matching session count does not match trial count.}",
        r"\label{tab:common-support}",
        r"\begin{tabularx}{0.80\textwidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}",
        r"\toprule",
        r"Contrast & Participants & Mean AUC difference [95\% CI] \\", r"\midrule",
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

    sensitivity_rows = []
    for label, column in (("LOSO $-$ immediately previous", "delta_last"),
                          ("LOSO $-$ first session", "delta_first")):
        p = wide[~wide["is_structural_zero"]].groupby("subject_uid")[column].mean()
        s = t_summary(p)
        sensitivity_rows.append(
            f"{label} & {s['n']} & {s['mean']:+.3f} "
            f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] \\\\"
        )
    if not ma_frame.empty:
        for unit, estimand in (("block", "conditional"), ("day", "conditional"),
                               ("block", "all_origin"), ("day", "all_origin")):
            row = ma_frame[(ma_frame["time_unit"] == unit) &
                           (ma_frame["metric"] == "auc") &
                           (ma_frame["estimand"] == estimand)].iloc[0]
            label = f"Ma2020 {unit}, {estimand.replace('_', ' ')}"
            sensitivity_rows.append(
                f"{label} & {int(row['n'])} & {row['mean']:+.3f} "
                f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}] \\\\"
            )
    (PAPER / "table_sensitivity.tex").write_text("\n".join([
        "% Generated by scripts/revision_analysis.py; do not edit.",
        r"\begin{table}[!t]", r"\centering", r"\settableformat",
        r"\caption{Alternative training policies and Ma2020 time-unit sensitivity.}",
        r"\label{tab:sensitivity}", r"\par\smallskip",
        r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}",
        r"\toprule", r"Analysis & $n$ & Mean AUC difference [95\% CI] \\", r"\midrule",
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

    (PAPER / "revision_results.tex").write_text(
        "% Generated by scripts/revision_analysis.py; do not edit.\n" +
        "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()) +
        "\n"
    )


def main():
    wide = pd.read_csv(RESULTS / "gap_wide.csv")
    wide = wide[wide["model"].isin(CLASSICAL)].copy()
    long_df = pd.read_csv(RESULTS / "auc_long.csv")
    conditional_rows = wide[~wide["is_structural_zero"]].copy()
    all_rows = wide.copy()

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
    forest = figure_forest(all_rows)
    figure_estimands(
        {"conditional": cond_bundle, "all_origin": all_bundle},
        cond_participant, all_participant,
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

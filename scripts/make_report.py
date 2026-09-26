#!/usr/bin/env python3
"""Aggregate raw results, run the statistical tests, and render all figures.

Outputs (under results/ and results/figures/):
  auc_long.csv, transfer_long.csv, gap_wide.csv
  summary.json                      -- all headline numbers used by the report
  per_dataset_slopes.csv, model_ranking.csv, protocol_means.csv
  Each figure is written as an editable SVG source, a vector PDF mirror for
  pdfLaTeX, and a PNG preview for the HTML report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import RAW_DIR, RESULTS_DIR, FIG_DIR, DEPLOYABLE_REF  # noqa: E402
from src import stats as st  # noqa: E402

PROTO_ORDER = ["loso", "loso_matched", "future_matched", "forward_expanding",
               "last_session", "first_session"]
PROTO_LABEL = {"loso": "LOSO (all T-1 sessions)",
               "loso_matched": "LOSO pool, size-matched to t",
               "future_matched": "future-only, size-matched to t",
               "forward_expanding": "forward (expanding, t past)",
               "last_session": "last-session-only", "first_session": "first-session-only"}
MODEL_LABEL = {"csp_lda": "CSP+LDA", "ts_lr": "TS+LR", "eegnet": "EEGNet"}
DS_COLORS = {"BNCI2014_004": "#4477AA", "Zhou2016": "#EE6677",
             "Zhou2020": "#66CCEE", "Kumar2024": "#228833",
             "Ma2020": "#CCBB44", "Stieger2021": "#AA3377",
             "Shin2017A": "#BBBBBB"}
FIGSIZE = (9.0, 4.8)
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


def _save_figure(fig, stem):
    """Write one fixed-size figure as editable SVG, vector PDF, and preview PNG."""
    fig.set_size_inches(*FIGSIZE, forward=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.svg", format="svg")
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", format="png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- IO
def load_long():
    frames = [pd.read_csv(p) for p in sorted(RAW_DIR.glob("auc_*.csv"))]
    if not frames:
        raise SystemExit("no auc_*.csv results yet")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS_DIR / "auc_long.csv", index=False)
    return df


def load_transfer():
    frames = [pd.read_csv(p) for p in sorted(RAW_DIR.glob("transfer_*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS_DIR / "transfer_long.csv", index=False)
    return df


def build_gap_wide(long_df):
    d = long_df[long_df["target_session"] >= 1].copy()
    keys = ["dataset", "subject", "model", "target_session", "drift"]
    wide = d.pivot_table(index=keys, columns="protocol", values="auc").reset_index()
    wide = wide.rename_axis(None, axis=1)
    for ref, name in [("forward_expanding", "delta"),
                      ("last_session", "delta_last"),
                      ("first_session", "delta_first")]:
        if "loso" in wide and ref in wide:
            wide[name] = wide["loso"] - wide[ref]

    Tmap = (long_df.groupby(["dataset", "subject"])["target_session"].max() + 1).rename("T")
    wide = wide.merge(Tmap, on=["dataset", "subject"], how="left")
    wide["sessions_after"] = (wide["T"] - 1 - wide["target_session"]).astype(float)

    # STRUCTURAL ZERO. At t = T-1 the LOSO training set {0..T-2} is *identical* to
    # forward chaining's {0..T-2}, so delta is 0 by construction, not by evidence
    # (verified empirically: mean and sd are exactly 0.00000 on those rows). They
    # carry no information about H1 and would dilute every estimate, so the
    # headline tests run on interior targets 1 <= t < T-1 only.
    wide["is_structural_zero"] = wide["target_session"] == wide["T"] - 1

    # Mechanism covariates. LOSO always trains on T-1 sessions; forward chaining
    # trains on t. Separating log(t) from log(T-1) asks whether the gap is about
    # forward's own starvation or about LOSO's abundance. These are only separable
    # because T varies (3,5,6,7,11,15) -- and, within Stieger2021 alone, T is 7 or
    # 11, which identifies log(T-1) free of any between-dataset confound.
    wide["log_t"] = np.log(wide["target_session"])
    wide["log_Tm1"] = np.log(wide["T"] - 1)
    wide["log_ratio"] = wide["log_Tm1"] - wide["log_t"]

    wide = st.add_derived(wide)
    wide.to_csv(RESULTS_DIR / "gap_wide.csv", index=False)
    return wide


def interior(wide):
    """Rows that actually carry evidence for H1/H2 (drops structural zeros)."""
    return wide[~wide["is_structural_zero"]].copy()


# ---------------------------------------------------------------------- FIGURES
def _subject_ci(df, group_cols, value):
    """Per-participant mean, then two-sided 95% t CI within group."""
    sm = df.dropna(subset=[value]).groupby(group_cols + ["subject_uid"])[value].mean().reset_index()
    g = sm.groupby(group_cols)[value].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["crit"] = g["count"].map(
        lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
    g["lo"] = g["mean"] - g["crit"] * g["se"]
    g["hi"] = g["mean"] + g["crit"] * g["se"]
    return g


def fig_gap_by_dataset_model(wide):
    g = _subject_ci(wide, ["dataset", "model"], "delta")
    datasets = [d for d in DS_COLORS if d in g["dataset"].unique()]
    models = [m for m in MODEL_LABEL if m in g["model"].unique()]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(datasets)); w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = g[g["model"] == m].set_index("dataset").reindex(datasets)
        ax.bar(x + i * w, sub["mean"], w, label=MODEL_LABEL.get(m, m),
               yerr=(sub["mean"] - sub["lo"]).abs(), capsize=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=10)
    ax.set_ylabel(r"$\Delta$ = AUC$_{LOSO}$ - AUC$_{forward}$")
    ax.set_title("LOSO minus forward-chaining AUC\n(mean over participants and interior sessions, 95% CI)")
    ax.legend(title="model")
    _save_figure(fig, "fig1_gap_by_dataset_model")


def fig_gap_vs_drift(wide):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    d = wide.dropna(subset=["delta", "drift_z"])
    for ds, sub in d.groupby("dataset"):
        ax.scatter(sub["drift_z"], sub["delta"], s=14, alpha=0.5,
                   color=DS_COLORS.get(ds, "#888"), label=ds)
    # pooled OLS line
    if len(d) > 2:
        b1, b0 = np.polyfit(d["drift_z"], d["delta"], 1)
        xs = np.linspace(d["drift_z"].min(), d["drift_z"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, "k-", lw=2, label=f"pooled slope={b1:.3f}")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("session drift  z( $d_R(\\bar C_t,\\bar C_{t-1})$ )  [within dataset]")
    ax.set_ylabel(r"$\Delta$ = AUC$_{LOSO}$ - AUC$_{forward}$")
    ax.set_title("Protocol gap and consecutive-session covariance distance\n"
                 "(raw observations and pooled descriptive line)")
    ax.legend(fontsize=8)
    _save_figure(fig, "fig2_gap_vs_drift")


def fig_gap_by_session(wide):
    g = _subject_ci(wide, ["target_session"], "delta")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(g["target_session"], g["mean"],
                yerr=(g["mean"] - g["lo"]).abs(), fmt="o-", capsize=3, color="#4477AA")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("target session index t")
    ax.set_ylabel(r"mean $\Delta$")
    ax.set_title("LOSO--forward difference by target session\n(mechanically 0 at t=T-1 where LOSO=forward)")
    _save_figure(fig, "fig3_gap_by_session")


def fig_ranking_inversion(long_df, dataset="Stieger2021"):
    """Slope chart of per-model means under two protocols on Stieger2021."""
    d = long_df[(long_df["dataset"] == dataset) & (long_df["target_session"] >= 1)]
    cnt = d.groupby("subject")["model"].nunique()
    subs = cnt[cnt == d["model"].nunique()].index
    d = d[d["subject"].isin(subs)]
    if d.empty:
        return
    means = d[d["protocol"].isin(["loso", DEPLOYABLE_REF])].groupby(
        ["protocol", "model"])["auc"].mean().unstack()
    if means.empty or "loso" not in means.index:
        return
    colors = {"csp_lda": "#4477AA", "ts_lr": "#0c6e77", "eegnet": "#AA3377"}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = [0, 1]
    for m in means.columns:
        y = [means.loc["loso", m], means.loc[DEPLOYABLE_REF, m]]
        ax.plot(x, y, "o-", lw=2.4, ms=8, color=colors.get(m, "#888"),
                label=MODEL_LABEL.get(m, m))
        ax.annotate(f"{y[0]:.3f}", (0, y[0]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9)
        ax.annotate(f"{y[1]:.3f}", (1, y[1]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["LOSO", "forward chaining"], fontsize=9)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylabel("mean AUC")
    loso_order = means.loc["loso"].sort_values(ascending=False).index.tolist()
    fwd_order = means.loc["forward_expanding"].sort_values(ascending=False).index.tolist()
    relation = "model order differs between protocols" if loso_order != fwd_order else "model order is unchanged"
    ax.set_title(f"Model means under two evaluation protocols ({dataset}, n={len(subs)})\n{relation}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=max(1, len(means.columns)), frameon=False)
    _save_figure(fig, "fig12_ranking_inversion")


def fig_deployment(long_df, dataset="Ma2020"):
    """Session-wise LOSO and forward estimates on the longest trajectory."""
    d = long_df[(long_df["dataset"] == dataset) & (long_df["model"] != "eegnet")].copy()
    if d.empty:
        return None
    d["uid"] = d["dataset"] + "/" + d["subject"].astype(str)
    piv = d.pivot_table(index=["uid", "target_session"], columns="protocol",
                        values="auc").reset_index()
    # per-session mean over subjects x classical models
    g = piv.groupby("target_session").agg(
        loso=("loso", "mean"), forward=("forward_expanding", "mean")).dropna()
    g = g[g.index >= 1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(g.index, g["loso"], "o-", color="#AA3377", label="LOSO")
    ax.plot(g.index, g["forward"], "o-", color="#4477AA",
            label="forward chaining")
    ax.fill_between(g.index, g["forward"], g["loso"], color="#AA3377", alpha=0.12)
    # annotate the earliest-session difference
    t1 = g.index.min()
    gap1 = g.loc[t1, "loso"] - g.loc[t1, "forward"]
    ax.annotate(f"session {int(t1)} difference\n{gap1*100:.1f} AUC points",
                xy=(t1, (g.loc[t1, "loso"] + g.loc[t1, "forward"]) / 2),
                xytext=(t1 + 2.5, g.loc[t1, "loso"] + 0.01), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#AA3377"))
    ax.set_xlabel(f"deployment session on {dataset} (T={int(g.index.max())+1})")
    ax.set_ylabel("AUC")
    ax.set_title("LOSO and forward-chaining estimates by target session\n(the shaded region is their difference)")
    ax.legend(fontsize=8, loc="lower right")
    _save_figure(fig, "fig11_deployment")
    # return a small practitioner table for the report
    rows = []
    for t in [tt for tt in (1, 2, 3, 5) if tt in g.index]:
        rows.append({"session": int(t), "loso": float(g.loc[t, "loso"]),
                     "forward": float(g.loc[t, "forward"]),
                     "difference": float(g.loc[t, "loso"] - g.loc[t, "forward"])})
    return rows


def fig_transfer_curve(tm):
    if tm.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
    # (a) descriptive: raw AUC vs signed lag (confounded by test-session difficulty)
    g = tm.dropna(subset=["auc"]).groupby("lag")["auc"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    ax[0].errorbar(g["lag"], g["mean"], yerr=1.96 * g["se"], fmt="o-", capsize=3, color="#228833")
    ax[0].axvline(0, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel("donor lag = train - test session  (<0 past, >0 future)")
    ax[0].set_ylabel("single-session transfer AUC")
    ax[0].set_title("(a) descriptive: AUC by signed lag\n(mixes different test sessions)")
    # (b) clean within-test-session future - past contrast, by |lag|
    _, wide = st.transfer_symmetry(tm)
    if len(wide):
        h = wide.groupby("abslag")["future_minus_past"].agg(["mean", "std", "count"]).reset_index()
        h["se"] = h["std"] / np.sqrt(h["count"].clip(lower=1))
        ax[1].errorbar(h["abslag"], h["mean"], yerr=1.96 * h["se"], fmt="s-", capsize=3, color="#AA3377")
        ax[1].axhline(0, color="k", lw=1)
        ax[1].set_xlabel("|lag| (sessions)")
        ax[1].set_ylabel("AUC(future donor) - AUC(past donor)")
        ax[1].set_title("(b) same test session, matched size\n>0 = future look-ahead helps")
    _save_figure(fig, "fig4_transfer_curve")


def fig_protocol_auc(long_df):
    d = long_df[(long_df["target_session"] >= 1) &
                (long_df["model"] != "eegnet")].copy()
    d["subject_uid"] = d["dataset"].astype(str) + "/" + d["subject"].astype(str)
    g = _subject_ci(d, ["protocol"], "auc")
    g["protocol"] = pd.Categorical(g["protocol"], PROTO_ORDER, ordered=True)
    g = g.sort_values("protocol")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["#AA3377", "#4477AA", "#66CCEE", "#CCBB44", "#228833", "#999933"]
    ax.bar([PROTO_LABEL.get(p, str(p)) for p in g["protocol"]], g["mean"],
           yerr=(g["mean"] - g["lo"]).abs(), capsize=4, color=colors[:len(g)])
    ax.set_ylabel("mean AUC (interior sessions)")
    ax.set_ylim(0.5, max(0.9, g["hi"].max() + 0.02))
    ax.set_title("Mean decoding AUC by evaluation protocol")
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    _save_figure(fig, "fig6_protocol_auc")


def fig_dose_response(summary):
    """Delta vs LOSO's data surplus -- the cold-start dose-response (headline)."""
    dr = summary.get("dose_response", [])
    if not dr:
        return
    sa = [r["sessions_after"] for r in dr]
    mean = [r["mean"] for r in dr]
    lo = [r["ci"][0] for r in dr]; hi = [r["ci"][1] for r in dr]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(sa, mean, yerr=[np.subtract(mean, lo), np.subtract(hi, mean)],
                fmt="o-", capsize=3, color="#AA3377")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("extra sessions LOSO trains on beyond deployment  $(T-1)-t$")
    ax.set_ylabel(r"optimism gap $\Delta$")
    ax.set_title("LOSO gap by the number of unavailable future sessions\n"
                 "(descriptive subject-level means and 95% confidence intervals)")
    _save_figure(fig, "fig7_dose_response")


def fig_history_mechanisms(summary, tm):
    """Pair the protocol dose curve with a same-target donor-direction contrast."""
    dr = summary.get("dose_response", [])
    if not dr or tm.empty:
        return

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)

    # (a) Protocol separation by the exact number of sessions that LOSO can use
    # but forward evaluation cannot. Open markers identify bins supported by only
    # one dataset, making the narrowing support visible rather than implicit.
    dose = pd.DataFrame(dr).sort_values("sessions_after")
    x = dose["sessions_after"].to_numpy()
    y = dose["mean"].to_numpy()
    lo = np.array([v[0] for v in dose["ci"]])
    hi = np.array([v[1] for v in dose["ci"]])
    ax[0].errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", capsize=3,
                   color="#AA3377", lw=1.4)
    multi = dose["n_datasets"].to_numpy() > 1
    ax[0].plot(x, y, "-", color="#AA3377", lw=1.8)
    ax[0].scatter(x[multi], y[multi], s=42, color="#AA3377",
                  label="multiple datasets")
    ax[0].scatter(x[~multi], y[~multi], s=42, facecolors="white",
                  edgecolors="#AA3377", linewidths=1.5,
                  label="one dataset")
    ax[0].axhline(0, color="k", lw=0.9)
    ax[0].set_xlabel("future sessions available to LOSO but not forward evaluation")
    ax[0].set_ylabel(r"mean paired gap $\Delta$")
    ax[0].set_title("(a) Protocol gap by unavailable history")
    ax[0].legend(fontsize=8, frameon=False)

    # (b) One-session donors equidistant from the same target. Aggregation is at
    # participant level before the CI, matching the manuscript's inferential unit.
    _, paired = st.transfer_symmetry(tm)
    paired["subject_uid"] = (paired["dataset"].astype(str) + "/" +
                             paired["subject"].astype(str))
    lag = _subject_ci(paired, ["abslag"], "future_minus_past")
    ax[1].errorbar(lag["abslag"], lag["mean"],
                   yerr=[lag["mean"] - lag["lo"], lag["hi"] - lag["mean"]],
                   fmt="s-", capsize=3, color="#4477AA", lw=1.8, ms=5)
    ax[1].axhline(0, color="k", lw=0.9)
    ax[1].set_xlabel("absolute donor lag from the target (sessions)")
    ax[1].set_ylabel("AUC(future donor) - AUC(past donor)")
    ax[1].set_title("(b) Same target and matched donor size")

    _save_figure(fig, "fig13_history_mechanisms")


def fig_decomposition(summary):
    """Total gap = quantity + sampled composition, with size-matched contrasts."""
    sm = summary.get("size_matched", {})
    h1 = summary.get("H1_vs_forward", {})
    if not sm or "matched_vs_forward" not in sm:
        return
    bars = [("total\n(loso - forward)", h1.get("subject_mean", np.nan), "#AA3377"),
            ("size\n(loso - loso_matched)", sm["loso_vs_matched"]["subject_mean"], "#4477AA"),
            ("composition\n(loso_matched - forward)", sm["matched_vs_forward"]["subject_mean"], "#CCBB44")]
    cis = [h1.get("ci"),
           sm["loso_vs_matched"]["ci"], sm["matched_vs_forward"]["ci"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (lab, val, col) in enumerate(bars):
        err = None if cis[i] is None else [[val - cis[i][0]], [cis[i][1] - val]]
        ax.bar(i, val, color=col, yerr=err, capsize=4)
        ax.text(i, val + 0.001, f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("AUC")
    ax.set_title("Decomposition of the LOSO-forward gap\n"
                 "same-size composition contrast is estimated with uncertainty")
    _save_figure(fig, "fig8_decomposition")


# ------------------------------------------------------- MECHANISM (primary)
def _lmm(formula, df, group):
    import statsmodels.formula.api as smf
    r = smf.mixedlm(formula, df, groups=df[group]).fit(reml=False, method="lbfgs")
    return r


def mechanism_decomposition(wide):
    """Is the gap forward's starvation log(t), or LOSO's abundance log(T-1)?

    C(dataset) is mandatory here: T is nearly collinear with dataset identity, so
    without it log_Tm1 silently absorbs between-dataset differences in gap size
    and looks strongly positive. With it, log_Tm1 is identified only where T
    varies within a dataset -- i.e. Stieger2021's T=7 vs T=11 subjects.
    """
    d = interior(wide).dropna(subset=["delta", "log_t", "log_Tm1", "drift_z"])
    out = {"n": int(len(d)), "n_subjects": int(d["subject_uid"].nunique())}
    specs = {
        "drift_only": "delta ~ drift_z + C(model) + C(dataset)",
        "t_only": "delta ~ target_session + C(model) + C(dataset)",
        "log_t_only": "delta ~ log_t + C(model) + C(dataset)",
        "full": "delta ~ log_t + log_Tm1 + drift_z + C(model) + C(dataset)",
    }
    for name, fo in specs.items():
        try:
            r = _lmm(fo, d, "subject_uid")
            terms = {}
            for k in ("log_t", "log_Tm1", "drift_z", "target_session"):
                if k in r.params:
                    ci = r.conf_int().loc[k]
                    terms[k] = dict(beta=float(r.params[k]), p=float(r.pvalues[k]),
                                    ci=[float(ci[0]), float(ci[1])])
            out[name] = {"formula": fo, "aic": float(r.aic), "terms": terms}
        except Exception as e:
            out[name] = {"formula": fo, "error": str(e)}
    return out


def size_matched_contrast(wide):
    """Size-matched composition analysis: t training sessions, varying the pool.

    delta_matched = AUC(loso_matched) - AUC(forward_expanding): both train on
    exactly t sessions, but loso_matched may draw from the future. A delta_matched
    of ~0 is consistent with a predominantly training-size gap under this sampling
    scheme. delta_future = AUC(future_matched) - AUC(forward_expanding) is the
    sharper same-size past-vs-future contrast (defined for t <= (T-1)/2), but both
    composition terms can retain temporal practice or session-quality trends.
    """
    out = {}
    for col, ref, key in [("loso_matched", "forward_expanding", "matched_vs_forward"),
                          ("future_matched", "forward_expanding", "future_vs_forward"),
                          ("loso", "loso_matched", "loso_vs_matched")]:
        if col not in wide or ref not in wide:
            continue
        d = interior(wide).dropna(subset=[col, ref]).copy()
        if d.empty:
            continue
        d["g"] = d[col] - d[ref]
        sp = d.groupby("subject_uid")["g"].mean()
        t, p = stats.ttest_1samp(sp, 0)
        crit = stats.t.ppf(0.975, len(sp) - 1) if len(sp) > 1 else np.nan
        rec = {"contrast": f"{col} - {ref}", "n": int(len(d)),
               "n_subjects": int(len(sp)), "mean": float(d["g"].mean()),
               "subject_mean": float(sp.mean()), "p_twosided": float(p),
               "ci": [float(sp.mean() - crit * sp.sem()),
                      float(sp.mean() + crit * sp.sem())],
               "by_dataset": {ds: float(g.groupby("subject_uid")["g"].mean().mean())
                              for ds, g in d.groupby("dataset")}}
        # For the two composition contrasts, bound the effect with equivalence
        # tests and report power to detect a 1-point (0.01 AUC) difference.
        if key in ("matched_vs_forward", "future_vs_forward"):
            rec["tost"] = {f"margin_{m}": st.tost_equivalence(sp.values, m)
                           for m in (0.01, 0.02)}
            rec["power_01"] = st.power_to_detect(sp.values, 0.01)
        out[key] = rec
    return out


def common_support_models(wide):
    """Compare models only on subjects where every model actually ran.

    A raw per-model comparison can be confounded whenever model coverage differs
    by dataset. Restricting to subjects with every fitted model makes the model
    contrast interpretable, at the cost of support.
    """
    d = interior(wide).dropna(subset=["delta"])
    per = d.groupby(["subject_uid", "model"])["delta"].mean().unstack()
    common = per.dropna()
    out = {"n_subjects_common": int(len(common)),
           "models": sorted(per.columns.tolist()), "per_model": {}}
    for m in common.columns:
        t, p = stats.ttest_1samp(common[m], 0)
        crit = stats.t.ppf(0.975, len(common) - 1) if len(common) > 1 else np.nan
        out["per_model"][m] = {"mean": float(common[m].mean()),
                               "p_twosided": float(p),
                               "ci": [float(common[m].mean() - crit * common[m].sem()),
                                      float(common[m].mean() + crit * common[m].sem())]}
    return out


def dose_response(wide):
    """Delta as a function of how many extra sessions LOSO gets (sessions_after).

    sessions_after = (T-1) - t is exactly LOSO's data surplus at target t. At
    sessions_after=1 the two protocols differ by a single session -- the minimal
    asymmetry. We fit a dataset-adjusted slope as the headline dose statistic and
    record which datasets contribute at each descriptive bin; no monotonic shape is
    assumed.
    """
    d = interior(wide).dropna(subset=["delta"])
    rows = []
    for sa, g in d.groupby("sessions_after"):
        sp = g.groupby("subject_uid")["delta"].mean()
        if len(sp) < 5:
            continue
        t, p = stats.ttest_1samp(sp, 0)
        crit = stats.t.ppf(0.975, len(sp) - 1)
        rows.append({"sessions_after": int(sa), "mean": float(sp.mean()),
                     "p_twosided": float(p), "n_subjects": int(len(sp)),
                     "n_datasets": int(g["dataset"].nunique()),
                     "datasets": sorted(g["dataset"].unique().tolist()),
                     "ci": [float(sp.mean() - crit * sp.sem()),
                            float(sp.mean() + crit * sp.sem())]})
    return rows


def dose_slope(wide):
    """Dataset-adjusted slope of Delta on sessions_after (the headline dose number,
    robust to the between-dataset confound in the high-surplus bins)."""
    d = interior(wide).dropna(subset=["delta", "sessions_after"])
    try:
        r = _lmm("delta ~ sessions_after + C(dataset) + C(model)", d, "subject_uid")
        ci = r.conf_int().loc["sessions_after"]
        return {"beta": float(r.params["sessions_after"]), "p": float(r.pvalues["sessions_after"]),
                "ci": [float(ci[0]), float(ci[1])], "n": int(len(d))}
    except Exception as e:
        return {"error": str(e)}


def size_share_range(summary):
    """Share of the total gap attributable to training size, with the uncertainty
    that the look-ahead CI implies. size_share = size / total; the look-ahead CI
    upper bound pushes the size share down (and vice versa)."""
    sm = summary.get("size_matched", {})
    total = summary["H1_vs_forward"]["subject_mean"]
    if "loso_vs_matched" not in sm or total <= 0:
        return {}
    size = sm["loso_vs_matched"]["subject_mean"]
    look = sm["matched_vs_forward"]
    lo_look, hi_look = look["ci"]
    # size share is (total - look)/total; look ranges over its CI (clamped >=0)
    def share(lk):
        return max(0.0, min(1.0, (total - max(0.0, lk)) / total))
    return {"point": share(look["subject_mean"]),
            "range_from_lookahead_ci": [share(hi_look), share(lo_look)],
            "note": "share falls as the look-ahead component rises to its CI upper bound"}


def leave_dataset_out(wide):
    """Sensitivity: recompute H1 dropping each dataset, and dropping both long-T ones."""
    d = interior(wide).dropna(subset=["delta"])
    out = []
    for ds in sorted(d["dataset"].unique()):
        sub = d[d["dataset"] != ds]
        sp = sub.groupby("subject_uid")["delta"].mean()
        t, p = stats.ttest_1samp(sp, 0)
        out.append({"dropped": ds, "mean": float(sp.mean()),
                    "p_onesided": float(p / 2 if t > 0 else 1 - p / 2),
                    "n_subjects": int(len(sp))})
    sub = d[~d["dataset"].isin(["Ma2020", "Stieger2021"])]
    if not sub.empty:
        sp = sub.groupby("subject_uid")["delta"].mean()
        t, p = stats.ttest_1samp(sp, 0)
        out.append({"dropped": "Ma2020+Stieger2021 (both long-T)",
                    "mean": float(sp.mean()),
                    "p_onesided": float(p / 2 if t > 0 else 1 - p / 2),
                    "n_subjects": int(len(sp))})
    return out


def natural_experiment(long_df):
    """Stieger2021 T=7 vs T=11 at matched t: does LOSO's extra data help?

    Same dataset and same target session; only the size of LOSO's training set
    changes (6 vs 10 sessions). Forward trains on t either way. This is the one
    place where LOSO's abundance is manipulated free of any dataset confound.
    """
    d = long_df[(long_df["dataset"] == "Stieger2021") &
                (long_df["model"] != "eegnet")].copy()
    if d.empty:
        return {"skipped": "Stieger2021 absent"}
    d["T"] = d.groupby("subject")["target_session"].transform("max") + 1
    w = d.pivot_table(index=["subject", "model", "target_session", "T", "drift"],
                      columns="protocol", values="auc").reset_index()
    w = w[(w["target_session"] >= 1) & (w["target_session"] < w["T"] - 1)]
    w["delta"] = w["loso"] - w["forward_expanding"]
    ov = w[w["target_session"] <= 4].dropna(subset=["delta"])   # t range both groups share
    s7 = ov[ov["T"] == 7].groupby("subject")["delta"].mean()
    s11 = ov[ov["T"] == 11].groupby("subject")["delta"].mean()
    t, p = stats.ttest_ind(s11, s7)
    res = {"n_T7": int(len(s7)), "n_T11": int(len(s11)),
           "mean_T7": float(s7.mean()), "mean_T11": float(s11.mean()),
           "diff_T11_minus_T7": float(s11.mean() - s7.mean()),
           "p_twosided": float(p)}
    try:
        r = _lmm("delta ~ C(T) + target_session + drift", ov, "subject")
        k = [i for i in r.params.index if "T)[T.11]" in i][0]
        res["lmm_T11_effect"] = float(r.params[k])
        res["lmm_p"] = float(r.pvalues[k])
    except Exception as e:
        res["lmm_error"] = str(e)
    return res


def donor_direction(tm):
    """Future vs past donor at a FIXED test session and matched |lag|.

    A positive intercept describes an average future-donor advantage. A positive
    slope in L means the contrast grows as the two donors move farther apart in
    session index; either pattern may still contain practice or quality trends.
    """
    t = tm.copy()
    t["L"] = t["lag"].abs()
    t["dir"] = np.where(t["lag"] > 0, "future", "past")
    t["uid"] = t["dataset"] + "_S" + t["subject"].astype(str)
    pv = t.pivot_table(index=["dataset", "uid", "model", "test_session", "L"],
                       columns="dir", values="auc").dropna().reset_index()
    if pv.empty or "future" not in pv or "past" not in pv:
        return {"skipped": "no matched donor pairs"}
    pv["diff"] = pv["future"] - pv["past"]
    sp = pv.groupby("uid")["diff"].mean()
    tt, pp = stats.ttest_1samp(sp, 0)
    crit = stats.t.ppf(0.975, len(sp) - 1) if len(sp) > 1 else np.nan
    out = {"n_pairs": int(len(pv)), "n_subjects": int(len(sp)),
           "mean_future_minus_past": float(pv["diff"].mean()),
           "subject_mean": float(sp.mean()), "p_twosided": float(pp),
           "ci": [float(sp.mean() - crit * sp.sem()), float(sp.mean() + crit * sp.sem())],
           "by_lag": []}
    for L in sorted(pv["L"].unique()):
        g = pv[pv["L"] == L].groupby("uid")["diff"].mean()
        if len(g) < 5:
            continue
        tl, pl = stats.ttest_1samp(g, 0)
        out["by_lag"].append({"L": int(L), "mean": float(g.mean()),
                              "p": float(pl), "n_subjects": int(len(g))})
    try:
        r = _lmm("diff ~ L + C(model) + C(dataset)", pv, "uid")
        out["lmm_intercept"] = float(r.params["Intercept"])
        out["lmm_intercept_p"] = float(r.pvalues["Intercept"])
        out["lmm_L_slope"] = float(r.params["L"])
        out["lmm_L_p"] = float(r.pvalues["L"])
    except Exception as e:
        out["lmm_error"] = str(e)
    return out


def ranking_inversion(long_df):
    """Does LOSO reorder models relative to the deployable protocol?

    For each dataset, rank models by mean AUC under LOSO and under forward
    chaining (restricted to subjects that have every model, so the comparison is
    paired). Report whether the orderings differ and, for the top-2 pair, a paired
    subject-level test under each protocol. A reversal means only that the two
    protocol-specific mean rankings differ; it does not identify why a model
    changes rank.
    """
    out = []
    d = long_df[long_df["target_session"] >= 1]
    for ds, g in d.groupby("dataset"):
        models = sorted(g["model"].unique())
        if len(models) < 2:
            continue
        # subjects with every model present (paired support)
        cnt = g.groupby("subject")["model"].nunique()
        subs = cnt[cnt == len(models)].index
        gg = g[g["subject"].isin(subs)]
        if gg.empty:
            continue
        rec = {"dataset": ds, "n_subjects": int(len(subs)), "models": models}
        order = {}
        for proto in ["loso", DEPLOYABLE_REF]:
            piv = gg[gg["protocol"] == proto].pivot_table(
                index="subject", columns="model", values="auc")
            if piv.empty:
                break
            means = piv.mean().sort_values(ascending=False)
            order[proto] = list(means.index)
            rec[f"means_{proto}"] = {m: float(v) for m, v in means.items()}
        if len(order) < 2:
            continue
        rec["order_loso"] = order["loso"]
        rec["order_forward"] = order[DEPLOYABLE_REF]
        rec["inverted"] = order["loso"] != order[DEPLOYABLE_REF]
        # paired test on the LOSO-winner vs forward-winner pair
        a, b = order["loso"][0], order[DEPLOYABLE_REF][0]
        if a != b:
            for proto in ["loso", DEPLOYABLE_REF]:
                piv = gg[gg["protocol"] == proto].pivot_table(
                    index="subject", columns="model", values="auc")
                if a in piv and b in piv:
                    diff = (piv[a] - piv[b]).dropna()
                    if len(diff) > 1:
                        t, p = stats.ttest_1samp(diff, 0)
                        rec[f"top2_{proto}"] = {
                            "pair": f"{a} - {b}", "diff": float(diff.mean()),
                            "p_twosided": float(p), "n": int(len(diff))}
        out.append(rec)
    return out


def practice_trend(tm):
    """Are later sessions intrinsically better (as target, and as donor)?"""
    rows = []
    for ds, s in tm.groupby("dataset"):
        by_test = s.groupby("test_session")["auc"].mean()
        by_train = s.groupby("train_session")["auc"].mean()
        rec = {"dataset": ds}
        for tag, ser in [("as_test_target", by_test), ("as_donor", by_train)]:
            if len(ser) > 2:
                r, p = stats.pearsonr(ser.index.values.astype(float), ser.values)
                rec[f"{tag}_r"] = float(r)
                rec[f"{tag}_p"] = float(p)
        rows.append(rec)
    return rows


# ------------------------------------------------------------------------ STATS
def model_ranking(long_df):
    d = long_df[long_df["target_session"] >= 1].copy()
    d["subject_uid"] = d["dataset"].astype(str) + "/" + d["subject"].astype(str)
    # EEGNet is a targeted Stieger2021 analysis. Global model means would compare
    # its 62 participants against classical models from all 138 participants, so
    # restrict this convenience table to participants with every model.
    n_models = d["model"].nunique()
    complete = d.groupby("subject_uid")["model"].nunique()
    d = d[d["subject_uid"].isin(complete[complete == n_models].index)]
    rows = []
    for proto in ["loso", DEPLOYABLE_REF]:
        sub = d[d["protocol"] == proto].groupby("model")["auc"].mean().sort_values(ascending=False)
        rows.append({"protocol": proto, "n_subjects_common": int(d["subject_uid"].nunique()),
                     "ranking": " > ".join(sub.index),
                     **{f"auc_{m}": float(sub.get(m, np.nan)) for m in MODEL_LABEL}})
    return pd.DataFrame(rows)


def main():
    long_df = load_long()
    tm = load_transfer()
    wide = build_gap_wide(long_df)

    # Every headline test runs on INTERIOR targets only. At t = T-1 the two
    # training sets coincide, so those rows are exact zeros by construction.
    inner = interior(wide)
    # The headline gap and every mechanism analysis are estimated on the two
    # CLASSICAL models, which run on every dataset. EEGNet enters as a separate
    # ranking-stability analysis so stochastic training and model coverage cannot
    # silently change the prespecified classical headline. wide_clf mirrors wide
    # for functions that call interior() themselves.
    inner_clf = inner[inner["model"] != "eegnet"].copy()
    wide_clf = wide[wide["model"] != "eegnet"].copy()

    summary = {}
    summary["datasets"] = sorted(long_df["dataset"].unique().tolist())
    summary["models"] = sorted(long_df["model"].unique().tolist())
    summary["headline_models"] = sorted(inner_clf["model"].unique().tolist())
    summary["n_auc_rows"] = int(len(long_df))
    summary["n_gap_rows"] = int(len(wide))
    summary["n_interior_rows"] = int(len(inner))
    summary["n_interior_rows_classical"] = int(len(inner_clf))
    summary["n_structural_zero_rows"] = int(wide["is_structural_zero"].sum())
    summary["n_subjects"] = int(wide["subject_uid"].nunique())
    summary["T_values"] = sorted(int(x) for x in wide["T"].unique())

    # H1: mean gap vs each deployable reference (interior, classical models)
    for col, key in [("delta", "vs_forward"), ("delta_last", "vs_last"), ("delta_first", "vs_first")]:
        if col in inner_clf:
            summary[f"H1_{key}"] = st.test_mean_gap(inner_clf, value=col)
    # per-model uses ALL models (this is the ranking check, not the headline)
    summary["H1_by_model"] = {}
    for m, g in inner.groupby("model"):
        summary["H1_by_model"][m] = st.test_mean_gap(g, value="delta")
    summary["H1_by_dataset"] = {}
    for ds, g in inner_clf.groupby("dataset"):
        summary["H1_by_dataset"][ds] = st.test_mean_gap(g, value="delta")

    # H2 (drift) and the mechanism decomposition that supersedes it (classical)
    summary["H2_model"] = st.fit_gap_model(inner_clf, value="delta")
    summary["H2_mechanism"] = st.fit_mechanism_model(inner_clf, value="delta")
    slopes = st.per_dataset_drift_slopes(inner_clf, value="delta")
    slopes.to_csv(RESULTS_DIR / "per_dataset_slopes.csv", index=False)
    summary["H2_per_dataset"] = slopes.to_dict(orient="records")

    # PRIMARY mechanism evidence (classical models; matched arms are classical-only)
    summary["mechanism_decomposition"] = mechanism_decomposition(wide_clf)
    summary["natural_experiment_stieger"] = natural_experiment(long_df)
    summary["size_matched"] = size_matched_contrast(wide_clf)
    summary["size_share"] = size_share_range(summary)
    # Robustness demanded by adversarial review: model contrast on common support,
    # dose-response in LOSO's data surplus, and leave-one-dataset-out sensitivity.
    summary["common_support_models"] = common_support_models(wide)  # all models
    summary["dose_response"] = dose_response(wide_clf)
    summary["dose_slope"] = dose_slope(wide_clf)
    summary["leave_dataset_out"] = leave_dataset_out(wide_clf)

    # transfer: donor direction + the practice trend that explains it
    if not tm.empty:
        res, _ = st.transfer_symmetry(tm)
        summary["transfer_symmetry"] = res
        summary["donor_direction"] = donor_direction(tm)
        summary["practice_trend"] = practice_trend(tm)

    # protocol means (classical models, to match the headline sample) & model ranking
    d1 = long_df[(long_df["target_session"] >= 1) & (long_df["model"] != "eegnet")].copy()
    d1["subject_uid"] = d1["dataset"].astype(str) + "/" + d1["subject"].astype(str)
    pm = _subject_ci(d1, ["protocol"], "auc")[["protocol", "mean", "lo", "hi", "count"]]
    pm.to_csv(RESULTS_DIR / "protocol_means.csv", index=False)
    summary["protocol_means"] = pm.set_index("protocol")["mean"].to_dict()
    summary["ranking_inversion"] = ranking_inversion(long_df)
    mr = model_ranking(long_df)
    mr.to_csv(RESULTS_DIR / "model_ranking.csv", index=False)
    summary["model_ranking"] = mr.to_dict(orient="records")

    # per (dataset x model) mean gap table
    gm = _subject_ci(wide, ["dataset", "model"], "delta")
    gm.to_csv(RESULTS_DIR / "gap_by_dataset_model.csv", index=False)

    # figures
    fig_gap_by_dataset_model(inner_clf)
    fig_gap_vs_drift(inner_clf)
    fig_gap_by_session(wide_clf)  # keeps t=T-1 to show the structural zero
    fig_transfer_curve(tm)
    fig_protocol_auc(long_df)
    fig_dose_response(summary)
    fig_history_mechanisms(summary, tm)
    fig_decomposition(summary)
    fig_ranking_inversion(long_df, dataset="Stieger2021")
    summary["deployment_table"] = fig_deployment(long_df, dataset="Ma2020")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: summary[k] for k in
                      ["datasets", "n_subjects", "n_interior_rows", "T_values", "H1_vs_forward",
                       "mechanism_decomposition", "natural_experiment_stieger",
                       "donor_direction"] if k in summary},
                     indent=2, default=float)[:2500])
    print("\nfigures ->", FIG_DIR)


if __name__ == "__main__":
    main()

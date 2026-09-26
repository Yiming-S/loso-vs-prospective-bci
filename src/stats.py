"""Hypothesis tests and mixed-effects models for the optimism gap.

H1:  E(Delta) != 0           (the expected direction is LOSO above forward)
H2:  Delta increases with session drift D_{i,t}

Delta_{i,t,m} = b0 + b1 * drift_z + b2 * t + b3 * model + u_subject + eps

We reuse the modelling pattern from ShiftDx (utils.fit_lmm_or_fallback) and the
Shifts-Impact paper (run_claims._fit_single_slope_model): statsmodels MixedLM
with random per-subject intercepts, standardising drift within dataset to
drift_z, and an OLS + cluster-robust fallback when the LMM will not identify.
Dataset enters as a fixed effect (few levels); subject (nested in dataset) is the
random grouping factor.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except Exception:
    _HAS_SM = False


def add_derived(gap_df: pd.DataFrame) -> pd.DataFrame:
    d = gap_df.copy()
    d["subject_uid"] = d["dataset"].astype(str) + "/" + d["subject"].astype(str)
    # standardise drift within dataset (drift scales differ across channel counts)
    d["drift_z"] = np.nan
    for ds, g in d.groupby("dataset"):
        v = g["drift"].astype(float)
        mu, sd = v.mean(), v.std(ddof=0)
        d.loc[g.index, "drift_z"] = 0.0 if (sd == 0 or np.isnan(sd)) else (v - mu) / sd
    return d


# --- H1: does the mean gap differ from zero? --------------------------------
def test_mean_gap(d: pd.DataFrame, value="delta"):
    """Test whether E(Delta) differs from zero, retaining directional values."""
    x = d[value].dropna().values
    out = {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
           "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan}

    # (a) subject-level paired one-sample t-test (avoids pseudo-replication)
    subj = d.dropna(subset=[value]).groupby("subject_uid")[value].mean()
    out["n_subjects"] = int(subj.shape[0])
    out["subject_mean"] = float(subj.mean())
    if subj.shape[0] > 1:
        t, p = sps.ttest_1samp(subj.values, 0.0)
        se = float(subj.sem())
        crit = float(sps.t.ppf(0.975, subj.shape[0] - 1))
        out["ci"] = [float(subj.mean() - crit * se),
                     float(subj.mean() + crit * se)]
        out["subj_t"] = float(t)
        out["subj_p_twosided"] = float(p)
        out["subj_p_onesided"] = float(p / 2 if t > 0 else 1 - p / 2)
        w2 = (sps.wilcoxon(subj.values, alternative="two-sided")
              if np.any(subj.values != 0) else None)
        w1 = (sps.wilcoxon(subj.values, alternative="greater")
              if np.any(subj.values != 0) else None)
        out["subj_wilcoxon_p_twosided"] = float(w2.pvalue) if w2 is not None else np.nan
        out["subj_wilcoxon_p"] = float(w1.pvalue) if w1 is not None else np.nan

    # (b) LMM intercept-only with random subject intercept.
    # Guard against singular / boundary fits: on small or low-variance support the
    # group-variance estimate collapses and statsmodels can return a degenerate
    # intercept of exactly 0.0 (p=0.5). Emitting that as if it were an estimate is
    # misleading (it silently contradicts subject_mean), so we detect it via the
    # convergence flag and a 0-intercept-vs-nonzero-subject-mean sanity check and
    # report the fit as non-converged instead.
    if _HAS_SM and d["subject_uid"].nunique() > 1:
        try:
            m = smf.mixedlm(f"{value} ~ 1", d.dropna(subset=[value]),
                            groups=d.dropna(subset=[value])["subject_uid"])
            r = m.fit(reml=False, method="lbfgs")
            b = float(r.params["Intercept"]); p2 = float(r.pvalues["Intercept"])
            converged = bool(getattr(r, "converged", True))
            degenerate = (abs(b) < 1e-8 and abs(out.get("subject_mean", 0.0)) > 1e-4)
            if (not converged) or degenerate:
                out["lmm_intercept"] = None
                out["lmm_p_twosided"] = None
                out["lmm_p_onesided"] = None
                out["lmm_converged"] = False
            else:
                out["lmm_intercept"] = b
                out["lmm_p_twosided"] = p2
                out["lmm_p_onesided"] = float(p2 / 2 if b > 0 else 1 - p2 / 2)
                out["lmm_converged"] = True
        except Exception as e:
            out["lmm_error"] = str(e)[:120]
            out["lmm_converged"] = False
    return out


# --- Equivalence (TOST) + power, for bounding a near-null effect -------------
def tost_equivalence(subject_values, margin: float, alpha: float = 0.05):
    """Two one-sided tests that a subject-level mean lies within [-margin, +margin].

    Returns the TOST p-value (equivalent within the margin iff p_tost < alpha) and
    the (1-2*alpha) CI, which lies inside the margin exactly when equivalence holds.
    Used to turn "look-ahead is not significant" into the defensible "look-ahead is
    bounded to a negligible band" -- or to show honestly that it is NOT equivalent
    at a tight margin (i.e. only underpowered)."""
    x = np.asarray(subject_values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    out = {"n": int(n), "margin": float(margin), "mean": float(np.mean(x)) if n else np.nan}
    if n < 2:
        return out
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    df = n - 1
    t_lower = (np.mean(x) + margin) / se          # H0: mu <= -margin
    t_upper = (np.mean(x) - margin) / se          # H0: mu >= +margin
    p_lower = float(sps.t.sf(t_lower, df))        # reject if mean > -margin
    p_upper = float(sps.t.cdf(t_upper, df))       # reject if mean < +margin
    p_tost = max(p_lower, p_upper)
    tcrit = sps.t.ppf(1 - alpha, df)
    ci = [float(np.mean(x) - tcrit * se), float(np.mean(x) + tcrit * se)]  # 90% CI at alpha=.05
    out.update(se=se, p_tost=float(p_tost),
               equivalent=bool(p_tost < alpha),
               ci90=ci, ci90_within_margin=bool(ci[0] > -margin and ci[1] < margin))
    return out


def power_to_detect(subject_values, effect: float, alpha: float = 0.05):
    """Achieved power of the one-sided subject-level t-test to detect a true mean
    of size `effect`, given the observed subject-level SD and n."""
    x = np.asarray(subject_values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": int(n), "power": np.nan}
    sd = float(np.std(x, ddof=1))
    df = n - 1
    ncp = effect / (sd / np.sqrt(n))
    tcrit = sps.t.ppf(1 - alpha, df)
    power = float(sps.nct.sf(tcrit, df, ncp))
    return {"n": int(n), "effect": float(effect), "sd": sd, "power": power}


# --- H2: gap vs drift (and t, model, dataset) -------------------------------
def fit_gap_model(d: pd.DataFrame, value="delta"):
    """Full mixed model Delta ~ drift_z + t + C(model) + C(dataset), RE=subject.
    Falls back to cluster-robust OLS if the LMM fails to identify."""
    dd = d.dropna(subset=[value, "drift_z"]).copy()
    info = {"n": int(len(dd)), "n_subjects": int(dd["subject_uid"].nunique())}
    n_model = dd["model"].nunique()
    n_ds = dd["dataset"].nunique()
    terms = ["drift_z", "target_session"]
    if n_model > 1:
        terms.append("C(model)")
    if n_ds > 1:
        terms.append("C(dataset)")
    formula = f"{value} ~ " + " + ".join(terms)
    info["formula"] = formula

    if _HAS_SM and dd["subject_uid"].nunique() > 1:
        try:
            r = smf.mixedlm(formula, dd, groups=dd["subject_uid"]).fit(
                reml=False, method="lbfgs")
            if not bool(getattr(r, "converged", True)):
                raise RuntimeError("MixedLM did not converge")
            info["backend"] = "mixedlm"
            info["params"] = r.params.to_dict()
            info["pvalues"] = r.pvalues.to_dict()
            ci = r.conf_int()
            info["drift_beta"] = float(r.params.get("drift_z", np.nan))
            info["drift_p"] = float(r.pvalues.get("drift_z", np.nan))
            if "drift_z" in ci.index:
                info["drift_ci"] = [float(ci.loc["drift_z", 0]), float(ci.loc["drift_z", 1])]
            return info
        except Exception as e:
            info["mixedlm_error"] = str(e)[:120]
    # fallback: OLS with cluster-robust SE by subject
    if _HAS_SM:
        try:
            r = smf.ols(formula, dd).fit(cov_type="cluster",
                                         cov_kwds={"groups": dd["subject_uid"]})
            info["backend"] = "ols_cluster"
            info["params"] = r.params.to_dict()
            info["pvalues"] = r.pvalues.to_dict()
            info["drift_beta"] = float(r.params.get("drift_z", np.nan))
            info["drift_p"] = float(r.pvalues.get("drift_z", np.nan))
        except Exception as e:
            info["ols_error"] = str(e)[:120]
    return info


def fit_mechanism_model(d: pd.DataFrame, value="delta"):
    """Secondary model exposing the leak mechanism:
    Delta ~ drift_z + sessions_after + C(model) + C(dataset), RE = subject.
    sessions_after = number of future sessions LOSO uses that forward cannot."""
    need = [value, "drift_z", "sessions_after"]
    dd = d.dropna(subset=need).copy()
    info = {"n": int(len(dd))}
    terms = ["drift_z", "sessions_after"]
    if dd["model"].nunique() > 1:
        terms.append("C(model)")
    if dd["dataset"].nunique() > 1:
        terms.append("C(dataset)")
    formula = f"{value} ~ " + " + ".join(terms)
    info["formula"] = formula
    if _HAS_SM and dd["subject_uid"].nunique() > 1:
        try:
            r = smf.mixedlm(formula, dd, groups=dd["subject_uid"]).fit(reml=False, method="lbfgs")
            if not bool(getattr(r, "converged", True)):
                raise RuntimeError("MixedLM did not converge")
            info["backend"] = "mixedlm"
            for k in ["drift_z", "sessions_after"]:
                info[f"{k}_beta"] = float(r.params.get(k, np.nan))
                info[f"{k}_p"] = float(r.pvalues.get(k, np.nan))
            info["params"] = r.params.to_dict()
            info["pvalues"] = r.pvalues.to_dict()
            return info
        except Exception as e:
            info["mixedlm_error"] = str(e)[:120]
    if _HAS_SM:
        try:
            r = smf.ols(formula, dd).fit(cov_type="cluster", cov_kwds={"groups": dd["subject_uid"]})
            info["backend"] = "ols_cluster"
            for k in ["drift_z", "sessions_after"]:
                info[f"{k}_beta"] = float(r.params.get(k, np.nan))
                info[f"{k}_p"] = float(r.pvalues.get(k, np.nan))
        except Exception as e:
            info["ols_error"] = str(e)[:120]
    return info


def per_dataset_drift_slopes(d: pd.DataFrame, value="delta"):
    """Per-dataset OLS slope of Delta on drift_z (cluster-robust by subject)."""
    rows = []
    for ds, g in d.groupby("dataset"):
        gg = g.dropna(subset=[value, "drift_z"])
        rec = {"dataset": ds, "n": len(gg), "n_subjects": gg["subject_uid"].nunique(),
               "mean_delta": float(gg[value].mean()) if len(gg) else np.nan}
        if _HAS_SM and len(gg) >= 6 and gg["drift_z"].nunique() > 1 and gg["subject_uid"].nunique() > 1:
            try:
                r = smf.ols(f"{value} ~ drift_z", gg).fit(
                    cov_type="cluster", cov_kwds={"groups": gg["subject_uid"]})
                rec["slope"] = float(r.params["drift_z"])
                rec["slope_p"] = float(r.pvalues["drift_z"])
            except Exception:
                rec["slope"] = np.nan; rec["slope_p"] = np.nan
        else:
            rec["slope"] = np.nan; rec["slope_p"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --- Transfer-matrix symmetry: matched donor-direction comparison -------------
def transfer_symmetry(tm_df: pd.DataFrame):
    """Directional information test at matched training size (one session).

    For each (dataset, subject, model, TEST session t, k) we pair the future
    donor (train on t+k) against the past donor (train on t-k), both scored on
    the SAME test session t. Pairing within t removes the test-session-difficulty
    confound (future donors necessarily test earlier sessions, past donors later
    ones). future_minus_past > 0 would mean a future session is more informative
    about t than an equidistant past session at equal training size. This does
    not by itself distinguish future access from practice, period, or
    session-quality trends."""
    d = tm_df.dropna(subset=["auc"]).copy()
    d["abslag"] = d["lag"].abs()
    d["dir"] = np.where(d["lag"] > 0, "future", "past")
    # index includes test_session so future/past share the same target t
    wide = d.pivot_table(index=["dataset", "subject", "model", "test_session", "abslag"],
                         columns="dir", values="auc").reset_index()
    wide = wide.dropna(subset=["future", "past"])
    wide["future_minus_past"] = wide["future"] - wide["past"]
    res = {"n_pairs": int(len(wide)),
           "mean_future_minus_past": float(wide["future_minus_past"].mean()) if len(wide) else np.nan}
    if len(wide) > 1:
        t, p = sps.ttest_1samp(wide["future_minus_past"].values, 0.0)
        res["t"] = float(t); res["p_twosided"] = float(p)
        # subject-level to avoid pseudo-replication
        subj = wide.groupby(d["subject"].name if False else ["dataset", "subject"])["future_minus_past"].mean()
        if subj.shape[0] > 1:
            ts, ps = sps.ttest_1samp(subj.values, 0.0)
            res["subj_t"] = float(ts); res["subj_p_twosided"] = float(ps)
            res["n_subjects"] = int(subj.shape[0])
    return res, wide

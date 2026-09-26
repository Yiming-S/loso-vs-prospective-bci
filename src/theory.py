"""Toy comparison of LOSO and forward mean estimators under parameter drift.

Toy mean-estimation model. The session-specific
parameter follows a drift process

    theta_t = theta_{t-1} + mu + eps_t,     eps_t ~ N(0, tau^2 I)          (RW + trend)

and each session yields a noisy finite-sample estimate

    hat_theta_j^obs = theta_j + eta_j,      eta_j ~ N(0, sigma^2 I).

A pooled decoder trained on a set S of sessions estimates theta_t by the
(equally weighted) mean  hat_theta_t^S = mean_{j in S} hat_theta_j^obs. Then, per
dimension,

    E|| hat_theta_t^S - theta_t ||^2
        = (tau^2 / |S|^2) * sum_{j,k in S} overlap(j,k)      # drift tracking
        + sigma^2 / |S|                                       # estimation variance
        + (trend bias term, zero when mu = 0 OR when S is symmetric about t)

where overlap(j,k) = number of shared random-walk increments on the paths from t
to j and to k: min(|t-j|,|t-k|) if j,k are the same side of t, else 0. This
decomposition is verified against Monte Carlo to <0.3% relative error.

Two forces bear on the LOSO-vs-forward comparison, and only ONE uniformly favours
LOSO:
  * variance (ALWAYS favours LOSO): |S_LOSO| = T-1 >= |S_forward| = t, so
    sigma^2/|S| is smaller for LOSO. Pure data-quantity effect.
  * drift tracking + trend bias (favour EITHER protocol, depending on t): LOSO's
    donor set is symmetric about t only near the timeline midpoint, where
    past/future straddle pairs have overlap 0 and drift error partly cancels. At
    the cold-start target t=1, LOSO's donors sit almost entirely in the future
    (far from t), so its drift/bias terms are LARGER than one-sided forward's --
    i.e. these terms favour FORWARD there. Any LOSO advantage observed at t=1 must
    therefore come from the variance term (data quantity), not drift coverage.

Within this model, the per-t shape (``analytic_gap_per_t``) is
monotone-decreasing from a t=1 peak for some parameter choices and becomes an
inverted-U that is negative at t=1 for others. The selected Monte-Carlo grid
illustrates cases with aggregate
E[MSE_LOSO] < E[MSE_forward] (averaged over interior t), but this is not a
universal statement and is not used to infer the empirical drift regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pooled_mse(theta, obs, t, S):
    est = obs[S].mean(axis=0)
    return float(np.sum((est - theta[t]) ** 2))


def _overlap(j, k, t):
    """Shared random-walk increments on the paths t->j and t->k."""
    return min(abs(t - j), abs(t - k)) if (j - t) * (k - t) >= 0 else 0


def analytic_mse(S, t, tau, sigma, mu):
    """Closed-form per-coordinate E||hat_theta_t^S - theta_t||^2 for the model
    above: drift-variance + estimation-variance + squared trend bias. Verified
    against Monte Carlo to <0.3% relative error (see TEST/verify_theory)."""
    S = list(S)
    n = len(S)
    drift = sum(_overlap(j, k, t) for j in S for k in S) * tau ** 2 / n ** 2
    est = sigma ** 2 / n
    bias = (mu * float(np.mean([j - t for j in S]))) ** 2
    return drift + est + bias


def analytic_gap_per_t(T, tau, sigma, mu):
    """Analytic forward-MSE minus LOSO-MSE difference at each interior target t.

    When estimation variance dominates for this mean estimator, the gap can peak
    at t=1; for other parameter choices the curve can become an inverted U and be
    negative at t=1. This sensitivity plot is not a regime diagnostic for the EEG
    results: the empirical decoders and their AUC are not identified by the toy
    estimator's MSE curve."""
    rows = []
    for t in range(1, T - 1):
        S_loso = [j for j in range(T) if j != t]
        S_fwd = list(range(0, t))
        gap = analytic_mse(S_fwd, t, tau, sigma, mu) - analytic_mse(S_loso, t, tau, sigma, mu)
        rows.append({"t": t, "gap": gap})
    return pd.DataFrame(rows)


def simulate(T=8, p=20, tau=0.3, sigma=0.5, mu=0.0, reps=4000, seed=0):
    """Monte-Carlo E[MSE] for LOSO vs forward-expanding pooled estimators,
    averaged over interior targets t = 1..T-2 (where the protocols differ)."""
    rng = np.random.default_rng(seed)
    mu_vec = np.full(p, mu) if np.ndim(mu) == 0 else np.asarray(mu)
    rows = []
    for _ in range(reps):
        eps = rng.normal(0, tau, size=(T, p))
        theta = np.zeros((T, p))
        for j in range(1, T):
            theta[j] = theta[j - 1] + mu_vec + eps[j]
        obs = theta + rng.normal(0, sigma, size=(T, p))
        for t in range(1, T - 1):        # interior targets: both protocols defined and differ
            S_loso = [j for j in range(T) if j != t]
            S_fwd = list(range(0, t))
            rows.append((t, _pooled_mse(theta, obs, t, S_loso),
                         _pooled_mse(theta, obs, t, S_fwd)))
    df = pd.DataFrame(rows, columns=["t", "mse_loso", "mse_forward"])
    return df


def summarize(df: pd.DataFrame):
    g = df.mean(numeric_only=True)
    out = {
        "mse_loso": float(g["mse_loso"]),
        "mse_forward": float(g["mse_forward"]),
        "delta_mse": float(g["mse_forward"] - g["mse_loso"]),
        "ratio": float(g["mse_forward"] / g["mse_loso"]),
        "frac_reps_loso_better": float((df["mse_loso"] < df["mse_forward"]).mean()),
    }
    per_t = df.groupby("t")[["mse_loso", "mse_forward"]].mean()
    per_t["delta"] = per_t["mse_forward"] - per_t["mse_loso"]
    return out, per_t


def run_grid(seed=0):
    """Return a tidy table over drift regimes."""
    configs = [
        dict(name="random_walk",       tau=0.3, sigma=0.5, mu=0.0),
        dict(name="random_walk_strong", tau=0.6, sigma=0.5, mu=0.0),
        dict(name="trend",             tau=0.3, sigma=0.5, mu=0.15),
        dict(name="near_stationary",   tau=0.05, sigma=0.5, mu=0.0),
    ]
    recs = []
    for c in configs:
        df = simulate(tau=c["tau"], sigma=c["sigma"], mu=c["mu"], seed=seed)
        s, _ = summarize(df)
        recs.append({**c, **s})
    return pd.DataFrame(recs)


# ===========================================================================
# Bridge to the real classifier + metric: tangent-space logistic regression, AUC
# ===========================================================================
# The parameter-MSE result above concerns a *mean estimator*. The actual study
# uses a tangent-space logistic decoder and scores AUC. This section shows the
# whether qualitatively similar behavior can occur in one controlled generative
# example. It is an illustration, not a theorem about the empirical pipeline.
#
# Model. After the AIRM tangent-space map, each trial is a feature x in R^p.
# Class-conditional Gaussians with shared covariance (take identity WLOG after
# whitening): x | (session s, class c) ~ N(sign_c * d_s / 2, I), c in {0,1}. The
# discriminative direction d_s = mu_{s,1} - mu_{s,0} drifts across sessions,
#     d_s = d_{s-1} + trend + N(0, tau^2 I),
# exactly the random-walk-with-trend of the parameter theory (now on the LDA
# direction). A decoder trained on a session set S sees finite samples, so its
# fitted weight estimates d_t with the same pooled-mean error structure; and for
# a linear decoder on Gaussian classes the test AUC depends on the signed
# alignment and scale of the fitted weight with the true d_t. Parameter MSE alone
# does not order AUC in general. The simulation below therefore supplies only a
# model-specific numerical bridge by fitting sklearn LogisticRegression under
# LOSO / forward / size-matched protocols; it cannot transfer the MSE result to
# empirical AUC as a general monotonic implication.
def simulate_tslr(T=12, p=10, n_per_class=30, tau=0.15, trend=0.0, sep=1.2,
                  reps=60, seed=0):
    """Fit real logistic regression on drifting tangent-space Gaussian features
    under LOSO / forward / size-matched protocols; return per-protocol AUC.

    Decomposes the synthetic gap using the same bookkeeping as the empirical
    study. The output is sensitive to the chosen generator parameters and must
    not be used to identify an empirical drift value or mechanism share."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n_draws = 5
    rows = []
    for _ in range(reps):
        # drifting discriminative direction d_s (unit base direction * sep)
        base = rng.normal(size=p)
        base = sep * base / np.linalg.norm(base)
        d = np.zeros((T, p))
        d[0] = base
        trend_vec = np.full(p, trend) if np.ndim(trend) == 0 else np.asarray(trend)
        for s in range(1, T):
            d[s] = d[s - 1] + trend_vec + rng.normal(0, tau, size=p)

        def gen(s, n):
            y = np.r_[np.zeros(n), np.ones(n)].astype(int)
            X = np.vstack([rng.normal(-d[s] / 2, 1.0, size=(n, p)),
                           rng.normal(+d[s] / 2, 1.0, size=(n, p))])
            return X, y

        train = {s: gen(s, n_per_class) for s in range(T)}
        test = {s: gen(s, 200) for s in range(T)}  # large test set for stable AUC

        def fit_auc(Ssess, t):
            Xtr = np.vstack([train[j][0] for j in Ssess])
            ytr = np.concatenate([train[j][1] for j in Ssess])
            if len(np.unique(ytr)) < 2:
                return np.nan
            clf = LogisticRegression(penalty="l2", max_iter=5000)
            clf.fit(Xtr, ytr)
            Xte, yte = test[t]
            return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

        for t in range(1, T - 1):
            loso = [j for j in range(T) if j != t]
            fwd = list(range(0, t))
            # size-matched: draw t sessions from LOSO pool / from strict future
            pool = [j for j in range(T) if j != t]
            fut_pool = list(range(t + 1, T))
            lm = np.mean([fit_auc(sorted(rng.choice(pool, t, replace=False)), t)
                          for _ in range(n_draws)])
            fm = (np.mean([fit_auc(sorted(rng.choice(fut_pool, t, replace=False)), t)
                           for _ in range(n_draws)]) if len(fut_pool) >= t else np.nan)
            rows.append(dict(t=t, tau=tau,
                             auc_loso=fit_auc(loso, t), auc_forward=fit_auc(fwd, t),
                             auc_loso_matched=lm, auc_future_matched=fm))
    df = pd.DataFrame(rows)
    return df


def tslr_decomposition(df):
    """Total / size / look-ahead AUC decomposition from a simulate_tslr frame."""
    d = df.dropna(subset=["auc_loso", "auc_forward", "auc_loso_matched"])
    total = (d["auc_loso"] - d["auc_forward"]).mean()
    size = (d["auc_loso"] - d["auc_loso_matched"]).mean()
    look = (d["auc_loso_matched"] - d["auc_forward"]).mean()
    fut = (df.dropna(subset=["auc_future_matched", "auc_forward"]).eval(
        "auc_future_matched - auc_forward")).mean()
    return {"total": float(total), "size": float(size), "look_ahead": float(look),
            "future_only": float(fut), "size_share": float(size / total) if total else np.nan}

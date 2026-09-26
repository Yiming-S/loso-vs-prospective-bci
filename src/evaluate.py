"""Per-(dataset, subject) evaluation across models x protocols.

Produces one long-format row per (dataset, subject, model, protocol, target
session t): AUC on the held-out session, plus bookkeeping (n train sessions /
trials). Drift D_t is attached per (subject, t). Everything a fresh clone per fit.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .data import load_subject_sessions, describe_sessions
from .drift import cached_session_drift
from .models import build_model
from .splitters import protocol_train_sets, MATCHED_PROTOCOLS
from .config import PROTOCOLS, MATCHED_DRAWS, RANDOM_STATE

warnings.filterwarnings("ignore")
try:
    import mne
    mne.set_log_level("ERROR")
except Exception:
    pass


def _fit_score(model_name, n_ch, X_tr, y_tr, X_te, y_te,
               random_state=RANDOM_STATE):
    """Fit a fresh model, return test AUC (or NaN on failure)."""
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return np.nan
    est = build_model(model_name, n_ch, random_state=random_state)
    try:
        est = clone(est) if model_name != "eegnet" else est  # eegnet clone-safe already
    except Exception:
        pass
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_te)
    # positive class = second sorted class (index 1)
    return float(roc_auc_score(y_te, proba[:, 1]))


def evaluate_subject(dataset_name, subject, models, protocols=PROTOCOLS,
                     resample=None, random_state=RANDOM_STATE, log=print,
                     compute_drift=True):
    """Return (rows_df, meta_dict) for one subject. resample=None uses config default."""
    kw = {} if resample is None else {"resample": resample}
    sessions = load_subject_sessions(dataset_name, subject, **kw)
    T = len(sessions)
    n_ch = sessions[0]["X"].shape[1]
    if T < 2:
        log(f"[{dataset_name} S{subject}] only {T} session(s); skipped")
        return pd.DataFrame(), {"skipped": True, "T": T}

    log(f"[{dataset_name} S{subject}] {describe_sessions(sessions)}")

    # drift trajectory (once per subject)
    drift = (cached_session_drift(dataset_name, subject, sessions)
             if compute_drift else {})  # drift is not needed for seed-only ranking checks

    Xs = [s["X"] for s in sessions]
    ys = [s["y"] for s in sessions]

    rows = []
    for model_name in models:
        t0 = time.perf_counter()
        for protocol in protocols:
            if protocol in MATCHED_PROTOCOLS:
                # Size-matched contrast: average AUC over random draws of exactly
                # t training sessions, so the comparison against forward_expanding
                # holds training size fixed and varies only the allowed pool.
                gen = MATCHED_PROTOCOLS[protocol]
                acc = {}
                for t, _d, train_idx in gen(T, n_draws=MATCHED_DRAWS,
                                            seed=random_state + int(subject)):
                    X_tr = np.concatenate([Xs[j] for j in train_idx], axis=0)
                    y_tr = np.concatenate([ys[j] for j in train_idx], axis=0)
                    auc = _fit_score(model_name, n_ch, X_tr, y_tr, Xs[t], ys[t],
                                     random_state=random_state)
                    a = acc.setdefault(t, {"aucs": [], "ntr": []})
                    a["aucs"].append(auc)
                    a["ntr"].append(len(y_tr))
                for t, a in acc.items():
                    rows.append(dict(
                        dataset=dataset_name, subject=int(subject), model=model_name,
                        protocol=protocol, target_session=int(t),
                        n_train_sessions=int(t),
                        n_train_trials=float(np.mean(a["ntr"])),
                        n_test_trials=int(len(ys[t])),
                        drift=float(drift.get(t, np.nan)),
                        auc=float(np.mean(a["aucs"])),
                        auc_sd_draws=float(np.std(a["aucs"])),
                        n_draws=len(a["aucs"]),
                        seed=int(random_state),
                    ))
                continue
            for t, train_idx in protocol_train_sets(protocol, T):
                X_tr = np.concatenate([Xs[j] for j in train_idx], axis=0)
                y_tr = np.concatenate([ys[j] for j in train_idx], axis=0)
                X_te, y_te = Xs[t], ys[t]
                auc = _fit_score(model_name, n_ch, X_tr, y_tr, X_te, y_te,
                                 random_state=random_state)
                rows.append(dict(
                    dataset=dataset_name, subject=int(subject), model=model_name,
                    protocol=protocol, target_session=int(t),
                    n_train_sessions=len(train_idx), n_train_trials=int(len(y_tr)),
                    n_test_trials=int(len(y_te)),
                    drift=float(drift.get(t, np.nan)), auc=auc,
                    seed=int(random_state),
                ))
        log(f"    {model_name}: {time.perf_counter() - t0:.1f}s")
    return pd.DataFrame(rows), {"skipped": False, "T": T, "n_ch": n_ch}


def transfer_matrix_subject(dataset_name, subject, models, resample=None, log=print):
    """Single-session -> single-session transfer AUC.

    For every ordered pair (train session j, test session t), fit on session j
    alone and score AUC on session t. This estimates donor-direction differences
    at a fixed training size of one session: comparing lag +k (a future
    donor, j = t+k) against lag -k (a past donor, j = t-k) removes the
    data-quantity confound that the expanding-window protocols carry, but can
    retain temporal practice/quality trends. Returns a
    long df with columns dataset, subject, model, train_session, test_session,
    lag (=j-t), auc. Intended for the cheap classical models only.
    """
    kw = {} if resample is None else {"resample": resample}
    sessions = load_subject_sessions(dataset_name, subject, **kw)
    T = len(sessions)
    n_ch = sessions[0]["X"].shape[1]
    if T < 2:
        return pd.DataFrame()
    Xs = [s["X"] for s in sessions]
    ys = [s["y"] for s in sessions]
    rows = []
    for model_name in models:
        for j in range(T):
            for t in range(T):
                if j == t:
                    continue
                auc = _fit_score(model_name, n_ch, Xs[j], ys[j], Xs[t], ys[t])
                rows.append(dict(
                    dataset=dataset_name, subject=int(subject), model=model_name,
                    train_session=j, test_session=t, lag=int(j - t), auc=auc,
                ))
    return pd.DataFrame(rows)


def gap_table(long_df, ref="forward_expanding"):
    """Wide optimism-gap table: one row per (dataset, subject, model, t) with
    auc_loso, auc_<ref>, delta = auc_loso - auc_ref, and drift. t>=1 only."""
    d = long_df[long_df["target_session"] >= 1].copy()
    keys = ["dataset", "subject", "model", "target_session", "drift"]
    piv = d.pivot_table(index=keys, columns="protocol", values="auc").reset_index()
    if "loso" not in piv or ref not in piv:
        return pd.DataFrame()
    piv = piv.rename_axis(None, axis=1)
    piv["delta"] = piv["loso"] - piv[ref]
    return piv

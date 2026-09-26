"""Session-level covariance drift metric.

D_{i,t} = d_R( Cbar_{i,t}, Cbar_{i,t-1} )

where Cbar is the affine-invariant Frechet (geometric) mean of the trial
covariance matrices in a session and d_R is the affine-invariant Riemannian
(AIRM) geodesic distance.

The implementation calls pyriemann's tested mean-covariance and distance
functions directly rather than maintaining a separate SPD-geometry routine.
"""
from __future__ import annotations

import json
import os

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann

from .config import CACHE_DIR, COV_ESTIMATOR, PREPROCESS_VERSION


def trial_covariances(X: np.ndarray, estimator: str = COV_ESTIMATOR) -> np.ndarray:
    """(n_trials, n_ch, n_times) -> (n_trials, n_ch, n_ch) SPD covariances."""
    return Covariances(estimator=estimator).transform(X)


def frechet_mean_cov(X: np.ndarray, estimator: str = COV_ESTIMATOR) -> np.ndarray:
    """Affine-invariant Frechet mean covariance of one session's trials."""
    covs = trial_covariances(X, estimator)
    return mean_covariance(covs, metric="riemann")


def riemann_drift(X_prev: np.ndarray, X_curr: np.ndarray,
                  estimator: str = COV_ESTIMATOR) -> float:
    """AIRM distance between the Frechet-mean covariances of two sessions."""
    Cp = frechet_mean_cov(X_prev, estimator)
    Cc = frechet_mean_cov(X_curr, estimator)
    return float(distance_riemann(Cp, Cc))


def session_drift_trajectory(sessions, estimator: str = COV_ESTIMATOR):
    """Per-session drift D_t = d_R(Cbar_t, Cbar_{t-1}) for t >= 1.

    sessions : ordered list of dicts each with key 'X' = (n_trials, n_ch, n_times).
    Returns dict {t (int, 1..T-1): drift}. t indexes the *target* session.
    Frechet means are cached so each session mean is computed once.
    """
    means = [frechet_mean_cov(s["X"], estimator) for s in sessions]
    out = {}
    for t in range(1, len(sessions)):
        out[t] = float(distance_riemann(means[t - 1], means[t]))
    return out


def cached_session_drift(dataset: str, subject: int, sessions,
                         estimator: str = COV_ESTIMATOR):
    """Load or compute the preprocessing-versioned drift trajectory.

    The AIRM Frechet means are expensive for 62-channel trajectories. Drift is a
    deterministic property of the cached epochs and is shared by the main and
    size-matched protocol runs, so persist it once per dataset/participant.
    A process-specific temporary file followed by an atomic replace prevents a
    partial JSON file if a run is interrupted.
    """
    path = CACHE_DIR / (
        f"drift_{dataset}_S{int(subject)}_{PREPROCESS_VERSION}_{estimator}.json")
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            values = {int(k): float(v) for k, v in payload["drift"].items()}
            if (payload.get("preprocess_version") == PREPROCESS_VERSION and
                    set(values) == set(range(1, len(sessions))) and
                    all(np.isfinite(list(values.values())))):
                return values
        except Exception:
            pass
    values = session_drift_trajectory(sessions, estimator)
    payload = {"dataset": dataset, "subject": int(subject),
               "preprocess_version": PREPROCESS_VERSION,
               "estimator": estimator, "drift": values}
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return values

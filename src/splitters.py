"""Session-level evaluation protocols.

Given T sessions ordered chronologically s_0, ..., s_{T-1}, each protocol maps a
*target* session t to the set of *training* sessions:

  loso               train = {all sessions except t}     test = {t}   (t = 0..T-1)
  forward_expanding  train = {0, ..., t-1}               test = {t}   (t = 1..T-1)
  last_session       train = {t-1}                       test = {t}   (t = 1..T-1)
  first_session      train = {0}                          test = {t}   (t = 1..T-1)

LOSO is MOABB's default cross-session evaluation: to score an early session it is
allowed to train on *later* sessions. The three forward protocols respect
chronological availability. "forward_expanding" is what the proposal calls
forward chaining (a.k.a. expanding-window forward chaining) and is the reference
against which the optimism gap Delta = AUC_loso - AUC_forward_expanding is defined.
"last_session" (calibrate on the most recent session only) and "first_session"
(calibrate once and deploy forever) are the two operationally cheaper extremes.

The LOSO--forward difference is only defined for target sessions t >= 1, because the forward
protocols cannot score session 0 (it has no past). So all cross-protocol
comparisons are restricted to t in 1..T-1.

`ForwardSessionSplitter` is a scikit-learn / MOABB-compatible cross-validator: it
consumes the MOABB `metadata` frame's 'session' column and yields (train_idx,
test_idx) index arrays, so it can be dropped into MOABB's CrossSessionEvaluation
via cv_class / cv_kwargs.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin


# --- Protocol -> training-session sets ---------------------------------------
def protocol_train_sets(protocol: str, T: int):
    """Yield (t, train_session_indices) for a protocol over T ordered sessions."""
    if protocol == "loso":
        for t in range(T):
            train = [j for j in range(T) if j != t]
            if train:
                yield t, train
    elif protocol == "forward_expanding":
        for t in range(1, T):
            yield t, list(range(0, t))
    elif protocol == "last_session":
        for t in range(1, T):
            yield t, [t - 1]
    elif protocol == "first_session":
        for t in range(1, T):
            yield t, [0]
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")


# --- Size-matched contrasts (separate size from sampled composition) ----------
# The plain LOSO-vs-forward gap confounds two things: at target t, LOSO trains on
# T-1 sessions while forward trains on t, so LOSO enjoys BOTH a look-ahead pool
# and up to 14x more training data. Delta therefore cannot, on its own, be
# attributed to look-ahead. These two protocols hold the training-set SIZE fixed
# at exactly t sessions -- forward's size -- and vary only WHICH sessions are
# allowed. This removes the size difference, while the remaining composition
# effect can still reflect practice or session-quality trends.
#
#   loso_matched   : draw t sessions from LOSO's pool {all}\{t}  (past AND future)
#   future_matched : draw t sessions from the strict future {t+1..T-1}
#
# Both are averaged over `n_draws` random draws. Note loso_matched degenerates to
# forward_expanding at t = T-1 (the pool is then exactly {0..T-2} and k = T-1), so
# it reproduces the same structural zero -- a useful internal consistency check.
def matched_loso_train_sets(T: int, n_draws: int = 5, seed: int = 0):
    """Yield (t, draw, train) drawing t sessions from {0..T-1} \\ {t}."""
    rng = np.random.default_rng(seed)
    for t in range(1, T):
        pool = [j for j in range(T) if j != t]
        if t > len(pool):
            continue
        for d in range(n_draws):
            yield t, d, sorted(rng.choice(pool, size=t, replace=False).tolist())


def future_matched_train_sets(T: int, n_draws: int = 5, seed: int = 0):
    """Yield (t, draw, train) drawing t sessions from the strict future of t.

    Only defined while the future is big enough (T-1-t >= t, i.e. t <= (T-1)/2).
    Paired against forward_expanding this is a same-size past-vs-future contrast.
    """
    rng = np.random.default_rng(seed + 1000)
    for t in range(1, T):
        pool = list(range(t + 1, T))
        if t > len(pool):
            continue
        for d in range(n_draws):
            yield t, d, sorted(rng.choice(pool, size=t, replace=False).tolist())


MATCHED_PROTOCOLS = {
    "loso_matched": matched_loso_train_sets,
    "future_matched": future_matched_train_sets,
}


# --- MOABB / sklearn-compatible splitter -------------------------------------
class ForwardSessionSplitter(GroupsConsumerMixin, BaseCrossValidator):
    """Causal forward-chaining cross-session splitter for MOABB.

    Parameters
    ----------
    mode : {'expanding', 'last', 'first'}
        expanding -> train on all past sessions (0..t-1);
        last      -> train on the immediately preceding session (t-1);
        first     -> train on the first session only (0).
    session_order : list, optional
        Explicit chronological order of session labels. If None, sessions are
        ordered by the first integer in the label, then lexically.

    Usage with MOABB:
        CrossSessionEvaluation(..., cv_class=ForwardSessionSplitter,
                               cv_kwargs={'mode': 'expanding'})
    The splitter reads the per-trial session labels from the `metadata` frame that
    MOABB passes as the 3rd positional argument to split().
    """

    _MODE_TO_PROTOCOL = {
        "expanding": "forward_expanding",
        "last": "last_session",
        "first": "first_session",
    }

    def __init__(self, mode: str = "expanding", session_order=None):
        if mode not in self._MODE_TO_PROTOCOL:
            raise ValueError(f"mode must be one of {list(self._MODE_TO_PROTOCOL)}")
        self.mode = mode
        self.session_order = session_order

    # -- helpers --
    @staticmethod
    def _default_key(label):
        import re
        m = re.search(r"\d+", str(label))
        return (int(m.group()) if m else 10**9, str(label))

    def _ordered_sessions(self, sessions):
        uniq = list(dict.fromkeys(sessions))
        if self.session_order is not None:
            order = [s for s in self.session_order if s in set(uniq)]
            # append any sessions not listed, in default order
            order += [s for s in sorted(uniq, key=self._default_key) if s not in order]
            return order
        return sorted(uniq, key=self._default_key)

    def _sessions_from_args(self, X, y=None, groups=None):
        # Direct use may pass the MOABB metadata frame; MOABB 1.5 wraps this class
        # as an inner sklearn cross-validator and passes the per-trial session labels
        # as `groups`. Supporting both forms keeps the class useful on its own and
        # makes `CrossSessionEvaluation(cv_class=ForwardSessionSplitter)` work.
        if groups is not None and "session" in getattr(groups, "columns", []):
            return np.asarray(groups["session"])
        if groups is not None:
            arr = np.asarray(groups)
            if arr.ndim == 1:
                return arr
        raise ValueError(
            "ForwardSessionSplitter requires per-trial session labels in `groups` "
            "or a metadata frame with a 'session' column.")

    # -- sklearn API --
    def get_n_splits(self, X=None, y=None, groups=None):
        sess = self._sessions_from_args(X, y, groups)
        return max(0, len(np.unique(sess)) - 1)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        sess = self._sessions_from_args(X, y, groups)
        order = self._ordered_sessions(sess.tolist())
        T = len(order)
        protocol = self._MODE_TO_PROTOCOL[self.mode]
        idx_all = np.arange(len(sess))
        for t, train_pos in protocol_train_sets(protocol, T):
            train_labels = {order[j] for j in train_pos}
            test_label = order[t]
            train_idx = idx_all[np.isin(sess, list(train_labels))]
            test_idx = idx_all[sess == test_label]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx

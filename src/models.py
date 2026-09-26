"""The three decoders, all exposing fit / predict_proba on (n_trials, n_ch, n_times).

  csp_lda : mne CSP (log-variance) + shrinkage LDA        (classical, spatial)
  ts_lr   : pyriemann Covariances(lwf) + TangentSpace(riemann) + L2 logistic reg
  eegnet  : a compact, self-contained EEGNet-v4 in torch    (deep, sanity/ranking)

csp_lda and ts_lr are the exact pipelines used in CrossDA
(crossda/pipelines/pipeline_utils.py); we build them from mne/pyriemann/sklearn
directly. EEGNet is implemented here (braindecode is not installed) following
Lawhern et al. 2018.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from .config import COV_ESTIMATOR, RANDOM_STATE


# --- Classical pipelines -----------------------------------------------------
def build_csp_lda(n_channels: int):
    # CSP needs n_components <= n_channels; keep it even and modest.
    n_comp = min(8, n_channels)
    if n_comp % 2 == 1:
        n_comp -= 1
    n_comp = max(2, n_comp)
    csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return make_pipeline(csp, lda)


def build_ts_lr(n_channels: int = None):
    return make_pipeline(
        Covariances(estimator=COV_ESTIMATOR),
        TangentSpace(metric="riemann"),
        LogisticRegression(penalty="l2", max_iter=20000, solver="lbfgs"),
    )


# --- EEGNet ------------------------------------------------------------------
def _torch_device():
    import torch
    requested = os.environ.get("EEGNET_DEVICE", "").strip().lower()
    if requested:
        if requested not in {"cpu", "mps", "cuda"}:
            raise ValueError(
                "EEGNET_DEVICE must be one of cpu, mps, or cuda; "
                f"got {requested!r}")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("EEGNET_DEVICE=mps requested, but MPS is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EEGNET_DEVICE=cuda requested, but CUDA is unavailable")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_eegnet_module(n_ch, n_times, n_classes=2,
                         F1=8, D=2, F2=16, kern_len=64, drop=0.25):
    import torch
    import torch.nn as nn

    kern_len = min(kern_len, n_times)
    sep_len = min(16, n_times)

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
                nn.BatchNorm2d(F1),
                # depthwise spatial conv over channels
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(drop),
            )
            self.block2 = nn.Sequential(
                # separable conv = depthwise (temporal) + pointwise
                nn.Conv2d(F1 * D, F1 * D, (1, sep_len), padding=(0, sep_len // 2),
                          groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(drop),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_ch, n_times)
                flat = self.block2(self.block1(dummy)).flatten(1).shape[1]
            self.classify = nn.Linear(flat, n_classes)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            return self.classify(x)

    return EEGNet()


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style EEGNet. X = (n_trials, n_ch, n_times)."""

    def __init__(self, max_epochs=250, batch_size=32, lr=1e-3, patience=40,
                 val_frac=0.15, random_state=RANDOM_STATE, verbose=False):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state
        self.verbose = verbose

    def _standardize_fit(self, X):
        self.mu_ = X.mean(axis=(0, 2), keepdims=True)
        self.sd_ = X.std(axis=(0, 2), keepdims=True) + 1e-7
        return (X - self.mu_) / self.sd_

    def _standardize(self, X):
        return (X - self.mu_) / self.sd_

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        self.classes_ = np.unique(y)
        n_ch, n_times = X.shape[1], X.shape[2]
        Xs = self._standardize_fit(X)

        # internal validation split for early stopping (stratified best-effort)
        n = len(Xs)
        idx = rng.permutation(n)
        n_val = max(2, int(round(self.val_frac * n)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
            tr_idx, val_idx = idx, idx  # too small to split; validate on train

        dev = _torch_device()
        self.device_ = dev
        net = _build_eegnet_module(n_ch, n_times, n_classes=len(self.classes_)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()

        def to_t(a):
            return torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(1)

        Xtr = to_t(Xs[tr_idx]); ytr = torch.tensor(y[tr_idx], device=dev)
        Xva = to_t(Xs[val_idx]); yva = torch.tensor(y[val_idx], device=dev)

        best_state, best_val, bad = None, np.inf, 0
        bs = self.batch_size
        for ep in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for b in range(0, len(Xtr), bs):
                sel = perm[b:b + bs]
                opt.zero_grad()
                out = net(Xtr[sel])
                loss = lossf(out, ytr[sel])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = lossf(net(Xva), yva).item()
            if vloss < best_val - 1e-4:
                best_val, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        return self

    def predict_proba(self, X):
        import torch
        X = self._standardize(np.asarray(X, dtype=np.float32))
        xt = torch.tensor(X, dtype=torch.float32, device=self.device_).unsqueeze(1)
        with torch.no_grad():
            logits = self.net_(xt)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_eegnet(n_channels: int = None, random_state: int = RANDOM_STATE):
    return EEGNetClassifier(random_state=random_state)


def build_model(name: str, n_channels: int, random_state: int = RANDOM_STATE):
    if name == "csp_lda":
        return build_csp_lda(n_channels)
    if name == "ts_lr":
        return build_ts_lr(n_channels)
    if name == "eegnet":
        return build_eegnet(n_channels, random_state=random_state)
    raise ValueError(f"Unknown model {name!r}")

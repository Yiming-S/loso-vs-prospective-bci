"""MOABB session loading + caching.

Re-expresses the pattern from CrossDA's `_load_subject_sessions`
(crossda/core/workers.py): LeftRightImagery(fmin, fmax).get_data(...) then split
by the metadata['session'] column into per-session trial arrays. We add a common
resample rate (so EEGNet sees a fixed time axis), a robust chronological session
sort, and a per-(dataset, subject) pickle cache.
"""
from __future__ import annotations

import contextlib
import pickle
import re
import warnings
from pathlib import Path

import numpy as np

from .config import (CACHE_DIR, DATASETS, FMIN, FMAX, TMIN, TMAX, RESAMPLE_HZ,
                     PREPROCESS_VERSION, PROPOSAL_ALIASES,
                     LOCAL_EEG_ROOT, MA2020_EVENT_CODES, MA2020_TMIN, MA2020_TMAX,
                     MA2020_CLASS_NAMES, MA2020_NON_EEG_CHANNELS,
                     STIEGER_LOCAL_DIR)

warnings.filterwarnings("ignore")
try:
    import mne
    mne.set_log_level("ERROR")
    import logging
    for _n in ("moabb", "mne", "moabb.datasets", "moabb.paradigms"):
        logging.getLogger(_n).setLevel(logging.ERROR)
except Exception:
    pass


def _session_sort_key(name):
    """Chronological order for MOABB session labels.

    Session labels are strings like '0','1','session_0','0train','1test'. We sort
    by the first integer found (chronology), falling back to the raw string. For
    all datasets used here this equals recording order.
    """
    s = str(name)
    m = re.search(r"\d+", s)
    return (int(m.group()) if m else 10**9, s)


def _resolve(dataset_name: str) -> str:
    return PROPOSAL_ALIASES.get(dataset_name, dataset_name)


@contextlib.contextmanager
def _no_download_guard(active: bool = True):
    """Make any attempted MOABB/pooch download a hard error.

    Used when a dataset is served from the read-only local store: MOABB only
    fetches files it cannot find, so a fetch attempt means the store is
    incomplete. Downloading would write into that store, which this project must
    never do -- so we raise instead.
    """
    if not active:
        yield
        return
    import pooch
    orig = pooch.retrieve

    def _blocked(*a, **k):
        raise RuntimeError(
            "Download blocked: a file is missing from the local read-only store "
            f"(fname={k.get('fname')}). This project never writes there. "
            "Mount the store or exclude this subject.")

    pooch.retrieve = _blocked
    try:
        yield
    finally:
        pooch.retrieve = orig


def _load_ma2020_sessions(subject: int, resample: float | None):
    """Load one Ma2020 subject's 15 sessions from the local .cnt store.

    This audited local-data path follows CrossDA's `_load_ma2020_sessions`
    (crossda/core/workers.py:111): read .cnt with MNE, keep EEG channels, band-pass
    to the MI band, epoch the two MI annotation codes. We additionally resample to
    the project's common rate before epoching -- annotations are stored in seconds,
    so resampling the raw preserves event timing.

    Filenames are `sub-XXX_ses-NN_task-motorimagery_eeg.cnt` with zero-padded NN,
    so plain filename sort is chronological.
    """
    import mne
    mne.set_log_level("ERROR")

    cfg = DATASETS["Ma2020"]
    subj_dir = LOCAL_EEG_ROOT / cfg["subdir"] / f"sub-{subject:03d}"
    if not subj_dir.is_dir():
        raise FileNotFoundError(
            f"Ma2020 subject dir not found: {subj_dir}. Is the SSD mounted?")

    cnt_files = sorted(p for p in subj_dir.iterdir()
                       if p.suffix == ".cnt" and "motorimagery" in p.name.lower())
    if not cnt_files:
        raise FileNotFoundError(f"No Ma2020 .cnt files in {subj_dir}")

    sessions = []
    for cnt in cnt_files:
        raw = mne.io.read_raw_cnt(str(cnt), preload=True, verbose=False)
        # CNT channel types are unreliable here: HEO/VEO/M2 and, for some files,
        # EMG1/EMG2 are all tagged as EEG. Drop them explicitly before selecting
        # EEG so every subject uses the documented 62-channel scalp montage.
        drop = [ch for ch in MA2020_NON_EEG_CHANNELS if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)
        raw.pick("eeg")
        if len(raw.ch_names) != DATASETS["Ma2020"]["n_channels"]:
            raise RuntimeError(
                f"Ma2020 {cnt.name}: expected 62 EEG channels after dropping "
                f"auxiliaries, found {len(raw.ch_names)} ({raw.ch_names})")
        raw.filter(FMIN, FMAX, verbose=False)
        if resample:
            raw.resample(resample, verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        keep = {k: v for k, v in event_id.items() if k in MA2020_EVENT_CODES}
        if len(keep) < 2:
            continue
        epochs = mne.Epochs(raw, events, event_id=keep,
                            tmin=MA2020_TMIN, tmax=MA2020_TMAX,
                            baseline=None, preload=True, verbose=False)
        if len(epochs) < 4:
            continue

        codes = sorted(keep, key=lambda k: MA2020_EVENT_CODES.index(k))
        code_to_idx = {keep[c]: i for i, c in enumerate(codes)}
        y = np.array([code_to_idx[v] for v in epochs.events[:, 2]], dtype=np.int64)
        if len(np.unique(y)) < 2:
            continue

        # ses-NN from the filename is the chronological session label.
        m = re.search(r"ses-(\d+)", cnt.name)
        sessions.append(dict(
            X=epochs.get_data().astype(np.float32),
            y=y,
            session=(m.group(1) if m else cnt.stem),
            classes=list(MA2020_CLASS_NAMES),
            ch_names=list(epochs.ch_names),
            sfreq=float(epochs.info["sfreq"]),
        ))

    if not sessions:
        raise RuntimeError(f"No valid Ma2020 sessions for subject {subject}")
    sessions.sort(key=lambda s: _session_sort_key(s["session"]))
    return sessions


def _finalize_sessions(sessions, resample):
    """Enforce the common time axis and attach auditable preprocessing metadata."""
    if not sessions:
        return sessions
    expected = int(round((TMAX - TMIN) * float(resample))) if resample else None
    n_channels = {int(s["X"].shape[1]) for s in sessions}
    if len(n_channels) != 1:
        raise RuntimeError(f"Channel count changes within subject: {sorted(n_channels)}")
    for s in sessions:
        X = np.asarray(s["X"], dtype=np.float32)
        if expected is not None:
            if X.shape[2] < expected:
                raise RuntimeError(
                    f"Session {s['session']} has {X.shape[2]} samples; expected at least {expected}")
            X = X[:, :, :expected]
        s["X"] = X
        s["preprocess_version"] = PREPROCESS_VERSION
        s["analysis_window_s"] = (TMIN, TMAX)
        s["band_hz"] = (FMIN, FMAX)
        s["sfreq"] = float(resample) if resample else float(s.get("sfreq", np.nan))
    return sessions


def load_subject_sessions(dataset_name: str, subject: int,
                          resample: float | None = RESAMPLE_HZ,
                          cache: bool = True):
    """Return an ordered list of per-session dicts for one subject.

    Each dict: {'X': (n_trials, n_ch, n_times) float32,
                'y': (n_trials,) int {0,1},
                'session': str,
                'classes': [name0, name1]}
    Sorted chronologically. Only the two MI classes are kept.
    """
    dataset_name = _resolve(dataset_name)
    tag = (f"{dataset_name}_S{subject}_rs{int(resample) if resample else 'native'}_"
           f"{PREPROCESS_VERSION}.pkl")
    cache_path = CACHE_DIR / tag
    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            sessions = pickle.load(f)
        if dataset_name == "Ma2020":
            for session in sessions:
                session["classes"] = list(MA2020_CLASS_NAMES)
        return sessions

    # Stieger2021 already used the exact 0--3 s native interval in the previous
    # analysis and its cached arrays are exactly 384 samples. They were produced
    # by the same /opt/anaconda3 environment and preprocessing settings; reusing
    # them avoids re-filtering 600+ hours of continuous raw EEG. Longer legacy
    # epochs are deliberately not migrated because cropping after filtering is
    # not numerically equivalent (verified on BNCI2014_004).
    legacy_path = CACHE_DIR / (
        f"{dataset_name}_S{subject}_rs{int(resample) if resample else 'native'}.pkl")
    expected = int(round((TMAX - TMIN) * float(resample))) if resample else None
    if cache and dataset_name == "Stieger2021" and legacy_path.exists():
        with open(legacy_path, "rb") as f:
            legacy = pickle.load(f)
        if legacy and expected is not None and all(
                s["X"].shape[2] == expected for s in legacy):
            sessions = _finalize_sessions(legacy, resample)
            for session in sessions:
                session["cache_migrated_from"] = legacy_path.name
            with open(cache_path, "wb") as f:
                pickle.dump(sessions, f)
            return sessions

    if DATASETS[dataset_name].get("loader") == "ma2020":
        sessions = _load_ma2020_sessions(subject, resample)
    else:
        import moabb.datasets as mds
        from moabb.paradigms import LeftRightImagery

        ds_cls = getattr(mds, DATASETS[dataset_name]["moabb_cls"])
        ds = ds_cls()
        paradigm = LeftRightImagery(
            fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX, resample=resample)
        # Datasets served from the read-only local store must never be written to.
        local_served = (dataset_name == "Stieger2021" and STIEGER_LOCAL_DIR.is_dir())
        with _no_download_guard(local_served):
            X, y, meta = paradigm.get_data(dataset=ds, subjects=[subject])

        classes = sorted(np.unique(y).tolist())      # ['left_hand','right_hand']
        y_int = np.array([classes.index(v) for v in y], dtype=np.int64)

        sessions = []
        for sess in sorted(meta["session"].unique(), key=_session_sort_key):
            m = (meta["session"] == sess).to_numpy()
            sessions.append(dict(
                X=X[m].astype(np.float32),
                y=y_int[m],
                session=str(sess),
                classes=classes,
            ))

    sessions = _finalize_sessions(sessions, resample)

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(sessions, f)
    return sessions


def subject_list(dataset_name: str):
    dataset_name = _resolve(dataset_name)
    cfg = DATASETS[dataset_name]
    if cfg.get("loader") == "ma2020":
        # Subjects are the sub-XXX directories actually present on the local store.
        root = LOCAL_EEG_ROOT / cfg["subdir"]
        if root.is_dir():
            return sorted(int(p.name.split("-")[1]) for p in root.iterdir()
                          if p.is_dir() and re.fullmatch(r"sub-\d+", p.name))
        # A review clone may contain audited versioned caches but not licensed raw
        # EEG. Infer the available subject identifiers without embedding a
        # machine-specific source-data path in the repository.
        pattern = f"Ma2020_S*_rs{int(RESAMPLE_HZ)}_{PREPROCESS_VERSION}.pkl"
        cached = []
        for path in CACHE_DIR.glob(pattern):
            match = re.search(r"Ma2020_S(\d+)_", path.name)
            if match:
                cached.append(int(match.group(1)))
        if cached:
            return sorted(set(cached))
        raise FileNotFoundError(
            f"Ma2020 store not found under {root}. Set LOCAL_EEG_ROOT to the "
            "licensed source-data directory or populate the audited caches.")
    import moabb.datasets as mds
    ds = getattr(mds, DATASETS[dataset_name]["moabb_cls"])()
    return list(ds.subject_list)


def describe_sessions(sessions) -> str:
    T = len(sessions)
    n_ch = sessions[0]["X"].shape[1]
    n_t = sessions[0]["X"].shape[2]
    trials = [s["X"].shape[0] for s in sessions]
    return f"T={T} sessions, {n_ch}ch, {n_t} samples, trials/session={trials}"

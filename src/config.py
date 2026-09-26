"""Central configuration: dataset registry, band-pass, resampling, paths."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = ROOT / "report"
for _d in (CACHE_DIR, RESULTS_DIR, RAW_DIR, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Data policy: this project only ever *writes* inside its own folder. New MOABB
# downloads therefore go into cache/mne_data. Where a dataset is already present
# in a MOABB-readable store on disk, we point that dataset's path env var at it so
# it is *read* (never re-downloaded, never rewritten) instead of re-fetched.
MNE_DATA_DIR = CACHE_DIR / "mne_data"
MNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MNE_DATA"] = str(MNE_DATA_DIR)
os.environ["MOABB_RESULTS"] = str(CACHE_DIR / "moabb_results")

# mne.get_config prefers env vars over its json config; the json on this machine
# points several datasets at paths outside this project (and some non-existent),
# so we override every relevant dataset path to the project cache. Nothing is ever
# downloaded or written outside this folder; datasets are (re)fetched on demand.
_DATASET_SIGNS = [
    "BNCI", "ZHOU2016", "ZHOU2020", "KUMAR2024",
    "MA2020", "STIEGER2021",
]
for _sign in _DATASET_SIGNS:
    os.environ[f"MNE_DATASETS_{_sign}_PATH"] = str(MNE_DATA_DIR)

# Local, already-downloaded EEG store (read-only). Nothing is ever written here.
LOCAL_EEG_ROOT = Path(os.environ.get(
    "LOCAL_EEG_ROOT", str(ROOT / "external_data")))

# Stieger2021 (352 GB) is already on the local store. MOABB builds its path as
# <MNE_DATASETS_STIEGER2021_PATH>/MNE-Stieger2021-data/S<sub>_Session_<n>.mat and
# only calls pooch when a file is ABSENT -- so pointing at an intact local store
# means every file is read and none is written. data.py additionally hard-blocks
# pooch during loading, so a missing file raises instead of downloading onto the
# SSD. If the store is not mounted we leave the project cache path in place.
STIEGER_LOCAL_DIR = LOCAL_EEG_ROOT / "MNE-Stieger2021-data"
if STIEGER_LOCAL_DIR.is_dir():
    os.environ["MNE_DATASETS_STIEGER2021_PATH"] = str(LOCAL_EEG_ROOT)

# --- Signal preprocessing ----------------------------------------------------
FMIN, FMAX = 8.0, 35.0          # motor-imagery mu/beta band
TMIN, TMAX = 0.0, 3.0           # common task-relative analysis window (seconds)
RESAMPLE_HZ = 128.0             # common rate so EEGNet sees a fixed time axis
PREPROCESS_VERSION = "v2_8-35Hz_0-3s_62eeg"
COV_ESTIMATOR = "lwf"           # Ledoit-Wolf shrinkage covariance (pyriemann)

# --- Dataset registry --------------------------------------------------------
# "loader" selects the ingest path:
#   "moabb"  -> moabb.datasets.<moabb_cls> via LeftRightImagery
#   "ma2020" -> local .cnt files read directly with MNE to enforce the audited
#               62-channel montage before preprocessing
# All are binary motor imagery.
# Ordered roughly by size so the runner does cheap datasets first.
DATASETS = {
    "BNCI2014_004": dict(loader="moabb", moabb_cls="BNCI2014_004", n_channels=3,  note="9 subj, 5 sessions, 3 bipolar ch"),
    "Zhou2016":     dict(loader="moabb", moabb_cls="Zhou2016",     n_channels=14, note="4 subj, 3 sessions"),
    "Zhou2020":     dict(loader="moabb", moabb_cls="Zhou2020",     n_channels=None,
                         note="20 subj, 6/7 sessions, 41/26 EEG ch"),
    "Kumar2024":    dict(loader="moabb", moabb_cls="Kumar2024",    n_channels=None, note="18 subj, 6 sessions"),
    # Ma2020: the long-trajectory dataset. 25 subj x 15 sessions x 62 EEG ch @1000 Hz,
    # binary MI (annotation codes "1"/"2"), ~40 trials/session, .cnt format.
    # 15 sessions gives 13 non-structural interior targets per subject. Read from
    # LOCAL_EEG_ROOT.
    "Ma2020":       dict(loader="ma2020", moabb_cls=None, n_channels=62,
                         subdir="MNE-ma2020-data", n_subjects=25,
                         note="25 subj, 15 sessions, 62 EEG ch (local SSD)"),
    "Stieger2021":  dict(loader="moabb", moabb_cls="Stieger2021",  n_channels=60,
                         note="62 subj, 7/11 sessions, 60 retained EEG ch"),
}

# Dataset names are never silently aliased. Zhou2016 and Zhou2020 are distinct
# datasets and both are included so the provenance of every result stays explicit.
PROPOSAL_ALIASES = {}

PRIMARY_DATASETS = [
    "BNCI2014_004", "Zhou2016", "Zhou2020", "Kumar2024", "Ma2020",
    "Stieger2021",
]
HEAVY_DATASETS = ["Zhou2020", "Ma2020", "Stieger2021"]

# Ma2020 event annotation codes -> class index. The dataset is binary MI; the
# .cnt annotations carry the literal codes "1" and "2".
MA2020_EVENT_CODES = ("1", "2")
MA2020_CLASS_NAMES = ("right_hand", "right_elbow")
MA2020_TMIN, MA2020_TMAX = 0.0, 3.0
# MNE's CNT reader marks every channel as EEG in these files. These names are
# auxiliary/reference channels in the Ma2020 release and must be removed to match
# the 62-channel EEG montage used by the official MOABB loader.
MA2020_NON_EEG_CHANNELS = ("HEO", "VEO", "M2", "EMG1", "EMG2")

# --- Models & protocols ------------------------------------------------------
MODELS = ["csp_lda", "ts_lr", "eegnet"]
CLASSICAL_MODELS = ["csp_lda", "ts_lr"]

PROTOCOLS = ["loso", "forward_expanding", "last_session", "first_session"]
# Size-matched descriptive references. These hold the training set at exactly t
# sessions and vary the eligible donor pool. Their contrasts depend on the
# sampling distribution, matching unit, and path, so the revised manuscript does
# not interpret them as unique quantity/composition mechanisms. The existing
# audited results use MATCHED_DRAWS fixed draws per participant-target.
MATCHED_PROTOCOL_NAMES = ["loso_matched", "future_matched"]
MATCHED_DRAWS = 5
# The optimism gap Delta is defined against the deployable expanding-window arm.
DEPLOYABLE_REF = "forward_expanding"

RANDOM_STATE = 20260716

# The neural-network seed check is deliberately secondary and computationally
# bounded. This fixed, balanced Stieger2021 subset was selected once with
# np.random.RandomState(RANDOM_STATE): six participants from each trajectory
# length (T=7 and T=11). The canonical ranking analysis still uses all 62.
ROBUSTNESS_SUBJECTS = (5, 21, 42, 52, 57, 60, 4, 8, 9, 11, 30, 36)
ROBUSTNESS_EXPECTED_ROWS = 6 * (2 * 7 - 1) + 6 * (2 * 11 - 1)  # 204

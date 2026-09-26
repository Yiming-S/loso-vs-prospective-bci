# Chronology-aware cross-session evaluation for motor-imagery EEG

Author-specified repository: https://github.com/Yiming-S/loso-vs-prospective-bci
Unauthenticated public access has not been verified. The attached review
supplement is the self-contained source-and-results distribution.

This repository compares leave-one-session-out (LOSO) evaluation with
chronological, expanding-window forward evaluation. LOSO measures whether a held-out
session is compatible with all other sessions from the participant; forward chaining
estimates what could have been achieved using only sessions already observed at the
target time. The difference is reported as a difference between estimands, not as a
generic model-accuracy improvement.

The corrected analysis uses six distinct longitudinal datasets:

| Dataset | Participants | Sessions | Retained EEG channels | Binary task |
|---|---:|---:|---:|---|
| BNCI2014_004 | 9 | 5 | 3 bipolar | left vs right hand |
| Zhou2016 | 4 | 3 | 14 | left vs right hand |
| Zhou2020 | 20 | 6 or 7 | 41 or 26 | left vs right hand subset |
| Kumar2024 | 18 | 6 | 22 | left vs right hand |
| Ma2020 | 25 | 15 blocks over 3 days | 62 | right hand vs right elbow |
| Stieger2021 | 62 | 7 or 11 | 60 retained | left vs right hand |

Zhou2016 and Zhou2020 are separate MOABB datasets. Ma2020 is Xuelin Ma et al.'s
same-limb joint-imagery dataset; the loader explicitly removes HEO, VEO, M2, and any
EMG channels that MNE's CNT reader otherwise labels as EEG.

## Analysis definition

All trials are filtered to 8--35 Hz, resampled to 128 Hz, and restricted to the
task-relative interval `[0, 3)` s (384 samples). The common protocols are:

- `loso`: train on all sessions except the target;
- `forward_expanding`: train only on sessions before the target;
- `last_session`: train on the immediately preceding session;
- `first_session`: train on the first session only.

The revised paper reports a primary sequence average and a conditional summary
with the two classical pipelines (CSP+LDA and tangent-space logistic regression):

- `all_origin` (primary deployment-sequence summary): participant-level mean
  `AUC_loso - AUC_forward_expanding` over every prospective target, including
  the final target where the two training pools coincide and the contrast is
  exactly zero;
- `conditional` (supporting): the same participant-level mean over targets for
  which at least one later session exists.

Both estimands are also reported with target, dataset, and random-effects
weighting. Participant distributions, threshold crossings, and classical-decoder
ranking changes provide practical-impact summaries.

Two sampled contrasts hold the number of training sessions fixed. `loso_matched`
draws from the LOSO-eligible pool, while `future_matched` draws from the strict future
when enough sessions exist. The revised paper treats these as a reference-dependent
descriptive audit. It reports their support, sample-based marginal draw standard
errors, and a triangle-inequality bound computed from those estimated standard
errors. The saved draw-level AUC summaries do not identify the full covariance
needed for an exact aggregate Monte Carlo standard error.

The manuscript's model-comparison results use the two fixed-configuration classical
pipelines available for every participant in all six datasets. Exploratory neural-
network outputs are outside the submitted analysis and are not included in the
IEEE Access supplement.

## Reproduce the submitted figures and tables from saved results

This route uses the result-level CSV files included in the supplement. It does
not download EEG, refit decoders, or require MOABB, PyRiemann, MNE, or PyTorch.
`scripts/revision_analysis.py` reads `results/gap_wide.csv`,
`results/auc_long.csv`, and `results/sensitivity/ma2020_day_level.csv` and
regenerates the manuscript numbers, tables, and fixed-size SVG/PDF figures.

The regenerated results use Python 3.12.14 and the pinned packages in
`requirements-results-only.txt`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-results-only.txt
python scripts/revision_analysis.py
python scripts/audit_results.py --scope submitted-classical
python -m unittest TEST.test_reproducibility -v
# Requires a TeX installation with latexmk and the packages used by main.tex.
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

The submitted-classical audit writes `results/submitted_audit.json`. It checks
classical-model result keys, participant/target coverage, paired AUC agreement,
structural zeros, saved estimand means, matched-reference draw metadata,
day-level sensitivity coverage, and manuscript vector assets. It cannot verify
EEG channel counts, epoch length, preprocessing, or fitted models without the
original EEG. Those checks belong to the separate raw-cache audit below.
`results/audit.json` is the historical raw-workspace audit, not the output of
the result-only command.

## Refit from raw EEG

`requirements-ieee-access.txt` records the direct package versions from the
original classical-model analysis (reported Python 3.13.9). It is separate from
the Python 3.12 result-level regeneration environment above; neither file is a
complete transitive lockfile.

MOABB downloads are routed into `cache/mne_data/`. Ma2020 and Stieger2021 are read
from the local store configured by `LOCAL_EEG_ROOT` and are never written there.
Set that environment variable to the directory containing
`MNE-ma2020-data/` and `MNE-Stieger2021-data/` when raw-data refits are required.

```bash
python -m pip install -r requirements-ieee-access.txt
# Classical main protocols and donor-transfer matrices
python scripts/run_experiment.py \
  --datasets BNCI2014_004 Zhou2016 Zhou2020 Kumar2024 Ma2020 Stieger2021 \
  --models csp_lda ts_lr --transfer

# Size-matched protocols
python scripts/run_experiment.py \
  --datasets BNCI2014_004 Zhou2016 Zhou2020 Kumar2024 Ma2020 Stieger2021 \
  --models csp_lda ts_lr \
  --protocols loso_matched future_matched --tag _matched

# Aggregate and write paper numbers
python scripts/make_report.py
python scripts/run_ma2020_day_sensitivity.py --workers 4
python scripts/revision_analysis.py
python scripts/audit_results.py --scope submitted-classical
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

The historical full-workspace check is
`python scripts/audit_results.py --scope raw-cache`. It additionally requires
the original raw result tables, EEG caches, `requirements.txt`, and exploratory
Stieger EEGNet robustness outputs. These large or out-of-scope materials are
not part of the result-level submission supplement, so that command is not the
entry point for reproducing the submitted classical-model analysis.

`scripts/reuse_exact_stieger_results.py` is a narrowly scoped migration for the
existing Stieger2021 outputs. It fails unless all 62 legacy caches contain exactly
60 channels and 384 samples and all result tables have complete, duplicate-free
coverage. Its machine-readable provenance is written to
`results/stieger_exact_reuse_manifest.json`. No other dataset's legacy output is
eligible because its corrected preprocessing or channel selection changed.

Tests requiring the raw-analysis software dependencies:

```bash
python -m pytest TEST -q
python TEST/verify_theory.py
```

`results/inclusion_manifest.csv` records result coverage for the included
participants. It is not a prospective dataset-screening or exclusion log.

The formal manuscript source is `paper/main.tex`. Generated tables and figures live
under `results/`; the checked reader-facing PDF is written to
`output/pdf/loso_evaluation.pdf`. Pre-correction outputs are retained, but isolated
from all aggregation, under `archive/pre_v2_20260719/`.

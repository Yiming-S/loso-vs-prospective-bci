# Chronology-aware cross-session evaluation for motor-imagery EEG

Author-specified repository: https://github.com/Yiming-S/loso-vs-prospective-bci
Unauthenticated public access has not been verified. The attached review
supplement is the self-contained source-and-results distribution.

This repository compares leave-one-session-out (LOSO) evaluation with
chronological, expanding-window forward evaluation. LOSO measures whether a held-out
session can be decoded using all other sessions from the participant; expanding-window evaluation
estimates what could have been achieved using only sessions already observed at the
target time. The difference is reported as a difference between estimands, not as a
generic model-accuracy improvement.

The study builds on established rolling-origin methods and previous chronological
EEG evaluations. Its contribution is a unified paired audit across six
motor-imagery datasets, with explicit averaging populations, temporal units,
cross-dataset heterogeneity, and decoder-margin interpretation. It does not
introduce LOSO-versus-chronological comparison or a new decoding algorithm.

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

The paper reports two participant-weighted summaries of
`AUC_loso - AUC_forward_expanding` with the two classical pipelines (CSP+LDA and
tangent-space logistic regression):

- `conditional` (primary): participant-level mean over targets at which LOSO
  has at least one later session available, i.e. the targets where the two
  protocols can differ (1,001 participant-targets);
- `all_origin` (sensitivity): the same mean including each participant's final
  target, where the two training pools coincide and the contrast is exactly
  zero; within a participant it equals `(T-2)/(T-1)` times the conditional mean.

Both summaries are also reported with target, dataset, and random-effects
weighting. Participant distributions, threshold crossings, and classical-decoder
ranking changes provide practical-impact summaries.

`results/leave_one_dataset_out.csv` reports point estimates after each dataset is
omitted, keeping the original within-participant averaging and giving the remaining
participants equal weight. These are collection-composition sensitivity summaries,
not new confidence intervals or tests. Their aggregate records appear under
`leave_one_dataset_out` in `results/revision_summary.json`.

Two sampled reference arms hold the number of training sessions fixed at the
expanding-window count `t`. `loso_matched` draws the `t` sessions from the
LOSO-eligible pool (five draws, fixed seed), while `future_matched` draws them
from the strict future when enough later sessions exist. The paper partitions the
conditional difference under this reference as
`(loso - loso_matched) + (loso_matched - forward_expanding)`: the complete
donor set versus the matched reference (about 73% of the total) and the matched
reference versus the prior-history set (about 27%). Both components are defined
for the specified reference (session-count matching, uniform draws).
Intervals for the sampled arms are participant-level t intervals conditional on
the five realized draws. Each participant draws from its own seed
(`RANDOM_STATE + subject`), so draw noise enters the participant scatter as
independent error; only 3 of 135 (subject, T) pairs share a draw pattern across
datasets. Per-draw AUCs are not stored, so the aggregate Monte Carlo covariance
is not identifiable from the cached summaries.
`scripts/donor_sampling_uncertainty.py` reports the marginal per-row draw SE,
the donor-sampling component under independent draws across targets, the
covariance-agnostic triangle bound, and the seed-sharing count
(`paper/donor_sampling_results.tex`, `results/donor_sampling_uncertainty.json`).
The main LOSO-minus-expanding estimate uses no donor-reference sampling.

The manuscript's model-comparison results use the two fixed-configuration classical
pipelines available for every participant in all six datasets. The single-seed
EEGNet run on Stieger2021 and its three-seed 12-participant subset check are
reported in the appendix.

`scripts/moabb_forward_demo.py` runs `src/splitters.py::ForwardSessionSplitter`
inside MOABB 1.5.0's unmodified `CrossSessionEvaluation`
(`cv_class=ForwardSessionSplitter, cv_kwargs={"mode": "expanding"}`) on
BNCI2014_004 (four expanding-window splits per participant, 36 scores per
decoder, 72 in total) and checks the per-session AUCs against the manuscript's
`forward_expanding` results (`results/moabb_demo/`; agreement within 0.004 AUC,
the residual coming from MOABB's 385-sample epochs versus the loader's 384).

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
python scripts/donor_sampling_uncertainty.py
python scripts/audit_results.py --scope submitted-classical
python -m unittest TEST.test_reproducibility -v
# Requires a TeX installation with latexmk and the packages used by main.tex.
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

The additional result-statistics tests cover averaging, support, decoder margins,
Monte Carlo diagnostics, and dataset omission. They use the same environment plus
the tested `pytest==9.1.1` runner:

```bash
python -m pip install pytest==9.1.1
python -m pytest TEST/test_revision_statistics.py TEST/test_reproducibility.py -q
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
python scripts/donor_sampling_uncertainty.py
python scripts/moabb_forward_demo.py
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

The formal manuscript source is `paper/main.tex`. Generated LaTeX tables and
macro files are written to `paper/` (`table_*.tex`, `revision_results.tex`,
`donor_sampling_results.tex`); figures live under `results/figures/`; the
checked reader-facing PDF is written to `output/pdf/loso_evaluation.pdf`. Pre-correction outputs are retained, but isolated
from all aggregation, under `archive/pre_v2_20260719/`.

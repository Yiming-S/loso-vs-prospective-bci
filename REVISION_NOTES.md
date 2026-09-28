# Manuscript and reproducibility revision

## Scientific changes

- Added the directly overlapping chronological EEG study by Egger et al. (2024), and distinguished its executed-gesture comparison and Wimpff et al.'s chronological fine-tuning from this cross-dataset motor-imagery audit.
- Reorganized the abstract, contribution paragraph, results, discussion, and conclusion around paired protocol differences, recording-design heterogeneity, and averaging targets. Ranking changes and five-draw donor-reference contrasts are supporting analyses.
- Added result-only leave-one-dataset-out point summaries using unchanged participant weighting; all-origin estimates remain positive in every five-dataset subset. The exported CSV and generated manuscript macros provide the full calculation trail.
- Moved the Ma2020 temporal-unit figure into the primary results sequence, routed the expanding-window connector around the unavailable future session, and added near-origin decoder-margin insets. SVG sources and vector-PDF mirrors share fixed-size canvases.
- Stated the conditioning on realized donor draws directly in the decomposition table caption and added published sources for prediction intervals and HKSJ sensitivity formulas.

- Retained the all-origin primary estimand and supporting conditional estimand; added their participant-specific structural-zero scaling identity.
- Reframed the contribution as a paired empirical audit of training information sets at identical test sessions.
- Reported the uniform donor-session-count decomposition on all 1,001 conditional targets, with the 560-target strict-future comparison separately identified.
- Distinguished session-count matching from trial-count matching and removed mechanism-level interpretations of the descriptive partition.
- Corrected marginal Monte Carlo standard errors for cached ddof=0 standard deviations. Reported covariance-free plug-in bounds under estimated marginal variances; removed the unsupported low-aggregate-error claim.
- Added observed decoder-margin plots and bilateral-margin sensitivity. All-origin ranking changes number 14; only 1 exceeds 0.01 AUC on both protocol-specific margins.
- Added explicit meta-analysis interval formulas and residual-scale sensitivity, a five-field reporting specification, and bidirectional illustrative threshold crossings.
- Updated all figures as fixed-size editable SVGs with corresponding vector PDFs; revised table labels and support layout.

## Reproduction checks

The revised archive was extracted into a clean directory with no raw EEG or cached fits. The result-analysis script and submitted-classical audit completed successfully in the Python 3.12.14 environment specified by requirements-results-only.txt. Regenerated revision_summary.json, revision_results.tex, and leave_one_dataset_out.csv matched the prepared outputs byte for byte. All 16 targeted statistics and reproducibility tests passed. The 11-page manuscript compiled from the extracted source; the prepared PDF was visually inspected page by page. These checks validate result-level regeneration; raw-data refitting was not performed in this revision. Every pre-existing revision_summary.json value, raw score table, and model/preprocessing source file remained unchanged.

The archive contains a per-file SHA-256 manifest. Historical raw-cache audit and Stieger reuse provenance are labeled separately from the current submitted-result audit.

## Funding

The author confirmed that this research received no external funding. The manuscript includes this statement in its Funding and Acknowledgment section.

## Author decisions still required

- Confirm public repository creation and upload before describing a GitHub URL as publicly accessible.
- Select a project code license; no license grant has been assigned by this revision.

## 2026-09-27 restructure (cold-read fixes)

- Primary estimand is now the conditional summary (targets with at least one later
  donor, 1,001 participant-targets; +0.035 AUC, 95% CI [+0.030, +0.040]); the
  all-origin summary and the structural-zero scaling identity are reported as a
  sensitivity analysis. Figure 1(b), Figure 2, Figure 3, and Table 2 were
  regenerated with the conditional summary first; `revision_results.tex` is unchanged.
- The donor-count-matched partition moved from an appendix to the Methods
  (Section IV-D, Equation 6) and a Results subsection (V-B) and into the abstract:
  about three quarters of the difference (+0.025) comes from LOSO training on more
  sessions, about one quarter (+0.009, 95% CI [+0.005, +0.014]) from access to later
  sessions.
- `scripts/donor_sampling_uncertainty.py` replaces the triangle-inequality plug-in
  bound with a donor-sampling standard error under independent draws across
  targets (shared draws across decoders within a target): 0.0007 AUC on the
  conditional support, about 10% of the composition estimate's variance. Intervals
  for the sampled arms in the text include this variance
  (`paper/donor_sampling_results.tex`, `results/donor_sampling_uncertainty.json`).
- The EEGNet ordering reversal was withdrawn from the Discussion; the single-seed
  run and its 12-participant three-seed check remain an appendix illustration.
- `scripts/moabb_forward_demo.py` runs `ForwardSessionSplitter` inside MOABB
  1.5.0's unmodified `CrossSessionEvaluation` on BNCI2014_004 and matches the
  manuscript's expanding-window and LOSO AUCs within 0.004 (Appendix A-F,
  `results/moabb_demo/`).
- `TEST/test_revision_statistics.py` updated for the new Table 2 column order.

## 2026-09-28 claim and statistics pass

- Donor-sampling uncertainty: withdrew the combined participant-plus-donor interval
  and the "about 10% of variance" statement. Sampled-arm intervals are participant-level
  t intervals conditional on the five realized draws (Table 3, Appendix B-B). Seeds are
  participant-specific (`RANDOM_STATE + subject`; 3 of 135 participant-length pairs share
  a draw pattern across datasets); per-draw AUCs are not stored, so the aggregate draw
  covariance is not identifiable. Appendix B-B reports the independent-draws component
  (0.0007 / 0.0012 AUC) and the covariance-agnostic bound (0.0180 / 0.0221).
- Claims rewritten as comparisons under the specified session-count-matched reference
  (components about 73% and 27% of the total; the future-donor arm scored higher on early
  targets), replacing causal wording. Thesis sentence added to the introduction.
- Abstract compressed to +0.035 [0.030, 0.040] plus three qualitative results.
- Figures/tables: Figure 2 now shows the matched-reference comparisons (panel a) and the
  participant distributions (panel b); Figure 5, Table 6, Table 7 conditional first;
  Table 8 labels the alternative policies as conditional targets; Figure 3/4/6 heights reduced.
- EEGNet: appendix keeps the two facts (single-seed full-cohort ordering; three-seed
  12-participant near-tie under LOSO); repeated negations removed from Results/Discussion.
- Corrected MOABB check wording: 36 scores per decoder, 72 in total. Conclusion qualifies
  +0.035 as targets where LOSO had a later session available.

## 2026-09-28 prose pass

- Language-only revision of abstract, Sections I-VII, and Appendices A-B for plain,
  natural prose (no changes to numbers, macros, references, or citations; verified by
  a regex bag diff against the pre-polish source). Three-lens diagnostic and a
  four-lens adversarial verification were used for checking; all edits were made by hand.
- Code Availability now reads "will be released at" the GitHub URL, because the
  repository is not yet publicly reachable (HTTP 404 on 2026-09-28).
- `table_support.tex` placement changed from `[!t]` to `[t]` in the generator to stop
  page 6 from overflowing into the footer.

## 2026-09-28 IEEE Access references

- Added 13 IEEE Access references (Crossref-verified DOIs, abstracts read before citing):
  Hernandez-Rojas 2022, Maswanganyi 2022, Meng 2017 (Introduction: repeated use and
  session-to-session variability); Keutayeva 2023, Sugata 2026 (acronym note: LOSO often
  means leave-one-subject-out elsewhere); Sugata 2026, Li 2026 (Related Work: protocol
  dependence of reported accuracy); Lee 2020, Heskebeck 2026, Jiang 2020 (Related Work:
  donor use and adaptive pipelines); Singh 2019 (Discussion: small-sample behaviour of
  Riemannian classifiers); George 2022, Raza 2025, Suemitsu 2023 (Discussion: decoder
  comparisons and reporting). Cross-domain candidates were excluded by the author's choice.
- References now 48; manuscript 13 pages.

## 2026-09-28 number formatting conventions

- AUC levels, AUC differences, and their CIs: three decimals in text and tables
  (Table 3 changed from four to three decimals; its caption now says components sum
  to the total "up to rounding").
- Four decimals only for standard errors and near-zero quantities: Appendix B
  donor-sampling SEs and bound, the drift coefficient, and the EEGNet seed table
  (Table 10) with its text range.
- p-values: two decimals, three below 0.01, and "p < 0.001" below 0.001
  (`pvalue()` now returns a math-mode relation).
- Signs: a number stated as a difference carries its sign; when the direction is
  given in words ("0.025 AUC above"), no plus sign. Ten `...Unsigned` macros were
  added for the six sentences of that form.
- True minus signs in all text-mode macros (`text_minus()`); leave-one-dataset-out
  macros at three decimals.

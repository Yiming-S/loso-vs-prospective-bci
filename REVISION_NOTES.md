# Manuscript and reproducibility revision

## Scientific changes

- Retained the all-origin primary estimand and supporting conditional estimand; added their participant-specific structural-zero scaling identity.
- Reframed the contribution as a paired empirical audit of training information sets at identical test sessions.
- Reported the uniform donor-session-count decomposition on all 1,001 conditional targets, with the 560-target strict-future comparison separately identified.
- Distinguished session-count matching from trial-count matching and removed mechanism-level interpretations of the descriptive partition.
- Corrected marginal Monte Carlo standard errors for cached ddof=0 standard deviations. Reported covariance-free plug-in bounds under estimated marginal variances; removed the unsupported low-aggregate-error claim.
- Added observed decoder-margin plots and bilateral-margin sensitivity. All-origin ranking changes number 14; only 1 exceeds 0.01 AUC on both protocol-specific margins.
- Added explicit meta-analysis interval formulas and residual-scale sensitivity, a five-field reporting specification, and bidirectional illustrative threshold crossings.
- Updated all figures as fixed-size editable SVGs with corresponding vector PDFs; revised table labels and support layout.

## Reproduction checks

The submitted archive was extracted into a clean directory with no raw EEG or cached fits. The result-analysis script and submitted-classical audit completed successfully in the Python 3.12.14 environment specified by requirements-results-only.txt. Regenerated revision_summary.json and revision_results.tex matched the prepared outputs byte for byte. The 11 targeted statistics and reproducibility tests passed. The manuscript compiled from the extracted source. These checks validate result-level regeneration; raw-data refitting was not performed in this revision.

The archive contains a per-file SHA-256 manifest. Historical raw-cache audit and Stieger reuse provenance are labeled separately from the current submitted-result audit.

## Funding

The author confirmed that this research received no external funding. The manuscript includes this statement in its Funding and Acknowledgment section.

## Author decisions still required

- Confirm public repository creation and upload before describing a GitHub URL as publicly accessible.
- Select a project code license; no license grant has been assigned by this revision.

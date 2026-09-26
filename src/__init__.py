"""How Optimistic Is Leave-One-Session-Out Evaluation in Motor-Imagery BCI?

Self-contained experiment package. Reuses only stable third-party libraries
(moabb, pyriemann, mne, scikit-learn, statsmodels, torch); the domain logic that
exists in the related research code (session loading and pyriemann drift math,
ShiftDx/Shifts-Impact mixed-effects modelling) is re-expressed here in a minimal,
dependency-light form so the study stands on its own.
"""

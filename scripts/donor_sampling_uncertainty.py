"""Donor-sampling uncertainty for the donor-count-matched decomposition.

The uniform donor-count-matched reference averages five random donor draws per
(participant, target, decoder) row.  results/auc_long.csv stores the draw
standard deviation of each row mean (ddof=0), so the marginal standard error of
a row mean is  e_r = s_r0 / sqrt(B - 1)  with B = 5.

Aggregation follows the manuscript: targets and decoders are averaged within
participant, then participants receive equal weight, so the composition
estimate is  sum_r a_r x_r  with  a_r = 1 / (N * n_rows_in_participant).

Two donor-sampling standard errors are reported for that aggregate:

  triangle  : sum_r a_r e_r                       (covariance-agnostic bound,
                                                    as already in the paper)
  realistic : sqrt( sum_targets (sum_{r in target} a_r e_r)^2 )
              Draws are independent across targets (each target draws its own
              donor sets), while the two decoder rows of one target share the
              same drawn donor sets and are treated as perfectly correlated.
              This is the correct SE under independent target draws and an
              upper bound for the within-target decoder correlation.

The donor-sampling SE is then combined with the participant-level sampling SE
of the composition contrast (matched - expanding) as  sqrt(se_p^2 + se_d^2)
and a t(N-1) interval is reported.  The same is done for the strict-future
arm (future_matched - forward_expanding) on the strict-future support.

Writes paper/donor_sampling_results.tex (LaTeX macros) and
results/donor_sampling_uncertainty.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
CLASSICAL = ["csp_lda", "ts_lr"]
KEYS = ["dataset", "subject", "model", "target_session"]


def participant_contrast(frame, left, right):
    x = frame.dropna(subset=[left, right]).copy()
    x["contrast_value"] = x[left] - x[right]
    return x.groupby("subject_uid")["contrast_value"].mean()


def donor_se(frame, long_df, protocol):
    """Triangle bound and realistic SE of the participant-weighted mean of a
    sampled-reference arm, using stored per-row draw SDs."""
    sampled = long_df[
        long_df["model"].isin(CLASSICAL)
        & (long_df["protocol"] == protocol)
        & (long_df["target_session"] >= 1)
    ][KEYS + ["auc_sd_draws", "n_draws"]]
    f = frame[KEYS + ["subject_uid"]].merge(sampled, on=KEYS, how="left", validate="one_to_one")
    if f[["auc_sd_draws", "n_draws"]].isna().any().any():
        raise ValueError(f"missing draw variability for {protocol}")
    if (f["n_draws"] <= 1).any():
        raise ValueError("need at least two draws")
    f["e"] = f["auc_sd_draws"] / np.sqrt(f["n_draws"] - 1)
    n_part = f["subject_uid"].nunique()
    rows_per_part = f.groupby("subject_uid")["subject_uid"].transform("size")
    f["a"] = 1.0 / (n_part * rows_per_part)
    if not np.isclose(f["a"].sum(), 1.0):
        raise AssertionError("weights do not sum to one")
    f["ae"] = f["a"] * f["e"]
    triangle = float(f["ae"].sum())
    per_target = f.groupby(["dataset", "subject", "target_session"])["ae"].sum()
    realistic = float(np.sqrt((per_target ** 2).sum()))
    independent_rows = float(np.sqrt((f["ae"] ** 2).sum()))
    return {
        "participants": int(n_part),
        "rows": int(len(f)),
        "targets": int(len(per_target)),
        "draws": sorted(f["n_draws"].astype(int).unique().tolist()),
        "triangle_se_bound": triangle,
        "realistic_se_shared_decoder_draws": realistic,
        "independent_rows_se": independent_rows,
    }


def combined(participant_values, se_donor):
    x = np.asarray(participant_values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    se_p = float(stats.sem(x))
    se_tot = math.sqrt(se_p ** 2 + se_donor ** 2)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return {
        "n": n,
        "mean": mean,
        "se_participant": se_p,
        "se_donor": se_donor,
        "se_total": se_tot,
        "ci_conditional_on_draws": [mean - tcrit * se_p, mean + tcrit * se_p],
        "ci_with_donor_sampling": [mean - tcrit * se_tot, mean + tcrit * se_tot],
        "donor_share_of_variance": se_donor ** 2 / se_tot ** 2,
    }


def fmt(v, d=4, signed=True):
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def main():
    wide = pd.read_csv(RESULTS / "gap_wide.csv")
    wide = wide[wide["model"].isin(CLASSICAL)].copy()
    long_df = pd.read_csv(RESULTS / "auc_long.csv")
    conditional = wide[~wide["is_structural_zero"]].copy()
    uniform = conditional.dropna(subset=["loso_matched"])
    strict = conditional.dropna(subset=["future_matched", "loso_matched"])

    out = {}
    # Conditional support: composition = loso_matched - forward_expanding
    se_u = donor_se(uniform, long_df, "loso_matched")
    comp_u = combined(participant_contrast(uniform, "loso_matched", "forward_expanding"),
                      se_u["realistic_se_shared_decoder_draws"])
    count_u = combined(participant_contrast(uniform, "loso", "loso_matched"),
                       se_u["realistic_se_shared_decoder_draws"])
    total_u = participant_contrast(uniform, "loso", "forward_expanding")
    out["conditional"] = {"donor_se": se_u, "composition": comp_u, "count": count_u,
                          "total_mean": float(total_u.mean())}
    # Strict-future support
    se_s = donor_se(strict, long_df, "loso_matched")
    comp_s = combined(participant_contrast(strict, "loso_matched", "forward_expanding"),
                      se_s["realistic_se_shared_decoder_draws"])
    se_f = donor_se(strict, long_df, "future_matched")
    fut_s = combined(participant_contrast(strict, "future_matched", "forward_expanding"),
                     se_f["realistic_se_shared_decoder_draws"])
    total_s = participant_contrast(strict, "loso", "forward_expanding")
    count_s = combined(participant_contrast(strict, "loso", "loso_matched"),
                       se_s["realistic_se_shared_decoder_draws"])
    out["strict_future"] = {"donor_se_matched": se_s, "donor_se_future": se_f,
                            "composition": comp_s, "count": count_s, "future_minus_expanding": fut_s,
                            "total_mean": float(total_s.mean())}
    # additivity checks
    assert np.isclose(out["conditional"]["total_mean"], comp_u["mean"] + count_u["mean"])

    # Seed sharing: draws use seed RANDOM_STATE + participant index, so two
    # participants share the same draw pattern only if they have the same index
    # and trajectory length T in different datasets.
    T_by = wide.groupby(["dataset", "subject"])["T"].first().reset_index()
    pair_counts = T_by.groupby(["subject", "T"])["dataset"].nunique()
    shared_pairs = int((pair_counts > 1).sum())
    out["seed_sharing"] = {
        "subject_T_pairs": int(len(pair_counts)),
        "pairs_shared_across_datasets": shared_pairs,
        "note": "per-draw AUCs are not stored; the aggregate Monte Carlo covariance is not identifiable from the cached summaries",
    }
    (RESULTS / "donor_sampling_uncertainty.json").write_text(json.dumps(out, indent=2))

    macros = {
        "CondCompDrawSE": fmt(se_u["realistic_se_shared_decoder_draws"], 4, False),
        "CondCompParticipantSE": fmt(comp_u["se_participant"], 4, False),
        "StrictCompDrawSE": fmt(se_s["realistic_se_shared_decoder_draws"], 4, False),
        "StrictFutureDrawSE": fmt(se_f["realistic_se_shared_decoder_draws"], 4, False),
        "CondCountShare": fmt(100 * count_u["mean"] / out["conditional"]["total_mean"], 0, False),
        "CondCompShare": fmt(100 * comp_u["mean"] / out["conditional"]["total_mean"], 0, False),
        "StrictCountShare": fmt(100 * (total_s.mean() - comp_s["mean"]) / total_s.mean(), 0, False),
        "SharedSeedPairs": str(shared_pairs),
        "SubjectTPairs": str(int(len(pair_counts))),
    }
    lines = ["% Generated by scripts/donor_sampling_uncertainty.py; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (PAPER / "donor_sampling_results.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

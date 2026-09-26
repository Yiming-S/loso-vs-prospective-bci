#!/usr/bin/env python3
"""Write LaTeX result macros from the audited aggregate outputs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from src.config import RANDOM_STATE, ROBUSTNESS_SUBJECTS  # noqa: E402

SUMMARY = ROOT / "results" / "summary.json"
GAP_WIDE = ROOT / "results" / "gap_wide.csv"
OUT = ROOT / "paper" / "results.tex"
DATASET_TABLE = ROOT / "paper" / "table_dataset_protocols.tex"
CONTRAST_TABLE = ROOT / "paper" / "table_protocol_contrasts.tex"


def num(value, digits=3, signed=False):
    return f"{value:{'+' if signed else ''}.{digits}f}"


def pvalue(value):
    if value is None or not math.isfinite(value):
        return "NA"
    if value <= 0:
        return r"\ensuremath{<10^{-300}}"
    if value < 1e-4:
        exponent = math.floor(math.log10(value))
        mantissa = value / (10 ** exponent)
        return rf"\ensuremath{{{mantissa:.1f}\times10^{{{exponent}}}}}"
    return f"{value:.4f}"


def main():
    summary = json.loads(SUMMARY.read_text())
    primary = summary["H1_vs_forward"]
    drift_model = summary["H2_model"]
    matched = summary["size_matched"]
    size = matched["loso_vs_matched"]
    composition = matched["matched_vs_forward"]
    future = matched["future_vs_forward"]
    slope = summary["dose_slope"]
    natural = summary["natural_experiment_stieger"]
    donor = summary["donor_direction"]

    macros = {
        "NSubjects": str(summary["n_subjects"]),
        "NAUCRows": str(summary["n_auc_rows"]),
        "GapMean": num(primary["subject_mean"], signed=True),
        "GapLo": num(primary["ci"][0], signed=True),
        "GapHi": num(primary["ci"][1], signed=True),
        "GapP": pvalue(primary["subj_p_onesided"]),
        "GapPTwo": pvalue(primary.get(
            "subj_p_twosided", 2 * primary["subj_p_onesided"])),
        "SizeMean": num(size["subject_mean"], signed=True),
        "SizeLo": num(size["ci"][0], signed=True),
        "SizeHi": num(size["ci"][1], signed=True),
        "SizeP": pvalue(size["p_twosided"]),
        "CompositionMean": num(composition["subject_mean"], signed=True),
        "CompositionLo": num(composition["ci"][0], signed=True),
        "CompositionHi": num(composition["ci"][1], signed=True),
        "CompositionP": pvalue(composition["p_twosided"]),
        "FutureMean": num(future["subject_mean"], signed=True),
        "FutureLo": num(future["ci"][0], signed=True),
        "FutureHi": num(future["ci"][1], signed=True),
        "FutureP": pvalue(future["p_twosided"]),
        "DriftBeta": num(drift_model["drift_beta"], digits=4, signed=True),
        "DriftP": pvalue(drift_model["drift_p"]),
        "DoseBeta": num(slope["beta"], digits=4, signed=True),
        "DoseLo": num(slope["ci"][0], digits=4, signed=True),
        "DoseHi": num(slope["ci"][1], digits=4, signed=True),
        "DoseP": pvalue(slope["p"]),
        "NaturalDiff": num(natural["diff_T11_minus_T7"], signed=True),
        "NaturalP": pvalue(natural["p_twosided"]),
        "DonorMean": num(donor["subject_mean"], signed=True),
        "DonorLo": num(donor["ci"][0], signed=True),
        "DonorHi": num(donor["ci"][1], signed=True),
        "DonorP": pvalue(donor["p_twosided"]),
        "DonorLagBeta": num(donor["lmm_L_slope"], digits=4, signed=True),
        "DonorLagP": pvalue(donor["lmm_L_p"]),
        "DriftLo": num(drift_model["drift_ci"][0], digits=4, signed=True),
        "DriftHi": num(drift_model["drift_ci"][1], digits=4, signed=True),
    }

    dose_rows = {int(row["sessions_after"]): row
                 for row in summary.get("dose_response", [])}
    for sessions, prefix in [(1, "DoseOne"), (4, "DoseFour")]:
        row = dose_rows[sessions]
        macros.update({
            prefix + "Mean": num(row["mean"], signed=True),
            prefix + "Lo": num(row["ci"][0], signed=True),
            prefix + "Hi": num(row["ci"][1], signed=True),
            prefix + "N": str(row["n_subjects"]),
        })

    # Absolute protocol levels on the same participant, model, and interior-target
    # support as the primary paired contrast.
    wide = pd.read_csv(GAP_WIDE)
    interior = wide[
        (~wide["is_structural_zero"]) &
        (wide["model"].isin(["csp_lda", "ts_lr"]))
    ]
    participant = interior.groupby(
        ["dataset", "subject"], as_index=False
    )[["loso", "forward_expanding"]].mean()
    for column, prefix in [
        ("loso", "LosoLevel"),
        ("forward_expanding", "ForwardLevel"),
    ]:
        values = participant[column].dropna()
        mean = values.mean()
        half_width = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
        macros.update({
            prefix + "Mean": num(mean),
            prefix + "Lo": num(mean - half_width),
            prefix + "Hi": num(mean + half_width),
        })

    leave_one_out = [r for r in summary["leave_dataset_out"]
                     if "+" not in r["dropped"]]
    drop_long = next(r for r in summary["leave_dataset_out"]
                     if r["dropped"].startswith("Ma2020+Stieger2021"))
    macros.update({
        "LeaveDatasetMin": num(min(r["mean"] for r in leave_one_out), signed=True),
        "LeaveDatasetMax": num(max(r["mean"] for r in leave_one_out), signed=True),
        "DropLongMean": num(drop_long["mean"], signed=True),
    })

    robustness = []
    macros["RobustN"] = str(len(ROBUSTNESS_SUBJECTS))
    robust_dir = ROOT / "results" / "robustness"
    long_path = ROOT / "results" / "auc_long.csv"
    if robust_dir.is_dir() and long_path.exists():
        long = pd.read_csv(long_path)
        stieger = long[(long["dataset"] == "Stieger2021") &
                       (long["target_session"] >= 1)]
        paired = stieger.groupby(
            ["subject", "protocol", "model"])["auc"].mean().unstack("model")
        macros["StiegerN"] = str(paired.reset_index()["subject"].nunique())
        for protocol, prefix in [("loso", "StiegerLoso"),
                                 ("forward_expanding", "StiegerForward")]:
            frame = paired.xs(protocol, level="protocol").dropna(
                subset=["eegnet", "ts_lr"])
            difference = frame["eegnet"] - frame["ts_lr"]
            _, p = stats.ttest_1samp(difference, 0.0)
            macros[prefix + "Diff"] = num(difference.mean(), signed=True)
            macros[prefix + "P"] = pvalue(p)

        ts = stieger[stieger["model"] == "ts_lr"].groupby(
            ["subject", "protocol"])["auc"].mean().unstack()
        for path in sorted(robust_dir.glob("eegnet_Stieger2021_seed*.csv")):
            frame = pd.read_csv(path)
            if set(frame["subject"].astype(int)) != set(ROBUSTNESS_SUBJECTS):
                continue
            eg = frame[frame["target_session"] >= 1].groupby(
                ["subject", "protocol"])["auc"].mean().unstack()
            common = eg.index.intersection(ts.index)
            if len(common) == len(ROBUSTNESS_SUBJECTS) and \
                    {"loso", "forward_expanding"}.issubset(eg.columns) and \
                    {"loso", "forward_expanding"}.issubset(ts.columns):
                robustness.append({
                    "seed": int(frame["seed"].iloc[0]),
                    "loso": float((eg.loc[common, "loso"] -
                                   ts.loc[common, "loso"]).mean()),
                    "forward": float((eg.loc[common, "forward_expanding"] -
                                      ts.loc[common, "forward_expanding"]).mean()),
                })
    macros["SeedCount"] = str(len(robustness))
    macros.update({
        "SeedLosoReversals": "0", "SeedForwardReversals": "0",
        "SeedLosoDiffMin": "NA", "SeedLosoDiffMax": "NA",
        "SeedForwardDiffMin": "NA", "SeedForwardDiffMax": "NA",
        "SubsetCanonicalLosoDiff": "NA", "SubsetCanonicalForwardDiff": "NA",
    })
    if robustness:
        loso = [r["loso"] for r in robustness]
        forward = [r["forward"] for r in robustness]
        macros.update({
            "SeedLosoReversals": str(sum(x > 0 for x in loso)),
            "SeedForwardReversals": str(sum(x < 0 for x in forward)),
            "SeedLosoDiffMin": num(min(loso), digits=4, signed=True),
            "SeedLosoDiffMax": num(max(loso), digits=4, signed=True),
            "SeedForwardDiffMin": num(min(forward), digits=4, signed=True),
            "SeedForwardDiffMax": num(max(forward), digits=4, signed=True),
        })
        canonical_subset = next(
            (row for row in robustness if row["seed"] == RANDOM_STATE), None)
        if canonical_subset:
            macros.update({
                "SubsetCanonicalLosoDiff": num(
                    canonical_subset["loso"], digits=4, signed=True),
                "SubsetCanonicalForwardDiff": num(
                    canonical_subset["forward"], digits=4, signed=True),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by scripts/write_paper_results.py; do not edit by hand."]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items())
    OUT.write_text("\n".join(lines) + "\n")

    # Dataset-level absolute protocol values and paired gaps, all computed after
    # averaging the two classical decoders and interior targets within participant.
    participant = interior.groupby(
        ["dataset", "subject"], as_index=False
    )[["loso", "forward_expanding"]].mean()
    participant["gap"] = participant["loso"] - participant["forward_expanding"]
    table_rows = []
    for dataset, frame in participant.groupby("dataset", sort=True):
        values = frame["gap"].dropna()
        half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
        dataset_label = dataset.replace("_", r"\_")
        table_rows.append(
            f"{dataset_label} & {len(values)} & "
            f"{frame['loso'].mean():.3f} & {frame['forward_expanding'].mean():.3f} & "
            f"{values.mean():+.3f} [{values.mean()-half:+.3f}, {values.mean()+half:+.3f}] \\\\"
        )
    values = participant["gap"].dropna()
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    table_rows.append(
        f"\\midrule Pooled & {len(values)} & {participant['loso'].mean():.3f} & "
        f"{participant['forward_expanding'].mean():.3f} & "
        f"{values.mean():+.3f} [{values.mean()-half:+.3f}, {values.mean()+half:+.3f}] \\\\"
    )
    DATASET_TABLE.write_text("\n".join([
        "% Generated by scripts/write_paper_results.py; do not edit by hand.",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Absolute protocol levels and paired differences by dataset. AUCs are averaged over the two classical decoders and interior targets within participant. Intervals refer to participant-level paired differences.}",
        r"\label{tab:dataset-protocols}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Dataset & $n$ & LOSO & Forward & Difference [95\% CI] \\",
        r"\midrule", *table_rows, r"\bottomrule", r"\end{tabular}",
        r"\end{table}", ""
    ]))

    primary = summary["H1_vs_forward"]
    contrasts = [
        ("LOSO $-$ forward", primary["n_subjects"], primary["subject_mean"],
         primary["ci"], primary["subj_p_twosided"]),
        ("LOSO $-$ size-matched LOSO", size["n_subjects"], size["subject_mean"],
         size["ci"], size["p_twosided"]),
        ("Size-matched LOSO $-$ forward", composition["n_subjects"],
         composition["subject_mean"], composition["ci"], composition["p_twosided"]),
        ("Strict future $-$ forward", future["n_subjects"], future["subject_mean"],
         future["ci"], future["p_twosided"]),
    ]
    contrast_rows = [
        f"{label} & {n} & {estimate:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] & {pvalue(p)} \\\\"
        for label, n, estimate, ci, p in contrasts
    ]
    CONTRAST_TABLE.write_text("\n".join([
        "% Generated by scripts/write_paper_results.py; do not edit by hand.",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Participant-level protocol contrasts. The quantity contrast changes donor count; the two composition contrasts hold donor count fixed.}",
        r"\label{tab:protocol-contrasts}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Contrast & $n$ & Mean AUC [95\% CI] & Two-sided $p$ \\",
        r"\midrule", *contrast_rows, r"\bottomrule", r"\end{tabular}",
        r"\end{table}", ""
    ]))

    print(f"wrote {OUT} ({len(macros)} macros)")
    print(f"wrote {DATASET_TABLE}")
    print(f"wrote {CONTRAST_TABLE}")


if __name__ == "__main__":
    main()

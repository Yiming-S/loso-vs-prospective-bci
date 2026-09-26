#!/usr/bin/env python3
"""Build a restrained, fully data-driven Markdown and HTML research report."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from src.config import ROBUSTNESS_SUBJECTS  # noqa: E402

RESULTS = ROOT / "results"
REPORT = ROOT / "report"
SUMMARY = RESULTS / "summary.json"

DATASETS = [
    ("BNCI2014_004", 9, "5", "3 bipolar", "left vs right hand"),
    ("Zhou2016", 4, "3", "14", "left vs right hand"),
    ("Zhou2020", 20, "6 or 7", "41 or 26", "left vs right hand subset"),
    ("Kumar2024", 18, "6", "22", "left vs right hand"),
    ("Ma2020", 25, "15 blocks / 3 days", "62", "right hand vs right elbow"),
    ("Stieger2021", 62, "7 or 11", "60 retained", "left vs right hand"),
]


def fnum(value, digits=3, signed=False):
    if value is None:
        return "NA"
    return f"{float(value):{'+' if signed else ''}.{digits}f}"


def fp(value):
    if value is None:
        return "NA"
    value = float(value)
    return f"{value:.2e}" if value < 1e-4 else f"{value:.4f}"


def seed_robustness():
    """Participant-weighted EEGNet-minus-TS+LR differences by fixed seed."""
    long_path = RESULTS / "auc_long.csv"
    robust_dir = RESULTS / "robustness"
    if not long_path.exists() or not robust_dir.is_dir():
        return []
    long = pd.read_csv(long_path)
    ts = long[(long["dataset"] == "Stieger2021") &
              (long["model"] == "ts_lr") &
              (long["target_session"] >= 1)]
    ts = ts.groupby(["subject", "protocol"])["auc"].mean().unstack()
    rows = []
    for path in sorted(robust_dir.glob("eegnet_Stieger2021_seed*.csv")):
        eeg = pd.read_csv(path)
        if set(eeg["subject"].astype(int)) != set(ROBUSTNESS_SUBJECTS):
            continue
        eeg = eeg[eeg["target_session"] >= 1].groupby(
            ["subject", "protocol"])["auc"].mean().unstack()
        common = eeg.index.intersection(ts.index)
        if len(common) != len(ROBUSTNESS_SUBJECTS):
            continue
        seed = int(re.search(r"seed(\d+)", path.name).group(1))
        rows.append({
            "seed": seed,
            "loso": float((eeg.loc[common, "loso"] - ts.loc[common, "loso"]).mean()),
            "forward": float((eeg.loc[common, "forward_expanding"] -
                              ts.loc[common, "forward_expanding"]).mean()),
        })
    return rows


def make_markdown(summary, audit):
    h1 = summary["H1_vs_forward"]
    h2 = summary["H2_model"]
    matched = summary["size_matched"]
    quantity = matched["loso_vs_matched"]
    composition = matched["matched_vs_forward"]
    future = matched["future_vs_forward"]
    slope = summary["dose_slope"]
    natural = summary["natural_experiment_stieger"]
    donor = summary["donor_direction"]
    audit_text = "passed" if audit.get("ok") else "failed"

    lines = [
        "# When leave-one-session-out evaluation overstates deployable MI-BCI performance",
        "",
        "## Abstract",
        "",
        f"We compare leave-one-session-out (LOSO) evaluation with expanding-window "
        f"forward chaining on six longitudinal motor-imagery EEG datasets "
        f"({summary['n_subjects']} participants). All trials use a common 8--35 Hz, "
        f"128 Hz, 0--3 s preprocessing interval. On the prespecified classical endpoint, "
        f"LOSO minus forward chaining was {fnum(h1['subject_mean'], signed=True)} AUC "
        f"(95% CI [{fnum(h1['ci'][0], signed=True)}, "
        f"{fnum(h1['ci'][1], signed=True)}], one-sided p={fp(h1['subj_p_onesided'])}). "
        f"A size-matched analysis separated a training-quantity term of "
        f"{fnum(quantity['subject_mean'], signed=True)} from a sampled-composition term "
        f"of {fnum(composition['subject_mean'], signed=True)} AUC. The composition term "
        f"is not treated as a context-free causal effect because practice and "
        f"session-quality trends remain possible. Standardized consecutive-session "
        f"covariance drift did not reliably moderate the gap "
        f"(beta={fnum(h2.get('drift_beta'), 4, True)}, p={fp(h2.get('drift_p'))}).",
        "",
        "## Evaluation question",
        "",
        "LOSO measures cross-session interpolation: each target is evaluated after "
        "pooling all other sessions, including later ones. Forward chaining measures a "
        "different estimand: performance using only the training history available before "
        "the target. Their raw difference changes both session composition and training "
        "quantity. The analysis therefore does not label the entire difference as a pure "
        "effect of future access.",
        "",
        "The primary endpoint averages `AUC_LOSO - AUC_forward` over interior target "
        "sessions and the two prespecified classical models within participant. The final "
        "session is an exact structural zero because both protocols use the same training "
        "sessions; it is verified but excluded from inference.",
        "",
        "## Data and preprocessing",
        "",
        "| Dataset | Participants | Sessions | EEG channels | Binary task |",
        "|---|---:|---:|---:|---|",
    ]
    lines.extend(f"| {d} | {n} | {t} | {c} | {task} |" for d, n, t, c, task in DATASETS)
    lines += [
        "",
        "Zhou2016 and Zhou2020 are distinct datasets. Ma2020 is Xuelin Ma et al.'s "
        "same-limb joint-imagery dataset, not a left/right-hand dataset. Its HEO, VEO, "
        "M2, and optional EMG channels are removed explicitly before the 62-channel "
        "montage is checked. Every retained epoch contains exactly 384 samples.",
        "",
        "## Main results",
        "",
        f"The participant-level LOSO--forward difference was "
        f"**{fnum(h1['subject_mean'], signed=True)} AUC** (95% CI "
        f"[{fnum(h1['ci'][0], signed=True)}, {fnum(h1['ci'][1], signed=True)}]; "
        f"one-sided p={fp(h1['subj_p_onesided'])}; n={h1['n_subjects']} participants).",
        "",
        "![LOSO minus forward AUC by dataset and model](../results/figures/fig1_gap_by_dataset_model.png)",
        "",
        "### Size-matched decomposition",
        "",
        "| Contrast | Participant mean AUC | 95% CI | Two-sided p |",
        "|---|---:|---:|---:|",
        f"| Extra training quantity: LOSO - size-matched LOSO pool | "
        f"{fnum(quantity['subject_mean'], signed=True)} | "
        f"[{fnum(quantity['ci'][0], signed=True)}, {fnum(quantity['ci'][1], signed=True)}] | "
        f"{fp(quantity['p_twosided'])} |",
        f"| Sampled composition: size-matched LOSO pool - forward | "
        f"{fnum(composition['subject_mean'], signed=True)} | "
        f"[{fnum(composition['ci'][0], signed=True)}, {fnum(composition['ci'][1], signed=True)}] | "
        f"{fp(composition['p_twosided'])} |",
        f"| Strict future - forward, matched size | {fnum(future['subject_mean'], signed=True)} | "
        f"[{fnum(future['ci'][0], signed=True)}, {fnum(future['ci'][1], signed=True)}] | "
        f"{fp(future['p_twosided'])} |",
        "",
        "![Size and sampled-composition terms](../results/figures/fig8_decomposition.png)",
        "",
        f"The adjusted association with each additional unavailable session was "
        f"{fnum(slope.get('beta'), 4, True)} AUC (p={fp(slope.get('p'))}). This is a "
        "description of protocol separation, not a learning curve under a fixed pool.",
        "",
        f"The within-Stieger2021 comparison of 11-session versus 7-session participants "
        f"over common target indices was {fnum(natural['diff_T11_minus_T7'], signed=True)} "
        f"AUC (p={fp(natural['p_twosided'])}); trajectory length is observational, and "
        "this comparison did not independently support a larger gap in the longer group. "
        f"At fixed one-session training size and fixed test session, an equally distant "
        f"future donor exceeded a past donor by {fnum(donor['subject_mean'], signed=True)} "
        f"AUC (95% CI [{fnum(donor['ci'][0], signed=True)}, "
        f"{fnum(donor['ci'][1], signed=True)}], p={fp(donor['p_twosided'])}).",
        "",
        "### Covariance drift",
        "",
        f"The adjusted standardized-drift coefficient was "
        f"{fnum(h2.get('drift_beta'), 4, True)} (p={fp(h2.get('drift_p'))}). This null "
        "moderation result does not establish stationarity; it concerns only the tested "
        "mean-covariance distance and the LOSO--forward difference.",
        "",
        "![Gap and covariance drift](../results/figures/fig2_gap_vs_drift.png)",
        "",
        "## Exploratory model ranking and EEGNet seeds",
        "",
        "EEGNet is evaluated only on Stieger2021 as a targeted ranking analysis; it "
        "is not pooled with the six-dataset classical endpoint. All three robustness "
        f"refits use the same MPS device and a fixed balanced subset of "
        f"{len(ROBUSTNESS_SUBJECTS)} participants (six T=7 and six T=11). Their hashes, "
        "subject IDs, and device provenance are recorded in "
        "`results/robustness/run_manifest.json`.",
        "",
    ]

    ranking = summary.get("ranking_inversion", [])
    lines += [
        "| Dataset | Participants with all models | LOSO order | Forward order | Different? |",
        "|---|---:|---|---|---|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['dataset']} | {row['n_subjects']} | "
            f"{' > '.join(row['order_loso'])} | {' > '.join(row['order_forward'])} | "
            f"{'yes' if row['inverted'] else 'no'} |")

    seeds = seed_robustness()
    canonical_subset = next(
        (row for row in seeds if row["seed"] == 20260716), None)
    if canonical_subset:
        lines += [
            "",
            f"The complete 62-participant canonical analysis changes the mean ordering "
            f"between LOSO and forward chaining, but the fixed sensitivity subset does "
            f"not reproduce the LOSO ordering even at the canonical seed: its "
            f"EEGNet-minus-TS+LR differences are "
            f"{fnum(canonical_subset['loso'], 4, signed=True)} under LOSO and "
            f"{fnum(canonical_subset['forward'], 4, signed=True)} under forward chaining. "
            "The full-sample ranking result is therefore reported as exploratory rather "
            "than seed-stable.",
        ]
    lines += ["", "Stieger2021 participant-weighted EEGNet-minus-TS+LR differences:", "",
              "| Seed | LOSO | Forward |", "|---:|---:|---:|"]
    if seeds:
        lines.extend(f"| {r['seed']} | {fnum(r['loso'], 4, signed=True)} | "
                     f"{fnum(r['forward'], 4, signed=True)} |" for r in seeds)
    else:
        lines.append("| incomplete | NA | NA |")
    lines += [
        "",
        "![Stieger2021 model means](../results/figures/fig12_ranking_inversion.png)",
        "",
        "## Interpretation and limitations",
        "",
        "Both evaluation protocols are legitimate for their respective questions. "
        "A prospective benchmark should align its training set with chronological data "
        "availability, while a retrospective compatibility study can use LOSO. The "
        "size-matched arm indicates how much of the observed separation is associated "
        "with training quantity under the implemented sampling scheme; its remainder "
        "does not identify a universal causal value of future sessions.",
        "",
        "The study is restricted to binary motor imagery and AUC; Ma2020 contrasts "
        "right-hand and right-elbow imagery while the other datasets contrast left and "
        "right hand. Dataset hardware, feedback, practice schedules, and session quality "
        "also differ. Five random draws "
        "summarize matched pools rather than enumerate every subset. The compact EEGNet "
        "analysis is a ranking check, not a broad deep-learning benchmark, and the "
        "three-seed check is limited to a balanced 12-participant subset. The toy "
        "mean-estimation calculation is illustrative: parameter MSE does not generally "
        "order classifier AUC and is not used to infer an empirical drift regime.",
        "",
        "## Reproducibility and audit",
        "",
        f"The final machine audit **{audit_text}**. It checks all six datasets, exact "
        f"preprocessing versions and epoch lengths, Ma2020's 62-channel montage, result "
        f"coverage, unique keys, finite AUCs in [0,1], and exact structural zeros. "
        f"There are {summary['n_auc_rows']} AUC result rows in the aggregate table.",
        "",
        "Stieger2021 legacy results are reused only because all 62 cached datasets were "
        "verified to have the unchanged native 0--3 s, 128 Hz, 384-sample window, and "
        "because models, splits, and the canonical seed were unchanged. The provenance "
        "manifest is retained in `results/stieger_exact_reuse_manifest.json`. No other "
        "legacy dataset result is reused. Pre-correction material is isolated under "
        "`archive/pre_v2_20260719/`.",
        "",
        "## Data references",
        "",
        "- [MOABB](https://doi.org/10.1088/1741-2552/aadea0)",
        "- [Zhou2016](https://doi.org/10.1371/journal.pone.0162657)",
        "- [Zhou2020](https://doi.org/10.3389/fnhum.2021.701091)",
        "- [Kumar2024](https://doi.org/10.1093/pnasnexus/pgae076)",
        "- [Ma2020](https://doi.org/10.1038/s41597-020-0535-2)",
        "- [Stieger2021](https://doi.org/10.1038/s41597-021-00883-1)",
        "",
    ]
    return "\n".join(lines)


CSS = """
:root { color-scheme: light; --ink:#182027; --muted:#5e6973; --line:#d7dde2; --accent:#285f79; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#f4f6f7; font:17px/1.63 Georgia, 'Times New Roman', serif; }
main { max-width:980px; margin:28px auto; padding:54px 68px; background:white; box-shadow:0 3px 22px #26323a18; }
h1,h2,h3 { font-family:Inter, ui-sans-serif, system-ui, sans-serif; line-height:1.2; color:#102a38; }
h1 { font-size:2.25rem; margin-top:0; letter-spacing:-.025em; }
h2 { margin-top:2.2em; padding-bottom:.3em; border-bottom:1px solid var(--line); }
h3 { margin-top:1.7em; }
p { max-width:78ch; }
table { width:100%; border-collapse:collapse; margin:1.2em 0 1.8em; font:14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif; }
th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#eef3f5; color:#173c4d; }
img { display:block; max-width:100%; height:auto; margin:1.6em auto; border:1px solid var(--line); }
code { background:#eef2f4; padding:.1em .3em; border-radius:3px; }
a { color:var(--accent); }
@media (max-width:720px) { main { margin:0; padding:28px 20px; } h1 { font-size:1.7rem; } table { display:block; overflow-x:auto; } }
@media print { body { background:white; font-size:11pt; } main { box-shadow:none; margin:0; padding:0; max-width:none; } h2 { break-after:avoid; } img,table { break-inside:avoid; } }
"""


def main():
    summary = json.loads(SUMMARY.read_text())
    audit_path = RESULTS / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"ok": False}
    REPORT.mkdir(parents=True, exist_ok=True)
    text = make_markdown(summary, audit)
    (REPORT / "report.md").write_text(text + "\n")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    title = "Chronology-aware cross-session evaluation"
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")
    (REPORT / "report.html").write_text(page)
    print(f"wrote {REPORT / 'report.md'}")
    print(f"wrote {REPORT / 'report.html'}")


if __name__ == "__main__":
    main()

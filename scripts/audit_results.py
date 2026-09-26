#!/usr/bin/env python3
"""Audit submitted result tables, or the original raw-cache workspace explicitly.

The default scope requires result-level files only. It does not claim to verify
EEG channels, epoch length, preprocessing, or the correctness of fitted models.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import importlib
import importlib.metadata
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (CLASSICAL_MODELS, MATCHED_DRAWS, MODELS,
                        PREPROCESS_VERSION, PRIMARY_DATASETS, RAW_DIR,
                        PROPOSAL_ALIASES, PROTOCOLS, RANDOM_STATE,
                        RESAMPLE_HZ, RESULTS_DIR,
                        ROBUSTNESS_SUBJECTS)  # noqa: E402


def check_frame(path, keys, errors):
    if not path.exists():
        errors.append(f"missing {path.relative_to(ROOT)}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.duplicated(keys).any():
        errors.append(f"duplicate keys in {path.name}: {keys}")
    if "auc" in frame:
        bad = ~np.isfinite(frame["auc"]) | ~frame["auc"].between(0, 1)
        if bad.any():
            errors.append(f"{int(bad.sum())} invalid AUC rows in {path.name}")
    return frame


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_cache_audit():
    """Historical full-workspace audit, including exploratory EEGNet outputs."""
    from src.data import load_subject_sessions, subject_list
    errors = []
    detail = {}
    all_main = []
    stieger_robust_expected = 0
    if PROPOSAL_ALIASES:
        errors.append(f"dataset aliases must be empty, found {PROPOSAL_ALIASES}")
    for dataset in PRIMARY_DATASETS:
        subjects = subject_list(dataset)
        expected_main = expected_matched = expected_transfer = 0
        dataset_models = MODELS if dataset == "Stieger2021" else CLASSICAL_MODELS
        cache_shapes = {}
        for subject in subjects:
            sessions = load_subject_sessions(dataset, subject, cache=True)
            T = len(sessions)
            shapes = sorted({tuple(s["X"].shape[1:]) for s in sessions})
            cache_shapes[str(subject)] = shapes
            if any(shape[1] != int(3 * RESAMPLE_HZ) for shape in shapes):
                errors.append(f"{dataset} S{subject}: non-384 epoch {shapes}")
            versions = {s.get("preprocess_version") for s in sessions}
            if versions != {PREPROCESS_VERSION}:
                errors.append(f"{dataset} S{subject}: cache version {versions}")
            if dataset == "Ma2020" and any(shape[0] != 62 for shape in shapes):
                errors.append(f"Ma2020 S{subject}: expected 62 EEG channels, got {shapes}")
            if dataset == "Ma2020" and any(
                    s.get("classes") != ["right_hand", "right_elbow"] for s in sessions):
                errors.append(f"Ma2020 S{subject}: incorrect class labels")
            if any(set(np.unique(s["y"]).tolist()) != {0, 1} for s in sessions):
                errors.append(f"{dataset} S{subject}: non-binary or missing class in a session")
            expected_main += len(dataset_models) * (4 * T - 3)
            expected_matched += len(CLASSICAL_MODELS) * ((T - 1) + ((T - 1) // 2))
            expected_transfer += len(CLASSICAL_MODELS) * T * (T - 1)
            if dataset == "Stieger2021" and int(subject) in ROBUSTNESS_SUBJECTS:
                stieger_robust_expected += 2 * T - 1

        main = check_frame(
            RAW_DIR / f"auc_{dataset}.csv",
            ["dataset", "subject", "model", "protocol", "target_session"], errors)
        matched = check_frame(
            RAW_DIR / f"auc_matched_{dataset}.csv",
            ["dataset", "subject", "model", "protocol", "target_session"], errors)
        transfer = check_frame(
            RAW_DIR / f"transfer_{dataset}.csv",
            ["dataset", "subject", "model", "train_session", "test_session"], errors)
        for name, frame, expected in [
            ("main", main, expected_main), ("matched", matched, expected_matched),
            ("transfer", transfer, expected_transfer),
        ]:
            if len(frame) != expected:
                errors.append(f"{dataset} {name}: {len(frame)} rows, expected {expected}")
        if not main.empty:
            if set(main["model"]) != set(dataset_models):
                errors.append(f"{dataset} main: model set {sorted(main['model'].unique())}")
            if set(main["protocol"]) != set(PROTOCOLS):
                errors.append(f"{dataset} main: protocol set {sorted(main['protocol'].unique())}")
            if "seed" not in main or main["seed"].isna().any():
                errors.append(f"{dataset} main: missing seed provenance")
        if not matched.empty:
            if set(matched["model"]) != set(CLASSICAL_MODELS):
                errors.append(f"{dataset} matched: model set {sorted(matched['model'].unique())}")
            if set(matched["protocol"]) != {"loso_matched", "future_matched"}:
                errors.append(f"{dataset} matched: unexpected protocols")
            if "n_draws" not in matched or set(matched["n_draws"].astype(int)) != {MATCHED_DRAWS}:
                errors.append(f"{dataset} matched: expected n_draws={MATCHED_DRAWS}")
            if "seed" not in matched or matched["seed"].isna().any():
                errors.append(f"{dataset} matched: missing seed provenance")
        if not main.empty:
            all_main.append(main)
        detail[dataset] = {
            "subjects": len(subjects), "cache_shapes": cache_shapes,
            "rows": {"main": len(main), "matched": len(matched),
                     "transfer": len(transfer)},
        }

    robustness_detail = {}
    for seed in [RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2]:
        path = RESULTS_DIR / "robustness" / f"eegnet_Stieger2021_seed{seed}.csv"
        frame = check_frame(
            path, ["dataset", "subject", "model", "protocol", "target_session"],
            errors)
        if len(frame) != stieger_robust_expected:
            errors.append(
                f"Stieger2021 robustness seed {seed}: {len(frame)} rows, "
                f"expected {stieger_robust_expected}")
        if not frame.empty:
            if set(frame["subject"].astype(int)) != set(ROBUSTNESS_SUBJECTS):
                errors.append(
                    f"Stieger2021 robustness seed {seed}: invalid subject subset")
            if set(frame["model"]) != {"eegnet"} or set(frame["protocol"]) != {
                    "loso", "forward_expanding"}:
                errors.append(f"Stieger2021 robustness seed {seed}: invalid arms")
            if "seed" not in frame or set(frame["seed"].astype(int)) != {seed}:
                errors.append(f"Stieger2021 robustness seed {seed}: invalid seed column")
        robustness_detail[str(seed)] = len(frame)

    manifest_path = RESULTS_DIR / "robustness" / "run_manifest.json"
    if not manifest_path.exists():
        errors.append("missing results/robustness/run_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
            runs = {int(run["seed"]): run for run in manifest.get("runs", [])}
            expected_seeds = {RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2}
            if set(runs) != expected_seeds:
                errors.append(f"robustness manifest seeds {sorted(runs)}")
            for seed in expected_seeds.intersection(runs):
                run = runs[seed]
                path = ROOT / run["file"]
                if run.get("compute_device") != "mps":
                    errors.append(f"robustness seed {seed}: device is not mps")
                if set(map(int, run.get("subject_ids", []))) != set(ROBUSTNESS_SUBJECTS):
                    errors.append(f"robustness seed {seed}: manifest subject mismatch")
                if not path.exists() or run.get("sha256") != sha256(path):
                    errors.append(f"robustness seed {seed}: manifest hash mismatch")
                if int(run.get("rows", -1)) != stieger_robust_expected:
                    errors.append(f"robustness seed {seed}: manifest row count mismatch")
        except Exception as exc:
            errors.append(f"invalid robustness manifest: {exc}")

    if all_main:
        frame = pd.concat(all_main, ignore_index=True)
        if set(frame["dataset"]) != set(PRIMARY_DATASETS):
            errors.append("main results do not contain exactly the six registered datasets")
        pivot = frame.pivot_table(
            index=["dataset", "subject", "model", "target_session"],
            columns="protocol", values="auc").reset_index()
        max_t = pivot.groupby(["dataset", "subject"])["target_session"].transform("max")
        structural = pivot[pivot["target_session"] == max_t]
        if not structural.empty:
            delta = (structural["loso"] - structural["forward_expanding"]).abs()
            if not np.all(delta.fillna(np.inf).to_numpy() == 0):
                errors.append(f"structural zeros fail; max |delta|={delta.max()}")

    # Major-revision outputs: Ma2020 day-level refits, revised estimands, and
    # fixed-size editable SVG sources used by the manuscript.
    day_path = RESULTS_DIR / "sensitivity" / "ma2020_day_level.csv"
    day = check_frame(
        day_path, ["dataset", "subject", "model", "protocol", "target_day"],
        errors)
    if len(day) != 200:
        errors.append(f"Ma2020 day sensitivity: {len(day)} rows, expected 200")
    if not day.empty:
        if set(day["subject"].astype(int)) != set(range(1, 26)):
            errors.append("Ma2020 day sensitivity: incomplete subject set")
        if set(day["model"]) != set(CLASSICAL_MODELS):
            errors.append("Ma2020 day sensitivity: incorrect model set")
        if set(day["protocol"]) != {"loso", "forward_expanding"}:
            errors.append("Ma2020 day sensitivity: incorrect protocol set")
        bad_ba = ~np.isfinite(day["balanced_accuracy"]) | \
            ~day["balanced_accuracy"].between(0, 1)
        if bad_ba.any():
            errors.append(f"Ma2020 day sensitivity: {int(bad_ba.sum())} invalid BA rows")

    revision_path = RESULTS_DIR / "revision_summary.json"
    revision = {}
    if not revision_path.exists():
        errors.append("missing results/revision_summary.json")
    else:
        revision = json.loads(revision_path.read_text())
        if int(revision.get("structural_zero_rows", -1)) != 2 * 138:
            errors.append("revision summary: expected 276 classical structural-zero rows")
        estimands = revision.get("estimands", {})
        if set(estimands) != {"conditional", "all_origin"}:
            errors.append("revision summary: missing conditional/all-origin estimands")

    svg_names = [
        "fig0_protocol_logic.svg", "fig1_gap_forest.svg", "fig2_estimands_distribution.svg",
        "fig3_history_by_dataset.svg", "fig4_drift_facets.svg",
        "fig5_ma_time_unit.svg",
    ]
    svg_dimensions = set()
    for name in svg_names:
        path = RESULTS_DIR / "figures" / name
        if not path.exists():
            errors.append(f"missing results/figures/{name}")
            continue
        match = re.search(r'<svg[^>]+width="([^"]+)"[^>]+height="([^"]+)"',
                          path.read_text()[:2000])
        if match is None:
            errors.append(f"cannot read SVG dimensions for {name}")
        else:
            svg_dimensions.add(match.groups())
    if len(svg_dimensions) > 1:
        errors.append(f"revised SVG canvases differ: {sorted(svg_dimensions)}")

    module_names = {
        "moabb": "moabb", "mne": "mne", "pyriemann": "pyriemann",
        "scikit-learn": "sklearn", "statsmodels": "statsmodels",
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "torch": "torch", "matplotlib": "matplotlib",
    }
    packages = {}
    for package, module_name in module_names.items():
        try:
            packages[package] = str(importlib.import_module(module_name).__version__)
        except (ImportError, AttributeError):
            packages[package] = None
    locked = {}
    for raw_line in (ROOT / "requirements.txt").read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if "==" in line:
            package, locked_version = line.split("==", 1)
            locked[package.strip()] = locked_version.strip()
    for package, locked_version in locked.items():
        if packages.get(package) != locked_version:
            errors.append(
                f"runtime {package}={packages.get(package)}, expected {locked_version}")
    audit = {
        "preprocess_version": PREPROCESS_VERSION,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "packages": packages},
        "datasets": list(PRIMARY_DATASETS),
        "matched_draws": MATCHED_DRAWS,
        "robustness_rows": robustness_detail,
        "ma2020_day_rows": len(day),
        "revision_svg_dimensions": sorted(svg_dimensions),
        "ok": not errors,
        "errors": errors,
        "detail": detail,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "audit.json", "w") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps({"ok": audit["ok"], "errors": errors,
                      "rows": {d: v["rows"] for d, v in detail.items()}}, indent=2))
    raise SystemExit(0 if audit["ok"] else 1)


def submitted_classical_audit(root=ROOT):
    """Return independent consistency checks using only submitted result tables."""
    root = Path(root)
    results = root / "results"
    errors = []
    expected_subjects = {
        "BNCI2014_004": 9, "Zhou2016": 4, "Zhou2020": 20,
        "Kumar2024": 18, "Ma2020": 25, "Stieger2021": 62,
    }
    keys = ["dataset", "subject", "model", "target_session"]

    def read_frame(relative, required, unique):
        path = results / relative
        if not path.is_file():
            errors.append(f"missing results/{relative}")
            return pd.DataFrame()
        frame = pd.read_csv(path)
        absent = set(required) - set(frame.columns)
        if absent:
            errors.append(f"results/{relative}: missing columns {sorted(absent)}")
            return pd.DataFrame()
        if "model" in frame:
            frame = frame[frame["model"].isin(CLASSICAL_MODELS)].copy()
        if frame.duplicated(unique).any():
            errors.append(f"results/{relative}: duplicate keys {unique}")
        return frame

    wide = read_frame("gap_wide.csv", keys + [
        "T", "is_structural_zero", "delta", "loso", "forward_expanding",
        "loso_matched", "future_matched", "subject_uid"], keys)
    long_df = read_frame("auc_long.csv", keys + [
        "protocol", "auc", "seed", "auc_sd_draws", "n_draws"], keys + ["protocol"])
    day = read_frame("sensitivity/ma2020_day_level.csv", [
        "dataset", "subject", "model", "protocol", "target_day", "auc"],
        ["dataset", "subject", "model", "protocol", "target_day"])
    summary_path = results / "revision_summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read results/revision_summary.json: {exc}")
        summary = {}

    detail = {}
    for name, frame in (("auc_long.csv", long_df), ("ma2020_day_level.csv", day)):
        if not frame.empty and (not np.isfinite(frame["auc"]).all()
                                or not frame["auc"].between(0, 1).all()):
            errors.append(f"{name}: non-finite or out-of-range AUC")

    if wide.empty or long_df.empty or day.empty:
        errors.append("submitted classical input tables must be nonempty")
    else:
        if set(wide["dataset"]) != set(expected_subjects):
            errors.append("gap_wide.csv: expected exactly the six submitted datasets")
        for dataset, n_subjects in expected_subjects.items():
            frame = wide[wide["dataset"] == dataset]
            observed = set(frame["subject"].astype(int))
            if observed != set(range(1, n_subjects + 1)):
                errors.append(f"{dataset}: incomplete participant IDs")
            detail[dataset] = {"participants": len(observed), "wide_rows": len(frame)}
        for (dataset, subject), frame in wide.groupby(["dataset", "subject"]):
            if frame["T"].nunique() != 1:
                errors.append(f"{dataset}/{subject}: inconsistent trajectory length")
                continue
            T = int(frame["T"].iloc[0])
            if T < 3 or set(frame["model"]) != set(CLASSICAL_MODELS):
                errors.append(f"{dataset}/{subject}: incomplete classical decoder coverage")
            for model, model_frame in frame.groupby("model"):
                if set(model_frame["target_session"]) != set(range(1, T)):
                    errors.append(f"{dataset}/{subject}/{model}: incomplete target coverage")
        structural = wide["target_session"] == wide["T"] - 1
        if not np.array_equal(wide["is_structural_zero"].to_numpy(), structural.to_numpy()):
            errors.append("gap_wide.csv: structural-zero labels disagree with endpoint")
        if not np.all(wide.loc[structural, "delta"].to_numpy() == 0):
            errors.append("gap_wide.csv: final-origin differences are not exactly zero")
        if not np.allclose(wide["delta"], wide["loso"] - wide["forward_expanding"],
                           atol=1e-12, rtol=0):
            errors.append("gap_wide.csv: delta disagrees with the paired AUC difference")
        for protocol in ("loso", "forward_expanding", "loso_matched", "future_matched"):
            valid = wide[protocol].dropna()
            if not np.isfinite(valid).all() or not valid.between(0, 1).all():
                errors.append(f"gap_wide.csv: invalid {protocol} AUC")
            observed = long_df[long_df["protocol"] == protocol][keys + ["auc"]]
            paired = wide[keys + [protocol]].merge(
                observed, on=keys, how="left", validate="one_to_one"
            ) if not observed.duplicated(keys).any() and not wide.duplicated(keys).any() else None
            if paired is not None and not np.allclose(
                    paired[protocol], paired["auc"], atol=1e-12, rtol=0, equal_nan=True):
                errors.append(f"{protocol}: auc_long.csv and gap_wide.csv disagree")
        matched = long_df[long_df["protocol"].isin(["loso_matched", "future_matched"])]
        if (matched.empty or matched["n_draws"].isna().any()
                or set(matched["n_draws"].astype(int)) != {MATCHED_DRAWS}
                or matched["auc_sd_draws"].isna().any()
                or not np.isfinite(matched["auc_sd_draws"]).all()
                or (matched["auc_sd_draws"] < 0).any()):
            errors.append("auc_long.csv: invalid matched-reference draw metadata")
        if long_df["seed"].isna().any():
            errors.append("auc_long.csv: missing seed provenance")
        if len(day) != 200 or set(day["subject"]) != set(range(1, 26)):
            errors.append("Ma2020 day results: expected 25 participants and 200 rows")
        if (set(day["model"]) != set(CLASSICAL_MODELS)
                or set(day["protocol"]) != {"loso", "forward_expanding"}
                or set(day["target_day"]) != {1, 2}):
            errors.append("Ma2020 day results: incorrect decoder/protocol/target coverage")
        for label, frame in (("conditional", wide[~structural]), ("all_origin", wide)):
            participant = frame.groupby(["dataset", "subject"])["delta"].mean()
            reported = summary.get("estimands", {}).get(label, {}).get("participant_weighted", {})
            if (reported.get("n") != len(participant)
                    or not np.isclose(reported.get("mean", np.nan), participant.mean(),
                                      atol=1e-12, rtol=0)):
                errors.append(f"revision_summary.json: {label} participant estimate disagrees")
        if summary.get("structural_zero_rows") != int(structural.sum()):
            errors.append("revision_summary.json: structural-zero row count disagrees")
        conditional = wide[~structural].dropna(subset=["loso_matched"])
        strict = conditional.dropna(subset=["future_matched"])
        supports = {"primary_conditional": conditional, "strict_future": strict}
        for support_name, frame in supports.items():
            reported = summary.get("common_support_contrasts", {}).get(support_name, {})
            contrasts = {
                "total": ("loso", "forward_expanding"),
                "quantity_uniform": ("loso", "loso_matched"),
                "composition_uniform": ("loso_matched", "forward_expanding"),
            }
            if support_name == "strict_future":
                contrasts["strict_future"] = ("future_matched", "forward_expanding")
            for contrast_name, (left, right) in contrasts.items():
                values = (frame[left] - frame[right]).groupby(
                    [frame["dataset"], frame["subject"]]).mean()
                result = reported.get(contrast_name, {})
                if (result.get("n_fits") != len(frame) or result.get("n") != len(values)
                        or not np.isclose(result.get("mean", np.nan), values.mean(),
                                          atol=1e-12, rtol=0)):
                    errors.append(f"revision_summary.json: {support_name}/{contrast_name} disagrees")
        mc = summary.get("matched_monte_carlo", {})
        if "aggregate_mcse" in mc:
            errors.append("revision_summary.json: obsolete independence-based aggregate_mcse")
        uniform_draws = matched[matched["protocol"] == "loso_matched"]
        if not uniform_draws.duplicated(keys).any() and not wide.duplicated(keys).any():
            for label, frame in (("conditional", conditional), ("strict_future", strict)):
                draws = frame[keys].merge(uniform_draws, on=keys, how="left", validate="one_to_one")
                # Saved SD used ddof=0. Divide by sqrt(B-1) for the sample-based
                # marginal standard error; preserve shared-draw dependence.
                marginal_se = draws["auc_sd_draws"] / np.sqrt(draws["n_draws"] - 1)
                participant_bound = marginal_se.groupby([draws["dataset"], draws["subject"]]).mean()
                reported = mc.get("by_support", {}).get(label, {})
                if reported.get("draws") != [MATCHED_DRAWS]:
                    errors.append(f"revision_summary.json: {label} missing matched draw count")
                expected_mc = {
                    "participants": len(participant_bound),
                    "participant_targets": len(frame[["dataset", "subject", "target_session"]].drop_duplicates()),
                    "model_target_rows": len(frame),
                    "median_marginal_se": marginal_se.median(),
                    "q95_marginal_se": marginal_se.quantile(0.95),
                    "max_marginal_se": marginal_se.max(),
                    "triangle_se_bound_plugin": participant_bound.mean(),
                }
                for key, expected in expected_mc.items():
                    if not np.isclose(reported.get(key, np.nan), expected, atol=1e-12, rtol=0):
                        errors.append(f"revision_summary.json: {label}/{key} disagrees")

    svg_dimensions = set()
    manuscript = root / "paper" / "main.tex"
    if not manuscript.is_file():
        errors.append("missing paper/main.tex")
    else:
        figure_names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                                  manuscript.read_text())
        for name in figure_names:
            figure = results / "figures" / Path(name).name
            for suffix in (".pdf", ".svg"):
                if not figure.with_suffix(suffix).is_file():
                    errors.append(f"missing results/figures/{figure.stem}{suffix}")
            svg = figure.with_suffix(".svg")
            if svg.is_file():
                match = re.search(r'<svg[^>]+width="([^"]+)"[^>]+height="([^"]+)"',
                                  svg.read_text()[:2000])
                if match:
                    svg_dimensions.add(match.groups())
                else:
                    errors.append(f"cannot read {svg.name} canvas dimensions")
    if len(svg_dimensions) > 1:
        errors.append("manuscript SVG canvas dimensions differ")
    packages = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "statsmodels"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "scope": "submitted-classical", "ok": not errors, "errors": errors,
        "checked": ["classical result keys and participant/target coverage", "paired AUC agreement",
                    "final-origin structural zeros", "saved estimand and support-specific contrast means",
                    "matched-draw metadata and marginal-SE/triangle-bound summaries",
                    "Ma2020 day result coverage", "manuscript vector figure presence and dimensions"],
        "not_checked": ["raw EEG", "epoch length or channel count", "model refitting",
                        "preprocessing correctness", "historical cache provenance"],
        "environment": {"python": sys.version.split()[0], "packages": packages},
        "detail": detail,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["submitted-classical", "raw-cache"],
                        default="submitted-classical")
    args = parser.parse_args()
    if args.scope == "raw-cache":
        raw_cache_audit()
        return
    audit = submitted_classical_audit()
    path = RESULTS_DIR / "submitted_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["ok"] else 1)


if __name__ == "__main__":
    main()

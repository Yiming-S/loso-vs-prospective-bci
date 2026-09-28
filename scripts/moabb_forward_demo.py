"""Run ForwardSessionSplitter inside MOABB's CrossSessionEvaluation.

Demonstrates that the chronological expanding-window protocol can be obtained
from the unmodified MOABB benchmark by passing
    CrossSessionEvaluation(..., cv_class=ForwardSessionSplitter,
                           cv_kwargs={"mode": "expanding"})
and checks the resulting per-session AUCs against the manuscript's
forward_expanding results for BNCI2014_004 (results/raw/auc_BNCI2014_004.csv).
The default LOSO evaluation (cv_class=None) is run for comparison.

Preprocessing matches the manuscript: 8--35 Hz, [0, 3) s, 128 Hz.  Decoders are
the manuscript's fixed CSP+LDA and TS+LR pipelines from src/models.py.
Writes results/moabb_demo/moabb_forward_demo.csv and a comparison JSON.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import src.config as cfg  # noqa: E402  (routes MNE/MOABB caches into the project)
from src.models import build_csp_lda, build_ts_lr  # noqa: E402
from src.splitters import ForwardSessionSplitter  # noqa: E402

import mne  # noqa: E402
mne.set_log_level("ERROR")
from moabb import set_log_level  # noqa: E402
from moabb.datasets import BNCI2014_004  # noqa: E402
from moabb.evaluations import CrossSessionEvaluation  # noqa: E402
from moabb.paradigms import LeftRightImagery  # noqa: E402

set_log_level("error")
OUT = ROOT / "results" / "moabb_demo"
OUT.mkdir(parents=True, exist_ok=True)


def session_index(label: str) -> int:
    import re
    m = re.search(r"\d+", str(label))
    return int(m.group()) if m else -1


def run(cv_class, cv_kwargs, suffix):
    paradigm = LeftRightImagery(fmin=cfg.FMIN, fmax=cfg.FMAX, tmin=cfg.TMIN,
                                tmax=cfg.TMAX, resample=cfg.RESAMPLE_HZ)
    dataset = BNCI2014_004()
    pipelines = {"CSP+LDA": build_csp_lda(3), "TS+LR": build_ts_lr()}
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm, datasets=[dataset], overwrite=True, suffix=suffix,
        hdf5_path=str(OUT / "moabb_results"), cv_class=cv_class, cv_kwargs=cv_kwargs,
        n_jobs=1,
    )
    t0 = time.time()
    res = evaluation.process(pipelines)
    res["protocol"] = "forward_expanding" if cv_class is not None else "loso"
    res["target_session"] = res["session"].map(session_index)
    res["subject"] = res["subject"].astype(int)
    print(f"{suffix}: {len(res)} rows in {time.time()-t0:.0f}s")
    return res


def main():
    fwd = run(ForwardSessionSplitter, {"mode": "expanding"}, "fwd")
    loso = run(None, None, "loso")
    demo = pd.concat([fwd, loso], ignore_index=True)
    demo.to_csv(OUT / "moabb_forward_demo.csv", index=False)

    paper = pd.read_csv(ROOT / "results" / "raw" / "auc_BNCI2014_004.csv")
    paper = paper[paper["model"].isin(["csp_lda", "ts_lr"])]
    name_map = {"csp_lda": "CSP+LDA", "ts_lr": "TS+LR"}
    paper = paper.assign(pipeline=paper["model"].map(name_map))
    merged = demo.merge(
        paper[["subject", "pipeline", "protocol", "target_session", "auc"]],
        on=["subject", "pipeline", "protocol", "target_session"], how="outer",
        suffixes=("_moabb", "_paper"), indicator=True)
    merged["abs_diff"] = (merged["score"] - merged["auc"]).abs()
    merged.to_csv(OUT / "moabb_vs_paper.csv", index=False)

    summary = {}
    for (proto, pipe), g in merged.groupby(["protocol", "pipeline"]):
        summary[f"{proto}/{pipe}"] = {
            "n_moabb": int(g["score"].notna().sum()),
            "n_paper": int(g["auc"].notna().sum()),
            "n_matched": int((g["_merge"] == "both").sum()),
            "mean_auc_moabb": float(g["score"].mean()),
            "mean_auc_paper": float(g["auc"].mean()),
            "max_abs_diff": float(g["abs_diff"].max()),
            "mean_abs_diff": float(g["abs_diff"].mean()),
        }
    # Expanding-window splits per subject: one per target t>=1 (4 for 5 sessions)
    fwd_counts = fwd.groupby(["subject", "pipeline"]).size().unique().tolist()
    summary["forward_splits_per_subject_pipeline"] = fwd_counts
    summary["forward_min_target_session"] = int(fwd["target_session"].min())
    summary["forward_n_train_at_t1"] = None
    (OUT / "moabb_vs_paper_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

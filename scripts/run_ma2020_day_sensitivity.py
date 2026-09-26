#!/usr/bin/env python3
"""Re-evaluate Ma2020 after grouping its 15 blocks into three recording days.

The released files are ordered as five blocks on each of three days.  This
sensitivity analysis keeps every trial, concatenates blocks 1--5, 6--10, and
11--15, and then repeats LOSO and expanding-window evaluation at the day level.
It is deliberately separate from the six-dataset primary result because the
primary analysis uses the released block identifier as its repeated unit.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE  # noqa: E402
from src.data import load_subject_sessions, subject_list  # noqa: E402
from src.models import build_model  # noqa: E402


def _fit_metrics(model_name, n_channels, X_train, y_train, X_test, y_test):
    estimator = build_model(model_name, n_channels, random_state=RANDOM_STATE)
    estimator.fit(X_train, y_train)
    probability = estimator.predict_proba(X_test)[:, 1]
    prediction = estimator.predict(X_test)
    return {
        "auc": float(roc_auc_score(y_test, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
    }


def _one_subject(subject):
    sessions = load_subject_sessions("Ma2020", int(subject))
    if len(sessions) != 15:
        raise RuntimeError(f"Ma2020 S{subject}: expected 15 blocks, found {len(sessions)}")

    days = []
    for day in range(3):
        blocks = sessions[day * 5:(day + 1) * 5]
        days.append({
            "X": np.concatenate([block["X"] for block in blocks], axis=0),
            "y": np.concatenate([block["y"] for block in blocks], axis=0),
        })

    rows = []
    n_channels = int(days[0]["X"].shape[1])
    for model in ("csp_lda", "ts_lr"):
        for target in (1, 2):
            pools = {
                "loso": [j for j in range(3) if j != target],
                "forward_expanding": list(range(target)),
            }
            for protocol, donors in pools.items():
                X_train = np.concatenate([days[j]["X"] for j in donors], axis=0)
                y_train = np.concatenate([days[j]["y"] for j in donors], axis=0)
                metrics = _fit_metrics(
                    model, n_channels, X_train, y_train,
                    days[target]["X"], days[target]["y"],
                )
                rows.append({
                    "dataset": "Ma2020",
                    "subject": int(subject),
                    "model": model,
                    "time_unit": "recording_day",
                    "protocol": protocol,
                    "target_day": int(target),
                    "n_train_days": len(donors),
                    "n_train_trials": int(len(y_train)),
                    "n_test_trials": int(len(days[target]["y"])),
                    **metrics,
                    "seed": RANDOM_STATE,
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    subjects = subject_list("Ma2020")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(_one_subject, subject): subject for subject in subjects}
        for job in as_completed(jobs):
            subject = jobs[job]
            result = job.result()
            rows.extend(result)
            print(f"Ma2020 S{subject}: day-level sensitivity complete", flush=True)

    frame = pd.DataFrame(rows).sort_values(
        ["subject", "model", "target_day", "protocol"]
    )
    out_dir = ROOT / "results" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ma2020_day_level.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")


if __name__ == "__main__":
    main()

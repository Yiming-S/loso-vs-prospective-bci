#!/usr/bin/env python3
"""Main driver: evaluate every (dataset, subject) across models x protocols.

Writes results incrementally to results/raw/auc_<dataset>.csv (and, for classical
models, transfer_<dataset>.csv) so progress survives interruption and re-runs
resume by skipping (subject, model) pairs already recorded. Safe to run in the
background.

Examples
--------
  python scripts/run_experiment.py --datasets BNCI2014_004 Zhou2016 Zhou2020 Kumar2024 Ma2020 Stieger2021 \
         --models csp_lda ts_lr --transfer
  python scripts/run_experiment.py --datasets BNCI2014_004 Zhou2016 --models eegnet
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (RAW_DIR, PRIMARY_DATASETS, CLASSICAL_MODELS,
                        RANDOM_STATE)  # noqa: E402
from src.data import subject_list  # noqa: E402
from src.evaluate import evaluate_subject, transfer_matrix_subject  # noqa: E402


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _append(df: pd.DataFrame, path: Path):
    if df is None or len(df) == 0:
        return
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _done_pairs(path: Path, models):
    """Return set of (subject, model) already present, for resume."""
    if not path.exists():
        return set()
    try:
        d = pd.read_csv(path, usecols=["subject", "model"])
        return set(map(tuple, d.drop_duplicates().values.tolist()))
    except Exception:
        return set()


def run(datasets, models, subjects_limit=None, do_transfer=False, resample=None,
        protocols=None, tag="", random_state=RANDOM_STATE):
    """tag lets a protocol set write to its own raw file with its own resume state
    (e.g. tag="_matched" -> auc_matched_<ds>.csv), so adding protocols later does
    not require recomputing the ones already recorded."""
    for ds in datasets:
        auc_path = RAW_DIR / f"auc{tag}_{ds}.csv"
        tm_path = RAW_DIR / f"transfer_{ds}.csv"
        try:
            subs = subject_list(ds)
        except Exception as e:
            _log(f"!! {ds}: cannot list subjects: {e}")
            continue
        if subjects_limit:
            subs = subs[:subjects_limit]
        done = _done_pairs(auc_path, models)
        tm_done = _done_pairs(tm_path, models)
        _log(f"=== {ds}: {len(subs)} subjects, models={models}, "
             f"{len(done)} (subj,model) already done ===")

        for s in subs:
            todo = [m for m in models if (s, m) not in done]
            if todo:
                t0 = time.time()
                try:
                    kw = {} if protocols is None else {"protocols": protocols}
                    df, meta = evaluate_subject(
                        ds, s, models=todo, resample=resample,
                        random_state=random_state, log=_log, **kw)
                    _append(df, auc_path)
                    _log(f"  {ds} S{s} {todo} done in {time.time()-t0:.1f}s  meta={meta}")
                except Exception:
                    _log(f"  !! {ds} S{s} FAILED:\n{traceback.format_exc()}")
            # transfer matrix (classical models only)
            if do_transfer:
                tm_models = [m for m in models
                             if m in CLASSICAL_MODELS and (s, m) not in tm_done]
                if tm_models:
                    try:
                        tm = transfer_matrix_subject(ds, s, models=tm_models, resample=resample, log=_log)
                        _append(tm, tm_path)
                    except Exception:
                        _log(f"  !! {ds} S{s} transfer FAILED:\n{traceback.format_exc()}")
        _log(f"=== {ds} complete -> {auc_path.name} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PRIMARY_DATASETS)
    ap.add_argument("--models", nargs="+", default=CLASSICAL_MODELS)
    ap.add_argument("--subjects-limit", type=int, default=None)
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--resample", type=float, default=None)
    ap.add_argument("--protocols", nargs="+", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()
    _log(f"START datasets={args.datasets} models={args.models} "
         f"transfer={args.transfer} limit={args.subjects_limit}")
    run(args.datasets, args.models, args.subjects_limit, args.transfer,
        args.resample, protocols=args.protocols, tag=args.tag,
        random_state=args.seed)
    _log("ALL DONE")


if __name__ == "__main__":
    main()

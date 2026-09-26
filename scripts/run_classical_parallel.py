#!/usr/bin/env python3
"""Parallel, resumable subject-level runner for deterministic classical models.

Workers compute complete subject/model arms, while only the parent process appends
CSV output. This avoids concurrent writes and is intended for the computationally
large classical and size-matched analyses after caches have been populated.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASSICAL_MODELS, PRIMARY_DATASETS, RANDOM_STATE, RAW_DIR  # noqa: E402
from src.data import subject_list  # noqa: E402
from src.evaluate import evaluate_subject, transfer_matrix_subject  # noqa: E402


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def done_pairs(path):
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["subject", "model"]).drop_duplicates()
    return set(map(tuple, frame.itertuples(index=False, name=None)))


def append(frame, path):
    if frame is not None and not frame.empty:
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def work(job):
    dataset, subject, models, transfer_models, protocols, seed = job
    result = transfer = pd.DataFrame()
    if models:
        kwargs = {} if protocols is None else {"protocols": protocols}
        result, meta = evaluate_subject(
            dataset, subject, models=models, random_state=seed,
            log=lambda _message: None, **kwargs)
    else:
        meta = {"skipped": "main already complete"}
    if transfer_models:
        transfer = transfer_matrix_subject(
            dataset, subject, models=transfer_models, log=lambda _message: None)
    return dataset, subject, result, transfer, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=PRIMARY_DATASETS)
    parser.add_argument("--models", nargs="+", default=CLASSICAL_MODELS)
    parser.add_argument("--subjects", nargs="+", type=int,
                        help="optional explicit subject subset (use with one dataset)")
    parser.add_argument("--protocols", nargs="+")
    parser.add_argument("--tag", default="")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    jobs = []
    paths = {}
    for dataset in args.datasets:
        auc_path = RAW_DIR / f"auc{args.tag}_{dataset}.csv"
        transfer_path = RAW_DIR / f"transfer_{dataset}.csv"
        paths[dataset] = (auc_path, transfer_path)
        main_done = done_pairs(auc_path)
        transfer_done = done_pairs(transfer_path)
        subjects = args.subjects if args.subjects is not None else subject_list(dataset)
        for subject in subjects:
            models = [m for m in args.models if (subject, m) not in main_done]
            transfer_models = ([m for m in args.models
                                if (subject, m) not in transfer_done]
                               if args.transfer else [])
            if models or transfer_models:
                jobs.append((dataset, subject, models, transfer_models,
                             args.protocols, args.seed))
    log(f"scheduled {len(jobs)} subject jobs with {args.workers} workers")

    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_jobs = {pool.submit(work, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_jobs):
            job = future_jobs[future]
            try:
                dataset, subject, frame, transfer, meta = future.result()
                auc_path, transfer_path = paths[dataset]
                append(frame, auc_path)
                append(transfer, transfer_path)
                log(f"{dataset} S{subject}: main={len(frame)} transfer={len(transfer)} {meta}")
            except Exception:
                failures.append((job[0], job[1]))
                log(f"!! {job[0]} S{job[1]} failed:\n{traceback.format_exc()}")
    if failures:
        raise SystemExit(f"failed jobs: {failures}")
    log("all scheduled jobs complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repeat EEGNet LOSO/forward fits across fixed seeds for ranking robustness.

The canonical analysis uses one preregistered seed. This targeted check repeats
the long-trajectory datasets where ranking reversal is scientifically relevant
and writes one resumable file per seed under results/robustness/.
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

from src.config import RANDOM_STATE, RAW_DIR, RESULTS_DIR  # noqa: E402
from src.data import subject_list  # noqa: E402
from src.evaluate import evaluate_subject  # noqa: E402


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def reuse_canonical_seed(dataset, subjects, seed, path):
    """Materialize the canonical fit as one robustness seed when complete.

    This avoids fitting the identical seed twice.  The source remains the
    canonical raw result file, and only the two protocols requested by this
    robustness analysis are selected.
    """
    if seed != RANDOM_STATE or path.exists():
        return False
    canonical = RAW_DIR / f"auc_{dataset}.csv"
    if not canonical.exists():
        return False
    frame = pd.read_csv(canonical)
    keep = frame[
        (frame["model"] == "eegnet")
        & frame["protocol"].isin(["loso", "forward_expanding"])
        & frame["subject"].isin(subjects)
    ].copy()
    if set(keep["subject"].astype(int)) != set(map(int, subjects)):
        return False
    keep["seed"] = seed
    keep["robustness_provenance"] = "canonical_seed_reuse"
    keep.to_csv(path, index=False)
    log(f"{dataset} seed={seed}: reused complete canonical fits -> {path.name}")
    return True


def evaluate_job(job):
    dataset, subject, seed = job
    frame, _ = evaluate_subject(
        dataset, subject, models=["eegnet"],
        protocols=["loso", "forward_expanding"],
        random_state=seed, compute_drift=False, log=lambda _message: None)
    return subject, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Stieger2021"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2])
    parser.add_argument("--subjects-limit", type=int)
    parser.add_argument("--subjects", nargs="+", type=int,
                        help="optional explicit participant IDs (for resumable batches)")
    parser.add_argument("--refit-subjects", nargs="+", type=int,
                        help="remove and refit these participant IDs in each selected seed")
    parser.add_argument("--workers", type=int, default=1,
                        help="participant processes; 1 is safest for a single MPS device")
    parser.add_argument("--refit-canonical", action="store_true",
                        help="fit RANDOM_STATE instead of reusing its canonical result")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        subjects = subject_list(dataset)
        if args.subjects:
            requested = set(args.subjects)
            unknown = requested.difference(map(int, subjects))
            if unknown:
                raise SystemExit(f"{dataset}: unknown subjects {sorted(unknown)}")
            subjects = [subject for subject in subjects if int(subject) in requested]
        if args.subjects_limit:
            subjects = subjects[:args.subjects_limit]
        for seed in args.seeds:
            path = out_dir / f"eegnet_{dataset}_seed{seed}.csv"
            if not args.refit_canonical:
                reuse_canonical_seed(dataset, subjects, seed, path)
            if args.subjects and path.exists():
                old = pd.read_csv(path)
                selected = set(map(int, subjects))
                keep = old[old["subject"].astype(int).isin(selected)]
                if len(keep) != len(old):
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    keep.to_csv(tmp, index=False)
                    tmp.replace(path)
                    log(f"{dataset} seed={seed}: restricted existing rows to "
                        f"{len(selected)} requested subjects")
            if args.refit_subjects and path.exists():
                old = pd.read_csv(path)
                forced = set(args.refit_subjects)
                keep = old[~old["subject"].astype(int).isin(forced)]
                if len(keep) != len(old):
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    keep.to_csv(tmp, index=False)
                    tmp.replace(path)
                    log(f"{dataset} seed={seed}: removed subjects "
                        f"{sorted(forced)} for forced refit")
            done = set()
            if path.exists():
                done = set(pd.read_csv(path, usecols=["subject"])["subject"].astype(int))
            log(f"{dataset} seed={seed}: {len(done)}/{len(subjects)} subjects complete")
            todo = [int(subject) for subject in subjects if int(subject) not in done]
            if args.workers == 1:
                for subject in todo:
                    try:
                        log(f"[{dataset} S{subject}] EEGNet seed={seed} start")
                        _, frame = evaluate_job((dataset, subject, seed))
                        frame.to_csv(path, mode="a", header=not path.exists(), index=False)
                        log(f"[{dataset} S{subject}] seed={seed} complete ({len(frame)} rows)")
                    except Exception:
                        log(f"!! {dataset} S{subject} seed={seed} failed:\n"
                            f"{traceback.format_exc()}")
            else:
                jobs = [(dataset, subject, seed) for subject in todo]
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=args.workers) as pool:
                    futures = {pool.submit(evaluate_job, job): job[1] for job in jobs}
                    for future in concurrent.futures.as_completed(futures):
                        subject = futures[future]
                        try:
                            _, frame = future.result()
                            frame.to_csv(path, mode="a", header=not path.exists(), index=False)
                            log(f"[{dataset} S{subject}] seed={seed} complete ({len(frame)} rows)")
                        except Exception:
                            log(f"!! {dataset} S{subject} seed={seed} failed:\n"
                                f"{traceback.format_exc()}")
            log(f"complete -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Populate subject-level project caches in parallel without running models."""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import describe_sessions, load_subject_sessions, subject_list  # noqa: E402


def cache_one(dataset_subject):
    dataset, subject = dataset_subject
    sessions = load_subject_sessions(dataset, subject, cache=True)
    return subject, describe_sessions(sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subjects", nargs="+", type=int)
    args = parser.parse_args()
    subjects = args.subjects or subject_list(args.dataset)
    jobs = [(args.dataset, subject) for subject in subjects]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(cache_one, job): job[1] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            subject = futures[future]
            try:
                _, description = future.result()
                print(f"{args.dataset} S{subject}: {description}", flush=True)
            except Exception as exc:
                print(f"{args.dataset} S{subject}: FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()

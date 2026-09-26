#!/usr/bin/env python3
"""Reuse Stieger2021 results whose preprocessing is exactly unchanged in v2.

Stieger2021's MOABB interval was already the native [0, 3) s interval at 128 Hz,
so every legacy epoch has exactly 384 samples.  The model definitions, protocol
splits, and default random seed are also unchanged.  This script makes reuse
explicit and fail-closed: it verifies every subject cache and the expected result
coverage before copying data into the active result directory with provenance
columns.  No other dataset is eligible for this migration.
"""
from __future__ import annotations

import json
import hashlib
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR, PREPROCESS_VERSION, RANDOM_STATE, RAW_DIR  # noqa: E402

ARCHIVE = ROOT / "archive" / "pre_v2_20260719" / "results_raw"
DATASET = "Stieger2021"
N_SUBJECTS = 62
EXPECTED_ROWS = {
    "auc_Stieger2021.csv": 6618,
    "auc_matched_Stieger2021.csv": 1608,
    "transfer_Stieger2021.csv": 10784,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_legacy_caches() -> list[str]:
    checked = []
    for subject in range(1, N_SUBJECTS + 1):
        path = CACHE_DIR / f"{DATASET}_S{subject}_rs128.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            sessions = pickle.load(handle)
        if not sessions:
            raise RuntimeError(f"Empty cache: {path}")
        bad = [s.get("session") for s in sessions if s["X"].shape[2] != 384]
        channels = {int(s["X"].shape[1]) for s in sessions}
        if bad or channels != {60}:
            raise RuntimeError(
                f"Ineligible cache {path.name}: bad_sessions={bad}, channels={channels}")
        checked.append(path.name)
    return checked


def verify_and_copy(name: str) -> dict:
    source = ARCHIVE / name
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    if len(data) != EXPECTED_ROWS[name]:
        raise RuntimeError(f"{name}: expected {EXPECTED_ROWS[name]} rows, found {len(data)}")
    if set(data["subject"].unique()) != set(range(1, N_SUBJECTS + 1)):
        raise RuntimeError(f"{name}: subject coverage is incomplete")
    key = [c for c in ["dataset", "subject", "model", "protocol",
                       "target_session", "train_session", "test_session"]
           if c in data]
    if data.duplicated(key).any():
        raise RuntimeError(f"{name}: duplicate result keys")
    if "auc" in data and not data["auc"].between(0, 1).all():
        raise RuntimeError(f"{name}: AUC outside [0, 1]")

    if name.startswith("auc_"):
        data["seed"] = RANDOM_STATE
    data["preprocess_version"] = PREPROCESS_VERSION
    data["result_provenance"] = "reused_exact_native_window_v1"
    destination = RAW_DIR / name
    data.to_csv(destination, index=False)
    return {"source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "destination": str(destination.relative_to(ROOT)),
            "rows": len(data), "models": sorted(data["model"].unique().tolist())}


def main() -> None:
    checked = verify_legacy_caches()
    copied = {name: verify_and_copy(name) for name in EXPECTED_ROWS}
    manifest = {
        "dataset": DATASET,
        "reason": ("Legacy Stieger epochs already equal the audited 0-3 s, 128 Hz, "
                   "384-sample analysis window; model definitions, protocol sets, "
                   "and the canonical random seed are unchanged."),
        "preprocess_version": PREPROCESS_VERSION,
        "random_state": RANDOM_STATE,
        "verified_legacy_caches": len(checked),
        "current_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [ROOT / "src" / "data.py", ROOT / "src" / "models.py",
                         ROOT / "src" / "evaluate.py", ROOT / "src" / "splitters.py"]
        },
        "copied": copied,
    }
    path = ROOT / "results" / "stieger_exact_reuse_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

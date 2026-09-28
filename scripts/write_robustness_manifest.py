#!/usr/bin/env python3
"""Record compute-device and file provenance for the three EEGNet refits."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from src.config import ROBUSTNESS_EXPECTED_ROWS, ROBUSTNESS_SUBJECTS  # noqa: E402

ROBUST = ROOT / "results" / "robustness"
SEEDS = (20260716, 20260717, 20260718)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    entries = []
    for seed in SEEDS:
        path = ROBUST / f"eegnet_Stieger2021_seed{seed}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        frame = pd.read_csv(path)
        if set(frame["subject"].astype(int)) != set(ROBUSTNESS_SUBJECTS):
            raise SystemExit(f"{path.name}: incomplete or incorrect subject subset")
        if len(frame) != ROBUSTNESS_EXPECTED_ROWS:
            raise SystemExit(
                f"{path.name}: {len(frame)} rows, expected {ROBUSTNESS_EXPECTED_ROWS}")
        entries.append({
            "seed": seed,
            "file": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "rows": int(len(frame)),
            "subjects": int(frame["subject"].nunique()),
            "subject_ids": sorted(frame["subject"].astype(int).unique().tolist()),
            "compute_device": "mps",
            "protocols": sorted(frame["protocol"].unique().tolist()),
        })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Stieger2021",
        "scope": "targeted EEGNet ranking-stability analysis",
        "subset_definition": {
            "n": len(ROBUSTNESS_SUBJECTS),
            "subject_ids": list(ROBUSTNESS_SUBJECTS),
            "selection": (
                "Fixed random sample using seed 20260716, balanced as six "
                "T=7 and six T=11 participants."
            ),
        },
        "device_note": (
            "All three robustness files were freshly fit on one MPS device with identical "
            "architecture, optimizer, early stopping, and protocol definitions. "
            "The primary canonical EEGNet rows in results/raw remain the separately "
            "audited canonical analysis."
        ),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        },
        "runs": entries,
    }
    ROBUST.mkdir(parents=True, exist_ok=True)
    out = ROBUST / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

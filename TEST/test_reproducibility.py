"""Independent result-level audit tests; no raw EEG or decoder dependencies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submitted_audit", ROOT / "scripts" / "audit_results.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SubmittedClassicalAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="loso-submitted-test-")
        self.root = Path(self.tmp.name)
        files = [
            "results/gap_wide.csv", "results/auc_long.csv",
            "results/sensitivity/ma2020_day_level.csv",
            "results/revision_summary.json", "paper/main.tex",
        ]
        for name in files:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        shutil.copytree(ROOT / "results" / "figures", self.root / "results" / "figures")

    def tearDown(self):
        self.tmp.cleanup()

    def test_saved_results_pass_without_raw_or_neural_files(self):
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse((self.root / "cache").exists())
        self.assertFalse((self.root / "results" / "raw").exists())
        self.assertFalse((self.root / "results" / "robustness").exists())
        self.assertIn("epoch length or channel count", result["not_checked"])

    def test_corrupted_paired_difference_fails(self):
        path = self.root / "results" / "gap_wide.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "delta"] += 0.1
        frame.to_csv(path, index=False)
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertFalse(result["ok"])
        self.assertTrue(any("delta disagrees" in item for item in result["errors"]))

    def test_duplicate_long_result_key_fails(self):
        path = self.root / "results" / "auc_long.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertFalse(result["ok"])
        self.assertTrue(any("duplicate keys" in item for item in result["errors"]))

    def test_missing_input_fails(self):
        (self.root / "results" / "auc_long.csv").unlink()
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertFalse(result["ok"])
        self.assertIn("missing results/auc_long.csv", result["errors"])

    def test_stale_summary_fails(self):
        path = self.root / "results" / "revision_summary.json"
        summary = json.loads(path.read_text())
        summary["estimands"]["conditional"]["participant_weighted"]["mean"] += 0.1
        path.write_text(json.dumps(summary))
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertFalse(result["ok"])
        self.assertTrue(any("conditional participant estimate disagrees" in item
                            for item in result["errors"]))


if __name__ == "__main__":
    unittest.main()

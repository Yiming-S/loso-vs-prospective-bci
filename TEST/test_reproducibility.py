"""Independent result-level audit tests; no raw EEG or decoder dependencies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

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

    def replace_svg_dimensions(self, stem, **dimensions):
        path = self.root / "results" / "figures" / f"{stem}.svg"
        document = ET.parse(path)
        document.getroot().attrib.update(dimensions)
        document.write(path)

    def test_wrong_scientific_figure_width_fails(self):
        for stem in ("fig2_estimands_distribution", "fig5_ma_time_unit"):
            with self.subTest(figure=stem):
                self.replace_svg_dimensions(stem, width="123pt")
                result = AUDIT.submitted_classical_audit(self.root)
                self.assertFalse(result["ok"])
                self.assertTrue(any(f"{stem}.svg: width" in item
                                    for item in result["errors"]))
                self.replace_svg_dimensions(
                    stem, width=f"{AUDIT.SCIENTIFIC_FIGURE_WIDTHS_PT[stem]}pt")

    def test_missing_vector_companion_fails(self):
        for suffix in (".svg", ".pdf"):
            with self.subTest(suffix=suffix):
                name = f"fig6_decoder_margins{suffix}"
                path = self.root / "results" / "figures" / name
                path.unlink()
                result = AUDIT.submitted_classical_audit(self.root)
                self.assertFalse(result["ok"])
                self.assertIn(f"missing results/figures/{name}", result["errors"])
                shutil.copy2(ROOT / "results" / "figures" / name, path)

    def test_different_positive_figure_heights_pass(self):
        self.replace_svg_dimensions("fig2_estimands_distribution", height="180pt")
        self.replace_svg_dimensions("fig5_ma_time_unit", height="210pt")
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertTrue(result["ok"], result["errors"])

    def test_nonpositive_or_nonfinite_figure_height_fails(self):
        for height in ("0pt", "-1pt", "nanpt", "infpt"):
            with self.subTest(height=height):
                self.replace_svg_dimensions("fig6_decoder_margins", height=height)
                result = AUDIT.submitted_classical_audit(self.root)
                self.assertFalse(result["ok"])
                self.assertTrue(any("height must be positive and finite" in item
                                    for item in result["errors"]))

    def test_raster_portraits_do_not_require_vector_companions(self):
        manuscript = self.root / "paper" / "main.tex"
        manuscript.write_text(manuscript.read_text() + "\n" +
                              r"\includegraphics{test_portrait.jpg}" + "\n" +
                              r"\includegraphics{test_portrait.png}" + "\n")
        result = AUDIT.submitted_classical_audit(self.root)
        self.assertTrue(result["ok"], result["errors"])

    def test_raw_cache_figure_inventory_includes_decoder_margins(self):
        self.assertIn("fig6_decoder_margins", AUDIT.SCIENTIFIC_FIGURE_WIDTHS_PT)


if __name__ == "__main__":
    unittest.main()

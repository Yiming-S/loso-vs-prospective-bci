"""Regression tests for the corrected common-window contract."""
from __future__ import annotations

import unittest

import numpy as np

from src.config import PREPROCESS_VERSION
from src.data import _finalize_sessions


def session(n_channels=3, n_times=500, label="0"):
    return {
        "X": np.zeros((8, n_channels, n_times), dtype=np.float64),
        "y": np.tile([0, 1], 4),
        "session": label,
        "classes": ["left", "right"],
    }


class FinalizeSessionsTests(unittest.TestCase):
    def test_trims_to_exact_three_seconds_and_records_contract(self):
        sessions = _finalize_sessions([session()], 128.0)
        self.assertEqual(sessions[0]["X"].shape, (8, 3, 384))
        self.assertEqual(sessions[0]["X"].dtype, np.float32)
        self.assertEqual(sessions[0]["analysis_window_s"], (0.0, 3.0))
        self.assertEqual(sessions[0]["preprocess_version"], PREPROCESS_VERSION)

    def test_rejects_short_epoch(self):
        with self.assertRaisesRegex(RuntimeError, "expected at least 384"):
            _finalize_sessions([session(n_times=383)], 128.0)

    def test_rejects_channel_count_change_within_subject(self):
        with self.assertRaisesRegex(RuntimeError, "Channel count changes"):
            _finalize_sessions([session(3, label="0"), session(4, label="1")], 128.0)


if __name__ == "__main__":
    unittest.main()

"""Unit and MOABB-integration tests for the causal session splitter."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.splitters import ForwardSessionSplitter


class ForwardSessionSplitterTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1, 0, 1])
        self.sessions = np.array(["0", "0", "1", "1", "2", "2"])
        self.X = np.arange(len(self.y))

    def test_expanding_groups_interface(self):
        cv = ForwardSessionSplitter(mode="expanding")
        got = [(tr.tolist(), te.tolist()) for tr, te in cv.split(
            self.X, self.y, groups=self.sessions)]
        self.assertEqual(got, [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])])

    def test_all_modes_with_metadata_frame(self):
        metadata = pd.DataFrame({"session": self.sessions})
        expected = {
            "expanding": [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])],
            "last": [([0, 1], [2, 3]), ([2, 3], [4, 5])],
            "first": [([0, 1], [2, 3]), ([0, 1], [4, 5])],
        }
        for mode, want in expected.items():
            with self.subTest(mode=mode):
                got = [(tr.tolist(), te.tolist()) for tr, te in
                       ForwardSessionSplitter(mode=mode).split(
                           self.X, self.y, groups=metadata)]
                self.assertEqual(got, want)

    def test_moabb_cross_session_wrapper(self):
        from moabb.evaluations import CrossSessionSplitter

        metadata = pd.DataFrame({
            "subject": [1] * len(self.y),
            "session": self.sessions,
        })
        cv = CrossSessionSplitter(
            cv_class=ForwardSessionSplitter, mode="expanding")
        got = [(tr.tolist(), te.tolist()) for tr, te in cv.split(self.y, metadata)]
        self.assertEqual(got, [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])])


if __name__ == "__main__":
    unittest.main()

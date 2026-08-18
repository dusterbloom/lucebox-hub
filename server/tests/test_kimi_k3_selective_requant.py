#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_kimi_k3_selective_requant import sampled_equal, window_offsets  # noqa: E402


class FakeTensor:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class SelectiveRequantVerifierTest(unittest.TestCase):
    def test_windows_cover_edges_and_are_deterministic(self) -> None:
        first = window_offsets(10000, seed=17, window=4096)
        second = window_offsets(10000, seed=17, window=4096)
        self.assertEqual(first, second)
        self.assertIn(0, first)
        self.assertIn(10000 - 4096, first)

    def test_sampled_equal_detects_changed_untouched_bytes(self) -> None:
        left = np.arange(10000, dtype=np.uint8)
        right = left.copy()
        equal, checked = sampled_equal(FakeTensor(left), FakeTensor(right), 3, 4096)
        self.assertTrue(equal)
        self.assertGreaterEqual(checked, 4096)
        right[0] ^= 1
        equal, _ = sampled_equal(FakeTensor(left), FakeTensor(right), 3, 4096)
        self.assertFalse(equal)


if __name__ == "__main__":
    unittest.main()

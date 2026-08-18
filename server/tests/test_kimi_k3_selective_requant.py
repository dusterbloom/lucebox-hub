#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_kimi_k3_selective_requant import (  # noqa: E402
    parse_shards,
    sampled_equal,
    shard_number,
    window_offsets,
)
from plan_kimi_k3_kda_requant import parse_layers  # noqa: E402


class FakeTensor:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class SelectiveRequantVerifierTest(unittest.TestCase):
    def test_layer_subset_parser_defaults_to_all(self) -> None:
        available = {0, 1, 2, 4}
        self.assertEqual(parse_layers(None, available), available)
        self.assertEqual(parse_layers("4,1,4", available), {1, 4})

    def test_layer_subset_parser_rejects_invalid_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a recurrent KDA layer"):
            parse_layers("3", {0, 1, 2, 4})
        with self.assertRaisesRegex(ValueError, "empty entry"):
            parse_layers("1,", {0, 1, 2, 4})

    def test_shard_subset_parser(self) -> None:
        self.assertIsNone(parse_shards(None))
        self.assertEqual(parse_shards("13,12,13"), {12, 13})
        self.assertEqual(
            shard_number("Kimi-K3-KDA-Q4_K-00012-of-00014.gguf"), 12
        )
        with self.assertRaisesRegex(ValueError, "positive shard IDs"):
            parse_shards("0")

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

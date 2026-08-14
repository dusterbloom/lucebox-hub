#!/usr/bin/env python3
"""Synthetic regression for the identifiable Kimi H16 Fisher fitter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fit_kimi_h16_fisher import Samples, fit_rank_ladder  # noqa: E402


def make_samples(
    generator: np.random.Generator,
    count: int,
    split: str,
) -> Samples:
    dimension = 16
    scales = np.linspace(2.0, 0.4, dimension)
    delta = generator.normal(size=(count, dimension)) * scales
    behavioral_weights = np.asarray([1.5, 0.9, 0.5, 0.25])
    kl = 0.5 * np.square(delta[:, :4]) @ behavioral_weights
    return Samples(
        delta=delta,
        kl=kl,
        family=np.full(count, "synthetic", dtype=object),
        split=np.full(count, split, dtype=object),
        sequence=np.asarray([
            f"{split}-{index // 8}" for index in range(count)
        ], dtype=object),
        sequence_row=np.arange(count, dtype=np.int64) % 8,
    )


class KimiH16FisherTest(unittest.TestCase):
    def test_low_rank_metric_recovers_behavioral_order(self) -> None:
        generator = np.random.default_rng(20260814)
        train = make_samples(generator, 512, "calibration")
        validation = make_samples(generator, 128, "validation")
        test = make_samples(generator, 128, "test")
        ladder, artifacts = fit_rank_ladder(
            train, validation, test, "synthetic"
        )
        rank_four = next(row for row in ladder if row["rank"] == 4)
        rank_sixteen = next(row for row in ladder if row["rank"] == 16)
        self.assertEqual(rank_four["status"], "MEASURED")
        self.assertGreater(rank_four["test"]["spearman"], 0.8)
        self.assertGreater(rank_sixteen["test"]["spearman"], 0.95)
        self.assertEqual(artifacts["synthetic_B_rank_4"].shape, (4, 16))
        self.assertEqual(artifacts["synthetic_B_rank_16"].shape, (16, 16))


if __name__ == "__main__":
    unittest.main()

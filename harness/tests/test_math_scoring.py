from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "harness"))

import client_test_runner  # noqa: E402
from math_scoring import _math_equiv  # noqa: E402


class MathScoringTests(unittest.TestCase):
    def test_math01_chained_inequality_matches_half_open_interval(self) -> None:
        correct, _ = client_test_runner._score_math_response(
            r"\boxed{2\le x<5}", "[2,5)"
        )
        self.assertTrue(correct)
        self.assertTrue(_math_equiv(r"x\in[2,5)", r"2\leq x<5"))

    def test_math04_and_math08_rational_forms_match(self) -> None:
        self.assertTrue(_math_equiv(r"-2/3", r"-\frac{2}{3}"))
        self.assertTrue(_math_equiv(r"\frac{-2}{3}", r"-\frac{2}{3}"))
        self.assertTrue(_math_equiv(r"x=\frac{20}{3}", "20/3"))
        self.assertTrue(_math_equiv(r"6\frac{2}{3}", "20/3"))

    def test_strictly_rejects_nearby_but_wrong_answers(self) -> None:
        self.assertFalse(_math_equiv(r"2<x<5", "[2,5)"))
        self.assertFalse(_math_equiv(r"[2,5]", "[2,5)"))
        self.assertFalse(_math_equiv(r"x=2<x<5", "[2,5)"))
        self.assertFalse(_math_equiv(r"x=9", "8"))
        self.assertFalse(_math_equiv(r"\frac{19}{3}", "20/3"))


if __name__ == "__main__":
    unittest.main()

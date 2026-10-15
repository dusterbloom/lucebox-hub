"""Pure tests for r2b_report on synthetic cells (no files): python3 -I -m pytest -q lucebox/r2b/test_r2b_report.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2b_report as rep  # noqa: E402


def cells(n_art=10, per=5, loss8=0):
    keys = [(f"a{a}", q) for a in range(n_art) for q in range(per)]
    base = {k: 1 for k in keys}
    drop = {k: (0 if k[1] < loss8 else 1) for k in keys}
    return {("SHIP", 0): base, ("SHIP", 1): base, ("SHIP", 8): drop, ("FULL", 0): base, ("FULL", 1): base, ("FULL", 8): base}


def test_primary_contrast_sign_and_count():
    r = rep.contrast(cells(loss8=1), ("SHIP", 8), ("SHIP", 0))
    assert r["n_q"] == 50 and r["n_art"] == 10 and abs(r["mean"] + 20.0) < 1e-9 and r["only_b"] == 10
    assert rep.decide(r["lo"], r["hi"]) == "FAIL"
    z = rep.contrast(cells(), ("SHIP", 8), ("SHIP", 0))
    assert z["mean"] == 0 and rep.decide(z["lo"], z["hi"]) == "PASS"


def test_decide_and_secondary():
    assert rep.decide(-3.0, 1) == "PASS" and rep.decide(-3.1, 0) == "INCONCLUSIVE" and rep.decide(-5, -3.1) == "FAIL"
    assert rep.secondary_reading(6.0).startswith("quote") and rep.secondary_reading(6.1).startswith("compaction")


def test_missing_cells_give_none():
    assert rep.contrast({}, ("SHIP", 8), ("SHIP", 0)) is None and rep.fmt(None).startswith("not run")

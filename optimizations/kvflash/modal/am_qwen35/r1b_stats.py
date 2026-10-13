"""R1b statistics (PREREG 'Statistics (all rungs)' and 'R1(b): statistics and pass criteria'). Pure numpy, no GPU.

Unit: question. Cluster: article. Paired cluster bootstrap, 10,000 resamples, percentile 95%.
"""
import json
import math
from collections import defaultdict

import numpy as np

MARGIN = 3.0  # points


def boot_mean(values, clusters, n_boot=10000, seed=0):
    """Mean of per-question `values` and its 95% percentile CI, resampling whole clusters (articles)."""
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    m = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi),
            "n_q": int(len(values)), "n_art": int(len(uniq))}


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p for paired binary vectors; returns (only_a, only_b, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n = x + y
    if n == 0:
        return x, y, 1.0
    p = 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, p)


def load_cell(paths):
    """jsonl file(s) of per-question rows -> {(article_id, qi): correct(0/1)}."""
    out = {}
    for path in [paths] if isinstance(paths, str) else paths:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r["article_id"], r["qi"])] = int(r["correct"])
    return out


def align(*cells):
    """Per-question arrays over the questions present in every cell, plus article ids (sorted, deterministic)."""
    keys = sorted(set.intersection(*[set(c) for c in cells]))
    return [np.array([c[k] for k in keys], float) for c in cells], [k[0] for k in keys]


def degradation(orig, x, clusters, **kw):
    """D(X) = acc(original) - acc(X) in points, with CI."""
    r = boot_mean((np.asarray(orig) - np.asarray(x)) * 100, clusters, **kw)
    return r


def delta_vs_reference(orig_a, x_a, orig_b, x_b, clusters, **kw):
    """Delta(X) = D_A(X) - D_B(X) per question (model A = 27B, model B = 4B), points, with CI and verdict."""
    v = ((np.asarray(orig_a) - np.asarray(x_a)) - (np.asarray(orig_b) - np.asarray(x_b))) * 100
    r = boot_mean(v, clusters, **kw)
    r["verdict"] = verdict(r)
    return r


def verdict(r, margin=MARGIN):
    """PASS: upper bound <= +margin. FAIL: lower bound > 0 and point > margin. Else inconclusive."""
    if r["hi"] <= margin:
        return "PASS"
    if r["lo"] > 0 and r["mean"] > margin:
        return "FAIL"
    return "inconclusive"


def online_vs_ceiling(online, ceiling, clusters, **kw):
    """R0 rule: lower bound of (online - ceiling) >= -margin -> within margin; CI wider than the margin -> inconclusive."""
    r = boot_mean((np.asarray(online) - np.asarray(ceiling)) * 100, clusters, **kw)
    r["verdict"] = "within 3 points" if r["lo"] >= -MARGIN else ("worse" if r["hi"] < -MARGIN else "inconclusive")
    return r


def am_vs_control(am, ctl, clusters, **kw):
    """R0 rule: lower bound > 0 -> AM better; inside +-margin -> equivalent; else inconclusive."""
    r = boot_mean((np.asarray(am) - np.asarray(ctl)) * 100, clusters, **kw)
    r["verdict"] = ("AM better" if r["lo"] > 0 else
                    "equivalent" if (r["lo"] >= -MARGIN and r["hi"] <= MARGIN) else "inconclusive")
    return r


def per_article_acc(values, clusters):
    d = defaultdict(list)
    for v, c in zip(values, clusters):
        d[c].append(v)
    return {c: float(np.mean(v)) for c, v in d.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(20), 18)
    o = (rng.random(360) < 0.8).astype(float)
    same = boot_mean(o - o, cl)
    assert same["mean"] == 0 and same["lo"] == 0 and same["hi"] == 0
    x = o * (rng.random(360) < 0.9)  # degrades ~ 8 points
    d = degradation(o, x, cl)
    assert 0 < d["lo"] < d["mean"] < d["hi"] and d["n_art"] == 20 and d["n_q"] == 360
    assert delta_vs_reference(o, x, o, x, cl)["verdict"] == "PASS"  # identical degradation -> Delta 0
    assert verdict({"mean": 5, "lo": 1, "hi": 9}) == "FAIL" and verdict({"mean": 5, "lo": -1, "hi": 9}) == "inconclusive"
    assert mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])[:2] == (3, 0) and mcnemar_exact([1, 0], [1, 0])[2] == 1.0
    print("r1b_stats ok")

"""R2b report (PREREG "Rung R2b", driver changes item 3). Pure numpy.

  python3 -I lucebox/r2b/r2b_report.py <R2b rows dir> <R1e rows dir>
Each dir is either flat or holds held/ and sel/ subdirectories (evidence/am/R2b/{held,sel}, evidence/am/R1e/{held,sel}).
R2b rows: {FULL-CHK,F-BLK-UNI}__k{1,8}__t0.1__<split>.jsonl (+ F-BLK-UNI__k0 for the repeatability check).
k = 0 comes from R1(e): ORIG (FULL) and F-UNI-LOCAL (shipping arm). All statistics through r1b_stats (cluster = article,
10,000 resamples, seed 0, 95% percentile; McNemar exact secondary).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../modal/am_qwen35"))
import numpy as np  # noqa: E402

from r1b_stats import MARGIN, align, boot_mean, mcnemar_exact  # noqa: E402

SHIP, FULL = "F-BLK-UNI", "FULL-CHK"
SPLITS = ("held", "sel")
CONTRASTS = [
    ("ship(8) - ship(0)  [PRIMARY]", ("SHIP", 8), ("SHIP", 0)),
    ("ship(1) - ship(0)", ("SHIP", 1), ("SHIP", 0)),
    ("ship(8) - ship(1)", ("SHIP", 8), ("SHIP", 1)),
    ("FULL(8) - ship(8)  [secondary G]", ("FULL", 8), ("SHIP", 8)),
    ("FULL(1) - ship(1)", ("FULL", 1), ("SHIP", 1)),
    ("FULL(1) - ORIG(0)  (distractors alone, chunked-prefill noise)", ("FULL", 1), ("FULL", 0)),
    ("FULL(8) - ORIG(0)  (distractors alone, chunked-prefill noise)", ("FULL", 8), ("FULL", 0)),
]


def decide(lo, hi, margin=MARGIN):
    """PASS lo >= -margin; FAIL hi < -margin; else INCONCLUSIVE (stage 1: go to stage 2; pooled: ship on point estimate)."""
    return "PASS" if lo >= -margin else ("FAIL" if hi < -margin else "INCONCLUSIVE")


def secondary_reading(g_mean):
    return ("compaction costs more with a long context than R1(e): add a 'compaction off' arm to the agentic gate"
            if g_mean > 6 else "quote about 4 to 5 points against full KV")


def _file(d, split, name):
    for p in (os.path.join(d, split, name), os.path.join(d, name)):
        if os.path.exists(p):
            return p
    return None


def read_rows(d, split, arm, k):
    name = f"{arm}__t0.1__{split}.jsonl" if k is None else f"{arm}__k{k}__t0.1__{split}.jsonl"
    p = _file(d, split, name)
    return [json.loads(l) for l in open(p) if l.strip()] if p else []


def vec(rows):
    return {(r["article_id"], r["qi"]): int(r["correct"]) for r in rows}


def load(r2b, r1e, splits):
    """({(arm, k): {(article, qi): correct}}, {(arm, k): rows}) merged over `splits`. k = 0 rows are R1(e)'s, restricted to
    the articles R2b has rows for."""
    cells, raw = {}, {}
    for sp in splits:
        parts = {("FULL", k): read_rows(r2b, sp, FULL, k) for k in (1, 8)}
        parts.update({("SHIP", k): read_rows(r2b, sp, SHIP, k) for k in (1, 8)})
        arts = {r["article_id"] for rows in parts.values() for r in rows}
        parts[("FULL", 0)] = [r for r in read_rows(r1e, sp, "ORIG", None) if r["article_id"] in arts]
        parts[("SHIP", 0)] = [r for r in read_rows(r1e, sp, "F-UNI-LOCAL", None) if r["article_id"] in arts]
        for key, rows in parts.items():
            cells.setdefault(key, {}).update(vec(rows))
            raw.setdefault(key, []).extend(rows)
    return cells, raw


def contrast(cells, a, b):
    if not cells.get(a) or not cells.get(b):
        return None
    (x, y), arts = align(cells[a], cells[b])
    if not len(x):
        return None
    r = boot_mean((x - y) * 100, arts)
    r["only_a"], r["only_b"], r["p"] = mcnemar_exact(x, y)
    return r


def fmt(r):
    return "not run / no rows" if r is None else (
        f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] points, n_q {r['n_q']} n_art {r['n_art']}, "
        f"McNemar only-first {r['only_a']} only-second {r['only_b']} p {r['p']:.3f}")


def counts_table(cells, raw):
    out = ["| arm | k | correct | N questions | articles | ctx tokens (mean) | slots effective (mean) | prefill s | "
           "compact s | answer s | peak GB (max) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("FULL", "SHIP"):
        for k in (0, 1, 8):
            c, rows = cells.get((arm, k)), raw.get((arm, k), [])
            if not c:
                out.append(f"| {arm} | {k} | - | - | - | - | - | - | - | - | - |")
                continue
            per = list({r["article_id"]: r for r in rows}.values())
            m = lambda key: (f"{np.mean([a[key] for a in per if key in a]):.1f}" if any(key in a for a in per) else "-")
            out.append(f"| {arm} | {k} | {sum(c.values())} | {len(c)} | {len({q[0] for q in c})} | {m('seq_len')} | "
                       f"{m('slots_effective_mean')} | {m('prefill_s')} | {m('compact_s')} | {m('answer_s')} | "
                       f"{max(a.get('peak_gb', 0) for a in per)} |")
    return "\n".join(out)


def repeatability(r2b, r1e, splits=("held",)):
    """Shipping arm re-run at k = 0 against R1(e)'s F-UNI-LOCAL rows: how many parsed answers differ."""
    out = []
    for sp in splits:
        new = {(r["article_id"], r["qi"]): r["parsed"] for r in read_rows(r2b, sp, SHIP, 0)}
        old = {(r["article_id"], r["qi"]): r["parsed"] for r in read_rows(r1e, sp, "F-UNI-LOCAL", None)}
        keys = sorted(set(new) & set(old))
        if keys:
            diff = sum(new[k] != old[k] for k in keys)
            out.append(f"{sp}: {diff} of {len(keys)} answers differ from R1(e) F-UNI-LOCAL on {len({k[0] for k in keys})} "
                       f"articles (expected 0; any difference is the greedy-flip floor)")
    return "\n".join(out) or "repeatability rows not present"


def report(r2b, r1e):
    out = ["# R2b report", ""]
    held, raw_h = load(r2b, r1e, ("held",))
    sel, raw_s = load(r2b, r1e, ("sel",))
    both, raw_b = load(r2b, r1e, SPLITS)
    for name, c, raw in (("Held-out (stage 1, decides)", held, raw_h), ("Selection (stage 2)", sel, raw_s),
                         ("Pooled (stage 1 + stage 2)", both, raw_b)):
        out += [f"## {name}", "", counts_table(c, raw), ""]
        out += [f"- {label}: {fmt(contrast(c, a, b))}" for label, a, b in CONTRASTS]
        out.append("")
    d = contrast(held, ("SHIP", 8), ("SHIP", 0))
    out += ["## Decision rule", ""]
    if d is None:
        out.append("Stage 1 has no ship(8) / ship(0) pairs yet.")
    else:
        v1 = decide(d["lo"], d["hi"])
        out.append(f"Stage 1 (held-out, {d['n_art']} articles, {d['n_q']} questions): D = {d['mean']:+.2f} "
                   f"[{d['lo']:+.2f}, {d['hi']:+.2f}] -> {v1}")
        if v1 == "INCONCLUSIVE":
            p = contrast(both, ("SHIP", 8), ("SHIP", 0))
            if p and p["n_art"] > d["n_art"]:
                out.append(f"Pooled ({p['n_art']} articles, {p['n_q']} questions): {p['mean']:+.2f} "
                           f"[{p['lo']:+.2f}, {p['hi']:+.2f}] -> {decide(p['lo'], p['hi'])}")
            else:
                out.append("Stage 2 not complete: pooled verdict pending.")
    g = contrast(held, ("FULL", 8), ("SHIP", 8))
    if g:
        out.append(f"Secondary G = FULL(8) - ship(8), held-out: {g['mean']:+.2f} -> {secondary_reading(g['mean'])}")
    out += ["", "## Repeatability (shipping arm, k = 0, vs R1(e))", "", repeatability(r2b, r1e)]
    return "\n".join(out)


if __name__ == "__main__":
    assert decide(-3.0, 1) == "PASS" and decide(-3.1, 0) == "INCONCLUSIVE" and decide(-5, -3.1) == "FAIL"
    assert secondary_reading(6.0).startswith("quote") and secondary_reading(6.1).startswith("compaction")
    print(report(sys.argv[1], sys.argv[2]))

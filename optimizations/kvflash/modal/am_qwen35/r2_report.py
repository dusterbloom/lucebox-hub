"""R2 report: counts, timing/memory and (with --stats) the pre-registered paired statistics. Pure numpy.

  python3 -I modal/am_qwen35/r2_report.py <r2 rows dir> [--k0 evidence/am/R1b/27b] [--stats]
Rows: <arm>__k<k>__t0.1__sel.jsonl (R2) and F__t0.1__sel.jsonl / orig__t0.1__sel.jsonl (k = 0, R1b).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from r1b_stats import boot_mean, mcnemar_exact  # noqa: E402

ARMS = ("FULL", "F-SEG", "F-BLK", "F-RE")


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def load(d, k0):
    """{(arm, k): rows}; k = 0 rows are the R1b ones (FULL <- orig, every compacting arm <- F)."""
    cells = {}
    for arm in ARMS:
        for k in (1, 2, 4, 8):
            rows = read(f"{d}/{arm}__k{k}__t0.1__sel.jsonl")
            if rows:
                cells[(arm, k)] = rows
    if k0:
        orig, f = read(f"{k0}/orig__t0.1__sel.jsonl"), read(f"{k0}/F__t0.1__sel.jsonl")
        arts = {r["article_id"] for rows in cells.values() for r in rows}
        for arm in ARMS:
            cells[(arm, 0)] = [r for r in (orig if arm == "FULL" else f) if r["article_id"] in arts]
    return cells


def vec(rows):
    return {(r["article_id"], r["qi"]): int(r["correct"]) for r in rows}


def paired(a, b):
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys], float), np.array([b[k] for k in keys], float), [k[0] for k in keys]


def table(cells):
    out = ["| arm | k | correct | N | ctx tokens | slots tensor / effective | prefill s | compact s | answer s | peak GB | GPU |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, k) in sorted(cells, key=lambda x: (ARMS.index(x[0]), x[1])):
        rows = cells[(arm, k)]
        r0 = rows[0]
        per_art = {}
        for r in rows:
            per_art[r["article_id"]] = r
        arts = list(per_art.values())
        mean = lambda key: (round(float(np.mean([a[key] for a in arts])), 1) if key in arts[0] else "-")
        out.append(f"| {arm} | {k} | {sum(r['correct'] for r in rows)} | {len(rows)} | {mean('seq_len')} | "
                   f"{mean('slots_mean')} / {mean('slots_effective_mean')} | {mean('prefill_s')} | {mean('compact_s')} | "
                   f"{mean('answer_s')} | {max([a.get('peak_gb', 0) for a in arts])} | {r0.get('gpu', '-')} |")
    return "\n".join(out)


def stats(cells):
    out = []
    ci = lambda r: f"{r['mean']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]"
    for arm in ARMS[1:]:
        for k in (1, 2, 4, 8):
            if (arm, k) not in cells or ("FULL", k) not in cells:
                continue
            f, x, cl = paired(vec(cells[("FULL", k)]), vec(cells[(arm, k)]))
            d = boot_mean((f - x) * 100, cl)
            line = f"| {arm} | {k} | D = FULL - arm | {ci(d)} | n_q {d['n_q']} n_art {d['n_art']} |"
            if (arm, 0) in cells:
                a0, ak, cl = paired(vec(cells[(arm, 0)]), vec(cells[(arm, k)]))
                r = boot_mean((ak - a0) * 100, cl)
                ox, oy, p = mcnemar_exact(ak, a0)
                line += f" arm(k) - arm(0) {ci(r)}, McNemar only-k {ox} only-0 {oy} p {p:.3f} |"
                if arm == "F-SEG" and k == 8:
                    line += f" Q1 lower bound >= -3: {r['lo'] >= -3.0} |"
            out.append(line)
    return "\n".join(["| arm | k | measure | points [95% CI] | detail |", "|---|---|---|---|---|"] + out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--k0", default="")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    c = load(a.dir, a.k0)
    print(table(c))
    if a.stats:
        print()
        print(stats(c))

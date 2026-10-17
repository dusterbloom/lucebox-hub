"""Selector v2 derisk (PREREG_selv2.md): fact coverage at v1's budget (60 chunks) for every (queries, query agg, plane agg,
unit) variant on the multi-query exact dump. Variants are chosen on split A (trajectories), reported on split B.
    python3 -I offline_mq.py <worktree> <mq dump dir>
<mq dump dir> holds QK16KND.jsonl and exact_dump/{exact_dump,exact_dump_mq}.jsonl from run_mq_dump.sh."""
import hashlib
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_gran as g  # noqa: E402

DIR = sys.argv[2]
reqs = [json.loads(l) for l in open(os.path.join(DIR, "QK16KND.jsonl"))]
mq = {r["id"]: json.loads(l) for r, l in zip(reqs, open(os.path.join(DIR, "exact_dump", "exact_dump_mq.jsonl")))}
assert len(mq) == len(reqs), (len(mq), len(reqs))

QUERIES, QAGG, PAGG, UNITS = (1, 4, 16, 64), ("mean", "max"), ("mean", "top", "max"), (10, 2, 1)
V1 = (1, "mean", "mean", 10)   # the shipped selector: last token, mean over planes, 640-token blocks


def score(row, n, qagg, pagg):
    rows = row[pagg][-n:]
    f = (lambda v: sum(v) / len(v)) if qagg == "mean" else max
    return [f(col) for col in zip(*rows)]


def split(i):
    """A or B by a hash of the trajectory id (the probe id before its first ':')."""
    return "AB"[int(hashlib.sha1(i.split(":")[0].encode()).hexdigest(), 16) % 2]


def sign_p(wins, losses):
    """Two-sided exact sign test on discordant probes."""
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0


variants = sorted({(n, "mean" if n == 1 else qa, pa, u) for n, qa, pa, u in itertools.product(QUERIES, QAGG, PAGG, UNITS)})
hit = {}   # (variant, id) -> fact fully held
rows = []
for i, band, fwc, fact, msg in g.out_of_window():
    if i not in mq:
        continue
    rows.append((i, band))
    for v in variants:
        hit[(v, i)] = fact <= g.units(score(mq[i], *v[:3]), fwc, v[3])


def count(v, band, half=None):
    return sum(hit[(v, i)] for i, b in rows if b == band and (half is None or split(i) == half))


n_of = {(b, h): sum(1 for i, bb in rows if bb == b and (h is None or split(i) == h)) for b in ("beyond", "within") for h in ("A", "B", None)}
print("fact fully held at a 60-chunk budget; n per cell:", {f"{b}/{h or 'all'}": n for (b, h), n in n_of.items()})
print(f"{'queries':>7} {'q agg':>5} {'planes':>6} {'unit':>4} | {'beyond A':>8} {'B':>4} {'all':>4} | {'within A':>8} {'B':>4} {'all':>4}")
for v in variants:
    print(f"{v[0]:>7} {v[1]:>5} {v[2]:>6} {v[3]:>4} | {count(v, 'beyond', 'A'):>8} {count(v, 'beyond', 'B'):>4} {count(v, 'beyond'):>4} | "
          f"{count(v, 'within', 'A'):>8} {count(v, 'within', 'B'):>4} {count(v, 'within'):>4}" + ("   <- v1" if v == V1 else ""))
# Choice on A: most beyond hits; ties go to fewer queries, then mean over max, then the larger unit (the simpler selector).
rank = {"mean": 0, "top": 1, "max": 2}
best = max(variants, key=lambda v: (count(v, "beyond", "A"), -v[0], -rank[v[1]], -rank[v[2]], v[3]))
print("\nchosen on A:", best)
for band in ("beyond", "within"):
    ids = [i for i, b in rows if b == band and split(i) == "B"]
    w = sum(hit[(best, i)] and not hit[(V1, i)] for i in ids)
    l_ = sum(hit[(V1, i)] and not hit[(best, i)] for i in ids)
    print(f"held-out B {band}: n {len(ids)}  v1 {count(V1, band, 'B')}  chosen {count(best, band, 'B')}  net {w - l_:+d} "
          f"(gains {w}, losses {l_}, sign test p {sign_p(w, l_):.3g})")

"""Exploratory: rank of the probe's best fact chunk among the eligible chunks [10, fwc) under mass_end (last prompt token),
mass_r (token R-1) and their sum; then fact coverage when both queries vote.
    python3 -I rank_miss.py <worktree>"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_gran as g  # noqa: E402

BINS = (1, 10, 30, 60, 120, 250, 500, 10**9)
QUERIES = {"end": lambda d: d["mass_end"], "r": lambda d: d["mass_r"],
           "end+r": lambda d: [a + b for a, b in zip(d["mass_end"], d["mass_r"])]}


def rank(s, fwc, fact):
    order = sorted(range(g.CPB, fwc), key=lambda c: (-s[c], -c))
    return 1 + min(order.index(x) for x in fact)


h, rel = Counter(), {q: [] for q in QUERIES}
for i, band, fwc, fact, msg in g.out_of_window():
    if band != "beyond":
        continue
    for q, f in QUERIES.items():
        k = rank(f(g.dump[i]), fwc, fact)
        h[(q, next(x for x in BINS if k <= x))] += 1
        rel[q].append(k / (fwc - g.CPB))
print("beyond-pool probes: rank of the fact chunk among the eligible chunks (the budget is 60)")
print(f"{'query':7} " + " ".join(f"{'<=' + (str(x) if x < 10**9 else 'all'):>6}" for x in BINS))
for q in QUERIES:
    print(f"{q:7} " + " ".join(f"{h[(q, x)]:>6}" for x in BINS))
for q, v in rel.items():
    v.sort()
    print(f"{q:7} relative rank p25 p50 p75 p90:", [round(v[int(p * len(v))], 3) for p in (.25, .5, .75, .9)])
print()
g.report(lambda i: QUERIES["end+r"](g.dump[i]),
         {k: v for k, v in g.VARIANTS.items() if k.startswith("units g=10") or k.startswith("units g=1 ") or k == "seed +1"})

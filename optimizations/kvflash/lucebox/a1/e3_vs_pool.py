"""E3 next to B0 POOL16K on the same probes: per probe, the oracle-swap answer (E3), the as-specified swap (E3SPEC) and the exact 16K
pool, scored with the harness scorer, by distance.  python3 -I e3_vs_pool.py <E3E2 evidence dir> <POOL16K.jsonl> <tasks.jsonl>"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agentic_eval"))
import probes  # noqa: E402

d, pool_path, tasks_path = sys.argv[1:4]
tasks = {t["id"]: t for t in map(json.loads, open(tasks_path))}


def rows(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def ok(r):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]))


e3, spec, pool = rows(os.path.join(d, "E3.jsonl")), rows(os.path.join(d, "E3SPEC.jsonl")), rows(pool_path)
BINS = [(0, 4096), (4096, 8192), (8192, 16384), (16384, 10 ** 9)]
tab = {b: [0, 0, 0, 0] for b in BINS}       # E3, E3SPEC, POOL16K correct, n (probes with a POOL16K row)
missing = 0
for i in e3:
    if i not in pool:
        missing += 1
        continue
    t = tasks[i]
    dist = t["distance_tok"]
    b = next(b for b in BINS if b[0] <= dist < b[1])
    tab[b][0] += ok(e3[i])
    tab[b][1] += ok(spec.get(i))
    tab[b][2] += ok(pool[i])
    tab[b][3] += 1
tot = [sum(v[k] for v in tab.values()) for k in range(4)]
for b, (a, s, p, n) in tab.items():
    print(f"{b[0]:>6}-{b[1] if b[1] < 10 ** 9 else 'inf':<6} n={n:3d}  E3 {a:3d}  E3SPEC {s:3d}  POOL16K {p:3d}")
print(f"all          n={tot[3]:3d}  E3 {tot[0]:3d}  E3SPEC {tot[1]:3d}  POOL16K {tot[2]:3d}   (probes without a POOL16K row: {missing})")

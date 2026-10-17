"""PREREG_selv2.md checks 1, 2', 3' (amended) on a run_mq_dump.sh output dir; the 2026-10-16 dump comparison is informational.
    python3 -I prereg_checks.py <mq dump dir> [<old dump dir> [<prs8 OFF dir>]]
Masses below 1e-6 are left out of the relative checks (the near-zero tail); the max over all chunks is printed too."""
import collections
import json
import os
import sys

NEW = sys.argv[1]
OLD = sys.argv[2] if len(sys.argv) > 2 else "/mnt/kimi-k3/split-work/rlru/dump"
OFF = sys.argv[3] if len(sys.argv) > 3 else "/mnt/kimi-k3/split-work/rlru4k/v2/off"


def load(d, f):
    return [json.loads(l) for l in open(os.path.join(d, f))]


def rel(a, b, floor=1e-6):
    assert len(a) == len(b), (len(a), len(b))
    return max((abs(x - y) / max(abs(x), abs(y)) for x, y in zip(a, b) if max(abs(x), abs(y)) >= floor), default=0.0)


reqs, dump, mq = load(NEW, "QK16KND.jsonl"), load(NEW, "exact_dump/exact_dump.jsonl"), load(NEW, "exact_dump/exact_dump_mq.jsonl")
errors = sum("error" in r for r in reqs)
c1 = len(reqs) == len(dump) == len(mq) and errors == 0
print(f"check 1: {len(reqs)} requests, {len(dump)} dump rows, {len(mq)} mq rows, {errors} errors -> {'PASS' if c1 else 'FAIL'}")

self_rel = max(rel(m["mean"][-1], d["mass_end"]) for m, d in zip(mq, dump))
self_all = max(rel(m["mean"][-1], d["mass_end"], 0.0) for m, d in zip(mq, dump))
c2 = self_rel <= 1e-3
print(f"check 2': last mq mean row vs mass_end max rel {self_rel:.2e} (all chunks {self_all:.2e}) -> {'PASS' if c2 else 'FAIL'}")
old = {r["id"]: d for r, d in zip(load(OLD, "QK16KND.jsonl"), load(OLD, "exact_dump/exact_dump.jsonl"))}
vs_old = [rel(d["mass_end"], old[r["id"]]["mass_end"]) for r, d in zip(reqs, dump) if r["id"] in old]
print(f"  info: mass_end vs the 2026-10-16 dump (prs-cuda) on {len(vs_old)}/{len(reqs)} requests max rel {max(vs_old, default=0):.2e}")

off = {r["id"]: r for r in load(OFF, "QK16KND.jsonl")}
key = lambda r: (r.get("content"), r.get("tool_calls"))
probes = [r for r in reqs if r.get("kind") == "probe" and r["id"] in off]
same = sum(key(r) == key(off[r["id"]]) for r in probes)
c3 = bool(probes) and same >= 0.99 * len(probes)
print(f"check 3': answers equal to prs8 OFF on {same}/{len(probes)} probes -> {'PASS' if c3 else 'FAIL'}")
print("n_q per request:", dict(sorted(collections.Counter(m["n_q"] for m in mq).items())))
print("RUN", "VALID" if c1 and c2 and c3 else "VOID")

"""Smokes (c) and (d) (PREREG.md): every probe that fits the pool answers identically in two runs.
  c <drift_off dir> <drift_on2 dir>: same history; fitting = ON2 logged "mode=4: skip, fits" for that request
  d <drift_off dir> <full off dir>: reduced vs full history; fitting = the drift_on2 run's skip set (same requests)
Also asserts the fitting probes took the same restore path (cached prefix, pt) in both runs, else the comparison is void.
    python3 -I gate_cd.py c|d <dir A> <dir B> [<drift_on2 dir>]"""
import json
import os
import sys

which, A, B = sys.argv[1:4]
D = os.path.dirname(os.path.abspath(__file__))
on2 = sys.argv[4] if len(sys.argv) > 4 else os.path.join(D, "smoke", "drift_on2")
rows = [json.loads(l) for l in open(os.path.join(on2, "QK16KND.jsonl"))]
blocks = open(os.path.join(on2, "QK16KND.server.log")).read().split("] chat START ")[1:]
assert len(rows) == len(blocks), (len(rows), len(blocks))
fits = [r["id"] for r, b in zip(rows, blocks) if r.get("kind") == "probe" and "mode=4: skip, fits" in b]
a, b = ({r["id"]: r for r in map(json.loads, open(os.path.join(d, "QK16KND.jsonl")))} for d in (A, B))
key = lambda r: (r["timings"].get("cached_prefix_tokens"), r["timings"].get("prefilled_tokens"))
path = [i[-10:] for i in fits if key(a[i]) != key(b[i])]
diff = [i[-10:] for i in fits if a[i].get("content") != b[i].get("content")]
print(f"smoke {which}: fitting probes {len(fits)}; restore path differs {path}; answers differ {diff}")
sys.exit(0 if fits and not path and not diff else 1)

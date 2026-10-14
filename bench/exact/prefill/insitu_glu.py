#!/usr/bin/env python3
"""Routed gate/up ledger from a rocprofv3 kernel trace of one prefill request (the last request, >1 s idle before it):
calls / total / median us of the gate/up kernels (legacy two-launch or single-launch), the tile builders, the
request wall and the summed kernel time, plus the top kernels by total time.
usage: insitu_glu.py <kernel_trace.csv> [top]"""
import csv, re, statistics as st, sys
from collections import defaultdict

rows = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(sys.argv[1])))
start = 0
for i in range(1, len(rows)):
    if rows[i][0] - rows[i - 1][1] > 1e9:
        start = i
rows = rows[start:]
pats = [
    ("glu legacy <64,128>", r"mmb_routed_glu_kernel<64, 128"),
    ("glu legacy <64,64>", r"mmb_routed_glu_kernel<64, 64"),
    ("glu single-launch", r"mmb_routed_glu_q4k_kernel"),
    ("glu F16 single-launch", r"mmb_routed_glu_f16_kernel"),
    ("cvt bf16->F16 act", r"mmb_cvt_bf16_f16"),
    ("mmb_build_tiles", r"mmb_build_tiles"),
    ("mmb_build_desc2", r"mmb_build_desc2"),
]
d = defaultdict(list)
for s, e, n in rows:
    for k, p in pats:
        if re.search(p, n):
            d[k].append((e - s) / 1000.0)
            break
wall = (rows[-1][1] - rows[0][0]) / 1e6
busy = sum(e - s for s, e, _ in rows) / 1e6
print(f"last request: {len(rows)} kernels, wall {wall:.1f} ms, summed kernel time {busy:.1f} ms")
for k, _ in pats:
    v = d.get(k)
    if v:
        print(f"  {k:22s} calls {len(v):4d} total {sum(v)/1000:8.2f} ms median {st.median(v):8.1f} us mean {st.mean(v):8.1f} min {min(v):8.1f} max {max(v):8.1f}")
tot = defaultdict(float); cnt = defaultdict(int)
for s, e, n in rows:
    tot[n] += (e - s) / 1e6; cnt[n] += 1
print("top kernels by total ms:")
for n, t in sorted(tot.items(), key=lambda kv: -kv[1])[: int(sys.argv[2]) if len(sys.argv) > 2 else 12]:
    print(f"  {t:8.1f} ms {cnt[n]:5d} x  {n[:110]}")

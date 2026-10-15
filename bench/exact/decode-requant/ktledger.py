#!/usr/bin/env python3
"""Per-kernel ledger of one decode token from a rocprofv3 --kernel-trace CSV of luce_server (prof.sh): token
boundaries are the argmax kernel (LUCE_GPU_ARGMAX, one per token), the window is the last N tokens before the last
argmax. Per (kernel, grid_x, workgroup_x): launches / token, us / token (raw duration sum, streams may overlap).
usage: ktledger.py <kt_kernel_trace.csv> [ntokens=120] [filter-substring]"""
import csv, sys
from collections import defaultdict

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
flt = sys.argv[3] if len(sys.argv) > 3 else None

rows = []
with open(path, newline="") as f:
    r = csv.reader(f); hdr = next(r)
    ix = {k: i for i, k in enumerate(hdr)}
    kn, st, en, gx, wx = ix["Kernel_Name"], ix["Start_Timestamp"], ix["End_Timestamp"], ix["Grid_Size_X"], ix["Workgroup_Size_X"]
    for row in r:
        rows.append((int(row[st]), int(row[en]), row[kn], int(row[gx]), int(row[wx])))
rows.sort()
arg = [i for i, x in enumerate(rows) if "argmax" in x[2]]
if len(arg) < N + 1:
    sys.exit(f"only {len(arg)} argmax launches")
lo, hi = arg[-N - 1], arg[-1]           # N tokens: kernels after argmax[-N-1] up to and including argmax[-1]
win = rows[lo + 1:hi + 1]
wall = (rows[hi][1] - rows[lo][1]) / 1e3 / N
agg = defaultdict(lambda: [0, 0.0])
for s, e, name, g, w in win:
    k = ((name.split(">(")[0] + ">" if ">(" in name else name.split("(")[0])[:120], g, w)
    agg[k][0] += 1; agg[k][1] += (e - s) / 1e3
tot = sum(v[1] for v in agg.values()) / N
print(f"window: {N} tokens, {len(win)} launches ({len(win)/N:.1f}/token), wall {wall:.1f} us/token, kernel-sum {tot:.1f} us/token")
print(f"{'launch/tok':>10} {'us/tok':>9} {'us/launch':>9} {'grid_x':>8} {'wg':>4}  kernel")
for (name, g, w), (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if flt and flt not in name: continue
    print(f"{n/N:10.2f} {us/N:9.1f} {us/n:9.2f} {g:8d} {w:4d}  {name}")

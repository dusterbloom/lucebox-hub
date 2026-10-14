#!/usr/bin/env python3
"""Per-kernel stats for the routed gate/up GEMV from a rocprofv3 kernel trace of the server.
Restricts to the last request (>1 s idle before it) like decode/ledger.py, then prints calls,
median/mean us for the gate/up kernels (legacy 8-warp mul_mat_vec_q<...,8> on Q4_K with fusion,
or mul_mat_vec_q_routed_gate_up) and the Q5_1/Q8_0 down kernels for scale.
usage: insitu.py <kernel_trace.csv>"""
import csv, re, statistics as st, sys
from collections import defaultdict

rows = []
for r in csv.DictReader(open(sys.argv[1])):
    rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
# last request: after the last gap > 1 s
start = 0
for i in range(1, len(rows)):
    if rows[i][0] - rows[i - 1][1] > 1e9:
        start = i
rows = rows[start:]
pats = {
    "gate/up legacy 8-warp": r"mul_mat_vec_q<\(ggml_type\)12, 1, true, false, 0, false, false, false, false, false, 8>",
    "gate/up two-wave": r"mul_mat_vec_q_routed_gate_up",
    "down Q5_1 8-warp": r"mul_mat_vec_q<\(ggml_type\)7, 1, false, false, 0, false, false, false, false, false, 8>",
    "down Q8_0 8-warp": r"mul_mat_vec_q<\(ggml_type\)8, 1, false, false, 0, false, false, false, false, false, 8>",
}
d = defaultdict(list)
for s, e, n in rows:
    for k, p in pats.items():
        if re.search(p, n):
            d[k].append((e - s) / 1000.0)
            break
tot = (rows[-1][1] - rows[0][0]) / 1e6 if rows else 0
print(f"last request: {len(rows)} kernels, {tot:.1f} ms wall")
for k, v in d.items():
    print(f"{k:24s} calls {len(v):5d} median {st.median(v):7.1f} us mean {st.mean(v):7.1f} min {min(v):7.1f} max {max(v):7.1f}")

#!/usr/bin/env python3
"""Per (kernel, grid_x) duration stats of a rocprofv3 --kernel-trace CSV (gemvbench runs): n, median / mean / min
us and GB/s for a given MB per launch. usage: ktsum.py <csv> <MB per launch> [skip=8 (warm-up launches)]"""
import csv, statistics as st, sys
from collections import defaultdict

path, mb = sys.argv[1], float(sys.argv[2])
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 8
d = defaultdict(list)
with open(path, newline="") as f:
    r = csv.reader(f); hdr = next(r); ix = {k: i for i, k in enumerate(hdr)}
    for row in r:
        name = row[ix["Kernel_Name"]]
        if "mul_mat_vec" not in name and "mmvq" not in name and "kq_dense" not in name: continue
        d[(name.split(">(")[0] + ">" if ">(" in name else name.split("(")[0], int(row[ix["Grid_Size_X"]]), int(row[ix["Workgroup_Size_X"]]))].append(
            (int(row[ix["End_Timestamp"]]) - int(row[ix["Start_Timestamp"]])) / 1e3)
for (name, g, w), v in d.items():
    v = v[skip:] if len(v) > skip else v
    med = st.median(v)
    print(f"n={len(v):4d} median {med:8.2f} us mean {st.mean(v):8.2f} min {min(v):8.2f} -> {mb / med * 1e3:6.1f} GB/s  grid {g} wg {w}  {name[:100]}")

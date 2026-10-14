#!/usr/bin/env python3
"""Print rocprofv3 kernel_stats rows for the mul_mat_vec kernels (calls, avg/min/max us)."""
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    n = r["Name"]
    if "mul_mat_vec" in n:
        print(f'{r["Calls"]:>6} {float(r["AverageNs"])/1000:8.1f} {float(r["MinNs"])/1000:8.1f} {float(r["MaxNs"])/1000:8.1f}  {n[:120]}')

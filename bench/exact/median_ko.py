#!/usr/bin/env python3
import re, sys, glob, statistics

base = "/home/duster/qwen4exp-exact-stack/logs/ko"
tags = ["ko_base", "ko_hc_A", "ko_hc_B", "ko_q8_C"]
for t in tags:
    all_vals = []
    for r in (1, 2, 3):
        path = f"{base}/{t}_rep{r}.log"
        try:
            vals = []
            for line in open(path):
                m = re.search(r"mode=1.*ms_token=([0-9.]+)", line)
                if m:
                    vals.append(float(m.group(1)))
        except FileNotFoundError:
            print(f"{t} rep={r}: MISSING")
            continue
        vals.sort()
        med = statistics.median(vals) if vals else float("nan")
        print(f"{t} rep={r} n={len(vals)} median={med:.3f} vals={vals}")
        all_vals.extend(vals)
    all_vals.sort()
    if all_vals:
        print(f"{t} ALL n={len(all_vals)} median={statistics.median(all_vals):.3f} mean={statistics.mean(all_vals):.3f}")
    print()

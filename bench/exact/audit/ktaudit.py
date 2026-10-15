#!/usr/bin/env python3
"""Kernel ledger of a rocprofv3 --kernel-trace CSV (trace.sh) for the fusion audit.
  pf  : the measured p4096 request = the last segment after an idle gap > 1 s (req1 warm-up, sleep 3, req2);
        per-kernel count / ms per request, class buckets, every copy / convert / quantize kernel with grid.
  dec : the last N decode tokens (argmax kernel = token boundary, as ktledger.py); us per token.
usage: ktaudit.py <kt_kernel_trace.csv> pf|dec [N=100] [name-filter]"""
import csv, re, sys
from collections import defaultdict

path, mode = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
flt = sys.argv[4] if len(sys.argv) > 4 else None

rows = []
with open(path, newline="") as f:
    r = csv.reader(f); hdr = next(r); ix = {k: i for i, k in enumerate(hdr)}
    kn, st, en, gx, wx = ix["Kernel_Name"], ix["Start_Timestamp"], ix["End_Timestamp"], ix["Grid_Size_X"], ix["Workgroup_Size_X"]
    for row in r:
        rows.append((int(row[st]), int(row[en]), row[kn], int(row[gx]), int(row[wx])))
rows.sort()

def short(name):
    n = name.split(">(")[0] + ">" if ">(" in name else name.split("(")[0]
    n = re.sub(r"^(void |__global__ )", "", n)
    return n[:110]

CLASSES = [
    ("copy/cont", r"cpy_|k_repeat|convert_|mmb_cvt_|cont_|transpose|dup_"),
    ("quantize", r"quantize_"),
    ("dequant", r"dequantize_|k_chunked_dequant|mmb_dq_"),
    ("norm", r"rms_norm|gated_rms_norm|norm_f32|gdn_tail|layer_norm"),
    ("rope", r"rope_"),
    ("attn", r"qsa3_|flash_attn|fattn|softmax"),
    ("gdn", r"gated_delta_net|gdn_conv|gdn_concat|gdn_ab"),
    ("hc", r"hc_"),
    ("gemm", r"mmb_|mul_mat|mmq|mmvq|mmvf|gemm|wmma"),
    ("moe-route", r"mm_ids|mmid|topk|argsort|sort|k_bin_bcast_i32|moe_|route|ids_"),
    ("elementwise", r"k_bin_bcast|unary|silu|swiglu|glu_|scale_|add_|mul_|op_"),
    ("argmax", r"argmax"),
]
def klass(n):
    for c, pat in CLASSES:
        if re.search(pat, n): return c
    return "other"

if mode == "pf":
    # split on idle gaps > 1 s, take the last segment that has > 1000 launches
    segs, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[1] > 1_000_000_000: segs.append(cur); cur = []
        cur.append(b)
    segs.append(cur)
    big = [s for s in segs if len(s) > 1000]
    win = big[-1]
    denom, unit = 1.0, "ms/req"
    print(f"segments: {[len(s) for s in segs]} -> measured = last with >1000 launches ({len(win)} launches), "
          f"span {(win[-1][1]-win[0][0])/1e6:.1f} ms")
else:
    arg = [i for i, x in enumerate(rows) if "argmax" in x[2]]
    if len(arg) < N + 1: sys.exit(f"only {len(arg)} argmax launches")
    lo, hi = arg[-N - 1], arg[-1]
    win = rows[lo + 1:hi + 1]
    denom, unit = N / 1000.0, "us/tok"      # ms -> us per token
    print(f"window: {N} tokens, {len(win)} launches ({len(win)/N:.1f}/token), wall {(rows[hi][1]-rows[lo][1])/1e3/N:.1f} us/token")

agg = defaultdict(lambda: [0, 0.0]); cls = defaultdict(float); grids = defaultdict(lambda: defaultdict(int))
for s, e, name, g, w in win:
    k = short(name); agg[k][0] += 1; agg[k][1] += (e - s) / 1e6; cls[klass(k)] += (e - s) / 1e6
    grids[k][(g, w)] += 1
tot = sum(v[1] for v in agg.values())
print(f"kernel-sum {tot/denom:.1f} {unit}")
print("\n== classes ==")
for c, ms in sorted(cls.items(), key=lambda kv: -kv[1]):
    print(f"{ms/denom:10.2f} {unit} {100*ms/tot:5.1f}%  {c}")
print(f"\n== kernels (count, {unit}, {unit.split('/')[0]}/launch) ==")
for k, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if flt and flt not in k: continue
    print(f"{n:7d} {ms/denom:10.3f} {ms/n*(1000.0 if mode!='pf' else 1):9.3f} {100*ms/tot:5.2f}%  {klass(k):11s} {k}")
print("\n== copy / convert / quantize / dequant launches by grid ==")
for k, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if klass(k) not in ("copy/cont", "quantize", "dequant"): continue
    gl = ", ".join(f"{g}x{w}:{c}" for (g, w), c in sorted(grids[k].items(), key=lambda kv: -kv[1])[:6])
    print(f"{n:7d} {ms/denom:10.3f} {100*ms/tot:5.2f}%  {k}  [{gl}]")

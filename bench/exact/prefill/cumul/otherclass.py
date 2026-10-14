#!/usr/bin/env python3
"""Classify every kernel of the last request in two rocprofv3 traces (luce, gufo) and
print a ranked per-class table (ms, delta) with the top 3 kernels per class on each side,
plus inter-kernel gap time.  usage: otherclass.py <luce_trace.csv> <gufo_trace.csv> [--unmatched]"""
import csv, re, sys
from collections import defaultdict

CLASSES = [  # first match wins
    ("MoE routed expert GEMMs", r"mmb_routed_glu|mmb_routed_kernel|RoutedF16GEMMKernel|mmid|mul_mat_id"),
    ("MoE routing + MoE combine (topk/sort/softmax/counts/combine)", r"k_argsort|k_topk|mm_ids_route|RoutedPad|RoutedScatter|RouterTopK|ExpertCounts|SelectScore|SelectMark|soft_max|hc_combine_norm_moe|moe_fused_combine|HcCombineMoe|reduce_rows|mmb_build_desc"),
    ("HC combine (non-MoE layers)", r"hc_combine_norm_f32|HcCombineVec4|HcCombine"),
    ("HC mix / small-M GEMM / router GEMM", r"hc_gate_mix|mmb_small_m|HcMix|DenseF16GEMMKernel<\d+, \d+, \d+, \d+, \d+, \d+, true|Cijk_|mul_mat_vec|mmvq|qfn_mmq|hc_mix"),
    ("dense GEMM (prefill tiles)", r"mmb_q8f16_kernel|mmb_w8a8_kernel|mmb_f32split|mmb_dense_kernel|DenseF16GEMMKernel|W8A8BlockedWmmaGEMMKernel|WKQuantA8BlockedWmmaGEMMKernel"),
    ("GDN recurrence", r"gated_delta_net|GdnRowSplit|GdnEpilogue"),
    ("GDN conv / prep / concat", r"gdn_conv|SsmConv|GdnPrep|gdn_concat|concat_dim0|arange"),
    ("attention (incl. rope, indexer)", r"qsa|WmmaCausal|rope|Rope|indexer|PoolBlocks|StoreRows|HistoryShift"),
    ("norms / quant / convert", r"rms_norm|gated_rms_norm|RmsNorm|Quantize|mmb_cvt|NarrowKernel|ResToHalf|ResToFloat|cpy_scalar|quantize"),
    ("elementwise / copies", r"k_bin_bcast|unary|Swiglu|copyBuffer|fillBuffer|CopyKernel|cpy_|scale"),
    ("PLE / embed (once per request)", r"ple_|Ple|Embed"),
]

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["Kind"] != "KERNEL_DISPATCH":
                continue
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    start = 0
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][1] > 1_000_000_000:
            start = i
    return rows[start:]

def classify(name):
    for cls, rx in CLASSES:
        if re.search(rx, name):
            return cls
    return "unclassified"

def short(name):
    name = name.replace("(anonymous namespace)::", "").replace("void ", "")
    name = re.sub(r"^(gufo|ggml)[:\w]*::", "", name)
    depth = 0
    for i, ch in enumerate(name):   # cut at the argument list, not at '(' inside template args
        if ch == "<": depth += 1
        elif ch == ">": depth -= 1
        elif ch == "(" and depth == 0:
            name = name[:i]; break
    return name[:80]

def analyse(rows):
    per_class = defaultdict(float)
    per_kernel = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    gap = 0.0
    for i, (s, e, n) in enumerate(rows):
        ms = (e - s) / 1e6
        c = classify(n)
        per_class[c] += ms
        k = per_kernel[c][short(n)]
        k[0] += 1
        k[1] += ms
        if i:
            gap += max(0, s - rows[i - 1][1]) / 1e6
    wall = (rows[-1][1] - rows[0][0]) / 1e6
    return per_class, per_kernel, gap, wall, len(rows)

luce, gufo = load(sys.argv[1]), load(sys.argv[2])
L = analyse(luce)
G = analyse(gufo)
classes = sorted(set(L[0]) | set(G[0]), key=lambda c: -(L[0].get(c, 0) - G[0].get(c, 0)))
print("| class | luce ms | gufo ms | delta ms (luce - gufo) |")
print("|---|---:|---:|---:|")
for c in classes:
    print(f"| {c} | {L[0].get(c, 0):.1f} | {G[0].get(c, 0):.1f} | {L[0].get(c, 0) - G[0].get(c, 0):+.1f} |")
print(f"| inter-kernel gaps (sum of idle between consecutive kernels) | {L[2]:.1f} | {G[2]:.1f} | {L[2] - G[2]:+.1f} |")
print(f"| wall (first start -> last end) | {L[3]:.1f} | {G[3]:.1f} | {L[3] - G[3]:+.1f} |")
print(f"| kernel launches | {L[4]} | {G[4]} | {L[4] - G[4]:+d} |")
print()
for c in classes:
    print(f"### {c}")
    for side, A in (("luce", L), ("gufo", G)):
        top = sorted(A[1][c].items(), key=lambda kv: -kv[1][1])[:4]
        for name, (n, ms) in top:
            print(f"- {side}: {ms:.1f} ms = {n} x {ms / n:.3f}  `{name}`")
if "--unmatched" in sys.argv:
    for side, A in (("luce", L), ("gufo", G)):
        print(f"\nunclassified {side}:")
        for name, (n, ms) in sorted(A[1]["unclassified"].items(), key=lambda kv: -kv[1][1])[:40]:
            print(f"  {ms:8.1f} ms  {n:5d} x  {name}")

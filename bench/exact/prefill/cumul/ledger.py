#!/usr/bin/env python3
"""Ranked per-class ledger from 2 rocprofv3 traces per engine: luce1 luce2 gufo1 gufo2.
Prints class | luce t1 / t2 | gufo t1 / t2 | delta (mean luce - mean gufo) | trace-to-trace variation per engine."""
import sys, importlib.util
spec = importlib.util.spec_from_file_location("oc", sys.argv[1]); oc = importlib.util.module_from_spec(spec)
import io, contextlib
sys.argv = [sys.argv[0], "luce1.csv", "gufo1.csv"]   # otherclass runs its own main on import; silence it
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(oc)
L = [oc.analyse(oc.load(f)) for f in ("luce1.csv", "luce2.csv")]
G = [oc.analyse(oc.load(f)) for f in ("gufo1.csv", "gufo2.csv")]
classes = set()
for A in L + G: classes |= set(A[0])
def m(A, c): return A[0].get(c, 0.0)
rows = []
for c in classes:
    l1, l2, g1, g2 = m(L[0], c), m(L[1], c), m(G[0], c), m(G[1], c)
    rows.append((c, l1, l2, g1, g2, (l1 + l2) / 2 - (g1 + g2) / 2, abs(l1 - l2), abs(g1 - g2)))
rows.sort(key=lambda r: -r[5])
print("| class | luce t1 / t2 ms | gufo t1 / t2 ms | delta ms (luce - gufo, means) | luce t1-t2 | gufo t1-t2 |")
print("|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r[0]} | {r[1]:.1f} / {r[2]:.1f} | {r[3]:.1f} / {r[4]:.1f} | {r[5]:+.1f} | {r[6]:.1f} | {r[7]:.1f} |")
lw = [A[3] for A in L]; gw = [A[3] for A in G]; lg = [A[2] for A in L]; gg = [A[2] for A in G]
print(f"| inter-kernel gaps (rocprof artifact) | {lg[0]:.1f} / {lg[1]:.1f} | {gg[0]:.1f} / {gg[1]:.1f} | {(sum(lg)-sum(gg))/2:+.1f} | {abs(lg[0]-lg[1]):.1f} | {abs(gg[0]-gg[1]):.1f} |")
print(f"| kernel sum (wall - gaps) | {lw[0]-lg[0]:.1f} / {lw[1]-lg[1]:.1f} | {gw[0]-gg[0]:.1f} / {gw[1]-gg[1]:.1f} | {(sum(lw)-sum(lg)-sum(gw)+sum(gg))/2:+.1f} | {abs(lw[0]-lg[0]-lw[1]+lg[1]):.1f} | {abs(gw[0]-gg[0]-gw[1]+gg[1]):.1f} |")
print(f"| wall (first start -> last end) | {lw[0]:.1f} / {lw[1]:.1f} | {gw[0]:.1f} / {gw[1]:.1f} | {(sum(lw)-sum(gw))/2:+.1f} | {abs(lw[0]-lw[1]):.1f} | {abs(gw[0]-gw[1]):.1f} |")
print(f"| kernel launches | {L[0][4]} / {L[1][4]} | {G[0][4]} / {G[1][4]} | | | |")
print()
print("Top kernels per class (trace 1 / trace 2, ms = calls x mean):")
for r in rows:
    c = r[0]
    print(f"### {c}")
    for side, AA in (("luce", L), ("gufo", G)):
        names = {}
        for A in AA:
            for name, (n, ms) in A[1][c].items(): names.setdefault(name, []).append((n, ms))
        top = sorted(names.items(), key=lambda kv: -max(x[1] for x in kv[1]))[:5]
        for name, lst in top:
            print(f"- {side}: " + " / ".join(f"{ms:.1f} = {n} x {ms/n:.3f}" for n, ms in lst) + f"  `{name}`")

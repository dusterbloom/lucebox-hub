#!/usr/bin/env python3
"""Summarise cumul.sh output: per arm median (min-max) of p4096 / p16384 / decode, per-round medians,
A-vs-A and B-vs-B spreads (max - min of the round medians, % of the arm median), deltas vs A,
greedy-64 first divergence, and the caching proof from the server logs.  usage: cumsum.py <outdir>"""
import glob, json, os, re, statistics as st, sys

out = sys.argv[1]
print(open(f"{out}/binary.md5").read().strip()); print(open(f"{out}/arms.txt").read().strip())

def rounds(arm, metric, key):
    rr = {}
    for p in sorted(glob.glob(f"{out}/{arm}[0-9]*_{metric}.jsonl")):
        tag = os.path.basename(p).split("_")[0]
        rr[tag] = [json.loads(l)[key] for l in open(p) if l.strip()]
    return rr

for metric, key in [("p4096", "prefill_tok_s"), ("p16384", "prefill_tok_s"), ("decode", "decode_tok_s")]:
    print(f"\n== {metric} tok/s ==")
    med = {}
    for arm in "ABCD":
        rr = rounds(arm, metric, key)
        if not rr: continue
        flat = [v for r in rr.values() for v in r]
        med[arm] = st.median(flat)
        rm = [st.median(v) for v in rr.values()]
        spread = (max(rm) - min(rm)) / med[arm] * 100 if len(rm) > 1 else float("nan")
        print(f"  {arm}: median {med[arm]:8.2f}  min {min(flat):8.2f}  max {max(flat):8.2f}  n={len(flat)}  round medians {[round(x, 1) for x in rm]}  spread {spread:.2f}%")
        if metric == "decode": print(f"     ms/token {1000 / med[arm]:.2f}")
    for arm in "BCD":
        if arm in med and "A" in med: print(f"  {arm} vs A: {(med[arm] / med['A'] - 1) * 100:+.2f}%")
    if "C" in med and "B" in med: print(f"  C vs B: {(med['C'] / med['B'] - 1) * 100:+.2f}%")

print("\n== greedy-64 ==")
for p in ("p4096", "d512"):
    texts = {os.path.basename(f).split("_")[0]: json.load(open(f))["text"] for f in sorted(glob.glob(f"{out}/*_greedy64_{p}.json"))}
    for a, b in (("A", "B"), ("B", "C")):
        if a in texts and b in texts:
            ta, tb = texts[a], texts[b]
            if ta == tb: print(f"  {p} {a} vs {b}: identical ({len(ta)} chars)")
            else:
                i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
                print(f"  {p} {a} vs {b}: differ at char {i}: {a}={ta[max(0, i - 20):i + 40]!r} {b}={tb[max(0, i - 20):i + 40]!r}")

print("\n== caching proof / server logs ==")
for f in sorted(glob.glob(f"{out}/serve_*.log")):
    lines = open(f, errors="replace").read().splitlines()
    pl = sorted({m.group(0) for l in lines for m in [re.search(r"prefix_len=\d+", l)] if m})
    ca = sorted({m.group(0) for l in lines for m in [re.search(r"cache=\w+ cached_tokens=\d+", l)] if m})
    es = [l for l in lines if "exact-stack defaults" in l]
    print(f"  {os.path.basename(f)}: prefix_len {pl} cache {ca} | {es[-1].strip()[:90] if es else ''}")

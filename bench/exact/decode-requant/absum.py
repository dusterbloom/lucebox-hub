#!/usr/bin/env python3
"""Summarise ab.sh output: per arm median / mean / min / max of p4096, p16384 and decode tok/s, per-round
medians and their spread, deltas vs A (luce exact) and C vs B / C vs D (gufo), greedy-64 divergence, and the
caching + fusion proof from the server logs. usage: absum.py <outdir>"""
import glob, json, os, re, statistics as st, sys

out = sys.argv[1]
print(open(f"{out}/binary.md5").read().strip()); arms_txt = open(f"{out}/arms.txt").read().strip(); print(arms_txt)
# ab.sh arms are A-D; ab2.sh writes one NAME=ENV/FLAGS line per arm, every arm is compared with the first.
ARMS = [l.split("=", 1)[0] for l in arms_txt.splitlines()] if ": " not in arms_txt else list("ABCD")
BASE = ARMS[0]

def rounds(arm, metric, key):
    rr = {}
    for p in sorted(glob.glob(f"{out}/{arm}[0-9]*_{metric}.jsonl")):
        b = os.path.basename(p)   # arm A must not pick up A13's files
        if any(len(o) > len(arm) and b.startswith(o) for o in ARMS): continue
        rr[os.path.basename(p).split("_")[0]] = [json.loads(l)[key] for l in open(p) if l.strip()]
    return rr

for metric, key in [("p4096", "prefill_tok_s"), ("p16384", "prefill_tok_s"), ("decode", "decode_tok_s")]:
    print(f"\n== {metric} tok/s ==")
    med = {}
    for arm in ARMS:
        rr = rounds(arm, metric, key)
        if not rr: continue
        flat = [v for r in rr.values() for v in r]
        med[arm] = st.median(flat)
        rm = [st.median(v) for v in rr.values()]
        spread = (max(rm) - min(rm)) / med[arm] * 100 if len(rm) > 1 else float("nan")
        print(f"  {arm}: median {med[arm]:8.2f} mean {st.mean(flat):8.2f} min {min(flat):8.2f} max {max(flat):8.2f} n={len(flat)}"
              f" round medians {[round(x, 1) for x in rm]} spread {spread:.2f}%  raw {[round(x, 1) for x in flat]}")
        if metric == "decode": print(f"     ms/token {1000 / med[arm]:.2f}")
    for arm in ARMS[1:]:
        if arm in med and BASE in med: print(f"  {arm} vs {BASE}: {(med[arm] / med[BASE] - 1) * 100:+.2f}%")
    if "C" in med and "B" in med: print(f"  C vs B: {(med['C'] / med['B'] - 1) * 100:+.2f}%")
    if "D" in med and "C" in med: print(f"  C vs D: {(med['C'] / med['D'] - 1) * 100:+.2f}%")

print("\n== greedy-64 ==")
for p in ("p4096", "d512"):
    texts = {os.path.basename(f).split("_")[0]: json.load(open(f))["text"] for f in sorted(glob.glob(f"{out}/*_greedy64_{p}.json"))}
    for a, b in [(BASE, x) for x in ARMS[1:]] + [("B", "C")]:
        if a in texts and b in texts:
            ta, tb = texts[a], texts[b]
            if ta == tb: print(f"  {p} {a} vs {b}: identical ({len(ta)} chars)")
            else:
                i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
                print(f"  {p} {a} vs {b}: differ at char {i}: {a}={ta[max(0, i - 20):i + 40]!r} {b}={tb[max(0, i - 20):i + 40]!r}")

print("\n== caching / fusion proof ==")
for f in sorted(glob.glob(f"{out}/serve_*.log")):
    lines = open(f, errors="replace").read().splitlines()
    pl = sorted({m.group(0) for l in lines for m in [re.search(r"prefix_len=\d+", l)] if m})
    ca = sorted({m.group(0) for l in lines for m in [re.search(r"cache=\w+ cached_tokens=\d+", l)] if m})
    fu = [l.strip()[:100] for l in lines if "capture] sites" in l or "decode copies" in l]
    print(f"  {os.path.basename(f)}: prefix_len {pl} cache {ca} {fu}")

#!/usr/bin/env python3
"""Summarise mtp_ab.sh output: per arm and workload the client decode tok/s (median / mean / min / max, per-round
medians and their spread), the server's own decode tok/s, acceptance rate, tokens per step, draft ms and the
width histogram from the [qwen4exp-mtp] lines; p4096 / p16384 prefill when present; greedy text identity between
arms; deltas of every arm against the first. usage: mtp_absum.py <outdir>"""
import glob, json, os, re, statistics as st, sys

out = sys.argv[1]
print(open(f"{out}/binary.md5").read().strip())
arms_txt = open(f"{out}/arms.txt").read().strip(); print(arms_txt)
ARMS = [l.split("=", 1)[0] for l in arms_txt.splitlines()]
BASE = ARMS[0]
WLS = ["code16k", "prose16k", "counting"]

def files(arm, suffix):
    rr = {}
    for p in sorted(glob.glob(f"{out}/{arm}[0-9]*_{suffix}")):
        b = os.path.basename(p)
        if any(len(o) > len(arm) and b.startswith(o) for o in ARMS): continue
        rr[b.split("_")[0]] = p
    return rr

def mtp_fields(line):
    return {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", line)}

def stats(label, rr_vals, unit="tok/s"):
    flat = [v for r in rr_vals.values() for v in r]
    if not flat: return None
    rm = [st.median(v) for v in rr_vals.values()]
    med = st.median(flat)
    spread = (max(rm) - min(rm)) / med * 100 if len(rm) > 1 else float("nan")
    print(f"  {label}: median {med:8.2f} mean {st.mean(flat):8.2f} min {min(flat):8.2f} max {max(flat):8.2f} n={len(flat)}"
          f" round medians {[round(x, 1) for x in rm]} spread {spread:.2f}%")
    return med

for wl in WLS:
    print(f"\n== {wl}: decode tok/s (client) ==")
    med = {}
    for arm in ARMS:
        rr = files(arm, f"{wl}.tsv")
        if not rr: continue
        client, server, rate, tps, draft, hist = {}, [], [], [], [], {}
        for r, p in rr.items():
            rows = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
            client[r] = [json.loads(x[0])["decode_tok_s"] for x in rows]
            for x in rows:
                if len(x) < 2 or "qwen4exp-mtp" not in x[1]: continue
                f = mtp_fields(x[1])
                server.append(f["decode"]); rate.append(f["rate"]); tps.append(f["tokens_per_step"]); draft.append(f["draft_ms"])
                for k in range(1, 8): hist[k] = hist.get(k, 0) + int(f.get(f"steps_k{k}", 0))
        med[arm] = stats(arm, client)
        if server:
            tot = sum(hist.values()) or 1
            print(f"     server decode median {st.median(server):.2f} tok/s  accept rate {st.mean(rate):.3f}"
                  f"  tokens/step {st.mean(tps):.3f}  draft {st.mean(draft):.2f} ms/draft"
                  f"  widths {{{', '.join(f'k{k}:{100 * n / tot:.0f}%' for k, n in sorted(hist.items()) if n)}}}")
    for arm in ARMS[1:]:
        if med.get(arm) and med.get(BASE): print(f"  {arm} vs {BASE}: {(med[arm] / med[BASE] - 1) * 100:+.2f}%")

for metric in ("p4096", "p16384"):
    rr_any = any(files(a, f"{metric}.jsonl") for a in ARMS)
    if not rr_any: continue
    print(f"\n== {metric} prefill tok/s ==")
    med = {}
    for arm in ARMS:
        rr = {r: [json.loads(l)["prefill_tok_s"] for l in open(p) if l.strip()] for r, p in files(arm, f"{metric}.jsonl").items()}
        if rr: med[arm] = stats(arm, rr)
    for arm in ARMS[1:]:
        if med.get(arm) and med.get(BASE): print(f"  {arm} vs {BASE}: {(med[arm] / med[BASE] - 1) * 100:+.2f}%")

print("\n== greedy identity ==")
for wl in WLS:
    texts = {os.path.basename(f).split("_")[0]: json.load(open(f))["text"] for f in sorted(glob.glob(f"{out}/*_greedy_{wl}.json"))}
    for b in ARMS[1:]:
        if BASE in texts and b in texts:
            ta, tb = texts[BASE], texts[b]
            if ta == tb: print(f"  {wl} {BASE} vs {b}: identical ({len(ta)} chars)")
            else:
                i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
                print(f"  {wl} {BASE} vs {b}: differ at char {i}: {BASE}={ta[max(0, i - 20):i + 40]!r} {b}={tb[max(0, i - 20):i + 40]!r}")

print("\n== server log proof ==")
for f in sorted(glob.glob(f"{out}/serve_*.log")):
    lines = open(f, errors="replace").read().splitlines()
    pl = sorted({m.group(0) for l in lines for m in [re.search(r"prefix_len=\d+", l)] if m})
    fu = [l.strip()[:110] for l in lines if "capture] sites" in l or "decode copies" in l or "verify-width" in l]
    print(f"  {os.path.basename(f)}: prefix_len {pl} {fu[:3]}")

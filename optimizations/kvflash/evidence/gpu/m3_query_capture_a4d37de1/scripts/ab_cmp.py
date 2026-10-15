import json, re, statistics as S
O = "/mnt/kimi-k3/kvflash-gate-m3/"
ARMS = {a: ["%s_%d" % (a, i) for i in range(1, 6)] for a in ("m2", "m3", "m3cap")}
RUNS = ARMS["m2"] + ARMS["m3"] + ARMS["m3cap"]
DONE = re.compile(r"chat DONE .*? in=(\d+) effective_in=(\d+) out=(\d+) ([\d.]+)s .*?prefix_len=(-?\d+) prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)")
HIT = re.compile(r"disk_hit=(\w+)")
R = {}
for k in RUNS:
    rows, hits = [], []
    for l in open(O + k + "/gate4_A.log", errors="replace"):
        m = DONE.search(l)
        if m:
            i, e, o, w, p, pf, d, t = m.groups()
            rows.append(dict(eff=int(e), out=int(o), wall=float(w), prefix=int(p), pf=float(pf), tps=float(t)))
        h = HIT.search(l)
        if h and "chat CACHE" in l:
            hits.append(h.group(1))
    R[k] = (rows[1:], hits[1:])  # drop the 1319-token warm-up request; turn 1 is rows[0] after this
ans = {k: [r["answer"] for r in json.load(open(O + k + "/gate4_A.json"))] for k in RUNS}
print("answers byte-identical across the 10 m2+m3 runs:", all(ans[k] == ans[RUNS[0]] for k in ARMS["m2"] + ARMS["m3"]), "(turns: %d)" % len(ans[RUNS[0]]))
print("answers of m3cap runs identical to m2:", all(ans[k] == ans[RUNS[0]] for k in ARMS["m3cap"]), "(chunk cuts differ under capture, so this may legitimately be False)")
p0 = [r["prefix"] for r in R[RUNS[0]][0]]
print("prefix_len identical turn for turn:", all([r["prefix"] for r in R[k][0]] == p0 for k in RUNS), p0)
print("disk_hit=false everywhere:", all(h == "false" for k in RUNS for h in R[k][1]), "(%d lines)" % sum(len(R[k][1]) for k in RUNS))
print("turns per run:", {k: len(R[k][0]) for k in RUNS})

def pt(r): return (r["eff"] - r["prefix"]) / r["pf"]
def stats(v): return "%9.2f %9.2f %9.2f %9.2f" % (S.median(v), S.mean(v), min(v), max(v))
print("\n%-34s %9s %9s %9s %9s" % ("metric (per-run value pooled by arm)", "median", "mean", "min", "max"))
for tag, ks in ARMS.items():
    t1 = [pt(R[k][0][0]) for k in ks]
    rest = [S.median(pt(r) for r in R[k][0][1:]) for k in ks]  # per-run median over turns 2-10
    restp = [pt(r) for k in ks for r in R[k][0][1:]]
    dec = [r["tps"] for k in ks for r in R[k][0]]
    wall = [r["wall"] for k in ks for r in R[k][0]]
    print(tag)
    print("  %-32s %s" % ("turn-1 prefill tok/s (5 runs)", stats(t1)))
    print("  %-32s %s" % ("turns 2-10 prefill tok/s (pooled)", stats(restp)))
    print("  %-32s %s" % ("turns 2-10 prefill, per-run medians", stats(rest)))
    print("  %-32s %s" % ("decode tok/s (all turns pooled)", stats(dec)))
    print("  %-32s %s" % ("wall s (all turns pooled)", stats(wall)))
    print("  turn-1 prefill in run order:", ["%.0f" % v for v in t1])
    print("  turn-1 prefill seconds:", ["%.2f" % R[k][0][0]["pf"] for k in ks])
    print("  turn-1 wall s:", ["%.2f" % R[k][0][0]["wall"] for k in ks])

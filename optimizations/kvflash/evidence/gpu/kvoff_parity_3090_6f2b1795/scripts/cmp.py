"""Compare main vs E runs. usage: python3 -I cmp.py <run dir>  (holds main-run1, E-run1, main-run2, E-run2)"""
import json, re, sys, statistics as S
O = sys.argv[1].rstrip("/") + "/"
DONE = re.compile(r"chat DONE .*? in=(\d+) effective_in=(\d+) out=(\d+) ([\d.]+)s .*?prefix_len=(-?\d+) prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)")
J, R, DH = {}, {}, {}
for k in ("main-run1", "E-run1", "main-run2", "E-run2"):
    J[k] = json.load(open(O + k + "/gate4_A.json"))
    rows = []
    txt = open(O + k + "/gate4_A.log", errors="replace").read()
    for l in txt.splitlines():
        m = DONE.search(l)
        if m:
            i, e, o, w, p, pf, d, t = m.groups()
            rows.append(dict(inp=int(i), eff=int(e), prefix=int(p), pf=float(pf), tps=float(t), wall=float(w)))
    R[k] = rows[1:]  # drop warm-up
    DH[k] = (txt.count("disk_hit=false"), txt.count("disk_hit=true"))
    assert len(R[k]) == 10, (k, len(R[k]))
pt = lambda r: (r["eff"] - r["prefix"]) / r["pf"] if r["pf"] > 0 else float("nan")
for n in ("1", "2"):
    a, b = "main-run" + n, "E-run" + n
    print(f"=== pair {n}: {a} vs {b}")
    print("turn in prefix_main prefix_E pf_main pf_E pftps_main pftps_E tps_main tps_E wall_main wall_E identical")
    same = 0
    for i in range(10):
        ok = J[a][i]["answer"] == J[b][i]["answer"]
        same += ok
        m, s = R[a][i], R[b][i]
        print(i + 1, m["inp"], m["prefix"], s["prefix"], m["pf"], s["pf"], "%.0f" % pt(m), "%.0f" % pt(s), m["tps"], s["tps"], m["wall"], s["wall"], ok)
    pm = [r["prefix"] for r in R[a]]
    pe = [r["prefix"] for r in R[b]]
    print(f"{'IDENTICAL' if same == 10 else 'DIFFERENT'} {same}/10; prefix_len match all turns: {pm == pe}; input tokens match: {[r['inp'] for r in R[a]] == [r['inp'] for r in R[b]]}")
print("=== disk_hit (false count, true count) over the whole server log incl. warm-up:", DH)
print("=== prefix_len per arm:")
for k in R:
    print(k, [r["prefix"] for r in R[k]])


def stats(X):
    pts = [pt(r) for r in X if r["pf"] > 0]
    t = [r["tps"] for r in X]
    w = [r["wall"] for r in X]
    return "prefill tok/s med %.0f (n=%d) | decode med %.2f mean %.2f best %.2f min %.2f | wall med %.2f mean %.2f" % (
        S.median(pts), len(pts), S.median(t), S.mean(t), max(t), min(t), S.median(w), S.mean(w))


print("=== medians per run (prefill tok/s = (effective_in - prefix_len) / prefill s; 'cold' turn 1 excluded in the second line)")
for k in R:
    print(k, stats(R[k]))
    print(k, "turns 2-10:", stats(R[k][1:]))
    print(k, "cold turn 1: prefill %.2f s, %.0f tok/s" % (R[k][0]["pf"], pt(R[k][0])))
print("=== pooled (both runs, 20 turns)")
for arm, ks in (("main", ("main-run1", "main-run2")), ("E", ("E-run1", "E-run2"))):
    X = R[ks[0]] + R[ks[1]]
    print(arm, stats(X))
    print(arm, "turns 2-10:", stats(R[ks[0]][1:] + R[ks[1]][1:]))

import json, re, statistics as S
O = "/mnt/kimi-k3/kvflash-gate-kvoff/"
N = {"main": "main-af64", "stack": "stack-fc5563d1"}
J = {k: json.load(open(O + v + "/gate4_A.json")) for k, v in N.items()}
DONE = re.compile(r"chat DONE .*? in=(\d+) effective_in=(\d+) out=(\d+) ([\d.]+)s .*?prefix_len=(-?\d+) prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)")
R = {}
for k, v in N.items():
    rows = []
    for l in open(O + v + "/gate4_A.log", errors="replace"):
        m = DONE.search(l)
        if m:
            i, e, o, w, p, pf, d, t = m.groups()
            rows.append(dict(inp=int(i), eff=int(e), out=int(o), wall=float(w), prefix=int(p), pf=float(pf), tps=float(t)))
    R[k] = rows[1:]  # drop warm-up
    print(k, "rows", len(R[k]))
same = 0
print("turn in_main in_stack prefix_main prefix_stack pf_main pf_stack pftps_main pftps_stack tps_main tps_stack wall_main wall_stack ident")
for i in range(10):
    a, b = J["main"][i], J["stack"][i]
    ok = a["answer"] == b["answer"]
    same += ok
    m, s = R["main"][i], R["stack"][i]
    pt = lambda r: (r["eff"] - r["prefix"]) / r["pf"] if r["pf"] > 0 else float("nan")
    print(i + 1, m["inp"], s["inp"], m["prefix"], s["prefix"], m["pf"], s["pf"], "%.0f" % pt(m), "%.0f" % pt(s), m["tps"], s["tps"], m["wall"], s["wall"], ok)
print("identical", same, "/10; prefix match all:", [r["prefix"] for r in R["main"]] == [r["prefix"] for r in R["stack"]])
for k in N:
    X = R[k]
    pts = [(r["eff"] - r["prefix"]) / r["pf"] for r in X if r["pf"] > 0]
    t = [r["tps"] for r in X]
    print(k, "prefill tok/s med %.0f | decode med %.2f mean %.2f best %.2f worst %.2f | wall med %.2f" % (
        S.median(pts), S.median(t), S.mean(t), max(t), min(t), S.median(r["wall"] for r in X)))

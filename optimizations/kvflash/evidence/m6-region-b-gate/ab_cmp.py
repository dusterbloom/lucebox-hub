import json, re, statistics as st, sys
R = "/mnt/kimi-k3/kvflash-gate-m6"
arms = {}
for arm in ("main", "m6"):
    rows = []
    for i in range(1, 6):
        for r in json.load(open(f"{R}/{arm}_{i}/m6_run.json")):
            d = r["done"][0]; c = r["cache"][0]
            g = lambda k: re.search(k + r"=([\d.]+)", d).group(1)
            ein, pre = int(g("effective_in")), int(re.search(r"prefix_len=(\d+)", c).group(1))
            rows.append(dict(run=i, p=r["prompt"], text=r["text"], ein=ein, pre=pre,
                             hit=re.search(r"disk_hit=(\w+)", c).group(1),
                             pf=(ein - pre) / float(g("prefill")) if False else (ein - pre) / float(re.search(r"prefill=([\d.]+)s", d).group(1)),
                             dec=float(re.search(r"decode=[\d.]+s\(([\d.]+)tok/s", d).group(1)),
                             wall=float(re.search(r"out=\d+ ([\d.]+)s", d).group(1))))
    arms[arm] = rows
a, b = arms["main"], arms["m6"]
print("disk_hit values:", {x["hit"] for x in a + b})
print("prefix_len/effective_in equal across arms:", all((x["pre"], x["ein"]) == (y["pre"], y["ein"]) for x, y in zip(a, b)))
ident = all(x["text"] == y["text"] for x, y in zip(a, b))
print("greedy text identical across arms, all 25 pairs:", ident)
for p in range(5):
    t = {x["text"] for x in a + b if x["p"] == p}
    print(f"  prompt {p}: distinct texts across 10 runs = {len(t)}")
print(f"{'metric':<14}{'arm':<6}{'median':>9}{'mean':>9}{'min':>9}{'max':>9}")
for k, n in (("pf", "prefill tok/s"), ("dec", "decode tok/s"), ("wall", "wall s")):
    for arm in ("main", "m6"):
        v = [x[k] for x in arms[arm]]
        print(f"{n:<14}{arm:<6}{st.median(v):9.2f}{st.mean(v):9.2f}{min(v):9.2f}{max(v):9.2f}")
    d = st.median([x[k] for x in b]) / st.median([x[k] for x in a]) - 1
    print(f"{'':<14}m6 vs main median: {d*100:+.2f}%")

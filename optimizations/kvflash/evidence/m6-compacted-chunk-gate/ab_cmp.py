import json, re, statistics as st
R = "/mnt/kimi-k3/kvflash-gate-m6c"
ARMS = ("main", "m6c", "neutral")
arms = {}
for arm in ARMS:
    rows = []
    for i in range(1, 6):
        for r in json.load(open(f"{R}/{arm}_{i}/m6_run.json")):
            d, c = r["done"][0], r["cache"][0]
            ein = int(re.search(r"effective_in=(\d+)", d).group(1))
            pre = int(re.search(r"prefix_len=(\d+)", c).group(1))
            rows.append(dict(run=i, p=r["prompt"], text=r["text"], ein=ein, pre=pre,
                             hit=re.search(r"disk_hit=(\w+)", c).group(1),
                             pf=(ein - pre) / float(re.search(r"prefill=([\d.]+)s", d).group(1)),
                             dec=float(re.search(r"decode=[\d.]+s\(([\d.]+)tok/s", d).group(1)),
                             wall=float(re.search(r"out=\d+ ([\d.]+)s", d).group(1))))
    arms[arm] = rows
a, b, n = (arms[k] for k in ARMS)
print("disk_hit values:", {x["hit"] for k in ARMS for x in arms[k]})
print("prefix_len/effective_in equal across the three arms:",
      all((x["pre"], x["ein"]) == (y["pre"], y["ein"]) == (z["pre"], z["ein"]) for x, y, z in zip(a, b, n)))
print("GATE (a) greedy text identical, main vs m6c, all 25 pairs:", all(x["text"] == y["text"] for x, y in zip(a, b)))
print("GATE (b) greedy text identical, neutral vs m6c, all 25 pairs:", all(x["text"] == y["text"] for x, y in zip(b, n)))
for p in range(5):
    print(f"  prompt {p}: distinct texts across 15 runs = {len({x['text'] for k in ARMS for x in arms[k] if x['p'] == p})}")
print(f"{'metric':<14}{'arm':<9}{'median':>9}{'mean':>9}{'min':>9}{'max':>9}{'vs main':>10}")
for k, name in (("pf", "prefill tok/s"), ("dec", "decode tok/s"), ("wall", "wall s")):
    base = st.median([x[k] for x in a])
    for arm in ARMS:
        v = [x[k] for x in arms[arm]]
        print(f"{name:<14}{arm:<9}{st.median(v):9.2f}{st.mean(v):9.2f}{min(v):9.2f}{max(v):9.2f}{(st.median(v) / base - 1) * 100:+9.2f}%")

"""AIME Table 9 analysis: python3 -I analyze_t9.py <dir with run subdirs> (e.g. _dl/qwen3-4b)."""
import glob, json, re, sys, random, statistics as st

root = sys.argv[1]
T3 = 3.182  # t(0.975, df=3)
runs = {}  # (name, seed) -> {problem_idx: correct}
for f in glob.glob(f"{root}/*/*.json"):
    if f.endswith("manifest.json"):
        continue
    d = json.load(open(f))
    rs = d.get("results") or []
    if not rs or "is_correct" not in rs[0]:
        continue
    nm = f.split("/")[-2]
    m = re.match(r"(t9_compact_2048_t0\.5_c\d|baseline_(?:2048|4096|8192))_s(\d)(?:_p\d)?$", nm)
    if not m:
        continue
    cfg, seed = m.group(1), int(m.group(2))
    for r in rs:
        runs.setdefault((cfg, seed), {})[r["problem_idx"]] = int(r["is_correct"])

cfgs = sorted({c for c, _ in runs})
def per_seed(c):
    return [sum(runs[(c, s)].values()) for s in sorted(s for cc, s in runs if cc == c)]
def mci(xs):
    m = st.mean(xs); h = T3 * st.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h
paper = {"baseline_2048": 1.25, "baseline_4096": 7.75, "baseline_8192": 13,
         "t9_compact_2048_t0.5_c2": 8, "t9_compact_2048_t0.5_c6": 13}
print("config                      per-seed(/30)     mean  95%CI(t,df3)      paper  paper_in_CI")
for c in cfgs:
    ps = per_seed(c); m, lo, hi = mci(ps)
    p = paper.get(c)
    print(f"{c:27s} {str(ps):17s} {m:5.2f} [{lo:5.2f},{hi:5.2f}]  {p}  {lo <= p <= hi if p is not None else ''}")

def seedavg(c):
    seeds = sorted(s for cc, s in runs if cc == c)
    return [st.mean(runs[(c, s)][i] for s in seeds) for i in range(30)]
random.seed(0)
for comp, base in (("t9_compact_2048_t0.5_c2", "baseline_4096"), ("t9_compact_2048_t0.5_c6", "baseline_8192")):
    a, b = seedavg(comp), seedavg(base)
    diff = [x - y for x, y in zip(a, b)]
    ds = []
    for _ in range(10000):
        smp = [diff[random.randrange(30)] for _ in range(30)]
        ds.append(sum(smp))  # in problems of 30
    ds.sort()
    print(f"paired {comp} - {base}: mean {sum(diff):+.2f} problems, bootstrap95 [{ds[250]:+.2f},{ds[9750]:+.2f}] (threshold lower >= -3)")

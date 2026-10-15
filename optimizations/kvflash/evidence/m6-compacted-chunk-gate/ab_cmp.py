"""Gate comparison. Exits non-zero on ANY failure: a missing or extra row, a row
whose request did not finish ok / produce 256 tokens / run uncached, a mismatch
in prefix_len or effective_in, or any greedy text difference (main vs m6c, and
neutral vs m6c). Rows are keyed by (run, prompt), never paired by position."""
import json, re, statistics as st, sys

R = "/mnt/kimi-k3/kvflash-gate-m6c"
ARMS = ("main", "m6c", "neutral")
RUNS, PROMPTS = range(1, 6), range(5)
fails = []


def fail(msg):
    fails.append(msg)
    print("FAIL:", msg)


def grab(rx, s, what):
    m = re.search(rx, s)
    if not m:
        raise ValueError(f"{what}: no match for {rx!r} in {s!r}")
    return m.group(1)


arms = {}
for arm in ARMS:
    rows = {}
    for i in RUNS:
        for r in json.load(open(f"{R}/{arm}_{i}/m6_run.json")):
            key = (i, r["prompt"])
            if key in rows:
                fail(f"{arm}: duplicate row {key}")
                continue
            if len(r["done"]) != 1 or len(r["cache"]) != 1:
                fail(f"{arm} {key}: {len(r['done'])} DONE lines and {len(r['cache'])} CACHE lines, want 1 and 1")
                continue
            d, c = r["done"][0], r["cache"][0]
            if "ok=true" not in d or "finish=length" not in d or r["ct"] != 256:
                fail(f"{arm} {key}: not a clean 256-token run: {d}")
            ein = int(grab(r"effective_in=(\d+)", d, "effective_in"))
            pre = int(grab(r"prefix_len=(\d+)", c, "prefix_len"))
            rows[key] = dict(text=r["text"], ein=ein, pre=pre, hit=grab(r"disk_hit=(\w+)", c, "disk_hit"),
                             pf=(ein - pre) / float(grab(r"prefill=([\d.]+)s", d, "prefill")),
                             dec=float(grab(r"decode=[\d.]+s\(([\d.]+)tok/s", d, "decode")),
                             wall=float(grab(r"out=\d+ ([\d.]+)s", d, "wall")))
    arms[arm] = rows

want = {(i, p) for i in RUNS for p in PROMPTS}
for arm in ARMS:
    got = set(arms[arm])
    if got != want:
        fail(f"{arm}: rows missing {sorted(want - got)} extra {sorted(got - want)}")
    print(f"{arm}: {len(got)} rows")

hits = {x["hit"] for k in ARMS for x in arms[k].values()}
print("disk_hit values:", hits)
if hits != {"false"}:
    fail(f"disk_hit must be false everywhere, saw {hits}")

common = sorted(want & set.intersection(*(set(arms[k]) for k in ARMS)))
for key in common:
    a, b, n = (arms[k][key] for k in ARMS)
    if not (a["pre"], a["ein"]) == (b["pre"], b["ein"]) == (n["pre"], n["ein"]):
        fail(f"{key}: prefix_len/effective_in differ across arms")
    if a["text"] != b["text"]:
        fail(f"GATE (a) {key}: main and m6c greedy text differ")
    if b["text"] != n["text"]:
        fail(f"GATE (b) {key}: neutral and m6c greedy text differ")
print(f"GATE (a) main == m6c greedy text on {len(common)} keyed pairs; GATE (b) neutral == m6c on {len(common)}")
for p in PROMPTS:
    print(f"  prompt {p}: distinct texts across {len(RUNS) * 3} runs = "
          f"{len({x['text'] for k in ARMS for key, x in arms[k].items() if key[1] == p})}")

print(f"{'metric':<14}{'arm':<9}{'median':>9}{'mean':>9}{'min':>9}{'max':>9}{'vs main':>10}")
for k, name in (("pf", "prefill tok/s"), ("dec", "decode tok/s"), ("wall", "wall s")):
    base = st.median([x[k] for x in arms["main"].values()])
    for arm in ARMS:
        v = [x[k] for x in arms[arm].values()]
        print(f"{name:<14}{arm:<9}{st.median(v):9.2f}{st.mean(v):9.2f}{min(v):9.2f}{max(v):9.2f}"
              f"{(st.median(v) / base - 1) * 100:+9.2f}%")

if fails:
    print(f"RESULT: FAIL ({len(fails)} problems)")
    sys.exit(1)
print("RESULT: PASS")

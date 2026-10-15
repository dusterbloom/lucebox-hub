"""M7 speed comparison. Exits non-zero on ANY failure: a missing or extra row, a row whose request did not
finish ok / produce 256 tokens / run uncached, a mismatch in prefix_len or effective_in, or a greedy text
difference between main and m7 (compaction off). The m7on arm (compaction on) is informational: its text may
differ, and it must report compacted==sealed blocks and empty nan_layers. Rows are keyed by (run, prompt)."""
import json, re, statistics as st, sys

R = "/mnt/kimi-k3/kvflash-gate-m7/speed"
ARMS = ("main", "m7", "m7on")
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
        for r in json.load(open(f"{R}/{arm}_{i}/speed_run.json")):
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
            row = dict(text=r["text"], ein=ein, pre=pre, hit=grab(r"disk_hit=(\w+)", c, "disk_hit"),
                       pf=(ein - pre) / float(grab(r"prefill=([\d.]+)s", d, "prefill")),
                       dec=float(grab(r"decode=[\d.]+s\(([\d.]+)tok/s", d, "decode")),
                       wall=float(grab(r"out=\d+ ([\d.]+)s", d, "wall")))
            if arm == "m7on":
                if len(r["am"]) != 1:
                    fail(f"{arm} {key}: {len(r['am'])} compaction summaries, want 1")
                else:
                    m = re.search(r"sealed=(\d+) compacted=(\d+) skipped=(\d+) failed=(\d+).*per_block=([\d.]+)ms nan_layers=\[(.*?)\]", r["am"][0])
                    s_, c_, k_, f_, pb, nl = m.groups()
                    row.update(per_block=float(pb))
                    if s_ != c_ or f_ != "0" or nl:
                        fail(f"{arm} {key}: sealed {s_} compacted {c_} skipped {k_} failed {f_} nan_layers [{nl}]")
            elif r["am"]:
                fail(f"{arm} {key}: compaction ran with the ratio unset: {r['am']}")
            rows[key] = row
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
        fail(f"GATE {key}: main and m7 (compaction off) greedy text differ")
print(f"GATE main == m7 (compaction off) greedy text on {len(common)} keyed pairs")
print("m7on greedy text equal to main on", sum(arms["main"][k]["text"] == arms["m7on"][k]["text"] for k in common), "of", len(common), "(informational)")
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
pb = [x["per_block"] for x in arms["m7on"].values() if "per_block" in x]
if pb:
    print(f"compaction per sealed block (ms), m7on: median {st.median(pb):.1f} min {min(pb):.1f} max {max(pb):.1f}")
sys.exit(1 if fails else 0)

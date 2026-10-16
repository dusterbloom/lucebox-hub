"""Read rules of PREREG_v2.md (rlru4k-v2), fixed before any row. Arms off, on2, on6, on10 under this folder, all on prs8,
ON arms with LUCE_KVFLASH_SCORE_POOL_RECENT=1. prs7 arms (../off, ../on2, ../on6, ../on10) are read for the speed
comparison and the mechanism table only.
    python3 -P analyze_v2.py <worktree>   (-P: numpy lives in the user site)"""
import json
import os
import re
import statistics as st
import sys

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
K4 = os.path.join(D, "..")
R4 = os.path.join(K4, "..", "rlru4")
sys.path.insert(0, os.path.join(R4, "agentic_eval"))
sys.path.insert(0, os.path.join(W, "optimizations/kvflash/lucebox/a1"))
sys.path.insert(0, D)
import probes  # noqa: E402
from a1_core import boot_ci  # noqa: E402
import resident_cmp  # noqa: E402

CT, CPB, OLD = 64, 10, 16384
ARMS = ("off", "on2", "on6", "on10")
ON = ARMS[1:]
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
retained = json.load(open(os.path.join(R4, "retained.json")))
diag27 = set(json.load(open(os.path.join(K4, "diag27.json"))))
pp = json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))
span = {r["id"]: r["span"]["fact"] for r in pp}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]: (\d+) page events"
                 r"(?:; (\d+) in \(([0-9.]+) MiB H2D\))?.* scorer ([0-9.]+) ms")
REC = re.compile(r"prefill reselect recent-pool @\d+: (\d+) page events; (\d+) in \(([0-9.]+) MiB H2D\)")


def rows_of(d):
    return [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]


def blocks_of(d, n):
    b = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(b) == n, (d, len(b), n)
    return b


allrows = {a: rows_of(os.path.join(D, a)) for a in ARMS}
bad = {a: sum(bool(r["timings"].get("disk_hit")) for r in allrows[a] if "timings" in r) for a in ARMS}
assert not any(bad.values()), f"disk_hit true on rows: {bad}"
arms = {a: {r["id"]: r for r in allrows[a] if r.get("kind", "probe") == "probe"} for a in ARMS}
ids = [i for i in retained if all(i in arms[a] for a in ARMS)]
print(f"probes on all arms: {len(ids)} of {len(retained)}; errors " + ", ".join(f"{a} {sum('error' in arms[a][i] for i in ids)}" for a in ARMS))

# Structural check first: no answer number counts unless every ON arm's resident sets equal OFF's on every request.
struct = {a: resident_cmp.main(os.path.join(D, "off"), os.path.join(D, a)) == 0 for a in ON}
print("structural check (0 mismatches):", struct)


def ok(r, i):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


v = {a: {i: ok(arms[a][i], i) for i in ids} for a in ARMS}
within = [i for i in ids if tasks[i]["distance_tok"] <= OLD]
within_x = [i for i in within if i not in diag27]
beyond = [i for i in ids if tasks[i]["distance_tok"] > OLD]
cl = lambda S: [tasks[i]["traj"] for i in S]
diff = lambda x, y, S: boot_ci([v[x][i] - v[y][i] for i in S], cl(S))
out = {"counts": {a: {"overall": sum(v[a][i] for i in ids), "within": sum(v[a][i] for i in within),
                      "within_excl27": sum(v[a][i] for i in within_x), "beyond": sum(v[a][i] for i in beyond)} for a in ARMS}}
for a in ARMS:
    c = out["counts"][a]
    print(f"{a:5} overall {c['overall']:>4}/{len(ids)}  within {c['within']:>4}/{len(within)}  "
          f"within excl. 27 {c['within_excl27']:>4}/{len(within_x)}  beyond {c['beyond']:>4}/{len(beyond)}")
pct = lambda d: f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
cmp = {"on6-on2": diff("on6", "on2", ids), "on10-on6": diff("on10", "on6", ids)}
for a in ON:
    cmp[f"{a}-off"] = diff(a, "off", ids)
    cmp[f"{a}-off within"] = diff(a, "off", within)
    cmp[f"{a}-off within excl27"] = diff(a, "off", within_x)
    cmp[f"{a}-off beyond"] = diff(a, "off", beyond)
for k, d in cmp.items():
    print(f"  {k:24} {pct(d)} pts")
out["diffs"] = cmp


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def speeds(d, a):
    rows = rows_of(d)
    pr = [r for r in rows if r.get("kind", "probe") == "probe" and r["id"] in set(ids) and "error" not in r]
    s = {"prefill_ms": med([r["timings"].get("prefill_ms") for r in pr]),
         "decode_tok_s": med([r["timings"].get("decode_tokens_per_sec") for r in pr]),
         "client_wall_s": med([r.get("client_wall_s") for r in pr]),
         "pt": med([r["timings"].get("prefilled_tokens") for r in pr]),
         "cached_prefix_tokens": med([r["timings"].get("cached_prefix_tokens") for r in pr])}
    if a != "off":
        blk = blocks_of(d, len(rows))
        sel = [(r, SEL.search(b)) for r, b in zip(rows, blk)]
        hit = [m for _, m in sel if m]
        s["scorer_ms"] = med([float(m[8]) for m in hit])
        s["wall_minus_scorer_s"] = med([r["client_wall_s"] - (float(m[8]) / 1000 if m else 0) for r, m in sel
                                        if r.get("kind", "probe") == "probe" and r.get("client_wall_s") is not None])
        s["page_events_per_reselect"] = med([int(m[5]) for m in hit])
        s["page_ins_per_reselect"] = med([int(m[6]) for m in hit if m[6]])
        s["h2d_mib_per_reselect"] = med([float(m[7]) for m in hit if m[7]])
        s["requests_reselected"] = f"{len(hit)}/{len(sel)}"
        rec = [m for b in blk for m in REC.findall(b)]
        if rec:
            s["recency_steps"] = len(rec)
            s["recency_page_events_median"] = med([int(x[0]) for x in rec])
            s["recency_h2d_mib_median"] = med([float(x[2]) for x in rec])
            s["recency_h2d_mib_max"] = max(float(x[2]) for x in rec)
    return s


speed = {a: speeds(os.path.join(D, a), a) for a in ARMS}
speed_prs7 = {a: speeds(os.path.join(K4, a), a) for a in ARMS}
print("prs8 (v2):", json.dumps(speed, indent=1))
print("prs7 (rlru4k, ON arms flag off) for comparison:", json.dumps(speed_prs7, indent=1))
print("clock: WSL tsc ~7% slow, absolute tok/s ~8% high; same-box ratios hold")


def mechanism(d):
    """Out-of-window probes: engine picks hold the source? dump's top k (OFF state, mass_end) holds it?"""
    R = os.path.join(K4, "..", "rlru", "dump")
    dump = {r["id"]: x for r, x in zip(map(json.loads, open(os.path.join(R, "QK16KND.jsonl"))),
                                        map(json.loads, open(os.path.join(R, "exact_dump", "exact_dump.jsonl"))))}
    rows = rows_of(d)
    tab = {}
    for r, b in zip(rows, blocks_of(d, len(rows))):
        i, m = r["id"], SEL.search(b)
        if r.get("kind") != "probe" or i not in span or i not in dump or not m or i not in set(ids):
            continue
        k, tail, r4, picked = int(m[1]), int(m[2]), int(m[3]), {int(x) for x in m[4].split(",") if x}
        fwc = (r4 - 1) // CT - tail
        fa, fb = span[i]
        if fb // CT >= fwc:
            continue
        src = {c // CPB for c in range(fa // CT, fb // CT + 1)}
        s = dump[i]["mass_end"]
        top = [x for _, x in sorted(((sum(s[x * CPB:(x + 1) * CPB]), x) for x in range(1, fwc // CPB)), key=lambda t: (-t[0], -t[1]))[:k]]
        key = f"{'beyond' if tasks[i]['distance_tok'] > OLD else 'within'} engine {'hit' if src & picked else 'miss'} dump {'hit' if src & set(top) else 'miss'}"
        tab[key] = tab.get(key, 0) + 1
    return dict(sorted(tab.items()))


mech = {a: {"prs8_v2": mechanism(os.path.join(D, a)), "prs7": mechanism(os.path.join(K4, a))} for a in ON}
print("mechanism (out-of-window sources):", json.dumps(mech, indent=1))

guard = {a: cmp[f"{a}-off within"]["lo"] >= -0.020 and cmp[f"{a}-off within excl27"]["lo"] >= -0.020 for a in ON}
bar = {a: out["counts"][a]["overall"] >= 719 and guard[a] for a in ON}
rule1 = cmp["on6-on2"]["lo"] > 0
rule2 = rule1 and cmp["on10-on6"]["lo"] > 0
valid = all(struct.values())
ship = ("void: structural check failed" if not valid else
        "k=10 (flag on)" if (rule1 and rule2 and guard["on10"] and bar["on10"]) else
        "k=6 (flag on)" if (rule1 and guard["on6"] and bar["on6"]) else
        "k=2 (flag on)" if (guard["on2"] and bar["on2"]) else "k=2 (flag off, rlru4 design)")
w2 = speed["on2"]["wall_minus_scorer_s"]
verdict = {"structural_check": struct, "rule1_on6_beats_on2": rule1, "rule2_on10_beats_on6": rule2,
           "rule3_guard_all_and_excl27": guard, "rule4_bar": bar,
           "rule6_speed_flag_vs_on2": {a: bool(w2 and speed[a]["wall_minus_scorer_s"] > 1.25 * w2) for a in ("on6", "on10")},
           "rule7_pt_equal": len({speed[a]["pt"] for a in ARMS}) == 1, "ship": ship}
print("predictions: see PREREG_v2.md; actual " + ", ".join(f"{a} {out['counts'][a]['overall']}" for a in ON))
print(json.dumps(verdict, indent=1))
json.dump({**out, "speed": speed, "speed_prs7": speed_prs7, "mechanism": mech, "verdict": verdict},
          open(os.path.join(D, "report_v2.json"), "w"), indent=1, default=str)

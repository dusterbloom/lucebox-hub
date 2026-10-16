"""Read rules of PREREG.md (this folder, larger k), fixed before any row. Arms off, on2, on6, on10 under this folder.
    python3 -P analyze4k.py <worktree>   (-P: numpy lives in the user site)"""
import json
import os
import re
import statistics as st
import sys

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.join(D, "..", "rlru4")
sys.path.insert(0, os.path.join(R4, "agentic_eval"))
sys.path.insert(0, os.path.join(W, "optimizations/kvflash/lucebox/a1"))
import probes  # noqa: E402
from a1_core import boot_ci  # noqa: E402

CT, OLD = 64, 16384
ARMS = ("off", "on2", "on6", "on10")
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
retained = json.load(open(os.path.join(R4, "retained.json")))
span = {r["id"]: r["span"]["fact"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[[0-9,]*\]: (\d+) page events(?:; (\d+) in \(([0-9.]+) MiB H2D\))?.* scorer ([0-9.]+) ms")

allrows = {a: [json.loads(l) for l in open(os.path.join(D, a, "QK16KND.jsonl"))] for a in ARMS}
bad = {a: sum(bool(r["timings"].get("disk_hit")) for r in allrows[a] if "timings" in r) for a in ARMS}
assert not any(bad.values()), f"disk_hit true on rows: {bad}"
arms = {a: {r["id"]: r for r in allrows[a] if r.get("kind", "probe") == "probe"} for a in ARMS}
ids = [i for i in retained if all(i in arms[a] for a in ARMS)]
print(f"probes on all arms: {len(ids)} of {len(retained)}; errors " + ", ".join(f"{a} {sum('error' in arms[a][i] for i in ids)}" for a in ARMS))


def ok(r, i):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


v = {a: {i: ok(arms[a][i], i) for i in ids} for a in ARMS}
within = [i for i in ids if tasks[i]["distance_tok"] <= OLD]
beyond = [i for i in ids if tasks[i]["distance_tok"] > OLD]
cl = lambda S: [tasks[i]["traj"] for i in S]
diff = lambda x, y, S: boot_ci([v[x][i] - v[y][i] for i in S], cl(S))
out = {"counts": {a: {"overall": sum(v[a][i] for i in ids), "within": sum(v[a][i] for i in within), "beyond": sum(v[a][i] for i in beyond)} for a in ARMS}}
for a in ARMS:
    c = out["counts"][a]
    print(f"{a:5} overall {c['overall']:>4}/{len(ids)}  within {c['within']:>4}/{len(within)}  beyond {c['beyond']:>4}/{len(beyond)}")
pct = lambda d: f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
cmp = {"on6-on2": diff("on6", "on2", ids), "on10-on6": diff("on10", "on6", ids)}
for a in ("on2", "on6", "on10"):
    cmp[f"{a}-off"] = diff(a, "off", ids)
    cmp[f"{a}-off within"] = diff(a, "off", within)
    cmp[f"{a}-off beyond"] = diff(a, "off", beyond)
for k, d in cmp.items():
    print(f"  {k:18} {pct(d)} pts")
out["diffs"] = cmp


def blocks(a):
    b = open(os.path.join(D, a, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(b) == len(allrows[a]), (a, len(b), len(allrows[a]))
    return b


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


speed, strata = {}, {}
for a in ARMS:
    pr = [arms[a][i] for i in ids if "error" not in arms[a][i]]
    s = {"prefill_ms": med([r["timings"].get("prefill_ms") for r in pr]),
         "decode_tok_s": med([r["timings"].get("decode_tokens_per_sec") for r in pr]),
         "client_wall_s": med([r.get("client_wall_s") for r in pr]),
         "pt": med([r["timings"].get("prefilled_tokens") for r in pr]),
         "cached_prefix_tokens": med([r["timings"].get("cached_prefix_tokens") for r in pr])}
    if a != "off":
        sel = [(r, SEL.search(b)) for r, b in zip(allrows[a], blocks(a))]
        hit = [(r, m) for r, m in sel if m]
        s["scorer_ms"] = med([float(m[7]) for _, m in hit])
        s["wall_minus_scorer_s"] = med([r["client_wall_s"] - (float(m[7]) / 1000 if m else 0) for r, m in sel
                                        if r.get("kind", "probe") == "probe" and r.get("client_wall_s") is not None])
        s["page_events_per_reselect"] = med([int(m[4]) for _, m in hit])
        s["page_ins_per_reselect"] = med([int(m[5]) for _, m in hit if m[5]])
        s["h2d_mib_per_reselect"] = med([float(m[6]) for _, m in hit if m[6]])
        s["requests_reselected"] = f"{len(hit)}/{len(sel)}"
        # stratum: within-16K probes whose fact lies outside this arm's own window
        S = []
        for r, m in sel:
            i = r["id"]
            if r.get("kind") == "probe" and m and i in span and i in within:
                fwc = (int(m[3]) - 1) // CT - int(m[2])
                if span[i][1] // CT < fwc:
                    S.append(i)
        strata[a] = {"n": len(S), "off": sum(v["off"][i] for i in S), "on2": sum(v["on2"][i] for i in S), a: sum(v[a][i] for i in S)}
    speed[a] = s
print(json.dumps(speed, indent=1))
print("stratum (within 16K, source outside the arm's window):", json.dumps(strata))

guard = {a: cmp[f"{a}-off within"]["lo"] >= -0.020 for a in ("on2", "on6", "on10")}
bar = {a: out["counts"][a]["overall"] >= 719 and guard[a] for a in ("on2", "on6", "on10")}
rule1 = cmp["on6-on2"]["lo"] > 0
rule2 = rule1 and cmp["on10-on6"]["lo"] > 0
ship = "k=10" if (rule1 and rule2 and guard["on10"] and bar["on10"]) else "k=6" if (rule1 and guard["on6"] and bar["on6"]) else "k=2"
w2 = speed["on2"]["wall_minus_scorer_s"]
verdict = {"rule1_on6_beats_on2": rule1, "rule2_on10_beats_on6": rule2, "rule3_guard": guard, "rule4_bar": bar,
           "rule6_speed_flag_vs_on2": {a: bool(w2 and speed[a]["wall_minus_scorer_s"] > 1.25 * w2) for a in ("on6", "on10")},
           "rule7_pt_equal": len({speed[a]["pt"] for a in ARMS}) == 1, "ship": ship}
print(f"predictions: ON2 near 766, ON6 804, ON10 820; actual " + ", ".join(f"{a} {out['counts'][a]['overall']}" for a in ("on2", "on6", "on10")))
print(json.dumps(verdict, indent=1))
json.dump({**out, "speed": speed, "strata": strata, "verdict": verdict}, open(os.path.join(D, "report.json"), "w"), indent=1, default=str)

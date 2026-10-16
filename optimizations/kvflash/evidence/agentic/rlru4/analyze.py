"""Read rules of PREREG.md (this folder, R-LRU mode 4), fixed before any mode-4 row. Copied from ../rlru/analyze.py: log regex for mode=4, prediction 726, per-request scorer, wall minus scorer, share rewound.
    python3 -P analyze.py <worktree>   (-P, not -I: numpy lives in the user site)   (the kvflash-am worktree: harness probes.py, a1_core.boot_ci, B0 and E3E2 rows)"""
import json
import os
import re
import statistics as st
import sys

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "agentic_eval"))
sys.path.insert(0, os.path.join(W, "optimizations/kvflash/lucebox/a1"))
import probes  # noqa: E402
from a1_core import boot_ci  # noqa: E402

OLD = 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
retained = json.load(open(os.path.join(D, "retained.json")))


def rows(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if r.get("kind", "probe") == "probe"}


def ok(r, i):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


arms = {a: rows(os.path.join(D, a, "QK16KND.jsonl")) for a in ("off", "on")}
b0 = rows(os.path.join(W, "optimizations/kvflash/evidence/agentic/B0_3090/POOL16K.jsonl"))
e3 = rows(os.path.join(W, "optimizations/kvflash/evidence/agentic/E3E2/E3.jsonl"))
ids = [i for i in retained if all(i in arms[a] for a in arms)]
print(f"probes with rows on both arms: {len(ids)} of {len(retained)}; errors off {sum('error' in arms['off'][i] for i in ids)}, "
      f"on {sum('error' in arms['on'][i] for i in ids)}")
v = {a: {i: ok(arms[a][i], i) for i in ids} for a in arms}
v["b0"] = {i: ok(b0.get(i), i) for i in ids}
subs = {"overall": ids, "within 16K": [i for i in ids if tasks[i]["distance_tok"] <= OLD],
        "beyond 16K": [i for i in ids if tasks[i]["distance_tok"] > OLD],
        "beyond 16K, shared with E3": [i for i in ids if tasks[i]["distance_tok"] > OLD and i in e3 and i in b0]}
out = {}
for name, S in subs.items():
    cl = [tasks[i]["traj"] for i in S]
    res = {a: {"correct": sum(v[a][i] for i in S), "n": len(S)} for a in ("off", "on", "b0")}
    for a in ("off", "on"):
        res[a]["ci"] = boot_ci([v[a][i] for i in S], cl) if S else None
    res["on_minus_off"] = boot_ci([v["on"][i] - v["off"][i] for i in S], cl) if S else None
    if "E3" in name:
        res["e3"] = {"correct": sum(ok(e3[i], i) for i in S), "n": len(S)}
    out[name] = res
    d = res["on_minus_off"]
    print(f"{name:28} OFF {res['off']['correct']:>4}/{len(S):<4} ON {res['on']['correct']:>4}/{len(S):<4} B0 {res['b0']['correct']:>4}"
          + (f"  ON-OFF {100 * d['mean']:+.1f} pts [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]" if d else "")
          + (f"  E3 {res['e3']['correct']}" if "e3" in res else ""))


def med(a, f):
    xs = [f(arms[a][i]) for i in ids if "error" not in arms[a][i]]
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


speed = {a: {"prefill_ms": med(a, lambda r: r["timings"].get("prefill_ms")),
             "decode_tok_s": med(a, lambda r: r["timings"].get("decode_tokens_per_sec")),
             "client_wall_s": med(a, lambda r: r.get("client_wall_s")),
             "prefilled_tokens (pt)": med(a, lambda r: r["timings"].get("prefilled_tokens")),
             "cached_prefix_tokens": med(a, lambda r: r["timings"].get("cached_prefix_tokens")),
             "cache_hit_share": sum(bool(arms[a][i]["timings"].get("cache_hit")) for i in ids) / len(ids),
             "disk_hit_true": sum(bool(arms[a][i]["timings"].get("disk_hit")) for i in ids)} for a in arms}
# Per request (the log is in row order; one block per "chat START"): scorer ms, rewound or skipped.
blocks = open(os.path.join(D, "on", "QK16KND.server.log")).read().split("] chat START ")[1:]
on_rows = [json.loads(l) for l in open(os.path.join(D, "on", "QK16KND.jsonl"))]
assert len(blocks) == len(on_rows), (len(blocks), len(on_rows))
req = []
for r, b in zip(on_rows, blocks):
    m = re.search(r"mode=4 scorer=exact .* scorer ([0-9.]+) ms", b)
    req.append({"id": r["id"], "kind": r.get("kind"), "scorer_ms": float(m.group(1)) if m else 0.0,
                "rewound": bool(re.search(r"mode=4 @\d+: re-prefill \d+ tokens ok", b)), "wall": r.get("client_wall_s")})
scorer = [q["scorer_ms"] for q in req if q["scorer_ms"]]
speed["on"]["scorer_ms_median"] = st.median(scorer) if scorer else None
speed["on"]["client_wall_s_minus_scorer"] = st.median([q["wall"] - q["scorer_ms"] / 1000 for q in req if q["kind"] == "probe" and q["wall"] is not None])
speed["on"]["requests_rewound"] = f"{sum(q['rewound'] for q in req)}/{len(req)}"
for band, S in (("within", subs["within 16K"]), ("beyond", subs["beyond 16K"])):
    S = set(S)
    speed["on"][f"probes_rewound_{band}"] = f"{sum(q['rewound'] for q in req if q['id'] in S)}/{len(S)}"
print(json.dumps(speed, indent=1))
print(f"prediction (PREREG step 1): 726 overall, 679 within 16K, 46 beyond; "
      f"actual {out['overall']['on']['correct']}, {out['within 16K']['on']['correct']}, {out['beyond 16K']['on']['correct']}")
ov, wi = out["overall"], out["within 16K"]
drop = (wi["off"]["correct"] - wi["on"]["correct"]) / len(subs["within 16K"])
pt_off, pt_on = speed["off"]["prefilled_tokens (pt)"], speed["on"]["prefilled_tokens (pt)"]
verdict = {
    "rule1_primary_win": ov["on_minus_off"]["lo"] > 0,
    "rule2_beyond16K_win": out["beyond 16K"]["on_minus_off"]["lo"] > 0,
    "rule3_within16K_drop_pts": 100 * drop, "rule3_guard_ok": drop <= 0.020,
    "rule4_on_retention": ov["on"]["correct"] / len(ids), "rule4_clears_bar": ov["on"]["correct"] / len(ids) >= 719 / 951
    and ov["on_minus_off"]["lo"] > 0 and drop <= 0.020,
    "rule6_prefill_flag": bool(speed["on"]["prefill_ms"] and speed["off"]["prefill_ms"] and speed["on"]["prefill_ms"] > 1.5 * speed["off"]["prefill_ms"]),
    "rule7_speed_void": bool(pt_off and pt_on and abs(pt_on - pt_off) / pt_off > 0.10)}
print(json.dumps(verdict, indent=1))
json.dump({"subsets": out, "speed": speed, "verdict": verdict}, open(os.path.join(D, "report.json"), "w"), indent=1)

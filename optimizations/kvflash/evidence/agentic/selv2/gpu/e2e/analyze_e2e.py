"""Read rules of PREREG_e2e.md, fixed before any row. v2 against on6ref (same binary and session); while on6ref is missing
or short of the retained probes, against the shipped prs8 ON6 arm, marked provisional. prs8 OFF is the structural reference.
    python3 -P analyze_e2e.py   (-P: numpy lives in the user site)"""
import json
import math
import os
import re
import statistics as st
import sys

E = os.path.dirname(os.path.abspath(__file__))
SW = "/mnt/kimi-k3/split-work"
K = f"{SW}/rlru4k/v2"
R4 = f"{SW}/rlru4"
W = "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am"
sys.path.insert(0, f"{R4}/agentic_eval")
sys.path.insert(0, f"{W}/optimizations/kvflash/lucebox/a1")
sys.path.insert(0, K)
import probes  # noqa: E402
from a1_core import boot_ci  # noqa: E402
import resident_cmp  # noqa: E402

CT, OLD = 64, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
retained = json.load(open(f"{R4}/retained.json"))
diag27 = set(json.load(open(f"{SW}/rlru4k/diag27.json")))
spans = {r["id"]: r["span"]["fact"] for r in
         json.load(open(f"{W}/optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json"))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]: (\d+) page events"
                 r"(?:; (\d+) in \(([0-9.]+) MiB H2D\))?.* scorer ([0-9.]+) ms")


def rows_of(d):
    return [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]


def probes_of(d):
    return {r["id"]: r for r in rows_of(d) if r.get("kind", "probe") == "probe"}


on6ref = os.path.join(E, "on6ref")
final = os.path.exists(os.path.join(on6ref, "QK16KND.jsonl")) and all(i in probes_of(on6ref) for i in retained)
DIRS = {"v2": (os.path.join(E, "v2"), 2), "ref": (on6ref if final else os.path.join(K, "on6"), 10),
        "on6_prs8": (os.path.join(K, "on6"), 10), "off_prs8": (os.path.join(K, "off"), None)}
print(f"reference: {'on6ref (same binary, same session) - FINAL' if final else 'prs8 ON6 (rlru4k/v2/on6) - PROVISIONAL'}")
arms = {a: probes_of(d) for a, (d, _) in DIRS.items()}
ids = [i for i in retained if all(i in arms[a] for a in arms)]
errs = {a: sum("error" in arms[a][i] for i in ids) for a in arms}
disk = {a: sum(bool(r.get("timings", {}).get("disk_hit")) for r in rows_of(d)) for a, (d, _) in DIRS.items()}
print(f"probes on all arms: {len(ids)} of {len(retained)}; errors {errs}; disk_hit rows {disk}")
struct = {a: resident_cmp.main(DIRS["off_prs8"][0], DIRS[a][0]) == 0 for a in ("v2", "ref")}
print("structural check (0 mismatches against prs8 OFF):", struct)


def ok(r, i):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


v = {a: {i: ok(arms[a][i], i) for i in ids} for a in arms}
within = [i for i in ids if tasks[i]["distance_tok"] <= OLD]
within_x = [i for i in within if i not in diag27]
beyond = [i for i in ids if tasks[i]["distance_tok"] > OLD]
bands = {"overall": ids, "within": within, "within_excl27": within_x, "beyond": beyond}
counts = {a: {b: sum(v[a][i] for i in S) for b, S in bands.items()} for a in arms}
for a in arms:
    print(f"{a:9} " + "  ".join(f"{b} {counts[a][b]:>4}/{len(bands[b])}" for b in bands))


def sign_p(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n)


diffs = {}
for b, S in bands.items():
    d = boot_ci([v["v2"][i] - v["ref"][i] for i in S], [tasks[i]["traj"] for i in S])
    d["v2_only"] = sum(v["v2"][i] > v["ref"][i] for i in S)
    d["ref_only"] = sum(v["v2"][i] < v["ref"][i] for i in S)
    d["sign_p"] = sign_p(d["v2_only"], d["ref_only"])
    diffs[b] = d
    print(f"  v2-ref {b:14} {100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] pts; "
          f"discordant v2 {d['v2_only']} ref {d['ref_only']} (sign p {d['sign_p']:.2g})")
same = sum(arms["ref"][i].get("content") == arms["on6_prs8"][i].get("content") for i in ids)
print(f"reference vs prs8 ON6: answers equal {same}/{len(ids)}" + ("" if final else " (same arm)"))


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def per_request(d):
    rows = rows_of(d)
    b = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(b) == len(rows), (d, len(b), len(rows))
    return list(zip(rows, b))


def speeds(a):
    d, unit = DIRS[a]
    reqs = per_request(d)
    pr = [r for r, _ in reqs if r.get("kind", "probe") == "probe" and r["id"] in set(ids) and "error" not in r]
    s = {"prefill_ms": med([r["timings"].get("prefill_ms") for r in pr]),
         "decode_tok_s": med([r["timings"].get("decode_tokens_per_sec") for r in pr]),
         "client_wall_s": med([r.get("client_wall_s") for r in pr]),
         "pt": med([r["timings"].get("prefilled_tokens") for r in pr]),
         "cached_prefix_tokens": med([r["timings"].get("cached_prefix_tokens") for r in pr])}
    if unit:
        hit = [m for _, b in reqs for m in [SEL.search(b)] if m]
        s["scorer_ms"] = med([float(m[8]) for m in hit])
        s["scorer_ms_max"] = max((float(m[8]) for m in hit), default=None)
        s["h2d_mib_per_reselect"] = med([float(m[7]) for m in hit if m[7]])
        s["requests_reselected"] = f"{len(hit)}/{len(reqs)}"
    return s


speed = {a: speeds(a) for a in ("v2", "ref", "off_prs8")}
print("speeds (medians over probes; WSL clock: same-box ratios hold):", json.dumps(speed, indent=1))


def mechanism(a):
    """Out-of-window facts: fully held by the picks, partly, missed; probes and correct answers per class."""
    d, unit = DIRS[a]
    tab = {}
    for r, b in per_request(d):
        i, m = r["id"], SEL.search(b)
        if r.get("kind", "probe") != "probe" or i not in spans or i not in set(ids):
            continue
        band = "beyond" if tasks[i]["distance_tok"] > OLD else "within"
        if not m:
            cls = "fits" if "mode=4: skip, fits" in b else "no reselect"
        else:
            tail, r4 = int(m[2]), int(m[3])
            picked = {int(x) for x in m[4].split(",") if x}
            fwc = (r4 - 1) // CT - tail
            fa, fb = spans[i]
            held = [c >= fwc or c // unit in picked for c in range(fa // CT, fb // CT + 1)]
            cls = "in window" if fb // CT >= fwc else "held" if all(held) else "part" if any(held) else "missed"
        n, c = tab.get(f"{band} {cls}", (0, 0))
        tab[f"{band} {cls}"] = (n + 1, c + v[a][i])
    return dict(sorted(tab.items()))


mech = {a: mechanism(a) for a in ("v2", "ref")}
print("mechanism (class: probes, correct):", json.dumps(mech, indent=1))

valid = (all(struct.values()) and len(ids) == len(retained) and not errs["v2"] and not errs["ref"]
         and not disk["v2"] and not disk["ref"] and speed["v2"]["pt"] == speed["ref"]["pt"])
guard = diffs["within"]["lo"] >= -0.020 and diffs["within_excl27"]["lo"] >= -0.020
win = diffs["beyond"]["lo"] > 0 and guard and diffs["overall"]["mean"] > 0
verdict = ("void: a validity check failed" if not valid else "WIN" if win else "HARM" if not guard else "NULL")
cost_ok = win and speed["v2"]["scorer_ms"] is not None and speed["v2"]["scorer_ms"] <= 200
print(json.dumps({"final": final, "valid": valid, "guard": guard, "verdict": verdict,
                  "cost_line": "shippable as default" if cost_ok else "not shippable as default (scorer cost or no WIN)"}, indent=1))
json.dump({"final": final, "counts": counts, "diffs": diffs, "speed": speed, "mechanism": mech, "struct": struct,
           "verdict": verdict, "cost_ok": cost_ok}, open(os.path.join(E, "report_e2e.json" if final else "report_e2e.provisional.json"), "w"),
          indent=1, default=str)

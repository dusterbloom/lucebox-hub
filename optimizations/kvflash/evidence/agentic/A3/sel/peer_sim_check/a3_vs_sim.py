"""A3 stage 1 measured rows against the offline A3 simulator (A3_sim/a3_sim.py), probe by probe, W=2.
    python3 -I a3_vs_sim.py <worktree> [W]
Sim class: all (k >= candidates), ex (source in P/W/tail), fetch (a_last rank <= k), miss; predicted answer = sim 'lo'.
Measured class: from the route row's src_status / source_fetch_coverage; measured answer = probes.score_exact."""
import collections
import json
import os
import sys

W_TREE = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
EV = os.path.join(W_TREE, "optimizations/kvflash/evidence/agentic")
sys.path.insert(0, os.path.join(W_TREE, "optimizations/kvflash/agentic_eval"))
sys.path.insert(0, os.path.join(EV, "A3_sim"))
import probes  # noqa: E402
from a3_sim import BLK, MAX_NEW, plan  # noqa: E402


def rows(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
e2, e3 = rows(os.path.join(EV, "E3E2/e2.jsonl")), rows(os.path.join(EV, "E3E2/E3.jsonl"))
ship = rows(os.path.join(EV, "A1/pilot/SHIP.jsonl"))
ans, route = rows(os.path.join(EV, f"A3/sel/A3_W{W}.jsonl")), rows(os.path.join(EV, f"A3/sel/A3_W{W}_route.jsonl"))


def ok(r, i, key="content"):
    return int(r is not None and "error" not in r and probes.score_exact(r.get(key) or r.get("text") or "", tasks[i]["answer"]))


ids = [i for i in e3 if i in e2 and e2[i]["located"]]
q = {}
for i in ids:
    k = (e2[i]["traj"], e2[i]["ctx_len"])
    q[k] = max(q.get(k, 0), e3[i]["prompt_tokens"] - e3[i]["ctx_len"])
conf = collections.Counter()
lines = []
for i in ids:
    r = e2[i]
    nb, tail = r["n_blocks"], r["ctx_len"] - BLK * r["n_blocks"]
    ex, cand, k = plan(nb, tail, q[(r["traj"], r["ctx_len"])] + MAX_NEW, W, None)
    src = set(r["src_blocks"])
    if k >= cand:
        sc, sv = "all", 1
    elif not src or src <= ex:
        sc, sv = "ex", ok(e3[i], i)
    elif r["ranks"]["a_last"] <= k:
        sc, sv = "fetch", ok(e3[i], i)
    else:
        sc, sv = "miss", ok(ship.get(i), i)
    ro, a = route.get(i), ans.get(i)
    if ro is None or a is None:
        conf[(sc, "absent", sv, None)] += 1
        continue
    mc = {"exact_resident": "ex", "candidate": {"all": "fetch", "partial": "partial"}.get(ro["source_fetch_coverage"], "miss")}.get(ro["src_status"], ro["src_status"])
    mv = ok(a, i, "text")
    conf[(sc, mc, sv, mv)] += 1
    if (sc, sv) != (mc if mc != "partial" else "fetch", mv) or sc != mc:
        lines.append(f"{tasks[i]['distance_tok']:>6} {i[-22:]} sim {sc:<5} k {k:>2} rank {r['ranks']['a_last']!s:>3} pred {sv} | "
                     f"meas {mc:<7} k {a['k']:>2} rank {ro.get('rank_a_last')!s:>4} got {mv}  e3 {ok(e3[i], i)} ship {ok(ship.get(i), i)}")
print(f"W={W}: {len(ids)} probes; sim predicted {sum(v for (_, _, v, _), n in conf.items() for _ in range(n))}, "
      f"measured {sum((m or 0) for (_, _, _, m), n in conf.items() for _ in range(n))}")
print("sim class -> measured class (pred, got): count")
for (sc, mc, sv, mv), n in sorted(conf.items(), key=str):
    print(f"  {sc:<5} -> {mc:<8} pred {sv} got {mv}: {n}")
print("disagreements (class or answer):")
print("\n".join(sorted(lines)))

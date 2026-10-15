import sys, json
W = "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/"
sys.path.insert(0, W + "evidence/agentic/A3_sim"); sys.path.insert(0, W + "agentic_eval")
import a3_sim as S, probes
d = W + "evidence/agentic/E3E2"
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
e2, e3, ship = S.rows(d + "/e2.jsonl"), S.rows(d + "/E3.jsonl"), S.rows(W + "evidence/agentic/A1/pilot/SHIP.jsonl")
ok = lambda r: int(r is not None and probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]))
q = {}
for i, r in e2.items():
    q[(r["traj"], r["ctx_len"])] = max(q.get((r["traj"], r["ctx_len"]), 0), e3[i]["prompt_tokens"] - e3[i]["ctx_len"])
for idx in (True, False):
    c = {"all_exact": [0, 0], "src_in_P_W_tail": [0, 0], "fetched": [0, 0], "routing_miss": [0, 0]}
    for i, r in e2.items():
        nb = r["n_blocks"]
        ex, cand, k = S.plan(nb, r["ctx_len"] - 640 * nb, q[(r["traj"], r["ctx_len"])] + 64, 2, None, idx)
        src = set(r["src_blocks"])
        if k >= cand:
            key, v = "all_exact", 1
        elif not src or src <= ex:
            key, v = "src_in_P_W_tail", ok(e3[i])
        elif r["ranks"]["a_last"] <= k:
            key, v = "fetched", ok(e3[i])
        else:
            key, v = "routing_miss", ok(ship.get(i))
        c[key][0] += v
        c[key][1] += 1
    print("index" if idx else "no index", {k: f"{a}/{n}" for k, (a, n) in c.items()})

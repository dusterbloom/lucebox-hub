"""Estimate (not a measurement): recent exact window + k routed 640-token blocks, no compaction, at the 16K-row budget.
Per probe: distance <= window -> this run's OFF answer (LRU, everything recent exact); else recalled if the router's rank of the source
block over ALL sealed blocks is <= k (pessimistic: rank among blocks older than the window can only be better) -> correct with the
engine-measured rate for an old fact whose source is resident (P_RECALL); else wrong (0% measured when nothing is resident).
    python3 -I rlru_est.py <worktree>"""
import json
import os
import sys

W_TREE = sys.argv[1]
R = "/mnt/kimi-k3/split-work/prefill_reselect_run"
sys.path.insert(0, os.path.join(R, "agentic_eval"))
import probes  # noqa: E402

BLK, CAP, RESERVE, P_RECALL = 640, 16384, 400, 0.81
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
off = {r["id"]: probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"])
       for r in map(json.loads, open(os.path.join(R, "off/QK16KND.jsonl"))) if r.get("kind") == "probe"}
e9 = json.load(open(os.path.join(W_TREE, "optimizations/kvflash/evidence/agentic/E3E2/e_report.json")))["E9_exact_mass_router"]["per_probe"]
pp = [p for p in e9 if p["id"] in off]
print(f"E9 probes {len(e9)}, in this run's 951: {len(pp)}; OFF correct on them: {sum(off[p['id']] for p in pp)} "
      f"(beyond 16K: {sum(off[p['id']] for p in pp if p['dist'] > 16384)}/{sum(p['dist'] > 16384 for p in pp)})")
print(f"{'k':>2} {'window':>7} | " + " | ".join(f"{r:>18}" for r in ("a_last", "x_last", "x_max")))
for k in (0, 1, 2, 4, 6, 8):
    win = CAP - k * BLK - RESERVE
    cells = []
    for router in ("a_last", "x_last", "x_max"):
        exp, old = 0.0, 0.0
        for p in pp:
            if p["dist"] <= win:
                v = off[p["id"]]
            else:
                v = P_RECALL if p[router] is not None and p[router] <= k else 0.0
                old += v if p["dist"] > 16384 else 0
            exp += v
        cells.append(f"{exp:6.1f} (old {old:4.1f})")
    print(f"{k:>2} {win:>7} | " + " | ".join(f"{c:>18}" for c in cells))

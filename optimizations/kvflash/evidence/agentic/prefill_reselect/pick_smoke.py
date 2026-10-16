"""Preflight smoke ids: 5 retained probes of one (traj, cut), 3 beyond 16K and 2 within, ON arm only, rows not counted.
    python3 -I pick_smoke.py   -> smoke_ids.json"""
import collections
import json

D = "/mnt/kimi-k3/split-work/prefill_reselect_run"
ret = set(json.load(open(D + "/retained.json")))
g = collections.defaultdict(list)
for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl")):
    if t["id"] in ret and t["kind"] == "probe":
        g[(t["traj"], t["cut_index"])].append(t)
best = max(g.values(), key=lambda v: (min(3, sum(t["distance_tok"] > 16384 for t in v)), min(2, sum(t["distance_tok"] <= 16384 for t in v))))
pick = [t for t in best if t["distance_tok"] > 16384][:3] + [t for t in best if t["distance_tok"] <= 16384][:2]
print(best[0]["traj"], best[0]["cut_index"], [(t["id"], t["distance_tok"]) for t in pick])
json.dump([t["id"] for t in pick], open(D + "/smoke_ids.json", "w"))

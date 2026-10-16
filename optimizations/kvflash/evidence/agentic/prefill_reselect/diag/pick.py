"""Diagnostic subset: whole (traj, cut) groups with the most within-16K losses in the prefill-reselect run (OFF ok, ON wrong),
plus the group with the most old-fact gains, until >= 60 probes.   python3 -I pick.py  -> ids.json"""
import collections
import json
import os
import sys

R = "/mnt/kimi-k3/split-work/prefill_reselect_run"
sys.path.insert(0, os.path.join(R, "agentic_eval"))
import probes  # noqa: E402

tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
ok = {a: {r["id"]: probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"])
          for r in map(json.loads, open(os.path.join(R, a, "QK16KND.jsonl"))) if r.get("kind") == "probe"} for a in ("off", "on")}
groups = collections.defaultdict(list)
for i in ok["off"]:
    groups[(tasks[i]["traj"], tasks[i]["cut_index"])].append(i)
lost = lambda g: sum(tasks[i]["distance_tok"] <= 16384 and ok["off"][i] and not ok["on"][i] for i in groups[g])
gain = lambda g: sum(tasks[i]["distance_tok"] > 16384 and ok["on"][i] and not ok["off"][i] for i in groups[g])
pick = [max(groups, key=gain)]
for g in sorted(groups, key=lambda g: -lost(g)):
    if sum(len(groups[p]) for p in pick) >= 60:
        break
    if g not in pick:
        pick.append(g)
ids = [i for g in pick for i in groups[g]]
for g in pick:
    print(g[0][-12:], g[1], "n", len(groups[g]), "within-lost", lost(g), "old-gained", gain(g))
print(len(ids), "probes")
json.dump(ids, open(os.path.dirname(os.path.abspath(__file__)) + "/ids.json", "w"))

import json, sys
sys.path.insert(0, "/mnt/kimi-k3/split-work/rlru4/agentic_eval")
sys.path.insert(0, "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/lucebox/a1")
import probes
from a1_core import boot_ci
D = "/mnt/kimi-k3/split-work/rlru4k"
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
ids = json.load(open("/mnt/kimi-k3/split-work/rlru4/retained.json"))
v = {}
for a in ("off", "on2", "on6"):
    r = {x["id"]: x for x in map(json.loads, open(f"{D}/{a}/QK16KND.jsonl")) if x.get("kind") == "probe"}
    v[a] = {i: int(probes.score_exact(r[i].get("content") or "", tasks[i]["answer"])) for i in ids}
W = [i for i in ids if tasks[i]["distance_tok"] <= 16384]
B = [i for i in ids if tasks[i]["distance_tok"] > 16384]
f = lambda d: f"{100*d['mean']:+.1f} [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}]"
for x, y, S, n in (("on6", "on2", ids, "all"), ("on6", "off", W, "within"), ("on2", "off", W, "within"), ("on6", "off", B, "beyond"), ("on6", "on2", W, "within")):
    print(f"{x}-{y} {n}: {f(boot_ci([v[x][i] - v[y][i] for i in S], [tasks[i]['traj'] for i in S]))}")

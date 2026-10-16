"""Interim read of v2 OFF and ON2 (no verdict until all arms end). Counts, the ON2 guard read twice, and ON2 v2 vs prs7 ON2."""
import json, sys
sys.path.insert(0, "/mnt/kimi-k3/split-work/rlru4/agentic_eval")
sys.path.insert(0, "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/lucebox/a1")
import probes
from a1_core import boot_ci
D = "/mnt/kimi-k3/split-work/rlru4k"
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
ids = json.load(open("/mnt/kimi-k3/split-work/rlru4/retained.json"))
d27 = set(json.load(open(f"{D}/diag27.json")))
v = {}
for a, p in (("off", "v2/off"), ("on2", "v2/on2"), ("on2_prs7", "on2"), ("on6", "v2/on6"), ("on6_prs7", "on6")):
    r = {x["id"]: x for x in map(json.loads, open(f"{D}/{p}/QK16KND.jsonl")) if x.get("kind") == "probe"}
    v[a] = {i: int("error" not in r[i] and probes.score_exact(r[i].get("content") or "", tasks[i]["answer"])) for i in ids}
W = [i for i in ids if tasks[i]["distance_tok"] <= 16384]
WX = [i for i in W if i not in d27]
B = [i for i in ids if tasks[i]["distance_tok"] > 16384]
for a in v:
    print(f"{a:9} overall {sum(v[a][i] for i in ids)}/{len(ids)}  within {sum(v[a][i] for i in W)}/{len(W)}  "
          f"within excl27 {sum(v[a][i] for i in WX)}/{len(WX)}  beyond {sum(v[a][i] for i in B)}/{len(B)}")
f = lambda d: f"{100*d['mean']:+.1f} [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}]"
for x, y, S, n in (("on2", "off", ids, "all"), ("on2", "off", W, "within"), ("on2", "off", WX, "within excl27"),
                   ("on2", "off", B, "beyond"), ("on2", "on2_prs7", ids, "all"),
                   ("on6", "off", ids, "all"), ("on6", "off", W, "within"), ("on6", "off", WX, "within excl27"),
                   ("on6", "off", B, "beyond"), ("on6", "on2", ids, "all"), ("on6", "on6_prs7", ids, "all")):
    print(f"{x}-{y} {n}: {f(boot_ci([v[x][i] - v[y][i] for i in S], [tasks[i]['traj'] for i in S]))}")

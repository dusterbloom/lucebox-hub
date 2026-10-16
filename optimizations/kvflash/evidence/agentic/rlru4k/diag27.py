"""Freeze the in-sample diagnosis set for PREREG_v2: within-16K probes whose source lay out of ON6's window (prs7, rlru4k/on6)
and that the engine did not recall. These are the 27 that motivated SPEC_recent_pool.md. Writes diag27.json.
    python3 -I diag27.py <worktree>"""
import json, os, re, sys
W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
CT, CPB, OLD = 64, 10, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
spans = {r["id"]: r["span"]["fact"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
on = [json.loads(l) for l in open(os.path.join(D, "on6", "QK16KND.jsonl"))]
blocks = open(os.path.join(D, "on6", "QK16KND.server.log")).read().split("] chat START ")[1:]
assert len(blocks) == len(on)
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")
out = []
for r, b in zip(on, blocks):
    i, m = r["id"], SEL.search(b)
    if r.get("kind") != "probe" or i not in spans or not m or tasks[i]["distance_tok"] > OLD:
        continue
    tail, r4, picked = int(m[2]), int(m[3]), {int(x) for x in m[4].split(",") if x}
    fwc = (r4 - 1) // CT - tail
    fa, fb = spans[i]
    if fb // CT < fwc and not ({c // CPB for c in range(fa // CT, fb // CT + 1)} & picked):
        out.append(i)
assert len(out) == 27, len(out)
json.dump(sorted(out), open(os.path.join(D, "diag27.json"), "w"), indent=0)
print(len(out), "ids written")

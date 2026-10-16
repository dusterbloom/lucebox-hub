"""Post hoc (listed in PREREG_v2 as reported, not computed by analyze_v2.py): the stratum the window gives up, as in
../analyze4k.py. Per ON arm, within-16K probes whose fact lies outside the arm's own logged window, with OFF, ON2 and arm answers.
    python3 -I stratum_v2.py <worktree>"""
import json
import os
import re
import sys

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "..", "rlru4", "agentic_eval"))
import probes  # noqa: E402

CT, OLD = 64, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
span = {r["id"]: r["span"]["fact"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) ")
rows = {a: [json.loads(l) for l in open(os.path.join(D, a, "QK16KND.jsonl"))] for a in ("off", "on2", "on6", "on10")}
v = {a: {r["id"]: int("error" not in r and probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]))
         for r in rs if r.get("kind") == "probe"} for a, rs in rows.items()}
out = {}
for a in ("on2", "on6", "on10"):
    blocks = open(os.path.join(D, a, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(blocks) == len(rows[a])
    S = []
    for r, b in zip(rows[a], blocks):
        i, m = r["id"], SEL.search(b)
        if r.get("kind") == "probe" and m and i in span and tasks[i]["distance_tok"] <= OLD:
            if span[i][1] // CT < (int(m[3]) - 1) // CT - int(m[2]):
                S.append(i)
    out[a] = {"n": len(S), "off": sum(v["off"][i] for i in S), "on2": sum(v["on2"][i] for i in S), a: sum(v[a][i] for i in S)}
print("stratum (within 16K, source outside the arm's window), v2 on prs8:", json.dumps(out))

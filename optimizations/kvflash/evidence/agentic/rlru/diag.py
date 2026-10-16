"""Post-hoc (not preregistered): why R-LRU missed its prediction. Per probe, from the ON arm's own reselect log (picked blocks, tail,
R), classify the fact as in window / source block recalled / missed, and compare ON and OFF answers per class; also how often the
engine's picks contain the source block the rank dump (OFF state, mass_end) put in its top 6.
    python3 -I diag.py <worktree>"""
import json
import os
import re
import sys
from collections import Counter

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

CT, CPB, OLD = 64, 10, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
msgs = {r["id"]: r["span"]["msg"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
spans = {r["id"]: r["span"]["fact"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
rows = {a: [json.loads(l) for l in open(os.path.join(D, a, "QK16KND.jsonl"))] for a in ("off", "on", "dump")}
sel = [(int(t), int(r), [int(b) for b in p.split(",") if b]) for t, r, p in
       re.findall(r"mode=3 scorer=exact k=6 tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]", open(os.path.join(D, "on", "QK16KND.server.log")).read())]
dump = [json.loads(l) for l in open(os.path.join(D, "dump", "exact_dump", "exact_dump.jsonl"))]
assert len(sel) == len(rows["on"]) == len(dump), (len(sel), len(rows["on"]), len(dump))
off = {r["id"]: r for r in rows["off"]}


def ok(r, i):
    return int(probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


def top(scores, nb, k):
    return [b for _, b in sorted(((sum(scores[b * CPB:(b + 1) * CPB]), b) for b in range(1, nb)), key=lambda x: (-x[0], -x[1]))[:k]]


cells = Counter()
agree = Counter()
for r, (tail, R, picked), d in zip(rows["on"], sel, dump):
    i = r["id"]
    if r.get("kind") != "probe" or i not in spans:
        continue
    assert d["prompt_len"] == r["prompt_tokens"]
    cur = (R - 1) // CT                     # pager cur_chunk after truncate(R)
    fwc = cur - tail                        # first window chunk
    fa, fb = spans[i]
    src = sorted({c // CPB for c in range(fa // CT, fb // CT + 1)})
    if fb // CT >= fwc:
        cls = "in window"
    elif any(b in picked for b in src):
        ma, mb = msgs[i]
        whole = all(c >= fwc or c // CPB in picked for c in range(ma // CT, mb // CT + 1))
        cls = "recalled" if whole else "rec part"
    else:
        cls = "missed"
    band = "beyond" if tasks[i]["distance_tok"] > OLD else "within"
    cells[(band, cls, "n")] += 1
    cells[(band, cls, "on")] += ok(r, i)
    cells[(band, cls, "off")] += ok(off[i], i)
    if cls != "in window":
        dtop = top(d["mass_end"], fwc // CPB, 6)
        agree[(band, "dump top6 has src")] += any(b in dtop for b in src)
        agree[(band, "engine picked src")] += cls != "missed"
        agree[(band, "out of window")] += 1
print(f"{'band':7} {'class':10} {'n':>4} {'OFF':>4} {'ON':>4}  ON rate")
for band in ("within", "beyond"):
    for cls in ("in window", "recalled", "rec part", "missed"):
        n = cells[(band, cls, "n")]
        if n:
            print(f"{band:7} {cls:10} {n:>4} {cells[(band, cls, 'off')]:>4} {cells[(band, cls, 'on')]:>4}  {100 * cells[(band, cls, 'on')] / n:5.1f}%")
for band in ("within", "beyond"):
    n = agree[(band, "out of window")]
    print(f"{band}: out of window {n}; dump's top-6 (OFF state) holds the source {agree[(band, 'dump top6 has src')]}, "
          f"engine's picks (ON state) hold it {agree[(band, 'engine picked src')]}")

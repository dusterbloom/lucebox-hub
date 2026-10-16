"""Post-hoc (not preregistered; PREREG "reported separately"): ../rlru/diag.py's recall classes on a mode-4 ON log. Per probe, from
its own request block in the server log (split on "chat START", as analyze.py does): fits (skip, nothing evicted), in window, whole
source message recalled, part recalled, missed, or no reselect. ON vs OFF answers per class; and how often the engine's picks hold
the source block that the rank dump (OFF state, mass_end, ../rlru/dump) put in its top k.
    python3 -I diag4.py <worktree> [<on dir> <off dir>]   (default on/ and off/; the smoke dirs work too)"""
import json
import os
import re
import sys
from collections import Counter

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
ON, OFF_DIR = (sys.argv[2:4] if len(sys.argv) > 3 else (os.path.join(D, "on"), os.path.join(D, "off")))
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

CT, CPB, OLD = 64, 10, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
pp = json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))
msgs, spans = {r["id"]: r["span"]["msg"] for r in pp}, {r["id"]: r["span"]["fact"] for r in pp}
on = [json.loads(l) for l in open(os.path.join(ON, "QK16KND.jsonl"))]
off = {r["id"]: r for r in map(json.loads, open(os.path.join(OFF_DIR, "QK16KND.jsonl")))}
blocks = open(os.path.join(ON, "QK16KND.server.log")).read().split("] chat START ")[1:]
assert len(blocks) == len(on), (len(blocks), len(on))
R = os.path.join(D, "..", "rlru", "dump")
dump = {r["id"]: d for r, d in zip(map(json.loads, open(os.path.join(R, "QK16KND.jsonl"))), map(json.loads, open(os.path.join(R, "exact_dump", "exact_dump.jsonl"))))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")


def ok(r, i):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


def top(scores, nb, k):
    return [b for _, b in sorted(((sum(scores[b * CPB:(b + 1) * CPB]), b) for b in range(1, nb)), key=lambda x: (-x[0], -x[1]))[:k]]


cells, agree = Counter(), Counter()
for r, b in zip(on, blocks):
    i = r["id"]
    if r.get("kind") != "probe" or i not in spans:
        continue
    m = SEL.search(b)
    fa, fb = spans[i]
    src = sorted({c // CPB for c in range(fa // CT, fb // CT + 1)})
    if "mode=4: skip, fits" in b:
        cls = "fits"
    elif not m:
        cls = "no reselect"
    else:
        k, tail, r4, picked = int(m[1]), int(m[2]), int(m[3]), [int(x) for x in m[4].split(",") if x]
        fwc = (r4 - 1) // CT - tail         # first window chunk: protected are c > cur - 1 - tail, cur = pager cur_chunk after truncate(R4)
        if fb // CT >= fwc:
            cls = "in window"
        elif any(x in picked for x in src):
            ma, mb = msgs[i]
            cls = "recalled" if all(c >= fwc or c // CPB in picked for c in range(ma // CT, mb // CT + 1)) else "rec part"
        else:
            cls = "missed"
        if cls != "in window" and i in dump:
            agree[(tasks[i]["distance_tok"] > OLD, "dump")] += any(x in top(dump[i]["mass_end"], fwc // CPB, k) for x in src)
            agree[(tasks[i]["distance_tok"] > OLD, "engine")] += cls != "missed"
            agree[(tasks[i]["distance_tok"] > OLD, "n")] += 1
    band = "beyond" if tasks[i]["distance_tok"] > OLD else "within"
    cells[(band, cls, "n")] += 1
    cells[(band, cls, "on")] += ok(r, i)
    cells[(band, cls, "off")] += ok(off.get(i), i)
print(f"{'band':7} {'class':12} {'n':>4} {'OFF':>4} {'ON':>4}  ON rate")
for band in ("within", "beyond"):
    for cls in ("fits", "in window", "recalled", "rec part", "missed", "no reselect"):
        n = cells[(band, cls, "n")]
        if n:
            print(f"{band:7} {cls:12} {n:>4} {cells[(band, cls, 'off')]:>4} {cells[(band, cls, 'on')]:>4}  {100 * cells[(band, cls, 'on')] / n:5.1f}%")
for far in (False, True):
    print(f"{'beyond' if far else 'within'}: out of window {agree[(far, 'n')]}; dump's top-k (OFF state) holds the source {agree[(far, 'dump')]}, "
          f"engine's picks (ON state) hold it {agree[(far, 'engine')]}")

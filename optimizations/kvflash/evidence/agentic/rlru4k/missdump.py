"""Per ON arm: for out-of-window probes the engine missed, does the dump's top k (OFF state, mass_end) hold the source? And the reverse.
    python3 -I missdump.py <worktree> <on dir> <off dir>"""
import json, os, re, sys
from collections import Counter
W, ON, OFF = sys.argv[1:4]
sys.path.insert(0, "/mnt/kimi-k3/split-work/rlru4/agentic_eval")
import probes
CT, CPB, OLD = 64, 10, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
pp = json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))
spans = {r["id"]: r["span"]["fact"] for r in pp}
on = [json.loads(l) for l in open(os.path.join(ON, "QK16KND.jsonl"))]
off = {r["id"]: r for r in map(json.loads, open(os.path.join(OFF, "QK16KND.jsonl")))}
blocks = open(os.path.join(ON, "QK16KND.server.log")).read().split("] chat START ")[1:]
R = "/mnt/kimi-k3/split-work/rlru/dump"
dump = {r["id"]: d for r, d in zip(map(json.loads, open(os.path.join(R, "QK16KND.jsonl"))), map(json.loads, open(os.path.join(R, "exact_dump", "exact_dump.jsonl"))))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")
ok = lambda r, i: int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))
def top(s, nb, k):
    return [b for _, b in sorted(((sum(s[b * CPB:(b + 1) * CPB]), b) for b in range(1, nb)), key=lambda x: (-x[0], -x[1]))[:k]]
c = Counter()
for r, b in zip(on, blocks):
    i = r["id"]
    m = SEL.search(b)
    if r.get("kind") != "probe" or i not in spans or i not in dump or not m:
        continue
    k, tail, r4, picked = int(m[1]), int(m[2]), int(m[3]), [int(x) for x in m[4].split(",") if x]
    fwc = (r4 - 1) // CT - tail
    fa, fb = spans[i]
    if fb // CT >= fwc:
        continue
    src = {x // CPB for x in range(fa // CT, fb // CT + 1)}
    band = "beyond" if tasks[i]["distance_tok"] > OLD else "within"
    eng, dmp = bool(src & set(picked)), bool(src & set(top(dump[i]["mass_end"], fwc // CPB, k)))
    key = (band, f"engine {'hit' if eng else 'miss'}", f"dump {'hit' if dmp else 'miss'}")
    c[key + ("n",)] += 1
    c[key + ("off",)] += ok(off.get(i), i)
    c[key + ("on",)] += ok(r, i)
for key in sorted({k[:3] for k in c}):
    print(*key, "n", c[key + ("n",)], "OFF", c[key + ("off",)], "ON", c[key + ("on",)], sep="  ")

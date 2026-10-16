"""PREREG_v2 predictions, computed from prs7 rows only (rlru4k ../off, ../on<k>) before any v2 row.
Assumption: with the recency scoring pool, the engine's picks equal the dump's top k (OFF state, mass_end) on every
out-of-window probe. Per band (within / beyond 16K) and arm:
  engine miss, dump hit -> becomes a hit, expected correct = n * (ON rate on prs7's engine-hit, dump-hit probes);
  engine hit, dump miss -> becomes a miss, expected correct = n * (ON rate on prs7's engine-miss, dump-miss probes);
  everything else unchanged. Prediction = prs7 count + sum of (expected - prs7 correct) over the moved cells.
    python3 -I predict_v2.py <worktree> <k>..."""
import json
import os
import re
import sys

W, KS = sys.argv[1], sys.argv[2:]
D = os.path.dirname(os.path.abspath(__file__))
K4 = os.path.join(D, "..")
sys.path.insert(0, os.path.join(K4, "..", "rlru4", "agentic_eval"))
import probes  # noqa: E402

CT, CPB, OLD = 64, 10, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
retained = set(json.load(open(os.path.join(K4, "..", "rlru4", "retained.json"))))
span = {r["id"]: r["span"]["fact"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
R = os.path.join(K4, "..", "rlru", "dump")
dump = {r["id"]: x for r, x in zip(map(json.loads, open(os.path.join(R, "QK16KND.jsonl"))), map(json.loads, open(os.path.join(R, "exact_dump", "exact_dump.jsonl"))))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")
ok = lambda r, i: int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))

for k in KS:
    d = os.path.join(K4, f"on{k}")
    rows = [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]
    blocks = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(rows) == len(blocks)
    base = {"within": 0, "beyond": 0}
    cell = {}
    for r, b in zip(rows, blocks):
        i = r["id"]
        if r.get("kind") != "probe" or i not in retained:
            continue
        band = "beyond" if tasks[i]["distance_tok"] > OLD else "within"
        c = ok(r, i)
        base[band] += c
        m = SEL.search(b)
        if i not in span or i not in dump or not m:
            continue
        tail, r4, picked = int(m[2]), int(m[3]), {int(x) for x in m[4].split(",") if x}
        fwc = (r4 - 1) // CT - tail
        fa, fb = span[i]
        if fb // CT >= fwc:
            continue
        src = {x // CPB for x in range(fa // CT, fb // CT + 1)}
        s = dump[i]["mass_end"]
        top = [x for _, x in sorted(((sum(s[x * CPB:(x + 1) * CPB]), x) for x in range(1, fwc // CPB)), key=lambda t: (-t[0], -t[1]))[:int(k)]]
        key = (band, bool(src & picked), bool(src & set(top)))
        n, cc = cell.get(key, (0, 0))
        cell[key] = (n + 1, cc + c)
    rate = lambda band, e, dmp: cell.get((band, e, dmp), (0, 0))[1] / max(1, cell.get((band, e, dmp), (0, 0))[0])
    pred = {}
    for band in ("within", "beyond"):
        n1, c1 = cell.get((band, False, True), (0, 0))
        n2, c2 = cell.get((band, True, False), (0, 0))
        pred[band] = base[band] + (n1 * rate(band, True, True) - c1) + (n2 * rate(band, False, False) - c2)
    print(f"k={k}: prs7 within {base['within']} beyond {base['beyond']} total {base['within'] + base['beyond']}; "
          f"cells {dict(sorted(cell.items()))}; predicted within {pred['within']:.0f} beyond {pred['beyond']:.0f} "
          f"total {pred['within'] + pred['beyond']:.0f}")

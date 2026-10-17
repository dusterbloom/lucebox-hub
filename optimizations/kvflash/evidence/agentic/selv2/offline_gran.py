"""Exploratory (post hoc, same probes v1 was tuned on): at v1's budget of 60 chunks (6 blocks x 10 x 64 tokens), how often
does each selection unit hold the probe's source fact? Scores are the rank dump's mass_end (OFF state, last prompt token's
query, ../rlru/dump); the window start per request comes from the shipped ON6 run's server log (rlru4k/v2/on6).
    python3 -I offline_gran.py <worktree>"""
import json
import os
import re
import sys
from collections import Counter

W = sys.argv[1]
CT, CPB, OLD, BUDGET = 64, 10, 16384, 60
S = "/mnt/kimi-k3/split-work"
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
pp = json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))
spans = {r["id"]: (r["span"]["fact"], r["span"]["msg"]) for r in pp}
ON = f"{S}/rlru4k/v2/on6"
on = [json.loads(l) for l in open(f"{ON}/QK16KND.jsonl")]
logs = open(f"{ON}/QK16KND.server.log").read().split("] chat START ")[1:]
assert len(logs) == len(on)
dump = {r["id"]: d for r, d in zip(map(json.loads, open(f"{S}/rlru/dump/QK16KND.jsonl")),
                                   map(json.loads, open(f"{S}/rlru/dump/exact_dump/exact_dump.jsonl")))}
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")


def units(s, fwc, g):
    """Top units of g chunks (aligned to g, inside [CPB, fwc)) by summed mass, BUDGET // g of them."""
    u = [(sum(s[a:a + g]), a) for a in range(-(-CPB // g) * g, fwc - g + 1, g)]
    return {c for _, a in sorted(u, key=lambda x: (-x[0], -x[1]))[:BUDGET // g] for c in range(a, a + g)}


def seeds(s, fwc, left, right):
    """Greedy: chunks by mass, each seed brings [c-left, c+right]; stop at the budget."""
    out = set()
    for c in sorted(range(CPB, fwc), key=lambda c: (-s[c], -c)):
        for x in range(max(CPB, c - left), min(fwc, c + right + 1)):
            if len(out) < BUDGET:
                out.add(x)
        if len(out) >= BUDGET:
            break
    return out


def hier(s, fwc, nb):
    """Top nb blocks by summed mass, then the best BUDGET chunks inside them."""
    blocks = sorted(range(1, fwc // CPB), key=lambda b: (-sum(s[b * CPB:(b + 1) * CPB]), -b))[:nb]
    cand = [c for b in blocks for c in range(b * CPB, (b + 1) * CPB)]
    return set(sorted(cand, key=lambda c: (-s[c], -c))[:BUDGET])


VARIANTS = {f"units g={g:<2} k={BUDGET // g}": (lambda s, f, g=g: units(s, f, g)) for g in (10, 5, 4, 3, 2, 1)}
VARIANTS.update({"seed +-1": lambda s, f: seeds(s, f, 1, 1), "seed +1": lambda s, f: seeds(s, f, 0, 1),
                 "seed +-2": lambda s, f: seeds(s, f, 2, 2), "hier 16 blk": lambda s, f: hier(s, f, 16),
                 "hier 30 blk": lambda s, f: hier(s, f, 30)})

def out_of_window():
    """(id, band, fwc, fact chunks, msg chunks before the window) for probes whose fact lies before the window."""
    for r, b in zip(on, logs):
        i = r["id"]
        m = SEL.search(b)
        if r.get("kind") != "probe" or i not in spans or i not in dump or not m:
            continue
        tail, r4 = int(m[2]), int(m[3])
        fwc = (r4 - 1) // CT - tail
        (fa, fb), (ma, mb) = spans[i]
        if fb // CT < fwc:
            yield (i, "beyond" if tasks[i]["distance_tok"] > OLD else "within", fwc, set(range(fa // CT, fb // CT + 1)),
                   [x for x in range(ma // CT, mb // CT + 1) if x < fwc])


def report(score_of, variants=VARIANTS):
    """Fact coverage per variant; score_of(id) gives the per-chunk scores."""
    c = Counter()
    for i, band, fwc, fact, msg in out_of_window():
        s = score_of(i)
        c[(band, "n")] += 1
        for name, fn in variants.items():
            sel = fn(s, fwc)
            assert len(sel) <= BUDGET and all(CPB <= x < fwc for x in sel), name
            full, any_ = fact <= sel, bool(fact & sel)
            c[(band, name, "full")] += full
            c[(band, name, "part")] += any_ and not full
            c[(band, name, "msgcov")] += sum(x in sel for x in msg) / max(1, len(msg))
    for band in ("beyond", "within"):
        n = c[(band, "n")]
        print(f"{band}: {n} probes with the source outside the window; budget {BUDGET} chunks")
        print(f"  {'variant':16} {'fact held':>9} {'part':>5} {'miss':>5}  mean msg cover")
        for name in variants:
            f, p = c[(band, name, "full")], c[(band, name, "part")]
            print(f"  {name:16} {f:>9} {p:>5} {n - f - p:>5}  {c[(band, name, 'msgcov')] / n:6.2f}")


if __name__ == "__main__":
    report(lambda i: dump[i]["mass_end"])

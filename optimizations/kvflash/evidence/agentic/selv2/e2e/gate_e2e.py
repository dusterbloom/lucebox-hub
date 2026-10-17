"""PREREG_e2e checks 1, 2 (stdlib only).
    python3 -I gate_e2e.py shadow <prs8 dir> <dir>   flags off: same picks and answers as prs8, no selector v2 line
    python3 -I gate_e2e.py v2 <dir>                  every mode-4 reselect is selector v2 at 64 queries, units of 2, 30 picks"""
import json
import os
import re
import sys
from collections import Counter

SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")
V2 = re.compile(r"\[kvflash\] selector v2: (\d+) of (\d+) queries")
UNITS = "[kvflash] prefill reselect picked units of 2 chunks"
REPRE = re.compile(r"mode=4 @\d+: re-prefill \d+ tokens ok")


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]
    blocks = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(rows) == len(blocks), (d, len(rows), len(blocks))
    return rows, blocks


def clean(rows):
    return sum("error" in r for r in rows), sum(bool(r.get("timings", {}).get("disk_hit")) for r in rows)


def shadow(ref, d):
    ra, ba = load(ref)
    rb, bb = load(d)
    assert [r["id"] for r in ra] == [r["id"] for r in rb], "request sequences differ"
    picks = sum(SEL.findall(x) != SEL.findall(y) for x, y in zip(ba, bb))
    ans = sum(x.get("content") != y.get("content") for x, y in zip(ra, rb))
    n, v2 = sum(len(SEL.findall(b)) for b in bb), sum(len(V2.findall(b)) for b in bb)
    err, disk = clean(rb)
    print(f"shadow: {len(rb)} requests, {n} reselects; pick mismatches {picks}; answers differ {ans}; "
          f"selector v2 lines {v2}; errors {err}; disk_hit {disk}")
    return 0 if n and not (picks or ans or v2 or err or disk) else 1


def v2(d, nq=64, k=6, unit=2):
    rows, blocks = load(d)
    bad, hist = [], Counter()
    for r, b in zip(rows, blocks):
        sel, v = SEL.findall(b), V2.findall(b)
        why = []
        if not (len(sel) == len(v) == b.count(UNITS) == len(REPRE.findall(b))):
            why.append(f"lines sel {len(sel)} v2 {len(v)} units {b.count(UNITS)} re-prefill {len(REPRE.findall(b))}")
        for n, of in v:
            hist[int(n)] += 1
            if int(n) <= 0 or int(of) != nq:
                why.append(f"v2 {n} of {of}")
        for kk, _, _, p in sel:
            ids = [int(x) for x in p.split(",") if x]
            if int(kk) != k or len(ids) != 10 * k // unit or len(set(ids)) != len(ids) or min(ids, default=0) < 10 // unit:
                why.append(f"picks k={kk} n={len(ids)} min={min(ids, default=None)}")
        if "prefill reselect" in b and "failed" in b:
            why.append("failed")
        if why:
            bad.append((r["id"], why))
    text = "".join(blocks)
    rec, res = text.count("prefill reselect recent-pool @"), text.count("[kvflash] resident @")
    n, nsel = sum(hist.values()), len(SEL.findall("".join(blocks)))
    err, disk = clean(rows)
    print(f"v2 gate: {len(rows)} requests, {nsel} reselects, {n} with a selector v2 line; bad requests {len(bad)}; recency {rec} / resident {res}; "
          f"errors {err}; disk_hit {disk}; queries per reselect: {nq} on {hist[nq]}, fewer on {n - hist[nq]} "
          f"(min {min(hist, default=None)})")
    for i, w in bad[:3]:
        print(" ", i, w)
    return 0 if nsel and not bad and rec == res and not (err or disk) else 1


if __name__ == "__main__":
    sys.exit(shadow(*sys.argv[2:4]) if sys.argv[1] == "shadow" else v2(sys.argv[2]))

"""PREREG_v2 shadow check: prs8 with the recency flag off picks exactly the blocks prs7 picked, request by request
(the `picked=[...]` of every mode-4 reselect line, plus the k/tail/@R4 fields). Answers are reported, not gated.
    python3 -I shadow_cmp.py <prs7 dir> <prs8 dir>"""
import json
import os
import re
import sys

SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]
    blocks = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(rows) == len(blocks), (d, len(rows), len(blocks))
    return rows, [SEL.findall(b) for b in blocks]


def main(a, b):
    ra, pa = load(a)
    rb, pb = load(b)
    assert [r["id"] for r in ra] == [r["id"] for r in rb], "request sequences differ"
    bad = [(r["id"], x, y) for r, x, y in zip(ra, pa, pb) if x != y]
    ans = sum(x.get("content") != y.get("content") for x, y in zip(ra, rb))
    print(f"shadow: {len(ra)} requests, {sum(map(len, pa))} reselects in prs7, {sum(map(len, pb))} in prs8; "
          f"pick mismatches {len(bad)}; answers differ {ans}")
    for i, x, y in bad[:3]:
        print(" ", i, "prs7", x, "prs8", y)
    return 0 if sum(map(len, pa)) and not bad else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))

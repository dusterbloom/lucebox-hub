"""PREREG_v2 structural check: on every request, the resident chunk set logged at the first prefill batch
(`[kvflash] resident @<kv_offset>: n=<N> chunks=<ranges>`, LUCE_KVFLASH_LOG_RESIDENT=1) is the same in arm B as in arm A.
Requests are aligned by order (both arms send the same request sequence; asserted by id). Fails on any mismatch,
or if neither arm logged a resident line (the env did not take effect).
    python3 -I resident_cmp.py <A dir> <B dir>"""
import json
import os
import re
import sys

LINE = re.compile(r"\[kvflash\] resident @(\d+): n=(\d+) chunks=(\S*)")


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]
    blocks = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(rows) == len(blocks), (d, len(rows), len(blocks))
    return [r["id"] for r in rows], [LINE.findall(b) for b in blocks]


def main(a, b):
    ia, la = load(a)
    ib, lb = load(b)
    assert ia == ib, "request sequences differ"
    bad = [(i, x, y) for i, x, y in zip(ia, la, lb) if x != y]
    same_off = [t for t in bad if [m[0] for m in t[1]] == [m[0] for m in t[2]]]
    pooled = sum(bool(x) for x in la)
    print(f"resident sets: {len(ia)} requests, {pooled} pooled in A, {sum(bool(y) for y in lb)} in B; "
          f"mismatches {len(bad)} ({len(same_off)} at the same kv_offset, {len(bad) - len(same_off)} at a different one)")
    for i, x, y in bad[:3]:
        print(" ", i, "A", [m[:2] for m in x], "B", [m[:2] for m in y])
    return 0 if pooled and not bad else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))

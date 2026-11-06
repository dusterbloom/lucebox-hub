#!/usr/bin/env python3
"""Compare two LUCE_LAYERDIFF dumps (see src/common/layerdiff.h).

usage: layerdiff_compare.py ar.tsv spec.tsv [--max-report N]

A spec row is comparable when it is verify column 0 (the committed token), or
when every earlier column of the same call carries the token plain decode
committed at that position. For each comparable (pos, tok) row the nodes are
walked in the spec graph order and the first node whose hash differs from the
AR dump is reported, then every later differing node of that row is counted so
you can see whether the rest follows from the first.
"""
import sys
from collections import defaultdict

FIELDS = ("call slot pos tok col ncols key name op shape hash sum amax srcs").split()


def load(path):
    rows = defaultdict(list)  # (call, slot, pos, tok, col, ncols) -> [node dict]
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(FIELDS):
                p += [""] * (len(FIELDS) - len(p))
            d = dict(zip(FIELDS, p))
            k = (d["call"], int(d["slot"]), int(d["pos"]), int(d["tok"]), int(d["col"]), int(d["ncols"]))
            rows[k].append(d)
    return rows


def main():
    ar_path, sp_path = sys.argv[1], sys.argv[2]
    ar, sp = load(ar_path), load(sp_path)
    ar_by_pos = {}  # (slot, pos) -> (tok, nodes)
    for (call, slot, pos, tok, col, ncols), nodes in ar.items():
        ar_by_pos.setdefault((slot, pos), (tok, nodes))
    ok_prefix = {}  # (call, slot) -> columns still matching AR's committed tokens
    report = []
    n_rows = n_same = 0
    for key in sorted(sp, key=lambda k: (k[2], int(k[0][1:]), k[4])):
        call, slot, pos, tok, col, ncols = key
        a = ar_by_pos.get((slot, pos))
        if a is None:
            continue
        valid = ok_prefix.get((call, slot), True) if col else True
        ok_prefix[(call, slot)] = valid and a[0] == tok
        if not valid or a[0] != tok:
            continue
        n_rows += 1
        amap = {d["key"]: d for d in a[1]}
        diffs, missing, matched = [], [], 0
        for d in sp[key]:
            o = amap.get(d["key"])
            if o is None:
                missing.append(d["key"])
            elif (matched := matched + 1) and o["hash"] != d["hash"]:
                diffs.append((d, o))
        if matched < 0.5 * len(sp[key]):
            print(f"warning: pos {pos} only {matched}/{len(sp[key])} nodes matched by key")
        if not diffs:
            n_same += 1
            continue
        report.append((pos, call, col, ncols, diffs, missing))
    print(f"comparable rows {n_rows}, bit-identical rows {n_same}, differing rows {len(report)}")
    for pos, call, col, ncols, diffs, missing in report[: int(sys.argv[4]) if len(sys.argv) > 4 else 5]:
        d, o = diffs[0]
        print(f"\npos {pos} call {call} col {col}/{ncols}: {len(diffs)} differing nodes, {len(missing)} spec-only nodes")
        print(f"  first: {d['key']} op={d['op']} name={d['name']!r} out={d['shape']}")
        print(f"    inputs: {d['srcs']}")
        print(f"    spec sum={d['sum']} amax={d['amax']}   ar sum={o['sum']} amax={o['amax']}")
        later = [x[0]["key"] for x in diffs[1:6]]
        print(f"  next differing: {later}")
        if missing[:3]:
            print(f"  spec-only nodes (no AR counterpart): {missing[:3]}")


if __name__ == "__main__":
    main()

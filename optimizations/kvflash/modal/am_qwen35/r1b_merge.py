"""Merge per-job result dirs (parallel jobs write separate dirs) into one jsonl per cell.

  python3 r1b_merge.py <dst> <src_dir> [<src_dir> ...]   (src dirs hold {cell}__t{target}__{split}.jsonl)
Appends; refuses to write a (article_id, qi, cell) row twice.
"""
import glob
import json
import os
import sys

dst, srcs = sys.argv[1], sys.argv[2:]
os.makedirs(dst, exist_ok=True)
seen = {}
for s in srcs:
    for p in sorted(glob.glob(f"{s}/*__t*__*.jsonl")):
        name = os.path.basename(p)
        keys = seen.setdefault(name, set())
        with open(f"{dst}/{name}", "a") as out:
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["article_id"], r["qi"])
                assert k not in keys, (name, k)
                keys.add(k)
                out.write(line)
print({n: len(k) for n, k in sorted(seen.items())})

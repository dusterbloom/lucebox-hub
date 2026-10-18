"""Probe answers of two arms equal on content + tool calls with the random tool-call ids dropped.
    python3 -I cmp_noid.py <a/QK16KND.jsonl> <b/QK16KND.jsonl>"""
import json
import sys

a, b = ({r["id"]: r for r in map(json.loads, open(p)) if r.get("kind") == "probe"} for p in sys.argv[1:3])
key = lambda r: (r.get("content"), json.dumps([{k: v for k, v in t.items() if k != "id"} for t in r.get("tool_calls") or []]))
ids = [i for i in a if i in b]
diff = [i for i in ids if key(a[i]) != key(b[i])]
print(f"{len(ids) - len(diff)}/{len(ids)} shared probes equal (tool-call ids ignored); differing: {len(diff)}")

"""Answers of two arms on their shared probe ids (content + tool calls), and the first differing id.
    python3 -I cmp_answers.py <a/QK16KND.jsonl> <b/QK16KND.jsonl>"""
import json
import sys

a, b = ({r["id"]: r for r in map(json.loads, open(p)) if r.get("kind") == "probe"} for p in sys.argv[1:3])
ids = [i for i in a if i in b]
key = lambda r: (r.get("content"), r.get("tool_calls"))
diff = [i for i in ids if key(a[i]) != key(b[i])]
print(f"{len(ids) - len(diff)}/{len(ids)} shared probes equal; first differing: {diff[:3]}")

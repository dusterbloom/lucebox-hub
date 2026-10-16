"""Smoke (d) gate: f9b0's primer has the same cached prefix and pt in orders A and B (else the test is void), and every f9b0
probe answers identically in A and B.   python3 -I cmp_d.py <A QK16KND.jsonl> <B QK16KND.jsonl>"""
import json
import sys

F = "chatcmpl-f9b0f6072abe7d7d31ee12992f5fd055"
a, b = ({r["id"]: r for r in map(json.loads, open(p))} for p in sys.argv[1:3])
f = sorted(i for i in a if i.startswith(F))
key = lambda r: (r["timings"].get("cached_prefix_tokens"), r["timings"].get("prefilled_tokens"))
same_prefix = all(i in b and key(a[i]) == key(b[i]) for i in f)
probes = [i for i in f if a[i].get("kind") == "probe"]
diff = [i[-8:] for i in probes if a[i].get("content") != b.get(i, {}).get("content")]
errors = [i for i in list(a) + list(b) if "error" in a.get(i, b.get(i, {}))]
print(f"smoke d: f9b0 requests {len(f)}, cached prefix and pt equal in A and B: {same_prefix}; "
      f"probes {len(probes)}, answers differ: {diff}; errors {errors}")
sys.exit(0 if same_prefix and len(probes) == 5 and not diff and not errors else 1)

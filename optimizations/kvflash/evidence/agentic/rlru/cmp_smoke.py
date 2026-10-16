"""Smoke (a) gate: mode 3 with a no-op reselect (tail 254 = the LRU set, 0 page events) must reproduce OFF character for character.
    python3 -I cmp_smoke.py <off QK16KND.jsonl> <noop QK16KND.jsonl> <noop QK16KND.server.log>"""
import json
import re
import sys

off, noop = ({r["id"]: r for r in map(json.loads, open(p))} for p in sys.argv[1:3])
log = open(sys.argv[3]).read()
events = [int(e) for e in re.findall(r"mode=3 .*picked=\[[^\]]*\]: (\d+) page events.*; rewind 1 token", log)]
reruns = log.count("rerun ok")
bad = [i for i in off if "error" in off[i] or "error" in noop.get(i, {"error": 1}) or off[i].get("content") != noop[i].get("content")]
print(f"smoke a: {len(off)} probes, {len(events)} rewinds, {reruns} reruns, page events {events}, mismatched {bad}")
sys.exit(0 if events and reruns == len(events) and not any(events) and not bad else 1)

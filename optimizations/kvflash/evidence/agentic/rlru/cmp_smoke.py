"""Smoke (a) gate: mode 3 with a no-op reselect (every chunk protected, resident wins ties) must show 0 page events and a rerun
whose logits equal the first pass's bit for bit, on every rewind. This checks the rewind inside one request, so state carried
over between requests (the score ledger steering the next live restore) cannot confound it. Also reports how many probe answers
differ from OFF: the flip rate from the 1-token final batch, which ON vs OFF also contains (the reselect hook is request-scoped since prs5).
    python3 -I cmp_smoke.py <off QK16KND.jsonl> <noop QK16KND.jsonl> <noop QK16KND.server.log>"""
import json
import re
import sys

off, noop = ({r["id"]: r for r in map(json.loads, open(p))} for p in sys.argv[1:3])
log = open(sys.argv[3]).read()
events = [int(e) for e in re.findall(r"mode=3 .*picked=\[[^\]]*\]: (\d+) page events.*; rewind 1 token", log)]
diffs = [float(d) for d in re.findall(r"rerun ok, logits max\|diff\| (\S+) vs first pass", log)]
errors = [i for i in noop if "error" in noop[i]]
probes = [i for i in off if off[i].get("kind") == "probe"]
flips = [i for i in probes if off[i].get("content") != noop.get(i, {}).get("content")]
print(f"smoke a: {len(noop)} requests, {len(events)} rewinds, page events {events}, rerun logit max|diff| {diffs}, errors {errors}")
print(f"flip rate vs OFF (1-token final batch; more than 1 of 5 must be stated before the arms): {len(flips)}/{len(probes)} probes {flips}")
sys.exit(0 if events and len(diffs) == len(events) and not any(events) and not any(diffs) and not errors else 1)

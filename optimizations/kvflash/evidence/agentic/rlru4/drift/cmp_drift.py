"""Per probe, answer equal to the full-run OFF? for the drift arms (off, on, noop), with the mode-4 status from each arm's log.
    python3 -I cmp_drift.py"""
import json
import os

O = os.path.dirname(os.path.abspath(__file__))
full = {r["id"]: r.get("content") for r in map(json.loads, open(os.path.join(O, "..", "off", "QK16KND.jsonl")))}
for arm in ("off", "on", "noop"):
    rows = [json.loads(l) for l in open(os.path.join(O, arm, "QK16KND.jsonl"))]
    blocks = open(os.path.join(O, arm, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(rows) == len(blocks)
    print(arm)
    for r, b in zip(rows, blocks):
        st = "skip" if "skip, fits" in b else "rewound" if "re-prefill" in b else "-"
        eq = "" if r.get("kind") != "probe" else ("= full OFF" if r.get("content") == full[r["id"]] else "DIFFERS from full OFF")
        print(f"  {r['id'][-34:]:34} {r.get('kind'):7} {st:7} {eq}")

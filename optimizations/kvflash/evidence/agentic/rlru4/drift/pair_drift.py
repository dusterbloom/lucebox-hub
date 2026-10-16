"""Within the drift repro (same history on every arm): which probes does on / noop answer differently from off?
    python3 -I pair_drift.py"""
import json
import os

O = os.path.dirname(os.path.abspath(__file__))
arm = {a: {r["id"]: r for r in map(json.loads, open(os.path.join(O, a, "QK16KND.jsonl")))} for a in ("off", "on", "noop")}
for a in ("on", "noop"):
    d = [i[-34:] for i, r in arm[a].items() if r.get("kind") == "probe" and r.get("content") != arm["off"][i].get("content")]
    print(f"{a} vs off (same history): {len(d)} of {sum(r.get('kind') == 'probe' for r in arm[a].values())} probes differ {d}")

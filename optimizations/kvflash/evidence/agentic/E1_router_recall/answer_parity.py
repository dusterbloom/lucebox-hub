"""E1 parity check: probe recall of the E1 run (QK16K, probes only) against B0 POOL16K, with the harness scorer.
Usage: python3 -I answer_parity.py <E1 rows.jsonl> <B0 POOL16K.jsonl> <tasks.jsonl> <out.json>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "agentic_eval")))
import probes  # noqa: E402

rows_p, b0_p, tasks_p, out_p = sys.argv[1:5]
tasks = {t["id"]: t for t in map(json.loads, open(tasks_p)) if t["kind"] == "probe"}
e1 = {r["id"]: r for r in map(json.loads, open(rows_p)) if r.get("kind") == "probe"}
b0 = {r["id"]: r for r in map(json.loads, open(b0_p)) if r.get("kind") == "probe"}
out = {}
for name, f in (("older_than_16K", lambda t: t["distance_tok"] > 16384), ("within_16K", lambda t: t["distance_tok"] <= 16384), ("all", lambda t: True)):
    ids = [i for i in e1 if i in b0 and i in tasks and f(tasks[i])]
    ok = lambda rows, i: probes.score_exact(rows[i].get("content") or "", tasks[i]["answer"])
    out[name] = {"n": len(ids), "E1_correct": sum(ok(e1, i) for i in ids), "B0_POOL16K_correct": sum(ok(b0, i) for i in ids),
                 "same_text": sum((e1[i].get("content") or "") == (b0[i].get("content") or "") for i in ids),
                 "both_correct": sum(ok(e1, i) and ok(b0, i) for i in ids)}
json.dump(out, open(out_p, "w"), indent=1)
print(json.dumps(out, indent=1))

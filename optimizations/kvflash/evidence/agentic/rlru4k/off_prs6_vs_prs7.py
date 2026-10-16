"""Post-hoc: OFF on prs6 (../rlru4/off) vs OFF on prs7 (off/): answers that changed, correct counts, speed medians.
prs7 adds the truncate layout fix and the page-out race fence; these rows cannot tell the two apart.
    python3 -I off_prs6_vs_prs7.py"""
import json
import os
import statistics as st
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "rlru4", "agentic_eval"))
import probes  # noqa: E402

tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
p6, p7 = ({r["id"]: r for r in map(json.loads, open(p)) if r.get("kind") == "probe"} for p in
          (os.path.join(D, "..", "rlru4", "off", "QK16KND.jsonl"), os.path.join(D, "off", "QK16KND.jsonl")))
ok = lambda r, i: int(probes.score_exact(r.get("content") or "", tasks[i]["answer"]))
ch = [i for i in p6 if p6[i].get("content") != p7[i].get("content")]
print(f"answers changed {len(ch)}/{len(p6)}; correct prs6 {sum(ok(p6[i], i) for i in p6)}, prs7 {sum(ok(p7[i], i) for i in p7)}; "
      f"among changed: prs6 {sum(ok(p6[i], i) for i in ch)}, prs7 {sum(ok(p7[i], i) for i in ch)}")
for k in ("prefill_ms", "decode_tokens_per_sec"):
    print(k, "median prs6", st.median(r["timings"][k] for r in p6.values()), "prs7", st.median(r["timings"][k] for r in p7.values()))
print("client_wall_s median prs6", st.median(r["client_wall_s"] for r in p6.values()), "prs7", st.median(r["client_wall_s"] for r in p7.values()))
W = [i for i in ch if ok(p7[i], i) and not ok(p6[i], i)]
L = [i for i in ch if ok(p6[i], i) and not ok(p7[i], i)]
band = lambda S, far: sum((tasks[i]["distance_tok"] > 16384) == far for i in S)
print(f"prs7 gains {len(W)} (within {band(W, False)}, beyond {band(W, True)}), losses {len(L)} (within {band(L, False)}, beyond {band(L, True)}); changed but both wrong or both right {len(ch) - len(W) - len(L)}")

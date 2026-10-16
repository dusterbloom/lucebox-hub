"""Post-hoc diagnostics (not preregistered): ON vs OFF by distance bin and probe type, and samples of within-16K losses.
    python3 -I diag.py <worktree>"""
import collections
import json
import os
import sys

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
arm = {a: {r["id"]: r for r in map(json.loads, open(os.path.join(D, a, "QK16KND.jsonl"))) if r.get("kind") == "probe"} for a in ("off", "on")}
ok = {a: {i: int(probes.score_exact(r.get("content") or "", tasks[i]["answer"])) for i, r in arm[a].items()} for a in arm}
ids = sorted(ok["off"])
BINS = [0, 1024, 2048, 4096, 8192, 12288, 16384, 10 ** 9]
by = collections.defaultdict(lambda: [0, 0, 0, 0])   # n, off, on, lost (off ok, on wrong)
for i in ids:
    d = tasks[i]["distance_tok"]
    b = next(f"{BINS[k] // 1024}-{BINS[k + 1] // 1024 if BINS[k + 1] < 10 ** 9 else 'inf'}K" for k in range(len(BINS) - 1) if BINS[k] <= d < BINS[k + 1])
    for key in (("dist", b), ("type", tasks[i]["type"]), ("cut", tasks[i]["cut_index"])):
        c = by[key]
        c[0] += 1; c[1] += ok["off"][i]; c[2] += ok["on"][i]; c[3] += ok["off"][i] and not ok["on"][i]
for key in sorted(by, key=lambda k: (k[0], str(k[1]))):
    n, f, o, lost = by[key]
    print(f"{key[0]:5} {str(key[1]):>10}  n {n:>4}  OFF {f:>4}  ON {o:>4}  lost {lost:>3}  gained {o - f + lost:>3}")
print("\nwithin-16K losses, sample:")
lost = [i for i in ids if tasks[i]["distance_tok"] <= 16384 and ok["off"][i] and not ok["on"][i]]
for i in lost[:25]:
    t = tasks[i]
    print(f"{t['distance_tok']:>6} {t['type']:<12} want {str(t['answer'])[:50]!r}\n{'':19} got  {(arm['on'][i].get('content') or '')[:80]!r}")

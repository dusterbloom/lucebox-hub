"""Mode-4 (rewind the turn) estimate per k from the exact-mass rank dump (OFF state, mass_end query = mode 4's query).
    python3 -I rank4.py <worktree>
Window = POOL - 1 - 10k - reserve - 1 chunks, reserve = ceil(pt / 64) + 1 (pt = tokens this request prefilled, ~ end - R4).
Per probe: fact chunk in window -> OFF answer; else the source message whole in window + top-k blocks -> P_WHOLE[band]; fact block
in top k but message cut -> P_PART; else 0. Rates are engine-measured with the source resident while the question was prefilled
(prefill_reselect_diag: whole message 88/93 within, 17/21 beyond; fact chunk only 16/30 within)."""
import json
import os
import sys

W = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

CT, CPB, POOL, OLD = 64, 10, 256, 16384
P_WHOLE, P_PART = {"within": 0.95, "beyond": 0.81}, 0.53
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
span = {r["id"]: r["span"] for r in json.load(open(os.path.join(W, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
reqs = [json.loads(l) for l in open(os.path.join(D, "..", "rlru", "dump", "QK16KND.jsonl"))]
dump = [json.loads(l) for l in open(os.path.join(D, "..", "rlru", "dump", "exact_dump", "exact_dump.jsonl"))]
rows = [(r, d) for r, d in zip(reqs, dump) if r.get("kind") == "probe" and r["id"] in span]
assert len(rows) == 951 and all(d["prompt_len"] == r["prompt_tokens"] for r, d in rows)


def top(scores, nb, k):
    return set(b for _, b in sorted(((sum(scores[b * CPB:(b + 1) * CPB]), b) for b in range(1, nb)), key=lambda x: (-x[0], -x[1]))[:k])


off = {"within": 0, "beyond": 0}
for r, _ in rows:
    off["beyond" if tasks[r["id"]]["distance_tok"] > OLD else "within"] += probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"])
print(f"OFF (dump arm): within {off['within']}, beyond {off['beyond']}")
print(f"{'k':>2} | {'window':>6} | {'within est':>10} {'drop':>5} | {'beyond est':>10} | {'total':>6} | whole/part/missed out of window")
for k in (2, 4, 6):
    est = {"within": 0.0, "beyond": 0.0}
    cnt = [0, 0, 0]
    wins = []
    for r, d in rows:
        i = r["id"]
        band = "beyond" if tasks[i]["distance_tok"] > OLD else "within"
        reserve = -(-r["timings"]["prefilled_tokens"] // CT) + 1
        tail = POOL - 1 - CPB * k - reserve - 1
        fwc = (d["r"] - 1) // CT - tail
        wins.append(tail)
        fa, fb = span[i]["fact"]
        ma, mb = span[i]["msg"]
        if fb // CT >= fwc:
            est[band] += probes.score_exact(r.get("content") or "", tasks[i]["answer"])
            continue
        picked = top(d["mass_end"], fwc // CPB, k)
        if any(c // CPB in picked for c in range(fa // CT, fb // CT + 1)):
            whole = all(c >= fwc or c // CPB in picked for c in range(ma // CT, mb // CT + 1))
            est[band] += P_WHOLE[band] if whole else P_PART
            cnt[0 if whole else 1] += 1
        else:
            cnt[2] += 1
    print(f"{k:>2} | {sorted(wins)[len(wins) // 2] * CT:>6} | {est['within']:>10.0f} {off['within'] - est['within']:>5.0f} | "
          f"{est['beyond']:>10.0f} | {est['within'] + est['beyond']:>6.0f} | {cnt[0]}/{cnt[1]}/{cnt[2]}")

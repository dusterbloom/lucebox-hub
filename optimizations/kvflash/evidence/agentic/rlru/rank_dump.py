"""Exact-mass router ranks on the engine's own LRU state, from the LUCE_KVFLASH_EXACT_DUMP run (no reselect; answers = OFF).
    python3 -I rank_dump.py <worktree> [dump arm dir]
Per probe, the source = the 640-token block(s) holding the fact span (E1 spans; same harness, same token positions). For a recall
budget k, the window keeps the newest 256 - 1 - 10k - 2 chunks; a probe is 'in window' if the fact chunk is in it, else 'recalled' if a
source block ranks within the top k among eligible blocks (entirely older than the window, not block 0), by the dump's mass_r
(q at R-1) or mass_end (q at end-1). Estimate = OFF answer if in window, P_RECALL if recalled, 0 otherwise (engine-measured rates)."""
import json
import os
import statistics as st
import sys

W_TREE = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
ARM = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "dump")
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

CT, CPB, POOL, P_RECALL = 64, 10, 256, 0.81
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
recs = {r["id"]: r for r in json.load(open(os.path.join(W_TREE, "optimizations/kvflash/evidence/agentic/E1_router_recall/summary.per_probe.json")))}
reqs = [json.loads(l) for l in open(os.path.join(ARM, "QK16KND.jsonl"))]
dump = [json.loads(l) for l in open(os.path.join(ARM, "exact_dump", sorted(os.listdir(os.path.join(ARM, "exact_dump")))[0]))]
assert len(dump) == len(reqs), (len(dump), len(reqs))
rows = []
for r, d in zip(reqs, dump):
    assert d["prompt_len"] == r["prompt_tokens"], (r["id"], d["prompt_len"], r["prompt_tokens"])
    if r.get("kind") == "probe" and r["id"] in recs:
        rows.append((r, d))
print(f"probes with a dump line and a span: {len(rows)}")


def block_rank(scores, src_blocks, first_window_chunk):
    """1-based rank of the best source block among eligible blocks; None if a source chunk is already in the window."""
    nb = first_window_chunk // CPB                      # blocks entirely below the window
    if any(b >= nb for b in src_blocks):
        return None
    bs = sorted(((sum(scores[b * CPB:(b + 1) * CPB]), b) for b in range(1, nb)), key=lambda x: (-x[0], -x[1]))
    order = {b: i + 1 for i, (_, b) in enumerate(bs)}
    return min(order.get(b, 10 ** 9) for b in src_blocks)


print(f"{'k':>2} {'query':>8} | {'in window':>9} {'recalled':>8} {'missed':>6} | {'est. correct':>12} {'old (207)':>9} | recall@k of out-of-window sources")
for k in (1, 2, 4, 6, 8):
    for q in ("mass_r", "mass_end"):
        inw = rec = miss = 0
        est = old = 0.0
        for r, d in rows:
            i, cur = r["id"], (d["r"] - 1) // CT        # pager cur_chunk at the reselect point R
            tail = POOL - 1 - CPB * k - 2 - 1          # engine: n_pool - sink - 10k - reserve(2 for a 4-token tail) - 1
            fwc = cur - tail
            fa, fb = recs[i]["span"]["fact"]
            ok = probes.score_exact(r.get("content") or "", tasks[i]["answer"])
            rank = block_rank(d[q], sorted({c // CPB for c in range(fa // CT, fb // CT + 1)}), fwc)
            if rank is None:
                inw += 1
                v = ok
            elif rank <= k:
                rec += 1
                v = P_RECALL
            else:
                miss += 1
                v = 0.0
            est += v
            old += v if tasks[i]["distance_tok"] > 16384 else 0
        print(f"{k:>2} {q:>8} | {inw:>9} {rec:>8} {miss:>6} | {est:>12.0f} {old:>9.0f} | {100 * rec / max(1, rec + miss):5.1f}%"
              f"  (random {100 * st.mean(min(1, k / max(1, (d['r'] - 1) // CT // CPB - 1)) for _, d in rows):4.1f}%)")

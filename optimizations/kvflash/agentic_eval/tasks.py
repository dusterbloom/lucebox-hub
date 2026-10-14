"""Build the request list (probe candidates + decision points) per trajectory. Deterministic (fixed seeds).
Usage: python3 -I tasks.py <trajectories.jsonl> <out tasks.jsonl> [n_traj] [dp_per_traj=24] [probes_per_P=12]
Order inside a trajectory is ascending by cut position so the server's prefix cache is reused."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes  # noqa: E402

PROBE_POINTS = [12 * 1024, 24 * 1024, 36 * 1024]
DP_MIN_TOK = 12 * 1024
SEED = 20261014


def build_for_trajectory(rec, dp_n=24, probes_n=12, seed=SEED):
    msgs, bnd = rec["messages"], rec["boundaries"]
    tid = rec["trajectory_id"]
    items = []

    def tool_before(j):
        return j > 0 and msgs[j - 1]["role"] == "tool"

    for P in PROBE_POINTS:
        ok = [int(j) for j, p in bnd.items() if p <= P and tool_before(int(j))]
        if not ok:
            continue
        k = max(ok)
        if bnd[str(k)] < P - 4096:  # no boundary near P (trajectory shorter than P): skip this P
            continue
        cands = probes.extract_candidates(msgs[:k], rec["tools"], bnd, bnd[str(k)])
        pick = probes.sample_probes(cands, f"{seed}:{tid}:{P}", probes_n)
        for n, c in enumerate(pick):
            items.append({"id": f"{tid}:p{P // 1024}:{n}", "traj": tid, "kind": "probe", "P": P, "cut_index": k,
                          "cut_tok": bnd[str(k)], "n_cands": len(cands), **c})
    # decision points: one random assistant boundary per equal-count bin, past DP_MIN_TOK
    dps = sorted(int(j) for j, p in bnd.items() if p > DP_MIN_TOK and tool_before(int(j)))
    rng = random.Random(f"{seed}:{tid}:dp")
    if len(dps) <= dp_n:
        pick = dps
    else:
        edges = [round(i * len(dps) / dp_n) for i in range(dp_n + 1)]
        pick = [rng.choice(dps[edges[i]:edges[i + 1]]) for i in range(dp_n)]
    for j in pick:
        gc = None
        if msgs[j].get("tool_calls") if j < len(msgs) else None:
            gc = probes.parse_response_call({"tool_calls": msgs[j]["tool_calls"]})
        items.append({"id": f"{tid}:d{j}", "traj": tid, "kind": "decision", "cut_index": j, "cut_tok": bnd[str(j)],
                      "gold_key": gc})
    items.sort(key=lambda x: (x["cut_tok"], x["kind"] != "decision", x["id"]))
    return items


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n_traj = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    dp_n = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    pn = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    n = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            if n >= n_traj:
                break
            items = build_for_trajectory(json.loads(line), dp_n, pn)
            n += 1
            kinds = {}
            for it in items:
                g.write(json.dumps(it) + "\n")
                kk = it.get("type", it["kind"])
                kinds[kk] = kinds.get(kk, 0) + 1
            print(n, items[0]["traj"] if items else "-", kinds, flush=True)


if __name__ == "__main__":
    main()

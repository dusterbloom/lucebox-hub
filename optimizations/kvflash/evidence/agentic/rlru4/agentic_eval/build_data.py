"""Select + convert + truncate trajectories. Run: python3 -I build_data.py [n_select=60]
Reads the untrusted parquet (own dir) with -I; writes derived JSONL to a separate new directory.
Sampling: fixed-seed permutation of all rows; walk it, keep rows whose WHOLE trajectory renders to >= 32K tokens
(Qwen3.6/3.8 vocab tokenizer.json), convert, truncate at the last assistant boundary <= 48K (minus a margin for the
generation prompt / server-side tojson drift)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert  # noqa: E402  (adds user site for pyarrow)
import pyarrow.parquet as pq  # noqa: E402

SRC = "/mnt/kimi-k3/datasets/swe-rebench-openhands-raw/trajectories.parquet"
OUT_DIR = "/mnt/kimi-k3/datasets/agentic-eval-prepared"
SEED = 20261014
MIN_TOK = 32 * 1024
CUT_TOK = 48 * 1024 - 512  # 48K minus margin


def main():
    n_sel = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    pf = pq.ParquetFile(SRC)
    n = pf.metadata.num_rows
    perm = list(range(n))
    random.Random(SEED).shuffle(perm)
    cand = perm[: n_sel * 3]  # more than needed (>96% are >= 32K); we stop at n_sel in permutation order
    # map global row -> (row group, local)
    starts, acc = [], 0
    for g in range(pf.metadata.num_row_groups):
        starts.append(acc)
        acc += pf.metadata.row_group(g).num_rows
    import bisect
    by_g = {}
    for r in cand:
        g = bisect.bisect_right(starts, r) - 1
        by_g.setdefault(g, []).append(r - starts[g])
    rows = {}
    for g, locs in by_g.items():
        t = pf.read_row_group(g, columns=["trajectory_id", "instance_id", "repo", "trajectory", "tools", "resolved", "exit_status"])
        for r, d in zip(locs, t.take(locs).to_pylist()):
            rows[starts[g] + r] = d
        del t
    os.makedirs(OUT_DIR, exist_ok=True)
    kept, skipped = [], {"nonjson_args": 0, "short": 0, "no_cut": 0}
    for r in cand:
        if len(kept) >= n_sel:
            break
        d = rows[r]
        msgs = convert.convert_messages(d["trajectory"])
        if msgs is None:
            skipped["nonjson_args"] += 1
            continue
        tools = convert.convert_tools(d["tools"])
        full_tok, pos = convert.boundary_positions(msgs, tools)
        if full_tok < MIN_TOK:
            skipped["short"] += 1
            continue
        ok = [j for j, p in pos.items() if p <= CUT_TOK and msgs[j - 1]["role"] == "tool"]
        if not ok:
            skipped["no_cut"] += 1
            continue
        k = max(ok)  # messages[:k] is the truncated trajectory (ends with a tool result)
        kept.append({"row": r, "trajectory_id": d["trajectory_id"], "instance_id": d["instance_id"], "repo": d["repo"],
                     "resolved": d["resolved"], "exit_status": d["exit_status"], "full_tokens": full_tok,
                     "cut_index": k, "cut_tokens": pos[k], "boundaries": {str(j): p for j, p in pos.items() if j <= k},
                     "tools": tools, "messages": msgs[:k]})
        print(len(kept), d["instance_id"], full_tok, pos[k], flush=True)
    with open(os.path.join(OUT_DIR, "trajectories.jsonl"), "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    print("selected", len(kept), "skipped", skipped, "seed", SEED)


if __name__ == "__main__":
    main()

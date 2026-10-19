"""ABBA decode A/B of the pager copy path: packed (strided 2D) vs legacy (per-head copies), same binary f619149b, 303 probes,
compaction before each block. Paired per probe: mean(A1, A2) - mean(B1, B2); median with a bootstrap CI over probes.
    python3 -I analyze_ab.py /mnt/kimi-k3/split-work/pager/ab"""
import json
import random
import statistics as st
import sys

B = ["A1_packed", "B1_legacy", "B2_legacy", "A2_packed"]
ab = sys.argv[1]
rows = {b: {r["id"]: r for r in map(json.loads, open(f"{ab}/{b}/QK16KND.jsonl")) if r.get("kind", "probe") == "probe"} for b in B}
ids = [i for i in rows[B[0]] if all(i in rows[b] and "error" not in rows[b][i] for b in B)]
t = lambda b, i: rows[b][i].get("timings") or {}
wall = lambda b, i: rows[b][i]["client_wall_s"]
dec_ids = [i for i in ids if all((rows[b][i].get("completion_tokens") or 0) >= 16 for b in B)]
d4 = lambda v: f"med {st.median(v):.2f} mean {st.mean(v):.2f} best {max(v):.2f} min {min(v):.2f}"
print(f"probes on all blocks {len(ids)}, decode subset (>= 16 tokens on every block) {len(dec_ids)}")
for b in B:
    print(f"{b:10s} decode tok/s {d4([t(b, i)['decode_tokens_per_sec'] for i in dec_ids])} | prefill ms "
          f"{d4([t(b, i)['prefill_ms'] for i in ids])} | wall s {d4([wall(b, i) for i in ids])}")
key = lambda b, i: (rows[b][i].get("content"), [c["function"] for c in rows[b][i].get("tool_calls") or []])
print("answers identical on all 4 blocks (tool-call ids aside):", sum(len({json.dumps(key(b, i)) for b in B}) == 1 for i in ids), "of", len(ids))
print("same prompt and cached tokens on all 4 blocks:", sum(len({(t(b, i).get("cached_prefix_tokens"), rows[b][i].get("prompt_tokens")) for b in B}) == 1 for i in ids))


def paired(name, S, f):
    d = [(f("A1_packed", i) + f("A2_packed", i)) / 2 - (f("B1_legacy", i) + f("B2_legacy", i)) / 2 for i in S]
    rng = random.Random(1)
    bs = sorted(st.median(rng.choices(d, k=len(d))) for _ in range(2000))
    print(f"packed - legacy {name}: median {st.median(d):+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}], mean {st.mean(d):+.3f} (n {len(d)})")


paired("decode tok/s", dec_ids, lambda b, i: t(b, i)["decode_tokens_per_sec"])
paired("prefill ms", ids, lambda b, i: t(b, i)["prefill_ms"])
paired("wall s", ids, wall)
for x, y in (("A2_packed", "A1_packed"), ("B2_legacy", "B1_legacy")):
    print(f"drift {x} - {y} decode: {st.median(t(x, i)['decode_tokens_per_sec'] - t(y, i)['decode_tokens_per_sec'] for i in dec_ids):+.3f}")

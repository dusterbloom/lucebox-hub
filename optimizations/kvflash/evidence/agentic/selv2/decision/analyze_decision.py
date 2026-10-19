"""Decision-step eval: agreement with FULL per arm, paired differences with cluster-bootstrap CIs, cost, instrumentation.
    python3 -I analyze_decision.py <arms_dir> <R>      (arms_dir/<ARM>/QK16KND.jsonl or FULL.jsonl, plus .server.log)
    python3 -I analyze_decision.py --selftest"""
import json
import random
import re
import statistics as st
import sys

ARMS = ["FULL", "ON6", "V2", "V2PU", "V2P", "OFFP"]
PAIRS = [("V2PU", "ON6"), ("V2", "ON6"), ("V2PU", "V2P"), ("V2P", "V2"), ("OFFP", "FULL")]
SEED, DRAWS = 20261019, 10000


def first_call(r):
    """(name, canonical args) of the first tool call, or ('TEXT', content) when there is none."""
    tc = r.get("tool_calls") or []
    if tc:
        f = tc[0]["function"]
        try:
            args = json.dumps(json.loads(f["arguments"]), sort_keys=True)
        except (ValueError, TypeError):
            args = str(f["arguments"])
        return (f["name"], args)
    return ("TEXT", (r.get("content") or "").strip())


def agree(a, b):
    return first_call(a) == first_call(b)


def agree_name(a, b):
    return first_call(a)[0] == first_call(b)[0]


def boot(diffs_by_traj):
    """95% CI of the mean paired difference (points), resampling trajectories."""
    trajs = list(diffs_by_traj)
    rng = random.Random(SEED)
    stats = []
    for _ in range(DRAWS):
        pick = [diffs_by_traj[rng.choice(trajs)] for _ in trajs]
        flat = [d for g in pick for d in g]
        stats.append(100.0 * sum(flat) / len(flat))
    stats.sort()
    return stats[int(0.025 * DRAWS)], stats[int(0.975 * DRAWS) - 1]


def load(arms_dir, arm):
    f = "FULL" if arm == "FULL" else "QK16KND"
    rows = [json.loads(l) for l in open(f"{arms_dir}/{arm}/{f}.jsonl")]
    log = open(f"{arms_dir}/{arm}/{f}.server.log", errors="replace").read()
    return {r["id"]: r for r in rows if r["kind"] == "decision"}, rows, log


def describe(v):
    return f"med {st.median(v):.1f} mean {st.mean(v):.1f} best {max(v):.1f} min {min(v):.1f}" if v else "n/a"


def main(arms_dir, R):
    data = {a: load(arms_dir, a) for a in ARMS}
    full = data["FULL"][0]
    ids = [i for i in full if all(i in data[a][0] and "error" not in data[a][0][i] for a in ARMS)]
    print(f"steps with a row on every arm: {len(ids)} (errors per arm: "
          + ", ".join(f"{a} {sum('error' in r for r in data[a][1])}" for a in ARMS) + ")")
    print("steps stopped by the token cap (no complete action): "
          + ", ".join(f"{a} {sum(data[a][0][i].get('finish') == 'length' for i in ids)}" for a in ARMS))
    agr = {a: {i: agree(data[a][0][i], full[i]) for i in ids} for a in ARMS}
    agn = {a: {i: agree_name(data[a][0][i], full[i]) for i in ids} for a in ARMS}
    for a in ARMS:
        print(f"{a:5s} agreement with FULL: exact {sum(agr[a].values())}/{len(ids)} = {100 * sum(agr[a].values()) / len(ids):.1f}%, "
              f"tool name {100 * sum(agn[a].values()) / len(ids):.1f}%")
    for x, y in PAIRS:
        by = {}
        for i in ids:
            by.setdefault(full[i]["traj"], []).append(int(agr[x][i]) - int(agr[y][i]))
        flat = [d for g in by.values() for d in g]
        lo, hi = boot(by)
        print(f"{x} - {y}: {100 * sum(flat) / len(flat):+.1f} points [{lo:+.1f}, {hi:+.1f}], "
              f"discordant {sum(d > 0 for d in flat)} vs {sum(d < 0 for d in flat)}")
    for a in ARMS:
        rs = [data[a][0][i] for i in ids]
        tm = [r.get("timings") or {} for r in rs]
        print(f"{a:5s} prefill ms {describe([t['prefill_ms'] for t in tm if 'prefill_ms' in t])}; "
              f"decode tok/s {describe([t['decode_tokens_per_sec'] for t, r in zip(tm, rs) if (r.get('completion_tokens') or 0) >= 32])}; "
              f"wall s {describe([r['client_wall_s'] for r in rs])}")
        log = data[a][2]
        hits, misses = len(re.findall(r"\[kvflash\] uq hit", log)), len(re.findall(r"\[kvflash\] uq miss", log))
        pins = re.findall(r"\[kvflash\] preamble pin end=(\d+) pinned_chunks=(\d+) reserve=(\d+)", log)
        sc = [float(x) for x in re.findall(r"gpu scorer: .*? total ([\d.]+) ms", log)]
        print(f"      uq hit {hits} miss {misses}" + (f" ({100 * hits / (hits + misses):.1f}% hit)" if hits + misses else "")
              + f"; preamble pin lines {len(pins)}, pinned_chunks {sorted({int(p[1]) for p in pins})}"
              + f", reserve {sorted({int(p[2]) for p in pins})} (R={R})"
              + (f"; scorer median {st.median(sc):.1f} ms" if sc else "")
              + f"; disk_hit {sorted(set(re.findall(r'disk_hit=(\w+)', log)))}")


def selftest():
    tc = lambda n, a: {"tool_calls": [{"function": {"name": n, "arguments": json.dumps(a)}}], "content": "x"}
    assert agree(tc("bash", {"a": 1, "b": 2}), {"tool_calls": [{"function": {"name": "bash", "arguments": '{"b": 2, "a": 1}'}}]})
    assert not agree(tc("bash", {"a": 1}), tc("bash", {"a": 2}))
    assert agree_name(tc("bash", {"a": 1}), tc("bash", {"a": 2}))
    assert agree({"content": " hi "}, {"content": "hi", "tool_calls": None})
    assert not agree({"content": "hi"}, tc("bash", {}))
    lo, hi = boot({"t1": [1, 0], "t2": [0, 0], "t3": [1, 1]})
    assert 0 <= lo <= 50 <= hi <= 100
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1] == "--selftest" else main(sys.argv[1], int(sys.argv[2]))

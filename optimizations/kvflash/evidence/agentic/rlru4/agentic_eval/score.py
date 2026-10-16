"""Score arms. Usage: python3 -I score.py <tasks.jsonl> <results_dir> <out.json> [--retained-out retained_probes.json] [--arms FULL,POOL8K,...]
b (recall): retained probes = FULL answered correctly (decided on FULL alone). Per arm: retention = P(correct | retained).
c (next tool call): reference = FULL's own first parsed call; arm agrees if its first parsed call has the same key.
   Points where FULL produced no parsable call are excluded (counted); an arm parse failure counts as disagreement.
Statistics: paired cluster bootstrap over trajectories (10,000 resamples, pooled ratio), exact McNemar.
NOTE on construction: FULL defines the retained set, so FULL's own b-score is 1 and its c-agreement is 1 by definition;
POOL - FULL is therefore (arm score - 1) and every discordant pair is FULL-right/arm-wrong. A FULLB arm (same config as
FULL, run second) gives the non-degenerate greedy-nondeterminism floor for the same statistics."""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def cluster_bootstrap_diff(by_traj, n_boot=10000, seed=1):
    """by_traj: {traj: (num_arm, num_ref, den)}. Returns (point_diff, lo, hi) for arm-ref pooled-ratio difference."""
    ts = sorted(by_traj)
    tot_den = sum(by_traj[t][2] for t in ts)
    if tot_den == 0:
        return None
    point = (sum(by_traj[t][0] for t in ts) - sum(by_traj[t][1] for t in ts)) / tot_den
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = [ts[rng.randrange(len(ts))] for _ in ts]
        den = sum(by_traj[t][2] for t in s)
        if den:
            diffs.append((sum(by_traj[t][0] - by_traj[t][1] for t in s)) / den)
    diffs.sort()
    return point, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def mcnemar_exact(b, c):
    """Two-sided exact binomial p for discordant counts b (ref right, arm wrong) and c (ref wrong, arm right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def probe_ok(rec, task):
    return rec is not None and "error" not in rec and probes.score_exact(rec.get("content") or "", task["answer"])


def call_of(rec):
    if rec is None or "error" in rec:
        return None
    return probes.parse_response_call({"tool_calls": rec.get("tool_calls")})


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def timing_summary(recs):
    ok = [r for r in recs if r.get("timings")]
    out = {}
    for kind in ("probe", "decision", "primer", "all"):
        rs = [r for r in ok if kind == "all" or r["kind"] == kind]
        if not rs:
            continue
        pf = [r["timings"]["prefill_ms"] / 1000 for r in rs]
        dc = [r["timings"]["decode_ms"] / 1000 for r in rs]
        pft = [r["timings"].get("prefilled_tokens", 0) for r in rs]
        dct = [r.get("completion_tokens") or 0 for r in rs]
        pf_tps = [t / s for t, s in zip(pft, pf) if s > 0.5 and t > 500]
        dc_tps = [t / s for t, s in zip(dct, dc) if s > 0.3 and t >= 8]
        out[kind] = {"n": len(rs), "prefill_s_sum": sum(pf), "decode_s_sum": sum(dc), "wall_s_sum": sum(pf) + sum(dc),
                     "prefill_s_median": pctl(pf, .5), "decode_s_median": pctl(dc, .5),
                     "wall_s_median": pctl([a + b for a, b in zip(pf, dc)], .5),
                     "prefilled_tokens_sum": sum(pft), "completion_tokens_sum": sum(dct),
                     "prefill_tps_median": pctl(pf_tps, .5), "decode_tps_median": pctl(dc_tps, .5),
                     "cache_hit_frac": sum(1 for r in rs if (r["timings"].get("cached_prefix_tokens") or 0) > 6000) / len(rs)}
    return out


def analyze(tasks, results, arms):
    T = {t["id"]: t for t in tasks}
    R = {a: {r["id"]: r for r in results.get(a, []) if r["kind"] in ("probe", "decision")} for a in arms}
    full = R["FULL"]
    out = {"arms": arms, "errors": {a: sum(1 for r in R[a].values() if "error" in r) for a in arms}}
    # ---- b
    probes_all = [t for t in tasks if t["kind"] == "probe"]
    attempted = [t for t in probes_all if t["id"] in full]
    retained = [t for t in attempted if probe_ok(full[t["id"]], t)]
    out["b"] = {"candidates": len(probes_all), "full_attempted": len(attempted), "retained": len(retained),
                "yield": len(retained) / max(1, len(attempted)), "by_type_retained": {}, "arms": {}}
    for ty in probes.TYPES:
        a_ = [t for t in attempted if t["type"] == ty]
        out["b"]["by_type_retained"][ty] = {"attempted": len(a_), "retained": sum(1 for t in a_ if probe_ok(full[t["id"]], t))}
    for a in arms:
        if a == "FULL":
            continue
        have = [t for t in retained if t["id"] in R[a]]
        ok = {t["id"]: probe_ok(R[a][t["id"]], t) for t in have}
        by_traj = {}
        for t in have:
            n, r_, d = by_traj.get(t["traj"], (0, 0, 0))
            by_traj[t["traj"]] = (n + ok[t["id"]], r_ + 1, d + 1)
        k = sum(ok.values())
        bb = sum(1 for v in ok.values() if not v)
        d = {"n": len(have), "correct": k, "retention": k / max(1, len(have)), "wilson95": wilson(k, len(have)),
             "diff_vs_FULL": cluster_bootstrap_diff(by_traj), "mcnemar_b_ref_right_arm_wrong": bb, "mcnemar_c": 0,
             "mcnemar_p": mcnemar_exact(bb, 0), "by_type": {}, "by_P": {}, "by_distance": {}}
        for ty in probes.TYPES:
            s = [t for t in have if t["type"] == ty]
            if s:
                d["by_type"][ty] = [sum(ok[t["id"]] for t in s), len(s)]
        for P in (12288, 24576, 36864):
            s = [t for t in have if t["P"] == P]
            if s:
                d["by_P"][P // 1024] = [sum(ok[t["id"]] for t in s), len(s)]
        for lo, hi in ((0, 4096), (4096, 8192), (8192, 16384), (16384, 10**9)):
            s = [t for t in have if lo <= t["distance_tok"] < hi]
            if s:
                d["by_distance"]["%d-%s" % (lo, hi if hi < 10**9 else "inf")] = [sum(ok[t["id"]] for t in s), len(s)]
        out["b"]["arms"][a] = d
    # ---- c
    dps = [t for t in tasks if t["kind"] == "decision" and t["id"] in full]
    ref = {t["id"]: call_of(full[t["id"]]) for t in dps}
    valid = [t for t in dps if ref[t["id"]] is not None]
    out["c"] = {"decision_points": len(dps), "full_parsed": len(valid), "full_parse_fail": len(dps) - len(valid), "arms": {}}
    gold_ok = sum(1 for t in valid if t.get("gold_key") and tuple(t["gold_key"]) == ref[t["id"]])
    out["c"]["full_agrees_with_gold_full_key"] = gold_ok
    out["c"]["full_name_agrees_with_gold"] = sum(1 for t in valid if t.get("gold_key") and t["gold_key"][0] == ref[t["id"]][0])
    for a in arms:
        if a == "FULL":
            continue
        have = [t for t in valid if t["id"] in R[a]]
        calls = {t["id"]: call_of(R[a][t["id"]]) for t in have}
        ag = {t["id"]: calls[t["id"]] is not None and calls[t["id"]] == ref[t["id"]] for t in have}
        ag_name = {t["id"]: calls[t["id"]] is not None and calls[t["id"]][0] == ref[t["id"]][0] for t in have}
        by_traj, by_traj_n = {}, {}
        for t in have:
            n, r_, d_ = by_traj.get(t["traj"], (0, 0, 0))
            by_traj[t["traj"]] = (n + ag[t["id"]], r_ + 1, d_ + 1)
            n, r_, d_ = by_traj_n.get(t["traj"], (0, 0, 0))
            by_traj_n[t["traj"]] = (n + ag_name[t["id"]], r_ + 1, d_ + 1)
        k = sum(ag.values())
        d = {"n": len(have), "agree": k, "agreement": k / max(1, len(have)), "wilson95": wilson(k, len(have)),
             "diff_vs_FULL": cluster_bootstrap_diff(by_traj), "name_agreement": sum(ag_name.values()) / max(1, len(have)),
             "name_diff_vs_FULL": cluster_bootstrap_diff(by_traj_n), "arm_parse_fail": sum(1 for v in calls.values() if v is None),
             "mcnemar_b_ref_right_arm_wrong": len(have) - k, "mcnemar_c": 0, "mcnemar_p": mcnemar_exact(len(have) - k, 0),
             "agrees_with_gold_full_key": sum(1 for t in have if t.get("gold_key") and calls[t["id"]] == tuple(t["gold_key"]))}
        out["c"]["arms"][a] = d
    out["timing"] = {a: timing_summary(results.get(a, [])) for a in arms}
    return out


def main():
    tasks_p, res_dir, out_p = sys.argv[1:4]
    arms = ["FULL", "POOL8K", "POOL16K", "FULLB"]
    if "--arms" in sys.argv:
        arms = sys.argv[sys.argv.index("--arms") + 1].split(",")
    tasks = load_jsonl(tasks_p)
    results = {a: load_jsonl(os.path.join(res_dir, a + ".jsonl")) for a in arms}
    arms = [a for a in arms if results[a]]
    out = analyze(tasks, results, arms)
    if "--retained-out" in sys.argv:
        full = {r["id"]: r for r in results["FULL"]}
        ids = [t["id"] for t in tasks if t["kind"] == "probe" and t["id"] in full and probe_ok(full[t["id"]], t)]
        json.dump(ids, open(sys.argv[sys.argv.index("--retained-out") + 1], "w"))
    json.dump(out, open(out_p, "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()

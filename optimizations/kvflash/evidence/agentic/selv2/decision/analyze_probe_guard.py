"""Probe guard of PREREG_decision.md: V2PU - ON6 exact answer match on the 710 retained probes cut past 16,384 tokens
(analyze_e2e.py method: probes.score_exact, boot_ci over trajectories). Guard holds if the overall lower bound >= -2.0 points.
    python3 -P analyze_probe_guard.py <arms_dir>   (-P: numpy lives in the user site)"""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/kimi-k3/split-work/rlru4/agentic_eval")
sys.path.insert(0, "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/lucebox/a1")
import probes  # noqa: E402
from a1_core import boot_ci  # noqa: E402

tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
retained = json.load(open(f"{D}/retained_paged.json"))


def ok(r, i):
    return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[i]["answer"]))


def main(arms_dir):
    rows = {a: [json.loads(l) for l in open(f"{arms_dir}/probe_{a}/QK16KND.jsonl")] for a in ("ON6", "V2PU")}
    arms = {a: {r["id"]: r for r in rs if r.get("kind", "probe") == "probe"} for a, rs in rows.items()}
    ids = [i for i in retained if all(i in arms[a] for a in arms)]
    print(f"probes on both arms: {len(ids)} of {len(retained)}; errors "
          + ", ".join(f"{a} {sum('error' in arms[a][i] for i in ids)}" for a in arms)
          + "; disk_hit rows " + ", ".join(f"{a} {sum(bool((r.get('timings') or {}).get('disk_hit')) for r in rs)}"
                                           for a, rs in rows.items()))
    v = {a: {i: ok(arms[a][i], i) for i in ids} for a in arms}
    bands = {"overall": ids, "within": [i for i in ids if tasks[i]["distance_tok"] <= 16384],
             "beyond": [i for i in ids if tasks[i]["distance_tok"] > 16384]}
    for b, S in bands.items():
        d = boot_ci([v["V2PU"][i] - v["ON6"][i] for i in S], [tasks[i]["traj"] for i in S])
        print(f"{b:8s} n {len(S)}: ON6 {sum(v['ON6'][i] for i in S)}, V2PU {sum(v['V2PU'][i] for i in S)}; V2PU - ON6 "
              f"{100 * d['mean']:+.1f} points [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}], discordant "
              f"{sum(v['V2PU'][i] > v['ON6'][i] for i in S)} vs {sum(v['V2PU'][i] < v['ON6'][i] for i in S)}")
        if b == "overall":
            guard = 100 * d["lo"] >= -2.0
    print("probe guard:", "HOLDS" if guard else "FAILS", "(overall lower bound >= -2.0 points)")


if __name__ == "__main__":
    main(sys.argv[1])

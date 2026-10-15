"""A1 proper-run stage-1 report (20 trajectories): reference yield, retention with 95% and 99.9% intervals, by distance and probe type,
B0 POOL16K on the same trajectories and retained probes, the block-512 control's agreement with FULL-CHK, time and memory.
Decides nothing beyond the preregistered early look already in checks.json.

    python3 a1_proper_report.py <a1_dir> <b0_dir>      -> prints JSON, writes <a1_dir>/proper_report.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "modal", "am_qwen35"))
import a1_core as C  # noqa: E402
from a1_pilot_report import by_type, pool_retention  # noqa: E402


def control_agreement(H, tasks, full, fullb):
    """Block-512 against block-640 FULL-CHK on the probes both answered (PREREG A1 section 4, item 2): of the probes the reference got right
    and the control answered, how many the control flips; and the all-probe correctness / identical-answer agreement."""
    F = {r["id"]: r for r in full}
    B = {r["id"]: r for r in fullb}
    ok = lambda t, r: int("error" not in r and H.probes.score_exact(r.get("content") or "", t["answer"]))  # noqa: E731
    both = [t for t in tasks if t["kind"] == "probe" and t["id"] in F and t["id"] in B]
    ret = [t for t in both if ok(t, F[t["id"]])]
    flips = [t["id"] for t in ret if not ok(t, B[t["id"]])]
    same_ids = sum(1 for t in both if F[t["id"]].get("gen_ids") == B[t["id"]].get("gen_ids"))
    agree = sum(1 for t in both if ok(t, F[t["id"]]) == ok(t, B[t["id"]]))
    return {"trajectories": len({t["traj"] for t in both}), "probes_with_both_rows": len(both),
            "reference_correct_retained": len(ret), "control_flips_of_retained": len(flips),
            "flip_share_of_retained": len(flips) / len(ret) if ret else None, "flip_ids": flips,
            "correctness_agreement": [agree, len(both)], "identical_answer_token_ids": [same_ids, len(both)],
            "soft_reference_if_over_5_points": bool(ret and len(flips) / len(ret) > 0.05)}


def main(out, b0):
    import r1b_stats
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(out, "FULL.jsonl"))
    ship = C.read_rows(os.path.join(out, "SHIP.jsonl"))
    fullb = C.read_rows(os.path.join(out, "FULLB.jsonl"))
    trajs = json.load(open(os.path.join(out, "summary.json")))["trajectories"]
    T = set(trajs)
    ok, att = C.reference_yield(H, tasks, full, T)
    vals, cl, ids = C.retention_vectors(H, tasks, full, ship, T)
    ret = C.retained_probe_ids(H, tasks, full)
    b0_ret = set(json.load(open(os.path.join(b0, "retained.json"))))
    pv, pc, pids = pool_retention(H, tasks, b0_ret, C.read_rows(os.path.join(b0, "POOL16K.jsonl")), T)
    both = [i for i in ids if i in set(pids)]
    S, P = dict(zip(ids, vals)), dict(zip(pids, pv))
    x, y, p = r1b_stats.mcnemar_exact([S[i] for i in both], [P[i] for i in both])
    prog = [json.loads(l) for l in open(os.path.join(out, "progress.jsonl"))]
    arms = ("FULL", "SHIP", "FULLB")
    rep = {
        "trajectories": len(trajs), "reference_yield": [ok, att],
        "retained_probes_in_20": sum(1 for t in tasks if t["id"] in ret and t["traj"] in T),
        "retention": {"ci95": C.boot_ci(vals, cl, 95.0), "ci99.9": C.boot_ci(vals, cl, 99.9), "correct": sum(vals), "n": len(vals), "bar": C.BAR},
        "retention_by_distance": C.retention_by_distance(tasks, ids, vals), "retention_by_type": by_type(tasks, ids, vals),
        "retained_without_ship_row": sorted(i for i in ret if i not in {r["id"] for r in ship} and i.split(":")[0] in T),
        "b0_pool16k_same_trajs": {"ci95": C.boot_ci(pv, pc, 95.0), "correct": sum(pv), "n": len(pv)},
        "b0_pool16k_by_distance": C.retention_by_distance(tasks, pids, pv),
        "same_probe": {"n": len(both), "ship": sum(S[i] for i in both), "pool16k": sum(P[i] for i in both),
                       "mcnemar_only_ship_only_pool_p": [x, y, p]},
        "control_fullb_vs_full_chk": control_agreement(H, tasks, [r for r in full if r["traj"] in set(r2["traj"] for r2 in fullb)], fullb),
        "seconds_note": "summed over progress.jsonl rows (a unit interrupted by a watchdog stop and rerun appears once per completed run)",
        "seconds": {a: {k: round(sum(q[k] for q in prog if q.get("arm") == a), 1) for k in ("build_s", "answer_s", "wall_s")} for a in arms},
        "peak_gib": {a: {"alloc": max(q["peak_alloc_gib"] for q in prog if q.get("arm") == a),
                         "reserved": max(q["peak_reserved_gib"] for q in prog if q.get("arm") == a)} for a in arms},
        "checks": json.load(open(os.path.join(out, "checks.json"))),
    }
    print(json.dumps(rep, indent=1))
    with open(os.path.join(out, "proper_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

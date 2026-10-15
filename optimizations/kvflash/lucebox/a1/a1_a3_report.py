"""A3 report (PREREG "A3 routed exact fetch"): retention with its intervals, by distance and probe type, next to B0 POOL16K on the same
probes, router recall of the located source block, fetch count and rows per cut, and provenance times. Prints JSON and writes
<dir>/a3_report.json.

  python3 a1_a3_report.py sel    <a3sel_dir> <pilot_dir> <b0_dir>      stage 1: the four configurations, eligibility and the selection
  python3 a1_a3_report.py proper <a3_dir>    <a1_dir>    <b0_dir>      stage 2: retention, early look, A1 section 6 verdict, comparisons

Stage 1 decides only the configuration, never the bar (PREREG)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "modal", "am_qwen35"))
import a1_a3  # noqa: E402
import a1_core as C  # noqa: E402
import a1_e  # noqa: E402
from a1_pilot_report import by_type, pool_retention  # noqa: E402


# ---------------------------------------------------------------- pure tables over rows

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def cuts_table(rows):
    """One entry per (trajectory, cut): rows kept, k used and requested, over-budget flag. Rows of one cut share all four."""
    cuts = {}
    for r in rows:
        cuts[(r["traj"], r["cut_index"])] = (r["rows"], r["k"], r["k_req"], r["over_budget"], r["n_cand"])
    v = list(cuts.values())
    hist = {}
    for _, k, _, _, _ in v:
        hist[str(k)] = hist.get(str(k), 0) + 1
    return {"cuts": len(v), "mean_rows": _mean(x[0] for x in v), "max_rows": max((x[0] for x in v), default=None),
            "cuts_over_cap_at_k0": sum(1 for x in v if x[3]), "cuts_k_lowered": sum(1 for x in v if x[1] < min(x[2], x[4])),
            "cuts_without_candidates": sum(1 for x in v if x[4] == 0), "k_used_histogram": hist, "mean_k_used": _mean(x[1] for x in v)}


def router_tables(route_rows):
    """Router recall of the located source block (E3's definition), a_last. Probes whose block is already exact (P or W), in the
    open tail, or not located are counted and left out of the recall."""
    status = {}
    for r in route_rows:
        status[r["src_status"]] = status.get(r["src_status"], 0) + 1
    cand = [r for r in route_rows if r["src_status"] == "candidate" and r["rank_a_last"] is not None]
    far = [r for r in cand if r["distance_tok"] > a1_e.FAR]
    return {"probes": len(route_rows), "source_status": status,
            "a_last": {"candidate_probes": a1_e.recall_table([r["rank_a_last"] for r in cand]),
                       "candidate_probes_over_16K": a1_e.recall_table([r["rank_a_last"] for r in far])},
            "mean_candidates_per_probe": _mean(r["n_cand"] for r in route_rows)}


def exact_at_answer(route_rows, k):
    """Share of located probes whose source block(s) are exact in the answer state (prefix, window or fetched)."""
    loc = [r for r in route_rows if r["located"] and r["src_blocks"]]
    hit = sum(1 for r in loc if r["per_k"][str(k)]["source_exact_at_answer"])
    return {"located_in_sealed_blocks": len(loc), "source_exact_at_answer": hit}


def provenance(rows):
    """Never a gate: routing-pass prefill time, fetch bytes, wall per probe."""
    return {"route_s_mean": _mean(r["route_s"] for r in rows), "fetch_bytes_mean": _mean(r["fetch_bytes"] for r in rows),
            "probe_wall_s_mean": _mean(r["probe_wall_s"] for r in rows), "answer_decode_tokens_mean": _mean(r["decode_tokens"] for r in rows)}


def correct(H, tasks_by_id, r):
    return int("error" not in r and H.probes.score_exact(r.get("content") or "", tasks_by_id[r["id"]]["answer"]))


# ---------------------------------------------------------------- one arm

def arm_summary(H, tasks, full, rows, T):
    vals, cl, ids = C.retention_vectors(H, tasks, full, rows, T)
    by = {t["id"]: t for t in tasks}
    used = [r for r in rows if r["id"] in set(ids)]
    over = {r["id"]: r["over_budget"] for r in used}
    within = [(v, i) for v, i in zip(vals, ids) if not over[i]]
    return {"n": len(vals), "correct": sum(vals), "retention": C.boot_ci(vals, cl) if vals else None,
            "retention_ci99.9": C.boot_ci(vals, cl, 99.9) if vals else None,
            "by_distance": C.retention_by_distance(tasks, ids, vals), "by_type": by_type(tasks, ids, vals),
            "probes_over_cap_at_k0": sum(1 for r in used if r["over_budget"]),
            "within_cap_only": {"n": len(within), "correct": sum(v for v, _ in within)},
            "mean_rows": _mean(r["rows"] for r in used), "cuts": cuts_table(used), "provenance": provenance(used) if used else None,
            "_ids": ids, "_vals": vals}


def pool_block(H, tasks, b0, T, arm_ids, arm_vals):
    import r1b_stats
    b0_ret = set(json.load(open(os.path.join(b0, "retained.json"))))
    pv, pc, pids = pool_retention(H, tasks, b0_ret, C.read_rows(os.path.join(b0, "POOL16K.jsonl")), T)
    S, P = dict(zip(arm_ids, arm_vals)), dict(zip(pids, pv))
    both = [i for i in arm_ids if i in P]
    x, y, p = r1b_stats.mcnemar_exact([S[i] for i in both], [P[i] for i in both]) if both else (0, 0, 1.0)
    return {"pool16k_same_trajectories": {"ci95": C.boot_ci(pv, pc, 95.0) if pv else None, "correct": sum(pv), "n": len(pv)},
            "pool16k_by_distance": C.retention_by_distance(tasks, pids, pv),
            "same_probe": {"n": len(both), "a3": sum(S[i] for i in both), "pool16k": sum(P[i] for i in both),
                           "mcnemar_a3_only_pool_only_p": [x, y, p]},
            "same_probe_by_distance": {"a3": C.retention_by_distance(tasks, both, [S[i] for i in both]),
                                       "pool16k": C.retention_by_distance(tasks, both, [P[i] for i in both])}}


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------- stages

def report_sel(sel, pilot, b0):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(pilot, "FULL.jsonl"))
    T = set(list(H.load_trajs())[:a1_a3.PILOT_N])
    rep, table = {"stage": "sel", "trajectories": a1_a3.PILOT_N, "arms": {}, "router": {}}, {}
    ws = sorted({int(f.split("_")[1][1:]) for f in os.listdir(sel) if f.startswith("A3_W") and f.endswith("_route.jsonl")})
    for W in ws:
        route = [r for r in C.read_rows(os.path.join(sel, a1_a3.route_name(W) + ".jsonl")) if r["traj"] in T]
        rep["router"][f"W{W}"] = router_tables(route)
        for k in sorted({int(f.split("_K")[1].split(".")[0]) for f in os.listdir(sel) if f.startswith(f"A3_W{W}_K")}):
            name = f"W{W}_K{k}"
            rows = [r for r in C.read_rows(os.path.join(sel, a1_a3.arm_name(W, k) + ".jsonl")) if r["traj"] in T]
            s = arm_summary(H, tasks, full, rows, T)
            s["source_exact_at_answer"] = exact_at_answer(route, k)
            s["pool16k"] = pool_block(H, tasks, b0, T, s["_ids"], s["_vals"])
            table[name] = {"correct": s["correct"], "n": s["n"], "mean_rows": s["mean_rows"] if s["mean_rows"] is not None else 0.0,
                           "eligible": s["probes_over_cap_at_k0"] == 0 and s["n"] > 0}
            rep["arms"][name] = strip(s)
    rep["table"] = table
    rep["selection"] = a1_a3.select_config(table)
    rep["selection_rule"] = ("highest retention among configurations with no cut over the row cap at k = 0; ties by fewer mean rows; "
                             "'selected_unrestricted' ignores the eligibility restriction and is for the record only")
    return rep


def report_proper(out, a1, b0):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(out, "FULL.jsonl"))
    summ = json.load(open(os.path.join(out, "summary.json")))
    T = set(summ["trajectories"])
    W, k = summ["W"], summ["k"]
    rows = [r for r in C.read_rows(os.path.join(out, a1_a3.arm_name(W, k) + ".jsonl")) if r["traj"] in T]
    route = [r for r in C.read_rows(os.path.join(out, a1_a3.route_name(W) + ".jsonl")) if r["traj"] in T]
    s = arm_summary(H, tasks, full, rows, T)
    ok, att = C.reference_yield(H, tasks, full, T)
    n_ret = sum(1 for t in tasks if t["id"] in C.retained_probe_ids(H, tasks, full) and t["traj"] in T)
    ci95 = s["retention"]
    early = (summ.get("stopped") or {}).get("detail")
    if summ.get("stopped") and summ["stopped"]["reason"].startswith("early-"):
        decision = {"basis": f"early look at {a1_a3.core.EARLY_AT} trajectories, 99.9% interval", "verdict": early["verdict"]}
    else:
        decision = {"basis": "final, 95% interval (A1 section 6)", "verdict": a1_a3.final_verdict(ci95) if ci95 else None}
    meets = len(T) >= 50 and n_ret >= 600
    rep = {"stage": "proper", "arm": summ["arm"], "W": W, "k_requested": k, "trajectories": len(T), "reference_yield": [ok, att],
           "retained_probes": n_ret, "retention": {k_: s[k_] for k_ in ("n", "correct", "retention", "retention_ci99.9")}, "bar": C.BAR,
           "decision": decision, "size_requirement_50_trajectories_and_600_retained_met": meets,
           "size_note": None if meets or decision["basis"].startswith("early") else "fewer than 50 trajectories or 600 retained probes: a prefix, say so",
           "early_look": early, "by_distance": s["by_distance"], "by_type": s["by_type"], "probes_over_cap_at_k0": s["probes_over_cap_at_k0"],
           "within_cap_only": s["within_cap_only"], "mean_rows": s["mean_rows"], "cuts": s["cuts"], "provenance": s["provenance"],
           "router": router_tables(route), "source_exact_at_answer": exact_at_answer(route, k),
           "pool16k": pool_block(H, tasks, b0, T, s["_ids"], s["_vals"]),
           "b0_pool16k_bar_by_distance": "B0: 96%, 91%, 96%, 6% (PREREG A1 section 1); the same-trajectory values are in pool16k"}
    return rep


def main(argv):
    mode, d, ref, b0 = argv[1:5]
    rep = report_sel(d, ref, b0) if mode == "sel" else report_proper(d, ref, b0)
    print(json.dumps(rep, indent=1))
    with open(os.path.join(d, "a3_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main(sys.argv)

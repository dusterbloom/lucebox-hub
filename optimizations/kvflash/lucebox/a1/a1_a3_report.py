"""A3 report (PREREG "A3 routed exact fetch"): retention with its intervals, by distance and probe type, next to B0 POOL16K on the same
probes, by distance bin and by k used, router recall of the located source block (ranked, and within the fetched set), k and rows
per cut, and provenance times. Prints JSON and writes
<dir>/a3_report.json.

  python3 a1_a3_report.py sel    <a3sel_dir> <pilot_dir> <b0_dir>      stage 1: the two windows (2, 6 blocks), eligibility, selection
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


class IncompleteRun(ValueError):
    """A run (or a mix of attempts) that cannot back a selection or a verdict."""


def check_windows(found):
    want = sorted(f"W{w}" for w in a1_a3.SEL_WBS)
    if sorted(found) != want:
        raise IncompleteRun(f"stage 1 needs exactly the arms {want}, found {sorted(found)}")


def check_complete(name, answers, route, expected):
    """Exactly one answer record and one router record per expected probe, from the same attempt. Refuses otherwise."""
    problems = []
    for label, rows in (("answer", answers), ("router", route)):
        ids = [r["id"] for r in rows]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        miss, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
        for what, xs in (("missing", miss), ("unexpected", extra), ("duplicate", dup)):
            if xs:
                problems.append(f"{label} records {what}: {len(xs)} (e.g. {xs[0]})")
    a, r = {x["id"]: x.get("attempt") for x in answers}, {x["id"]: x.get("attempt") for x in route}
    mixed = sorted(i for i in a if i in r and (a[i] != r[i] or a[i] is None))
    if mixed:
        problems.append(f"answer and router records from different attempts (or none): {len(mixed)} (e.g. {mixed[0]})")
    if problems:
        raise IncompleteRun(f"{name}: " + "; ".join(problems))


def cuts_table(rows):
    """One entry per (trajectory, cut): rows kept, reserve, k used, candidates. Rows of one cut share all of them."""
    cuts = {}
    for r in rows:
        cuts[(r["traj"], r["cut_index"])] = (r["rows"], r["k"], r["n_cand"], r["reserve"], r["peak_rows"])
    v = list(cuts.values())
    hist = {}
    for x in v:
        hist[str(x[1])] = hist.get(str(x[1]), 0) + 1
    return {"cuts": len(v), "mean_rows": _mean(x[0] for x in v), "max_rows": max((x[0] for x in v), default=None),
            "mean_reserve": _mean(x[3] for x in v), "max_reserve": max((x[3] for x in v), default=None),
            "mean_peak_rows": _mean(x[4] for x in v), "max_peak_rows": max((x[4] for x in v), default=None),
            "cuts_k_below_candidates": sum(1 for x in v if x[1] < x[2]),
            "cuts_without_candidates": sum(1 for x in v if x[2] == 0), "k_used_histogram": hist, "mean_k_used": _mean(x[1] for x in v)}


def by_k_used(rows, ids, vals):
    """Retention counts grouped by the k of the probe's cut."""
    k_of = {r["id"]: r["k"] for r in rows}
    out = {}
    for i, v in zip(ids, vals):
        d = out.setdefault(str(k_of[i]), {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += v
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def fetch_coverage_vs_retention(route_rows, val_by_id):
    """Candidate probes (source block compacted before the fetch) by how much of their candidate source blocks the fetch restored
    (none, partial, all), with the number answered correctly. Descriptive: it does not say why a probe failed (the other blocks, the
    carried DeltaNet state and the history's K/V are still affected by compaction)."""
    cand = [r for r in route_rows if r["src_status"] == "candidate" and r["id"] in val_by_id]
    out = {"candidate_probes": len(cand)}
    for name in ("none", "partial", "all"):
        sel = [r for r in cand if r["source_fetch_coverage"] == name]
        out[name] = {"n": len(sel), "correct": sum(val_by_id[r["id"]] for r in sel)}
    return out


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


def exact_at_answer(route_rows):
    """Share of located probes whose source block(s) are exact in the answer state (prefix, window or fetched)."""
    loc = [r for r in route_rows if r["located"] and r["src_blocks"]]
    return {"located_in_sealed_blocks": len(loc), "source_exact_at_answer": sum(1 for r in loc if r["source_exact_at_answer"])}


def provenance(rows):
    """Never a gate: routing-pass prefill time, fetch bytes, wall per probe."""
    return {"route_s_mean": _mean(r["route_s"] for r in rows), "fetch_bytes_mean": _mean(r["fetch_bytes"] for r in rows),
            "probe_wall_s_mean": _mean(r["probe_wall_s"] for r in rows), "answer_decode_tokens_mean": _mean(r["decode_tokens"] for r in rows)}


def correct(H, tasks_by_id, r):
    return int("error" not in r and H.probes.score_exact(r.get("content") or "", tasks_by_id[r["id"]]["answer"]))


# ---------------------------------------------------------------- one arm

def arm_summary(H, tasks, full, rows, route, T):
    vals, cl, ids = C.retention_vectors(H, tasks, full, rows, T)
    by = {t["id"]: t for t in tasks}
    used = [r for r in rows if r["id"] in set(ids)]
    return {"n": len(vals), "correct": sum(vals), "retention": C.boot_ci(vals, cl) if vals else None,
            "retention_ci99.9": C.boot_ci(vals, cl, 99.9) if vals else None,
            "by_distance": C.retention_by_distance(tasks, ids, vals), "by_type": by_type(tasks, ids, vals),
            "by_k_used": by_k_used(used, ids, vals), "source_fetch_coverage_vs_retention": fetch_coverage_vs_retention(route, dict(zip(ids, vals))),
            "source_exact_at_answer": exact_at_answer(route),
            "mean_rows": _mean(r["rows"] for r in used), "mean_peak_rows": _mean(r["peak_rows"] for r in used), "cuts": cuts_table(used), "provenance": provenance(used) if used else None,
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
    try:                                                     # the fixed universe: registered reference file, 6 trajectories, 118 probes
        a1_a3.check_reference(os.path.join(pilot, "FULL.jsonl"), a1_a3.PILOT_REF_MD5)
        want = a1_a3.universe_ids(H, tasks, full, T)
        a1_a3.check_universe(want, a1_a3.PILOT_PROBES, "pilot reference")
    except SystemExit as e:
        raise IncompleteRun(str(e))
    wbs = sorted({int(f.split("_")[1][1:]) for f in os.listdir(sel) if f.startswith("A3_W") and f.endswith("_route.jsonl")})
    check_windows([f"W{wb}" for wb in wbs])
    for wb in wbs:
        name = f"W{wb}"
        route = [r for r in C.read_rows(os.path.join(sel, a1_a3.route_name(wb) + ".jsonl")) if r["traj"] in T]
        rep["router"][name] = router_tables(route)
        rows = [r for r in C.read_rows(os.path.join(sel, a1_a3.arm_name(wb) + ".jsonl")) if r["traj"] in T]
        check_complete(name, rows, route, want)
        s = arm_summary(H, tasks, full, rows, route, T)
        s["pool16k"] = pool_block(H, tasks, b0, T, s["_ids"], s["_vals"])
        table[name] = {"correct": s["correct"], "n": s["n"], "mean_rows": s["mean_rows"] if s["mean_rows"] is not None else 0.0,
                       }
        rep["arms"][name] = strip(s)
    rep["table"] = table
    rep["complete"] = True            # exactly W2 and W6, each with one record per probe of the fixed universe (not of whatever rows exist)
    rep["expected_probes"] = len(want)
    rep["reference_md5"] = a1_a3.PILOT_REF_MD5
    rep["selection"] = a1_a3.select_config(table)
    rep["selection_rule"] = "highest retention; ties by fewer mean rows, then the smaller W (a cut over budget at k = 0 aborts the run, so there is no eligibility test)"
    return rep


def report_proper(out, a1, b0):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(out, "FULL.jsonl"))
    summ = json.load(open(os.path.join(out, "summary.json")))
    T = set(summ["trajectories"])
    wb = summ["window_blocks"]
    rows = [r for r in C.read_rows(os.path.join(out, a1_a3.arm_name(wb) + ".jsonl")) if r["traj"] in T]
    route = [r for r in C.read_rows(os.path.join(out, a1_a3.route_name(wb) + ".jsonl")) if r["traj"] in T]
    try:                                                     # held-out 7-20: the registered A1 reference, 219 probes; 21+ come from this run's own FULL arm
        a1_a3.check_reference(os.path.join(a1, "FULL.jsonl"), a1_a3.A1_REF_MD5)
        order = list(H.load_trajs())[a1_a3.PILOT_N:]
        a1_a3.check_universe(a1_a3.universe_ids(H, tasks, full, set(order[:a1_a3.A1_HELDOUT_END - a1_a3.PILOT_N])),
                             a1_a3.A1_HELDOUT_PROBES, "A1 proper reference, held-out trajectories 7-20")
    except SystemExit as e:
        raise IncompleteRun(str(e))
    check_complete(summ["arm"], rows, route, a1_a3.universe_ids(H, tasks, full, T))
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    s = arm_summary(H, tasks, full, rows, route, T)
    ok, att = C.reference_yield(H, tasks, full, T)
    n_ret = sum(1 for t in tasks if t["id"] in C.retained_probe_ids(H, tasks, full) and t["traj"] in T)
    ci95 = s["retention"]
    early = (summ.get("stopped") or {}).get("detail")
    if summ.get("stopped") and summ["stopped"]["reason"].startswith("early-"):
        decision = {"basis": f"early look at {a1_a3.core.EARLY_AT} trajectories, 99.9% interval", "verdict": early["verdict"]}
    else:
        decision = {"basis": "final, 95% interval (A1 section 6)", "verdict": a1_a3.final_verdict(ci95) if ci95 else None}
    meets = len(T) >= 50 and n_ret >= 600
    rep = {"stage": "proper", "arm": summ["arm"], "window_blocks": wb, "selection_report_sha256": manifest["sel_report_sha256"], "trajectories": len(T), "reference_yield": [ok, att],
           "retained_probes": n_ret, "retention": {k_: s[k_] for k_ in ("n", "correct", "retention", "retention_ci99.9")}, "bar": C.BAR,
           "decision": decision, "size_requirement_50_trajectories_and_600_retained_met": meets,
           "size_note": None if meets or decision["basis"].startswith("early") else "fewer than 50 trajectories or 600 retained probes: a prefix, say so",
           "early_look": early, "by_distance": s["by_distance"], "by_type": s["by_type"], "by_k_used": s["by_k_used"],
           "source_fetch_coverage_vs_retention": s["source_fetch_coverage_vs_retention"],
           "mean_rows": s["mean_rows"], "mean_peak_rows": s["mean_peak_rows"], "cuts": s["cuts"], "provenance": s["provenance"],
           "router": router_tables(route), "source_exact_at_answer": s["source_exact_at_answer"],
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

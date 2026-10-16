"""A4 report (PREREG "A4 R-LRU offline pilot"): per arm, retention against the pilot FULL-CHK rows with the trajectory-clustered bootstrap
interval, overall, by distance bin and by probe type; the router rank tables of the two routed arms; the paired A4_K4 vs A4_K0 counts
(McNemar exact); and the registered numbers next to them (A3 W=2, B0 POOL16K on the same trajectories, E3, the engine's pure LRU).
Decides nothing. Prints JSON and writes <dir>/a4_report.json.

  python3 a1_a4_report.py <a4_dir> <pilot_dir> <b0_dir> [<a3sel_dir> [<e3e2_dir>]]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
import a1_a3  # noqa: E402
import a1_a3_report as r3  # noqa: E402
import a1_a4  # noqa: E402
import a1_core as C  # noqa: E402
import a1_e  # noqa: E402
from a1_pilot_report import pool_retention  # noqa: E402

# Registered reference numbers (counts), quoted next to the arms. The engine anchor is the 3090 OFF arm of the prefill-reselect run.
REGISTERED = {"A3_W2": "95/118 (A3 stage 1)", "POOL16K_same_trajectories": "88/111 (B0)", "E3": "99/118 (oracle swap)",
              "engine_pure_lru_3090": "700/951 overall, 688/744 within 16K (prefill-reselect OFF arm)"}


# ---------------------------------------------------------------- pure tables

def bin_label(lo, hi):
    return f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"


def cell(vals, clusters):
    """{correct, n, retention: bootstrap interval clustered by trajectory}."""
    return {"correct": int(sum(vals)), "n": len(vals), "retention": C.boot_ci(vals, clusters) if vals else None}


def grouped(keys, vals, clusters):
    """cell() per distinct key (insertion order)."""
    out = {}
    for k in dict.fromkeys(keys):
        idx = [i for i, x in enumerate(keys) if x == k]
        out[k] = cell([vals[i] for i in idx], [clusters[i] for i in idx])
    return out


def retention_cells(tasks, ids, vals, clusters):
    """Overall, by distance bin (0-4K, 4-8K, 8-16K, over 16K) and by probe type, each with its interval."""
    by = {t["id"]: t for t in tasks}
    dist = [by[i]["distance_tok"] for i in ids]
    labels = [next(bin_label(lo, hi) for lo, hi in C.DISTANCE_BINS if lo <= d < hi) for d in dist]
    types = [by[i].get("probe_type") or by[i].get("type") or "?" for i in ids]
    by_d = grouped(labels, vals, clusters)
    return {"overall": cell(vals, clusters),
            "by_distance": {bin_label(lo, hi): by_d.get(bin_label(lo, hi), {"correct": 0, "n": 0, "retention": None}) for lo, hi in C.DISTANCE_BINS},
            "by_type": grouped(types, vals, clusters)}


def paired(a, b):
    """Paired counts of two {id: 0/1} maps over their common ids, with McNemar exact (r1b_stats): both, only a, only b, neither."""
    import r1b_stats
    ids = [i for i in a if i in b]
    x, y, p = r1b_stats.mcnemar_exact([a[i] for i in ids], [b[i] for i in ids]) if ids else (0, 0, 1.0)
    return {"n": len(ids), "both": sum(a[i] and b[i] for i in ids), "only_a": x, "only_b": y,
            "neither": sum((not a[i]) and (not b[i]) for i in ids), "mcnemar_exact_p": p}


def router_table(route):
    """Rank tables of one routed arm. recall@k over the probes whose source block is a candidate (rank not None), overall and for facts
    older than 16K; the status counts (resident = pinned prefix, sink, window or tail); the share inside the fetched k of the cut."""
    status = {}
    for r in route:
        status[r["src_status"]] = status.get(r["src_status"], 0) + 1
    cand = [r for r in route if r["src_status"] == "candidate" and r.get("ranks")]
    far = [r for r in cand if r["distance_tok"] > a1_e.FAR]
    out = {"probes": len(route), "source_status": status, "source_candidate_with_ranks": len(cand), "mean_candidates": _mean(r["n_cand"] for r in route),
           "mean_k": _mean(r["k"] for r in route)}
    for name in a1_a4.RANKERS:
        rs = [r["ranks"][name] for r in cand if r["ranks"].get(name) is not None]
        fr = [r["ranks"][name] for r in far if r["ranks"].get(name) is not None]
        out[name] = {"all": a1_e.recall_table(rs), "over_16K": a1_e.recall_table(fr),
                     "inside_k_optimistic": [sum(r["ranks"][name] <= r["k_plan"] for r in cand if r["ranks"].get(name) is not None), len(rs)]}
    cov = {"none": 0, "partial": 0, "all": 0}
    for r in cand:
        c = fetch_coverage(r.get("src_cand_blocks"), r.get("fetched"))
        if c:
            cov[c] += 1
    out["x_last_fetch_coverage"] = cov
    return out


def fetch_coverage(src_cand, fetched):
    """How much of the fact's candidate source blocks the real fetch (ties broken by lower block id) restored: None, "none", "partial", "all"."""
    if not src_cand:
        return None
    n = sum(1 for j in src_cand if j in (fetched or []))
    return "none" if n == 0 else "all" if n == len(src_cand) else "partial"


def cuts_table(cuts):
    """Per arm, one record per (trajectory, cut): window blocks at build time and at the cut, candidates, k, reserve and the row bound."""
    def col(name):
        return [c[name] for c in cuts]
    return {"cuts": len(cuts), "cuts_with_candidates": sum(1 for c in cuts if c["n_cand"] > 0),
            "window_build_mean": _mean(col("window_build")), "window_cut_mean": _mean(col("window_cut")),
            "window_cut_min": min(col("window_cut"), default=None), "candidates_mean": _mean(col("n_cand")), "candidates_max": max(col("n_cand"), default=None),
            "k_histogram": {str(k): col("k").count(k) for k in sorted(set(col("k")))}, "reserve_mean": _mean(col("reserve")),
            "rows_bound_max": max(col("rows_bound"), default=None), "peak_bound_max": max(col("peak_bound"), default=None)}


def row_stats(rows):
    """Peak rows actually held (rows + reserve) and the share of answers reused from the sibling arm (identical state)."""
    return {"mean_rows": _mean(r["rows"] for r in rows), "mean_peak_rows": _mean(r["peak_rows"] for r in rows),
            "max_peak_rows": max((r["peak_rows"] for r in rows), default=None), "answers_reused": sum(1 for r in rows if r.get("answer_reused")),
            "probes_without_candidates": sum(1 for r in rows if r["n_cand"] == 0)}


def no_candidate_identity(rows, full_by_id):
    """Sanity: where nothing was evicted at the cut (no candidates) the state is the full-KV state, so the answer text should equal the
    reference's. Counts the probes whose content equals the reference content."""
    nc = [r for r in rows if r["n_cand"] == 0 and r["id"] in full_by_id]
    return {"n": len(nc), "content_equal_to_reference": sum(1 for r in nc if (r.get("content") or "") == (full_by_id[r["id"]].get("content") or ""))}


def by_candidates(rows, ids, vals, clusters):
    """Retention split by whether the probe's cut evicted anything: 'none' cuts hold the full-KV state (the sanity set), 'some' cuts are
    the informative ones."""
    n_cand = {r["id"]: r["n_cand"] for r in rows}
    return grouped(["some" if n_cand[i] else "none" for i in ids], vals, clusters)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- report

def values_of(H, tasks, full, rows, T):
    vals, cl, ids = C.retention_vectors(H, tasks, full, rows, T)
    return vals, cl, ids


def main(d, pilot, b0, d3=None, d32=None):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(pilot, "FULL.jsonl"))
    a1_a3.check_reference(os.path.join(pilot, "FULL.jsonl"), a1_a3.PILOT_REF_MD5)
    a1_a3.check_reference(os.path.join(d, "FULL.jsonl"), a1_a3.PILOT_REF_MD5)           # the copy in the output directory is the registered file
    T = set(list(H.load_trajs())[:a1_a3.PILOT_N])
    want = a1_a3.universe_ids(H, tasks, full, T)
    a1_a3.check_universe(want, a1_a3.PILOT_PROBES, "pilot reference")
    rep = {"reference_md5": a1_a3.PILOT_REF_MD5, "expected_probes": len(want), "arms": {}, "router": {}, "cuts": {}, "registered": REGISTERED}
    vmap, full_by_id = {}, {r["id"]: r for r in full}
    for arm in a1_a4.ARMS:
        rows = [r for r in C.read_rows(os.path.join(d, arm + ".jsonl")) if r["traj"] in T]
        if arm in a1_a4.ROUTED:
            route = [r for r in C.read_rows(os.path.join(d, arm + "_route.jsonl")) if r["traj"] in T]
            r3.check_complete(arm, rows, route, want)
            rep["router"][arm] = router_table(route)
        else:
            r3.check_complete(arm, rows, rows, want)
        vals, cl, ids = values_of(H, tasks, full, rows, T)
        vmap[arm] = dict(zip(ids, vals))
        build = a1_a4.ARMS[arm]["build"]
        cuts = [c for c in C.read_rows(os.path.join(d, f"A4_cuts_{build}.jsonl")) if c["arm"] == arm and c["traj"] in T]
        rep["cuts"][arm] = cuts_table(cuts)
        rep["arms"][arm] = {**retention_cells(tasks, ids, vals, cl), **row_stats(rows), "no_candidate_identity": no_candidate_identity(rows, full_by_id),
                            "by_cut_has_candidates": by_candidates(rows, ids, vals, cl)}
    rep["paired"] = {"A4_K4_vs_A4_K0": paired(vmap["A4_K4"], vmap["A4_K0"]), "A4_K4P_vs_A4_K4": paired(vmap["A4_K4P"], vmap["A4_K4"])}
    nxt = {"registered": REGISTERED}
    b0_ret = set(json.load(open(os.path.join(b0, "retained.json"))))
    pv, pc, pids = pool_retention(H, tasks, b0_ret, C.read_rows(os.path.join(b0, "POOL16K.jsonl")), T)
    nxt["POOL16K_same_trajectories"] = {**cell(pv, pc), "by_distance": C.retention_by_distance(tasks, pids, pv)}
    pmap = dict(zip(pids, pv))
    nxt["paired_vs_POOL16K"] = {a: paired(vmap[a], pmap) for a in a1_a4.ARMS}
    for name, path in (("A3_W2", d3 and os.path.join(d3, "A3_W2.jsonl")), ("E3", d32 and os.path.join(d32, "E3.jsonl"))):
        if path and os.path.exists(path):
            v, c, i = values_of(H, tasks, full, C.read_rows(path), T)
            nxt[name] = {**cell(v, c), "by_distance": C.retention_by_distance(tasks, i, v)}
            nxt["paired_" + name] = {a: paired(vmap[a], dict(zip(i, v))) for a in a1_a4.ARMS}
    rep["next_to"] = nxt
    print(json.dumps(rep, indent=1))
    with open(os.path.join(d, "a4_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])

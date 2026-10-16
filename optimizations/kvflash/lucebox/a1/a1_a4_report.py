"""A4 report (PREREG "A4 R-LRU offline pilot"): per arm, retention against the pilot FULL-CHK rows with the trajectory-clustered bootstrap
interval, overall, by distance bin and by probe type; the router rank tables of the two routed arms; the paired A4_K4 vs A4_K0 counts
(McNemar exact); and the registered numbers next to them (A3 W=2, B0 POOL16K on the same trajectories, E3, the engine's pure LRU).
Decides nothing. Prints JSON and writes <dir>/a4_report.json.

  python3 a1_a4_report.py <a4_dir> <pilot_dir> <b0_dir> [<a3sel_dir> [<e3e2_dir>]]
  python3 a1_a4_report.py --partial <a4_dir> <pilot_dir>     (A4_K0 and A4_CT complete: prints the primary rule and the arms that are complete)
Amendment 1 (compact-on-evict arms A4_CT, A4_CTK4, A4_CA, A4_CT0): when their row files exist the report adds `compact_on_evict`: per arm the
retention over the reference-correct probes split within the pool (distance under 16K) and beyond it, each paired against A4_K0 (counts,
McNemar, and the bootstrap interval of the paired difference clustered by trajectory), the mean compacted blocks, the drops and the
delayed-query subsample counts, and the registered primary rule.
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


def router_table(route, rankers=None):
    """Rank tables of one routed arm. recall@k over the probes whose source block is a candidate (rank not None), overall and for facts
    older than 16K; the status counts (resident = pinned prefix, sink, window or tail); the share inside the fetched k of the cut."""
    status = {}
    for r in route:
        status[r["src_status"]] = status.get(r["src_status"], 0) + 1
    cand = [r for r in route if r["src_status"] in ("candidate", "compacted") and r.get("ranks")]      # "compacted" is A4_CTK4's name for a routable source
    far = [r for r in cand if r["distance_tok"] >= C.DISTANCE_BINS[-1][0]]                          # 16,384 and over, as the pool split and the distance bins
    out = {"probes": len(route), "source_status": status, "source_candidate_with_ranks": len(cand), "mean_candidates": _mean(r["n_cand"] for r in route),
           "mean_k": _mean(r["k"] for r in route)}
    for name in (rankers or a1_a4.RANKERS):
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


# ---------------------------------------------------------------- Amendment 1: compact-on-evict tables

def pool_split(tasks, ids):
    """'within' (the fact is under 16K tokens back: inside the 16,384-row pool) or 'beyond' (the last distance bin, 16K and over), per id."""
    by = {t["id"]: t for t in tasks}
    far = C.DISTANCE_BINS[-1][0]
    return ["beyond" if by[i]["distance_tok"] >= far else "within" for i in ids]


def paired_difference(a, b, clusters_by_id):
    """Bootstrap interval (clustered by trajectory) of mean(a - b) over the common ids of two {id: 0/1} maps; None when there are none."""
    ids = [i for i in a if i in b]
    return C.boot_ci([a[i] - b[i] for i in ids], [clusters_by_id[i] for i in ids]) if ids else None


def split_vs_k0(vmap, k0, split, clusters_by_id):
    """Per pool split: the arm's cell, A4_K0's cell on the same probes, the paired counts and the interval of the paired difference."""
    out = {}
    for name in ("within", "beyond"):
        ids = [i for i in vmap if split[i] == name]
        a, b = {i: vmap[i] for i in ids}, {i: k0[i] for i in ids if i in k0}
        out[name] = {"arm": cell(list(a.values()), [clusters_by_id[i] for i in a]), "A4_K0": cell(list(b.values()), [clusters_by_id[i] for i in b]),
                     "paired": paired(a, b), "difference": paired_difference(a, b, clusters_by_id)}
    return out


def ct_stats(cuts, rows):
    """Compacted blocks (mean per cut and per probe), drops (build: the last cut of each trajectory; cut: summed over cuts, ephemeral) and the delayed-query
    counts of the build (compactions, those subsampled to the 1024 cap, mean tokens available and used; last cut of each trajectory)."""
    last = {}
    for c in cuts:
        if c["traj"] not in last or c["q"]["compactions"] >= last[c["traj"]]["q"]["compactions"]:
            last[c["traj"]] = c
    q = [c["q"] for c in last.values()]
    n_comp = sum(x["compactions"] for x in q)
    return {"cuts": len(cuts), "mean_compacted_blocks_per_cut": _mean(c["n_compacted"] for c in cuts),
            "mean_compacted_blocks_per_probe": _mean(r["n_compacted"] for r in rows), "max_compacted_blocks": max((c["n_compacted"] for c in cuts), default=None),
            "drops_build": sum(c["drops_build"] for c in last.values()), "drops_cut_total": sum(c["drops_cut"] for c in cuts),
            "cut_compactions": sum(c["n_cut_compact"] for c in cuts), "build_compactions": n_comp,
            "q_cut_compactions": sum(c["q_cut"]["compactions"] for c in cuts), "q_cut_subsampled": sum(c["q_cut"]["subsampled"] for c in cuts), "q_subsampled": sum(x["subsampled"] for x in q),
            "q_avail_mean": sum(x["q_avail_mean"] * x["compactions"] for x in q if x["compactions"]) / n_comp if n_comp else None,
            "q_used_mean": sum(x["q_used_mean"] * x["compactions"] for x in q if x["compactions"]) / n_comp if n_comp else None,
            "window_cut_mean": _mean(c["window_cut"] for c in cuts)}


WITHIN_POOL_NET_LOSS = 2


def primary_rule(split_ct, bar=C.BAR, slack=WITHIN_POOL_NET_LOSS):
    """Amendment 1. POSITIVE needs both: (a) the lower bound of the paired difference A4_CT - A4_K0 beyond the pool is above 0, and (b) over ALL
    within-pool probes (distance under 16,384, cuts after evictions included) K0-only correct - CT-only correct <= `slack` (at most two net losses).
    A preregistered descriptive screen whose role is to decide a held-out run, not confirmatory evidence (18 beyond-pool probes in 6 clusters).
    The A1 bar (75.6%, 719/951) is unchanged and printed next to it."""
    d = (split_ct["beyond"]["difference"] or {})
    w = split_ct["within"]["paired"]
    k0_only = w["n"] - w["both"] - w["neither"] - w["only_a"]
    guard = {"n": w["n"], "k0_only": k0_only, "ct_only": w["only_a"], "net_loss": k0_only - w["only_a"], "limit": slack}
    guard["ok"] = guard["net_loss"] <= slack
    lower_ok = bool(d) and d["lo"] > 0
    return {"beyond_pool_difference_vs_K0": d, "beyond_n": split_ct["beyond"]["paired"]["n"], "beyond_lower_bound_above_0": lower_ok, "within_pool_guard": guard,
            "primary": "POSITIVE" if lower_ok and guard["ok"] else "NOT_POSITIVE", "bar": bar}


def check_no_eviction(arm, cuts, rows, k0_cuts, k0_rows):
    """The invariant, enforced: at every cut where an arm evicted nothing, its state digest (K/V and beta of the full-attention layers, DeltaNet
    state) and the generated ids of every probe answered there equal A4_K0's. A cut that is evictionless in one and not the other, or a cut A4_K0 lacks,
    is a violation too. Exits non-zero naming the cut."""
    k0c = {(c["traj"], c["cut_index"]): c for c in k0_cuts}
    bad, quiet = [], set()

    def is_quiet(c, who, key):
        """Nothing evicted, compacted or dropped, from the plan's own counts; the digest must then be there (fail closed) and only then."""
        if any(c.get(f) is None for f in ("evicted", "compacted", "dropped")):
            bad.append(f"{who} cut {key}: no eviction counts in the cut record")
            return None
        q = c["evicted"] == 0 and c["compacted"] == 0 and c["dropped"] == 0
        if q != bool(c.get("digest")):
            bad.append(f"{who} cut {key}: eviction counts say {'no eviction' if q else 'eviction'} but the digest is {'missing' if q else 'present'}")
        return q

    for c in cuts:
        key = (c["traj"], c["cut_index"])
        o = k0c.get(key)
        if o is None:
            bad.append(f"{arm} cut {key}: A4_K0 has no such cut")
            continue
        q, q0 = is_quiet(c, arm, key), is_quiet(o, "A4_K0", key)
        if q is None or q0 is None:
            continue
        if q != q0:
            bad.append(f"{arm} cut {key}: nothing evicted in only one of {arm} and A4_K0")
        elif q:
            quiet.add(key)
            if c["digest"] != o["digest"]:
                bad.append(f"{arm} cut {key}: state digest differs from A4_K0")
    k0g = {r["id"]: r["gen_ids"] for r in k0_rows}
    for r in rows:
        if (r["traj"], r["cut_index"]) in quiet and r["gen_ids"] != k0g.get(r["id"]):
            bad.append(f"{arm} cut {(r['traj'], r['cut_index'])} probe {r['id']}: generated ids differ from A4_K0")
    if bad:
        sys.exit("A4 no-eviction invariant broken, refusing to report: " + "; ".join(bad))
    return {"cuts_compared": len(quiet), "probes_compared": sum(1 for r in rows if (r["traj"], r["cut_index"]) in quiet)}


def check_integrity(build, arm_rows, cuts, done, trajs, expect_cuts=None):
    """Done markers and cut records against the rows: one marker per expected trajectory, its n_rows equal to the rows of the build's arms, the rows
    of one attempt, and per arm and trajectory exactly the cut records of the cuts that have rows, all from the marker's attempt, and (with
    `expect_cuts`, trajectory -> the task file's cut indices of the universe) exactly the cuts the tasks have."""
    problems = []
    dt = [r["traj"] for r in done]
    for t in sorted(trajs):
        if dt.count(t) != 1:
            problems.append(f"done marker for {t}: {dt.count(t)}")
        mk = [r for r in done if r["traj"] == t]
        n = sum(1 for rows in arm_rows.values() for r in rows if r["traj"] == t)
        if mk and mk[0]["n_rows"] != n:
            problems.append(f"{t}: done marker says {mk[0]['n_rows']} rows, the files hold {n}")
        if mk and {r.get("attempt") for rows in arm_rows.values() for r in rows if r["traj"] == t} != {mk[0].get("attempt")}:
            problems.append(f"{t}: rows and done marker are from different attempts")
    for arm, rows in arm_rows.items():
        for t in sorted(trajs):
            want = sorted({r["cut_index"] for r in rows if r["traj"] == t})
            got = sorted(c["cut_index"] for c in cuts if c["arm"] == arm and c["traj"] == t)
            if got != want:
                problems.append(f"{arm} {t}: cut records {got} against the cuts with rows {want}")
            if expect_cuts is not None and want != sorted(expect_cuts[t]):
                problems.append(f"{arm} {t}: rows at cuts {want} against the task file's cuts {sorted(expect_cuts[t])}")
            mk = [r for r in done if r["traj"] == t]
            if mk and {c.get("attempt") for c in cuts if c["arm"] == arm and c["traj"] == t} - {mk[0].get("attempt")}:
                problems.append(f"{arm} {t}: cut records from a different attempt than the done marker")
    if problems:
        raise r3.IncompleteRun(f"build {build}: " + "; ".join(problems))


def integrity_for_build(d, build, T, trajs, expect_cuts=None):
    arm_rows = {a: [r for r in C.read_rows(os.path.join(d, a + ".jsonl")) if r["traj"] in T] for a in a1_a4.arms_of(build)}
    cuts = [c for c in C.read_rows(os.path.join(d, f"A4_cuts_{build}.jsonl")) if c["traj"] in T]
    done = [r for r in C.read_rows(os.path.join(d, f"done_{build}.jsonl")) if r["traj"] in T]
    check_integrity(build, arm_rows, cuts, done, trajs, expect_cuts)


def task_cuts(tasks, want):
    out = {}
    for t in tasks:
        if t["id"] in want:
            out.setdefault(t["traj"], set()).add(t["cut_index"])
    return out


def load_k0(d, T):
    return ([r for r in C.read_rows(os.path.join(d, "A4_K0.jsonl")) if r["traj"] in T],
            [c for c in C.read_rows(os.path.join(d, f"A4_cuts_{a1_a4.ARMS['A4_K0']['build']}.jsonl")) if c["arm"] == "A4_K0" and c["traj"] in T])


def compact_on_evict(d, H, tasks, full, T, want, vmap, full_by_id, partial=False):
    """The Amendment 1 section; arms whose row file is absent are left out (with `partial`, also the arms that are not complete)."""
    by = {t["id"]: t for t in tasks}
    cl = {i: by[i]["traj"] for i in want}
    trajs = {by[i]["traj"] for i in want}
    ec = task_cuts(tasks, want)
    k0_rows, k0_cuts = load_k0(d, T)
    out = {}
    for arm, spec in a1_a4.CT_ARMS.items():
        path = os.path.join(d, arm + ".jsonl")
        if not os.path.exists(path):
            if not partial and arm == "A4_CT":
                sys.exit("A4_CT.jsonl is missing: the full report needs the CT build")
            continue
        rows = [r for r in C.read_rows(path) if r["traj"] in T]
        route = [r for r in C.read_rows(os.path.join(d, arm + "_route.jsonl")) if r["traj"] in T] if arm in a1_a4.ROUTED else rows
        if partial and {r["id"] for r in rows} != set(want):
            out[arm] = {"incomplete": f"{len({r['id'] for r in rows})} of {len(want)} probes"}
            continue
        try:
            integrity_for_build(d, spec["build"], T, trajs, ec)
            r3.check_complete(arm, rows, route, want)
        except r3.IncompleteRun as e:
            if not partial or arm == "A4_CT":
                raise
            out[arm] = {"incomplete": str(e)}                                   # an optional build still running
            continue
        vals, cls, ids = values_of(H, tasks, full, rows, T)
        vm = dict(zip(ids, vals))
        cuts = [c for c in C.read_rows(os.path.join(d, f"A4_cuts_{spec['build']}.jsonl")) if c["arm"] == arm and c["traj"] in T]
        split = dict(zip(ids, pool_split(tasks, ids)))
        sv = split_vs_k0(vm, vmap["A4_K0"], split, cl)
        inv = check_no_eviction(arm, cuts, rows, k0_cuts, k0_rows)
        out[arm] = {**retention_cells(tasks, ids, vals, cls), **row_stats(rows), "compaction": ct_stats(cuts, rows), "vs_A4_K0": sv, "no_eviction_identity_vs_K0": inv,
                    "no_candidate_identity": no_candidate_identity(rows, full_by_id), "descriptive_vs_bar": a1_a3.final_verdict(cell(vals, cls)["retention"]),
                    "source_status": {k: sum(1 for r in rows if r["src_status"] == k) for k in sorted({r["src_status"] for r in rows})}}
        vmap[arm] = vm
        if arm in a1_a4.ROUTED:
            out[arm]["router"] = router_table(route, ("x_last",))
    if "vs_A4_K0" in out.get("A4_CT", {}):
        out["primary_rule"] = primary_rule(out["A4_CT"]["vs_A4_K0"])
        out["paired_vs_A4_CT"] = {a: paired(vmap[a], vmap["A4_CT"]) for a in out if a in vmap and a != "A4_CT"}
    return out


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
    by_id = {t["id"]: t for t in tasks}
    for build in sorted({s["build"] for s in a1_a4.ARMS.values()}):          # includes S0 (A4_K0): an absent file fails its done-marker count
        integrity_for_build(d, build, T, {by_id[i]["traj"] for i in want}, task_cuts(tasks, want))
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
    ct = compact_on_evict(d, H, tasks, full, T, want, vmap, full_by_id)
    if ct:
        rep["compact_on_evict"] = ct
    nxt = {"registered": REGISTERED}
    b0_ret = set(json.load(open(os.path.join(b0, "retained.json"))))
    pv, pc, pids = pool_retention(H, tasks, b0_ret, C.read_rows(os.path.join(b0, "POOL16K.jsonl")), T)
    nxt["POOL16K_same_trajectories"] = {**cell(pv, pc), "by_distance": C.retention_by_distance(tasks, pids, pv)}
    pmap = dict(zip(pids, pv))
    nxt["paired_vs_POOL16K"] = {a: paired(vmap[a], pmap) for a in vmap}
    for name, path in (("A3_W2", d3 and os.path.join(d3, "A3_W2.jsonl")), ("E3", d32 and os.path.join(d32, "E3.jsonl"))):
        if path and os.path.exists(path):
            v, c, i = values_of(H, tasks, full, C.read_rows(path), T)
            nxt[name] = {**cell(v, c), "by_distance": C.retention_by_distance(tasks, i, v)}
            nxt["paired_" + name] = {a: paired(vmap[a], dict(zip(i, v))) for a in vmap}
    rep["next_to"] = nxt
    print(json.dumps(rep, indent=1))
    with open(os.path.join(d, "a4_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


def partial(d, pilot):
    """As soon as A4_K0 and A4_CT are complete: the primary rule, with every other complete arm next to it. Writes <dir>/a4_partial.json."""
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(pilot, "FULL.jsonl"))
    a1_a3.check_reference(os.path.join(pilot, "FULL.jsonl"), a1_a3.PILOT_REF_MD5)
    T = set(list(H.load_trajs())[:a1_a3.PILOT_N])
    want = a1_a3.universe_ids(H, tasks, full, T)
    a1_a3.check_universe(want, a1_a3.PILOT_PROBES, "pilot reference")
    by = {t["id"]: t for t in tasks}
    k0_rows, _ = load_k0(d, T)
    if {r["id"] for r in k0_rows} != set(want):
        sys.exit(f"A4_K0 is not complete ({len({r['id'] for r in k0_rows})} of {len(want)} probes): no partial report yet")
    integrity_for_build(d, a1_a4.ARMS["A4_K0"]["build"], T, {by[i]["traj"] for i in want}, task_cuts(tasks, want))
    r3.check_complete("A4_K0", k0_rows, k0_rows, want)
    vals, cl, ids = values_of(H, tasks, full, k0_rows, T)
    vmap = {"A4_K0": dict(zip(ids, vals))}
    ct = compact_on_evict(d, H, tasks, full, T, want, vmap, {r["id"]: r for r in full}, partial=True)
    if "vs_A4_K0" not in ct.get("A4_CT", {}):
        sys.exit("A4_CT is not complete: no partial report yet")
    rep = {"partial": True, "expected_probes": len(want), "A4_K0": cell(vals, cl), "compact_on_evict": ct}
    print(json.dumps(rep, indent=1))
    with open(os.path.join(d, "a4_partial.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--partial":
        partial(*sys.argv[2:4])
    else:
        main(*sys.argv[1:6])

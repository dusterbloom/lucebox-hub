"""Report for diagnostics E3, E3SPEC, E2 (dir 1) and E5 (dir 2) against the pilot's FULL-CHK reference and shipping rows.
    python3 a1_e_report.py <e3e2_dir> <e5_dir> <pilot_dir> [<e6_dir> <e7_dir> <e8_dir> <e9_dir>]    -> prints JSON, writes <e3e2_dir>/e_report.json
Decides nothing (PREREG "Diagnostics E2, E3, E5 before A2" and "Diagnostics E6, E7"; E8: "Diagnostic E8").
E9 reads <e9_dir>/e9.jsonl (exact-mass router, PREREG "Diagnostic E9") and checks its FULL.jsonl md5 the same way.
E8 reads <e8_dir>/SHIP.jsonl and checks that <e8_dir>/FULL.jsonl is byte-equal to the pilot FULL.jsonl (md5)."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as C  # noqa: E402
import a1_e  # noqa: E402
from a1_pilot_report import by_type  # noqa: E402


def retention(H, tasks, full, rows, T):
    vals, cl, ids = C.retention_vectors(H, tasks, full, rows, T)
    return {"n": len(vals), "correct": sum(vals), "retention": C.boot_ci(vals, cl) if vals else None,
            "by_distance": C.retention_by_distance(tasks, ids, vals), "by_type": by_type(tasks, ids, vals)}


def e2_tables(rows):
    el = [r for r in rows if r["eligible"]]
    far = [r for r in el if r["distance_tok"] > a1_e.FAR]
    out = {"probes": len(rows), "eligible": len(el), "not_located": sum(1 for r in rows if not r["located"]),
           "source_in_tail": sum(1 for r in rows if r["located"] and not r["src_blocks"]), "far_eligible": len(far)}
    for name in a1_e.ROUTERS:
        out[name] = {"all": a1_e.recall_table([r["ranks"][name] for r in el]),
                     "far_over_16K": a1_e.recall_table([r["ranks"][name] for r in far])}
    out["mean_blocks_per_cut"] = sum(r["n_blocks"] for r in el) / len(el) if el else None
    return out


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def e9_tables(rows):
    """E9: recall tables per router over every block (E2's definition), the over-16K subset, the exact-mass rank against the a_last rank
    per probe, and the share of probes whose source block is inside A3's fetched k, per W and router."""
    el = [r for r in rows if r.get("eligible")]
    far = [r for r in el if r["distance_tok"] > a1_e.FAR]
    out = {"probes": len(rows), "eligible": len(el), "far_eligible": len(far)}
    for name in sorted({k for r in el for k in r["ranks"]}):
        out[name] = {"all": a1_e.recall_table([r["ranks"][name] for r in el]),
                     "far_over_16K": a1_e.recall_table([r["ranks"][name] for r in far])}
    for agg in ("last", "max"):
        a, x = [r["ranks"]["a_" + agg] for r in el], [r["ranks"]["x_" + agg] for r in el]
        out["x_vs_a_" + agg] = {"x_better": sum(i < j for i, j in zip(x, a)), "equal": sum(i == j for i, j in zip(x, a)),
                                "x_worse": sum(i > j for i, j in zip(x, a))}
    out["per_probe"] = [{"id": r["id"], "dist": r["distance_tok"], "a_last": r["ranks"]["a_last"], "x_last": r["ranks"]["x_last"],
                         "x_max": r["ranks"]["x_max"]} for r in el]
    bud = {}
    for wb in sorted({w for r in el for w in r.get("budget", {})}, key=int):
        rs = [r["budget"][wb] for r in el if wb in r["budget"]]
        inc = [b for b in rs if b["src_cand"]]
        bud["W" + wb] = {"probes": len(rs), "source_in_candidates": len(inc), "mean_k": sum(b["k"] for b in rs) / len(rs),
                         "inside_k": {r: [sum(b["in_k"][r] for b in inc), len(inc)] for r in ("a_last", "x_last", "x_max", "b_last")} if inc else None}
    out["budget"] = bud
    return out


def main(d32, d5, pilot, d6=None, d7=None, d8=None, d9=None):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(pilot, "FULL.jsonl"))
    T = set(list(H.load_trajs())[:6])
    rep = {"pilot_ship_ratio_0.10": retention(H, tasks, full, C.read_rows(os.path.join(pilot, "SHIP.jsonl")), T)}
    for arm in ("E3", "E3SPEC"):
        rows = C.read_rows(os.path.join(d32, arm + ".jsonl"))
        rep[arm] = {**retention(H, tasks, full, rows, T), "located_share": [sum(1 for r in rows if r["located"]), len(rows)],
                    "mean_blocks_swapped": sum(len(r["swapped_blocks"]) for r in rows) / len(rows) if rows else None}
    if os.path.exists(os.path.join(d5, "SHIP.jsonl")):
        rep["E5_ratio_0.25"] = retention(H, tasks, full, C.read_rows(os.path.join(d5, "SHIP.jsonl")), T)
    for name, d in (("E6_repeat_queries", d6), ("E7_system_prefix_exact", d7)):
        f = os.path.join(d, name[:2] + ".jsonl") if d else None
        if f and os.path.exists(f):
            rep[name] = retention(H, tasks, full, C.read_rows(f), T)
            if name[:2] == "E7":
                pr = [r for r in C.read_rows(os.path.join(d, "progress.jsonl")) if "exact_tokens" in r]
                rep[name]["exact_tokens_per_trajectory"] = {r["traj"][-8:]: r["exact_tokens"] for r in pr}
    if d8 and os.path.exists(os.path.join(d8, "SHIP.jsonl")):
        ref = md5(os.path.join(pilot, "FULL.jsonl"))
        assert md5(os.path.join(d8, "FULL.jsonl")) == ref, "E8 reference is not the pilot FULL.jsonl"
        rep["E8_select_only"] = {**retention(H, tasks, full, C.read_rows(os.path.join(d8, "SHIP.jsonl")), T), "reference_md5": ref}
    if d9 and os.path.exists(os.path.join(d9, "e9.jsonl")):
        assert md5(os.path.join(d9, "FULL.jsonl")) == md5(os.path.join(pilot, "FULL.jsonl")), "E9 reference is not the pilot FULL.jsonl"
        rep["E9_exact_mass_router"] = e9_tables(C.read_rows(os.path.join(d9, "e9.jsonl")))
    rep["E2"] = e2_tables(C.read_rows(os.path.join(d32, "e2.jsonl")))
    sc = os.path.join(d32, "ship_check.json")
    rep["ship_check"] = json.load(open(sc)) if os.path.exists(sc) else None
    print(json.dumps(rep, indent=1))
    with open(os.path.join(d32, "e_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:8])

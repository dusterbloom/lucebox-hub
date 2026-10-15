"""A1 pilot report (Amendment 2): reference yield, retention with its indicative interval, retention by distance and by
probe type, determinism, time and memory, and B0 POOL16K on the same trajectories for reading. Decides nothing.

    python3 a1_pilot_report.py <out_dir> <b0_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as C  # noqa: E402


def pool_retention(H, tasks, b0_retained, pool_rows, trajs):
    """B0's own retention for POOL16K: correct among the probes B0 full KV got right, on `trajs`."""
    P = {r["id"]: r for r in pool_rows}
    vals, cl, ids = [], [], []
    for t in tasks:
        if t["id"] in b0_retained and t["traj"] in trajs and t["id"] in P:
            r = P[t["id"]]
            vals.append(int("error" not in r and H.probes.score_exact(r.get("content") or "", t["answer"])))
            cl.append(t["traj"])
            ids.append(t["id"])
    return vals, cl, ids


def by_type(tasks, ids, vals):
    by = {t["id"]: t for t in tasks}
    out = {}
    for i, v in zip(ids, vals):
        k = by[i].get("probe_type") or by[i].get("type") or "?"
        s = out.setdefault(k, [0, 0])
        s[0] += v
        s[1] += 1
    return out


def main(out, b0):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    full = C.read_rows(os.path.join(out, "FULL.jsonl"))
    ship = C.read_rows(os.path.join(out, "SHIP.jsonl"))
    trajs = json.load(open(os.path.join(out, "summary.json")))["trajectories"]
    T = set(trajs)
    ok, att = C.reference_yield(H, tasks, full, T)
    vals, cl, ids = C.retention_vectors(H, tasks, full, ship, T)
    ret = C.retained_probe_ids(H, tasks, full)
    missing = sorted(i for i in ret if i not in {r["id"] for r in ship})
    b0_ret = set(json.load(open(os.path.join(b0, "retained.json"))))
    pv, pc, pids = pool_retention(H, tasks, b0_ret, C.read_rows(os.path.join(b0, "POOL16K.jsonl")), T)
    both = set(ids) & set(pids)
    S = dict(zip(ids, vals))
    Pm = dict(zip(pids, pv))
    rep_rows = [r for r in full + ship if r["traj"] == trajs[0] and "repeat_equal" in r]
    prog = [json.loads(l) for l in open(os.path.join(out, "progress.jsonl"))]
    rep = {
        "trajectories": len(trajs),
        "reference_yield": [ok, att],
        "retention": C.boot_ci(vals, cl) if vals else None,
        "retention_by_distance": C.retention_by_distance(tasks, ids, vals),
        "retention_by_type": by_type(tasks, ids, vals),
        "retained_without_ship_row": missing,
        "b0_pool16k_same_trajs": C.boot_ci(pv, pc) if pv else None,
        "b0_pool16k_by_distance": C.retention_by_distance(tasks, pids, pv),
        "same_probe": {"n": len(both), "ship": sum(S[i] for i in both), "pool16k": sum(Pm[i] for i in both)},
        "determinism": {"rows": len(rep_rows), "equal": sum(1 for r in rep_rows if r["repeat_equal"])},
        "seconds": {a: {"build": round(sum(p["build_s"] for p in prog if p["arm"] == a), 1),
                        "answer": round(sum(p["answer_s"] for p in prog if p["arm"] == a), 1),
                        "wall": round(sum(p["wall_s"] for p in prog if p["arm"] == a), 1)} for a in ("FULL", "SHIP")},
        "peak_gib": {a: {"alloc": max(p["peak_alloc_gib"] for p in prog if p["arm"] == a),
                         "reserved": max(p["peak_reserved_gib"] for p in prog if p["arm"] == a)} for a in ("FULL", "SHIP")},
    }
    print(json.dumps(rep, indent=1))
    with open(os.path.join(out, "pilot_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

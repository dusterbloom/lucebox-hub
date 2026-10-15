"""D1 identity diagnostic report (outside A1): the shipping-arm generator keeping every slot (ratio 1.0), scored like the pilot's
retention (correct among the probes FULL-CHK got right). SHIP.jsonl in the D1 directory holds the D1 rows.
    python3 a1_d1_report.py <d1_dir>      -> prints JSON, writes <d1_dir>/d1_report.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as C  # noqa: E402


def main(out):
    H = C.Harness(C.default_paths())
    tasks = H.load_tasks()
    by = {t["id"]: t for t in tasks}
    full = C.read_rows(os.path.join(out, "FULL.jsonl"))
    d1 = C.read_rows(os.path.join(out, "SHIP.jsonl"))
    F, D = {r["id"]: r for r in full}, {r["id"]: r for r in d1}
    trajs = list(dict.fromkeys(r["traj"] for r in full))
    vals, cl, ids = C.retention_vectors(H, tasks, full, d1, set(trajs))
    per = {}
    for i, v, c in zip(ids, vals, cl):
        s = per.setdefault(c, [0, 0])
        s[0] += v
        s[1] += 1
    misses = [{"id": i, "gold": by[i]["answer"], "full": F[i].get("content"), "d1": D[i].get("content")}
              for i, v in zip(ids, vals) if not v]
    prog = [json.loads(l) for l in open(os.path.join(out, "progress.jsonl"))]
    rep = {"retention_overall": [sum(vals), len(vals)], "per_trajectory": per, "misses": misses,
           "units": [{k: p.get(k) for k in ("traj", "arm", "n_rows", "build_s", "answer_s", "wall_s", "peak_alloc_gib",
                                            "peak_reserved_gib")} for p in prog],
           "reference_correct_without_d1_row": sorted(C.retained_probe_ids(H, tasks, full) - set(D))}
    json.dump(rep, open(os.path.join(out, "d1_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

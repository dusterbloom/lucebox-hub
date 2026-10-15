"""Offline policy simulator for routed exact fetch (A3 family), from rows that already exist: E2 router ranks, E3 oracle answers,
the pilot shipping answers and B0 POOL16K. No GPU, no new model calls; an estimate, not a measurement.
    python3 -I a3_sim.py <E3E2 dir> <pilot dir> <POOL16K.jsonl> <tasks.jsonl>
Per probe: the source block is exact if it lies in the system prefix P, the recent window W, or the router's top k candidates.
Predicted answer: all blocks exact -> correct (the state equals the reference); source exact -> E3's answer; else the shipping answer.
The router rank is E2's a_last rank over ALL sealed blocks; A3 ranks candidates only, so the true rank is between
rank - (exact blocks) and rank. 'lo' uses the rank as is (pessimistic), 'hi' assumes every exact block ranked above the source."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agentic_eval"))
import probes  # noqa: E402

BLK, SLOTS, CAP, P_BLOCKS, MAX_NEW = 640, 64, 16384, 10, 64   # PREREG A3: 5,921-token prefix = 10 blocks; 64-token answers
BINS = [(0, 4096), (4096, 8192), (8192, 16384), (16384, 10 ** 9)]


def rows(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def plan(nb, tail, reserve, W, k_cap=None):
    """(exact set without fetch, candidates, k) under the A3 row budget; W=None means the window fills the budget after k_cap."""
    P = set(range(min(P_BLOCKS, nb)))
    if W is None:                                   # hybrid: reserve k_cap fetch blocks, then grow the window as far as rows allow
        W = 0
        while W < nb:
            E = len(P | set(range(nb - W - 1, nb)))
            k = min(k_cap, nb - E)
            if BLK * (E + k) + SLOTS * (nb - E - k) + tail + reserve > CAP:
                break
            W += 1
    ex = P | set(range(nb - W, nb))
    cand = nb - len(ex)
    k = 0
    while k < cand and BLK * (len(ex) + k + 1) + SLOTS * (nb - len(ex) - k - 1) + tail + reserve <= CAP:
        k += 1
    if k_cap is not None:
        k = min(k, k_cap)
    return ex, cand, k


def main(d, pilot, pool_path, tasks_path):
    tasks = {t["id"]: t for t in map(json.loads, open(tasks_path))}
    e2, e3, ship, pool = rows(os.path.join(d, "e2.jsonl")), rows(os.path.join(d, "E3.jsonl")), rows(os.path.join(pilot, "SHIP.jsonl")), rows(pool_path)

    def ok(r):
        return int(r is not None and "error" not in r and probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]))

    ids = [i for i in e3 if i in e2 and e2[i]["located"]]
    q = {}                                          # PREREG A3 reserve: longest question suffix of the cut + max_new
    for i in ids:
        key = (e2[i]["traj"], e2[i]["ctx_len"])
        q[key] = max(q.get(key, 0), e3[i]["prompt_tokens"] - e3[i]["ctx_len"])
    policies = [("SHIP (all compacted)", "ship"), ("E3 oracle", "oracle"), ("A3 W=2", (2, None)), ("A3 W=6", (6, None))]
    policies += [(f"hybrid k={k}, W fills", (None, k)) for k in (2, 4, 8)]
    print(f"probes {len(ids)} (E3 rows with a located source); POOL16K rows for {sum(i in pool for i in ids)}")
    print(f"{'policy':24} {'est. correct lo-hi':>20} {'0-4K':>9} {'4-8K':>9} {'8-16K':>9} {'>16K':>9}  mean k")
    for name, pol in policies:
        lo = hi = 0
        binlo, n_bin, ks = {b: 0 for b in BINS}, {b: 0 for b in BINS}, []
        for i in ids:
            r, dist = e2[i], tasks[i]["distance_tok"]
            b = next(b for b in BINS if b[0] <= dist < b[1])
            n_bin[b] += 1
            if pol == "ship":
                v = (ok(ship.get(i)),) * 2
            elif pol == "oracle":
                v = (ok(e3[i]),) * 2
            else:
                nb, tail = r["n_blocks"], r["ctx_len"] - BLK * r["n_blocks"]
                ex, cand, k = plan(nb, tail, q[(r["traj"], r["ctx_len"])] + MAX_NEW, *pol)
                ks.append(k)
                src = set(r["src_blocks"])
                if k >= cand:                       # every block exact: the reference state
                    v = (1, 1)
                elif not src or src <= ex:          # source in the open tail, P or W
                    v = (ok(e3[i]),) * 2
                else:
                    rank = r["ranks"]["a_last"]
                    v = tuple(ok(e3[i]) if rr <= k else ok(ship.get(i)) for rr in (rank, max(1, rank - len(ex))))
            lo, hi = lo + v[0], hi + v[1]
            binlo[b] += v[0]
        mk = f"{sum(ks) / len(ks):6.1f}" if ks else "     -"
        cells = " ".join(f"{binlo[b]:>3}/{n_bin[b]:<3}  " for b in BINS)
        print(f"{name:24} {lo:>4}-{hi:<4} ({100 * lo / len(ids):4.1f}-{100 * hi / len(ids):4.1f}%) {cells} {mk}")
    pl = [i for i in ids if i in pool]
    print(f"POOL16K actual on its {len(pl)} probes: {sum(ok(pool[i]) for i in pl)} ({100 * sum(ok(pool[i]) for i in pl) / len(pl):.1f}%)")


if __name__ == "__main__":
    main(*sys.argv[1:5])

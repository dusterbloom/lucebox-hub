"""Post-hoc (not preregistered): which residency feature predicts a correct answer, and what other pool policies would score.
Residency is replayed from E1's dump (scores_last, LRU resident set) as in ../prefill_reselect_sim; answers are this run's OFF (LRU) and ON
(reselect, W=4). A policy's expected correct = sum over probes of P(correct | feature cell), calibrated on OFF+ON jointly.
    python3 -I calib.py <worktree>"""
import collections
import json
import lzma
import os
import sys

W_TREE = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(W_TREE, "optimizations/kvflash/evidence/agentic/E1_router_recall")
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

BLOCKS, CT, OLD = 256, 64, 16384
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
recs = {r["id"]: r for r in json.load(open(os.path.join(E1, "summary.per_probe.json")))}
rows = [json.loads(l) for l in lzma.open(os.path.join(E1, "rows_QK16K.jsonl.xz"), "rt")]
dumps = [json.loads(l) for l in lzma.open(os.path.join(E1, "qk_dump.jsonl.xz"), "rt")]
matched, di = {}, 0
for r in rows:
    if "error" in r or r.get("prompt_tokens") is None:
        continue
    while di < len(dumps) and dumps[di]["prompt_len"] != r["prompt_tokens"]:
        di += 1
    if di < len(dumps):
        matched[r["id"]] = dumps[di]
        di += 1
ans = {a: {r["id"]: int(probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]))
           for r in map(json.loads, open(os.path.join(D, a, "QK16KND.jsonl"))) if r.get("kind") == "probe"} for a in ("off", "on")}
ids = [i for i in recs if i in matched and i in ans["on"]]


def lru(d):
    return set(d["resident"]) | {d["committed"] // CT}


def reselect(d, W, dil=0, budget=BLOCKS):
    """Protect chunk 0 and the last W chunks; fill the rest by score, a chunk scoring as the max over +-dil neighbours."""
    n, cur, s = d["n_chunks"], d["committed"] // CT, d["scores_last"]
    prot = {0} | {c for c in range(n + 1) if c > cur - 1 - W}
    sd = [max(s[max(0, c - dil):c + dil + 1]) for c in range(n)] if dil else s
    rest = sorted((c for c in range(n) if c not in prot), key=lambda c: (-sd[c], -c))
    return prot | set(rest[:max(0, budget - len(prot))])


def feats(i, res):
    r = recs[i]
    fa, fb = r["span"]["fact"]
    ma, mb = r["span"]["msg"]
    n = matched[i]["n_chunks"]
    fact = all(c in res for c in range(fa // CT, fb // CT + 1))
    lead = all(c in res for c in range(ma // CT, fb // CT + 1))           # message start through the fact, contiguous
    mc = range(ma // CT, min(mb // CT, n - 1) + 1)
    frac = sum(c in res for c in mc) / len(mc)
    near = all(c in res for c in range(max(0, fa // CT - 2), min(n - 1, fb // CT + 2) + 1))  # fact +-2 chunks
    return (fact, near, lead, "all" if frac == 1 else ">=.5" if frac >= .5 else "<.5")


cell = collections.defaultdict(lambda: [0, 0])
for i in ids:
    for a, res in (("off", lru(matched[i])), ("on", reselect(matched[i], 4))):
        c = cell[feats(i, res)]
        c[0] += 1
        c[1] += ans[a][i]
print("feature cell (fact chunk, fact+-2, msg start..fact, msg share) -> n, correct, rate  [OFF+ON pooled]")
for k in sorted(cell, key=str):
    n, o = cell[k]
    print(f"  {str(k):38} n {n:>4} ok {o:>4} {100 * o / n:5.1f}%")
P = {k: o / n for k, (n, o) in cell.items()}
base = sum(cell[k][1] for k in cell) / sum(cell[k][0] for k in cell)


def score(pol):
    out = {"within": 0.0, "old": 0.0}
    for i in ids:
        out["old" if tasks[i]["distance_tok"] > OLD else "within"] += P.get(feats(i, pol(matched[i])), base)
    return out


print(f"\nprobes {len(ids)}; measured: OFF within {sum(ans['off'][i] for i in ids if tasks[i]['distance_tok'] <= OLD)} "
      f"old {sum(ans['off'][i] for i in ids if tasks[i]['distance_tok'] > OLD)}; ON within "
      f"{sum(ans['on'][i] for i in ids if tasks[i]['distance_tok'] <= OLD)} old {sum(ans['on'][i] for i in ids if tasks[i]['distance_tok'] > OLD)}")
print(f"{'policy':34} {'within/744':>10} {'old/207':>8} {'total':>7}")
pols = [("LRU (OFF)", lru), ("reselect W=4 (ON)", lambda d: reselect(d, 4))]
pols += [(f"reselect W={w} dil={x}", (lambda w, x: lambda d: reselect(d, w, x))(w, x))
         for w in (128, 160, 192, 208, 224, 240) for x in (0, 1, 2, 4)]
for name, pol in pols:
    s = score(pol)
    print(f"{name:34} {s['within']:>10.0f} {s['old']:>8.0f} {s['within'] + s['old']:>7.0f}")


# Model 2: the cell also carries the share of the last-16K chunks left resident (recent context), in fifths.
def recent_share(d, res):
    cur = d["committed"] // CT
    rec = range(max(0, cur - BLOCKS + 1), cur + 1)
    return sum(c in res for c in rec) / len(rec)


def feats2(i, res):
    f = feats(i, res)
    return (f[0] and f[1] and f[2], f[3] == "all", min(4, int(recent_share(matched[i], res) * 5)))


cell2 = collections.defaultdict(lambda: [0, 0])
for i in ids:
    for a, res in (("off", lru(matched[i])), ("on", reselect(matched[i], 4))):
        c = cell2[feats2(i, res)]
        c[0] += 1
        c[1] += ans[a][i]
P2 = {k: o / n for k, (n, o) in cell2.items()}
print("\nmodel 2 cells (fact+lead+-2, msg all, recent-share fifth) -> n, rate")
for k in sorted(cell2, key=str):
    print(f"  {str(k):24} n {cell2[k][0]:>4} {100 * cell2[k][1] / cell2[k][0]:5.1f}%")
print(f"{'policy':34} {'within/744':>10} {'old/207':>8} {'total':>7}   (model 2)")
for name, pol in pols:
    s = {"within": 0.0, "old": 0.0}
    for i in ids:
        s["old" if tasks[i]["distance_tok"] > OLD else "within"] += P2.get(feats2(i, pol(matched[i])), base)
    print(f"{name:34} {s['within']:>10.0f} {s['old']:>8.0f} {s['within'] + s['old']:>7.0f}")

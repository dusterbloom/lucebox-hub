"""QuALITY analysis: python3 -I analyze_quality.py <root> <dir-glob> [ref_method_substring] [margin]
Merges all result JSONs under root/<dir-glob>/, per method: n questions, accuracy, cluster-bootstrap (articles) 95% CI,
mean GPU-s per article (compaction + query generation). Paired diff of every method vs the reference (matched questions,
article-cluster bootstrap, 10,000 resamples) and an exact McNemar p."""
import glob, json, random, sys
from math import comb

root, pat = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None
data = {}   # method -> {qid: (article, correct)}
cost = {}   # method -> [seconds per article]
for f in glob.glob(f"{root}/{pat}/*_2026*.json"):
    for r in json.load(open(f))["results"]:
        m = r["method"] if not __import__("os").environ.get("KEY_DIR") else f.split("/")[-2].replace("_t0.1","") + ":" + r["method"].replace("highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy","HA").replace("omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy","OMPF")
        cost.setdefault(m, []).append(r.get("compaction_time", 0) + r.get("query_generation_time", 0))
        for q in r["qa_results"]["results_per_question"]:
            data.setdefault(m, {})[q["question_id"]] = (r["article_id"], int(bool(q["is_correct"])))
random.seed(0)

def boot(vals_by_art, B=10000):
    arts = list(vals_by_art)
    out = []
    for _ in range(B):
        s = n = 0
        for a in random.choices(arts, k=len(arts)):
            s += sum(vals_by_art[a]); n += len(vals_by_art[a])
        out.append(s / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]

def by_art(pairs):
    d = {}
    for a, v in pairs:
        d.setdefault(a, []).append(v)
    return d

print(f"{'method':58s} nq  n_art  acc%   95%CI           GPU-s/article")
for m in sorted(data):
    v = list(data[m].values()); d = by_art(v)
    lo, hi = boot(d)
    print(f"{m:58s} {len(v):4d} {len(d):3d} {100*sum(x for _, x in v)/len(v):6.1f} [{100*lo:5.1f},{100*hi:5.1f}] {sum(cost[m])/len(cost[m]):8.1f}")
if ref:
    rm = [m for m in data if ref in m]
    if rm:
        rm = rm[0]
        for m in sorted(data):
            if m == rm:
                continue
            ids = sorted(set(data[m]) & set(data[rm]))
            d = {}
            for q in ids:
                a = data[m][q][0]
                d.setdefault(a, []).append(data[m][q][1] - data[rm][q][1])
            diffs = [x for v in d.values() for x in v]
            lo, hi = boot(d)
            b = sum(1 for x in diffs if x > 0); c = sum(1 for x in diffs if x < 0)
            n = b + c
            p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
            print(f"paired {m} - {rm}: n={len(ids)} diff {100*sum(diffs)/len(diffs):+.2f} pts "
                  f"95%CI [{100*lo:+.2f},{100*hi:+.2f}] discordant {n} (+{b}/-{c}) McNemar p={p:.3f}")

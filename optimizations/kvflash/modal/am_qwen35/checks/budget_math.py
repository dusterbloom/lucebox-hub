"""CPU budget math reimplemented from reading am-compaction base.py apply_max_ratio_cap + per_layer_head.py.
Pure; no repo code is imported or run.

usage: python3 -I budget_math.py <budget.json> <cell.jsonl> <target>
"""
import json, sys, collections


def cap(props, max_ratio, target_ratio, total_heads):
    if max_ratio >= 1.0 and target_ratio * total_heads <= 1.0:
        return props
    mp = max_ratio / (target_ratio * total_heads)
    if max(props.values()) <= mp:
        return props
    r = dict(props)
    for _ in range(total_heads + 1):
        ex = 0.0
        un = 0.0
        capped = set()
        for k, p in r.items():
            if p > mp:
                ex += p - mp
                capped.add(k)
            else:
                un += p
        if ex == 0:
            break
        if un == 0:
            for k in r:
                r[k] = min(r[k], mp)
            break
        for k in r:
            r[k] = mp if k in capped else r[k] + r[k] / un * ex
    return r


def budgets(path, target, seq_len, n_art, max_ratio=0.75):
    d = json.load(open(path))
    keep = seq_len - n_art
    T = max(1, int(n_art * target))
    H = len(d)
    p = cap(d, max_ratio, T / n_art, H)
    b = {k: int(v * T * H) for k, v in p.items()}
    layers = collections.defaultdict(list)
    for k, v in b.items():
        l, _ = k[1:].split("H")
        layers[int(l)].append(v)
    return T, keep, H, layers


def main(path, jsonl, target):
    seen = {}
    for line in open(jsonl):
        x = json.loads(line)
        seen[x["article_id"]] = x
    out = []
    for a, x in seen.items():
        T, keep, H, layers = budgets(path, target, x["seq_len"], x["article_tokens"])
        nl = len(layers)
        nh = len(next(iter(layers.values())))
        padded = sum(max(v) for v in layers.values()) / nl + keep
        eff = sum(sum(v) for v in layers.values()) / (nl * nh) + keep
        tot = sum(sum(v) for v in layers.values())
        out.append((x["article_tokens"], x.get("full_layer_len_mean"), padded, eff, tot, T * H,
                    min(min(v) for v in layers.values()), max(max(v) for v in layers.values()), nl, nh))
    n = len(out)
    art = sum(o[0] for o in out)
    print("n", n, "layers", out[0][8], "heads/layer", out[0][9], "mean article", art / n)
    print("jsonl len mean", sum(o[1] for o in out if o[1]) / n, "recomputed padded", sum(o[2] for o in out) / n,
          "max abs diff", max(abs(o[1] - o[2]) for o in out if o[1]))
    print("padded/article", sum(o[2] for o in out) / art)
    print("effective attended per head (+keep) mean", sum(o[3] for o in out) / n, "ratio", sum(o[3] for o in out) / art)
    print("sum head budgets/(T*H)", sum(o[4] for o in out) / sum(o[5] for o in out),
          "head budget min/max", min(o[6] for o in out), max(o[7] for o in out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))

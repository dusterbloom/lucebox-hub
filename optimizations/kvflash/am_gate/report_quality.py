#!/usr/bin/env python3
"""M7 QuALITY gate report: engine arms vs the R1(e) Modal rows. Stdlib only.

usage: report_quality.py <orig.jsonl> <auto.jsonl> <R1e/sel dir> [out.md]
"""
import json, random, statistics as st, sys


def load(p):
    return [json.loads(l) for l in open(p)]


def key(r):
    return (r["article_index"], r["qi"])


def acc(rows):
    return sum(r["correct"] for r in rows) / len(rows)


def med(xs):
    xs = [x for x in xs if x is not None]
    return (st.median(xs), min(xs), max(xs)) if xs else (None, None, None)


def cluster_bootstrap(a_rows, b_rows, n=10000, seed=0):
    """D = acc(b) - acc(a) over questions, resampling ARTICLES with replacement (same draw for both arms)."""
    by = {}
    for r in a_rows:
        by.setdefault(r["article_index"], {})[r["qi"]] = [r["correct"], None]
    for r in b_rows:
        by[r["article_index"]][r["qi"]][1] = r["correct"]
    arts = sorted(by)
    tot = {a: (len(by[a]), sum(v[0] for v in by[a].values()), sum(v[1] for v in by[a].values())) for a in arts}
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        picks = [arts[rng.randrange(len(arts))] for _ in arts]
        q = sum(tot[a][0] for a in picks)
        ds.append((sum(tot[a][2] for a in picks) - sum(tot[a][1] for a in picks)) / q)
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def main():
    orig, auto, refdir = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    out = []
    P = out.append
    mo = {key(r): r for r in load(f"{refdir}/ORIG__t0.1__sel.jsonl")}
    mf = {key(r): r for r in load(f"{refdir}/F-UNI-LOCAL__t0.1__sel.jsonl")}
    eo = {key(r): r for r in orig}
    ef = {key(r): r for r in auto}
    keys = sorted(mo)
    miss = [k for k in keys if k not in eo or k not in ef]
    P(f"questions: modal {len(keys)}, orig_engine {len(eo)}, F_engine {len(ef)}, missing {len(miss)}")
    keys = [k for k in keys if k in eo and k in ef]
    ro = [eo[k] for k in keys]
    rf = [ef[k] for k in keys]
    n = len(keys)
    P("")
    P("| arm | correct | accuracy |")
    P("|---|---|---|")
    P(f"| orig_engine | {sum(r['correct'] for r in ro)}/{n} | {acc(ro):.4f} |")
    P(f"| F_engine (auto 0.10, P=0 W=0) | {sum(r['correct'] for r in rf)}/{n} | {acc(rf):.4f} |")
    P(f"| orig_Modal | {sum(mo[k]['correct'] for k in keys)}/{n} | {sum(mo[k]['correct'] for k in keys)/n:.4f} |")
    P(f"| F_Modal (F-UNI-LOCAL) | {sum(mf[k]['correct'] for k in keys)}/{n} | {sum(mf[k]['correct'] for k in keys)/n:.4f} |")
    lo, hi = cluster_bootstrap(ro, rf)
    d = acc(rf) - acc(ro)
    P("")
    P(f"D_engine = F_engine - orig_engine = {d:+.4f} ({sum(r['correct'] for r in rf) - sum(r['correct'] for r in ro):+d} questions), "
      f"95% cluster bootstrap (20 articles, 10000 resamples, seed 0) [{lo:+.4f}, {hi:+.4f}]")
    mro = [mo[k] for k in keys]
    mrf = [mf[k] for k in keys]
    lo2, hi2 = cluster_bootstrap(mro, mrf)
    P(f"D_Modal  = F_Modal - orig_Modal   = {acc(mrf) - acc(mro):+.4f}, [{lo2:+.4f}, {hi2:+.4f}]")
    P("")
    ag_f = sum(ef[k]["parsed"] == mf[k]["parsed"] for k in keys)
    ag_o = sum(eo[k]["parsed"] == mo[k]["parsed"] for k in keys)
    P(f"per-question agreement of the parsed choice: F_engine vs F_Modal {ag_f}/{n} ({ag_f/n:.3f}); "
      f"orig_engine vs orig_Modal {ag_o}/{n} ({ag_o/n:.3f})")
    ag_fo = sum(ef[k]["parsed"] == eo[k]["parsed"] for k in keys)
    P(f"agreement F_engine vs orig_engine {ag_fo}/{n}; F_Modal vs orig_Modal {sum(mf[k]['parsed'] == mo[k]['parsed'] for k in keys)}/{n}")
    P("")
    P("## Parity and safety checks")
    for name, rows in (("orig_engine", ro), ("F_engine", rf)):
        P(f"- {name}: prefix_len==0 and disk_hit false and restore false in "
          f"{sum(r.get('prefix_len') == 0 and r.get('disk_hit') is False and r.get('restore') is False for r in rows)}/{len(rows)}; "
          f"server prompt_tokens == HF tokens in {sum(r['prompt_tokens'] == r['hf_prompt_len'] for r in rows)}/{len(rows)}; "
          f"non-empty nan_layers in {sum(bool(r.get('nan_layers')) for r in rows)}; "
          f"unparsed answers {sum(r['parsed'] is None for r in rows)}")
    P(f"- F_engine compaction: sealed blocks per prompt min/max {min(r['am_sealed'] for r in rf)}/{max(r['am_sealed'] for r in rf)}; "
      f"compacted==sealed in {sum(r['am_compacted'] == r['am_sealed'] for r in rf)}/{n}; failed blocks {sum(r['am_failed'] for r in rf)}; "
      f"skipped blocks {sum(r['am_skipped'] for r in rf)}")
    exp = [(r["ctx_len"] // 640) for r in rf]
    P(f"- sealed context blocks per question equal the reference's ctx_len//640 in {sum(r['am_compacted'] == e for r, e in zip(rf, exp))}/{n}")
    P("")
    P("## Cost (server-side, per question)")
    for name, rows in (("orig_engine", ro), ("F_engine", rf)):
        P(f"- {name}: prefill_s median/min/max {med([r.get('prefill_s') for r in rows])}; "
          f"decode tok/s median/min/max {med([r.get('decode_tps') for r in rows])}; wall_s {med([r['wall_s'] for r in rows])}")
    P(f"- compaction per sealed block (ms, total = select + fit + io + publish): median/min/max "
      f"{med([r['am_per_block_ms'] for r in rf])}; select {med([r['am_select_ms'] / r['am_compacted'] for r in rf])}; "
      f"fit {med([r['am_fit_ms'] / r['am_compacted'] for r in rf])}; io {med([r['am_io_ms'] / r['am_compacted'] for r in rf])}")
    txt = "\n".join(out)
    print(txt)
    if len(sys.argv) > 4:
        open(sys.argv[4], "w").write(txt + "\n")


if __name__ == "__main__":
    main()

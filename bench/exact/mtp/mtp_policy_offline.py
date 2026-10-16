#!/usr/bin/env python3
"""Offline optimum of the MTP verify width from LUCE_MTP_TRACE=1 server logs ([qwen4exp-mtp-step] lines, one per
verify step: k, accepted, ms). Per workload (requests are matched to workloads by their prompt_tokens in the
[qwen4exp-mtp] summary order given on the command line):
  - cost curve: median step ms per width over every traced arm, plus the AR step from an AR arm's decode tok/s;
  - acceptance: per-depth conditional acceptance from the fixed k=7 arm (the only arm that reaches every depth);
  - best fixed width under stationarity (expected tokens / cost), hindsight fixed width on the k=7 steps
    (min(accepted, k) + 1 tokens at cost(k), step boundaries ignored) and the per-step oracle (k = accepted).
usage: mtp_policy_offline.py <k7 outdir> [<ab outdir> ...]   (outdirs of mtp_ab.sh; k7 arm named k7)"""
import glob, json, os, re, statistics as st, sys

STEP = re.compile(r"\[qwen4exp-mtp-step\] step=(\d+) k=(\d+) accepted=(\d+) ms=([\d.]+)")
SUMMARY = re.compile(r"\[qwen4exp-mtp\] k=\d+ drafts=")
WL = os.environ.get("MTP_WL", "code16k prose16k counting").split()   # the run's WORKLOADS, in order


def requests(log):
    """Split a server log into per-request step lists, in order."""
    reqs, cur = [], []
    for line in open(log, errors="replace"):
        m = STEP.search(line)
        if m:
            cur.append((int(m.group(2)), int(m.group(3)), float(m.group(4))))
        elif SUMMARY.search(line):
            reqs.append(cur)
            cur = []
    return reqs


def by_workload(outdir, arm):
    """mtp_ab.sh order per round: warm-up + REPS per workload, in WL order; warm-ups dropped."""
    out = {w: [] for w in WL}
    for log in sorted(glob.glob(f"{outdir}/serve_{arm}_*.log")):
        reqs = requests(log)
        tsv = {w: sum(1 for _ in open(f"{outdir}/{arm}{log.split('_')[-1][:-4]}_{w}.tsv")) for w in WL}
        i = 0
        for w in WL:
            n = tsv[w] + 1
            out[w] += reqs[i + 1:i + n]
            i += n
    return out


def ar_ms(outdir):
    out = {}
    for w in WL:
        v = []
        for f in glob.glob(f"{outdir}/ar[0-9]*_{w}.tsv"):
            v += [1000.0 / json.loads(l.split("\t")[0])["decode_tok_s"] for l in open(f) if l.strip()]
        if v: out[w] = st.median(v)
    return out


k7dir = sys.argv[1]
arms = [(d, a) for d in sys.argv[1:] for a in sorted({f.split("serve_")[1].rsplit("_", 1)[0] for f in glob.glob(f"{d}/serve_*.log")})]
ar = {}
for d in sys.argv[1:]:
    ar.update(ar_ms(d))
k7 = by_workload(k7dir, "k7")
for w in WL:
    cost = {}
    for d, a in arms:
        if a == "ar": continue
        for req in by_workload(d, a)[w]:
            for k, acc, ms in req:
                cost.setdefault(k, []).append(ms)
    med = {k: st.median(v) for k, v in sorted(cost.items())}
    if w in ar: med[0] = ar[w]
    xs = [k for k in med if k >= 1]
    ys = [med[k] for k in xs]
    mx, my = st.mean(xs), st.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    base = my - slope * mx
    print(f"\n== {w}: step ms per width (median, n): " + ", ".join(f"k{k}={v:.1f}({len(cost[k]) if k in cost else 'AR'})" for k, v in sorted(med.items())))
    print(f"   line fit over k>=1: {base:.1f} + {slope:.1f} k")
    steps = [s for req in k7[w] for s in req]
    if not steps:
        print("   no k=7 trace"); continue
    accs = [acc for k, acc, ms in steps if k == 7]
    reached = [sum(1 for a in accs if a >= d) for d in range(0, 8)]
    p = [reached[d] / reached[d - 1] if reached[d - 1] else 0.0 for d in range(1, 8)]
    print(f"   k=7 steps {len(accs)}, per-depth conditional acceptance " + " ".join(f"{x:.3f}" for x in p)
          + ", survival " + " ".join(f"{x:.3f}" for x in [eval('*'.join(map(str, p[:d + 1]))) for d in range(7)]))
    c = lambda k: med.get(k, base + slope * k)
    exp_tokens, surv = [1.0], 1.0
    for d in range(7):
        surv *= p[d]; exp_tokens.append(exp_tokens[-1] + surv)
    stat = {k: exp_tokens[k] / c(k) for k in range(1, 8)}
    hind = {k: sum(min(a, k) + 1 for a in accs) / (len(accs) * c(k)) for k in range(1, 8)}
    oracle_tok = sum(a + 1 for a in accs)
    oracle_ms = sum(c(a) for a in accs)
    print("   stationary fixed  tok/s: " + " ".join(f"k{k}={1000 * v:.2f}" for k, v in stat.items()) + f"  best k={max(stat, key=stat.get)}")
    print("   hindsight fixed   tok/s: " + " ".join(f"k{k}={1000 * v:.2f}" for k, v in hind.items()) + f"  best k={max(hind, key=hind.get)}")
    print(f"   per-step oracle (k = accepted, AR step for 0): {1000 * oracle_tok / oracle_ms:.2f} tok/s")

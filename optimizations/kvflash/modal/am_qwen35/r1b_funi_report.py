"""PREREG R1b Amendment 3 tables (F-UNI). python3 r1b_funi_report.py <dir27b>. Pure numpy."""
import sys

import r1b_stats as st

d = sys.argv[1]
CELLS = ("orig", "F", "FUNI", "UAM")


def load(c, t, s):
    try:
        return st.load_cell(f"{d}/{c}__t{t}__{s}.jsonl")
    except FileNotFoundError:
        return None


def f(r, neg=False):
    m, lo, hi = (-r["mean"], -r["hi"], -r["lo"]) if neg else (r["mean"], r["lo"], r["hi"])
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


print("| split | target | n q | " + " | ".join(CELLS) + " |\n|---|---|---|" + "---|" * len(CELLS))
rows = []
for s in ("sel", "held"):
    for t in ("0.1", "0.05"):
        c = {k: load(k, t, s) for k in CELLS}
        keys = sorted(set.intersection(*[set(v) for v in c.values() if v]))
        cnt = {k: (sum(v[x] for x in keys) if v else None) for k, v in c.items()}
        print(f"| {s} | {t} | {len(keys)} | " + " | ".join(
            "-" if cnt[k] is None else f"{cnt[k]} ({100 * cnt[k] / len(keys):.1f}%)" for k in CELLS) + " |")
        rows.append((s, t, c))
print()
print("| split | target | comparison | n q | diff points [95% CI] | McNemar only-A/only-B, p | verdict |\n"
      "|---|---|---|---|---|---|---|")
for s, t, c in rows:
    (o, F, U), cl = st.align(c["orig"], c["F"], c["FUNI"])
    r = st.degradation(o, U, cl)
    mc = st.mcnemar_exact(o, U)
    print(f"| {s} | {t} | D = orig - F-UNI | {r['n_q']} | {r['mean']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}] | "
          f"{mc[0]}/{mc[1]}, {mc[2]:.3f} | |")
    (F, U), cl = st.align(c["F"], c["FUNI"])
    r = st.online_vs_ceiling(U, F, cl)  # lower bound of (F-UNI - F) >= -3  <=>  upper bound of (F - F-UNI) <= 3
    mc = st.mcnemar_exact(F, U)
    v = {"within 3 points": "uniform within 3 points of F", "inconclusive": "inconclusive"}.get(
        r["verdict"], "uniform worse than F by more than 3")
    print(f"| {s} | {t} | F - F-UNI | {r['n_q']} | {f(r, True)} | {mc[0]}/{mc[1]}, {mc[2]:.3f} | {v} |")
    if s == "sel":
        (U, A), cl = st.align(c["FUNI"], c["UAM"])
        r = st.am_vs_control(U, A, cl)
        mc = st.mcnemar_exact(U, A)
        print(f"| {s} | {t} | F-UNI - U-AM | {r['n_q']} | {f(r)} | {mc[0]}/{mc[1]}, {mc[2]:.3f} | {r['verdict']} |")

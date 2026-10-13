"""Markdown tables for evidence/am/R1b.md from the per-question jsonl files (local copies of the volume dirs).

  python3 r1b_report.py <dir27b/r1b> [<dir4b/r1b>]
Pure numpy (r1b_stats); no GPU.
"""
import glob
import os
import sys

import numpy as np

import r1b_stats as st

TARGETS = ("0.1", "0.05")
SPLITS = ("sel", "held")


def cells_in(d):
    out = {}
    for p in glob.glob(f"{d}/*__t*__*.jsonl"):
        c, t, s = os.path.basename(p)[:-6].split("__")
        out[(c, t[1:], s)] = p
    return out


def timing(path):
    import json
    rows = [json.loads(l) for l in open(path) if l.strip()]
    per_art = {}
    for r in rows:
        per_art[r["article_id"]] = (r.get("compaction_s", 0.0), r.get("query_gen_s", 0.0), r.get("cell_wall_s", 0.0))
    a = np.array(list(per_art.values()))
    return a.mean(0) if len(a) else np.zeros(3)


def fmt(r, sign=False):
    f = "{:+.1f}" if sign else "{:.1f}"
    return f"{f.format(r['mean'])} [{f.format(r['lo'])}, {f.format(r['hi'])}]"


def main(d27, d4=None):
    c27 = cells_in(d27)
    c4 = cells_in(d4) if d4 else {}
    lines = []
    for split in SPLITS:
        for t in TARGETS:
            if ("orig", t, split) not in c27:
                continue
            orig = st.load_cell(c27[("orig", t, split)])
            lines += [f"### 27B, {split}, target {t}", "",
                      "| cell | n q | n art | acc % | D vs own original, points [95% CI] | McNemar only-orig/only-X, p | "
                      "compaction s / art | query gen s / art |", "|---|---|---|---|---|---|---|---|"]
            for (c, tt, s), p in sorted(c27.items()):
                if tt != t or s != split:
                    continue
                x = st.load_cell(p)
                (o, xv), cl = st.align(orig, x)
                d = st.degradation(o, xv, cl)
                mc = st.mcnemar_exact(o, xv)
                tm = timing(p)
                lines.append(f"| {c} | {d['n_q']} | {d['n_art']} | {xv.mean() * 100:.1f} | {fmt(d)} | "
                             f"{mc[0]}/{mc[1]}, {mc[2]:.3f} | {tm[0]:.0f} | {tm[1]:.0f} |")
            lines.append("")
            if c4:
                lines += [f"Delta vs matched-protocol 4B (27B degradation minus 4B degradation, points; PASS if upper "
                          f"bound <= +3.0)", "", "| cell | n q | n art | D 27B | D 4B | Delta [95% CI] | verdict |",
                          "|---|---|---|---|---|---|---|"]
                for c in ("F", "O1", "O2", "C"):
                    ks = [("orig", t, split), (c, t, split)]
                    if not all(k in c27 and k in c4 for k in ks):
                        continue
                    cells = [st.load_cell(c27[k]) for k in ks] + [st.load_cell(c4[k]) for k in ks]
                    (o27, x27, o4, x4), cl = st.align(*cells)
                    r = st.delta_vs_reference(o27, x27, o4, x4, cl)
                    lines.append(f"| {c} | {r['n_q']} | {r['n_art']} | {(o27 - x27).mean() * 100:.1f} | "
                                 f"{(o4 - x4).mean() * 100:.1f} | {fmt(r, True)} | {r['verdict']} |")
                lines.append("")
            if ("C", t, split) in c27:
                lines += ["Online vs ceiling (online minus C, points; R0 rule)", "",
                          "| cell | n q | diff [95% CI] | verdict |", "|---|---|---|---|"]
                for c in ("O1", "O2", "F"):
                    if (c, t, split) in c27:
                        (a, b), cl = st.align(st.load_cell(c27[(c, t, split)]), st.load_cell(c27[("C", t, split)]))
                        r = st.online_vs_ceiling(a, b, cl)
                        lines.append(f"| {c} | {r['n_q']} | {fmt(r, True)} | {r['verdict']} |")
                lines.append("")
            if ("UAM", t, split) in c27:
                lines += ["U-AM vs controls (AM minus control, points; R0 rule)", "",
                          "| control | n q | diff [95% CI] | verdict |", "|---|---|---|---|"]
                for c in ("RND", "TRN"):
                    if (c, t, split) in c27:
                        (a, b), cl = st.align(st.load_cell(c27[("UAM", t, split)]), st.load_cell(c27[(c, t, split)]))
                        r = st.am_vs_control(a, b, cl)
                        lines.append(f"| {c} | {r['n_q']} | {fmt(r, True)} | {r['verdict']} |")
                lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])

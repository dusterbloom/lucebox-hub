"""R1(e) report (PREREG 'R1(e)', item 6). Usage: r1e_report.py <rows dir> <split>

Rows: <dir>/{ORIG,F-UNI-GLOBAL,F-UNI-LOCAL}__t0.1__<split>.jsonl. Unit question, cluster article,
paired cluster bootstrap (r1b_stats). Decision on held-out: LOCAL - GLOBAL lower 95% bound > -3 points.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../modal/am_qwen35"))
from r1b_stats import MARGIN, align, boot_mean, load_cell, mcnemar_exact  # noqa: E402

ARMS = ("ORIG", "F-UNI-GLOBAL", "F-UNI-LOCAL")


def decide(lo, margin=MARGIN):
    return "SHIP segment-local" if lo > -margin else "global or query-reservoir design"


def report(rows_dir, split):
    cells = [load_cell(os.path.join(rows_dir, f"{a}__t0.1__{split}.jsonl")) for a in ARMS]
    (orig, glob, local), arts = align(*cells)
    out = [f"R1(e) {split}: {len(orig)} questions, {len(set(arts))} articles (questions present in all arms)", ""]
    out.append("| Arm | Correct | Accuracy | Drop vs ORIG, points [95% CI] |")
    out.append("|---|---|---|---|")
    for name, x in zip(ARMS, (orig, glob, local)):
        d = boot_mean((orig - x) * 100, arts) if name != "ORIG" else None
        drop = "" if d is None else f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"
        out.append(f"| {name} | {int(x.sum())} | {100 * x.mean():.1f}% | {drop} |")
    r = boot_mean((local - glob) * 100, arts)
    only_l, only_g, p = mcnemar_exact(local, glob)
    out += ["", f"LOCAL - GLOBAL: {r['mean']:+.2f} points [{r['lo']:+.2f}, {r['hi']:+.2f}], "
                f"McNemar only-LOCAL {only_l} / only-GLOBAL {only_g}, p = {p:.3f}",
            f"Rule (lower bound > -{MARGIN:g}): {decide(r['lo'])}"
            + ("" if split == "held" else " (selection split: sanity only, does not decide)")]
    return "\n".join(out)


if __name__ == "__main__":
    assert decide(-2.9) == "SHIP segment-local" and decide(-3.0) != "SHIP segment-local"
    print(report(sys.argv[1], sys.argv[2]))

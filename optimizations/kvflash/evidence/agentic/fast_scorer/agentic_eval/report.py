"""Render the evidence markdown tables from score.json + arm meta files.
Usage: python3 -I report.py <full_dir> <score.json> <n_traj>   (prints markdown to stdout)"""
import json
import os
import sys


def f(x, p=3):
    return "n/a" if x is None else ("%.*f" % (p, x))


def ci(d):
    if not d:
        return "n/a"
    pt, lo, hi = d
    return "%+.1f pts [%+.1f, %+.1f]" % (100 * pt, 100 * lo, 100 * hi)


def main():
    out_dir, score_p, n_traj = sys.argv[1], sys.argv[2], int(sys.argv[3])
    S = json.load(open(score_p))
    arms = S["arms"]
    metas = {a: json.load(open(os.path.join(out_dir, a + ".meta.json"))) for a in arms if os.path.exists(os.path.join(out_dir, a + ".meta.json"))}
    L = []
    L.append("### Counts")
    b, c = S["b"], S["c"]
    L.append(f"- Trajectories: {n_traj}. Probe candidates {b['candidates']}, FULL attempted {b['full_attempted']}, retained (FULL correct) {b['retained']} (yield {100 * b['yield']:.1f}%).")
    L.append(f"- Decision points {c['decision_points']}; FULL produced a parsable call at {c['full_parsed']} (excluded {c['full_parse_fail']}, all `finish=length` at 256 tokens).")
    L.append(f"- FULL vs gold (OpenHands/Qwen3-Coder call) full key {c['full_agrees_with_gold_full_key']}/{c['full_parsed']}, tool name {c['full_name_agrees_with_gold']}/{c['full_parsed']}.")
    L.append(f"- Request errors per arm: {S['errors']}")
    L.append("")
    L.append("Retained probes by type (retained / attempted): " + ", ".join(f"{k} {v['retained']}/{v['attempted']}" for k, v in b["by_type_retained"].items()))
    L.append("")
    L.append("### b. Recall retention, P(correct | FULL correct)")
    L.append("| arm | n | correct | retention | Wilson 95% | arm - FULL (cluster bootstrap 95%) | McNemar b/c, exact p |")
    L.append("|---|---|---|---|---|---|---|")
    for a, d in b["arms"].items():
        w = d["wilson95"]
        L.append(f"| {a} | {d['n']} | {d['correct']} | {100 * d['retention']:.1f}% | {f(w[0] * 100, 1) if w else 'n/a'}-{f(w[1] * 100, 1) if w else 'n/a'}% | {ci(d['diff_vs_FULL'])} | {d['mcnemar_b_ref_right_arm_wrong']}/{d['mcnemar_c']}, {d['mcnemar_p']:.2g} |")
    for key, title in (("by_type", "type"), ("by_P", "P (K tokens)"), ("by_distance", "distance of fact from cut (tokens)")):
        L.append("")
        L.append(f"Retention by {title} (correct/n):")
        keys = []
        for d in b["arms"].values():
            for k in d[key]:
                if k not in keys:
                    keys.append(k)
        L.append("| arm | " + " | ".join(str(k) for k in keys) + " |")
        L.append("|---|" + "---|" * len(keys))
        for a, d in b["arms"].items():
            L.append(f"| {a} | " + " | ".join("%d/%d (%.0f%%)" % (d[key][k][0], d[key][k][1], 100 * d[key][k][0] / d[key][k][1]) if k in d[key] else "-" for k in keys) + " |")
    L.append("")
    L.append("### c. Next-tool-call agreement with FULL's own call")
    L.append("| arm | n | agree (name+key) | agreement | Wilson 95% | arm - FULL (cluster bootstrap 95%) | name-only agreement | name-only arm - FULL | arm parse failures | McNemar b/c, exact p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for a, d in c["arms"].items():
        w = d["wilson95"]
        L.append(f"| {a} | {d['n']} | {d['agree']} | {100 * d['agreement']:.1f}% | {f(w[0] * 100, 1) if w else 'n/a'}-{f(w[1] * 100, 1) if w else 'n/a'}% | {ci(d['diff_vs_FULL'])} | {100 * d['name_agreement']:.1f}% | {ci(d['name_diff_vs_FULL'])} | {d['arm_parse_fail']} | {d['mcnemar_b_ref_right_arm_wrong']}/{d['mcnemar_c']}, {d['mcnemar_p']:.2g} |")
    L.append("")
    L.append("### Server-side timings (usage.timings: prefill_ms, decode_ms; wall = prefill + decode)")
    L.append("| arm | requests | prefill s (sum) | decode s (sum) | wall s (sum) | wall per trajectory | median prefill s | median decode s | median wall s | median prefill tok/s* | median decode tok/s* | prefilled tokens (sum) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        t = S["timing"][a].get("all")
        if not t:
            continue
        ntr = n_traj if a != "FULLB" else 20
        L.append(f"| {a} | {t['n']} | {t['prefill_s_sum']:.0f} | {t['decode_s_sum']:.0f} | {t['wall_s_sum']:.0f} | {t['wall_s_sum'] / ntr:.0f} s | {f(t['prefill_s_median'], 2)} | {f(t['decode_s_median'], 2)} | {f(t['wall_s_median'], 2)} | {f(t['prefill_tps_median'], 0)} | {f(t['decode_tps_median'], 1)} | {t['prefilled_tokens_sum']} |")
    L.append("")
    L.append("*tok/s medians over requests with prefill > 500 tokens and > 0.5 s (prefill) or >= 8 completion tokens and > 0.3 s (decode).")
    L.append("")
    L.append("By request kind (sum wall s / median wall s):")
    L.append("| arm | probe | decision | primer |")
    L.append("|---|---|---|---|")
    for a in arms:
        t = S["timing"][a]
        L.append(f"| {a} | " + " | ".join(f"{t[k]['wall_s_sum']:.0f} / {t[k]['wall_s_median']:.2f} (n={t[k]['n']})" if k in t else "-" for k in ("probe", "decision", "primer")) + " |")
    L.append("")
    L.append("Binary and launch: " + "; ".join(f"{a}: md5 {m['binary_md5']}, pool {m['pool']}, max-ctx {m['max_ctx']}" for a, m in metas.items()))
    print("\n".join(L))


if __name__ == "__main__":
    main()

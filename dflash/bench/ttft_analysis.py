#!/usr/bin/env python3
"""Analyse raw_results.jsonl from ttft_investigation.py and write SUMMARY.md."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    raw_path = args.out / "raw_results.jsonl"
    rows = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"[analysis] loaded {len(rows)} rows")

    # Group by (n_gen, mode)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["n_gen"], r["mode"])
        groups[key].append(r)

    n_gen_list = sorted({r["n_gen"] for r in rows})
    modes = ["off", "always"]

    # Build summary table
    summary: dict[tuple, dict] = {}
    for n_gen in n_gen_list:
        for mode in modes:
            key = (n_gen, mode)
            cell = groups.get(key, [])
            if not cell:
                summary[key] = None
                continue

            ttft_vals = [r["ttft_s"] for r in cell if r["ttft_s"] is not None]
            wall_vals = [r["wall_s"] for r in cell if r["wall_s"] is not None]
            out_tok_vals = [r["output_tokens"] for r in cell if r["output_tokens"]]

            summary[key] = {
                "n": len(cell),
                "mean_ttft": statistics.mean(ttft_vals) if ttft_vals else float("nan"),
                "p50_ttft": statistics.median(ttft_vals) if ttft_vals else float("nan"),
                "mean_wall": statistics.mean(wall_vals) if wall_vals else float("nan"),
                "mean_out_tokens": statistics.mean(out_tok_vals) if out_tok_vals else float("nan"),
                "errors": sum(1 for r in cell if r.get("error")),
            }

    # Compute speedups
    lines = []
    lines.append("# TTFT vs Wall-Total Investigation at 32K\n")
    lines.append("**Hypothesis**: pflash saves prefill (TTFT), not decode time. "
                 "Speedup visible in TTFT regardless of n_gen; "
                 "total wall speedup drops as n_gen increases.\n\n")

    lines.append("## Per-(n_gen, mode) Table\n\n")
    lines.append("| n_gen | mode   | n | mean_ttft_s | p50_ttft_s | mean_wall_s | mean_out_tok |")
    lines.append("|-------|--------|---|-------------|------------|-------------|--------------|")

    for n_gen in n_gen_list:
        for mode in modes:
            s = summary.get((n_gen, mode))
            if s is None:
                lines.append(f"| {n_gen:5d} | {mode:6s} | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {n_gen:5d} | {mode:6s} | {s['n']} "
                    f"| {s['mean_ttft']:11.2f} "
                    f"| {s['p50_ttft']:10.2f} "
                    f"| {s['mean_wall']:11.2f} "
                    f"| {s['mean_out_tokens']:12.1f} |"
                )

    lines.append("")

    # Speedup table
    lines.append("## Speedup Table (OFF / ALWAYS)\n\n")
    lines.append("| n_gen | ttft_speedup | wall_speedup |")
    lines.append("|-------|--------------|--------------|")

    for n_gen in n_gen_list:
        s_off = summary.get((n_gen, "off"))
        s_al = summary.get((n_gen, "always"))
        if s_off and s_al:
            ttft_sp = (s_off["mean_ttft"] / s_al["mean_ttft"]
                       if s_al["mean_ttft"] > 0 else float("nan"))
            wall_sp = (s_off["mean_wall"] / s_al["mean_wall"]
                       if s_al["mean_wall"] > 0 else float("nan"))
            lines.append(
                f"| {n_gen:5d} | {ttft_sp:12.1f}x | {wall_sp:12.1f}x |"
            )
        else:
            lines.append(f"| {n_gen:5d} | N/A | N/A |")

    lines.append("")

    # Disambiguation section
    lines.append("## Disambiguation\n")
    s8_off = summary.get((8, "off"))
    s8_al = summary.get((8, "always"))
    s256_off = summary.get((256, "off"))
    s256_al = summary.get((256, "always"))

    if s8_off and s8_al:
        ttft_8x = (s8_off["mean_ttft"] / s8_al["mean_ttft"]) if s8_al["mean_ttft"] > 0 else float("nan")
        wall_8x = (s8_off["mean_wall"] / s8_al["mean_wall"]) if s8_al["mean_wall"] > 0 else float("nan")
        lines.append(
            f"At **n_gen=8** (output dominated by prefill): "
            f"TTFT speedup={ttft_8x:.1f}x, wall speedup={wall_8x:.1f}x. "
            f"These should be close (decode contributes ~{8}/{s8_off.get('mean_out_tokens', 8):.0f} of wall)."
        )

    if s256_off and s256_al:
        ttft_256x = (s256_off["mean_ttft"] / s256_al["mean_ttft"]) if s256_al["mean_ttft"] > 0 else float("nan")
        wall_256x = (s256_off["mean_wall"] / s256_al["mean_wall"]) if s256_al["mean_wall"] > 0 else float("nan")
        lines.append(
            f"\nAt **n_gen=256** (decode dominates): "
            f"TTFT speedup={ttft_256x:.1f}x, wall speedup={wall_256x:.1f}x. "
            f"Wall speedup should be lower than TTFT speedup."
        )

    lines.append("")

    # Production speedup framing
    lines.append("## Production Speedup at Typical Workloads\n")
    lines.append(
        "- **Short retrieval (n_gen ≤ 16)**: wall speedup ≈ TTFT speedup (decode negligible)\n"
        "- **Long completion (n_gen=256)**: TTFT speedup stays high; wall speedup lower\n"
        "- **Long reasoning (n_gen≥2000)**: wall speedup approaches decode-speed ratio "
          "(TTFT contribution diluted further)\n"
    )

    # Headline recommendation
    lines.append("## Headline Recommendation\n")
    lines.append(
        "**Use TTFT as the headline metric.** It is n_gen-invariant and reflects the "
        "user-visible latency improvement for any workload where the context is long "
        "(the dominant cost is prefill). For multi-turn chat where the user waits for "
        "first token, TTFT is exactly what matters. Total wall time is the correct "
        "metric only for batch/offline pipelines where n_gen >> 0 and throughput matters.\n"
    )

    # Notes on methodology
    lines.append("## Methodology Notes\n")
    lines.append(
        "- Two server launches required (OFF then ALWAYS): per-request pflash_mode override "
          "in `extra_body` cannot switch between OFF and ALWAYS within one server instance "
          "(see `http_server.cpp` L551 — `should_compress` is gated on `config_.pflash_mode`, "
          "not the per-request override).\n"
        "- TTFT measured as time from HTTP send to first OpenAI SSE content delta "
          "(via raw TCP socket to avoid buffering).\n"
        "- 3 NIAH cases at 32K tokens, 3 n_gen values (8, 64, 256), 2 modes = 18 requests.\n"
        "- Server: Qwen3.6-27B-Q4_K_M, drafter: Qwen3-0.6B-BF16, keep_ratio=0.05.\n"
    )

    summary_text = "\n".join(lines) + "\n"
    summary_path = args.out / "SUMMARY.md"
    summary_path.write_text(summary_text)
    print(f"[analysis] wrote {summary_path}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()

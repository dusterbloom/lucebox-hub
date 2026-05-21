#!/usr/bin/env python3
"""Generate SUMMARY.md from runs.jsonl."""
import sys
import json
import os

results_dir = sys.argv[1]
jsonl_path = os.path.join(results_dir, "runs.jsonl")

runs = []
with open(jsonl_path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except Exception:
                pass


def fmt_rate(v):
    if v is None or v == "—":
        return "—"
    return f"{v:.3f}"


def fmt_wall(v):
    if v is None or v == "—":
        return "—"
    return f"{v:.2f}s"


by_label = {r["label"]: r for r in runs}

lines = ["# Composition Bench Summary — 2026-05-21", ""]
lines += ["## Composition Table (PFlash OFF vs ALWAYS per speculator)", ""]
lines += ["| Speculator | Mode | keep | mean_accept_rate | mean_wall/turn | ok_done | turns |"]
lines += ["|-----------|------|------|-----------------|---------------|---------|-------|"]

comp_rows = [
    ("MTP γ=2", "OFF",    "n/a",  "mtp-off"),
    ("MTP γ=2", "ALWAYS", "0.05", "mtp-always-k05"),
    ("DFlash",  "OFF",    "n/a",  "dflash-off"),
    ("DFlash",  "ALWAYS", "0.05", "dflash-always-k05"),
]
for spec, mode, keep, label in comp_rows:
    r = by_label.get(label, {})
    err = r.get("error", "")
    if err:
        lines.append(f"| {spec} | {mode} | {keep} | ERR: {err} | — | — | — |")
    else:
        lines.append(
            f"| {spec} | {mode} | {keep} "
            f"| {fmt_rate(r.get('mean_accept_rate'))} "
            f"| {fmt_wall(r.get('mean_wall_per_turn_s'))} "
            f"| {r.get('ok_done_seen', '—')} "
            f"| {r.get('num_turns', '—')} |"
        )

lines += [""]
lines += ["## Keep Sweep on MTP (keep_ratio effect on decode)", ""]
lines += ["| keep | mean_accept_rate | mean_wall/turn | mean_completion_tokens | ok_done |"]
lines += ["|------|-----------------|---------------|------------------------|---------|"]

sweep_rows = [
    ("0.025", "mtp-always-k025"),
    ("0.050", "mtp-always-k05"),
    ("0.100", "mtp-always-k10"),
    ("0.200", "mtp-always-k20"),
]
for keep, label in sweep_rows:
    r = by_label.get(label, {})
    err = r.get("error", "")
    if err:
        lines.append(f"| {keep} | ERR: {err} | — | — | — |")
    else:
        lines.append(
            f"| {keep} "
            f"| {fmt_rate(r.get('mean_accept_rate'))} "
            f"| {fmt_wall(r.get('mean_wall_per_turn_s'))} "
            f"| {r.get('mean_completion_tokens', '—')} "
            f"| {r.get('ok_done_seen', '—')} |"
        )

lines += [""]
lines += ["## Per-Run Summary", ""]
lines += ["| label | ok_done_seen | num_turns | mean_accept | mean_wall/turn | status |"]
lines += ["|-------|-------------|-----------|------------|---------------|--------|"]
for r in runs:
    label = r.get("label", "?")
    err = r.get("error", "")
    if err:
        lines.append(f"| {label} | — | — | — | — | CRASH: {err} |")
    else:
        ok_count = r.get("ok_done_count", 0) or 0
        num_turns = r.get("num_turns", 0) or 0
        if ok_count == num_turns and num_turns > 0:
            status = "OK"
        elif ok_count > 0:
            status = f"partial ({ok_count}/{num_turns} OK_DONE)"
        else:
            status = "FAIL"
        lines.append(
            f"| {label} | {r.get('ok_done_seen', '—')} "
            f"| {r.get('num_turns', '—')} "
            f"| {fmt_rate(r.get('mean_accept_rate'))} "
            f"| {fmt_wall(r.get('mean_wall_per_turn_s'))} "
            f"| {status} |"
        )

summary_path = os.path.join(results_dir, "SUMMARY.md")
with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[gen_summary] written: {summary_path}")
for line in lines:
    print(line)

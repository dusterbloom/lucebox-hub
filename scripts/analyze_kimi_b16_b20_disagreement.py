#!/usr/bin/env python3
"""Test route12/B16 versus B20 disagreement on native-aligned terminal rows."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_kimi_native_terminal_reuse_suite import analyze_arm, percentile
from analyze_kimi_native_tool_first_token import terminal_metrics
from analyze_kimi_progressive_tool_rescue import digest


def score(teacher: dict, teacher_logits: list[float], arm: dict,
          logits: list[float]) -> dict:
    aligned = arm["generated_ids"][:-1] == teacher["generated_ids"][:-1]
    arm["logit_history_aligned"] = aligned
    arm["generated_sequence_exact"] = arm["generated_ids"] == teacher["generated_ids"]
    arm["terminal_metrics"] = terminal_metrics(teacher_logits, logits) if aligned else None
    return arm


def aggregate(rows: list[dict], policy: str) -> dict:
    arms = [row[policy] for row in rows]
    metrics = [arm["terminal_metrics"] for arm in arms
               if arm["terminal_metrics"] is not None]
    divergences = [item["terminal_kl"] for item in metrics]
    positions = sum(arm["traffic"]["provider_positions"] for arm in arms)
    logical = sum(arm["traffic"]["total_provider_bytes"] for arm in arms)
    physical = sum(arm["physical"]["direct_physical_bytes"] for arm in arms)
    return {
        "fixtures": len(arms),
        "logit_history_aligned": len(metrics),
        "generated_sequence_exact": sum(arm["generated_sequence_exact"] for arm in arms),
        "terminal_top1_agreement": sum(item["top1_agreement"] for item in metrics),
        "terminal_kl": ({
            "mean": statistics.fmean(divergences),
            "median": statistics.median(divergences),
            "p95": percentile(divergences, 0.95),
            "maximum": max(divergences),
        } if divergences else None),
        "provider_positions": positions,
        "logical_gib_per_position": logical / positions / (1024 ** 3),
        "physical_gib_per_position": physical / positions / (1024 ** 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.prereg.read_text())

    rows = []
    b16_commits = set()
    binary_hashes = set()
    for fixture in prereg["fixtures"]:
        teacher, teacher_logits = analyze_arm(Path(fixture["teacher_root"]), fixture)
        b20, b20_logits = analyze_arm(Path(fixture["b20_root"]), fixture)
        b16, b16_logits = analyze_arm(Path(fixture["b16_root"]), fixture)
        if teacher["manifest_sha256"] != fixture["teacher_manifest_sha256"]:
            raise ValueError(f"teacher manifest changed: {fixture['id']}")
        if b20["manifest_sha256"] != fixture["b20_manifest_sha256"]:
            raise ValueError(f"B20 manifest changed: {fixture['id']}")
        if (b16["environment"].get("DFLASH_KIMI_H22_LAYER_BUDGETS")
                != prereg["policies"]["b16"]["path"]
                or b16["environment"].get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12"):
            raise ValueError(f"B16 policy changed: {fixture['id']}")
        b20 = score(teacher, teacher_logits, b20, b20_logits)
        b16 = score(teacher, teacher_logits, b16, b16_logits)
        rows.append({"id": fixture["id"], "kind": fixture["kind"],
                     "teacher": teacher, "route12_budget20": b20,
                     "route12_budget16": b16})
        b16_commits.add(b16["source_commit"])
        binary_hashes.update((teacher["executable_sha256"], b20["executable_sha256"],
                              b16["executable_sha256"]))
    if len(b16_commits) != 1 or len(binary_hashes) != 1:
        raise ValueError("B16 commit or executable identity changed")

    ordinary = [row for row in rows if row["kind"] == "ordinary"]
    tool = next(row for row in rows if row["kind"] == "tool_boundary")
    aggregates = {
        "ordinary": {
            policy: aggregate(ordinary, policy) for policy in
            ("route12_budget20", "route12_budget16")},
        "all": {
            policy: aggregate(rows, policy) for policy in
            ("route12_budget20", "route12_budget16")},
    }
    base = aggregates["ordinary"]["route12_budget16"]
    base_gate = (base["generated_sequence_exact"] == len(ordinary)
                 and base["logical_gib_per_position"] < 0.9)
    ordinary_agreement = all(
        row["route12_budget16"]["terminal_metrics"]["candidate_top1"]
        == row["route12_budget20"]["terminal_metrics"]["candidate_top1"]
        for row in ordinary)
    tool_b16 = tool["route12_budget16"]["terminal_metrics"]
    tool_b20 = tool["route12_budget20"]["terminal_metrics"]
    risk_gate = (base_gate and ordinary_agreement
                 and tool_b16["candidate_top1"] != tool_b20["candidate_top1"]
                 and not tool_b16["top1_agreement"] and tool_b20["top1_agreement"])
    status = ("MEASURED_B16_B20_DISAGREEMENT_GO" if risk_gate else
              "MEASURED_B16_BASE_ONLY_GO" if base_gate else
              "MEASURED_B16_B20_NO_GO")
    result = {
        "schema": "kimi-k3-b16-b20-disagreement-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": prereg["source"],
        "fixtures": rows,
        "aggregate": aggregates,
        "gate": {
            "b16_ordinary_base_pass": base_gate,
            "ordinary_b16_b20_top1_agreement": ordinary_agreement,
            "tool_b16_b20_disagreement_detects_native_failure": risk_gate,
            "interpretation": (
                "Earn a one-pass predictor or incremental B16-to-B20 hydration experiment."
                if risk_gate else "Do not implement a disagreement runtime from this gate."),
        },
        "limitations": [
            "Running complete B16 and B20 passes is a discriminator, not an efficient policy.",
            "Four terminal rows cannot estimate the production rescue rate.",
            "The tool boundary was previously known and is not a held-out risk example.",
            "Cold token-sequential timings are not serving throughput.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": status,
                      "aggregate": aggregates, "risk_gate": risk_gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

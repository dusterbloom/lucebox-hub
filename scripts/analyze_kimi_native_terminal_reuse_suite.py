#!/usr/bin/env python3
"""Score frozen K3 policies against reused, history-aligned Full192 logits."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_kimi_native_tool_first_token import read_logits, terminal_metrics
from analyze_kimi_progressive_tool_rescue import (
    P20,
    digest,
    i32le_digest,
    read_environment,
    token_traces,
    traffic,
    verify_manifest,
)


def analyze_arm(root: Path, fixture: dict) -> tuple[dict, list[float]]:
    verify_manifest(root)
    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    if len(traces["prompt_ids"]) != fixture["prompt_tokens"]:
        raise ValueError(f"prompt length changed: {root}")
    if i32le_digest(traces["prompt_ids"]) != fixture["prompt_token_ids_i32le_sha256"]:
        raise ValueError(f"prompt token hash changed: {root}")
    if len(traces["generated_ids"]) != fixture["generated_tokens"]:
        raise ValueError(f"generated token count changed: {root}")
    byte_metrics = traffic(root / "traffic.tsv")
    p20 = P20.findall(stderr)
    if len(p20) > 1:
        raise ValueError(f"multiple P20 summaries: {root}")
    physical = None
    if p20:
        reads, physical_bytes, io_ns = map(int, p20[0])
        physical = {
            "explicit_provider_reads": reads,
            "direct_physical_bytes": physical_bytes,
            "direct_physical_gib_per_position": physical_bytes
            / byte_metrics["provider_positions"] / (1024 ** 3),
            "direct_io_ns": io_ns,
        }
    logits_path = root / "final.f32"
    response = json.loads((root / "response.json").read_text())
    return ({
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "response_sha256": digest(root / "response.json"),
        "logits_sha256": digest(logits_path),
        "traffic_sha256": digest(root / "traffic.tsv"),
        "prompt_token_ids_i32le_sha256": i32le_digest(traces["prompt_ids"]),
        "generated_ids": traces["generated_ids"],
        "content": response["choices"][0]["message"].get("content"),
        "finish_reason": response["choices"][0]["finish_reason"],
        "environment": {key: value for key, value in read_environment(
            root / "environment.nul").items() if key.startswith("DFLASH_KIMI")},
        "usage": response["usage"],
        "traffic": byte_metrics,
        "physical": physical,
    }, read_logits(logits_path))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate(rows: list[dict], policy: str) -> dict:
    arms = [row[policy] for row in rows]
    aligned = [arm for arm in arms if arm["history_aligned"]]
    positions = sum(arm["traffic"]["provider_positions"] for arm in arms)
    logical = sum(arm["traffic"]["total_provider_bytes"] for arm in arms)
    physical_values = [arm["physical"] for arm in arms]
    physical = (sum(item["direct_physical_bytes"] for item in physical_values)
                if all(item is not None for item in physical_values) else None)
    divergences = [arm["terminal_metrics"]["terminal_kl"] for arm in aligned]
    return {
        "fixtures": len(arms),
        "history_aligned": len(aligned),
        "generated_sequence_exact": sum(arm["generated_sequence_exact"] for arm in arms),
        "terminal_top1_agreement": sum(
            arm["terminal_metrics"]["top1_agreement"] for arm in aligned),
        "terminal_kl": ({
            "mean": statistics.fmean(divergences),
            "median": statistics.median(divergences),
            "p95": percentile(divergences, 0.95),
            "maximum": max(divergences),
        } if divergences else None),
        "provider_positions": positions,
        "logical_gib_per_position": logical / positions / (1024 ** 3),
        "physical_gib_per_position": (
            physical / positions / (1024 ** 3) if physical is not None else None),
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
    candidate_commits = set()
    binary_hashes = set()
    for fixture in prereg["fixtures"]:
        teacher, teacher_logits = analyze_arm(Path(fixture["teacher_root"]), fixture)
        if teacher["manifest_sha256"] != fixture["teacher_manifest_sha256"]:
            raise ValueError(f"teacher manifest changed: {fixture['id']}")
        row = {"id": fixture["id"], "teacher": teacher}
        for policy in ("h23_aggressive1p8", "h23_moonshot1p2", "route12_budget20"):
            arm, logits = analyze_arm(Path(fixture["candidate_roots"][policy]), fixture)
            environment = arm["environment"]
            expected = prereg["policies"][policy]
            if environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != expected["path"]:
                raise ValueError(f"policy changed: {fixture['id']} {policy}")
            if environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != expected.get("route_limit"):
                raise ValueError(f"route limit changed: {fixture['id']} {policy}")
            aligned = arm["generated_ids"] == teacher["generated_ids"]
            arm["history_aligned"] = aligned
            arm["generated_sequence_exact"] = aligned
            arm["terminal_metrics"] = terminal_metrics(teacher_logits, logits) if aligned else None
            row[policy] = arm
            candidate_commits.add(arm["source_commit"])
            binary_hashes.add(arm["executable_sha256"])
        binary_hashes.add(teacher["executable_sha256"])
        rows.append(row)
    if len(candidate_commits) != 1 or len(binary_hashes) != 1:
        raise ValueError("candidate commit or executable identity changed")

    aggregates = {policy: aggregate(rows, policy) for policy in
                  ("h23_aggressive1p8", "h23_moonshot1p2", "route12_budget20")}
    route = aggregates["route12_budget20"]
    h23_kl = aggregates["h23_aggressive1p8"]["terminal_kl"]
    behavior_gate = (route["generated_sequence_exact"] == len(rows)
                     and route["logical_gib_per_position"] < 1.2)
    strong_gate = (behavior_gate
                   and route["terminal_kl"] is not None
                   and h23_kl is not None
                   and route["terminal_kl"]["mean"]
                   <= h23_kl["mean"])
    status = ("MEASURED_NATIVE_REUSE_STRONG_GO" if strong_gate else
              "MEASURED_NATIVE_REUSE_BEHAVIORAL_GO" if behavior_gate else
              "MEASURED_NATIVE_REUSE_NO_GO")
    result = {
        "schema": "kimi-k3-native-terminal-reuse-suite-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": prereg["source"],
        "fixtures": rows,
        "aggregate": aggregates,
        "gate": {
            "behavioral_pass": behavior_gate,
            "strong_distributional_pass": strong_gate,
            "interpretation": ("Advance route12/B20 to a wider native KL suite."
                               if behavior_gate else "Stop route12/B20."),
        },
        "limitations": [
            "Only the final captured distribution of each exact eight-token history is scored.",
            "A fixture with candidate history divergence is excluded from KL rather than compared on mismatched histories.",
            "Three short fixtures do not establish coding-agent or long-context reliability.",
            "Cold token-sequential timings are not serving throughput.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": status,
                      "aggregate": aggregates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

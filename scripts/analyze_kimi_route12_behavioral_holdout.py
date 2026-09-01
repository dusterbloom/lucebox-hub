#!/usr/bin/env python3
"""Score route12 on the preregistered unseen behavioral holdout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import digest, token_traces
from analyze_kimi_route12_native_success import first_divergence, task_success
from analyze_kimi_schema_rescue import analyze_arm


ROUTE_MARKER = re.compile(
    r"\[kimi-k3-route-limit\] top-routes=(\d+) weights=unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--root", action="append", required=True,
        help="task-id=artifact-root; repeat once per preregistered task")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    baseline = json.loads(args.baseline.read_text())
    repo = Path(__file__).resolve().parent.parent
    for path, expected in (
            (Path(__file__), prereg["source"]["analyzer_sha256"]),
            (repo / "scripts/analyze_kimi_route12_native_success.py",
             prereg["source"]["scorer_module_sha256"]),
            (repo / "scripts/run_kimi_progressive_tool_rescue.sh",
             prereg["source"]["runner_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"registered harness hash changed: {path}")
    if digest(args.baseline) != prereg["native_baseline"]["sha256"]:
        raise ValueError("native baseline hash changed")
    if digest(args.policy) != prereg["representation"]["policy_sha256"]:
        raise ValueError("Budget20 policy hash changed")
    native_rows = {row["id"]: row for row in baseline["sequences"]}
    fixture_rows = {row["id"]: row for row in prereg["fixtures"]}
    root_args = dict(item.split("=", 1) for item in args.root)
    if set(root_args) != set(native_rows) or set(root_args) != set(fixture_rows):
        raise ValueError("task/root set does not match preregistration")

    sequences = []
    source_commits = set()
    executable_hashes = set()
    total_positions = total_logical = total_fallback = 0
    total_physical = total_direct_io_ns = 0
    for identifier in (row["id"] for row in prereg["fixtures"]):
        spec = fixture_rows[identifier]
        fixture = Path(spec["path"])
        root = Path(root_args[identifier])
        if digest(fixture) != spec["sha256"]:
            raise ValueError(f"fixture hash changed: {identifier}")
        if digest(root / "request.json") != spec["sha256"]:
            raise ValueError(f"measured fixture changed: {identifier}")
        arm = analyze_arm(root)
        source_commits.add(arm["source_commit"])
        executable_hashes.add(arm["executable_sha256"])
        environment = arm["environment"]
        if (environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12" or
                environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1" or
                environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(args.policy)):
            raise ValueError(f"environment changed: {identifier}")
        if "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment:
            raise ValueError(f"position intervention present: {identifier}")
        stderr = (root / "server.stderr").read_text()
        route_markers = [int(value) for value in ROUTE_MARKER.findall(stderr)]
        if (route_markers != [12] or arm["schema_prefix_configurations"] != 0 or
                arm["schema_rescue_markers"] or arm["static_position_markers"] != 0 or
                arm["request_wide_b96_markers"] != 0):
            raise ValueError(f"route/schema marker contract changed: {identifier}")
        traces = token_traces(stderr)
        native = native_rows[identifier]
        response = json.loads((root / "response.json").read_text())
        content = response["choices"][0]["message"].get("content", "")
        success = task_success(identifier, content)
        divergence = first_divergence(native["output_tokens"], arm["generated_ids"])
        prompt_equal = traces["prompt_ids"] == native["prompt_tokens"]
        positions = arm["traffic"]["provider_positions"]
        if positions != len(traces["prompt_ids"]) + len(arm["generated_ids"]) - 1:
            raise ValueError(f"provider-position alignment changed: {identifier}")
        total_positions += positions
        total_logical += arm["traffic"]["total_provider_bytes"]
        total_fallback += arm["traffic"]["exact_fallback_bytes"]
        total_physical += arm["traffic"]["direct_physical_bytes"]
        total_direct_io_ns += arm["traffic"]["direct_io_ns"]
        sequences.append({
            "id": identifier,
            "artifact_root": str(root),
            "manifest_sha256": arm["manifest_sha256"],
            "response_sha256": digest(root / "response.json"),
            "logits_sha256": arm["logits_sha256"],
            "traffic_sha256": arm["traffic_sha256"],
            "native_prompt_token_count": len(native["prompt_tokens"]),
            "candidate_prompt_token_count": len(traces["prompt_ids"]),
            "prompt_tokens_equal_native": prompt_equal,
            "native_tokens": native["output_tokens"],
            "candidate_tokens": arm["generated_ids"],
            "output_ids_equal_frozen_native": divergence is None,
            "prompt_aligned_exact_sequence": prompt_equal and divergence is None,
            "first_generated_token_divergence": divergence,
            "candidate_text": content,
            "task_success": success,
            "route_limit_markers": route_markers,
            "provider_positions": positions,
            "logical_gib_per_position": arm["traffic"]["logical_gib_per_position"],
            "physical_gib_per_position": (
                arm["traffic"]["direct_physical_bytes"] / positions / (1024 ** 3)),
            "client_time_seconds": float(
                (root / "client.time.tsv").read_text().split("\t", 1)[0]),
        })

    if len(source_commits) != 1:
        raise ValueError("arms used different source commits")
    if executable_hashes != {prereg["source"]["executable_sha256"]}:
        raise ValueError("arms used a different executable")
    logical_gib = total_logical / total_positions / (1024 ** 3)
    physical_gib = total_physical / total_positions / (1024 ** 3)
    tasks_passed = sum(row["task_success"] for row in sequences)
    gate_passed = tasks_passed == len(sequences) and logical_gib < 1.2
    result = {
        "schema": "kimi-k3-route12-behavioral-holdout-result-v1",
        "status": ("MEASURED_ROUTE12_BEHAVIORAL_HOLDOUT_GO" if gate_passed else
                   "MEASURED_ROUTE12_BEHAVIORAL_HOLDOUT_NO_GO"),
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": next(iter(source_commits)),
        },
        "native_baseline": prereg["native_baseline"],
        "candidate": {
            "tasks_passed": tasks_passed,
            "tasks": len(sequences),
            "output_ids_equal_frozen_native": sum(
                row["output_ids_equal_frozen_native"] for row in sequences),
            "prompt_aligned_exact_sequences": sum(
                row["prompt_aligned_exact_sequence"] for row in sequences),
            "prompt_aligned_tasks": sum(
                row["prompt_tokens_equal_native"] for row in sequences),
            "sequences": sequences,
        },
        "traffic": {
            "provider_positions": total_positions,
            "logical_authoritative_bytes": total_logical,
            "logical_gib_per_position": logical_gib,
            "exact_fallback_bytes": total_fallback,
            "exact_fallback_gib_per_position": (
                total_fallback / total_positions / (1024 ** 3)),
            "physical_direct_read_bytes": total_physical,
            "physical_gib_per_position": physical_gib,
            "direct_io_ns": total_direct_io_ns,
        },
        "terminal_kl": {
            "available": False,
            "reason": "The immutable native artifact retains IDs/text but not native logits; prompt-token drift is also possible and is reported per task.",
        },
        "gate": {
            "passed": gate_passed,
            "all_behaviors_retained": tasks_passed == len(sequences),
            "schema_false_activation_absent": True,
            "logical_below_1_2_gib": logical_gib < 1.2,
            "decision": (
                "Proceed to tool-declared false-positive and broader structured/agentic gates; keep research-only."
                if gate_passed else
                "Stop route12 production consideration."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "tasks_passed": f"{tasks_passed}/{len(sequences)}",
        "logical_gib_per_position": logical_gib,
        "physical_gib_per_position": physical_gib,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

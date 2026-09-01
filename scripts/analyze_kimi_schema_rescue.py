#!/usr/bin/env python3
"""Validate the preregistered K3 tool-schema rescue gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    P20,
    RESCUE,
    digest,
    i32le_digest,
    read_environment,
    token_traces,
    traffic,
    valid_weather_call,
    verify_manifest,
)


SCHEMA_RESCUE = re.compile(
    r"\[kimi-k3-progressive-schema-rescue\] base-pos=(\d+) slab-budget=(\d+)")


def analyze_arm(root: Path) -> dict:
    verify_manifest(root)
    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    p20 = P20.findall(stderr)
    if len(p20) != 1:
        raise ValueError(f"{root}: expected one P20 summary")
    explicit_reads, physical_bytes, direct_io_ns = map(int, p20[0])
    response = json.loads((root / "response.json").read_text())
    byte_metrics = traffic(root / "traffic.tsv")
    byte_metrics.update({
        "explicit_provider_reads": explicit_reads,
        "direct_physical_bytes": physical_bytes,
        "direct_io_ns": direct_io_ns,
    })
    choice = response["choices"][0]
    return {
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "environment": {
            key: value for key, value in read_environment(
                root / "environment.nul").items()
            if key.startswith("DFLASH_KIMI_EXPERIMENT_")
        },
        "prompt_count": len(traces["prompt_ids"]),
        "prompt_token_ids_i32le_sha256": i32le_digest(traces["prompt_ids"]),
        "generated_ids": traces["generated_ids"],
        "response": {
            "finish_reason": choice["finish_reason"],
            "content": choice["message"].get("content", ""),
            "usage": response["usage"],
        },
        "schema_prefix_configurations": stderr.count(
            "[kimi-k3-schema-rescue] configured-prefixes="),
        "schema_rescue_markers": [
            [int(pos), int(budget)]
            for pos, budget in SCHEMA_RESCUE.findall(stderr)
        ],
        "static_position_markers": len(RESCUE.findall(stderr)),
        "request_wide_b96_markers": stderr.count(
            "request minimum slab budget=96"),
        "logits_sha256": digest(root / "final.f32"),
        "traffic_sha256": digest(root / "traffic.tsv"),
        "traffic": byte_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--positional-root", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--non-tool-off-root", type=Path, required=True)
    parser.add_argument("--non-tool-on-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    tool = analyze_arm(args.tool_root)
    off = analyze_arm(args.non_tool_off_root)
    on = analyze_arm(args.non_tool_on_root)
    if len({tool["source_commit"], off["source_commit"], on["source_commit"]}) != 1:
        raise ValueError("gate arms used different source commits")
    if len({tool["executable_sha256"], off["executable_sha256"],
            on["executable_sha256"]}) != 1:
        raise ValueError("gate arms used different executables")

    tool_environment = tool["environment"]
    if tool_environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1":
        raise ValueError("tool schema trigger was not enabled")
    if "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in tool_environment:
        raise ValueError("tool arm contains a static position override")
    expected_prompt = prereg["fixtures"]["tool"][
        "frozen_prompt_token_ids_i32le_sha256"]
    if (tool["prompt_count"] != 147 or
            tool["prompt_token_ids_i32le_sha256"] != expected_prompt):
        raise ValueError("tool prompt alignment changed")
    if (tool["schema_prefix_configurations"] != 1 or
            tool["schema_rescue_markers"] != [[158, 96]] or
            tool["static_position_markers"] != 0 or
            tool["request_wide_b96_markers"] != 0):
        raise ValueError("tool trigger marker contract changed")
    tool_response = json.loads((args.tool_root / "response.json").read_text())
    tool_valid, tool_call = valid_weather_call(tool_response)
    if not tool_valid:
        raise ValueError("schema trigger did not recover get_weather")

    verify_manifest(args.positional_root)
    positional_traces = token_traces(
        (args.positional_root / "server.stderr").read_text())
    mechanism_equal = {
        "generated_ids": tool["generated_ids"] == positional_traces["generated_ids"],
        "logits": tool["logits_sha256"] == digest(
            args.positional_root / "final.f32"),
        "traffic": tool["traffic_sha256"] == digest(
            args.positional_root / "traffic.tsv"),
    }
    if not all(mechanism_equal.values()):
        raise ValueError("runtime trigger differs from positional singleton")

    for name, arm, enabled in (("off", off, False), ("on", on, True)):
        present = arm["environment"].get(
            "DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") == "1"
        if present != enabled:
            raise ValueError(f"non-tool {name}: schema env changed")
        if (arm["schema_prefix_configurations"] != 0 or
                arm["schema_rescue_markers"] or
                arm["static_position_markers"] != 0 or
                arm["request_wide_b96_markers"] != 0):
            raise ValueError(f"non-tool {name}: unexpected marker")
    non_tool_equal = {
        "prompt_token_ids": (
            off["prompt_token_ids_i32le_sha256"] ==
            on["prompt_token_ids_i32le_sha256"]),
        "generated_ids": off["generated_ids"] == on["generated_ids"],
        "logits": off["logits_sha256"] == on["logits_sha256"],
        "traffic": off["traffic_sha256"] == on["traffic_sha256"],
        "logical_bytes": (
            off["traffic"]["total_provider_bytes"] ==
            on["traffic"]["total_provider_bytes"]),
        "fallback_bytes": (
            off["traffic"]["exact_fallback_bytes"] ==
            on["traffic"]["exact_fallback_bytes"]),
        "physical_bytes": (
            off["traffic"]["direct_physical_bytes"] ==
            on["traffic"]["direct_physical_bytes"]),
    }
    if not all(non_tool_equal.values()):
        raise ValueError("non-tool on/off control diverged")

    result = {
        "schema": "kimi-k3-tool-schema-rescue-result-v1",
        "status": "MEASURED_SCHEMA_RESCUE_GATE_GO",
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": tool["source_commit"],
            "executable_sha256": tool["executable_sha256"],
        },
        "tool_schema_trigger": {
            **tool,
            "tool_call_valid": tool_valid,
            "tool_call": tool_call,
            "positional_singleton_artifact_root": str(args.positional_root),
            "positional_mechanism_byte_identical": mechanism_equal,
        },
        "non_tool_off": off,
        "non_tool_on": on,
        "non_tool_on_off_byte_identical": non_tool_equal,
        "gate": {
            "passed": True,
            "decision": "Run a broader native-success tool suite and tool-declared false-positive controls. Keep research-only until that gate passes.",
        },
        "limitations": [
            "Only one multi-token tool name has a measured rescue.",
            "The no-tools control proves inertness outside tool requests, not precision inside all tool-declared requests.",
            "One-token tool names are outside this trigger's scope.",
            "No terminal full-vocabulary KL teacher capture was made for the generated sequence.",
            "Cold timings are not serving throughput.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "tool_logical_gib_per_position": (
            tool["traffic"]["logical_gib_per_position"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

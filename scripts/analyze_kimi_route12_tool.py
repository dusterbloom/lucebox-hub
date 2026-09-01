#!/usr/bin/env python3
"""Validate the preregistered K3 route12 full-tool follow-up."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import digest, valid_weather_call
from analyze_kimi_schema_rescue import analyze_arm


ROUTE_MARKER = re.compile(
    r"\[kimi-k3-route-limit\] top-routes=(\d+) weights=unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--route12-root", type=Path, required=True)
    parser.add_argument("--budget24-root", type=Path, required=True)
    parser.add_argument("--budget20-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    if digest(args.policy) != prereg["arm"]["policy_sha256"]:
        raise ValueError("Budget20 policy hash changed")
    route12 = analyze_arm(args.route12_root)
    budget24 = analyze_arm(args.budget24_root)
    budget20 = analyze_arm(args.budget20_root)
    if digest(args.route12_root / "request.json") != prereg["arm"][
            "fixture_sha256"]:
        raise ValueError("tool fixture hash changed")
    if route12["executable_sha256"] != prereg["source"][
            "executable_sha256"]:
        raise ValueError("route12 executable hash changed")
    if (route12["prompt_count"] != 147 or
            route12["prompt_token_ids_i32le_sha256"] !=
            budget24["prompt_token_ids_i32le_sha256"]):
        raise ValueError("tool prompt alignment changed")

    environment = route12["environment"]
    if (environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12" or
            environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1" or
            environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(args.policy)):
        raise ValueError("route12 experiment environment changed")
    if "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment:
        raise ValueError("route12 arm contains a position intervention")
    stderr = (args.route12_root / "server.stderr").read_text()
    route_markers = [int(value) for value in ROUTE_MARKER.findall(stderr)]
    if (route_markers != [12] or route12["schema_prefix_configurations"] != 1 or
            route12["schema_rescue_markers"] != [[158, 96]] or
            route12["static_position_markers"] != 0 or
            route12["request_wide_b96_markers"] != 0):
        raise ValueError("route/schema marker contract changed")

    response = json.loads((args.route12_root / "response.json").read_text())
    tool_valid, tool_call = valid_weather_call(response)
    logical = route12["traffic"]["logical_gib_per_position"]
    physical = (route12["traffic"]["direct_physical_bytes"] /
                route12["traffic"]["provider_positions"] / (1024 ** 3))
    gate_passed = tool_valid and logical < 1.2
    result = {
        "schema": "kimi-k3-route12-tool-result-v1",
        "status": ("MEASURED_ROUTE12_TOOL_GO" if gate_passed else
                   "MEASURED_ROUTE12_TOOL_NO_GO"),
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": route12["source_commit"],
        },
        "route12": {
            **route12,
            "route_limit_markers": route_markers,
            "tool_call_valid": tool_valid,
            "tool_call": tool_call,
            "physical_gib_per_position": physical,
        },
        "comparison": {
            "budget24_generated_ids_equal": (
                route12["generated_ids"] == budget24["generated_ids"]),
            "budget24_final_logits_equal": (
                route12["logits_sha256"] == budget24["logits_sha256"]),
            "logical_gib_per_position": {
                "invalid_budget20_route16": budget20["traffic"][
                    "logical_gib_per_position"],
                "valid_route12": logical,
                "valid_budget24": budget24["traffic"][
                    "logical_gib_per_position"],
            },
            "physical_gib_per_position": {
                "valid_route12": physical,
                "valid_budget24": (
                    budget24["traffic"]["direct_physical_bytes"] /
                    budget24["traffic"]["provider_positions"] / (1024 ** 3)),
            },
        },
        "gate": {
            "passed": gate_passed,
            "tool_valid": tool_valid,
            "logical_below_1_2_gib": logical < 1.2,
            "decision": (
                "Freeze route12 and validate held-out native-success quality plus additional tool-declared controls."
                if gate_passed else
                "Stop route12 behavioral promotion."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "tool_valid": tool_valid,
        "logical_gib_per_position": logical,
        "physical_gib_per_position": physical,
        "budget24_generated_ids_equal": result["comparison"][
            "budget24_generated_ids_equal"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

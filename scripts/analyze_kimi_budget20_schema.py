#!/usr/bin/env python3
"""Validate the one-shot K3 Budget20 schema interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import digest, valid_weather_call
from analyze_kimi_schema_rescue import analyze_arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--budget16-root", type=Path, required=True)
    parser.add_argument("--budget20-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    if digest(args.policy) != prereg["policy"]["sha256"]:
        raise ValueError("Budget20 policy hash changed")
    control = analyze_arm(args.control_root)
    budget16 = analyze_arm(args.budget16_root)
    budget20 = analyze_arm(args.budget20_root)
    if len({control["executable_sha256"], budget16["executable_sha256"],
            budget20["executable_sha256"]}) != 1:
        raise ValueError("comparison arms used different executables")
    if (budget20["prompt_count"] != 147 or
            budget20["prompt_token_ids_i32le_sha256"] !=
            prereg["fixture"]["prompt_token_ids_i32le_sha256"]):
        raise ValueError("Budget20 prompt alignment changed")
    environment = budget20["environment"]
    if environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1":
        raise ValueError("Budget20 schema trigger was not enabled")
    if environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(args.policy):
        raise ValueError("Budget20 policy path changed")
    if "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment:
        raise ValueError("Budget20 contains a static position override")
    if (budget20["schema_prefix_configurations"] != 1 or
            budget20["schema_rescue_markers"] or
            budget20["static_position_markers"] != 0 or
            budget20["request_wide_b96_markers"] != 0):
        raise ValueError("Budget20 marker contract changed")

    response = json.loads((args.budget20_root / "response.json").read_text())
    tool_valid, tool_call = valid_weather_call(response)
    if tool_valid:
        raise ValueError("Budget20 unexpectedly passed its NO-GO path")
    traffic20 = budget20["traffic"]
    traffic24 = control["traffic"]
    physical20 = (traffic20["direct_physical_bytes"] /
                  traffic20["provider_positions"] / (1024 ** 3))
    physical24 = (traffic24["direct_physical_bytes"] /
                  traffic24["provider_positions"] / (1024 ** 3))
    comparison = {
        "budget16_and_budget20_first_six_ids_equal": (
            budget16["generated_ids"][:6] == budget20["generated_ids"][:6]),
        "budget20_and_budget24_generated_ids_equal": (
            budget20["generated_ids"] == control["generated_ids"]),
        "logical_gib_per_position_budget24":
            traffic24["logical_gib_per_position"],
        "logical_gib_per_position_budget20":
            traffic20["logical_gib_per_position"],
        "logical_reduction_fraction": 1.0 - (
            traffic20["logical_gib_per_position"] /
            traffic24["logical_gib_per_position"]),
        "physical_gib_per_position_budget24": physical24,
        "physical_gib_per_position_budget20": physical20,
        "physical_reduction_fraction": 1.0 - physical20 / physical24,
    }
    result = {
        "schema": "kimi-k3-budget20-schema-rescue-result-v1",
        "status": "MEASURED_BUDGET20_SCHEMA_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": budget20["source_commit"],
            "executable_sha256": budget20["executable_sha256"],
            "policy_sha256": digest(args.policy),
        },
        "budget24_schema_control": control,
        "budget16_context": budget16,
        "budget20_schema": {
            **budget20,
            "tool_call_valid": tool_valid,
            "tool_call": tool_call,
        },
        "comparison": comparison,
        "gate": {
            "traffic_below_1_2_gib":
                traffic20["logical_gib_per_position"] < 1.2,
            "tool_valid": False,
            "joint_go": False,
            "decision": "Stop uniform-budget interpolation on this fixture; do not try Budget21/22/23. Investigate prompt-terminal allocation or cheap/rich pre-token disagreement.",
        },
        "limitations": [
            "One tool fixture cannot establish broad quality.",
            "No prompt-terminal teacher KL was captured.",
            "Cold timing is not serving throughput."
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "logical_gib_per_position": traffic20["logical_gib_per_position"],
        "physical_gib_per_position": physical20,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

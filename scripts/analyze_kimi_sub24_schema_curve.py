#!/usr/bin/env python3
"""Validate the stopped K3 sub-24 schema-rescue curve."""

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
    parser.add_argument("--budget16-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    policies = {entry["budget"]: entry for entry in prereg["policies"]}
    if digest(args.budget16_policy) != policies[16]["sha256"]:
        raise ValueError("Budget16 policy hash changed")
    for budget in (12, 8):
        if Path(policies[budget]["artifact_root"]).exists():
            raise ValueError(f"Budget{budget} ran after the stop condition")

    control = analyze_arm(args.control_root)
    budget16 = analyze_arm(args.budget16_root)
    if control["executable_sha256"] != budget16["executable_sha256"]:
        raise ValueError("control and Budget16 used different executables")
    if (budget16["prompt_count"] != 147 or
            budget16["prompt_token_ids_i32le_sha256"] !=
            prereg["fixture"]["prompt_token_ids_i32le_sha256"]):
        raise ValueError("Budget16 prompt alignment changed")

    environment = budget16["environment"]
    if environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1":
        raise ValueError("Budget16 schema trigger was not enabled")
    if environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(
            args.budget16_policy):
        raise ValueError("Budget16 policy path changed")
    if "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment:
        raise ValueError("Budget16 contains a static position override")
    if (budget16["schema_prefix_configurations"] != 1 or
            budget16["schema_rescue_markers"] or
            budget16["static_position_markers"] != 0 or
            budget16["request_wide_b96_markers"] != 0):
        raise ValueError("Budget16 marker contract changed")

    response = json.loads((args.budget16_root / "response.json").read_text())
    tool_valid, tool_call = valid_weather_call(response)
    if tool_valid:
        raise ValueError("Budget16 unexpectedly passed; continue curve instead")
    b16_traffic = budget16["traffic"]
    control_traffic = control["traffic"]
    comparison = {
        "generated_ids_equal": (
            budget16["generated_ids"] == control["generated_ids"]),
        "tool_valid_control": valid_weather_call(json.loads(
            (args.control_root / "response.json").read_text()))[0],
        "tool_valid_budget16": tool_valid,
        "logical_gib_per_position_delta": (
            b16_traffic["logical_gib_per_position"] -
            control_traffic["logical_gib_per_position"]),
        "logical_gib_per_position_reduction_fraction": 1.0 - (
            b16_traffic["logical_gib_per_position"] /
            control_traffic["logical_gib_per_position"]),
        "physical_gib_per_position_control": (
            control_traffic["direct_physical_bytes"] /
            control_traffic["provider_positions"] / (1024 ** 3)),
        "physical_gib_per_position_budget16": (
            b16_traffic["direct_physical_bytes"] /
            b16_traffic["provider_positions"] / (1024 ** 3)),
    }
    comparison["physical_gib_per_position_reduction_fraction"] = 1.0 - (
        comparison["physical_gib_per_position_budget16"] /
        comparison["physical_gib_per_position_control"])

    result = {
        "schema": "kimi-k3-sub24-schema-rescue-result-v1",
        "status": "MEASURED_SUB24_SCHEMA_B16_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": budget16["source_commit"],
            "executable_sha256": budget16["executable_sha256"],
            "budget16_policy_sha256": digest(args.budget16_policy),
        },
        "budget24_schema_control": control,
        "budget16_schema": {
            **budget16,
            "tool_call_valid": tool_valid,
            "tool_call": tool_call,
        },
        "comparison": comparison,
        "stop": {
            "fired": True,
            "budget12_run": False,
            "budget8_run": False,
            "interpretation": "Budget16 diverged before the declared tool-name prefix, so a final-name schema rescue cannot repair the earlier causal damage. Do not lower the base further on this fixture.",
        },
        "gates": {
            "below_budget24_quality": False,
            "below_1_2_gib_traffic": (
                b16_traffic["logical_gib_per_position"] < 1.2),
            "joint_quality_and_traffic": False,
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "logical_gib_per_position": b16_traffic["logical_gib_per_position"],
        "physical_gib_per_position": comparison[
            "physical_gib_per_position_budget16"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the preregistered K3 periodic-refresh discriminator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    analyze_arm,
    digest,
    read_environment,
)
from analyze_kimi_prompt_tail import markers, read_logits, terminal_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--budget20-root", type=Path, required=True)
    parser.add_argument("--budget24-root", type=Path, required=True)
    parser.add_argument("--period8-root", type=Path, required=True)
    parser.add_argument("--period4-root", type=Path, required=True)
    parser.add_argument("--budget20-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    if digest(args.budget20_policy) != prereg["policies"]["base"]["sha256"]:
        raise ValueError("Budget20 policy hash changed")
    expected_prompt = prereg["fixture"]["prompt_token_ids_i32le_sha256"]
    period8_positions = prereg["arms"][0]["position_budgets"]
    period4_positions = prereg["arms"][1]["position_budgets"]
    specs = {
        "budget20_control": (args.budget20_root, ""),
        "budget24_reference": (args.budget24_root, ""),
        "period8": (args.period8_root, period8_positions),
        "period4": (args.period4_root, period4_positions),
    }
    arms = {
        name: analyze_arm(root, positions, expected_prompt)
        for name, (root, positions) in specs.items()
    }
    if any(digest(root / "request.json") != prereg["fixture"]["sha256"]
           for root, _ in specs.values()):
        raise ValueError("first-token fixture hash changed")
    if len({arm["executable_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("arms used different executables")
    if arms["budget20_control"]["source_commit"] != prereg[
            "controls"]["control_source_commit"]:
        raise ValueError("retained control source changed")
    if arms["budget20_control"]["executable_sha256"] != prereg[
            "controls"]["executable_sha256"]:
        raise ValueError("retained executable changed")
    if len({arms[name]["source_commit"] for name in ("period8", "period4")}) != 1:
        raise ValueError("periodic arms used different source commits")
    if any(len(arm["generated_ids"]) != 1 for arm in arms.values()):
        raise ValueError("expected one generated token per arm")

    for name, (root, _) in specs.items():
        environment = read_environment(root / "environment.nul")
        if "DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE" in environment:
            raise ValueError(f"{name}: schema rescue must be disabled")
        arms[name]["environment"] = {
            key: value for key, value in environment.items()
            if key.startswith("DFLASH_KIMI_EXPERIMENT_") or
            key == "DFLASH_KIMI_H22_LAYER_BUDGETS"
        }

    expected_markers = {
        "budget20_control": [],
        "budget24_reference": [],
        "period8": [[position, 24] for position in range(7, 147, 8)],
        "period4": [[position, 24] for position in range(3, 147, 4)],
    }
    for name, (root, _) in specs.items():
        observed = markers(root)
        if observed != expected_markers[name]:
            raise ValueError(f"{name}: refresh markers changed")
        arms[name]["position_budget_markers"] = observed

    reference = read_logits(args.budget24_root / "final.f32")
    metrics = {
        name: terminal_metrics(reference, read_logits(root / "final.f32"))
        for name, (root, _) in specs.items()
    }
    control_kl = metrics["budget20_control"][
        "kl_budget24_reference_to_arm"]
    for name in ("period8", "period4"):
        metrics[name]["kl_reduction_vs_budget20_fraction"] = 1.0 - (
            metrics[name]["kl_budget24_reference_to_arm"] / control_kl)

    period4 = metrics["period4"]
    no_go = bool(not period4["top1_agrees_with_budget24"] or
                 period4["kl_reduction_vs_budget20_fraction"] <= 0.0 or
                 arms["period4"]["traffic"]["logical_gib_per_position"] >= 1.2)
    if not no_go:
        raise ValueError("period4 unexpectedly passed; analyzer expects NO-GO path")

    result = {
        "schema": "kimi-k3-periodic-refresh-result-v1",
        "status": "MEASURED_PERIODIC_REFRESH_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": arms["period4"]["source_commit"],
            "executable_sha256": arms["period4"]["executable_sha256"],
        },
        "arms": arms,
        "terminal_metrics": metrics,
        "gate": {
            "no_go": no_go,
            "decision": "Stop static periodic refresh. Neither one-in-eight nor one-in-four uniform-Budget24 rows restored the Budget24 first token; occasional rich rows do not reset the accumulated cheap-state error.",
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "period8_kl_reduction": metrics["period8"][
            "kl_reduction_vs_budget20_fraction"],
        "period4_kl_reduction": metrics["period4"][
            "kl_reduction_vs_budget20_fraction"],
        "period4_logical_gib_per_position": arms["period4"]["traffic"][
            "logical_gib_per_position"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

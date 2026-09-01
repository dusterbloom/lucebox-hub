#!/usr/bin/env python3
"""Validate the preregistered route12 plus uniform-Budget16 discriminator."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    analyze_arm,
    digest,
    read_environment,
)
from analyze_kimi_prompt_tail import read_logits, terminal_metrics


ROUTE_MARKER = re.compile(
    r"\[kimi-k3-route-limit\] top-routes=(\d+) weights=unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--budget24-root", type=Path, required=True)
    parser.add_argument("--route12-budget20-root", type=Path, required=True)
    parser.add_argument("--route16-budget16-root", type=Path, required=True)
    parser.add_argument("--route12-budget16-root", type=Path, required=True)
    parser.add_argument("--budget16-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    repo = Path(__file__).resolve().parent.parent
    for path, expected in (
            (Path(__file__), prereg["source"]["analyzer_sha256"]),
            (repo / "scripts/run_kimi_progressive_tool_rescue.sh",
             prereg["source"]["runner_sha256"]),
            (args.budget16_policy, prereg["policy"]["sha256"])):
        if digest(path) != expected:
            raise ValueError(f"registered input hash changed: {path}")

    expected_prompt = prereg["fixture"]["prompt_token_ids_i32le_sha256"]
    roots = {
        "budget24_reference": args.budget24_root,
        "route12_budget20_reference": args.route12_budget20_root,
        "route16_budget16": args.route16_budget16_root,
        "route12_budget16": args.route12_budget16_root,
    }
    arms = {
        name: analyze_arm(root, "", expected_prompt)
        for name, root in roots.items()
    }
    if any(digest(root / "request.json") != prereg["fixture"]["sha256"]
           for root in roots.values()):
        raise ValueError("first-token fixture hash changed")
    if any(len(arm["generated_ids"]) != 1 for arm in arms.values()):
        raise ValueError("expected one generated token per arm")
    if any(arm["traffic"]["provider_positions"] != 147
           for arm in arms.values()):
        raise ValueError("expected 147 provider positions per arm")

    for name, expected in prereg["retained_references"].items():
        arm = arms[name]
        if (arm["manifest_sha256"] != expected["manifest_sha256"] or
                arm["logits_sha256"] != expected["logits_sha256"] or
                arm["generated_ids"] != expected["generated_ids"]):
            raise ValueError(f"retained reference changed: {name}")

    new_names = ("route16_budget16", "route12_budget16")
    if len({arms[name]["source_commit"] for name in new_names}) != 1:
        raise ValueError("new arms used different source commits")
    if {arms[name]["executable_sha256"] for name in new_names} != {
            prereg["source"]["executable_sha256"]}:
        raise ValueError("new arms used a different executable")

    for name in new_names:
        root = roots[name]
        environment = read_environment(root / "environment.nul")
        observed = [int(value) for value in ROUTE_MARKER.findall(
            (root / "server.stderr").read_text())]
        route12 = name == "route12_budget16"
        if (environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") !=
                str(args.budget16_policy) or
                "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment or
                "DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE" in environment or
                (environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") == "12")
                != route12 or observed != ([12] if route12 else [])):
            raise ValueError(f"intervention contract changed: {name}")
        arm = arms[name]
        arm["route_limit"] = 12 if route12 else 16
        arm["route_limit_markers"] = observed
        arm["environment"] = {
            key: value for key, value in environment.items()
            if key.startswith("DFLASH_KIMI_EXPERIMENT_") or
            key == "DFLASH_KIMI_H22_LAYER_BUDGETS"
        }

    reference = read_logits(args.budget24_root / "final.f32")
    metrics = {
        name: terminal_metrics(reference, read_logits(root / "final.f32"))
        for name, root in roots.items()
    }
    control_kl = metrics["route16_budget16"][
        "kl_budget24_reference_to_arm"]
    candidate_kl = metrics["route12_budget16"][
        "kl_budget24_reference_to_arm"]
    kl_reduction = 1.0 - candidate_kl / control_kl if control_kl else 0.0
    candidate = arms["route12_budget16"]
    control = arms["route16_budget16"]
    candidate_metrics = metrics["route12_budget16"]
    byte_reduction = 1.0 - (
        candidate["traffic"]["total_provider_bytes"] /
        control["traffic"]["total_provider_bytes"])
    passed = bool(
        candidate_metrics["top1_agrees_with_budget24"] and
        candidate_metrics["budget24_top1_margin"] > 0.0 and
        kl_reduction >= prereg["gate"]["minimum_kl_reduction_fraction"] and
        candidate["traffic"]["logical_gib_per_position"] <=
        prereg["gate"]["maximum_logical_gib_per_position"] and
        byte_reduction > 0.0)
    status = ("MEASURED_ROUTE12_BUDGET16_GO" if passed else
              "MEASURED_ROUTE12_BUDGET16_NO_GO")
    result = {
        "schema": "kimi-k3-route12-budget16-result-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": arms["route16_budget16"]["source_commit"],
        },
        "arms": arms,
        "terminal_metrics": metrics,
        "comparison": {
            "kl_reduction_vs_route16_budget16_fraction": kl_reduction,
            "logical_byte_reduction_vs_route16_budget16_fraction":
                byte_reduction,
            "logical_gib_per_position_delta_vs_route12_budget20":
                candidate["traffic"]["logical_gib_per_position"] -
                arms["route12_budget20_reference"]["traffic"][
                    "logical_gib_per_position"],
        },
        "gate": {
            "passed": passed,
            "decision": (
                "Run one separately preregistered full get_weather sequence with route12/Budget16 plus the already validated schema rescue."
                if passed else
                "Stop route12/Budget16. Do not spend a broad suite on this joint allocation."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": status,
        "top1": candidate_metrics["top1"],
        "budget24_top1_margin": candidate_metrics["budget24_top1_margin"],
        "kl": candidate_kl,
        "kl_reduction": kl_reduction,
        "logical_gib_per_position": candidate["traffic"][
            "logical_gib_per_position"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

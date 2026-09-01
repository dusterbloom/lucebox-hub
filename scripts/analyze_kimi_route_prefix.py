#!/usr/bin/env python3
"""Validate the preregistered K3 route-prefix screen."""

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
    parser.add_argument("--prereg-v2", type=Path, required=True)
    parser.add_argument("--old-control-root", type=Path, required=True)
    parser.add_argument("--route16-root", type=Path, required=True)
    parser.add_argument("--budget24-root", type=Path, required=True)
    for count in (12, 8, 6, 4):
        parser.add_argument(f"--route{count}-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    amendment = json.loads(args.prereg_v2.read_text())
    if digest(args.policy) != prereg["policy"]["sha256"]:
        raise ValueError("Budget20 policy hash changed")
    expected_prompt = prereg["fixture"]["prompt_token_ids_i32le_sha256"]
    roots = {
        "old_route16_control": args.old_control_root,
        "route16": args.route16_root,
        "budget24_reference": args.budget24_root,
        "route12": args.route12_root,
        "route8": args.route8_root,
        "route6": args.route6_root,
        "route4": args.route4_root,
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

    closure = {
        "prompt_token_ids": (
            arms["old_route16_control"]["prompt_token_ids_i32le_sha256"] ==
            arms["route16"]["prompt_token_ids_i32le_sha256"]),
        "generated_ids": (
            arms["old_route16_control"]["generated_ids"] ==
            arms["route16"]["generated_ids"]),
        "logits": (arms["old_route16_control"]["logits_sha256"] ==
                   arms["route16"]["logits_sha256"]),
        "traffic": (arms["old_route16_control"]["traffic_sha256"] ==
                    arms["route16"]["traffic_sha256"]),
    }
    if not all(closure.values()):
        raise ValueError("route16 binary closure failed")

    new_names = ["route16", "route12", "route8", "route6", "route4"]
    if len({arms[name]["source_commit"] for name in new_names}) != 1:
        raise ValueError("new-binary arms used different source commits")
    if len({arms[name]["executable_sha256"] for name in new_names}) != 1:
        raise ValueError("new-binary arms used different executables")
    if arms["route16"]["executable_sha256"] != amendment["build"][
            "executable_sha256"]:
        raise ValueError("route-screen executable hash changed")

    for name, root in roots.items():
        environment = read_environment(root / "environment.nul")
        observed = [int(value) for value in ROUTE_MARKER.findall(
            (root / "server.stderr").read_text())]
        expected_count = (int(name.removeprefix("route"))
                          if name in new_names and name != "route16" else 16)
        if name in ("old_route16_control", "route16", "budget24_reference"):
            if "DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT" in environment or observed:
                raise ValueError(f"{name}: unexpected route intervention")
        else:
            if (environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") !=
                    str(expected_count) or observed != [expected_count]):
                raise ValueError(f"{name}: route intervention changed")
        arms[name]["route_limit"] = expected_count
        arms[name]["route_limit_markers"] = observed
        arms[name]["environment"] = {
            key: value for key, value in environment.items()
            if key.startswith("DFLASH_KIMI_EXPERIMENT_") or
            key == "DFLASH_KIMI_H22_LAYER_BUDGETS"
        }

    reference = read_logits(args.budget24_root / "final.f32")
    metrics = {
        name: terminal_metrics(reference, read_logits(root / "final.f32"))
        for name, root in roots.items()
        if name != "old_route16_control"
    }
    control_kl = metrics["route16"]["kl_budget24_reference_to_arm"]
    control_bytes = arms["route16"]["traffic"]["logical_gib_per_position"]
    for count in (12, 8, 6, 4):
        name = f"route{count}"
        metrics[name]["kl_reduction_vs_route16_fraction"] = 1.0 - (
            metrics[name]["kl_budget24_reference_to_arm"] / control_kl)
        metrics[name]["logical_byte_reduction_vs_route16_fraction"] = 1.0 - (
            arms[name]["traffic"]["logical_gib_per_position"] / control_bytes)

    candidates = [f"route{count}" for count in (12, 8, 6, 4)]
    go_arms = [name for name in candidates
               if metrics[name]["top1_agrees_with_budget24"] and
               metrics[name]["kl_reduction_vs_route16_fraction"] >= 0.5 and
               arms[name]["traffic"]["logical_gib_per_position"] <=
               control_bytes]
    partial_arms = [name for name in candidates
                    if not metrics[name]["top1_agrees_with_budget24"] and
                    metrics[name]["kl_reduction_vs_route16_fraction"] >= 0.5 and
                    metrics[name]["budget24_top1_margin"] >
                    metrics["route16"]["budget24_top1_margin"]]
    best = min(candidates, key=lambda name: metrics[name][
        "kl_budget24_reference_to_arm"])
    status = ("MEASURED_ROUTE_PREFIX_GO" if go_arms else
              "MEASURED_ROUTE_PREFIX_PARTIAL" if partial_arms else
              "MEASURED_ROUTE_PREFIX_NO_GO")
    result = {
        "schema": "kimi-k3-route-prefix-result-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "preregistration_v2_sha256": digest(args.prereg_v2),
        "source": {
            **prereg["source"],
            "implementation_commit": amendment["source"][
                "implementation_commit"],
            "measured_commit": arms["route16"]["source_commit"],
            "executable_sha256": arms["route16"]["executable_sha256"],
        },
        "binary_closure": closure,
        "arms": arms,
        "terminal_metrics": metrics,
        "gate": {
            "go_arms": go_arms,
            "partial_arms": partial_arms,
            "best_kl_arm": best,
            "decision": (
                "Run the best GO arm for full tool generation."
                if go_arms else
                "Only a separately preregistered retained-weight renormalization check is earned."
                if partial_arms else
                "Stop the unrenormalized global route-prefix policy."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": status,
        "go_arms": go_arms,
        "partial_arms": partial_arms,
        "best_kl_arm": best,
        "metrics": {name: {
            "top1": metrics[name]["top1"],
            "kl": metrics[name]["kl_budget24_reference_to_arm"],
            "kl_reduction": metrics[name].get(
                "kl_reduction_vs_route16_fraction"),
            "logical_gib_per_position": arms[name]["traffic"][
                "logical_gib_per_position"],
        } for name in ["route16", *candidates]},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

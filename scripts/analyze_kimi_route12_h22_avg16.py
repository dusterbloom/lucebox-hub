#!/usr/bin/env python3
"""Validate H22-ranked equal-average Budget16 policies under route12."""

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


def policy_average(path: Path) -> tuple[int, float, dict[int, int]]:
    budgets = []
    counts: dict[int, int] = {}
    layers = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        layer_text, budget_text = line.split()
        layer, budget = int(layer_text), int(budget_text)
        layers.append(layer)
        budgets.append(budget)
        counts[budget] = counts.get(budget, 0) + 1
    if layers != list(range(1, 93)):
        raise ValueError(f"policy does not cover layers 1--92: {path}")
    return sum(budgets), sum(budgets) / len(budgets), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--budget24-root", type=Path, required=True)
    parser.add_argument("--route12-budget20-root", type=Path, required=True)
    parser.add_argument("--uniform16-root", type=Path, required=True)
    parser.add_argument("--conservative-root", type=Path, required=True)
    parser.add_argument("--sharp-root", type=Path, required=True)
    parser.add_argument("--conservative-policy", type=Path, required=True)
    parser.add_argument("--sharp-policy", type=Path, required=True)
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
            (args.conservative_policy,
             prereg["policies"]["conservative"]["sha256"]),
            (args.sharp_policy, prereg["policies"]["sharp"]["sha256"])):
        if digest(path) != expected:
            raise ValueError(f"registered input hash changed: {path}")
    policy_summaries = {}
    for name, path in (("conservative", args.conservative_policy),
                       ("sharp", args.sharp_policy)):
        total, average, counts = policy_average(path)
        if total != 1472 or average != 16.0:
            raise ValueError(f"{name}: policy is not exact average 16")
        policy_summaries[name] = {
            "nominal_slab_sum": total,
            "average_nominal_slabs": average,
            "budget_counts": counts,
        }

    expected_prompt = prereg["fixture"]["prompt_token_ids_i32le_sha256"]
    roots = {
        "budget24_reference": args.budget24_root,
        "route12_budget20_reference": args.route12_budget20_root,
        "route12_uniform16_reference": args.uniform16_root,
        "route12_h22_conservative": args.conservative_root,
        "route12_h22_sharp": args.sharp_root,
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

    new_specs = {
        "route12_h22_conservative": args.conservative_policy,
        "route12_h22_sharp": args.sharp_policy,
    }
    if len({arms[name]["source_commit"] for name in new_specs}) != 1:
        raise ValueError("new arms used different source commits")
    if {arms[name]["executable_sha256"] for name in new_specs} != {
            prereg["source"]["executable_sha256"]}:
        raise ValueError("new arms used a different executable")
    expected_nominal = arms["route12_uniform16_reference"]["traffic"][
        "requested_nominal_slabs"]
    for name, policy in new_specs.items():
        root = roots[name]
        environment = read_environment(root / "environment.nul")
        observed = [int(value) for value in ROUTE_MARKER.findall(
            (root / "server.stderr").read_text())]
        if (environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(policy) or
                environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12" or
                "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment or
                "DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE" in environment or
                observed != [12] or
                arms[name]["traffic"]["requested_nominal_slabs"] !=
                expected_nominal):
            raise ValueError(f"intervention contract changed: {name}")
        arms[name]["route_limit"] = 12
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
    }
    uniform = arms["route12_uniform16_reference"]
    uniform_kl = metrics["route12_uniform16_reference"][
        "kl_budget24_reference_to_arm"]
    candidates = {}
    for name in new_specs:
        arm = arms[name]
        measured = metrics[name]
        kl = measured["kl_budget24_reference_to_arm"]
        kl_reduction = 1.0 - kl / uniform_kl if uniform_kl else 0.0
        logical = arm["traffic"]["logical_gib_per_position"]
        passed = bool(
            measured["top1_agrees_with_budget24"] and
            measured["budget24_top1_margin"] > 0.0 and
            kl_reduction >= prereg["gate"]["minimum_kl_reduction_fraction"] and
            logical <= prereg["gate"]["maximum_logical_gib_per_position"])
        candidates[name] = {
            "passed": passed,
            "kl_reduction_vs_uniform16_fraction": kl_reduction,
            "logical_gib_per_position_delta_vs_uniform16":
                logical - uniform["traffic"]["logical_gib_per_position"],
        }
    go_arms = [name for name, row in candidates.items() if row["passed"]]
    best = min(new_specs, key=lambda name: metrics[name][
        "kl_budget24_reference_to_arm"])
    status = ("MEASURED_ROUTE12_H22_AVG16_GO" if go_arms else
              "MEASURED_ROUTE12_H22_AVG16_NO_GO")
    result = {
        "schema": "kimi-k3-route12-h22-avg16-result-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": arms["route12_h22_conservative"][
                "source_commit"],
        },
        "policy_summaries": policy_summaries,
        "arms": arms,
        "terminal_metrics": metrics,
        "candidates": candidates,
        "gate": {
            "passed": bool(go_arms),
            "go_arms": go_arms,
            "best_kl_arm": best,
            "decision": (
                "Run only the best passing policy on one separately preregistered full get_weather sequence with schema rescue."
                if go_arms else
                "Stop H22 quartile average-16 allocation; the budget-96 layer atlas does not transfer strongly enough to this low-budget composed state."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": status,
        "go_arms": go_arms,
        "best": best,
        "candidates": {
            name: {
                "top1": metrics[name]["top1"],
                "margin": metrics[name]["budget24_top1_margin"],
                "kl": metrics[name]["kl_budget24_reference_to_arm"],
                "kl_reduction": row[
                    "kl_reduction_vs_uniform16_fraction"],
                "logical_gib_per_position": arms[name]["traffic"][
                    "logical_gib_per_position"],
            }
            for name, row in candidates.items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

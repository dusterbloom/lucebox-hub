#!/usr/bin/env python3
"""Validate the preregistered K3 prompt-tail discriminator."""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    RESCUE,
    analyze_arm,
    digest,
    read_environment,
)


def read_logits(path: Path) -> array.array:
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if not values:
        raise ValueError(f"empty logits: {path}")
    return values


def logsumexp(values: array.array) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(
        math.exp(value - maximum) for value in values))


def top_two(values: array.array) -> tuple[int, float, float]:
    top_index = max(range(len(values)), key=values.__getitem__)
    top = values[top_index]
    second = max(value for index, value in enumerate(values)
                 if index != top_index)
    return top_index, top, second


def terminal_metrics(reference: array.array,
                     candidate: array.array) -> dict[str, int | float | bool]:
    if len(reference) != len(candidate):
        raise ValueError("logit vocabulary sizes differ")
    ref_top, _, _ = top_two(reference)
    candidate_top, candidate_top_logit, candidate_second = top_two(candidate)
    ref_log_z = logsumexp(reference)
    candidate_log_z = logsumexp(candidate)
    kl = math.fsum(
        math.exp(ref - ref_log_z) *
        ((ref - ref_log_z) - (value - candidate_log_z))
        for ref, value in zip(reference, candidate))
    teacher_other = max(value for index, value in enumerate(candidate)
                        if index != ref_top)
    return {
        "vocabulary": len(reference),
        "top1": candidate_top,
        "top1_agrees_with_budget24": candidate_top == ref_top,
        "top1_margin": candidate_top_logit - candidate_second,
        "budget24_top1_logit": candidate[ref_top],
        "budget24_top1_margin": candidate[ref_top] - teacher_other,
        "kl_budget24_reference_to_arm": max(0.0, kl),
    }


def markers(root: Path) -> list[list[int]]:
    return [[int(position), int(budget)]
            for position, budget in RESCUE.findall(
                (root / "server.stderr").read_text())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--budget20-root", type=Path, required=True)
    parser.add_argument("--budget24-root", type=Path, required=True)
    parser.add_argument("--tail1-root", type=Path, required=True)
    parser.add_argument("--tail8-root", type=Path, required=True)
    parser.add_argument("--budget20-policy", type=Path, required=True)
    parser.add_argument("--budget24-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    expected_prompt = prereg["fixture"][
        "expected_prompt_token_ids_i32le_sha256"]
    if digest(args.budget20_policy) != prereg["policies"]["budget20"]["sha256"]:
        raise ValueError("Budget20 policy hash changed")
    if digest(args.budget24_policy) != prereg["policies"][
            "budget24_reference"]["sha256"]:
        raise ValueError("Budget24 policy hash changed")

    arm_specs = {
        "budget20_control": (args.budget20_root, ""),
        "budget24_reference": (args.budget24_root, ""),
        "budget20_last1_budget24": (args.tail1_root, "146:24"),
        "budget20_last8_budget24": (
            args.tail8_root,
            "139:24,140:24,141:24,142:24,143:24,144:24,145:24,146:24"),
    }
    arms = {
        name: analyze_arm(root, positions, expected_prompt)
        for name, (root, positions) in arm_specs.items()
    }
    if any(digest(root / "request.json") != prereg["fixture"]["sha256"]
           for root, _ in arm_specs.values()):
        raise ValueError("first-token fixture hash changed")
    if len({arm["source_commit"] for arm in arms.values()}) != 1:
        raise ValueError("arms used different source commits")
    if len({arm["executable_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("arms used different executables")
    if any(len(arm["generated_ids"]) != 1 for arm in arms.values()):
        raise ValueError("expected exactly one generated token per arm")
    if any(arm["traffic"]["provider_positions"] != 147
           for arm in arms.values()):
        raise ValueError("expected 147 provider positions per arm")
    for name, (root, _) in arm_specs.items():
        environment = read_environment(root / "environment.nul")
        if "DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE" in environment:
            raise ValueError(f"{name}: schema rescue must be disabled")
        expected_policy = (str(args.budget24_policy)
                           if name == "budget24_reference"
                           else str(args.budget20_policy))
        if environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != expected_policy:
            raise ValueError(f"{name}: policy path changed")
        arms[name]["environment"] = {
            key: value for key, value in environment.items()
            if key.startswith("DFLASH_KIMI_EXPERIMENT_") or
            key == "DFLASH_KIMI_H22_LAYER_BUDGETS"
        }

    expected_markers = {
        "budget20_control": [],
        "budget24_reference": [],
        "budget20_last1_budget24": [[146, 24]],
        "budget20_last8_budget24": [[position, 24]
                                     for position in range(139, 147)],
    }
    for name, arm in arms.items():
        observed = markers(arm_specs[name][0])
        if observed != expected_markers[name]:
            raise ValueError(f"{name}: intervention markers changed")
        arm["position_budget_markers"] = observed

    reference = read_logits(args.budget24_root / "final.f32")
    metrics = {
        name: terminal_metrics(reference, read_logits(root / "final.f32"))
        for name, (root, _) in arm_specs.items()
    }
    control_kl = metrics["budget20_control"][
        "kl_budget24_reference_to_arm"]
    for name in ("budget20_last1_budget24", "budget20_last8_budget24"):
        kl = metrics[name]["kl_budget24_reference_to_arm"]
        metrics[name]["kl_reduction_vs_budget20_fraction"] = (
            1.0 - kl / control_kl if control_kl else 0.0)

    tail1 = metrics["budget20_last1_budget24"]
    tail8 = metrics["budget20_last8_budget24"]
    strong_go = bool(tail1["top1_agrees_with_budget24"] and
                     tail1["kl_reduction_vs_budget20_fraction"] >= 0.5)
    partial_go = bool(tail1["top1_agrees_with_budget24"] and
                      tail1["kl_reduction_vs_budget20_fraction"] > 0.0)
    no_go = bool(not tail8["top1_agrees_with_budget24"] or
                 tail8["kl_reduction_vs_budget20_fraction"] <= 0.0)
    if not no_go:
        raise ValueError("tail8 unexpectedly passed; analyzer expects NO-GO path")

    result = {
        "schema": "kimi-k3-prompt-tail-rescue-result-v1",
        "status": "MEASURED_PROMPT_TAIL_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": next(iter(arms.values()))["source_commit"],
            "executable_sha256": next(iter(arms.values()))[
                "executable_sha256"],
        },
        "arms": arms,
        "terminal_metrics": metrics,
        "gate": {
            "strong_go": strong_go,
            "partial_go": partial_go,
            "no_go": no_go,
            "decision": "Stop static prompt-tail rescue. The Budget20 damage is not repaired by eight final Budget24 prompt rows; test a label-free recurrent-state refresh or cheap/rich disagreement signal instead of tuning the tail length.",
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "budget20_kl": control_kl,
        "tail1_kl_reduction": tail1["kl_reduction_vs_budget20_fraction"],
        "tail8_kl_reduction": tail8["kl_reduction_vs_budget20_fraction"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

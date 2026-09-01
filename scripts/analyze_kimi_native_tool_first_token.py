#!/usr/bin/env python3
"""Compare native Full192, H23 and route12/B20 on one aligned tool token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    P20,
    digest,
    i32le_digest,
    read_environment,
    token_traces,
    traffic,
    verify_manifest,
)
from analyze_kimi_terminal_full_screen import read_logits, terminal_metrics


def analyze_arm(root: Path, prompt_sha: str) -> tuple[dict, object]:
    verify_manifest(root)
    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    if len(traces["prompt_ids"]) != 147 or len(traces["generated_ids"]) != 1:
        raise ValueError(f"unexpected token geometry: {root}")
    if i32le_digest(traces["prompt_ids"]) != prompt_sha:
        raise ValueError(f"prompt token hash changed: {root}")
    logits_path = root / "final.f32"
    byte_metrics = traffic(root / "traffic.tsv")
    p20 = P20.findall(stderr)
    if len(p20) > 1:
        raise ValueError(f"multiple P20 summaries: {root}")
    physical = None
    if p20:
        explicit_reads, physical_bytes, direct_io_ns = map(int, p20[0])
        physical = {
            "explicit_provider_reads": explicit_reads,
            "direct_physical_bytes": physical_bytes,
            "direct_physical_gib_per_position": physical_bytes
            / byte_metrics["provider_positions"] / (1024 ** 3),
            "direct_io_ns": direct_io_ns,
        }
    response = json.loads((root / "response.json").read_text())
    return ({
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "command_sha256": digest(root / "command.nul"),
        "environment_sha256": digest(root / "environment.nul"),
        "response_sha256": digest(root / "response.json"),
        "logits_sha256": digest(logits_path),
        "traffic_sha256": digest(root / "traffic.tsv"),
        "prompt_token_ids_i32le_sha256": i32le_digest(traces["prompt_ids"]),
        "generated_ids": traces["generated_ids"],
        "environment": {key: value for key, value in read_environment(
            root / "environment.nul").items() if key.startswith("DFLASH_KIMI")},
        "usage": response["usage"],
        "traffic": byte_metrics,
        "physical": physical,
    }, read_logits(logits_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--h23-root", type=Path, required=True)
    parser.add_argument("--moonshot-root", type=Path, required=True)
    parser.add_argument("--route12-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    prompt_sha = prereg["fixture"]["prompt_token_ids_i32le_sha256"]
    native, native_logits = analyze_arm(args.native_root, prompt_sha)
    h23, h23_logits = analyze_arm(args.h23_root, prompt_sha)
    moonshot, moonshot_logits = analyze_arm(args.moonshot_root, prompt_sha)
    route12, route12_logits = analyze_arm(args.route12_root, prompt_sha)
    arms = {"native_full192": native, "h23_aggressive1p8": h23,
            "h23_moonshot1p2": moonshot, "route12_budget20": route12}
    if len({arm["source_commit"] for arm in arms.values()}) != 1:
        raise ValueError("arms used different source commits")
    if len({arm["executable_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("arms used different executables")

    native_env, h23_env, moonshot_env, route_env = (
        arm["environment"] for arm in arms.values())
    if (native_env.get("DFLASH_KIMI_P20_SLAB_BUDGET") != "192"
            or "DFLASH_KIMI_H22_LAYER_BUDGETS" in native_env
            or "DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT" in native_env):
        raise ValueError("native arm is not clean Full192")
    if h23_env.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != prereg["arms"]["h23"]["policy"]:
        raise ValueError("H23 policy changed")
    if "DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT" in h23_env:
        raise ValueError("H23 route count changed")
    if (moonshot_env.get("DFLASH_KIMI_H22_LAYER_BUDGETS")
            != prereg["arms"]["moonshot"]["policy"]
            or "DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT" in moonshot_env):
        raise ValueError("moonshot policy changed")
    if (route_env.get("DFLASH_KIMI_H22_LAYER_BUDGETS")
            != prereg["arms"]["route12_budget20"]["policy"]
            or route_env.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12"):
        raise ValueError("route12/B20 policy changed")

    h23_metrics = terminal_metrics(native_logits, h23_logits)
    moonshot_metrics = terminal_metrics(native_logits, moonshot_logits)
    route_metrics = terminal_metrics(native_logits, route12_logits)
    h23.update(h23_metrics)
    moonshot.update(moonshot_metrics)
    route12.update(route_metrics)
    teacher_top = int(native["generated_ids"][0])
    if (teacher_top != h23_metrics["teacher_top1"]
            or teacher_top != moonshot_metrics["teacher_top1"]
            or teacher_top != route_metrics["teacher_top1"]):
        raise ValueError("native emitted ID disagrees with captured native argmax")

    route_bytes = route12["traffic"]["logical_gib_per_position"]
    h23_bytes = h23["traffic"]["logical_gib_per_position"]
    behavior_gate = (
        route_metrics["top1_agreement"]
        and route_metrics["candidate_teacher_margin"] > 0
        and route_bytes < 1.2
    )
    strong_gate = behavior_gate and route_metrics["terminal_kl"] <= h23_metrics["terminal_kl"]
    if strong_gate:
        status = "MEASURED_NATIVE_KL_STRONG_GO"
    elif behavior_gate:
        status = "MEASURED_NATIVE_KL_BEHAVIORAL_GO"
    else:
        status = "MEASURED_NATIVE_KL_NO_GO"

    result = {
        "schema": "kimi-k3-native-tool-first-token-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": prereg["source"],
        "fixture": prereg["fixture"],
        "arms": arms,
        "comparison": {
            "route12_minus_h23_terminal_kl": route_metrics["terminal_kl"] - h23_metrics["terminal_kl"],
            "route12_over_h23_terminal_kl": (
                route_metrics["terminal_kl"] / h23_metrics["terminal_kl"]
                if h23_metrics["terminal_kl"] else None),
            "route12_minus_h23_logical_gib_per_position": route_bytes - h23_bytes,
            "route12_logical_byte_reduction_fraction": 1.0 - route_bytes / h23_bytes,
            "route12_minus_moonshot_terminal_kl": route_metrics["terminal_kl"] - moonshot_metrics["terminal_kl"],
            "route12_over_moonshot_terminal_kl": (
                route_metrics["terminal_kl"] / moonshot_metrics["terminal_kl"]
                if moonshot_metrics["terminal_kl"] else None),
            "route12_minus_moonshot_logical_gib_per_position": route_bytes
            - moonshot["traffic"]["logical_gib_per_position"],
        },
        "gate": {
            "behavioral_pass": behavior_gate,
            "strong_distributional_pass": strong_gate,
            "interpretation": (
                "Advance route12/B20 to broader source-matched native KL measurement."
                if behavior_gate else
                "Stop route12/B20; do not tune this fixture."
            ),
        },
        "limitations": [
            "One aligned tool-boundary position is a discriminator, not a quality frontier.",
            "A correct top-one token is not distributional equivalence.",
            "Cold request timing is not serving throughput.",
            "This comparison tests the frozen route/slab policies, not a new terminal selector.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": status,
        "h23_kl": h23_metrics["terminal_kl"],
        "moonshot_kl": moonshot_metrics["terminal_kl"],
        "route12_kl": route_metrics["terminal_kl"],
        "route12_gib_per_position": route_bytes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

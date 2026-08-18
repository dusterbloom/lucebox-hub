#!/usr/bin/env python3
"""Assemble the selective KDA-core requantization decision record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MAPPED_CORE = re.compile(r"mapped-core=([0-9.]+) GiB")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_core(path: Path) -> float:
    match = MAPPED_CORE.search(path.read_text())
    if not match:
        raise ValueError(f"mapped-core line absent: {path}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--baseline-stage", type=Path, required=True)
    parser.add_argument("--candidate-stage", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--candidate-quality", type=Path, required=True)
    parser.add_argument("--baseline-stderr", type=Path, required=True)
    parser.add_argument("--candidate-stderr", type=Path, required=True)
    parser.add_argument("--candidate-checksums", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "plan": args.plan,
        "verification": args.verification,
        "baseline_stage": args.baseline_stage,
        "candidate_stage": args.candidate_stage,
        "baseline_quality": args.baseline_quality,
        "candidate_quality": args.candidate_quality,
        "baseline_stderr": args.baseline_stderr,
        "candidate_stderr": args.candidate_stderr,
        "candidate_checksums": args.candidate_checksums,
    }
    plan = load(args.plan)
    verification = load(args.verification)
    baseline_stage = load(args.baseline_stage)
    candidate_stage = load(args.candidate_stage)
    baseline_quality = load(args.baseline_quality)
    candidate_quality = load(args.candidate_quality)
    if verification["status"] != "PASS":
        raise ValueError("selective requant integrity verification did not pass")
    if verification["non_target_full_hash"] != "PASS":
        raise ValueError("release-strength non-target hash was not completed")

    base_total = baseline_stage["stages"]["total_ms"]["median_ms"]
    candidate_total = candidate_stage["stages"]["total_ms"]["median_ms"]
    base_routed = baseline_stage["stages"]["routed_prep_ms"]["median_ms"]
    candidate_routed = candidate_stage["stages"]["routed_prep_ms"]["median_ms"]
    base_runtime = baseline_quality["runtime"]
    candidate_runtime = candidate_quality["runtime"]
    base_terminal = baseline_quality["candidate"]["aggregate_terminal"]
    candidate_terminal = candidate_quality["candidate"]["aggregate_terminal"]

    result = {
        "schema": "kimi-k3-p32-selective-kda-q4k-v1",
        "status": "MEASURED",
        "verdict": "PRACTICAL_GO_BROAD_GATE__NOT_QUALITY_EQUIVALENCE",
        "provenance": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "transformation": {
            "target": "KDA-only q/k/v/g/output matrices in 69 recurrent layers",
            "source_type": "Q6_K",
            "target_type": plan["target_type"],
            "changed_tensors": verification["changed_tensor_count"],
            "untouched_tensors": verification["non_target_tensor_count"],
            "untouched_full_hash": verification["non_target_full_hash"],
            "source_file_bytes": verification["source_total_bytes"],
            "candidate_file_bytes": verification["candidate_total_bytes"],
            "saved_file_bytes": verification["saved_file_bytes"],
            "saved_file_gib": verification["saved_file_bytes"] / 2**30,
            "mapped_core_gib_before": mapped_core(args.baseline_stderr),
            "mapped_core_gib_after": mapped_core(args.candidate_stderr),
        },
        "steady_decode_stage_profile": {
            "rows": candidate_stage["decode_rows"],
            "median_total_ms_before": base_total,
            "median_total_ms_after": candidate_total,
            "transition_rate_before": 1000.0 / base_total,
            "transition_rate_after": 1000.0 / candidate_total,
            "throughput_ratio": base_total / candidate_total,
            "median_routed_preparation_ms_before": base_routed,
            "median_routed_preparation_ms_after": candidate_routed,
            "routed_preparation_reduction_fraction": 1.0 - candidate_routed / base_routed,
        },
        "frozen_12_prompt_gate": {
            "baseline_task_successes": baseline_quality["candidate"]["native_successes_retained"],
            "candidate_task_successes": candidate_quality["candidate"]["native_successes_retained"],
            "task_denominator": candidate_quality["candidate"]["native_success_denominator"],
            "baseline_token_exact": baseline_quality["candidate"]["token_exact"],
            "candidate_token_exact": candidate_quality["candidate"]["token_exact"],
            "baseline_terminal": base_terminal,
            "candidate_terminal": candidate_terminal,
            "baseline_elapsed_seconds": base_runtime["elapsed_seconds"],
            "candidate_elapsed_seconds": candidate_runtime["elapsed_seconds"],
            "end_to_end_throughput_ratio": (
                base_runtime["elapsed_seconds"] / candidate_runtime["elapsed_seconds"]
            ),
            "baseline_peak_ram_kib": base_runtime["peak_ram_kib"],
            "candidate_peak_ram_kib": candidate_runtime["peak_ram_kib"],
            "peak_ram_saved_gib": (
                base_runtime["peak_ram_kib"] - candidate_runtime["peak_ram_kib"]
            ) / 2**20,
            "baseline_energy_joules": base_runtime["gpu_energy_joules"],
            "candidate_energy_joules": candidate_runtime["gpu_energy_joules"],
        },
        "limits": [
            "The frozen gate has 12 deterministic thinking-disabled prompts; it is not a model-card benchmark.",
            "Terminal KL is not zero and histories have different aligned-row counts after divergence.",
            "The candidate is a combined KDA-Q4_K plus frozen slab-policy deployment test.",
            "No four-token-per-second claim follows from this isolated 1.09x systems gain.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "verdict": result["verdict"],
        "task_success": (
            f"{result['frozen_12_prompt_gate']['candidate_task_successes']}/"
            f"{result['frozen_12_prompt_gate']['task_denominator']}"
        ),
        "steady_decode_ratio": result["steady_decode_stage_profile"]["throughput_ratio"],
        "broad_wall_ratio": result["frozen_12_prompt_gate"]["end_to_end_throughput_ratio"],
        "saved_gib": result["transformation"]["saved_file_gib"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

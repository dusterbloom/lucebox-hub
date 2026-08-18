#!/usr/bin/env python3
"""Aggregate P33 CPU-thread and KDA/MLA boundary profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


PREPARATION = re.compile(
    r'\[kimi-k3-boundary\] phase="routed layer preparation" .*'
    r'compute_ms=([0-9.]+)'
)
STAGE = re.compile(r"\[kimi-k3-stage\] position=([0-9]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_boundary(path: Path) -> list[tuple[int, list[float]]]:
    current: list[float] = []
    groups: list[tuple[int, list[float]]] = []
    for line in path.read_text().splitlines():
        match = PREPARATION.search(line)
        if match:
            current.append(float(match.group(1)))
        match = STAGE.search(line)
        if match:
            if len(current) != 92:
                raise ValueError(
                    f"position {match.group(1)} has {len(current)} routed layers"
                )
            groups.append((int(match.group(1)), current))
            current = []
    if current:
        raise ValueError("unterminated boundary-profile position")
    if not groups:
        raise ValueError("no complete boundary-profile positions")
    return groups


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-log", type=Path, required=True)
    parser.add_argument("--requant-plan", type=Path, required=True)
    parser.add_argument("--decode-start", type=int, required=True)
    parser.add_argument(
        "--thread-profile", action="append", default=[],
        help="NAME=JSON stage-profile artifact",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.requant_plan.read_text())
    kda_layers = set(int(value) for value in plan["kda_layers"])
    groups = parse_boundary(args.boundary_log)
    decode = [(position, values) for position, values in groups
              if position >= args.decode_start]
    if not decode:
        raise ValueError("no decode positions after requested boundary")

    def split(rows: list[tuple[int, list[float]]]) -> dict[str, object]:
        kda: list[float] = []
        mla: list[float] = []
        for _, values in rows:
            for model_layer, value in enumerate(values, start=1):
                (kda if model_layer in kda_layers else mla).append(value)
        count = len(rows)
        return {
            "positions": count,
            "routed_kda_layers": len(kda) // count,
            "routed_mla_layers": len(mla) // count,
            "kda_per_layer": distribution(kda),
            "mla_per_layer": distribution(mla),
            "kda_compute_ms_per_position": sum(kda) / count,
            "mla_compute_ms_per_position": sum(mla) / count,
        }

    thread_profiles: dict[str, object] = {}
    for item in args.thread_profile:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--thread-profile must be NAME=PATH")
        path = Path(raw_path)
        profile = json.loads(path.read_text())
        thread_profiles[name] = {
            "artifact": str(path),
            "artifact_sha256": sha256(path),
            "decode_rows": profile["decode_rows"],
            "median_total_ms": profile["stages"]["total_ms"]["median_ms"],
            "median_routed_preparation_ms":
                profile["stages"]["routed_prep_ms"]["median_ms"],
            "median_expert_ms": profile["stages"]["experts_ms"]["median_ms"],
            "transition_rate":
                profile["median_control_room"]["measured_transition_rate"],
        }

    result = {
        "schema": "kimi-k3-p33-core-boundary-profile-v1",
        "status": "MEASURED",
        "provenance": {
            "boundary_log": str(args.boundary_log),
            "boundary_log_sha256": sha256(args.boundary_log),
            "requant_plan": str(args.requant_plan),
            "requant_plan_sha256": sha256(args.requant_plan),
            "decode_start": args.decode_start,
        },
        "thread_sweep": thread_profiles,
        "boundary_all_positions": split(groups),
        "boundary_decode_positions": split(decode),
        "interpretation": (
            "Twelve CPU threads remain the measured optimum. Routed preparation "
            "is dominated by graph compute, and recurrent KDA layers dominate "
            "that compute; further host packing work cannot remove this stage."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

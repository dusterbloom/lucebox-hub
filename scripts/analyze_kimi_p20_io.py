#!/usr/bin/env python3
"""Attribute K3 calibrated-provider traffic without conflating byte classes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


GIB = 1024**3


def load_single_row(path: Path) -> dict[str, int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return {key: int(value) for key, value in rows[0].items()}


def load_traffic(path: Path) -> tuple[list[dict[str, int]], dict[str, int]]:
    with path.open(newline="") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    sums: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if key != "model_layer":
                sums[key] += value
    return rows, dict(sums)


def parse_native_scheduler(stderr: str) -> dict[str, int | float | str]:
    lines = [line for line in stderr.splitlines() if line.startswith("[moe-nvme]")]
    if not lines:
        return {}
    line = lines[-1]
    result: dict[str, int | float | str] = {"raw": line}
    for key in ("payload", "physical"):
        match = re.search(rf"\b{key}=([0-9.]+) GiB\b", line)
        if match:
            result[f"{key}_bytes"] = round(float(match.group(1)) * GIB)
    rate = re.search(r"\bactive-io-rate=([0-9.]+) GiB/s\b", line)
    if rate:
        result["active_io_gib_s"] = float(rate.group(1))
    integer_fields = {
        "requests", "reads", "cache-hit", "hits", "misses", "evictions"
    }
    for key, raw in re.findall(r"([a-z-]+)=([^ ]+)", line):
        value = raw.rstrip("%,")
        if key in integer_fields:
            try:
                result[key.replace("-", "_")] = int(value)
            except ValueError:
                result[key.replace("-", "_")] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--process", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_bytes: Counter[str] = Counter()
    trace_requests: Counter[str] = Counter()
    exact_fallback_logical = 0
    range_counts: Counter[tuple[str, int, int]] = Counter()
    selected_ranges: list[tuple[str, int, int]] = []
    with args.trace.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            region = row["region"]
            logical = int(row["logical_length"])
            explicit = int(row["explicit_read_bytes"])
            trace_requests[region] += 1
            trace_bytes[f"{region}_logical"] += logical
            trace_bytes[f"{region}_explicit"] += explicit
            if region == "native-exact-expert":
                exact_fallback_logical += logical
            elif region in {"gate", "up", "down"}:
                key = (row["file_path"], int(row["file_offset"]), logical)
                range_counts[key] += 1
                selected_ranges.append(key)

    traffic_rows, traffic = load_traffic(args.traffic)
    process = load_single_row(args.process)
    scheduler = parse_native_scheduler(args.stderr.read_text(errors="replace"))
    p20_line = ""
    for line in args.stderr.read_text(errors="replace").splitlines():
        if line.startswith("[kimi-k3-p20]"):
            p20_line = line
    p20_counters = {
        key.replace("-", "_"): int(value)
        for key, value in re.findall(r"([a-z0-9-]+)=([0-9]+)", p20_line)
    }
    telemetry = (
        json.loads(args.telemetry.read_text()) if args.telemetry else None
    )

    selected_logical = traffic.get("selected_sidecar_bytes", 0)
    fallback_logical = traffic.get("exact_fallback_bytes", 0)
    policy_logical = selected_logical + fallback_logical
    sidecar_explicit = sum(
        trace_bytes[f"{region}_explicit"] for region in ("gate", "up", "down")
    )
    aux_explicit = trace_bytes["slab-mean_explicit"]
    native_physical = int(scheduler.get("physical_bytes", 0))
    process_physical = process["process_read_bytes"]
    residual_process = max(
        0, process_physical - sidecar_explicit - aux_explicit - native_physical
    )
    attributed_or_bounded = (
        sidecar_explicit + aux_explicit + native_physical + residual_process
    )

    reference_full_weight_h2d = p20_counters.get(
        "reference_full_weight_h2d", 0)
    sparse_authoritative_h2d = p20_counters.get(
        "sparse_authoritative_h2d", 0)
    provider_authoritative_h2d = (
        sparse_authoritative_h2d or reference_full_weight_h2d)
    native_payload = int(scheduler.get("payload_bytes", 0))

    duplicate_instances = sum(count - 1 for count in range_counts.values())
    duplicate_bytes = sum(
        (count - 1) * key[2] for key, count in range_counts.items()
    )

    result = {
        "schema": "k3-p20-io-audit-v1",
        "status": "MEASURED_WITH_INFERRED_PROCESS_RESIDUAL",
        "inputs": {key: str(value) for key, value in vars(args).items()
                   if value is not None and key != "output"},
        "logical": {
            "selected_sidecar_bytes": selected_logical,
            "exact_fallback_bytes": fallback_logical,
            "total_policy_bytes": policy_logical,
        },
        "provider_explicit": {
            "selected_sidecar_bytes": sidecar_explicit,
            "auxiliary_mean_bytes": aux_explicit,
            "all_explicit_bytes": process["explicit_provider_read_bytes"],
            "request_counts_by_region": dict(trace_requests),
        },
        "native_scheduler": scheduler,
        "process": process,
        "amplification": {
            "process_over_policy": (
                process_physical / policy_logical if policy_logical else None
            ),
            "provider_selected_explicit_over_selected_logical": (
                sidecar_explicit / selected_logical if selected_logical else None
            ),
            "inferred_mapped_core_or_untracked_process_bytes": residual_process,
            "attributed_or_bounded_fraction": (
                attributed_or_bounded / process_physical
                if process_physical else None
            ),
        },
        "h2d": {
            "reference_full_reconstructed_weight_bytes": reference_full_weight_h2d,
            "sparse_authoritative_weight_bytes": sparse_authoritative_h2d,
            "native_scheduler_payload_bytes_upper_bound": native_payload,
            "authoritative_weight_over_policy": (
                (provider_authoritative_h2d + native_payload) / policy_logical
                if policy_logical else None
            ),
            "note": "native payload is an upper bound on fallback H2D because it includes the native scheduler's serving activity",
            "p20_runtime_counters": p20_counters,
        },
        "duplicates": {
            "selected_range_instances": len(selected_ranges),
            "unique_selected_ranges": len(range_counts),
            "duplicate_instances": duplicate_instances,
            "duplicate_bytes": duplicate_bytes,
        },
        "telemetry": telemetry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "process_over_policy": result["amplification"]["process_over_policy"],
        "sidecar_explicit_over_selected": result["amplification"]["provider_selected_explicit_over_selected_logical"],
        "authoritative_weight_h2d_over_policy": result["h2d"]["authoritative_weight_over_policy"],
        "duplicate_bytes": duplicate_bytes,
    }, indent=2))


if __name__ == "__main__":
    main()

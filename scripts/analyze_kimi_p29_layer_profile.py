#!/usr/bin/env python3
"""Recover per-layer K3 CPU preparation costs from an existing boundary log."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from gguf import GGUFReader


BOUNDARY = re.compile(
    r'^\[kimi-k3-boundary\] phase="(?P<phase>[^"]+)" .*?'
    r'compute_ms=(?P<compute>[0-9.]+) .*?total_ms=(?P<total>[0-9.]+)$'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def read_layer_types(model: Path) -> list[str]:
    reader = GGUFReader(str(model), "r")
    field = reader.fields["kimi-k3.attention.head_count_kv"]
    values = [int(field.parts[index][0]) for index in field.data]
    return ["KDA" if value == 0 else "MLA" for value in values]


def read_layer_weight_bytes(model: Path) -> dict[int, dict[str, int]]:
    match = re.match(r"^(.*-)[0-9]{5}(-of-[0-9]{5}\.gguf)$", model.name)
    paths = [model]
    if match:
        paths = [Path(path) for path in sorted(glob.glob(
            str(model.with_name(match.group(1) + "*" + match.group(2)))
        ))]
    offloaded_suffixes = {
        "ffn_routed_down.weight",
        "ffn_routed_norm.weight",
        "ffn_routed_up.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    }
    totals: dict[int, dict[str, int]] = {}
    name_pattern = re.compile(r"^blk\.([0-9]+)\.(.+)$")
    for path in paths:
        reader = GGUFReader(str(path), "r")
        for tensor in reader.tensors:
            tensor_match = name_pattern.match(tensor.name)
            if not tensor_match:
                continue
            layer = int(tensor_match.group(1))
            suffix = tensor_match.group(2)
            values = totals.setdefault(layer, {
                "cpu_preparation": 0,
                "already_offloaded": 0,
                "routed_expert_bank": 0,
            })
            size = int(tensor.data.nbytes)
            if "_exps." in suffix:
                values["routed_expert_bank"] += size
            elif suffix in offloaded_suffixes:
                values["already_offloaded"] += size
            else:
                values["cpu_preparation"] += size
    return totals


def read_rows(log: Path) -> list[list[dict[str, float]]]:
    rows: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] | None = None
    for line in log.read_text(errors="replace").splitlines():
        match = BOUNDARY.match(line)
        if not match:
            continue
        phase = match.group("phase")
        if phase == "embedding":
            if current is not None:
                rows.append(current)
            current = []
        elif phase == "routed layer preparation" and current is not None:
            current.append({
                "compute_ms": float(match.group("compute")),
                "total_ms": float(match.group("total")),
            })
    if current is not None:
        rows.append(current)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-rows", type=int, default=1)
    args = parser.parse_args()

    layer_types = read_layer_types(args.model)
    layer_weight_bytes = read_layer_weight_bytes(args.model)
    rows = read_rows(args.log)
    expected = len(layer_types) - 1
    if not rows or any(len(row) != expected for row in rows):
        raise SystemExit(
            f"expected {expected} routed preparations per row, got "
            f"{[len(row) for row in rows]}"
        )
    used_rows = rows[args.skip_rows:]
    if not used_rows:
        raise SystemExit("skip-rows removed every row")

    layers = []
    for index in range(expected):
        model_layer = index + 1
        values = [row[index]["compute_ms"] for row in used_rows]
        weights = layer_weight_bytes.get(model_layer, {})
        cpu_weight_bytes = int(weights.get("cpu_preparation", 0))
        layer_result = {
            "model_layer": model_layer,
            "attention": layer_types[model_layer],
            "attnres_phase": model_layer % 12,
            "cpu_preparation_weight_bytes": cpu_weight_bytes,
            "already_offloaded_weight_bytes": int(
                weights.get("already_offloaded", 0)
            ),
            **summary(values),
        }
        layer_result["median_ms_per_cpu_weight_gib"] = (
            layer_result["median_ms"] /
            (cpu_weight_bytes / (1024 ** 3))
            if cpu_weight_bytes else math.nan
        )
        layers.append(layer_result)

    family_values: dict[str, list[float]] = {"KDA": [], "MLA": []}
    family_row_totals: dict[str, list[float]] = {"KDA": [], "MLA": []}
    for family in family_values:
        family_layers = [
            index for index in range(expected)
            if layer_types[index + 1] == family
        ]
        family_values[family] = [
            row[index]["compute_ms"]
            for row in used_rows for index in family_layers
        ]
        family_row_totals[family] = [
            sum(row[index]["compute_ms"] for index in family_layers)
            for row in used_rows
        ]

    phase_values: dict[str, list[float]] = {}
    for index in range(expected):
        phase = str((index + 1) % 12)
        phase_values.setdefault(phase, []).extend(
            row[index]["compute_ms"] for row in used_rows
        )

    row_totals = [
        sum(item["compute_ms"] for item in row) for row in used_rows
    ]
    ranked_layers = sorted(
        layers,
        key=lambda item: item["median_ms_per_cpu_weight_gib"],
        reverse=True,
    )
    placement_ceilings = {}
    for budget_gib in (4, 6, 8):
        used_bytes = 0
        saved_ms = 0.0
        selected_layers = []
        for layer in ranked_layers:
            layer_bytes = int(layer["cpu_preparation_weight_bytes"])
            if used_bytes + layer_bytes > budget_gib * (1024 ** 3):
                continue
            used_bytes += layer_bytes
            saved_ms += float(layer["median_ms"])
            selected_layers.append(int(layer["model_layer"]))
        placement_ceilings[str(budget_gib)] = {
            "capacity_gib": budget_gib,
            "used_gib": used_bytes / (1024 ** 3),
            "selected_layers": selected_layers,
            "selected_layer_count": len(selected_layers),
            "perfect_cpu_compute_removed_ms": saved_ms,
            "warning": (
                "Optimistic CPU-time removal before CUDA compute, state "
                "transfer, synchronization, and arithmetic-quality costs."
            ),
        }
    result = {
        "schema": "k3-p29-layer-profile-v1",
        "source_log": str(args.log),
        "source_log_sha256": sha256(args.log),
        "model": str(args.model),
        "model_metadata_shard_sha256": sha256(args.model),
        "rows_total": len(rows),
        "rows_skipped": args.skip_rows,
        "rows_analyzed": len(used_rows),
        "routed_layers": expected,
        "row_total_compute": summary(row_totals),
        "attention_families": {
            family: {
                "layer_count": sum(
                    layer_types[index + 1] == family
                    for index in range(expected)
                ),
                "per_boundary": summary(family_values[family]),
                "per_row_total": summary(family_row_totals[family]),
            }
            for family in ("KDA", "MLA")
        },
        "attnres_phase": {
            phase: summary(values) for phase, values in phase_values.items()
        },
        "top_layers_by_median": sorted(
            layers, key=lambda item: item["median_ms"], reverse=True
        )[:20],
        "top_layers_by_median_ms_per_gib": ranked_layers[:20],
        "greedy_placement_ceilings": placement_ceilings,
        "layers": layers,
        "interpretation_boundary": (
            "Each preparation graph contains AttnRes mixing, KDA or MLA, "
            "normalization and the CPU router. These timings separate layers "
            "and attention families, not individual operators within a graph."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

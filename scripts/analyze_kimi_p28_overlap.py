#!/usr/bin/env python3
"""Compute an oracle delivery-overlap ceiling from an archived K3 I/O trace.

This is deliberately a trace analyzer, not a runtime prefetcher.  It preserves
the recorded route/slab requests and attributes the measured P27 aggregate
timers in proportion to the recorded bytes.  Exact-fallback execution remains
an opaque serial remainder because the P27 trace does not time it per route.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


GIB = float(1 << 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--p27-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    args = parse_args()
    p27 = json.loads(args.p27_results.read_text())
    measured = p27["thirty_two_row"]["p27"]
    transitions = int(p27["thirty_two_row"]["decoded_transitions"])
    rows = transitions + 1

    # The physical direct-read charge appears only once on each coalesced
    # gate/up/down record.  Logical length appears on every selected tensor.
    route_stats: dict[tuple[int, int, int], dict[str, int]] = defaultdict(
        lambda: {"logical": 0, "physical": 0, "fallback": 0}
    )
    layers: set[tuple[int, int]] = set()
    with args.trace.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            position = int(row["base_pos"])
            layer = int(row["model_layer"])
            expert = int(row["expert_id"])
            key = (position, layer, expert)
            layers.add((position, layer))
            if row["region"] == "native-exact-expert":
                route_stats[key]["fallback"] = max(
                    route_stats[key]["fallback"], int(row["logical_length"])
                )
            elif row["region"] in {"gate", "up", "down"}:
                route_stats[key]["logical"] += int(row["logical_length"])
                route_stats[key]["physical"] += int(row["explicit_read_bytes"])

    selected = [value for value in route_stats.values() if value["logical"]]
    submitted_aligned_bytes = sum(value["physical"] for value in selected)
    compact_trace_bytes = sum(value["logical"] for value in selected)
    trace_fallback_bytes = sum(value["fallback"] for value in route_stats.values())
    calibrated_routes = len(selected)

    traffic_rows = list(csv.DictReader(args.traffic.open(), delimiter="\t"))
    logical_bytes = sum(int(row["selected_sidecar_bytes"]) for row in traffic_rows)
    fallback_bytes = sum(int(row["exact_fallback_bytes"]) for row in traffic_rows)
    if fallback_bytes != trace_fallback_bytes:
        raise ValueError("fallback-byte mismatch between traffic and I/O traces")
    selected_scale = logical_bytes / compact_trace_bytes

    footer = args.stderr.read_text()
    footer_match = re.search(
        r"sparse-authoritative-h2d=(\d+).*?direct-physical-bytes=(\d+)", footer
    )
    if not footer_match:
        raise ValueError("P27 byte-accounting footer missing")
    h2d_bytes = int(footer_match.group(1))
    actual_physical_bytes = int(footer_match.group(2))
    if h2d_bytes != logical_bytes:
        raise ValueError("authoritative H2D does not match selected traffic")

    read_s = float(measured["direct_io_seconds"])
    upload_scatter_s = float(measured["compact_scatter_seconds"])
    graph_s = float(measured["expert_graph_seconds"])
    readback_s = float(measured["expert_readback_seconds"])
    decode_s = float(measured["decode_seconds"])
    selected_gpu_s = upload_scatter_s + graph_s + readback_s
    selected_serial_s = read_s + selected_gpu_s
    opaque_serial_s = decode_s - selected_serial_s

    # Per-layer byte distributions establish buffer requirements.  They are
    # also a control that the trace covers all 92 routed layers for every row.
    layer_selected: dict[tuple[int, int], int] = defaultdict(int)
    layer_submitted: dict[tuple[int, int], int] = defaultdict(int)
    layer_fallback: dict[tuple[int, int], int] = defaultdict(int)
    layer_routes: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for (position, layer, _expert), value in route_stats.items():
        layer_selected[(position, layer)] += round(value["logical"] * selected_scale)
        layer_submitted[(position, layer)] += value["physical"]
        layer_fallback[(position, layer)] += value["fallback"]
        if value["logical"]:
            layer_routes[(position, layer)].append(
                (_expert, value["physical"], value["logical"])
            )
    layer_provider = [
        layer_selected[key] + layer_fallback[key] for key in sorted(layers)
    ]
    layer_selected_values = [layer_selected[key] for key in sorted(layers)]
    layer_read_ms = [
        1000.0 * read_s * layer_submitted[key] / submitted_aligned_bytes
        for key in sorted(layers)
    ]

    # Ceiling 1: once a layer's router is known, selected route N+1 reads can
    # overlap route N's upload/scatter/graph/readback. Attribute measured time
    # by submitted/read bytes and replay a two-resource flow shop in frozen
    # expert-ID order. A layer barrier is retained, so startup/drain overhead is
    # paid 2,944 times instead of being wished away by max(total R, total GPU).
    within_selected_s = 0.0
    per_route_graph_readback_s = (graph_s + readback_s) / calibrated_routes
    for key in sorted(layer_routes):
        read_clock = 0.0
        gpu_clock = 0.0
        for _expert, route_physical, route_logical in sorted(layer_routes[key]):
            route_read_s = read_s * route_physical / submitted_aligned_bytes
            route_gpu_s = (
                upload_scatter_s * route_logical / compact_trace_bytes
                + per_route_graph_readback_s
            )
            read_clock += route_read_s
            gpu_clock = max(gpu_clock, read_clock) + route_gpu_s
        within_selected_s += gpu_clock
    within_total_s = opaque_serial_s + within_selected_s
    within_hidden_read_s = selected_serial_s - within_selected_s

    # Ceiling 2: with perfect one-layer route knowledge, all but the initial
    # layer's selected reads can be issued under preceding non-read work.  This
    # assumes storage and useful work do not contend; it does not assume H2D or
    # scatter can disappear.  Two-layer lookahead cannot improve this steady
    # state bound because one layer already covers the average read exposure.
    layer_count = len(layers)
    startup_read_s = read_s / layer_count if layer_count else 0.0
    lookahead_total_s = decode_s - read_s + startup_read_s
    lookahead_hidden_read_s = read_s - startup_read_s

    # A deliberately labelled ideal bound also overlaps upload/scatter.  It is
    # not an implementation forecast: P27 combines DMA and a CUDA scatter in
    # one timer, and the scatter shares GPU/memory resources with useful work.
    startup_delivery_s = (read_s + upload_scatter_s) / layer_count if layer_count else 0.0
    ideal_delivery_total_s = decode_s - read_s - upload_scatter_s + startup_delivery_s

    def rate(seconds: float) -> float:
        return transitions / seconds

    baseline_rate = rate(decode_s)

    def free_stage(seconds: float) -> dict[str, float]:
        reduced = decode_s - seconds
        return {
            "lower_bound_decode_seconds": reduced,
            "transitions_per_second": rate(reduced),
            "maximum_speedup": decode_s / reduced,
        }

    output = {
        "schema": "k3-p28-oracle-overlap-ceiling-v1",
        "classification": "PROJECTED_FROM_MEASURED_P27_TRACE",
        "inputs": {
            "trace": str(args.trace),
            "traffic": str(args.traffic),
            "stderr": str(args.stderr),
            "p27_results": str(args.p27_results),
            "rows": rows,
            "decoded_transitions": transitions,
            "routed_layer_rows": layer_count,
            "calibrated_routes": calibrated_routes,
        },
        "bytes": {
            "selected_logical": logical_bytes,
            "compact_trace_bytes": compact_trace_bytes,
            "submitted_aligned_bytes": submitted_aligned_bytes,
            "actual_direct_physical_bytes": actual_physical_bytes,
            "authoritative_h2d_bytes": h2d_bytes,
            "fallback_logical_opaque": fallback_bytes,
            "selected_logical_gib_per_row": logical_bytes / rows / GIB,
            "fallback_logical_gib_per_row": fallback_bytes / rows / GIB,
            "mean_selected_layer_mib": sum(layer_selected_values) / layer_count / (1 << 20),
            "p95_selected_layer_mib": percentile(layer_selected_values, 0.95) / (1 << 20),
            "max_selected_layer_mib": max(layer_selected_values) / (1 << 20),
            "mean_provider_layer_mib": sum(layer_provider) / layer_count / (1 << 20),
            "max_provider_layer_mib": max(layer_provider) / (1 << 20),
            "mean_attributed_read_ms_per_layer": sum(layer_read_ms) / layer_count,
            "p95_attributed_read_ms_per_layer": percentile(layer_read_ms, 0.95),
            "max_attributed_read_ms_per_layer": max(layer_read_ms),
            "mean_nonread_ms_per_layer": 1000.0 * (decode_s - read_s) / layer_count,
        },
        "measured_baseline": {
            "decode_seconds": decode_s,
            "transitions_per_second": baseline_rate,
            "selected_read_seconds": read_s,
            "selected_upload_scatter_seconds": upload_scatter_s,
            "selected_graph_seconds": graph_s,
            "selected_readback_seconds": readback_s,
            "opaque_serial_remainder_seconds": opaque_serial_s,
        },
        "control_room_free_stage_ceilings": {
            "selected_read_free": free_stage(read_s),
            "upload_scatter_free": free_stage(upload_scatter_s),
            "selected_graph_free": free_stage(graph_s),
            "selected_readback_free": free_stage(readback_s),
            "read_plus_upload_scatter_free": free_stage(read_s + upload_scatter_s),
        },
        "within_layer_route_pipeline": {
            "lower_bound_decode_seconds": within_total_s,
            "transitions_per_second": rate(within_total_s),
            "throughput_gain_fraction": rate(within_total_s) / baseline_rate - 1.0,
            "hidden_selected_read_seconds": within_hidden_read_s,
            "hidden_h2d_seconds": 0.0,
            "extra_pinned_mib": max(layer_selected_values) / (1 << 20),
            "extra_speculative_bytes": 0,
            "gate": "GO" if rate(within_total_s) / baseline_rate - 1.0 >= 0.15 else "NO_GO_BORDERLINE",
            "scheduling": "two-resource flow shop, frozen expert-ID order, barrier after every routed layer",
        },
        "one_layer_oracle_read_lookahead": {
            "lower_bound_decode_seconds": lookahead_total_s,
            "transitions_per_second": rate(lookahead_total_s),
            "throughput_gain_fraction": rate(lookahead_total_s) / baseline_rate - 1.0,
            "hidden_selected_read_seconds": lookahead_hidden_read_s,
            "hidden_h2d_seconds": 0.0,
            "extra_pinned_mib": max(layer_selected_values) / (1 << 20),
            "extra_speculative_bytes": 0,
            "gate": "STRONG_GO" if rate(lookahead_total_s) / baseline_rate - 1.0 >= 0.25 else "GO",
        },
        "two_layer_oracle_read_lookahead": {
            "lower_bound_decode_seconds": lookahead_total_s,
            "transitions_per_second": rate(lookahead_total_s),
            "incremental_gain_over_one_layer": 0.0,
            "extra_pinned_mib": 2.0 * max(layer_selected_values) / (1 << 20),
            "extra_speculative_bytes": 0,
            "verdict": "NO_INCREMENTAL_VALUE_AT_THIS_CEILING",
        },
        "ideal_read_plus_upload_scatter_bound": {
            "lower_bound_decode_seconds": ideal_delivery_total_s,
            "transitions_per_second": rate(ideal_delivery_total_s),
            "throughput_gain_fraction": rate(ideal_delivery_total_s) / baseline_rate - 1.0,
            "warning": "Upper bound only: the measured timer combines asynchronous H2D with a CUDA scatter.",
        },
        "limitations": [
            "No GPU or integrated replay was run; values are oracle ceilings, not measured speedups.",
            "Exact-fallback storage/compute is retained inside the opaque serial remainder.",
            "The trace contains no per-request latency, so aggregate P27 timers are byte-attributed.",
            "One-layer oracle knowledge is not deployable until a predictor earns it.",
            "Completion may be asynchronous, but final expert-ID accumulation must remain deterministic.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

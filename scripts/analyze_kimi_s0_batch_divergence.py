#!/usr/bin/env python3
"""Compare stacked sequential K3 trace rows with one causal verify batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_kimi_h17_divergence import array_metrics, load_trace


BASE_STAGES = (
    "layer_input",
    "pre_moe_hidden",
    "router_logits",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-position", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.base_position < 0 or args.width <= 1:
        raise ValueError("invalid S0 base position or width")

    header, records = load_trace(args.trace)
    batched = {
        record.layer: record
        for record in records
        if record.base_position == args.base_position
        and record.token_count == args.width
    }
    sequential: dict[int, list[object]] = {}
    for record in records:
        token_offset = record.base_position - args.base_position
        if record.token_count != 1 or not 0 <= token_offset < args.width:
            continue
        rows = sequential.setdefault(record.layer, [None] * args.width)
        if rows[token_offset] is not None:
            raise ValueError(
                f"duplicate sequential row layer={record.layer} "
                f"position={record.base_position}"
            )
        rows[token_offset] = record

    layers = sorted(set(batched) & set(sequential))
    if not layers:
        raise ValueError("trace contains no paired sequential/batched layers")
    rows: list[dict[str, object]] = []
    first_numerical: dict[str, object] | None = None
    first_route_order: dict[str, object] | None = None
    first_route_membership: dict[str, object] | None = None
    for layer in layers:
        seq_records = sequential[layer]
        if any(record is None for record in seq_records):
            raise ValueError(f"incomplete sequential rows for layer {layer}")
        batch = batched[layer]
        stages = list(BASE_STAGES)
        if header["version"] >= 2:
            stages.append("pre_expert_latent")
        stages.extend(("routed_latent", "moe_output", "post_moe_hidden"))
        for stage in stages:
            reference = np.concatenate(
                [getattr(record, stage) for record in seq_records], axis=0
            )
            candidate = getattr(batch, stage)
            metrics = array_metrics(reference, candidate)
            row = {"model_layer": layer, "stage": stage, **metrics}
            rows.append(row)
            if first_numerical is None and not metrics["bit_identical"]:
                difference = np.not_equal(reference, candidate)
                token_indices = np.nonzero(np.any(difference, axis=1))[0]
                row["first_mismatch_token"] = (
                    int(token_indices[0]) if token_indices.size else -1
                )
                first_numerical = row.copy()

        reference_ids = np.concatenate(
            [record.selected_ids for record in seq_records], axis=0
        )
        for token in range(args.width):
            left = reference_ids[token]
            right = batch.selected_ids[token]
            if np.array_equal(left, right):
                continue
            item = {
                "model_layer": layer,
                "token": token,
                "absolute_position": args.base_position + token,
                "sequential_ids": [int(value) for value in left],
                "batch_ids": [int(value) for value in right],
                "ordered_agreement": int(np.count_nonzero(left == right)),
                "set_overlap": len(set(map(int, left)) & set(map(int, right))),
            }
            if first_route_order is None:
                first_route_order = item
            if first_route_membership is None and item["set_overlap"] < len(left):
                first_route_membership = item

    report = {
        "schema": "k3-s0-sequential-vs-batch-divergence-v1",
        "trace": str(args.trace),
        "header": header,
        "base_position": args.base_position,
        "width": args.width,
        "paired_layers": layers,
        "first_numerical_divergence": first_numerical,
        "first_router_order_divergence": first_route_order,
        "first_router_membership_divergence": first_route_membership,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "first_numerical_divergence": first_numerical,
        "first_router_order_divergence": first_route_order,
        "first_router_membership_divergence": first_route_membership,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

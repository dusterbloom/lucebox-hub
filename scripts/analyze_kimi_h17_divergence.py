#!/usr/bin/env python3
"""Locate the first native-versus-all-192 Kimi trajectory divergence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FILE_HEADER = struct.Struct("<8s7I")
RECORD_HEADER = struct.Struct("<iiiI")
MAGIC_V1 = b"K3DVT001"
MAGIC_V2 = b"K3DVT002"
ATTN_RES_BOUNDARY = 1 << 0


@dataclass
class TraceRecord:
    layer: int
    base_position: int
    token_count: int
    flags: int
    layer_input: np.ndarray
    pre_moe_hidden: np.ndarray
    router_logits: np.ndarray
    selected_ids: np.ndarray
    pre_expert_latent: np.ndarray | None
    routed_latent: np.ndarray
    moe_output: np.ndarray
    post_moe_hidden: np.ndarray


def take_array(
    raw: bytes,
    offset: int,
    dtype: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    count = math.prod(shape)
    item_size = np.dtype(dtype).itemsize
    end = offset + count * item_size
    if end > len(raw):
        raise ValueError("truncated H17 divergence trace")
    values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
    return values.reshape(shape), end


def load_trace(path: Path) -> tuple[dict[str, int], list[TraceRecord]]:
    raw = path.read_bytes()
    if len(raw) < FILE_HEADER.size:
        raise ValueError(f"{path}: truncated file header")
    (
        magic,
        version,
        hidden,
        latent,
        experts,
        top_k,
        block_size,
        reserved,
    ) = FILE_HEADER.unpack_from(raw)
    if magic not in (MAGIC_V1, MAGIC_V2) or reserved != 0:
        raise ValueError(f"{path}: unsupported trace header")
    expected_version = 1 if magic == MAGIC_V1 else 2
    if version != expected_version:
        raise ValueError(f"{path}: inconsistent trace magic/version")
    if min(hidden, latent, experts, top_k, block_size) <= 0:
        raise ValueError(f"{path}: invalid trace dimensions")

    offset = FILE_HEADER.size
    records: list[TraceRecord] = []
    while offset < len(raw):
        if offset + RECORD_HEADER.size > len(raw):
            raise ValueError(f"{path}: truncated record header")
        layer, base_position, token_count, flags = RECORD_HEADER.unpack_from(
            raw, offset
        )
        offset += RECORD_HEADER.size
        if layer < 1 or token_count <= 0 or flags & ~ATTN_RES_BOUNDARY:
            raise ValueError(f"{path}: invalid record metadata")
        layer_input, offset = take_array(
            raw, offset, "<f4", (token_count, hidden)
        )
        pre_moe_hidden, offset = take_array(
            raw, offset, "<f4", (token_count, hidden)
        )
        router_logits, offset = take_array(
            raw, offset, "<f4", (token_count, experts)
        )
        selected_ids, offset = take_array(
            raw, offset, "<i4", (token_count, top_k)
        )
        pre_expert_latent = None
        if version >= 2:
            pre_expert_latent, offset = take_array(
                raw, offset, "<f4", (token_count, latent)
            )
        routed_latent, offset = take_array(
            raw, offset, "<f4", (token_count, latent)
        )
        moe_output, offset = take_array(
            raw, offset, "<f4", (token_count, hidden)
        )
        post_moe_hidden, offset = take_array(
            raw, offset, "<f4", (token_count, hidden)
        )
        records.append(
            TraceRecord(
                layer=layer,
                base_position=base_position,
                token_count=token_count,
                flags=flags,
                layer_input=layer_input,
                pre_moe_hidden=pre_moe_hidden,
                router_logits=router_logits,
                selected_ids=selected_ids,
                pre_expert_latent=pre_expert_latent,
                routed_latent=routed_latent,
                moe_output=moe_output,
                post_moe_hidden=post_moe_hidden,
            )
        )
    return {
        "version": version,
        "hidden_dimension": hidden,
        "latent_dimension": latent,
        "expert_count": experts,
        "top_k": top_k,
        "attn_res_block_size": block_size,
    }, records


def array_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError("trace array shapes differ")
    left = reference.astype(np.float64, copy=False).reshape(-1)
    right = candidate.astype(np.float64, copy=False).reshape(-1)
    difference = right - left
    left_norm2 = float(np.dot(left, left))
    right_norm2 = float(np.dot(right, right))
    error_norm2 = float(np.dot(difference, difference))
    denominator = math.sqrt(left_norm2)
    cosine_denominator = math.sqrt(left_norm2 * right_norm2)
    return {
        "bit_identical": bool(reference.tobytes() == candidate.tobytes()),
        "rel_l2": math.sqrt(error_norm2) / max(denominator, 1.0e-300),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "cosine": (
            float(np.dot(left, right)) / max(cosine_denominator, 1.0e-300)
        ),
    }


def record_key(record: TraceRecord) -> tuple[int, int, int]:
    return record.base_position, record.token_count, record.layer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--slab192", type=Path, required=True)
    parser.add_argument("--terminal-comparison", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    exact_header, exact_records = load_trace(args.exact)
    slab_header, slab_records = load_trace(args.slab192)
    if exact_header != slab_header:
        raise ValueError("native and slab192 trace headers differ")
    if len(exact_records) != len(slab_records):
        raise ValueError("native and slab192 record counts differ")

    stage_names = [
        "layer_input",
        "pre_moe_hidden",
        "router_logits",
    ]
    if exact_header["version"] >= 2:
        stage_names.append("pre_expert_latent")
    stage_names.extend(("routed_latent", "moe_output", "post_moe_hidden"))
    rows: list[dict[str, object]] = []
    first_numerical: dict[str, object] | None = None
    first_router_order: dict[str, object] | None = None
    first_router_set: dict[str, object] | None = None
    exact_next_layer_identity = True
    slab_next_layer_identity = True

    previous_exact: TraceRecord | None = None
    previous_slab: TraceRecord | None = None
    for record_index, (exact, slab) in enumerate(
        zip(exact_records, slab_records, strict=True)
    ):
        if record_key(exact) != record_key(slab) or exact.flags != slab.flags:
            raise ValueError("native and slab192 record metadata differ")
        if (
            previous_exact is not None
            and previous_exact.base_position == exact.base_position
            and previous_exact.token_count == exact.token_count
            and previous_exact.layer + 1 == exact.layer
        ):
            exact_next_layer_identity &= (
                previous_exact.post_moe_hidden.tobytes()
                == exact.layer_input.tobytes()
            )
            slab_next_layer_identity &= (
                previous_slab is not None
                and previous_slab.post_moe_hidden.tobytes()
                == slab.layer_input.tobytes()
            )

        for stage in stage_names:
            metrics = array_metrics(getattr(exact, stage), getattr(slab, stage))
            row = {
                "record_index": record_index,
                "base_position": exact.base_position,
                "token_count": exact.token_count,
                "model_layer": exact.layer,
                "attn_res_boundary": bool(exact.flags & ATTN_RES_BOUNDARY),
                "stage": stage,
                **metrics,
            }
            rows.append(row)
            if first_numerical is None and not metrics["bit_identical"]:
                first_numerical = row.copy()

        for token in range(exact.token_count):
            exact_ids = exact.selected_ids[token]
            slab_ids = slab.selected_ids[token]
            ordered_equal = bool(np.array_equal(exact_ids, slab_ids))
            set_equal = set(map(int, exact_ids)) == set(map(int, slab_ids))
            if ordered_equal:
                continue
            divergence = {
                "record_index": record_index,
                "base_position": exact.base_position,
                "token_offset": token,
                "absolute_token_position": exact.base_position + token,
                "model_layer": exact.layer,
                "exact_top16_ids": [int(value) for value in exact_ids],
                "slab192_top16_ids": [int(value) for value in slab_ids],
                "ordered_id_agreement": int(np.count_nonzero(exact_ids == slab_ids)),
                "set_overlap": len(set(map(int, exact_ids)) & set(map(int, slab_ids))),
                "pre_router_hidden_error": array_metrics(
                    exact.pre_moe_hidden[token], slab.pre_moe_hidden[token]
                ),
                "router_logit_error": array_metrics(
                    exact.router_logits[token], slab.router_logits[token]
                ),
            }
            if first_router_order is None:
                first_router_order = divergence
            if first_router_set is None and not set_equal:
                first_router_set = divergence

        previous_exact = exact
        previous_slab = slab

    growth: dict[str, object] | None = None
    growth_router = first_router_set or first_router_order
    if growth_router is not None:
        target_base = int(growth_router["base_position"])
        target_count = exact_records[int(growth_router["record_index"])].token_count
        token = int(growth_router["token_offset"])
        layer_rows = []
        for exact, slab in zip(exact_records, slab_records, strict=True):
            if exact.base_position != target_base or exact.token_count != target_count:
                continue
            layer_rows.append(
                {
                    "model_layer": exact.layer,
                    "pre_router_rel_l2": array_metrics(
                        exact.pre_moe_hidden[token], slab.pre_moe_hidden[token]
                    )["rel_l2"],
                    "routed_latent_rel_l2": array_metrics(
                        exact.routed_latent[token], slab.routed_latent[token]
                    )["rel_l2"],
                    "post_moe_rel_l2": array_metrics(
                        exact.post_moe_hidden[token], slab.post_moe_hidden[token]
                    )["rel_l2"],
                    "router_ids_equal": bool(
                        np.array_equal(
                            exact.selected_ids[token], slab.selected_ids[token]
                        )
                    ),
                }
            )
        divergence_layer = int(growth_router["model_layer"])
        current = next(
            item for item in layer_rows if item["model_layer"] == divergence_layer
        )
        prior = next(
            (
                item
                for item in reversed(layer_rows)
                if item["model_layer"] < divergence_layer
            ),
            None,
        )
        growth = {
            "trajectory": layer_rows,
            "first_router_layer_pre_to_post_error_ratio": (
                float(current["post_moe_rel_l2"])
                / max(float(current["pre_router_rel_l2"]), 1.0e-300)
            ),
            "pre_router_error_ratio_vs_previous_routed_layer": (
                None
                if prior is None
                else float(current["pre_router_rel_l2"])
                / max(float(prior["pre_router_rel_l2"]), 1.0e-300)
            ),
        }

    terminal = None
    if args.terminal_comparison:
        terminal = json.loads(args.terminal_comparison.read_text())

    result = {
        "schema": "kimi-k3-h17-divergence-localization-v1",
        "claim_status": "MEASURED",
        "exact_trace": str(args.exact),
        "slab192_trace": str(args.slab192),
        "header": exact_header,
        "record_count": len(exact_records),
        "first_numerical_divergence": first_numerical,
        "first_router_order_divergence": first_router_order,
        "first_router_top16_membership_divergence": first_router_set,
        "error_growth_at_first_router_divergence": growth,
        "next_layer_input_identity_within_each_run": {
            "exact": exact_next_layer_identity,
            "slab192": slab_next_layer_identity,
        },
        "terminal_comparison": terminal,
        "secondary_teacher_note": (
            "The slab192 trace is retained only to separate recomposition error "
            "from future omission error. It is not an approximation-quality teacher."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

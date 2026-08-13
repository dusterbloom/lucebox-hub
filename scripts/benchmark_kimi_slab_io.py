#!/usr/bin/env python3
"""Direct-I/O benchmark for matched whole-expert and slab-prefix traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import mmap
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

from train_kimi_panel_directional import load_data


HEADER = struct.Struct("<8s8I5Q")
MAGIC = b"K3SLB001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--slab-budget", type=int, default=96)
    return parser.parse_args()


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def inverse_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    values = np.broadcast_to(np.arange(order.shape[1]), order.shape)
    np.put_along_axis(ranks, order, values, axis=1)
    return ranks


def main() -> int:
    args = parse_args()
    with args.sidecar.open("rb", buffering=0) as source:
        raw = source.read(HEADER.size)
    (
        magic, version, layer, experts, dimension, width, slab_size,
        slab_count, alignment, index_offset, index_bytes, payload_offset,
        slab_bytes, record_bytes,
    ) = HEADER.unpack(raw)
    if magic != MAGIC or version != 1 or args.sidecar.stat().st_size < payload_offset:
        raise ValueError("invalid slab sidecar")
    if args.slab_budget <= 0 or args.slab_budget > 16 * slab_count:
        raise ValueError("slab budget is out of range")
    data = load_data(args.capture, args.teacher)
    tokens = data.validation_indices[: args.tokens]
    if tokens.size != args.tokens:
        raise ValueError("not enough validation tokens")
    with np.load(args.fit_state, allow_pickle=False) as state:
        importance = state["slab_expected_residual_norm"]
        full_norm = state["native_expected_norm"]
    ids = data.expert_ids[tokens]
    weights = data.router_weights[tokens]

    slab_scores = weights[:, :, None] * importance[ids]
    slab_rank = inverse_order(
        np.argsort(-slab_scores.reshape(tokens.size, -1), axis=1, kind="stable")
    ).reshape(tokens.size, data.top_k, slab_count)
    slab_counts = (slab_rank < args.slab_budget).sum(axis=2)
    whole_order = np.argsort(
        -(weights * full_norm[ids]), axis=1, kind="stable"
    )[:, : args.slab_budget // slab_count]

    adaptive_trace: list[list[tuple[int, int]]] = []
    whole_trace: list[list[tuple[int, int]]] = []
    for token in range(tokens.size):
        adaptive_reads: list[tuple[int, int]] = []
        for rank in range(data.top_k):
            count = int(slab_counts[token, rank])
            if not count:
                continue
            expert = int(ids[token, rank])
            adaptive_reads.append(
                (
                    payload_offset + expert * record_bytes,
                    align(count * slab_bytes, alignment),
                )
            )
        adaptive_trace.append(adaptive_reads)
        whole_trace.append(
            [
                (
                    payload_offset + int(ids[token, rank]) * record_bytes,
                    record_bytes,
                )
                for rank in whole_order[token]
            ]
        )

    flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
    descriptor = os.open(args.sidecar, flags)
    local = threading.local()

    def read_one(request: tuple[int, int]) -> int:
        offset, length = request
        buffer = getattr(local, "buffer", None)
        if buffer is None:
            buffer = mmap.mmap(-1, record_bytes)
            local.buffer = buffer
        view = memoryview(buffer)[:length]
        observed = os.preadv(descriptor, [view], offset)
        view.release()
        if observed != length:
            raise IOError("short direct read")
        return observed

    def run(name: str, trace: list[list[tuple[int, int]]]) -> dict[str, object]:
        started = time.monotonic()
        total = 0
        reads = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            for requests in trace:
                total += sum(pool.map(read_one, requests))
                reads += len(requests)
        elapsed = time.monotonic() - started
        return {
            "name": name,
            "tokens": len(trace),
            "read_operations": reads,
            "physical_bytes": total,
            "logical_bytes": (
                args.slab_budget * slab_bytes * len(trace)
            ),
            "alignment_overhead_bytes": (
                total - args.slab_budget * slab_bytes * len(trace)
            ),
            "elapsed_seconds": elapsed,
            "gib_per_second": total / (1 << 30) / elapsed,
            "tokens_per_second_if_one_layer": len(trace) / elapsed,
        }

    # Alternate order on a second pass so neither policy always benefits from
    # the drive's first-run state. O_DIRECT bypasses the page cache.
    results = [run("adaptive_slab_prefix", adaptive_trace)]
    results.append(run("whole_expert", whole_trace))
    results.append(run("whole_expert_repeat", whole_trace))
    results.append(run("adaptive_slab_prefix_repeat", adaptive_trace))
    os.close(descriptor)

    result = {
        "schema": "kimi-k3-layer01-progressive-slab-direct-io-v1",
        "status": "EXPLORATORY",
        "sidecar": str(args.sidecar),
        "model_layer": layer,
        "validation_tokens": args.tokens,
        "workers": args.workers,
        "direct_io": bool(getattr(os, "O_DIRECT", 0)),
        "slab_budget": args.slab_budget,
        "active_slabs": data.top_k * slab_count,
        "nominal_payload_fraction": args.slab_budget / (data.top_k * slab_count),
        "results": results,
        "warning": "This isolates direct storage reads for one layer; it does not include dequantization, GPU transfer, or expert arithmetic.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    for row in results:
        print(
            f"[slab-io] {row['name']} {row['gib_per_second']:.3f} GiB/s "
            f"reads={row['read_operations']} elapsed={row['elapsed_seconds']:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

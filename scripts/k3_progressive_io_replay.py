#!/usr/bin/env python3
"""Replay P20 sidecar ranges independently of K3 model compute.

The first implementation deliberately offers only honest buffered-pread modes.
Pinned/O_DIRECT/io_uring modes are added only after the request geometry is
measured; this tool never labels ordinary Python buffers as pinned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


GIB = 1024**3


@dataclass(frozen=True)
class Request:
    path: str
    offset: int
    length: int
    layer: int


def proc_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def load_requests(path: Path, include_means: bool) -> tuple[list[Request], int]:
    requests: list[Request] = []
    unresolved_fallback_bytes = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            region = row["region"]
            length = int(row["logical_length"])
            if region == "native-exact-expert":
                unresolved_fallback_bytes += length
                continue
            if region == "slab-mean" and not include_means:
                continue
            if region not in {"gate", "up", "down", "slab-mean"}:
                continue
            requests.append(Request(
                row["file_path"], int(row["file_offset"]), length,
                int(row["model_layer"])))
    return requests, unresolved_fallback_bytes


def coalesce_adjacent(requests: list[Request]) -> list[Request]:
    if not requests:
        return []
    result = [requests[0]]
    for item in requests[1:]:
        previous = result[-1]
        if (item.path == previous.path and item.layer == previous.layer and
                item.offset == previous.offset + previous.length):
            result[-1] = Request(
                previous.path, previous.offset,
                previous.length + item.length, previous.layer)
        else:
            result.append(item)
    return result


def drop_cache(paths: set[str]) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def replay(requests: list[Request], queue_depth: int) -> dict[str, object]:
    descriptors = {path: os.open(path, os.O_RDONLY) for path in
                   sorted({item.path for item in requests})}
    latencies_ms: list[float] = []

    def read_one(item: Request) -> tuple[int, float]:
        started = time.perf_counter_ns()
        payload = os.pread(descriptors[item.path], item.length, item.offset)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if len(payload) != item.length:
            raise OSError(
                f"short read {item.path}@{item.offset}: "
                f"{len(payload)} != {item.length}")
        return len(payload), elapsed_ms

    before = proc_read_bytes()
    started = time.perf_counter()
    submitted = 0
    try:
        if queue_depth <= 1:
            for item in requests:
                read_bytes, latency = read_one(item)
                submitted += read_bytes
                latencies_ms.append(latency)
        else:
            with ThreadPoolExecutor(max_workers=queue_depth) as pool:
                for read_bytes, latency in pool.map(read_one, requests):
                    submitted += read_bytes
                    latencies_ms.append(latency)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    elapsed = time.perf_counter() - started
    physical = max(0, proc_read_bytes() - before)
    return {
        "submitted_bytes": submitted,
        "os_physical_bytes": physical,
        "elapsed_seconds": elapsed,
        "submitted_gib_s": submitted / GIB / elapsed if elapsed else 0.0,
        "os_physical_gib_s": physical / GIB / elapsed if elapsed else 0.0,
        "request_latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--mode", choices=("current", "batched-pread"),
                        default="current")
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--include-means", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queue_depth <= 0:
        parser.error("--queue-depth must be positive")

    raw, unresolved = load_requests(args.trace, args.include_means)
    requests = coalesce_adjacent(raw) if args.mode == "batched-pread" else raw
    if args.cold:
        drop_cache({item.path for item in requests})
    result = replay(requests, args.queue_depth)
    logical = sum(item.length for item in requests)
    result.update({
        "schema": "k3-progressive-io-replay-v1",
        "backend": args.mode,
        "destination_mode": "ordinary-host-buffer",
        "cold_cache_requested": args.cold,
        "queue_depth": args.queue_depth,
        "input_requests": len(raw),
        "submitted_requests": len(requests),
        "logical_bytes": logical,
        "unresolved_exact_fallback_bytes": unresolved,
        "physical_over_logical": (
            result["os_physical_bytes"] / logical if logical else None),
        "note": "exact fallback model-shard ranges are not replayed by this sidecar-only baseline",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

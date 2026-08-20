#!/usr/bin/env python3
"""Replay K3 sidecar reads with the production one-layer dependency boundary.

The P20 trace charges physical bytes on the first component row of each
aligned slab read.  This tool reconstructs those reads, filters prompt/prefill
positions from the suite manifest, and compares the current per-slab plan with
an exact same-layer plan that deduplicates and merges only overlapping or
adjacent aligned ranges.  It never merges across a model layer, token, prompt,
or file.

Use --plan-only before a timed replay.  Timed reads use O_DIRECT and preadv()
into page-aligned anonymous mappings, matching the alignment contract of the
production P20 path without populating the page cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


GIB = 1024 ** 3


@dataclass(frozen=True)
class Request:
    path: str
    offset: int
    length: int


@dataclass
class LayerGroup:
    sequence: int
    position: int
    layer: int
    requests: list[Request]


def prompt_lengths(manifest: Path) -> list[int]:
    value = json.loads(manifest.read_text())
    rows = value.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise ValueError("suite manifest has no sequences")
    result = []
    for row in rows:
        count = row.get("prompt_token_count") if isinstance(row, dict) else None
        if not isinstance(count, int) or count <= 0:
            raise ValueError("suite manifest has an invalid prompt length")
        result.append(count)
    return result


def load_prefill_groups(trace: Path, lengths: list[int]) -> list[LayerGroup]:
    groups: list[LayerGroup] = []
    current_key: tuple[int, int, int] | None = None
    current: LayerGroup | None = None
    sequence = 0
    last_base: int | None = None
    with trace.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "base_pos", "token_index", "model_layer", "file_path",
            "aligned_offset", "aligned_length", "explicit_read_bytes",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("I/O trace is missing required P20 columns")
        for row_number, row in enumerate(reader, 2):
            try:
                base = int(row["base_pos"])
                token = int(row["token_index"])
                layer = int(row["model_layer"])
                offset = int(row["aligned_offset"])
                length = int(row["aligned_length"])
                physical = int(row["explicit_read_bytes"])
            except ValueError as error:
                raise ValueError(f"malformed trace row {row_number}: {error}")
            if last_base is not None and base < last_base:
                sequence += 1
            last_base = base
            position = base + token
            if sequence >= len(lengths) or position >= lengths[sequence]:
                continue
            if physical <= 0 or offset < 0 or length <= 0:
                continue
            if offset % mmap.PAGESIZE or length % mmap.PAGESIZE:
                raise ValueError(
                    f"unaligned physical trace row {row_number}: "
                    f"offset={offset} length={length}")
            key = (sequence, position, layer)
            if key != current_key:
                current = LayerGroup(sequence, position, layer, [])
                groups.append(current)
                current_key = key
            assert current is not None
            current.requests.append(Request(row["file_path"], offset, length))
    return groups


def coalesce(requests: list[Request]) -> list[Request]:
    result: list[Request] = []
    for item in sorted(set(requests), key=lambda value: (
            value.path, value.offset, value.length)):
        if (result and result[-1].path == item.path and
                item.offset <= result[-1].offset + result[-1].length):
            previous = result[-1]
            end = max(previous.offset + previous.length,
                      item.offset + item.length)
            result[-1] = Request(previous.path, previous.offset,
                                 end - previous.offset)
        else:
            result.append(item)
    return result


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def proc_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def direct_read(fd: int, request: Request, keep_bytes: bool) -> tuple[bytes | None, float]:
    buffer = mmap.mmap(-1, request.length)
    view = memoryview(buffer)
    started = time.perf_counter_ns()
    try:
        got = os.preadv(fd, [view], request.offset)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if got != request.length:
            raise OSError(
                f"short direct read {request.path}@{request.offset}: "
                f"{got} != {request.length}")
        payload = bytes(view) if keep_bytes else None
    finally:
        view.release()
        buffer.close()
    return payload, elapsed_ms


def digest_original_requests(
        originals: list[Request], spans: list[Request],
        payloads: list[bytes]) -> bytes:
    digest = hashlib.sha256()
    by_path: dict[str, list[tuple[Request, bytes]]] = {}
    for span, payload in zip(spans, payloads, strict=True):
        by_path.setdefault(span.path, []).append((span, payload))
    for item in originals:
        for span, payload in by_path.get(item.path, []):
            if (span.offset <= item.offset and
                    item.offset + item.length <= span.offset + span.length):
                relative = item.offset - span.offset
                digest.update(payload[relative:relative + item.length])
                break
        else:
            raise ValueError("coalesced replay did not cover an original read")
    return digest.digest()


def replay(
        groups: list[LayerGroup], mode: str, queue_depth: int,
        verify: bool) -> dict[str, object]:
    paths = sorted({item.path for group in groups for item in group.requests})
    direct_flag = getattr(os, "O_DIRECT", 0)
    if not direct_flag:
        raise OSError("O_DIRECT is unavailable on this platform")
    descriptors = {
        path: os.open(path, os.O_RDONLY | direct_flag) for path in paths
    }
    latencies: list[float] = []
    submitted_requests = 0
    submitted_bytes = 0
    digest = hashlib.sha256()
    before = proc_read_bytes()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=queue_depth) as pool:
            for group in groups:
                requests = (coalesce(group.requests)
                            if mode == "coalesced" else group.requests)
                futures = [
                    pool.submit(
                        direct_read, descriptors[item.path], item, verify)
                    for item in requests
                ]
                payloads: list[bytes] = []
                for future in futures:
                    payload, latency = future.result()
                    latencies.append(latency)
                    if verify:
                        assert payload is not None
                        payloads.append(payload)
                if verify:
                    digest.update(digest_original_requests(
                        group.requests, requests, payloads))
                submitted_requests += len(requests)
                submitted_bytes += sum(item.length for item in requests)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    elapsed = time.perf_counter() - started
    physical = max(0, proc_read_bytes() - before)
    return {
        "mode": mode,
        "queue_depth": queue_depth,
        "groups": len(groups),
        "submitted_requests": submitted_requests,
        "submitted_bytes": submitted_bytes,
        "elapsed_seconds": elapsed,
        "submitted_gib_per_second": submitted_bytes / GIB / elapsed,
        "os_physical_bytes": physical,
        "os_physical_gib_per_second": physical / GIB / elapsed,
        "request_latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "verification_sha256": digest.hexdigest() if verify else None,
    }


def plan_summary(groups: list[LayerGroup]) -> dict[str, object]:
    current_requests = sum(len(group.requests) for group in groups)
    current_bytes = sum(
        item.length for group in groups for item in group.requests)
    coalesced = [coalesce(group.requests) for group in groups]
    coalesced_requests = sum(len(items) for items in coalesced)
    coalesced_bytes = sum(item.length for items in coalesced for item in items)
    return {
        "layer_groups": len(groups),
        "current": {
            "requests": current_requests,
            "bytes": current_bytes,
        },
        "coalesced": {
            "requests": coalesced_requests,
            "bytes": coalesced_bytes,
        },
        "request_reduction_fraction": (
            1.0 - coalesced_requests / current_requests
            if current_requests else 0.0),
        "byte_reduction_fraction": (
            1.0 - coalesced_bytes / current_bytes
            if current_bytes else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("current", "coalesced"))
    parser.add_argument("--queue-depth", type=int, default=16)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.queue_depth <= 0:
        parser.error("--queue-depth must be positive")
    if args.limit_groups is not None and args.limit_groups <= 0:
        parser.error("--limit-groups must be positive")

    groups = load_prefill_groups(args.trace, prompt_lengths(args.manifest))
    if args.limit_groups is not None:
        groups = groups[:args.limit_groups]
    result: dict[str, object] = {
        "schema": "k3-layer-direct-read-replay-v1",
        "trace": str(args.trace),
        "manifest": str(args.manifest),
        "scope": "prefill-one-token-one-layer",
        "plan": plan_summary(groups),
    }
    if args.mode:
        result["replay"] = replay(
            groups, args.mode, args.queue_depth, args.verify)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

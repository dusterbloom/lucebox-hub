#!/usr/bin/env python3
"""Read-only O_DIRECT rooflines for K3 sidecars and frozen P56 traffic.

The raw lane reads one complete sidecar sequentially with an aligned,
persistent buffer.  The trace lane reconstructs the exact unmerged physical
requests from the frozen P56 prefill trace, retains the one-token/one-layer
dependency boundary, and replays each group through a fixed pool of aligned
buffers.  Neither lane writes to the measured device, drops caches, or runs
model arithmetic.

O_DIRECT bypasses the page cache.  Results also record process physical-read
counters, underlying block-device counters, swap activity, NVMe temperature,
link state, and scheduler state so a fast but contaminated run is not silently
promoted as a roofline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


GIB = 1024 ** 3
SECTOR_BYTES = 512


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


def read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 2:
                try:
                    result[fields[0].rstrip(":" )] = int(fields[1])
                except ValueError:
                    continue
    except OSError:
        pass
    return result


def proc_io() -> dict[str, int]:
    return read_key_values(Path("/proc/self/io"))


def vmstat() -> dict[str, int]:
    return read_key_values(Path("/proc/vmstat"))


def meminfo() -> dict[str, int]:
    return read_key_values(Path("/proc/meminfo"))


def block_stats(device: str) -> dict[str, int]:
    path = Path("/sys/class/block") / device / "stat"
    try:
        fields = [int(value) for value in path.read_text().split()]
    except (OSError, ValueError):
        return {}
    if len(fields) < 11:
        return {}
    return {
        "read_operations": fields[0],
        "read_bytes": fields[2] * SECTOR_BYTES,
        "write_operations": fields[4],
        "write_bytes": fields[6] * SECTOR_BYTES,
        "io_busy_ms": fields[9],
        "weighted_io_ms": fields[10],
    }


def nvme_temperatures() -> dict[str, int]:
    result: dict[str, int] = {}
    for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if root.joinpath("name").read_text().strip() != "nvme":
                continue
        except OSError:
            continue
        labels: dict[str, str] = {}
        for label in root.glob("temp*_label"):
            try:
                labels[label.stem.removesuffix("_label")] = (
                    label.read_text().strip())
            except OSError:
                pass
        for sensor in root.glob("temp*_input"):
            try:
                value = int(sensor.read_text().strip())
            except (OSError, ValueError):
                continue
            stem = sensor.stem.removesuffix("_input")
            result[f"{root.name}:{labels.get(stem, stem)}"] = value
    return result


def first_parent_value(start: Path, name: str) -> str | None:
    try:
        current = start.resolve()
    except OSError:
        current = start
    for parent in (current, *current.parents):
        candidate = parent / name
        try:
            return candidate.read_text().strip()
        except OSError:
            continue
    return None


def platform_description(device: str) -> dict[str, object]:
    root = Path("/sys/class/block") / device
    scheduler = None
    try:
        scheduler = root.joinpath("queue/scheduler").read_text().strip()
    except OSError:
        pass
    device_root = root / "device"
    return {
        "block_device": device,
        "model": first_parent_value(device_root, "model"),
        "serial": first_parent_value(device_root, "serial"),
        "firmware_revision": first_parent_value(device_root, "firmware_rev"),
        "current_link_speed": first_parent_value(
            device_root, "current_link_speed"),
        "current_link_width": first_parent_value(
            device_root, "current_link_width"),
        "scheduler": scheduler,
        "page_size": mmap.PAGESIZE,
        "o_direct_constant": getattr(os, "O_DIRECT", 0),
    }


def snapshot(device: str) -> dict[str, object]:
    memory = meminfo()
    return {
        "monotonic_ns": time.monotonic_ns(),
        "process_io": proc_io(),
        "block": block_stats(device),
        "vmstat": vmstat(),
        "nvme_temperature_millicelsius": nvme_temperatures(),
        "memory_kib": {
            key: memory.get(key, 0)
            for key in ("MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback")
        },
    }


def delta(after: dict[str, int], before: dict[str, int], key: str) -> int:
    return max(0, after.get(key, 0) - before.get(key, 0))


def controls(
        device: str, submitted_bytes: int,
        before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    process_before = before["process_io"]
    process_after = after["process_io"]
    block_before = before["block"]
    block_after = after["block"]
    vm_before = before["vmstat"]
    vm_after = after["vmstat"]
    assert isinstance(process_before, dict) and isinstance(process_after, dict)
    assert isinstance(block_before, dict) and isinstance(block_after, dict)
    assert isinstance(vm_before, dict) and isinstance(vm_after, dict)
    process_read_bytes = delta(
        process_after, process_before, "read_bytes")
    device_read_bytes = delta(block_after, block_before, "read_bytes")
    device_write_bytes = delta(block_after, block_before, "write_bytes")
    background_read_bytes = max(0, device_read_bytes - process_read_bytes)
    physical_fraction = (
        process_read_bytes / submitted_bytes if submitted_bytes else 0.0)
    background_fraction = (
        background_read_bytes / device_read_bytes if device_read_bytes else 0.0)
    swap_in_pages = delta(vm_after, vm_before, "pswpin")
    swap_out_pages = delta(vm_after, vm_before, "pswpout")
    return {
        "block_device": device,
        "process_physical_read_bytes": process_read_bytes,
        "device_read_bytes": device_read_bytes,
        "device_write_bytes": device_write_bytes,
        "background_read_bytes_upper_bound": background_read_bytes,
        "process_physical_over_submitted": physical_fraction,
        "background_over_device_read": background_fraction,
        "swap_in_pages": swap_in_pages,
        "swap_out_pages": swap_out_pages,
        "io_busy_ms": delta(block_after, block_before, "io_busy_ms"),
        "temperature_before_millicelsius": before[
            "nvme_temperature_millicelsius"],
        "temperature_after_millicelsius": after[
            "nvme_temperature_millicelsius"],
        "memory_before_kib": before["memory_kib"],
        "memory_after_kib": after["memory_kib"],
        "clean_gate": (
            0.98 <= physical_fraction <= 1.02 and
            background_fraction <= 0.02 and
            swap_in_pages == 0 and swap_out_pages == 0),
        "gate_rule": (
            "process physical/submitted within 2%; background reads <=2%; "
            "zero swap-in and swap-out"),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


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


def request_fingerprint(groups: list[LayerGroup]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        digest.update(
            f"G\0{group.sequence}\0{group.position}\0{group.layer}\n".encode())
        for item in group.requests:
            digest.update(
                f"R\0{item.path}\0{item.offset}\0{item.length}\n".encode())
    return digest.hexdigest()


def direct_descriptors(paths: list[str]) -> dict[str, int]:
    direct_flag = getattr(os, "O_DIRECT", 0)
    if not direct_flag:
        raise OSError("O_DIRECT is unavailable on this platform")
    return {
        path: os.open(path, os.O_RDONLY | os.O_CLOEXEC | direct_flag)
        for path in paths
    }


class PersistentDirectPool:
    def __init__(self, descriptors: dict[str, int], queue_depth: int,
                 buffer_bytes: int):
        self.descriptors = descriptors
        self.queue_depth = queue_depth
        self.buffer_bytes = buffer_bytes
        self.local = threading.local()
        self.buffers: list[mmap.mmap] = []
        self.buffers_lock = threading.Lock()

    def read(self, request: Request) -> float:
        buffer = getattr(self.local, "buffer", None)
        if buffer is None:
            buffer = mmap.mmap(-1, self.buffer_bytes)
            self.local.buffer = buffer
            with self.buffers_lock:
                self.buffers.append(buffer)
        view = memoryview(buffer)[:request.length]
        started = time.perf_counter_ns()
        try:
            observed = os.preadv(
                self.descriptors[request.path], [view], request.offset)
        finally:
            view.release()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if observed != request.length:
            raise OSError(
                f"short direct read {request.path}@{request.offset}: "
                f"{observed} != {request.length}")
        return elapsed_ms

    def close(self) -> None:
        self.local = threading.local()
        for buffer in self.buffers:
            buffer.close()


def raw_pass(file_path: Path, block_bytes: int, device: str) -> dict[str, object]:
    aligned_bytes = file_path.stat().st_size // mmap.PAGESIZE * mmap.PAGESIZE
    if aligned_bytes <= 0:
        raise ValueError("raw input file has no aligned data")
    descriptors = direct_descriptors([str(file_path)])
    descriptor = descriptors[str(file_path)]
    buffer = mmap.mmap(-1, block_bytes)
    before = snapshot(device)
    latencies: list[float] = []
    started = time.perf_counter()
    offset = 0
    try:
        while offset < aligned_bytes:
            length = min(block_bytes, aligned_bytes - offset)
            view = memoryview(buffer)[:length]
            read_started = time.perf_counter_ns()
            try:
                observed = os.preadv(descriptor, [view], offset)
            finally:
                view.release()
            latencies.append((time.perf_counter_ns() - read_started) / 1e6)
            if observed != length:
                raise OSError(
                    f"short raw direct read @{offset}: {observed} != {length}")
            offset += length
    finally:
        buffer.close()
        os.close(descriptor)
    elapsed = time.perf_counter() - started
    after = snapshot(device)
    return {
        "block_bytes": block_bytes,
        "submitted_requests": len(latencies),
        "submitted_bytes": aligned_bytes,
        "elapsed_seconds": elapsed,
        "submitted_gib_per_second": aligned_bytes / GIB / elapsed,
        "request_latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "controls": controls(device, aligned_bytes, before, after),
    }


def trace_pass(groups: list[LayerGroup], queue_depth: int,
               device: str) -> dict[str, object]:
    paths = sorted({item.path for group in groups for item in group.requests})
    descriptors = direct_descriptors(paths)
    max_request = max(
        item.length for group in groups for item in group.requests)
    pool = PersistentDirectPool(descriptors, queue_depth, max_request)
    submitted_requests = 0
    submitted_bytes = 0
    latencies: list[float] = []
    before = snapshot(device)
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=queue_depth) as executor:
            for group in groups:
                # Intentionally retain every original request and the barrier
                # between causal token/layer groups.  P56R already rejected
                # larger same-layer coalesced spans.
                futures = [executor.submit(pool.read, item)
                           for item in group.requests]
                for future in futures:
                    latencies.append(future.result())
                submitted_requests += len(group.requests)
                submitted_bytes += sum(item.length for item in group.requests)
    finally:
        pool.close()
        for descriptor in descriptors.values():
            os.close(descriptor)
    elapsed = time.perf_counter() - started
    after = snapshot(device)
    return {
        "queue_depth": queue_depth,
        "persistent_aligned_buffers": len(pool.buffers),
        "groups": len(groups),
        "submitted_requests": submitted_requests,
        "submitted_bytes": submitted_bytes,
        "elapsed_seconds": elapsed,
        "submitted_gib_per_second": submitted_bytes / GIB / elapsed,
        "request_latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "controls": controls(device, submitted_bytes, before, after),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-device", default="nvme0n1")
    parser.add_argument("--output", type=Path)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("raw")
    raw.add_argument("--file", type=Path, required=True)
    raw.add_argument("--block-mib", type=int, nargs="+", default=[1, 4, 8])
    raw.add_argument("--passes", type=int, default=3)
    add_common(raw)
    trace = commands.add_parser("trace")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--manifest", type=Path, required=True)
    trace.add_argument("--queue-depth", type=int, nargs="+", default=[4, 8, 16, 32])
    trace.add_argument("--passes", type=int, default=1)
    trace.add_argument("--limit-groups", type=int)
    add_common(trace)
    args = parser.parse_args()
    if args.passes <= 0:
        parser.error("--passes must be positive")

    result: dict[str, object] = {
        "schema": "k3-xg7000-storage-roofline-v1",
        "scope": "READ_ONLY_NO_MODEL_NO_CACHE_DROP",
        "command": args.command,
        "platform": platform_description(args.block_device),
    }
    if args.command == "raw":
        if any(value <= 0 for value in args.block_mib):
            parser.error("--block-mib values must be positive")
        for value in args.block_mib:
            if (value << 20) % mmap.PAGESIZE:
                parser.error("raw block sizes must be page aligned")
        result["source"] = {
            "file": str(args.file),
            "file_bytes": args.file.stat().st_size,
        }
        arms = []
        for pass_index in range(args.passes):
            order = args.block_mib if pass_index % 2 == 0 else list(
                reversed(args.block_mib))
            for block_mib in order:
                arm = raw_pass(
                    args.file, block_mib << 20, args.block_device)
                arm["pass"] = pass_index + 1
                arms.append(arm)
        result["arms"] = arms
    else:
        if any(value <= 0 or value > 256 for value in args.queue_depth):
            parser.error("queue depths must be in 1..256")
        if args.limit_groups is not None and args.limit_groups <= 0:
            parser.error("--limit-groups must be positive")
        groups = load_prefill_groups(
            args.trace, prompt_lengths(args.manifest))
        if args.limit_groups is not None:
            groups = groups[:args.limit_groups]
        plan_requests = sum(len(group.requests) for group in groups)
        plan_bytes = sum(
            item.length for group in groups for item in group.requests)
        result["source"] = {
            "trace": str(args.trace),
            "trace_sha256": sha256(args.trace),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
        }
        result["plan"] = {
            "dependency_boundary": "one-prompt-position/model-layer",
            "coalescing": False,
            "groups": len(groups),
            "requests": plan_requests,
            "bytes": plan_bytes,
            "request_fingerprint_sha256": request_fingerprint(groups),
        }
        arms = []
        for pass_index in range(args.passes):
            order = args.queue_depth if pass_index % 2 == 0 else list(
                reversed(args.queue_depth))
            for queue_depth in order:
                arm = trace_pass(groups, queue_depth, args.block_device)
                arm["pass"] = pass_index + 1
                arms.append(arm)
        result["arms"] = arms

    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

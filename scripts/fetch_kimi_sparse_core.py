#!/usr/bin/env python3
"""Plan and fetch only the allocated ranges of a sparse Kimi-K3 GGUF core."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


COPY_CHUNK = 8 << 20
ROUTED_SUFFIXES = (
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
    ".ffn_gate_up_exps.weight",
)


@dataclass(frozen=True)
class ByteRange:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(COPY_CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def digest_range(path: Path, span: ByteRange) -> str:
    result = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        cursor = 0
        while cursor < span.length:
            block = os.pread(
                descriptor, min(COPY_CHUNK, span.length - cursor),
                span.offset + cursor,
            )
            if not block:
                raise IOError(f"short read from {path} at {span.offset + cursor}")
            result.update(block)
            cursor += len(block)
    finally:
        os.close(descriptor)
    return result.hexdigest()


def digest_ranges(path: Path, spans: list[ByteRange]) -> str:
    result = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        for span in spans:
            cursor = 0
            while cursor < span.length:
                block = os.pread(
                    descriptor, min(COPY_CHUNK, span.length - cursor),
                    span.offset + cursor,
                )
                if not block:
                    raise IOError(
                        f"short read from {path} at {span.offset + cursor}"
                    )
                result.update(block)
                cursor += len(block)
    finally:
        os.close(descriptor)
    return result.hexdigest()


def digest_range_rows(path: Path, spans: list[ByteRange]) -> tuple[list[str], str]:
    combined = hashlib.sha256()
    rows = []
    descriptor = os.open(path, os.O_RDONLY)
    try:
        for span in spans:
            current = hashlib.sha256()
            cursor = 0
            while cursor < span.length:
                block = os.pread(
                    descriptor, min(COPY_CHUNK, span.length - cursor),
                    span.offset + cursor,
                )
                if not block:
                    raise IOError(
                        f"short read from {path} at {span.offset + cursor}"
                    )
                current.update(block)
                combined.update(block)
                cursor += len(block)
            rows.append(current.hexdigest())
    finally:
        os.close(descriptor)
    return rows, combined.hexdigest()


def split_paths(first: Path) -> list[Path]:
    match = re.match(r"^(.*-)[0-9]{5}(-of-[0-9]{5}\.gguf)$", first.name)
    if not match:
        return [first]
    paths = sorted(first.parent.glob(match.group(1) + "*" + match.group(2)))
    if not paths:
        raise ValueError("no GGUF shards found")
    return paths


def is_routed(name: str) -> bool:
    return any(suffix in name for suffix in ROUTED_SUFFIXES)


def coalesce(ranges: list[ByteRange], maximum_gap: int = 4096) -> list[ByteRange]:
    ordered = sorted(ranges, key=lambda value: value.offset)
    if not ordered:
        return []
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.offset <= previous.end + maximum_gap:
            merged[-1] = ByteRange(
                previous.offset, max(previous.end, current.end) - previous.offset
            )
        else:
            merged.append(current)
    return merged


def plan(args: argparse.Namespace) -> int:
    from gguf import GGUFReader

    if args.output.exists():
        raise FileExistsError(args.output)
    source_manifest = json.loads(args.sparse_manifest.read_text())
    expected = source_manifest["shards_detail"]
    sources = split_paths(args.model)
    if len(sources) != len(expected):
        raise ValueError("source shards disagree with sparse manifest")

    shards = []
    for index, (source, reference) in enumerate(zip(sources, expected), start=1):
        reader = GGUFReader(source, "r")
        spans = [ByteRange(0, int(reader.data_offset))]
        spans.extend(
            ByteRange(int(tensor.data_offset), int(tensor.n_bytes))
            for tensor in reader.tensors
            if not is_routed(tensor.name)
        )
        spans = coalesce(spans)
        copied = sum(span.length for span in spans)
        if source.stat().st_size != int(reference["logical_bytes"]):
            raise ValueError(f"logical size changed: {source}")
        if copied != int(reference["copied_bytes"]):
            raise ValueError(f"copied byte count changed: {source}")
        range_digests, combined = digest_range_rows(source, spans)
        rows = []
        for range_index, (span, range_digest) in enumerate(
                zip(spans, range_digests)):
            rows.append({
                "index": range_index,
                "offset": span.offset,
                "length": span.length,
                "sha256": range_digest,
            })
        if combined != reference["source_range_sha256"]:
            raise ValueError(f"source range digest changed: {source}")
        shards.append({
            "index": index,
            "name": source.name,
            "url": f"{args.base_url.rstrip('/')}/{source.name}",
            "logical_bytes": source.stat().st_size,
            "copied_bytes": copied,
            "combined_range_sha256": combined,
            "ranges": rows,
        })
        print(f"[{index}/{len(sources)}] {source.name}: {copied} bytes", flush=True)
        del reader

    payload = {
        "schema": "kimi-k3-sparse-range-fetch-plan-v1",
        "created_unix": int(time.time()),
        "source_model": str(args.model.resolve()),
        "source_sparse_manifest": str(args.sparse_manifest.resolve()),
        "source_sparse_manifest_sha256": digest(args.sparse_manifest),
        "base_url": args.base_url,
        "shards": shards,
        "totals": {
            "shards": len(shards),
            "ranges": sum(len(shard["ranges"]) for shard in shards),
            "logical_bytes": sum(shard["logical_bytes"] for shard in shards),
            "copied_bytes": sum(shard["copied_bytes"] for shard in shards),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], sort_keys=True))
    return 0


def load_state(path: Path, plan_sha256: str) -> dict:
    if not path.exists():
        return {"schema": "kimi-k3-sparse-range-fetch-state-v1",
                "plan_sha256": plan_sha256, "completed": []}
    state = json.loads(path.read_text())
    if state.get("plan_sha256") != plan_sha256:
        raise ValueError("fetch state belongs to a different plan")
    return state


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, path)


def fetch_one(url: str, destination: Path, span: ByteRange, expected: str) -> None:
    end = span.end - 1
    for attempt in range(1, 4):
        result = hashlib.sha256()
        descriptor = None
        try:
            descriptor = os.open(destination, os.O_WRONLY)
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={span.offset}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                content_range = response.headers.get("Content-Range")
                expected_range = f"bytes {span.offset}-{end}/"
                if (response.status != 206 or content_range is None or
                        not content_range.startswith(expected_range)):
                    raise IOError(f"server ignored byte range: HTTP {response.status}")
                cursor = 0
                while cursor < span.length:
                    block = response.read(min(COPY_CHUNK, span.length - cursor))
                    if not block:
                        break
                    written = os.pwrite(descriptor, block, span.offset + cursor)
                    if written != len(block):
                        raise IOError(f"short write to {destination}")
                    result.update(block)
                    cursor += len(block)
                if cursor != span.length:
                    raise IOError(f"short HTTP range: {cursor} != {span.length}")
            os.fsync(descriptor)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt)
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if result.hexdigest() == expected:
            return
        if attempt == 3:
            raise ValueError(f"range SHA-256 mismatch for {destination}@{span.offset}")


def fetch(args: argparse.Namespace) -> int:
    plan_path = args.plan
    payload = json.loads(plan_path.read_text())
    if payload.get("schema") != "kimi-k3-sparse-range-fetch-plan-v1":
        raise ValueError("unsupported plan schema")
    plan_sha256 = digest(plan_path)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    state_path = args.output_directory / "fetch-state.json"
    state = load_state(state_path, plan_sha256)
    completed = set(state["completed"])
    lock = threading.Lock()

    jobs = []
    for shard in payload["shards"]:
        destination = args.output_directory / shard["name"]
        if destination.exists():
            if destination.stat().st_size != shard["logical_bytes"]:
                raise ValueError(f"wrong existing logical size: {destination}")
        else:
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
            os.ftruncate(descriptor, shard["logical_bytes"])
            os.close(descriptor)
        for row in shard["ranges"]:
            key = f"{shard['index']}:{row['index']}"
            span = ByteRange(row["offset"], row["length"])
            if key in completed:
                if digest_range(destination, span) == row["sha256"]:
                    continue
                completed.remove(key)
            jobs.append((key, shard["url"], destination, span, row["sha256"]))

    started = time.monotonic()

    def worker(job: tuple) -> str:
        key, url, destination, span, expected = job
        fetch_one(url, destination, span, expected)
        with lock:
            completed.add(key)
            state["completed"] = sorted(completed)
            save_state(state_path, state)
            print(f"[{len(completed)}/{payload['totals']['ranges']}] {key}", flush=True)
        return key

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, jobs))

    verification = []
    for shard in payload["shards"]:
        destination = args.output_directory / shard["name"]
        spans = [ByteRange(row["offset"], row["length"])
                 for row in shard["ranges"]]
        actual = digest_ranges(destination, spans)
        if actual != shard["combined_range_sha256"]:
            raise ValueError(f"combined range SHA-256 mismatch: {destination}")
        stat = destination.stat()
        verification.append({
            "name": shard["name"],
            "logical_bytes": stat.st_size,
            "allocated_bytes": stat.st_blocks * 512,
            "combined_range_sha256": actual,
        })

    result = {
        "schema": "kimi-k3-sparse-range-fetch-result-v1",
        "status": "VERIFIED_COMPLETE",
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "command": sys.argv,
        "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "shards": verification,
        "totals": {
            "logical_bytes": sum(row["logical_bytes"] for row in verification),
            "allocated_bytes": sum(row["allocated_bytes"] for row in verification),
        },
    }
    result_path = args.output_directory / "fetch-result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    planner = subparsers.add_parser("plan")
    planner.add_argument("model", type=Path)
    planner.add_argument("sparse_manifest", type=Path)
    planner.add_argument("base_url")
    planner.add_argument("output", type=Path)
    planner.set_defaults(run=plan)

    downloader = subparsers.add_parser("fetch")
    downloader.add_argument("plan", type=Path)
    downloader.add_argument("output_directory", type=Path)
    downloader.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    downloader.set_defaults(run=fetch)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())

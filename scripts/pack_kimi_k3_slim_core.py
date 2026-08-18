#!/usr/bin/env python3
"""Build sparse Kimi-K3 GGUF shards containing metadata and non-routed core.

The output keeps every source file's logical size and GGUF tensor metadata so
the existing loader can bind the normal core tensors. Routed expert payload
ranges are filesystem holes and must only be used with the fail-closed
DFLASH_KIMI_SIDECAR_AUTHORITATIVE provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from gguf import GGUFReader


COPY_CHUNK = 8 << 20
EXPECTED_ROUTED_TENSORS = 92 * 3
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


def coalesce_ranges(ranges: list[ByteRange], maximum_gap: int = 4096) -> list[ByteRange]:
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


def digest_ranges(path: Path, ranges: list[ByteRange]) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        for span in ranges:
            cursor = 0
            while cursor < span.length:
                block = os.pread(
                    descriptor, min(COPY_CHUNK, span.length - cursor),
                    span.offset + cursor,
                )
                if not block:
                    raise IOError(f"short read from {path} at {span.offset + cursor}")
                digest.update(block)
                cursor += len(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def copy_sparse_ranges(
    source: Path, destination: Path, logical_size: int,
    ranges: list[ByteRange], verify: bool,
) -> dict[str, object]:
    temporary = destination.with_name(destination.name + ".partial")
    source_fd = os.open(source, os.O_RDONLY)
    destination_fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    source_digest = hashlib.sha256()
    copied = 0
    try:
        os.ftruncate(destination_fd, logical_size)
        for span in ranges:
            cursor = 0
            while cursor < span.length:
                block = os.pread(
                    source_fd, min(COPY_CHUNK, span.length - cursor),
                    span.offset + cursor,
                )
                if not block:
                    raise IOError(
                        f"short read from {source} at {span.offset + cursor}"
                    )
                written = os.pwrite(destination_fd, block, span.offset + cursor)
                if written != len(block):
                    raise IOError(
                        f"short write to {temporary} at {span.offset + cursor}"
                    )
                source_digest.update(block)
                copied += len(block)
                cursor += len(block)
        os.fsync(destination_fd)
    except BaseException:
        os.close(destination_fd)
        os.close(source_fd)
        raise
    os.close(destination_fd)
    os.close(source_fd)
    source_sha = source_digest.hexdigest()
    destination_sha = digest_ranges(temporary, ranges) if verify else None
    if verify and destination_sha != source_sha:
        raise ValueError(f"copied-range hash mismatch for {destination.name}")
    os.replace(temporary, destination)
    stat = destination.stat()
    return {
        "logical_bytes": logical_size,
        "copied_bytes": copied,
        "allocated_bytes": stat.st_blocks * 512,
        "copied_ranges": len(ranges),
        "source_range_sha256": source_sha,
        "destination_range_sha256": destination_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="first source GGUF shard")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    sources = split_paths(args.model)
    if args.output_directory.exists():
        raise FileExistsError(
            f"refusing existing output directory: {args.output_directory}"
        )
    if args.manifest.exists():
        raise FileExistsError(f"refusing existing manifest: {args.manifest}")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    shard_results: list[dict[str, object]] = []
    routed_names: list[str] = []
    seen_names: set[str] = set()
    total_non_routed_tensor_bytes = 0
    total_routed_tensor_bytes = 0

    for index, source in enumerate(sources, start=1):
        reader = GGUFReader(source, "r")
        source_size = source.stat().st_size
        ranges = [ByteRange(0, int(reader.data_offset))]
        shard_routed = 0
        shard_non_routed = 0
        for tensor in reader.tensors:
            if tensor.name in seen_names:
                raise ValueError(f"duplicate tensor name: {tensor.name}")
            seen_names.add(tensor.name)
            # GGUFReader exposes tensor.data_offset as an absolute file
            # offset. The first tensor commonly begins exactly at
            # reader.data_offset; adding the data base twice corrupts every
            # later range calculation.
            absolute = int(tensor.data_offset)
            length = int(tensor.n_bytes)
            if absolute < int(reader.data_offset) or absolute + length > source_size:
                raise ValueError(f"tensor range is outside {source}: {tensor.name}")
            if is_routed(tensor.name):
                routed_names.append(tensor.name)
                shard_routed += length
                total_routed_tensor_bytes += length
            else:
                ranges.append(ByteRange(absolute, length))
                shard_non_routed += length
                total_non_routed_tensor_bytes += length
        copied_ranges = coalesce_ranges(ranges)
        destination = args.output_directory / source.name
        result = copy_sparse_ranges(
            source, destination, source_size, copied_ranges,
            verify=not args.skip_verify,
        )
        result.update({
            "index": index,
            "source": str(source.resolve()),
            "destination": str(destination),
            "header_bytes": int(reader.data_offset),
            "non_routed_tensor_bytes": shard_non_routed,
            "routed_tensor_hole_bytes": shard_routed,
        })
        shard_results.append(result)
        print(
            f"[{index}/{len(sources)}] {source.name}: "
            f"copied={result['copied_bytes']} allocated={result['allocated_bytes']} "
            f"routed-hole={shard_routed}",
            flush=True,
        )
        del reader

    if len(routed_names) != EXPECTED_ROUTED_TENSORS:
        raise ValueError(
            f"expected {EXPECTED_ROUTED_TENSORS} routed tensors, "
            f"found {len(routed_names)}"
        )
    total_logical = sum(int(value["logical_bytes"]) for value in shard_results)
    total_copied = sum(int(value["copied_bytes"]) for value in shard_results)
    total_allocated = sum(int(value["allocated_bytes"]) for value in shard_results)
    manifest = {
        "schema": "kimi-k3-slim-core-sparse-gguf-v1",
        "classification": "EXACT_NON_ROUTED_COPY_ROUTED_PAYLOAD_HOLES",
        "created_unix": int(time.time()),
        "elapsed_seconds": time.monotonic() - started,
        "source_first_shard": str(args.model.resolve()),
        "output_directory": str(args.output_directory),
        "verification": "full copied-range SHA-256" if not args.skip_verify else "skipped",
        "required_runtime": {
            "environment": "DFLASH_KIMI_SIDECAR_AUTHORITATIVE=1",
            "failure_policy": "fail closed; never evaluate routed holes",
        },
        "totals": {
            "shards": len(shard_results),
            "tensors": len(seen_names),
            "routed_tensors": len(routed_names),
            "logical_bytes": total_logical,
            "copied_bytes": total_copied,
            "allocated_bytes": total_allocated,
            "non_routed_tensor_bytes": total_non_routed_tensor_bytes,
            "routed_tensor_hole_bytes": total_routed_tensor_bytes,
        },
        "routed_tensor_names": sorted(routed_names),
        "shards_detail": shard_results,
    }
    temporary_manifest = args.manifest.with_name(args.manifest.name + ".partial")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, args.manifest)
    print(json.dumps(manifest["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

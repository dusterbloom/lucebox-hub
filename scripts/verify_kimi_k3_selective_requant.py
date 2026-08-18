#!/usr/bin/env python3
"""Verify that a selective Kimi-K3 requant changed only preregistered tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader


QTYPE_NAMES = {value: value.name.lower() for value in GGMLQuantizationType}


def split_paths(first: Path) -> list[Path]:
    match = re.match(r"^(.*-)[0-9]{5}(-of-[0-9]{5}\.gguf)$", first.name)
    if not match:
        return [first]
    paths = sorted(first.parent.glob(match.group(1) + "*" + match.group(2)))
    if not paths:
        raise ValueError(f"no GGUF shards found beside {first}")
    return paths


def tensor_map(paths: list[Path]) -> tuple[dict[str, object], list[GGUFReader]]:
    readers: list[GGUFReader] = []
    tensors: dict[str, object] = {}
    for path in paths:
        reader = GGUFReader(path, "r")
        readers.append(reader)
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise ValueError(f"duplicate tensor: {tensor.name}")
            tensors[tensor.name] = tensor
    return tensors, readers


def window_offsets(length: int, seed: int, window: int) -> list[int]:
    if length <= window:
        return [0]
    rng = random.Random(seed)
    candidates = {0, (length - window) // 2, length - window}
    candidates.add(rng.randrange(0, length - window + 1))
    return sorted(candidates)


def sampled_equal(left: object, right: object, seed: int, window: int) -> tuple[bool, int]:
    left_view = memoryview(left.data).cast("B")
    right_view = memoryview(right.data).cast("B")
    if len(left_view) != len(right_view):
        return False, 0
    checked = 0
    for offset in window_offsets(len(left_view), seed, window):
        end = min(offset + window, len(left_view))
        checked += end - offset
        if left_view[offset:end] != right_view[offset:end]:
            return False, checked
    return True, checked


def sha256_tensor(tensor: object, chunk_bytes: int = 8 << 20) -> str:
    view = memoryview(tensor.data).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(view), chunk_bytes):
        digest.update(view[offset : offset + chunk_bytes])
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--window-bytes", type=int, default=4096)
    parser.add_argument("--hash-all-nontarget", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    target_names = {row["name"] for row in plan["targets"]}
    expected_type = plan["target_type"]
    source_paths = split_paths(args.source)
    candidate_paths = split_paths(args.candidate)
    source, source_readers = tensor_map(source_paths)
    candidate, candidate_readers = tensor_map(candidate_paths)
    # Keep the readers alive: the tensor arrays are views into their mmaps.
    _readers = source_readers + candidate_readers

    if source.keys() != candidate.keys():
        missing = sorted(source.keys() - candidate.keys())
        extra = sorted(candidate.keys() - source.keys())
        raise ValueError(f"tensor-name mismatch: missing={missing[:5]}, extra={extra[:5]}")
    if target_names - source.keys():
        raise ValueError("plan contains targets absent from source")

    changed_types: list[str] = []
    sampled_bytes = 0
    sampled_failures: list[str] = []
    full_hash_failures: list[str] = []
    for index, name in enumerate(sorted(source)):
        left = source[name]
        right = candidate[name]
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError(f"shape changed: {name}: {left.shape} -> {right.shape}")
        left_type = QTYPE_NAMES[left.tensor_type]
        right_type = QTYPE_NAMES[right.tensor_type]
        if name in target_names:
            if left_type != "q6_k" or right_type != expected_type:
                raise ValueError(
                    f"target qtype mismatch: {name}: {left_type} -> {right_type}"
                )
            changed_types.append(name)
            continue
        if left_type != right_type:
            raise ValueError(f"non-target qtype changed: {name}: {left_type} -> {right_type}")
        equal, checked = sampled_equal(left, right, index, args.window_bytes)
        sampled_bytes += checked
        if not equal:
            sampled_failures.append(name)
        if args.hash_all_nontarget and sha256_tensor(left) != sha256_tensor(right):
            full_hash_failures.append(name)

    if set(changed_types) != target_names:
        raise ValueError("changed target set differs from preregistered plan")
    if sampled_failures or full_hash_failures:
        raise ValueError(
            f"non-target byte mismatch: sampled={sampled_failures[:5]}, "
            f"full_hash={full_hash_failures[:5]}"
        )

    result = {
        "schema": "kimi-k3-selective-requant-verification-v1",
        "status": "PASS",
        "source_first_shard": str(args.source),
        "candidate_first_shard": str(args.candidate),
        "source_shards": [
            {"path": str(path), "bytes": path.stat().st_size} for path in source_paths
        ],
        "candidate_shards": [
            {"path": str(path), "bytes": path.stat().st_size} for path in candidate_paths
        ],
        "source_total_bytes": sum(path.stat().st_size for path in source_paths),
        "candidate_total_bytes": sum(path.stat().st_size for path in candidate_paths),
        "saved_file_bytes": (
            sum(path.stat().st_size for path in source_paths)
            - sum(path.stat().st_size for path in candidate_paths)
        ),
        "tensor_count": len(source),
        "changed_tensor_count": len(changed_types),
        "changed_tensor_names": sorted(changed_types),
        "target_type": expected_type,
        "non_target_tensor_count": len(source) - len(changed_types),
        "non_target_sample_window_bytes": args.window_bytes,
        "non_target_sampled_bytes": sampled_bytes,
        "non_target_sample_result": "PASS",
        "non_target_full_hash": "PASS" if args.hash_all_nontarget else "NOT_RUN",
        "semantic_quality": "OPEN",
        "runtime_performance": "OPEN",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "tensor_count": result["tensor_count"],
        "changed_tensor_count": result["changed_tensor_count"],
        "saved_gib": result["saved_file_bytes"] / 2**30,
        "sampled_nontarget_mib": sampled_bytes / 2**20,
        "full_hash": result["non_target_full_hash"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

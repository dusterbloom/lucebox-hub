#!/usr/bin/env python3
"""Benchmark lossless compression on real per-expert IQ1_S byte slices."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zstandard
from gguf import GGUFReader


EXPERT_COUNT = 896
COMPONENTS = ("down", "gate", "up")
LEVELS = (-5, 1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--sample-experts", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reader = GGUFReader(args.shard, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expert_indices = np.linspace(
        0, EXPERT_COUNT - 1, args.sample_experts, dtype=np.int64
    )
    samples: list[tuple[str, int, memoryview]] = []
    component_bytes: dict[str, int] = {}
    tensor_bytes: dict[str, int] = {}
    for component in COMPONENTS:
        name = f"blk.{args.layer}.ffn_{component}_exps.weight"
        tensor = tensors.get(name)
        if tensor is None:
            raise ValueError(f"missing tensor {name} in {args.shard}")
        raw = tensor.data.view(np.uint8).reshape(-1)
        if raw.size % EXPERT_COUNT:
            raise ValueError(f"tensor {name} is not expert-major divisible")
        stride = raw.size // EXPERT_COUNT
        component_bytes[component] = int(stride)
        tensor_bytes[component] = int(raw.size)
        for expert in expert_indices:
            begin = int(expert) * stride
            samples.append(
                (component, int(expert), memoryview(raw[begin : begin + stride]))
            )

    results: dict[str, dict[str, object]] = {}
    for level in LEVELS:
        compressor = zstandard.ZstdCompressor(level=level)
        decompressor = zstandard.ZstdDecompressor()
        compressed: list[tuple[str, int, bytes, memoryview]] = []
        original_bytes = 0
        compressed_bytes = 0
        start = time.perf_counter()
        for component, expert, source in samples:
            encoded = compressor.compress(source)
            compressed.append((component, expert, encoded, source))
            original_bytes += len(source)
            compressed_bytes += len(encoded)
        compression_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for component, expert, encoded, source in compressed:
            decoded = decompressor.decompress(encoded, max_output_size=len(source))
            if decoded != source:
                raise ValueError(
                    f"lossless verification failed for {component} expert {expert}"
                )
        decompression_seconds = time.perf_counter() - start
        ratio = compressed_bytes / original_bytes
        results[f"zstd_level_{level}"] = {
            "sample_original_bytes": original_bytes,
            "sample_compressed_bytes": compressed_bytes,
            "compressed_fraction": ratio,
            "space_saving_fraction": 1.0 - ratio,
            "compression_gib_per_second": (
                original_bytes / compression_seconds / (1 << 30)
            ),
            "decompression_gib_per_second": (
                original_bytes / decompression_seconds / (1 << 30)
            ),
            "projected_495_26_gib_pool_gib": 495.26 * ratio,
            "projected_8_844_gib_per_token": 8.844 * ratio,
            "verified_byte_exact": True,
        }

    result = {
        "schema": "kimi-k3-iq1s-lossless-compression-v1",
        "status": "EXPLORATORY",
        "shard": str(args.shard),
        "model_layer": args.layer,
        "expert_count": EXPERT_COUNT,
        "sample_experts": list(map(int, expert_indices)),
        "sample_components": list(COMPONENTS),
        "component_bytes_per_expert": component_bytes,
        "tensor_bytes": tensor_bytes,
        "framing": "one independent Zstandard frame per expert component",
        "levels": results,
        "warning": "Python throughput is indicative, not a production C++ decoder benchmark.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    best = min(results.values(), key=lambda value: value["compressed_fraction"])
    print(
        f"best-compressed-fraction={best['compressed_fraction']:.6f} "
        f"fast-decode-gib-s={results['zstd_level_-5']['decompression_gib_per_second']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create deterministic synthetic shards for validating the LSA toolchain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .dataset import DatasetMetadata, float_to_bf16_bits, write_shard
except ImportError:
    from dataset import DatasetMetadata, float_to_bf16_bits, write_shard


def make_shard(
    path: Path,
    *,
    seed: int,
    examples: int,
    blocks: int,
    hidden_size: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    rng = np.random.default_rng(seed)
    metadata = DatasetMetadata(
        model_fingerprint="synthetic-qwen35",
        hidden_size=hidden_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    keys = rng.normal(size=(blocks, kv_heads, head_dim)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True).clip(1e-6)
    hidden = rng.normal(size=(examples, hidden_size)).astype(np.float32)
    projection = rng.normal(
        scale=1 / np.sqrt(hidden_size),
        size=(hidden_size, kv_heads * head_dim),
    ).astype(np.float32)
    query = (hidden @ projection).reshape(examples, kv_heads, head_dim)
    query /= np.linalg.norm(query, axis=-1, keepdims=True).clip(1e-6)

    labels: list[np.ndarray] = []
    visible = np.empty(examples, dtype=np.int32)
    boundary = np.empty(examples, dtype=np.int32)
    offsets = [0]
    for row in range(examples):
        count = max(4, blocks - examples + row + 1)
        count = min(count, blocks)
        visible[row] = count
        boundary[row] = count * metadata.block_size
        score = np.einsum("hd,bhd->bh", query[row], keys[:count]).max(axis=1)
        mass = 1 / (1 + np.exp(-10 * (score - 0.25)))
        labels.append(mass.astype(np.float16))
        offsets.append(offsets[-1] + count)

    write_shard(
        path,
        metadata,
        block_keys=keys.astype(np.float16),
        query_hidden_bf16=float_to_bf16_bits(hidden),
        boundary_pos=boundary,
        visible_blocks=visible,
        label_offsets=np.asarray(offsets, dtype=np.int64),
        label_mass=np.concatenate(labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    args = parser.parse_args()
    make_shard(
        args.output,
        seed=args.seed,
        examples=args.examples,
        blocks=args.blocks,
        hidden_size=args.hidden_size,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )


if __name__ == "__main__":
    main()

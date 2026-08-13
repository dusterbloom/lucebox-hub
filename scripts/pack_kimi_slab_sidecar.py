#!/usr/bin/env python3
"""Repack one real Kimi expert layer into progressive expert/slab order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader


EXPERT_COUNT = 896
EXPERT_WIDTH = 3072
DIMENSION = 3584
SLAB_SIZE = 256
SLAB_COUNT = 12
BLOCK_ALIGNMENT = 4096
HEADER = struct.Struct("<8s8I5Q")
MAGIC = b"K3SLB001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    reader = GGUFReader(args.shard, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate = tensors[f"blk.{args.layer}.ffn_gate_exps.weight"]
    up = tensors[f"blk.{args.layer}.ffn_up_exps.weight"]
    down = tensors[f"blk.{args.layer}.ffn_down_exps.weight"]
    if gate.data.shape != (EXPERT_COUNT, EXPERT_WIDTH, 700):
        raise ValueError(f"unexpected gate layout: {gate.data.shape}")
    if up.data.shape != gate.data.shape:
        raise ValueError("up layout disagrees")
    if down.data.shape != (EXPERT_COUNT, DIMENSION, 600):
        raise ValueError(f"unexpected down layout: {down.data.shape}")
    with np.load(args.fit_state, allow_pickle=False) as state:
        importance = state["slab_expected_residual_norm"]
    if importance.shape != (EXPERT_COUNT, SLAB_COUNT):
        raise ValueError("fit-state slab importance shape disagrees")
    order = np.argsort(-importance, axis=1, kind="stable").astype("<u2")

    component_slab_bytes = SLAB_SIZE * 700
    down_slab_bytes = DIMENSION * 50
    if component_slab_bytes != down_slab_bytes:
        raise ValueError("Kimi slab components are not byte balanced")
    slab_bytes = 3 * component_slab_bytes
    record_bytes = SLAB_COUNT * slab_bytes
    if record_bytes % BLOCK_ALIGNMENT:
        raise ValueError("expert record is not direct-I/O aligned")
    index_offset = BLOCK_ALIGNMENT
    index_bytes = order.nbytes
    payload_offset = align(index_offset + index_bytes, BLOCK_ALIGNMENT)
    payload_bytes = EXPERT_COUNT * record_bytes
    file_bytes = payload_offset + payload_bytes

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    digest = hashlib.sha256()
    header = HEADER.pack(
        MAGIC,
        1,
        args.layer,
        EXPERT_COUNT,
        DIMENSION,
        EXPERT_WIDTH,
        SLAB_SIZE,
        SLAB_COUNT,
        BLOCK_ALIGNMENT,
        index_offset,
        index_bytes,
        payload_offset,
        slab_bytes,
        record_bytes,
    )
    with temporary.open("wb", buffering=0) as output:
        header_block = header + bytes(BLOCK_ALIGNMENT - len(header))
        output.write(header_block)
        digest.update(header_block)
        index_raw = order.tobytes(order="C")
        output.write(index_raw)
        digest.update(index_raw)
        padding = bytes(payload_offset - index_offset - index_bytes)
        output.write(padding)
        digest.update(padding)
        for expert in range(EXPERT_COUNT):
            for slab_raw in order[expert]:
                slab = int(slab_raw)
                begin = slab * SLAB_SIZE
                end = begin + SLAB_SIZE
                byte_begin = slab * 50
                byte_end = byte_begin + 50
                for raw in (
                    gate.data[expert, begin:end].tobytes(order="C"),
                    up.data[expert, begin:end].tobytes(order="C"),
                    down.data[expert, :, byte_begin:byte_end].tobytes(order="C"),
                ):
                    output.write(raw)
                    digest.update(raw)
            if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
                print(
                    f"[slab-pack] experts={expert + 1}/{EXPERT_COUNT} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != file_bytes:
        raise ValueError("sidecar length disagrees with its registered layout")
    temporary.replace(args.output)

    manifest = {
        "schema": "kimi-k3-progressive-slab-sidecar-v1",
        "status": "EXPERIMENTAL",
        "source_shard": str(args.shard),
        "source_shard_bytes": args.shard.stat().st_size,
        "fit_state": str(args.fit_state),
        "fit_state_sha256": sha256(args.fit_state),
        "output": str(args.output),
        "output_bytes": file_bytes,
        "output_sha256": digest.hexdigest(),
        "model_layer": args.layer,
        "expert_count": EXPERT_COUNT,
        "dimension": DIMENSION,
        "expert_width": EXPERT_WIDTH,
        "slab_size": SLAB_SIZE,
        "slabs_per_expert": SLAB_COUNT,
        "slab_bytes": slab_bytes,
        "record_bytes": record_bytes,
        "alignment": BLOCK_ALIGNMENT,
        "index_offset": index_offset,
        "index_bytes": index_bytes,
        "payload_offset": payload_offset,
        "payload_bytes": payload_bytes,
        "ordering": "descending calibration mean slab residual norm within expert",
        "quantized_weight_bytes_unchanged": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[slab-pack] bytes={file_bytes} sha256={digest.hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

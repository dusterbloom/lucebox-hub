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
HEADER_V1 = struct.Struct("<8s8I5Q")
HEADER_V2 = struct.Struct("<8s8I8Q")
MAGIC = b"K3SLB001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--gate-shard", type=Path)
    parser.add_argument("--up-shard", type=Path)
    parser.add_argument("--down-shard", type=Path)
    parser.add_argument(
        "--natural-order",
        action="store_true",
        help=(
            "store slabs in physical neuron order; the fit-state positional "
            "argument is ignored. This is intended only for the all-192 "
            "numerical control, where ordering cannot affect selection."
        ),
    )
    parser.add_argument(
        "--legacy-natural-v1",
        action="store_true",
        help=(
            "emit the registered version-1 natural-order header used by the "
            "uniform IQ1_S reference sidecars; requires --natural-order and "
            "three equal legacy component sizes"
        ),
    )
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
    if args.legacy_natural_v1 and not args.natural_order:
        raise ValueError("--legacy-natural-v1 requires --natural-order")
    started = time.monotonic()
    shard_paths = {
        "gate": args.gate_shard or args.shard,
        "up": args.up_shard or args.shard,
        "down": args.down_shard or args.shard,
    }
    readers: dict[Path, GGUFReader] = {}
    tensor_names = {
        "gate": f"blk.{args.layer}.ffn_gate_exps.weight",
        "up": f"blk.{args.layer}.ffn_up_exps.weight",
        "down": f"blk.{args.layer}.ffn_down_exps.weight",
    }
    selected_tensors: dict[str, object] = {}
    for component, shard_path in shard_paths.items():
        if shard_path not in readers:
            readers[shard_path] = GGUFReader(shard_path, "r")
        reader = readers[shard_path]
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        if tensor_names[component] not in tensors:
            raise KeyError(
                f"{tensor_names[component]} is absent from {shard_path}"
            )
        selected_tensors[component] = tensors[tensor_names[component]]
    gate = selected_tensors["gate"]
    up = selected_tensors["up"]
    down = selected_tensors["down"]
    if gate.data.shape[:2] != (EXPERT_COUNT, EXPERT_WIDTH):
        raise ValueError(f"unexpected gate layout: {gate.data.shape}")
    if up.data.shape[:2] != (EXPERT_COUNT, EXPERT_WIDTH):
        raise ValueError(f"unexpected up layout: {up.data.shape}")
    if down.data.shape[:2] != (EXPERT_COUNT, DIMENSION):
        raise ValueError(f"unexpected down layout: {down.data.shape}")
    if args.natural_order:
        order = np.broadcast_to(
            np.arange(SLAB_COUNT, dtype="<u2"),
            (EXPERT_COUNT, SLAB_COUNT),
        ).copy()
        fit_state_path = None
        fit_state_sha256 = None
        ordering = "natural neuron order (all-192 numerical control only)"
    else:
        with np.load(args.fit_state, allow_pickle=False) as state:
            importance = state["slab_expected_residual_norm"]
        if importance.shape != (EXPERT_COUNT, SLAB_COUNT):
            raise ValueError("fit-state slab importance shape disagrees")
        order = np.argsort(-importance, axis=1, kind="stable").astype("<u2")
        fit_state_path = str(args.fit_state)
        fit_state_sha256 = sha256(args.fit_state)
        ordering = (
            "descending calibration mean slab residual norm within expert"
        )

    gate_row_bytes = int(gate.data.shape[2])
    up_row_bytes = int(up.data.shape[2])
    down_row_bytes = int(down.data.shape[2])
    if down_row_bytes % SLAB_COUNT:
        raise ValueError("down tensor rows do not split into twelve byte blocks")
    gate_slab_bytes = SLAB_SIZE * gate_row_bytes
    up_slab_bytes = SLAB_SIZE * up_row_bytes
    down_slab_bytes = DIMENSION * (down_row_bytes // SLAB_COUNT)
    if args.legacy_natural_v1:
        legacy_component_bytes = SLAB_SIZE * 700
        if (
            gate_slab_bytes,
            up_slab_bytes,
            down_slab_bytes,
        ) != (legacy_component_bytes,) * 3:
            raise ValueError(
                "legacy natural v1 requires equal IQ1_S slab component sizes"
            )
    slab_bytes = gate_slab_bytes + up_slab_bytes + down_slab_bytes
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
    use_natural_v2 = args.natural_order and not args.legacy_natural_v1
    header_struct = HEADER_V2 if use_natural_v2 else HEADER_V1
    header_values = [
        MAGIC,
        2 if use_natural_v2 else 1,
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
    ]
    if use_natural_v2:
        header_values.extend(
            [gate_slab_bytes, up_slab_bytes, down_slab_bytes]
        )
    header = header_struct.pack(*header_values)
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
                down_block_bytes = down_row_bytes // SLAB_COUNT
                byte_begin = slab * down_block_bytes
                byte_end = byte_begin + down_block_bytes
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
        "source_shards": {
            component: {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            for component, path in shard_paths.items()
        },
        "fit_state": fit_state_path,
        "fit_state_sha256": fit_state_sha256,
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
        "gate_slab_bytes": gate_slab_bytes,
        "up_slab_bytes": up_slab_bytes,
        "down_slab_bytes": down_slab_bytes,
        "record_bytes": record_bytes,
        "alignment": BLOCK_ALIGNMENT,
        "index_offset": index_offset,
        "index_bytes": index_bytes,
        "payload_offset": payload_offset,
        "payload_bytes": payload_bytes,
        "ordering": ordering,
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

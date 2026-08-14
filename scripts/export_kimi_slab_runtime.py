#!/usr/bin/env python3
"""Export the registered Kimi layer-one calibration state for C++ runtime use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


MAGIC = b"K3AUX001"
HEADER = struct.Struct("<8s8I10Q")
ALIGNMENT = 4096
EXPERT_COUNT = 896
DIMENSION = 3584
SLAB_SIZE = 256
SLAB_COUNT = 12


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.fit_state, allow_pickle=False) as state:
        slab_means = np.asarray(state["slab_means"], dtype="<f4")
        importance = np.asarray(
            state["slab_expected_residual_norm"], dtype="<f4"
        )
        native_means = np.asarray(state["native_means"], dtype="<f4")
        native_importance = np.asarray(
            state["native_expected_norm"], dtype="<f4"
        )
    if slab_means.shape != (EXPERT_COUNT, SLAB_COUNT, DIMENSION):
        raise ValueError(f"unexpected slab_means shape {slab_means.shape}")
    if importance.shape != (EXPERT_COUNT, SLAB_COUNT):
        raise ValueError(f"unexpected slab importance shape {importance.shape}")
    if native_means.shape != (EXPERT_COUNT, DIMENSION):
        raise ValueError(f"unexpected native_means shape {native_means.shape}")
    if native_importance.shape != (EXPERT_COUNT,):
        raise ValueError(
            f"unexpected native importance shape {native_importance.shape}"
        )

    order = np.argsort(-importance, axis=1, kind="stable").astype("<u2")
    ordered_means = np.take_along_axis(
        slab_means, order[:, :, None].astype(np.int64), axis=1
    ).astype("<f4", copy=False)
    ordered_importance = np.take_along_axis(
        importance, order.astype(np.int64), axis=1
    ).astype("<f4", copy=False)

    arrays = [
        ("order", order),
        ("ordered_slab_means", ordered_means),
        ("ordered_residual_importance", ordered_importance),
        ("native_means", native_means),
        ("native_expected_norm", native_importance),
    ]
    offsets: list[int] = []
    sizes: list[int] = []
    cursor = ALIGNMENT
    for _, array in arrays:
        cursor = align(cursor)
        offsets.append(cursor)
        sizes.append(array.nbytes)
        cursor += array.nbytes

    header = HEADER.pack(
        MAGIC,
        1,
        args.layer,
        EXPERT_COUNT,
        DIMENSION,
        SLAB_SIZE,
        SLAB_COUNT,
        0,  # float32 statistics
        ALIGNMENT,
        *[value for pair in zip(offsets, sizes) for value in pair],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb", buffering=0) as output:
        output.write(header)
        output.write(bytes(ALIGNMENT - len(header)))
        position = ALIGNMENT
        for (_, array), offset in zip(arrays, offsets):
            output.write(bytes(offset - position))
            raw = array.tobytes(order="C")
            output.write(raw)
            position = offset + len(raw)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(args.output)

    metadata = {
        "schema": "kimi-k3-progressive-slab-runtime-v1",
        "model_layer": args.layer,
        "fit_state": str(args.fit_state),
        "fit_state_sha256": digest(args.fit_state),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": digest(args.output),
        "expert_count": EXPERT_COUNT,
        "dimension": DIMENSION,
        "slab_size": SLAB_SIZE,
        "slabs_per_expert": SLAB_COUNT,
        "ordering": "descending calibration mean slab residual norm within expert",
        "arrays": {
            name: {"offset": offset, "bytes": size, "dtype": str(array.dtype)}
            for (name, array), offset, size in zip(arrays, offsets, sizes)
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

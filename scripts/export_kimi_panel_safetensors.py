#!/usr/bin/env python3
"""Convert the resumable panel runner's float32 artifact to safetensors."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


HEADER = struct.Struct("<8sIiIIII")
MAGIC = b"K3FIT001"
ARRAY_NAMES = (
    "codeword_offset",
    "unweighted_offset",
    "unweighted_gain",
    "router_weighted_offset",
    "router_weighted_gain",
)


def load_panel(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    with path.open("rb") as source:
        raw_header = source.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("panel header is truncated")
        magic, version, layer, experts, dimension, arrays, reserved = (
            HEADER.unpack(raw_header)
        )
        if (
            magic != MAGIC
            or version != 1
            or layer < 0
            or experts == 0
            or dimension == 0
            or arrays != len(ARRAY_NAMES)
            or reserved != 0
        ):
            raise ValueError("panel header is invalid or unsupported")
        value_count = experts * dimension
        tensors: dict[str, np.ndarray] = {}
        for name in ARRAY_NAMES:
            data = np.fromfile(source, dtype="<f4", count=value_count)
            if data.size != value_count:
                raise ValueError(f"panel array {name} is truncated")
            data = data.reshape(experts, dimension)
            if not np.isfinite(data).all():
                raise ValueError(f"panel array {name} contains non-finite values")
            tensors[name] = data
        if source.read(1):
            raise ValueError("panel artifact contains trailing bytes")
    return {
        "version": version,
        "model_layer": layer,
        "expert_count": experts,
        "latent_dimension": dimension,
    }, tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    metadata, arrays = load_panel(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: torch.from_numpy(array.copy()).to(torch.bfloat16)
        for name, array in arrays.items()
    }
    save_file(
        tensors,
        str(args.output),
        metadata={
            "schema": "kimi-k3-diagonal-panel-v1",
            "model_layer": str(metadata["model_layer"]),
            "expert_count": str(metadata["expert_count"]),
            "latent_dimension": str(metadata["latent_dimension"]),
            "source_storage": "float32",
            "export_storage": "bfloat16",
        },
    )
    expected_values = (
        len(ARRAY_NAMES)
        * metadata["expert_count"]
        * metadata["latent_dimension"]
    )
    if not args.output.is_file() or args.output.stat().st_size <= 0:
        raise RuntimeError("safetensors export was not created")
    print(
        f"exported {expected_values} values for layer "
        f"{metadata['model_layer']} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

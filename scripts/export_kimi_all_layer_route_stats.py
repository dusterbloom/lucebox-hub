#!/usr/bin/env python3
"""Export compact native-route means/importances for the all-layer K3 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


FIRST_LAYER = 1
LAST_LAYER = 92
EXPERT_COUNT = 896
DIMENSION = 3584
ALIGNMENT = 4096
MAGIC = b"K3ROUTE1"
HEADER = struct.Struct("<8s6I4Q32s")


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_layer(fit_state: Path, output: Path, layer: int) -> dict[str, object]:
    fit_sha = sha256(fit_state)
    with np.load(fit_state, allow_pickle=False) as state:
        means = np.asarray(state["native_means"], dtype="<f4")
        importance = np.asarray(state["native_expected_norm"], dtype="<f4")
    if means.shape != (EXPERT_COUNT, DIMENSION) or not np.isfinite(means).all():
        raise ValueError(f"layer {layer}: invalid native means")
    if (
        importance.shape != (EXPERT_COUNT,)
        or not np.isfinite(importance).all()
        or np.any(importance < 0)
    ):
        raise ValueError(f"layer {layer}: invalid native importance")

    means_offset = ALIGNMENT
    importance_offset = align(means_offset + means.nbytes)
    header = HEADER.pack(
        MAGIC,
        1,
        layer,
        EXPERT_COUNT,
        DIMENSION,
        0,
        ALIGNMENT,
        means_offset,
        means.nbytes,
        importance_offset,
        importance.nbytes,
        bytes.fromhex(fit_sha),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb", buffering=0) as sink:
        first = header + bytes(means_offset - len(header))
        sink.write(first)
        digest.update(first)
        raw_means = means.tobytes(order="C")
        sink.write(raw_means)
        digest.update(raw_means)
        padding = bytes(importance_offset - means_offset - len(raw_means))
        sink.write(padding)
        digest.update(padding)
        raw_importance = importance.tobytes(order="C")
        sink.write(raw_importance)
        digest.update(raw_importance)
        sink.flush()
        os.fsync(sink.fileno())
    temporary.replace(output)
    return {
        "layer": layer,
        "fit_state": str(fit_state),
        "fit_state_sha256": fit_sha,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "native_means_offset": means_offset,
        "native_means_bytes": means.nbytes,
        "native_importance_offset": importance_offset,
        "native_importance_bytes": importance.nbytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_state_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--first-layer", type=int, default=FIRST_LAYER)
    parser.add_argument("--last-layer", type=int, default=LAST_LAYER)
    args = parser.parse_args()
    if not 1 <= args.first_layer <= args.last_layer <= LAST_LAYER:
        raise ValueError("layer range must be within 1..92")

    records = []
    for layer in range(args.first_layer, args.last_layer + 1):
        stem = f"kimi_layer{layer:02d}"
        fit_state = (
            args.fit_state_root / f"{stem}_neuron_slabs_calibration.npz"
        )
        if not fit_state.is_file():
            raise FileNotFoundError(fit_state)
        output = args.output_root / f"{stem}_route_stats.k3route"
        record = export_layer(fit_state, output, layer)
        manifest = output.with_suffix(".json")
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(
            f"[route-stats-export] layer={layer} bytes={record['output_bytes']}",
            flush=True,
        )

    aggregate = {
        "schema": "kimi-k3-all-layer-route-stats-v1",
        "status": "PILOT_2048_TOKEN_CALIBRATION",
        "quality_claim": "NONE",
        "first_layer": args.first_layer,
        "last_layer": args.last_layer,
        "layer_count": len(records),
        "total_bytes": sum(int(record["output_bytes"]) for record in records),
        "layers": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "all_layers_route_stats_manifest.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in aggregate.items() if key != "layers"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

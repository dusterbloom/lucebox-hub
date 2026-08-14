#!/usr/bin/env python3
"""Build resumable natural-order slab sidecars for all routed Kimi layers.

The artifacts are a numerical-control substrate, not a deployable ordering:
natural order is valid only at the all-192 budget, where every active slab is
read and the order cannot change which bytes are selected.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from gguf import GGUFReader


FIRST_ROUTED_LAYER = 1
LAST_ROUTED_LAYER = 92
SIDECAR_BYTES = 5_780_303_872


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--pack-script",
        type=Path,
        default=Path(__file__).with_name("pack_kimi_slab_sidecar.py"),
    )
    parser.add_argument("--first-layer", type=int, default=FIRST_ROUTED_LAYER)
    parser.add_argument("--last-layer", type=int, default=LAST_ROUTED_LAYER)
    return parser.parse_args()


def discover_tensor_shards(model_directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    shards = sorted(model_directory.glob("*.gguf"))
    if not shards:
        raise FileNotFoundError(f"no GGUF shards in {model_directory}")
    for shard in shards:
        reader = GGUFReader(shard, "r")
        names = {tensor.name for tensor in reader.tensors}
        for name in names:
            if not name.endswith("_exps.weight"):
                continue
            if name in result:
                raise ValueError(f"tensor {name} occurs in multiple shards")
            result[name] = shard
    expected = {
        f"blk.{layer}.ffn_{component}_exps.weight"
        for layer in range(FIRST_ROUTED_LAYER, LAST_ROUTED_LAYER + 1)
        for component in ("gate", "up", "down")
    }
    missing = sorted(expected - result.keys())
    if missing:
        raise ValueError(f"missing routed expert tensors {missing}")
    return result


def complete(output: Path, manifest: Path, layer: int) -> bool:
    if not output.is_file() or output.stat().st_size != SIDECAR_BYTES:
        return False
    if not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("model_layer") == layer
        and record.get("output_bytes") == SIDECAR_BYTES
        and record.get("ordering")
        == "natural neuron order (all-192 numerical control only)"
    )


def main() -> int:
    args = parse_args()
    if not (
        FIRST_ROUTED_LAYER <= args.first_layer <= args.last_layer <= LAST_ROUTED_LAYER
    ):
        raise ValueError("layer range must lie within routed model layers 1..92")
    tensor_shards = discover_tensor_shards(args.model_directory)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    pending = []
    for layer in range(args.first_layer, args.last_layer + 1):
        output = args.output_directory / f"kimi_layer{layer:02d}_natural_slabs.k3slab"
        manifest = args.output_directory / f"kimi_layer{layer:02d}_natural_slabs.json"
        if not complete(output, manifest, layer):
            pending.append((layer, output, manifest))
    required = len(pending) * SIDECAR_BYTES
    free = shutil.disk_usage(args.output_directory).free
    print(
        f"[all-slab-pack] pending={len(pending)} required={required} free={free}",
        flush=True,
    )
    if free < required + 64 * (1 << 30):
        raise OSError("insufficient free space for sidecars plus 64 GiB safety margin")

    for index, (layer, output, manifest) in enumerate(pending, start=1):
        component_shards = {
            component: tensor_shards[
                f"blk.{layer}.ffn_{component}_exps.weight"
            ]
            for component in ("gate", "up", "down")
        }
        command = [
            sys.executable,
            str(args.pack_script),
            str(component_shards["gate"]),
            "/dev/null",
            str(output),
            str(manifest),
            "--layer",
            str(layer),
            "--natural-order",
            "--gate-shard",
            str(component_shards["gate"]),
            "--up-shard",
            str(component_shards["up"]),
            "--down-shard",
            str(component_shards["down"]),
        ]
        print(
            f"[all-slab-pack] layer={layer} item={index}/{len(pending)} "
            "shards=" + ",".join(
                component_shards[name].name
                for name in ("gate", "up", "down")
            ),
            flush=True,
        )
        subprocess.run(command, check=True)

    aggregate = {
        "schema": "kimi-k3-all-layer-natural-slab-sidecars-v1",
        "status": "EXPERIMENTAL_NUMERICAL_CONTROL_ONLY",
        "model_directory": str(args.model_directory),
        "first_layer": args.first_layer,
        "last_layer": args.last_layer,
        "layer_count": args.last_layer - args.first_layer + 1,
        "sidecar_bytes_per_layer": SIDECAR_BYTES,
        "total_sidecar_bytes": (
            (args.last_layer - args.first_layer + 1) * SIDECAR_BYTES
        ),
        "path_pattern": str(
            args.output_directory / "kimi_layer%02d_natural_slabs.k3slab"
        ),
        "ordering": "natural neuron order",
        "valid_budget": 192,
    }
    (args.output_directory / "all_layers_manifest.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

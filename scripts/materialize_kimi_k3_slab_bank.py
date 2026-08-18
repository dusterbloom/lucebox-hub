#!/usr/bin/env python3
"""Materialize the Kimi K3 natural slab bank without a second full checkpoint.

The source GGUF shards are public and temporary.  Each layer is packed into a
hash-registered natural-order sidecar as soon as its component shards exist.
Only after the sidecar matches the registered reference are that layer's routed
tensor ranges hole-punched.  A temporary shard is removed once every routed
tensor it supplies has a verified sidecar.

The deployment root is marker-bound and every operation is resumable.  This is
intended for capacity-constrained Lucebox installs that receive the separately
verified sparse P32 core by another channel.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from gguf import GGUFReader


FIRST_ROUTED_LAYER = 1
LAST_ROUTED_LAYER = 92
DEFAULT_MINIMUM_FREE_BYTES = 32 << 30
MARKER_SCHEMA = "kimi-k3-streamed-sidecar-deployment-v1"
STATE_SCHEMA = "kimi-k3-streamed-sidecar-layer-v1"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    size: int


@dataclass(frozen=True)
class LayerSpec:
    layer: int
    output_name: str
    output_bytes: int
    output_sha256: str
    components: dict[str, SourceSpec]


@dataclass(frozen=True)
class TensorRange:
    component: str
    tensor_name: str
    source: Path
    offset: int
    length: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_reference(reference_directory: Path) -> tuple[list[LayerSpec], dict[str, SourceSpec]]:
    layers: list[LayerSpec] = []
    sources: dict[str, SourceSpec] = {}
    for layer in range(FIRST_ROUTED_LAYER, LAST_ROUTED_LAYER + 1):
        path = reference_directory / f"kimi_layer{layer:02d}_natural_slabs.json"
        record = json.loads(path.read_text())
        if record.get("model_layer") != layer:
            raise ValueError(f"reference layer mismatch in {path}")
        if record.get("ordering") != "natural neuron order (all-192 numerical control only)":
            raise ValueError(f"reference ordering mismatch in {path}")
        components: dict[str, SourceSpec] = {}
        for component in ("gate", "up", "down"):
            source_record = record["source_shards"][component]
            spec = SourceSpec(
                name=Path(source_record["path"]).name,
                size=int(source_record["bytes"]),
            )
            previous = sources.get(spec.name)
            if previous is not None and previous != spec:
                raise ValueError(f"inconsistent source registration for {spec.name}")
            sources[spec.name] = spec
            components[component] = spec
        layers.append(
            LayerSpec(
                layer=layer,
                output_name=f"kimi_layer{layer:02d}_natural_slabs.k3slab",
                output_bytes=int(record["output_bytes"]),
                output_sha256=str(record["output_sha256"]),
                components=components,
            )
        )
    return layers, sources


def ensure_root(root: Path, reference_directory: Path) -> dict[str, Path]:
    resolved = root.resolve()
    if resolved == Path("/") or resolved == Path.home().resolve():
        raise ValueError("refusing broad deployment root")
    marker = resolved / "deployment-marker.json"
    if resolved.exists() and not marker.is_file():
        if any(resolved.iterdir()):
            raise FileExistsError(f"refusing unmarked nonempty deployment root: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record.get("schema") != MARKER_SCHEMA:
            raise ValueError("deployment marker schema mismatch")
    else:
        atomic_json(
            marker,
            {
                "schema": MARKER_SCHEMA,
                "reference_directory": str(reference_directory.resolve()),
            },
        )
    paths = {
        "root": resolved,
        "source": resolved / "temporary-source",
        "sidecars": resolved / "natural-sidecars",
        "state": resolved / "state",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def source_receipt(state_directory: Path, name: str) -> Path:
    return state_directory / "downloads" / f"{name}.json"


def retired_receipt(state_directory: Path, name: str) -> Path:
    return state_directory / "retired" / f"{name}.json"


def download_source(
    source_directory: Path,
    state_directory: Path,
    spec: SourceSpec,
    base_url: str,
) -> None:
    destination = source_directory / spec.name
    receipt = source_receipt(state_directory, spec.name)
    retired = retired_receipt(state_directory, spec.name)
    if retired.is_file():
        if destination.exists():
            raise ValueError(f"retired source unexpectedly exists: {destination}")
        return
    if receipt.is_file():
        if not destination.is_file() or destination.stat().st_size != spec.size:
            raise ValueError(f"registered source is missing or truncated: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"unregistered source file: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and partial.stat().st_size > spec.size:
        raise ValueError(f"partial source exceeds registered size: {partial}")
    url = base_url.rstrip("/") + "/" + urllib.parse.quote(spec.name)
    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "8",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    print(f"[materialize] download {spec.name} bytes={spec.size}", flush=True)
    subprocess.run(command, check=True)
    if partial.stat().st_size != spec.size:
        raise ValueError(
            f"download size mismatch for {spec.name}: "
            f"{partial.stat().st_size} != {spec.size}"
        )
    with partial.open("rb") as source:
        os.fsync(source.fileno())
    os.replace(partial, destination)
    atomic_json(
        receipt,
        {"schema": MARKER_SCHEMA, "name": spec.name, "bytes": spec.size, "url": url},
    )


def layer_output_valid(spec: LayerSpec, sidecar_directory: Path, state_directory: Path) -> bool:
    output = sidecar_directory / spec.output_name
    receipt = state_directory / "layers" / f"layer{spec.layer:02d}.json"
    if not receipt.is_file():
        return False
    record = json.loads(receipt.read_text())
    return (
        record.get("schema") == STATE_SCHEMA
        and record.get("layer") == spec.layer
        and record.get("output_bytes") == spec.output_bytes
        and record.get("output_sha256") == spec.output_sha256
        and output.is_file()
        and output.stat().st_size == spec.output_bytes
    )


def tensor_ranges(spec: LayerSpec, source_directory: Path) -> list[TensorRange]:
    names = {
        "gate": f"blk.{spec.layer}.ffn_gate_exps.weight",
        "up": f"blk.{spec.layer}.ffn_up_exps.weight",
        "down": f"blk.{spec.layer}.ffn_down_exps.weight",
    }
    readers: dict[Path, GGUFReader] = {}
    result: list[TensorRange] = []
    try:
        for component in ("gate", "up", "down"):
            source = source_directory / spec.components[component].name
            if source not in readers:
                readers[source] = GGUFReader(source, "r")
            selected = None
            for tensor in readers[source].tensors:
                if tensor.name == names[component]:
                    selected = tensor
                    break
            if selected is None:
                raise KeyError(f"{names[component]} is absent from {source}")
            offset = int(selected.data_offset)
            length = int(selected.n_bytes)
            if offset < 0 or length <= 0 or offset + length > source.stat().st_size:
                raise ValueError(f"invalid tensor range for {names[component]}")
            result.append(
                TensorRange(component, names[component], source, offset, length)
            )
    finally:
        readers.clear()
    return result


def interior_is_hole(descriptor: int, offset: int, length: int) -> bool:
    block = 4096
    begin = ((offset + block - 1) // block) * block
    end = ((offset + length) // block) * block
    if begin >= end:
        return True
    try:
        data = os.lseek(descriptor, begin, os.SEEK_DATA)
    except OSError as error:
        if error.errno == errno.ENXIO:
            return True
        raise
    return data >= end


def range_is_punched(value: TensorRange) -> bool:
    descriptor = os.open(value.source, os.O_RDONLY)
    try:
        edge = min(4096, value.length)
        return (
            interior_is_hole(descriptor, value.offset, value.length)
            and not any(os.pread(descriptor, edge, value.offset))
            and not any(
                os.pread(descriptor, edge, value.offset + value.length - edge)
            )
        )
    finally:
        os.close(descriptor)


def punch_ranges(ranges: list[TensorRange]) -> tuple[int, int]:
    sources = {value.source for value in ranges}
    before = sum(source.stat().st_blocks * 512 for source in sources)
    newly_punched: list[TensorRange] = []
    for value in ranges:
        if range_is_punched(value):
            continue
        subprocess.run(
            [
                "fallocate",
                "--punch-hole",
                "--keep-size",
                "--offset",
                str(value.offset),
                "--length",
                str(value.length),
                str(value.source),
            ],
            check=True,
        )
        if not range_is_punched(value):
            raise ValueError(f"tensor range was not fully punched: {value.tensor_name}")
        newly_punched.append(value)
    for source in sources:
        descriptor = os.open(source, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    after = sum(source.stat().st_blocks * 512 for source in sources)
    expected = sum(value.length for value in newly_punched) - 2 * 4096 * len(
        newly_punched
    )
    reclaimed = before - after
    if reclaimed < max(0, expected):
        raise ValueError(f"hole punching reclaimed {reclaimed}, expected at least {expected}")
    return before, after


def pack_layer(
    spec: LayerSpec,
    source_directory: Path,
    sidecar_directory: Path,
    state_directory: Path,
    pack_script: Path,
    minimum_free_bytes: int,
) -> None:
    output = sidecar_directory / spec.output_name
    manifest = sidecar_directory / output.with_suffix(".json").name
    if output.is_file():
        if output.stat().st_size != spec.output_bytes or sha256(output) != spec.output_sha256:
            raise ValueError(f"existing sidecar disagrees with reference: {output}")
    else:
        free = shutil.disk_usage(sidecar_directory).free
        if free < spec.output_bytes + minimum_free_bytes:
            raise OSError(
                f"insufficient free space for layer {spec.layer}: "
                f"free={free} required={spec.output_bytes + minimum_free_bytes}"
            )
        sources = {
            component: source_directory / spec.components[component].name
            for component in ("gate", "up", "down")
        }
        command = [
            sys.executable,
            str(pack_script),
            str(sources["gate"]),
            "/dev/null",
            str(output),
            str(manifest),
            "--layer",
            str(spec.layer),
            "--natural-order",
            "--gate-shard",
            str(sources["gate"]),
            "--up-shard",
            str(sources["up"]),
            "--down-shard",
            str(sources["down"]),
        ]
        print(f"[materialize] pack layer={spec.layer}", flush=True)
        subprocess.run(command, check=True)
        if output.stat().st_size != spec.output_bytes:
            raise ValueError(f"sidecar size mismatch for layer {spec.layer}")
        actual = sha256(output)
        if actual != spec.output_sha256:
            raise ValueError(
                f"sidecar SHA mismatch for layer {spec.layer}: "
                f"{actual} != {spec.output_sha256}"
            )
    ranges = tensor_ranges(spec, source_directory)
    before, after = punch_ranges(ranges)
    atomic_json(
        state_directory / "layers" / f"layer{spec.layer:02d}.json",
        {
            "schema": STATE_SCHEMA,
            "layer": spec.layer,
            "output": str(output),
            "output_bytes": spec.output_bytes,
            "output_sha256": spec.output_sha256,
            "source_allocated_before": before,
            "source_allocated_after": after,
            "ranges": [
                {
                    "component": value.component,
                    "tensor_name": value.tensor_name,
                    "source": value.source.name,
                    "offset": value.offset,
                    "length": value.length,
                }
                for value in ranges
            ],
        },
    )


def retire_unused_sources(
    layers: list[LayerSpec],
    source_directory: Path,
    sidecar_directory: Path,
    state_directory: Path,
) -> None:
    by_source: dict[str, list[LayerSpec]] = {}
    for layer in layers:
        for source in set(value.name for value in layer.components.values()):
            by_source.setdefault(source, []).append(layer)
    for name, required_layers in by_source.items():
        retired = retired_receipt(state_directory, name)
        if retired.is_file():
            continue
        if not all(layer_output_valid(layer, sidecar_directory, state_directory) for layer in required_layers):
            continue
        source = source_directory / name
        allocated = source.stat().st_blocks * 512
        source.unlink()
        atomic_json(
            retired,
            {
                "schema": MARKER_SCHEMA,
                "name": name,
                "retired_allocated_bytes": allocated,
                "reason": "all referenced routed tensors have verified sidecars",
            },
        )
        print(f"[materialize] retired {name} allocated={allocated}", flush=True)


def process_ready_layers(
    layers: list[LayerSpec],
    paths: dict[str, Path],
    pack_script: Path,
    minimum_free_bytes: int,
) -> None:
    for spec in layers:
        if layer_output_valid(spec, paths["sidecars"], paths["state"]):
            continue
        required = [paths["source"] / value.name for value in spec.components.values()]
        if not all(path.is_file() for path in required):
            continue
        pack_layer(
            spec,
            paths["source"],
            paths["sidecars"],
            paths["state"],
            pack_script,
            minimum_free_bytes,
        )
        retire_unused_sources(layers, paths["source"], paths["sidecars"], paths["state"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment_root", type=Path)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("--download-base-url", required=True)
    parser.add_argument(
        "--pack-script",
        type=Path,
        default=Path(__file__).with_name("pack_kimi_slab_sidecar.py"),
    )
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=DEFAULT_MINIMUM_FREE_BYTES
    )
    args = parser.parse_args()

    layers, sources = load_reference(args.reference_directory)
    paths = ensure_root(args.deployment_root, args.reference_directory)
    for source in sorted(sources.values(), key=lambda value: value.name):
        download_source(
            paths["source"],
            paths["state"],
            source,
            args.download_base_url,
        )
        process_ready_layers(layers, paths, args.pack_script, args.minimum_free_bytes)
    process_ready_layers(layers, paths, args.pack_script, args.minimum_free_bytes)
    incomplete = [
        value.layer
        for value in layers
        if not layer_output_valid(value, paths["sidecars"], paths["state"])
    ]
    if incomplete:
        raise ValueError(f"incomplete sidecar layers: {incomplete}")
    retire_unused_sources(layers, paths["source"], paths["sidecars"], paths["state"])
    aggregate = {
        "schema": "kimi-k3-streamed-sidecar-bank-v1",
        "status": "COMPLETE",
        "layer_count": len(layers),
        "total_sidecar_bytes": sum(value.output_bytes for value in layers),
        "reference_directory": str(args.reference_directory.resolve()),
        "sidecar_directory": str(paths["sidecars"]),
        "temporary_sources_remaining": sorted(
            value.name for value in paths["source"].glob("*.gguf")
        ),
    }
    if aggregate["temporary_sources_remaining"]:
        raise ValueError("verified deployment still retains temporary GGUF shards")
    atomic_json(paths["sidecars"] / "all_layers_manifest.json", aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plan a selective Kimi-K3 KDA requantization without touching experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from gguf import GGMLQuantizationType, GGML_QUANT_SIZES, GGUFReader


TARGET_SUFFIXES = {
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ssm_g.weight",
    "attn_output.weight",
}
QTYPE_NAMES = {
    value: value.name.lower() for value in GGMLQuantizationType
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_paths(first: Path) -> list[Path]:
    match = re.match(r"^(.*-)[0-9]{5}(-of-[0-9]{5}\.gguf)$", first.name)
    if not match:
        return [first]
    paths = sorted(first.parent.glob(match.group(1) + "*" + match.group(2)))
    if not paths:
        raise ValueError("no GGUF shards found")
    return paths


def metadata_ints(reader: GGUFReader, key: str) -> list[int]:
    field = reader.fields[key]
    return [int(field.parts[index][0]) for index in field.data]


def parse_layers(raw: str | None, available: set[int]) -> set[int]:
    if raw is None:
        return set(available)
    selected: set[int] = set()
    for value in raw.split(","):
        if not value:
            raise ValueError("--layers contains an empty entry")
        layer = int(value)
        if layer not in available:
            raise ValueError(f"layer {layer} is not a recurrent KDA layer")
        selected.add(layer)
    if not selected:
        raise ValueError("--layers selected no recurrent KDA layers")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor_types", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--target-type", default="q4_k")
    parser.add_argument("--source-type", default="q6_k")
    parser.add_argument(
        "--layers",
        help="comma-separated recurrent model-layer IDs; default is all",
    )
    parser.add_argument(
        "--checksum-file",
        type=Path,
        default=Path(__file__).with_name("kimi_k3_ud_iq1s.sha256"),
    )
    args = parser.parse_args()

    paths = split_paths(args.model)
    first = GGUFReader(paths[0], "r")
    head_count_kv = metadata_ints(first, "kimi-k3.attention.head_count_kv")
    kda_layers = {index for index, value in enumerate(head_count_kv) if value == 0}
    selected_layers = parse_layers(args.layers, kda_layers)
    type_by_name = {value: key for key, value in QTYPE_NAMES.items()}
    if args.source_type not in type_by_name or args.target_type not in type_by_name:
        raise ValueError("unknown source or target quantization type")
    source_block, source_block_bytes = GGML_QUANT_SIZES[
        type_by_name[args.source_type]
    ]
    target_block, target_block_bytes = GGML_QUANT_SIZES[
        type_by_name[args.target_type]
    ]
    if source_block != target_block:
        raise ValueError("source and target quantization block sizes differ")

    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    targets: list[dict[str, object]] = []
    all_input_bytes = 0
    target_input_bytes = 0
    target_output_bytes = 0

    for path in paths:
        reader = GGUFReader(path, "r")
        for tensor in reader.tensors:
            if tensor.name in seen:
                raise ValueError(f"duplicate tensor name: {tensor.name}")
            seen.add(tensor.name)
            current = QTYPE_NAMES.get(tensor.tensor_type)
            if current is None:
                raise ValueError(f"unsupported qtype {tensor.tensor_type}")
            selected = False
            layer_match = re.match(r"^blk\.([0-9]+)\.(.+)$", tensor.name)
            if layer_match:
                layer = int(layer_match.group(1))
                selected = (
                    layer in selected_layers and
                    layer_match.group(2) in TARGET_SUFFIXES
                )
            desired = args.target_type if selected else current
            rows.append((f"^{re.escape(tensor.name)}$", desired))
            tensor_bytes = int(tensor.data.nbytes)
            all_input_bytes += tensor_bytes
            if selected:
                if current != args.source_type:
                    raise ValueError(
                        f"target tensor is {current}, expected "
                        f"{args.source_type}: {tensor.name}"
                    )
                if tensor.data.size % source_block_bytes:
                    raise ValueError(
                        f"source byte extent is not block aligned: {tensor.name}"
                    )
                output_bytes = (
                    tensor.data.size // source_block_bytes * target_block_bytes
                )
                target_input_bytes += tensor_bytes
                target_output_bytes += output_bytes
                targets.append({
                    "name": tensor.name,
                    "source_shard": str(path),
                    "source_type": current,
                    "target_type": args.target_type,
                    "input_bytes": tensor_bytes,
                    "projected_output_bytes": output_bytes,
                })

    expected_targets = len(selected_layers) * len(TARGET_SUFFIXES)
    if len(targets) != expected_targets:
        raise ValueError(
            f"expected {expected_targets} KDA targets, found {len(targets)}"
        )

    args.tensor_types.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_types.write_text(
        "\n".join(f"{pattern}={qtype}" for pattern, qtype in rows) + "\n"
    )
    checksums: dict[str, str] = {}
    for line in args.checksum_file.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[Path(name.lstrip("* ")).name] = digest
    if any(path.name not in checksums for path in paths):
        raise ValueError("checksum registry does not cover every source shard")

    manifest = {
        "schema": "kimi-k3-selective-kda-requant-plan-v1",
        "status": "PROJECTED_NOT_QUANTIZED",
        "model": str(args.model),
        "shards": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "registered_sha256": checksums[path.name],
            }
            for path in paths
        ],
        "checksum_file": str(args.checksum_file),
        "checksum_file_sha256": sha256(args.checksum_file),
        "tensor_count": len(rows),
        "kda_layers": sorted(kda_layers),
        "selected_layers": sorted(selected_layers),
        "target_suffixes": sorted(TARGET_SUFFIXES),
        "target_type": args.target_type,
        "source_type": args.source_type,
        "source_block_bytes": source_block_bytes,
        "target_block_bytes": target_block_bytes,
        "target_tensor_count": len(targets),
        "all_input_tensor_bytes": all_input_bytes,
        "target_input_bytes": target_input_bytes,
        "projected_target_output_bytes": target_output_bytes,
        "projected_saved_bytes": target_input_bytes - target_output_bytes,
        "tensor_type_file": str(args.tensor_types),
        "tensor_type_file_sha256": sha256(args.tensor_types),
        "targets": targets,
        "hard_gate": (
            "Every non-target tensor must remain byte-identical; routed expert "
            "and sidecar semantics are frozen. Quality and speed are OPEN."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "target_tensors": len(targets),
        "target_input_gib": target_input_bytes / 2**30,
        "target_output_gib": target_output_bytes / 2**30,
        "projected_saved_gib": (target_input_bytes - target_output_bytes) / 2**30,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

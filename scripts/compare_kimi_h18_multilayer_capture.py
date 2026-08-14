#!/usr/bin/env python3
"""Compare a multi-layer capture prefix with existing v1 single-layer captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8sIiIIQQII4Q")
RECORD = struct.Struct("<IB3sI")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_record(path: Path) -> dict[str, object]:
    with path.open("rb") as source:
        raw = source.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"truncated capture header: {path}")
        (magic, version, layer, latent_dim, top_k, sequence_count,
         token_count, latent_storage, weight_storage, *_reserved) = HEADER.unpack(raw)
        if magic != b"K3PNL001" or version != 1 or sequence_count < 1:
            raise ValueError(f"invalid capture header: {path}")
        raw = source.read(RECORD.size)
        if len(raw) != RECORD.size:
            raise ValueError(f"truncated record header: {path}")
        id_bytes, split, _reserved_record, record_tokens = RECORD.unpack(raw)
        identifier = source.read(id_bytes).decode("utf-8")
        tokens = np.frombuffer(
            source.read(record_tokens * 4), dtype="<i4").copy()
        latent = np.frombuffer(
            source.read(record_tokens * latent_dim * 2), dtype="<u2").copy()
        expert_ids = np.frombuffer(
            source.read(record_tokens * top_k * 4), dtype="<i4").copy()
        weights = np.frombuffer(
            source.read(record_tokens * top_k * 4), dtype="<f4").copy()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "layer": layer,
        "latent_dim": latent_dim,
        "top_k": top_k,
        "sequence_count": sequence_count,
        "token_count": token_count,
        "split": split,
        "id": identifier,
        "record_tokens": record_tokens,
        "tokens": tokens,
        "latent": latent,
        "expert_ids": expert_ids,
        "weights": weights,
    }


def compare(candidate_path: Path, reference_path: Path) -> dict[str, object]:
    candidate = first_record(candidate_path)
    reference = first_record(reference_path)
    prefix_tokens = int(candidate["record_tokens"])
    latent_values = prefix_tokens * int(candidate["latent_dim"])
    route_values = prefix_tokens * int(candidate["top_k"])
    candidate_latent = candidate["latent"]
    reference_latent = reference["latent"][:latent_values]
    candidate_weights = candidate["weights"]
    reference_weights = reference["weights"][:route_values]
    latent_left = (candidate_latent.astype(np.uint32) << 16).view(np.float32)
    latent_right = (reference_latent.astype(np.uint32) << 16).view(np.float32)
    latent_difference = latent_left.astype(np.float64) - latent_right.astype(np.float64)
    weight_difference = (
        candidate_weights.astype(np.float64) - reference_weights.astype(np.float64)
    )
    return {
        "layer": int(candidate["layer"]),
        "candidate": {"path": candidate["path"], "sha256": candidate["sha256"]},
        "reference": {"path": reference["path"], "sha256": reference["sha256"]},
        "prefix_tokens": prefix_tokens,
        "sequence_id_equal": candidate["id"] == reference["id"],
        "split_equal": candidate["split"] == reference["split"],
        "token_ids_bit_equal": bool(np.array_equal(
            candidate["tokens"], reference["tokens"][:prefix_tokens])),
        "latent_bf16_bit_equal": bool(np.array_equal(
            candidate_latent, reference_latent)),
        "latent_max_abs": float(np.max(np.abs(latent_difference), initial=0.0)),
        "expert_ids_bit_equal": bool(np.array_equal(
            candidate["expert_ids"], reference["expert_ids"][:route_values])),
        "router_weights_bit_equal": bool(np.array_equal(
            candidate_weights.view(np.uint32), reference_weights.view(np.uint32))),
        "router_weights_max_abs": float(
            np.max(np.abs(weight_difference), initial=0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("multi_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--layer1-reference", type=Path,
        default=Path("/mnt/kimi-k3/captures/kimi_layer01_10000.bin"))
    parser.add_argument(
        "--layer12-reference", type=Path,
        default=Path("/mnt/kimi-k3/captures/kimi_layer12_10000.bin"))
    args = parser.parse_args()
    comparisons = [
        compare(args.multi_root / "kimi_layer01_8.bin", args.layer1_reference),
        compare(args.multi_root / "kimi_layer12_8.bin", args.layer12_reference),
    ]
    gates = {
        "token_ids": all(row["token_ids_bit_equal"] for row in comparisons),
        "latent_bf16": all(row["latent_bf16_bit_equal"] for row in comparisons),
        "expert_ids": all(row["expert_ids_bit_equal"] for row in comparisons),
        "router_weights": all(
            row["router_weights_bit_equal"] for row in comparisons),
    }
    result = {
        "schema": "kimi-k3-h18-multilayer-capture-comparison-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate one completed expert-response/teacher layer and write a receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kimi_h23_capture_chunks", ROOT / "scripts/kimi_h23_capture_chunks.py"
)
CAPTURE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CAPTURE)

TEACHER = struct.Struct("<8sIiIIQQ2Q")
RANK = struct.Struct("<8sIiIIQII3Q")
PANEL = struct.Struct("<8sIiIIII")
RESPONSE = struct.Struct("<8sIiiIQII2Q")
DIMENSION = 3584
EXPERTS = 896
TOP_K = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def header(path: Path, layout: struct.Struct) -> tuple:
    with path.open("rb") as source:
        raw = source.read(layout.size)
    if len(raw) != layout.size:
        raise ValueError(f"truncated header: {path}")
    return layout.unpack(raw)


def validate(layer: int, capture: Path, responses: Path, state: Path, prefix: Path) -> dict[str, object]:
    capture_info = CAPTURE.inspect_capture(capture, layer)
    token_count = capture_info["token_count"]
    sequence_count = capture_info["sequence_count"]
    result_path = Path(str(prefix) + ".json")
    teacher_path = Path(str(prefix) + ".teacher.f32")
    rank_path = Path(str(prefix) + ".validation_by_rank.f32")
    panel_path = Path(str(prefix) + ".panel.f32")
    result = json.loads(result_path.read_text())
    if (
        result.get("schema") != "kimi-k3-layer-panel-fit-v1"
        or result.get("model_layer") != layer
        or Path(result.get("capture_path", "")) != capture
        or result.get("calibration_tokens", 0) + result.get("validation_tokens", 0) != token_count
    ):
        raise ValueError(f"fit result disagrees at layer {layer}")

    teacher = header(teacher_path, TEACHER)
    if (
        teacher[0] != b"K3TGT001"
        or teacher[1] != 1
        or teacher[2] != layer
        or teacher[3] != DIMENSION
        or teacher[4] != 0
        or teacher[5] != sequence_count
        or teacher[6] != token_count
        or any(teacher[7:])
        or teacher_path.stat().st_size != TEACHER.size + token_count * DIMENSION * 4
    ):
        raise ValueError(f"teacher artifact disagrees at layer {layer}")
    rank = header(rank_path, RANK)
    validation_tokens = result["validation_tokens"]
    if (
        rank[0] != b"K3RNK001"
        or rank[1] != 1
        or rank[2] != layer
        or rank[3] != DIMENSION
        or rank[4] != TOP_K
        or rank[5] != validation_tokens
        or rank[6] != 0
        or rank[7] != 0
        or any(rank[8:])
        or rank_path.stat().st_size != RANK.size + validation_tokens * TOP_K * DIMENSION * 4
    ):
        raise ValueError(f"rank teacher artifact disagrees at layer {layer}")
    panel = header(panel_path, PANEL)
    if (
        panel[:7] != (b"K3FIT001", 1, layer, EXPERTS, DIMENSION, 5, 0)
        or panel_path.stat().st_size != PANEL.size + EXPERTS * DIMENSION * 5 * 4
    ):
        raise ValueError(f"panel artifact disagrees at layer {layer}")

    response_bytes = 0
    route_count = 0
    for expert in range(EXPERTS):
        path = responses / f"expert_{expert:04d}.responses.f32"
        values = header(path, RESPONSE)
        routes = values[5]
        if (
            values[0] != b"K3RSP001"
            or values[1] != 1
            or values[2] != layer
            or values[3] != expert
            or values[4] != DIMENSION
            or values[6] != 0
            or values[7] != 0
            or any(values[8:])
            or path.stat().st_size != RESPONSE.size + routes * (16 + DIMENSION * 4)
        ):
            raise ValueError(f"expert response disagrees: {path}")
        route_count += routes
        response_bytes += path.stat().st_size
    if route_count != token_count * TOP_K:
        raise ValueError(f"response route total disagrees at layer {layer}")

    stats = list(state.glob("expert_*.stats"))
    if len(stats) != EXPERTS:
        raise ValueError(f"fit-state expert count disagrees at layer {layer}")
    return {
        "schema": "kimi-k3-h23-fit-layer-receipt-v1",
        "layer": layer,
        "capture": str(capture),
        "capture_sha256": sha256(capture),
        "capture_tokens": token_count,
        "capture_sequences": sequence_count,
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "teacher": str(teacher_path),
        "teacher_sha256": sha256(teacher_path),
        "validation_by_rank": str(rank_path),
        "validation_by_rank_bytes": rank_path.stat().st_size,
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "responses": str(responses),
        "response_files": EXPERTS,
        "response_routes": route_count,
        "response_bytes": response_bytes,
        "fit_state": str(state),
        "fit_state_files": len(stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        raise ValueError("layer must be in 1..92")
    value = validate(args.layer, args.capture, args.responses, args.state, args.prefix)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(args.receipt)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

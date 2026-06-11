"""Reader and converter for luce.lsa.qwen35.raw.v1 capture directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .dataset import DatasetMetadata, write_shard
    from .oracle import cross_layer_oracle, layer_block_attention_mass
except ImportError:
    from dataset import DatasetMetadata, write_shard
    from oracle import cross_layer_oracle, layer_block_attention_mass

RAW_SCHEMA = "luce.lsa.qwen35.raw.v1"


@dataclass(frozen=True)
class RawCapture:
    path: Path
    manifest: dict
    boundary_pos: np.memmap
    query_hidden_bf16: np.memmap
    key_pre: np.memmap
    key_post: dict[int, np.memmap]
    query_post: dict[int, np.memmap]


def _mapped(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes, found {path.stat().st_size}"
        )
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _fnv1a64(path: Path) -> str:
    value = 14695981039346656037
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            for byte in chunk:
                value ^= byte
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def load_raw_capture(path: Path, *, verify_checksums: bool = False) -> RawCapture:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != RAW_SCHEMA:
        raise ValueError(f"unsupported raw schema: {manifest.get('schema')!r}")
    if manifest.get("endianness") != "little":
        raise ValueError("only little-endian raw captures are supported")

    tokens = int(manifest["tokens"])
    examples = int(manifest["examples"])
    hidden_size = int(manifest["hidden_size"])
    kv_heads = int(manifest["kv_heads"])
    query_heads = int(manifest["query_heads"])
    head_dim = int(manifest["head_dim"])
    horizon = int(manifest["lookahead_horizon"])
    layers = tuple(int(layer) for layer in manifest["oracle_layers"])
    if min(tokens, hidden_size, kv_heads, query_heads, head_dim, horizon) <= 0:
        raise ValueError("raw capture geometry is invalid")
    chunk_path = path / "chunk_tokens.i32"
    if chunk_path.stat().st_size % np.dtype("<i4").itemsize:
        raise ValueError("chunk token stream is truncated")
    chunk_tokens = np.fromfile(chunk_path, dtype="<i4")
    if np.any(chunk_tokens <= 0) or int(chunk_tokens.sum()) != tokens:
        raise ValueError("chunk token stream does not cover the capture")
    if verify_checksums:
        for name, properties in manifest.get("files", {}).items():
            if _fnv1a64(path / name) != properties["fnv1a64"]:
                raise ValueError(f"{name}: checksum mismatch")

    boundary = _mapped(path / "boundary_pos.i32", "<i4", (examples,))
    hidden = _mapped(
        path / "query_hidden.bf16", "<u2", (examples, hidden_size)
    )
    key_pre = _mapped(
        path / "key_pre.f16", "<f2", (tokens, kv_heads, head_dim)
    )
    key_post = {
        layer: _mapped(
            path / f"layer_{layer:02d}.key_post.f16",
            "<f2",
            (tokens, kv_heads, head_dim),
        )
        for layer in layers
    }
    query_post = {
        layer: _mapped(
            path / f"layer_{layer:02d}.query_post.f16",
            "<f2",
            (examples, horizon, query_heads, head_dim),
        )
        for layer in layers
    }
    if np.any(boundary < 0) or np.any(boundary + horizon > tokens):
        raise ValueError("raw capture boundary is outside the token stream")
    if boundary.size > 1 and np.any(np.diff(boundary) <= 0):
        raise ValueError("raw capture boundaries must be strictly increasing")
    return RawCapture(
        path=path,
        manifest=manifest,
        boundary_pos=boundary,
        query_hidden_bf16=hidden,
        key_pre=key_pre,
        key_post=key_post,
        query_post=query_post,
    )


def build_block_keys(
    capture: RawCapture, *, sink_tokens: int = 64
) -> np.ndarray:
    block = int(capture.manifest["block_size"])
    complete = int(capture.manifest["tokens"]) // block
    first = (sink_tokens + block - 1) // block
    source = np.asarray(capture.key_pre[: complete * block], dtype=np.float32)
    pooled = source.reshape(
        complete, block, source.shape[1], source.shape[2]
    ).mean(axis=1)
    norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
    pooled /= np.maximum(norm, 1e-12)
    return pooled[first:].astype(np.float16)


def convert_raw_capture(
    capture: RawCapture,
    output: Path,
    *,
    device: str = "cpu",
    sink_tokens: int = 64,
    recent_tokens: int = 8192,
) -> None:
    manifest = capture.manifest
    block = int(manifest["block_size"])
    horizon = int(manifest["lookahead_horizon"])
    layers = tuple(int(layer) for layer in manifest["oracle_layers"])
    block_keys = build_block_keys(capture, sink_tokens=sink_tokens)
    device_obj = torch.device(device)

    visible_blocks: list[int] = []
    label_rows: list[np.ndarray] = []
    for row, boundary_value in enumerate(capture.boundary_pos):
        boundary = int(boundary_value)
        key_end = boundary + horizon
        query_positions = torch.arange(
            boundary, key_end, dtype=torch.int64, device=device_obj
        )
        key_positions = torch.arange(
            key_end, dtype=torch.int64, device=device_obj
        )
        layer_rows = []
        for layer in layers:
            query = torch.from_numpy(
                np.asarray(capture.query_post[layer][row], dtype=np.float32)
            ).to(device_obj)
            key = torch.from_numpy(
                np.asarray(capture.key_post[layer][:key_end], dtype=np.float32)
            ).to(device_obj)
            layer_rows.append(
                layer_block_attention_mass(
                    query,
                    key,
                    query_positions,
                    key_positions,
                    block_size=block,
                    sink_tokens=sink_tokens,
                    recent_tokens=recent_tokens,
                    boundary_position=boundary,
                )
            )
        result = cross_layer_oracle(torch.stack(layer_rows, dim=0))
        labels = result.label_mass.detach().cpu().numpy().astype(np.float16)
        visible_blocks.append(labels.size)
        label_rows.append(labels)

    offsets = np.zeros(len(label_rows) + 1, dtype=np.int64)
    for index, labels in enumerate(label_rows):
        offsets[index + 1] = offsets[index] + labels.size
    label_mass = (
        np.concatenate(label_rows)
        if label_rows
        else np.empty((0,), dtype=np.float16)
    )
    metadata = DatasetMetadata(
        model_fingerprint=str(manifest["model_fingerprint"]),
        hidden_size=int(manifest["hidden_size"]),
        kv_heads=int(manifest["kv_heads"]),
        head_dim=int(manifest["head_dim"]),
        block_size=block,
        retrieval_interval=block,
        lookahead_horizon=horizon,
        oracle_layers=layers,
    )
    write_shard(
        output,
        metadata,
        block_keys=block_keys,
        query_hidden_bf16=np.asarray(capture.query_hidden_bf16).copy(),
        boundary_pos=np.asarray(capture.boundary_pos).copy(),
        visible_blocks=np.asarray(visible_blocks, dtype=np.int32),
        label_offsets=offsets,
        label_mass=label_mass,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-tokens", type=int, default=8192)
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    convert_raw_capture(
        load_raw_capture(args.raw, verify_checksums=args.verify_checksums),
        args.output,
        device=args.device,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
    )


if __name__ == "__main__":
    main()

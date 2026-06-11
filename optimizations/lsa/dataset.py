"""Versioned NPZ training-shard contract for the Qwen3.5 LSA encoder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

SCHEMA = "luce.lsa.qwen35.npz.v1"


@dataclass(frozen=True)
class DatasetMetadata:
    schema: str = SCHEMA
    model_fingerprint: str = "unknown"
    hidden_size: int = 5120
    kv_heads: int = 4
    head_dim: int = 256
    block_size: int = 64
    retrieval_interval: int = 64
    lookahead_horizon: int = 64
    hidden_tap: str = "layer46.post_ffn"
    key_tap: str = "layer47.k_norm.pre_rope"
    oracle_layers: tuple[int, ...] = tuple(range(3, 64, 4))

    @property
    def key_size(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class Shard:
    path: Path
    metadata: DatasetMetadata
    block_keys: np.ndarray
    query_hidden_bf16: np.ndarray
    boundary_pos: np.ndarray
    visible_blocks: np.ndarray
    label_offsets: np.ndarray
    label_mass: np.ndarray

    def example_count(self) -> int:
        return int(self.query_hidden_bf16.shape[0])


def _metadata_array(metadata: DatasetMetadata) -> np.ndarray:
    payload = json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":")).encode()
    return np.frombuffer(payload, dtype=np.uint8).copy()


def _parse_metadata(value: np.ndarray) -> DatasetMetadata:
    if value.dtype != np.uint8 or value.ndim != 1:
        raise ValueError("metadata_json must be a one-dimensional uint8 array")
    raw = json.loads(value.tobytes().decode())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"unsupported LSA dataset schema: {raw.get('schema')!r}")
    raw["oracle_layers"] = tuple(raw["oracle_layers"])
    return DatasetMetadata(**raw)


def float_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))
    return tensor.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


def bf16_bits_to_float(value: np.ndarray) -> torch.Tensor:
    bits = np.asarray(value, dtype=np.uint16)
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).float()


def write_shard(
    path: Path,
    metadata: DatasetMetadata,
    *,
    block_keys: np.ndarray,
    query_hidden_bf16: np.ndarray,
    boundary_pos: np.ndarray,
    visible_blocks: np.ndarray,
    label_offsets: np.ndarray,
    label_mass: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata_json=_metadata_array(metadata),
        block_keys=np.asarray(block_keys, dtype=np.float16),
        query_hidden_bf16=np.asarray(query_hidden_bf16, dtype=np.uint16),
        boundary_pos=np.asarray(boundary_pos, dtype=np.int32),
        visible_blocks=np.asarray(visible_blocks, dtype=np.int32),
        label_offsets=np.asarray(label_offsets, dtype=np.int64),
        label_mass=np.asarray(label_mass, dtype=np.float16),
    )
    load_shard.cache_clear()
    load_shard(path)


@lru_cache(maxsize=8)
def load_shard(path: Path) -> Shard:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "metadata_json",
            "block_keys",
            "query_hidden_bf16",
            "boundary_pos",
            "visible_blocks",
            "label_offsets",
            "label_mass",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays: {sorted(missing)}")
        metadata = _parse_metadata(data["metadata_json"])
        shard = Shard(
            path=path,
            metadata=metadata,
            block_keys=data["block_keys"].copy(),
            query_hidden_bf16=data["query_hidden_bf16"].copy(),
            boundary_pos=data["boundary_pos"].copy(),
            visible_blocks=data["visible_blocks"].copy(),
            label_offsets=data["label_offsets"].copy(),
            label_mass=data["label_mass"].copy(),
        )
    validate_shard(shard)
    return shard


def validate_shard(shard: Shard) -> None:
    meta = shard.metadata
    keys = shard.block_keys
    hidden = shard.query_hidden_bf16
    examples = hidden.shape[0] if hidden.ndim == 2 else -1

    if keys.dtype != np.float16 or keys.ndim != 3:
        raise ValueError(f"{shard.path}: block_keys must be float16 [blocks, heads, dim]")
    if keys.shape[1:] != (meta.kv_heads, meta.head_dim):
        raise ValueError(f"{shard.path}: block key geometry does not match metadata")
    if hidden.dtype != np.uint16 or hidden.ndim != 2 or hidden.shape[1] != meta.hidden_size:
        raise ValueError(f"{shard.path}: query_hidden_bf16 geometry does not match metadata")
    if shard.boundary_pos.shape != (examples,) or shard.boundary_pos.dtype != np.int32:
        raise ValueError(f"{shard.path}: boundary_pos must be int32 [examples]")
    if shard.visible_blocks.shape != (examples,) or shard.visible_blocks.dtype != np.int32:
        raise ValueError(f"{shard.path}: visible_blocks must be int32 [examples]")
    if shard.label_offsets.shape != (examples + 1,) or shard.label_offsets.dtype != np.int64:
        raise ValueError(f"{shard.path}: label_offsets must be int64 [examples + 1]")
    if shard.label_mass.dtype != np.float16 or shard.label_mass.ndim != 1:
        raise ValueError(f"{shard.path}: label_mass must be float16 [labels]")
    if shard.label_offsets[0] != 0 or shard.label_offsets[-1] != shard.label_mass.size:
        raise ValueError(f"{shard.path}: label offsets do not cover label_mass")
    if np.any(np.diff(shard.label_offsets) != shard.visible_blocks):
        raise ValueError(f"{shard.path}: each label row must match visible_blocks")
    if np.any(shard.visible_blocks < 0) or np.any(shard.visible_blocks > keys.shape[0]):
        raise ValueError(f"{shard.path}: visible_blocks is outside the key catalog")
    if not np.isfinite(keys).all() or not np.isfinite(shard.label_mass).all():
        raise ValueError(f"{shard.path}: shard contains non-finite values")
    if np.any(shard.label_mass < 0) or np.any(shard.label_mass > 1):
        raise ValueError(f"{shard.path}: label mass must be in [0, 1]")


def discover_shards(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            out.extend(sorted(path.glob("*.npz")))
        elif path.suffix == ".npz":
            out.append(path)
    if not out:
        raise ValueError("no LSA .npz shards found")
    return out


class LsaExampleDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = discover_shards(paths)
        self.index: list[tuple[Path, int]] = []
        self.metadata = load_shard(self.paths[0]).metadata
        for path in self.paths:
            shard = load_shard(path)
            if shard.metadata != self.metadata:
                raise ValueError(f"{path}: metadata does not match the first shard")
            self.index.extend((path, i) for i in range(shard.example_count()))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, row = self.index[index]
        shard = load_shard(path)
        visible = int(shard.visible_blocks[row])
        begin = int(shard.label_offsets[row])
        end = int(shard.label_offsets[row + 1])
        return {
            "hidden": bf16_bits_to_float(shard.query_hidden_bf16[row]),
            "keys": torch.from_numpy(shard.block_keys[:visible].copy()).float(),
            "target": torch.from_numpy(shard.label_mass[begin:end].copy()).float(),
            "boundary_pos": int(shard.boundary_pos[row]),
            "source": str(path),
        }

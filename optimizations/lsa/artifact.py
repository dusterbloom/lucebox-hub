"""Versioned encoder artifact shared by training and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from .dataset import DatasetMetadata
    from .model import CompactQwen35Encoder
except ImportError:
    from dataset import DatasetMetadata
    from model import CompactQwen35Encoder

ENCODER_SCHEMA = "luce.lsa.qwen35.encoder.v1"
WEIGHT_NAME = "encoder.f16.bin"
MANIFEST_NAME = "encoder.json"


def fnv1a64_bytes(value: bytes) -> str:
    checksum = 14695981039346656037
    for byte in value:
        checksum ^= byte
        checksum = (checksum * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{checksum:016x}"


def write_encoder_artifact(
    directory: Path,
    model: CompactQwen35Encoder,
    metadata: DatasetMetadata,
) -> dict[str, object]:
    if (
        metadata.hidden_size != model.hidden_size
        or metadata.kv_heads != model.kv_heads
        or metadata.head_dim != model.head_dim
    ):
        raise ValueError("encoder geometry does not match dataset metadata")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    down = (
        state["down.weight"]
        .detach()
        .cpu()
        .to(torch.float16)
        .numpy()
        .astype("<f2", copy=False)
    )
    up = (
        state["up.weight"]
        .detach()
        .cpu()
        .to(torch.float16)
        .numpy()
        .astype("<f2", copy=False)
    )
    packed = down.tobytes(order="C") + up.tobytes(order="C")

    weight_tmp = directory / f"{WEIGHT_NAME}.tmp"
    manifest_tmp = directory / f"{MANIFEST_NAME}.tmp"
    weight_tmp.write_bytes(packed)

    config: dict[str, object] = {
        "schema": ENCODER_SCHEMA,
        "dataset": asdict(metadata),
        "rank": model.rank,
        "score_temperature": model.score_temperature,
        "decision_threshold": model.decision_threshold,
        "logit_scale": model.logit_scale,
        "parameters": model.parameter_count(),
        "weight_file": {
            "name": WEIGHT_NAME,
            "dtype": "float16-le",
            "fnv1a64": fnv1a64_bytes(packed),
            "layout": [
                {
                    "name": "down.weight",
                    "shape": list(down.shape),
                    "offset_bytes": 0,
                },
                {
                    "name": "up.weight",
                    "shape": list(up.shape),
                    "offset_bytes": down.nbytes,
                },
            ],
            "size_bytes": len(packed),
        },
    }
    manifest_tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    weight_tmp.replace(directory / WEIGHT_NAME)
    manifest_tmp.replace(directory / MANIFEST_NAME)
    return config


def load_encoder_artifact(
    directory: Path, device: torch.device
) -> CompactQwen35Encoder:
    directory = Path(directory)
    config = json.loads((directory / MANIFEST_NAME).read_text())
    if config.get("schema") != ENCODER_SCHEMA:
        raise ValueError(f"unsupported encoder schema: {config.get('schema')!r}")
    dataset = config["dataset"]
    hidden_size = int(dataset["hidden_size"])
    rank = int(config["rank"])
    kv_heads = int(dataset["kv_heads"])
    head_dim = int(dataset["head_dim"])
    model = CompactQwen35Encoder(
        hidden_size=hidden_size,
        rank=rank,
        kv_heads=kv_heads,
        head_dim=head_dim,
        score_temperature=config["score_temperature"],
        decision_threshold=config["decision_threshold"],
        logit_scale=config["logit_scale"],
    )
    weight = config["weight_file"]
    if weight.get("dtype") != "float16-le":
        raise ValueError(f"unsupported encoder dtype: {weight.get('dtype')!r}")
    weight_name = Path(weight["name"])
    if weight_name.is_absolute() or weight_name.name != str(weight_name):
        raise ValueError("encoder weight path must be a local file name")
    packed = (directory / weight_name).read_bytes()
    if len(packed) != weight["size_bytes"]:
        raise ValueError("encoder weight size does not match manifest")
    if fnv1a64_bytes(packed) != weight["fnv1a64"]:
        raise ValueError("encoder weight checksum does not match manifest")
    down_count = rank * hidden_size
    up_count = kv_heads * head_dim * rank
    expected_size = (down_count + up_count) * np.dtype("<f2").itemsize
    layout = weight.get("layout")
    expected_layout = [
        {
            "name": "down.weight",
            "shape": [rank, hidden_size],
            "offset_bytes": 0,
        },
        {
            "name": "up.weight",
            "shape": [kv_heads * head_dim, rank],
            "offset_bytes": down_count * np.dtype("<f2").itemsize,
        },
    ]
    if len(packed) != expected_size or layout != expected_layout:
        raise ValueError("encoder weight layout does not match manifest")
    values = np.frombuffer(packed, dtype="<f2")
    down = torch.from_numpy(values[:down_count].copy()).reshape(rank, hidden_size)
    up = torch.from_numpy(values[down_count:].copy()).reshape(
        kv_heads * head_dim, rank
    )
    model.load_state_dict({"down.weight": down, "up.weight": up})
    return model.to(device).eval()

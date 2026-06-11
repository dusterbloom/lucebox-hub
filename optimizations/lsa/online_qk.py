#!/usr/bin/env python3
"""No-training online QK selector for Qwen3.5 LSA raw captures."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .oracle import cross_layer_oracle, layer_block_attention_mass
    from .raw_dataset import RawCapture, load_raw_capture
except ImportError:
    from oracle import cross_layer_oracle, layer_block_attention_mass
    from raw_dataset import RawCapture, load_raw_capture

REPORT_SCHEMA = "luce.lsa.qwen35.online_qk_report.v1"


def cold_block_geometry(
    *,
    boundary_position: int,
    block_size: int = 64,
    sink_tokens: int = 64,
    recent_tokens: int = 8192,
) -> tuple[int, int]:
    if block_size <= 0 or sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("cold block geometry is invalid")
    if boundary_position < 0:
        raise ValueError("boundary position must be non-negative")
    cold_end = max(sink_tokens, boundary_position - recent_tokens)
    first_block = (sink_tokens + block_size - 1) // block_size
    block_count = cold_end // block_size
    return first_block, max(0, block_count - first_block)


def pooled_cold_keys(
    k_post_rope: torch.Tensor,
    *,
    boundary_position: int,
    block_size: int = 64,
    sink_tokens: int = 64,
    recent_tokens: int = 8192,
) -> torch.Tensor:
    if k_post_rope.ndim != 3:
        raise ValueError("K capture must have shape [tokens, kv_heads, head_dim]")
    first_block, candidate_blocks = cold_block_geometry(
        boundary_position=boundary_position,
        block_size=block_size,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
    )
    if candidate_blocks == 0:
        return torch.empty(
            (0, k_post_rope.shape[1], k_post_rope.shape[2]),
            dtype=torch.float32,
            device=k_post_rope.device,
        )
    begin = first_block * block_size
    end = (first_block + candidate_blocks) * block_size
    if end > k_post_rope.shape[0]:
        raise ValueError("K capture does not cover sealed cold blocks")
    blocks = k_post_rope[begin:end].float().reshape(
        candidate_blocks,
        block_size,
        k_post_rope.shape[1],
        k_post_rope.shape[2],
    )
    return F.normalize(blocks.mean(dim=1), dim=-1)


def qk_block_scores(
    q_post_rope: torch.Tensor,
    k_post_rope: torch.Tensor,
    *,
    boundary_position: int,
    block_size: int = 64,
    sink_tokens: int = 64,
    recent_tokens: int = 8192,
    pooling: str = "max",
) -> torch.Tensor:
    """Score sealed cold blocks with current/future Q against pooled post-RoPE K."""

    if q_post_rope.ndim != 3:
        raise ValueError("Q capture must have shape [future, q_heads, head_dim]")
    if q_post_rope.shape[-1] != k_post_rope.shape[-1]:
        raise ValueError("Q and K head dimensions do not match")
    if q_post_rope.shape[1] % k_post_rope.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if pooling not in {"max", "mean", "logsumexp"}:
        raise ValueError("pooling must be max, mean, or logsumexp")

    keys = pooled_cold_keys(
        k_post_rope,
        boundary_position=boundary_position,
        block_size=block_size,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
    )
    if keys.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=k_post_rope.device)

    group_size = q_post_rope.shape[1] // k_post_rope.shape[1]
    query = F.normalize(q_post_rope.float(), dim=-1).reshape(
        q_post_rope.shape[0],
        k_post_rope.shape[1],
        group_size,
        q_post_rope.shape[-1],
    )
    per_query = torch.einsum("thgd,bhd->tbhg", query, keys).flatten(start_dim=2)
    if pooling == "max":
        return per_query.amax(dim=(0, 2))
    if pooling == "mean":
        return per_query.mean(dim=(0, 2))
    return torch.logsumexp(per_query.flatten(start_dim=1), dim=0)


def aggregate_layer_scores(
    layer_scores: Iterable[torch.Tensor],
    *,
    mode: str = "max",
) -> torch.Tensor:
    scores = list(layer_scores)
    if not scores:
        raise ValueError("at least one layer score tensor is required")
    if mode not in {"max", "mean"}:
        raise ValueError("aggregation mode must be max or mean")
    shapes = {tuple(score.shape) for score in scores}
    if len(shapes) != 1:
        raise ValueError("layer score tensors must have matching shapes")
    stacked = torch.stack(scores, dim=0)
    if mode == "max":
        return stacked.amax(dim=0)
    return stacked.mean(dim=0)


def top_budget_indices(
    scores: torch.Tensor,
    *,
    keep_ratio: float | None = None,
    budget: int | None = None,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if scores.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=scores.device)
    if budget is None:
        if keep_ratio is None or not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1] when budget is absent")
        budget = math.ceil(scores.numel() * keep_ratio)
    if budget <= 0:
        raise ValueError("budget must be positive")
    count = min(scores.numel(), budget)
    return torch.topk(scores, count, sorted=False).indices


def mass_recall(target: torch.Tensor, selected: torch.Tensor) -> float:
    total = float(target.sum())
    if total <= 0:
        return 1.0
    return float(target[selected].sum()) / total


def evaluate_raw_capture(
    capture: RawCapture,
    *,
    keep_ratios: Iterable[float] = (0.10, 0.20, 0.50),
    device: str = "cpu",
    sink_tokens: int = 64,
    recent_tokens: int = 8192,
    score_pooling: str = "max",
    layer_aggregation: str = "max",
    max_examples: int = 0,
    seed: int = 7,
) -> dict[str, object]:
    manifest = capture.manifest
    block_size = int(manifest["block_size"])
    horizon = int(manifest["lookahead_horizon"])
    layers = tuple(int(layer) for layer in manifest["oracle_layers"])
    device_obj = torch.device(device)
    rng = torch.Generator().manual_seed(seed)
    metrics: dict[str, list[float]] = defaultdict(list)
    examples = int(capture.boundary_pos.shape[0])
    if max_examples > 0:
        examples = min(examples, max_examples)

    with torch.no_grad():
        for row in range(examples):
            boundary = int(capture.boundary_pos[row])
            key_end = boundary + horizon
            query_positions = torch.arange(
                boundary, key_end, dtype=torch.int64, device=device_obj
            )
            key_positions = torch.arange(
                key_end, dtype=torch.int64, device=device_obj
            )
            layer_mass = []
            layer_scores = []
            for layer in layers:
                query = torch.from_numpy(
                    np.asarray(capture.query_post[layer][row], dtype=np.float32)
                ).to(device_obj)
                key = torch.from_numpy(
                    np.asarray(capture.key_post[layer][:key_end], dtype=np.float32)
                ).to(device_obj)
                layer_mass.append(
                    layer_block_attention_mass(
                        query,
                        key,
                        query_positions,
                        key_positions,
                        block_size=block_size,
                        sink_tokens=sink_tokens,
                        recent_tokens=recent_tokens,
                        boundary_position=boundary,
                    )
                )
                layer_scores.append(
                    qk_block_scores(
                        query,
                        key,
                        boundary_position=boundary,
                        block_size=block_size,
                        sink_tokens=sink_tokens,
                        recent_tokens=recent_tokens,
                        pooling=score_pooling,
                    )
                )
            target = cross_layer_oracle(torch.stack(layer_mass, dim=0)).label_mass
            scores = aggregate_layer_scores(layer_scores, mode=layer_aggregation)
            if scores.numel() != target.numel():
                raise ValueError("selector score count does not match oracle target")
            for ratio in keep_ratios:
                suffix = f"{ratio:.3f}"
                selected = top_budget_indices(scores, keep_ratio=ratio)
                count = selected.numel()
                recent = torch.arange(
                    max(0, scores.numel() - count),
                    scores.numel(),
                    device=scores.device,
                )
                random_order = torch.randperm(
                    scores.numel(), generator=rng, device=scores.device
                )[:count]
                metrics[f"qk_recall@{suffix}"].append(mass_recall(target, selected))
                metrics[f"recent_recall@{suffix}"].append(mass_recall(target, recent))
                metrics[f"random_recall@{suffix}"].append(
                    mass_recall(target, random_order)
                )
            metrics["candidate_blocks"].append(float(scores.numel()))
            metrics["target_mass"].append(float(target.sum()))

    means = {name: float(np.mean(values)) for name, values in sorted(metrics.items())}
    return {
        "schema": REPORT_SCHEMA,
        "raw": str(capture.path),
        "examples": examples,
        "device": device,
        "sink_tokens": sink_tokens,
        "recent_tokens": recent_tokens,
        "score_pooling": score_pooling,
        "layer_aggregation": layer_aggregation,
        "keep_ratios": list(keep_ratios),
        "metrics": means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.10, 0.20, 0.50])
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-tokens", type=int, default=8192)
    parser.add_argument("--score-pooling", choices=("max", "mean", "logsumexp"), default="max")
    parser.add_argument("--layer-aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(ratio <= 0 or ratio > 1 for ratio in args.keep_ratios):
        parser.error("--keep-ratios values must be in (0, 1]")
    report = evaluate_raw_capture(
        load_raw_capture(args.raw, verify_checksums=args.verify_checksums),
        keep_ratios=args.keep_ratios,
        device=args.device,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        score_pooling=args.score_pooling,
        layer_aggregation=args.layer_aggregation,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()

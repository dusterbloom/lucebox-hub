#!/usr/bin/env python3
"""Evaluate an LSA encoder against fixed-budget retrieval baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from .artifact import load_encoder_artifact
    from .dataset import LsaExampleDataset
except ImportError:
    from artifact import load_encoder_artifact
    from dataset import LsaExampleDataset


def mass_recall(target: torch.Tensor, selected: torch.Tensor) -> float:
    total = float(target.sum())
    if total <= 0:
        return 1.0
    return float(target[selected].sum()) / total


def top_indices(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    count = min(scores.numel(), max(1, math.ceil(scores.numel() * keep_ratio)))
    return torch.topk(scores, count, sorted=False).indices


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    dataset = LsaExampleDataset(args.shards)
    device = torch.device(args.device)
    model = load_encoder_artifact(args.model, device)
    generator = torch.Generator().manual_seed(args.seed)
    metrics: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for example in dataset:
            hidden = example["hidden"].to(device)
            keys = example["keys"].to(device)
            target = example["target"]
            scores = torch.sigmoid(model(hidden, keys)).cpu()
            metrics["threshold_keep"].append(
                float((scores >= args.threshold).float().mean())
            )
            for keep_ratio in args.keep_ratios:
                suffix = f"{keep_ratio:.3f}"
                learned = top_indices(scores, keep_ratio)
                count = learned.numel()
                recent = torch.arange(scores.numel() - count, scores.numel())
                random_order = torch.randperm(scores.numel(), generator=generator)[:count]
                metrics[f"learned_recall@{suffix}"].append(mass_recall(target, learned))
                metrics[f"recent_recall@{suffix}"].append(mass_recall(target, recent))
                metrics[f"random_recall@{suffix}"].append(
                    mass_recall(target, random_order)
                )

    means = {name: float(np.mean(values)) for name, values in sorted(metrics.items())}
    return {
        "schema": "luce.lsa.qwen35.evaluation.v1",
        "examples": len(dataset),
        "threshold": args.threshold,
        "metrics": means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(ratio <= 0 or ratio > 1 for ratio in args.keep_ratios):
        parser.error("--keep-ratios values must be in (0, 1]")
    report = evaluate(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()

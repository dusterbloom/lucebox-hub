#!/usr/bin/env python3
"""Train the compact Qwen3.5 LSA query encoder from extracted NPZ shards."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

try:
    from .artifact import write_encoder_artifact
    from .dataset import LsaExampleDataset
    from .model import CompactQwen35Encoder, focal_mass_loss
except ImportError:
    from artifact import write_encoder_artifact
    from dataset import LsaExampleDataset
    from model import CompactQwen35Encoder, focal_mass_loss


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = LsaExampleDataset(args.shards)
    metadata = dataset.metadata
    device = torch.device(args.device)
    model = CompactQwen35Encoder(
        hidden_size=metadata.hidden_size,
        rank=args.rank,
        kv_heads=metadata.kv_heads,
        head_dim=metadata.head_dim,
        score_temperature=args.score_temperature,
        decision_threshold=args.decision_threshold,
        logit_scale=args.logit_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    order = list(range(len(dataset)))
    step = 0
    pending = 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(order)
        running = 0.0
        examples_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for index in order:
            example = dataset[index]
            hidden = example["hidden"].to(device)
            keys = example["keys"].to(device)
            target = example["target"].to(device)
            logits = model(hidden, keys)
            loss = focal_mass_loss(
                logits,
                target,
                gamma=args.focal_gamma,
                positive_weight=args.positive_weight,
            )
            (loss / args.accumulate).backward()
            running += float(loss.detach())
            examples_seen += 1
            step += 1
            pending += 1
            if pending == args.accumulate:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            if args.max_steps and step >= args.max_steps:
                break
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        print(f"epoch={epoch + 1} steps={step} mean_loss={running / examples_seen:.6f}")
        if args.max_steps and step >= args.max_steps:
            break

    write_encoder_artifact(args.output, model, metadata)
    print(f"wrote {args.output} ({model.parameter_count():,} parameters)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--score-temperature", type=float, default=0.1)
    parser.add_argument("--decision-threshold", type=float, default=0.02)
    parser.add_argument("--logit-scale", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.accumulate <= 0 or args.epochs <= 0:
        parser.error("--accumulate and --epochs must be positive")
    train(args)


if __name__ == "__main__":
    main()

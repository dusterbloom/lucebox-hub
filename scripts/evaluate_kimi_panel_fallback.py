#!/usr/bin/env python3
"""Measure exact-top-rank and confidence fallback on held-out Kimi tokens."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from export_kimi_panel_safetensors import load_panel
from train_kimi_panel_directional import load_data, summarize


RANK_HEADER = struct.Struct("<8sIiIIQII3Q")
RANK_MAGIC = b"K3RNK001"


def read_rank_teacher(
    path: Path, model_layer: int, dimension: int, top_k: int, tokens: int
) -> np.ndarray:
    with path.open("rb") as source:
        raw = source.read(RANK_HEADER.size)
        if len(raw) != RANK_HEADER.size:
            raise ValueError("rank teacher header is truncated")
        (
            magic,
            version,
            layer,
            stored_dimension,
            stored_top_k,
            stored_tokens,
            storage,
            reserved,
            *reserved64,
        ) = RANK_HEADER.unpack(raw)
        if (
            magic != RANK_MAGIC
            or version != 1
            or layer != model_layer
            or stored_dimension != dimension
            or stored_top_k != top_k
            or stored_tokens != tokens
            or storage != 0
            or reserved != 0
            or any(reserved64)
        ):
            raise ValueError("rank teacher does not match the capture")
        values = np.fromfile(
            source, dtype="<f4", count=top_k * tokens * dimension
        )
        if values.size != top_k * tokens * dimension or source.read(1):
            raise ValueError("rank teacher payload is truncated or extended")
    return values.reshape(top_k, tokens, dimension)


def metric(exact: torch.Tensor, estimate: torch.Tensor) -> dict[str, object]:
    cosine = functional.cosine_similarity(exact, estimate, dim=1)
    relative = (
        torch.linalg.vector_norm(estimate - exact, dim=1)
        / torch.linalg.vector_norm(exact, dim=1).clamp_min(1e-12)
    )
    return {
        "cosine": summarize(cosine.float().cpu().numpy()),
        "relative_l2": summarize(relative.float().cpu().numpy()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("rank_teacher", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()

    data = load_data(args.capture, args.teacher)
    metadata, arrays = load_panel(args.panel)
    if metadata["model_layer"] != data.model_layer:
        raise ValueError("panel layer does not match capture")
    validation_count = data.validation_indices.size
    exact_by_rank = read_rank_teacher(
        args.rank_teacher, data.model_layer, data.dimension,
        data.top_k, validation_count
    )
    device = torch.device(args.device)
    validation = data.validation_indices
    latent = torch.from_numpy(data.latent[validation].copy()).to(device)
    ids = torch.from_numpy(data.expert_ids[validation].copy()).to(device)
    weights = torch.from_numpy(data.router_weights[validation].copy()).to(device)
    exact = torch.from_numpy(data.teacher[validation].copy()).to(device)
    rank_exact = torch.from_numpy(exact_by_rank.copy()).to(device)
    offset = torch.from_numpy(arrays["unweighted_offset"].copy()).to(device)
    gain = torch.from_numpy(arrays["unweighted_gain"].copy()).to(device)

    approximate_by_rank: list[torch.Tensor] = []
    approximate = torch.zeros_like(exact)
    for rank in range(data.top_k):
        expert = ids[:, rank]
        contribution = weights[:, rank, None] * (
            offset[expert] + gain[expert] * latent
        )
        approximate_by_rank.append(contribution)
        approximate += contribution

    reconstructed_exact = rank_exact.sum(dim=0)
    exact_reconstruction = {
        "maximum_absolute_difference": float(
            (reconstructed_exact - exact).abs().max().cpu()
        ),
        "mean_cosine": float(
            functional.cosine_similarity(reconstructed_exact, exact, dim=1)
            .mean().cpu()
        ),
    }

    top_rank_ladder: list[dict[str, object]] = []
    hybrid = approximate.clone()
    top_rank_ladder.append(
        {
            "exact_ranks": 0,
            "exact_expert_evaluations_per_token": 0,
            "fraction_of_exact_routed_traffic": 0.0,
            "metrics": metric(exact, hybrid),
        }
    )
    for rank in range(data.top_k):
        hybrid += rank_exact[rank] - approximate_by_rank[rank]
        top_rank_ladder.append(
            {
                "exact_ranks": rank + 1,
                "exact_expert_evaluations_per_token": rank + 1,
                "fraction_of_exact_routed_traffic": (rank + 1) / data.top_k,
                "metrics": metric(exact, hybrid),
            }
        )

    token_cosine = functional.cosine_similarity(approximate, exact, dim=1)
    confidence = weights.max(dim=1).values
    fractions = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

    def token_fallback_ladder(order: torch.Tensor) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for fraction in fractions:
            count = round(fraction * validation_count)
            candidate = approximate.clone()
            if count:
                selected = order[:count]
                candidate[selected] = exact[selected]
            rows.append(
                {
                    "exact_token_fraction": fraction,
                    "exact_tokens": count,
                    "average_exact_expert_evaluations_per_token": (
                        count * data.top_k / validation_count
                    ),
                    "metrics": metric(exact, candidate),
                }
            )
        return rows

    result = {
        "schema": "kimi-k3-panel-fallback-v1",
        "status": "exploratory_followup_not_registered_primary_gate",
        "model_layer": data.model_layer,
        "validation_tokens": int(validation_count),
        "panel_variant": "unweighted_diagonal",
        "exact_rank_reconstruction": exact_reconstruction,
        "exact_top_router_rank_ladder": top_rank_ladder,
        "whole_token_fallback": {
            "highest_router_confidence_first": token_fallback_ladder(
                torch.argsort(confidence, descending=True)
            ),
            "lowest_router_confidence_first_control": token_fallback_ladder(
                torch.argsort(confidence, descending=False)
            ),
            "oracle_worst_panel_cosine_first_upper_bound": token_fallback_ladder(
                torch.argsort(token_cosine, descending=False)
            ),
        },
        "final_logit_kl": None,
        "final_logit_kl_unavailable_reason": (
            "The capture stops before the remaining ninety-one model layers."
        ),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("exact ranks -> mean held-out cosine")
    for row in top_rank_ladder:
        print(
            f"{row['exact_ranks']:2d} -> "
            f"{row['metrics']['cosine']['mean']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

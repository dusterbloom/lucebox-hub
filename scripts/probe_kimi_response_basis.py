#!/usr/bin/env python3
"""Measure a layer-shared basis for exact Kimi routed-expert responses.

If responses from all experts occupy a common output subspace, an atlas can
store coordinates instead of 3,584-dimensional answers and local charts can
map into that same small coordinate space.  This is a storage/computation
geometry probe; it does not alter model execution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from probe_kimi_response_atlas import (
    pair_cosine,
    read_expert_responses,
    response_path,
    summarize,
)
from train_kimi_panel_directional import load_data


EXPERT_COUNT = 896
MODEL_MOE_LAYERS = 92
BASIS_RANKS = (8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-routes", type=int, default=16384)
    parser.add_argument("--pca-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260813)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_data(args.capture, args.teacher)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    token_count = data.latent.shape[0]
    dimension = data.dimension
    validation_mask = np.zeros(token_count, dtype=bool)
    validation_mask[data.validation_indices] = True
    calibration_route_count = (
        token_count - data.validation_indices.size
    ) * data.top_k
    sample_probability = min(1.0, args.sample_routes / calibration_route_count)
    sampled: list[np.ndarray] = []

    for expert in range(EXPERT_COUNT):
        records, outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        calibration_rows = np.flatnonzero(~validation_mask[token_indices])
        keep = rng.random(calibration_rows.size) < sample_probability
        if np.any(keep):
            sampled.append(outputs[calibration_rows[keep]].copy())
    sample = np.concatenate(sampled)
    if sample.shape[0] < max(BASIS_RANKS):
        raise ValueError("too few calibration responses sampled")
    # Keep a deterministic exactly bounded subset if Bernoulli sampling ran high.
    if sample.shape[0] > args.sample_routes:
        selected = rng.choice(sample.shape[0], args.sample_routes, replace=False)
        sample = sample[selected]

    sample_tensor = torch.from_numpy(sample).to(device)
    output_mean = sample_tensor.mean(dim=0, keepdim=True)
    centered = sample_tensor - output_mean
    total_sample_energy = float(centered.square().sum().item())
    torch.manual_seed(args.seed)
    _, singular_values, basis = torch.pca_lowrank(
        centered,
        q=max(BASIS_RANKS),
        center=False,
        niter=args.pca_iterations,
    )
    explained = singular_values.square().cumsum(dim=0) / total_sample_energy

    validation_count = data.validation_indices.size
    aggregate = {
        rank: np.zeros((validation_count, dimension), dtype=np.float32)
        for rank in BASIS_RANKS
    }
    individual_cosines: dict[int, list[np.ndarray]] = {
        rank: [] for rank in BASIS_RANKS
    }
    validation_position = np.full(token_count, -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(
        validation_count, dtype=np.int64
    )

    for expert in range(EXPERT_COUNT):
        records, outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        validation_rows = np.flatnonzero(validation_mask[token_indices])
        if validation_rows.size == 0:
            continue
        validation_tokens = token_indices[validation_rows]
        route_weights = records["router_weight"][validation_rows].astype(
            np.float32, copy=False
        )
        exact = torch.from_numpy(outputs[validation_rows]).to(device)
        centered_exact = exact - output_mean
        coordinates = centered_exact @ basis
        positions = validation_position[validation_tokens]
        for rank in BASIS_RANKS:
            estimate = output_mean + coordinates[:, :rank] @ basis[:, :rank].T
            individual_cosines[rank].append(
                torch.nn.functional.cosine_similarity(
                    estimate, exact, dim=1
                ).detach().cpu().numpy()
            )
            aggregate[rank][positions] += (
                route_weights[:, None] * estimate.detach().cpu().numpy()
            )

    exact_aggregate = data.teacher[data.validation_indices]
    variants: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for rank in BASIS_RANKS:
        name = f"shared_response_rank{rank:03d}"
        aggregate_cosine = pair_cosine(aggregate[rank], exact_aggregate)
        expert_cosine = np.concatenate(individual_cosines[rank])
        # Global per-layer mean+basis, plus rank BF16 coordinates per stored
        # atlas entry.  Report atlas K=64 and K=all.
        fixed_elements_layer = dimension + dimension * rank
        atlas_projection: dict[str, dict[str, float | int]] = {}
        for atlas_name, entries_layer in (
            ("k64", 64 * EXPERT_COUNT),
            ("kall_calibration", calibration_route_count),
        ):
            stored_elements_layer = fixed_elements_layer + entries_layer * rank
            atlas_projection[atlas_name] = {
                "entries_layer": entries_layer,
                "projected_bfloat16_all_layers_bytes": (
                    stored_elements_layer * 2 * MODEL_MOE_LAYERS
                ),
                "projected_bfloat16_one_response_bytes_per_token": (
                    rank * 2 * data.top_k * MODEL_MOE_LAYERS
                ),
                "reconstruction_multiply_accumulates_per_token": (
                    rank * dimension * data.top_k * MODEL_MOE_LAYERS
                ),
            }
        variants[name] = {
            "rank": rank,
            "sample_cumulative_centered_energy": float(explained[rank - 1].item()),
            "routed_aggregate_cosine": summarize(aggregate_cosine),
            "individual_expert_cosine": summarize(expert_cosine),
            "atlas_projection": atlas_projection,
        }
        csv_rows.append(
            {
                "variant": name,
                "rank": rank,
                "sample_cumulative_centered_energy": float(
                    explained[rank - 1].item()
                ),
                "aggregate_mean_cosine": float(aggregate_cosine.mean()),
                "aggregate_p05_cosine": float(np.quantile(aggregate_cosine, 0.05)),
                "expert_mean_cosine": float(expert_cosine.mean()),
                "atlas_k64_all_layers_gib": (
                    atlas_projection["k64"][
                        "projected_bfloat16_all_layers_bytes"
                    ]
                    / (1 << 30)
                ),
                "atlas_kall_all_layers_gib": (
                    atlas_projection["kall_calibration"][
                        "projected_bfloat16_all_layers_bytes"
                    ]
                    / (1 << 30)
                ),
                "one_response_kib_per_token": (
                    atlas_projection["k64"][
                        "projected_bfloat16_one_response_bytes_per_token"
                    ]
                    / (1 << 10)
                ),
            }
        )

    best_name = max(
        variants,
        key=lambda name: variants[name]["routed_aggregate_cosine"]["mean"],
    )
    result = {
        "schema": "kimi-k3-layer01-shared-response-basis-v1",
        "status": "EXPLORATORY",
        "capture": str(args.capture),
        "teacher": str(args.teacher),
        "response_directory": str(args.response_directory),
        "model_layer": data.model_layer,
        "dimension": dimension,
        "top_k": data.top_k,
        "calibration_route_count": calibration_route_count,
        "validation_route_count": validation_count * data.top_k,
        "sample_routes_requested": args.sample_routes,
        "sample_routes_used": int(sample.shape[0]),
        "pca_iterations": args.pca_iterations,
        "seed": args.seed,
        "basis": "centered exact individual routed-expert responses",
        "best_variant": best_name,
        "variants": variants,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)
    best = variants[best_name]["routed_aggregate_cosine"]
    print(
        f"best={best_name} mean={best['mean']:.9f} "
        f"p05={best['p05']:.9f} sample={sample.shape[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

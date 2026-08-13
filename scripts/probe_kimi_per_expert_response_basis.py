#!/usr/bin/env python3
"""Measure held-out response subspaces independently for every Kimi expert."""

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
BASIS_RANKS = (1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260813)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_data(args.capture, args.teacher)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    token_count = data.latent.shape[0]
    dimension = data.dimension
    validation_mask = np.zeros(token_count, dtype=bool)
    validation_mask[data.validation_indices] = True
    validation_position = np.full(token_count, -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(
        data.validation_indices.size, dtype=np.int64
    )
    variant_names = [f"per_expert_rank{rank:02d}" for rank in BASIS_RANKS]
    aggregate = {
        name: np.zeros(
            (data.validation_indices.size, dimension), dtype=np.float32
        )
        for name in variant_names
    }
    individual_cosines: dict[str, list[np.ndarray]] = {
        name: [] for name in variant_names
    }
    realized_ranks: dict[int, list[int]] = {rank: [] for rank in BASIS_RANKS}
    fixed_elements: dict[int, int] = {rank: 0 for rank in BASIS_RANKS}
    atlas_k64_elements: dict[int, int] = {rank: 0 for rank in BASIS_RANKS}
    atlas_all_elements: dict[int, int] = {rank: 0 for rank in BASIS_RANKS}
    validation_rank_sum: dict[int, int] = {rank: 0 for rank in BASIS_RANKS}
    training_counts: list[int] = []

    for expert in range(EXPERT_COUNT):
        records, outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        if not np.all(data.expert_ids[token_indices, ranks] == expert):
            raise ValueError(f"response metadata disagrees for expert {expert}")
        is_validation = validation_mask[token_indices]
        training_rows = np.flatnonzero(~is_validation)
        validation_rows = np.flatnonzero(is_validation)
        training_counts.append(int(training_rows.size))
        if training_rows.size == 0:
            raise ValueError(f"expert {expert} has no calibration responses")

        y_training = torch.from_numpy(outputs[training_rows]).to(device)
        y_mean = y_training.mean(dim=0, keepdim=True)
        centered = y_training - y_mean
        maximum_rank = min(max(BASIS_RANKS), max(0, training_rows.size - 1))
        if maximum_rank > 0:
            _, _, basis = torch.pca_lowrank(
                centered,
                q=maximum_rank,
                center=False,
                niter=args.pca_iterations,
            )
        else:
            basis = torch.empty((dimension, 0), device=device)

        if validation_rows.size:
            y_validation = torch.from_numpy(outputs[validation_rows]).to(device)
            centered_validation = y_validation - y_mean
            coordinates = centered_validation @ basis
            validation_tokens = token_indices[validation_rows]
            positions = validation_position[validation_tokens]
            route_weights = records["router_weight"][validation_rows].astype(
                np.float32, copy=False
            )

        for requested_rank in BASIS_RANKS:
            name = f"per_expert_rank{requested_rank:02d}"
            realized_rank = min(requested_rank, maximum_rank)
            realized_ranks[requested_rank].append(realized_rank)
            fixed_elements[requested_rank] += dimension * (1 + realized_rank)
            atlas_k64_elements[requested_rank] += (
                min(64, training_rows.size) * realized_rank
            )
            atlas_all_elements[requested_rank] += training_rows.size * realized_rank
            validation_rank_sum[requested_rank] += (
                validation_rows.size * realized_rank
            )
            if validation_rows.size == 0:
                continue
            estimate = y_mean.expand(validation_rows.size, -1)
            if realized_rank:
                estimate = estimate + (
                    coordinates[:, :realized_rank]
                    @ basis[:, :realized_rank].T
                )
            individual_cosines[name].append(
                torch.nn.functional.cosine_similarity(
                    estimate, y_validation, dim=1
                ).detach().cpu().numpy()
            )
            aggregate[name][positions] += (
                route_weights[:, None] * estimate.detach().cpu().numpy()
            )

    exact_aggregate = data.teacher[data.validation_indices]
    validation_routes = data.validation_indices.size * data.top_k
    variants: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for requested_rank in BASIS_RANKS:
        name = f"per_expert_rank{requested_rank:02d}"
        aggregate_cosine = pair_cosine(aggregate[name], exact_aggregate)
        expert_cosine = np.concatenate(individual_cosines[name])
        realized = np.asarray(realized_ranks[requested_rank])
        average_validation_rank = (
            validation_rank_sum[requested_rank] / validation_routes
        )
        projections: dict[str, dict[str, float | int]] = {}
        for atlas_name, coordinate_elements in (
            ("k64", atlas_k64_elements[requested_rank]),
            ("kall_calibration", atlas_all_elements[requested_rank]),
        ):
            layer_elements = fixed_elements[requested_rank] + coordinate_elements
            projections[atlas_name] = {
                "projected_bfloat16_all_layers_bytes": (
                    layer_elements * 2 * MODEL_MOE_LAYERS
                ),
                "projected_mean_coordinate_bytes_per_token": (
                    average_validation_rank
                    * 2
                    * data.top_k
                    * MODEL_MOE_LAYERS
                ),
                "reconstruction_multiply_accumulates_per_token": (
                    average_validation_rank
                    * dimension
                    * data.top_k
                    * MODEL_MOE_LAYERS
                ),
            }
        variants[name] = {
            "requested_rank": requested_rank,
            "realized_rank_per_expert": summarize(realized),
            "validation_route_weighted_average_rank": average_validation_rank,
            "routed_aggregate_cosine": summarize(aggregate_cosine),
            "individual_expert_cosine": summarize(expert_cosine),
            "atlas_projection": projections,
        }
        csv_rows.append(
            {
                "variant": name,
                "requested_rank": requested_rank,
                "mean_realized_rank": float(realized.mean()),
                "validation_weighted_rank": average_validation_rank,
                "aggregate_mean_cosine": float(aggregate_cosine.mean()),
                "aggregate_p05_cosine": float(np.quantile(aggregate_cosine, 0.05)),
                "expert_mean_cosine": float(expert_cosine.mean()),
                "atlas_k64_all_layers_gib": (
                    projections["k64"]["projected_bfloat16_all_layers_bytes"]
                    / (1 << 30)
                ),
                "atlas_kall_all_layers_gib": (
                    projections["kall_calibration"][
                        "projected_bfloat16_all_layers_bytes"
                    ]
                    / (1 << 30)
                ),
                "mean_coordinate_kib_per_token": (
                    projections["k64"][
                        "projected_mean_coordinate_bytes_per_token"
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
        "schema": "kimi-k3-layer01-per-expert-response-basis-v1",
        "status": "EXPLORATORY",
        "capture": str(args.capture),
        "teacher": str(args.teacher),
        "response_directory": str(args.response_directory),
        "model_layer": data.model_layer,
        "dimension": dimension,
        "top_k": data.top_k,
        "tokens": token_count,
        "calibration_tokens": int(token_count - data.validation_indices.size),
        "validation_tokens": int(data.validation_indices.size),
        "pca_iterations": args.pca_iterations,
        "seed": args.seed,
        "training_routes_per_expert": summarize(np.asarray(training_counts)),
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
        f"p05={best['p05']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

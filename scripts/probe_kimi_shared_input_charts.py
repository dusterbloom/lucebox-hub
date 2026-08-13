#!/usr/bin/env python3
"""Fit compute-light per-expert charts in one shared Kimi latent basis.

This probe asks whether a small, layer-shared address projection is sufficient
to select a linearly varying expert answer.  Each expert stores an intercept and
a map from q shared principal coordinates to the 3,584-dimensional response.
The native router and its weights remain authoritative.
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
CHART_RANKS = (2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-fraction", type=float, default=1.0e-2)
    parser.add_argument("--pca-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260813)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ridge_fraction <= 0:
        raise ValueError("ridge fraction must be positive")
    data = load_data(args.capture, args.teacher)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dimension = data.dimension
    token_count = data.latent.shape[0]
    validation_mask = np.zeros(token_count, dtype=bool)
    validation_mask[data.validation_indices] = True
    calibration_indices = np.flatnonzero(~validation_mask)
    validation_position = np.full(token_count, -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(
        data.validation_indices.size, dtype=np.int64
    )

    latent = torch.from_numpy(data.latent).to(device)
    latent_mean = latent[calibration_indices].mean(dim=0, keepdim=True)
    centered_calibration = latent[calibration_indices] - latent_mean
    _, singular_values, basis = torch.pca_lowrank(
        centered_calibration,
        q=max(CHART_RANKS),
        center=False,
        niter=args.pca_iterations,
    )
    features = (latent - latent_mean) @ basis
    total_latent_energy = float(centered_calibration.square().sum().item())
    explained = singular_values.square().cumsum(dim=0) / total_latent_energy

    variant_names = ["mean"] + [f"shared_pca_rank{rank:02d}" for rank in CHART_RANKS]
    aggregate = {
        name: np.zeros((data.validation_indices.size, dimension), dtype=np.float32)
        for name in variant_names
    }
    individual_cosines: dict[str, list[np.ndarray]] = {
        name: [] for name in variant_names
    }
    training_counts: list[int] = []
    validation_counts: list[int] = []

    for expert in range(EXPERT_COUNT):
        records, outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        route_weights = records["router_weight"].astype(np.float32, copy=False)
        if (
            np.any(token_indices >= token_count)
            or np.any(ranks >= data.top_k)
            or not np.all(data.expert_ids[token_indices, ranks] == expert)
            or not np.array_equal(
                data.router_weights[token_indices, ranks], route_weights
            )
        ):
            raise ValueError(f"response metadata disagrees with capture for expert {expert}")
        is_validation = validation_mask[token_indices]
        training_rows = np.flatnonzero(~is_validation)
        validation_rows = np.flatnonzero(is_validation)
        if training_rows.size == 0:
            raise ValueError(f"expert {expert} has no calibration responses")
        training_counts.append(int(training_rows.size))
        validation_counts.append(int(validation_rows.size))
        if validation_rows.size == 0:
            continue

        training_tokens = token_indices[training_rows]
        validation_tokens = token_indices[validation_rows]
        x_training = features[training_tokens]
        x_validation = features[validation_tokens]
        y_training = torch.from_numpy(outputs[training_rows]).to(device)
        y_validation = torch.from_numpy(outputs[validation_rows]).to(device)
        y_mean = y_training.mean(dim=0, keepdim=True)
        positions = validation_position[validation_tokens]
        weights = route_weights[validation_rows, None]

        mean_estimate = y_mean.expand(validation_rows.size, -1)
        individual_cosines["mean"].append(
            torch.nn.functional.cosine_similarity(
                mean_estimate, y_validation, dim=1
            ).detach().cpu().numpy()
        )
        aggregate["mean"][positions] += weights * mean_estimate.detach().cpu().numpy()

        y_centered = y_training - y_mean
        for rank in CHART_RANKS:
            name = f"shared_pca_rank{rank:02d}"
            x_train_rank = x_training[:, :rank]
            x_mean = x_train_rank.mean(dim=0, keepdim=True)
            x_centered = x_train_rank - x_mean
            x_scale = x_centered.square().mean(dim=0, keepdim=True).sqrt()
            x_scale = x_scale.clamp_min(1.0e-6)
            x_standardized = x_centered / x_scale
            gram = x_standardized.T @ x_standardized
            mean_eigenvalue = gram.trace() / rank
            regularized = gram + torch.eye(rank, device=device) * (
                args.ridge_fraction * mean_eigenvalue.clamp_min(1.0)
            )
            chart = torch.linalg.solve(
                regularized,
                x_standardized.T @ y_centered,
            )
            estimate = y_mean + (
                (x_validation[:, :rank] - x_mean) / x_scale
            ) @ chart
            individual_cosines[name].append(
                torch.nn.functional.cosine_similarity(
                    estimate, y_validation, dim=1
                ).detach().cpu().numpy()
            )
            aggregate[name][positions] += weights * estimate.detach().cpu().numpy()

    exact = data.teacher[data.validation_indices]
    variants: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for name in variant_names:
        rank = 0 if name == "mean" else int(name[-2:])
        aggregate_cosine = pair_cosine(aggregate[name], exact)
        expert_cosine = np.concatenate(individual_cosines[name])
        # One layer-shared latent mean and basis, plus an output intercept,
        # feature intercept, and q-by-D response chart for every expert.
        stored_elements_layer = (
            dimension
            + dimension * rank
            + EXPERT_COUNT * (dimension + 2 * rank + rank * dimension)
        )
        bytes_read_per_route = (
            dimension + 2 * rank + rank * dimension
        ) * 2
        variants[name] = {
            "rank": rank,
            "pca_cumulative_input_energy": (
                0.0 if rank == 0 else float(explained[rank - 1].item())
            ),
            "projected_bfloat16_all_layers_bytes": (
                stored_elements_layer * 2 * MODEL_MOE_LAYERS
            ),
            "projected_bfloat16_bytes_per_token": (
                bytes_read_per_route * data.top_k * MODEL_MOE_LAYERS
            ),
            "multiply_accumulates_per_token": (
                rank * dimension * data.top_k * MODEL_MOE_LAYERS
            ),
            "routed_aggregate_cosine": summarize(aggregate_cosine),
            "individual_expert_cosine": summarize(expert_cosine),
        }
        csv_rows.append(
            {
                "variant": name,
                "rank": rank,
                "pca_cumulative_input_energy": variants[name][
                    "pca_cumulative_input_energy"
                ],
                "aggregate_mean_cosine": float(aggregate_cosine.mean()),
                "aggregate_p05_cosine": float(np.quantile(aggregate_cosine, 0.05)),
                "expert_mean_cosine": float(expert_cosine.mean()),
                "projected_bfloat16_all_layers_gib": (
                    stored_elements_layer * 2 * MODEL_MOE_LAYERS / (1 << 30)
                ),
                "projected_bfloat16_mebibytes_per_token": (
                    bytes_read_per_route
                    * data.top_k
                    * MODEL_MOE_LAYERS
                    / (1 << 20)
                ),
                "multiply_accumulates_per_token": variants[name][
                    "multiply_accumulates_per_token"
                ],
            }
        )

    best_name = max(
        variants,
        key=lambda name: variants[name]["routed_aggregate_cosine"]["mean"],
    )
    result = {
        "schema": "kimi-k3-layer01-shared-input-charts-v1",
        "status": "EXPLORATORY",
        "capture": str(args.capture),
        "teacher": str(args.teacher),
        "response_directory": str(args.response_directory),
        "model_layer": data.model_layer,
        "dimension": dimension,
        "top_k": data.top_k,
        "tokens": token_count,
        "calibration_tokens": int(calibration_indices.size),
        "validation_tokens": int(data.validation_indices.size),
        "ridge_fraction_of_mean_gram_eigenvalue": args.ridge_fraction,
        "pca_iterations": args.pca_iterations,
        "seed": args.seed,
        "training_routes_per_expert": summarize(np.asarray(training_counts)),
        "validation_routes_per_expert": summarize(np.asarray(validation_counts)),
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

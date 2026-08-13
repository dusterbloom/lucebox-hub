#!/usr/bin/env python3
"""Probe reading only a subset of Kimi's 16 exact routed experts."""

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
ROUTE_BUDGETS = (0, 1, 2, 4, 8, 12, 15, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--greedy-batch", type=int, default=128)
    return parser.parse_args()


def gather_by_order(
    contributions: np.ndarray,
    approximate: np.ndarray,
    order: np.ndarray,
) -> dict[int, np.ndarray]:
    estimate = approximate.sum(axis=1).copy()
    results = {0: estimate.copy()}
    budgets = set(ROUTE_BUDGETS[1:])
    rows = np.arange(contributions.shape[0])
    for count in range(1, contributions.shape[1] + 1):
        rank = order[:, count - 1]
        estimate += (
            contributions[rows, rank] - approximate[rows, rank]
        )
        if count in budgets:
            results[count] = estimate.copy()
    return results


def greedy_oracle(
    contributions: np.ndarray,
    approximate: np.ndarray,
    teacher: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    outputs = {
        budget: np.zeros_like(teacher)
        for budget in ROUTE_BUDGETS
    }
    for begin in range(0, teacher.shape[0], batch_size):
        end = min(teacher.shape[0], begin + batch_size)
        exact_batch = torch.from_numpy(contributions[begin:end]).to(device)
        approximate_batch = torch.from_numpy(approximate[begin:end]).to(device)
        target = torch.from_numpy(teacher[begin:end]).to(device)
        delta = exact_batch - approximate_batch
        current = approximate_batch.sum(dim=1)
        outputs[0][begin:end] = current.cpu().numpy()
        used = torch.zeros(
            (end - begin, contributions.shape[1]),
            dtype=torch.bool,
            device=device,
        )
        rows = torch.arange(end - begin, device=device)
        for count in range(1, contributions.shape[1] + 1):
            candidates = current[:, None, :] + delta
            scores = torch.nn.functional.cosine_similarity(
                candidates, target[:, None, :], dim=2
            )
            scores.masked_fill_(used, -torch.inf)
            selected = scores.argmax(dim=1)
            current = current + delta[rows, selected]
            used[rows, selected] = True
            if count in outputs:
                outputs[count][begin:end] = current.cpu().numpy()
    return outputs


def main() -> int:
    args = parse_args()
    data = load_data(args.capture, args.teacher)
    device = torch.device(args.device)
    validation_count = data.validation_indices.size
    dimension = data.dimension
    validation_mask = np.zeros(data.latent.shape[0], dtype=bool)
    validation_mask[data.validation_indices] = True
    validation_position = np.full(data.latent.shape[0], -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(
        validation_count, dtype=np.int64
    )
    contributions = np.zeros(
        (validation_count, data.top_k, dimension), dtype=np.float32
    )
    approximate = np.zeros_like(contributions)
    expected_norm = np.zeros(EXPERT_COUNT, dtype=np.float32)

    for expert in range(EXPERT_COUNT):
        records, outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        weights = records["router_weight"].astype(np.float32, copy=False)
        if not np.all(data.expert_ids[token_indices, ranks] == expert):
            raise ValueError(f"response metadata disagrees for expert {expert}")
        is_validation = validation_mask[token_indices]
        training_rows = np.flatnonzero(~is_validation)
        validation_rows = np.flatnonzero(is_validation)
        mean_output = outputs[training_rows].mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
        expected_norm[expert] = np.linalg.norm(
            outputs[training_rows], axis=1
        ).mean()
        positions = validation_position[token_indices[validation_rows]]
        route_ranks = ranks[validation_rows]
        route_weights = weights[validation_rows]
        contributions[positions, route_ranks] = (
            route_weights[:, None] * outputs[validation_rows]
        )
        approximate[positions, route_ranks] = (
            route_weights[:, None] * mean_output[None, :]
        )

    teacher = data.teacher[data.validation_indices]
    reconstructed = contributions.sum(axis=1)
    if not np.allclose(reconstructed, teacher, rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("individual responses do not reconstruct exact teacher")

    router_order = np.broadcast_to(
        np.arange(data.top_k, dtype=np.int64),
        (validation_count, data.top_k),
    )
    expected_scores = (
        data.router_weights[data.validation_indices]
        * expected_norm[data.expert_ids[data.validation_indices]]
    )
    expected_order = np.argsort(-expected_scores, axis=1, kind="stable")
    oracle_norm_order = np.argsort(
        -np.linalg.norm(contributions, axis=2), axis=1, kind="stable"
    )

    estimates: dict[str, dict[int, np.ndarray]] = {}
    for tail_name, base in (
        ("zero_tail", np.zeros_like(approximate)),
        ("mean_tail", approximate),
    ):
        estimates[f"router_{tail_name}"] = gather_by_order(
            contributions, base, router_order
        )
        estimates[f"expected_norm_{tail_name}"] = gather_by_order(
            contributions, base, expected_order
        )
        estimates[f"oracle_norm_{tail_name}"] = gather_by_order(
            contributions, base, oracle_norm_order
        )
        estimates[f"oracle_greedy_{tail_name}"] = greedy_oracle(
            contributions, base, teacher, device, args.greedy_batch
        )

    exact_expert_payload_per_token = 8.844 * (1 << 30)
    methods: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for method, ladder in estimates.items():
        rows: list[dict[str, object]] = []
        for budget in ROUTE_BUDGETS:
            cosine = pair_cosine(ladder[budget], teacher)
            row = {
                "exact_routes": budget,
                "exact_fraction": budget / data.top_k,
                "projected_exact_expert_bytes_per_token": (
                    exact_expert_payload_per_token * budget / data.top_k
                ),
                "routed_aggregate_cosine": summarize(cosine),
            }
            rows.append(row)
            csv_rows.append(
                {
                    "method": method,
                    "exact_routes": budget,
                    "exact_fraction": budget / data.top_k,
                    "mean_cosine": float(cosine.mean()),
                    "p05_cosine": float(np.quantile(cosine, 0.05)),
                    "p01_cosine": float(np.quantile(cosine, 0.01)),
                    "projected_exact_expert_gib_per_token": (
                        8.844 * budget / data.top_k
                    ),
                }
            )
        methods[method] = {
            "selection": (
                "native router order" if method.startswith("router_")
                else "router weight times calibration mean response norm"
                if method.startswith("expected_norm_")
                else "held-out exact contribution norm (oracle)"
                if method.startswith("oracle_norm_")
                else "held-out greedy cosine improvement (oracle)"
            ),
            "tail": "zero" if method.endswith("zero_tail") else "calibration expert mean",
            "ladder": rows,
        }

    result = {
        "schema": "kimi-k3-layer01-sparse-exact-routes-v1",
        "status": "EXPLORATORY",
        "capture": str(args.capture),
        "teacher": str(args.teacher),
        "response_directory": str(args.response_directory),
        "model_layer": data.model_layer,
        "dimension": dimension,
        "top_k": data.top_k,
        "validation_tokens": int(validation_count),
        "full_model_exact_expert_payload_gib_per_token": 8.844,
        "warning": "Layer-one directional agreement is not final-logit quality.",
        "methods": methods,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)
    best_non_oracle = max(
        (row for row in csv_rows if not row["method"].startswith("oracle_") and row["exact_routes"] == 8),
        key=lambda row: row["mean_cosine"],
    )
    print(
        f"best-non-oracle-k8={best_non_oracle['method']} "
        f"mean={best_non_oracle['mean_cosine']:.9f} "
        f"p05={best_non_oracle['p05_cosine']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

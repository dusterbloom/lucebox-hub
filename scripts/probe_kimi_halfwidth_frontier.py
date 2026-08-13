#!/usr/bin/env python3
"""Compare a published half-width Kimi expert bank with full-width refinements."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, quants

from probe_kimi_response_atlas import (
    pair_cosine,
    read_expert_responses,
    response_path,
    summarize,
)
from train_kimi_panel_directional import load_data


EXPERT_COUNT = 896
FULL_EXPERT_WIDTH = 3072
HALF_EXPERT_WIDTH = 1536
MODEL_MOE_LAYERS = 92
FULL_ROUTED_GIB_PER_TOKEN = 8.844
REFINEMENT_BUDGETS = (0, 4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("halfwidth_shard", type=Path)
    parser.add_argument("teacher_spine_shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("full_response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--rms-epsilon", type=float, default=1.0e-6)
    return parser.parse_args()


def dequantize(data: np.ndarray, tensor_type: object) -> torch.Tensor:
    return torch.from_numpy(
        quants.dequantize(np.ascontiguousarray(data), tensor_type)
    )


def situ_expert(
    z: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    gate_value = z @ gate.T
    up_value = z @ up.T
    nonlinear = 4.0 * torch.tanh(gate_value / 4.0) * torch.sigmoid(gate_value)
    linear = 25.0 * torch.tanh(up_value / 25.0)
    return (nonlinear * linear) @ down.T


def inverse_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    values = np.broadcast_to(np.arange(order.shape[1]), order.shape)
    np.put_along_axis(ranks, order, values, axis=1)
    return ranks


def relative_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left - right, axis=1) / np.maximum(
        np.linalg.norm(right, axis=1), 1.0e-30
    )


def pair_summary(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    return {
        "cosine": summarize(pair_cosine(left, right)),
        "relative_l2": summarize(relative_l2(left, right)),
    }


def metric_bundle(
    estimate: np.ndarray,
    teacher: np.ndarray,
    gamma: torch.Tensor,
    projection: torch.Tensor,
    epsilon: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    norm_cosines: list[np.ndarray] = []
    norm_relative: list[np.ndarray] = []
    projection_cosines: list[np.ndarray] = []
    projection_relative: list[np.ndarray] = []
    for begin in range(0, teacher.shape[0], batch_size):
        end = min(teacher.shape[0], begin + batch_size)
        candidate = torch.from_numpy(estimate[begin:end]).to(device)
        target = torch.from_numpy(teacher[begin:end]).to(device)
        candidate_norm = (
            candidate
            * torch.rsqrt(candidate.square().mean(dim=1, keepdim=True) + epsilon)
            * gamma
        )
        target_norm = (
            target
            * torch.rsqrt(target.square().mean(dim=1, keepdim=True) + epsilon)
            * gamma
        )
        norm_cosines.append(
            torch.nn.functional.cosine_similarity(
                candidate_norm, target_norm, dim=1
            ).cpu().numpy()
        )
        norm_relative.append(
            (
                torch.linalg.vector_norm(candidate_norm - target_norm, dim=1)
                / torch.linalg.vector_norm(target_norm, dim=1).clamp_min(1.0e-30)
            ).cpu().numpy()
        )
        candidate_projection = candidate_norm @ projection.T
        target_projection = target_norm @ projection.T
        projection_cosines.append(
            torch.nn.functional.cosine_similarity(
                candidate_projection, target_projection, dim=1
            ).cpu().numpy()
        )
        projection_relative.append(
            (
                torch.linalg.vector_norm(
                    candidate_projection - target_projection, dim=1
                )
                / torch.linalg.vector_norm(
                    target_projection, dim=1
                ).clamp_min(1.0e-30)
            ).cpu().numpy()
        )
    return {
        "routed_aggregate": pair_summary(estimate, teacher),
        "post_rmsnorm": {
            "cosine": summarize(np.concatenate(norm_cosines)),
            "relative_l2": summarize(np.concatenate(norm_relative)),
        },
        "post_shared_up_projection": {
            "cosine": summarize(np.concatenate(projection_cosines)),
            "relative_l2": summarize(np.concatenate(projection_relative)),
        },
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    data = load_data(args.capture, args.teacher)
    if data.dimension != 3584 or data.top_k != 16:
        raise ValueError("unexpected Kimi routed shape")
    validation_count = data.validation_indices.size
    validation_mask = np.zeros(data.latent.shape[0], dtype=bool)
    validation_mask[data.validation_indices] = True
    validation_position = np.full(data.latent.shape[0], -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(validation_count)

    reader = GGUFReader(args.halfwidth_shard, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate_tensor = tensors[f"blk.{args.layer}.ffn_gate_exps.weight"]
    up_tensor = tensors[f"blk.{args.layer}.ffn_up_exps.weight"]
    down_tensor = tensors[f"blk.{args.layer}.ffn_down_exps.weight"]
    if gate_tensor.data.shape != (EXPERT_COUNT, HALF_EXPERT_WIDTH, 700):
        raise ValueError(f"unexpected half-width gate shape: {gate_tensor.data.shape}")
    if up_tensor.data.shape != gate_tensor.data.shape:
        raise ValueError("half-width up shape disagrees")
    if down_tensor.data.shape != (EXPERT_COUNT, data.dimension, 300):
        raise ValueError(f"unexpected half-width down shape: {down_tensor.data.shape}")
    spine_reader = GGUFReader(args.teacher_spine_shard, "r")
    spine_tensors = {tensor.name: tensor for tensor in spine_reader.tensors}
    norm_tensor = spine_tensors[f"blk.{args.layer}.ffn_routed_norm.weight"]
    projection_tensor = spine_tensors[f"blk.{args.layer}.ffn_routed_up.weight"]

    device = torch.device(args.device)
    half_validation = np.zeros(
        (validation_count, data.top_k, data.dimension), dtype=np.float32
    )
    full_validation = np.zeros_like(half_validation)
    calibration_delta_norm = np.zeros(EXPERT_COUNT, dtype=np.float32)
    calibration_full_norm = np.zeros(EXPERT_COUNT, dtype=np.float32)

    print("[halfwidth] evaluating all real layer-one routes", flush=True)
    for expert in range(EXPERT_COUNT):
        records, full_outputs = read_expert_responses(
            response_path(args.full_response_directory, expert),
            data.model_layer,
            expert,
            data.dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        weights = records["router_weight"].astype(np.float32, copy=False)
        z = torch.from_numpy(data.latent[token_indices]).to(device)
        gate = dequantize(gate_tensor.data[expert], gate_tensor.tensor_type).to(device)
        up = dequantize(up_tensor.data[expert], up_tensor.tensor_type).to(device)
        down = dequantize(down_tensor.data[expert], down_tensor.tensor_type).to(device)
        half_outputs = situ_expert(z, gate, up, down).cpu().numpy()
        del z, gate, up, down

        calibration_rows = np.flatnonzero(~validation_mask[token_indices])
        validation_rows = np.flatnonzero(validation_mask[token_indices])
        delta_calibration = full_outputs[calibration_rows] - half_outputs[calibration_rows]
        calibration_delta_norm[expert] = np.linalg.norm(
            delta_calibration, axis=1
        ).mean()
        calibration_full_norm[expert] = np.linalg.norm(
            full_outputs[calibration_rows], axis=1
        ).mean()
        if validation_rows.size:
            positions = validation_position[token_indices[validation_rows]]
            route_ranks = ranks[validation_rows]
            route_weights = weights[validation_rows, None]
            half_validation[positions, route_ranks] = (
                route_weights * half_outputs[validation_rows]
            )
            full_validation[positions, route_ranks] = (
                route_weights * full_outputs[validation_rows]
            )
        if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
            print(
                f"[halfwidth] experts={expert + 1}/{EXPERT_COUNT} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    teacher = data.teacher[data.validation_indices]
    full_reconstructed = full_validation.sum(axis=1)
    if not np.allclose(full_reconstructed, teacher, rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("full response records no longer reconstruct teacher")
    half_aggregate = half_validation.sum(axis=1)
    delta = full_validation - half_validation
    ids = data.expert_ids[data.validation_indices]
    weights = data.router_weights[data.validation_indices]
    causal_score = weights * calibration_delta_norm[ids]
    causal_rank = inverse_order(
        np.argsort(-causal_score, axis=1, kind="stable")
    )
    actual_delta_norm = np.linalg.norm(delta, axis=2)
    diagnostic_rank = inverse_order(
        np.argsort(-actual_delta_norm, axis=1, kind="stable")
    )
    full_contribution_score = weights * calibration_full_norm[ids]
    contribution_rank = inverse_order(
        np.argsort(-full_contribution_score, axis=1, kind="stable")
    )

    policies = {
        "calibration_delta_refinement": (
            causal_rank,
            "router weight times calibration mean full-minus-half output norm",
            False,
        ),
        "calibration_full_contribution_refinement": (
            contribution_rank,
            "router weight times calibration mean full output norm",
            False,
        ),
        "heldout_delta_norm_refinement": (
            diagnostic_rank,
            "actual held-out weighted full-minus-half output norm",
            True,
        ),
    }
    gamma = torch.from_numpy(np.asarray(norm_tensor.data, dtype=np.float32)).to(device)
    projection = dequantize(
        projection_tensor.data, projection_tensor.tensor_type
    ).to(device)
    methods: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for name, (rank, description, diagnostic) in policies.items():
        ladder: list[dict[str, object]] = []
        for budget in REFINEMENT_BUDGETS:
            selected = rank < budget
            estimate = half_aggregate + (delta * selected[:, :, None]).sum(axis=1)
            metrics = metric_bundle(
                estimate, teacher, gamma, projection,
                args.rms_epsilon, device, args.metric_batch,
            )
            payload_fraction = 0.5 + 0.5 * budget / data.top_k
            row = {
                "fullwidth_refinements": budget,
                "halfwidth_routes": data.top_k - budget,
                "routed_payload_fraction": payload_fraction,
                "nominal_routed_gib_per_token": (
                    FULL_ROUTED_GIB_PER_TOKEN * payload_fraction
                ),
                "metrics_against_native_fullwidth_teacher": metrics,
            }
            ladder.append(row)
            csv_rows.append(
                {
                    "method": name,
                    "fullwidth_refinements": budget,
                    "payload_fraction": payload_fraction,
                    "nominal_gib_per_token": FULL_ROUTED_GIB_PER_TOKEN * payload_fraction,
                    "mean_cosine": metrics["routed_aggregate"]["cosine"]["mean"],
                    "p05_cosine": metrics["routed_aggregate"]["cosine"]["p05"],
                    "post_norm_mean_cosine": metrics["post_rmsnorm"]["cosine"]["mean"],
                    "post_projection_mean_cosine": metrics[
                        "post_shared_up_projection"
                    ]["cosine"]["mean"],
                    "diagnostic_uses_heldout_answers": diagnostic,
                }
            )
        methods[name] = {
            "description": description,
            "diagnostic_uses_heldout_answers": diagnostic,
            "ladder": ladder,
        }

    result = {
        "schema": "kimi-k3-layer01-halfwidth-refinement-frontier-v1",
        "status": "EXPLORATORY",
        "source_repository": "vcruz305/Kimi-K3-GGUF",
        "source_revision": "6c27987818cb244ef8b57ea5ed14d31fe9482c27",
        "model_layer": data.model_layer,
        "halfwidth_shard": str(args.halfwidth_shard),
        "teacher_spine_shard": str(args.teacher_spine_shard),
        "capture": str(args.capture),
        "teacher": str(args.teacher),
        "calibration_tokens": int(data.latent.shape[0] - validation_count),
        "validation_tokens": int(validation_count),
        "sequence_disjoint_validation": True,
        "expert_widths": {
            "full_teacher": FULL_EXPERT_WIDTH,
            "published_halfwidth": HALF_EXPERT_WIDTH,
        },
        "halfwidth_all_routes_against_teacher": pair_summary(
            half_aggregate, teacher
        ),
        "methods": methods,
        "warnings": [
            "The published half-width artifact is a different model and cannot preserve full-width bit exactness.",
            "The external expert outputs use Python-dequantized IQ1_S weights while the teacher responses use Lucebox native quantized kernels.",
            "This isolates the routed expert boundary on the same latent inputs and routes; it does not reproduce the external model's own upstream hidden-state distribution.",
            "Routed RMS normalization and the shared up projection come from the full-width teacher, so only the routed expert provider changes.",
            "Layer-one directional agreement is not final-logit quality.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(csv_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(csv_rows)
    print("[halfwidth] refinement frontier", flush=True)
    for row in csv_rows:
        if row["method"] == "calibration_delta_refinement":
            print(
                f"[halfwidth] refinements={row['fullwidth_refinements']} "
                f"mean={row['mean_cosine']:.9f} p05={row['p05_cosine']:.9f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

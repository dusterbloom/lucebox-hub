#!/usr/bin/env python3
"""Measure whole-route versus neuron-slab allocation on real Kimi experts.

The experiment keeps the original router and uses the same sequence-disjoint
layer-one capture as the registered panel probe.  An IQ1_S expert with 3,072
internal neurons is split into twelve independently decodable 256-neuron
slabs.  At matched bytes, it compares complete experts, a static equal-width
allocation, a calibration-only adaptive allocation, and a held-out diagnostic
that ranks the actual residual norms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
MODEL_MOE_LAYERS = 92
ORIGINAL_EXPERT_WIDTH = 3072
DEFAULT_SLAB_SIZE = 256
SLAB_BUDGETS = (48, 72, 96, 120, 144, 168, 192)
WHOLE_ROUTE_SLAB_BUDGETS = (48, 96, 144, 192)
FULL_ROUTED_GIB_PER_TOKEN = 8.844


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "shard",
        type=Path,
        help=(
            "any GGUF shard in the model directory (or the directory itself); "
            "the requested layer tensors are resolved across all sibling shards"
        ),
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--fit-state", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument(
        "--exact-fallback-uncalibrated",
        action="store_true",
        help=(
            "keep validation routes through experts absent from calibration "
            "exact and report their additional byte cost"
        ),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--slab-size", type=int, default=DEFAULT_SLAB_SIZE)
    parser.add_argument("--audit-experts", type=int, default=16)
    parser.add_argument("--smoke-expert", type=int, default=-1)
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--rms-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def inverse_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    values = np.broadcast_to(np.arange(order.shape[1]), order.shape)
    np.put_along_axis(ranks, order, values, axis=1)
    return ranks


def relative_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(left - right, axis=1)
    denominator = np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1.0e-30)


def summarize_pair(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    return {
        "cosine": summarize(pair_cosine(left, right)),
        "relative_l2": summarize(relative_l2(left, right)),
    }


def dequantize_part(data: np.ndarray, tensor_type: object) -> torch.Tensor:
    contiguous = np.ascontiguousarray(data)
    return torch.from_numpy(quants.dequantize(contiguous, tensor_type))


def resolve_layer_tensors(
    shard_or_directory: Path,
    layer: int,
) -> tuple[dict[str, object], dict[str, Path], list[GGUFReader]]:
    """Find one K3 layer's tensors in its split GGUF model directory.

    The IQ1_S model distributes routed layers across fourteen files.  The
    original layer-one probe happened to work when given shard 1, but later
    layers are not guaranteed to live there.  Scan only GGUF headers and keep
    the exact source file for every tensor in the result metadata.
    """
    model_directory = (
        shard_or_directory if shard_or_directory.is_dir() else shard_or_directory.parent
    )
    shards = sorted(model_directory.glob("*.gguf"))
    if not shards:
        raise FileNotFoundError(f"no GGUF shards in {model_directory}")
    names = (
        f"blk.{layer}.ffn_gate_exps.weight",
        f"blk.{layer}.ffn_up_exps.weight",
        f"blk.{layer}.ffn_down_exps.weight",
        f"blk.{layer}.ffn_routed_norm.weight",
        f"blk.{layer}.ffn_routed_up.weight",
    )
    wanted = set(names)
    tensors: dict[str, object] = {}
    sources: dict[str, Path] = {}
    readers: list[GGUFReader] = []
    for shard in shards:
        reader = GGUFReader(shard, "r")
        readers.append(reader)
        for tensor in reader.tensors:
            if tensor.name not in wanted:
                continue
            if tensor.name in tensors:
                raise ValueError(f"tensor {tensor.name} occurs in multiple shards")
            tensors[tensor.name] = tensor
            sources[tensor.name] = shard
        if len(tensors) == len(names):
            break
    missing = [name for name in names if name not in tensors]
    if missing:
        raise KeyError(
            f"layer {layer} tensors missing from {model_directory}: {missing}"
        )
    return tensors, sources, readers


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


def eval_slab(
    z: torch.Tensor,
    gate_tensor: object,
    up_tensor: object,
    down_tensor: object,
    expert: int,
    slab: int,
    slab_size: int,
    device: torch.device,
) -> np.ndarray:
    begin = slab * slab_size
    end = begin + slab_size
    block_bytes = slab_size * 50 // 256
    byte_begin = slab * block_bytes
    byte_end = byte_begin + block_bytes
    gate = dequantize_part(
        gate_tensor.data[expert, begin:end], gate_tensor.tensor_type
    ).to(device)
    up = dequantize_part(
        up_tensor.data[expert, begin:end], up_tensor.tensor_type
    ).to(device)
    down = dequantize_part(
        down_tensor.data[expert, :, byte_begin:byte_end], down_tensor.tensor_type
    ).to(device)
    output = situ_expert(z, gate, up, down).cpu().numpy()
    del gate, up, down
    return output


def eval_full_dequantized(
    z: torch.Tensor,
    gate_tensor: object,
    up_tensor: object,
    down_tensor: object,
    expert: int,
    device: torch.device,
) -> np.ndarray:
    gate = dequantize_part(
        gate_tensor.data[expert], gate_tensor.tensor_type
    ).to(device)
    up = dequantize_part(
        up_tensor.data[expert], up_tensor.tensor_type
    ).to(device)
    down = dequantize_part(
        down_tensor.data[expert], down_tensor.tensor_type
    ).to(device)
    output = situ_expert(z, gate, up, down).cpu().numpy()
    del gate, up, down
    return output


def metric_bundle(
    estimate: np.ndarray,
    teacher: np.ndarray,
    gamma: torch.Tensor,
    projection: torch.Tensor,
    epsilon: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    latent = summarize_pair(estimate, teacher)
    post_norm_cosine: list[np.ndarray] = []
    post_norm_relative_l2: list[np.ndarray] = []
    post_projection_cosine: list[np.ndarray] = []
    post_projection_relative_l2: list[np.ndarray] = []
    with torch.no_grad():
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
            post_norm_cosine.append(
                torch.nn.functional.cosine_similarity(
                    candidate_norm, target_norm, dim=1
                ).cpu().numpy()
            )
            post_norm_relative_l2.append(
                (
                    torch.linalg.vector_norm(candidate_norm - target_norm, dim=1)
                    / torch.linalg.vector_norm(target_norm, dim=1).clamp_min(1.0e-30)
                ).cpu().numpy()
            )
            candidate_projected = candidate_norm @ projection.T
            target_projected = target_norm @ projection.T
            post_projection_cosine.append(
                torch.nn.functional.cosine_similarity(
                    candidate_projected, target_projected, dim=1
                ).cpu().numpy()
            )
            post_projection_relative_l2.append(
                (
                    torch.linalg.vector_norm(
                        candidate_projected - target_projected, dim=1
                    )
                    / torch.linalg.vector_norm(
                        target_projected, dim=1
                    ).clamp_min(1.0e-30)
                ).cpu().numpy()
            )
            del candidate, target, candidate_norm, target_norm
            del candidate_projected, target_projected
    return {
        "routed_aggregate": latent,
        "post_rmsnorm": {
            "cosine": summarize(np.concatenate(post_norm_cosine)),
            "relative_l2": summarize(np.concatenate(post_norm_relative_l2)),
        },
        "post_shared_up_projection": {
            "cosine": summarize(np.concatenate(post_projection_cosine)),
            "relative_l2": summarize(
                np.concatenate(post_projection_relative_l2)
            ),
        },
    }


def add_corrections(
    estimates: dict[int, np.ndarray],
    positions: np.ndarray,
    corrections: np.ndarray,
    selection_ranks: np.ndarray,
    budgets: tuple[int, ...],
) -> None:
    for budget in budgets:
        selected = selection_ranks < budget
        if np.any(selected):
            estimates[budget][positions[selected]] += corrections[selected]


def make_base(
    expert_means: np.ndarray,
    expert_ids: np.ndarray,
    router_weights: np.ndarray,
) -> np.ndarray:
    base = np.zeros(
        (expert_ids.shape[0], expert_means.shape[1]), dtype=np.float32
    )
    for rank in range(expert_ids.shape[1]):
        base += (
            router_weights[:, rank, None]
            * expert_means[expert_ids[:, rank]]
        )
    return base


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    if (
        args.slab_size <= 0
        or ORIGINAL_EXPERT_WIDTH % args.slab_size
        or args.slab_size % 256
    ):
        raise ValueError("slab size must be a positive multiple of 256 dividing 3072")
    slab_count = ORIGINAL_EXPERT_WIDTH // args.slab_size
    active_slab_count = 16 * slab_count
    if slab_count != 12 or active_slab_count != 192:
        raise ValueError("the registered budget ladder currently requires 12 slabs")
    if args.rms_epsilon <= 0:
        raise ValueError("RMS epsilon must be positive")

    data = load_data(args.capture, args.teacher)
    if data.top_k != 16 or data.dimension != 3584:
        raise ValueError("unexpected Kimi routed shape")
    validation_count = data.validation_indices.size
    validation_mask = np.zeros(data.latent.shape[0], dtype=bool)
    validation_mask[data.validation_indices] = True
    validation_position = np.full(data.latent.shape[0], -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(validation_count)
    calibration_route_counts = np.bincount(
        data.expert_ids[~validation_mask].reshape(-1), minlength=EXPERT_COUNT
    )
    validation_route_counts = np.bincount(
        data.expert_ids[validation_mask].reshape(-1), minlength=EXPERT_COUNT
    )
    capture_calibrated_experts = calibration_route_counts > 0
    missing_calibration_experts = int(
        np.count_nonzero(~capture_calibrated_experts)
    )
    if missing_calibration_experts and not args.exact_fallback_uncalibrated:
        raise ValueError(
            f"capture has {missing_calibration_experts} experts without "
            "calibration routes; use more calibration data or explicitly "
            "enable exact fallback"
        )

    tensors, tensor_sources, _tensor_readers = resolve_layer_tensors(
        args.shard, args.layer
    )
    gate_tensor = tensors[f"blk.{args.layer}.ffn_gate_exps.weight"]
    up_tensor = tensors[f"blk.{args.layer}.ffn_up_exps.weight"]
    down_tensor = tensors[f"blk.{args.layer}.ffn_down_exps.weight"]
    norm_tensor = tensors[f"blk.{args.layer}.ffn_routed_norm.weight"]
    projection_tensor = tensors[f"blk.{args.layer}.ffn_routed_up.weight"]
    expected_gate_shape = (EXPERT_COUNT, ORIGINAL_EXPERT_WIDTH, 700)
    expected_down_shape = (EXPERT_COUNT, data.dimension, 600)
    if gate_tensor.data.shape != expected_gate_shape:
        raise ValueError(f"unexpected gate bytes: {gate_tensor.data.shape}")
    if up_tensor.data.shape != expected_gate_shape:
        raise ValueError(f"unexpected up bytes: {up_tensor.data.shape}")
    if down_tensor.data.shape != expected_down_shape:
        raise ValueError(f"unexpected down bytes: {down_tensor.data.shape}")

    device = torch.device(args.device)
    if args.smoke_expert >= 0:
        if args.smoke_expert >= EXPERT_COUNT:
            raise ValueError("smoke expert is out of range")
        records, native_outputs = read_expert_responses(
            response_path(args.response_directory, args.smoke_expert),
            data.model_layer,
            args.smoke_expert,
            data.dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        rows = np.flatnonzero(validation_mask[token_indices])[:32]
        if rows.size == 0:
            raise ValueError("smoke expert has no validation routes")
        z = torch.from_numpy(data.latent[token_indices[rows]]).to(device)
        slab_sum = np.zeros((rows.size, data.dimension), dtype=np.float32)
        for slab in range(slab_count):
            slab_sum += eval_slab(
                z, gate_tensor, up_tensor, down_tensor,
                args.smoke_expert, slab, args.slab_size, device,
            )
        full = eval_full_dequantized(
            z, gate_tensor, up_tensor, down_tensor,
            args.smoke_expert, device,
        )
        smoke = {
            "schema": "kimi-k3-neuron-slab-smoke-v1",
            "expert": args.smoke_expert,
            "routes": int(rows.size),
            "slab_sum_vs_full_dequantized": summarize_pair(slab_sum, full),
            "full_dequantized_vs_native": summarize_pair(
                full, native_outputs[rows]
            ),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(smoke, indent=2) + "\n")
        print(json.dumps(smoke, indent=2), flush=True)
        return 0

    expected_state_shapes = {
        "slab_means": (EXPERT_COUNT, slab_count, data.dimension),
        "slab_expected_norm": (EXPERT_COUNT, slab_count),
        "slab_expected_residual_norm": (EXPERT_COUNT, slab_count),
        "native_means": (EXPERT_COUNT, data.dimension),
        "native_expected_norm": (EXPERT_COUNT,),
    }
    calibrated_state_shape = (EXPERT_COUNT,)
    if args.fit_state and args.fit_state.exists():
        print(f"[slabs] loading calibration state {args.fit_state}", flush=True)
        with np.load(args.fit_state, allow_pickle=False) as state:
            state_fields = set(state.files)
            expected_fields = set(expected_state_shapes)
            if state_fields not in (
                expected_fields,
                expected_fields | {"calibrated_experts"},
            ):
                raise ValueError("calibration state fields disagree")
            loaded = {name: state[name] for name in expected_state_shapes}
            loaded_calibrated = (
                state["calibrated_experts"]
                if "calibrated_experts" in state.files
                else np.ones(calibrated_state_shape, dtype=np.uint8)
            )
        for name, shape in expected_state_shapes.items():
            if loaded[name].shape != shape or loaded[name].dtype != np.float32:
                raise ValueError(f"calibration state shape/type disagrees: {name}")
            if not np.isfinite(loaded[name]).all():
                raise ValueError(f"calibration state is non-finite: {name}")
        slab_means = loaded["slab_means"]
        slab_expected_norm = loaded["slab_expected_norm"]
        slab_expected_residual_norm = loaded["slab_expected_residual_norm"]
        native_means = loaded["native_means"]
        native_expected_norm = loaded["native_expected_norm"]
        if (
            loaded_calibrated.shape != calibrated_state_shape
            or loaded_calibrated.dtype != np.uint8
            or not np.all((loaded_calibrated == 0) | (loaded_calibrated == 1))
        ):
            raise ValueError("calibrated expert mask disagrees")
        calibrated_experts = loaded_calibrated.astype(bool, copy=False)
        if not np.array_equal(
            calibrated_experts, capture_calibrated_experts
        ):
            raise ValueError(
                "calibration-state expert coverage disagrees with capture"
            )
    else:
        slab_means = np.zeros(
            expected_state_shapes["slab_means"], dtype=np.float32
        )
        slab_expected_norm = np.zeros(
            expected_state_shapes["slab_expected_norm"], dtype=np.float32
        )
        slab_expected_residual_norm = np.zeros_like(slab_expected_norm)
        native_means = np.zeros(
            expected_state_shapes["native_means"], dtype=np.float32
        )
        native_expected_norm = np.zeros(
            expected_state_shapes["native_expected_norm"], dtype=np.float32
        )
        calibrated_experts = capture_calibrated_experts

        print("[slabs] pass 1/3: calibration means and importance", flush=True)
        for expert in range(EXPERT_COUNT):
            if not calibrated_experts[expert]:
                continue
            records, native_outputs = read_expert_responses(
                response_path(args.response_directory, expert),
                data.model_layer,
                expert,
                data.dimension,
            )
            token_indices = records["token_index"].astype(np.int64, copy=False)
            calibration_rows = np.flatnonzero(~validation_mask[token_indices])
            if calibration_rows.size == 0:
                raise ValueError(f"expert {expert} has no calibration routes")
            native_calibration = native_outputs[calibration_rows]
            native_means[expert] = native_calibration.mean(
                axis=0, dtype=np.float64
            ).astype(np.float32)
            native_expected_norm[expert] = np.linalg.norm(
                native_calibration, axis=1
            ).mean()
            z = torch.from_numpy(
                data.latent[token_indices[calibration_rows]]
            ).to(device)
            for slab in range(slab_count):
                output = eval_slab(
                    z, gate_tensor, up_tensor, down_tensor,
                    expert, slab, args.slab_size, device,
                )
                mean = output.mean(axis=0, dtype=np.float64).astype(np.float32)
                slab_means[expert, slab] = mean
                slab_expected_norm[expert, slab] = np.linalg.norm(
                    output, axis=1
                ).mean()
                slab_expected_residual_norm[expert, slab] = np.linalg.norm(
                    output - mean, axis=1
                ).mean()
            del z
            if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
                print(
                    f"[slabs] calibration experts={expert + 1}/{EXPERT_COUNT} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        if args.fit_state:
            args.fit_state.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.fit_state.with_suffix(args.fit_state.suffix + ".tmp.npz")
            np.savez(
                temporary,
                slab_means=slab_means,
                slab_expected_norm=slab_expected_norm,
                slab_expected_residual_norm=slab_expected_residual_norm,
                native_means=native_means,
                native_expected_norm=native_expected_norm,
                calibrated_experts=calibrated_experts.astype(np.uint8),
            )
            temporary.replace(args.fit_state)
            print(f"[slabs] saved calibration state {args.fit_state}", flush=True)
    if args.calibration_only:
        if not args.fit_state:
            raise ValueError("calibration-only requires --fit-state")
        print("[slabs] calibration-only complete", flush=True)
        return 0

    validation_ids = data.expert_ids[data.validation_indices]
    validation_weights = data.router_weights[data.validation_indices]
    slab_expert_means = slab_means.sum(axis=1)
    slab_base = make_base(
        slab_expert_means, validation_ids, validation_weights
    )
    native_base = make_base(native_means, validation_ids, validation_weights)

    static_order = np.argsort(
        -slab_expected_residual_norm, axis=1, kind="stable"
    )
    static_rank = inverse_order(static_order)
    adaptive_scores = (
        validation_weights[:, :, None]
        * slab_expected_residual_norm[validation_ids]
    )
    adaptive_scores = np.where(
        calibrated_experts[validation_ids, None],
        adaptive_scores,
        -np.inf,
    )
    adaptive_rank = inverse_order(
        np.argsort(
            -adaptive_scores.reshape(validation_count, -1),
            axis=1,
            kind="stable",
        )
    ).reshape(validation_count, data.top_k, slab_count)
    whole_scores = validation_weights * native_expected_norm[validation_ids]
    whole_scores = np.where(
        calibrated_experts[validation_ids], whole_scores, -np.inf
    )
    whole_rank = inverse_order(
        np.argsort(-whole_scores, axis=1, kind="stable")
    )

    static_estimates = {
        budget: slab_base.copy() for budget in SLAB_BUDGETS
    }
    adaptive_estimates = {
        budget: slab_base.copy() for budget in SLAB_BUDGETS
    }
    whole_estimates = {
        budget: native_base.copy() for budget in WHOLE_ROUTE_SLAB_BUDGETS
    }
    dequantized_teacher = np.zeros_like(slab_base)
    exact_fallback_aggregate = np.zeros_like(slab_base)
    actual_residual_norm = np.zeros(
        (validation_count, data.top_k, slab_count), dtype=np.float32
    )
    audit_full_vs_slab_cosines: list[np.ndarray] = []
    audit_full_vs_slab_relative_l2: list[np.ndarray] = []
    audit_full_vs_native_cosines: list[np.ndarray] = []

    print("[slabs] pass 2/3: causal policies and held-out norms", flush=True)
    for expert in range(EXPERT_COUNT):
        if (
            not calibrated_experts[expert]
            and validation_route_counts[expert] == 0
        ):
            continue
        records, native_outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            data.dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        route_ranks = records["rank"].astype(np.int64, copy=False)
        route_weights = records["router_weight"].astype(np.float32, copy=False)
        rows = np.flatnonzero(validation_mask[token_indices])
        if rows.size == 0:
            continue
        positions = validation_position[token_indices[rows]]
        ranks = route_ranks[rows]
        weights = route_weights[rows]
        if not calibrated_experts[expert]:
            exact_contribution = weights[:, None] * native_outputs[rows]
            exact_fallback_aggregate[positions] += exact_contribution
            dequantized_teacher[positions] += exact_contribution
            for estimates in (
                static_estimates, adaptive_estimates, whole_estimates
            ):
                for estimate in estimates.values():
                    estimate[positions] += exact_contribution
            continue
        native_correction = weights[:, None] * (
            native_outputs[rows] - native_means[expert]
        )
        for budget in WHOLE_ROUTE_SLAB_BUDGETS:
            selected = whole_rank[positions, ranks] < budget // slab_count
            if np.any(selected):
                whole_estimates[budget][positions[selected]] += (
                    native_correction[selected]
                )
        z = torch.from_numpy(data.latent[token_indices[rows]]).to(device)
        slab_sum = np.zeros((rows.size, data.dimension), dtype=np.float32)
        for slab in range(slab_count):
            output = eval_slab(
                z, gate_tensor, up_tensor, down_tensor,
                expert, slab, args.slab_size, device,
            )
            slab_sum += output
            weighted_output = weights[:, None] * output
            dequantized_teacher[positions] += weighted_output
            correction = weights[:, None] * (
                output - slab_means[expert, slab]
            )
            actual_residual_norm[positions, ranks, slab] = np.linalg.norm(
                correction, axis=1
            )
            static_ranks = np.full(
                rows.size, static_rank[expert, slab] * data.top_k,
                dtype=np.int64,
            )
            add_corrections(
                static_estimates, positions, correction,
                static_ranks, SLAB_BUDGETS,
            )
            add_corrections(
                adaptive_estimates, positions, correction,
                adaptive_rank[positions, ranks, slab], SLAB_BUDGETS,
            )
        if expert < args.audit_experts:
            full = eval_full_dequantized(
                z, gate_tensor, up_tensor, down_tensor, expert, device
            )
            audit_full_vs_slab_cosines.append(pair_cosine(slab_sum, full))
            audit_full_vs_slab_relative_l2.append(relative_l2(slab_sum, full))
            audit_full_vs_native_cosines.append(
                pair_cosine(full, native_outputs[rows])
            )
        del z
        if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
            print(
                f"[slabs] causal experts={expert + 1}/{EXPERT_COUNT} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    oracle_rank = inverse_order(
        np.argsort(
            -actual_residual_norm.reshape(validation_count, -1),
            axis=1,
            kind="stable",
        )
    ).reshape(validation_count, data.top_k, slab_count)
    oracle_estimates = {
        budget: slab_base.copy() + exact_fallback_aggregate
        for budget in SLAB_BUDGETS
    }

    print("[slabs] pass 3/3: held-out residual-norm diagnostic", flush=True)
    for expert in range(EXPERT_COUNT):
        if not calibrated_experts[expert]:
            continue
        records, _ = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            data.dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        route_ranks = records["rank"].astype(np.int64, copy=False)
        route_weights = records["router_weight"].astype(np.float32, copy=False)
        rows = np.flatnonzero(validation_mask[token_indices])
        if rows.size == 0:
            continue
        positions = validation_position[token_indices[rows]]
        ranks = route_ranks[rows]
        weights = route_weights[rows]
        z = torch.from_numpy(data.latent[token_indices[rows]]).to(device)
        for slab in range(slab_count):
            output = eval_slab(
                z, gate_tensor, up_tensor, down_tensor,
                expert, slab, args.slab_size, device,
            )
            correction = weights[:, None] * (
                output - slab_means[expert, slab]
            )
            add_corrections(
                oracle_estimates, positions, correction,
                oracle_rank[positions, ranks, slab], SLAB_BUDGETS,
            )
        del z
        if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
            print(
                f"[slabs] diagnostic experts={expert + 1}/{EXPERT_COUNT} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    audit_cosine = np.concatenate(audit_full_vs_slab_cosines)
    audit_relative_l2 = np.concatenate(audit_full_vs_slab_relative_l2)
    audit_native = np.concatenate(audit_full_vs_native_cosines)
    if audit_cosine.mean() < 0.99999 or audit_relative_l2.mean() > 1.0e-3:
        raise ValueError("the twelve-slab reconstruction failed its numerical gate")

    gamma = torch.from_numpy(
        np.asarray(norm_tensor.data, dtype=np.float32).copy()
    ).to(device)
    projection = dequantize_part(
        projection_tensor.data, projection_tensor.tensor_type
    ).to(device)
    teacher = data.teacher[data.validation_indices]
    methods: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    fallback_routes = int(
        validation_route_counts[~calibrated_experts].sum()
    )
    fallback_route_mask = ~calibrated_experts[validation_ids]
    fallback_tokens = int(np.count_nonzero(fallback_route_mask.any(axis=1)))
    fallback_slab_equivalents_per_token = (
        fallback_routes * slab_count / max(1, validation_count)
    )
    minimum_active_expert_calibration_hits = calibration_route_counts[
        validation_ids
    ].min(axis=1)

    def register(
        name: str,
        description: str,
        estimates: dict[int, np.ndarray],
        diagnostic: bool,
    ) -> None:
        ladder: list[dict[str, object]] = []
        for budget, estimate in sorted(estimates.items()):
            metrics = metric_bundle(
                estimate, teacher, gamma, projection,
                args.rms_epsilon, device, args.metric_batch,
            )
            row = {
                "slab_budget": budget,
                "active_slabs": active_slab_count,
                "routed_payload_fraction": budget / active_slab_count,
                "measured_payload_fraction_with_exact_fallback": (
                    budget + fallback_slab_equivalents_per_token
                ) / active_slab_count,
                "nominal_routed_gib_per_token": (
                    FULL_ROUTED_GIB_PER_TOKEN * budget / active_slab_count
                ),
                "metrics_against_native_teacher": metrics,
                "routed_aggregate_by_minimum_active_expert_calibration_hits": {
                    str(threshold): {
                        "tokens": int(np.count_nonzero(mask)),
                        "metrics": summarize_pair(
                            estimate[mask], teacher[mask]
                        )["cosine"],
                    }
                    for threshold in (1, 5, 10, 30, 100)
                    if np.any(
                        mask := (
                            minimum_active_expert_calibration_hits >= threshold
                        )
                    )
                },
            }
            if name == "whole_expert_expected_mean_tail":
                row["complete_experts"] = budget // slab_count
            if name == "uniform_static_slab_mean_tail":
                row["slabs_per_active_expert"] = budget // data.top_k
            ladder.append(row)
            csv_rows.append(
                {
                    "method": name,
                    "slab_budget": budget,
                    "payload_fraction": budget / active_slab_count,
                    "payload_fraction_with_exact_fallback": (
                        budget + fallback_slab_equivalents_per_token
                    ) / active_slab_count,
                    "nominal_gib_per_token": (
                        FULL_ROUTED_GIB_PER_TOKEN * budget / active_slab_count
                    ),
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

    register(
        "whole_expert_expected_mean_tail",
        "Complete experts ranked by router weight times calibration mean native output norm; omitted experts use their calibration mean.",
        whole_estimates,
        False,
    )
    register(
        "uniform_static_slab_mean_tail",
        "Every active expert receives the same slab count; per-expert slabs are ranked by calibration mean residual norm.",
        static_estimates,
        False,
    )
    register(
        "adaptive_expected_residual_slab_mean_tail",
        "The global token budget is allocated by router weight times calibration mean slab residual norm.",
        adaptive_estimates,
        False,
    )
    register(
        "heldout_residual_norm_slab_mean_tail",
        "Diagnostic selection ranks the unknown held-out weighted slab residual norm; it is not a deployable oracle for direction.",
        oracle_estimates,
        True,
    )

    bytes_per_component_slab = args.slab_size * 3584 * 50 // 256
    bytes_per_slab = 3 * bytes_per_component_slab
    mean_tail_bf16_bytes = (
        MODEL_MOE_LAYERS * EXPERT_COUNT * slab_count * data.dimension * 2
    )
    result = {
        "schema": "kimi-k3-neuron-slab-frontier-v2",
        "status": "EXPLORATORY",
        "model_layer": data.model_layer,
        "capture": str(args.capture),
        "teacher": str(args.teacher),
        "shard": str(args.shard),
        "tensor_sources": {
            name: str(path) for name, path in sorted(tensor_sources.items())
        },
        "response_directory": str(args.response_directory),
        "calibration_tokens": int(data.latent.shape[0] - validation_count),
        "validation_tokens": int(validation_count),
        "sequence_disjoint_validation": True,
        "expert_coverage": {
            "calibrated_experts": int(np.count_nonzero(calibrated_experts)),
            "uncalibrated_experts": int(np.count_nonzero(~calibrated_experts)),
            "exact_fallback_enabled": args.exact_fallback_uncalibrated,
            "validation_exact_fallback_routes": fallback_routes,
            "validation_exact_fallback_tokens": fallback_tokens,
            "validation_routes": int(validation_count * data.top_k),
            "mean_extra_slab_equivalents_per_token": (
                fallback_slab_equivalents_per_token
            ),
            "validation_tokens_by_minimum_active_expert_calibration_hits": {
                str(threshold): int(
                    np.count_nonzero(
                        minimum_active_expert_calibration_hits >= threshold
                    )
                )
                for threshold in (1, 5, 10, 30, 100)
            },
        },
        "layout": {
            "expert_width": ORIGINAL_EXPERT_WIDTH,
            "slab_size": args.slab_size,
            "slabs_per_expert": slab_count,
            "active_experts": data.top_k,
            "active_slabs": active_slab_count,
            "iq1_s_bytes_per_component_slab": bytes_per_component_slab,
            "iq1_s_bytes_per_complete_slab": bytes_per_slab,
            "iq1_s_bytes_per_complete_expert": bytes_per_slab * slab_count,
            "all_layer_bf16_slab_mean_tail_bytes": mean_tail_bf16_bytes,
            "all_layer_bf16_slab_mean_tail_gib": mean_tail_bf16_bytes / (1 << 30),
        },
        "reconstruction_audit": {
            "experts": args.audit_experts,
            "twelve_slab_sum_vs_one_full_dequantized_matmul": {
                "cosine": summarize(audit_cosine),
                "relative_l2": summarize(audit_relative_l2),
            },
            "one_full_dequantized_matmul_vs_native_quantized_kernel": {
                "cosine": summarize(audit_native),
            },
            "all_validation_slab_sum_vs_native_teacher": summarize_pair(
                dequantized_teacher, teacher
            ),
        },
        "selection_importance": {
            "causal": "router weight times calibration expected slab residual norm",
            "static": "calibration expected slab residual norm within each expert",
            "heldout_diagnostic": "actual weighted slab residual norm",
        },
        "methods": methods,
        "warnings": [
            "Layer-one directional agreement is not final-logit quality.",
            "Python dequantization and split reductions need not be bit-identical to the native quantized GPU kernel.",
            "Physical progressive reads require an expert-slab sidecar or repacked bank; standard GGUF tensor layout is not the intended serving layout.",
            "The held-out residual-norm ranking is diagnostic and is not an oracle for cosine-optimal subset selection.",
            "The slab mean tail adds storage and memory traffic beyond the routed IQ1_S payload; its all-layer BF16 size is reported explicitly.",
            "Experts absent from calibration remain exact on validation when explicit fallback is enabled; their additional routed bytes are reported and are not included in the nominal budget.",
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

    matched = [
        row for row in csv_rows
        if row["slab_budget"] == 96
        and not row["diagnostic_uses_heldout_answers"]
    ]
    print("[slabs] matched 50% frontier", flush=True)
    for row in matched:
        print(
            f"[slabs] {row['method']} mean={row['mean_cosine']:.9f} "
            f"p05={row['p05_cosine']:.9f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

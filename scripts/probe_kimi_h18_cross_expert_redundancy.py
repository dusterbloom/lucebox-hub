#!/usr/bin/env python3
"""H18 cross-expert functional-redundancy kill test.

For 32 predeclared layer-one experts, evaluate every native neuron on the same
512 inputs and find its best replacement neuron in a *different* expert under
the exact rank-one contribution cost.  The complete 98,304-square matrix is
never materialized; expert pairs are streamed as 3,072-square GEMMs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, quants

from train_kimi_panel_directional import read_capture


WIDTH = 3072
OUTPUT_DIMENSION = 3584
PROBE_COUNT = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("predeclaration", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_matches", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--activation-batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_cost(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values64.mean()),
        "minimum": float(values64.min()),
        "median": float(np.quantile(values64, 0.50)),
        "p75": float(np.quantile(values64, 0.75)),
        "p90": float(np.quantile(values64, 0.90)),
        "p95": float(np.quantile(values64, 0.95)),
        "p99": float(np.quantile(values64, 0.99)),
        "maximum": float(values64.max()),
    }


def summarize_similarity(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values64.mean()),
        "minimum": float(values64.min()),
        "p01": float(np.quantile(values64, 0.01)),
        "p05": float(np.quantile(values64, 0.05)),
        "median": float(np.quantile(values64, 0.50)),
        "p75": float(np.quantile(values64, 0.75)),
        "p90": float(np.quantile(values64, 0.90)),
        "p95": float(np.quantile(values64, 0.95)),
        "p99": float(np.quantile(values64, 0.99)),
        "maximum": float(values64.max()),
    }


def coverage(cost: np.ndarray) -> dict[str, float]:
    return {
        "C_le_0.01": float(np.mean(cost <= 0.01)),
        "C_le_0.05": float(np.mean(cost <= 0.05)),
        "C_le_0.10": float(np.mean(cost <= 0.10)),
        "C_le_0.20": float(np.mean(cost <= 0.20)),
    }


def common_probe_indices(records: list[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    calibration: list[tuple[str, int, int]] = []
    cursor = 0
    for record in records:
        count = int(record["latent"].shape[0])
        if int(record["split"]) == 0 and count >= 6:
            calibration.append((str(record["id"]), cursor, count))
        cursor += count
    if not calibration:
        raise ValueError("capture has no calibration sequences")
    selected: list[int] = []
    selected_sequences: list[str] = []
    for round_index in range(6):
        for identifier, begin, count in calibration:
            local = min(count - 1, int((round_index + 0.5) * count / 6.0))
            selected.append(begin + local)
            selected_sequences.append(identifier)
            if len(selected) == PROBE_COUNT:
                indices = np.asarray(selected, dtype=np.int64)
                if np.unique(indices).size != PROBE_COUNT:
                    raise ValueError("common-probe selection contains duplicates")
                return indices, selected_sequences
    raise ValueError("capture does not provide 512 predeclared probes")


def dequantize_expert(tensor: object, expert: int) -> torch.Tensor:
    values = quants.dequantize(
        np.ascontiguousarray(tensor.data[expert]), tensor.tensor_type
    )
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


@torch.no_grad()
def expert_geometry(
    latent: np.ndarray,
    gate_tensor: object,
    up_tensor: object,
    down_tensor: object,
    expert: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = dequantize_expert(gate_tensor, expert).to(device)
    up = dequantize_expert(up_tensor, expert).to(device)
    down = dequantize_expert(down_tensor, expert).to(device)
    activation_parts: list[torch.Tensor] = []
    for begin in range(0, latent.shape[0], batch_size):
        z = torch.from_numpy(
            np.ascontiguousarray(latent[begin : begin + batch_size], dtype=np.float32)
        ).to(device)
        gate_value = z @ gate.T
        up_value = z @ up.T
        nonlinear = 4.0 * torch.tanh(gate_value / 4.0) * torch.sigmoid(gate_value)
        linear = 25.0 * torch.tanh(up_value / 25.0)
        activation_parts.append(nonlinear * linear)
    # Neuron-major for expert-pair GEMMs.
    activation = torch.cat(activation_parts).T.contiguous()
    down_columns = down.T.contiguous()
    if activation.shape != (WIDTH, PROBE_COUNT):
        raise ValueError(f"unexpected activation shape {activation.shape}")
    if down_columns.shape != (WIDTH, OUTPUT_DIMENSION):
        raise ValueError(f"unexpected down-column shape {down_columns.shape}")
    activation_norm = torch.linalg.vector_norm(activation, dim=1)
    down_norm = torch.linalg.vector_norm(down_columns, dim=1)
    if torch.any(activation_norm == 0.0) or torch.any(down_norm == 0.0):
        raise ValueError(f"expert {expert} has a zero-norm native neuron")
    activation.div_(activation_norm[:, None])
    down_columns.div_(down_norm[:, None])
    return activation, down_columns, activation_norm, down_norm


def update_rows(
    score: torch.Tensor,
    signed: torch.Tensor,
    source: int,
    target: int,
    best_score: torch.Tensor,
    best_signed: torch.Tensor,
    best_target_expert: torch.Tensor,
    best_target_neuron: torch.Tensor,
) -> None:
    candidate, target_neuron = torch.max(score, dim=1)
    row = torch.arange(WIDTH, device=score.device)
    improve = candidate > best_score[source]
    if torch.any(improve):
        best_score[source, improve] = candidate[improve]
        best_signed[source, improve] = signed[row, target_neuron][improve]
        best_target_expert[source, improve] = target
        best_target_neuron[source, improve] = target_neuron[improve].to(torch.int16)


def verify_direct_cost(
    activation: torch.Tensor,
    down: torch.Tensor,
    activation_norm: torch.Tensor,
    down_norm: torch.Tensor,
    source_expert: int,
    source_neuron: int,
    target_expert: int,
    target_neuron: int,
) -> dict[str, float | int]:
    source_a = (
        activation[source_expert, source_neuron]
        * activation_norm[source_expert, source_neuron]
    ).cpu().numpy()
    target_a = (
        activation[target_expert, target_neuron]
        * activation_norm[target_expert, target_neuron]
    ).cpu().numpy()
    source_d = (
        down[source_expert, source_neuron] * down_norm[source_expert, source_neuron]
    ).cpu().numpy()
    target_d = (
        down[target_expert, target_neuron] * down_norm[target_expert, target_neuron]
    ).cpu().numpy()
    rho = float(
        activation[source_expert, source_neuron]
        @ activation[target_expert, target_neuron]
    )
    cosine_d = float(
        down[source_expert, source_neuron] @ down[target_expert, target_neuron]
    )
    shortcut = 1.0 - (rho * cosine_d) ** 2
    alpha = float(
        (rho * cosine_d)
        * (
            float(activation_norm[source_expert, source_neuron])
            * float(down_norm[source_expert, source_neuron])
        )
        / (
            float(activation_norm[target_expert, target_neuron])
            * float(down_norm[target_expert, target_neuron])
        )
    )
    source_contribution = source_a[:, None] * source_d[None, :]
    target_contribution = target_a[:, None] * target_d[None, :]
    direct = float(
        np.square(source_contribution - alpha * target_contribution, dtype=np.float64).sum()
        / np.square(source_contribution, dtype=np.float64).sum()
    )
    return {
        "source_expert_index": source_expert,
        "source_neuron": source_neuron,
        "target_expert_index": target_expert,
        "target_neuron": target_neuron,
        "alpha": alpha,
        "shortcut_cost": shortcut,
        "direct_cost": direct,
        "absolute_difference": abs(shortcut - direct),
    }


def concentration(
    expert_ids: np.ndarray,
    best_target_expert: np.ndarray,
) -> dict[str, object]:
    directed = Counter()
    targets = Counter()
    for source_index, source_expert in enumerate(expert_ids):
        for target_index in best_target_expert[source_index]:
            target_expert = int(expert_ids[int(target_index)])
            directed[(int(source_expert), target_expert)] += 1
            targets[target_expert] += 1
    total = int(best_target_expert.size)
    pair_counts = sorted(directed.items(), key=lambda item: (-item[1], item[0]))
    target_counts = sorted(targets.items(), key=lambda item: (-item[1], item[0]))
    shares = np.asarray([count / total for _, count in pair_counts], dtype=np.float64)
    return {
        "directed_pair_count": len(pair_counts),
        "top_directed_pairs": [
            {
                "source_expert": pair[0],
                "target_expert": pair[1],
                "neurons": count,
                "fraction": count / total,
            }
            for pair, count in pair_counts[:20]
        ],
        "top1_pair_fraction": float(shares[0]),
        "top5_pair_fraction": float(shares[:5].sum()),
        "pair_herfindahl": float(np.square(shares).sum()),
        "top_target_experts": [
            {"expert": expert, "neurons": count, "fraction": count / total}
            for expert, count in target_counts[:10]
        ],
    }


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        # The primary nearest-neighbor search is true FP32, not a TF32/FP16
        # candidate approximation.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    declaration = json.loads(args.predeclaration.read_text())
    if declaration.get("status") != "PREDECLARED_BEFORE_REDUNDANCY_MEASUREMENT":
        raise ValueError("experts must be predeclared before redundancy measurement")
    if sha256(args.capture) != declaration["capture"]["sha256"]:
        raise ValueError("capture hash differs from the predeclaration")
    if sha256(args.shard) != declaration["model_shard"]["sha256"]:
        raise ValueError("model shard hash differs from the predeclaration")
    expert_ids = np.asarray(
        declaration["experts"]["top16_by_total_registered_route_count"]
        + declaration["experts"]["random16"],
        dtype=np.int32,
    )
    if expert_ids.size != 32 or np.unique(expert_ids).size != 32:
        raise ValueError("predeclaration must contain exactly 32 unique experts")
    header, records = read_capture(args.capture)
    if header["model_layer"] != 1 or header["dimension"] != OUTPUT_DIMENSION:
        raise ValueError("expected the registered layer-one 3584-dimensional capture")
    latent = np.concatenate([record["latent"] for record in records])
    probe_indices, probe_sequences = common_probe_indices(records)
    probe_latent = np.ascontiguousarray(latent[probe_indices], dtype=np.float32)

    reader = GGUFReader(args.shard, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate_tensor = tensors["blk.1.ffn_gate_exps.weight"]
    up_tensor = tensors["blk.1.ffn_up_exps.weight"]
    down_tensor = tensors["blk.1.ffn_down_exps.weight"]
    activation = torch.empty(
        (32, WIDTH, PROBE_COUNT), dtype=torch.float32, device=device
    )
    down = torch.empty(
        (32, WIDTH, OUTPUT_DIMENSION), dtype=torch.float32, device=device
    )
    activation_norm = torch.empty((32, WIDTH), dtype=torch.float32, device=device)
    down_norm = torch.empty((32, WIDTH), dtype=torch.float32, device=device)
    for index, expert in enumerate(expert_ids):
        geometry = expert_geometry(
            probe_latent,
            gate_tensor,
            up_tensor,
            down_tensor,
            int(expert),
            args.activation_batch,
            device,
        )
        activation[index], down[index], activation_norm[index], down_norm[index] = geometry
        print(f"geometry {index + 1}/32 expert={int(expert)}", flush=True)

    shape = (32, WIDTH)
    best_joint = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    best_product = torch.zeros(shape, dtype=torch.float32, device=device)
    best_target_expert = torch.full(
        shape, -1, dtype=torch.int16, device=device
    )
    best_target_neuron = torch.full(
        shape, -1, dtype=torch.int16, device=device
    )
    best_activation = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    best_activation_signed = torch.zeros(shape, dtype=torch.float32, device=device)
    best_activation_expert = torch.full(
        shape, -1, dtype=torch.int16, device=device
    )
    best_activation_neuron = torch.full(
        shape, -1, dtype=torch.int16, device=device
    )

    pair_count = 0
    for left in range(32):
        for right in range(left + 1, 32):
            rho = activation[left] @ activation[right].T
            absolute_rho = torch.abs(rho)
            update_rows(
                absolute_rho,
                rho,
                left,
                right,
                best_activation,
                best_activation_signed,
                best_activation_expert,
                best_activation_neuron,
            )
            update_rows(
                absolute_rho.T,
                rho.T,
                right,
                left,
                best_activation,
                best_activation_signed,
                best_activation_expert,
                best_activation_neuron,
            )
            cosine_d = down[left] @ down[right].T
            product = rho * cosine_d
            absolute_product = torch.abs(product)
            update_rows(
                absolute_product,
                product,
                left,
                right,
                best_joint,
                best_product,
                best_target_expert,
                best_target_neuron,
            )
            update_rows(
                absolute_product.T,
                product.T,
                right,
                left,
                best_joint,
                best_product,
                best_target_expert,
                best_target_neuron,
            )
            pair_count += 1
        print(f"pair-search source={left + 1}/32 pairs={pair_count}/496", flush=True)

    target_activation = activation[
        best_target_expert.long(), best_target_neuron.long()
    ]
    target_down = down[best_target_expert.long(), best_target_neuron.long()]
    best_rho = torch.sum(activation * target_activation, dim=2)
    best_cosine_d = torch.sum(down * target_down, dim=2)
    joint = torch.square(best_rho * best_cosine_d)
    replacement_cost = torch.clamp(1.0 - joint, 0.0, 1.0)
    alpha = (
        (best_rho * best_cosine_d)
        * (activation_norm * down_norm)
        / (
            activation_norm[
                best_target_expert.long(), best_target_neuron.long()
            ]
            * down_norm[best_target_expert.long(), best_target_neuron.long()]
        )
    )
    activation_only_similarity = torch.square(best_activation)
    activation_only_cost = torch.clamp(1.0 - activation_only_similarity, 0.0, 1.0)

    direct_controls: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(260814)
    for _ in range(8):
        left = int(rng.integers(0, 32))
        right = int(rng.integers(0, 31))
        if right >= left:
            right += 1
        control = verify_direct_cost(
            activation,
            down,
            activation_norm,
            down_norm,
            left,
            int(rng.integers(0, WIDTH)),
            right,
            int(rng.integers(0, WIDTH)),
        )
        control["kind"] = "random_pair"
        direct_controls.append(control)
    for flat in np.linspace(0, 32 * WIDTH - 1, 8, dtype=np.int64):
        left, neuron = np.unravel_index(int(flat), shape)
        control = verify_direct_cost(
            activation,
            down,
            activation_norm,
            down_norm,
            int(left),
            int(neuron),
            int(best_target_expert[left, neuron].item()),
            int(best_target_neuron[left, neuron].item()),
        )
        control["kind"] = "best_match"
        direct_controls.append(control)
    direct_max_error = max(float(control["absolute_difference"]) for control in direct_controls)
    if direct_max_error > 2.0e-5:
        raise ValueError(f"analytic replacement cost failed direct check: {direct_max_error}")

    best_target_expert_np = best_target_expert.cpu().numpy()
    best_target_neuron_np = best_target_neuron.cpu().numpy()
    best_activation_expert_np = best_activation_expert.cpu().numpy()
    best_activation_neuron_np = best_activation_neuron.cpu().numpy()
    best_rho_np = best_rho.cpu().numpy()
    best_cosine_d_np = best_cosine_d.cpu().numpy()
    joint_np = joint.cpu().numpy()
    replacement_cost_np = replacement_cost.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    activation_only_similarity_np = activation_only_similarity.cpu().numpy()
    activation_only_cost_np = activation_only_cost.cpu().numpy()
    flat_cost = replacement_cost_np.reshape(-1)
    measured_coverage = coverage(flat_cost)
    if measured_coverage["C_le_0.10"] < 0.50:
        verdict = "WEAK_NO_GO_EXTREME_POOLING"
        next_step = "STOP before medoid rate-distortion"
    elif (
        measured_coverage["C_le_0.10"] >= 0.75
        and measured_coverage["C_le_0.05"] >= 0.50
    ):
        verdict = "STRONG"
        next_step = "medoid rate-distortion earned"
    else:
        verdict = "AMBIGUOUS"
        next_step = "one predeclared 64-expert replication before clustering"

    by_source: list[dict[str, object]] = []
    for index, expert in enumerate(expert_ids):
        by_source.append(
            {
                "expert": int(expert),
                "cost": summarize_cost(replacement_cost_np[index]),
                "coverage": coverage(replacement_cost_np[index]),
            }
        )

    args.output_matches.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_matches,
        expert_ids=expert_ids,
        probe_token_indices=probe_indices,
        best_target_expert_index=best_target_expert_np,
        best_target_expert_id=expert_ids[best_target_expert_np],
        best_target_neuron=best_target_neuron_np,
        alpha=alpha_np.astype(np.float32),
        rho_squared=np.square(best_rho_np).astype(np.float32),
        down_cosine_squared=np.square(best_cosine_d_np).astype(np.float32),
        joint_similarity=joint_np.astype(np.float32),
        normalized_cost=replacement_cost_np.astype(np.float32),
        activation_only_target_expert_id=expert_ids[best_activation_expert_np],
        activation_only_target_neuron=best_activation_neuron_np,
        activation_only_similarity=activation_only_similarity_np.astype(np.float32),
        activation_only_cost=activation_only_cost_np.astype(np.float32),
    )
    matches_hash = sha256(args.output_matches)
    result = {
        "schema": "kimi-k3-h18-cross-expert-redundancy-v1",
        "verdict": verdict,
        "next_step": next_step,
        "claim": "MEASURED cross-expert algebraic replacement microscope; no runtime or storage claim",
        "model_layer": 1,
        "expert_ids": list(map(int, expert_ids)),
        "expert_count": 32,
        "neurons_per_expert": WIDTH,
        "total_neurons": int(32 * WIDTH),
        "common_probe_count": PROBE_COUNT,
        "common_probe_unique_sequences": len(set(probe_sequences)),
        "common_probe_token_indices_sha256": hashlib.sha256(probe_indices.tobytes()).hexdigest(),
        "cost": "E||a_i d_i - alpha a_j d_j||^2 / E||a_i d_i||^2",
        "optimal_alpha": "E[a_i a_j] (d_i^T d_j) / (E[a_j^2] ||d_j||^2)",
        "shortcut": "C = 1 - rho(a_i,a_j)^2 cos(d_i,d_j)^2",
        "different_expert_constraint": True,
        "joint_replacement_cost": summarize_cost(flat_cost),
        "coverage": measured_coverage,
        "activation_only": {
            "similarity_squared": summarize_similarity(activation_only_similarity_np.reshape(-1)),
            "cost": summarize_cost(activation_only_cost_np.reshape(-1)),
            "coverage": coverage(activation_only_cost_np.reshape(-1)),
        },
        "best_joint_components": {
            "rho_squared": summarize_similarity(np.square(best_rho_np).reshape(-1)),
            "down_cosine_squared": summarize_similarity(np.square(best_cosine_d_np).reshape(-1)),
            "joint": summarize_similarity(joint_np.reshape(-1)),
        },
        "by_source_expert": by_source,
        "match_concentration": concentration(expert_ids, best_target_expert_np),
        "direct_shortcut_control": {
            "cases": direct_controls,
            "maximum_absolute_difference": direct_max_error,
            "tolerance": 2.0e-5,
            "status": "PASS",
        },
        "gate": declaration["gate"],
        "predeclaration": str(args.predeclaration),
        "predeclaration_sha256": sha256(args.predeclaration),
        "matches": {"path": str(args.output_matches), "sha256": matches_hash},
        "artifacts": {"capture": str(args.capture), "shard": str(args.shard)},
        "device": str(device),
        "matmul_precision": "FP32 with CUDA TF32 disabled" if device.type == "cuda" else "FP32 CPU",
        "memory_design": "~1.6 GiB normalized activations/down columns; stream 496 expert-pair matrices; never materialize 98,304 squared",
        "elapsed_seconds": time.time() - started,
        "medoid_rate_distortion_reached": verdict == "STRONG",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"verdict={verdict} median_C={result['joint_replacement_cost']['median']:.6f} "
        f"coverage_C10={measured_coverage['C_le_0.10']:.6f} "
        f"coverage_C05={measured_coverage['C_le_0.05']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

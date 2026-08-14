#!/usr/bin/env python3
"""H18 Lane B: optimistic native-neuron output-LS compression ceiling.

The experiment deliberately stops before sparse fusion.  It chooses native
SiTU-GLU activation columns using pivoted QR on registered training sequences,
fits a ridge output map, chooses ridge only on development sequences, and
opens registered validation sequences exactly once for the reported test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from gguf import GGUFReader, quants

from probe_kimi_response_atlas import (
    pair_cosine,
    read_expert_responses,
    response_path,
)
from train_kimi_panel_directional import load_data


WIDTH = 3072
DEFAULT_RIDGE_MULTIPLIERS = (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("predeclaration", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--activation-batch", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray, *, cosine: bool = False) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    result = {
        "mean": float(values64.mean()),
        "median": float(np.quantile(values64, 0.50)),
        "minimum": float(values64.min()),
        "maximum": float(values64.max()),
    }
    if cosine:
        result["p05"] = float(np.quantile(values64, 0.05))
        result["p01"] = float(np.quantile(values64, 0.01))
    else:
        result["p95"] = float(np.quantile(values64, 0.95))
        result["p99"] = float(np.quantile(values64, 0.99))
    return result


def output_metrics(prediction: np.ndarray, teacher: np.ndarray) -> dict[str, object]:
    cosine = pair_cosine(prediction, teacher)
    teacher_norm = np.linalg.norm(teacher, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    relative_l2 = np.linalg.norm(prediction - teacher, axis=1) / np.maximum(
        teacher_norm, 1.0e-30
    )
    norm_ratio = prediction_norm / np.maximum(teacher_norm, 1.0e-30)
    return {
        "relative_l2": summarize(relative_l2),
        "cosine": summarize(cosine, cosine=True),
        "norm_ratio": summarize(norm_ratio),
    }


def dequantize_expert(tensor: object, expert: int) -> torch.Tensor:
    values = quants.dequantize(
        np.ascontiguousarray(tensor.data[expert]), tensor.tensor_type
    )
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


@torch.no_grad()
def native_activations(
    latent: np.ndarray,
    gate: torch.Tensor,
    up: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for begin in range(0, latent.shape[0], batch_size):
        z = torch.from_numpy(
            np.ascontiguousarray(latent[begin : begin + batch_size], dtype=np.float32)
        )
        gate_value = z @ gate.T
        up_value = z @ up.T
        nonlinear = 4.0 * torch.tanh(gate_value / 4.0) * torch.sigmoid(gate_value)
        linear = 25.0 * torch.tanh(up_value / 25.0)
        parts.append((nonlinear * linear).numpy())
    activation = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    if activation.shape != (latent.shape[0], WIDTH):
        raise ValueError(f"unexpected activation shape {activation.shape}")
    return activation


@torch.no_grad()
def offline_full_output(
    activation: np.ndarray,
    down: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for begin in range(0, activation.shape[0], batch_size):
        values = torch.from_numpy(
            np.ascontiguousarray(
                activation[begin : begin + batch_size], dtype=np.float32
            )
        )
        parts.append((values @ down.T).numpy())
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def split_rows(
    token_indices: np.ndarray,
    data: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_by_token = np.full(data.latent.shape[0], -1, dtype=np.int8)
    split_by_token[data.train_indices] = 0
    split_by_token[data.development_indices] = 1
    split_by_token[data.validation_indices] = 2
    route_split = split_by_token[token_indices]
    if np.any(route_split < 0):
        raise ValueError("expert routes are not covered by registered splits")
    return tuple(np.flatnonzero(route_split == split) for split in range(3))


def pivot_order(train: np.ndarray) -> np.ndarray:
    # scipy preserves float32 here.  The pivot order is determined from train
    # sequences only and is fixed before any development/test fitting score.
    _, _, pivots = scipy.linalg.qr(
        np.array(train, dtype=np.float32, order="F", copy=True),
        mode="economic",
        pivoting=True,
        overwrite_a=True,
        check_finite=False,
    )
    return np.asarray(pivots, dtype=np.int64)


def ridge_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_x: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, int]:
    # Thin FP32 SVD supports both k < n and the diagnostic k > n case without
    # creating a k-by-k inverse.  This is the stated output-LS ceiling.
    u, singular, vt = scipy.linalg.svd(
        np.asarray(train_x, dtype=np.float32),
        full_matrices=False,
        overwrite_a=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    tolerance = (
        np.finfo(np.float32).eps
        * max(train_x.shape)
        * float(singular[0])
    )
    if ridge == 0.0:
        gain = np.zeros_like(singular)
        keep = singular > tolerance
        gain[keep] = 1.0 / singular[keep]
    else:
        gain = singular / (np.square(singular) + np.float32(ridge))
    projected_y = u.T @ np.asarray(train_y, dtype=np.float32)
    coefficient = (vt.T * gain[None, :]) @ projected_y
    prediction = np.asarray(target_x, dtype=np.float32) @ coefficient
    return np.asarray(prediction, dtype=np.float32), int(np.sum(singular > tolerance))


def evaluate_budget(
    activation: np.ndarray,
    output: np.ndarray,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    test_rows: np.ndarray,
    order: np.ndarray,
    budget: int,
) -> dict[str, object]:
    selected = order[:budget]
    train_x = np.ascontiguousarray(activation[train_rows][:, selected], dtype=np.float32)
    development_x = np.ascontiguousarray(
        activation[development_rows][:, selected], dtype=np.float32
    )
    test_x = np.ascontiguousarray(activation[test_rows][:, selected], dtype=np.float32)
    train_y = np.ascontiguousarray(output[train_rows], dtype=np.float32)
    development_y = np.ascontiguousarray(output[development_rows], dtype=np.float32)
    test_y = np.ascontiguousarray(output[test_rows], dtype=np.float32)
    ridge_scale = float(np.square(train_x, dtype=np.float64).sum() / train_x.shape[0])
    validation: list[dict[str, object]] = []
    chosen: tuple[float, dict[str, object], np.ndarray, int] | None = None
    for multiplier in DEFAULT_RIDGE_MULTIPLIERS:
        ridge = float(multiplier * ridge_scale)
        prediction, numerical_rank = ridge_prediction(
            train_x, train_y, development_x, ridge
        )
        metrics = output_metrics(prediction, development_y)
        validation.append(
            {
                "ridge_multiplier": multiplier,
                "ridge": ridge,
                "numerical_rank": numerical_rank,
                "metrics": metrics,
            }
        )
        key = float(metrics["relative_l2"]["mean"])
        if chosen is None or key < chosen[0]:
            chosen = (key, validation[-1], prediction, numerical_rank)
    assert chosen is not None
    selected_ridge = float(chosen[1]["ridge"])
    test_prediction, numerical_rank = ridge_prediction(
        train_x, train_y, test_x, selected_ridge
    )
    return {
        "native_activation_count": budget,
        "compression_factor": WIDTH / budget,
        "selected_neuron_indices": list(map(int, selected)),
        "subset_method": "training-only column-pivoted QR",
        "fit": "FP32 ridge output least squares without intercept",
        "ridge_scale": ridge_scale,
        "ridge_selection": "minimum development mean output relative L2",
        "selected_ridge_multiplier": float(chosen[1]["ridge_multiplier"]),
        "selected_ridge": selected_ridge,
        "train_design_numerical_rank": numerical_rank,
        "development_candidates": validation,
        "test": output_metrics(test_prediction, test_y),
    }


def load_expert_data(
    expert: int,
    data: object,
    tensors: dict[str, object],
    response_directory: Path,
    activation_batch: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, object],
]:
    records, output = read_expert_responses(
        response_path(response_directory, expert),
        data.model_layer,
        expert,
        data.dimension,
    )
    token_indices = records["token_index"].astype(np.int64, copy=False)
    ranks = records["rank"].astype(np.int64, copy=False)
    if not np.all(data.expert_ids[token_indices, ranks] == expert):
        raise ValueError(f"expert {expert} response metadata differs from capture")
    gate = dequantize_expert(
        tensors[f"blk.{data.model_layer}.ffn_gate_exps.weight"], expert
    )
    up = dequantize_expert(
        tensors[f"blk.{data.model_layer}.ffn_up_exps.weight"], expert
    )
    down = dequantize_expert(
        tensors[f"blk.{data.model_layer}.ffn_down_exps.weight"], expert
    )
    activation = native_activations(
        data.latent[token_indices], gate, up, activation_batch
    )
    native_output = np.ascontiguousarray(output, dtype=np.float32)
    dequantized_output = offline_full_output(activation, down, activation_batch)
    fidelity = output_metrics(dequantized_output, native_output)
    fidelity["flag"] = (
        "PASS"
        if float(fidelity["cosine"]["mean"]) >= 0.999
        and float(fidelity["cosine"]["p05"]) >= 0.995
        else "MATERIALLY_MISMATCHED"
    )
    return (
        activation,
        native_output,
        split_rows(token_indices, data),
        fidelity,
    )


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.activation_batch <= 0:
        raise ValueError("threads and activation batch must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    declaration = json.loads(args.predeclaration.read_text())
    if declaration.get("status") != "PREDECLARED_BEFORE_COMPRESSION":
        raise ValueError("Lane B expert must be predeclared before fitting")
    expert = int(declaration["expert_id"])
    data = load_data(args.capture, args.teacher)
    reader = GGUFReader(args.shard, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    activation, output, rows, teacher_fidelity = load_expert_data(
        expert,
        data,
        tensors,
        args.response_directory,
        args.activation_batch,
    )
    train_rows, development_rows, test_rows = rows
    observed_counts = {
        "total": int(output.shape[0]),
        "train": int(train_rows.size),
        "development": int(development_rows.size),
        "test": int(test_rows.size),
    }
    if observed_counts != declaration["route_counts"]:
        raise ValueError(
            f"predeclared and observed route counts differ: {observed_counts}"
        )
    order = pivot_order(activation[train_rows])
    primary = {"192": evaluate_budget(
        activation, output, train_rows, development_rows, test_rows, order, 192
    )}
    cosine192 = primary["192"]["test"]["cosine"]
    median192 = float(cosine192["median"])
    p05192 = float(cosine192["p05"])
    replication: dict[str, object] = {}
    if median192 < 0.95 or p05192 < 0.80:
        verdict = "NO_GO_16X"
        gate = "k192_failed_hard_gate"
        for budget in (384, 768):
            primary[str(budget)] = evaluate_budget(
                activation,
                output,
                train_rows,
                development_rows,
                test_rows,
                order,
                budget,
            )
    elif median192 >= 0.98 and p05192 >= 0.90:
        verdict = "MAJOR_GO_PENDING_REPLICATION"
        gate = "k192_passed_major_go_gate"
        passing = 1
        replication[str(expert)] = primary["192"]
        for replicate_expert in declaration["conditional_replication_experts"]:
            (
                candidate_activation,
                candidate_output,
                candidate_rows,
                candidate_fidelity,
            ) = load_expert_data(
                int(replicate_expert),
                data,
                tensors,
                args.response_directory,
                args.activation_batch,
            )
            candidate_train, candidate_development, candidate_test = candidate_rows
            candidate_order = pivot_order(candidate_activation[candidate_train])
            candidate = evaluate_budget(
                candidate_activation,
                candidate_output,
                candidate_train,
                candidate_development,
                candidate_test,
                candidate_order,
                192,
            )
            replication[str(replicate_expert)] = candidate
            candidate["offline_full_dequantized_vs_native_teacher"] = (
                candidate_fidelity
            )
            candidate_cosine = candidate["test"]["cosine"]
            if (
                float(candidate_cosine["median"]) >= 0.98
                and float(candidate_cosine["p05"]) >= 0.90
            ):
                passing += 1
        verdict = "MAJOR_GO" if passing == 8 else "REPLICATION_FAILED"
        replication["passing_experts"] = passing
        replication["total_experts"] = 8
    else:
        verdict = "INTERMEDIATE_STOP"
        gate = "k192_between_hard_no_go_and_major_go_thresholds"

    result = {
        "schema": "kimi-k3-h18-neuron-fusion-ceiling-v1",
        "verdict": verdict,
        "gate": gate,
        "claim": "MEASURED output-LS ceiling; no sparse fusion or runtime speed claim",
        "predeclaration": str(args.predeclaration),
        "predeclaration_sha256": sha256(args.predeclaration),
        "model_layer": data.model_layer,
        "primary_expert": expert,
        "route_counts": observed_counts,
        "offline_full_dequantized_vs_native_teacher": teacher_fidelity,
        "whole_sequence_split": True,
        "test_policy": "registered validation sequences untouched until subset and ridge selection completed",
        "activation_source": "FP32 CPU evaluation of dequantized IQ1_S gate/up rows with native SiTU-GLU formula",
        "output_teacher": "captured native IQ1_S expert output",
        "primary": primary,
        "conditional_replication": replication,
        "hard_gates": {
            "no_go": "median cosine < 0.95 OR p05 cosine < 0.80",
            "major_go": "median cosine >= 0.98 AND p05 cosine >= 0.90",
        },
        "artifacts": {
            "capture": str(args.capture),
            "teacher": str(args.teacher),
            "response_directory": str(args.response_directory),
            "shard": str(args.shard),
        },
        "elapsed_seconds": time.time() - started,
        "sparse_fusion_reached": False,
        "native_intervention_reached": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"expert={expert} test={test_rows.size} "
        f"k192_mean={cosine192['mean']:.6f} "
        f"median={median192:.6f} p05={p05192:.6f} verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Held-out rate/distortion and linear omitted-tail ceiling for Kimi slabs.

This is deliberately an *offline ceiling*, not a runtime tail provider.  It
uses the exact retained routed-slab aggregate as its only learned feature, fits
on whole training sequences, chooses ridge on development sequences, and
reports only untouched validation sequences.  Its job is to tell whether a
layer is limited by slab selection, byte budget, or an unpredictable omitted
aggregate before any learned runtime correction is considered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from probe_kimi_neuron_slabs import (
    EXPERT_COUNT,
    ORIGINAL_EXPERT_WIDTH,
    DEFAULT_SLAB_SIZE,
    eval_slab,
    inverse_order,
    resolve_layer_tensors,
)
from probe_kimi_response_atlas import read_expert_responses, response_path
from train_kimi_panel_directional import load_data


SLAB_BUDGETS = (96, 120, 144, 168, 192)
RIDGE_MULTIPLIERS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slab-size", type=int, default=DEFAULT_SLAB_SIZE)
    parser.add_argument("--metric-batch", type=int, default=256)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.50)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def pair_metrics(candidate: np.ndarray, teacher: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    numerator = np.einsum("ij,ij->i", candidate, teacher, dtype=np.float64)
    denominator = np.linalg.norm(candidate, axis=1) * np.linalg.norm(teacher, axis=1)
    cosine = numerator / np.maximum(denominator, 1.0e-30)
    relative_l2 = np.linalg.norm(candidate - teacher, axis=1) / np.maximum(
        np.linalg.norm(teacher, axis=1), 1.0e-30
    )
    return {
        "cosine": summarize(cosine),
        "relative_l2": summarize(relative_l2),
    }, cosine.astype(np.float32), relative_l2.astype(np.float32)


def load_state(path: Path, dimension: int, slab_count: int) -> dict[str, np.ndarray]:
    expected = {
        "slab_means": (EXPERT_COUNT, slab_count, dimension),
        "slab_expected_residual_norm": (EXPERT_COUNT, slab_count),
        "calibrated_experts": (EXPERT_COUNT,),
    }
    with np.load(path, allow_pickle=False) as state:
        missing = set(expected) - set(state.files)
        if missing:
            raise ValueError(f"fit state lacks fields {sorted(missing)}")
        result = {name: np.asarray(state[name]) for name in expected}
    for name, shape in expected.items():
        if result[name].shape != shape:
            raise ValueError(f"fit state shape mismatch for {name}")
    if result["slab_means"].dtype != np.float32 or result[
        "slab_expected_residual_norm"
    ].dtype != np.float32:
        raise ValueError("fit state uses unexpected statistics dtype")
    calibrated = result["calibrated_experts"]
    if calibrated.dtype != np.uint8 or not np.all((calibrated == 0) | (calibrated == 1)):
        raise ValueError("fit state calibrated-expert mask is invalid")
    if not np.isfinite(result["slab_means"]).all() or not np.isfinite(
        result["slab_expected_residual_norm"]
    ).all():
        raise ValueError("fit state contains non-finite values")
    return result


def add_rows(destination: np.ndarray, rows: np.ndarray, values: np.ndarray) -> None:
    # A routed expert occurs at most once in a K3 token's top-16 list.
    if rows.size != np.unique(rows).size:
        raise ValueError("duplicate expert route in one token")
    destination[rows] += values


def ridge_tail_ceiling(
    feature: np.ndarray,
    residual: np.ndarray,
    train: np.ndarray,
    development: np.ndarray,
    validation: np.ndarray,
    base: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Fit an optimistic dense ridge map from retained exact sum to tail.

    The linear map is intentionally dense and is never emitted as a runtime
    artifact.  Development sequences select regularization; validation
    sequences are never used in either fitting or selection.
    """
    if train.size <= 1 or development.size <= 1 or validation.size <= 1:
        raise ValueError("all three sequence-disjoint splits need at least two rows")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x_train = torch.from_numpy(feature[train]).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(residual[train]).to(device=device, dtype=torch.float32)
    x_dev = torch.from_numpy(feature[development]).to(device=device, dtype=torch.float32)
    y_dev = torch.from_numpy(residual[development]).to(device=device, dtype=torch.float32)
    x_val = torch.from_numpy(feature[validation]).to(device=device, dtype=torch.float32)
    mean_x = x_train.mean(dim=0, keepdim=True)
    mean_y = y_train.mean(dim=0, keepdim=True)
    xc = x_train - mean_x
    yc = y_train - mean_y
    gram = xc.T @ xc
    cross = xc.T @ yc
    diagonal_scale = float(torch.trace(gram).item() / gram.shape[0])
    if not math.isfinite(diagonal_scale) or diagonal_scale <= 0:
        raise ValueError("retained-feature Gram scale is invalid")
    identity = torch.eye(gram.shape[0], device=device, dtype=torch.float32)
    best: tuple[float, float, torch.Tensor] | None = None
    development_rows: list[dict[str, float]] = []
    for multiplier in RIDGE_MULTIPLIERS:
        ridge = multiplier * diagonal_scale
        system = gram + ridge * identity
        try:
            weights = torch.linalg.solve(system, cross)
        except RuntimeError as error:
            development_rows.append({
                "ridge_multiplier": multiplier,
                "ridge": ridge,
                "development_relative_l2": float("inf"),
                "error": str(error),
            })
            continue
        predicted = (x_dev - mean_x) @ weights + mean_y
        difference = predicted - y_dev
        score = float(
            (torch.linalg.vector_norm(difference, dim=1)
             / torch.clamp(torch.linalg.vector_norm(y_dev, dim=1), min=1.0e-30)).mean().item()
        )
        development_rows.append({
            "ridge_multiplier": multiplier,
            "ridge": ridge,
            "development_relative_l2": score,
        })
        if best is None or score < best[0]:
            best = (score, ridge, weights)
    if best is None:
        raise RuntimeError("all dense ridge solves failed")
    _, selected_ridge, weights = best
    prediction = np.empty((validation.size, feature.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, validation.size, batch_size):
            end = min(validation.size, begin + batch_size)
            output = (x_val[begin:end] - mean_x) @ weights + mean_y
            prediction[begin:end] = output.cpu().numpy()
    reconstructed = base[validation] + prediction
    metrics, cosine, relative_l2 = pair_metrics(
        reconstructed, base[validation] + residual[validation]
    )
    result = {
        "method": "dense_linear_ridge_omitted_tail_ceiling",
        "feature": "exact retained routed-slab aggregate only",
        "target": "native routed aggregate minus slab-mean-tail reconstruction",
        "runtime_deployable": False,
        "training_tokens": int(train.size),
        "development_tokens": int(development.size),
        "heldout_validation_tokens": int(validation.size),
        "ridge_selection": "minimum development relative L2",
        "selected_ridge": selected_ridge,
        "development_curve": development_rows,
        "heldout_metrics_against_native": metrics,
    }
    del x_train, y_train, x_dev, y_dev, x_val, mean_x, mean_y, xc, yc, gram, cross, identity, weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, np.stack((cosine, relative_l2), axis=0)


def main() -> int:
    args = parse_args()
    if args.slab_size <= 0 or ORIGINAL_EXPERT_WIDTH % args.slab_size:
        raise ValueError("slab size must exactly divide K3's routed-expert width")
    if args.metric_batch <= 0:
        raise ValueError("metric batch must be positive")
    slab_count = ORIGINAL_EXPERT_WIDTH // args.slab_size
    active_slabs = 16 * slab_count
    if tuple(sorted(SLAB_BUDGETS)) != SLAB_BUDGETS or SLAB_BUDGETS[-1] != active_slabs:
        raise ValueError("registered rate/distortion ladder disagrees with K3")
    started = time.monotonic()
    data = load_data(args.capture, args.teacher)
    if data.model_layer != args.layer or data.top_k != 16:
        raise ValueError("capture layer or routed top-k disagrees with requested K3 probe")
    state = load_state(args.fit_state, data.dimension, slab_count)
    calibrated = state["calibrated_experts"].astype(bool)
    capture_calibrated = np.bincount(
        data.expert_ids[np.concatenate((data.train_indices, data.development_indices))].reshape(-1),
        minlength=EXPERT_COUNT,
    ) > 0
    if not np.array_equal(calibrated, capture_calibrated):
        raise ValueError("calibration coverage differs from the capture split")
    tensors, tensor_sources, _readers = resolve_layer_tensors(args.shard, args.layer)
    gate = tensors[f"blk.{args.layer}.ffn_gate_exps.weight"]
    up = tensors[f"blk.{args.layer}.ffn_up_exps.weight"]
    down = tensors[f"blk.{args.layer}.ffn_down_exps.weight"]
    expected_gate = (EXPERT_COUNT, ORIGINAL_EXPERT_WIDTH, 700)
    expected_down = (EXPERT_COUNT, data.dimension, 600)
    if gate.data.shape != expected_gate or up.data.shape != expected_gate or down.data.shape != expected_down:
        raise ValueError("unexpected routed-expert IQ1_S tensor shape")

    token_count = data.latent.shape[0]
    importance = state["slab_expected_residual_norm"]
    static_order = np.argsort(-importance, axis=1, kind="stable")
    static_rank = inverse_order(static_order)
    score = data.router_weights[:, :, None] * importance[data.expert_ids]
    score = np.where(calibrated[data.expert_ids, None], score, -np.inf)
    adaptive_rank = inverse_order(
        np.argsort(-score.reshape(token_count, -1), axis=1, kind="stable")
    ).reshape(token_count, data.top_k, slab_count)

    tail_mean = np.zeros((token_count, data.dimension), dtype=np.float32)
    exact_fallback = np.zeros_like(tail_mean)
    adaptive_retained = {
        budget: np.zeros_like(tail_mean) for budget in SLAB_BUDGETS
    }
    adaptive_selected_mean = {
        budget: np.zeros_like(tail_mean) for budget in SLAB_BUDGETS
    }
    uniform_retained = {
        budget: np.zeros_like(tail_mean) for budget in (96, 144)
    }
    uniform_selected_mean = {
        budget: np.zeros_like(tail_mean) for budget in (96, 144)
    }
    oracle_scores = np.full(
        (token_count, data.top_k, slab_count), -np.inf, dtype=np.float32
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print("[tail-ceiling] pass 1/2: exact retained aggregates", flush=True)
    for expert in range(EXPERT_COUNT):
        records, native = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            data.dimension,
        )
        tokens = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        weights = records["router_weight"].astype(np.float32, copy=False)
        if not np.all(data.expert_ids[tokens, ranks] == expert):
            raise ValueError(f"response metadata disagrees with capture for expert {expert}")
        if not np.allclose(data.router_weights[tokens, ranks], weights, atol=1.0e-6):
            raise ValueError(f"router weights disagree with capture for expert {expert}")
        if not calibrated[expert]:
            add_rows(exact_fallback, tokens, weights[:, None] * native)
            continue
        expert_mean = state["slab_means"][expert].sum(axis=0)
        add_rows(tail_mean, tokens, weights[:, None] * expert_mean)
        z = torch.from_numpy(data.latent[tokens]).to(device)
        for slab in range(slab_count):
            output = eval_slab(z, gate, up, down, expert, slab, args.slab_size, device)
            contribution = weights[:, None] * output
            mean_contribution = weights[:, None] * state["slab_means"][expert, slab]
            oracle_scores[tokens, ranks, slab] = np.linalg.norm(
                contribution - mean_contribution, axis=1
            )
            for budget in SLAB_BUDGETS:
                selected = adaptive_rank[tokens, ranks, slab] < budget
                if np.any(selected):
                    add_rows(adaptive_retained[budget], tokens[selected], contribution[selected])
                    add_rows(adaptive_selected_mean[budget], tokens[selected], mean_contribution[selected])
            for budget in (96, 144):
                selected = static_rank[expert, slab] < budget // data.top_k
                if selected:
                    add_rows(uniform_retained[budget], tokens, contribution)
                    add_rows(uniform_selected_mean[budget], tokens, mean_contribution)
        del z
        if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
            print(f"[tail-ceiling] pass1 expert={expert + 1}/{EXPERT_COUNT}", flush=True)

    oracle_rank = inverse_order(
        np.argsort(-oracle_scores.reshape(token_count, -1), axis=1, kind="stable")
    ).reshape(token_count, data.top_k, slab_count)
    oracle_retained = {budget: np.zeros_like(tail_mean) for budget in SLAB_BUDGETS}
    oracle_selected_mean = {budget: np.zeros_like(tail_mean) for budget in SLAB_BUDGETS}
    print("[tail-ceiling] pass 2/2: held-out selector diagnostic", flush=True)
    for expert in range(EXPERT_COUNT):
        if not calibrated[expert]:
            continue
        records, _ = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            data.dimension,
        )
        tokens = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        weights = records["router_weight"].astype(np.float32, copy=False)
        z = torch.from_numpy(data.latent[tokens]).to(device)
        for slab in range(slab_count):
            output = eval_slab(z, gate, up, down, expert, slab, args.slab_size, device)
            contribution = weights[:, None] * output
            mean_contribution = weights[:, None] * state["slab_means"][expert, slab]
            for budget in SLAB_BUDGETS:
                selected = oracle_rank[tokens, ranks, slab] < budget
                if np.any(selected):
                    add_rows(oracle_retained[budget], tokens[selected], contribution[selected])
                    add_rows(oracle_selected_mean[budget], tokens[selected], mean_contribution[selected])
        del z
        if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
            print(f"[tail-ceiling] pass2 expert={expert + 1}/{EXPERT_COUNT}", flush=True)

    base = tail_mean + exact_fallback
    validation = data.validation_indices
    row_errors: dict[str, np.ndarray] = {}
    methods: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    adaptive_approximations: dict[int, np.ndarray] = {}
    for budget in SLAB_BUDGETS:
        candidate = base + adaptive_retained[budget] - adaptive_selected_mean[budget]
        adaptive_approximations[budget] = candidate
        metrics, cosine, relative_l2 = pair_metrics(candidate[validation], data.teacher[validation])
        methods[f"adaptive_{budget}"] = {
            "kind": "deployable static residual-norm selector plus slab-mean tail",
            "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "heldout_metrics_against_native": metrics,
        }
        row_errors[f"adaptive_{budget}_cosine"] = cosine
        row_errors[f"adaptive_{budget}_relative_l2"] = relative_l2
        csv_rows.append({
            "method": "adaptive", "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "mean_cosine": metrics["cosine"]["mean"],
            "p05_cosine": metrics["cosine"]["p05"],
            "mean_relative_l2": metrics["relative_l2"]["mean"],
            "diagnostic": False,
        })
        oracle_candidate = base + oracle_retained[budget] - oracle_selected_mean[budget]
        oracle_metrics, oracle_cosine, oracle_relative_l2 = pair_metrics(
            oracle_candidate[validation], data.teacher[validation]
        )
        methods[f"oracle_{budget}"] = {
            "kind": "held-out residual-norm selector plus slab-mean tail",
            "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "runtime_deployable": False,
            "heldout_metrics_against_native": oracle_metrics,
        }
        row_errors[f"oracle_{budget}_cosine"] = oracle_cosine
        row_errors[f"oracle_{budget}_relative_l2"] = oracle_relative_l2
        csv_rows.append({
            "method": "oracle", "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "mean_cosine": oracle_metrics["cosine"]["mean"],
            "p05_cosine": oracle_metrics["cosine"]["p05"],
            "mean_relative_l2": oracle_metrics["relative_l2"]["mean"],
            "diagnostic": True,
        })
    for budget in (96, 144):
        candidate = base + uniform_retained[budget] - uniform_selected_mean[budget]
        metrics, cosine, relative_l2 = pair_metrics(candidate[validation], data.teacher[validation])
        methods[f"uniform_{budget}"] = {
            "kind": "uniform per-expert static-prefix control plus slab-mean tail",
            "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "heldout_metrics_against_native": metrics,
        }
        row_errors[f"uniform_{budget}_cosine"] = cosine
        row_errors[f"uniform_{budget}_relative_l2"] = relative_l2
        csv_rows.append({
            "method": "uniform", "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "mean_cosine": metrics["cosine"]["mean"],
            "p05_cosine": metrics["cosine"]["p05"],
            "mean_relative_l2": metrics["relative_l2"]["mean"],
            "diagnostic": False,
        })

    for budget in (96, 120, 144, 168):
        residual = data.teacher - adaptive_approximations[budget]
        ceiling, errors = ridge_tail_ceiling(
            adaptive_retained[budget] + exact_fallback,
            residual,
            data.train_indices,
            data.development_indices,
            validation,
            adaptive_approximations[budget],
            device,
            args.metric_batch,
        )
        methods[f"linear_tail_ceiling_{budget}"] = ceiling | {
            "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
        }
        row_errors[f"linear_tail_ceiling_{budget}_cosine"] = errors[0]
        row_errors[f"linear_tail_ceiling_{budget}_relative_l2"] = errors[1]
        metrics = ceiling["heldout_metrics_against_native"]
        csv_rows.append({
            "method": "linear_tail_ceiling", "budget": budget,
            "exact_byte_fraction": budget / active_slabs,
            "mean_cosine": metrics["cosine"]["mean"],
            "p05_cosine": metrics["cosine"]["p05"],
            "mean_relative_l2": metrics["relative_l2"]["mean"],
            "diagnostic": True,
        })
        del residual

    def recovery(budget: int) -> dict[str, float]:
        baseline = methods[f"adaptive_{budget}"]["heldout_metrics_against_native"]
        ceiling = methods[f"linear_tail_ceiling_{budget}"]["heldout_metrics_against_native"]
        baseline_error = float(baseline["relative_l2"]["mean"])
        ceiling_error = float(ceiling["relative_l2"]["mean"])
        return {
            "baseline_relative_l2": baseline_error,
            "ceiling_relative_l2": ceiling_error,
            "relative_l2_reduction_fraction": (
                (baseline_error - ceiling_error) / max(baseline_error, 1.0e-30)
            ),
        }

    diagnostics = {
        "selection_headroom_96": {
            "adaptive_mean_cosine": methods["adaptive_96"]["heldout_metrics_against_native"]["cosine"]["mean"],
            "oracle_mean_cosine": methods["oracle_96"]["heldout_metrics_against_native"]["cosine"]["mean"],
        },
        "selection_headroom_144": {
            "adaptive_mean_cosine": methods["adaptive_144"]["heldout_metrics_against_native"]["cosine"]["mean"],
            "oracle_mean_cosine": methods["oracle_144"]["heldout_metrics_against_native"]["cosine"]["mean"],
        },
        "linear_tail_recovery": {
            str(budget): recovery(budget) for budget in (96, 120, 144, 168)
        },
        "interpretation": [
            "A large oracle-versus-adaptive gap indicates selector headroom, not a deployable result.",
            "A large held-out dense-linear recovery indicates aggregate omitted-tail predictability, not a runtime design.",
            "This measures routed-output geometry only. Final-logit KL remains a later whole-model gate.",
        ],
    }
    result = {
        "schema": "kimi-k3-omitted-tail-ceiling-v1",
        "status": "EXPLORATORY_HELDOUT_CEILING",
        "model_layer": args.layer,
        "registered_ladder": list(SLAB_BUDGETS),
        "capture": str(args.capture),
        "capture_sha256": sha256(args.capture),
        "teacher": str(args.teacher),
        "teacher_sha256": sha256(args.teacher),
        "fit_state": str(args.fit_state),
        "fit_state_sha256": sha256(args.fit_state),
        "response_directory": str(args.response_directory),
        "tensor_sources": {name: str(path) for name, path in sorted(tensor_sources.items())},
        "splits": {
            "train_tokens": int(data.train_indices.size),
            "development_tokens": int(data.development_indices.size),
            "heldout_validation_tokens": int(validation.size),
            "whole_sequence_separation": True,
            "calibration_state_uses": "train plus development sequences only",
        },
        "coverage": {
            "calibrated_experts": int(np.count_nonzero(calibrated)),
            "uncalibrated_experts": int(np.count_nonzero(~calibrated)),
            "uncalibrated_routes_are": "preserved as exact fallback in every approximation",
        },
        "methods": methods,
        "diagnostics": diagnostics,
        "elapsed_seconds": time.monotonic() - started,
        "warnings": [
            "The dense linear ceiling is intentionally much larger than an acceptable runtime corrector and must not be called deployable.",
            "The oracle selector sees held-out slab residual norms and is a selector ceiling only.",
            "All quality metrics here are local routed-output measures, not whole-model final-logit KL.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)
    if args.output_npz:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **row_errors)
    print(json.dumps({
        "layer": args.layer,
        "adaptive96": methods["adaptive_96"]["heldout_metrics_against_native"],
        "linear96": methods["linear_tail_ceiling_96"]["heldout_metrics_against_native"],
        "adaptive144": methods["adaptive_144"]["heldout_metrics_against_native"],
        "linear144": methods["linear_tail_ceiling_144"]["heldout_metrics_against_native"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

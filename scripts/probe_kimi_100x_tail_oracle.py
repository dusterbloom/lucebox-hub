#!/usr/bin/env python3
"""Falsify the K3 100x two-route aggregate-tail proposal.

The registered policy keeps the two routes with the largest causal
router-weight times calibration-response-norm score.  At AttnRes block starts
both experts are complete; elsewhere the intended runtime keeps six calibrated
slabs per expert.  This first gate is deliberately layer 12, the first routed
block start, so its exact retained result can be assembled from the archived
native per-expert responses without evaluating weights again.

The oracle fits a tail subspace on training sequences only.  Held-out
coefficients are projected from the true omitted aggregate and are therefore
not deployable.  A directional variant targets the shortest correction from
the retained result to the teacher ray, matching K3's routed RMSNorm geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from probe_kimi_neuron_slabs import (
    EXPERT_COUNT,
    FULL_ROUTED_GIB_PER_TOKEN,
    dequantize_part,
    metric_bundle,
    resolve_layer_tensors,
)
from probe_kimi_response_atlas import read_expert_responses, response_path
from train_kimi_panel_directional import load_data


ORACLE_RANKS = (64, 128, 256)
TOP_ROUTES = 2
ROUTED_LAYERS = 92
ROUTED_BLOCK_STARTS = 7  # zero-indexed layers 12,24,...,84; layer 0 is dense
REGISTERED_MEAN_GATE = 0.99
REGISTERED_P05_GATE = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=260815)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fit_state(path: Path, dimension: int) -> dict[str, np.ndarray]:
    expected = {
        "native_means": (EXPERT_COUNT, dimension),
        "native_expected_norm": (EXPERT_COUNT,),
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
    if not np.isfinite(result["native_means"]).all() or not np.isfinite(
        result["native_expected_norm"]
    ).all():
        raise ValueError("fit state contains non-finite values")
    calibrated = result["calibrated_experts"]
    if calibrated.dtype != np.uint8 or not np.all(
        (calibrated == 0) | (calibrated == 1)
    ):
        raise ValueError("invalid calibrated-expert mask")
    return result


def add_unique_rows(
    destination: np.ndarray, rows: np.ndarray, values: np.ndarray
) -> None:
    if rows.size != np.unique(rows).size:
        raise ValueError("one expert occurs twice in a token's native routes")
    destination[rows] += values


def shortest_ray_residual(base: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    """Return the minimum-L2 correction from base to teacher's positive ray."""
    numerator = np.einsum("ij,ij->i", base, teacher, dtype=np.float64)
    denominator = np.einsum("ij,ij->i", teacher, teacher, dtype=np.float64)
    scale = numerator / np.maximum(denominator, 1.0e-30)
    # A negative ray is not equivalent under RMSNorm.  The tiny positive floor
    # retains the correct direction while making the gate maximally optimistic.
    scale = np.maximum(scale, 1.0e-6).astype(np.float32)
    return scale[:, None] * teacher - base


def pca_oracle_candidates(
    residual: np.ndarray,
    base: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[int, np.ndarray]:
    maximum_rank = max(ORACLE_RANKS)
    if train.size <= maximum_rank:
        raise ValueError("not enough training rows for registered oracle ranks")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    training = torch.from_numpy(residual[train]).to(
        device=device, dtype=torch.float32
    )
    heldout = torch.from_numpy(residual[validation]).to(
        device=device, dtype=torch.float32
    )
    mean = training.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(
        training - mean, q=maximum_rank, center=False, niter=4
    )
    candidates: dict[int, np.ndarray] = {}
    with torch.no_grad():
        centered = heldout - mean
        for rank in ORACLE_RANKS:
            components = basis[:, :rank]
            projected = (centered @ components) @ components.T + mean
            candidates[rank] = base[validation] + projected.cpu().numpy()
    del training, heldout, mean, basis
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates


def main() -> int:
    args = parse_args()
    if args.layer != 12 or args.layer % 12 != 0:
        raise ValueError(
            "the first registered gate must be zero-indexed layer 12, an "
            "AttnRes block start where the two retained experts are complete"
        )
    if args.metric_batch <= 0:
        raise ValueError("metric batch must be positive")
    started = time.monotonic()
    data = load_data(args.capture, args.teacher)
    if data.model_layer != args.layer or data.top_k != 16:
        raise ValueError("capture layer/top-k disagrees with the registered gate")
    state = load_fit_state(args.fit_state, data.dimension)
    calibrated = state["calibrated_experts"].astype(bool)
    calibrated_norm = state["native_expected_norm"][calibrated]
    if not calibrated_norm.size:
        raise ValueError("fit state contains no calibrated experts")
    fallback_norm = float(np.median(calibrated_norm))
    fallback_mean = state["native_means"][calibrated].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    route_norm = np.where(
        calibrated, state["native_expected_norm"], fallback_norm
    )
    route_score = data.router_weights * route_norm[data.expert_ids]
    selected_rank = np.argsort(-route_score, axis=1, kind="stable")[:, :TOP_ROUTES]
    selected_mask = np.zeros_like(data.expert_ids, dtype=bool)
    np.put_along_axis(selected_mask, selected_rank, True, axis=1)

    base = np.zeros_like(data.teacher)
    coverage_control_base = np.zeros_like(data.teacher)
    selected_routes_seen = 0
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
            raise ValueError(f"response metadata mismatch for expert {expert}")
        mean = state["native_means"][expert] if calibrated[expert] else fallback_mean
        add_unique_rows(base, tokens, weights[:, None] * mean)
        if calibrated[expert]:
            add_unique_rows(
                coverage_control_base, tokens, weights[:, None] * mean
            )
        else:
            add_unique_rows(
                coverage_control_base, tokens, weights[:, None] * native
            )
        chosen = selected_mask[tokens, ranks]
        if np.any(chosen):
            correction = weights[chosen, None] * (native[chosen] - mean)
            add_unique_rows(base, tokens[chosen], correction)
            if calibrated[expert]:
                add_unique_rows(
                    coverage_control_base, tokens[chosen], correction
                )
            selected_routes_seen += int(np.count_nonzero(chosen))
    if selected_routes_seen != data.teacher.shape[0] * TOP_ROUTES:
        raise ValueError("did not reconstruct exactly two live routes per token")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tensors, tensor_sources, _readers = resolve_layer_tensors(
        args.shard, args.layer
    )
    gamma = torch.from_numpy(
        np.asarray(
            tensors[f"blk.{args.layer}.ffn_routed_norm.weight"].data,
            dtype=np.float32,
        ).copy()
    ).to(device)
    projection_tensor = tensors[f"blk.{args.layer}.ffn_routed_up.weight"]
    projection = dequantize_part(
        projection_tensor.data, projection_tensor.tensor_type
    ).to(device)
    validation = data.validation_indices
    methods: dict[str, object] = {
        "two_complete_routes_plus_mean_tail": {
            "runtime_deployable": True,
            "selection": "top two router-weight times calibration native-response norm",
            "uncalibrated_policy": "global calibrated mean/norm; no exact fallback",
            "metrics": metric_bundle(
                base[validation], data.teacher[validation], gamma, projection,
                1.0e-6, device, args.metric_batch,
            ),
        },
        "two_routes_with_exact_uncalibrated_control": {
            "runtime_deployable": False,
            "selection": "registered top two, but all uncalibrated routes are exact",
            "purpose": "falsify calibration coverage as the cause of oracle failure",
            "metrics": metric_bundle(
                coverage_control_base[validation], data.teacher[validation],
                gamma, projection, 1.0e-6, device, args.metric_batch,
            ),
        },
    }

    residuals = {
        "euclidean": data.teacher - base,
        "directional_ray": shortest_ray_residual(base, data.teacher),
    }
    for objective, residual in residuals.items():
        candidates = pca_oracle_candidates(
            residual, base, data.train_indices, validation, device, args.seed
        )
        for rank, candidate in candidates.items():
            methods[f"{objective}_pca_oracle_r{rank}"] = {
                "runtime_deployable": False,
                "basis_fit": "training sequences only",
                "heldout_coefficients": "oracle projection of true held-out correction",
                "metrics": metric_bundle(
                    candidate, data.teacher[validation], gamma, projection,
                    1.0e-6, device, args.metric_batch,
                ),
            }

    control_residual = shortest_ray_residual(
        coverage_control_base, data.teacher
    )
    control_candidates = pca_oracle_candidates(
        control_residual, coverage_control_base, data.train_indices,
        validation, device, args.seed
    )
    methods["directional_ray_pca_oracle_r128_exact_uncalibrated_control"] = {
        "runtime_deployable": False,
        "basis_fit": "training sequences only",
        "heldout_coefficients": "oracle projection of true held-out correction",
        "purpose": "optimistic exact-coverage falsification control",
        "metrics": metric_bundle(
            control_candidates[128], data.teacher[validation], gamma,
            projection, 1.0e-6, device, args.metric_batch,
        ),
    }

    registered = methods["directional_ray_pca_oracle_r128"]["metrics"][
        "post_shared_up_projection"
    ]["cosine"]
    passed = (
        float(registered["mean"]) >= REGISTERED_MEAN_GATE
        and float(registered["p05"]) >= REGISTERED_P05_GATE
    )
    average_exact_gib = (
        FULL_ROUTED_GIB_PER_TOKEN
        * TOP_ROUTES / data.top_k
        * 0.5
        + FULL_ROUTED_GIB_PER_TOKEN
        * TOP_ROUTES / data.top_k
        * 0.5
        * ROUTED_BLOCK_STARTS / ROUTED_LAYERS
    )
    result = {
        "schema": "kimi-k3-100x-two-route-tail-oracle-v1",
        "status": "ORACLE_GATE_PASS" if passed else "ORACLE_GATE_FAIL",
        "layer": args.layer,
        "policy": {
            "ordinary_layer": "two dynamic routes times six calibrated slabs",
            "block_start": "two dynamic routes complete (twelve slabs)",
            "all_other_routes": "no authoritative bytes; aggregate directional tail",
            "exact_fallback": False,
            "projected_average_authoritative_gib_per_token": average_exact_gib,
            "projected_expert_io_roofline_tokens_per_second_at_5_257_gib_s": (
                5.257263242095205 / average_exact_gib
            ),
            "projection_only": True,
        },
        "registered_gate": {
            "method": "directional_ray_pca_oracle_r128",
            "metric": "post-routed-up-projection cosine",
            "mean_at_least": REGISTERED_MEAN_GATE,
            "p05_at_least": REGISTERED_P05_GATE,
            "passed": passed,
        },
        "splits": {
            "training_tokens": int(data.train_indices.size),
            "development_tokens_unused": int(data.development_indices.size),
            "heldout_validation_tokens": int(validation.size),
            "whole_sequence_separation": True,
        },
        "coverage": {
            "calibrated_experts": int(np.count_nonzero(calibrated)),
            "uncalibrated_experts": int(np.count_nonzero(~calibrated)),
            "uncalibrated_routes": int(np.count_nonzero(
                ~calibrated[data.expert_ids]
            )),
            "selected_uncalibrated_routes": int(np.count_nonzero(
                selected_mask & ~calibrated[data.expert_ids]
            )),
            "selected_routes": selected_routes_seen,
        },
        "methods": methods,
        "artifacts": {
            "capture": str(args.capture),
            "capture_sha256": sha256(args.capture),
            "teacher": str(args.teacher),
            "teacher_sha256": sha256(args.teacher),
            "fit_state": str(args.fit_state),
            "fit_state_sha256": sha256(args.fit_state),
            "response_directory": str(args.response_directory),
            "tensor_sources": {
                name: str(path) for name, path in sorted(tensor_sources.items())
            },
        },
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": (
            "The rank-128 oracle sees the true held-out correction coefficients. "
            "Failure closes the registered block-Observer continuation because "
            "a causal predictor cannot exceed this fixed-basis ceiling."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "baseline": methods["two_complete_routes_plus_mean_tail"]["metrics"],
        "directional_r128": methods[
            "directional_ray_pca_oracle_r128"
        ]["metrics"],
        "projected_average_gib_per_token": average_exact_gib,
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

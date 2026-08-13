#!/usr/bin/env python3
"""Probe a storage-heavy per-expert response atlas on held-out Kimi states.

The native router remains authoritative.  For each selected expert, this probe
addresses calibration responses by cosine similarity in the real latent input,
retrieves one or several stored expert outputs, and performs the original
router-weighted reduction.  It never changes the registered diagonal result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

from train_kimi_panel_directional import load_data


RESPONSE_HEADER = struct.Struct("<8sIiiIQII2Q")
RESPONSE_MAGIC = b"K3RSP001"
RESPONSE_RECORD = np.dtype(
    [("token_index", "<u8"), ("rank", "<u4"), ("router_weight", "<f4")]
)
ATLAS_BUDGETS = (1, 4, 16, 64)
NEIGHBOR_COUNTS = (1, 4, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=260813)
    return parser.parse_args()


def response_path(directory: Path, expert: int) -> Path:
    return directory / f"expert_{expert:04d}.responses.f32"


def read_expert_responses(
    path: Path, expected_layer: int, expected_expert: int, expected_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as source:
        raw = source.read(RESPONSE_HEADER.size)
        if len(raw) != RESPONSE_HEADER.size:
            raise ValueError(f"truncated expert response header: {path}")
        (
            magic,
            version,
            model_layer,
            expert,
            dimension,
            route_count,
            storage,
            reserved,
            reserved0,
            reserved1,
        ) = RESPONSE_HEADER.unpack(raw)
        if (
            magic != RESPONSE_MAGIC
            or version != 1
            or model_layer != expected_layer
            or expert != expected_expert
            or dimension != expected_dimension
            or route_count <= 0
            or storage != 0
            or reserved != 0
            or reserved0 != 0
            or reserved1 != 0
        ):
            raise ValueError(f"invalid expert response header: {path}")
        records = np.fromfile(source, dtype=RESPONSE_RECORD, count=route_count)
        outputs = np.fromfile(
            source, dtype="<f4", count=route_count * expected_dimension
        )
        if (
            records.size != route_count
            or outputs.size != route_count * expected_dimension
            or source.read(1)
        ):
            raise ValueError(f"truncated or extended expert response: {path}")
    outputs = outputs.reshape(route_count, expected_dimension)
    if not np.isfinite(outputs).all():
        raise ValueError(f"non-finite expert response: {path}")
    return records, outputs


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(values, dim=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.50)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def pair_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1.0e-30)


def atlas_variants() -> list[tuple[str, int | None, int, str]]:
    variants: list[tuple[str, int | None, int, str]] = [
        ("mean", 0, 1, "none")
    ]
    for budget in ATLAS_BUDGETS:
        for neighbors in NEIGHBOR_COUNTS:
            if neighbors <= budget:
                variants.append(
                    (
                        f"exemplar_k{budget:03d}_n{neighbors:02d}",
                        budget,
                        neighbors,
                        "latent_cosine",
                    )
                )
    for neighbors in NEIGHBOR_COUNTS:
        variants.append(
            (f"exemplar_kall_n{neighbors:02d}", None, neighbors, "latent_cosine")
        )
    for budget in (64, None):
        label = "all" if budget is None else f"{budget:03d}"
        for neighbors in NEIGHBOR_COUNTS:
            variants.append(
                (
                    f"oracle_output_k{label}_n{neighbors:02d}",
                    budget,
                    neighbors,
                    "output_cosine_oracle",
                )
            )
    return variants


def main() -> int:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    data = load_data(args.capture, args.teacher)
    dimension = data.dimension
    token_count = data.latent.shape[0]
    validation_mask = np.zeros(token_count, dtype=bool)
    validation_mask[data.validation_indices] = True
    validation_position = np.full(token_count, -1, dtype=np.int64)
    validation_position[data.validation_indices] = np.arange(
        data.validation_indices.size, dtype=np.int64
    )
    variants = atlas_variants()
    aggregate = {
        name: np.zeros((data.validation_indices.size, dimension), dtype=np.float32)
        for name, _, _, _ in variants
    }
    individual_cosines: dict[str, list[np.ndarray]] = {
        name: [] for name, _, _, _ in variants
    }
    entries_by_variant = {name: 0 for name, _, _, _ in variants}
    route_records = 0
    calibration_records = 0
    validation_records = 0
    device = torch.device(args.device)

    for expert in range(896):
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
            np.any(token_indices < 0)
            or np.any(token_indices >= token_count)
            or np.any(ranks < 0)
            or np.any(ranks >= data.top_k)
            or not np.all(data.expert_ids[token_indices, ranks] == expert)
            or not np.array_equal(data.router_weights[token_indices, ranks], route_weights)
        ):
            raise ValueError(f"response metadata disagrees with capture for expert {expert}")
        is_validation = validation_mask[token_indices]
        calibration_rows = np.flatnonzero(~is_validation)
        validation_rows = np.flatnonzero(is_validation)
        if calibration_rows.size == 0:
            raise ValueError(f"expert {expert} has no calibration responses")
        route_records += records.size
        calibration_records += calibration_rows.size
        validation_records += validation_rows.size
        if validation_rows.size == 0:
            for name, budget, _, _ in variants:
                entries_by_variant[name] += (
                    1 if name == "mean" else min(calibration_rows.size, budget or calibration_rows.size)
                )
            continue

        calibration_tokens = token_indices[calibration_rows]
        validation_tokens = token_indices[validation_rows]
        z_calibration = normalize_rows(
            torch.from_numpy(data.latent[calibration_tokens]).to(device)
        )
        z_validation = normalize_rows(
            torch.from_numpy(data.latent[validation_tokens]).to(device)
        )
        y_calibration = torch.from_numpy(outputs[calibration_rows]).to(device)
        y_validation = torch.from_numpy(outputs[validation_rows]).to(device)
        similarity_all = z_validation @ z_calibration.T
        output_similarity_all = (
            normalize_rows(y_validation) @ normalize_rows(y_calibration).T
        )
        generator = torch.Generator(device=device).manual_seed(args.seed + expert)
        permutation = torch.randperm(
            calibration_rows.size, generator=generator, device=device
        )

        for name, budget, neighbors, address in variants:
            if name == "mean":
                estimate = y_calibration.mean(dim=0, keepdim=True).expand(
                    validation_rows.size, -1
                )
                entry_count = 1
            else:
                entry_count = min(
                    calibration_rows.size,
                    calibration_rows.size if budget is None else budget,
                )
                selected = permutation[:entry_count]
                similarities = (
                    output_similarity_all[:, selected]
                    if address == "output_cosine_oracle"
                    else similarity_all[:, selected]
                )
                used_neighbors = min(neighbors, entry_count)
                scores, local_indices = similarities.topk(used_neighbors, dim=1)
                selected_outputs = y_calibration[selected[local_indices]]
                if used_neighbors == 1:
                    estimate = selected_outputs[:, 0, :]
                else:
                    weights = torch.softmax(scores * args.temperature, dim=1)
                    estimate = (selected_outputs * weights[:, :, None]).sum(dim=1)
            entries_by_variant[name] += entry_count
            expert_cosine = torch.nn.functional.cosine_similarity(
                estimate, y_validation, dim=1
            ).detach().cpu().numpy()
            individual_cosines[name].append(expert_cosine)
            estimate_cpu = estimate.detach().cpu().numpy()
            positions = validation_position[validation_tokens]
            aggregate[name][positions] += route_weights[validation_rows, None] * estimate_cpu

    if route_records != token_count * data.top_k:
        raise ValueError("expert response set does not cover every native route")
    if validation_records != data.validation_indices.size * data.top_k:
        raise ValueError("expert response validation coverage is incomplete")
    exact = data.teacher[data.validation_indices]
    result_variants: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for name, _, neighbors, address in variants:
        aggregate_cosine = pair_cosine(aggregate[name], exact)
        expert_cosine = np.concatenate(individual_cosines[name])
        entries = entries_by_variant[name]
        bf16_response_layer_bytes = entries * dimension * 2
        bf16_address_layer_bytes = (
            0 if name == "mean" else bf16_response_layer_bytes
        )
        bf16_total_layer_bytes = (
            bf16_response_layer_bytes + bf16_address_layer_bytes
        )
        response_bytes_per_token = (
            92 * data.top_k * dimension * 2 * neighbors
        )
        address_scan_bytes_per_token = (
            0
            if name == "mean"
            else 92 * data.top_k * (entries / 896) * dimension * 2
        )
        result_variants[name] = {
            "stored_entries_layer": entries,
            "neighbors_read_per_route": neighbors,
            "address": address,
            "projected_bfloat16_response_all_layers_bytes": (
                bf16_response_layer_bytes * 92
            ),
            "projected_bfloat16_address_all_layers_bytes": (
                bf16_address_layer_bytes * 92
            ),
            "projected_bfloat16_all_layers_bytes": bf16_total_layer_bytes * 92,
            "projected_bfloat16_response_bytes_per_token": response_bytes_per_token,
            "projected_bfloat16_address_scan_bytes_per_token": (
                address_scan_bytes_per_token
            ),
            "projected_bfloat16_streamed_bytes_per_token": (
                response_bytes_per_token + address_scan_bytes_per_token
            ),
            "routed_aggregate_cosine": summarize(aggregate_cosine),
            "individual_expert_cosine": summarize(expert_cosine),
        }
        csv_rows.append(
            {
                "variant": name,
                "stored_entries_layer": entries,
                "neighbors_read_per_route": neighbors,
                "address": address,
                "aggregate_mean_cosine": float(aggregate_cosine.mean()),
                "aggregate_p05_cosine": float(np.quantile(aggregate_cosine, 0.05)),
                "expert_mean_cosine": float(expert_cosine.mean()),
                "projected_bfloat16_all_layers_gib": (
                    bf16_total_layer_bytes * 92 / (1 << 30)
                ),
                "projected_bfloat16_response_mebibytes_per_token": (
                    response_bytes_per_token / (1 << 20)
                ),
                "projected_bfloat16_address_scan_mebibytes_per_token": (
                    address_scan_bytes_per_token / (1 << 20)
                ),
            }
        )

    best_name = max(
        result_variants,
        key=lambda name: result_variants[name]["routed_aggregate_cosine"]["mean"],
    )
    result = {
        "schema": "kimi-k3-layer01-response-atlas-v1",
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
        "calibration_route_records": calibration_records,
        "validation_route_records": validation_records,
        "address": "full-latent cosine, plus explicitly labeled output-cosine oracle controls",
        "interpolation": f"softmax(similarity * {args.temperature:g})",
        "prototype_selection": f"nested deterministic random exemplars, seed {args.seed}",
        "best_variant": best_name,
        "variants": result_variants,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)
    best = result_variants[best_name]["routed_aggregate_cosine"]
    print(
        f"best={best_name} mean={best['mean']:.9f} "
        f"p05={best['p05']:.9f} entries={entries_by_variant[best_name]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

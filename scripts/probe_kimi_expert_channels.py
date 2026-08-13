#!/usr/bin/env python3
"""Probe internal channel sparsity on sampled real Kimi IQ1_S experts."""

from __future__ import annotations

import argparse
import json
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
KEEP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--sample-experts", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_data(args.capture, args.teacher)
    device = torch.device(args.device)
    reader = GGUFReader(args.shard, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate_tensor = tensors[f"blk.{args.layer}.ffn_gate_exps.weight"]
    up_tensor = tensors[f"blk.{args.layer}.ffn_up_exps.weight"]
    down_tensor = tensors[f"blk.{args.layer}.ffn_down_exps.weight"]
    sample_experts = np.linspace(
        0, EXPERT_COUNT - 1, args.sample_experts, dtype=np.int64
    )
    validation_mask = np.zeros(data.latent.shape[0], dtype=bool)
    validation_mask[data.validation_indices] = True
    native_full_cosines: list[np.ndarray] = []
    activation_energy: dict[float, list[np.ndarray]] = {
        fraction: [] for fraction in KEEP_FRACTIONS
    }
    dequant_full_cosines: dict[float, list[np.ndarray]] = {
        fraction: [] for fraction in KEEP_FRACTIONS
    }
    native_cosines: dict[float, list[np.ndarray]] = {
        fraction: [] for fraction in KEEP_FRACTIONS
    }
    route_counts: list[int] = []

    for expert_raw in sample_experts:
        expert = int(expert_raw)
        records, exact_outputs = read_expert_responses(
            response_path(args.response_directory, expert),
            data.model_layer,
            expert,
            data.dimension,
        )
        token_indices = records["token_index"].astype(np.int64, copy=False)
        rows = np.flatnonzero(validation_mask[token_indices])
        if rows.size == 0:
            continue
        route_counts.append(int(rows.size))
        z = torch.from_numpy(data.latent[token_indices[rows]]).to(device)
        native = torch.from_numpy(exact_outputs[rows]).to(device)
        gate = torch.from_numpy(
            quants.dequantize(gate_tensor.data[expert], gate_tensor.tensor_type)
        ).to(device)
        up = torch.from_numpy(
            quants.dequantize(up_tensor.data[expert], up_tensor.tensor_type)
        ).to(device)
        down = torch.from_numpy(
            quants.dequantize(down_tensor.data[expert], down_tensor.tensor_type)
        ).to(device)
        gate_value = z @ gate.T
        up_value = z @ up.T
        nonlinear = 4.0 * torch.tanh(gate_value / 4.0) * torch.sigmoid(gate_value)
        linear = 25.0 * torch.tanh(up_value / 25.0)
        activated = nonlinear * linear
        dequant_full = activated @ down.T
        native_full_cosines.append(
            torch.nn.functional.cosine_similarity(
                dequant_full, native, dim=1
            ).cpu().numpy()
        )
        energy_total = activated.square().sum(dim=1).clamp_min(1.0e-30)
        order = activated.abs().argsort(dim=1, descending=True)
        for fraction in KEEP_FRACTIONS:
            keep = max(1, int(round(activated.shape[1] * fraction)))
            indices = order[:, :keep]
            retained = torch.gather(activated, 1, indices)
            activation_energy[fraction].append(
                (retained.square().sum(dim=1) / energy_total).cpu().numpy()
            )
            truncated = torch.zeros_like(activated)
            truncated.scatter_(1, indices, retained)
            estimate = truncated @ down.T
            dequant_full_cosines[fraction].append(
                torch.nn.functional.cosine_similarity(
                    estimate, dequant_full, dim=1
                ).cpu().numpy()
            )
            native_cosines[fraction].append(
                torch.nn.functional.cosine_similarity(
                    estimate, native, dim=1
                ).cpu().numpy()
            )
        del gate, up, down, gate_value, up_value, nonlinear, linear, activated

    native_full = np.concatenate(native_full_cosines)
    variants: dict[str, dict[str, object]] = {}
    for fraction in KEEP_FRACTIONS:
        energy = np.concatenate(activation_energy[fraction])
        to_dequant = np.concatenate(dequant_full_cosines[fraction])
        to_native = np.concatenate(native_cosines[fraction])
        # Gate and up are always needed to discover active channels. Only the
        # down-projection fraction can be paged after that discovery.
        payload_fraction = 2.0 / 3.0 + fraction / 3.0
        variants[f"keep_{fraction:g}"] = {
            "channel_fraction": fraction,
            "channels": int(round(3072 * fraction)),
            "activation_energy_fraction": summarize(energy),
            "cosine_to_dequantized_full_output": summarize(to_dequant),
            "cosine_to_native_exact_output": summarize(to_native),
            "ideal_expert_payload_fraction": payload_fraction,
            "ideal_expert_payload_gib_per_token": 8.844 * payload_fraction,
            "ideal_maximum_speedup_if_only_io_bound": 1.0 / payload_fraction,
        }

    result = {
        "schema": "kimi-k3-layer01-expert-channel-sparsity-v1",
        "status": "EXPLORATORY",
        "shard": str(args.shard),
        "capture": str(args.capture),
        "response_directory": str(args.response_directory),
        "model_layer": data.model_layer,
        "sample_experts": list(map(int, sample_experts)),
        "sample_validation_routes": int(sum(route_counts)),
        "validation_routes_per_sampled_expert": summarize(np.asarray(route_counts)),
        "dequantized_full_vs_native_exact_cosine": summarize(native_full),
        "selection": "oracle top absolute SiTU-GLU activation per route",
        "variants": variants,
        "warnings": [
            "Oracle channel selection is an upper bound, not an implementable predictor.",
            "Gate and up tensors must still be read before exact active channels are known.",
            "Quantized down rows are block packed, so practical scattered reads may exceed the ideal byte fraction.",
            "Python dequantization may not be bit-identical to the native GPU quantized kernel.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    ten = variants["keep_0.1"]
    print(
        f"sample-routes={sum(route_counts)} "
        f"full-vs-native={native_full.mean():.6f} "
        f"keep10-native={ten['cosine_to_native_exact_output']['mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

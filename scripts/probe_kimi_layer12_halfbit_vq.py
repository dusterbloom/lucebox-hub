#!/usr/bin/env python3
"""Matched-byte full-width weight-codec probe for one real Kimi K3 layer.

This is deliberately a *rate/distortion* experiment, not an inference path.
It keeps every routed expert's 3,072 live intermediate neurons and replaces
the current IQ1_S bytes with a deterministic vector-quantized representation
whose logical payload is exactly half the current routed-weight traffic:

  * 8-weight vectors, 64 entries: 6 / 8 = 0.75 bits per weight;
  * one FP16 RMS scale per 512 weights: 0.03125 bits per weight.

The resulting 0.78125 bits/weight is 3,225,600 bytes per active expert, or
51,609,600 bytes for K3's 16 live experts.  That is the same raw active-bank
budget as retaining 96/192 exact 256-neuron slabs.  Codebooks are shared per
layer/component and charged separately as resident metadata.

This probe starts from the already-deployed IQ1_S/IQ2_XXS weights, not native
BF16 K3 weights.  It reports:

  (a) native captured teacher vs Python-dequantized full expert control;
  (b) VQ full-width aggregate vs that same teacher;
  (c) held-out comparison against the recorded 96-slab mean-tail and its
      non-deployable selector oracle.

No weight bytes are physically packed or served by this script, so it makes
no NVMe or speed claim.  A positive result earns a packed-codec implementation
and a terminal-logit replay; a negative static codec result is not a theorem
against every possible activation-aware sub-one-bit quantizer.
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

from probe_kimi_halfwidth_frontier import metric_bundle
from probe_kimi_neuron_slabs import (
    EXPERT_COUNT,
    ORIGINAL_EXPERT_WIDTH,
    dequantize_part,
    resolve_layer_tensors,
    situ_expert,
)
from probe_kimi_response_atlas import read_expert_responses, response_path
from train_kimi_panel_directional import load_data


TOP_K = 16
VECTOR_WIDTH = 8
CODEBOOK_ENTRIES = 64
SCALE_GROUP = 512
SCALE_BYTES = 2
CODE_BITS = 6
DEFAULT_SEED = 260815


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("response_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rms-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--sample-vectors-per-expert", type=int, default=128)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--nearest-batch", type=int, default=65536)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    dot = np.einsum("ij,ij->i", candidate, teacher, dtype=np.float64)
    denom = np.linalg.norm(candidate, axis=1) * np.linalg.norm(teacher, axis=1)
    cosine = dot / np.maximum(denom, 1.0e-30)
    relative_l2 = np.linalg.norm(candidate - teacher, axis=1) / np.maximum(
        np.linalg.norm(teacher, axis=1), 1.0e-30
    )
    return {
        "cosine": summarize(cosine),
        "relative_l2": summarize(relative_l2),
    }, cosine.astype(np.float32), relative_l2.astype(np.float32)


def normalized_vectors(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized 8-vectors and its per-512-value RMS scales.

    Every routed K3 component has its input dimension divisible by 512:
    gate/up are [3072, 3584] and down is [3584, 3072].  Keeping scales inside
    each such group makes the byte count exact and avoids a hidden FP32 scale
    side channel.
    """
    if weight.ndim != 2 or weight.shape[1] % SCALE_GROUP:
        raise ValueError(f"weight shape cannot use 512-value groups: {tuple(weight.shape)}")
    rows, columns = weight.shape
    groups = weight.reshape(rows, columns // SCALE_GROUP, SCALE_GROUP)
    scales = torch.sqrt(torch.mean(groups.square(), dim=2, keepdim=True)).clamp_min(1.0e-12)
    vectors = (groups / scales).reshape(-1, VECTOR_WIDTH)
    return vectors, scales.squeeze(-1)


def nearest_labels(vectors: torch.Tensor, codebook: torch.Tensor, batch: int) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError("unexpected vector shape")
    labels = torch.empty(vectors.shape[0], dtype=torch.int64, device=vectors.device)
    code_norm = codebook.square().sum(dim=1)
    for begin in range(0, vectors.shape[0], batch):
        end = min(vectors.shape[0], begin + batch)
        part = vectors[begin:end]
        distances = part.square().sum(dim=1, keepdim=True) + code_norm - 2.0 * (part @ codebook.T)
        labels[begin:end] = torch.argmin(distances, dim=1)
    return labels


def train_codebook(
    samples: np.ndarray,
    device: torch.device,
    iterations: int,
    nearest_batch: int,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """Small deterministic Lloyd k-means with no hidden library dependency."""
    if samples.ndim != 2 or samples.shape[1] != VECTOR_WIDTH:
        raise ValueError("invalid codebook samples")
    if samples.shape[0] < CODEBOOK_ENTRIES:
        raise ValueError("insufficient samples for codebook")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    points = torch.from_numpy(samples).to(device=device, dtype=torch.float32)
    initial = torch.randperm(points.shape[0], generator=generator, device=device)[:CODEBOOK_ENTRIES]
    codebook = points[initial].clone()
    objectives: list[float] = []
    for _ in range(iterations):
        sums = torch.zeros_like(codebook)
        counts = torch.zeros(CODEBOOK_ENTRIES, device=device, dtype=torch.float32)
        squared_error = 0.0
        count = 0
        for begin in range(0, points.shape[0], nearest_batch):
            end = min(points.shape[0], begin + nearest_batch)
            part = points[begin:end]
            labels = nearest_labels(part, codebook, nearest_batch)
            reconstruction = codebook[labels]
            squared_error += float((part - reconstruction).square().sum().item())
            count += int(part.numel())
            sums.index_add_(0, labels, part)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        nonempty = counts > 0
        codebook[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = torch.nonzero(~nonempty, as_tuple=False).flatten()
        if empty.numel():
            resample = torch.randint(points.shape[0], (empty.numel(),), generator=generator, device=device)
            codebook[empty] = points[resample]
        objectives.append(squared_error / max(count, 1))
    return codebook, objectives


def reconstruct_vq(
    weight: torch.Tensor,
    codebook: torch.Tensor,
    nearest_batch: int,
) -> torch.Tensor:
    # The ledger charges FP16 scales and codebooks.  Recreate from those exact
    # stored precisions rather than quietly using the FP32 training values.
    rows, columns = weight.shape
    groups = weight.reshape(rows, columns // SCALE_GROUP, SCALE_GROUP)
    scales = torch.sqrt(torch.mean(groups.square(), dim=2)).clamp_min(1.0e-12)
    scales = scales.to(torch.float16).to(torch.float32)
    vectors = (groups / scales[:, :, None]).reshape(-1, VECTOR_WIDTH)
    labels = nearest_labels(vectors, codebook, nearest_batch)
    reconstructed = codebook[labels].reshape(
        weight.shape[0], weight.shape[1] // SCALE_GROUP, SCALE_GROUP
    ) * scales[:, :, None]
    return reconstructed.reshape_as(weight)


def component_samples(
    tensor: object,
    component: str,
    sample_vectors_per_expert: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collect a deterministic, equal-expert static weight sample for VQ."""
    pieces: list[np.ndarray] = []
    for expert in range(EXPERT_COUNT):
        weight = dequantize_part(tensor.data[expert], tensor.tensor_type)
        vectors, _ = normalized_vectors(weight)
        values = vectors.cpu().numpy()
        if values.shape[0] < sample_vectors_per_expert:
            raise ValueError(f"{component} expert {expert} has too few vectors")
        chosen = rng.choice(values.shape[0], size=sample_vectors_per_expert, replace=False)
        pieces.append(values[chosen])
        if (expert + 1) % 128 == 0:
            print(f"[halfbit-vq] sampling {component}: {expert + 1}/{EXPERT_COUNT}", flush=True)
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def aggregate_validation(
    data: object,
    response_directory: Path,
    gate_tensor: object,
    up_tensor: object,
    down_tensor: object,
    codebooks: dict[str, torch.Tensor],
    device: torch.device,
    nearest_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native reconstructed, full-dequantized, and VQ aggregates.

    Evaluation uses only the untouched validation sequences.  Static VQ
    codebooks are learned from checkpoint weights only, never validation
    activations or outputs.
    """
    token_count = data.latent.shape[0]
    validation = data.validation_indices
    position = np.full(token_count, -1, dtype=np.int64)
    position[validation] = np.arange(validation.size, dtype=np.int64)
    native_total = np.zeros((validation.size, data.dimension), dtype=np.float32)
    full_total = np.zeros_like(native_total)
    vq_total = np.zeros_like(native_total)
    mask = np.zeros(token_count, dtype=bool)
    mask[validation] = True

    for expert in range(EXPERT_COUNT):
        records, native = read_expert_responses(
            response_path(response_directory, expert), data.model_layer, expert, data.dimension
        )
        tokens = records["token_index"].astype(np.int64, copy=False)
        ranks = records["rank"].astype(np.int64, copy=False)
        weights = records["router_weight"].astype(np.float32, copy=False)
        take = mask[tokens]
        if not np.any(take):
            continue
        tokens = tokens[take]
        ranks = ranks[take]
        weights = weights[take]
        native = native[take]
        if not np.all(data.expert_ids[tokens, ranks] == expert):
            raise ValueError(f"response metadata disagrees for expert {expert}")
        positions = position[tokens]
        z = torch.from_numpy(data.latent[tokens]).to(device=device, dtype=torch.float32)
        gate = dequantize_part(gate_tensor.data[expert], gate_tensor.tensor_type).to(device)
        up = dequantize_part(up_tensor.data[expert], up_tensor.tensor_type).to(device)
        down = dequantize_part(down_tensor.data[expert], down_tensor.tensor_type).to(device)
        with torch.no_grad():
            full = situ_expert(z, gate, up, down).cpu().numpy()
            vq_gate = reconstruct_vq(gate, codebooks["gate"], nearest_batch)
            vq_up = reconstruct_vq(up, codebooks["up"], nearest_batch)
            vq_down = reconstruct_vq(down, codebooks["down"], nearest_batch)
            vq = situ_expert(z, vq_gate, vq_up, vq_down).cpu().numpy()
        native_total[positions] += weights[:, None] * native
        full_total[positions] += weights[:, None] * full
        vq_total[positions] += weights[:, None] * vq
        del z, gate, up, down, vq_gate, vq_up, vq_down, full, vq
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (expert + 1) % 32 == 0 or expert + 1 == EXPERT_COUNT:
            print(f"[halfbit-vq] validation experts={expert + 1}/{EXPERT_COUNT}", flush=True)
    return native_total, full_total, vq_total


def sequence_bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    data: object,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """Paired whole-sequence bootstrap for mean-cosine improvement."""
    validation = data.validation_indices
    seq_positions: list[np.ndarray] = []
    for start, end in data.sequence_ranges:
        selected = np.flatnonzero((validation >= start) & (validation < end))
        if selected.size:
            seq_positions.append(selected)
    if len(seq_positions) < 2:
        raise ValueError("need multiple held-out sequences for block bootstrap")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    distribution = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        choices = rng.integers(0, len(seq_positions), size=len(seq_positions))
        rows = np.concatenate([seq_positions[choice] for choice in choices])
        distribution[index] = float(delta[rows].mean())
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "replicates": int(repeats),
        "resampling_unit": "whole held-out sequence",
    }


def byte_ledger() -> dict[str, object]:
    values_per_expert = 3 * ORIGINAL_EXPERT_WIDTH * 3584
    # gate/up are 3072 x 3584, down is 3584 x 3072; all have equal count.
    if values_per_expert % SCALE_GROUP or values_per_expert % VECTOR_WIDTH:
        raise ValueError("codec groups do not tile K3 expert")
    # Each 6-bit code labels one *eight-weight* vector, not one scalar.
    index_bytes = values_per_expert // VECTOR_WIDTH * CODE_BITS // 8
    scale_bytes = values_per_expert // SCALE_GROUP * SCALE_BYTES
    packed = index_bytes + scale_bytes
    exact_active = packed * TOP_K * 2
    codebook = 3 * CODEBOOK_ENTRIES * VECTOR_WIDTH * SCALE_BYTES
    return {
        "values_per_expert": values_per_expert,
        "codec": {
            "vector_width": VECTOR_WIDTH,
            "codebook_entries": CODEBOOK_ENTRIES,
            "code_bits_per_vector": CODE_BITS,
            "scale_group_values": SCALE_GROUP,
            "scale_storage": "FP16",
            "indices_bytes_per_expert": index_bytes,
            "scale_bytes_per_expert": scale_bytes,
            "packed_bytes_per_expert": packed,
            "packed_bits_per_weight": packed * 8 / values_per_expert,
            "resident_codebook_bytes_per_layer": codebook,
        },
        "active_route": {
            "experts": TOP_K,
            "current_iq1s_bytes": packed * TOP_K * 2,
            "halfbyte_target_bytes": packed * TOP_K,
            "codec_bytes": packed * TOP_K,
            "codec_fraction_of_current_iq1s": 0.5,
            "includes": "indices plus FP16 per-512-value scales",
            "excludes": "resident layer codebooks; 3,072 bytes per layer",
        },
    }


def main() -> int:
    args = parse_args()
    if args.layer < 1 or args.sample_vectors_per_expert <= 0:
        raise ValueError("invalid layer or sample count")
    if args.kmeans_iterations <= 0 or args.nearest_batch <= 0:
        raise ValueError("invalid VQ settings")
    for output in (args.output_json, args.output_csv):
        if output and output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
    for input_path in (args.shard, args.capture, args.teacher, args.baseline_json, args.baseline_npz):
        if not input_path.exists():
            raise FileNotFoundError(input_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    data = load_data(args.capture, args.teacher)
    if data.model_layer != args.layer or data.top_k != TOP_K or data.dimension != 3584:
        raise ValueError("capture does not describe the requested K3 routed layer")
    tensors, sources, _ = resolve_layer_tensors(args.shard, args.layer)
    gate_tensor = tensors[f"blk.{args.layer}.ffn_gate_exps.weight"]
    up_tensor = tensors[f"blk.{args.layer}.ffn_up_exps.weight"]
    down_tensor = tensors[f"blk.{args.layer}.ffn_down_exps.weight"]
    norm_tensor = tensors[f"blk.{args.layer}.ffn_routed_norm.weight"]
    projection_tensor = tensors[f"blk.{args.layer}.ffn_routed_up.weight"]

    rng = np.random.default_rng(args.seed)
    codebooks: dict[str, torch.Tensor] = {}
    training: dict[str, object] = {}
    for offset, (name, tensor) in enumerate((
        ("gate", gate_tensor), ("up", up_tensor), ("down", down_tensor),
    )):
        print(f"[halfbit-vq] collecting static {name} codebook sample", flush=True)
        samples = component_samples(tensor, name, args.sample_vectors_per_expert, rng)
        print(f"[halfbit-vq] Lloyd k-means {name} ({samples.shape[0]} vectors)", flush=True)
        codebook, objectives = train_codebook(
            samples, device, args.kmeans_iterations, args.nearest_batch, args.seed + offset
        )
        # Pack contract: the three layer-level codebooks are FP16.  Decode them
        # back to FP32 here only to make the arithmetic comparison stable.
        codebooks[name] = codebook.to(torch.float16).to(torch.float32)
        training[name] = {
            "sample_vectors": int(samples.shape[0]),
            "sample_vectors_per_expert": args.sample_vectors_per_expert,
            "iterations": args.kmeans_iterations,
            "objective_per_scalar": objectives,
        }
        del samples
        if device.type == "cuda":
            torch.cuda.empty_cache()

    native, full, vq = aggregate_validation(
        data, args.response_directory, gate_tensor, up_tensor, down_tensor,
        codebooks, device, args.nearest_batch,
    )
    teacher = data.teacher[data.validation_indices]
    if not np.allclose(native, teacher, rtol=1.0e-5, atol=1.0e-5):
        maxabs = float(np.max(np.abs(native - teacher)))
        raise ValueError(f"native response files fail to reconstruct validation teacher: maxabs={maxabs}")
    gamma = dequantize_part(norm_tensor.data, norm_tensor.tensor_type).to(device)
    projection = dequantize_part(projection_tensor.data, projection_tensor.tensor_type).to(device)
    native_full, full_cosine, full_rel = pair_metrics(full, teacher)
    native_vq, vq_cosine, vq_rel = pair_metrics(vq, teacher)
    vq_full, vq_full_cosine, vq_full_rel = pair_metrics(vq, full)
    downstream = metric_bundle(vq, teacher, gamma, projection, args.rms_epsilon, device, args.metric_batch)
    baseline = json.loads(args.baseline_json.read_text())
    baseline_npz = np.load(args.baseline_npz, allow_pickle=False)
    expected_rows = data.validation_indices.size
    oracle96 = np.asarray(baseline_npz["oracle_96_cosine"], dtype=np.float32)
    mean96 = np.asarray(baseline_npz["adaptive_96_cosine"], dtype=np.float32)
    if oracle96.shape != (expected_rows,) or mean96.shape != (expected_rows,):
        raise ValueError("baseline NPZ validation rows disagree with current capture")
    comparison = {
        "mean_tail_96": baseline["methods"]["adaptive_96"]["heldout_metrics_against_native"],
        "selector_oracle_96": baseline["methods"]["oracle_96"]["heldout_metrics_against_native"],
        "vq_minus_mean_tail96_cosine": sequence_bootstrap_delta(
            vq_cosine, mean96, data, args.bootstrap_replicates, args.seed + 1
        ),
        "vq_minus_selector_oracle96_cosine": sequence_bootstrap_delta(
            vq_cosine, oracle96, data, args.bootstrap_replicates, args.seed + 2
        ),
        "pass_rule": (
            "only a positive whole-sequence 95% lower confidence bound against the "
            "selector-oracle-96 cosine earns terminal-KL replay"
        ),
    }
    result = {
        "schema": "kimi-k3-layer-halfbit-vq-v1",
        "status": "EXPLORATORY_LOCAL_RATE_DISTORTION_ONLY",
        "purpose": "all 192 live functions at half current active expert-weight bytes versus 96 exact slabs plus mean tail",
        "model_layer": args.layer,
        "split": {
            "heldout_validation_tokens": int(expected_rows),
            "whole_sequence_separation": True,
            "static_codebook_training": "checkpoint weights only; no captured activation or target output was used",
        },
        "input": {
            "shard": str(args.shard),
            "capture": str(args.capture),
            "capture_sha256": sha256(args.capture),
            "teacher": str(args.teacher),
            "teacher_sha256": sha256(args.teacher),
            "baseline_json": str(args.baseline_json),
            "baseline_json_sha256": sha256(args.baseline_json),
            "baseline_npz": str(args.baseline_npz),
            "baseline_npz_sha256": sha256(args.baseline_npz),
            "response_directory": str(args.response_directory),
            "tensor_sources": {name: str(path) for name, path in sorted(sources.items())},
        },
        "codec_training": training,
        "byte_ledger": byte_ledger(),
        "metrics": {
            "full_python_dequantized_vs_native_teacher": native_full,
            "vq_fullwidth_vs_native_teacher": native_vq,
            "vq_fullwidth_vs_python_dequantized": vq_full,
            "vq_fullwidth_downstream_local": downstream,
        },
        "comparisons": comparison,
        "row_metrics": {
            "full_dequantized_cosine": full_cosine,
            "full_dequantized_relative_l2": full_rel,
            "vq_cosine": vq_cosine,
            "vq_relative_l2": vq_rel,
            "vq_vs_full_cosine": vq_full_cosine,
            "vq_vs_full_relative_l2": vq_full_rel,
        },
        "elapsed_seconds": time.monotonic() - started,
        "warnings": [
            "This is a logical codec byte ledger, not a packed sidecar or NVMe benchmark.",
            "The source is the deployed IQ1_S/IQ2_XXS weight representation, not the official native checkpoint.",
            "Local routed-output fidelity is not final-logit KL or free-generation quality.",
            "A static weight-only VQ failure does not falsify every possible activation-aware sub-one-bit codec.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            columns = ["method", "mean_cosine", "p05_cosine", "mean_relative_l2"]
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for method, metrics in (
                ("full_python_dequantized_vs_native_teacher", native_full),
                ("vq_fullwidth_vs_native_teacher", native_vq),
                ("vq_fullwidth_vs_python_dequantized", vq_full),
                ("mean_tail_96", comparison["mean_tail_96"]),
                ("selector_oracle_96", comparison["selector_oracle_96"]),
            ):
                writer.writerow({
                    "method": method,
                    "mean_cosine": metrics["cosine"]["mean"],
                    "p05_cosine": metrics["cosine"]["p05"],
                    "mean_relative_l2": metrics["relative_l2"]["mean"],
                })
    print(json.dumps({
        "vq_fullwidth_vs_native_teacher": native_vq,
        "vq_minus_oracle96": comparison["vq_minus_selector_oracle96_cosine"],
        "byte_ledger": result["byte_ledger"]["active_route"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

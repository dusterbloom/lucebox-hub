#!/usr/bin/env python3
"""Cheap captured-state screen for STQ1_0 K3 slab coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, quants


CAPTURE_HEADER = struct.Struct("<8sIiIIQQII4Q")
CAPTURE_RECORD = struct.Struct("<IB3sI")
RESPONSE_HEADER = struct.Struct("<8sIiiIQII2Q")
RESPONSE_RECORD = np.dtype(
    [("token_index", "<u8"), ("rank", "<u4"), ("router_weight", "<f4")]
)
DIMENSION = 3584
TOP_K = 16
ROUTE_LIMIT = 12
SLAB_SIZE = 256
SLAB_COUNT = 12
IQ1S_COMPONENT_BYTES = 179_200
STQ_COMPONENT_BYTES = 150_528

POLICIES = {
    "route12_iq1s_b16": (16, "exact", 16 * 3 * IQ1S_COMPONENT_BYTES, 16),
    "route12_iq1s_b20": (20, "exact", 20 * 3 * IQ1S_COMPONENT_BYTES, 20),
    "route12_stq1_b19": (19, "stq", 19 * 3 * STQ_COMPONENT_BYTES, 0),
    "route12_stq1_gu_iq1s_down_b18": (
        18,
        "stq_gu",
        18 * (2 * STQ_COMPONENT_BYTES + IQ1S_COMPONENT_BYTES),
        0,
    ),
    "route12_iq1s_b8_stq1_tail9": (
        17,
        "stq",
        8 * 3 * IQ1S_COMPONENT_BYTES + 9 * 3 * STQ_COMPONENT_BYTES,
        8,
    ),
    "route12_iq1s_b6_stq1_tail12": (
        18,
        "stq",
        6 * 3 * IQ1S_COMPONENT_BYTES + 12 * 3 * STQ_COMPONENT_BYTES,
        6,
    ),
    "route12_iq1s_b8_stq1_gu_iq1s_down_tail9": (
        17,
        "stq_gu",
        8 * 3 * IQ1S_COMPONENT_BYTES +
        9 * (2 * STQ_COMPONENT_BYTES + IQ1S_COMPONENT_BYTES),
        8,
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def combined_digest(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(path.name.encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                value.update(block)
    return value.hexdigest()


def bf16_to_float(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32, copy=False) << 16).view(np.float32)


def read_validation_rows(
        path: Path, expected_layer: int, count: int, skip: int = 0,
        step: int = 1) -> dict[str, object]:
    selected: dict[str, list[object]] = {
        "token_index": [], "token_id": [], "latent": [], "expert_ids": [],
        "router_weights": [], "sequence_id": [],
    }
    with path.open("rb") as source:
        raw = source.read(CAPTURE_HEADER.size)
        if len(raw) != CAPTURE_HEADER.size:
            raise ValueError("truncated capture header")
        (magic, version, layer, dimension, top_k, sequence_count, token_count,
         latent_storage, weight_storage, *reserved) = CAPTURE_HEADER.unpack(raw)
        if (magic != b"K3PNL001" or version != 1 or layer != expected_layer or
                dimension != DIMENSION or top_k != TOP_K or latent_storage != 1 or
                weight_storage != 0 or any(reserved)):
            raise ValueError("unsupported capture geometry")
        cursor = 0
        validation_cursor = 0
        for _ in range(sequence_count):
            raw = source.read(CAPTURE_RECORD.size)
            if len(raw) != CAPTURE_RECORD.size:
                raise ValueError("truncated capture record")
            identifier_bytes, split, padding, rows = CAPTURE_RECORD.unpack(raw)
            if not identifier_bytes or split not in (0, 1) or padding != b"\0\0\0" or not rows:
                raise ValueError("invalid capture record")
            identifier = source.read(identifier_bytes).decode()
            tokens = np.fromfile(source, dtype="<i4", count=rows)
            latent = np.fromfile(source, dtype="<u2", count=rows * DIMENSION)
            experts = np.fromfile(source, dtype="<i4", count=rows * TOP_K)
            weights = np.fromfile(source, dtype="<f4", count=rows * TOP_K)
            if (tokens.size != rows or latent.size != rows * DIMENSION or
                    experts.size != rows * TOP_K or weights.size != rows * TOP_K):
                raise ValueError("truncated capture payload")
            if split == 1 and len(selected["token_index"]) < count:
                local = np.arange(rows, dtype=np.int64)
                ordinal = validation_cursor + local
                chosen = local[(ordinal >= skip) & ((ordinal - skip) % step == 0)]
                chosen = chosen[:count - len(selected["token_index"])]
                selected["token_index"].extend((cursor + chosen).tolist())
                selected["token_id"].extend(tokens[chosen].tolist())
                selected["latent"].extend(
                    bf16_to_float(latent).reshape(rows, DIMENSION)[chosen])
                selected["expert_ids"].extend(
                    experts.reshape(rows, TOP_K)[chosen])
                selected["router_weights"].extend(
                    weights.reshape(rows, TOP_K)[chosen])
                selected["sequence_id"].extend([identifier] * len(chosen))
            if split == 1:
                validation_cursor += rows
            cursor += rows
        if cursor != token_count or source.read(1):
            raise ValueError("capture extent mismatch")
    if len(selected["token_index"]) != count:
        raise ValueError(f"capture has fewer than {count} validation rows")
    result = {
        "token_index": np.asarray(selected["token_index"], dtype=np.int64),
        "token_id": np.asarray(selected["token_id"], dtype=np.int32),
        "latent": np.asarray(selected["latent"], dtype=np.float32),
        "expert_ids": np.asarray(selected["expert_ids"], dtype=np.int32),
        "router_weights": np.asarray(selected["router_weights"], dtype=np.float32),
        "sequence_id": selected["sequence_id"],
    }
    if not np.allclose(result["router_weights"].sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("router weights do not sum to one")
    return result


def git_provenance(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> bytes:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True, capture_output=True,
        ).stdout

    head = git("rev-parse", "HEAD").decode().strip()
    branch = git("branch", "--show-current").decode().strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    patch = hashlib.sha256()
    patch.update(git("diff", "--binary", "HEAD"))
    for raw in status.decode().splitlines():
        if not raw.startswith("?? "):
            continue
        relative = raw[3:]
        path = root / relative
        patch.update(relative.encode())
        patch.update(path.read_bytes())
    return {
        "branch": branch,
        "commit": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "patch_sha256": patch.hexdigest(),
    }


def read_responses(path: Path, layer: int, expert: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as source:
        raw = source.read(RESPONSE_HEADER.size)
        if len(raw) != RESPONSE_HEADER.size:
            raise ValueError(f"truncated response header: {path}")
        (magic, version, observed_layer, observed_expert, dimension, routes,
         storage, reserved, reserved0, reserved1) = RESPONSE_HEADER.unpack(raw)
        if (magic != b"K3RSP001" or version != 1 or observed_layer != layer or
                observed_expert != expert or dimension != DIMENSION or storage or
                reserved or reserved0 or reserved1):
            raise ValueError(f"invalid response header: {path}")
        records = np.fromfile(source, dtype=RESPONSE_RECORD, count=routes)
        outputs = np.fromfile(source, dtype="<f4", count=routes * DIMENSION)
        if records.size != routes or outputs.size != routes * DIMENSION or source.read(1):
            raise ValueError(f"response extent mismatch: {path}")
    return records, outputs.reshape(routes, DIMENSION)


def resolve_tensors(model: Path, layer: int) -> tuple[dict[str, object], list[GGUFReader], list[Path]]:
    directory = model if model.is_dir() else model.parent
    names = {
        "gate": f"blk.{layer}.ffn_gate_exps.weight",
        "up": f"blk.{layer}.ffn_up_exps.weight",
        "down": f"blk.{layer}.ffn_down_exps.weight",
    }
    found: dict[str, object] = {}
    readers: list[GGUFReader] = []
    sources: list[Path] = []
    for shard in sorted(directory.glob("*.gguf")):
        reader = GGUFReader(shard, "r")
        readers.append(reader)
        for tensor in reader.tensors:
            for family, name in names.items():
                if tensor.name == name:
                    found[family] = tensor
                    sources.append(shard)
        if len(found) == len(names):
            break
    if set(found) != set(names):
        raise KeyError(f"missing layer {layer} tensors: {sorted(set(names) - set(found))}")
    return found, readers, sorted(set(sources))


def dequantize(data: np.ndarray, tensor_type: object) -> np.ndarray:
    return np.asarray(
        quants.dequantize(np.ascontiguousarray(data), tensor_type),
        dtype=np.float32,
    )


def stq1_emulate(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Reproduce Hy4's three-iteration LS STQ values without packing bytes."""
    source = np.asarray(values, dtype=np.float32)
    if source.shape[-1] % 256:
        raise ValueError("STQ1_0 requires rows divisible by 256")
    blocks = source.reshape(-1, 256)
    groups = blocks.reshape(-1, 4, 4, 16).transpose(0, 1, 3, 2)
    sumx2 = np.sum(blocks * blocks, axis=1, dtype=np.float64).astype(np.float32)
    sigma2 = (2.0 / 256.0) * sumx2
    weights = np.sqrt(sigma2[:, None, None, None] + groups * groups)
    scale = np.max(np.abs(blocks), axis=1).astype(np.float32)
    selection = np.zeros_like(groups, dtype=np.float32)
    lanes = np.arange(4)
    for _ in range(3):
        cost = weights * (groups * groups - (np.abs(groups) - scale[:, None, None, None]) ** 2)
        zero = np.argmin(cost, axis=-1)
        selection = np.where(groups < 0.0, -1.0, 1.0).astype(np.float32)
        selection[lanes[None, None, None, :] == zero[..., None]] = 0.0
        numerator = np.sum(weights * selection * groups, axis=(1, 2, 3), dtype=np.float64)
        denominator = np.sum(weights * selection * selection, axis=(1, 2, 3), dtype=np.float64)
        updated = np.divide(numerator, denominator, out=scale.astype(np.float64), where=denominator > 0)
        scale = np.where(updated > 0.0, updated, scale).astype(np.float32)
    stored_scale = scale.astype(np.float16).astype(np.float32)
    reconstructed = (
        selection * stored_scale[:, None, None, None]
    ).transpose(0, 1, 3, 2).reshape(source.shape)
    zero_fraction = float(np.count_nonzero(selection == 0.0) / selection.size)
    if not math.isclose(zero_fraction, 0.25, abs_tol=1e-12):
        raise ValueError(f"STQ 3:4 invariant failed: zero_fraction={zero_fraction}")
    error = reconstructed - source
    relative_l2 = float(
        np.linalg.norm(error.astype(np.float64)) /
        max(np.linalg.norm(source.astype(np.float64)), 1e-30)
    )
    return reconstructed, {
        "relative_l2": relative_l2,
        "zero_fraction": zero_fraction,
        "blocks": int(blocks.shape[0]),
        "encoded_bytes": int(blocks.shape[0] * 42),
    }


def situ_expert(z: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    gate_value = z @ gate.T
    up_value = z @ up.T
    sigmoid = np.empty_like(gate_value)
    positive = gate_value >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-gate_value[positive]))
    exp_value = np.exp(gate_value[~positive])
    sigmoid[~positive] = exp_value / (1.0 + exp_value)
    nonlinear = 4.0 * np.tanh(gate_value / 4.0) * sigmoid
    linear = 25.0 * np.tanh(up_value / 25.0)
    return (nonlinear * linear) @ down.T


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    numerator = np.linalg.norm(difference, axis=1)
    denominator = np.linalg.norm(reference.astype(np.float64), axis=1)
    relative = numerator / np.maximum(denominator, 1e-30)
    cosine = np.einsum("ij,ij->i", candidate, reference, dtype=np.float64) / np.maximum(
        np.linalg.norm(candidate, axis=1) * denominator, 1e-30
    )
    return {
        "relative_l2": {
            "mean": float(relative.mean()), "median": float(np.median(relative)),
            "p95": float(np.quantile(relative, 0.95)), "maximum": float(relative.max()),
            "rows": [float(value) for value in relative],
        },
        "cosine": {
            "mean": float(cosine.mean()), "minimum": float(cosine.min()),
            "rows": [float(value) for value in cosine],
        },
    }


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        generator = np.random.default_rng(260901)
        source = generator.normal(size=(3, 512)).astype(np.float32)
        reconstructed, diagnostics = stq1_emulate(source)
        assert reconstructed.shape == source.shape
        assert np.isfinite(reconstructed).all()
        assert diagnostics["blocks"] == 6
        assert diagnostics["encoded_bytes"] == 252
        assert diagnostics["zero_fraction"] == 0.25
        print(json.dumps({"self_test": "PASS", **diagnostics}, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("responses", type=Path)
    parser.add_argument("fit_state", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--row-step", type=int, default=1)
    parser.add_argument(
        "--gate-arm", choices=tuple(POLICIES),
        default="route12_iq1s_b8_stq1_gu_iq1s_down_tail9",
    )
    parser.add_argument("--gate-mode", choices=("screen", "confirm"), default="confirm")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.skip_rows < 0:
        raise ValueError("--skip-rows must be nonnegative")
    if args.row_step <= 0:
        raise ValueError("--row-step must be positive")
    started = time.monotonic()
    capture = read_validation_rows(
        args.capture, args.layer, args.rows, args.skip_rows, args.row_step)
    with np.load(args.fit_state, allow_pickle=False) as state:
        slab_means = state["slab_means"].astype(np.float32, copy=False)
        residual = state["slab_expected_residual_norm"].astype(np.float32, copy=False)
        calibrated = state["calibrated_experts"].astype(bool, copy=False)
    if slab_means.shape != (896, SLAB_COUNT, DIMENSION) or residual.shape != (896, SLAB_COUNT):
        raise ValueError("fit-state geometry mismatch")
    experts = capture["expert_ids"][:, :ROUTE_LIMIT]
    route_weights = capture["router_weights"][:, :ROUTE_LIMIT]
    if not calibrated[experts].all():
        raise ValueError("frozen rows contain an uncalibrated route")
    natural_order = np.argsort(-residual, axis=1, kind="stable")

    selections: dict[str, list[dict[tuple[int, int], str]]] = {}
    tasks: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for name, (budget, tail_mode, _bytes, exact_prefix) in POLICIES.items():
        selections[name] = []
        for position in range(args.rows):
            candidates: list[tuple[float, int, int, int]] = []
            for route in range(ROUTE_LIMIT):
                expert = int(experts[position, route])
                for rank in range(SLAB_COUNT):
                    natural = int(natural_order[expert, rank])
                    candidates.append((
                        abs(float(route_weights[position, route])) * float(residual[expert, natural]),
                        route, rank, natural,
                    ))
            chosen = sorted(candidates, key=lambda item: -item[0])[:budget]
            selected = {
                (route, natural): "exact" if ordinal < exact_prefix else tail_mode
                for ordinal, (_, route, _, natural) in enumerate(chosen)
            }
            selections[name].append(selected)
            for route, natural in selected:
                key = (int(experts[position, route]), natural)
                tasks.setdefault(key, set()).add((position, route))

    reference = np.zeros((args.rows, DIMENSION), dtype=np.float32)
    coverage = np.zeros((args.rows, ROUTE_LIMIT), dtype=bool)
    token_to_position = {
        int(token): position for position, token in enumerate(capture["token_index"])
    }
    response_paths: list[Path] = []
    for expert in sorted(set(int(value) for value in experts.flat)):
        path = args.responses / f"expert_{expert:04d}.responses.f32"
        records, outputs = read_responses(path, args.layer, expert)
        response_paths.append(path)
        for index, record in enumerate(records):
            position = token_to_position.get(int(record["token_index"]))
            route = int(record["rank"])
            if position is None or route >= ROUTE_LIMIT:
                continue
            if int(experts[position, route]) != expert:
                raise ValueError("response route identity mismatch")
            if not math.isclose(
                    float(record["router_weight"]), float(route_weights[position, route]),
                    rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("response route weight mismatch")
            reference[position] += float(record["router_weight"]) * outputs[index]
            coverage[position, route] = True
    if not coverage.all():
        raise ValueError("native route12 response reference is incomplete")

    base = np.zeros_like(reference)
    for position in range(args.rows):
        for route in range(ROUTE_LIMIT):
            expert = int(experts[position, route])
            base[position] += float(route_weights[position, route]) * slab_means[expert].sum(axis=0)
    estimates = {name: base.copy() for name in POLICIES}
    tensors, readers, tensor_sources = resolve_tensors(args.model, args.layer)
    stq_diagnostics: dict[str, list[float]] = {"gate": [], "up": [], "down": []}
    print(json.dumps({"tasks": len(tasks), "rows": args.rows}), flush=True)
    for ordinal, ((expert, natural), occurrences) in enumerate(sorted(tasks.items()), 1):
        positions = sorted({position for position, _ in occurrences})
        z = capture["latent"][positions]
        begin, end = natural * SLAB_SIZE, (natural + 1) * SLAB_SIZE
        gate = dequantize(tensors["gate"].data[expert, begin:end], tensors["gate"].tensor_type)
        up = dequantize(tensors["up"].data[expert, begin:end], tensors["up"].tensor_type)
        down_row_bytes = int(tensors["down"].data.shape[2]) // SLAB_COUNT
        down = dequantize(
            tensors["down"].data[expert, :, natural * down_row_bytes:(natural + 1) * down_row_bytes],
            tensors["down"].tensor_type,
        )
        exact_output = situ_expert(z, gate, up, down)
        stq_gate, gate_diag = stq1_emulate(gate)
        stq_up, up_diag = stq1_emulate(up)
        stq_down, down_diag = stq1_emulate(down)
        stq_output = situ_expert(z, stq_gate, stq_up, stq_down)
        stq_gu_output = situ_expert(z, stq_gate, stq_up, down)
        stq_diagnostics["gate"].append(gate_diag["relative_l2"])
        stq_diagnostics["up"].append(up_diag["relative_l2"])
        stq_diagnostics["down"].append(down_diag["relative_l2"])
        output_by_mode = {"exact": exact_output, "stq": stq_output, "stq_gu": stq_gu_output}
        position_index = {position: index for index, position in enumerate(positions)}
        mean = slab_means[expert, natural]
        for name, (_budget, _tail_mode, _bytes, _exact_prefix) in POLICIES.items():
            for position, route in occurrences:
                selected_mode = selections[name][position].get((route, natural))
                if selected_mode is None:
                    continue
                estimates[name][position] += float(route_weights[position, route]) * (
                    output_by_mode[selected_mode][position_index[position]] - mean
                )
        if ordinal % 8 == 0 or ordinal == len(tasks):
            print(json.dumps({"completed": ordinal, "tasks": len(tasks)}), flush=True)

    policy_results = {}
    for name, (budget, tail_mode, payload_bytes, exact_prefix) in POLICIES.items():
        policy_results[name] = {
            "budget": budget,
            "exact_prefix": exact_prefix,
            "tail_mode": tail_mode,
            "payload_bytes_per_routed_layer": payload_bytes,
            "metrics_vs_native_route12": metrics(estimates[name], reference),
        }
    b16 = policy_results["route12_iq1s_b16"]["metrics_vs_native_route12"]["relative_l2"]["mean"]
    stq_arms = tuple(name for name in POLICIES if "stq1" in name)
    best_stq_name = min(
        stq_arms,
        key=lambda name: policy_results[name]["metrics_vs_native_route12"]["relative_l2"]["mean"],
    )
    mixed_arms = tuple(name for name in stq_arms if "tail" in name)
    best_mixed_name = min(
        mixed_arms,
        key=lambda name: policy_results[name]["metrics_vs_native_route12"]["relative_l2"]["mean"],
    )
    control_metrics = policy_results["route12_iq1s_b16"]["metrics_vs_native_route12"]["relative_l2"]
    gate_metrics = policy_results[args.gate_arm]["metrics_vs_native_route12"]["relative_l2"]
    gate_arm = gate_metrics["mean"]
    gate_bytes = policy_results[args.gate_arm]["payload_bytes_per_routed_layer"]
    b16_bytes = policy_results["route12_iq1s_b16"]["payload_bytes_per_routed_layer"]
    row_wins = sum(
        candidate < control
        for candidate, control in zip(gate_metrics["rows"], control_metrics["rows"])
    )
    within_bytes = gate_bytes <= 1.01 * b16_bytes
    if args.gate_mode == "confirm":
        robust = gate_metrics["median"] <= control_metrics["median"] and row_wins > args.rows // 2
        interpretation = (
            "GO_TO_TERMINAL" if gate_arm <= b16 and robust and within_bytes
            else "RETAIN_LOW_PRIORITY" if gate_arm <= b16 and within_bytes
            else "NO_GO"
        )
    else:
        interpretation = (
            "GO_TO_TERMINAL" if gate_arm <= b16 and within_bytes
            else "RETAIN_LOW_PRIORITY" if gate_arm <= 1.2 * b16 and within_bytes
            else "NO_GO"
        )
    result = {
        "schema": "dflash.kimi-k3.stq1-tail-local-screen.v1",
        "status": "MEASURED_LOCAL_SCREEN",
        "interpretation": interpretation,
        "best_stq_arm": best_stq_name,
        "best_mixed_arm": best_mixed_name,
        "gate_arm": args.gate_arm,
        "gate_mode": args.gate_mode,
        "gate_arm_row_wins": row_wins,
        "source": {
            **git_provenance(Path(__file__).resolve().parent.parent),
            "native_model_revision": "a0836360ce58dfec088d966a97f2ddc8a606279b",
            "hy4_repository_revision": "779242edccdedc2109a0b36b164263a88f015bfa",
        },
        "command": sys.argv,
        "hardware": {"platform": platform.platform(), "numpy": np.__version__},
        "inputs": {
            "model": str(args.model),
            "tensor_sources": [str(path) for path in tensor_sources],
            "capture": str(args.capture), "capture_sha256": digest(args.capture),
            "fit_state": str(args.fit_state), "fit_state_sha256": digest(args.fit_state),
            "responses": str(args.responses),
            "used_response_files": len(response_paths),
            "used_responses_sha256": combined_digest(response_paths),
            "token_indices": capture["token_index"].tolist(),
            "token_ids": capture["token_id"].tolist(),
            "sequence_ids": capture["sequence_id"],
        },
        "selector": {
            "route_limit": ROUTE_LIMIT,
            "definition": "abs(router_weight) * 10K mean omitted-slab residual norm",
            "tail": "10K per-expert per-slab mean",
        },
        "stq": {
            "definition": "Hy4 three-iteration weighted-LS STQ1_0 without imatrix term; fp16 block scale",
            "physical_reads_measured": False,
            "hip_kernel_used": False,
            "component_reconstruction_relative_l2": {
                family: {
                    "mean": float(np.mean(values)),
                    "maximum": float(np.max(values)),
                    "samples": len(values),
                }
                for family, values in stq_diagnostics.items()
            },
        },
        "policies": policy_results,
        "elapsed_seconds": time.monotonic() - started,
        "terminal_kl": None,
        "limitations": [
            "Local routed-output error is not terminal KL and is known to rank slab value weakly.",
            "STQ values are emulated after dequantizing the IQ1_S teacher; no STQ file or physical byte path was exercised.",
            "This result cannot establish behavior, output quality, bytes/token, decode speed, prefill speed, or a HIP-kernel GO.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "interpretation": interpretation}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

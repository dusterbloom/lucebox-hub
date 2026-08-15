#!/usr/bin/env python3
"""Turn an isolated-layer terminal-KL sweep into H22 budget tables.

The sweep suite is allocation calibration, never the final quality set.  Each
row changes one routed layer at budget 96 while the paired native pass restores
the exact trajectory.  Local calibration curves supply the shape across the
registered budget ladder; terminal KL supplies the layer-specific behavioral
weight.  No result from the later end-to-end prompts enters this planner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from compare_kimi_logits import load_trace, log_softmax


BUDGETS = np.asarray([48, 72, 96, 120, 144, 168, 192], dtype=np.int32)
LAYERS = 92
EXPERTS = 896
TOP_K = 16
CAPTURE_HEADER = struct.Struct("<8sIiIIQQII4Q")
CAPTURE_RECORD = struct.Struct("<IB3sI")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_routes(path: Path, expected_layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_ids: list[np.ndarray] = []
    validation_ids: list[np.ndarray] = []
    validation_weights: list[np.ndarray] = []
    with path.open("rb") as source:
        raw = source.read(CAPTURE_HEADER.size)
        if len(raw) != CAPTURE_HEADER.size:
            raise ValueError(f"truncated capture {path}")
        (magic, version, layer, dimension, top_k, sequences, tokens,
         latent_storage, weight_storage, *reserved) = CAPTURE_HEADER.unpack(raw)
        if (
            magic != b"K3PNL001" or version != 1 or layer != expected_layer
            or dimension != 3584 or top_k != TOP_K or sequences <= 0
            or tokens <= 0 or latent_storage != 1 or weight_storage != 0
            or any(reserved)
        ):
            raise ValueError(f"incompatible capture {path}")
        observed = 0
        for _ in range(sequences):
            raw = source.read(CAPTURE_RECORD.size)
            if len(raw) != CAPTURE_RECORD.size:
                raise ValueError(f"truncated capture record {path}")
            identifier_bytes, split, record_reserved, count = CAPTURE_RECORD.unpack(raw)
            if split not in (0, 1) or record_reserved != b"\0\0\0" or count <= 0:
                raise ValueError(f"invalid capture record {path}")
            source.seek(identifier_bytes + count * 4 + count * dimension * 2, 1)
            ids = np.fromfile(source, dtype="<i4", count=count * top_k).reshape(count, top_k)
            weights = np.fromfile(source, dtype="<f4", count=count * top_k).reshape(count, top_k)
            if split == 0:
                calibration_ids.append(ids.reshape(-1))
            else:
                validation_ids.append(ids)
                validation_weights.append(weights)
            observed += count
        if observed != tokens or source.tell() != path.stat().st_size:
            raise ValueError(f"capture extent mismatch {path}")
    return (
        np.concatenate(calibration_ids),
        np.concatenate(validation_ids),
        np.concatenate(validation_weights),
    )


def local_proxy_curve(capture: Path, fit_state: Path, layer: int) -> dict[str, object]:
    calibration_ids, validation_ids, validation_weights = read_routes(capture, layer)
    hits = np.bincount(calibration_ids, minlength=EXPERTS)
    runtime_calibrated = hits >= 8
    with np.load(fit_state, allow_pickle=False) as state:
        importance = np.asarray(state["slab_expected_residual_norm"], dtype=np.float64)
    samples = np.zeros((validation_ids.shape[0], BUDGETS.size), dtype=np.float64)
    fallback_routes = np.zeros(validation_ids.shape[0], dtype=np.int32)
    for token, (ids, weights) in enumerate(zip(validation_ids, validation_weights)):
        calibrated = runtime_calibrated[ids]
        fallback_routes[token] = int((~calibrated).sum())
        scores = (
            np.abs(weights[calibrated, None].astype(np.float64))
            * importance[ids[calibrated]]
        ).reshape(-1)
        scores.sort()
        scores = scores[::-1]
        total = float(scores.sum())
        if total == 0.0:
            continue
        cumulative = np.cumsum(scores)
        for index, budget in enumerate(BUDGETS):
            retained = float(cumulative[min(int(budget), scores.size) - 1]) if scores.size else 0.0
            samples[token, index] = max(0.0, (total - retained) / total)
    return {
        "curve": samples.mean(axis=0),
        "runtime_calibrated_experts": int(runtime_calibrated.sum()),
        "validation_fallback_routes_per_token": float(fallback_routes.mean()),
    }


def terminal_metrics(teacher_path: Path, candidate_path: Path) -> dict[str, object]:
    teacher_header, teacher, teacher_raw = load_trace(teacher_path)
    candidate_header, candidate, _ = load_trace(candidate_path)
    if teacher_header != candidate_header or teacher.shape != candidate.shape:
        raise ValueError("teacher/candidate trace shape mismatch")
    teacher_logp = log_softmax(teacher.astype(np.float64))
    candidate_logp = log_softmax(candidate.astype(np.float64))
    probability = np.exp(teacher_logp)
    kl = np.sum(probability * (teacher_logp - candidate_logp), axis=1)
    teacher_top = teacher.argmax(axis=1)
    candidate_top = candidate.argmax(axis=1)
    return {
        "rows": int(teacher.shape[0]),
        "kl_mean": float(kl.mean()),
        "kl_maximum": float(kl.max()),
        "top1_agreement": int((teacher_top == candidate_top).sum()),
        "maximum_absolute_logit_difference": float(np.abs(teacher - candidate).max()),
        "teacher_trace_sha256": hashlib.sha256(teacher_raw).hexdigest(),
        "candidate_trace_sha256": sha256(candidate_path),
    }


def optimize(cost: np.ndarray, target_average: int) -> np.ndarray:
    """Exact dynamic program in 24-slab units at a fixed nominal budget."""
    capacity = LAYERS * target_average // 24
    infinity = np.inf
    table = np.full((LAYERS + 1, capacity + 1), infinity, dtype=np.float64)
    previous = np.full((LAYERS + 1, capacity + 1), -1, dtype=np.int16)
    table[0, 0] = 0.0
    for layer in range(LAYERS):
        for used in np.flatnonzero(np.isfinite(table[layer])):
            for index, budget in enumerate(BUDGETS):
                next_used = int(used + budget // 24)
                if next_used > capacity:
                    continue
                value = table[layer, used] + cost[layer, index]
                if value < table[layer + 1, next_used]:
                    table[layer + 1, next_used] = value
                    previous[layer + 1, next_used] = index
    if not np.isfinite(table[LAYERS, capacity]):
        raise ValueError(f"cannot allocate average budget {target_average}")
    allocation = np.empty(LAYERS, dtype=np.int32)
    used = capacity
    for layer in range(LAYERS, 0, -1):
        index = int(previous[layer, used])
        if index < 0:
            raise ValueError("broken allocation backtrace")
        allocation[layer - 1] = BUDGETS[index]
        used -= int(BUDGETS[index] // 24)
    return allocation


def write_budget_table(path: Path, allocation: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write("# H22 behavioral allocation: model_layer nominal_slab_budget\n")
        for layer, budget in enumerate(allocation, 1):
            output.write(f"{layer} {int(budget)}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas_directory", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("fit_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("budget_directory", type=Path)
    args = parser.parse_args()

    manifest_path = args.atlas_directory / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sequences = manifest.get("sequences", [])
    if len(sequences) != LAYERS:
        raise ValueError("H22 atlas must contain 92 layer rows")
    by_layer = {int(row["model_layer"]): row for row in sequences}
    if set(by_layer) != set(range(1, LAYERS + 1)):
        raise ValueError("H22 atlas must name each layer exactly once")

    rows: list[dict[str, object]] = []
    exact_hashes: set[str] = set()
    for layer in range(1, LAYERS + 1):
        sequence = by_layer[layer]
        terminal = terminal_metrics(
            Path(sequence["teacher_logits"]), Path(sequence["candidate_logits"])
        )
        exact_hashes.add(str(terminal["teacher_trace_sha256"]))
        proxy = local_proxy_curve(
            args.capture_root / f"kimi_layer{layer:02d}_2048.bin",
            args.fit_root / f"kimi_layer{layer:02d}_neuron_slabs_calibration.npz",
            layer,
        )
        curve = np.asarray(proxy.pop("curve"), dtype=np.float64)
        q96 = max(float(curve[2]), 1e-12)
        sensitivity = float(terminal["kl_mean"]) / (q96 * q96)
        predicted = sensitivity * curve * curve
        rows.append({
            "model_layer": layer,
            **terminal,
            **proxy,
            "local_omitted_proxy": curve.tolist(),
            "behavioral_sensitivity": sensitivity,
            "predicted_cost": predicted.tolist(),
        })
    if len(exact_hashes) != 1:
        raise ValueError("native atlas teacher was not repeatable across layers")

    cost = np.asarray([row["predicted_cost"] for row in rows], dtype=np.float64)
    allocations: dict[str, object] = {}
    for target in (96, 120, 144):
        allocation = optimize(cost, target)
        path = args.budget_directory / f"h22_behavioral_avg{target}.txt"
        write_budget_table(path, allocation)
        allocations[str(target)] = {
            "average_nominal_slabs": target,
            "table": str(path),
            "table_sha256": sha256(path),
            "budget_counts": {
                str(int(budget)): int((allocation == budget).sum())
                for budget in BUDGETS
            },
            "predicted_additive_cost": float(sum(
                cost[layer, int(np.flatnonzero(BUDGETS == allocation[layer])[0])]
                for layer in range(LAYERS)
            )),
        }
        for layer, budget in enumerate(allocation):
            rows[layer][f"allocation_avg{target}"] = int(budget)

    result = {
        "schema": "kimi-k3-h22-layer-behavior-atlas-v1",
        "status": "MEASURED_ATLAS_PROJECTED_ALLOCATIONS",
        "protocol": {
            "atlas_role": "allocation calibration only",
            "intervention": "one routed layer at calibrated budget 96; every other layer exact",
            "teacher": "paired native pass on restored exact state",
            "local_curve": "mean omitted sum(abs(router_weight)*calibration_residual_norm) fraction",
            "cost_model": "KL_at_96 * (local_proxy_at_budget/local_proxy_at_96)^2",
            "registered_budgets": BUDGETS.tolist(),
            "limitations": [
                "A one-stream atlas is a pilot behavioral prior, not a quality result.",
                "Predicted layer costs are not assumed additive under all-layer composition.",
                "End-to-end prompts are held out from allocation construction.",
            ],
        },
        "provenance": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "repeatable_teacher_trace_sha256": next(iter(exact_hashes)),
        },
        "allocations": allocations,
        "layers": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as output:
        fieldnames = [
            "model_layer", "kl_mean", "kl_maximum", "top1_agreement",
            "maximum_absolute_logit_difference", "behavioral_sensitivity",
            "runtime_calibrated_experts", "validation_fallback_routes_per_token",
            *[f"proxy_{budget}" for budget in BUDGETS],
            "allocation_avg96", "allocation_avg120", "allocation_avg144",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fieldnames if key in row}
            for index, budget in enumerate(BUDGETS):
                flat[f"proxy_{budget}"] = row["local_omitted_proxy"][index]
            writer.writerow(flat)
    print(json.dumps({
        "status": result["status"],
        "teacher_sha256": next(iter(exact_hashes)),
        "allocations": allocations,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

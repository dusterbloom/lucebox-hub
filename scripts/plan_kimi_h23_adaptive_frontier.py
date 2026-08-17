#!/usr/bin/env python3
"""Project H23 slab-only global policies from the frozen H22 evidence.

This is deliberately a planner, not a quality evaluator.  The only measured
behavioral point per layer is H22's isolated terminal KL at budget 96.  Other
budgets are priced by the frozen local omitted-residual curve, exactly as in
H22.  Physical byte costs come from an archived real P27 route trace, including
the observed per-token exact-fallback decisions and mixed-qtype layer sizes.

The output therefore distinguishes MEASURED byte geometry from PROJECTED
behavioral damage.  It never calls a projected policy quality-preserving.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


LAYERS = 92
EXPERTS = 896
TOP_K = 16
DIMENSION = 3584
SLABS = 12
BUDGETS = np.asarray([24, 48, 72, 96, 120, 144, 168, 192], dtype=np.int32)
CAPTURE_HEADER = struct.Struct("<8sIiIIQQII4Q")
CAPTURE_RECORD = struct.Struct("<IB3sI")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_calibration_ids_and_validation_routes(
    path: Path, expected_layer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            or dimension != DIMENSION or top_k != TOP_K or sequences <= 0
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
            source.seek(identifier_bytes + count * 4 + count * DIMENSION * 2, 1)
            ids = np.fromfile(source, dtype="<i4", count=count * TOP_K).reshape(count, TOP_K)
            weights = np.fromfile(source, dtype="<f4", count=count * TOP_K).reshape(count, TOP_K)
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


def local_proxy_curve(capture: Path, fit_state: Path, layer: int) -> np.ndarray:
    calibration_ids, validation_ids, validation_weights = (
        read_calibration_ids_and_validation_routes(capture, layer)
    )
    hits = np.bincount(calibration_ids, minlength=EXPERTS)
    calibrated = hits >= 8
    with np.load(fit_state, allow_pickle=False) as state:
        importance = np.asarray(state["slab_expected_residual_norm"], dtype=np.float64)
    samples = np.zeros((validation_ids.shape[0], BUDGETS.size), dtype=np.float64)
    for token, (ids, weights) in enumerate(zip(validation_ids, validation_weights)):
        active = calibrated[ids]
        scores = (
            np.abs(weights[active, None].astype(np.float64))
            * importance[ids[active]]
        ).reshape(-1)
        scores.sort()
        scores = scores[::-1]
        total = float(scores.sum())
        if total == 0.0:
            continue
        cumulative = np.cumsum(scores)
        for index, budget in enumerate(BUDGETS):
            count = min(int(budget), scores.size)
            retained = float(cumulative[count - 1]) if count else 0.0
            samples[token, index] = max(0.0, (total - retained) / total)
    return samples.mean(axis=0)


def read_trace_fallbacks(path: Path) -> dict[int, list[int]]:
    """Return exact-fallback route counts for every layer/position group."""
    groups: dict[tuple[str, int, int, int], int] = defaultdict(int)
    seen: set[tuple[str, int, int, int]] = set()
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {
            "prompt_id", "base_pos", "token_index", "model_layer",
            "exact_fallback",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"incompatible I/O trace {path}")
        for row in reader:
            key = (
                row["prompt_id"], int(row["base_pos"]),
                int(row["token_index"]), int(row["model_layer"]),
            )
            seen.add(key)
            if int(row["exact_fallback"]):
                groups[key] += 1
    by_layer: dict[int, list[int]] = {layer: [] for layer in range(1, LAYERS + 1)}
    for key in sorted(seen):
        by_layer[key[3]].append(groups[key])
    lengths = {len(values) for values in by_layer.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("route trace does not cover every layer equally")
    return by_layer


def read_trace_routes(path: Path) -> dict[int, list[list[int]]]:
    """Return all 16 routed expert IDs for every layer/position group.

    The physical I/O trace may contain multiple rows for one expert (for
    example gate/up/down) or a metadata-only row for a zero-depth prefix.  A
    set therefore recovers the logical route without counting physical tensor
    regions.  This lets a newer calibration mask be replayed on the exact same
    frozen trajectory instead of inheriting stale fallback decisions.
    """
    groups: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {
            "prompt_id", "base_pos", "token_index", "model_layer", "expert_id",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"incompatible I/O trace {path}")
        for row in reader:
            key = (
                row["prompt_id"], int(row["base_pos"]),
                int(row["token_index"]), int(row["model_layer"]),
            )
            groups[key].add(int(row["expert_id"]))
    by_layer: dict[int, list[list[int]]] = {
        layer: [] for layer in range(1, LAYERS + 1)
    }
    for key in sorted(groups):
        routed = sorted(groups[key])
        if len(routed) != TOP_K:
            raise ValueError(
                f"trace group {key} has {len(routed)} experts, expected {TOP_K}"
            )
        by_layer[key[3]].append(routed)
    lengths = {len(values) for values in by_layer.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("route trace does not cover every layer equally")
    return by_layer


def recompute_trace_fallbacks(
    path: Path, fit_root: Path
) -> dict[int, list[int]]:
    routes = read_trace_routes(path)
    result: dict[int, list[int]] = {}
    for layer in range(1, LAYERS + 1):
        state_path = (
            fit_root / f"kimi_layer{layer:02d}_neuron_slabs_calibration.npz"
        )
        with np.load(state_path, allow_pickle=False) as state:
            calibrated = np.asarray(state["calibrated_experts"], dtype=bool)
        if calibrated.shape != (EXPERTS,):
            raise ValueError(f"invalid calibrated mask in {state_path}")
        result[layer] = [
            sum(not calibrated[expert] for expert in routed)
            for routed in routes[layer]
        ]
    return result


def read_layer_geometry(path: Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for row in reader:
            layer = int(row["model_layer"])
            selected = int(row["selected_slab_records"])
            selected_bytes = int(row["selected_sidecar_bytes"])
            fallback = int(row["exact_fallback_routes"])
            fallback_bytes = int(row["exact_fallback_bytes"])
            if selected <= 0 or selected_bytes % selected:
                raise ValueError(f"cannot infer slab bytes for layer {layer}")
            slab_bytes = selected_bytes // selected
            if fallback:
                if fallback_bytes % fallback:
                    raise ValueError(f"cannot infer expert bytes for layer {layer}")
                expert_bytes = fallback_bytes // fallback
            else:
                expert_bytes = slab_bytes * SLABS
            if expert_bytes != slab_bytes * SLABS:
                raise ValueError(f"layer {layer} expert/slab geometry is not exact")
            result[layer] = (slab_bytes, expert_bytes)
    if set(result) != set(range(1, LAYERS + 1)):
        raise ValueError("traffic table must contain all 92 layers")
    return result


def measured_option_bytes(
    fallback_counts: list[int], slab_bytes: int, expert_bytes: int, budget: int
) -> float:
    total = 0
    for fallback in fallback_counts:
        if budget == TOP_K * SLABS:
            total += TOP_K * expert_bytes
            continue
        calibrated = TOP_K - fallback
        selected = min(budget, calibrated * SLABS)
        total += fallback * expert_bytes + selected * slab_bytes
    return total / len(fallback_counts)


def optimize(cost: np.ndarray, byte_cost: np.ndarray, target_bytes: int) -> np.ndarray | None:
    """Minimize projected cost under a conservative one-MiB byte capacity."""
    unit = 1 << 20
    weights = np.ceil(byte_cost / unit).astype(np.int32)
    capacity = target_bytes // unit
    minimum = int(weights.min(axis=1).sum())
    if minimum > capacity:
        return None
    table = np.full(capacity + 1, np.inf, dtype=np.float64)
    table[0] = 0.0
    previous: list[np.ndarray] = []
    for layer in range(LAYERS):
        next_table = np.full_like(table, np.inf)
        choice = np.full(capacity + 1, -1, dtype=np.int16)
        parent = np.full(capacity + 1, -1, dtype=np.int32)
        for index in range(BUDGETS.size):
            weight = int(weights[layer, index])
            source = table[: capacity + 1 - weight]
            candidate = source + cost[layer, index]
            destination = next_table[weight:]
            improve = candidate < destination
            positions = np.flatnonzero(improve)
            if positions.size:
                destination[positions] = candidate[positions]
                targets = positions + weight
                choice[targets] = index
                parent[targets] = positions
        table = next_table
        previous.append(np.stack((choice, parent)))
    used = int(np.nanargmin(table))
    if not np.isfinite(table[used]):
        return None
    allocation = np.empty(LAYERS, dtype=np.int32)
    for layer in range(LAYERS - 1, -1, -1):
        index = int(previous[layer][0, used])
        parent = int(previous[layer][1, used])
        if index < 0 or parent < 0:
            raise ValueError("broken H23 allocation backtrace")
        allocation[layer] = BUDGETS[index]
        used = parent
    return allocation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-tokens", type=int, default=2048)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument(
        "--fallback-source", choices=("trace", "fit-state"), default="trace",
        help="replay recorded fallback decisions or recompute them from fit masks",
    )
    parser.add_argument("--io-trace", type=Path, required=True)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--policy-directory", type=Path, required=True)
    args = parser.parse_args()

    atlas = json.loads(args.atlas.read_text())
    layers = atlas.get("layers", [])
    if len(layers) != LAYERS:
        raise ValueError("H22 atlas must contain 92 layers")
    atlas_by_layer = {int(row["model_layer"]): row for row in layers}
    fallbacks = (
        read_trace_fallbacks(args.io_trace)
        if args.fallback_source == "trace"
        else recompute_trace_fallbacks(args.io_trace, args.fit_root)
    )
    geometry = read_layer_geometry(args.traffic)

    costs = np.zeros((LAYERS, BUDGETS.size), dtype=np.float64)
    byte_costs = np.zeros_like(costs)
    option_rows: list[dict[str, object]] = []
    for layer in range(1, LAYERS + 1):
        row = atlas_by_layer[layer]
        curve = local_proxy_curve(
            args.capture_root / f"kimi_layer{layer:02d}_{args.capture_tokens}.bin",
            args.fit_root / f"kimi_layer{layer:02d}_neuron_slabs_calibration.npz",
            layer,
        )
        q96 = max(float(curve[3]), 1e-12)
        sensitivity = float(row["kl_mean"]) / (q96 * q96)
        costs[layer - 1] = sensitivity * curve * curve
        slab_bytes, expert_bytes = geometry[layer]
        for index, budget in enumerate(BUDGETS):
            measured_bytes = measured_option_bytes(
                fallbacks[layer], slab_bytes, expert_bytes, int(budget)
            )
            byte_costs[layer - 1, index] = measured_bytes
            option_rows.append({
                "model_layer": layer,
                "slab_budget": int(budget),
                "measured_bytes_per_position": int(round(measured_bytes)),
                "projected_behavioral_cost": float(costs[layer - 1, index]),
                "measured_kl_at_budget": float(row["kl_mean"]) if budget == 96 else "",
                "mean_exact_fallback_routes": float(np.mean(fallbacks[layer])),
            })

    exact_bytes = float(byte_costs[:, -1].sum())
    fallback_floor = float(sum(
        np.mean(fallbacks[layer]) * geometry[layer][1]
        for layer in range(1, LAYERS + 1)
    ))

    targets = [
        ("SAFE", 4.0),
        ("MEDIUM", 2.5),
        ("AGGRESSIVE", 1.8),
        ("MOONSHOT", 1.2),
    ]
    policies: dict[str, object] = {}
    args.policy_directory.mkdir(parents=True, exist_ok=True)
    for name, gib in targets:
        target = int(gib * (1 << 30))
        allocation = optimize(costs, byte_costs, target)
        if allocation is None:
            policies[name] = {
                "status": "INFEASIBLE_WITH_REGISTERED_BUDGETS_AND_OBSERVED_FALLBACKS",
                "target_gib_per_position": gib,
                "minimum_registered_budget_gib_per_position": float(
                    byte_costs[:, 0].sum() / (1 << 30)
                ),
                "observed_exact_fallback_floor_gib_per_position": fallback_floor / (1 << 30),
            }
            continue
        indices = np.asarray([
            int(np.flatnonzero(BUDGETS == value)[0]) for value in allocation
        ])
        actual_bytes = float(sum(
            byte_costs[layer, indices[layer]] for layer in range(LAYERS)
        ))
        projected_cost = float(sum(
            costs[layer, indices[layer]] for layer in range(LAYERS)
        ))
        path = args.policy_directory / f"h23_{name.lower()}_{str(gib).replace('.', '_')}gib.txt"
        with path.open("w") as output:
            output.write("# H23 PROJECTED slab-only allocation: model_layer nominal_slab_budget\n")
            for layer, budget in enumerate(allocation, 1):
                output.write(f"{layer} {int(budget)}\n")
        policies[name] = {
            "status": "PROJECTED_POLICY_REQUIRES_END_TO_END_QUALITY_RUN",
            "target_gib_per_position": gib,
            "measured_trace_bytes_gib_per_position": actual_bytes / (1 << 30),
            "fraction_of_exact_trace_bytes": actual_bytes / exact_bytes,
            "projected_additive_behavioral_cost": projected_cost,
            "average_nominal_slabs": float(allocation.mean()),
            "budget_counts": {
                str(int(budget)): int((allocation == budget).sum()) for budget in BUDGETS
            },
            "table": str(path),
            "table_sha256": sha256(path),
        }

    result = {
        "schema": "kimi-k3-h23-projected-byte-frontier-v1",
        "status": "MEASURED_BYTES_PROJECTED_BEHAVIOR_INCOMPLETE_ROUTE_AXIS",
        "provenance": {
            "h22_atlas": str(args.atlas),
            "h22_atlas_sha256": sha256(args.atlas),
            "p27_io_trace": str(args.io_trace),
            "p27_io_trace_sha256": sha256(args.io_trace),
            "p27_traffic": str(args.traffic),
            "p27_traffic_sha256": sha256(args.traffic),
            "capture_root": str(args.capture_root),
            "capture_tokens": args.capture_tokens,
            "fit_root": str(args.fit_root),
            "fit_manifest_sha256": sha256(
                args.fit_root / "all_layers_calibration_manifest.json"
            ),
        },
        "method": {
            "byte_cost": (
                "measured P27 route trace and mixed-qtype bytes; fallback decisions "
                f"sourced from {args.fallback_source}"
            ),
            "behavioral_anchor": "measured isolated terminal KL at budget 96",
            "other_budget_cost": "PROJECTED via frozen H22 omitted-residual curve squared",
            "optimization": "exact layer choice dynamic program with conservative 1 MiB byte bins",
            "slab_budgets": BUDGETS.tolist(),
            "route_axis": "NOT OPTIMIZED: no per-layer terminal-damage atlas exists for route counts 4/6/8/12",
        },
        "trace": {
            "model_positions": len(fallbacks[1]),
            "exact_routed_gib_per_position": exact_bytes / (1 << 30),
            "observed_exact_fallback_floor_gib_per_position": fallback_floor / (1 << 30),
            "observed_exact_fallback_fraction": fallback_floor / exact_bytes,
            "minimum_registered_budget_gib_per_position": float(
                byte_costs[:, 0].sum() / (1 << 30)
            ),
        },
        "policies": policies,
        "hard_gaps": [
            "Only budget 96 has measured isolated terminal KL per layer; every other slab budget is projected.",
            "The 32-position P27 trace is one trajectory, not a domain-balanced byte distribution.",
            "No per-layer behavioral atlas exists for exact-route counts 4, 6, 8, or 12, so route/slab choices cannot yet be safely mixed.",
            "The registered 24-slab minimum and observed exact-fallback floor make 1.8 and 1.2 GiB targets infeasible on this trace.",
            "No projected policy has broad official-template quality evidence.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(option_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(option_rows)
    print(json.dumps({"status": result["status"], "trace": result["trace"], "policies": policies}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

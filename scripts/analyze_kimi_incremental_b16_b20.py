#!/usr/bin/env python3
"""Score the preregistered B16 plus four-record B20 correction."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from analyze_kimi_terminal_full_screen import digest, read_logits, terminal_metrics


MARKER = re.compile(
    r"\[kimi-k3-incremental-slab\] layer=(\d+) base-pos=(\d+) "
    r"token=(\d+) base=(\d+) full=(\d+) delta=(\d+)")


def verify_manifest(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        if digest(path) != expected:
            raise ValueError(f"checksum mismatch: {path}")


def environment(root: Path) -> dict[str, str]:
    result = {}
    for item in (root / "environment.nul").read_bytes().split(b"\0"):
        if item:
            key, value = item.decode().split("=", 1)
            result[key] = value
    return result


def selected_records(root: Path, layer: int) -> set[tuple[int, int]]:
    with (root / "candidate-plan.tsv").open(newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if int(row["model_layer"]) == layer]
    terminal_pos = max(int(row["base_pos"]) for row in rows)
    rows = [row for row in rows if int(row["base_pos"]) == terminal_pos]
    if len(rows) != 12:
        raise ValueError(f"layer {layer}: expected 12 route rows, found {len(rows)}")
    return {(int(row["expert"]), int(rank))
            for row in rows
            for rank in filter(None, row["selected_ranks"].split(","))}


def traffic(root: Path, layer: int) -> dict[str, int]:
    with (root / "candidate-traffic.tsv").open(newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if int(row["model_layer"]) == layer]
    if len(rows) != 1:
        raise ValueError(f"layer {layer}: expected one traffic row")
    return {key: int(value) for key, value in rows[0].items()
            if key != "model_layer"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument(
        "--pair", action="append", required=True,
        help="LAYER=CONTROL_ROOT,INCREMENTAL_ROOT; repeat for each layer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    expected_layers = prereg["protocol"]["layers"]
    supplied = {}
    for raw in args.pair:
        layer_text, separator, roots = raw.partition("=")
        control, comma, incremental = roots.partition(",")
        layer = int(layer_text)
        if not separator or not comma or layer in supplied:
            raise ValueError(f"invalid or duplicate --pair: {raw}")
        supplied[layer] = (Path(control), Path(incremental))
    if sorted(supplied) != sorted(expected_layers):
        raise ValueError("supplied layers differ from preregistration")

    frozen_exact = prereg["fixture"].get("frozen_exact_terminal_sha256")
    rows = []
    for layer in expected_layers:
        control_root, incremental_root = supplied[layer]
        verify_manifest(control_root)
        verify_manifest(incremental_root)
        control_env = environment(control_root)
        incremental_env = environment(incremental_root)
        if control_env.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12" or \
                incremental_env.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12":
            raise ValueError(f"layer {layer}: route limit changed")
        if "DFLASH_KIMI_EXPERIMENT_INCREMENTAL_BASE_BUDGET" in control_env or \
                incremental_env.get(
                    "DFLASH_KIMI_EXPERIMENT_INCREMENTAL_BASE_BUDGET") != "16":
            raise ValueError(f"layer {layer}: incremental environment changed")
        if control_env.get("DFLASH_KIMI_EXPERIMENT_ACTIVE_LAYER") != str(layer) or \
                incremental_env.get("DFLASH_KIMI_EXPERIMENT_ACTIVE_LAYER") != str(layer):
            raise ValueError(f"layer {layer}: active layer changed")

        control_exact_path = control_root / "exact-terminal.f32"
        incremental_exact_path = incremental_root / "exact-terminal.f32"
        control_exact_hash = digest(control_exact_path)
        incremental_exact_hash = digest(incremental_exact_path)
        exact_equal = control_exact_hash == incremental_exact_hash and (
            frozen_exact is None or control_exact_hash == frozen_exact)
        control_plan = selected_records(control_root, layer)
        incremental_plan = selected_records(incremental_root, layer)
        control_traffic = traffic(control_root, layer)
        incremental_traffic = traffic(incremental_root, layer)
        markers = [tuple(map(int, match)) for match in MARKER.findall(
            (incremental_root / "run.stderr").read_text())]
        marker_ok = len(markers) == 1 and markers[0][0] == layer and \
            markers[0][3:] == (16, 20, 4)

        teacher = read_logits(control_exact_path)
        control = read_logits(control_root / "candidate-terminal.f32")
        incremental = read_logits(
            incremental_root / "candidate-terminal.f32")
        control_teacher = terminal_metrics(teacher, control)
        incremental_teacher = terminal_metrics(teacher, incremental)
        composition = terminal_metrics(control, incremental)
        teacher_kl_delta = abs(
            float(incremental_teacher["terminal_kl"]) -
            float(control_teacher["terminal_kl"]))
        structural = (
            exact_equal and control_plan == incremental_plan and
            len(control_plan) == 20 and
            control_traffic["selected_slab_records"] == 20 and
            incremental_traffic["selected_slab_records"] == 20 and
            control_traffic["total_provider_bytes"] ==
                incremental_traffic["total_provider_bytes"] and marker_ok)
        quality = (
            control_teacher["candidate_top1"] ==
                incremental_teacher["candidate_top1"] and
            float(composition["terminal_kl"]) <= 1e-6 and
            teacher_kl_delta <= 1e-4)
        rows.append({
            "layer": layer,
            "control_root": str(control_root),
            "incremental_root": str(incremental_root),
            "control_manifest_sha256": digest(control_root / "SHA256SUMS"),
            "incremental_manifest_sha256": digest(
                incremental_root / "SHA256SUMS"),
            "exact_trajectory_sha256": control_exact_hash,
            "exact_trajectory_equal": exact_equal,
            "selected_set_equal": control_plan == incremental_plan,
            "selected_records": len(control_plan),
            "incremental_marker": markers,
            "logical_authoritative_bytes":
                control_traffic["total_provider_bytes"],
            "control_logits_sha256": digest(
                control_root / "candidate-terminal.f32"),
            "incremental_logits_sha256": digest(
                incremental_root / "candidate-terminal.f32"),
            "max_abs_logit_delta": float(np.max(np.abs(control - incremental))),
            "control_vs_incremental": composition,
            "control_vs_exact": control_teacher,
            "incremental_vs_exact": incremental_teacher,
            "absolute_teacher_kl_delta": teacher_kl_delta,
            "structural_gate_pass": structural,
            "quality_gate_pass": quality,
            "gate_pass": structural and quality,
        })

    passed = all(row["gate_pass"] for row in rows)
    result = {
        "schema": "kimi-k3-incremental-b16-b20-equivalence-result-v1",
        "status": "MEASURED_GO" if passed else "MEASURED_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "layers": rows,
        "gate": {
            "passed": passed,
            "decision": (
                "Earn an all-layer behavioral incremental-rescue test; do not "
                "promote production code."
                if passed else
                "Close the current incremental composition order; do not build "
                "the progressive runtime."
            ),
        },
        "limitations": [
            "Frozen exact trajectories are a causal arithmetic discriminator, not on-policy sequence quality.",
            "Host-reference recomposition intentionally does not measure serving throughput.",
            "The candidate reads the same 20 records as control; selective rescue traffic is not measured here."
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

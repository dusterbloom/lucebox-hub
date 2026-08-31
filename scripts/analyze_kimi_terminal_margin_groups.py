#!/usr/bin/env python3
"""Validate and score the four preregistered K3 margin-oracle groups."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from analyze_kimi_terminal_full_screen import digest, read_logits, terminal_metrics


P20 = re.compile(
    r"explicit-provider-reads=(\d+).*direct-physical-bytes=(\d+).*"
    r"direct-io-ns=(\d+)")


def verify_manifest(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        if digest(path) != expected:
            raise ValueError(f"checksum mismatch: {path}")


def read_environment(path: Path) -> dict[str, str]:
    result = {}
    for item in path.read_bytes().split(b"\0"):
        if not item:
            continue
        key, value = item.decode().split("=", 1)
        result[key] = value
    return result


def selected_records(path: Path, layer: int) -> set[tuple[int, int]]:
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if int(row["model_layer"]) == layer]
    terminal_pos = max(int(row["base_pos"]) for row in rows)
    rows = [row for row in rows if int(row["base_pos"]) == terminal_pos]
    if len(rows) != 16:
        raise ValueError(f"expected 16 terminal routes, found {len(rows)}")
    return {
        (int(row["expert"]), int(rank))
        for row in rows
        for rank in filter(None, row["selected_ranks"].split(","))
    }


def traffic(path: Path, layer: int) -> dict[str, int]:
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if int(row["model_layer"]) == layer]
    if len(rows) != 1:
        raise ValueError(f"expected one layer-{layer} traffic row")
    return {key: int(value) for key, value in rows[0].items()
            if key != "model_layer"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--group", action="append", required=True,
                        help="NAME=ARTIFACT_ROOT; repeat exactly four times")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    expected = {group["name"]: group for group in prereg["groups"]}
    supplied = {}
    for raw in args.group:
        name, separator, path = raw.partition("=")
        if not separator or name in supplied:
            raise ValueError(f"invalid or duplicate --group: {raw}")
        supplied[name] = Path(path)
    if supplied.keys() != expected.keys():
        raise ValueError(
            f"groups differ: expected {sorted(expected)}, found {sorted(supplied)}")

    teacher = read_logits(args.teacher)
    baseline = read_logits(args.baseline_root / "candidate-terminal.f32")
    baseline_metrics = terminal_metrics(teacher, baseline)
    if digest(args.teacher) != prereg["source"]["teacher_logits_sha256"]:
        raise ValueError("teacher logits changed")
    if digest(args.baseline_root / "candidate-terminal.f32") != (
            prereg["source"]["baseline_logits_sha256"]):
        raise ValueError("baseline logits changed")

    rows = []
    for name, group in expected.items():
        root = supplied[name]
        verify_manifest(root)
        if digest(root / "exact-terminal.f32") != (
                prereg["source"]["exact_trajectory_sha256"]):
            raise ValueError(f"{name}: exact trajectory changed")
        environment = read_environment(root / "environment.nul")
        if environment.get("DFLASH_KIMI_EXPERIMENT_SLAB_FORCE") != (
                group["force_environment_value"]):
            raise ValueError(f"{name}: force composition changed")
        selected = selected_records(root / "candidate-plan.tsv", layer=92)
        if len(selected) != 24:
            raise ValueError(f"{name}: expected 24 selected records")
        targets = {(member["expert"], member["ordered_rank"])
                   for member in group["members"]}
        if not targets <= selected:
            raise ValueError(f"{name}: one or more forced records were not selected")

        candidate = read_logits(root / "candidate-terminal.f32")
        metrics = terminal_metrics(teacher, candidate)
        p20_matches = P20.findall((root / "run.stderr").read_text())
        if len(p20_matches) != 1:
            raise ValueError(f"{name}: expected one P20 physical-I/O row")
        explicit_reads, physical_bytes, direct_io_ns = map(int, p20_matches[0])
        byte_row = traffic(root / "candidate-traffic.tsv", layer=92)
        if (byte_row["requested_nominal_slabs"] != 24 or
                byte_row["selected_slab_records"] != 24 or
                byte_row["exact_fallback_bytes"] != 0):
            raise ValueError(f"{name}: Budget24 traffic contract changed")
        rows.append({
            "name": name,
            "artifact_root": str(root),
            "member_count": group["member_count"],
            "candidate_logits_sha256": digest(root / "candidate-terminal.f32"),
            "exact_trajectory_sha256": digest(root / "exact-terminal.f32"),
            "candidate_top1": int(metrics["candidate_top1"]),
            "teacher_top1": int(metrics["teacher_top1"]),
            "teacher_top1_recovered": bool(metrics["top1_agreement"]),
            "terminal_kl": float(metrics["terminal_kl"]),
            "relative_kl_reduction": (
                (float(baseline_metrics["terminal_kl"]) -
                 float(metrics["terminal_kl"])) /
                float(baseline_metrics["terminal_kl"])),
            "candidate_teacher_margin": float(metrics["candidate_teacher_margin"]),
            "teacher_margin_recovered": (
                float(metrics["candidate_teacher_margin"]) -
                float(baseline_metrics["candidate_teacher_margin"])),
            "margin_gate_pass": float(metrics["candidate_teacher_margin"]) > 0.0,
            "logical_authoritative_bytes": byte_row["total_provider_bytes"],
            "selected_slab_records": byte_row["selected_slab_records"],
            "exact_fallback_bytes": byte_row["exact_fallback_bytes"],
            "explicit_provider_reads": explicit_reads,
            "direct_physical_bytes": physical_bytes,
            "direct_io_ns": direct_io_ns,
            "command_sha256": digest(root / "command.nul"),
            "environment_sha256": digest(root / "environment.nul"),
            "run_manifest_sha256": digest(root / "SHA256SUMS"),
        })

    passed = any(row["teacher_top1_recovered"] or row["margin_gate_pass"]
                 for row in rows)
    result = {
        "schema": "kimi-k3-terminal-margin-group-result-v1",
        "status": "MEASURED_GATE_A_GO" if passed else "MEASURED_GATE_A_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "teacher_logits_sha256": digest(args.teacher),
        "baseline": {
            **baseline_metrics,
            "candidate_logits_sha256": digest(
                args.baseline_root / "candidate-terminal.f32"),
        },
        "groups": rows,
        "gate_a": {
            "passed": passed,
            "decision": (
                "Captured-tail discriminator is earned as research only."
                if passed else
                "Stop static B24 selection at layer 92; proceed to offline "
                "low-bit complement or progressive rescue."),
        },
        "limitations": prereg["scope"] + [
            "Direct-I/O time was measured under the recorded machine state and is not serving throughput.",
            "No full-model bytes/token, decode tok/s, or prefill tok/s is claimed.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "groups": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

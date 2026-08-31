#!/usr/bin/env python3
"""Analyze one immutable 192-record K3 terminal-slab screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_logits(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != 163840 or not np.isfinite(values).all():
        raise ValueError(f"invalid raw terminal logits: {path}")
    return values.astype(np.float64)


def log_prob(values: np.ndarray) -> np.ndarray:
    return values - np.logaddexp.reduce(values)


def terminal_metrics(teacher: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    teacher_logp = log_prob(teacher)
    candidate_logp = log_prob(candidate)
    probability = np.exp(teacher_logp)
    divergence = max(0.0, float(np.sum(probability * (teacher_logp - candidate_logp))))
    teacher_top = int(np.argmax(teacher))
    candidate_top = int(np.argmax(candidate))
    teacher_second = float(np.max(np.where(
        np.arange(teacher.size) == teacher_top, -np.inf, teacher)))
    candidate_other = float(np.max(np.where(
        np.arange(candidate.size) == teacher_top, -np.inf, candidate)))
    return {
        "terminal_kl": divergence,
        "teacher_top1": teacher_top,
        "candidate_top1": candidate_top,
        "top1_agreement": teacher_top == candidate_top,
        "teacher_top1_margin": float(teacher[teacher_top] - teacher_second),
        "candidate_teacher_margin": float(candidate[teacher_top] - candidate_other),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        tied = np.flatnonzero(values == value)
        result[tied] = result[tied].mean()
    return result


def correlation(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if len(rows) < 3:
        return None
    local = np.asarray([float(row["local_score"]) for row in rows])
    terminal = np.asarray([float(row["equal_byte_terminal_value"]) for row in rows])
    if np.ptp(local) == 0 or np.ptp(terminal) == 0:
        return None
    return {
        "n": len(rows),
        "pearson": float(np.corrcoef(local, terminal)[0, 1]),
        "spearman": float(np.corrcoef(average_ranks(local), average_ranks(terminal))[0, 1]),
    }


def read_terminal_routes(path: Path, layer: int) -> dict[int, dict[str, object]]:
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if int(row["model_layer"]) == layer]
    if not rows:
        raise ValueError(f"no layer {layer} rows: {path}")
    terminal_pos = max(int(row["base_pos"]) for row in rows)
    rows = [row for row in rows if int(row["base_pos"]) == terminal_pos]
    if len(rows) != 16:
        raise ValueError(f"expected 16 terminal routes, found {len(rows)}")
    return {
        int(row["expert"]): {
            "route": int(row["route"]),
            "router_weight": float(row["weight"]),
            "selected_ranks": [int(value) for value in row["selected_ranks"].split(",") if value],
        }
        for row in rows
    }


def read_aux_arrays(metadata_path: Path, binary_path: Path) -> dict[str, np.ndarray]:
    metadata = json.loads(metadata_path.read_text())
    definitions = metadata["arrays"]
    dtypes = {"uint16": "<u2", "float32": "<f4", "uint8": "u1", "uint32": "<u4"}
    result = {}
    for name in ("order", "ordered_residual_importance", "calibrated_experts",
                 "calibration_hit_counts"):
        definition = definitions[name]
        count = int(np.prod(definition["shape"]))
        values = np.fromfile(binary_path, dtype=dtypes[definition["dtype"]],
                             count=count, offset=int(definition["offset"]))
        result[name] = values.reshape(definition["shape"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--aux-json", type=Path, required=True)
    parser.add_argument("--aux-bin", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.layer < 1 or args.layer > 92:
        raise ValueError("layer must be in [1, 92]")
    completion = (args.screen_root / "COMPLETE").read_text().strip().split("\t")
    if len(completion) != 2 or completion[0] != "complete":
        raise ValueError("screen is not complete")
    expected_interventions = int(completion[1])

    teacher = read_logits(args.teacher)
    baseline_path = args.baseline_root / "candidate-terminal.f32"
    exact_path = args.baseline_root / "exact-terminal.f32"
    exact_sha256 = digest(exact_path)
    baseline = read_logits(baseline_path)
    baseline_metrics = terminal_metrics(teacher, baseline)
    routes = read_terminal_routes(args.baseline_root / "candidate-plan.tsv", args.layer)
    aux_metadata = json.loads(args.aux_json.read_text())
    arrays = read_aux_arrays(args.aux_json, args.aux_bin)
    slab_bytes = int(aux_metadata["layout"]["slab_bytes"])

    with (args.screen_root / "screen-plan.tsv").open(newline="") as source:
        interventions = list(csv.DictReader(source, delimiter="\t"))
    if len(interventions) != expected_interventions:
        raise ValueError(f"expected {expected_interventions} interventions, found {len(interventions)}")
    if any(not (Path(row["artifact_dir"]) / "SHA256SUMS").is_file()
           for row in interventions):
        raise ValueError("one or more indexed interventions lack SHA256SUMS")
    rows = []
    for intervention in interventions:
        layer, expert, rank = map(int, intervention["target"].split(":"))
        if layer != args.layer or expert not in routes:
            raise ValueError(f"intervention outside terminal routes: {intervention}")
        action = intervention["action"]
        selected = rank in routes[expert]["selected_ranks"]
        if selected != (action == "drop"):
            raise ValueError(f"action disagrees with baseline selection: {intervention}")
        artifact_dir = Path(intervention["artifact_dir"])
        candidate_path = artifact_dir / "candidate-terminal.f32"
        if digest(artifact_dir / "exact-terminal.f32") != exact_sha256:
            raise ValueError(f"exact trajectory changed: {artifact_dir}")
        candidate_routes = read_terminal_routes(artifact_dir / "candidate-plan.tsv", args.layer)
        if candidate_routes.keys() != routes.keys() or sum(
                len(route["selected_ranks"]) for route in candidate_routes.values()) != 24:
            raise ValueError(f"route set or Budget24 count changed: {artifact_dir}")
        target_selected = rank in candidate_routes[expert]["selected_ranks"]
        if bool(arrays["calibrated_experts"][expert]) and target_selected != (action == "force"):
            raise ValueError(f"intervention did not control its target: {artifact_dir}")
        metrics = terminal_metrics(teacher, read_logits(candidate_path))
        if action == "force":
            terminal_value = float(baseline_metrics["terminal_kl"]) - float(metrics["terminal_kl"])
        else:
            terminal_value = float(metrics["terminal_kl"]) - float(baseline_metrics["terminal_kl"])
        weight = float(routes[expert]["router_weight"])
        importance = float(arrays["ordered_residual_importance"][expert, rank])
        calibrated = bool(arrays["calibrated_experts"][expert])
        rows.append({
            "layer": layer,
            "route": int(routes[expert]["route"]),
            "expert": expert,
            "ordered_rank": rank,
            "natural_slab": int(arrays["order"][expert, rank]),
            "action": action,
            "selected_by_local_budget24": selected,
            "router_weight": weight,
            "residual_importance": importance,
            "local_score": abs(weight) * importance,
            "calibrated_expert": calibrated,
            "calibration_hits": int(arrays["calibration_hit_counts"][expert]),
            "authoritative_slab_bytes": slab_bytes if calibrated else 0,
            "candidate_logits_sha256": digest(candidate_path),
            "equal_byte_terminal_value": terminal_value,
            **metrics,
        })

    result = {
        "schema": "kimi-k3-terminal-full-screen-v1",
        "status": "MEASURED",
        "layer": args.layer,
        "teacher": {"path": str(args.teacher), "sha256": digest(args.teacher)},
        "baseline": {"root": str(args.baseline_root), "logits_sha256": digest(baseline_path),
                     "exact_trajectory_sha256": exact_sha256, **baseline_metrics},
        "screen": {
            "root": str(args.screen_root),
            "plan_sha256": digest(args.screen_root / "screen-plan.tsv"),
            "binary_sha256": [line.split()[0] for line in
                              (args.screen_root / "binary.sha256").read_text().splitlines()],
            "pair_script_sha256": [line.split()[0] for line in
                                   (args.screen_root / "pair-script.sha256").read_text().splitlines()],
            "source_commits": (args.screen_root / "source-commit.txt").read_text().splitlines(),
        },
        "aux": {"json": str(args.aux_json), "json_sha256": digest(args.aux_json),
                "binary": str(args.aux_bin), "binary_sha256": digest(args.aux_bin)},
        "rank_correlation": {
            "force_omitted": correlation([row for row in rows
                                            if row["action"] == "force" and row["calibrated_expert"]]),
            "drop_selected": correlation([row for row in rows if row["action"] == "drop"]),
        },
        "interventions": rows,
        "limitations": [
            "Equal-byte force and drop values are conditional on different displaced records and are not assumed additive.",
            "One frozen terminal trajectory is a causal discriminator, not an on-policy quality result.",
            "Authoritative slab bytes exclude any exact-fallback route bytes already present in the baseline."
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "interventions": len(rows),
                      "baseline_kl": baseline_metrics["terminal_kl"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
